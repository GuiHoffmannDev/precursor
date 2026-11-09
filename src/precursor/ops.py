"""Trace analytics, cost and run comparison over stored investigations (Phase 5).

Every investigation row keeps a per-step trace (node, kind llm/tool/step, tokens,
latency) and its final report. This module reads those rows and reports, per tag and
prompt version: run counts, invalid rate, latency percentiles, token use, the tool-call
mix, how often the code check rejects citations and the critic rejects claims, and an
estimated cost under a configurable price per 1k tokens.

`compare` matches two sets of runs on the same signal cells (as-of date, make, model,
model year, hazard) and reports recommendation agreement and budget ratios; `regressions`
turns a comparison into failures against thresholds. The live regression eval and
`precursor ops-report --compare` both use it, so a new prompt version is judged the same
way whichever path produced its runs.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import psycopg
from psycopg.rows import dict_row

RUN_SQL = """
    select id, as_of, make, model, model_year, hazard, tag, prompt_version, llm, blind, use_tools, status,
           recommendation, report, trace, llm_calls, tool_calls, prompt_tokens, completion_tokens, latency_ms
      from investigations
     order by id
"""

# Defaults for `regressions`: a candidate fails when it agrees with the baseline on fewer
# than min_agreement of matched cells, produces more invalid runs, or exceeds the median
# token or latency budget by these factors.
THRESHOLDS = {"min_agreement": 0.6, "max_invalid_rate_increase": 0.1, "max_token_ratio": 1.5,
              "max_latency_ratio": 2.0}


@dataclass(frozen=True)
class Pricing:
    """USD per 1k tokens. shortcut: illustrative defaults for a hosted ~30B open model;
    Ollama cloud bills by subscription, so pass your provider's real prices."""

    usd_per_1k_prompt: float = 0.0001
    usd_per_1k_completion: float = 0.0004

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens / 1000 * self.usd_per_1k_prompt + completion_tokens / 1000 * self.usd_per_1k_completion


def load_runs(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(RUN_SQL).fetchall()


def _pct(values, q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if len(values) else None


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _mean(values) -> float | None:
    return round(float(np.mean(values)), 2) if len(values) else None


def group_metrics(runs: list[dict], pricing: Pricing) -> dict:
    """Operational metrics for one group of runs."""
    n = len(runs)
    ok = [r for r in runs if r["status"] == "ok"]
    latency_s = [r["latency_ms"] / 1000 for r in runs]
    tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in runs]
    costs = [pricing.cost(r["prompt_tokens"], r["completion_tokens"]) for r in runs]

    nodes: dict[str, list[dict]] = {}
    tool_mix, tool_errors, parse_errors = Counter(), 0, 0
    first = Counter()  # first-draft verification: cited, invalid, claims, unsupported
    for r in runs:
        trace = r["trace"] or []
        for e in trace:
            if e["kind"] == "llm":
                nodes.setdefault(e["node"], []).append(e)
                parse_errors += "error" in e
            elif e["kind"] == "tool":
                tool_mix[e.get("name") or "?"] += 1
                tool_errors += "error" in e
        verify = next((e for e in trace if e["node"] == "verify"), None)
        if verify:
            first["cited"] += verify["cited"]
            first["invalid"] += verify["invalid"]
        rounds = [v for v in ((r["report"] or {}).get("verification") or []) if v.get("report")]
        if rounds:
            first["claims"] += rounds[0]["claims"]
            first["unsupported"] += rounds[0]["unsupported_claims"]
            first["critic_errors"] += rounds[0]["critic_error"] is not None

    reports = [r["report"] for r in ok if r["report"]]
    return {
        "runs": n, "ok": len(ok), "invalid": n - len(ok), "invalid_rate": _rate(n - len(ok), n),
        "recommendations": dict(Counter(r["recommendation"] or "invalid" for r in runs).most_common()),
        "latency_s": {"p50": _pct(latency_s, 50), "p95": _pct(latency_s, 95), "max": _pct(latency_s, 100)},
        "tokens_per_run": {"p50": _pct(tokens, 50), "p95": _pct(tokens, 95),
                           "mean_prompt": _mean([r["prompt_tokens"] for r in runs]),
                           "mean_completion": _mean([r["completion_tokens"] for r in runs])},
        "tokens_total": {"prompt": sum(r["prompt_tokens"] for r in runs),
                         "completion": sum(r["completion_tokens"] for r in runs)},
        "mean_llm_calls": _mean([r["llm_calls"] for r in runs]),
        "mean_tool_calls": _mean([r["tool_calls"] for r in runs]),
        "llm_nodes": {node: {"calls": len(es), "mean_prompt_tokens": _mean([e["prompt_tokens"] for e in es]),
                             "mean_completion_tokens": _mean([e["completion_tokens"] for e in es]),
                             "latency_p50_s": _pct([e["latency_ms"] / 1000 for e in es], 50)}
                      for node, es in sorted(nodes.items())},
        "llm_parse_errors": parse_errors,  # answers that failed validation and needed a repair
        "tool_mix": dict(tool_mix.most_common()), "tool_errors": tool_errors,
        "first_draft": {
            "claims": first["claims"], "cited": first["cited"],
            "citation_rejection_rate": _rate(first["invalid"], first["cited"]),
            "critic_rejection_rate": _rate(first["unsupported"], first["claims"]),
            "critic_errors": first["critic_errors"],
        },
        "revised_runs": sum(len((r["report"] or {}).get("verification") or []) > 1 for r in runs),
        "downgraded_escalations": sum(rep["recommendation"] != rep["llm_recommendation"] for rep in reports),
        "mean_claims_kept": _mean([len(rep["claims"]) for rep in reports]),
        "mean_claims_dropped": _mean([len(rep["dropped_claims"]) for rep in reports]),
        "cost_usd": {"total": round(sum(costs), 4), "per_run": round(float(np.mean(costs)), 6) if n else None},
    }


def report(runs: list[dict], pricing: Pricing, comparisons: list[tuple[str, str]] = ()) -> dict:
    """Metrics per `tag @ prompt_version`, per prompt version across tags, and requested comparisons."""
    groups: dict[str, list[dict]] = {}
    versions: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(f"{r['tag']}@{r['prompt_version']}", []).append(r)
        versions.setdefault(r["prompt_version"], []).append(r)
    return {
        "pricing": vars(pricing) | {"note": "USD per 1k tokens; illustrative unless set to a real provider price"},
        "runs": len(runs),
        "groups": {name: {"tag": rs[0]["tag"], "prompt_version": rs[0]["prompt_version"], "llm": rs[0]["llm"]}
                   | group_metrics(rs, pricing) for name, rs in sorted(groups.items())},
        "prompt_versions": {v: group_metrics(rs, pricing) for v, rs in sorted(versions.items())},
        "comparisons": [{"baseline": a, "candidate": b} | compare(select(runs, a), select(runs, b))
                        for a, b in comparisons],
    }


def cell_key(run: dict) -> tuple:
    return (str(run["as_of"]), run["make"], run["model"], int(run["model_year"]), run["hazard"])


def select(runs: list[dict], selector: str) -> dict[tuple, dict]:
    """Runs matching `tag` or `tag@prompt_version`, the latest run per cell."""
    tag, _, version = selector.partition("@")
    picked = {}
    for r in sorted(runs, key=lambda r: r["id"]):
        if r["tag"] == tag and (not version or r["prompt_version"] == version):
            picked[cell_key(r)] = r
    if not picked:
        raise ValueError(f"no runs match {selector!r}")
    return picked


def _side(runs: list[dict]) -> dict:
    tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in runs]
    latency = [r["latency_ms"] / 1000 for r in runs]
    return {"invalid_rate": _rate(sum(r["status"] != "ok" for r in runs), len(runs)),
            "escalate_rate": _rate(sum(r["recommendation"] == "escalate" for r in runs), len(runs)),
            "tokens_p50": _pct(tokens, 50), "latency_p50_s": _pct(latency, 50),
            "mean_llm_calls": _mean([r["llm_calls"] for r in runs]),
            "mean_tool_calls": _mean([r["tool_calls"] for r in runs])}


def compare(baseline: dict[tuple, dict], candidate: dict[tuple, dict]) -> dict:
    """Side-by-side metrics of two run sets on the cells they share."""
    keys = sorted(baseline.keys() & candidate.keys())
    if not keys:
        raise ValueError("the two run sets share no cells")
    base, cand = [baseline[k] for k in keys], [candidate[k] for k in keys]
    confusion: dict[str, Counter] = {}
    for b, c in zip(base, cand, strict=True):
        confusion.setdefault(b["recommendation"] or "invalid", Counter())[c["recommendation"] or "invalid"] += 1
    agree = sum(b["recommendation"] == c["recommendation"] for b, c in zip(base, cand, strict=True))
    return {"matched_cells": len(keys), "recommendation_agreement": _rate(agree, len(keys)),
            "confusion": {b: dict(c) for b, c in sorted(confusion.items())},
            "baseline_metrics": _side(base), "candidate_metrics": _side(cand)}


def regressions(comparison: dict, thresholds: dict = THRESHOLDS) -> list[str]:
    """Human-readable failures of a candidate against its baseline; empty when within thresholds."""
    b, c, out = comparison["baseline_metrics"], comparison["candidate_metrics"], []
    if comparison["recommendation_agreement"] < thresholds["min_agreement"]:
        out.append(f"recommendation agreement {comparison['recommendation_agreement']} "
                   f"< {thresholds['min_agreement']}")
    if c["invalid_rate"] - b["invalid_rate"] > thresholds["max_invalid_rate_increase"]:
        out.append(f"invalid rate {c['invalid_rate']} vs baseline {b['invalid_rate']}")
    for key, limit in (("tokens_p50", "max_token_ratio"), ("latency_p50_s", "max_latency_ratio")):
        if b[key] and c[key] / b[key] > thresholds[limit]:
            out.append(f"{key} {c[key]} is {c[key] / b[key]:.2f}x baseline {b[key]} (limit {thresholds[limit]}x)")
    return out
