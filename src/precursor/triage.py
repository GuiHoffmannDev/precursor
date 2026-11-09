"""Does agent triage raise the precision of Phase 3 alerts? A dev-period evaluation.

Alerts come from the Phase 3 weekly queue (default detector, K = 25 new cells per week)
in the 2019-2020 dev years; the 2021-2025 test years are never touched. An alert is a
hit when a recall on its cell is reported within 365 days, exactly as in the backtest.
A fixed, seeded random sample of hits and of misses is investigated as of each alert's
week, blinded (make and model masked, to limit what the LLM can recall from training),
by two arms: the full agent and the same pipeline without retrieval tools.

The sample is balanced, so its raw precision means nothing. Each arm's sensitivity (share
of hits escalated) and false-positive rate (share of misses escalated) are measured on
the sample and projected to the dev base rate b:
    precision = TPR * b / (TPR * b + FPR * (1 - b)).
"""

import hashlib
from dataclasses import asdict
import urllib.error
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import psycopg
from sklearn.metrics import roc_auc_score

from precursor import db, investigate, label, signals
from precursor.evidence import Signal
from precursor.signals import DEFAULT_DETECTOR, DEV, HEADLINE_BUDGET, WARMUP, YEAR, from_day, to_day

ARMS = {"agent": True, "no_tools": False}  # arm name -> use_tools
RANK = {"escalate": 2, "monitor": 1, "dismiss": 0, None: 1}  # invalid runs count as monitor
BOOTSTRAP = 2000


def dev_alerts(conn: psycopg.Connection) -> list[dict]:
    """Every dev-period alert of the default detector at K = 25, labeled hit or miss."""
    data = signals.load_signal_data(conn, DEV[1])
    recalls = signals.load_recalls(conn, data, date.today())  # later recalls only label alerts
    budget = HEADLINE_BUDGET
    alerts = signals.simulate(data, recalls, DEFAULT_DETECTOR, signals.weeks(WARMUP, DEV[1]), (budget,))[budget]
    recall_days: dict[int, list[int]] = {}
    for r in recalls:
        for c in r.cells.tolist():
            recall_days.setdefault(c, []).append(r.day)
    for days in recall_days.values():
        days.sort()
    out, scored = [], {}
    for t, cell, score in alerts:
        if not to_day(DEV[0]) <= t <= to_day(DEV[1]):
            continue
        days = recall_days.get(cell, [])
        if t not in scored:  # complaint counts and expectation for the signal brief
            cells, raw, expected, _ = signals.score(data, t, DEFAULT_DETECTOR)
            scored[t] = {int(c): (int(n), float(e)) for c, n, e in zip(cells, raw, expected, strict=True)}
        n, e = scored[t][cell]
        out.append(data.cell_name(cell) | {
            "as_of": from_day(t), "cell": cell, "score": round(score, 2), "complaints": n,
            "expected": round(e, 2), "hit": bisect_right(days, t + YEAR) > bisect_right(days, t),
        })
    return out


def sample(alerts: list[dict], n_per_class: int, seed: str) -> list[dict]:
    """A deterministic random sample of n hits and n misses."""
    def key(a: dict) -> str:
        return hashlib.md5(f"{seed}:{a['as_of']}:{a['cell']}".encode()).hexdigest()

    hits = sorted((a for a in alerts if a["hit"]), key=key)[:n_per_class]
    misses = sorted((a for a in alerts if not a["hit"]), key=key)[:n_per_class]
    return sorted(hits + misses, key=lambda a: (a["as_of"], a["cell"]))


def to_signal(alert: dict) -> Signal:
    return Signal(alert["as_of"], alert["make"], alert["model"], alert["model_year"], alert["hazard"],
                  window_weeks=DEFAULT_DETECTOR.window_weeks, complaints=alert["complaints"],
                  expected=alert["expected"], score=alert["score"])


EXISTING_SQL = """
    select id, recommendation, confidence, status, llm_calls, tool_calls, prompt_tokens, completion_tokens,
           latency_ms, report
      from investigations
     where tag = %s and prompt_version = %s and blind and use_tools = %s
       and as_of = %s and make = %s and model = %s and model_year = %s and hazard = %s
     order by id desc limit 1
"""


def _existing(conn, tag: str, use_tools: bool, alert: dict) -> dict | None:
    row = conn.execute(EXISTING_SQL, (tag, investigate.PROMPT_VERSION, use_tools, alert["as_of"], alert["make"],
                                      alert["model"], alert["model_year"], alert["hazard"])).fetchone()
    if row is None:
        return None
    keys = ("id", "recommendation", "confidence", "status", "llm_calls", "tool_calls", "prompt_tokens",
            "completion_tokens", "latency_ms", "report")
    return dict(zip(keys, row, strict=True))


def run_arm(arm: str, alerts: list[dict], *, chat, encode, workers: int, connect=db.connect) -> dict[int, dict]:
    """Investigate every alert in one arm; reuses stored runs, so an interrupted evaluation resumes."""
    use_tools, tag = ARMS[arm], f"triage-dev:{arm}"

    def one(i: int) -> tuple[int, dict]:
        with connect() as conn:
            found = _existing(conn, tag, use_tools, alerts[i])
            if found is None:
                deps = investigate.Deps(conn, chat=chat, encode=encode, use_tools=use_tools, blind=True)
                investigate.investigate(deps, to_signal(alerts[i]), tag=tag)
                found = _existing(conn, tag, use_tools, alerts[i])
            return i, found

    results, failed = {}, 0
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, i) for i in range(len(alerts))]
        for fut in as_completed(futures):
            try:
                i, row = fut.result()
            except (urllib.error.URLError, TimeoutError) as e:
                failed += 1  # not stored; a re-run retries it
                print(f"{arm}: LLM request failed: {e}", flush=True)
                continue
            results[i] = row
            if len(results) % 10 == 0:
                print(f"{arm}: {len(results)}/{len(alerts)} done", flush=True)
    if failed:
        raise RuntimeError(f"{arm}: {failed} investigations failed on LLM errors; re-run to resume")
    return results


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(center - half), 4), round(float(center + half), 4)]


def projected_precision(tpr: float, fpr: float, base_rate: float) -> float | None:
    denominator = tpr * base_rate + fpr * (1 - base_rate)
    return tpr * base_rate / denominator if denominator else None


def cutoff_metrics(hits: np.ndarray, misses: np.ndarray, base_rate: float, rng: np.random.Generator) -> dict:
    """Sensitivity, false-positive rate and projected precision for boolean flags on hits and misses."""
    tpr, fpr = hits.mean(), misses.mean()
    boot = []
    for _ in range(BOOTSTRAP):
        p = projected_precision(rng.choice(hits, len(hits)).mean(), rng.choice(misses, len(misses)).mean(), base_rate)
        if p is not None:
            boot.append(p)
    precision = projected_precision(tpr, fpr, base_rate)
    return {
        "sensitivity": round(float(tpr), 4), "sensitivity_ci95": wilson(int(hits.sum()), len(hits)),
        "false_positive_rate": round(float(fpr), 4), "false_positive_rate_ci95": wilson(int(misses.sum()), len(misses)),
        "flagged_share_projected": round(float(tpr * base_rate + fpr * (1 - base_rate)), 4),
        "projected_precision": None if precision is None else round(float(precision), 4),
        "projected_precision_ci95": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)] if boot else None,
        "lift_over_base_rate": None if precision is None else round(float(precision / base_rate), 2),
    }


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    return round(float(roc_auc_score(labels, scores)), 4) if len(set(labels)) == 2 else None


def arm_metrics(alerts: list[dict], rows: dict[int, dict], base_rate: float) -> dict:
    labels = np.array([a["hit"] for a in alerts])
    recs = [rows[i]["recommendation"] for i in range(len(alerts))]
    rng = np.random.default_rng(0)
    escalate = np.array([r == "escalate" for r in recs])
    not_dismissed = np.array([r in ("escalate", "monitor") for r in recs])

    def mean(key: str) -> float:
        return round(float(np.mean([rows[i][key] for i in range(len(alerts))])), 1)

    def claims(key: str) -> float:
        return round(float(np.mean([len((rows[i]["report"] or {}).get(key, [])) for i in range(len(alerts))])), 2)

    return {
        "recommendations": {
            "hits": {r: sum(x == r for x, h in zip(recs, labels, strict=True) if h) for r in (*investigate.RECOMMENDATIONS, None)},
            "misses": {r: sum(x == r for x, h in zip(recs, labels, strict=True) if not h) for r in (*investigate.RECOMMENDATIONS, None)},
        },
        "invalid_runs": sum(rows[i]["status"] != "ok" for i in range(len(alerts))),
        "escalate": cutoff_metrics(escalate[labels], escalate[~labels], base_rate, rng),
        "escalate_or_monitor": cutoff_metrics(not_dismissed[labels], not_dismissed[~labels], base_rate, rng),
        "auc_recommendation": _auc(labels.tolist(), [RANK[r] for r in recs]),
        "mean_claims_kept": claims("claims"), "mean_claims_dropped": claims("dropped_claims"),
        "mean_llm_calls": mean("llm_calls"), "mean_tool_calls": mean("tool_calls"),
        "mean_prompt_tokens": mean("prompt_tokens"), "mean_completion_tokens": mean("completion_tokens"),
        "mean_latency_s": round(mean("latency_ms") / 1000, 1),
    }


def run(conn: psycopg.Connection, *, n_per_class: int = 40, workers: int = 4, seed: str = "triage-v1",
        chat=None, encode=None, connect=db.connect) -> dict:
    """Sample dev alerts, investigate them in both arms, and compare against the detector score."""
    alerts = dev_alerts(conn)
    base_rate = sum(a["hit"] for a in alerts) / len(alerts)
    chosen = sample(alerts, n_per_class, seed)
    print(f"{len(alerts)} dev alerts, base rate {base_rate:.4f}; sampled {len(chosen)}", flush=True)
    chat = chat or label.ollama_chat
    encode = encode if encode is not None else investigate.lazy_encoder()
    rows = {arm: run_arm(arm, chosen, chat=chat, encode=encode, workers=workers, connect=connect) for arm in ARMS}
    labels = [a["hit"] for a in chosen]
    return {
        "protocol": {
            "period": [d.isoformat() for d in DEV], "detector": asdict(DEFAULT_DETECTOR) | {"k_per_week": HEADLINE_BUDGET},
            "label": "hit = a recall on the alert's cell reported within 365 days (as in the backtest)",
            "blind": True, "prompt_version": investigate.PROMPT_VERSION, "llm": label.MODEL,
            "seed": seed, "arms": {arm: {"use_tools": t} for arm, t in ARMS.items()},
            "max_tool_calls": investigate.MAX_TOOL_CALLS, "max_revisions": investigate.MAX_REVISIONS,
            "projection": "precision = TPR*b / (TPR*b + FPR*(1-b)), b = dev base rate; CI by bootstrap",
        },
        "sample": {"dev_alerts": len(alerts), "dev_hits": sum(a["hit"] for a in alerts),
                   "base_rate": round(base_rate, 4), "hits": sum(labels), "misses": len(labels) - sum(labels)},
        "arms": {arm: arm_metrics(chosen, r, base_rate) for arm, r in rows.items()},
        "detector_score": {"auc": _auc(labels, [a["score"] for a in chosen])},
        "alerts": [{k: a[k] for k in ("as_of", "make", "model", "model_year", "hazard", "score", "complaints", "hit")}
                   | {arm: {"id": rows[arm][i]["id"], "recommendation": rows[arm][i]["recommendation"],
                            "confidence": rows[arm][i]["confidence"]} for arm in ARMS}
                   for i, a in enumerate(chosen)],
    }
