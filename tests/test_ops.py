"""Trace analytics, cost and run comparison on synthetic investigation rows (no database)."""

from datetime import date

import pytest

from precursor import ops


def run(i: int, *, tag="t:agent", version="v1", rec="escalate", status="ok", year=2016, latency_ms=10_000,
        prompt=20_000, completion=1_000, invalid=0, unsupported=0, rounds=1) -> dict:
    trace = [
        {"node": "load", "kind": "step", "latency_ms": 50},
        {"node": "agent", "kind": "llm", "latency_ms": 1000, "prompt_tokens": 3000, "completion_tokens": 40,
         "tool_calls": ["search_complaints", "read_complaints"]},
        {"node": "tools", "kind": "tool", "name": "search_complaints", "results": ["1"], "latency_ms": 100},
        {"node": "tools", "kind": "tool", "name": "read_complaints", "error": "invalid ODINO", "latency_ms": 1},
        {"node": "report", "kind": "llm", "latency_ms": 3000, "prompt_tokens": 7000, "completion_tokens": 800,
         "tool_calls": [], **({"error": "no JSON"} if status != "ok" else {})},
        {"node": "verify", "kind": "step", "cited": 10, "invalid": invalid, "latency_ms": 5},
    ]
    verification = [{"round": r + 1, "report": True, "claims": 4, "invalid_citations": invalid,
                     "unsupported_claims": unsupported, "critic_error": None} for r in range(rounds)]
    report = None if status != "ok" else {
        "recommendation": rec, "llm_recommendation": rec, "claims": [{}] * 3, "dropped_claims": [{}],
        "verification": verification}
    return {"id": i, "as_of": date(2020, 1, 5), "make": "FORD", "model": "EXPLORER", "model_year": year,
            "hazard": "braking_failure", "tag": tag, "prompt_version": version, "llm": "m", "status": status,
            "recommendation": rec if status == "ok" else None, "report": report, "trace": trace,
            "llm_calls": 2, "tool_calls": 2, "prompt_tokens": prompt, "completion_tokens": completion,
            "latency_ms": latency_ms}


def test_group_metrics_count_rates_percentiles_and_cost():
    runs = [run(1, invalid=1, unsupported=1, rounds=2), run(2, latency_ms=30_000), run(3, status="invalid")]
    m = ops.group_metrics(runs, ops.Pricing(usd_per_1k_prompt=0.5, usd_per_1k_completion=1.0))
    assert m["runs"] == 3 and m["invalid"] == 1 and m["invalid_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["recommendations"] == {"escalate": 2, "invalid": 1}
    assert m["latency_s"]["p50"] == 10.0 and m["latency_s"]["max"] == 30.0
    assert m["tool_mix"] == {"search_complaints": 3, "read_complaints": 3} and m["tool_errors"] == 3
    assert m["llm_nodes"]["report"]["calls"] == 3 and m["llm_parse_errors"] == 1
    fd = m["first_draft"]
    assert fd["cited"] == 30 and fd["citation_rejection_rate"] == pytest.approx(1 / 30, abs=1e-4)
    assert fd["claims"] == 8 and fd["critic_rejection_rate"] == 0.125  # invalid run has no report
    assert m["revised_runs"] == 1 and m["mean_claims_kept"] == 3
    assert m["cost_usd"]["total"] == pytest.approx(3 * (20 * 0.5 + 1 * 1.0))


def test_report_groups_by_tag_and_prompt_version():
    runs = [run(1), run(2, version="v2"), run(3, tag="t:other")]
    r = ops.report(runs, ops.Pricing(), [("t:agent@v1", "t:agent@v2")])
    assert set(r["groups"]) == {"t:agent@v1", "t:agent@v2", "t:other@v1"}
    assert r["prompt_versions"]["v1"]["runs"] == 2 and r["prompt_versions"]["v2"]["runs"] == 1
    assert r["comparisons"][0]["matched_cells"] == 1 and r["comparisons"][0]["recommendation_agreement"] == 1.0


def test_select_takes_latest_run_per_cell_and_rejects_unknown_selectors():
    runs = [run(1, rec="dismiss"), run(2, rec="escalate"), run(3, version="v2", year=2017)]
    picked = ops.select(runs, "t:agent")
    assert len(picked) == 2 and picked[ops.cell_key(runs[0])]["id"] == 2
    assert list(ops.select(runs, "t:agent@v2").values())[0]["id"] == 3
    with pytest.raises(ValueError, match="no runs"):
        ops.select(runs, "t:agent@v9")


def test_compare_and_regressions_flag_drift_beyond_thresholds():
    base = {ops.cell_key(r): r for r in [run(i, year=2000 + i) for i in range(10)]}
    same = {k: dict(r) for k, r in base.items()}
    assert ops.regressions(ops.compare(base, same)) == []

    worse = [run(i, year=2000 + i, rec="dismiss" if i < 5 else "escalate", prompt=40_000, latency_ms=25_000,
                 status="invalid" if i == 9 else "ok") for i in range(10)]
    cmp = ops.compare(base, {ops.cell_key(r): r for r in worse})
    assert cmp["matched_cells"] == 10 and cmp["recommendation_agreement"] == 0.4
    assert cmp["confusion"]["escalate"] == {"dismiss": 5, "escalate": 4, "invalid": 1}
    problems = ops.regressions(cmp)
    assert len(problems) == 3 and problems[0].startswith("recommendation agreement 0.4")
    assert any(p.startswith("tokens_p50") for p in problems) and any(p.startswith("latency_p50_s") for p in problems)
    assert len(ops.regressions(cmp, ops.THRESHOLDS | {"max_invalid_rate_increase": 0.05})) == 4
    with pytest.raises(ValueError, match="share no cells"):
        ops.compare(base, {("x",): run(99)})
