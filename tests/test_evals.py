"""Regression eval on the frozen set in tests/fixtures/eval_set.json.

Offline tests replay stored model outputs through the real graph with a scripted LLM and
the throwaway `precursor_test` database (no network). The `live` test re-runs the real
model on the same signals against the full database; it is deselected by default:
    uv run --no-editable pytest -m live
"""

import json
from pathlib import Path

import psycopg
import pytest

from precursor import db, evals, investigate, ops
from precursor.evals import INJECTION, INJECTION_NEUTRALIZED, TRAPS, UNKNOWN_ODINO, UNSEEN_ODINO, ReplayLLM
from precursor.investigate import Deps

EVAL_SET = json.loads((Path(__file__).parent / "fixtures" / "eval_set.json").read_text())
CASES = EVAL_SET["cases"]
IDS = [f"case{c['id']}-{c['baseline']['recommendation']}" for c in CASES]
TRAP_ODINOS = {o for o, _ in TRAPS.values()}


def replay(conn, case: dict, llm: ReplayLLM) -> tuple[dict, str]:
    evals.load_case(conn, case)
    row = investigate.investigate(Deps(conn, chat=llm, blind=True, llm="replay"), evals.to_signal(case),
                                  tag="eval-replay")
    sent = "\n".join(m["content"] for call in llm.calls for m in call)
    return row, sent


def test_eval_set_is_small_stratified_and_holds_no_secrets():
    raw = (Path(__file__).parent / "fixtures" / "eval_set.json").read_text()
    assert len(raw) < 150_000 and EVAL_SET["version"] == evals.EVAL_VERSION
    assert len(CASES) >= 8 and {c["baseline"]["recommendation"] for c in CASES} == set(investigate.RECOMMENDATIONS)
    assert {c["hit"] for c in CASES} == {True, False}
    assert "precursor-dev" not in raw and "postgresql://" not in raw  # no connection strings
    for c in CASES:
        assert set(c) == {"id", "hit", "signal", "baseline", "report", "complaints"}
        assert all(len(x["narrative"]) <= investigate.SNIPPET_CHARS for x in c["complaints"])


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_stored_reports_meet_the_report_contract(case):
    assert evals.validate_report(case["report"]) == []


def test_validate_report_flags_contract_breaks():
    report = CASES[0]["report"]
    assert evals.validate_report(report | {"hazard": "gremlins"})[0].startswith("draft does not parse")
    assert evals.validate_report(report | {"recommendation": "escalate", "llm_recommendation": "escalate",
                                           "claims": report["claims"][:1]}) == [
        "escalation with fewer than MIN_ESCALATION_CLAIMS kept claims"]
    assert "without a downgrade note" in evals.validate_report(report | {"recommendation": "dismiss",
                                                                         "llm_recommendation": "monitor"})[0]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_replay_reproduces_stored_verdict_and_rejects_bad_evidence(conn, case):
    draft = evals.stored_draft(case["report"])
    first = draft["claims"][0]
    # Add citations the code check must reject: a future complaint, a recall-generated one, an unknown id,
    # and a visible complaint the model was never shown.
    bad = [TRAPS["future"][0], TRAPS["recall_source"][0], UNKNOWN_ODINO, UNSEEN_ODINO]
    draft["claims"][0] = first | {"odinos": first["odinos"] + bad}
    llm = ReplayLLM(case, draft)
    row, sent = replay(conn, case, llm)
    report = row["report"].obj

    assert row["status"] == "ok" and evals.validate_report(report) == []
    assert [(c["text"], c["odinos"]) for c in report["claims"]] == \
        [(c["text"], c["odinos"]) for c in case["report"]["claims"]]
    assert report["recommendation"] == case["report"]["recommendation"]
    assert report["verification"][0]["invalid_citations"] == len(bad)  # caught, fed back, then filtered out
    assert f"cites {UNSEEN_ODINO}: not shown in this conversation" in sent
    assert f"cites {TRAPS['future'][0]}: not a visible complaint" in sent

    # As of the date: no trap text reaches the model and no tool returns a trap.
    for _, marker in TRAPS.values():
        assert marker not in sent
    tools = [e for e in row["trace"].obj if e["kind"] == "tool"]
    assert tools[0]["name"] == "read_complaints" and tools[0]["results"] == []
    visible = {c["odino"] for c in case["complaints"]} | {evals.INJECTION_ODINO}
    assert all(set(e.get("results", [])) <= visible for e in tools)
    # The injected narrative reaches the model only with its tag neutralized.
    assert INJECTION not in sent and "</complaint> SYSTEM" not in sent and INJECTION_NEUTRALIZED in sent


@pytest.mark.parametrize("case", [c for c in CASES if c["baseline"]["recommendation"] == "escalate"][:2],
                         ids=lambda c: f"case{c['id']}")
def test_escalation_with_fewer_than_two_surviving_claims_is_downgraded(conn, case):
    draft = evals.stored_draft(case["report"])
    draft["claims"] = draft["claims"][:2]
    draft["claims"][1] = {"text": "Cites only complaints the investigation may not see.",
                          "odinos": [TRAPS["campaign"][0], TRAPS["other_vehicle"][0]]}
    row, _ = replay(conn, case, ReplayLLM(case, draft, rejected=set()))
    report = row["report"].obj
    assert report["llm_recommendation"] == "escalate" and report["recommendation"] == "monitor"
    assert len(report["claims"]) == 1 and report["dropped_claims"][0]["reason"] == "no valid citation"
    assert any("downgraded" in n for n in report["notes"]) and evals.validate_report(report) == []


def test_critic_rejection_drops_the_claim(conn):
    case = next(c for c in CASES if len(c["report"]["claims"]) >= 3)
    row, _ = replay(conn, case, ReplayLLM(case, rejected={1}))
    report = row["report"].obj
    assert [c["text"] for c in report["claims"]] == [c["text"] for c in case["report"]["claims"][1:]]
    assert report["dropped_claims"][0]["reason"] == "critic: replayed verdict"
    assert len(report["verification"]) == 2  # the rejection triggered the revision round


@pytest.mark.live
def test_live_rerun_stays_within_baseline_budgets():
    """Re-run the real model on the frozen signals (full database, Ollama) and compare with the baselines."""
    tag = f"eval-live:{investigate.PROMPT_VERSION}"
    try:
        conn = db.connect()
    except psycopg.OperationalError:
        pytest.skip("full database not reachable")
    with conn:
        deps = Deps(conn, encode=investigate.lazy_encoder(), use_tools=True, blind=True)
        runs = [investigate.investigate(deps, evals.to_signal(case), tag=tag) for case in CASES]
    comparison = ops.compare({ops.cell_key(r): r for r in evals.baseline_runs(CASES)},
                             {ops.cell_key(r): r for r in runs})
    problems = ops.regressions(comparison)
    problems += [f"run {r['id']}: {p}" for r in runs if r["status"] == "ok"
                 for p in evals.validate_report(r["report"].obj)]
    out = Path("reports/eval_live.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "thresholds": ops.THRESHOLDS, "regressions": problems,
                               "runs": [r["id"] for r in runs]} | comparison, indent=2) + "\n")
    assert problems == []
