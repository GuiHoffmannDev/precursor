import numpy as np

from precursor.classify import KINDS, evaluate, is_test


def synthetic(n: int = 400) -> dict:
    """Two hazards that both the text and the vectors separate perfectly."""
    rng = np.random.default_rng(0)
    hazard = np.array(["fire_or_fuel_leak", "loss_of_steering"] * (n // 2))
    words = {"fire_or_fuel_leak": "engine caught fire smoke", "loss_of_steering": "steering wheel locked up"}
    vectors = rng.normal(size=(n, 8)).astype(np.float32)
    vectors[:, 0] += np.where(hazard == "fire_or_fuel_leak", 5, -5)
    return {
        "odino": [str(10_000 + i) for i in range(n)],
        "narrative": [f"{words[h]} report {i}" for i, h in enumerate(hazard)],
        "vectors": vectors,
        "hazard": hazard,
        "severity": np.array(["potential", "incident"] * (n // 2)),
    }


def test_split_is_deterministic_and_about_20_percent():
    ids = [str(i) for i in range(10_000)]
    share = sum(map(is_test, ids)) / len(ids)
    assert 0.18 < share < 0.22
    assert [is_test(i) for i in ids[:50]] == [is_test(i) for i in ids[:50]]


def test_evaluate_reports_every_candidate_and_beats_majority():
    report = evaluate(synthetic())
    hazard = report["tasks"]["hazard"]
    assert set(hazard["results"]) == set(KINDS)
    assert hazard["results"]["majority"]["macro_f1"] < 0.5
    assert hazard["results"][hazard["best"]]["macro_f1"] == 1.0
    assert hazard["label_counts"] == {"fire_or_fuel_leak": 200, "loss_of_steering": 200}
    assert report["n_train"] + report["n_test"] == 400
