from dataclasses import replace
from datetime import date

import numpy as np

from precursor.signals import (
    NH, SIGNAL_HAZARDS, Detector, Recall, SignalData, evaluate, first_complaint_day, from_day,
    score, simulate, to_day,
)

H = SIGNAL_HAZARDS
T0 = to_day(date(2022, 1, 2))  # a Sunday
BRAKES = H.index("braking_failure")
VEHICLES = [("FORD", "EXPLORER", 2016), ("HONDA", "CIVIC", 2022), ("TOYOTA", "CAMRY", 2020)]


def make_data(rows: list[tuple[int, int, str, bool]]) -> SignalData:
    rows = sorted(rows)
    return SignalData(
        vehicles=VEHICLES,
        day=np.array([r[0] for r in rows], dtype=np.int64),
        vehicle=np.array([r[1] for r in rows], dtype=np.int64),
        hazard=np.array([H.index(r[2]) for r in rows], dtype=np.int64),
        severe=np.array([r[3] for r in rows], dtype=bool),
    )


# Steady background on six hazards for every vehicle, plus a braking surge on the
# Explorer in the 12 days before T0 and a later Camry steering surge.
BACKGROUND = [(T0 - 7 * w + d, v, H[(w + v + d) % 6], False) for w in range(60) for v in range(3) for d in range(3)]
SURGE = [(T0 - i, 0, "braking_failure", i % 2 == 0) for i in range(12)]
LATE = [(T0 + 20 + i, 2, "loss_of_steering", False) for i in range(10)]
DATA = make_data(BACKGROUND + SURGE + LATE)


def test_surge_cell_ranks_first_for_both_methods():
    for det in (Detector(), Detector("count", 4, 3, 1.0)):
        cells, raw, expected, scores = score(DATA, T0, det)
        assert cells[np.argmax(scores)] == 0 * NH + BRAKES
    cells, raw, expected, _ = score(DATA, T0, Detector())
    top = list(cells).index(BRAKES)
    assert raw[top] >= 12 and expected[top] < raw[top]


def test_score_never_reads_after_as_of():
    t = T0 - 5
    past = make_data([r for r in BACKGROUND + SURGE + LATE if r[0] <= t])
    for det in (Detector(), Detector("count", 13, 3, 1.0)):
        for a, b in zip(score(DATA, t, det), score(past, t, det), strict=True):
            np.testing.assert_array_equal(a, b)


def test_backtest_detects_recall_early_and_suppresses_known_recalls():
    steering = 1 * NH + H.index("loss_of_steering")
    camry_steering = 2 * NH + H.index("loss_of_steering")
    recalls = [
        Recall("22V001000", T0 + 60, ("braking_failure",), 1, np.array([BRAKES])),  # Explorer brakes
        Recall("21V900000", T0 - 10, ("loss_of_steering",), 1, np.array([steering])),  # already public
        Recall("22V002000", T0 + 25, ("loss_of_steering",), 1, np.array([camry_steering])),
    ]
    week_days = list(range(T0 - 14, T0 + 70, 7))
    alerts = simulate(DATA, recalls, Detector(min_count=2), week_days, budgets=(1, 3))
    cells = [c for _, c, _ in alerts[3]]
    assert len(cells) == len(set(cells))  # a cell alerts at most once a year
    assert steering not in [c for t, c, _ in alerts[3] if t >= T0 - 10]  # known recall is suppressed

    period = (from_day(T0 - 100), from_day(T0 + 100))
    report = evaluate(alerts[1], recalls, period, T0 + 1000, first_complaint_day(DATA))
    # The Explorer surge alerts in the first week it shows up, 67 days before its recall.
    assert report["recalls"]["n"] == 3
    assert report["recalls"]["median_lead_days"] is not None
    braking_alert = min(t for t, c, _ in alerts[1] if c == BRAKES)
    assert T0 + 60 - braking_alert >= 60
    assert report["precision"] > 0


def test_severity_weight_changes_score_but_not_min_count():
    det = Detector("count", 4, 3, 1.0)
    weighted = replace(det, severe_weight=3.0)
    cells, raw, _, plain = score(DATA, T0, det)
    _, raw_w, _, heavy = score(DATA, T0, weighted)
    np.testing.assert_array_equal(raw, raw_w)  # min_count uses raw counts
    assert heavy[list(cells).index(BRAKES)] > plain[list(cells).index(BRAKES)]
