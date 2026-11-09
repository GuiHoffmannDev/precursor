"""Signal detection over vehicle x hazard cells and a leakage-free recall backtest.

A cell is one vehicle (make, model, model year) and one hazard class predicted from the
complaint narrative. Every score is computed as of a date from complaints with
`available_at <= as_of` only. Complaints produced by recall processes, and narratives
that quote a campaign number, are excluded because they were written after a recall.
The component field is never used: NHTSA relabeled it retroactively in 2021.

Detector: Poisson disproportionality, in the spirit of the pharmacovigilance PRR. In a
trailing window, a cell's expected count is the vehicle's complaint volume times the
hazard's share among all other vehicles. The score is -log10 P(X >= observed). Severe
complaints (crash, fire, injury, death, or predicted severity incident/injury) can
count more than once. The naive baseline ranks cells by raw count.

Backtest: an analyst reviews the top-K new cells each week. A cell is not re-alerted
within a year, and cells with a recall reported in the past year are suppressed (that
recall is public as of the week). A recall is detected when one of its cells alerted
in the year before its Part 573 report date (RCDATE).
"""

import math
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import psycopg
from scipy.stats import poisson

from precursor.label import EXCLUDED_SOURCES, HAZARDS, RECALL_PROMPT_VERSION
from precursor.search import CAMPAIGN_RE

SIGNAL_HAZARDS = tuple(h for h in HAZARDS if h != "not_safety_related")
NH = len(SIGNAL_HAZARDS)
EPOCH = date(1970, 1, 1)
YEAR = 365
LEADS = (0, 30, 90, 180)  # days before RCDATE
BUDGETS = (10, 25, 50, 100)  # new alerts reviewed per week
HEADLINE_BUDGET = 25
DEV = (date(2019, 1, 1), date(2020, 12, 31))  # parameters are tuned here
TEST = (date(2021, 1, 1), date(2025, 12, 31))  # and reported here
WARMUP = date(2018, 1, 1)  # simulation starts early so dev recalls have a full lookback year


def to_day(d: date) -> int:
    return (d - EPOCH).days


def from_day(n: int) -> date:
    return EPOCH + timedelta(days=int(n))


@dataclass(frozen=True)
class Detector:
    method: str = "disproportionality"  # or "count"
    window_weeks: int = 13
    min_count: int = 3
    severe_weight: float = 2.0


# Chosen on the 2019-2020 dev years by `precursor backtest` (see reports/backtest.json).
DEFAULT_DETECTOR = Detector()


@dataclass
class SignalData:
    vehicles: list[tuple[str, str, int]]
    day: np.ndarray  # available_at as days since epoch, sorted ascending
    vehicle: np.ndarray  # index into vehicles
    hazard: np.ndarray  # index into SIGNAL_HAZARDS
    severe: np.ndarray  # bool

    @property
    def n_cells(self) -> int:
        return len(self.vehicles) * NH

    def cell_name(self, cell: int) -> dict:
        make, model, year = self.vehicles[cell // NH]
        return {"make": make, "model": model, "model_year": year, "hazard": SIGNAL_HAZARDS[cell % NH]}


COMPLAINTS_SQL = """
    select distinct c.odino, c.available_at, p.make, p.model, p.model_year, pr.hazard,
           (coalesce(c.crash, false) or coalesce(c.fire, false) or coalesce(c.injured, 0) > 0
            or coalesce(c.deaths, 0) > 0 or pr.severity in ('incident', 'injury')) as severe
      from complaints c
      join complaint_products p using (odino)
      join complaint_predictions pr using (odino)
     where c.available_at <= %(as_of)s
       and coalesce(c.source_type, '') <> all(%(excluded)s)
       and c.narrative !~* %(campaign_re)s
       and p.product_type = 'V'
       and p.make is not null and p.model is not null and p.model_year is not null
       and pr.hazard = any(%(hazards)s)
"""


def load_signal_data(conn: psycopg.Connection, as_of: date) -> SignalData:
    """Eligible complaints public by `as_of`, one row per complaint and vehicle."""
    rows = conn.execute(COMPLAINTS_SQL, {
        "as_of": as_of, "excluded": list(EXCLUDED_SOURCES), "campaign_re": CAMPAIGN_RE,
        "hazards": list(SIGNAL_HAZARDS),
    }).fetchall()
    rows.sort(key=lambda r: (r[1], r[0]))
    vehicles = sorted({(r[2], r[3], r[4]) for r in rows})
    vindex = {v: i for i, v in enumerate(vehicles)}
    hindex = {h: i for i, h in enumerate(SIGNAL_HAZARDS)}
    return SignalData(
        vehicles=vehicles,
        day=np.array([to_day(r[1]) for r in rows], dtype=np.int64),
        vehicle=np.array([vindex[(r[2], r[3], r[4])] for r in rows], dtype=np.int64),
        hazard=np.array([hindex[r[5]] for r in rows], dtype=np.int64),
        severe=np.array([bool(r[6]) for r in rows], dtype=bool),
    )


def score(data: SignalData, as_of: int, det: Detector):
    """Score every cell with at least `min_count` complaints in the window ending `as_of`.

    Returns (cells, raw counts, expected, scores). Only rows with day <= as_of are read.
    """
    lo = np.searchsorted(data.day, as_of - 7 * det.window_weeks, "right")
    hi = np.searchsorted(data.day, as_of, "right")
    v, h = data.vehicle[lo:hi], data.hazard[lo:hi]
    w = np.where(data.severe[lo:hi], det.severe_weight, 1.0)
    cell = v * NH + h
    raw = np.bincount(cell, minlength=data.n_cells)
    cells = np.flatnonzero(raw >= det.min_count)
    obs = np.bincount(cell, weights=w, minlength=data.n_cells)[cells]
    if det.method == "count":
        return cells, raw[cells], np.full(len(cells), np.nan), obs
    if det.method != "disproportionality":
        raise ValueError(f"unknown method {det.method!r}")
    n_v = np.bincount(v, weights=w, minlength=len(data.vehicles))
    c_h = np.bincount(h, weights=w, minlength=NH)
    vi, hi_ = np.divmod(cells, NH)
    # Hazard share among the other vehicles; the +0.5 / +1 keep the expectation positive.
    expected = n_v[vi] * (c_h[hi_] - obs + 0.5) / (w.sum() - n_v[vi] + 1)
    # shortcut: weighted counts are floored to an integer Poisson tail; fine for ranking.
    scores = -poisson.logsf(np.floor(obs) - 1, expected) / math.log(10)
    return cells, raw[cells], expected, scores


@dataclass
class Recall:
    campno: str
    day: int
    hazards: tuple[str, ...]
    n_vehicles: int
    cells: np.ndarray  # cells whose vehicle has eligible complaints
    odi: bool = False  # influenced by an NHTSA (ODI) investigation rather than the manufacturer


RECALLS_SQL = """
    select r.campno, r.report_received_date, r.make, r.model, r.model_year,
           l.label->>'hazard', l.label->>'secondary_hazard', r.influenced_by
      from recalls r
      join recall_labels l on l.campno = r.campno and l.prompt_version = %(version)s and l.label is not null
     where r.recall_type = 'V' and r.report_received_date <= %(as_of)s
"""


def load_recalls(conn: psycopg.Connection, data: SignalData, as_of: date) -> list[Recall]:
    """Labeled vehicle recalls reported by `as_of`, mapped to cells of `data`."""
    vindex = {v: i for i, v in enumerate(data.vehicles)}
    camps: dict[str, dict] = {}
    for campno, rcdate, make, model, year, h1, h2, influenced_by in conn.execute(
            RECALLS_SQL, {"version": RECALL_PROMPT_VERSION, "as_of": as_of}):
        c = camps.setdefault(campno, {"day": to_day(rcdate), "vehicles": set(), "odi": False,
                                      "hazards": tuple(h for h in (h1, h2) if h in SIGNAL_HAZARDS)})
        c["day"] = min(c["day"], to_day(rcdate))
        c["odi"] |= influenced_by == "ODI"
        if make and model and year:
            c["vehicles"].add((make, model, year))
    recalls = []
    for campno, c in sorted(camps.items(), key=lambda kv: (kv[1]["day"], kv[0])):
        cells = [vindex[v] * NH + SIGNAL_HAZARDS.index(h)
                 for v in c["vehicles"] if v in vindex for h in c["hazards"]]
        recalls.append(Recall(campno, c["day"], c["hazards"], len(c["vehicles"]),
                              np.array(sorted(cells), dtype=np.int64), c["odi"]))
    return recalls


def recalled_cells(recalls: list[Recall], start: int, end: int) -> np.ndarray:
    """Cells of recalls reported in (start, end]."""
    cells = [r.cells for r in recalls if start < r.day <= end]
    return np.unique(np.concatenate(cells)) if cells else np.array([], dtype=np.int64)


def signals(conn: psycopg.Connection, as_of: date, k: int = 25, det: Detector = DEFAULT_DETECTOR) -> list[dict]:
    """Top-k cells as of a date; cells with a recall in the past year are flagged, not ranked."""
    data = load_signal_data(conn, as_of)
    t = to_day(as_of)
    cells, raw, expected, scores = score(data, t, det)
    known = np.isin(cells, recalled_cells(load_recalls(conn, data, as_of), t - YEAR, t))
    order = np.lexsort((cells, -scores))
    out = []
    for i in order:
        if known[i]:
            continue
        out.append(data.cell_name(int(cells[i])) | {
            "complaints": int(raw[i]),
            "expected": None if np.isnan(expected[i]) else round(float(expected[i]), 2),
            "score": round(float(scores[i]), 2),
        })
        if len(out) == k:
            break
    return out


def weeks(start: date, end: date) -> list[int]:
    """Sundays from start to end, as days since epoch."""
    first = start + timedelta(days=(6 - start.weekday()) % 7)
    return list(range(to_day(first), to_day(end) + 1, 7))


def simulate(data: SignalData, recalls: list[Recall], det: Detector, week_days: list[int],
             budgets=BUDGETS) -> dict[int, list[tuple[int, int, float]]]:
    """Weekly triage queue: per budget K, the K best new cells each week as (day, cell, score)."""
    last = {k: np.full(data.n_cells, -10 * YEAR, dtype=np.int64) for k in budgets}
    alerts: dict[int, list] = {k: [] for k in budgets}
    by_day = sorted(recalls, key=lambda r: r.day)
    known, lo, hi = Counter(), 0, 0
    for t in week_days:
        while hi < len(by_day) and by_day[hi].day <= t:  # recalls public as of t
            known.update(by_day[hi].cells.tolist())
            hi += 1
        while lo < hi and by_day[lo].day <= t - YEAR:  # older than a year
            known.subtract(by_day[lo].cells.tolist())
            lo += 1
        known_cells = np.fromiter((c for c, n in known.items() if n > 0), dtype=np.int64)
        cells, _, _, scores = score(data, t, det)
        fresh = ~np.isin(cells, known_cells)
        order = np.lexsort((cells, -scores))
        for k in budgets:
            eligible = order[fresh[order] & (last[k][cells[order]] <= t - YEAR)][:k]
            last[k][cells[eligible]] = t
            alerts[k].extend((t, int(cells[i]), float(scores[i])) for i in eligible)
    return alerts


def evaluate(alerts: list[tuple[int, int, float]], recalls: list[Recall], period: tuple[date, date],
             data_end: int, first_complaint: np.ndarray) -> dict:
    """Recall at lead time, lead days, and alert precision for recalls and alerts in `period`."""
    start, end = to_day(period[0]), to_day(period[1])
    recall_days: dict[int, list[int]] = {}
    for r in recalls:
        for c in r.cells.tolist():
            recall_days.setdefault(c, []).append(r.day)
    for days in recall_days.values():
        days.sort()
    alert_days: dict[int, list[int]] = {}
    for t, c, _ in alerts:
        alert_days.setdefault(c, []).append(t)

    rows = []
    for r in recalls:
        if not start <= r.day <= end:
            continue
        cells = r.cells.tolist()
        earliest = min((t for c in cells for t in alert_days.get(c, ())
                        if r.day - YEAR <= t < r.day), default=None)
        rows.append({
            "lead": None if earliest is None else r.day - earliest,
            "matched_vehicle": bool(cells),
            "odi": r.odi,
            "prior_evidence": any(first_complaint[c] < r.day for c in cells),
            "repeat": any(bisect_left(recall_days[c], r.day - 2 * YEAR) < bisect_left(recall_days[c], r.day)
                          for c in cells),
        })

    def detection(subset: list[dict]) -> dict:
        leads = [x["lead"] for x in subset if x["lead"] is not None]
        return {
            "n": len(subset),
            "recall_at_lead": {str(d): _ratio(sum(lead >= max(d, 1) for lead in leads), len(subset)) for d in LEADS},
            "median_lead_days": float(np.median(leads)) if leads else None,
        }

    # An alert is a hit when a recall on its cell follows within a year. Alerts too close
    # to the end of the recall data cannot be judged and are left out.
    horizon = min(end, data_end - YEAR)
    judged = [(t, c) for t, c, _ in alerts if start <= t <= horizon]
    n_weeks = len(range(start, horizon + 1, 7))
    hits = sum(bisect_right(recall_days.get(c, []), t + YEAR) > bisect_right(recall_days.get(c, []), t)
               for t, c in judged)
    return {
        "recalls": detection(rows),
        "recalls_with_prior_evidence": detection([x for x in rows if x["prior_evidence"]]),
        "first_recalls": detection([x for x in rows if not x["repeat"]]),
        "repeat_recalls": detection([x for x in rows if x["repeat"]]),
        "odi_influenced_recalls": detection([x for x in rows if x["odi"]]),
        "non_odi_recalls": detection([x for x in rows if not x["odi"]]),
        "recalls_without_matching_vehicle": sum(not x["matched_vehicle"] for x in rows),
        "alerts": len([a for a in alerts if start <= a[0] <= end]),
        "alerts_judged": len(judged),
        "precision": _ratio(hits, len(judged)),
        "false_alarms_per_week": round((len(judged) - hits) / max(1, n_weeks), 2),
    }


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def first_complaint_day(data: SignalData) -> np.ndarray:
    first = np.full(data.n_cells, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, data.vehicle * NH + data.hazard, data.day)
    return first


def base_rate(data: SignalData, recalls: list[Recall], det: Detector, period: tuple[date, date],
              data_end: int) -> float | None:
    """Share of scored, unsuppressed cells followed by a recall within a year: precision of a random pick."""
    hits = total = 0
    for t in weeks(*period):
        if t > data_end - YEAR:
            break
        cells, _, _, _ = score(data, t, det)
        cells = cells[~np.isin(cells, recalled_cells(recalls, t - YEAR, t))]
        hits += int(np.isin(cells, recalled_cells(recalls, t, t + YEAR)).sum())
        total += len(cells)
    return _ratio(hits, total)


def grid() -> list[Detector]:
    count = [Detector("count", w, 3, 1.0) for w in (4, 13, 26)]
    disp = [Detector("disproportionality", w, m, s) for w in (4, 13, 26) for m in (3, 5) for s in (1.0, 2.0, 3.0)]
    return count + disp


def backtest(conn: psycopg.Connection, end: date = TEST[1], progress=print) -> dict:
    """Tune on the dev years, then report the selected detectors on the test years."""
    data = load_signal_data(conn, end)
    recalls = load_recalls(conn, data, date.today())  # recalls after `end` judge late alerts
    data_end = max(r.day for r in recalls)
    first = first_complaint_day(data)
    week_days = weeks(WARMUP, end)

    runs = []
    for det in grid():
        alerts = simulate(data, recalls, det, week_days)
        dev = evaluate(alerts[HEADLINE_BUDGET], recalls, DEV, data_end, first)
        runs.append((det, alerts, dev))
        progress(f"{det} dev recall@30d={dev['recalls']['recall_at_lead']['30']} precision={dev['precision']}")

    def objective(run):  # dev recall detected >= 30 days early, then precision
        dev = run[2]
        return dev["recalls"]["recall_at_lead"]["30"] or 0, dev["precision"] or 0

    families = {
        "count_baseline": [r for r in runs if r[0].method == "count"],
        "disproportionality_unweighted": [r for r in runs if r[0].method != "count" and r[0].severe_weight == 1],
        "disproportionality_severity_weighted": [r for r in runs if r[0].method != "count" and r[0].severe_weight > 1],
    }
    test = {}
    for name, family in families.items():
        det, alerts, dev = max(family, key=objective)
        test[name] = {
            "detector": asdict(det),
            "dev": dev,
            "test": {str(k): evaluate(alerts[k], recalls, TEST, data_end, first) for k in BUDGETS},
            "test_base_rate": base_rate(data, recalls, det, TEST, data_end),
        }
    test_recalls = [r for r in recalls if to_day(TEST[0]) <= r.day <= to_day(TEST[1])]
    return {
        "protocol": {
            "dev": [d.isoformat() for d in DEV], "test": [d.isoformat() for d in TEST],
            "warmup_from": WARMUP.isoformat(), "budgets_per_week": list(BUDGETS),
            "headline_budget": HEADLINE_BUDGET, "leads_days": list(LEADS),
            "selection": f"max dev recall detected >= 30 days early at K={HEADLINE_BUDGET}, then dev precision",
            "recall_label_prompt": RECALL_PROMPT_VERSION, "lookback_days": YEAR,
            "recall_data_end": from_day(data_end).isoformat(),
        },
        "data": {
            "complaint_vehicle_rows": int(len(data.day)), "vehicles": len(data.vehicles),
            "labeled_vehicle_recalls": len(recalls), "test_recalls": len(test_recalls),
            "test_recalls_without_signal_hazard": sum(not r.hazards for r in test_recalls),
        },
        "dev_grid": [{"detector": asdict(d), "recall_at_30d": dev["recalls"]["recall_at_lead"]["30"],
                      "precision": dev["precision"]} for d, _, dev in runs],
        "selected": test,
    }
