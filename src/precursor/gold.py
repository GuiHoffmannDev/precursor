"""Human gold labels: export a review sheet, import it with validation, score against it.

Only held-out (test split) complaints are exported, so no model has trained on them.
The sheet shows the LLM's answer as a suggestion; reviewers fill gold_hazard and
gold_severity (copying the suggestion when it is right). Blank rows are skipped.
"""

import csv
from pathlib import Path

import psycopg
from sklearn.metrics import accuracy_score, f1_score

from precursor.classify import is_test
from precursor.label import HAZARDS, PROMPT_VERSION, SEVERITIES

COLUMNS = ("odino", "narrative", "llm_hazard", "llm_severity", "gold_hazard", "gold_severity", "notes")

EXPORT_SQL = """
    select l.odino, c.narrative, l.label->>'hazard', l.label->>'severity'
      from llm_labels l join complaints c using (odino)
     where l.prompt_version = %(version)s and l.label is not null
       and not exists (select 1 from gold_labels g where g.odino = l.odino)
     order by md5(l.odino || 'gold')
"""


def export_sheet(conn: psycopg.Connection, path: Path, n: int) -> int:
    rows = [r for r in conn.execute(EXPORT_SQL, {"version": PROMPT_VERSION}) if is_test(r[0])][:n]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for odino, narrative, hazard, severity in rows:
            w.writerow((odino, narrative, hazard, severity, "", "", ""))
    legend = ["gold_hazard (pick one):", *(f"  {k}: {v}" for k, v in HAZARDS.items()),
              "", "gold_severity (pick one):", *(f"  {k}: {v}" for k, v in SEVERITIES.items())]
    path.with_suffix(".legend.txt").write_text("\n".join(legend) + "\n", encoding="utf-8")
    return len(rows)


def read_sheet(path: Path) -> list[tuple[str, str, str, str]]:
    """Parse a reviewed sheet. Raises ValueError listing every invalid row, imports nothing."""
    good, errors = [], []
    with path.open(newline="", encoding="utf-8-sig") as f:  # -sig: Excel adds a BOM
        for line_no, row in enumerate(csv.DictReader(f), 2):
            hazard = (row.get("gold_hazard") or "").strip()
            severity = (row.get("gold_severity") or "").strip()
            if not hazard and not severity:
                continue
            if hazard not in HAZARDS:
                errors.append(f"line {line_no}: gold_hazard {hazard!r} not in {sorted(HAZARDS)}")
            elif severity not in SEVERITIES:
                errors.append(f"line {line_no}: gold_severity {severity!r} not in {sorted(SEVERITIES)}")
            else:
                good.append((row["odino"], hazard, severity, (row.get("notes") or "").strip() or None))
    if errors:
        raise ValueError("\n".join(errors))
    return good


def import_sheet(conn: psycopg.Connection, path: Path) -> int:
    rows = read_sheet(path)
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "insert into gold_labels (odino, hazard, severity, notes) values (%s, %s, %s, %s) "
            "on conflict (odino) do update set hazard = excluded.hazard, "
            "severity = excluded.severity, notes = excluded.notes, created_at = now()",
            rows,
        )
    return len(rows)


SCORE_SQL = """
    select g.hazard, g.severity, l.label->>'hazard', l.label->>'severity', p.hazard, p.severity
      from gold_labels g
      join llm_labels l on l.odino = g.odino and l.prompt_version = %(version)s and l.label is not null
      left join complaint_predictions p on p.odino = g.odino
"""


def score(conn: psycopg.Connection) -> dict:
    """Accuracy and macro-F1 of the LLM and the distilled classifier against human labels."""
    rows = conn.execute(SCORE_SQL, {"version": PROMPT_VERSION}).fetchall()
    report = {"n": len(rows)}
    for task, gold_i, llm_i, clf_i in (("hazard", 0, 2, 4), ("severity", 1, 3, 5)):
        gold = [r[gold_i] for r in rows]
        report[task] = {"llm": _metrics(gold, [r[llm_i] for r in rows])}
        scored = [(r[gold_i], r[clf_i]) for r in rows if r[clf_i] is not None]
        if scored:
            report[task]["classifier"] = _metrics(*map(list, zip(*scored, strict=True)))
    return report


def _metrics(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }
