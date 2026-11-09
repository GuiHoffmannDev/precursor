"""As-of evidence for signal investigations.

The investigation date and the signal's vehicle always come from the caller (the graph
state), never from the LLM. Every function returns only complaints that signal
detection itself may use: public on or before the date (`available_at`), not generated
by recall or petition processes, and not quoting a campaign number. Recalls are visible
only from their Part 573 report date (`report_received_date`). Searches cover the
signal's make and model in all model years, so an investigator can tell which years
are affected. The component field is never used (NHTSA relabeled it in 2021).
"""

from dataclasses import dataclass
from datetime import date, timedelta

import psycopg
from psycopg.rows import dict_row

from precursor import search as bm25
from precursor.label import EXCLUDED_SOURCES
from precursor.search import CAMPAIGN_RE

RRF_K = 60  # reciprocal-rank-fusion constant from Cormack et al. (2009)
FUSION_DEPTH = 50  # candidates taken from each ranker before fusion
MAX_K = 20


@dataclass(frozen=True)
class Signal:
    """One vehicle x hazard cell to investigate as of a date, with its detector numbers if known."""

    as_of: date
    make: str
    model: str
    model_year: int
    hazard: str
    window_weeks: int = 13
    complaints: int | None = None
    expected: float | None = None
    score: float | None = None

    @property
    def window_start(self) -> date:
        """Exclusive start of the detector window, as in `signals.score`."""
        return self.as_of - timedelta(weeks=self.window_weeks)

    def params(self, **extra) -> dict:
        return {"as_of": self.as_of, "make": self.make, "model": self.model, "year": self.model_year,
                "hazard": self.hazard, "since": self.window_start, "excluded": list(EXCLUDED_SOURCES),
                "campaign_re": CAMPAIGN_RE} | extra


# A complaint the investigation may see: eligible, public by the date, on the signal's make and model.
VISIBLE = """
    c.available_at <= %(as_of)s
    and coalesce(c.source_type, '') <> all(%(excluded)s)
    and c.narrative !~* %(campaign_re)s
    and exists (select 1 from complaint_products p
                 where p.odino = c.odino and p.product_type = 'V'
                   and p.make = %(make)s and p.model = %(model)s)
"""
ON_YEAR = """
    exists (select 1 from complaint_products p
             where p.odino = c.odino and p.product_type = 'V'
               and p.make = %(make)s and p.model = %(model)s and p.model_year = %(year)s)
"""
COLUMNS = """
    c.odino, c.available_at, c.crash, c.fire, c.injured, c.deaths, c.narrative, pr.hazard, pr.severity,
    (select array_agg(distinct p.model_year order by p.model_year) from complaint_products p
      where p.odino = c.odino and p.make = %(make)s and p.model = %(model)s
        and p.model_year is not null) as model_years
"""


def _check_k(k: int) -> None:
    if not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise ValueError(f"k must be an integer between 1 and {MAX_K}")


def cell_complaints(conn: psycopg.Connection, sig: Signal) -> list[dict]:
    """The signal's own complaints: its model year and predicted hazard, inside the detector window."""
    sql = f"""
        select {COLUMNS}
          from complaints c join complaint_predictions pr using (odino)
         where {VISIBLE} and {ON_YEAR}
           and pr.hazard = %(hazard)s and c.available_at > %(since)s
         order by c.available_at desc, c.odino
    """
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(sql, sig.params()).fetchall()


def cell_stats(conn: psycopg.Connection, sig: Signal) -> dict:
    """Counts an investigator needs to judge the trend, the affected years and the hazard mix."""
    p = sig.params(year_ago=sig.as_of - timedelta(days=365))
    monthly = conn.execute(f"""
        select to_char(date_trunc('month', c.available_at), 'YYYY-MM'), count(*)
          from complaints c join complaint_predictions pr using (odino)
         where {VISIBLE} and {ON_YEAR} and pr.hazard = %(hazard)s and c.available_at > %(year_ago)s
         group by 1 order by 1""", p).fetchall()
    by_year = conn.execute(f"""
        select p2.model_year, count(distinct c.odino)
          from complaints c join complaint_predictions pr using (odino)
          join complaint_products p2 on p2.odino = c.odino
         where {VISIBLE} and pr.hazard = %(hazard)s and c.available_at > %(since)s
           and p2.product_type = 'V' and p2.make = %(make)s and p2.model = %(model)s
           and p2.model_year is not null
         group by 1 order by 1""", p).fetchall()
    mix = conn.execute(f"""
        select pr.hazard, count(*)
          from complaints c join complaint_predictions pr using (odino)
         where {VISIBLE} and {ON_YEAR} and c.available_at > %(since)s
         group by 1 order by 2 desc, 1""", p).fetchall()
    return {"monthly": monthly, "by_model_year": by_year, "hazard_mix": mix}


def read_complaints(conn: psycopg.Connection, sig: Signal, odinos: list[str]) -> list[dict]:
    """Visible complaints among `odinos`, in the given order; invisible or unknown ids are dropped."""
    if not odinos:
        return []
    sql = f"""
        select {COLUMNS}
          from complaints c left join complaint_predictions pr using (odino)
         where c.odino = any(%(odinos)s) and {VISIBLE}
    """
    with conn.cursor(row_factory=dict_row) as cur:
        rows = {r["odino"]: r for r in cur.execute(sql, sig.params(odinos=list(odinos)))}
    return [rows[o] for o in dict.fromkeys(odinos) if o in rows]


def visible_odinos(conn: psycopg.Connection, sig: Signal, odinos: list[str]) -> set[str]:
    """The citation check: ids that exist, are on the signal's make and model, and are eligible and public by the date."""
    if not odinos:
        return set()
    rows = conn.execute(f"select c.odino from complaints c where c.odino = any(%(odinos)s) and {VISIBLE}",
                        sig.params(odinos=list(odinos))).fetchall()
    return {r[0] for r in rows}


def bm25_ranked(conn: psycopg.Connection, sig: Signal, query: str, k: int) -> list[str]:
    hits = bm25.search(conn, query, as_of=sig.as_of, k=k, make=sig.make, model=sig.model, eligible=True)
    return [h["odino"] for h in hits]


def vector_ranked(conn: psycopg.Connection, sig: Signal, vector: str, k: int,
                  exclude: tuple[str, ...] = ()) -> list[str]:
    """Exact cosine kNN over the vehicle's visible complaints.

    The candidate set (one make and model) is small, so an exact scan is cheap and,
    unlike a filtered HNSW scan, never returns fewer than k rows when k exist.
    """
    sql = f"""
        with cand as materialized (
            select c.odino, e.embedding <=> %(vec)s::halfvec as dist
              from complaint_embeddings e join complaints c using (odino)
             where {VISIBLE} and c.odino <> all(%(exclude)s)
        )
        select odino from cand order by dist, odino limit %(k)s
    """
    return [r[0] for r in conn.execute(sql, sig.params(vec=vector, exclude=list(exclude), k=k))]


def hybrid_ranked(conn: psycopg.Connection, sig: Signal, query: str, vector: str | None, k: int) -> list[str]:
    """Reciprocal-rank fusion of BM25 and vector rankings; BM25 alone when no query vector is given."""
    _check_k(k)
    rankings = [bm25_ranked(conn, sig, query, FUSION_DEPTH)]
    if vector is not None:
        rankings.append(vector_ranked(conn, sig, vector, FUSION_DEPTH))
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, odino in enumerate(ranking, 1):
            fused[odino] = fused.get(odino, 0.0) + 1 / (RRF_K + rank)
    return sorted(fused, key=lambda o: (-fused[o], o))[:k]


def similar_ranked(conn: psycopg.Connection, sig: Signal, odino: str, k: int) -> list[str]:
    """Nearest visible complaints to a visible complaint, by its stored embedding."""
    _check_k(k)
    row = conn.execute(f"""
        select e.embedding::text from complaint_embeddings e join complaints c using (odino)
         where c.odino = %(odino)s and {VISIBLE}""", sig.params(odino=odino)).fetchone()
    if row is None:
        raise ValueError(f"complaint {odino} is not visible for this vehicle as of {sig.as_of}")
    return vector_ranked(conn, sig, row[0], k, exclude=(odino,))


def prior_recalls(conn: psycopg.Connection, sig: Signal, limit: int = 15) -> list[dict]:
    """Vehicle recalls of the signal's make and model reported on or before the date, newest first."""
    sql = """
        select campno, min(report_received_date) as reported,
               array_agg(distinct model_year order by model_year) filter (where model_year is not null) as model_years,
               max(influenced_by) as influenced_by, max(defect_summary) as defect,
               max(consequence_summary) as consequence
          from recalls
         where recall_type = 'V' and make = %(make)s and model = %(model)s
           and report_received_date <= %(as_of)s
         group by campno
         order by reported desc, campno
         limit %(limit)s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(sql, sig.params(limit=limit)).fetchall()
