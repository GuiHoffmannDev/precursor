"""BM25 baseline retrieval over complaint narratives (ParadeDB pg_search)."""

from datetime import date

import psycopg
from psycopg.rows import dict_row

from precursor.label import EXCLUDED_SOURCES

# Narratives quoting an NHTSA campaign number (14V373000, 14V-373, 18E012) postdate that recall.
CAMPAIGN_RE = r"\m\d{2}[VETC]-?\d{3}(000)?\M"


def search(conn: psycopg.Connection, query: str, *, as_of: date | None, k: int = 10,
           make: str | None = None, model: str | None = None, eligible: bool = False) -> list[dict]:
    """Top-k complaints for `query`, seeing only complaints public on or before `as_of`.

    `as_of` is keyword-only and has no default so every caller decides explicitly
    whether it may see the future. Pass None only for interactive use.
    `make` and `model` (together) restrict results to one vehicle, any model year.
    `eligible` drops complaints that signal detection excludes: recall/petition-generated
    ones and narratives quoting a campaign number.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= k <= 1000:
        raise ValueError("k must be between 1 and 1000")
    if (make is None) != (model is None):
        raise ValueError("make and model must be given together")
    # Kept as a literal condition (not "param is null or ...") so pg_search pushes it into the index.
    as_of_filter = "and c.available_at <= %(as_of)s" if as_of else ""
    vehicle_filter = """and exists (select 1 from complaint_products p
                                     where p.odino = c.odino and p.product_type = 'V'
                                       and p.make = %(make)s and p.model = %(model)s)""" if make else ""
    eligible_filter = """and coalesce(c.source_type, '') <> all(%(excluded)s)
                         and c.narrative !~* %(campaign_re)s""" if eligible else ""
    sql = f"""
        select c.odino, pdb.score(c.odino) as score, c.available_at, c.narrative,
               (select string_agg(distinct concat_ws(' ', p.make, p.model, p.model_year), '; ')
                  from complaint_products p where p.odino = c.odino) as vehicles,
               (select string_agg(distinct p.component, '; ')
                  from complaint_products p where p.odino = c.odino) as components
          from complaints c
         where c.narrative ||| %(q)s {as_of_filter} {vehicle_filter} {eligible_filter}
         order by score desc
         limit %(k)s
    """
    params = {"q": query, "as_of": as_of, "k": k, "make": make, "model": model,
              "excluded": list(EXCLUDED_SOURCES), "campaign_re": CAMPAIGN_RE}
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(sql, params).fetchall()
