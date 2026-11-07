"""BM25 baseline retrieval over complaint narratives (ParadeDB pg_search)."""

from datetime import date

import psycopg
from psycopg.rows import dict_row


def search(conn: psycopg.Connection, query: str, *, as_of: date | None, k: int = 10) -> list[dict]:
    """Top-k complaints for `query`, seeing only complaints public on or before `as_of`.

    `as_of` is keyword-only and has no default so every caller decides explicitly
    whether it may see the future. Pass None only for interactive use.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= k <= 1000:
        raise ValueError("k must be between 1 and 1000")
    # Kept as a literal condition (not "param is null or ...") so pg_search pushes it into the index.
    as_of_filter = "and c.available_at <= %(as_of)s" if as_of else ""
    sql = f"""
        select c.odino, pdb.score(c.odino) as score, c.available_at, c.narrative,
               (select string_agg(distinct concat_ws(' ', p.make, p.model, p.model_year), '; ')
                  from complaint_products p where p.odino = c.odino) as vehicles,
               (select string_agg(distinct p.component, '; ')
                  from complaint_products p where p.odino = c.odino) as components
          from complaints c
         where c.narrative ||| %(q)s {as_of_filter}
         order by score desc
         limit %(k)s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(sql, {"q": query, "as_of": as_of, "k": k}).fetchall()
