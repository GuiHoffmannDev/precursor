"""Idempotent loads: COPY into a temp staging table, then merge in one transaction.

Re-loading a file replaces every complaint (by ODINO) and every recall campaign
(by CAMPNO) it contains. Replacing whole groups, instead of upserting on NHTSA row
ids, is deliberate: CMPLID and RECORD_ID are not stable across file releases.
"""

from collections.abc import Iterable

import psycopg

from precursor.nhtsa import COMPLAINT_COLUMNS, RECALL_COLUMNS

PRODUCT_COLUMNS = ("cmplid", "mfr_name", "make", "model", "model_year", "component", "product_type")
COMPLAINT_TABLE_COLUMNS = tuple(c for c in COMPLAINT_COLUMNS if c not in PRODUCT_COLUMNS)


def load_complaints(conn: psycopg.Connection, rows: Iterable[tuple]) -> dict[str, int]:
    select = ", ".join(f"{'p' if c in PRODUCT_COLUMNS else 'c'}.{c}" for c in COMPLAINT_COLUMNS)
    cols = ", ".join(COMPLAINT_TABLE_COLUMNS)
    updates = ", ".join(f"{c} = excluded.{c}" for c in COMPLAINT_TABLE_COLUMNS if c != "odino")
    products = ", ".join(("odino", *PRODUCT_COLUMNS))
    with conn.transaction():
        conn.execute(
            f"create temp table stg_complaints on commit drop as select {select} "
            "from complaints c join complaint_products p using (odino) with no data"
        )
        staged = _copy(conn, "stg_complaints", COMPLAINT_COLUMNS, rows)
        upserted = conn.execute(
            f"insert into complaints ({cols}) "
            f"select distinct on (odino) {cols} from stg_complaints order by odino, cmplid "
            f"on conflict (odino) do update set {updates}, loaded_at = now()"
        ).rowcount
        conn.execute(
            "delete from complaint_products p "
            "using (select distinct odino from stg_complaints) s where p.odino = s.odino"
        )
        inserted = conn.execute(
            f"insert into complaint_products ({products}) select {products} from stg_complaints "
            "on conflict do nothing"
        ).rowcount
    return {"rows": staged, "complaints": upserted, "products": inserted}


def load_recalls(conn: psycopg.Connection, rows: Iterable[tuple]) -> dict[str, int]:
    cols = ", ".join(RECALL_COLUMNS)
    with conn.transaction():
        conn.execute(
            f"create temp table stg_recalls on commit drop as select {cols} from recalls with no data"
        )
        staged = _copy(conn, "stg_recalls", RECALL_COLUMNS, rows)
        conn.execute(
            "delete from recalls r "
            "using (select distinct campno from stg_recalls) s where r.campno = s.campno"
        )
        inserted = conn.execute(
            f"insert into recalls ({cols}) select {cols} from stg_recalls on conflict do nothing"
        ).rowcount
    return {"rows": staged, "recalls": inserted}


def _copy(conn: psycopg.Connection, table: str, columns: tuple[str, ...], rows: Iterable[tuple]) -> int:
    n = 0
    with conn.cursor().copy(f"copy {table} ({', '.join(columns)}) from stdin") as cp:
        for row in rows:
            cp.write_row(row)
            n += 1
    return n
