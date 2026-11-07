"""Integration tests against a throwaway `precursor_test` database.

Needs the compose database running (`docker compose up -d`); skipped otherwise.
"""

import os
from datetime import date

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from helpers import complaint_line, recall_line
from precursor import db, nhtsa
from precursor.load import load_complaints, load_recalls
from precursor.search import search

TEST_DB = "precursor_test"


@pytest.fixture(scope="module")
def test_url():
    base = os.environ.get("DATABASE_URL", db.DEFAULT_URL)
    try:
        admin = psycopg.connect(base, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("database not reachable; run `docker compose up -d`")
    with admin:
        admin.execute(f"drop database if exists {TEST_DB} with (force)")
        admin.execute(f"create database {TEST_DB}")
    url = make_conninfo(base, dbname=TEST_DB)
    with psycopg.connect(url) as conn:
        db.init_schema(conn)
    return url


@pytest.fixture
def conn(test_url):
    with psycopg.connect(test_url) as c:
        c.execute("truncate complaints, complaint_products, recalls")
        c.commit()
        yield c


def complaints(*lines: str) -> list[tuple]:
    return [nhtsa.parse_complaint(line.split("\t")) for line in lines]


def counts(conn) -> tuple[int, int]:
    return conn.execute(
        "select (select count(*) from complaints), (select count(*) from complaint_products)"
    ).fetchone()


def test_complaint_load_is_idempotent_and_survives_cmplid_changes(conn):
    # One complaint (ODINO) filed against two components.
    rows = complaints(
        complaint_line(f1="100", f12="STEERING"),
        complaint_line(f1="101", f12="ELECTRICAL SYSTEM"),
    )
    assert load_complaints(conn, rows) == {"rows": 2, "complaints": 1, "products": 2}
    load_complaints(conn, rows)
    assert counts(conn) == (1, 2)

    # A later file renumbers CMPLID and drops one component: replace, do not append.
    load_complaints(conn, complaints(complaint_line(f1="900", f12="STEERING")))
    assert counts(conn) == (1, 1)
    assert conn.execute("select cmplid from complaint_products").fetchone() == ("900",)


def test_search_respects_as_of(conn):
    load_complaints(conn, complaints(
        complaint_line(f2="1", f16="20240101", f17="20240101", f20="steering wheel locked on highway"),
        complaint_line(f2="2", f16="20250101", f17="20250101", f20="steering wheel locked in parking lot"),
        complaint_line(f2="3", f16="20240101", f17="20240101", f20="airbag light stays on"),
    ))
    assert {r["odino"] for r in search(conn, "steering locked", as_of=None)} == {"1", "2"}
    hits = search(conn, "steering locked", as_of=date(2024, 6, 1))
    assert [r["odino"] for r in hits] == ["1"]
    assert hits[0]["vehicles"] == "FORD EXPLORER 2016"


def test_search_rejects_empty_query(conn):
    with pytest.raises(ValueError):
        search(conn, "  ", as_of=None)


def test_recall_reload_replaces_whole_campaign(conn):
    def recalls(*lines):
        return [nhtsa.parse_recall(line.split("\t")) for line in lines]

    load_recalls(conn, recalls(recall_line(f1="1", f5="2016"), recall_line(f1="2", f5="2017")))
    load_recalls(conn, recalls(recall_line(f1="7", f5="2016")))
    assert conn.execute("select record_id, model_year from recalls").fetchall() == [(7, 2016)]
