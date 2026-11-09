"""Shared fixtures for integration tests against a throwaway `precursor_test` database.

Needs the compose database running (`docker compose up -d`); tests using them are skipped otherwise.
"""

import os

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from precursor import db

TEST_DB = "precursor_test"


@pytest.fixture(scope="session")
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
        c.execute("truncate complaints, recalls, investigations cascade")
        c.commit()
        yield c
