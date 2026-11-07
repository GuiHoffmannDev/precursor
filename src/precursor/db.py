import os
from importlib.resources import files

import psycopg

DEFAULT_URL = "postgresql://precursor:precursor-dev@localhost:5433/precursor"


def connect(url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(url or os.environ.get("DATABASE_URL", DEFAULT_URL))


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(files("precursor").joinpath("schema.sql").read_text())
    conn.commit()
