import argparse
import sys
import textwrap
import time
from datetime import date
from pathlib import Path

from precursor import db, nhtsa
from precursor.load import load_complaints, load_recalls
from precursor.search import search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="precursor")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create extensions, tables and indexes")

    ingest = sub.add_parser("ingest", help="download NHTSA flat files and load them")
    ingest.add_argument("dataset", choices=["complaints", "recalls"])
    ingest.add_argument("--period", action="append", default=[],
                        help="complaints chunk such as 2020-2024 (repeatable)")
    ingest.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ingest.add_argument("--refresh", action="store_true", help="re-download even if cached")

    find = sub.add_parser("search", help="BM25 search over complaint narratives")
    find.add_argument("query")
    find.add_argument("--as-of", type=date.fromisoformat, help="only complaints public by this date")
    find.add_argument("-k", type=int, default=10)

    args = parser.parse_args(argv)
    with db.connect() as conn:
        if args.command == "init-db":
            db.init_schema(conn)
            print("schema ready")
        elif args.command == "ingest":
            return _ingest(conn, args)
        else:
            for r in search(conn, args.query, as_of=args.as_of, k=args.k):
                print(f"[{r['score']:.2f}] {r['odino']} {r['available_at']} | {r['vehicles']} | {r['components']}")
                print(textwrap.indent(textwrap.shorten(r["narrative"], 300), "    "))
    return 0


def _ingest(conn, args) -> int:
    if args.dataset == "complaints":
        if not args.period:
            print("ingest complaints needs at least one --period, e.g. --period 2025-2026", file=sys.stderr)
            return 2
        jobs = [(nhtsa.complaints_url(p), nhtsa.parse_complaint, load_complaints) for p in args.period]
    else:
        jobs = [(nhtsa.RECALLS_URL, nhtsa.parse_recall, load_recalls)]

    for url, parse, load in jobs:
        start = time.monotonic()
        path = nhtsa.download(url, args.data_dir, refresh=args.refresh)
        skipped: list[int] = []
        counts = load(conn, nhtsa.read_rows(path, parse, skipped))
        print(f"{path.name}: {counts} skipped={len(skipped)} in {time.monotonic() - start:.0f}s")
        if skipped:
            print(f"  first skipped lines: {skipped[:10]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
