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

    emb = sub.add_parser("embed", help="embed narratives that are missing or stale")
    emb.add_argument("--limit", type=int, help="stop after this many complaints")
    emb.add_argument("--labeled-only", action="store_true", help="only complaints with LLM labels")

    lab = sub.add_parser("label", help="LLM weak labels for a random sample of complaints")
    lab.add_argument("--n", type=int, default=100, help="complaints to label")
    lab.add_argument("--workers", type=int, default=8, help="concurrent LLM requests")

    labr = sub.add_parser("label-recalls", help="LLM hazard labels for vehicle recall campaigns")
    labr.add_argument("--since", type=date.fromisoformat, default=date(2018, 1, 1),
                      help="only campaigns reported on or after this date")
    labr.add_argument("--limit", type=int, help="stop after this many campaigns")
    labr.add_argument("--workers", type=int, default=8, help="concurrent LLM requests")

    cls = sub.add_parser("classify", help="distill LLM labels into classifiers and report metrics")
    cls.add_argument("--report", type=Path, default=Path("reports/classifier.json"))
    cls.add_argument("--score", action="store_true", help="also score every embedded complaint")

    nerp = sub.add_parser("ner-eval", help="evaluate zero-shot GLiNER against LLM entity spans")
    nerp.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5])
    nerp.add_argument("--limit", type=int, help="max test documents")
    nerp.add_argument("--report", type=Path, default=Path("reports/ner.json"))
    nerp.add_argument("--model", default=None, help="local fine-tuned model dir (default: base GLiNER)")

    nert = sub.add_parser("ner-train", help="fine-tune GLiNER on the train split of LLM entity spans")
    nert.add_argument("--output", type=Path, default=Path("models/gliner-precursor"))
    nert.add_argument("--epochs", type=int, default=3)
    nert.add_argument("--batch-size", type=int, default=8,
                      help="lower it (e.g. 2) on 16 GB Macs: GPU memory is shared with Docker")

    gold = sub.add_parser("gold", help="human gold labels: export, import, score")
    gold.add_argument("action", choices=["export", "import", "score"])
    gold.add_argument("--path", type=Path, default=Path("data/review/gold_review.csv"))
    gold.add_argument("--n", type=int, default=200, help="rows to export")

    sig = sub.add_parser("signals", help="top vehicle x hazard signals as of a date")
    sig.add_argument("--as-of", type=date.fromisoformat, required=True)
    sig.add_argument("-k", type=int, default=25)

    bt = sub.add_parser("backtest", help="tune on 2019-2020, report recall detection on 2021-2025")
    bt.add_argument("--report", type=Path, default=Path("reports/backtest.json"))

    inv = sub.add_parser("investigate", help="agent investigation of one signal as of a date")
    inv.add_argument("--as-of", type=date.fromisoformat, required=True)
    inv.add_argument("--signal", type=int, default=1, help="rank in `precursor signals` (1 = top)")
    inv.add_argument("--make", help="investigate this vehicle x hazard cell instead of a ranked signal")
    inv.add_argument("--model")
    inv.add_argument("--year", type=int)
    inv.add_argument("--hazard")
    inv.add_argument("--blind", action="store_true", help="mask make and model in everything the LLM sees")
    inv.add_argument("--no-tools", action="store_true", help="report from the up-front evidence only")
    inv.add_argument("--no-vector", action="store_true", help="BM25-only search (skips loading the embedding model)")
    inv.add_argument("--markdown", type=Path, help="also write the report to this file")

    tri = sub.add_parser("triage-eval", help="agent triage of dev-period (2019-2020) alerts vs backtest labels")
    tri.add_argument("--n-per-class", type=int, default=40, help="alerts sampled among hits and among misses")
    tri.add_argument("--workers", type=int, default=4, help="concurrent investigations")
    tri.add_argument("--seed", default="triage-v1")
    tri.add_argument("--report", type=Path, default=Path("reports/triage.json"))

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
        elif args.command == "embed":
            _embed(conn, args.limit, args.labeled_only)
        elif args.command == "classify":
            import json

            from precursor import classify

            data = classify.load_dataset(conn)
            report = classify.evaluate(data)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            for task, t in report["tasks"].items():
                print(task, "best:", t["best"], t["results"])
            if args.score:
                print(f"scored {classify.score_corpus(conn, data, report)} complaints")
        elif args.command == "ner-eval":
            import json

            from precursor import ner

            name = args.model or ner.GLINER_MODEL
            report = ner.evaluate(conn, ner.load_model(name), thresholds=args.thresholds,
                                  limit=args.limit, model_name=name)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            for run in report["runs"]:
                print(run["prompts"], run["threshold"], "exact", run["exact"]["micro"], "overlap", run["overlap"]["micro"])
        elif args.command == "ner-train":
            from precursor import ner

            print(f"saved to {ner.finetune(conn, args.output, epochs=args.epochs, batch_size=args.batch_size)}")
        elif args.command == "gold":
            from precursor import gold as g

            if args.action == "export":
                print(f"wrote {g.export_sheet(conn, args.path, args.n)} rows to {args.path}")
            elif args.action == "import":
                try:
                    print(f"imported {g.import_sheet(conn, args.path)} gold labels")
                except ValueError as e:
                    print(f"nothing imported, fix these rows:\n{e}", file=sys.stderr)
                    return 1
            else:
                print(g.score(conn))
        elif args.command == "label":
            from precursor import label

            start = time.monotonic()
            stats = label.label_sample(conn, args.n, workers=args.workers)
            print(f"{stats} with {label.MODEL} ({label.PROMPT_VERSION}) in {time.monotonic() - start:.0f}s")
        elif args.command == "signals":
            from precursor import signals

            print(f"{'make':<14}{'model':<22}{'year':<6}{'hazard':<26}{'n':>5}{'expected':>10}{'score':>8}")
            for r in signals.signals(conn, args.as_of, k=args.k):
                exp = "" if r["expected"] is None else f"{r['expected']:.1f}"
                print(f"{r['make'][:13]:<14}{r['model'][:21]:<22}{r['model_year']:<6}{r['hazard']:<26}"
                      f"{r['complaints']:>5}{exp:>10}{r['score']:>8.1f}")
        elif args.command == "backtest":
            import json

            from precursor import signals

            start = time.monotonic()
            report = signals.backtest(conn, progress=lambda m: print(m, flush=True))
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            for name, sel in report["selected"].items():
                t = sel["test"][str(signals.HEADLINE_BUDGET)]
                print(name, sel["detector"], "test K=25:", t["recalls"], "precision", t["precision"],
                      "base rate", sel["test_base_rate"])
            print(f"wrote {args.report} in {time.monotonic() - start:.0f}s")
        elif args.command == "investigate":
            return _investigate(conn, args)
        elif args.command == "triage-eval":
            import json

            from precursor import triage

            report = triage.run(conn, n_per_class=args.n_per_class, workers=args.workers, seed=args.seed)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
            print(json.dumps({k: report[k] for k in ("sample", "arms", "detector_score")}, indent=2, default=str))
            print(f"wrote {args.report}")
        elif args.command == "label-recalls":
            from precursor import label

            start = time.monotonic()
            stats = label.label_recalls(conn, since=args.since.isoformat(), limit=args.limit, workers=args.workers)
            print(f"{stats} with {label.MODEL} ({label.RECALL_PROMPT_VERSION}) in {time.monotonic() - start:.0f}s")
        else:
            for r in search(conn, args.query, as_of=args.as_of, k=args.k):
                print(f"[{r['score']:.2f}] {r['odino']} {r['available_at']} | {r['vehicles']} | {r['components']}")
                print(textwrap.indent(textwrap.shorten(r["narrative"], 300), "    "))
    return 0


def _investigate(conn, args) -> int:
    from dataclasses import replace

    from precursor import evidence, signals
    from precursor import investigate as inv
    from precursor.evidence import Signal

    cell = (args.make, args.model, args.year, args.hazard)
    if any(cell):
        if not all(cell):
            print("--make, --model, --year and --hazard go together", file=sys.stderr)
            return 2
        if args.hazard not in signals.SIGNAL_HAZARDS:
            print(f"--hazard must be one of {', '.join(signals.SIGNAL_HAZARDS)}", file=sys.stderr)
            return 2
        sig = Signal(args.as_of, args.make.upper(), args.model.upper(), args.year, args.hazard,
                     window_weeks=signals.DEFAULT_DETECTOR.window_weeks)
        sig = replace(sig, complaints=len(evidence.cell_complaints(conn, sig)))
    else:
        if args.signal < 1:
            print("--signal must be 1 or more", file=sys.stderr)
            return 2
        ranked = signals.signals(conn, args.as_of, k=args.signal)
        if len(ranked) < args.signal:
            print(f"only {len(ranked)} signals as of {args.as_of}", file=sys.stderr)
            return 1
        s = ranked[args.signal - 1]
        sig = Signal(args.as_of, s["make"], s["model"], s["model_year"], s["hazard"],
                     window_weeks=signals.DEFAULT_DETECTOR.window_weeks,
                     complaints=s["complaints"], expected=s["expected"], score=s["score"])
    deps = inv.Deps(conn, encode=None if args.no_vector else inv.lazy_encoder(),
                    use_tools=not args.no_tools, blind=args.blind)
    row = inv.investigate(deps, sig)
    print(row["markdown"])
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(row["markdown"])
    print(f"stored investigation {row['id']} ({row['status']})")
    return 0 if row["status"] == "ok" else 1


def _embed(conn, limit: int | None, labeled_only: bool) -> None:
    from precursor import embed  # heavy import (torch), only for this command

    model = embed.load_model()
    start = time.monotonic()

    def progress(done: int) -> None:
        print(f"embedded {done} ({done / (time.monotonic() - start):.0f}/s)", flush=True)

    total = embed.embed_pending(conn, lambda texts: embed.encode(model, texts), limit=limit,
                                labeled_only=labeled_only, on_batch=progress)
    print(f"done: {total} complaints embedded with {embed.MODEL}")


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
