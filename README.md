# Precursor

Agentic product-safety signal detection: find emerging vehicle defect patterns in
NHTSA consumer complaints **before** they become recalls, then investigate each
signal with a multi-agent system that cites its evidence.

The project is evaluated with a leakage-free backtest: historical recalls (with
their report dates) are the labels, and every query runs "as of" a past date.

## Status

**Phase 1 — data foundation (done)**

- Idempotent ingestion of NHTSA complaint and recall flat files into Postgres.
- As-of schema: every complaint carries the first date it was public.
- BM25 baseline retrieval (ParadeDB `pg_search`) with the as-of filter pushed into the index.

Loaded locally (2015-01-01 → 2026-11-04): 792,953 complaints, 247,149 recall rows
(15,379 campaigns). Full 2020–2024 chunk loads in ~21 s; an as-of BM25 query over all
complaints runs in ~50 ms.

**Next:** enrichment (NER, hazard classification, embeddings) → signal detection →
backtest → LangGraph investigation agents → eval harness and LLMOps.

## Quickstart

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Python ≥ 3.12.

```bash
docker compose up -d --wait
uv sync
uv run precursor init-db
uv run precursor ingest recalls
uv run precursor ingest complaints --period 2020-2024 --period 2025-2026
uv run precursor search "steering wheel locks up" --as-of 2024-06-01 -k 5
uv run pytest
```

Postgres listens on `127.0.0.1:5433`. Override with `DATABASE_URL`.
Downloaded files are cached in `data/raw/` (use `--refresh` to re-download).
Complaint chunks are 5-year files named by NHTSA, for example `2015-2019` and `2020-2024`.

## Data decisions

These choices matter for the validity of the backtest, so they are recorded here.

| Decision | Why |
|---|---|
| One `complaints` row per `ODINO`, components in `complaint_products` | The flat file repeats a complaint once per component. Retrieval and clustering must see each narrative once. |
| Re-loads replace whole complaints (by `ODINO`) and campaigns (by `CAMPNO`) | NHTSA documents `CMPLID` as updatable, and the natural key has duplicates. Upserting on row ids would create duplicates across releases. |
| `available_at = greatest(LDATE, DATEA)` | This is the first day a complaint was public. All as-of queries filter on it. |
| Recall label date = `RCDATE` (Part 573 report received) | This is the earliest public date of a recall. |
| Driver name, city, VIN prefix and dealer contact details are not loaded | They are not needed, and they are personal data. |

### Known leakage risks (to handle in the backtest)

- **Process-generated complaints.** `source_type` values `RC` (recall complaint), `RP`
  (recall petition) and `DP` (defect petition) are created by the recall process
  itself. Signal detection must exclude them.
- **Retroactive relabeling.** NHTSA's May 2021 system update rewrote make, model and
  component names in historical records to their current values. Component labels
  may therefore reflect knowledge gained after a recall. The backtest should rely on
  narrative text and vehicle identity, not on the component field alone.

## Layout

```
src/precursor/
  schema.sql   tables and indexes (idempotent)
  nhtsa.py     download + parse NHTSA flat files
  load.py      staging COPY + transactional merge
  search.py    as-of BM25 search
  cli.py       `precursor` command
tests/         parser unit tests + DB integration tests (throwaway precursor_test database)
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'precursor'` on macOS with Python ≥ 3.13.**
  Some macOS setups set the `hidden` file flag on files inside `.venv`, and Python
  3.13+ skips hidden `.pth` files, which breaks editable installs. Tests are not
  affected (`pytest` uses `pythonpath = ["src"]`). For the CLI, use a non-editable
  install: `uv run --no-editable precursor ...` or `export UV_NO_EDITABLE=1`. The
  package is rebuilt automatically when `src/` changes.
- **`CERTIFICATE_VERIFY_FAILED`** is handled: downloads use the `certifi` CA bundle.
