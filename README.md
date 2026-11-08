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

**Phase 2 — enrichment (done; human gold review pending)**

- Narrative embeddings (`gte-modernbert-base`, 768-d `halfvec`, HNSW) in pgvector.
  The job is resumable and re-embeds narratives that change in later NHTSA releases.
- LLM weak labels (hazard, severity, entity spans) from `gemma4:31b-cloud` via Ollama
  on a deterministic random sample, excluding recall/petition-generated complaints.
  The cloud model ignores constrained decoding, so answers are validated in Python
  with one repair attempt. Entity spans must appear verbatim in the narrative.
- Distillation: majority / TF-IDF / embedding-probe classifiers compared on a fixed
  20% held-out split; the best scores the whole corpus (`reports/classifier.json`).
  Results on 3,000 LLM labels (2,410 train / 590 test), agreement with the LLM:

  | Task | Majority | TF-IDF + LR | Embedding probe |
  |---|---|---|---|
  | hazard (14 classes), macro-F1 / accuracy | 0.03 / 0.30 | 0.63 / 0.73 | **0.67 / 0.76** |
  | severity (4 classes), macro-F1 / accuracy | 0.19 / 0.59 | **0.69 / 0.79** | 0.69 / 0.71 |

  These numbers measure agreement with the LLM teacher, not with humans; the gold set
  below measures both against human labels. The best model per task is refit on all
  3,000 labels and scores all 792,953 complaints into `complaint_predictions`
  (hazard: embedding probe, severity: TF-IDF). Most common predicted hazards:
  loss of propulsion 22%, airbag/restraint 13%, structural/detachment 10%.
  The table scores every complaint, including recall-generated ones; signal
  detection filters them out.
- NER: zero-shot GLiNER vs GLiNER fine-tuned on the LLM spans (`reports/ner_zeroshot.json`,
  `reports/ner.json`). `gliner_small-v2.1`, 3 epochs on the train split, 590 test
  documents, micro-F1 against the LLM spans (best of two prompt sets × thresholds 0.3/0.5):

  | Model | Exact F1 | Overlap F1 | component | failure_mode | driving_condition |
  |---|---|---|---|---|---|
  | Zero-shot | 0.32 | 0.42 | 0.46 | 0.13 | 0.08 |
  | Fine-tuned | **0.69** | **0.79** | **0.77** | **0.65** | **0.56** |

  Per-label columns are exact-match F1. Zero-shot GLiNER finds parts but not the
  failure and context phrases this taxonomy needs; ~2,400 LLM-labeled narratives fix that.
- Human gold set: review sheet export/import with strict validation; LLM and
  classifier are scored against it (`precursor gold score`).

**Next:** signal detection → backtest → LangGraph investigation agents → eval harness and LLMOps.

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

Phase 2 (needs [Ollama](https://ollama.com) signed in for the cloud labeling model):

```bash
uv run precursor embed                      # all complaints, resumable (~3-4 h on an M-series Mac)
uv run precursor label --n 3000             # LLM weak labels
uv run precursor embed --labeled-only       # let training start before the full embed finishes
uv run precursor classify --score           # distill + score the corpus
uv run precursor ner-eval                   # zero-shot GLiNER
uv run precursor ner-train --batch-size 2 && uv run precursor ner-eval --model models/gliner-precursor
uv run precursor gold export                # review data/review/gold_review.csv, then:
uv run precursor gold import && uv run precursor gold score
```

Set `OLLAMA_URL` or `PRECURSOR_LABEL_MODEL` to use another Ollama server or model.

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
- **Narratives that cite the recall.** Hotline transcriptions (prefix `TL*`) often
  quote an existing campaign number ("NHTSA CAMPAIGN NUMBER: 14V373000"). Such
  complaints were written after the recall and must not count as early evidence.
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
  embed.py     resumable narrative embeddings
  label.py     LLM weak labeling with validation and repair
  classify.py  distillation, held-out evaluation, corpus scoring
  ner.py       GLiNER inference, fine-tuning, span metrics
  gold.py      human gold-label sheet and scoring
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
- **Training or embedding crawls, or Postgres restarts, on a 16 GB Mac.** Apple-silicon
  GPU memory is shared with the Docker VM. GLiNER fine-tuning at batch size 8 used
  ~9 GB, pushed the machine into swap (~100 s/step) and starved the database until it
  crashed. Run GPU jobs one at a time and use `ner-train --batch-size 2` (~3 s/step).
  Both `embed` and `label` resume where they stopped, so re-run them after a crash.
