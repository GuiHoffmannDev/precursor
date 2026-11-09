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

**Phase 3 — signal detection and leakage-free backtest (done)**

Question: do NHTSA complaints show a defect before the manufacturer files the Part 573
recall report (`RCDATE`)? Short answer: sometimes, mostly for recalls that NHTSA itself
pushed; the statistical detector does not beat a raw-count baseline.

- **Signal cell:** one vehicle (make, model, model year) × one of 13 hazards (the
  complaint taxonomy without `not_safety_related`). The hazard comes from the
  distilled narrative classifier, never from the component field.
- **Eligible complaints** (as of a date): `available_at <= as_of`, `source_type` not
  RC/RP/DP, and no campaign number quoted in the narrative
  (`\d{2}[VETC]-?\d{3}(000)?`, which removes 99,576 complaints, 12.6%). As of
  2025-12-31: 592,088 complaint × vehicle rows over 16,089 vehicles.
- **Detector:** Poisson disproportionality (PRR-style). In a trailing window, a cell's
  expected count is its vehicle's complaint volume times the hazard's share among all
  other vehicles; score = −log10 P(X ≥ observed). Severe complaints (crash, fire,
  injury, death, or predicted severity incident/injury) can count more than once.
  Baseline: raw count in the window.
- **Triage queue:** each week an analyst reviews the top-K *new* cells. A cell is not
  re-alerted within a year, and cells with a recall reported in the past year are
  suppressed (that recall is public as of the week, so this uses no future data).
- **Recall labels:** `gemma4:31b-cloud` maps every vehicle campaign reported since
  2018 (7,916) to a primary hazard and an optional secondary one from the defect and
  consequence summaries (`recall_labels`, prompt `recall-hazard-v1`; 0 invalid
  answers; 1,501 campaigns got a secondary hazard). Same validation and repair
  pattern as complaint labels.
- **Matching:** a recall is detected when any of its cells (each recalled vehicle ×
  each of its hazards) alerted in the 365 days before `RCDATE`. Lead = `RCDATE` minus
  the earliest such alert. An alert is a hit when a recall on its cell follows within
  365 days; alerts after 2025-11-02 (one year before the end of the recall data) are
  not judged.
- **Protocol:** window (4/13/26 weeks), minimum count (3/5) and severe weight (1/2/3)
  were tuned on 2019–2020 recalls (objective: share detected ≥ 30 days early at
  K = 25). Numbers below are on the untouched 2021–2025 recalls (4,656 vehicle
  campaigns). The whole grid runs in ~10 s (`reports/backtest.json`).

  Test 2021–2025, K = 25 new alerts per week:

  | Detector (selected on dev) | Detected before RCDATE | ≥ 30 d | ≥ 90 d | Median lead | Precision | Random-pick precision | False alarms / week |
  |---|---|---|---|---|---|---|---|
  | Raw count, 26-week window | 6.1% | 5.6% | 4.6% | 215 d | **8.7%** | 6.6% | 22.8 |
  | Disproportionality, 13 weeks | 6.1% | 5.6% | 4.5% | 198 d | 7.8% | 7.8% | 23.0 |
  | + severity weight 2 (default) | 6.2% | 5.7% | 4.4% | 186 d | 7.8% | 7.8% | 23.1 |

  Detected before `RCDATE` at K = 25, by recall type (severity-weighted / raw count):

  | Recalls | n | Weighted | Raw count |
  |---|---|---|---|
  | All | 4,656 | 6.2% | 6.1% |
  | Vehicle has ≥ 1 earlier complaint in a matching cell | 1,505 | 19.2% | 18.8% |
  | Influenced by an ODI investigation | 285 | **36.1%** | 34.0% |
  | First recall on the cell in two years | 3,365 | 3.8% | 3.7% |
  | Follows another recall on the cell within two years | 1,291 | 12.6% | 12.2% |

  With a larger budget, raw count reaches 9.3% of all recalls (42.8% of ODI recalls)
  at K = 50. At K = 10 it has the best precision, 11.6% against a 6.6% random pick
  (1.8× lift); disproportionality gets 9.1% against 7.8% (1.2×).

What this says, and what it does not:

- **The ceiling is the data, not the detector.** 1,740 test recalls (37%) cover no
  vehicle with an eligible complaint (trailers, RVs, buses, heavy trucks, low-volume
  makes), and only 1,505 (32%) have even one earlier complaint in a matching cell.
  Most recalls come from manufacturer testing and warranty data that complaints do
  not see.
- **Complaint-driven recalls are detectable.** About a third of ODI-influenced recalls
  alert before `RCDATE`, with a median lead of about seven months (213–228 days). Part of that lead may
  follow a public ODI investigation that itself drew complaints; investigation open
  dates are not loaded, so this is not separated yet.
- **Disproportionality does not beat raw counts.** Normalizing by vehicle volume
  surfaces anomalies on low-volume vehicles, but recalls concentrate on high-volume
  ones, so ranking by volume gives more precision. Severity weighting changes little.
  The detector is kept as the default because it is the dev-set winner and its
  rankings are more interpretable for an investigator; a hybrid is an open item.
- **Precision is a lower bound.** A true defect that ends in a service bulletin, an
  investigation that closes, or a recall outside the 365-day horizon counts as a
  false alarm.
- **Residual leakage risks.** The hazard classifier was trained on LLM labels sampled
  from all years, so its notion of each hazard saw later complaints (but no recall
  data). Make and model names were rewritten to current values in 2021 on both sides.
  The campaign-number filter is conservative: it also drops complaints that quote an
  older, unrelated recall. Recall labels are LLM output, not human-reviewed.
- **Matching leniency.** Large campaigns span many vehicles and up to two hazards, so
  they offer more cells to match; every detector gets the same leniency, so the
  comparison stays fair, but absolute recall is somewhat optimistic for those campaigns.

**Phase 4 — investigation agents with verified citations (done; triage does not beat the base rate)**

Question: can an LLM agent read a signal's complaints, write a report whose every claim
is backed by real complaints, and triage alerts better than the detector alone?

- **Graph (LangGraph):** `load → agent ⇄ tools → report → verify → critic → (one
  revision) → finalize`, driving `gemma4:31b-cloud` through Ollama's native tool calling.
  - *load* (code) builds the brief, all as of the investigation date: the signal's
    complaints in its window (severe first, 12 shown), monthly counts, counts by model
    year and by hazard, and the make/model's earlier recalls.
  - *agent* (LLM + tools, ≤ 8 tool calls, ≤ 6 turns) searches the vehicle's other
    complaints: hybrid BM25 + vector search (reciprocal-rank fusion), nearest
    neighbours of a complaint (exact cosine kNN over the vehicle's complaints), and full
    narratives.
  - *report* (LLM) writes JSON validated in Python with one repair: summary, hazard,
    failure mode, affected model years, severity, trend, claims with ODINOs, confidence,
    and escalate / monitor / dismiss.
  - *verify* (code) rejects every cited ODINO that does not exist, is not on the signal's
    make and model, is not eligible (RC/RP/DP or quotes a campaign), is not public by the
    date, or was never shown to the LLM in this run.
  - *critic* (LLM) checks each claim against the quoted text and metadata of its valid
    citations.
  - One revision round gets both kinds of feedback. Claims still invalid or rejected are
    dropped; an escalation with fewer than two surviving claims becomes *monitor*.
- **Strictly as of the date.** The date and vehicle live in the graph state and are
  bound by code; tools take only a query, an ODINO or a list of ODINOs, with validated
  arguments and no free SQL. Every query filters `available_at` (and recall
  `report_received_date`) and applies the same eligibility rules as signal detection.
- **Prompt injection.** Narratives and recall texts are wrapped in `<complaint>` /
  `<recall>` tags with angle brackets neutralized, and every prompt says they are
  untrusted data. The tools are read-only.
- **Output.** Each run is stored in `investigations`: the report (JSON and markdown),
  the recommendation, and a trace of every LLM and tool call with tokens and latency
  (for Phase 5). A median agent run: 4–5 LLM calls, 2 tool calls, ~21k prompt tokens,
  ~10 s. Example: [`reports/example_investigation.md`](reports/example_investigation.md)
  (2019 Subaru Forester windshields cracking with no impact, as of 2020-06-07: escalate,
  five verified claims; no recall on that cell followed in the data, so the backtest
  would count it as a false alarm).
- **Citation checks catch real errors.** On the 80 agent runs below, the first drafts
  had 429 claims: 2 cited complaints that failed the code check and 25 were rejected by
  the critic. 21 runs were revised, and only one claim was still dropped at the end.

- **Triage evaluation (`reports/triage.json`).** Does the agent's escalate / dismiss
  call raise the precision of the Phase 3 queue? Alerts: all 2,600 dev-period (2019–2020)
  alerts of the default detector at K = 25; 187 are hits (a recall on the cell within
  365 days), a 7.2% base rate. A seeded sample of 40 hits and 40 misses was investigated
  as of each alert's week, blinded (make and model replaced by `[VEHICLE]` in every
  prompt, tool result and quoted text). Two arms: the full agent, and the same pipeline
  without tools (brief only). Precision is projected to the base rate from each arm's
  sensitivity and false-positive rate; the intervals are Wilson (rates) and bootstrap
  (precision). The 2021–2025 test years were not used.

  | Arm | Escalated: hits | Escalated: misses | Projected precision (95% CI) | Lift | AUC | Mean LLM / tool calls | Mean prompt tokens |
  |---|---|---|---|---|---|---|---|
  | Agent with tools | 75% (60–86%) | 82.5% (68–91%) | 6.6% (5.3–8.1%) | 0.92× | 0.46 | 4.6 / 2.0 | 23.9k |
  | Brief only, no tools | 62.5% (47–76%) | 75% (60–86%) | 6.1% (4.5–8.0%) | 0.84× | 0.44 | 2.3 / 0 | 6.9k |
  | Detector score (same sample) | | | | | 0.46 | | |

  **Result: agent triage does not raise precision.** Both arms escalate about four in
  five alerts, hits and misses alike, and neither ranks hits above misses (AUC below
  0.5, within noise of chance on 80 alerts). Counting *monitor* as a flag does not help
  (6.2% and 6.8%). Tools more than triple the tokens and do not change that. The detector's own
  score does not separate them either, which matches Phase 3: inside the weekly top 25,
  more complaints do not mean a later recall.

  Why, as far as this sample shows:
  - **The agent judges complaint coherence, and most alerts are coherent.** A top-25
    cell is by construction a cluster of similar complaints, so "a specific, consistent
    failure mode" is nearly always true. Whether a recall follows depends on things
    complaints do not show: manufacturer testing and warranty data, cost, and whether
    an earlier campaign or service bulletin already deals with it.
  - **The label is narrow.** Plausible defects such as the Forester windshields above
    end without a recall within a year, so they count as false alarms (Phase 3's
    "precision is a lower bound" applies here too).
  - **The agent does the minimum.** It made exactly the two tool calls the prompt
    requires, almost always `search_complaints`, and the extra evidence rarely changed
    its call.
  - **Blinding removes a signal on purpose.** Without make and model the model cannot
    use what it knows about manufacturers' recall habits; that is the price of not
    letting it remember later recalls. An unblinded run would be easier to beat and
    harder to trust, so it was not used.

  What the agents are good for, on this evidence: turning an alert into a short,
  checkable dossier (failure mode, affected years, severity, cited complaints) for a
  human reviewer, not deciding which alerts to drop. Better triage likely needs
  features complaints lack (investigation and service-bulletin data, manufacturer
  recall history) or a model calibrated on labeled outcomes rather than prompted.

**Next:** eval harness and LLMOps (tracing, cost, regression tests) on top of the stored traces.

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

Phase 3:

```bash
uv run precursor label-recalls                  # LLM hazard labels for vehicle recalls since 2018 (~20 min)
uv run precursor signals --as-of 2024-06-02 -k 15
uv run precursor backtest                       # tune on 2019-2020, report 2021-2025
```

Phase 4 (same Ollama setup):

```bash
uv run precursor investigate --as-of 2020-06-07 --signal 1      # top signal of that week
uv run precursor investigate --as-of 2020-06-07 --make SUBARU --model FORESTER --year 2019 \
    --hazard visibility_or_lighting --markdown reports/example_investigation.md
uv run precursor triage-eval                                     # dev alerts, writes reports/triage.json (resumable)
```

`--blind` masks make and model in everything the LLM sees, `--no-tools` reports from the
brief alone, and `--no-vector` skips loading the embedding model (BM25-only search).

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
| Signals use the narrative hazard, not the component field | NHTSA rewrote component labels retroactively in 2021. |
| Complaints quoting a campaign number are excluded from signals | They were written after that recall; the filter is applied to every as-of query. |
| Tune on 2019–2020, report 2021–2025 | Headline numbers come from years no parameter was chosen on. |
| Agent tools get the date and vehicle from code, never from the LLM | The model cannot ask for the future or for another vehicle. |
| Citations are checked in code, then by a critic | A claim survives only with existing, visible, eligible complaints that were shown to the model and support it. |
| Triage is evaluated blinded, on dev alerts only | Masking make and model limits what the LLM recalls from training; the test years stay untouched. |

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
  signals.py   vehicle x hazard detector, weekly triage simulation, backtest
  evidence.py  as-of evidence for investigations: signal brief, hybrid/kNN search, recalls, citation check
  investigate.py  LangGraph investigation agent, report validation, verification, storage
  triage.py    agent triage of dev-period alerts against backtest labels
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
- **`DiskFull: could not resize shared memory segment`.** Docker gives containers a
  64 MB `/dev/shm`, too small for parallel joins over the whole corpus. The compose
  file sets `shm_size: 512mb`; recreate the container with `docker compose up -d`
  (data lives in the volume).
- **`CERTIFICATE_VERIFY_FAILED`** is handled: downloads use the `certifi` CA bundle.
- **Training or embedding crawls, or Postgres restarts, on a 16 GB Mac.** Apple-silicon
  GPU memory is shared with the Docker VM. GLiNER fine-tuning at batch size 8 used
  ~9 GB, pushed the machine into swap (~100 s/step) and starved the database until it
  crashed. Run GPU jobs one at a time and use `ner-train --batch-size 2` (~3 s/step).
  Both `embed` and `label` resume where they stopped, so re-run them after a crash.
