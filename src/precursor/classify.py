"""Distill LLM weak labels into cheap classifiers and score every complaint.

Three candidates per task are compared on a fixed held-out split:
majority class (floor), TF-IDF + logistic regression (classic baseline) and a
logistic-regression probe on the narrative embeddings (no extra inference cost,
because the corpus is already embedded). The best one by macro-F1 scores the corpus.

Metrics measure agreement with the LLM labels (distillation fidelity), not ground
truth. Accuracy against human labels is measured separately on the gold set.
"""

import hashlib
import json
from datetime import UTC, datetime

import numpy as np
import psycopg
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from precursor.embed import MODEL as EMBED_MODEL
from precursor.label import PROMPT_VERSION

TASKS = ("hazard", "severity")
KINDS = ("majority", "tfidf", "embedding_probe")

DATA_SQL = """
    select l.odino, c.narrative, e.embedding::text, l.label->>'hazard', l.label->>'severity'
      from llm_labels l
      join complaints c using (odino)
      left join complaint_embeddings e on e.odino = l.odino and e.model = %(model)s
     where l.prompt_version = %(version)s and l.label is not null
     order by l.odino
"""


def is_test(odino: str) -> bool:
    """Deterministic 20% held-out split, stable across runs and label additions."""
    return int(hashlib.md5(odino.encode()).hexdigest()[:8], 16) % 5 == 0


def parse_vectors(texts: list[str]) -> np.ndarray:
    return np.array([json.loads(t) for t in texts], dtype=np.float32)  # '[0.1,...]' is valid JSON


def make_model(kind: str):
    lr = dict(max_iter=3000, class_weight="balanced")
    if kind == "majority":
        return DummyClassifier(strategy="most_frequent")
    if kind == "tfidf":
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                             LogisticRegression(C=10, **lr))
    if kind == "embedding_probe":
        return GridSearchCV(LogisticRegression(**lr), {"C": [0.1, 1, 10]}, cv=3, scoring="f1_macro")
    raise ValueError(kind)


def features(kind: str, narratives: list[str], vectors: np.ndarray):
    return vectors if kind == "embedding_probe" else narratives


def load_dataset(conn: psycopg.Connection) -> dict:
    rows = conn.execute(DATA_SQL, {"model": EMBED_MODEL, "version": PROMPT_VERSION}).fetchall()
    missing = sum(r[2] is None for r in rows)
    if missing:
        raise RuntimeError(f"{missing} labeled complaints have no embedding; run `precursor embed --labeled-only`")
    if not rows:
        raise RuntimeError("no labels found; run `precursor label` first")
    return {
        "odino": [r[0] for r in rows],
        "narrative": [r[1] for r in rows],
        "vectors": parse_vectors([r[2] for r in rows]),
        "hazard": np.array([r[3] for r in rows]),
        "severity": np.array([r[4] for r in rows]),
    }


def evaluate(data: dict) -> dict:
    """Fit every candidate on the train split and report held-out metrics per task."""
    test = np.array([is_test(o) for o in data["odino"]])
    narratives = np.array(data["narrative"], dtype=object)
    report = {"prompt_version": PROMPT_VERSION, "embedding_model": EMBED_MODEL,
              "n_train": int((~test).sum()), "n_test": int(test.sum()), "tasks": {}}
    for task in TASKS:
        y = data[task]
        results = {}
        for kind in KINDS:
            if kind == "embedding_probe":
                x_train, x_test = data["vectors"][~test], data["vectors"][test]
            else:
                x_train, x_test = list(narratives[~test]), list(narratives[test])
            pred = make_model(kind).fit(x_train, y[~test]).predict(x_test)
            results[kind] = {
                "macro_f1": round(float(f1_score(y[test], pred, average="macro", zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(y[test], pred)), 4),
            }
        best = max(KINDS, key=lambda k: results[k]["macro_f1"])
        report["tasks"][task] = {"best": best, "results": results,
                                 "label_counts": dict(zip(*np.unique(y, return_counts=True), strict=True))}
    return json.loads(json.dumps(report, default=int))  # numpy ints -> plain ints


SCORE_SQL = """
    select c.odino, c.narrative, e.embedding::text
      from complaint_embeddings e join complaints c using (odino)
     where e.model = %(model)s
"""
MERGE_SQL = """
    insert into complaint_predictions (odino, model_version, hazard, hazard_conf, severity, severity_conf)
    select odino, model_version, hazard, hazard_conf, severity, severity_conf from stg_predictions
    on conflict (odino) do update set
        model_version = excluded.model_version, hazard = excluded.hazard,
        hazard_conf = excluded.hazard_conf, severity = excluded.severity,
        severity_conf = excluded.severity_conf, created_at = now()
"""


def score_corpus(conn: psycopg.Connection, data: dict, report: dict, batch: int = 20_000) -> int:
    """Refit the best model per task on all labels and score every embedded complaint."""
    models = {}
    for task in TASKS:
        kind = report["tasks"][task]["best"]
        models[task] = (kind, make_model(kind).fit(features(kind, data["narrative"], data["vectors"]), data[task]))
    version = f"{PROMPT_VERSION}:" + "+".join(models[t][0] for t in TASKS) + f":{datetime.now(UTC):%Y%m%d}"

    total = 0
    with conn.transaction():
        conn.execute("create temp table stg_predictions (like complaint_predictions including defaults) on commit drop")
        with conn.cursor(name="score") as src, conn.cursor() as dst:
            src.execute(SCORE_SQL, {"model": EMBED_MODEL})
            while rows := src.fetchmany(batch):
                narratives, vectors = [r[1] for r in rows], parse_vectors([r[2] for r in rows])
                out = {}
                for task, (kind, model) in models.items():
                    proba = model.predict_proba(features(kind, narratives, vectors))
                    out[task] = (model.classes_[proba.argmax(1)], proba.max(1))
                with dst.copy("copy stg_predictions (odino, model_version, hazard, hazard_conf, severity, severity_conf) from stdin") as cp:
                    for i, r in enumerate(rows):
                        cp.write_row((r[0], version, out["hazard"][0][i], float(out["hazard"][1][i]),
                                      out["severity"][0][i], float(out["severity"][1][i])))
                total += len(rows)
        conn.execute(MERGE_SQL)
    return total
