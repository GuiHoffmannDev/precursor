"""Entity extraction with GLiNER, evaluated against the LLM entity spans.

The LLM spans are weak labels, so these scores measure agreement with the LLM,
not correctness. Spans are character offsets into the original narrative.
"""

import re
from collections import Counter
from pathlib import Path

GLINER_MODEL = "urchade/gliner_small-v2.1"
WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")  # same splitter GLiNER uses internally
# Natural-language prompts per label; GLiNER matches spans against these strings.
LABEL_PROMPTS = {
    "component": "vehicle part",
    "failure_mode": "malfunction",
    "driving_condition": "driving situation",
}
PROMPT_SETS = {
    "short": LABEL_PROMPTS,
    "descriptive": {
        "component": "vehicle component",
        "failure_mode": "failure or malfunction",
        "driving_condition": "driving condition",
    },
}

TEST_LABELS_SQL = """
    select l.odino, c.narrative, l.label->'entities'
      from llm_labels l join complaints c using (odino)
     where l.prompt_version = %(version)s and l.label is not null
     order by l.odino
"""


def evaluate(conn, model, *, thresholds: list[float], limit: int | None = None,
             model_name: str = GLINER_MODEL) -> dict:
    """Score every prompt set x threshold on the held-out split of the LLM labels."""
    from precursor.classify import is_test
    from precursor.label import PROMPT_VERSION

    rows = [r for r in conn.execute(TEST_LABELS_SQL, {"version": PROMPT_VERSION}) if is_test(r[0])][:limit]
    texts = [r[1] for r in rows]
    gold = [[{k: e[k] for k in ("start", "end", "label")} for e in r[2]] for r in rows]
    report = {"model": model_name, "n_docs": len(rows), "n_gold_spans": sum(map(len, gold)), "runs": []}
    for name, prompts in PROMPT_SETS.items():
        for threshold in thresholds:
            pred = predict(model, texts, threshold=threshold, prompts=prompts)
            report["runs"].append({
                "prompts": name, "threshold": threshold,
                "exact": span_scores(gold, pred), "overlap": span_scores(gold, pred, overlap=True),
            })
    return report


def load_model(path: str | Path = GLINER_MODEL):
    import torch
    from gliner import GLiNER

    model = GLiNER.from_pretrained(str(path))
    return model.to("mps") if torch.backends.mps.is_available() else model


def to_training_example(text: str, spans: list[dict], prompts: dict[str, str] = LABEL_PROMPTS) -> dict:
    """Convert character spans to GLiNER's word-index format (inclusive end index)."""
    words = [(m.start(), m.end(), m.group()) for m in WORD.finditer(normalize(text))]
    ner = []
    for s in spans:
        idx = [i for i, (ws, we, _) in enumerate(words) if ws < s["end"] and s["start"] < we]
        if idx:
            ner.append([idx[0], idx[-1], prompts[s["label"]]])
    return {"tokenized_text": [w for _, _, w in words], "ner": ner}


def finetune(conn, output_dir: Path, *, epochs: int = 3, batch_size: int = 8) -> Path:
    """Fine-tune GLiNER on the train split of the LLM spans; the test split stays unseen."""
    from precursor.classify import is_test
    from precursor.label import PROMPT_VERSION

    rows = [r for r in conn.execute(TEST_LABELS_SQL, {"version": PROMPT_VERSION}) if not is_test(r[0])]
    train = [to_training_example(r[1], r[2]) for r in rows]
    model = load_model()
    model.train_model(
        train_dataset=train, eval_dataset=None, output_dir=str(output_dir),
        learning_rate=5e-6, others_lr=1e-5,  # small LRs: adapt, do not overwrite, the pretrained matcher
        per_device_train_batch_size=batch_size,
        max_steps=epochs * len(train) // batch_size,
        save_strategy="no", logging_steps=50, dataloader_num_workers=0,
    )
    model.save_pretrained(str(output_dir))
    return output_dir


def normalize(text: str) -> str:
    """Lowercase (many narratives are ALL CAPS), but only when offsets stay valid."""
    low = text.lower()
    return low if len(low) == len(text) else text


def predict(model, texts: list[str], *, threshold: float, prompts: dict[str, str] = LABEL_PROMPTS,
            batch_size: int = 8) -> list[list[dict]]:
    back = {v: k for k, v in prompts.items()}
    out = model.inference([normalize(t) for t in texts], labels=list(prompts.values()),
                          threshold=threshold, batch_size=batch_size)
    return [[{"start": e["start"], "end": e["end"], "label": back[e["label"]]} for e in doc] for doc in out]


def _match(g: dict, p: dict, overlap: bool) -> bool:
    if g["label"] != p["label"]:
        return False
    if overlap:
        return g["start"] < p["end"] and p["start"] < g["end"]
    return g["start"] == p["start"] and g["end"] == p["end"]


def span_scores(gold: list[list[dict]], pred: list[list[dict]], *, overlap: bool = False) -> dict:
    """Micro and per-label precision/recall/F1 with greedy one-to-one span matching."""
    tp, n_gold, n_pred = Counter(), Counter(), Counter()
    for g_doc, p_doc in zip(gold, pred, strict=True):
        used: set[int] = set()
        for p in p_doc:
            n_pred[p["label"]] += 1
            for i, g in enumerate(g_doc):
                if i not in used and _match(g, p, overlap):
                    used.add(i)
                    tp[p["label"]] += 1
                    break
        for g in g_doc:
            n_gold[g["label"]] += 1

    def prf(t: int, g: int, p: int) -> dict:
        precision = t / p if p else 0.0
        recall = t / g if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    labels = sorted(set(n_gold) | set(n_pred))
    return {
        "micro": prf(sum(tp.values()), sum(n_gold.values()), sum(n_pred.values())),
        "per_label": {k: prf(tp[k], n_gold[k], n_pred[k]) for k in labels},
    }
