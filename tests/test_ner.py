from precursor.ner import LABEL_PROMPTS, normalize, span_scores, to_training_example


def test_training_example_maps_char_spans_to_inclusive_word_indices():
    text = "THE STEERING WHEEL LOCKED UP, ON THE HIGHWAY."
    ex = to_training_example(text, [
        {"start": 4, "end": 18, "label": "component"},       # STEERING WHEEL
        {"start": 19, "end": 28, "label": "failure_mode"},   # LOCKED UP
        {"start": 30, "end": 44, "label": "driving_condition"},  # ON THE HIGHWAY
    ])
    assert ex["tokenized_text"][:6] == ["the", "steering", "wheel", "locked", "up", ","]
    assert ex["ner"] == [
        [1, 2, LABEL_PROMPTS["component"]],
        [3, 4, LABEL_PROMPTS["failure_mode"]],
        [6, 8, LABEL_PROMPTS["driving_condition"]],
    ]


def span(start: int, end: int, label: str = "component") -> dict:
    return {"start": start, "end": end, "label": label}


def test_exact_matching_counts_each_gold_span_once():
    gold = [[span(0, 5), span(10, 15, "failure_mode")]]
    pred = [[span(0, 5), span(0, 5), span(10, 14, "failure_mode")]]
    scores = span_scores(gold, pred)
    assert scores["micro"] == {"precision": 0.3333, "recall": 0.5, "f1": 0.4}
    assert scores["per_label"]["component"]["recall"] == 1.0


def test_overlap_matching_requires_same_label():
    gold = [[span(10, 15, "failure_mode")]]
    assert span_scores(gold, [[span(12, 20, "failure_mode")]], overlap=True)["micro"]["f1"] == 1.0
    assert span_scores(gold, [[span(12, 20, "component")]], overlap=True)["micro"]["f1"] == 0.0


def test_empty_predictions_score_zero_without_dividing_by_zero():
    assert span_scores([[span(0, 1)]], [[]])["micro"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_normalize_keeps_offsets():
    assert normalize("BRAKES FAILED") == "brakes failed"
    assert normalize("İ") == "İ"  # lower() would change the length, so keep the original
