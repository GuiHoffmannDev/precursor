import json

import pytest

from precursor.label import MAX_ATTEMPTS, LabelError, label_one, parse_label, parse_recall_label

NARRATIVE = "WHILE DRIVING ON THE HIGHWAY THE STEERING WHEEL LOCKED UP. Dealer could not fix it."


def answer(**overrides) -> str:
    data = {
        "hazard": "loss_of_steering",
        "severity": "potential",
        "entities": [
            {"text": "steering wheel", "label": "component"},
            {"text": "locked up", "label": "failure_mode"},
            {"text": "on the highway", "label": "driving_condition"},
        ],
    }
    return json.dumps(data | overrides)


def test_parse_label_aligns_entities_case_insensitively():
    label = parse_label(f"```json\n{answer()}\n```", NARRATIVE)
    assert label["hazard"] == "loss_of_steering"
    assert [e["text"] for e in label["entities"]] == ["STEERING WHEEL", "LOCKED UP", "ON THE HIGHWAY"]
    first = label["entities"][0]
    assert NARRATIVE[first["start"]:first["end"]] == "STEERING WHEEL"
    assert label["dropped_entities"] == 0


def test_parse_label_drops_entities_not_in_text():
    label = parse_label(answer(entities=[{"text": "brake caliper", "label": "component"}]), NARRATIVE)
    assert label["entities"] == [] and label["dropped_entities"] == 1


@pytest.mark.parametrize("content", [
    "no json here",
    "{not json}",
    answer(hazard="steering failure"),
    answer(severity="high"),
    answer(entities=[{"text": "steering wheel", "label": "part"}]),
    answer(entities="steering wheel"),
])
def test_parse_label_rejects_contract_violations(content):
    with pytest.raises(LabelError):
        parse_label(content, NARRATIVE)


def test_label_one_repairs_once_then_gives_up():
    replies = iter(["bad", answer()])
    label, error, attempts = label_one(NARRATIVE, chat=lambda messages: next(replies))
    assert label["hazard"] == "loss_of_steering" and error is None and attempts == 2

    seen = []

    def always_bad(messages):
        seen.append(messages)
        return "still bad"

    label, error, attempts = label_one(NARRATIVE, chat=always_bad)
    assert label is None and "no JSON" in error and attempts == MAX_ATTEMPTS
    assert "Invalid answer" in seen[-1][-1]["content"]  # repair prompt shows the error


def test_parse_recall_label_accepts_an_optional_distinct_secondary_hazard():
    assert parse_recall_label('{"hazard": "fire_or_fuel_leak", "secondary_hazard": null}', "") == {
        "hazard": "fire_or_fuel_leak", "secondary_hazard": None}
    both = parse_recall_label('{"hazard": "loss_of_propulsion", "secondary_hazard": "fire_or_fuel_leak"}', "")
    assert both["secondary_hazard"] == "fire_or_fuel_leak"
    same = parse_recall_label('{"hazard": "fire_or_fuel_leak", "secondary_hazard": "fire_or_fuel_leak"}', "")
    assert same["secondary_hazard"] is None


@pytest.mark.parametrize("content", [
    "no json",
    '{"hazard": "engine fire"}',
    '{"hazard": "fire_or_fuel_leak", "secondary_hazard": "smoke"}',
    '["fire_or_fuel_leak"]',
])
def test_parse_recall_label_rejects_contract_violations(content):
    with pytest.raises(LabelError):
        parse_recall_label(content, "")
