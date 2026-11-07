from datetime import date

import pytest

from helpers import complaint_line, recall_line, write_zip
from precursor import nhtsa


def parse(line: str, columns=nhtsa.COMPLAINT_COLUMNS, fn=nhtsa.parse_complaint) -> dict:
    return dict(zip(columns, fn(line.split("\t")), strict=True))


def test_parse_complaint_types_and_as_of_date():
    row = parse(complaint_line())
    assert row["odino"] == "11600001"
    assert row["received_date"] == date(2025, 1, 1)
    assert row["available_at"] == date(2025, 1, 2)  # DATEA later than LDATE wins
    assert row["incident_date"] == date(2024, 7, 1)
    assert row["crash"] is False and row["towed"] is True
    assert row["miles"] == 45000 and row["model_year"] == 2016
    assert row["state"] == "NV" and row["source_type"] == "IVOQ"


def test_parse_complaint_unknowns_and_dirty_text():
    row = parse(complaint_line(f6="9999", f8="00000000", f18="", f20="a\x00b ", f12="WHEELS:RIM "))
    assert row["model_year"] is None
    assert row["incident_date"] is None
    assert row["miles"] is None
    assert row["narrative"] == "ab"
    assert row["component"] == "WHEELS:RIM"


def test_parse_complaint_accepts_pre_2026_layout_with_49_fields():
    fields = complaint_line().split("\t")[:49]
    assert nhtsa.parse_complaint(fields)[0] == "11600001"


@pytest.mark.parametrize("line", [complaint_line(f17=""), complaint_line(f2=" "), "a\tb\tc"])
def test_parse_complaint_rejects_unusable_rows(line):
    with pytest.raises(ValueError):
        nhtsa.parse_complaint(line.split("\t"))


def test_parse_recall():
    row = parse(recall_line(f5="9999"), nhtsa.RECALL_COLUMNS, nhtsa.parse_recall)
    assert row["campno"] == "25V001000" and row["record_id"] == 1
    assert row["model_year"] is None
    assert row["report_received_date"] == date(2025, 2, 10)
    assert row["potentially_affected"] == 1200


def test_read_rows_splits_only_on_newline_and_reports_bad_lines(tmp_path):
    path = write_zip(tmp_path / "c.zip", [
        complaint_line(f20="first\rsecond\x0bthird"),
        "broken\trow",
        complaint_line(f2="11600002"),
    ])
    skipped: list[int] = []
    rows = list(nhtsa.read_rows(path, nhtsa.parse_complaint, skipped))
    assert [r[0] for r in rows] == ["11600001", "11600002"]
    assert rows[0][5] == "first\rsecond\x0bthird"
    assert skipped == [2]


def test_complaints_url_rejects_arbitrary_paths():
    assert nhtsa.complaints_url("2020-2024").endswith("COMPLAINTS_RECEIVED_2020-2024.zip")
    with pytest.raises(ValueError):
        nhtsa.complaints_url("../../evil")
