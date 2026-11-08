import csv

import pytest

from precursor.gold import COLUMNS, read_sheet


def write_sheet(path, rows, bom=False):
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: "" for c in COLUMNS} | r)
    return path


def test_read_sheet_skips_blank_rows_and_handles_excel_bom(tmp_path):
    path = write_sheet(tmp_path / "s.csv", [
        {"odino": "1", "narrative": 'multi\nline, "quoted"', "gold_hazard": "loss_of_steering",
         "gold_severity": "potential", "notes": " ok "},
        {"odino": "2"},
    ], bom=True)
    assert read_sheet(path) == [("1", "loss_of_steering", "potential", "ok")]


def test_read_sheet_rejects_whole_file_on_any_invalid_value(tmp_path):
    path = write_sheet(tmp_path / "s.csv", [
        {"odino": "1", "gold_hazard": "loss_of_steering", "gold_severity": "potential"},
        {"odino": "2", "gold_hazard": "steering", "gold_severity": "potential"},
        {"odino": "3", "gold_hazard": "fire_or_fuel_leak", "gold_severity": ""},
    ])
    with pytest.raises(ValueError) as e:
        read_sheet(path)
    assert "line 3" in str(e.value) and "line 4" in str(e.value)
