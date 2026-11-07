import zipfile
from pathlib import Path


def complaint_line(**overrides: str) -> str:
    """A 51-field complaint record. Keys are 1-based NHTSA field numbers, e.g. f20='text'."""
    f = [""] * 51
    defaults = {
        1: "100", 2: "11600001", 3: "Ford Motor Company", 4: "FORD", 5: "EXPLORER", 6: "2016",
        7: "N", 8: "20240701", 9: "N", 10: "0", 11: "0", 12: "STEERING", 14: "NV",
        16: "20250102", 17: "20250101", 18: "45000", 20: "Steering wheel locks up.", 21: "IVOQ",
        22: "N", 32: "35", 46: "V", 48: "N", 49: "Y",
    }
    for k, v in defaults.items():
        f[k - 1] = v
    for k, v in overrides.items():
        f[int(k[1:]) - 1] = v
    return "\t".join(f)


def recall_line(**overrides: str) -> str:
    """A 29-field recall record. Keys are 1-based NHTSA field numbers."""
    f = [""] * 29
    defaults = {
        1: "1", 2: "25V001000", 3: "FORD", 4: "EXPLORER", 5: "2016", 7: "STEERING",
        8: "Ford Motor Company", 11: "V", 12: "1200", 13: "20250301", 14: "MFR",
        16: "20250210", 20: "Steering column may lock.", 21: "Increased crash risk.",
        22: "Dealers will replace the column.", 23: "",
    }
    for k, v in defaults.items():
        f[k - 1] = v
    for k, v in overrides.items():
        f[int(k[1:]) - 1] = v
    return "\t".join(f)


def write_zip(path: Path, lines: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(path.stem + ".txt", "\r\n".join(lines) + "\r\n")
    return path
