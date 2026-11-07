"""Download and parse NHTSA ODI flat files.

Field layouts: https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt and
https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt (tab-delimited, UTF-8, YYYYMMDD dates).
"""

import re
import ssl
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path

import certifi

BASE_URL = "https://static.nhtsa.gov/odi/ffdd"
RECALLS_URL = f"{BASE_URL}/rcl/FLAT_RCL_POST_2010.zip"
USER_AGENT = "precursor/0.1 (research project)"

# Column order of the tuples produced by parse_complaint / parse_recall.
COMPLAINT_COLUMNS = (
    "odino", "cmplid", "received_date", "available_at", "incident_date", "narrative",
    "crash", "fire", "injured", "deaths", "medical_attn", "towed", "police_report",
    "miles", "vehicle_speed", "state", "source_type",
    "mfr_name", "make", "model", "model_year", "component", "product_type",
)
RECALL_COLUMNS = (
    "campno", "record_id", "make", "model", "model_year", "component", "mfr_name",
    "report_received_date", "owner_notification_date", "manufacture_begin",
    "manufacture_end", "recall_type", "influenced_by", "potentially_affected",
    "defect_summary", "consequence_summary", "corrective_action", "notes",
)


def complaints_url(period: str) -> str:
    """URL of one 5-year complaints chunk, e.g. period='2020-2024'."""
    if not re.fullmatch(r"\d{4}-\d{4}", period):
        raise ValueError(f"period must look like 2020-2024, got {period!r}")
    return f"{BASE_URL}/cmpl/COMPLAINTS_RECEIVED_{period}.zip"


def download(url: str, dest_dir: Path, refresh: bool = False) -> Path:
    """Download url into dest_dir, reusing an existing copy unless refresh=True."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / url.rsplit("/", 1)[1]
    if dest.exists() and not refresh:
        return dest
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    # certifi: python.org macOS builds ship without a CA store by default.
    tls = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=60, context=tls) as resp, part.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    with zipfile.ZipFile(part) as zf:  # fail now on a truncated download
        if zf.testzip() is not None:
            raise ValueError(f"corrupt zip downloaded from {url}")
    part.replace(dest)
    return dest


def read_rows(
    zip_path: Path, parse: Callable[[list[str]], tuple], skipped: list[int]
) -> Iterator[tuple]:
    """Yield parsed rows of the single file inside zip_path.

    Lines that fail to parse are skipped and their 1-based line numbers appended
    to `skipped`, so one malformed record does not abort a multi-million-row load.
    """
    with zipfile.ZipFile(zip_path) as zf:
        (name,) = zf.namelist()
        with zf.open(name) as fh:
            # Binary iteration splits on b"\n" only: narratives contain \r, \x0b, \x1c.
            for line_no, raw in enumerate(fh, 1):
                line = raw.rstrip(b"\r\n").decode("utf-8")
                if not line:
                    continue
                try:
                    yield parse(line.split("\t"))
                except ValueError:
                    skipped.append(line_no)


def parse_complaint(f: list[str]) -> tuple:
    if len(f) < 49:  # fields 50-51 only exist in files written after 2026-04-30
        raise ValueError(f"expected >= 49 fields, got {len(f)}")
    odino = f[1].strip()
    received = _date(f[16])
    if not odino or received is None:
        raise ValueError("missing ODINO or LDATE")
    added = _date(f[15])
    return (
        odino,
        f[0].strip(),
        received,
        max(received, added) if added else received,
        _date(f[7]),
        _text(f[19]) or "",
        _yn(f[6]), _yn(f[8]), _int(f[9]), _int(f[10]),
        _yn(f[47]), _yn(f[48]), _yn(f[21]),
        _int(f[17]), _int(f[31]),
        _text(f[13]), _text(f[20]),
        _text(f[2]), _text(f[3]), _text(f[4]), _year(f[5]), _text(f[11]), _text(f[45]),
    )


def parse_recall(f: list[str]) -> tuple:
    if len(f) < 23:
        raise ValueError(f"expected >= 23 fields, got {len(f)}")
    campno = f[1].strip()
    record_id = _int(f[0])
    if not campno or record_id is None:
        raise ValueError("missing CAMPNO or RECORD_ID")
    return (
        campno, record_id,
        _text(f[2]), _text(f[3]), _year(f[4]), _text(f[6]), _text(f[7]),
        _date(f[15]), _date(f[12]), _date(f[8]), _date(f[9]),
        _text(f[10]), _text(f[13]), _int(f[11]),
        _text(f[19]), _text(f[20]), _text(f[21]), _text(f[22]),
    )


def _text(s: str) -> str | None:
    s = s.replace("\x00", "").strip()  # Postgres text cannot hold NUL
    return s or None


def _int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


def _yn(s: str) -> bool | None:
    return {"Y": True, "N": False}.get(s.strip().upper())


def _date(s: str) -> date | None:
    s = s.strip()
    if len(s) != 8 or not s.isdigit():
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:]))
    except ValueError:  # e.g. 00000000
        return None


def _year(s: str) -> int | None:
    y = _int(s)
    return y if y is not None and 1900 <= y <= 2100 else None  # 9999 means unknown
