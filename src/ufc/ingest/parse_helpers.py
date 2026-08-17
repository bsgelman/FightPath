"""Pure parsing functions for raw UFC data strings."""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime


def parse_x_of_y(s: str | float) -> tuple[int, int]:
    """'9 of 12' -> (9, 12). '---' or NaN -> (0, 0)."""
    if not isinstance(s, str):
        return (0, 0)
    s = s.strip()
    if s in ("---", "--", "", "N/A"):
        return (0, 0)
    m = re.match(r"(\d+)\s+of\s+(\d+)", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # Sometimes it's just a number (e.g. accuracy pct stored differently)
    try:
        v = int(s)
        return (v, v)
    except ValueError:
        return (0, 0)


def parse_mm_ss(s: str | float) -> int:
    """'1:44' -> 104 seconds. '--' or NaN -> 0."""
    if not isinstance(s, str):
        return 0
    s = s.strip()
    if s in ("--", "", "---"):
        return 0
    m = re.match(r"(\d+):(\d+)", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return 0


def parse_height(s: str | float) -> float | None:
    """'5\' 11\"' -> 71.0 inches. '--' -> None."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("--", "", "N/A"):
        return None
    m = re.match(r"""(\d+)'\s*(\d+)""", s)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2))
    return None


def parse_weight(s: str | float) -> float | None:
    """'155 lbs.' -> 155.0. '--' -> None."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("--", "", "N/A"):
        return None
    m = re.match(r"([\d.]+)\s*lbs?\.?", s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def parse_reach(s: str | float) -> float | None:
    """'66\"' -> 66.0 inches. '--' -> None."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("--", "", "N/A"):
        return None
    m = re.match(r'([\d.]+)"?', s)
    if m:
        return float(m.group(1))
    return None


def parse_dob(s: str | float) -> date | None:
    """'Jul 13, 1978' or 'Jul 13 1978' -> date. '--' -> None."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("--", "", "N/A"):
        return None
    for fmt in ("%b %d, %Y", "%b %d %Y", "%B %d, %Y", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_event_date(s: str | float) -> date | None:
    """Handles 'September 06, 2025', '2025/09/06', 'Sep 6, 2025'."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y/%m/%d", "%Y-%m-%d", "%b. %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_pct(s: str | float) -> float | None:
    """'100%' -> 1.0. '---' -> None."""
    if not isinstance(s, str):
        if isinstance(s, (int, float)):
            v = float(s)
            # stored as integer pct in some sources
            return v / 100.0 if v > 1.0 else v
        return None
    s = s.strip()
    if s in ("---", "--", "", "N/A"):
        return None
    m = re.match(r"([\d.]+)%?", s)
    if m:
        v = float(m.group(1))
        return v / 100.0 if v > 1.0 else v
    return None


def parse_scheduled_rounds(time_format: str | float) -> int:
    """'3 Rnd (5-5-5)' -> 3. '5 Rnd (5-5-5-5-5)' -> 5."""
    if not isinstance(time_format, str):
        return 3
    m = re.match(r"(\d+)\s+Rnd", time_format.strip())
    if m:
        return int(m.group(1))
    return 3


def normalize_name(name: str) -> str:
    """Lowercase, ASCII-fold, strip punctuation, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def strip_ws_columns(df):
    """Strip leading/trailing whitespace from all string columns."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def extract_url_hex(url: str | float) -> str | None:
    """'http://ufcstats.com/fighter-details/abc123' -> 'abc123'."""
    if not isinstance(url, str):
        return None
    m = re.search(r"/([0-9a-zA-Z]+)\s*$", url.strip())
    return m.group(1) if m else None


def normalize_method(method: str | float) -> str:
    """Normalize method strings to canonical categories."""
    if not isinstance(method, str):
        return "NC"
    m = method.strip().upper()
    if "KO" in m or "TKO" in m or "DOCTOR" in m or "COULD NOT CONTINUE" in m:
        return "KO/TKO"
    if "SUB" in m:
        return "SUB"
    if "UNANIMOUS" in m:
        return "U-DEC"
    if "SPLIT" in m:
        return "S-DEC"
    if "MAJORITY" in m:
        return "M-DEC"
    if "DQ" in m or "DISQUALIF" in m:
        return "DQ"
    if "OVERTURNED" in m or "NC" in m or "NO CONTEST" in m:
        return "NC"
    if "DECISION" in m:
        return "U-DEC"
    return "NC"


def normalize_stance(stance: str | float) -> str:
    """Normalize stance to canonical categories."""
    if not isinstance(stance, str):
        return "UNKNOWN"
    s = stance.strip().upper()
    if "ORTHODOX" in s:
        return "ORTHO"
    if "SOUTHPAW" in s:
        return "SOUTH"
    if "SWITCH" in s:
        return "SWITCH"
    if "OPEN" in s:
        return "OPEN"
    return "UNKNOWN"
