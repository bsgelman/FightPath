"""Parse archive (Kaggle) data — ONLY for reach back-fill and DOB disambiguation.

Do NOT ingest career-aggregate stats (SLPM, STR_ACC, etc.) from here —
they are current-snapshot values and would leak future info.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ufc.io import paths
from ufc.ingest.parse_helpers import (
    parse_reach, parse_height, parse_dob, normalize_name, strip_ws_columns,
)


def _archive_dir() -> Path:
    return paths.archive_source()


def get_reach_backfill() -> pd.DataFrame:
    """Return (normalized_name, dob, reach_in) for fighters where reach is known."""
    csv = _archive_dir() / "fighter_details.csv"
    if not csv.exists():
        return pd.DataFrame(columns=["norm_name", "dob", "reach_in"])

    df = pd.read_csv(csv, dtype=str)
    df = strip_ws_columns(df)
    df.columns = [c.lower().strip() for c in df.columns]

    result = pd.DataFrame()
    result["norm_name"] = df["name"].apply(normalize_name)
    result["dob"] = df.get("dob", pd.Series(dtype=str)).apply(parse_dob)

    # Reach in archive is in cm — convert to inches
    if "reach" in df.columns:
        reach_cm = pd.to_numeric(df["reach"], errors="coerce")
        result["reach_in"] = (reach_cm / 2.54).round(1)
    else:
        result["reach_in"] = None

    # Height in cm -> inches
    if "height" in df.columns:
        height_cm = pd.to_numeric(df["height"], errors="coerce")
        result["height_in"] = (height_cm / 2.54).round(1)
    else:
        result["height_in"] = None

    result = result.dropna(subset=["norm_name"])
    result = result[result["reach_in"].notna() | result["height_in"].notna()]
    return result


def get_dob_lookup() -> pd.DataFrame:
    """Return (normalized_name, dob) for disambiguation."""
    csv = _archive_dir() / "fighter_details.csv"
    if not csv.exists():
        return pd.DataFrame(columns=["norm_name", "dob"])

    df = pd.read_csv(csv, dtype=str)
    df = strip_ws_columns(df)
    df.columns = [c.lower().strip() for c in df.columns]

    result = pd.DataFrame()
    result["norm_name"] = df["name"].apply(normalize_name)
    result["dob"] = df.get("dob", pd.Series(dtype=str)).apply(parse_dob)
    return result.dropna(subset=["norm_name", "dob"])
