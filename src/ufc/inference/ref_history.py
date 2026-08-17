"""Per-referee stoppage tendency table for inference-time feature activation."""
from __future__ import annotations

import pandas as pd

_CACHE: pd.DataFrame | None = None


def build_ref_history() -> pd.DataFrame:
    """Return [referee, referee_stoppage_threshold, _ref_norm] — each referee's most
    recent causal stoppage threshold, derived from features_props.parquet."""
    from ufc.io import paths, parquet
    fp = parquet.read(paths.processed("features_props"))
    sub = fp[["referee", "event_date", "referee_stoppage_threshold"]].copy()
    sub = sub[sub["referee"].astype(str).str.strip() != ""].dropna(subset=["referee"])
    sub["event_date"] = pd.to_datetime(sub["event_date"])
    tbl = (
        sub.sort_values("event_date")
           .groupby("referee", as_index=False)
           .last()[["referee", "referee_stoppage_threshold"]]
    )
    tbl["_ref_norm"] = tbl["referee"].astype(str).str.strip().str.lower()
    return tbl.reset_index(drop=True)


def get_ref_history() -> pd.DataFrame:
    """Process-level memoized accessor."""
    global _CACHE
    if _CACHE is None:
        _CACHE = build_ref_history()
    return _CACHE


def known_referee_names() -> list[str]:
    """Sorted list of referee canonical names for UI dropdowns."""
    return sorted(get_ref_history()["referee"].tolist())
