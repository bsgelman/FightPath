"""Pre-UFC record features for debut/early-career prior seeding.

Derives pre-UFC wins/losses from ufc_fighter_career_stats.csv (all-promotions)
minus the fighter's UFC ledger record, then applies shrinkage toward 0.5.

No new scraping needed — ufc_fighter_career_stats.csv already has wins_total/losses_total.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ufc.io import paths


def compute_pre_ufc_features(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame indexed by fighter_id with pre-UFC record features.

    Columns:
      pre_ufc_wins, pre_ufc_losses, pre_ufc_n,
      pre_ufc_win_rate_shrunk  (shrunk toward 0.5 by k=5)
    """
    csv_path = paths.raw_scraper() / "ufc_fighter_career_stats.csv"
    career = pd.read_csv(csv_path, dtype=str)

    # Normalize fighter_id column name
    id_col = "fighter_id" if "fighter_id" in career.columns else career.columns[0]
    career = career.rename(columns={id_col: "fighter_id"})

    # wins_total / losses_total = all-promotions record
    for col in ("wins_total", "losses_total"):
        career[col] = pd.to_numeric(career.get(col, 0), errors="coerce").fillna(0).astype(int)

    career = career.set_index("fighter_id")[["wins_total", "losses_total"]]

    # UFC W/L from ledger
    ufc_record = (
        ledger.groupby("fighter_id")["won"]
        .agg(ufc_wins=lambda x: (x == 1).sum(), ufc_fights=lambda x: x.notna().sum())
        .assign(ufc_losses=lambda d: d["ufc_fights"] - d["ufc_wins"])
    )

    merged = career.join(ufc_record, how="left").fillna(0)
    merged["pre_ufc_wins"] = (merged["wins_total"] - merged["ufc_wins"]).clip(lower=0).astype(int)
    merged["pre_ufc_losses"] = (merged["losses_total"] - merged["ufc_losses"]).clip(lower=0).astype(int)
    merged["pre_ufc_n"] = merged["pre_ufc_wins"] + merged["pre_ufc_losses"]

    k = 5.0
    merged["pre_ufc_win_rate_shrunk"] = (
        (merged["pre_ufc_wins"] + k / 2) / (merged["pre_ufc_n"] + k)
    )

    return merged[["pre_ufc_wins", "pre_ufc_losses", "pre_ufc_n", "pre_ufc_win_rate_shrunk"]]


def get_pre_ufc_lookup(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return pre-UFC features DataFrame (indexed by fighter_id), with NaN fill."""
    try:
        lu = compute_pre_ufc_features(ledger)
    except Exception:
        return pd.DataFrame(columns=["pre_ufc_n", "pre_ufc_win_rate_shrunk"])
    return lu


def seeded_elo_offset(win_rate_shrunk: float, pre_ufc_n: int,
                      k: float = 100.0, cap: float = 120.0) -> float:
    """Elo offset based on pre-UFC record: k * (p - 0.5) * log1p(n), clamped ±cap."""
    return float(np.clip(k * (win_rate_shrunk - 0.5) * np.log1p(pre_ufc_n), -cap, cap))
