"""Per-round granularity features from fight_rounds.parquet.

Extracts R1-R5 striking/activity metrics per fighter-fight, then rolls
them into standard _ctd/_l3/_l5/_2y/_decay flavors (shift(1) ensures
no current-fight leakage).

R4/R5 data is NaN for fights that ended earlier — correctly excluded
from rolling means (informative absence, not zero).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from ufc.io import paths
from ufc.features.windows import all_window_flavors


def _cfg():
    with open(paths.root() / "configs" / "features.yaml") as f:
        return yaml.safe_load(f)


def _resolve_fight_ids(
    rounds_df: pd.DataFrame,
    fights_df: pd.DataFrame,
    fighters_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join fight_rounds → fight_id + fighter_id via event_name + fighter_name."""
    fights_a = fights_df[["fight_id", "event_name", "fighter_a_name"]].rename(
        columns={"fighter_a_name": "fighter_name"}
    )
    fights_b = fights_df[["fight_id", "event_name", "fighter_b_name"]].rename(
        columns={"fighter_b_name": "fighter_name"}
    )
    fights_long = pd.concat([fights_a, fights_b], ignore_index=True)

    name_to_id = dict(zip(fighters_df["fighter_name"], fighters_df["fighter_id"]))

    resolved = rounds_df.merge(fights_long, on=["event_name", "fighter_name"], how="left")
    resolved["fighter_id"] = resolved["fighter_name"].map(name_to_id)
    return resolved.dropna(subset=["fight_id", "fighter_id"])


def join_round_raw(
    ledger: pd.DataFrame,
    fights_df: pd.DataFrame,
    fighters_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add raw per-round columns to the ledger before rolling.

    Adds columns: r{1-5}_sig_str_raw, r{1-5}_kd_raw, r1_td_raw, r1_sub_att_raw,
    r1_ctrl_sec_raw, pace_r1r3_raw, pace_r3r5_raw, early_vol_ratio_raw.

    These are per-fight actuals. Rolling is applied later in compute_round_features.
    """
    resolved = _resolve_fight_ids(rounds_df, fights_df, fighters_df)

    # Pivot: one row per (fight_id, fighter_id) with columns for each round
    metrics = {
        "SIGSTR_landed": "sig_str",
        "kd": "kd",
        "TD_landed": "td",
        "sub_att": "sub_att",
        "ctrl_sec": "ctrl_sec",
    }

    pivot_frames = []
    for src_col, short in metrics.items():
        piv = (
            resolved[resolved[src_col].notna()]
            .pivot_table(
                index=["fight_id", "fighter_id"],
                columns="round_num",
                values=src_col,
                aggfunc="sum",
            )
            .rename(columns={r: f"r{r}_{short}_raw" for r in range(1, 6)})
        )
        # Only keep rounds 1-5 columns that exist
        piv = piv[[c for c in piv.columns if c in [f"r{r}_{short}_raw" for r in range(1, 6)]]]
        pivot_frames.append(piv)

    if not pivot_frames:
        return ledger

    per_fight = pd.concat(pivot_frames, axis=1).reset_index()

    # Pace-decline features per fight (derived; NaN when fight didn't reach required round)
    def _col(df, name):
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    r1 = _col(per_fight, "r1_sig_str_raw").fillna(0)
    r3 = _col(per_fight, "r3_sig_str_raw")
    r5 = _col(per_fight, "r5_sig_str_raw")
    total = sum(
        _col(per_fight, f"r{r}_sig_str_raw").fillna(0) for r in range(1, 6)
    )

    per_fight["pace_r1r3_raw"] = np.where(
        r3.notna(), (r1 - r3.fillna(0)) / r1.clip(lower=1), np.nan
    )
    per_fight["pace_r3r5_raw"] = np.where(
        r5.notna(), (r3.fillna(0) - r5.fillna(0)) / r3.fillna(0).clip(lower=1), np.nan
    )
    per_fight["early_vol_ratio_raw"] = np.where(
        total > 0, r1 / total.clip(lower=1), np.nan
    )

    # Keep only R1-5 sig_str, R1 activity specifics, and pace
    keep_cols = (
        [f"r{r}_sig_str_raw" for r in range(1, 6)]
        + ["r1_kd_raw", "r1_td_raw", "r1_sub_att_raw", "r1_ctrl_sec_raw"]
        + ["pace_r1r3_raw", "pace_r3r5_raw", "early_vol_ratio_raw"]
    )
    keep_cols = [c for c in keep_cols if c in per_fight.columns]
    per_fight = per_fight[["fight_id", "fighter_id"] + keep_cols]

    out = ledger.merge(per_fight, on=["fight_id", "fighter_id"], how="left")
    return out


def compute_round_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rolling windows to raw per-round columns already in df.

    Expects join_round_raw to have been called first (columns ending in _raw).
    After rolling, drops the _raw columns (they'd leak into the wide format).
    """
    cfg = _cfg()
    hl = cfg["windows"]["decay_halflife_days"]
    w2y = cfg["windows"]["date_months"] * 30
    by, sort_col, date_col = "fighter_id", "event_rank", "event_date"

    raw_cols = [c for c in df.columns if c.endswith("_raw") and not c.startswith("_")]
    if not raw_cols:
        return df

    new_cols = {}
    for raw_col in raw_cols:
        prefix = raw_col[:-4]  # strip "_raw"
        flavors = all_window_flavors(
            df, by, sort_col, date_col, raw_col,
            halflife_days=hl,
            window_24mo_days=w2y,
            prefix=prefix,
        )
        new_cols.update(flavors)

    df = df.drop(columns=raw_cols, errors="ignore")
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df
