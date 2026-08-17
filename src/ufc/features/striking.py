"""Striking profile features — all in 5 window flavors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from ufc.io import paths
from ufc.features.windows import all_window_flavors, causal_expanding, causal_sum_expanding


def _cfg():
    with open(paths.root() / "configs" / "features.yaml") as f:
        return yaml.safe_load(f)


def compute_striking(ledger: pd.DataFrame) -> pd.DataFrame:
    """Add striking features to ledger. Returns new DataFrame with added columns."""
    df = ledger.copy()
    # pd.NA in nullable-int columns breaks np.where boolean checks; cast to float
    for col in df.select_dtypes(include=["Int8","Int16","Int32","Int64","UInt8","UInt16","UInt32","UInt64"]).columns:
        df[col] = df[col].astype("float64")
    cfg = _cfg()
    hl = cfg["windows"]["decay_halflife_days"]
    w2y = cfg["windows"]["date_months"] * 30

    by = "fighter_id"
    sort_col = "event_rank"
    date_col = "event_date"

    # Compute fight minutes for rate calculations
    df["fight_min"] = df["total_fight_sec"].fillna(0) / 60.0

    # ── Per-fight rates (used as raw values going into rolling) ─────────────
    # These are per-fight values; we then aggregate across fights

    # Strikes per minute (sig)
    df["_slpm_fight"] = np.where(
        df["fight_min"] > 0,
        df["sig_str_landed"] / df["fight_min"],
        np.nan,
    )
    # Absorbed per minute
    df["_sapm_fight"] = np.where(
        df["fight_min"] > 0,
        df["sig_str_absorbed_landed"] / df["fight_min"],
        np.nan,
    )
    # Accuracy
    df["_str_acc_fight"] = np.where(
        df["sig_str_attempted"] > 0,
        df["sig_str_landed"] / df["sig_str_attempted"],
        np.nan,
    )
    # Defense (1 - opponent accuracy against you)
    df["_str_def_fight"] = np.where(
        df["sig_str_absorbed_attempted"] > 0,
        1.0 - df["sig_str_absorbed_landed"] / df["sig_str_absorbed_attempted"],
        np.nan,
    )

    # Location shares
    for loc in ["head", "body", "leg", "distance", "clinch", "ground"]:
        df[f"_{loc}_share_fight"] = np.where(
            df["sig_str_landed"] > 0,
            df[f"{loc}_landed"] / df["sig_str_landed"],
            np.nan,
        )
        df[f"_{loc}_acc_fight"] = np.where(
            df.get(f"{loc}_attempted", pd.Series(0, index=df.index)) > 0,
            df[f"{loc}_landed"] / df[f"{loc}_attempted"],
            np.nan,
        )

    # Knockdown rate per 15 min
    df["_kd_per_15_fight"] = np.where(
        df["fight_min"] > 0,
        df["kd_for"] / df["fight_min"] * 15,
        np.nan,
    )
    # KD per sig strike landed (power proxy)
    df["_kd_per_sig_fight"] = np.where(
        df["sig_str_landed"] > 0,
        df["kd_for"] / df["sig_str_landed"],
        np.nan,
    )

    # Absorbed knockdowns per 15 min (chin durability inverse)
    df["_kd_against_per_15_fight"] = np.where(
        df["fight_min"] > 0,
        df["kd_against"].fillna(0) / df["fight_min"] * 15,
        np.nan,
    )

    # Total strikes attempted per minute (volume)
    df["_vol_attempted_pm_fight"] = np.where(
        df["fight_min"] > 0,
        df["sig_str_attempted"] / df["fight_min"],
        np.nan,
    )

    # ── Rolling windows for each rate metric ─────────────────────────────────
    raw_metrics = [
        ("_slpm_fight", "slpm"),
        ("_sapm_fight", "sapm"),
        ("_str_acc_fight", "str_acc"),
        ("_str_def_fight", "str_def"),
        ("_head_share_fight", "head_share"),
        ("_body_share_fight", "body_share"),
        ("_leg_share_fight", "leg_share"),
        ("_distance_share_fight", "distance_share"),
        ("_clinch_share_fight", "clinch_share"),
        ("_ground_share_fight", "ground_share"),
        ("_head_acc_fight", "head_acc"),
        ("_body_acc_fight", "body_acc"),
        ("_leg_acc_fight", "leg_acc"),
        ("_kd_per_15_fight", "kd_per_15"),
        ("_kd_per_sig_fight", "kd_per_sig"),
        ("_kd_against_per_15_fight", "kd_against_per_15"),
        ("_vol_attempted_pm_fight", "vol_attempted_pm"),
    ]

    new_cols = {}
    for raw_col, feat_name in raw_metrics:
        if raw_col not in df.columns:
            continue
        flavors = all_window_flavors(
            df, by, sort_col, date_col, raw_col,
            halflife_days=hl,
            window_24mo_days=w2y,
            prefix=feat_name,
        )
        new_cols.update(flavors)

    # Drop raw intermediate columns, then concat all new feature columns at once
    raw_cols = [c for c in df.columns if c.startswith("_") and c.endswith("_fight")]
    df = df.drop(columns=raw_cols, errors="ignore")
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df
