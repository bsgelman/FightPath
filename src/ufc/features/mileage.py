"""Mileage / physical-decay features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from ufc.io import paths
from ufc.features.windows import (
    causal_expanding, causal_sum_expanding, causal_count,
    causal_rolling, causal_date_window, all_window_flavors,
)
from ufc.features.weight_class import _WC_WEIGHT_LBS, weight_class_lbs  # noqa: F401


def _cfg():
    with open(paths.root() / "configs" / "features.yaml") as f:
        return yaml.safe_load(f)


def compute_mileage(ledger: pd.DataFrame) -> pd.DataFrame:
    df = ledger.copy()
    cfg = _cfg()
    hl = cfg["windows"]["decay_halflife_days"]
    dw = cfg["damage_weights"]

    by = "fighter_id"
    sort_col = "event_rank"
    date_col = "event_date"

    # ── Age features ─────────────────────────────────────────────────────
    df["age_years_sq"] = df["age_years"] ** 2

    # ── Career fight count (prior to current fight) ───────────────────────
    df["fights_career"] = causal_count(df, by, sort_col)

    # ── Cumulative rounds and strikes absorbed ────────────────────────────
    df["total_rounds_career"] = causal_sum_expanding(df, by, sort_col, "end_round")
    df["total_sig_str_absorbed_career"] = causal_sum_expanding(
        df, by, sort_col, "sig_str_absorbed_landed"
    )

    # ── Damage index: recency-decayed head/body/leg absorbed ─────────────
    df["_damage_raw"] = (
        df["head_absorbed"].fillna(0) * dw["head"]
        + df["body_absorbed"].fillna(0) * dw["body"]
        + df["leg_absorbed"].fillna(0) * dw["leg"]
    )
    from ufc.features.windows import causal_decay
    df["damage_index"] = causal_decay(df, by, date_col, "_damage_raw", halflife_days=hl)

    # ── Layoff days ───────────────────────────────────────────────────────
    df["event_date_dt"] = pd.to_datetime(df[date_col])
    df["prev_fight_date"] = (
        df.sort_values([by, sort_col])
        .groupby(by)["event_date_dt"]
        .shift(1)
    )
    df["layoff_days"] = (
        (df["event_date_dt"] - df["prev_fight_date"])
        .dt.days
        .clip(lower=0)
    )
    df = df.drop(columns=["prev_fight_date", "event_date_dt"], errors="ignore")

    # ── Recent war flag (combined sig strikes > 300 OR high-pace 5-round fight) ──
    df["_total_sig_str_both"] = df["sig_str_landed"] + df["sig_str_absorbed_landed"]
    high_pace = (
        df["vol_attempted_pm_decay"] > 5.0
        if "vol_attempted_pm_decay" in df.columns
        else pd.Series(False, index=df.index)
    )
    df["_is_war"] = (
        (df["_total_sig_str_both"] > 300)
        | ((df["scheduled_rounds"] >= 5) & high_pace)
    ).astype(float)
    df["recent_war_flag"] = causal_date_window(
        df, by, date_col, "_is_war", window_days=180, agg="max"
    ).fillna(0).astype(int)

    # ── Defense trend (slope of str_def over last 3 fights) ─────────────
    if "str_def_decay" in df.columns:
        df["_str_def_for_trend"] = df["str_def_decay"]
        df["def_pct_trend_l3"] = _compute_slope(df, by, sort_col, "_str_def_for_trend", 3)

    # ── Volume trend (slope of slpm over last 3 fights) ──────────────────
    if "slpm_decay" in df.columns:
        df["volume_trend_l3"] = _compute_slope(df, by, sort_col, "slpm_decay", 3)

    # ── Cardio score (late-round volume retention, causal decay) ─────────
    # cardio_ratio_fight = R4+R5 avg strikes / R1+R2 avg strikes per fight
    # Only populated for fights that reached R4+; NaN otherwise (correctly excluded)
    if "cardio_ratio_fight" in df.columns:
        df["cardio_score"] = causal_decay(
            df, by, date_col, "cardio_ratio_fight", halflife_days=hl
        )
    else:
        df["cardio_score"] = np.nan

    # ── Weight-class change (lbs): current WC minus fighter's previous WC ────
    # Positive = moving up (e.g. LW→WW = +15), negative = cutting down.
    # NaN for a fighter's first recorded fight (no prior WC).
    if "weight_class" in df.columns:
        df["_wc_lbs"] = df["weight_class"].map(weight_class_lbs)
        df["_prev_wc_lbs"] = (
            df.sort_values([by, sort_col])
            .groupby(by, sort=False)["_wc_lbs"]
            .shift(1)
        )
        df["weight_class_change_lbs"] = df["_wc_lbs"] - df["_prev_wc_lbs"]
    else:
        df["weight_class_change_lbs"] = np.nan

    # ── Layoff × age interaction: ring rust hurts older fighters more ──────
    if "age_years" in df.columns:
        df["layoff_age_interaction"] = df["layoff_days"] * df["age_years"]
    else:
        df["layoff_age_interaction"] = np.nan

    # Drop intermediate columns AND the raw cardio_ratio_fight column.
    # cardio_ratio_fight is computed from the CURRENT fight's R4+R5/R1+R2 strike ratio
    # and would leak the fight outcome (KO/SUB never reach R4+ → fillna(0)=label proxy).
    # cardio_score (the causal-decay version) is the safe replacement.
    drop_cols = [c for c in df.columns if c.startswith("_")]
    drop_cols.append("cardio_ratio_fight")
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def _compute_slope(df: pd.DataFrame, by: str, sort_col: str, value_col: str, window: int) -> pd.Series:
    """OLS slope of value over last `window` prior fights, per fighter."""
    sorted_df = df.sort_values([by, sort_col]).copy()
    result = pd.Series(np.nan, index=sorted_df.index)

    for fighter, grp in sorted_df.groupby(by, sort=False):
        vals = grp[value_col].values
        indices = np.arange(len(grp))
        slopes = np.full(len(grp), np.nan)

        for i in range(window, len(grp)):
            # Use prior window fights (not including current)
            y = vals[max(0, i-window):i]
            if len(y) < 2:
                continue
            x = np.arange(len(y), dtype=float)
            xm, ym = x.mean(), y.mean()
            denom = ((x - xm)**2).sum()
            if denom == 0:
                continue
            slopes[i] = ((x - xm) * (y - ym)).sum() / denom

        result.loc[grp.index] = slopes

    return result.reindex(df.index)
