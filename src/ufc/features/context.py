"""Contextual features: title, rounds, referee tendency, altitude."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from ufc.io import paths
from ufc.features.windows import causal_date_window


def _altitude_lookup() -> dict[str, float]:
    with open(paths.root() / "configs" / "features.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("altitude_lookup", {})


def compute_context(ledger: pd.DataFrame) -> pd.DataFrame:
    """Add context features to the per-fighter ledger."""
    df = ledger.copy()

    df["event_date_sort"] = pd.to_datetime(df["event_date"])
    # Main event = the fight with the lexicographically-largest fight_id per event_id.
    # UFC's scraper assigns fight_id in card-order top→bottom, so max==main.
    main_fid_per_event = df.groupby("event_id")["fight_id"].transform("max")
    df["is_main_event"] = (df["fight_id"] == main_fid_per_event)

    # ── Altitude ──────────────────────────────────────────────────────────
    alt_lookup = _altitude_lookup()
    def _alt(loc):
        if not isinstance(loc, str):
            return 0.0
        for city, alt in alt_lookup.items():
            if city.lower() in loc.lower():
                return float(alt)
        return 0.0

    df["altitude_meters"] = df["location"].apply(_alt)

    # ── Referee stoppage tendency ─────────────────────────────────────────
    # Early stoppage = fight ended before scheduled time (not a decision)
    df["_is_early_finish"] = (
        (df["method"] == "KO/TKO") | (df["method"] == "SUB")
    ).astype(float)

    # Need per-fight referee rows, so compute on fight_id level
    ref_per_fight = (
        df.drop_duplicates("fight_id")[["fight_id", "event_date", "referee", "_is_early_finish"]]
        .rename(columns={"_is_early_finish": "ref_early"})
    )
    ref_per_fight["ref_early_ctd"] = causal_date_window(
        ref_per_fight,
        by="referee",
        date_col="event_date",
        value_col="ref_early",
        window_days=365 * 3,
        agg="mean",
    )
    df = pd.merge(
        df,
        ref_per_fight[["fight_id", "ref_early_ctd"]].rename(
            columns={"ref_early_ctd": "referee_stoppage_threshold"}
        ),
        on="fight_id",
        how="left",
    )

    df = df.drop(columns=["_is_early_finish", "event_date_sort"], errors="ignore")
    return df


def compute_home_advantage(ledger: pd.DataFrame, nationality_lookup: dict | None = None) -> pd.DataFrame:
    """Add is_home_country flag. Defaults to 0 — hook for future enrichment."""
    ledger = ledger.copy()
    ledger["is_home_country"] = 0
    return ledger


def compute_era_baselines(ledger: pd.DataFrame) -> pd.DataFrame:
    """Add fight-level era and weight-class baselines (all causal).

    - era_avg_sig_str_l12mo: global UFC mean sig_str_landed per fighter-fight
      over the prior 365 days. Gives the model an exogenous "era pace" knob
      to lift modern-era count predictions.
    - wc_finish_share_l2y: per-weight-class fraction of finishes (KO+SUB)
      over the prior 730 days. LW is decision-heavy, HW is KO-heavy.
    - wc_5rd_dec_rate: per-weight-class fraction of 5-round fights ending in
      DEC over the prior 730 days. Replaces the lost cardio leak signal for
      duration model.
    """
    df = ledger.copy()
    df["event_date_dt"] = pd.to_datetime(df["event_date"])

    # ── Global era pace (one number per event_date) ──────────────────────
    # Aggregate to event-date level first to avoid double-counting fighter-fights.
    by_date = (
        df.dropna(subset=["sig_str_landed"])
        .groupby("event_date_dt")["sig_str_landed"].mean()
        .sort_index()
    )
    # 365-day rolling mean, strictly prior (shift to exclude same-day events).
    era = by_date.shift(1).rolling("365D", min_periods=20).mean()
    # Build a lookup series indexed by date, then map onto df
    era_expanding = by_date.expanding().mean().shift(1)
    date_to_era = era.combine_first(era_expanding)
    df["era_avg_sig_str_l12mo"] = df["event_date_dt"].map(date_to_era).fillna(40.0)

    # ── Weight-class finish share (causal, last 2y) ──────────────────────
    df["_is_finish"] = df["method"].isin(["KO/TKO", "SUB"]).astype(float)
    df = df.sort_values(["weight_class", "event_date_dt"])
    df["wc_finish_share_l2y"] = (
        df.groupby("weight_class", sort=False)["_is_finish"]
        .transform(lambda s: s.shift(1).rolling(2000, min_periods=20).mean())
        .fillna(0.5)
    )

    # ── Weight-class 5-round dec rate (causal, last 2y) ──────────────────
    df["_is_dec_5rd"] = (
        (df["scheduled_rounds"] >= 5)
        & df["method"].isin(["U-DEC", "S-DEC", "M-DEC"])
    ).astype(float)
    df["_is_5rd"] = (df["scheduled_rounds"] >= 5).astype(float)
    # Causal expanding ratio conditional on 5-round
    grp = df.groupby("weight_class", sort=False)
    num = grp["_is_dec_5rd"].transform(lambda s: s.shift(1).rolling(2000, min_periods=10).sum())
    den = grp["_is_5rd"].transform(lambda s: s.shift(1).rolling(2000, min_periods=10).sum())
    df["wc_5rd_dec_rate"] = (num / den.replace(0, np.nan)).fillna(0.55)

    # ── V7.2: Global era KO and SUB share (causal, last 24 months) ──────────
    # Aggregate to one row per fight (dedup by fight_id) to avoid double-counting
    # the same fight twice (one row per fighter).
    per_fight = df.drop_duplicates("fight_id")[["fight_id", "event_date_dt", "method"]].copy()
    per_fight = per_fight.sort_values("event_date_dt")
    per_fight["_is_ko"] = per_fight["method"].eq("KO/TKO").astype(float)
    per_fight["_is_sub"] = per_fight["method"].eq("SUB").astype(float)

    by_date_ko = per_fight.groupby("event_date_dt")["_is_ko"].mean().sort_index()
    by_date_sub = per_fight.groupby("event_date_dt")["_is_sub"].mean().sort_index()

    # 730-day rolling mean, shift(1) to be strictly causal
    era_ko = by_date_ko.shift(1).rolling("730D", min_periods=20).mean()
    era_sub = by_date_sub.shift(1).rolling("730D", min_periods=20).mean()

    # Fall back to expanding mean for early rows without 20 fights in window
    era_ko_exp = by_date_ko.expanding().mean().shift(1)
    era_sub_exp = by_date_sub.expanding().mean().shift(1)
    date_to_ko = era_ko.combine_first(era_ko_exp)
    date_to_sub = era_sub.combine_first(era_sub_exp)

    df["era_ko_share_l24mo"] = df["event_date_dt"].map(date_to_ko).fillna(0.28)
    df["era_sub_share_l24mo"] = df["event_date_dt"].map(date_to_sub).fillna(0.14)

    df = df.drop(columns=["event_date_dt", "_is_finish", "_is_dec_5rd", "_is_5rd"], errors="ignore")
    return df
