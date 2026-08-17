"""Assemble all feature groups into model-ready wide parquets.

Produces:
  data/processed/features_winner.parquet  - per (fight, perspective) rows
  data/processed/features_props.parquet   - per (fight, fighter) rows
  data/processed/pre_fight_state.parquet  - latest state per fighter (for inference)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import yaml

from ufc.io import paths, parquet
from ufc.features import (
    ratings, striking, grappling, style, mileage, context, interactions, physical
)
from ufc.features.finishes import compute_finish_rates, fill_sparse_history
from ufc.features.interactions import compute_opponent_quality
from ufc.features import round_detail
from ufc.features.transitivity import compute_transitivity
from ufc.features.pre_ufc import get_pre_ufc_lookup


def _cfg():
    with open(paths.root() / "configs" / "split.yaml") as f:
        return yaml.safe_load(f)


# Columns a sentinel row keeps from the fighter's last real row. Everything
# else is nulled: a sentinel describes the FIGHTER (identity/bio/division),
# never a fight. Nulled stats are safe — causal windows exclude the current
# row, and no row sorts after a sentinel within its fighter group.
SENTINEL_KEEP_COLS = {
    "fight_id", "fighter_id", "event_date", "event_rank", "is_sentinel",
    "weight_class", "stance", "age_years", "reach_in", "height_in", "weight_lbs",
}

# Plain-int columns that get an explicit placeholder on sentinels instead of
# null (numpy int has no null slot; nulling would promote REAL rows to float64).
# 0 = "no common opponents with the not-yet-known next opponent" — honest for
# an upcoming pairing, and matches what the serve path should see.
SENTINEL_INT_ZERO_COLS = {"n_common_opps"}


def append_sentinel_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    """Append one no-outcome sentinel row per fighter, dated 1 day after their
    last fight and event-ranked after every real fight.

    The per-fighter causal feature build then produces, for each sentinel, a
    feature row covering the fighter's ENTIRE completed history — this becomes
    pre_fight_state (previously: the pre-fight row of the LAST fight, which
    silently dropped that fight's result from serving; see RUNS.md).
    Sentinels carry is_sentinel=True and MUST NOT reach the training tables.
    """
    df = ledger.copy()
    df["is_sentinel"] = False
    sent = (
        df.sort_values(["fighter_id", "event_date", "fight_id"])
        .groupby("fighter_id", sort=False)
        .tail(1)
        .copy()
    )
    sent["fight_id"] = "sentinel_" + sent["fighter_id"].astype(str)
    sent["event_date"] = pd.to_datetime(sent["event_date"]) + pd.Timedelta(1, unit="D")
    sent["event_rank"] = df["event_rank"].max() + 1
    sent["is_sentinel"] = True
    for c in sent.columns:
        if c in SENTINEL_KEEP_COLS:
            continue
        dt = sent[c].dtype
        if pd.api.types.is_datetime64_any_dtype(dt):
            sent[c] = pd.NaT
        elif isinstance(dt, pd.api.extensions.ExtensionDtype):
            # Nullable Int/Float/boolean columns: keep the dtype, null with pd.NA
            # (np.nan assignment would coerce the column to float64 and pd.concat
            # would then promote the REAL rows' dtype too).
            sent[c] = pd.array([pd.NA] * len(sent), dtype=dt)
        elif pd.api.types.is_bool_dtype(dt):
            # Plain numpy bool has no missing-value slot; nulling with NaN/None
            # would force pd.concat to upcast the REAL rows' dtype too (bool ->
            # object/boolean), same class of bug as the nullable-Int case above.
            # Default the sentinel placeholder to False to keep dtype stable.
            sent[c] = False
        elif pd.api.types.is_integer_dtype(dt) and c in SENTINEL_INT_ZERO_COLS:
            sent[c] = 0
        elif pd.api.types.is_integer_dtype(dt):
            # Plain numpy int has no null slot — nulling would promote the REAL
            # rows to float64 via pd.concat, breaking the byte-identical
            # invariant. Fail loudly: convert the column to a nullable Int dtype
            # in the ledger builder (see build_ledger.py) before adding it here.
            raise TypeError(
                f"append_sentinel_rows: column '{c}' is non-nullable {dt}; "
                "make it a nullable Int dtype in the ledger or add it to "
                "SENTINEL_KEEP_COLS."
            )
        elif pd.api.types.is_numeric_dtype(dt):
            sent[c] = np.nan
        else:
            sent[c] = None
    return pd.concat([df, sent], ignore_index=True)


def build_per_fighter_features(ledger: pd.DataFrame,
                                train_mask: pd.Series | None = None) -> pd.DataFrame:
    """Compute all per-fighter rolling features from the ledger.

    Returns a long DataFrame (one row per fight per fighter) with all features.
    """
    print("  Computing pre-UFC record features...")
    pre_ufc_lu = get_pre_ufc_lookup(ledger)
    print("  Computing ELO ratings (seeded from pre-UFC record)...")
    df = ratings.compute_elo(ledger, pre_ufc_lookup=pre_ufc_lu)
    print("  Computing Glicko-2 (seeded from pre-UFC record)...")
    df = ratings.compute_glicko2(df, pre_ufc_lookup=pre_ufc_lu)
    print("  Computing TrueSkill ratings...")
    df = ratings.compute_trueskill(df, pre_ufc_lookup=pre_ufc_lu)
    # Join pre-UFC features as stand-alone columns
    df = df.join(pre_ufc_lu[["pre_ufc_n", "pre_ufc_win_rate_shrunk"]], on="fighter_id", how="left")
    df["pre_ufc_n"] = df["pre_ufc_n"].fillna(0.0)
    df["pre_ufc_win_rate_shrunk"] = df["pre_ufc_win_rate_shrunk"].fillna(0.5)
    print("  Computing striking features...")
    df = striking.compute_striking(df)
    print("  Computing grappling features...")
    df = grappling.compute_grappling(df)
    print("  Computing finish-rate features...")
    df = compute_finish_rates(df)
    df = fill_sparse_history(df, train_mask=train_mask)
    print("  Computing style scores...")
    df = style.compute_style_scores(df, train_mask=train_mask)
    print("  Computing mileage features...")
    df = mileage.compute_mileage(df)
    print("  Computing context features...")
    df = context.compute_context(df)
    df = context.compute_home_advantage(df)

    # Per-round rolling features (from raw cols pre-joined onto ledger)
    if any(c.endswith("_raw") for c in df.columns):
        print("  Computing per-round rolling features...")
        df = round_detail.compute_round_features(df)

    # cardio_score is now computed in mileage.compute_mileage() from cardio_ratio_fight
    # Fill any remaining NaN with 1.0 (neutral — no fade information available)
    if "cardio_score" not in df.columns:
        df["cardio_score"] = 1.0
    else:
        df["cardio_score"] = df["cardio_score"].fillna(1.0)

    # ── Stance win rate vs opponent stance ────────────────────────────────
    # Join opponent's stance onto each row, then causal rolling win rate per (fighter, opp_stance)
    _stance_lu = (
        df[["fight_id", "fighter_id", "stance"]].drop_duplicates()
        .rename(columns={"fighter_id": "opponent_id", "stance": "opp_stance"})
    )
    df = df.merge(_stance_lu, on=["fight_id", "opponent_id"], how="left")

    df["_won_float"] = pd.to_numeric(df["won"], errors="coerce").astype("float64")
    df = df.sort_values(["fighter_id", "event_rank"])
    df["stance_wr_vs_opp_stance"] = (
        df.groupby(["fighter_id", "opp_stance"], sort=False)["_won_float"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0.5)
    )
    df = df.drop(columns=["_won_float"], errors="ignore")

    return df


def build_wide_fight_features(per_fighter_df: pd.DataFrame,
                               fights_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create wide (A vs B) DataFrame from long (per-fighter) DataFrame.

    A/B assignment is taken from fights_df's (fighter_a_id, fighter_b_id) if provided,
    so the wide layout matches the source corner ordering deterministically.
    Falls back to alphabetical fighter_id ordering if fights_df is None.
    """
    pf = per_fighter_df.copy()

    # Columns that are per-fighter features (not fight-level)
    # We'll separate them by suffixing with _a (this fighter) and _b (opponent)
    fight_cols = ["fight_id", "event_id", "event_date", "event_rank",
                  "weight_class", "scheduled_rounds", "is_title", "is_main_event",
                  "method", "end_round", "end_time_sec", "total_fight_sec",
                  "referee", "location", "altitude_meters", "referee_stoppage_threshold",
                  "era_avg_sig_str_l12mo", "wc_finish_share_l2y", "wc_5rd_dec_rate", "injury_freak"]

    fighter_cols = [c for c in pf.columns if c not in fight_cols + [
        "fighter_id", "opponent_id", "won",
        "sig_str_landed", "sig_str_attempted", "sig_str_absorbed_landed", "sig_str_absorbed_attempted",
        "td_landed", "td_attempted", "td_absorbed_landed", "td_absorbed_attempted",
        "ctrl_sec", "ctrl_sec_absorbed", "kd_for", "kd_against", "sub_att_for", "sub_att_against",
        "rev_for", "rev_against", "head_landed", "body_landed", "leg_landed",
        "distance_landed", "clinch_landed", "ground_landed",
        "head_absorbed", "body_absorbed", "leg_absorbed",
        "distance_absorbed", "clinch_absorbed", "ground_absorbed",
        "head_attempted", "body_attempted", "leg_attempted",
        "distance_attempted", "clinch_attempted", "ground_attempted",
        "fight_min",
        "r1_sig_str_landed", "r1_td_landed",
    ]]

    # Build canonical (fight_id -> a_id) lookup
    a_id_map: dict = {}
    if fights_df is not None and {"fight_id", "fighter_a_id"} <= set(fights_df.columns):
        a_id_map = dict(zip(fights_df["fight_id"], fights_df["fighter_a_id"]))

    fight_groups = pf.groupby("fight_id", sort=False)

    rows = []
    for fid, grp in fight_groups:
        if len(grp) != 2:
            continue
        canonical_a = a_id_map.get(fid)
        if canonical_a is not None and canonical_a in grp["fighter_id"].values:
            row_a = grp[grp["fighter_id"] == canonical_a].iloc[0]
            row_b = grp[grp["fighter_id"] != canonical_a].iloc[0]
        else:
            grp_sorted = grp.sort_values("fighter_id")
            row_a = grp_sorted.iloc[0]
            row_b = grp_sorted.iloc[1]

        row = {}
        # Fight-level columns (same for both)
        for c in fight_cols:
            if c in pf.columns:
                row[c] = row_a.get(c)

        # A-fighter features
        row["fighter_id_a"] = row_a["fighter_id"]
        row["fighter_id_b"] = row_b["fighter_id"]
        row["won_a"] = row_a.get("won")  # label: did fighter A win?

        # Raw stats (for prop target labels)
        for stat in ["sig_str_landed", "td_landed", "ctrl_sec", "sub_att_for", "r1_sig_str_landed",
                     "kd_for", "body_landed", "leg_landed", "r1_td_landed"]:
            row[f"{stat}_a"] = row_a.get(stat, 0)
            row[f"{stat}_b"] = row_b.get(stat, 0)

        # Per-fighter feature columns
        for c in fighter_cols:
            if c in pf.columns:
                row[f"{c}_a"] = row_a.get(c)
                row[f"{c}_b"] = row_b.get(c)

        rows.append(row)

    wide = pd.DataFrame(rows)
    return wide


def assemble(ledger: pd.DataFrame | None = None, gitsha: str = "latest") -> None:
    """Full pipeline: read ledger → compute features → write parquets."""
    if ledger is None:
        ledger = parquet.read(paths.processed("ledger"))

    # Ensure date is parsed
    ledger["event_date"] = pd.to_datetime(ledger["event_date"])
    ledger = ledger.sort_values(["event_date", "fight_id"]).reset_index(drop=True)

    # ── Load interim tables (used for round-detail join and R1 label join) ──
    rounds_df = fights_df = fighters_df = None
    try:
        rounds_df = parquet.read(paths.interim("fight_rounds"))
        fights_df = parquet.read(paths.interim("fights"))
        fighters_df = parquet.read(paths.interim("fighters"))
    except Exception as e:
        print(f"  WARNING: Could not load interim tables: {e}")

    # ── Enrich ledger with per-round raw stats (rolling applied later) ──────
    if rounds_df is not None and fights_df is not None and fighters_df is not None:
        try:
            print("  Joining per-round raw stats to ledger...")
            ledger = round_detail.join_round_raw(ledger, fights_df, fighters_df, rounds_df)
            n_raw = len([c for c in ledger.columns if c.endswith("_raw")])
            print(f"  Added {n_raw} per-round raw columns")
        except Exception as e:
            print(f"  WARNING: Could not join per-round stats: {e}")

    # ── Common-opponent transitivity (causal: only uses prior fights) ────────
    print("  Computing common-opponent transitivity...")
    ledger = compute_transitivity(ledger)

    # ── Era + weight-class baselines (all causal rolling) ────────────────────
    print("  Computing era + weight-class baselines...")
    ledger = context.compute_era_baselines(ledger)

    # Restore chronological order with positional labels. compute_era_baselines
    # returns the frame sorted by (weight_class, event_date); downstream rating
    # passes reset the index via merge, which desynchronized the label-aligned
    # train_mask in fill_sparse_history — its "train fold" silently included
    # post-train_end rows (1,926 leaked rows as of 2026-07). Order + labels must
    # be in lockstep before the mask is computed.
    ledger = ledger.sort_values(["event_date", "fight_id"]).reset_index(drop=True)

    # ── Sentinel rows: one per fighter, so pre_fight_state includes the last fight ──
    print("  Appending per-fighter sentinel rows (inference state)...")
    ledger = append_sentinel_rows(ledger)

    # ── Compute train mask BEFORE per-fighter feature build so causal stats use it ──
    split_cfg = _cfg()
    train_end = pd.to_datetime(split_cfg["train_end"])
    ledger_train_mask = (
        (pd.to_datetime(ledger["event_date"]) <= train_end)
        & ~ledger["is_sentinel"]
    )

    print("\n[A] Computing per-fighter features...")
    pf = build_per_fighter_features(ledger, train_mask=ledger_train_mask)

    # Save pre_fight_state (sentinel rows = full-history state per fighter)
    latest_state = (
        pf[pf["is_sentinel"].fillna(False)]
        .drop(columns=["is_sentinel"])
        .reset_index(drop=True)
    )
    assert latest_state["fighter_id"].is_unique, "expected exactly one sentinel per fighter"
    paths.processed("pre_fight_state").parent.mkdir(parents=True, exist_ok=True)
    parquet.write(latest_state, paths.processed("pre_fight_state"))
    print(f"  pre_fight_state: {len(latest_state)} fighters")

    # Training tables must never see sentinels.
    pf = pf[~pf["is_sentinel"].fillna(False)].drop(columns=["is_sentinel"]).reset_index(drop=True)

    # Join R1 stat labels per fighter from fight_rounds (prop target labels)
    if rounds_df is not None and fights_df is not None and fighters_df is not None:
        fights_a = fights_df[["fight_id", "event_name", "fighter_a_name"]].rename(
            columns={"fighter_a_name": "fighter_name"})
        fights_b = fights_df[["fight_id", "event_name", "fighter_b_name"]].rename(
            columns={"fighter_b_name": "fighter_name"})
        fights_long = pd.concat([fights_a, fights_b], ignore_index=True)
        name_to_id = dict(zip(fighters_df["fighter_name"], fighters_df["fighter_id"]))

        def _join_r1_stat(src_col: str, dst_col: str) -> None:
            try:
                r1_raw = (
                    rounds_df[rounds_df["round_num"] == 1]
                    .groupby(["event_name", "fighter_name"])[src_col]
                    .sum()
                    .reset_index()
                    .rename(columns={src_col: dst_col})
                )
                r1_raw = r1_raw.merge(fights_long, on=["event_name", "fighter_name"], how="left")
                r1_raw["fighter_id"] = r1_raw["fighter_name"].map(name_to_id)
                r1_joined = (
                    r1_raw.dropna(subset=["fight_id", "fighter_id"])
                    [["fight_id", "fighter_id", dst_col]]
                )
                nonlocal pf
                pf = pf.merge(r1_joined, on=["fight_id", "fighter_id"], how="left")
                pf[dst_col] = pf[dst_col].fillna(0)
                print(f"  Joined {dst_col} ({len(r1_joined)} fighter-fight rows)")
            except Exception as e:
                print(f"  WARNING: Could not join {dst_col}: {e}")
                pf[dst_col] = 0

        _join_r1_stat("SIGSTR_landed", "r1_sig_str_landed")
        _join_r1_stat("TD_landed", "r1_td_landed")
    else:
        pf["r1_sig_str_landed"] = 0
        pf["r1_td_landed"] = 0

    print("\n[B] Building wide fight features...")
    wide = build_wide_fight_features(pf, fights_df=fights_df)

    # Add physical deltas
    wide = physical.compute_physical(wide)

    # Add interaction features
    wide = interactions.compute_interactions(wide)

    # Add opponent quality features (use train-fold stats only — Fix #3)
    wide_train_mask = pd.to_datetime(wide["event_date"]) <= train_end
    wide = compute_opponent_quality(wide, train_mask=wide_train_mask)

    print("\n[C] Fitting PCA on style features (train fold)...")
    wide = style.fit_style_pca(wide, wide_train_mask, gitsha=gitsha)

    # ── Features winner ────────────────────────────────────────────────────
    # Drop raw fight outcome stats (leak) and keep only pre-fight features + labels
    target_cols = ["won_a", "method", "end_round", "end_time_sec", "total_fight_sec"]
    label_cols = target_cols + ["sig_str_landed_a", "sig_str_landed_b",
                                "td_landed_a", "td_landed_b"]

    # Drop rows where both fighters have no resolution
    wide_winner = wide.dropna(subset=["fighter_id_a", "fighter_id_b"]).copy()
    # Drop fights without outcome (upcoming / NC)
    wide_winner = wide_winner.dropna(subset=["won_a"])
    # Filter DQ/NC/Draw — corrupt labels for the binary "did A win?" classifier
    wide_winner = wide_winner[
        wide_winner["method"].isin(["KO/TKO", "SUB", "U-DEC", "S-DEC", "M-DEC"])
    ].copy()

    paths.processed("features_winner").parent.mkdir(parents=True, exist_ok=True)
    parquet.write(wide_winner, paths.processed("features_winner"))
    print(f"  features_winner: {len(wide_winner)} fight rows")

    # ── Features props ─────────────────────────────────────────────────────
    # For prop models we include rows with known fight outcomes too
    wide_props = wide.dropna(subset=["fighter_id_a", "fighter_id_b"]).copy()
    paths.processed("features_props").parent.mkdir(parents=True, exist_ok=True)
    parquet.write(wide_props, paths.processed("features_props"))
    print(f"  features_props: {len(wide_props)} fight rows")

    print("\n=== Feature assembly complete ===")
