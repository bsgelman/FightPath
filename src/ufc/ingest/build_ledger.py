"""Build data/processed/ledger.parquet — the foundational table.

One row per (fight_id, fighter_id) = 2 rows per fight.
All feature engineering reads exclusively from this table.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ufc.io import paths, parquet


def _compute_age(dob: date | None, event_date: date | None) -> float | None:
    if dob is None or event_date is None:
        return None
    if isinstance(dob, str) or isinstance(event_date, str):
        return None
    return (event_date - dob).days / 365.25


def build_ledger(
    events_df: pd.DataFrame,
    fights_df: pd.DataFrame,
    fight_rounds_df: pd.DataFrame,
    fighters_df: pd.DataFrame,
    name_map_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the ledger from parsed interim tables.

    Parameters
    ----------
    events_df       : from parse_scraper.parse_events()
    fights_df       : from parse_scraper.parse_fights()
    fight_rounds_df : from parse_scraper.parse_fight_rounds()
    fighters_df     : from parse_scraper.parse_fighters()
    name_map_df     : from name_match.build_name_map()
    """

    # ── 1. Attach event dates and IDs to fights ────────────────────────────
    events_clean = events_df[["event_name", "event_id", "event_date", "location"]].copy()
    fights_merged = pd.merge(fights_df, events_clean, on="event_name", how="left")

    # Attach fighter IDs from name_map
    fights_merged = pd.merge(
        fights_merged,
        name_map_df[["fight_id", "fighter_a_id", "fighter_b_id"]],
        on="fight_id",
        how="left",
    )

    # ── 2. Aggregate round stats per (bout, fighter) ───────────────────────
    # fight_rounds_df has (event_name, bout_str, round_num, fighter_name, ...)
    # We need to join to get fight_id, then aggregate
    bout_to_fight = (
        fights_df[["fight_id", "event_name"]]
        .copy()
        .assign(
            fighter_a_name=fights_df["fighter_a_name"],
            fighter_b_name=fights_df["fighter_b_name"],
        )
    )

    # Join rounds to fight_id via event_name + bout_str match
    # Build a mapping: (event_name, normalized_bout) -> fight_id
    def _norm_bout(a, b):
        return f"{str(a).strip()} vs. {str(b).strip()}"

    fight_bout_map = {}
    for _, row in bout_to_fight.iterrows():
        key = (row["event_name"], _norm_bout(row["fighter_a_name"], row["fighter_b_name"]))
        fight_bout_map[key] = row["fight_id"]
        key2 = (row["event_name"], _norm_bout(row["fighter_b_name"], row["fighter_a_name"]))
        fight_bout_map[key2] = row["fight_id"]

    fight_rounds_df["fight_id"] = fight_rounds_df.apply(
        lambda r: fight_bout_map.get((r["event_name"], r["bout_str"])), axis=1
    )
    fight_rounds_df = fight_rounds_df.dropna(subset=["fight_id"])

    # Resolve fighter_name in rounds to fighter_id using the same name_map that
    # populates the ledger. A global name→id dict breaks for duplicate names
    # (e.g. two "Bruno Silva" fighters): whichever id lands last in fighters_df
    # wins, but the ledger may use the other id → join miss → all stats NaN.
    # Using (fight_id, fighter_name)→fighter_id from name_map+fights_df ensures
    # round_agg and ledger always agree on which id to use for a given fight.
    _fights_for_map = fights_df[["fight_id", "fighter_a_name", "fighter_b_name"]].copy()
    # Only bring id columns from name_map to avoid fighter_a/b_name column collision after merge
    _nm = name_map_df[["fight_id", "fighter_a_id", "fighter_b_id"]].merge(
        _fights_for_map, on="fight_id", how="inner"
    )
    fight_name_id: dict[tuple, str] = {}
    for _, row in _nm.iterrows():
        fid = row["fight_id"]
        if row.get("fighter_a_id") and pd.notna(row.get("fighter_a_name")):
            fight_name_id[(fid, str(row["fighter_a_name"]).strip())] = row["fighter_a_id"]
        if row.get("fighter_b_id") and pd.notna(row.get("fighter_b_name")):
            fight_name_id[(fid, str(row["fighter_b_name"]).strip())] = row["fighter_b_id"]

    fight_rounds_df["fighter_id"] = fight_rounds_df.apply(
        lambda r: fight_name_id.get((r["fight_id"], str(r["fighter_name"]).strip())),
        axis=1,
    )

    # Compute cardio ratio (late-round volume / early-round volume) per (fight, fighter)
    # before aggregating rounds away — captures fade/no-fade tendency
    _rd = fight_rounds_df.dropna(subset=["fighter_id", "fight_id"]).copy()
    _rd["SIGSTR_attempted"] = pd.to_numeric(_rd["SIGSTR_attempted"], errors="coerce").fillna(0)
    _early = _rd[_rd["round_num"].isin([1, 2])].groupby(["fight_id", "fighter_id"])["SIGSTR_attempted"].mean().rename("_early_vol")
    _late  = _rd[_rd["round_num"].isin([4, 5])].groupby(["fight_id", "fighter_id"])["SIGSTR_attempted"].mean().rename("_late_vol")
    _cardio = _early.to_frame().join(_late, how="left")
    _cardio["cardio_ratio_fight"] = (
        _cardio["_late_vol"] / _cardio["_early_vol"].clip(lower=1.0)
    ).clip(lower=0.0, upper=3.0)
    _cardio = _cardio[["cardio_ratio_fight"]].reset_index()

    # Aggregate to fight level
    stat_cols = [c for c in fight_rounds_df.columns
                 if c.endswith("_landed") or c.endswith("_attempted")]
    sum_cols = stat_cols + ["kd", "sub_att", "rev", "ctrl_sec"]
    sum_cols = [c for c in sum_cols if c in fight_rounds_df.columns]

    round_agg = (
        fight_rounds_df.dropna(subset=["fighter_id"])
        .groupby(["fight_id", "fighter_id"])[sum_cols]
        .sum()
        .reset_index()
        .merge(_cardio, on=["fight_id", "fighter_id"], how="left")
    )

    # ── 3. Build per-fighter rows ──────────────────────────────────────────
    rows = []

    for _, fight in fights_merged.iterrows():
        fid = fight["fight_id"]
        event_date = fight.get("event_date")
        event_id = fight.get("event_id")
        location = fight.get("location", "")

        a_id = fight.get("fighter_a_id")
        b_id = fight.get("fighter_b_id")
        a_won = fight.get("fighter_a_won")

        method = fight.get("method", "NC")
        end_round = fight.get("end_round")
        end_time_sec = fight.get("end_time_sec")
        total_fight_sec = fight.get("total_fight_sec")
        scheduled_rounds = fight.get("scheduled_rounds", 3)
        weight_class = fight.get("weight_class", "")
        is_title = bool(fight.get("is_title", False))
        injury_freak = bool(fight.get("injury_freak", False))
        referee = fight.get("referee", "")

        for perspective, (this_id, opp_id, won_val) in enumerate([
            (a_id, b_id, a_won),
            (b_id, a_id, 0 if a_won == 1 else (1 if a_won == 0 else None)),
        ]):
            if not this_id or not opp_id:
                continue

            # Get fighter bio
            frow = fighters_df[fighters_df["fighter_id"] == this_id]
            dob = frow["dob"].iloc[0] if len(frow) > 0 else None
            reach_in = frow["reach_in"].iloc[0] if len(frow) > 0 else None
            height_in = frow["height_in"].iloc[0] if len(frow) > 0 else None
            stance = frow["stance"].iloc[0] if len(frow) > 0 else "UNKNOWN"
            weight_lbs = frow["weight_lbs"].iloc[0] if len(frow) > 0 else None

            # Compute age
            if isinstance(event_date, (date, pd.Timestamp)):
                ed = event_date.date() if hasattr(event_date, "date") else event_date
            else:
                ed = None
            db = dob.date() if hasattr(dob, "date") and dob is not None else dob
            age = _compute_age(db, ed)

            # Get stats for this fighter in this fight
            stats_row = round_agg[
                (round_agg["fight_id"] == fid) & (round_agg["fighter_id"] == this_id)
            ]
            opp_stats_row = round_agg[
                (round_agg["fight_id"] == fid) & (round_agg["fighter_id"] == opp_id)
            ]

            def _g(df, col, default=None):
                """Return int value or None when missing (so downstream rolling avgs skip)."""
                if len(df) == 0 or col not in df.columns:
                    return default
                v = df[col].iloc[0]
                if pd.isna(v):
                    return default
                return int(v)

            record = {
                "fight_id": fid,
                "event_id": event_id,
                "event_date": event_date,
                "fighter_id": this_id,
                "opponent_id": opp_id,
                "weight_class": weight_class,
                "scheduled_rounds": scheduled_rounds,
                "is_title": is_title,
                "injury_freak": injury_freak,
                "is_main_event": False,  # placeholder; set later if needed
                "won": won_val,
                "method": method,
                "end_round": end_round,
                "end_time_sec": end_time_sec,
                "total_fight_sec": total_fight_sec,
                # Own stats
                "sig_str_landed": _g(stats_row, "SIGSTR_landed"),
                "sig_str_attempted": _g(stats_row, "SIGSTR_attempted"),
                "td_landed": _g(stats_row, "TD_landed"),
                "td_attempted": _g(stats_row, "TD_attempted"),
                "ctrl_sec": _g(stats_row, "ctrl_sec"),
                "kd_for": _g(stats_row, "kd"),
                "sub_att_for": _g(stats_row, "sub_att"),
                "rev_for": _g(stats_row, "rev"),
                "head_landed": _g(stats_row, "HEAD_landed"),
                "body_landed": _g(stats_row, "BODY_landed"),
                "leg_landed": _g(stats_row, "LEG_landed"),
                "distance_landed": _g(stats_row, "DISTANCE_landed"),
                "clinch_landed": _g(stats_row, "CLINCH_landed"),
                "ground_landed": _g(stats_row, "GROUND_landed"),
                "head_attempted": _g(stats_row, "HEAD_attempted"),
                "body_attempted": _g(stats_row, "BODY_attempted"),
                "leg_attempted": _g(stats_row, "LEG_attempted"),
                "distance_attempted": _g(stats_row, "DISTANCE_attempted"),
                "clinch_attempted": _g(stats_row, "CLINCH_attempted"),
                "ground_attempted": _g(stats_row, "GROUND_attempted"),
                # Absorbed stats (from opponent's perspective)
                "sig_str_absorbed_landed": _g(opp_stats_row, "SIGSTR_landed"),
                "sig_str_absorbed_attempted": _g(opp_stats_row, "SIGSTR_attempted"),
                "td_absorbed_landed": _g(opp_stats_row, "TD_landed"),
                "td_absorbed_attempted": _g(opp_stats_row, "TD_attempted"),
                "ctrl_sec_absorbed": _g(opp_stats_row, "ctrl_sec"),
                "kd_against": _g(opp_stats_row, "kd"),
                "sub_att_against": _g(opp_stats_row, "sub_att"),
                "rev_against": _g(opp_stats_row, "rev"),
                "head_absorbed": _g(opp_stats_row, "HEAD_landed"),
                "body_absorbed": _g(opp_stats_row, "BODY_landed"),
                "leg_absorbed": _g(opp_stats_row, "LEG_landed"),
                "distance_absorbed": _g(opp_stats_row, "DISTANCE_landed"),
                "clinch_absorbed": _g(opp_stats_row, "CLINCH_landed"),
                "ground_absorbed": _g(opp_stats_row, "GROUND_landed"),
                # Cardio fade ratio (NaN for fights that didn't reach R4+)
                "cardio_ratio_fight": float(stats_row["cardio_ratio_fight"].iloc[0])
                    if len(stats_row) > 0 and "cardio_ratio_fight" in stats_row.columns
                    and not pd.isna(stats_row["cardio_ratio_fight"].iloc[0]) else None,
                # Bio snapshot
                "age_years": age,
                "reach_in": float(reach_in) if reach_in is not None and not (isinstance(reach_in, float) and np.isnan(reach_in)) else None,
                "height_in": float(height_in) if height_in is not None and not (isinstance(height_in, float) and np.isnan(height_in)) else None,
                "stance": stance,
                "weight_lbs": float(weight_lbs) if weight_lbs is not None and not (isinstance(weight_lbs, float) and np.isnan(weight_lbs)) else None,
                "referee": referee,
                "location": str(location) if location else "",
            }
            rows.append(record)

    ledger = pd.DataFrame(rows)

    # ── 4. Sort by event_date and assign event_rank ────────────────────────
    ledger["event_date"] = pd.to_datetime(ledger["event_date"])
    # A row whose bout is in ufc_fight_results.csv but whose EVENT never appears in
    # ufc_event_details.csv joins to no event and carries no date (2026-07-25: two
    # "UFC - Road to UFC 4.6" bouts — that series has never been in event_details).
    # Undated rows are unusable: the split, every temporal feature and transitivity
    # all key off event_date, and groupby(event_date).ngroup() drops NaT to NaN,
    # which crashed compute_transitivity. Drop them here, loudly.
    undated = ledger["event_date"].isna()
    if undated.any():
        print(f"  [warn] dropping {int(undated.sum())} ledger rows with no event_date "
              f"(bout present in fight_results but its event is absent from event_details)")
        ledger = ledger[~undated]
    ledger = ledger.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)
    ledger["event_rank"] = ledger.groupby("event_date", sort=False).ngroup()

    # Apply nullable integer dtypes (Int*) — preserves None semantics from _g()
    int_cols = [
        "end_round", "scheduled_rounds",
        "sig_str_landed", "sig_str_attempted",
        "td_landed", "td_attempted", "ctrl_sec",
        "kd_for", "sub_att_for", "rev_for",
        "head_landed", "body_landed", "leg_landed",
        "distance_landed", "clinch_landed", "ground_landed",
        "head_attempted", "body_attempted", "leg_attempted",
        "distance_attempted", "clinch_attempted", "ground_attempted",
        "sig_str_absorbed_landed", "sig_str_absorbed_attempted",
        "td_absorbed_landed", "td_absorbed_attempted",
        "ctrl_sec_absorbed", "kd_against", "sub_att_against", "rev_against",
        "head_absorbed", "body_absorbed", "leg_absorbed",
        "distance_absorbed", "clinch_absorbed", "ground_absorbed",
    ]
    for col in int_cols:
        if col in ledger.columns:
            ledger[col] = pd.to_numeric(ledger[col], errors="coerce").astype("Int32")
    if "won" in ledger.columns:
        ledger["won"] = pd.to_numeric(ledger["won"], errors="coerce").astype("Int8")
    if "end_time_sec" in ledger.columns:
        ledger["end_time_sec"] = pd.to_numeric(ledger["end_time_sec"], errors="coerce").astype("Int16")
    if "total_fight_sec" in ledger.columns:
        ledger["total_fight_sec"] = pd.to_numeric(ledger["total_fight_sec"], errors="coerce").astype("Int16")

    return ledger


def backfill_reach(ledger: pd.DataFrame, reach_lookup: pd.DataFrame) -> pd.DataFrame:
    """Fill missing reach_in values using archive data matched on normalized name."""
    from ufc.ingest.parse_helpers import normalize_name

    if reach_lookup.empty:
        return ledger

    fighters_df = pd.read_parquet(paths.interim("fighters"))
    fighters_df["norm_name"] = fighters_df["fighter_name"].apply(normalize_name)

    lookup_merged = pd.merge(
        fighters_df[["fighter_id", "norm_name"]],
        reach_lookup[["norm_name", "reach_in", "height_in"]],
        on="norm_name",
        how="left",
        suffixes=("", "_archive"),
    )
    reach_map = lookup_merged.dropna(subset=["reach_in"]).set_index("fighter_id")["reach_in"].to_dict()
    height_map = lookup_merged.dropna(subset=["height_in"]).set_index("fighter_id")["height_in"].to_dict()

    mask_reach = ledger["reach_in"].isna()
    ledger.loc[mask_reach, "reach_in"] = ledger.loc[mask_reach, "fighter_id"].map(reach_map)
    mask_height = ledger["height_in"].isna()
    ledger.loc[mask_height, "height_in"] = ledger.loc[mask_height, "fighter_id"].map(height_map)

    return ledger
