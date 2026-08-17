"""Parse scrape_ufc_stats-main CSVs into typed parquet files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ufc.io import paths, parquet
from ufc.ingest.parse_helpers import (
    parse_event_date, parse_x_of_y, parse_mm_ss, parse_height,
    parse_weight, parse_reach, parse_dob, parse_scheduled_rounds,
    normalize_method, normalize_stance, strip_ws_columns, extract_url_hex,
)


def _scraper_dir() -> Path:
    return paths.scraper_source()


def _injury_freak_flags(merged: pd.DataFrame,
                        curation: pd.DataFrame | None = None) -> pd.Series:
    """Freak-injury stoppage flag (spec: docs/superpowers/specs/2026-07-15-injury-stoppage-flag-design.md).

    Curation CSV wins; keyword-hit rows absent from it default to freak=1
    (tripwire) and are counted loudly EVERY run, including zero — the DETAILS
    text is untrusted and this rule fails open if its format drifts.
    """
    if curation is None:
        cur_path = paths.root() / "data" / "raw" / "manual" / "injury_stoppages.csv"
        curation = (pd.read_csv(cur_path, dtype={"fight_id": str})
                    if cur_path.exists()
                    else pd.DataFrame({"fight_id": [], "freak": []}))
    detail = merged.get("DETAILS", pd.Series("", index=merged.index)).fillna("")
    keyword = detail.str.contains("injur", case=False)
    freak_map = dict(zip(curation["fight_id"].astype(str),
                         curation["freak"].astype(int)))
    curated_flag = merged["fight_id"].map(freak_map)
    uncurated_kw = keyword & curated_flag.isna()
    print(f"  [injury tripwire] injury-keyword rows not in curation CSV: {int(uncurated_kw.sum())}")
    return (curated_flag.fillna(0).astype(bool) | uncurated_kw)


def parse_events() -> pd.DataFrame:
    src = _scraper_dir() / "ufc_event_details.csv"
    df = pd.read_csv(src, dtype=str)
    df = strip_ws_columns(df)
    df.columns = [c.strip() for c in df.columns]

    df["event_date"] = df["DATE"].apply(parse_event_date)
    df["event_name"] = df["EVENT"]
    df["location"] = df["LOCATION"]
    df["event_id"] = df["URL"].apply(extract_url_hex)
    df["event_url"] = df["URL"]

    df = df[["event_id", "event_name", "event_date", "location", "event_url"]].copy()
    df = df.dropna(subset=["event_date", "event_id"])
    df = df.sort_values("event_date").reset_index(drop=True)
    return df


def parse_fights() -> pd.DataFrame:
    details = pd.read_csv(_scraper_dir() / "ufc_fight_details.csv", dtype=str)
    results = pd.read_csv(_scraper_dir() / "ufc_fight_results.csv", dtype=str)
    details = strip_ws_columns(details)
    results = strip_ws_columns(results)
    details.columns = [c.strip() for c in details.columns]
    results.columns = [c.strip() for c in results.columns]

    # Merge on EVENT + BOUT (fight_details has the URL we need)
    merged = pd.merge(
        details[["EVENT", "BOUT", "URL"]],
        results[["EVENT", "BOUT", "OUTCOME", "WEIGHTCLASS", "METHOD",
                 "ROUND", "TIME", "TIME FORMAT", "REFEREE", "DETAILS"]],
        on=["EVENT", "BOUT"],
        how="left",
    )

    merged["fight_id"] = merged["URL"].apply(extract_url_hex)
    merged["event_name"] = merged["EVENT"]

    # Parse BOUT into two fighter names
    bout_split = merged["BOUT"].str.split(" vs. ", n=1, expand=True)
    merged["fighter_a_name"] = bout_split[0].str.strip()
    merged["fighter_b_name"] = bout_split[1].str.strip() if 1 in bout_split.columns else None

    # Outcome: "W/L" means fighter_a won
    merged["fighter_a_won"] = merged["OUTCOME"].apply(
        lambda x: 1 if isinstance(x, str) and x.strip().upper() == "W/L" else
                  (0 if isinstance(x, str) and x.strip().upper() == "L/W" else None)
    )

    # Method normalization
    merged["method"] = merged["METHOD"].apply(normalize_method)

    # Round / time
    merged["end_round"] = pd.to_numeric(merged["ROUND"], errors="coerce").astype("Int8")
    merged["end_time_sec"] = merged["TIME"].apply(parse_mm_ss).astype("Int16")
    merged["scheduled_rounds"] = merged["TIME FORMAT"].apply(parse_scheduled_rounds).astype("Int8")

    # Total fight seconds
    def _total_sec(row):
        r = row["end_round"]
        t = row["end_time_sec"]
        if pd.isna(r) or pd.isna(t):
            return None
        return int((r - 1) * 300 + t)

    merged["total_fight_sec"] = merged.apply(_total_sec, axis=1).astype("Int16")

    # Weight class
    merged["weight_class"] = merged["WEIGHTCLASS"].str.replace(" Bout", "", regex=False).str.strip()

    # Referee
    merged["referee"] = merged["REFEREE"]

    # Is title (look for 'Title' in weight class or details)
    merged["is_title"] = (
        merged["WEIGHTCLASS"].str.contains("Title", case=False, na=False) |
        merged.get("DETAILS", pd.Series(dtype=str)).str.contains("Title", case=False, na=False)
    )

    # Injury freak flag
    merged["injury_freak"] = _injury_freak_flags(merged)

    out_cols = [
        "fight_id", "event_name", "fighter_a_name", "fighter_b_name",
        "fighter_a_won", "method", "end_round", "end_time_sec",
        "scheduled_rounds", "total_fight_sec", "weight_class",
        "is_title", "injury_freak", "referee",
    ]
    df = merged[out_cols].copy()
    df = df.dropna(subset=["fight_id"])
    df = df.drop_duplicates(subset=["fight_id"])
    return df


def parse_fight_rounds() -> pd.DataFrame:
    src = _scraper_dir() / "ufc_fight_stats.csv"
    df = pd.read_csv(src, dtype=str)
    df = strip_ws_columns(df)
    df.columns = [c.strip() for c in df.columns]

    df["fight_id"] = df["EVENT"].apply(lambda _: None)  # filled by join below
    # Actually the stats CSV doesn't have the fight URL directly — join via EVENT+BOUT
    df["event_name"] = df["EVENT"]
    df["bout_str"] = df["BOUT"]

    # Round number
    df["round_num"] = df["ROUND"].str.extract(r"(\d+)").astype("Int8")

    # Fighter name
    df["fighter_name"] = df["FIGHTER"]

    # Parse "X of Y" columns
    sig_cols = ["SIG.STR.", "TOTAL STR.", "TD", "HEAD", "BODY", "LEG",
                "DISTANCE", "CLINCH", "GROUND"]
    for col in sig_cols:
        if col in df.columns:
            parsed = df[col].apply(parse_x_of_y)
            safe_col = col.replace(".", "").replace(" ", "_").upper()
            df[f"{safe_col}_landed"] = parsed.apply(lambda x: x[0]).astype("Int16")
            df[f"{safe_col}_attempted"] = parsed.apply(lambda x: x[1]).astype("Int16")

    # Accuracy % columns (keep raw then parse)
    if "SIG.STR. %" in df.columns:
        df["sig_str_pct_raw"] = df["SIG.STR. %"]
    if "TD %" in df.columns:
        df["td_pct_raw"] = df["TD %"]

    # Control time
    df["ctrl_sec"] = df["CTRL"].apply(parse_mm_ss).astype("Int16")

    # Knockdowns
    df["kd"] = pd.to_numeric(df["KD"], errors="coerce").fillna(0).astype("Int8")

    # Reversals
    if "REV." in df.columns:
        df["rev"] = pd.to_numeric(df["REV."], errors="coerce").fillna(0).astype("Int8")
    else:
        df["rev"] = 0

    # Sub attempts
    if "SUB.ATT" in df.columns:
        df["sub_att"] = pd.to_numeric(df["SUB.ATT"], errors="coerce").fillna(0).astype("Int8")
    else:
        df["sub_att"] = 0

    out_cols = [
        "event_name", "bout_str", "round_num", "fighter_name",
        "kd", "sub_att", "rev", "ctrl_sec",
        "SIGSTR_landed", "SIGSTR_attempted",
        "TOTAL_STR_landed", "TOTAL_STR_attempted",
        "TD_landed", "TD_attempted",
        "HEAD_landed", "HEAD_attempted",
        "BODY_landed", "BODY_attempted",
        "LEG_landed", "LEG_attempted",
        "DISTANCE_landed", "DISTANCE_attempted",
        "CLINCH_landed", "CLINCH_attempted",
        "GROUND_landed", "GROUND_attempted",
    ]
    # Keep only columns that exist
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols].copy()


def parse_fighters() -> pd.DataFrame:
    tott = pd.read_csv(_scraper_dir() / "ufc_fighter_tott.csv", dtype=str)
    details = pd.read_csv(_scraper_dir() / "ufc_fighter_details.csv", dtype=str)
    tott = strip_ws_columns(tott)
    details = strip_ws_columns(details)
    tott.columns = [c.strip() for c in tott.columns]
    details.columns = [c.strip() for c in details.columns]

    # tott has: FIGHTER, HEIGHT, WEIGHT, REACH, STANCE, DOB, URL
    tott["fighter_id"] = tott["URL"].apply(extract_url_hex)
    tott["fighter_name"] = tott["FIGHTER"]
    tott["height_in"] = tott["HEIGHT"].apply(parse_height)
    tott["weight_lbs"] = tott["WEIGHT"].apply(parse_weight)
    tott["reach_in"] = tott["REACH"].apply(parse_reach)
    tott["stance"] = tott["STANCE"].apply(normalize_stance)
    tott["dob"] = tott["DOB"].apply(parse_dob)
    tott["fighter_url"] = tott["URL"]

    # details has: FIRST, LAST, NICKNAME, URL
    details["fighter_id"] = details["URL"].apply(extract_url_hex)
    details["first_name"] = details.get("FIRST", "")
    details["last_name"] = details.get("LAST", "")
    details["nickname"] = details.get("NICKNAME", "")

    # Outer join so details-only fighters (new debutants without TOTT stats) are included
    merged = pd.merge(
        tott[["fighter_id", "fighter_name", "height_in", "weight_lbs",
              "reach_in", "stance", "dob", "fighter_url"]],
        details[["fighter_id", "first_name", "last_name", "nickname"]],
        on="fighter_id",
        how="outer",
    )
    # details wins whenever it has a non-empty name: ufc_fighter_tott.csv is
    # append-only (upstream scraper only fetches URLs it hasn't parsed before),
    # so a stale/wrong name there for a recycled fighter ID never self-corrects.
    # ufc_fighter_details.csv is fully rewritten from the live roster pages each
    # run, so it reflects the current name. Fall back to tott's FIGHTER when
    # details has no name (e.g. tott-only fighter with no details row).
    details_name = (
        merged["first_name"].fillna("") + " " + merged["last_name"].fillna("")
    ).str.strip()
    mask_details_name = details_name != ""
    merged["fighter_name"] = merged["fighter_name"].where(
        ~mask_details_name, details_name
    )
    merged["fighter_name"] = merged["fighter_name"].fillna(details_name)
    merged = merged.dropna(subset=["fighter_id"])
    merged = merged.drop_duplicates(subset=["fighter_id"])
    return merged
