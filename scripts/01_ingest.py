"""Phase 1: Ingest raw data into typed parquet files and build the ledger.

Run: python scripts/01_ingest.py
Idempotent — safe to re-run; overwrites output parquets.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from ufc.io import paths, parquet
from ufc.ingest import parse_scraper, parse_archive, name_match, build_ledger


def copy_raw_files():
    """Copy CSVs from their source folders into data/raw/."""
    import shutil

    scraper_src = paths.scraper_source()
    scraper_dst = paths.raw_scraper()
    scraper_dst.mkdir(parents=True, exist_ok=True)

    for csv in scraper_src.glob("*.csv"):
        dst = scraper_dst / csv.name
        if not dst.exists() or csv.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(csv, dst)
            print(f"  Copied {csv.name}")

    archive_src = paths.archive_source()
    archive_dst = paths.raw_archive()
    archive_dst.mkdir(parents=True, exist_ok=True)
    for csv in archive_src.glob("*.csv"):
        dst = archive_dst / csv.name
        if not dst.exists() or csv.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(csv, dst)
            print(f"  Copied {csv.name}")


def main():
    print("=== Phase 1: Ingestion ===")

    print("\n[0/5] Copying raw files...")
    copy_raw_files()

    print("\n[1/5] Parsing events...")
    events_df = parse_scraper.parse_events()
    print(f"  {len(events_df)} events, date range: {events_df['event_date'].min()} -> {events_df['event_date'].max()}")
    parquet.write(events_df, paths.interim("events"))

    print("\n[2/5] Parsing fights...")
    fights_df = parse_scraper.parse_fights()
    print(f"  {len(fights_df)} fights")
    parquet.write(fights_df, paths.interim("fights"))

    print("\n[3/5] Parsing fight rounds (round-by-round stats)...")
    rounds_df = parse_scraper.parse_fight_rounds()
    print(f"  {len(rounds_df)} round-fighter rows")
    parquet.write(rounds_df, paths.interim("fight_rounds"))

    print("\n[4/5] Parsing fighters...")
    fighters_df = parse_scraper.parse_fighters()
    print(f"  {len(fighters_df)} fighters")

    # Back-fill reach from archive
    print("  Back-filling reach from archive...")
    reach_lookup = parse_archive.get_reach_backfill()
    if not reach_lookup.empty:
        from ufc.ingest.parse_helpers import normalize_name
        fighters_df["norm_name"] = fighters_df["fighter_name"].apply(normalize_name)
        merged = pd.merge(
            fighters_df,
            reach_lookup[["norm_name", "reach_in", "height_in"]].rename(
                columns={"reach_in": "reach_archive", "height_in": "height_archive"}
            ),
            on="norm_name",
            how="left",
        )
        missing_reach = merged["reach_in"].isna() & merged["reach_archive"].notna()
        fighters_df.loc[missing_reach, "reach_in"] = merged.loc[missing_reach, "reach_archive"].values
        missing_height = fighters_df["height_in"].isna() & merged["height_archive"].notna()
        fighters_df.loc[missing_height, "height_in"] = merged.loc[missing_height, "height_archive"].values
        print(f"  Filled {missing_reach.sum()} reach values from archive")

    parquet.write(fighters_df, paths.interim("fighters"))

    print("\n[5/5] Resolving fighter names and building ledger...")
    name_map_df = name_match.build_name_map(fights_df, fighters_df, events_df)
    print(f"  Name map: {name_map_df['fighter_a_id'].notna().sum()}/{len(name_map_df)} fighter_a resolved, "
          f"{name_map_df['fighter_b_id'].notna().sum()}/{len(name_map_df)} fighter_b resolved")
    parquet.write(name_map_df, paths.interim("name_map"))

    ledger = build_ledger.build_ledger(events_df, fights_df, rounds_df, fighters_df, name_map_df)
    ledger = build_ledger.backfill_reach(ledger, reach_lookup)
    print(f"  Ledger: {len(ledger)} rows ({len(ledger)//2} fights)")
    print(f"  Date range: {ledger['event_date'].min()} -> {ledger['event_date'].max()}")
    print(f"  Fighters with won label: {ledger['won'].notna().sum()}")

    paths.processed("ledger").parent.mkdir(parents=True, exist_ok=True)
    parquet.write(ledger, paths.processed("ledger"))
    print(f"\n  Ledger written to {paths.processed('ledger')}")
    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    main()
