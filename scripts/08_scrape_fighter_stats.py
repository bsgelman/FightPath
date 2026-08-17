"""Scrape career-average stats from ufcstats.com fighter profile pages.

Writes data/raw/scraper/ufc_fighter_career_stats.csv used by the inference
layer to fill thin-data fighters' missing rolling features (see matchup.py).

Usage:
    python scripts/08_scrape_fighter_stats.py                  # skip already-scraped
    python scripts/08_scrape_fighter_stats.py --recent-days 14 # also re-scrape recent fighters
    python scripts/08_scrape_fighter_stats.py --refresh        # re-scrape all
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ufc.io import paths
from ufc.ingest.scrape_upcoming import _make_session, _solve_get
from ufc.ingest.scrape_fighter_stats import scrape_all_fighter_stats


def _recent_fighter_ids(days: int) -> set[str]:
    """Return fighter_ids who fought within the last `days` days (from ledger)."""
    try:
        import pandas as pd
        ledger_path = paths.processed("ledger")
        ledger = pd.read_parquet(ledger_path, columns=["fighter_id", "event_date"])
        cutoff = date.today() - timedelta(days=days)
        ledger["event_date"] = pd.to_datetime(ledger["event_date"]).dt.date
        recent = ledger[ledger["event_date"] >= cutoff]
        ids = set(recent["fighter_id"].astype(str).unique())
        print(f"  Recent fighters (last {days} days): {len(ids)}")
        return ids
    except Exception as exc:
        print(f"  [warn] could not load ledger for recent-days filter: {exc}")
        return set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="Re-scrape all fighters, not just new ones")
    parser.add_argument("--recent-days", type=int, default=0,
                        help="Also re-scrape fighters who fought within this many days (e.g. 14)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between requests (default 0.5)")
    args = parser.parse_args()

    fighter_csv = paths.raw_scraper() / "ufc_fighter_details.csv"
    output_csv = paths.raw_scraper() / "ufc_fighter_career_stats.csv"

    if not fighter_csv.exists():
        print(f"ERROR: {fighter_csv} not found. Run scripts/01_ingest.py first.")
        sys.exit(1)

    force_ids: set[str] = set()
    if args.recent_days > 0 and not args.refresh:
        force_ids = _recent_fighter_ids(args.recent_days)

    print("=== Scraping fighter career stats ===")
    print(f"  Source: {fighter_csv}")
    print(f"  Output: {output_csv}")
    print(f"  Refresh: {args.refresh}, recent-days: {args.recent_days}, delay: {args.delay}s")

    session = _make_session()
    df = scrape_all_fighter_stats(
        fighter_details_csv=fighter_csv,
        output_csv=output_csv,
        solve_get=_solve_get,
        session=session,
        delay=args.delay,
        refresh=args.refresh,
        force_ids=force_ids,
    )
    print(f"\nDone. {len(df)} fighters with career stats.")


if __name__ == "__main__":
    main()
