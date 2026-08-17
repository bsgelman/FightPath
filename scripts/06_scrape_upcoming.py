"""Scrape upcoming UFC cards from ufcstats.com and write to cards/upcoming/.

Run: python scripts/06_scrape_upcoming.py [--limit N] [--out DIR]
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.ingest.scrape_upcoming import scrape_upcoming_cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape upcoming UFC cards.")
    parser.add_argument("--limit", type=int, default=5,
                        help="Max number of upcoming events to scrape (default: 5).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: cards/upcoming/).")
    args = parser.parse_args()

    print(f"Scraping up to {args.limit} upcoming UFC events…")
    written = scrape_upcoming_cards(limit=args.limit, out_dir=args.out)

    if not written:
        print("No upcoming events found.")
        return

    for path in written:
        import json
        with open(path) as f:
            card = json.load(f)
        n_fights = len(card.get("matchups", []))
        print(f"  [ok] {card['event_name']}  ({card['event_date']})  - {n_fights} fights  ->  {path.name}")

    print(f"\nDone. {len(written)} card(s) written to {written[0].parent}.")


if __name__ == "__main__":
    main()
