"""Scrape per-fighter career-average stat blocks from ufcstats.com fighter profile pages.

Uses the existing PoW solver (_solve_get) from scrape_upcoming so the session
cookie is shared automatically.  Writes data/raw/scraper/ufc_fighter_career_stats.csv
with one row per fighter_id (URL key) and 8 career-average columns:

    slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg

Units match pre_fight_state rolling-window features so inference can fill
NaN values directly (accuracy/defense are fractions 0-1; rates are per-minute
or per-15-min as labelled).

Run via scripts/08_scrape_fighter_stats.py — idempotent, skips fighters whose
row already exists unless --refresh is passed.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LABEL_MAP: dict[str, str] = {
    "slpm":     "slpm",
    "str. acc": "str_acc",
    "sapm":     "sapm",
    "str. def": "str_def",
    "td avg":   "td_avg",
    "td acc":   "td_acc",
    "td def":   "td_def",
    "sub. avg": "sub_avg",
}

_CAREER_CSV_NAME = "ufc_fighter_career_stats.csv"
_CAREER_COLS = ["fighter_id", "slpm", "str_acc", "sapm", "str_def",
                "td_avg", "td_acc", "td_def", "sub_avg",
                "wins_total", "losses_total", "draws_total"]


# ---------------------------------------------------------------------------
# HTML parser (network-free, testable)
# ---------------------------------------------------------------------------

def parse_career_stats(html: str) -> dict[str, float]:
    """Extract career-average stats from a ufcstats fighter profile page HTML.

    Returns a dict with keys from _LABEL_MAP values.  Missing/unparseable
    fields are absent from the dict (caller decides the fallback).
    """
    soup = BeautifulSoup(html, "html.parser")
    stats: dict[str, float] = {}

    # Official MMA record (all promotions) from title-record span
    rec_span = soup.find(class_="b-content__title-record")
    if rec_span:
        rec_text = rec_span.get_text(" ", strip=True)
    else:
        rec_text = html  # fallback: search raw HTML
    m = re.search(r"Record:\s*(\d+)-(\d+)-(\d+)", rec_text)
    if not m:
        m = re.search(r"Record:\s*(\d+)-(\d+)", rec_text)
    if m:
        stats["wins_total"]   = float(m.group(1))
        stats["losses_total"] = float(m.group(2))
        stats["draws_total"]  = float(m.group(3)) if len(m.groups()) >= 3 else 0.0

    for li in soup.find_all("li"):
        text = li.get_text(" ", strip=True)
        if ":" not in text:
            continue
        label_raw, _, value_raw = text.partition(":")
        label_norm = label_raw.strip().lower().rstrip(".")
        # map label
        canonical: str | None = None
        for key, col in _LABEL_MAP.items():
            if label_norm.startswith(key):
                canonical = col
                break
        if canonical is None:
            continue
        value_clean = value_raw.strip().replace("%", "").replace("--", "").strip()
        if not value_clean:
            continue
        try:
            val = float(value_clean)
        except ValueError:
            continue
        # accuracy/defense columns come as percentages (e.g. 51) → convert to fraction
        if canonical in ("str_acc", "str_def", "td_acc", "td_def"):
            val = val / 100.0
        stats[canonical] = val
    return stats


# ---------------------------------------------------------------------------
# Scrape helpers
# ---------------------------------------------------------------------------

def _fighter_id_from_url(url: str) -> str:
    """Extract the hex fighter_id from a ufcstats fighter-details URL."""
    m = re.search(r"/fighter-details/([0-9a-f]+)", url)
    return m.group(1) if m else ""


def scrape_all_fighter_stats(
    fighter_details_csv: Path,
    output_csv: Path,
    solve_get,                  # callable(session, url) -> html str
    session: requests.Session,
    delay: float = 0.5,
    refresh: bool = False,
    force_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Scrape career-average stats for every fighter in fighter_details_csv.

    Args:
        fighter_details_csv: path to ufc_fighter_details.csv (FIRST,LAST,NICKNAME,URL)
        output_csv: where to write/update the results CSV
        solve_get: PoW-solving GET function from scrape_upcoming
        session: requests.Session carrying the PoW clearance cookie
        delay: seconds to sleep between requests (be polite)
        refresh: if True, re-scrape even fighters whose row already exists
        force_ids: fighter_ids to re-scrape even if already in the CSV (e.g. recently fought)

    Returns the full DataFrame including any previously scraped rows.
    """
    details = pd.read_csv(fighter_details_csv)
    details["fighter_id"] = details["URL"].apply(_fighter_id_from_url)
    details = details[details["fighter_id"] != ""].copy()

    # Load existing results (if any)
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
    else:
        existing = pd.DataFrame(columns=_CAREER_COLS)

    _force = force_ids or set()
    existing_ids: set[str] = (set(existing["fighter_id"].astype(str)) - _force) if not refresh else set()
    to_scrape = details[~details["fighter_id"].isin(existing_ids)].copy()
    if _force:
        print(f"  Force-rescraping {len(_force & set(details['fighter_id']))} recently-fought fighters")

    print(f"  Fighters to scrape: {len(to_scrape)}  (already have: {len(existing)})")

    new_rows: list[dict[str, Any]] = []
    for i, row in enumerate(to_scrape.itertuples(index=False), 1):
        fid = row.fighter_id
        url = row.URL
        if not url or pd.isna(url):
            continue
        try:
            html = solve_get(session, url)
            stats = parse_career_stats(html)
            stats["fighter_id"] = fid
            new_rows.append(stats)
        except Exception as exc:
            print(f"  [warn] {fid} ({url}): {exc}")
        if i % 50 == 0:
            print(f"    {i}/{len(to_scrape)} done...")
        time.sleep(delay)

    if new_rows:
        new_df = pd.DataFrame(new_rows)[_CAREER_COLS]
        combined = pd.concat([existing, new_df], ignore_index=True)
        # deduplicate on fighter_id, keep last (fresh scrape wins)
        combined = combined.drop_duplicates(subset=["fighter_id"], keep="last")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_csv, index=False)
        print(f"  Saved {len(combined)} fighters -> {output_csv}")
        return combined
    else:
        print("  No new fighters to scrape.")
        return existing
