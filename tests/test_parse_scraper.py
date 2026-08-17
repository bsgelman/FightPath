"""Tests for parse_scraper.parse_fighters()."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from ufc.ingest import parse_scraper


def test_parse_fighters_details_name_wins_over_stale_tott(tmp_path, monkeypatch):
    # ufc_fighter_tott.csv is append-only and can hold a stale name for a
    # recycled fighter ID (e.g. real-world "Regina Malpica" -> "Tyrell
    # Fortune"). ufc_fighter_details.csv is rewritten from the live roster
    # each run, so it must win when both disagree.
    tott = pd.DataFrame([
        {"FIGHTER": "Regina Malpica", "HEIGHT": "6' 0\"", "WEIGHT": "185 lbs.",
         "REACH": "72\"", "STANCE": "Orthodox", "DOB": "Jan 01, 1990",
         "URL": "http://ufcstats.com/fighter-details/abc123"},
        {"FIGHTER": "John Doe", "HEIGHT": "5' 11\"", "WEIGHT": "155 lbs.",
         "REACH": "70\"", "STANCE": "Southpaw", "DOB": "Feb 02, 1991",
         "URL": "http://ufcstats.com/fighter-details/def456"},
    ])
    details = pd.DataFrame([
        {"FIRST": "Tyrell", "LAST": "Fortune", "NICKNAME": "",
         "URL": "http://ufcstats.com/fighter-details/abc123"},
    ])
    tott.to_csv(tmp_path / "ufc_fighter_tott.csv", index=False)
    details.to_csv(tmp_path / "ufc_fighter_details.csv", index=False)
    monkeypatch.setattr(parse_scraper, "_scraper_dir", lambda: tmp_path)

    merged = parse_scraper.parse_fighters()
    by_id = merged.set_index("fighter_id")["fighter_name"]

    assert by_id["abc123"] == "Tyrell Fortune"
    # No details row for def456 -> falls back to tott's FIGHTER.
    assert by_id["def456"] == "John Doe"
