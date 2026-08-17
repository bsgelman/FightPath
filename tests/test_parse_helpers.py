"""Tests for parse_helpers.py pure functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
from datetime import date
from ufc.ingest.parse_helpers import (
    parse_x_of_y, parse_mm_ss, parse_height, parse_weight, parse_reach,
    parse_dob, parse_event_date, parse_pct, parse_scheduled_rounds,
    normalize_name, extract_url_hex, normalize_method, normalize_stance,
)


def test_parse_x_of_y_normal():
    assert parse_x_of_y("9 of 12") == (9, 12)
    assert parse_x_of_y("0 of 0") == (0, 0)
    assert parse_x_of_y("100 of 200") == (100, 200)


def test_parse_x_of_y_missing():
    assert parse_x_of_y("---") == (0, 0)
    assert parse_x_of_y("--") == (0, 0)
    assert parse_x_of_y(None) == (0, 0)
    assert parse_x_of_y("") == (0, 0)


def test_parse_mm_ss():
    assert parse_mm_ss("1:44") == 104
    assert parse_mm_ss("5:00") == 300
    assert parse_mm_ss("0:00") == 0
    assert parse_mm_ss("--") == 0
    assert parse_mm_ss(None) == 0


def test_parse_height():
    assert parse_height("5' 11\"") == 71.0
    assert parse_height("6' 0\"") == 72.0
    assert parse_height("--") is None
    assert parse_height(None) is None


def test_parse_weight():
    assert parse_weight("155 lbs.") == 155.0
    assert parse_weight("265 lbs.") == 265.0
    assert parse_weight("--") is None


def test_parse_reach():
    assert parse_reach("66\"") == 66.0
    assert parse_reach("72.5\"") == 72.5
    assert parse_reach("--") is None


def test_parse_dob():
    assert parse_dob("Jul 13, 1978") == date(1978, 7, 13)
    assert parse_dob("Jan 01, 1990") == date(1990, 1, 1)
    assert parse_dob("--") is None
    assert parse_dob(None) is None


def test_parse_event_date():
    assert parse_event_date("September 06, 2025") == date(2025, 9, 6)
    assert parse_event_date("2025/09/06") == date(2025, 9, 6)
    assert parse_event_date("Sep 6, 2025") == date(2025, 9, 6)


def test_parse_pct():
    assert parse_pct("100%") == pytest.approx(1.0)
    assert parse_pct("50%") == pytest.approx(0.5)
    assert parse_pct("---") is None
    assert parse_pct(None) is None


def test_parse_scheduled_rounds():
    assert parse_scheduled_rounds("3 Rnd (5-5-5)") == 3
    assert parse_scheduled_rounds("5 Rnd (5-5-5-5-5)") == 5
    assert parse_scheduled_rounds(None) == 3


def test_normalize_name():
    assert normalize_name("Ilia Topuria") == "ilia topuria"
    assert normalize_name("  Max  Holloway  ") == "max holloway"
    assert normalize_name("José Aldo") == "jose aldo"


def test_extract_url_hex():
    assert extract_url_hex("http://ufcstats.com/fighter-details/abc123def456") == "abc123def456"
    assert extract_url_hex("http://ufcstats.com/fight-details/xyz789") == "xyz789"
    assert extract_url_hex(None) is None


def test_normalize_method():
    assert normalize_method("KO/TKO") == "KO/TKO"
    assert normalize_method("Submission") == "SUB"
    assert normalize_method("Decision - Unanimous") == "U-DEC"
    assert normalize_method("Decision - Split") == "S-DEC"
    assert normalize_method("Decision - Majority") == "M-DEC"
    assert normalize_method("DQ") == "DQ"
    assert normalize_method("Overturned") == "NC"
    assert normalize_method(None) == "NC"


def test_normalize_stance():
    assert normalize_stance("Orthodox") == "ORTHO"
    assert normalize_stance("Southpaw") == "SOUTH"
    assert normalize_stance("Switch") == "SWITCH"
    assert normalize_stance("Open Stance") == "OPEN"
    assert normalize_stance(None) == "UNKNOWN"
