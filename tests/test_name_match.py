"""Tests for fighter name normalization and matching."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
import pandas as pd
from ufc.ingest.parse_helpers import normalize_name
from ufc.ingest.name_match import build_lookup, resolve_name


def _make_fighters() -> pd.DataFrame:
    return pd.DataFrame([
        {"fighter_id": "id_001", "fighter_name": "Conor McGregor"},
        {"fighter_id": "id_002", "fighter_name": "Khabib Nurmagomedov"},
        {"fighter_id": "id_003", "fighter_name": "Anderson Silva"},
        {"fighter_id": "id_004", "fighter_name": "Jose Aldo"},
        {"fighter_id": "id_005", "fighter_name": "Michael Johnson"},
        {"fighter_id": "id_006", "fighter_name": "Michael Johnson"},  # duplicate
        {"fighter_id": "id_007", "fighter_name": "Jon Jones"},
    ])


class TestNormalizeName:
    def test_basic(self):
        assert normalize_name("Conor McGregor") == "conor mcgregor"

    def test_accent_stripping(self):
        assert normalize_name("José Aldo") == "jose aldo"

    def test_extra_whitespace(self):
        assert normalize_name("  Jon  Jones  ") == "jon jones"

    def test_punctuation(self):
        assert normalize_name("T.J. Dillashaw") == "tj dillashaw"


class TestBuildLookup:
    def test_unique_names(self):
        fighters = _make_fighters()
        lookup = build_lookup(fighters)
        assert "conor mcgregor" in lookup
        assert lookup["conor mcgregor"] == ["id_001"]

    def test_duplicate_names(self):
        fighters = _make_fighters()
        lookup = build_lookup(fighters)
        # Both Michael Johnsons should be in the list
        mj = lookup.get("michael johnson", [])
        assert len(mj) == 2


class TestResolveName:
    def test_simple_resolve(self):
        fighters = _make_fighters()
        lookup = build_lookup(fighters)
        fid = resolve_name("Conor McGregor", None, lookup, fighters)
        assert fid == "id_001"

    def test_accent_resolve(self):
        fighters = _make_fighters()
        lookup = build_lookup(fighters)
        fid = resolve_name("Jose Aldo", None, lookup, fighters)
        assert fid == "id_004"

    def test_unknown_name_returns_none(self):
        fighters = _make_fighters()
        lookup = build_lookup(fighters)
        fid = resolve_name("Completely Unknown Fighter", None, lookup, fighters)
        assert fid is None

    def test_override_applied(self):
        fighters = _make_fighters()
        lookup = build_lookup(fighters)
        overrides = {"khabib nurmagomedov": "id_002"}
        fid = resolve_name("Khabib Nurmagomedov", None, lookup, fighters, overrides=overrides)
        assert fid == "id_002"
