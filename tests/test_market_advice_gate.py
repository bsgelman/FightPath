"""Tests for the per-kind Kalshi PAPER/LIVE gate lookup in ufc.api.app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.api.app import _advice_paper_for_kind


class TestAdvicePaperForKind:
    def test_exact_kind_match(self):
        cfg = {"status": "paper", "status_by_kind": {"winner": "live"}}
        assert _advice_paper_for_kind(cfg, "winner") is False

    def test_falls_back_to_family_for_round_indexed_kind(self):
        cfg = {"status": "paper", "status_by_kind": {"end_before_r": "live"}}
        assert _advice_paper_for_kind(cfg, "end_before_r2") is False
        assert _advice_paper_for_kind(cfg, "end_before_r5") is False

    def test_falls_back_to_global_status_when_kind_and_family_absent(self):
        cfg = {"status": "live", "status_by_kind": {"winner": "paper"}}
        assert _advice_paper_for_kind(cfg, "distance") is False
        assert _advice_paper_for_kind(cfg, "winner") is True

    def test_defaults_to_paper_on_empty_config(self):
        assert _advice_paper_for_kind({}, "winner") is True

    def test_unknown_status_value_fails_closed_to_paper(self):
        cfg = {"status_by_kind": {"winner": "yolo"}}
        assert _advice_paper_for_kind(cfg, "winner") is True

    def test_win_in_family_independent_of_end_before_family(self):
        cfg = {"status": "paper", "status_by_kind": {"win_in_r": "live"}}
        assert _advice_paper_for_kind(cfg, "win_in_r3") is False
        assert _advice_paper_for_kind(cfg, "end_before_r3") is True  # not live'd

    def test_case_insensitive_live(self):
        cfg = {"status_by_kind": {"winner": "LIVE"}}
        assert _advice_paper_for_kind(cfg, "winner") is False
