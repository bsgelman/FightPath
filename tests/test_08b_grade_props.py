"""Integration tests for scripts/08b_grade_props.py's _resolve_row routing:
Kalshi kinds route through kalshi_grading.settle(); the DFS lane (count props,
duration, finish family) must be byte-identical to before that routing was
added — pinned here so a future change to the Kalshi table can't silently
regress the shared ledger's DFS rows."""
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

grade_props = importlib.import_module("08b_grade_props")


def _fres(**overrides):
    base = dict(
        name_a="Fighter A", name_b="Fighter B",
        won_a=True, method="U-DEC", method_class="DEC",
        end_round=3, scheduled_rounds=3, total_fight_sec=900.0,
        stats={"sig_strikes": {"a": 80.0, "b": 60.0}},
    )
    base.update(overrides)
    return base


def _row(market, fighter="Fighter A", corner="red", side="over", line_value=0.5):
    return pd.Series({
        "market": market, "fighter": fighter, "corner": corner,
        "side": side, "line_value": line_value,
    })


class TestKalshiRoutingIntegration:
    def test_dq_hits_method_ko_not_method_dec(self):
        fres = _fres(method="DQ", method_class="DEC", end_round=2)
        realized, hit, status = grade_props._resolve_row(_row("method_ko"), fres)
        assert (hit, status) == (True, "resolved")
        realized, hit, status = grade_props._resolve_row(_row("method_dec"), fres)
        assert (hit, status) == (False, "resolved")

    def test_vicround_other_settles_yes_on_no_contest(self):
        fres = _fres(method="NC", method_class="DEC", won_a=None)
        realized, hit, status = grade_props._resolve_row(_row("vicround_other", fighter=""), fres)
        assert (realized, hit, status) == (1.0, True, "resolved")

    def test_winner_voids_on_no_contest(self):
        fres = _fres(method="NC", method_class="DEC", won_a=None)
        realized, hit, status = grade_props._resolve_row(_row("winner"), fres)
        assert (realized, hit, status) == (None, None, "void")

    def test_fight_level_row_never_calls_which_side(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("_which_side must not be called for a fight-level kind")
        monkeypatch.setattr(grade_props, "_which_side", _boom)
        fres = _fres(method="U-DEC", method_class="DEC", end_round=3)
        realized, hit, status = grade_props._resolve_row(_row("distance", fighter=""), fres)
        assert status == "resolved"

    def test_unknown_kalshi_style_kind_falls_through_to_pending(self):
        fres = _fres()
        assert grade_props._resolve_row(_row("not_a_kind"), fres) == (None, None, None)


class TestDfsLaneRegressionPin:
    """These assertions describe the DFS lane's behavior as it existed BEFORE the
    Kalshi routing was added — they must keep passing unchanged."""

    def test_duration_fight_level_over(self):
        fres = _fres(total_fight_sec=950.0)
        realized, hit, status = grade_props._resolve_row(
            _row("duration", side="over", line_value=900.0), fres)
        assert (realized, hit, status) == (950.0, True, "resolved")

    def test_duration_under(self):
        fres = _fres(total_fight_sec=850.0)
        realized, hit, status = grade_props._resolve_row(
            _row("duration", side="under", line_value=900.0), fres)
        assert (realized, hit, status) == (850.0, True, "resolved")

    def test_count_prop_sig_strikes(self):
        fres = _fres()
        realized, hit, status = grade_props._resolve_row(
            _row("sig_strikes", fighter="Fighter A", side="over", line_value=70.0), fres)
        assert (realized, hit, status) == (80.0, True, "resolved")

    def test_finish_family_ko_finish(self):
        fres = _fres(method="KO/TKO", method_class="KO/TKO", won_a=True)
        realized, hit, status = grade_props._resolve_row(
            _row("ko_finish", fighter="Fighter A", side="over", line_value=0.5), fres)
        assert (realized, hit, status) == (1.0, True, "resolved")

    def test_finish_family_voids_on_no_contest(self):
        fres = _fres(method="NC", method_class="DEC", won_a=None)
        realized, hit, status = grade_props._resolve_row(
            _row("ko_finish", fighter="Fighter A"), fres)
        assert (realized, hit, status) == (None, None, "void")

    def test_r2_finish_round_match(self):
        fres = _fres(method="SUB", method_class="SUB", won_a=True, end_round=2)
        realized, hit, status = grade_props._resolve_row(
            _row("r2_finish", fighter="Fighter A", side="over", line_value=0.5), fres)
        assert (realized, hit, status) == (1.0, True, "resolved")

    def test_unknown_market_still_pending(self):
        fres = _fres()
        assert grade_props._resolve_row(_row("totally_unknown_dfs_market"), fres) == (None, None, None)
