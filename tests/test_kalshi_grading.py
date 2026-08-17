"""Tests for kalshi_grading: the predicate table settling all 11 Kalshi market
kinds against a resolved fight outcome. One fixture per real-world scenario,
asserted across every kind it touches — this table is the single source of
truth 08b_grade_props.py routes into, so it must be exhaustively correct."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from ufc.valuation.kalshi_grading import FightOutcome, KindSpec, kind_spec, settle


def _outcome(fighter_won=None, raw_method="U-DEC", end_round=None,
             scheduled_rounds=3, total_fight_sec=None, is_draw=False):
    return FightOutcome(
        fighter_won=fighter_won, raw_method=raw_method, end_round=end_round,
        scheduled_rounds=scheduled_rounds, total_fight_sec=total_fight_sec, is_draw=is_draw,
    )


def _hit(market_kind, o):
    """Convenience: return just the hit bool/None (status assumed 'resolved' unless None)."""
    realized, hit, status = settle(market_kind, o)
    return hit, status


# ---------------------------------------------------------------------------
# kind_spec — needs_fighter routing (also covers scenario #13: fight-level
# kinds must never need the named-fighter resolution 08b's _which_side does)
# ---------------------------------------------------------------------------

class TestKindSpec:
    @pytest.mark.parametrize("kind", [
        "winner", "method_ko", "method_sub", "method_dec", "win_in_r1", "win_in_r3",
    ])
    def test_needs_fighter_true(self, kind):
        assert kind_spec(kind).needs_fighter is True

    @pytest.mark.parametrize("kind", [
        "distance", "mof_ko", "mof_sub", "mof_dec", "vicround_other",
        "end_before_r2", "end_before_r3", "end_before_r5",
    ])
    def test_needs_fighter_false(self, kind):
        assert kind_spec(kind).needs_fighter is False

    def test_only_vicround_other_settles_yes_on_nc(self):
        assert kind_spec("vicround_other").nc_hit is True
        for kind in ("winner", "method_ko", "method_sub", "method_dec", "distance",
                     "mof_ko", "mof_sub", "mof_dec", "end_before_r2", "win_in_r1"):
            assert kind_spec(kind).nc_hit is None

    def test_unknown_kind_returns_none(self):
        assert kind_spec("not_a_real_kind") is None
        assert kind_spec("sig_strikes") is None  # DFS market names are not Kalshi kinds


# ---------------------------------------------------------------------------
# Scenario 1 — clean unanimous decision
# ---------------------------------------------------------------------------

class TestCleanDecision:
    def _winner_side(self):
        return _outcome(fighter_won=True, raw_method="U-DEC", end_round=3, scheduled_rounds=3)

    def _loser_side(self):
        return _outcome(fighter_won=False, raw_method="U-DEC", end_round=3, scheduled_rounds=3)

    def test_distance_yes(self):
        assert _hit("distance", self._winner_side()) == (True, "resolved")

    def test_mof_dec_yes(self):
        assert _hit("mof_dec", self._winner_side()) == (True, "resolved")

    def test_vicround_other_yes(self):
        assert _hit("vicround_other", self._winner_side()) == (True, "resolved")

    def test_end_before_no(self):
        assert _hit("end_before_r2", self._winner_side()) == (False, "resolved")
        assert _hit("end_before_r3", self._winner_side()) == (False, "resolved")

    def test_win_in_no_for_both_sides(self):
        assert _hit("win_in_r3", self._winner_side()) == (False, "resolved")
        assert _hit("win_in_r3", self._loser_side()) == (False, "resolved")

    def test_winner_yes_and_no(self):
        assert _hit("winner", self._winner_side()) == (True, "resolved")
        assert _hit("winner", self._loser_side()) == (False, "resolved")

    def test_mof_ko_and_mof_sub_no(self):
        assert _hit("mof_ko", self._winner_side()) == (False, "resolved")
        assert _hit("mof_sub", self._winner_side()) == (False, "resolved")


# ---------------------------------------------------------------------------
# Scenario 2 — majority draw
# ---------------------------------------------------------------------------

class TestDraw:
    def _draw(self):
        return _outcome(fighter_won=None, raw_method="M-DEC", is_draw=True, end_round=None)

    def test_winner_no(self):
        assert _hit("winner", self._draw()) == (False, "resolved")

    def test_method_dec_no(self):
        assert _hit("method_dec", self._draw()) == (False, "resolved")

    def test_distance_yes(self):
        assert _hit("distance", self._draw()) == (True, "resolved")

    def test_mof_dec_no(self):
        assert _hit("mof_dec", self._draw()) == (False, "resolved")

    def test_vicround_other_yes(self):
        assert _hit("vicround_other", self._draw()) == (True, "resolved")

    def test_not_pending(self):
        # A draw must never be left pending on the fighter-dependent kinds —
        # fighter_won=None here means "nobody", not "unresolved".
        _, _, status = settle("winner", self._draw())
        assert status == "resolved"


# ---------------------------------------------------------------------------
# Scenario 3 — no contest: void everywhere EXCEPT vicround_other (settles YES)
# ---------------------------------------------------------------------------

class TestNoContest:
    def _nc(self, fighter_won=None):
        return _outcome(fighter_won=fighter_won, raw_method="NC", end_round=2)

    @pytest.mark.parametrize("kind", [
        "winner", "method_ko", "method_sub", "method_dec", "distance",
        "mof_ko", "mof_sub", "mof_dec", "end_before_r2", "win_in_r1",
    ])
    def test_voids(self, kind):
        realized, hit, status = settle(kind, self._nc())
        assert (realized, hit, status) == (None, None, "void")

    def test_vicround_other_settles_yes(self):
        realized, hit, status = settle("vicround_other", self._nc())
        assert (realized, hit, status) == (1.0, True, "resolved")

    def test_nc_voids_even_when_fighter_resolved(self):
        # NC must void regardless of whether a name happened to resolve.
        realized, hit, status = settle("winner", self._nc(fighter_won=True))
        assert status == "void"


# ---------------------------------------------------------------------------
# Scenario 4 — DQ win in round 2 (the grading fix: DQ != method_dec)
# ---------------------------------------------------------------------------

class TestDisqualification:
    def _winner_side(self):
        return _outcome(fighter_won=True, raw_method="DQ", end_round=2, scheduled_rounds=3)

    def _loser_side(self):
        return _outcome(fighter_won=False, raw_method="DQ", end_round=2, scheduled_rounds=3)

    def test_method_ko_yes_method_dec_no(self):
        assert _hit("method_ko", self._winner_side()) == (True, "resolved")
        assert _hit("method_dec", self._winner_side()) == (False, "resolved")

    def test_mof_ko_yes(self):
        assert _hit("mof_ko", self._winner_side()) == (True, "resolved")

    def test_distance_no(self):
        assert _hit("distance", self._winner_side()) == (False, "resolved")

    def test_end_before_r2_no_r3_yes(self):
        assert _hit("end_before_r2", self._winner_side()) == (False, "resolved")
        assert _hit("end_before_r3", self._winner_side()) == (True, "resolved")

    def test_win_in_r2_winner_yes_loser_no(self):
        assert _hit("win_in_r2", self._winner_side()) == (True, "resolved")
        assert _hit("win_in_r2", self._loser_side()) == (False, "resolved")

    def test_vicround_other_no(self):
        assert _hit("vicround_other", self._winner_side()) == (False, "resolved")


# ---------------------------------------------------------------------------
# Scenario 5 — technical decision (counts as distance/OTHER, not a round win)
# ---------------------------------------------------------------------------

class TestTechnicalDecision:
    def _o(self):
        return _outcome(fighter_won=True, raw_method="U-DEC", end_round=4,
                         scheduled_rounds=5, total_fight_sec=1400.0)

    def test_distance_yes(self):
        assert _hit("distance", self._o()) == (True, "resolved")

    def test_vicround_other_yes(self):
        assert _hit("vicround_other", self._o()) == (True, "resolved")

    def test_end_before_all_no(self):
        for r in (2, 3, 4, 5):
            assert _hit(f"end_before_r{r}", self._o()) == (False, "resolved")


# ---------------------------------------------------------------------------
# Scenario 6 — buzzer KO at 5:00 of round 1
# ---------------------------------------------------------------------------

class TestBuzzerKoRound1:
    def _winner_side(self):
        return _outcome(fighter_won=True, raw_method="KO/TKO", end_round=1, scheduled_rounds=3)

    def test_end_before_r2_yes(self):
        assert _hit("end_before_r2", self._winner_side()) == (True, "resolved")

    def test_win_in_r1_yes_r2_no(self):
        assert _hit("win_in_r1", self._winner_side()) == (True, "resolved")
        assert _hit("win_in_r2", self._winner_side()) == (False, "resolved")


# ---------------------------------------------------------------------------
# Scenario 7 — between-rounds retirement (end_round 2)
# ---------------------------------------------------------------------------

class TestBetweenRoundsRetirement:
    def _winner_side(self):
        return _outcome(fighter_won=True, raw_method="KO/TKO", end_round=2, scheduled_rounds=3)

    def test_end_before_r3_yes(self):
        assert _hit("end_before_r3", self._winner_side()) == (True, "resolved")

    def test_win_in_r2_yes(self):
        assert _hit("win_in_r2", self._winner_side()) == (True, "resolved")


# ---------------------------------------------------------------------------
# Scenario 8 — 5-round main event, R4 submission
# ---------------------------------------------------------------------------

class TestFiveRoundSubR4:
    def _winner_side(self):
        return _outcome(fighter_won=True, raw_method="SUB", end_round=4, scheduled_rounds=5)

    def test_end_before_r2_and_r3_no(self):
        assert _hit("end_before_r2", self._winner_side()) == (False, "resolved")
        assert _hit("end_before_r3", self._winner_side()) == (False, "resolved")

    def test_win_in_r4_yes(self):
        assert _hit("win_in_r4", self._winner_side()) == (True, "resolved")

    def test_distance_no_mof_sub_yes(self):
        assert _hit("distance", self._winner_side()) == (False, "resolved")
        assert _hit("mof_sub", self._winner_side()) == (True, "resolved")


# ---------------------------------------------------------------------------
# Scenario 9 — 5-round main event goes to a decision
# ---------------------------------------------------------------------------

class TestFiveRoundDecision:
    def _o(self):
        return _outcome(fighter_won=True, raw_method="S-DEC", end_round=5,
                         scheduled_rounds=5, total_fight_sec=1500.0)

    def test_distance_yes(self):
        assert _hit("distance", self._o()) == (True, "resolved")

    def test_end_before_all_no(self):
        for r in (2, 3, 4, 5):
            assert _hit(f"end_before_r{r}", self._o()) == (False, "resolved")


# ---------------------------------------------------------------------------
# Scenario 10 — missing end_round on a finish (round-boundary markets pend;
# fight-level markets still grade)
# ---------------------------------------------------------------------------

class TestMissingEndRound:
    def _winner_side(self):
        return _outcome(fighter_won=True, raw_method="KO/TKO", end_round=None, scheduled_rounds=3)

    def test_end_before_pending(self):
        realized, hit, status = settle("end_before_r2", self._winner_side())
        assert (realized, hit, status) == (None, None, None)

    def test_win_in_pending(self):
        realized, hit, status = settle("win_in_r2", self._winner_side())
        assert (realized, hit, status) == (None, None, None)

    def test_distance_and_mof_still_grade(self):
        assert _hit("distance", self._winner_side()) == (False, "resolved")
        assert _hit("mof_ko", self._winner_side()) == (True, "resolved")


# ---------------------------------------------------------------------------
# Scenario 11 — data gap: won_a NaN but NOT a draw (raw method is KO/TKO)
# ---------------------------------------------------------------------------

class TestDataGapNotADraw:
    def _o(self):
        return _outcome(fighter_won=None, raw_method="KO/TKO", is_draw=False, end_round=1)

    def test_winner_and_method_and_win_in_pending(self):
        for kind in ("winner", "method_ko", "method_sub", "method_dec", "win_in_r1"):
            realized, hit, status = settle(kind, self._o())
            assert (realized, hit, status) == (None, None, None), kind

    def test_mof_ko_yes_distance_no(self):
        assert _hit("mof_ko", self._o()) == (True, "resolved")
        assert _hit("distance", self._o()) == (False, "resolved")


# ---------------------------------------------------------------------------
# Scenario 12 — unknown market_kind never raises
# ---------------------------------------------------------------------------

class TestUnknownKind:
    def test_returns_none_triple(self):
        o = _outcome(fighter_won=True, raw_method="KO/TKO", end_round=1)
        assert settle("some_future_kalshi_market", o) == (None, None, None)
