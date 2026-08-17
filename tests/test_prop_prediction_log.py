"""Tests for prop_prediction_log key-building (no I/O)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.inference.prop_prediction_log import kalshi_key, legacy_kalshi_key, prop_key


class TestKalshiKey:
    def test_two_fighters_same_ask_get_distinct_keys(self):
        # The bug this fixes: win_in_r{N} markets for both fighters often price
        # near the same few cents — without `corner` in the key, one row silently
        # overwrote the other.
        k_red = kalshi_key("2026-07-11", "Red", "Blue", "win_in_r1", "red", "taker", 0.03)
        k_blue = kalshi_key("2026-07-11", "Red", "Blue", "win_in_r1", "blue", "taker", 0.03)
        assert k_red != k_blue

    def test_taker_and_maker_get_distinct_keys(self):
        k_taker = kalshi_key("2026-07-11", "Red", "Blue", "winner", "red", "taker", 0.55)
        k_maker = kalshi_key("2026-07-11", "Red", "Blue", "winner", "red", "maker", 0.55)
        assert k_taker != k_maker

    def test_fight_level_kind_uses_fight_corner(self):
        k = kalshi_key("2026-07-11", "Red", "Blue", "distance", "fight", "taker", 0.32)
        assert "|fight|" in k

    def test_order_independent_on_red_blue(self):
        # matchup_key is order-independent; kalshi_key inherits that.
        k1 = kalshi_key("2026-07-11", "Red", "Blue", "winner", "red", "taker", 0.55)
        k2 = kalshi_key("2026-07-11", "Blue", "Red", "winner", "red", "taker", 0.55)
        assert k1 == k2

    def test_differs_from_legacy_key(self):
        new = kalshi_key("2026-07-11", "Red", "Blue", "winner", "red", "taker", 0.55)
        old = legacy_kalshi_key("2026-07-11", "Red", "Blue", "winner", "taker", 0.55)
        assert new != old

    def test_differs_from_prop_key_shape(self):
        # Sanity: kalshi_key must not collide with a DFS prop_key for an
        # unrelated market/side/line combination.
        pk = prop_key("2026-07-11", "Red", "Blue", "sig_strikes", "over", 75.5)
        kk = kalshi_key("2026-07-11", "Red", "Blue", "winner", "red", "taker", 0.55)
        assert pk != kk


class TestPropKey:
    def test_two_fighters_same_finish_line_get_distinct_keys(self):
        # The bug this fixes (A6): finish props are always logged at
        # line_value=0.5 for BOTH fighters, so without `corner` in the key
        # one fighter's row silently overwrote the other on first-write-wins
        # dedup (verified live: 81/81 finish groups collapsed to one corner).
        k_red = prop_key("2026-07-11", "Red", "Blue", "ko_finish", "over", 0.5, corner="red")
        k_blue = prop_key("2026-07-11", "Red", "Blue", "ko_finish", "over", 0.5, corner="blue")
        assert k_red != k_blue

    def test_same_line_different_platform_get_distinct_keys(self):
        k_pp = prop_key("2026-07-11", "Red", "Blue", "sig_strikes", "over", 19.5,
                         corner="red", platform="powerplay")
        k_ud = prop_key("2026-07-11", "Red", "Blue", "sig_strikes", "over", 19.5,
                         corner="red", platform="flatmulti")
        assert k_pp != k_ud

    def test_backward_compatible_without_corner_or_platform(self):
        # Callers that don't pass corner/platform (e.g. legacy code, existing
        # tests) still get a valid, stable key shape.
        k = prop_key("2026-07-11", "Red", "Blue", "sig_strikes", "over", 75.5)
        assert k == "2026-07-11|blue|red|sig_strikes|over|75.5000"
