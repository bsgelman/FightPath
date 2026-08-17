"""Tests for valuation engine: payouts, edge calculation, Kelly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
import math
from ufc.valuation.payouts import get_payout_multiplier, get_odds_type_multiplier, implied_prob_per_leg
from ufc.valuation.edge import evaluate_line, Edge
from ufc.valuation.lines import Line


class MockCDF:
    """Simple mock CDF for testing."""
    def __init__(self, median=50.0, spread=10.0):
        self.median = median
        self.spread = spread

    def cdf(self, x):
        # Simple sigmoid-ish approximation
        import math
        z = (x - self.median) / self.spread
        return 1.0 / (1 + math.exp(-z * 2))

    def p_over(self, x):
        return 1.0 - self.cdf(x)

    def p_under(self, x):
        return self.cdf(x - 0.5)

    def uncertainty_band(self, x):
        p = self.p_over(x)
        se = (p * (1-p)) ** 0.5 / 30
        return (max(0, p - 1.28*se), min(1, p + 1.28*se))


class TestPayouts:
    def test_powerplay_2pick(self):
        mult = get_payout_multiplier("powerplay_power_2pick")
        assert abs(mult - 3.0) < 1e-6

    def test_powerplay_3pick(self):
        mult = get_payout_multiplier("powerplay_power_3pick")
        assert abs(mult - 5.0) < 1e-6

    def test_implied_prob_2pick(self):
        # For 2-pick 3x: implied per leg = 3^(-1/2) ≈ 0.5774
        p = implied_prob_per_leg("powerplay_power_2pick", n_legs=2)
        assert abs(p - 3.0**(-1/2)) < 1e-6

    def test_implied_prob_3pick(self):
        # For 3-pick 5x: implied per leg = 5^(-1/3) ≈ 0.5848
        p = implied_prob_per_leg("powerplay_power_3pick", n_legs=3)
        assert abs(p - 5.0**(-1/3)) < 1e-6

    def test_get_odds_type_multiplier_demon(self):
        m = get_odds_type_multiplier("powerplay", "demon")
        assert m is not None
        assert m > 0

    def test_get_odds_type_multiplier_goblin(self):
        m = get_odds_type_multiplier("powerplay", "goblin")
        assert m is not None
        assert m > 0

    def test_get_odds_type_multiplier_boost(self):
        m = get_odds_type_multiplier("flatmulti", "boost")
        assert m is not None
        assert m > 0

    def test_get_odds_type_multiplier_unknown(self):
        m = get_odds_type_multiplier("powerplay", "unknown_type")
        assert m is None

    def test_get_odds_type_multiplier_standard_absent(self):
        # "standard" is not a keyed entry in config (uses N-pick structure)
        m = get_odds_type_multiplier("powerplay", "standard")
        assert m is None


class TestEdge:
    def test_no_edge_at_median(self):
        """Line at model median → model_prob ≈ 0.50 → negative edge for Power Play 2-pick."""
        cdf = MockCDF(median=52.5, spread=10)
        line = Line(
            market="sig_strikes", side="over", line_value=52.5,
            payout_type="powerplay_power_2pick", payout_multiplier=3.0,
            fighter_id="fa",
        )
        edge = evaluate_line(line, cdf)
        # model_prob ≈ 0.50, implied ≈ 0.577
        assert edge.model_prob < 0.55
        assert edge.edge_pct < 0  # negative edge, no bet
        assert edge.kelly_fraction == 0.0  # Kelly caps at 0

    def test_positive_edge_below_median(self):
        """Line well below median → P(over) > 0.7 → positive edge."""
        cdf = MockCDF(median=60.0, spread=10)
        line = Line(
            market="sig_strikes", side="over", line_value=45.0,
            payout_type="powerplay_power_2pick", payout_multiplier=3.0,
            fighter_id="fa",
        )
        edge = evaluate_line(line, cdf)
        assert edge.model_prob > 0.6
        assert edge.edge_pct > 0  # positive edge

    def test_kelly_cap_respected(self):
        """Kelly fraction never exceeds kelly_cap."""
        from ufc.valuation.payouts import kelly_cap
        cdf = MockCDF(median=20.0, spread=5)  # line at 50 → near-certain over
        line = Line(
            market="sig_strikes", side="over", line_value=50.0,
            payout_type="powerplay_power_2pick", payout_multiplier=3.0,
            fighter_id="fa",
        )
        edge = evaluate_line(line, cdf)
        # Very negative edge — kelly should be 0
        assert edge.kelly_fraction >= 0
        assert edge.kelly_fraction <= kelly_cap()

    def test_confidence_band_valid(self):
        cdf = MockCDF(median=50.0, spread=10)
        line = Line(
            market="sig_strikes", side="over", line_value=52.5,
            payout_type="powerplay_power_3pick", payout_multiplier=5.0,
            fighter_id="fa",
        )
        edge = evaluate_line(line, cdf)
        lo, hi = edge.confidence_band
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= edge.model_prob <= hi + 0.1  # within band
