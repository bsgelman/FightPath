"""Tests for prop CDF objects: monotonicity, bounds, P(over)+P(under)≈1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pytest
from ufc.models.props_count import (
    PropCDF, QUANTILE_GRID, RateXDurationCDF,
    RateHurdleCountModel, predict_combined_count_cdf,
)


class TestPropCDF:
    def _make_cdf(self, median=50.0) -> PropCDF:
        """Create a reasonable CDF from linearly spaced quantiles."""
        qv = np.linspace(median - 30, median + 30, len(QUANTILE_GRID))
        return PropCDF(qv, np.array(QUANTILE_GRID), p_pos=1.0)

    def test_cdf_monotone(self):
        cdf = self._make_cdf(50)
        prev = -1.0
        for x in range(0, 150, 5):
            curr = cdf.cdf(x)
            assert curr >= prev - 1e-9, f"CDF not monotone at x={x}: {curr} < {prev}"
            prev = curr

    def test_cdf_bounds(self):
        cdf = self._make_cdf(50)
        for x in range(0, 200, 10):
            p = cdf.cdf(x)
            assert 0.0 <= p <= 1.0, f"CDF out of bounds at x={x}: {p}"

    def test_p_over_plus_cdf_approx_one(self):
        """P(X > x) + P(X <= x) ≈ 1."""
        cdf = self._make_cdf(50)
        for x in [30, 50, 70]:
            total = cdf.p_over(x) + cdf.cdf(x)
            assert abs(total - 1.0) < 1e-9, f"p_over + cdf != 1 at x={x}: {total}"

    def test_quantile_0_5_is_median(self):
        """The 50th percentile quantile should be near the median."""
        cdf = self._make_cdf(median=50.0)
        q50 = cdf.quantile(0.5)
        assert abs(q50 - 50.0) < 5.0, f"Median ({q50}) not close to 50"

    def test_pav_monotonize_enforced(self):
        """Quantile values are monotone even if raw values are not."""
        # Supply non-monotone quantile values (25 elements to match QUANTILE_GRID)
        qv = np.array([40, 50, 45, 60, 55, 70, 65, 80, 75, 90,
                        95, 100, 105, 108, 110, 115, 120, 125, 130, 135,
                        140, 145, 142, 150, 155])
        cdf = PropCDF(qv, np.array(QUANTILE_GRID), p_pos=1.0)
        # After PAV, quantile fn should be monotone (proxy via cdf)
        prev = -1.0
        for x in range(0, 300, 10):
            curr = cdf.cdf(x)
            assert curr >= prev - 1e-9, f"CDF not monotone at x={x}: {curr} < {prev}"
            prev = curr

    def test_uncertainty_band_contains_prob(self):
        cdf = self._make_cdf(50)
        p = cdf.p_over(45)
        lo, hi = cdf.uncertainty_band(45)
        assert lo <= p <= hi, f"Probability {p} not in band [{lo}, {hi}]"


class TestRateXDurationCDF:
    def _make_rxd(self, mean=5.0, n=2000) -> RateXDurationCDF:
        rng = np.random.default_rng(42)
        samples = rng.poisson(mean, size=n).astype(float)
        return RateXDurationCDF(samples, p_zero=(samples == 0).mean())

    def test_cdf_monotone(self):
        cdf = self._make_rxd()
        prev = -1.0
        for x in range(0, 30):
            curr = cdf.cdf(x)
            assert curr >= prev - 1e-9
            prev = curr

    def test_p_over_sums_to_one(self):
        cdf = self._make_rxd()
        for x in [2, 5, 10]:
            assert abs(cdf.p_over(x) + cdf.cdf(x) - 1.0) < 1e-9


class TestRateCeiling:
    """rate_ceiling clips per-rate draws in RateHurdleCountModel."""

    def test_ceiling_attribute(self):
        m = RateHurdleCountModel(target="ctrl_time", rate_ceiling=60.0)
        assert m.rate_ceiling == 60.0

    def test_no_ceiling_default(self):
        m = RateHurdleCountModel(target="sig_strikes")
        assert m.rate_ceiling is None


