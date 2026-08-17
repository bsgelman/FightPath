"""Tests for market_edge: Kalshi fee/breakeven/Kelly/liquidity math (no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

import numpy as np

from ufc.ingest.market_lines import MarketQuote, ResolvedMarketQuote
from ufc.valuation.market_edge import (
    kalshi_taker_fee,
    effective_cost_taker,
    effective_cost_maker,
    evaluate_market_quote,
    model_prob_for_quote,
    MarketEdge,
)


# ---------------------------------------------------------------------------
# kalshi_taker_fee
# ---------------------------------------------------------------------------

class TestKalshiTakerFee:
    def test_fee_at_50_cents_is_max(self):
        assert kalshi_taker_fee(0.50) == pytest.approx(0.0175)

    def test_fee_symmetric_10_and_90_cents(self):
        assert kalshi_taker_fee(0.10) == pytest.approx(kalshi_taker_fee(0.90))

    def test_fee_at_10_cents(self):
        assert kalshi_taker_fee(0.10) == pytest.approx(0.07 * 0.10 * 0.90)

    def test_fee_near_zero_at_extremes(self):
        assert kalshi_taker_fee(0.01) < kalshi_taker_fee(0.50)
        assert kalshi_taker_fee(0.99) < kalshi_taker_fee(0.50)

    def test_fee_never_negative(self):
        assert kalshi_taker_fee(0.01) >= 0
        assert kalshi_taker_fee(0.99) >= 0


# ---------------------------------------------------------------------------
# effective_cost_taker / effective_cost_maker
# ---------------------------------------------------------------------------

class TestEffectiveCost:
    def test_taker_cost_above_ask(self):
        ask = 0.56
        cost = effective_cost_taker(ask)
        assert cost > ask
        assert cost == pytest.approx(ask + kalshi_taker_fee(ask))

    def test_maker_cost_below_taker_cost(self):
        bid = 0.55
        maker = effective_cost_maker(bid)
        # taker cost at the corresponding ask (bid+1c) should be strictly higher
        taker = effective_cost_taker(bid + 0.01)
        assert maker < taker

    def test_maker_fee_is_quarter_of_taker(self):
        bid = 0.30
        post_price = bid + 0.01
        maker = effective_cost_maker(bid)
        expected = post_price + 0.25 * kalshi_taker_fee(post_price)
        assert maker == pytest.approx(expected)


# ---------------------------------------------------------------------------
# evaluate_market_quote
# ---------------------------------------------------------------------------

def _quote(yes_bid=0.55, yes_ask=0.56, depth_usd_3c=None):
    return MarketQuote(
        platform="kalshi", event_ticker="E", market_ticker="E-X", market_kind="winner",
        fighter_name="Test Fighter", yes_bid=yes_bid, yes_ask=yes_ask, last_price=yes_ask,
        volume=1000.0, open_interest=1000.0, depth_usd_3c=depth_usd_3c, fetched_at="",
    )


class TestEvaluateMarketQuote:
    def test_positive_edge_when_model_above_ask_plus_fee(self):
        edge = evaluate_market_quote(_quote(yes_ask=0.50), model_p=0.60)
        assert edge.edge_pct > 0
        assert edge.kelly > 0

    def test_zero_kelly_when_model_at_breakeven(self):
        # model exactly at the fee-adjusted breakeven -> zero edge, zero kelly
        ask = 0.50
        be = effective_cost_taker(ask)
        edge = evaluate_market_quote(_quote(yes_ask=ask), model_p=be)
        assert edge.kelly == 0.0

    def test_negative_edge_no_bet(self):
        edge = evaluate_market_quote(_quote(yes_ask=0.80), model_p=0.50)
        assert edge.edge_pct < 0
        assert edge.kelly == 0.0

    def test_kelly_capped(self):
        from ufc.valuation.payouts import kelly_cap
        edge = evaluate_market_quote(_quote(yes_ask=0.05), model_p=0.95)
        assert edge.kelly <= kelly_cap()

    def test_prices_against_ask_not_mid(self):
        # bid=0.40, ask=0.60 (wide spread) — breakeven must derive from ask=0.60, not mid=0.50
        edge = evaluate_market_quote(_quote(yes_bid=0.40, yes_ask=0.60), model_p=0.55)
        assert edge.breakeven == pytest.approx(effective_cost_taker(0.60))
        assert edge.edge_pct < 0  # 0.55 model < 0.60+fee breakeven

    def test_maker_breakeven_lower_than_taker(self):
        edge = evaluate_market_quote(_quote(yes_bid=0.55, yes_ask=0.56), model_p=0.60)
        assert edge.maker_breakeven < edge.breakeven

    def test_liq_tier_deep(self):
        edge = evaluate_market_quote(_quote(depth_usd_3c=15000.0), model_p=0.60)
        assert edge.liq_tier == "DEEP"

    def test_liq_tier_ok(self):
        edge = evaluate_market_quote(_quote(depth_usd_3c=5000.0), model_p=0.60)
        assert edge.liq_tier == "OK"

    def test_liq_tier_thin(self):
        edge = evaluate_market_quote(_quote(depth_usd_3c=500.0), model_p=0.60)
        assert edge.liq_tier == "THIN"

    def test_liq_tier_thin_when_depth_unknown(self):
        edge = evaluate_market_quote(_quote(depth_usd_3c=None), model_p=0.60)
        assert edge.liq_tier == "THIN"

    def test_stake_cap_passthrough(self):
        edge = evaluate_market_quote(_quote(depth_usd_3c=2500.0), model_p=0.60)
        assert edge.stake_cap_usd == pytest.approx(2500.0)


# ---------------------------------------------------------------------------
# model_prob_for_quote
# ---------------------------------------------------------------------------

class _FakePred:
    def __init__(self, prob_red=0.6, sim_samples=None):
        self.prob_red = prob_red
        self.prob_blue = 1.0 - prob_red
        self.sim_samples = sim_samples


def _resolved_quote(market_kind="winner", corner="red"):
    return ResolvedMarketQuote(
        platform="kalshi", event_ticker="E", market_ticker="E-X", market_kind=market_kind,
        fighter_name="Test Fighter", yes_bid=0.5, yes_ask=0.51, last_price=0.5,
        volume=100.0, open_interest=100.0, depth_usd_3c=None, fetched_at="",
        card_red="Red Fighter", card_blue="Blue Fighter", fight_idx=0, corner=corner,
    )


class TestModelProbForQuote:
    def test_winner_red_corner(self):
        pred = _FakePred(prob_red=0.7)
        p = model_prob_for_quote(_resolved_quote("winner", "red"), pred)
        assert p == pytest.approx(0.7)

    def test_winner_blue_corner(self):
        pred = _FakePred(prob_red=0.7)
        p = model_prob_for_quote(_resolved_quote("winner", "blue"), pred)
        assert p == pytest.approx(0.3)

    def test_method_without_sim_samples_returns_none(self):
        pred = _FakePred(sim_samples=None)
        p = model_prob_for_quote(_resolved_quote("method_ko", "red"), pred)
        assert p is None

    def test_method_ko_red_joint_probability(self):
        # 10 samples: red wins+KO in 3, red wins+DEC in 2, blue wins+KO in 1, rest DEC
        winner_a = np.array([True, True, True, True, False, False, False, False, False, False])
        method = np.array(["KO/TKO", "KO/TKO", "KO/TKO", "DEC", "KO/TKO", "DEC", "DEC", "DEC", "DEC", "DEC"])
        pred = _FakePred(sim_samples={"winner_a": winner_a, "method": method})
        p = model_prob_for_quote(_resolved_quote("method_ko", "red"), pred)
        assert p == pytest.approx(0.3)  # 3/10 red-wins-by-KO

    def test_method_ko_blue_joint_probability(self):
        winner_a = np.array([True, True, True, True, False, False, False, False, False, False])
        method = np.array(["KO/TKO", "KO/TKO", "KO/TKO", "DEC", "KO/TKO", "DEC", "DEC", "DEC", "DEC", "DEC"])
        pred = _FakePred(sim_samples={"winner_a": winner_a, "method": method})
        p = model_prob_for_quote(_resolved_quote("method_ko", "blue"), pred)
        assert p == pytest.approx(0.1)  # 1/10 blue-wins-by-KO

    def test_unknown_market_kind_returns_none(self):
        pred = _FakePred(sim_samples={"winner_a": np.array([True]), "method": np.array(["DEC"])})
        p = model_prob_for_quote(_resolved_quote("rounds", "red"), pred)
        assert p is None


# ---------------------------------------------------------------------------
# model_prob_for_quote — distance / mof_* / end_before_r{r} / win_in_r{r}
# ---------------------------------------------------------------------------

class _FakePredFull(_FakePred):
    """Adds method_probs + display_dur_cdf, the fallback-path attrs the new
    kinds need when sim_samples is unavailable."""
    def __init__(self, prob_red=0.6, sim_samples=None, method_probs=None, display_dur_cdf=None):
        super().__init__(prob_red=prob_red, sim_samples=sim_samples)
        self.method_probs = method_probs or {"KO/TKO": 0.35, "SUB": 0.15, "DEC": 0.50}
        self.display_dur_cdf = display_dur_cdf


def _sim(n_dec=5, n_ko=3, n_sub=2, red_win_frac_by_method=None):
    """Build a synthetic sim_samples dict with an exact, hand-countable composition."""
    methods = ["DEC"] * n_dec + ["KO/TKO"] * n_ko + ["SUB"] * n_sub
    # First half of each method-block is a red win (deterministic, hand-verifiable).
    winner_a, duration_sec = [], []
    for i, meth in enumerate(methods):
        winner_a.append(i % 2 == 0)
        if meth == "DEC":
            duration_sec.append(900.0)
        elif meth == "KO/TKO":
            duration_sec.append([120.0, 400.0, 750.0][i % 3])
        else:
            duration_sec.append([200.0, 500.0][i % 2])
    return {
        "winner_a": np.array(winner_a),
        "method": np.array(methods),
        "duration_sec": np.array(duration_sec),
    }


class TestModelProbForQuoteDistance:
    def test_distance_from_sim_is_dec_fraction(self):
        sim = _sim(n_dec=6, n_ko=2, n_sub=2)  # 6/10 DEC
        pred = _FakePredFull(sim_samples=sim)
        p = model_prob_for_quote(_resolved_quote("distance", "fight"), pred)
        assert p == pytest.approx(0.6)

    def test_distance_without_sim_falls_back_to_method_probs(self):
        pred = _FakePredFull(sim_samples=None, method_probs={"KO/TKO": 0.3, "SUB": 0.2, "DEC": 0.5})
        p = model_prob_for_quote(_resolved_quote("distance", "fight"), pred)
        assert p == pytest.approx(0.5)


class TestModelProbForQuoteVicroundOther:
    def test_vicround_other_is_dec_fraction_like_distance(self):
        sim = _sim(n_dec=7, n_ko=2, n_sub=1)
        pred = _FakePredFull(sim_samples=sim)
        p = model_prob_for_quote(_resolved_quote("vicround_other", "fight"), pred)
        assert p == pytest.approx(0.7)

    def test_vicround_other_without_sim_falls_back_to_method_probs(self):
        pred = _FakePredFull(sim_samples=None, method_probs={"KO/TKO": 0.4, "SUB": 0.1, "DEC": 0.5})
        p = model_prob_for_quote(_resolved_quote("vicround_other", "fight"), pred)
        assert p == pytest.approx(0.5)


class TestModelProbForQuoteMof:
    def test_mof_ko_from_sim(self):
        sim = _sim(n_dec=5, n_ko=3, n_sub=2)  # 3/10 KO
        pred = _FakePredFull(sim_samples=sim)
        p = model_prob_for_quote(_resolved_quote("mof_ko", "fight"), pred)
        assert p == pytest.approx(0.3)

    def test_mof_sub_from_sim(self):
        sim = _sim(n_dec=5, n_ko=3, n_sub=2)  # 2/10 SUB
        pred = _FakePredFull(sim_samples=sim)
        p = model_prob_for_quote(_resolved_quote("mof_sub", "fight"), pred)
        assert p == pytest.approx(0.2)

    def test_mof_dec_from_sim(self):
        sim = _sim(n_dec=5, n_ko=3, n_sub=2)
        pred = _FakePredFull(sim_samples=sim)
        p = model_prob_for_quote(_resolved_quote("mof_dec", "fight"), pred)
        assert p == pytest.approx(0.5)

    def test_mof_without_sim_falls_back_to_method_probs(self):
        pred = _FakePredFull(sim_samples=None, method_probs={"KO/TKO": 0.4, "SUB": 0.2, "DEC": 0.4})
        assert model_prob_for_quote(_resolved_quote("mof_ko", "fight"), pred) == pytest.approx(0.4)
        assert model_prob_for_quote(_resolved_quote("mof_sub", "fight"), pred) == pytest.approx(0.2)
        assert model_prob_for_quote(_resolved_quote("mof_dec", "fight"), pred) == pytest.approx(0.4)


class TestModelProbForQuoteEndBefore:
    def test_end_before_r2_counts_finishes_at_or_before_300s(self):
        # 4 finishes total (2 KO, 2 SUB); durations: KO@120/400, SUB@200/500 (n_ko=2,n_sub=2 -> i%3/i%2 cycles)
        winner_a = np.array([True, True, True, True, True, True])
        method = np.array(["DEC", "DEC", "KO/TKO", "KO/TKO", "SUB", "SUB"])
        duration_sec = np.array([900.0, 900.0, 120.0, 400.0, 200.0, 500.0])
        sim = {"winner_a": winner_a, "method": method, "duration_sec": duration_sec}
        pred = _FakePredFull(sim_samples=sim)
        # Finishes with duration <= 300: KO@120, SUB@200 -> 2/6
        p = model_prob_for_quote(_resolved_quote("end_before_r2", "fight"), pred)
        assert p == pytest.approx(2.0 / 6.0)

    def test_boundary_duration_exactly_300_counts(self):
        winner_a = np.array([True])
        method = np.array(["KO/TKO"])
        duration_sec = np.array([300.0])
        sim = {"winner_a": winner_a, "method": method, "duration_sec": duration_sec}
        pred = _FakePredFull(sim_samples=sim)
        p = model_prob_for_quote(_resolved_quote("end_before_r2", "fight"), pred)
        assert p == pytest.approx(1.0)

    def test_end_before_without_sim_falls_back_to_display_dur_cdf(self):
        class _FakeCdf:
            def cdf(self, t):
                assert t == pytest.approx(300.0)
                return 0.22
        pred = _FakePredFull(sim_samples=None, display_dur_cdf=_FakeCdf())
        p = model_prob_for_quote(_resolved_quote("end_before_r2", "fight"), pred)
        assert p == pytest.approx(0.22)

    def test_end_before_no_sim_no_cdf_returns_none(self):
        pred = _FakePredFull(sim_samples=None, display_dur_cdf=None)
        p = model_prob_for_quote(_resolved_quote("end_before_r2", "fight"), pred)
        assert p is None


class TestModelProbForQuoteWinIn:
    def test_win_in_r1_reuses_finish_prop_cdf_math(self):
        # red wins by KO at 120s (round 1) -> counts for win_in_r1, not r2
        winner_a = np.array([True, False])
        method = np.array(["KO/TKO", "DEC"])
        duration_sec = np.array([120.0, 900.0])
        sim = {"winner_a": winner_a, "method": method, "duration_sec": duration_sec}
        pred = _FakePredFull(sim_samples=sim)
        p_r1 = model_prob_for_quote(_resolved_quote("win_in_r1", "red"), pred)
        assert p_r1 == pytest.approx(0.5)  # 1/2 samples: red win + KO + dur in (0,300]

    def test_win_in_r_wrong_round_is_zero(self):
        winner_a = np.array([True])
        method = np.array(["KO/TKO"])
        duration_sec = np.array([120.0])  # round 1, not round 2
        sim = {"winner_a": winner_a, "method": method, "duration_sec": duration_sec}
        pred = _FakePredFull(sim_samples=sim)
        p = model_prob_for_quote(_resolved_quote("win_in_r2", "red"), pred)
        assert p == pytest.approx(0.0)
