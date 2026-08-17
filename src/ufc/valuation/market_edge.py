"""Fee/spread/depth-aware edge math for Kalshi prediction-market quotes.

Kalshi's fee schedule (verified 2026-07-02, https://kalshi.com/fee-schedule):
taker fee = 0.07 * C * (1-C) per contract (C = price in dollars), maker fee =
25% of the equivalent taker fee. Fees peak (1.75c) at a 50c price and shrink
toward the extremes — Kalshi's own order total is rounded up to the next cent;
that rounding is immaterial at the probability precision used here.

Pricing is always against the ASK (never the mid) — that's the price a taker
actually pays. The maker counterfactual (posting one cent inside the spread)
is computed for the ledger only, to distinguish "no edge" from "fees ate the
edge that existed at the mid."
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ufc.ingest.market_lines import MarketQuote
from ufc.valuation.payouts import kelly_cap

_LIQ_DEEP_USD = 10_000.0
_LIQ_OK_USD = 2_000.0

_METHOD_NAME = {"method_ko": "KO/TKO", "method_sub": "SUB", "method_dec": "DEC"}
_MOF_NAME = {"mof_ko": "KO/TKO", "mof_sub": "SUB", "mof_dec": "DEC"}
_END_BEFORE_RE = re.compile(r"^end_before_r(\d+)$")
_WIN_IN_RE = re.compile(r"^win_in_r(\d+)$")


def model_prob_for_quote(rq, pred) -> Optional[float]:
    """Model probability for one resolved Kalshi quote (ResolvedMarketQuote) against
    a FightPrediction. Winner markets read prob_red/prob_blue directly. Per-fighter
    method markets need the JOINT P(this fighter wins AND by this method) —
    pred.method_probs only gives P(fight ends by method X) with no fighter
    attribution, so this reads the Monte Carlo sim_samples (winner_a & method
    draws) instead. Fight-level markets (distance, mof_*, end_before_r{r},
    vicround_other) and per-fighter round-of-victory markets (win_in_r{r}) are
    priced the same way, plus an analytic fallback for when sim_samples isn't
    available (mirrors prop_cdf.py's MC-primary/analytic-fallback pattern).
    None if the market_kind is unrecognized, or a market needs sim_samples/a
    duration CDF that isn't available."""
    if rq.market_kind == "winner":
        return pred.prob_red if rq.corner == "red" else pred.prob_blue

    method_name = _METHOD_NAME.get(rq.market_kind)
    if method_name is not None:
        sim = getattr(pred, "sim_samples", None)
        if not sim:
            return None
        winner_a = sim["winner_a"]
        fighter_mask = winner_a if rq.corner == "red" else ~winner_a
        method_mask = sim["method"] == method_name
        return float((fighter_mask & method_mask).mean())

    if rq.market_kind in ("distance", "vicround_other"):
        sim = getattr(pred, "sim_samples", None)
        if sim:
            return float((sim["method"] == "DEC").mean())
        return pred.method_probs.get("DEC")

    mof_name = _MOF_NAME.get(rq.market_kind)
    if mof_name is not None:
        sim = getattr(pred, "sim_samples", None)
        if sim:
            return float((sim["method"] == mof_name).mean())
        return pred.method_probs.get(mof_name)

    m = _END_BEFORE_RE.match(rq.market_kind)
    if m is not None:
        r = int(m.group(1))
        sim = getattr(pred, "sim_samples", None)
        if sim:
            method = sim["method"]
            duration_sec = np.asarray(sim["duration_sec"])
            hit = (method != "DEC") & (duration_sec <= (r - 1) * 300.0)
            return float(hit.mean())
        dur_cdf = getattr(pred, "display_dur_cdf", None)
        if dur_cdf is not None:
            return dur_cdf.cdf((r - 1) * 300.0)
        return None

    m = _WIN_IN_RE.match(rq.market_kind)
    if m is not None:
        from ufc.valuation.prop_cdf import select_prop_cdf
        cdf = select_prop_cdf(pred, f"r{m.group(1)}_finish", rq.corner)
        return cdf.p_over(0.5) if cdf is not None else None

    return None


def kalshi_taker_fee(price: float) -> float:
    """Per-contract taker fee in dollars for a contract priced at `price` (0-1)."""
    return 0.07 * price * (1.0 - price)


def effective_cost_taker(ask: float) -> float:
    """Total cost per contract to take (buy at ask) — ask + taker fee."""
    return ask + kalshi_taker_fee(ask)


def effective_cost_maker(bid: float) -> float:
    """Total cost per contract for the maker counterfactual: post one cent inside
    the spread (bid + 1c) and pay the reduced (25%) maker fee on that price."""
    post_price = bid + 0.01
    return post_price + 0.25 * kalshi_taker_fee(post_price)


def _liq_tier(depth_usd_3c: Optional[float]) -> str:
    if depth_usd_3c is None:
        return "THIN"
    if depth_usd_3c >= _LIQ_DEEP_USD:
        return "DEEP"
    if depth_usd_3c >= _LIQ_OK_USD:
        return "OK"
    return "THIN"


@dataclass
class MarketEdge:
    model_p: float
    ask: float
    bid: Optional[float]
    breakeven: float          # taker fee-adjusted cost
    edge_pct: float           # model_p - breakeven
    kelly: float              # capped fractional Kelly, priced off breakeven
    maker_breakeven: float
    maker_edge_pct: float
    liq_tier: str
    stake_cap_usd: Optional[float]


def evaluate_market_quote(quote: MarketQuote, model_p: float) -> MarketEdge:
    """Price a Kalshi quote against the model probability. Always prices off the
    ask (taker cost) for the tradable edge/Kelly; the bid-side maker counterfactual
    is computed alongside for ledger/diagnostic use only."""
    ask = quote.yes_ask if quote.yes_ask is not None else quote.last_price
    breakeven = effective_cost_taker(ask)
    edge_pct = model_p - breakeven
    kelly = max(0.0, (model_p - breakeven) / (1.0 - breakeven)) if breakeven < 1.0 else 0.0
    kelly = min(kelly, kelly_cap())

    if quote.yes_bid is not None:
        maker_breakeven = effective_cost_maker(quote.yes_bid)
    else:
        maker_breakeven = breakeven
    maker_edge_pct = model_p - maker_breakeven

    return MarketEdge(
        model_p=model_p,
        ask=ask,
        bid=quote.yes_bid,
        breakeven=breakeven,
        edge_pct=edge_pct,
        kelly=kelly,
        maker_breakeven=maker_breakeven,
        maker_edge_pct=maker_edge_pct,
        liq_tier=_liq_tier(quote.depth_usd_3c),
        stake_cap_usd=quote.depth_usd_3c,
    )
