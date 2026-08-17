"""Map a prop market onto the right model CDF.

Split out of prop_cdf.py so the CDF routing survives independently of the
edge-evaluation layer that happened to host it. Consumers: api.serialize,
api.app, valuation.market_edge (Kalshi).

Leaf module - stdlib + logging only, no edge/line/payout imports.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


_FINISH_MARKETS = frozenset({
    "ko_finish", "sub_finish", "finish",
    "r1_finish", "r2_finish", "r3_finish", "r4_finish", "r5_finish",
    "r1_ko", "r2_ko", "r3_ko", "r4_ko", "r5_ko",
})


def _finish_round(market: str) -> int | None:
    """Return k for 'r{k}_finish' / 'r{k}_ko' markets, else None."""
    if market in ("ko_finish", "sub_finish", "finish"):
        return None
    if market.startswith("r") and (market.endswith("_finish") or market.endswith("_ko")):
        try:
            return int(market[1:market.index("_")])
        except (ValueError, IndexError):
            pass
    return None


def _is_round_ko(market: str) -> bool:
    """True for 'r{k}_ko' markets — KO/TKO-only, unlike 'r{k}_finish' (any finish)."""
    return market.startswith("r") and market.endswith("_ko")


def select_prop_cdf(pred: Any, market: str, corner: str) -> Any | None:
    """Pick the right CDF/float from a FightPrediction given market + corner."""
    if market in ("duration", "duration_sec", "rounds"):
        return pred.display_dur_cdf
    if market == "sig_strikes":
        return pred.ss_cdf_red if corner == "red" else pred.ss_cdf_blue
    if market == "r1_sig_strikes":
        return pred.r1_cdf_red if corner == "red" else pred.r1_cdf_blue
    if market == "takedowns":
        return pred.td_cdf_red if corner == "red" else pred.td_cdf_blue
    if market == "knockdowns":
        return pred.kd_cdf_red if corner == "red" else pred.kd_cdf_blue
    if market == "sub_attempts":
        return pred.sub_att_cdf_red if corner == "red" else pred.sub_att_cdf_blue
    if market == "r1_takedowns":
        return pred.r1_td_cdf_red if corner == "red" else pred.r1_td_cdf_blue
    if market == "body_sig_strikes":
        return pred.body_cdf_red if corner == "red" else pred.body_cdf_blue
    if market == "leg_sig_strikes":
        return pred.leg_cdf_red if corner == "red" else pred.leg_cdf_blue
    if market == "ctrl_time":
        return pred.ctrl_cdf_red if corner == "red" else pred.ctrl_cdf_blue
    if market == "sig_strikes_combo":
        return pred.ss_combo_cdf
    if market == "winner":
        return pred.prob_red if corner == "red" else pred.prob_blue

    # ── Finish props (v8.24): fighter-directional binary O/U-0.5 ──────────
    # All finish markets are directional: P(named fighter WINS by that finish).
    if market in _FINISH_MARKETS:
        return _finish_prop_cdf(pred, market, corner)

    return None


def _finish_prop_cdf(pred: Any, market: str, corner: str) -> Any:
    """Compute BernoulliPropCDF for a finish market.

    MC primary when sim_samples available (correct winner/method/duration joint).
    Analytic fallback: prob_corner × method_prob (winner ⊥ method approximation).
    """
    import numpy as np
    from ufc.models.props_duration import BernoulliPropCDF

    is_red = (corner == "red")
    mp = pred.method_probs
    k = _finish_round(market)

    # MC primary
    sim = getattr(pred, "sim_samples", None)
    if sim is not None:
        winner_a = np.asarray(sim.get("winner_a", []))
        method   = sim.get("method", np.array([]))
        dur      = np.asarray(sim.get("duration_sec", []))
        if len(winner_a) > 0 and len(method) > 0:
            win  = winner_a if is_red else ~winner_a
            if market == "ko_finish":
                hit = win & (method == "KO/TKO")
            elif market == "sub_finish":
                hit = win & (method == "SUB")
            elif market == "finish":
                hit = win & (method != "DEC")
            elif k is not None:
                is_fin = (method == "KO/TKO") if _is_round_ko(market) else (method != "DEC")
                hit = win & is_fin & (dur > (k - 1) * 300.0) & (dur <= k * 300.0)
            else:
                hit = win & (method != "DEC")
            return BernoulliPropCDF(float(hit.mean()))

    # Analytic fallback (no sim_samples on this prediction)
    if _is_round_ko(market):
        # Redistributes round-N finish mass by the OVERALL KO share of finishes,
        # but real KO/sub ratio is round-dependent (subs skew earlier) — biased
        # vs the MC path above. Log so a fallback-heavy pricing run is visible.
        logger.warning(
            "prop_edge: analytic fallback for round-KO market %r (corner=%s) — "
            "no sim_samples; round-independent KO-share approximation in use",
            market, corner,
        )
    pc = pred.prob_red if is_red else pred.prob_blue
    if market == "ko_finish":
        p = pc * mp.get("KO/TKO", 0.33)
    elif market == "sub_finish":
        p = pc * mp.get("SUB", 0.17)
    elif market == "finish":
        p = pc * (1.0 - mp.get("DEC", 0.50))
    elif k is not None:
        dc = pred.display_dur_cdf
        sched = int(getattr(pred, "rounds", 3) or 3)
        if dc is not None and 1 <= k <= sched:
            p_round = dc.cdf(k * 300.0) - dc.cdf((k - 1) * 300.0)
            if _is_round_ko(market):
                # No per-draw method/duration join available without sim_samples —
                # redistribute the round's finish mass by the overall KO share of
                # all finishes (winner ⊥ method ⊥ duration approximation, same
                # spirit as the other analytic-fallback branches above).
                p_fin = 1.0 - mp.get("DEC", 0.50)
                ko_share = (mp.get("KO/TKO", 0.33) / p_fin) if p_fin > 1e-9 else 0.0
                p = pc * p_round * ko_share
            else:
                p = pc * p_round
        else:
            p = 0.0
    else:
        p = 0.0
    return BernoulliPropCDF(p)


def _display_line(market: str, line_sec: float) -> float:
    """Convert canonical (seconds) line back to display units."""
    if market == "duration":
        return line_sec / 60.0
    if market == "rounds":
        return line_sec / 300.0
    if market == "ctrl_time":
        return line_sec / 60.0
    return line_sec
