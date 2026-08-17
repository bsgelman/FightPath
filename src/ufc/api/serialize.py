"""Serialize FightPrediction → JSON-safe fight shape expected by the React UI.

The UI design (FightPath Dashboard v3) expects a specific data contract.
This module owns the translation from CDF objects to precomputed arrays.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_QUANTILE_LABELS = ["10%", "25%", "50%", "75%", "90%"]
_QUANTILE_QS     = [0.10,  0.25,  0.50,  0.75,  0.90]
_N_CURVE_PTS     = 120


# ── Duration helpers ─────────────────────────────────────────────────────────

def _survival_curve_minutes(cdf_obj: Any, rounds: int) -> list[list[float]]:
    """Return [[minutes, survival], ...] for the UI POverCurve (x-axis = minutes).

    The decision point-mass is a Dirac spike at exactly sched_sec, not a smear.
    To represent it faithfully: sample up to (sched_sec - 1s) so the plateau at
    p_dec is visible, then append a coincident-x point at [max_min, 0.0].
    survivalAt2 then interpolates correctly for any L < sched (returns p_dec),
    and the chart renders a clean vertical drop at the scheduled max.
    """
    sched_sec = float(getattr(cdf_obj, "_scheduled_sec", rounds * 300))
    max_min = sched_sec / 60.0
    xs_min = np.linspace(0.0, max_min, _N_CURVE_PTS)
    pts = []
    for x_min in xs_min:
        t = min(x_min * 60.0, sched_sec - 1.0)
        s = float(cdf_obj.survival(t))
        pts.append([round(float(x_min), 3), round(s, 5)])
    # Vertical drop at the bell: coincident x, survival = 0.0
    pts.append([round(max_min, 3), 0.0])
    return pts


def _dur_quantiles(cdf_obj: Any) -> list[dict]:
    rows = []
    for label, q in zip(_QUANTILE_LABELS, _QUANTILE_QS):
        try:
            val_sec = _dur_quantile_sec(cdf_obj, q)
            rows.append({"label": label, "value": round(val_sec / 60.0, 2)})
        except Exception:
            rows.append({"label": label, "value": None})
    return rows


def _dur_quantile_sec(cdf_obj: Any, q: float) -> float:
    """Binary search on survival curve to find quantile in seconds."""
    sched_sec = float(getattr(cdf_obj, "_scheduled_sec", 900.0))
    lo, hi = 0.0, sched_sec
    target_survival = 1.0 - q
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if cdf_obj.survival(mid) > target_survival:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _round_distribution(cdf_obj: Any, rounds: int) -> list[float]:
    """Return [P(finish R1), P(finish R2), ..., P(finish Rn), P(DEC)] normalized."""
    scheduled_sec = float(getattr(cdf_obj, "_scheduled_sec", rounds * 300))
    dist = []
    for r in range(1, rounds + 1):
        start = (r - 1) * 300.0
        end = r * 300.0
        if end > scheduled_sec:
            end = scheduled_sec
        p_in = max(0.0, float(cdf_obj.survival(start)) - float(cdf_obj.survival(end)))
        dist.append(p_in)
    # DEC mass
    p_dec = float(getattr(cdf_obj, "_p_dec", cdf_obj.survival(scheduled_sec - 1)))
    # Normalize
    total = sum(dist) + p_dec
    if total > 1e-9:
        dist = [d / total for d in dist]
        p_dec = p_dec / total
    dist.append(p_dec)
    return [round(v, 4) for v in dist]


# ── Duration PDF helper ───────────────────────────────────────────────────────

def _duration_pdf_curve(cdf_obj: Any, rounds: int) -> list[list[float]]:
    """[[minutes, density], ...] — probability density from numerical CDF gradient."""
    try:
        from ufc.models.props_duration import _build_dur_cdf_grid
        sched_sec = float(getattr(cdf_obj, "_scheduled_sec", rounds * 300))
        t_grid, cdf_grid = _build_dur_cdf_grid(cdf_obj, sched_sec, n_pts=200)
        pdf_sec = np.gradient(cdf_grid, t_grid)
        pdf_sec = np.clip(pdf_sec, 0, None)
        x_min = t_grid / 60.0
        pdf_min = pdf_sec * 60.0
        area = float(np.trapezoid(pdf_min, x_min))
        if area > 0:
            pdf_min = pdf_min / area
        return [[round(float(x), 3), round(float(d), 6)] for x, d in zip(x_min, pdf_min)]
    except Exception:
        return []


# ── Count prop helpers ────────────────────────────────────────────────────────

def _count_survival_curve(cdf_obj: Any, x_max: float) -> list[list[float]]:
    """[[x, survival], ...] sampled at integers 0..x_max.

    Uses the CDF object's own coherent p_over() accessor when available
    (RateXDurationCDF.p_over — v8.35 discrete-coherence fix: p_over(line<1) =
    1 - p_zero, since integer counts have no mass in (0,1); v8.37's KD floor
    also only mutates _p_zero). Falling back to raw 1 - cdf() here silently
    undoes both fixes on the served prop-edge curve (10-21pp edge deflation).
    """
    pts = []
    n = int(x_max) + 1
    xs = np.linspace(0, x_max, min(n + 1, 90))
    for x in xs:
        s = (float(cdf_obj.p_over(float(x))) if hasattr(cdf_obj, "p_over")
             else float(1.0 - cdf_obj.cdf(float(x))))
        pts.append([round(float(x), 1), round(s, 5)])
    return pts


def _count_histogram(cdf_obj: Any, x_max: float, n_bins: int = 30) -> list[dict]:
    """[{x0, x1, count, frac}] from CDF differences (discrete bins)."""
    bin_w = max(1.0, x_max / n_bins)
    bins = np.arange(0.0, x_max + bin_w, bin_w)
    hist = []
    for i in range(len(bins) - 1):
        x0, x1 = bins[i], bins[i + 1]
        frac = max(0.0, float(cdf_obj.cdf(x1)) - float(cdf_obj.cdf(max(0.0, x0 - 0.5))))
        hist.append({"x0": round(x0, 1), "x1": round(x1, 1),
                     "count": 0, "frac": round(frac, 5)})
    return hist


def _count_quantiles(cdf_obj: Any) -> list[dict]:
    rows = []
    for label, q in zip(_QUANTILE_LABELS, _QUANTILE_QS):
        try:
            v = float(cdf_obj.quantile(q))
            rows.append({"label": label, "value": round(v, 1)})
        except Exception:
            rows.append({"label": label, "value": None})
    return rows


def _count_summary(cdf_obj: Any, x_max: float = 30.0) -> dict:
    """Summary stats for a count CDF: mean, sd, p0, quantiles."""
    p0 = float(getattr(cdf_obj, "_p_zero", 0.0))
    try:
        med = float(cdf_obj.quantile(0.5))
    except Exception:
        med = 0.0
    try:
        q25 = float(cdf_obj.quantile(0.25))
        q75 = float(cdf_obj.quantile(0.75))
    except Exception:
        q25 = q75 = med
    # Estimate mean from CDF by integrating 1 - F(x) over integers
    xs = np.arange(0, int(x_max) + 1)
    try:
        mean = float(np.sum([1.0 - cdf_obj.cdf(float(x)) for x in xs]))
    except Exception:
        mean = med
    sd = float((q75 - q25) / 1.35) if q75 > q25 else 0.0
    return {"mean": round(mean, 2), "sd": round(sd, 2), "p0": round(p0, 4),
            "q": {"p25": round(q25, 1), "p50": round(med, 1), "p75": round(q75, 1)}}


def _build_count_prop(cdf_obj: Any | None, x_max: float = 50.0) -> dict | None:
    if cdf_obj is None:
        return None
    try:
        return {
            "curve":     _count_survival_curve(cdf_obj, x_max),
            "hist":      _count_histogram(cdf_obj, x_max),
            "quantiles": _count_quantiles(cdf_obj),
            "summary":   _count_summary(cdf_obj, x_max),
            "pZero":     round(float(getattr(cdf_obj, "_p_zero", 0.0)), 4),
        }
    except Exception:
        logger.warning("_build_count_prop failed for %s", cdf_obj)
        return None


# ── Weight-class normalisation ────────────────────────────────────────────────
# Shared with feature-build (mileage.py) and inference (matchup.py) — see
# ufc.features.weight_class for the single source of truth.
from ufc.features.weight_class import clean_weight_class as _clean_weight_class  # noqa: E402


# ── Finish prop helper ────────────────────────────────────────────────────────

def _finish_probs(pred: Any) -> dict:
    """Per-corner ko/sub/any + per-round (r1..rounds) finish + per-round KO-only
    probabilities, plus a fight-level r1_finish (either winner). Per-corner
    r{k}_finish/r{k}_ko lets the UI price Flat Multi's per-fighter round-finish and
    round-knockout lines separately (e.g. McGregor R2 vs Holloway R2 are distinct
    events with distinct prices, and 'Round 2 Knockout' is KO-only — not any finish)."""
    from ufc.valuation.prop_cdf import _finish_prop_cdf
    out: dict[str, float | None] = {}
    rounds = int(getattr(pred, "rounds", 3) or 3)
    round_mkts = [f"r{k}_finish" for k in range(1, rounds + 1)]
    round_ko_mkts = [f"r{k}_ko" for k in range(1, rounds + 1)]
    for mkt in ("ko_finish", "sub_finish", "finish", *round_mkts, *round_ko_mkts):
        for corner in ("red", "blue"):
            try:
                cdf = _finish_prop_cdf(pred, mkt, corner)
                p = float(getattr(cdf, "p", 0.0))
                key = f"{corner}_{mkt}"
                out[key] = round(p, 4)
            except Exception:
                out[f"{corner}_{mkt}"] = None
    # R1 finish (fight-level, any winner) — kept for the Rounds tab + AI prompt panel.
    try:
        cdf_r = _finish_prop_cdf(pred, "r1_finish", "red")
        cdf_b = _finish_prop_cdf(pred, "r1_finish", "blue")
        out["r1_finish"] = round(
            float(getattr(cdf_r, "p", 0.0)) + float(getattr(cdf_b, "p", 0.0)), 4
        )
    except Exception:
        out["r1_finish"] = None
    return out


# ── Main serializer ───────────────────────────────────────────────────────────

def serialize_fight(pred: Any, slot: str = "main", idx: int = 0) -> dict:
    """Convert FightPrediction → JSON-safe dict matching the UI design data shape."""
    rounds = int(pred.rounds)
    dur_cdf = pred.display_dur_cdf
    sched_sec = float(getattr(dur_cdf, "_scheduled_sec", rounds * 300)) if dur_cdf else rounds * 300.0

    # ── Win probabilities ─────────────────────────────────────────────────────
    p_red = float(pred.prob_red)
    p_blue = float(pred.prob_blue)

    # Confidence band (80%  ~ 1.28σ from posterior assuming ~50k effective sim samples)
    n_eff = 50000
    se = np.sqrt(max(p_red * (1.0 - p_red), 1e-6) / n_eff)
    conf_lo = round(max(0.0, p_red - 1.28 * se), 3)
    conf_hi = round(min(1.0, p_red + 1.28 * se), 3)

    # ── Method ────────────────────────────────────────────────────────────────
    mp = pred.method_probs or {}
    ko  = round(float(mp.get("KO/TKO", 0.0)), 4)
    sub = round(float(mp.get("SUB",    0.0)), 4)
    dec = round(float(mp.get("DEC",    0.0)), 4)

    # ── Duration ─────────────────────────────────────────────────────────────
    dur_curve: list = []
    dur_quantiles: list = []
    round_dist: list = []
    inside = 0.0
    p_dec_val = 0.0
    median_min = None

    dur_pdf: list = []
    if dur_cdf is not None:
        try:
            dur_curve = _survival_curve_minutes(dur_cdf, rounds)
        except Exception:
            dur_curve = []
        try:
            dur_quantiles = _dur_quantiles(dur_cdf)
        except Exception:
            dur_quantiles = []
        try:
            round_dist = _round_distribution(dur_cdf, rounds)
        except Exception:
            round_dist = []
        try:
            p_dec_val = float(getattr(dur_cdf, "_p_dec",
                                      dur_cdf.survival(sched_sec - 1)))
        except Exception:
            p_dec_val = dec
        inside = round(1.0 - p_dec_val, 4)
        try:
            med_sec = _dur_quantile_sec(dur_cdf, 0.5)
            median_min = round(med_sec / 60.0, 2)
        except Exception:
            median_min = None
        try:
            dur_pdf = _duration_pdf_curve(dur_cdf, rounds)
        except Exception:
            dur_pdf = []

    # ── Count props ──────────────────────────────────────────────────────────
    sig_red  = _build_count_prop(pred.ss_cdf_red,  100.0)
    sig_blue = _build_count_prop(pred.ss_cdf_blue, 100.0)
    r1_red   = _build_count_prop(pred.r1_cdf_red,  80.0)
    r1_blue  = _build_count_prop(pred.r1_cdf_blue, 80.0)
    td_red   = _build_count_prop(pred.td_cdf_red,  15.0)
    td_blue  = _build_count_prop(pred.td_cdf_blue, 15.0)
    kd_red   = _build_count_prop(getattr(pred, "kd_cdf_red",  None), 3.0)
    kd_blue  = _build_count_prop(getattr(pred, "kd_cdf_blue", None), 3.0)
    subatt_red  = _build_count_prop(getattr(pred, "sub_att_cdf_red",  None), 4.0)
    subatt_blue = _build_count_prop(getattr(pred, "sub_att_cdf_blue", None), 4.0)
    r1td_red    = _build_count_prop(getattr(pred, "r1_td_cdf_red",    None), 4.0)
    r1td_blue   = _build_count_prop(getattr(pred, "r1_td_cdf_blue",   None), 4.0)
    body_red    = _build_count_prop(getattr(pred, "body_cdf_red",     None), 60.0)
    body_blue   = _build_count_prop(getattr(pred, "body_cdf_blue",    None), 60.0)
    leg_red     = _build_count_prop(getattr(pred, "leg_cdf_red",      None), 50.0)
    leg_blue    = _build_count_prop(getattr(pred, "leg_cdf_blue",     None), 50.0)
    # ctrl_time: stored in seconds server-side; serialize in minutes for UI
    _ctrl_r = getattr(pred, "ctrl_cdf_red",  None)
    _ctrl_b = getattr(pred, "ctrl_cdf_blue", None)

    class _MinutesCDF:
        """Thin adapter: converts second-scale CDF to minute-scale for serialize."""
        def __init__(self, inner):
            self._inner = inner
            self._p_zero = getattr(inner, "_p_zero", 0.0)
        def cdf(self, x):      return self._inner.cdf(x * 60.0)
        def p_over(self, x):   return self._inner.p_over(x * 60.0)
        def p_under(self, x):  return self._inner.p_under(x * 60.0)
        def quantile(self, q): return self._inner.quantile(q) / 60.0
        @property
        def median(self): return self._inner.median / 60.0
        def uncertainty_band(self, x): lo, hi = self._inner.uncertainty_band(x * 60.0); return lo, hi

    # x-axis spans the scheduled fight length (min) to match the UI axis (panels.jsx xMax)
    _ctrl_xmax = sched_sec / 60.0
    ctrl_red  = _build_count_prop(_MinutesCDF(_ctrl_r) if _ctrl_r  else None, _ctrl_xmax)
    ctrl_blue = _build_count_prop(_MinutesCDF(_ctrl_b) if _ctrl_b  else None, _ctrl_xmax)
    sig_combo = _build_count_prop(getattr(pred, "ss_combo_cdf", None), 160.0)

    # ── Finish props ──────────────────────────────────────────────────────────
    try:
        finish = _finish_probs(pred)
    except Exception:
        logger.warning("_finish_probs failed for %s vs %s", pred.red_name, pred.blue_name)
        finish = {}

    # ── Assemble output ───────────────────────────────────────────────────────
    return {
        "id":    f"fight_{idx}",
        "slot":  slot,
        "rounds": rounds,
        "isTitle": pred.is_title,
        "weightClass": _clean_weight_class(getattr(pred, "weight_class", "")),
        "a": {
            "name":   pred.red_name,
            "id":     pred.red_id,
            "pWin":   round(p_red,  4),
            "record": list(getattr(pred, "record_red",  None) or ()),
            "method": {"ko": ko, "sub": sub, "dec": dec},
            "sig":    sig_red,
            "r1sig":  r1_red,
            "td":     td_red,
            "kd":     kd_red,
            "subAtt": subatt_red,
            "r1td":   r1td_red,
            "bodySig": body_red,
            "legSig":  leg_red,
            "ctrl":    ctrl_red,
            "finish": {k.replace("red_", ""): v for k, v in finish.items() if k.startswith("red_")},
        },
        "b": {
            "name":   pred.blue_name,
            "id":     pred.blue_id,
            "pWin":   round(p_blue, 4),
            "record": list(getattr(pred, "record_blue", None) or ()),
            "method": {"ko": ko, "sub": sub, "dec": dec},
            "sig":    sig_blue,
            "r1sig":  r1_blue,
            "td":     td_blue,
            "kd":     kd_blue,
            "subAtt": subatt_blue,
            "r1td":   r1td_blue,
            "bodySig": body_blue,
            "legSig":  leg_blue,
            "ctrl":    ctrl_blue,
            "finish": {k.replace("blue_", ""): v for k, v in finish.items() if k.startswith("blue_")},
        },
        "method": {"ko": ko, "sub": sub, "dec": dec},
        "inside": inside,
        "pDec":   round(p_dec_val, 4),
        "confidence": {"lo": conf_lo, "hi": conf_hi},
        "durCurve":    dur_curve,
        "durPdf":      dur_pdf,
        "durQuantiles": dur_quantiles,
        "roundDist":   round_dist,
        "medianMin":   median_min,
        "schedSec":    round(sched_sec, 0),
        "r1Finish":    finish.get("r1_finish"),
        "sigCombo":    sig_combo,
        "simSamples":  None,  # not serialized (kept server-side for portfolio MC)
        "lowData":         bool(getattr(pred, "low_data",          False)),
        "nFightsRed":      int(getattr(pred,  "n_fights_red",      0)),
        "nFightsBlue":     int(getattr(pred,  "n_fights_blue",      0)),
        "methodEdgeScore": round(float(getattr(pred, "method_edge_score", 0.0)), 3),
        "hasMethodEdge":   bool(getattr(pred,  "has_method_edge",   False)),
        "winnerDrivers":   list(getattr(pred, "winner_drivers", None) or []),
    }
