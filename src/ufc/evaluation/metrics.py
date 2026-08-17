"""Evaluation metrics for winner and prop models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    log_loss, brier_score_loss, accuracy_score, roc_auc_score
)


def winner_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute all winner-model metrics."""
    y_pred = (y_pred_proba > 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_pred_proba)),
        "brier": float(brier_score_loss(y_true, y_pred_proba)),
        "auroc": float(roc_auc_score(y_true, y_pred_proba)),
        "ece": float(expected_calibration_error(y_true, y_pred_proba)),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        frac = mask.sum() / n
        mean_prob = y_prob[mask].mean()
        mean_outcome = y_true[mask].mean()
        ece += frac * abs(mean_prob - mean_outcome)
    return ece


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss."""
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, q * e, (q - 1) * e)))


def crps_from_quantiles(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantile_probs: np.ndarray,
) -> float:
    """Continuous Ranked Probability Score approximated from quantile predictions.

    quantile_preds: (n_samples, n_quantiles)
    quantile_probs: (n_quantiles,)
    """
    n = len(y_true)
    crps = 0.0
    for i in range(len(quantile_probs) - 1):
        q = quantile_probs[i]
        y_q = quantile_preds[:, i]
        crps += pinball_loss(y_true, y_q, q) * (quantile_probs[i+1] - quantile_probs[i])
    return crps * 2


def single_leg_hit_rate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    payout_type: str = "powerplay_power_3pick",
    threshold: float = 0.05,
    stake: float = 1.0,
) -> dict:
    """Single-leg win rate vs break-even implied probability.

    Step 11 replacement for roi_vs_line: drops the parlay multiplier that
    was previously applied to individual single-leg wins, which inflated ROI
    by 2.731x per winning leg.  The correct single-leg diagnostic is:
      profit = +stake if bet wins, -stake if bet loses (flat-bet).
    This tells you whether the model finds legs that clear the implied rate —
    not what a parlay pays.  Walk-forward parlay ROI is in parlay_backtest.py.

    Threshold is expressed as edge above the per-leg implied breakeven
    (e.g. threshold=0.05 → bet when model_prob > implied + 0.05).
    """
    from ufc.valuation.payouts import implied_prob_per_leg
    implied = implied_prob_per_leg(payout_type)
    bets = y_pred_proba > (implied + threshold)
    if not bets.any():
        return {"n_bets": 0, "win_rate": 0.0, "edge": 0.0, "flat_roi": 0.0,
                "payout_type": payout_type, "implied": implied}

    wins = y_true[bets].astype(bool)
    win_rate = float(wins.mean())
    edge = win_rate - implied
    profit = float(np.where(wins, stake, -stake).sum())
    flat_roi = profit / (bets.sum() * stake)
    return {
        "n_bets": int(bets.sum()),
        "win_rate": win_rate,
        "edge": edge,
        "flat_roi": float(flat_roi),
        "payout_type": payout_type,
        "implied": implied,
    }


def roi_vs_line(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    payout_type: str = "powerplay_power_3pick",
    threshold: float = 0.05,
    stake: float = 1.0,
) -> dict:
    """[DEPRECATED] Single-leg ROI using parlay multiplier — structurally inflated.

    Kept for backward compat (ROI curve plot only).  Use single_leg_hit_rate for
    correct diagnostics, and parlay_backtest.walk_forward_parlay for parlay ROI.
    """
    from ufc.valuation.payouts import implied_prob_per_leg, get_payout_multiplier
    implied = implied_prob_per_leg(payout_type)
    mult = get_payout_multiplier(payout_type)
    bets = y_pred_proba > (implied + threshold)
    if not bets.any():
        return {"n_bets": 0, "roi": 0.0, "win_rate": 0.0, "profit": 0.0,
                "payout_type": payout_type, "implied": implied}

    wins = y_true[bets].astype(bool)
    win_rate = float(wins.mean())
    profit = float(np.where(wins, stake * (mult - 1.0), -stake).sum())
    roi = profit / (bets.sum() * stake)
    return {
        "n_bets": int(bets.sum()),
        "win_rate": win_rate,
        "roi": float(roi),
        "profit": profit,
        "payout_type": payout_type,
        "implied": implied,
    }
