"""Per-weight-class confidence dampening for high-variance divisions.

HW and LHW have an intrinsically high entropy floor (single-punch KO variance).
Shrinking extreme win probabilities toward 0.5 for these divisions improves the
per-WC Brier score and signals the betting layer that stake sizing should be
conservative.

Usage:
    prob_red = apply_wc_temperature(prob_red, weight_class)
"""
from __future__ import annotations

import numpy as np

# Temperature > 1.0 shrinks logit toward 0 (i.e. probability toward 0.5).
# Calibrated to reduce LHW Brier 0.269 → ~0.255 target without overcorrecting.
_WC_TEMPERATURE: dict[str, float] = {
    "Heavyweight": 1.20,
    "Light Heavyweight": 1.15,
    "Women's Strawweight": 1.10,
}

_DEFAULT_TEMPERATURE = 1.0


def apply_wc_temperature(prob: float, weight_class: str) -> float:
    """Apply per-WC temperature scaling to a win probability.

    Temperature > 1.0 shrinks the logit toward 0, pulling extreme predictions
    toward 0.5. This is equivalent to a learned WC-specific confidence penalty.

    Parameters
    ----------
    prob : float
        P(fighter A wins), in (0, 1).
    weight_class : str
        UFC weight-class string (e.g. "Heavyweight", "Light Heavyweight").

    Returns
    -------
    float : temperature-adjusted win probability.
    """
    T = _WC_TEMPERATURE.get(weight_class, _DEFAULT_TEMPERATURE)
    if T == 1.0 or abs(prob - 0.5) < 1e-6:
        return float(prob)
    prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
    logit = np.log(prob / (1.0 - prob))
    new_logit = logit / T
    return float(1.0 / (1.0 + np.exp(-new_logit)))


def wc_prob_ci(prob: float, n_bootstrap: int = 1000, n_fights: int = 56,
               seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI for a win probability given limited WC sample size.

    Simulates Bernoulli draws to estimate calibration uncertainty. Useful for
    communicating model sharpness to the stake-sizing layer.

    Parameters
    ----------
    prob : float
        Point-estimate win probability.
    n_bootstrap : int
        Bootstrap replicates.
    n_fights : int
        Typical WC sample size (drives CI width).

    Returns
    -------
    (lower_95, upper_95) : tuple[float, float]
    """
    rng = np.random.default_rng(seed)
    samples = rng.binomial(n_fights, prob, size=n_bootstrap) / n_fights
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return lo, hi
