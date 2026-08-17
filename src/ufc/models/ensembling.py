"""Ensemble utilities — logit averaging, weight tuning."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit_average(probas: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Logit-average multiple probability arrays."""
    if weights is None:
        weights = [1.0 / len(probas)] * len(probas)
    weights = np.array(weights) / sum(weights)
    logit_avg = sum(w * logit(p) for w, p in zip(weights, probas))
    return sigmoid(logit_avg)


def tune_ensemble_weights(
    probas: list[np.ndarray], y_true: np.ndarray
) -> list[float]:
    """Find optimal ensemble weights by minimizing log-loss on validation set."""
    n = len(probas)

    def objective(w):
        w = np.abs(w)
        w = w / w.sum()
        combined = logit_average(probas, w.tolist())
        return log_loss(y_true, combined)

    result = minimize(
        objective,
        x0=[1.0 / n] * n,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-6},
    )
    weights = np.abs(result.x)
    weights = (weights / weights.sum()).tolist()
    return weights
