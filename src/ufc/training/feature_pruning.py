"""Per-model zero-importance feature pruning.

Each model writes its own dead-features file after training. Subsequent runs
prune accordingly. Previously a single global file from the winner ensemble
was applied to every model, which incorrectly pruned features (e.g.
r1_sub_win_rate_*) that the method classifier needed.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from ufc.io import paths


def _dead_path(model_name: str) -> Path:
    return paths.outputs_models() / f"dead_features_{model_name}.txt"


DEAD_THRESHOLD = 1e-6   # treat values below this as effectively zero


def compute_dead_features_from_importances(
    importances_by_estimator: dict[str, np.ndarray],
    feature_cols: list[str],
) -> set[str]:
    """A feature is dead only when its importance is ≈0 across ALL estimators."""
    dead = None
    for arr in importances_by_estimator.values():
        arr = np.asarray(arr, dtype=float)
        # Truncate / pad to feature_cols length (CatBoost may return importances
        # for extra cat-string features appended after numeric ones).
        if len(arr) > len(feature_cols):
            arr = arr[: len(feature_cols)]
        elif len(arr) < len(feature_cols):
            arr = np.concatenate([arr, np.zeros(len(feature_cols) - len(arr))])
        zero_set = {f for f, imp in zip(feature_cols, arr) if abs(imp) <= DEAD_THRESHOLD}
        dead = zero_set if dead is None else (dead & zero_set)
    return dead or set()


def compute_dead_features(ensemble) -> set[str]:
    """Compute dead features across LGBM, XGB, and CatBoost (intersection).

    Conservative: a feature is dead only if all three estimators agree it's unused.
    """
    importances = {
        "lgbm": np.asarray(ensemble.lgbm.feature_importances_, dtype=float),
        "xgb":  np.asarray(ensemble.xgb.feature_importances_, dtype=float),
    }
    if getattr(ensemble, "cat", None) is not None:
        importances["cat"] = np.asarray(ensemble.cat.get_feature_importance(), dtype=float)
    return compute_dead_features_from_importances(importances, ensemble.feature_cols)


def save_dead_features(dead: set[str], model_name: str) -> Path:
    """Persist dead features, accumulating monotonically.

    Once a feature lands in the dead list it stays there permanently.
    This prevents oscillation: removing feature X can cause feature Y to
    flip in/out of zero-importance on subsequent runs, creating a cycle
    that never converges.  The monotone union guarantees convergence after
    at most N passes (one per feature).
    """
    out = _dead_path(model_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = load_dead_features(model_name)   # load before overwriting
    merged = existing | dead                     # only add, never remove
    out.write_text("\n".join(sorted(merged)))
    return out


def load_dead_features(model_name: str) -> set[str]:
    p = _dead_path(model_name)
    if not p.exists():
        return set()
    return set(p.read_text().splitlines())


def prune_features(cols: list[str], model_name: str) -> list[str]:
    """Remove dead features for the given model."""
    dead = load_dead_features(model_name)
    if not dead:
        return cols
    pruned = [c for c in cols if c not in dead]
    n_removed = len(cols) - len(pruned)
    if n_removed > 0:
        print(f"  [pruning:{model_name}] Removed {n_removed} zero-importance features ({len(pruned)} remain)")
    return pruned
