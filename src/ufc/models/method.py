"""Method of victory classifier: KO/TKO, SUB, DEC.

v5-baseline: replaces per-class isotonic regression (which caused 0% KO degeneracy)
with temperature scaling (single scalar) + prior shrinkage at inference.

Temperature scaling: fit a single T on val by minimising log-loss of softmax(logits/T).
Prior shrinkage: p_final = (1-α)*p_temp + α*p_prior at α=0.05.
This guarantees a floor for all classes without per-class hackery.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

import lightgbm as lgb
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from ufc import SEED
from ufc.io import paths

METHOD_CLASSES = ["KO/TKO", "SUB", "DEC"]
METHOD_MAP = {"KO/TKO": 0, "SUB": 1, "U-DEC": 2, "S-DEC": 2, "M-DEC": 2, "DQ": 2, "NC": 2}

# Prior shrinkage: mix calibrated probs with empirical priors at α=0.05
_PRIOR_ALPHA = 0.05

# Rolling era window for prior estimation (months). Anchors shrinkage to modern
# KO/SUB/DEC rates rather than the full training history.
_PRIOR_WINDOW_MONTHS = 36


def _compute_rolling_era_prior(
    y_enc: np.ndarray,
    dates: "pd.Series",
    window_months: int = _PRIOR_WINDOW_MONTHS,
) -> "np.ndarray | None":
    """Prior from the most recent window_months of training data.

    Returns None if there are fewer than 50 fights or fewer than 5 of any class
    in the window — caller should fall back to all-time priors.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    cutoff = dates.max() - pd.DateOffset(months=window_months)
    recent_mask = (dates >= cutoff).values
    if recent_mask.sum() < 50:
        return None
    recent_y = y_enc[recent_mask]
    counts = np.bincount(recent_y, minlength=3).astype(float)
    if counts.min() < 5:
        return None
    return counts / counts.sum()


def _cfg():
    with open(paths.root() / "configs" / "model_props.yaml") as f:
        return yaml.safe_load(f)["method_clf"]


def _softmax_with_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to logits and return softmax probabilities."""
    scaled = logits / max(T, 1e-6)
    # Numerically stable softmax
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def _fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Find the temperature T that minimises log-loss on val set.

    logits: (n, 3) raw pre-softmax outputs from the LGBM model
    y_true: (n,) integer class labels {0, 1, 2}
    """
    def _objective(T):
        probs = _softmax_with_temperature(logits, T)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return log_loss(y_true, probs)

    result = minimize_scalar(_objective, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


class MethodClassifier:
    """LightGBM multinomial classifier with temperature scaling + prior shrinkage.

    Replaces per-class isotonic calibration (v4.x), which caused 0% KO degeneracy.
    """

    def __init__(self):
        self.model: lgb.LGBMClassifier | None = None
        self.feature_cols: list[str] = []
        self.temperature: float = 1.0
        self.class_priors: np.ndarray = np.array([1/3, 1/3, 1/3])

    def fit(self, X_train, y_train_method, X_val, y_val_method,
            feature_cols,
            sample_weight: np.ndarray | None = None,
            train_dates: "pd.Series | None" = None,
            temporal_oof: bool = False) -> "MethodClassifier":
        """Fit model; calibrate temperature on val set.

        Parameters
        ----------
        sample_weight : np.ndarray | None
            Per-row recency weight for training rows.
        train_dates : pd.Series | None
            Event dates for training rows. When provided, class priors are
            estimated from the most recent 36 months rather than the full
            training history, anchoring shrinkage to the modern era.
        temporal_oof : bool
            Prod tier. When True, temperature is fit on temporal-OOF logits
            (TimeSeriesSplit) instead of the in-sample val window. The prod
            split's val overlaps training (model trains on ALL data), so an
            in-sample temperature fit drives T to the floor (0.1) → grossly
            overconfident finish probabilities. OOF logits fix this, mirroring
            the winner model's prod calibration.
        """
        self.feature_cols = feature_cols
        cfg = _cfg()

        # Encode method to int
        y_tr = y_train_method.map(METHOD_MAP).fillna(2).astype(int)
        y_vl = y_val_method.map(METHOD_MAP).fillna(2).astype(int)

        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=cfg["n_estimators"],
            num_leaves=cfg["num_leaves"],
            learning_rate=cfg["learning_rate"],
            verbosity=cfg["verbosity"],
            random_state=SEED,
            # ── Determinism flags (Step 1) ───────────────────────────────
            deterministic=True,
            force_row_wise=True,
            num_threads=1,
            feature_fraction_seed=SEED,
            bagging_seed=SEED,
            data_random_seed=SEED,
            extra_seed=SEED,
            objective_seed=SEED,
        )
        sw_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        # Prod mode (temporal_oof): X_val/y_vl are in-sample (val ⊂ train) — early
        # stopping against them would inflate the fitted model's boosting rounds on
        # data it can already see. Carve a temporal holdout from the tail of train
        # instead, same pattern as _fit_temperature_oof's 18mo slice below.
        if temporal_oof and train_dates is not None:
            dates_vec = pd.to_datetime(train_dates).values
            cutoff = (dates_vec.max() - pd.DateOffset(months=6)).to_datetime64()
            fit_val_mask = dates_vec >= cutoff
            if fit_val_mask.sum() < 30:
                fit_val_mask = np.zeros(len(dates_vec), dtype=bool)
                fit_val_mask[np.argsort(dates_vec)[-30:]] = True
            fit_tr_mask = ~fit_val_mask
            Xtr_fit = X_train[feature_cols].fillna(0).values
            fit_X_tr, fit_y_tr = Xtr_fit[fit_tr_mask], y_tr.values[fit_tr_mask]
            fit_X_vl, fit_y_vl = Xtr_fit[fit_val_mask], y_tr.values[fit_val_mask]
            fit_sw_kw = ({"sample_weight": sample_weight[fit_tr_mask]}
                         if sample_weight is not None else {})
        else:
            fit_X_tr, fit_y_tr = X_train[feature_cols].fillna(0), y_tr
            fit_X_vl, fit_y_vl = X_val[feature_cols].fillna(0), y_vl
            fit_sw_kw = sw_kw
        self.model.fit(
            fit_X_tr, fit_y_tr,
            eval_set=[(fit_X_vl, fit_y_vl)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
            **fit_sw_kw,
        )

        # ── Temperature scaling on val ────────────────────────────────────
        # Use true pre-softmax scores (raw_score=True) rather than log(p) proxy.
        # The proxy softmax(log(p)/T) is a degraded power transform; true logits
        # give the calibration knob its full range.
        if temporal_oof and train_dates is not None:
            self.temperature = self._fit_temperature_oof(
                X_train, y_tr, feature_cols, train_dates, sample_weight
            )
            print(f"  Temperature T = {self.temperature:.4f} (temporal-OOF)")
        else:
            val_logits = self.model.booster_.predict(
                X_val[feature_cols].fillna(0).values, raw_score=True
            )
            if val_logits.ndim == 1:  # older LightGBM flattens to (n*n_classes,)
                val_logits = val_logits.reshape(len(X_val), -1)
            self.temperature = _fit_temperature(val_logits, y_vl.values)
            print(f"  Temperature T = {self.temperature:.4f}")

        # ── Class priors: rolling 36-month window (modern era) preferred ──────
        # Training-era priors pull shrinkage toward historical KO rates (too high).
        # Rolling 36-month window anchors shrinkage to the current era instead.
        counts = np.bincount(y_tr.values, minlength=3).astype(float)
        self.class_priors = counts / counts.sum()
        if train_dates is not None:
            rolling_prior = _compute_rolling_era_prior(
                y_tr.values,
                train_dates.reset_index(drop=True) if hasattr(train_dates, "reset_index") else pd.Series(train_dates),
            )
            if rolling_prior is not None:
                self.class_priors = rolling_prior
                print(
                    f"  Rolling 36mo priors: "
                    f"KO/TKO={rolling_prior[0]:.3f}, SUB={rolling_prior[1]:.3f}, DEC={rolling_prior[2]:.3f}"
                )
            else:
                print("  Rolling prior insufficient data — using all-time priors")
        print(f"  Priors: KO/TKO={self.class_priors[0]:.3f}, SUB={self.class_priors[1]:.3f}, DEC={self.class_priors[2]:.3f}")

        return self

    def _fit_temperature_oof(self, X_train, y_tr, feature_cols,
                             train_dates, sample_weight) -> float:
        """Temperature from temporal-OOF logits (prod tier).

        Re-fits the same LGBM on expanding-window time folds so the logits used
        for the temperature fit come from models that did NOT see those rows.
        Fits T on the most-recent 18-month OOF slice (current-era calibration).
        """
        from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415

        Xv = X_train[feature_cols].fillna(0).values
        yv = y_tr.values if hasattr(y_tr, "values") else np.asarray(y_tr)
        dates = pd.to_datetime(train_dates).values
        order = np.argsort(dates, kind="stable")
        n = len(order)
        oof_logits = np.full((n, 3), np.nan)
        covered = np.zeros(n, dtype=bool)
        params = self.model.get_params()

        tss = TimeSeriesSplit(n_splits=5)
        for tr_i, te_i in tss.split(order):
            tr, te = order[tr_i], order[te_i]
            fm = lgb.LGBMClassifier(**params)
            sw = {"sample_weight": sample_weight[tr]} if sample_weight is not None else {}
            fm.fit(Xv[tr], yv[tr], **sw)
            lg = fm.booster_.predict(Xv[te], raw_score=True)
            if lg.ndim == 1:
                lg = lg.reshape(len(te), -1)
            oof_logits[te] = lg
            covered[te] = True

        max_date = pd.to_datetime(train_dates).max()
        cutoff = (max_date - pd.DateOffset(months=18)).to_datetime64()
        recent = covered & (dates >= cutoff)
        if recent.sum() < 50:
            recent = covered
        return _fit_temperature(oof_logits[recent], yv[recent])

    def predict_proba_dict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return calibrated dict {method: P(method)} for each row.

        Applies temperature scaling + prior shrinkage (α=0.05).
        Guarantees all classes ≥ α * prior (absolute floor without per-class hacks).
        """
        Xf = X.reindex(columns=self.feature_cols, fill_value=0).fillna(0)
        raw_logits = self.model.booster_.predict(Xf.values, raw_score=True)
        if raw_logits.ndim == 1:
            raw_logits = raw_logits.reshape(len(Xf), -1)

        # Apply temperature scaling on true logits
        temp_probs = _softmax_with_temperature(raw_logits, self.temperature)

        # Prior shrinkage: mix with training priors at α=0.05
        alpha = _PRIOR_ALPHA
        final_probs = (1 - alpha) * temp_probs + alpha * self.class_priors[np.newaxis, :]

        # Renormalise (shrinkage already sums to 1, but floating-point noise)
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)

        return {cls: final_probs[:, i] for i, cls in enumerate(METHOD_CLASSES)}

    def save(self, path: Path, gitsha: str = "latest") -> Path:
        out = path / f"method_clf_{gitsha}.joblib"
        joblib.dump(self, out, compress=3)
        return out

    @staticmethod
    def load(path: Path) -> "MethodClassifier":
        return joblib.load(path)
