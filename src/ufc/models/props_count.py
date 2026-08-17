"""Distributional count models for sig strikes and takedowns.

v5-baseline:
- All count props now use HurdleCountModel (sig_strikes, r1_sig_strikes, takedowns).
  Previously only takedowns was hurdle; sig_strikes used CountModel (flat PropCDF).
  The hurdle structure was the only prop model consistently passing KS calibration.
- QUANTILE_GRID reduced from 25 to 11 points (faster, sufficient resolution).
- Dead code removed: NGB mixture in PropCDF, empirical zero rate plumbing.

v5.3 additions:
- RateHurdleCountModel: two-stage hurdle on per-minute rate (not raw count).
  Stage 2 target = log(count / active_minutes). At inference, integrates against
  a DurationCDF via Monte Carlo to produce a count CDF. Fixes the regime-mixture
  problem: the raw-count quantile model learned "when does the fight end" instead
  of "how active is this fighter".
- RateXDurationCDF: empirical count CDF from N_MC Monte Carlo samples.
  Mirrors HurdlePropCDF interface so downstream code (valuation, simulator) is
  unchanged except for the isinstance check in simulator._sample_counts.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

import scipy.special
import scipy.optimize
import lightgbm as lgb

from ufc import SEED
from ufc.io import paths


def _cfg():
    with open(paths.root() / "configs" / "model_props.yaml") as f:
        return yaml.safe_load(f)


# Sentinel distinguishing "no override passed" (use self.method_log_rate_adj)
# from an explicit override value (including None, meaning "no adjustment").
_UNSET = object()

# Count model quantile grid — 25 points for precision in both tails.
# (Duration model uses its own 11-point grid from props_duration.py)
QUANTILE_GRID = [
    0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
    0.90, 0.925, 0.95, 0.975, 0.99,
]


def _oof_best_n(
    estimator: "lgb.LGBMModel",
    X: np.ndarray,
    y: np.ndarray,
    dates: "pd.Series",
    sample_weight: "np.ndarray | None" = None,
    *,
    min_hold: int = 300,
    frac: float = 0.2,
) -> "int | None":
    """OOS early-stopping probe for prod models where val ⊂ train (in-sample leak).

    Fits `estimator` on the oldest (1-frac) of rows with the most-recent `frac`
    as a genuine holdout eval_set, returns best_iteration_.  Returns None when
    the training set is too small for a meaningful holdout (caller falls back to
    the standard val-set early-stopping path).  The caller must refit `estimator`
    on all rows at the returned n_estimators — this probe fit is a throw-away.
    """
    dts = pd.to_datetime(pd.Series(dates).values)
    order = np.argsort(dts.values, kind="stable")
    n_hold = max(min_hold, int(len(X) * frac))
    if n_hold >= len(X):
        return None
    hold_idx, fit_idx = order[-n_hold:], order[:-n_hold]
    sw = {"sample_weight": sample_weight[fit_idx]} if sample_weight is not None else {}
    estimator.fit(
        X[fit_idx], y[fit_idx],
        eval_set=[(X[hold_idx], y[hold_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        **sw,
    )
    best = getattr(estimator, "best_iteration_", None)
    return int(best) if (best is not None and best > 0) else estimator.n_estimators


class HurdlePropCDF:
    """CDF with hurdle logic: point mass at 0 + conditional quantile distribution.

    P(X <= x) = P(X=0)                       for x < 1
                P(X=0) + P(X>0) * P(X<=x | X>0)  for x >= 1

    The hurdle structure correctly handles zero-inflation (fights where a fighter
    throws/lands zero takedowns or zero sig strikes) without a spike in the PIT.
    """

    def __init__(self, qv_positive: np.ndarray, qs: np.ndarray, p_pos: float):
        self._qv_pos = np.maximum.accumulate(np.array(qv_positive))  # PAV monotonize
        self._qs = np.array(qs)
        self._p_pos = float(np.clip(p_pos, 0.0, 1.0))
        self._p_zero = 1.0 - self._p_pos

    def cdf(self, x: float) -> float:
        """P(X <= x)."""
        if x < 0:
            return 0.0
        if x < 1:
            # Only the zero mass lies below 1
            return self._p_zero
        p_pos_leq_x = float(np.interp(x, self._qv_pos, self._qs, left=0.0, right=1.0))
        return self._p_zero + self._p_pos * p_pos_leq_x

    def p_over(self, line: float) -> float:
        return 1.0 - self.cdf(line)

    def p_under(self, line: float) -> float:
        return self.cdf(line - 0.5)

    def quantile(self, q: float) -> float:
        if q <= self._p_zero:
            return 0.0
        q_pos = float(np.clip((q - self._p_zero) / self._p_pos, 0.0, 1.0))
        return float(np.interp(q_pos, self._qs, self._qv_pos))

    @property
    def median(self) -> float:
        return self.quantile(0.5)

    def uncertainty_band(self, line: float) -> tuple[float, float]:
        p = self.p_over(line)
        se = np.sqrt(p * (1 - p) / 1000)
        return (max(0, p - 1.28 * se), min(1, p + 1.28 * se))


# Backward-compatibility alias
PropCDF = HurdlePropCDF


class HurdleCountModel:
    """Two-stage hurdle model for zero-inflated counts.

    Stage 1: binary P(count > 0) via LGBMClassifier.
    Stage 2: quantile regression on positive rows only via LGBMRegressor.

    Used for ALL count props: sig_strikes, r1_sig_strikes, takedowns.
    """

    def __init__(self, target: str = "takedowns"):
        self.target = target
        self.pos_clf: lgb.LGBMClassifier | None = None
        self.quantile_models: list[lgb.LGBMRegressor] = []
        self.feature_cols: list[str] = []
        self.bias_shift: float = 0.0  # additive shift on positive-quantile predictions (sig_strikes only)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series,
            feature_cols: list[str],
            sample_weight=None) -> "HurdleCountModel":
        self.feature_cols = feature_cols
        cfg = _cfg()
        lgbm_cfg = cfg["lgbm_quantile"]

        X_tr = X_train[feature_cols].fillna(0).values
        y_tr = y_train.fillna(0).clip(lower=0).values.astype(float)
        X_vl = X_val[feature_cols].fillna(0).values
        y_vl = y_val.fillna(0).clip(lower=0).values.astype(float)

        print(f"  Fitting hurdle binary classifier for {self.target}...")
        y_pos_tr = (y_tr > 0).astype(int)
        y_pos_vl = (y_vl > 0).astype(int)
        self.pos_clf = lgb.LGBMClassifier(
            n_estimators=lgbm_cfg["n_estimators"],
            num_leaves=lgbm_cfg["num_leaves"],
            learning_rate=lgbm_cfg["learning_rate"],
            min_child_samples=lgbm_cfg["min_child_samples"],
            verbosity=-1,
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
        self.pos_clf.fit(
            X_tr, y_pos_tr,
            eval_set=[(X_vl, y_pos_vl)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=0)],
            **sw_kw,
        )

        print(f"  Fitting LGBM quantiles on positives for {self.target}...")
        pos_mask = y_tr > 0
        X_pos = X_tr[pos_mask]
        y_pos = y_tr[pos_mask]
        sw_pos = sample_weight[pos_mask] if sample_weight is not None else None

        self.quantile_models = []
        for q in QUANTILE_GRID:
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=lgbm_cfg["n_estimators"],
                learning_rate=lgbm_cfg["learning_rate"],
                num_leaves=lgbm_cfg["num_leaves"],
                min_child_samples=lgbm_cfg["min_child_samples"],
                verbosity=-1,
                random_state=SEED,
                # ── Determinism flags (Step 1) ───────────────────────────
                deterministic=True,
                force_row_wise=True,
                num_threads=1,
                feature_fraction_seed=SEED,
                bagging_seed=SEED,
                data_random_seed=SEED,
                extra_seed=SEED,
                objective_seed=SEED,
            )
            fit_kw_pos = {"sample_weight": sw_pos} if sw_pos is not None else {}
            if len(X_pos) > 20:
                m.fit(X_pos, y_pos,
                      eval_set=[(X_vl, y_vl)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=0)],
                      **fit_kw_pos)
            self.quantile_models.append(m)

        # Era-shift correction disabled: a flat additive bias on quantile values
        # shifts the full CDF uniformly but leaves tails miscalibrated, which
        # increases the KS stat. The era_avg_sig_str_l12mo feature (if used in
        # winner/method) carries the era trend without distorting quantile shape.
        # self.bias_shift remains 0.0 (default).

        return self

    def predict_cdf(self, X: pd.DataFrame) -> list[HurdlePropCDF]:
        """Return a HurdlePropCDF for each row in X."""
        Xf = X.reindex(columns=self.feature_cols, fill_value=0).fillna(0)
        Xa = Xf.values

        p_pos_all = self.pos_clf.predict_proba(Xa)[:, 1]
        q_preds = np.column_stack([m.predict(Xa) for m in self.quantile_models])
        q_preds = q_preds.clip(min=0)

        shift = getattr(self, "bias_shift", 0.0)
        cdfs = []
        for i in range(len(Xf)):
            qv_pos = np.maximum.accumulate(q_preds[i])
            if shift != 0.0:
                qv_pos = (qv_pos + shift).clip(min=0)
            cdfs.append(HurdlePropCDF(qv_pos, np.array(QUANTILE_GRID), float(p_pos_all[i])))

        return cdfs

    def save(self, path: Path, gitsha: str = "latest") -> Path:
        out = path / f"props_{self.target}_{gitsha}.joblib"
        joblib.dump(self, out, compress=3)
        return out

    @staticmethod
    def load(path: Path) -> "HurdleCountModel":
        return joblib.load(path)


# Backward-compatibility alias: CountModel was used by inference code.
# All count models are now hurdle; this alias avoids breaking predict.py / backtest.
CountModel = HurdleCountModel


# ── Rate × Duration decomposition ─────────────────────────────────────────────

_N_MC = 2000  # Monte Carlo draws for rate × duration integration


class RateXDurationCDF:
    """Count CDF derived from rate × duration Monte Carlo integration.

    Constructed with a sorted array of MC count draws; cdf(x) uses binary search.
    Mirrors the HurdlePropCDF public interface (cdf, p_over, p_under, quantile,
    median, uncertainty_band) so prop valuation and Monte Carlo simulator do not
    need to change.

    Parameters
    ----------
    count_samples : np.ndarray
        Raw (unsorted) MC count samples.
    p_zero : float
        P(count == 0) from the hurdle Stage 1; stored for reference.
    """

    def __init__(self, count_samples: np.ndarray, p_zero: float):
        self._samples = np.sort(count_samples.astype(float))
        self._p_zero = float(p_zero)
        self._n = len(self._samples)

    def cdf(self, x: float) -> float:
        """P(count <= x)."""
        if x < 0:
            return 0.0
        return float(np.searchsorted(self._samples, x, side="right") / self._n)

    def p_over(self, line: float) -> float:
        # Discrete-coherence at a sub-1 line (integer counts have NO mass in (0,1)):
        # the only 'over' event is count >= 1, whose probability is the separately-
        # calibrated hurdle P(>0) = 1 - p_zero.  The continuous rate x duration MC
        # leaks active mass into (0, 0.5], so 1 - cdf(0.5) deflates p_over(0.5) ~1-11%
        # below the hurdle (costs legitimate edge picks at the served 0.5 line for
        # td / kd / sub_att / r1_td) and breaks p_over + p_under == 1.  Match the
        # already-coherent HurdlePropCDF here; the Gate B PIT uses cdf() not p_over(),
        # so this is gate-neutral.  Lines >= 1 keep the empirical tail (the MC samples
        # for 'k landed' scatter around k, so 1 - cdf(floor(line)) would over-count).
        if line < 1.0:
            return 1.0 - self._p_zero
        return 1.0 - self.cdf(line)

    def p_under(self, line: float) -> float:
        return self.cdf(line - 0.5)

    def quantile(self, q: float) -> float:
        return float(np.quantile(self._samples, float(np.clip(q, 0.0, 1.0))))

    @property
    def median(self) -> float:
        return float(np.median(self._samples))

    def uncertainty_band(self, line: float) -> tuple[float, float]:
        p = self.p_over(line)
        se = np.sqrt(p * (1.0 - p) / 1000.0)
        return (max(0.0, p - 1.28 * se), min(1.0, p + 1.28 * se))


class RateHurdleCountModel:
    """Two-stage hurdle on per-minute rate, paired with a DurationCDF at inference.

    Stage 1: P(count > 0) via LGBMClassifier on raw features.
    Stage 2: 25-quantile LGBM on log(rate) for positive rows only,
             where rate = count / active_minutes.

    At predict_cdf time the caller supplies a DurationCDF per row.  The output
    is a RateXDurationCDF built from _N_MC Monte Carlo draws of (rate, duration).

    This fixes the regime-mixture problem in HurdleCountModel: the raw-count
    quantile model learned "when does the fight end" (dominated by
    referee_stoppage_threshold) rather than "how active is this fighter".
    By predicting rate and multiplying by sampled duration, the two sources of
    variation are separated.

    Parameters
    ----------
    target : str
        Prop name used for file naming and log messages.
    active_minutes_ceiling : float | None
        Hard ceiling on active minutes when integrating MC samples.
        None = use scheduled_rounds × 5 min (full fight).
        5.0 = cap at round 1 (for r1_sig_strikes).
    """

    def __init__(self, target: str = "sig_strikes",
                 active_minutes_ceiling: float | None = None,
                 rate_ceiling: float | None = None):
        self.target = target
        self.active_minutes_ceiling = active_minutes_ceiling
        # Maximum allowed rate (count/active_minute) — used for ctrl_time to enforce
        # per-fighter control ≤ fight duration on every MC draw. None = no clip.
        self.rate_ceiling: float | None = rate_ceiling
        self.pos_clf: lgb.LGBMClassifier | None = None
        self.quantile_models: list[lgb.LGBMRegressor] = []
        self.feature_cols: list[str] = []
        # bias_shift kept for backward-compat but never used (0.0).
        self.bias_shift: float = 0.0
        # ── Step 12: rate-duration coupling (ceiling-bounded models only) ────
        self._dur_alpha: float = 0.0
        self._log_mean_dur_frac: float = 0.0
        self._has_dur_coupling: bool = False
        # ── Method-conditional adjustments (v8.1+) ──────────────────────────
        # Fitted via fit_method_adjustments() after the main model is trained.
        # method_log_rate_adj: mean log-rate residual per method (kept for sig_strikes).
        # method_logodds_hurdle_adj: log-odds residual per method (stored but NOT
        #   applied in predict_cdf — double-counts duration effect for takedowns).
        self.method_log_rate_adj: dict[str, float] | None = None
        self.method_logodds_hurdle_adj: dict[str, float] | None = None
        # ── R1 finish-regime component (v8.5, replaces v8.2–v8.3 burst) ─────
        # For R1-finish draws the generative process is a burst (y ≈ a + b·t),
        # not rate×t accumulation. Fitted OLS on all R1-finish training rows.
        # At inference, finish draws are sampled as Poisson(max(a + b·t, 0.1))
        # — captures count structure and growing variance with no outlier risk.
        # r1_finish_fitted=True signals that a/b are available; the pool attribute
        # is kept as None (retired) for backward compat with any pickled models
        # that stored the empirical residual pool.
        self.r1_finish_intercept: float = 0.0
        self.r1_finish_slope: float = 0.0
        self.r1_finish_resid_pool: np.ndarray | None = None  # retired; use Poisson
        self.r1_finish_fitted: bool = False
        self.r1_burst_intercept: float = 0.0   # retired — backward compat only
        self.r1_burst_slope: float = 0.0        # retired — backward compat only
        # ── Duration-binned method rate adjustments (v8.5, takedowns only) ──
        # Replaces the flat per-method scalar (which was a duration confound) with
        # (method × duration-quartile) residuals. Applied per-draw at inference
        # when use_binned_rate_adj=True is passed to predict_cdf.
        self.method_log_rate_adj_binned: dict[str, list[float]] | None = None
        self._dur_bin_edges: np.ndarray | None = None
        # ── Finish-count head (v8.6, r1 only) ────────────────────────────────────
        # 25-quantile LGBM on raw R1 sig-strike count (NOT rate × t), trained on
        # r1_end rows. Removes the finish-time variance the rate×t coupling injects
        # for short finishes. Activated by use_finish_head=True in predict_cdf.
        self.finish_count_quantile_models: list = []
        self.finish_head_fitted: bool = False
        self._finish_head_has_t: bool = False
        self._finish_head_has_method: bool = False  # v8.9: method-conditional head
        self._finish_head_method_prior = None       # v8.9: [KO,SUB,DEC] fallback mix
        # ── Conditional hurdle (v8.6, full-fight models) ──────────────────────────
        # pos_clf_cond is trained on (features, method_onehot, log_active_min) so
        # that predict_cdf can assign per-draw P(>0 TD | sampled_method, sampled_dur)
        # in the MC loop. Activated by use_cond_hurdle=True in predict_cdf.
        self.pos_clf_cond = None
        self._cond_hurdle_aug_cols: list[str] = []
        self._has_cond_hurdle: bool = False
        # ── Val-anchored rate calibration factor (v8.10, sig_strikes only) ──────
        # Multiplicative factor applied to all log-rate predictions at inference.
        # Computed on the 2023 validation split: factor = mean(actual)/mean(predicted).
        # Corrects the ~4% in-sample rate-level over-prediction without chasing
        # the out-of-sample 2025 UFC pace drop. 1.0 = no-op (backward compat).
        self.rate_calib_factor: float = 1.0
        # Drift lever: marginal-path shrink of R1 finish-draw counts (r1 only). 1.0=no-op.
        self.finish_draw_scale: float = 1.0
        # v8.11: dispersion factor for the finish-count head (r1 only) ──────
        # Applied in predict_cdf to widen quantiles around the per-row median,
        # correcting under-dispersion in the r1_end conditional-null diagnostic
        # (oracle KS 0.242→~0.118 at factor 2.0; head spread was ~half empirical).
        # Tuned on the val split in fit_method_adjustments; 1.0 = no-op.
        self.finish_head_disp_factor: float = 1.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            active_minutes_train: np.ndarray,
            X_val: pd.DataFrame, y_val: pd.Series,
            active_minutes_val: np.ndarray,
            feature_cols: list[str],
            sample_weight: np.ndarray | None = None,
            temporal_oof: bool = False,
            train_dates: "pd.Series | None" = None) -> "RateHurdleCountModel":
        """Fit rate hurdle model.

        Parameters
        ----------
        active_minutes_train / active_minutes_val : np.ndarray
            Active fight minutes per row.  For sig_strikes: total_fight_sec / 60.
            For r1_sig_strikes: min(end_time_sec, 300) / 60 if end_round==1 else 5.0.
        temporal_oof : bool
            When True (prod mode only), use a genuine temporal holdout to find
            best_iteration_ for each LGBM, then refit on ALL train rows at that
            fixed tree count.  Prevents the in-sample-val early-stopping leak that
            occurs when split_prod.yaml val window overlaps the train window.
        train_dates : pd.Series | None
            event_date column aligned with X_train rows.  Required when temporal_oof=True.
        """
        self.feature_cols = feature_cols
        cfg = _cfg()
        lgbm_cfg = cfg["lgbm_quantile"]

        eps_min = 5.0 / 60.0  # 5-second floor prevents rate blow-up on instant stoppages

        y_tr = y_train.fillna(0).clip(lower=0).values.astype(float)
        y_vl = y_val.fillna(0).clip(lower=0).values.astype(float)

        X_tr = X_train[feature_cols].fillna(0).values
        X_vl = X_val[feature_cols].fillna(0).values

        act_tr = np.maximum(np.asarray(active_minutes_train, dtype=float), eps_min)
        act_vl = np.maximum(np.asarray(active_minutes_val, dtype=float), eps_min)

        rate_tr = y_tr / act_tr
        rate_vl = y_vl / act_vl  # used only for pos_clf eval set (binary label)

        # Stage 1: P(rate > 0) == P(count > 0)
        print(f"  Fitting hurdle binary classifier for {self.target} (rate)...")
        y_pos_tr = (rate_tr > 0).astype(int)
        y_pos_vl = (rate_vl > 0).astype(int)
        self.pos_clf = lgb.LGBMClassifier(
            n_estimators=lgbm_cfg["n_estimators"],
            num_leaves=lgbm_cfg["num_leaves"],
            learning_rate=lgbm_cfg["learning_rate"],
            min_child_samples=lgbm_cfg["min_child_samples"],
            verbosity=-1,
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
        if temporal_oof and train_dates is not None:
            _h_best_n = _oof_best_n(self.pos_clf, X_tr, y_pos_tr, train_dates, sample_weight)
            if _h_best_n is not None:
                print(f"    [temporal_oof] {self.target} hurdle best_n={_h_best_n}")
                self.pos_clf.set_params(n_estimators=_h_best_n)
                self.pos_clf.fit(X_tr, y_pos_tr, **sw_kw)
            else:
                self.pos_clf.fit(
                    X_tr, y_pos_tr,
                    eval_set=[(X_vl, y_pos_vl)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(period=0)],
                    **sw_kw,
                )
        else:
            self.pos_clf.fit(
                X_tr, y_pos_tr,
                eval_set=[(X_vl, y_pos_vl)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=0)],
                **sw_kw,
            )

        # Stage 2: 25-quantile regression on log(rate) for positive rows only.
        # v8.9 (Fix A): in R1-ceiling mode the rate model's sole job in the marginal
        # path is full-round pace (rate × 5min) — finish draws are owned by the
        # finish-count head.  Pooling r1_end rows injects inflated short-finish rates
        # (sub-60s finishes ≈ 10 strikes/min once divided by tiny active-minutes),
        # biasing rate×5 ~+18% high for past_r1.  Restrict the rate fit to full-round
        # rows (act >= ceiling) so the learned pace matches survivors (~3.3/min).
        # Stage-1 hurdle above stays on ALL rows (marginal P(count>0) is unchanged).
        print(f"  Fitting LGBM quantiles on positives for {self.target} (rate)...")
        pos_mask = rate_tr > 0
        pos_mask_vl = rate_vl > 0
        if self.active_minutes_ceiling is not None and self.active_minutes_ceiling <= 5.0:
            _ceil_min = float(self.active_minutes_ceiling)
            _full_tr = np.asarray(active_minutes_train, dtype=float) >= (_ceil_min - 0.01)
            _full_vl = np.asarray(active_minutes_val, dtype=float) >= (_ceil_min - 0.01)
            n_full_tr = int((pos_mask & _full_tr).sum())
            if n_full_tr >= 50:  # enough full-round rows to fit cleanly
                pos_mask = pos_mask & _full_tr
                pos_mask_vl = pos_mask_vl & _full_vl
                print(f"    [Fix A] r1 rate fit restricted to full-round rows: "
                      f"n={n_full_tr} (was {int((rate_tr > 0).sum())})")
        X_pos = X_tr[pos_mask]
        log_rate_pos = np.log(rate_tr[pos_mask])  # log(strikes / active_minute)
        sw_pos = sample_weight[pos_mask] if sample_weight is not None else None

        # Val positives for early stopping
        X_vl_pos = X_vl[pos_mask_vl]
        log_rate_vl_pos = np.log(rate_vl[pos_mask_vl])

        # Temporal-OOF: probe best tree count once (median q), reuse for all quantiles.
        _q_best_n: "int | None" = None
        if temporal_oof and train_dates is not None and len(X_pos) > 20:
            _dates_pos = pd.Series(
                pd.to_datetime(
                    train_dates.values if hasattr(train_dates, "values") else list(train_dates)
                )[pos_mask]
            )
            _probe_q = lgb.LGBMRegressor(
                objective="quantile", alpha=0.5,
                n_estimators=lgbm_cfg["n_estimators"],
                learning_rate=lgbm_cfg["learning_rate"],
                num_leaves=lgbm_cfg["num_leaves"],
                min_child_samples=lgbm_cfg["min_child_samples"],
                verbosity=-1, random_state=SEED,
                deterministic=True, force_row_wise=True, num_threads=1,
                feature_fraction_seed=SEED, bagging_seed=SEED,
                data_random_seed=SEED, extra_seed=SEED, objective_seed=SEED,
            )
            _q_best_n = _oof_best_n(_probe_q, X_pos, log_rate_pos, _dates_pos, sw_pos,
                                    min_hold=50, frac=0.2)
            if _q_best_n is not None:
                print(f"    [temporal_oof] {self.target} quantile best_n={_q_best_n}")

        self.quantile_models = []
        for q in QUANTILE_GRID:
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=lgbm_cfg["n_estimators"],
                learning_rate=lgbm_cfg["learning_rate"],
                num_leaves=lgbm_cfg["num_leaves"],
                min_child_samples=lgbm_cfg["min_child_samples"],
                verbosity=-1,
                random_state=SEED,
                # ── Determinism flags (Step 1) ───────────────────────────
                deterministic=True,
                force_row_wise=True,
                num_threads=1,
                feature_fraction_seed=SEED,
                bagging_seed=SEED,
                data_random_seed=SEED,
                extra_seed=SEED,
                objective_seed=SEED,
            )
            fit_kw_pos = {"sample_weight": sw_pos} if sw_pos is not None else {}
            if temporal_oof and _q_best_n is not None and len(X_pos) > 20:
                m.set_params(n_estimators=_q_best_n)
                m.fit(X_pos, log_rate_pos, **fit_kw_pos)
            elif len(X_pos) > 20 and len(X_vl_pos) > 5:
                m.fit(X_pos, log_rate_pos,
                      eval_set=[(X_vl_pos, log_rate_vl_pos)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=0)],
                      **fit_kw_pos)
            elif len(X_pos) > 20:
                m.fit(X_pos, log_rate_pos, **fit_kw_pos)
            self.quantile_models.append(m)

        return self

    def predict_cdf(
        self,
        X: pd.DataFrame,
        duration_cdfs: list | None = None,
        active_minutes_ceiling: float | None = None,
        scheduled_rounds: np.ndarray | None = None,
        method_proba: np.ndarray | None = None,
        duration_cdfs_by_method: dict | None = None,
        apply_burst: bool = True,
        apply_method_hurdle: bool = False,
        use_binned_rate_adj: bool = False,
        force_r1_end: bool = False,
        force_full_round: bool = False,
        use_finish_head: bool = False,
        use_cond_hurdle: bool = False,
        mean_preserve_cond_hurdle: bool = True,
        use_sub_count_head: bool = False,
        hurdle_floor: float | None = None,
        method_log_rate_adj_override: dict | None = _UNSET,
    ) -> list[RateXDurationCDF]:
        """Predict count CDF for each row by integrating rate × sampled duration.

        Parameters
        ----------
        X : pd.DataFrame
            Feature rows (same schema as training).
        duration_cdfs : list[DurationCDF] | None
            One DurationCDF per row (marginal, used when duration_cdfs_by_method is
            not supplied or in R1-ceiling mode).  If None, uniform fallback.
        active_minutes_ceiling : float | None
            Overrides self.active_minutes_ceiling per call.
        scheduled_rounds : np.ndarray | None
            Scheduled rounds per row (full-fight ceiling when no ceiling set).
        method_proba : np.ndarray | None
            Shape (n_rows, 3) [P(KO/TKO), P(SUB), P(DEC)].  When supplied,
            each draw samples a method used to select the duration CDF (if
            duration_cdfs_by_method is given) and apply per-method rate adjustment.
            The hurdle is NOT adjusted per-method unless apply_method_hurdle=True.
        duration_cdfs_by_method : dict[str, list[DurationCDF]] | None
            Keys "KO/TKO", "SUB", "DEC"; values are one DurationCDF per row.
            When supplied alongside method_proba, each draw samples duration from
            its method-specific CDF so KO draws are short and DEC draws are long,
            creating a coherent joint (method, duration, count).
        use_binned_rate_adj : bool
            If True and method_log_rate_adj_binned is fitted, apply per-draw
            (method × duration-bin) rate adjustments instead of the flat per-method
            scalar. Used for takedowns to replace the zeroed flat adjustment with a
            duration-stratified correction that avoids the duration confound.
        force_r1_end : bool
            If True (R1-ceiling mode only), every active draw samples a finish time
            from the duration CDF rather than using the Bernoulli split. Produces the
            conditional forecast F(count | fight ends in R1), used to compute the valid
            per-segment PIT null for the r1_end sub-segment.
        force_full_round : bool
            If True (R1-ceiling mode only), every active draw is a full-round draw
            (p_r1_end forced to 0.0, dur_sec = ceil_sec = 300s).  Produces the
            conditional forecast F(count | fight survives R1), symmetric to
            force_r1_end, used for the valid past_r1 conditional-null diagnostic.
        use_finish_head : bool
            If True and finish_head_fitted, replace rate×t for finish draws with
            inverse-CDF sampling from finish_count_quantile_models. This removes the
            finish-time variance source that rate×t coupling injects (the model learned
            y ≈ const, not y ≈ rate×t). Only active in R1-ceiling mode.
        use_cond_hurdle : bool
            If True and _has_cond_hurdle and method_proba supplied, replace the global
            hurdle P(>0) with a per-draw probability looked up from the conditional
            table P(>0 | row, method, dur). Mean-preserving rescaling ensures the
            MC-marginal matches pos_clf. Full-fight models only (takedowns).
        mean_preserve_cond_hurdle : bool
            If True (default), rescale _p_h_draws in logit space so their per-row
            mean equals p_pos_all[i] (the globally-calibrated hurdle).  Set False
            for forced-method override calls (KO/SUB/DEC-forced forecasts used for
            segment-conditional PIT) so they stay at P(>0|method) not at the
            method-marginal global hurdle.
        use_sub_count_head : bool
            If True and sub_head_fitted, replace rate×dur for SUB-method draws with
            samples from the SUB count head conditioned on log_active_min.  Captures
            the front-loaded TD distribution in submission fights.  Full-fight only.
        method_log_rate_adj_override : dict | None
            When passed (including None), use this value instead of
            self.method_log_rate_adj for this call only — does NOT mutate the
            instance.  Callers that previously did
            `m.method_log_rate_adj = None; try: ...; finally: restore` should pass
            `method_log_rate_adj_override=None` instead (thread-safe: no shared
            mutable state).

        Returns
        -------
        list[RateXDurationCDF]
        """
        # Lazy import to avoid circular dependency
        # (props_duration imports QUANTILE_GRID from this module)
        from ufc.models.props_duration import (  # noqa: PLC0415
            duration_inverse_cdf, _build_dur_cdf_grid,
        )

        Xf = X.reindex(columns=self.feature_cols, fill_value=0).fillna(0)
        Xa = Xf.values
        n_rows = len(Xf)

        # P(rate > 0) per row
        p_pos_all = self.pos_clf.predict_proba(Xa)[:, 1]

        # Hurdle floor (serving-only; gate scripts call predict_cdf without it so
        # Gate B is byte-identical).  Rescues thin-data fighters: when a fighter
        # has no round-level history, the R1 career features (r1_sig_str_ctd, …)
        # come in as 0, which the hurdle classifier reads as "lands zero R1 sig
        # strikes" → p_pos≈0 → a degenerate all-zero count CDF (P(under)=100%).
        # Floor to a sane near-certainty.  Empirical marginal P(>0 R1 sig)=0.965,
        # so a 0.90 floor never binds on healthy fighters (p_pos≈1.0) and only
        # corrects the OOD collapse.  Only meaningful for props where landing ≥1
        # is near-universal (r1_sig_strikes) — never pass for takedowns/finishes.
        if hurdle_floor is not None:
            p_pos_all = np.maximum(p_pos_all, float(hurdle_floor))

        qs = np.array(QUANTILE_GRID)
        n_q = len(qs)

        log_rate_preds = np.column_stack(
            [m.predict(Xa) for m in self.quantile_models]
        )
        log_rate_preds = np.maximum.accumulate(log_rate_preds, axis=1)

        # v8.10: val-anchored rate calibration (sig_strikes only; 1.0 no-op for others).
        _rcf = getattr(self, "rate_calib_factor", 1.0)
        if _rcf != 1.0:
            log_rate_preds = log_rate_preds + np.log(_rcf)

        # Ceiling in seconds per row
        ceiling = active_minutes_ceiling if active_minutes_ceiling is not None \
            else self.active_minutes_ceiling
        if ceiling is not None:
            ceiling_sec_arr = np.full(n_rows, float(ceiling) * 60.0)
        else:
            if scheduled_rounds is not None:
                sr = np.asarray(scheduled_rounds, dtype=float)
            elif "scheduled_rounds" in X.columns:
                sr = X["scheduled_rounds"].fillna(3).values.astype(float)
            else:
                sr = np.full(n_rows, 3.0)
            ceiling_sec_arr = sr * 300.0

        # ── Step 9: Pre-build CDF grids outside the row loop ─────────────────
        # _build_dur_cdf_grid uses one vectorised numpy interp instead of 512
        # Python-level cdf() calls.  Build one grid per row (each row has its
        # own DurationCDF with unique _lgbm_qv, so grids cannot be shared).
        prebuilt_grids: list[tuple[np.ndarray, np.ndarray] | None] = []
        p_r1_end_arr: list[float] = []  # for R1 ceiling rows

        for i in range(n_rows):
            dur_cdf_i = (duration_cdfs[i]
                         if duration_cdfs is not None and i < len(duration_cdfs)
                         else None)
            ceil_sec_i = float(ceiling_sec_arr[i])
            if dur_cdf_i is not None:
                effective_ceiling = min(ceil_sec_i, float(dur_cdf_i._scheduled_sec))
                prebuilt_grids.append(_build_dur_cdf_grid(dur_cdf_i, effective_ceiling))
                if ceil_sec_i <= 300.0:
                    p_r1_end_arr.append(float(min(1.0, dur_cdf_i.cdf(ceil_sec_i))))
                else:
                    p_r1_end_arr.append(0.0)
            else:
                prebuilt_grids.append(None)
                p_r1_end_arr.append(0.0)

        rng = np.random.default_rng(SEED)

        # Pre-cache rate adjustment array (hurdle adjustment dropped — double-counts duration)
        _effective_rate_adj = (
            self.method_log_rate_adj
            if method_log_rate_adj_override is _UNSET
            else method_log_rate_adj_override
        )
        _method_names = ["KO/TKO", "SUB", "DEC"]
        _rate_adj_arr = np.zeros(3)
        _has_rate_adj = bool(_effective_rate_adj)
        if _has_rate_adj:
            for _k, _m in enumerate(_method_names):
                _rate_adj_arr[_k] = _effective_rate_adj.get(_m, 0.0)

        # Pre-cache duration-binned rate adjustment tables (v8.5, takedowns).
        _binned_rate_adj = getattr(self, "method_log_rate_adj_binned", None)
        _dur_bin_edges = getattr(self, "_dur_bin_edges", None)
        _has_binned_rate_adj = (
            use_binned_rate_adj
            and _binned_rate_adj is not None
            and _dur_bin_edges is not None
            and len(_dur_bin_edges) >= 2
        )

        # ── Pre-build method-conditional CDF grids (takedowns coherence fix) ───
        # Three grid lists (KO/SUB/DEC), one entry per row.  Used in the MC loop
        # when method_proba and duration_cdfs_by_method are both supplied so each
        # draw samples duration from its method's CDF (coherent joint).
        _method_grids: list[list] | None = None
        if duration_cdfs_by_method is not None:
            _method_grids = [[], [], []]  # indices: 0=KO, 1=SUB, 2=DEC
            for _mc, _mn in enumerate(("KO/TKO", "SUB", "DEC")):
                _cdfs_m = duration_cdfs_by_method.get(_mn, [])
                for _mi in range(n_rows):
                    _dcdf_m = _cdfs_m[_mi] if _mi < len(_cdfs_m) else None
                    _csec_i = float(ceiling_sec_arr[_mi])
                    if _dcdf_m is not None:
                        _eff_ceil = min(_csec_i, float(_dcdf_m._scheduled_sec))
                        _method_grids[_mc].append(_build_dur_cdf_grid(_dcdf_m, _eff_ceil))
                    else:
                        _method_grids[_mc].append(None)

        # ── Method-conditional hurdle pre-computation (v8.4) ─────────────────
        # Fixes KO_finish zero-mass: KO fights skip grappling phases so
        # P(>0 TD | KO) < marginal.  Duration coupling cannot move the mass
        # at zero — only the hurdle stage can.
        # Mean-preserving re-centering: solve per-row c_i so that
        # Σ_k mprob[k] · sigmoid(logit_base + c_i + ladj[k]) = p_pos_i,
        # preserving the mixture-mean hurdle (and thus the overall KS gate).
        _p_pos_m: np.ndarray | None = None  # (n_rows, 3) per-method hurdle probs
        _apply_mh = bool(
            apply_method_hurdle
            and getattr(self, "method_logodds_hurdle_adj", None)
            and method_proba is not None
        )
        if _apply_mh:
            _ladj = np.array([
                self.method_logodds_hurdle_adj.get("KO/TKO", 0.0),
                self.method_logodds_hurdle_adj.get("SUB",    0.0),
                self.method_logodds_hurdle_adj.get("DEC",    0.0),
            ])
            _ladj = np.clip(_ladj, -4.0, 4.0)  # guard against extreme training residuals
            _eps = 1e-6
            _p_pos_safe = np.clip(p_pos_all, _eps, 1.0 - _eps)
            _logit_base = np.log(_p_pos_safe / (1.0 - _p_pos_safe))  # (n_rows,)
            _mprob = np.asarray(method_proba, dtype=float)[:n_rows]  # (n_rows, 3)
            # Vectorised bisection: 50 iterations → precision < 1e-14
            _lo = np.full(n_rows, -10.0)
            _hi = np.full(n_rows, 10.0)
            for _ in range(50):
                _mid = (_lo + _hi) * 0.5
                _logit_m = _logit_base[:, None] + _mid[:, None] + _ladj[None, :]
                _mix = np.sum(_mprob * (1.0 / (1.0 + np.exp(-_logit_m))), axis=1)
                _lo = np.where(_mix < _p_pos_safe, _mid, _lo)
                _hi = np.where(_mix < _p_pos_safe, _hi, _mid)
            _c = (_lo + _hi) * 0.5
            _logit_final = _logit_base[:, None] + _c[:, None] + _ladj[None, :]
            _p_pos_m = 1.0 / (1.0 + np.exp(-_logit_final))  # (n_rows, 3)

        # ── Pre-compute finish-count head predictions (v8.7, r1 only) ────────
        # v8.7: t-conditional head — (n_rows, n_t_grid, n_q) table so each
        # finish draw samples count conditioned on its sampled finish time t.
        # Safe in the mixed path: early-finish draws (~t→0) get ~2.5 strikes,
        # late-finish draws (~t=295s) get ~20.  No force_r1_end guard needed.
        # v8.6 legacy path kept for backward compat with old pickled models.
        _N_T_GRID = 10
        _T_GRID_SEC = np.linspace(5.0, 295.0, _N_T_GRID)  # finish-time grid [5s, 295s]
        _finish_head_preds: np.ndarray | None = None          # legacy (n_rows, n_q)
        _finish_head_t_preds: np.ndarray | None = None        # v8.7  (n_rows, n_t_grid, n_q)
        _finish_head_tm_preds: np.ndarray | None = None       # v8.9  (n_rows, 3, n_t_grid, n_q)
        _has_finish_head_method = (
            use_finish_head
            and getattr(self, "finish_head_fitted", False)
            and bool(self.finish_count_quantile_models)
            and getattr(self, "_finish_head_has_method", False)
        )
        _has_finish_head_t = (
            use_finish_head
            and getattr(self, "finish_head_fitted", False)
            and bool(self.finish_count_quantile_models)
            and getattr(self, "_finish_head_has_t", False)
            and not _has_finish_head_method
        )
        _has_finish_head_legacy = (
            use_finish_head
            and getattr(self, "finish_head_fitted", False)
            and bool(self.finish_count_quantile_models)
            and not getattr(self, "_finish_head_has_t", False)
            and not _has_finish_head_method
        )
        # v8.11: dispersion factor for the finish-count head (widen quantiles around median).
        _fhd_f = getattr(self, "finish_head_disp_factor", 1.0)
        _fhd_med_idx = int(np.argmin(np.abs(qs - 0.5)))  # index of q=0.50 in QUANTILE_GRID
        if _has_finish_head_method:
            # v8.9: (n_rows, 3-method, n_t_grid, n_q) table.  Feature order at fit
            # time was [features, t_finish_sec, ko, sub, dec]; build the same here.
            _finish_head_tm_preds = np.zeros((n_rows, 3, _N_T_GRID, n_q))
            for _mj in range(3):
                _oh = np.zeros((n_rows, 3)); _oh[:, _mj] = 1.0
                for _ti, _t_sec in enumerate(_T_GRID_SEC):
                    _t_col = np.full((n_rows, 1), _t_sec)
                    _Xa_tm = np.column_stack([Xa, _t_col, _oh])
                    _preds_tm = np.column_stack(
                        [m.predict(_Xa_tm) for m in self.finish_count_quantile_models]
                    )
                    _preds_tm = np.maximum.accumulate(np.maximum(_preds_tm, 0.0), axis=1)
                    if _fhd_f != 1.0:
                        _med_col = _preds_tm[:, _fhd_med_idx:_fhd_med_idx + 1]
                        _preds_tm = np.maximum.accumulate(
                            np.maximum(_med_col + _fhd_f * (_preds_tm - _med_col), 0.0), axis=1
                        )
                    _finish_head_tm_preds[:, _mj, _ti, :] = _preds_tm
        elif _has_finish_head_t:
            _finish_head_t_preds = np.zeros((n_rows, _N_T_GRID, n_q))
            for _ti, _t_sec in enumerate(_T_GRID_SEC):
                _t_col = np.full((n_rows, 1), _t_sec)
                _Xa_t = np.column_stack([Xa, _t_col])
                _preds_t = np.column_stack(
                    [m.predict(_Xa_t) for m in self.finish_count_quantile_models]
                )
                _preds_t = np.maximum.accumulate(np.maximum(_preds_t, 0.0), axis=1)
                _finish_head_t_preds[:, _ti, :] = _preds_t
        elif _has_finish_head_legacy:
            _finish_head_preds = np.column_stack(
                [m.predict(Xa) for m in self.finish_count_quantile_models]
            )
            _finish_head_preds = np.maximum.accumulate(
                np.maximum(_finish_head_preds, 0.0), axis=1
            )

        # ── Pre-compute SUB count head predictions (v8.7, full-fight only) ───
        # Shape (n_rows, n_dur_grid, n_q): quantile vectors at log-active-min grid.
        # Applied per-draw for SUB-method draws to replace rate×duration with a
        # head that captures the front-loaded TD distribution in SUB fights.
        _N_DUR_GRID_SUB = 10
        _SUB_LOG_DUR_GRID = np.linspace(np.log(5.0 / 60.0), np.log(25.0), _N_DUR_GRID_SUB)
        _sub_head_t_preds: np.ndarray | None = None
        _has_sub_head = (
            use_sub_count_head
            and getattr(self, "sub_head_fitted", False)
            and bool(getattr(self, "sub_count_quantile_models", None))
        )
        if _has_sub_head:
            _sub_head_t_preds = np.zeros((n_rows, _N_DUR_GRID_SUB, n_q))
            for _dgi, _log_d in enumerate(_SUB_LOG_DUR_GRID):
                _d_col = np.full((n_rows, 1), _log_d)
                _Xa_sub = np.column_stack([Xa, _d_col])
                _preds_sub = np.column_stack(
                    [m.predict(_Xa_sub) for m in self.sub_count_quantile_models]
                )
                _preds_sub = np.maximum.accumulate(np.maximum(_preds_sub, 0.0), axis=1)
                _sub_head_t_preds[:, _dgi, :] = _preds_sub

        # ── Pre-compute conditional hurdle tables (v8.6, full-fight models) ──
        # For each row: (3 methods × _N_DUR_PTS) P(>0 | features, method, log_dur).
        # Built via a small augmented LGBM predict; avoids N_MC×n_rows inference.
        _N_DUR_PTS = 11
        _cond_hurdle_tables: list[np.ndarray] | None = None
        _cond_hurdle_log_dur_grids: list[np.ndarray] | None = None
        _has_cond_hurdle_flag = (
            use_cond_hurdle
            and getattr(self, "_has_cond_hurdle", False)
            and self.pos_clf_cond is not None
            and method_proba is not None
        )
        if _has_cond_hurdle_flag:
            _cond_hurdle_tables = []
            _cond_hurdle_log_dur_grids = []
            for _ci in range(n_rows):
                _ceil_sec_ci = float(ceiling_sec_arr[_ci])
                _log_lo = np.log(5.0 / 60.0)
                _log_hi = np.log(max(_ceil_sec_ci / 60.0, 5.0 / 60.0 + 0.01))
                _log_dur_grid_ci = np.linspace(_log_lo, _log_hi, _N_DUR_PTS)
                _cond_hurdle_log_dur_grids.append(_log_dur_grid_ci)
                _n_aug = 3 * _N_DUR_PTS
                _Xa_row = np.tile(Xa[_ci : _ci + 1], (_n_aug, 1))
                _method_onehot_aug = np.zeros((_n_aug, 3))
                _log_act_aug = np.zeros(_n_aug)
                for _mci in range(3):
                    _s = _mci * _N_DUR_PTS
                    _e = _s + _N_DUR_PTS
                    _method_onehot_aug[_s:_e, _mci] = 1.0
                    _log_act_aug[_s:_e] = _log_dur_grid_ci
                _Xa_aug_ci = np.column_stack(
                    [_Xa_row, _method_onehot_aug, _log_act_aug[:, None]]
                )
                _probs_ci = self.pos_clf_cond.predict_proba(_Xa_aug_ci)[:, 1]
                _cond_hurdle_tables.append(_probs_ci.reshape(3, _N_DUR_PTS))

        cdfs: list[RateXDurationCDF] = []

        for i in range(n_rows):
            p_pos = float(p_pos_all[i])
            p_zero = 1.0 - p_pos
            ceil_sec = float(ceiling_sec_arr[i])

            # ── Method-conditional duration coupling (v8.3) ──────────────────
            # Sample per-draw method codes when needed for (a) method-conditional
            # duration CDFs or (b) per-draw rate adjustment.  The hurdle is NOT
            # adjusted per-method: the v8.2 hurdle adjustment double-counted the
            # duration effect, collapsing the decision-segment KS 0.043→0.218.
            mprob_i = (method_proba[i]
                       if method_proba is not None and i < len(method_proba)
                       else None)
            _use_method = (mprob_i is not None
                           and (_has_rate_adj or _method_grids is not None
                                or _apply_mh or _has_finish_head_method))
            if _use_method:
                ko_p = float(np.clip(mprob_i[0], 0.0, 1.0))
                sub_p = float(np.clip(mprob_i[1], 0.0, 1.0))
                u_m = rng.random(_N_MC)
                method_codes = np.where(u_m < ko_p, 0,
                               np.where(u_m < ko_p + sub_p, 1, 2))
            else:
                method_codes = None
            # ── Conditional hurdle fast-path (v8.6, full-fight models only) ──────
            # Sample ALL N_MC durations BEFORE the hurdle so each draw's P(>0 TD)
            # is looked up at its actual (method, dur) instead of a grid mean.
            # This avoids double-counting and correctly gives P(>0|KO,short)≈0.13.
            # The fast-path fires, appends the CDF, and continues to the next row.
            if (
                _has_cond_hurdle_flag
                and _cond_hurdle_tables is not None
                and mprob_i is not None
                and ceil_sec > 300.0
            ):
                _ko_p_ch = float(np.clip(mprob_i[0], 0.0, 1.0))
                _sub_p_ch = float(np.clip(mprob_i[1], 0.0, 1.0))
                # Sample method codes if not already done
                if method_codes is None:
                    u_m_ch = rng.random(_N_MC)
                    method_codes = np.where(
                        u_m_ch < _ko_p_ch, 0,
                        np.where(u_m_ch < _ko_p_ch + _sub_p_ch, 1, 2),
                    )
                # Step 1: Sample ALL N_MC durations (method-conditional)
                _dur_pre = np.full(_N_MC, ceil_sec, dtype=float)
                if _method_grids is not None:
                    for _mc_pre in range(3):
                        _m_pre = method_codes == _mc_pre
                        _n_pre = int(_m_pre.sum())
                        if _n_pre == 0:
                            continue
                        _g_pre = _method_grids[_mc_pre][i]
                        if _g_pre is not None:
                            _dur_pre[_m_pre] = duration_inverse_cdf(
                                None, rng.random(_n_pre), _prebuilt_grid=_g_pre
                            )
                elif prebuilt_grids[i] is not None:
                    _dur_pre = duration_inverse_cdf(
                        None, rng.random(_N_MC), _prebuilt_grid=prebuilt_grids[i]
                    )
                # Step 2: Per-draw P(>0) from (method, sampled_dur) table lookup
                _ch_table_i = _cond_hurdle_tables[i]  # (3, _N_DUR_PTS)
                _log_dur_grid_i = _cond_hurdle_log_dur_grids[i]
                _log_dur_pre = np.log(np.maximum(_dur_pre / 60.0, 5.0 / 3600.0))
                _p_h_draws = np.zeros(_N_MC)
                for _mc_pre in range(3):
                    _m_pre = method_codes == _mc_pre
                    if not _m_pre.any():
                        continue
                    _p_h_draws[_m_pre] = np.interp(
                        _log_dur_pre[_m_pre], _log_dur_grid_i, _ch_table_i[_mc_pre]
                    )
                # Mean-preserving rescale (v8.7): shift _p_h_draws in logit space
                # so mean(_p_h_draws) == p_pos_all[i], anchoring the cond-hurdle
                # marginal to the globally-calibrated hurdle.  Without this, the
                # conditional table extrapolates at long durations (5rd DEC at 25min
                # falls outside the training range) and the marginal P(>0) drifts,
                # breaking the 5rd gate.  Skipped for forced-method calls
                # (mean_preserve_cond_hurdle=False) so KO/SUB/DEC forecasts keep
                # their method-conditional P(>0), not the method-marginal target.
                if mean_preserve_cond_hurdle:
                    _eps_mp = 1e-6
                    _p_h_safe = np.clip(_p_h_draws, _eps_mp, 1.0 - _eps_mp)
                    _logit_h = np.log(_p_h_safe / (1.0 - _p_h_safe))
                    _target_safe = float(np.clip(p_pos_all[i], _eps_mp, 1.0 - _eps_mp))
                    # Bisection in logit space (50 iters → precision < 1e-14)
                    _lo_mp, _hi_mp = -10.0, 10.0
                    for _ in range(50):
                        _mid_mp = (_lo_mp + _hi_mp) * 0.5
                        _mix_mp = float(
                            np.mean(1.0 / (1.0 + np.exp(-(_logit_h + _mid_mp))))
                        )
                        if _mix_mp < _target_safe:
                            _lo_mp = _mid_mp
                        else:
                            _hi_mp = _mid_mp
                    _c_mp = (_lo_mp + _hi_mp) * 0.5
                    _p_h_draws = 1.0 / (1.0 + np.exp(-(_logit_h + _c_mp)))
                active_mask = rng.random(_N_MC) < _p_h_draws
                n_active_ch = int(active_mask.sum())
                _samples_ch = np.zeros(_N_MC, dtype=float)
                if n_active_ch > 0:
                    # Use pre-sampled durations for active draws
                    _dur_active = np.minimum(_dur_pre[active_mask], ceil_sec)
                    _act_min = np.maximum(_dur_active / 60.0, 5.0 / 60.0)
                    # Sample rate
                    _u_r = rng.random(n_active_ch)
                    _lrs = np.interp(_u_r, qs, log_rate_preds[i])
                    if _has_rate_adj and method_codes is not None:
                        _lrs += _rate_adj_arr[method_codes[active_mask]]
                    _rate_ch = np.exp(_lrs)
                    if self.rate_ceiling is not None:
                        _rate_ch = np.minimum(_rate_ch, self.rate_ceiling)
                    _samples_ch[active_mask] = np.maximum(_rate_ch * _act_min, 0.0)
                    # SUB count head override (v8.7): replace rate×dur for SUB draws
                    # with a sample from the duration-conditional SUB head.  Captures
                    # the front-loaded TD distribution (td/min falls with fight length).
                    if _has_sub_head and _sub_head_t_preds is not None:
                        _act_methods_ch = method_codes[active_mask]
                        _is_sub_ch = _act_methods_ch == 1  # index 1 = SUB
                        _n_sub_ch = int(_is_sub_ch.sum())
                        if _n_sub_ch > 0:
                            _sub_act_min = _act_min[_is_sub_ch]
                            _log_sub_durs = np.log(np.maximum(_sub_act_min, 5.0 / 60.0))
                            _sub_t_tbl = _sub_head_t_preds[i]  # (n_dur_grid, n_q)
                            _sub_q_vecs = np.zeros((_n_sub_ch, n_q))
                            for _qi in range(n_q):
                                _sub_q_vecs[:, _qi] = np.interp(
                                    _log_sub_durs, _SUB_LOG_DUR_GRID, _sub_t_tbl[:, _qi]
                                )
                            _sub_q_vecs = np.maximum.accumulate(
                                np.maximum(_sub_q_vecs, 0.0), axis=1
                            )
                            _u_sub_ch = rng.random(_n_sub_ch)
                            _idx_lo_s = np.clip(
                                np.searchsorted(qs, _u_sub_ch, side="right") - 1,
                                0, n_q - 2,
                            )
                            _alpha_s = (_u_sub_ch - qs[_idx_lo_s]) / np.maximum(
                                qs[_idx_lo_s + 1] - qs[_idx_lo_s], 1e-9
                            )
                            _sub_counts = np.maximum(
                                _sub_q_vecs[np.arange(_n_sub_ch), _idx_lo_s]
                                + _alpha_s * (
                                    _sub_q_vecs[np.arange(_n_sub_ch), _idx_lo_s + 1]
                                    - _sub_q_vecs[np.arange(_n_sub_ch), _idx_lo_s]
                                ),
                                0.0,
                            )
                            # Map active SUB positions to global MC index
                            _act_indices = np.where(active_mask)[0]
                            _sub_global_idx = _act_indices[_is_sub_ch]
                            _samples_ch[_sub_global_idx] = _sub_counts
                cdfs.append(RateXDurationCDF(_samples_ch, p_zero=p_zero))
                continue  # skip rest of per-row loop

            # Hurdle: method-conditional (v8.4) or global
            if _apply_mh and _p_pos_m is not None and method_codes is not None:
                active_mask = rng.random(_N_MC) < _p_pos_m[i][method_codes]
            else:
                active_mask = rng.random(_N_MC) < p_pos

            n_active = int(active_mask.sum())

            samples = np.zeros(_N_MC, dtype=float)
            if n_active > 0:
                # --- Sample duration ---
                grid_i = prebuilt_grids[i]
                if ceil_sec <= 300.0:
                    # R1-ceiling mode: Bernoulli decomposition.
                    # Method CDFs not used here (R1 props don't benefit from it).
                    # force_r1_end=True: every active draw is a finish draw,
                    # producing the conditional forecast F(count | ends in R1).
                    p_r1_end = (1.0 if force_r1_end
                                else 0.0 if force_full_round
                                else p_r1_end_arr[i])
                    is_r1_end_mask = rng.random(n_active) < p_r1_end
                    dur_sec = np.full(n_active, ceil_sec, dtype=float)
                    n_r1_end = int(is_r1_end_mask.sum())
                    if n_r1_end > 0 and grid_i is not None:
                        u_cond = rng.random(n_r1_end)
                        dur_sec[is_r1_end_mask] = duration_inverse_cdf(
                            None, u_cond, _prebuilt_grid=grid_i
                        )
                elif method_codes is not None and _method_grids is not None:
                    # Full-fight + method-conditional CDFs: each draw uses its
                    # method's CDF so KO draws are short, DEC draws are long.
                    dur_sec = np.zeros(n_active, dtype=float)
                    for _mc in range(3):
                        _m_mask = method_codes[active_mask] == _mc
                        _n_m = int(_m_mask.sum())
                        if _n_m == 0:
                            continue
                        _g_m = _method_grids[_mc][i]
                        if _g_m is not None:
                            _u_m = rng.random(_n_m)
                            dur_sec[_m_mask] = duration_inverse_cdf(
                                None, _u_m, _prebuilt_grid=_g_m
                            )
                        else:
                            _u_m = rng.random(_n_m)
                            dur_sec[_m_mask] = 1.0 + _u_m * (ceil_sec - 1.0)
                elif grid_i is not None:
                    # Full-fight, marginal duration CDF.
                    u_dur = rng.random(n_active)
                    dur_sec = duration_inverse_cdf(
                        None, u_dur, _prebuilt_grid=grid_i
                    )
                else:
                    # Fallback: uniform duration in [1, ceil_sec]
                    u_dur = rng.random(n_active)
                    dur_sec = 1.0 + u_dur * (ceil_sec - 1.0)

                dur_sec = np.asarray(dur_sec, dtype=float)
                active_min = np.minimum(dur_sec, ceil_sec) / 60.0
                active_min = np.maximum(active_min, 5.0 / 60.0)  # 5-second floor

                # --- Sample rate ---
                u_rate = rng.random(n_active)
                lqv = log_rate_preds[i]  # (n_q,) log-rate quantile values
                log_rate_samp = np.interp(u_rate, qs, lqv)

                # Apply per-draw method rate adjustment (v8.2, flat per-method)
                if _has_rate_adj and method_codes is not None:
                    log_rate_samp = log_rate_samp + _rate_adj_arr[method_codes[active_mask]]

                # Apply duration-binned method rate adjustment (v8.5, takedowns).
                # Replaces the flat scalar that was zeroed as a duration confound.
                # Each draw uses its (method, duration-bin) residual so short-KO
                # draws and long-SUB draws get the correct conditional correction.
                if _has_binned_rate_adj and method_codes is not None:
                    _active_methods = method_codes[active_mask]
                    _active_min_for_bin = active_min  # minutes per active draw
                    _dur_bins_draw = np.digitize(_active_min_for_bin, _dur_bin_edges[1:-1])
                    _n_bins = len(_dur_bin_edges) - 1
                    for _bmc, _bmn in enumerate(_method_names):
                        _bm_mask = _active_methods == _bmc
                        if not _bm_mask.any():
                            continue
                        _adj_list = _binned_rate_adj.get(_bmn, [0.0] * _n_bins)
                        _adj_arr = np.array(_adj_list, dtype=float)
                        _bins_clipped = np.clip(_dur_bins_draw[_bm_mask], 0, len(_adj_arr) - 1)
                        log_rate_samp[_bm_mask] += _adj_arr[_bins_clipped]

                rate_samp = np.exp(log_rate_samp)  # strikes / active_minute
                # getattr guard: older pickles predate rate_ceiling. A stale model
                # must degrade gracefully, never AttributeError the whole predict path.
                _rate_ceiling = getattr(self, "rate_ceiling", None)
                if _rate_ceiling is not None:
                    rate_samp = np.minimum(rate_samp, _rate_ceiling)

                counts = rate_samp * active_min

                # Drift lever: gentle shrink of finish-draw counts in the MARGINAL path.
                # The r1_end segment (R1 finishes) is over-predicted identically on val
                # AND test (meanPIT~0.42) — a stable finishing-burst bias, separable from
                # the 2025 survivor (past_r1) drift. Fires only in the real marginal
                # (not force_r1_end) for R1-ceiling models. Fit on val (self.finish_draw_scale).
                _fds = getattr(self, "finish_draw_scale", 1.0)
                if _fds != 1.0 and not force_r1_end and ceil_sec <= 300.0:
                    _fd_mask = dur_sec < (ceil_sec - 1.0)
                    if _fd_mask.any():
                        counts[_fd_mask] = counts[_fd_mask] * _fds

                # Finish-regime count correction — priority: t-head (v8.7) > legacy-head (v8.6) > OLS burst (v8.5)
                # v8.8 pivotal experiment: enabling the head in the marginal path
                # (removing the force_r1_end guard) caused overall KS to REGRESS
                # 0.060→0.112 — the head is miscalibrated (cond-null KS=0.212) and
                # applying a miscalibrated F_finish to p_r1_end fraction of marginal
                # draws makes the marginal worse.  Guard reinstated until F_finish is
                # fixed (Tier-2 diagnostics).  The guard is now conceptually clean:
                # it is not protecting an invalid metric (v8.8 evaluation now uses
                # valid symmetric conditional nulls for both r1_end and past_r1) but
                # simply preventing a miscalibrated component from contaminating the
                # marginal forecast.
                # v8.9 (Fix B): method+t-conditional head, force_r1_end-guarded
                # (diagnostic only).  EMPIRICAL FINDING: enabling the head in the
                # marginal REGRESSED overall KS 0.030→0.087 even after Fix A.  Once
                # the rate is de-inflated (Fix A), plain rate×t already calibrates the
                # marginal (overall=0.030 PASS, r1_end pred mean 13.5 vs emp 12.8).
                # The explicit finish head — bimodal, heavy-tailed, hard to calibrate
                # — over-predicts r1_end (14.9) and hurts the gate.  So the robust
                # rate×t approximation wins for the marginal; the method head is kept
                # only for the r1_end conditional-null diagnostic (fires under
                # force_r1_end).  Method-conditioning moved that diagnostic just
                # 0.212→0.207 — the residual is within-method/within-t dispersion
                # (a genuine Tier-3 trigger if the conditional null ever needs to pass).
                if _finish_head_tm_preds is not None and force_r1_end and ceil_sec <= 300.0:
                    is_finish_draw = dur_sec < (ceil_sec - 1.0)
                    if is_finish_draw.any():
                        _t_finish = dur_sec[is_finish_draw]  # seconds
                        _n_fin = int(is_finish_draw.sum())
                        # Method code per finish draw: prefer per-draw method_codes
                        # (sampled from method_proba); else fall back to the training
                        # method prior so the head still fires without method_proba.
                        if method_codes is not None:
                            _fin_m = method_codes[active_mask][is_finish_draw]
                        else:
                            _mp = (mprob_i if mprob_i is not None
                                   else getattr(self, "_finish_head_method_prior", None))
                            if _mp is None:
                                _mp = np.array([0.5, 0.2, 0.3])
                            _mp = np.clip(np.asarray(_mp, dtype=float), 1e-9, None)
                            _mp = _mp / _mp.sum()
                            _fin_m = np.searchsorted(
                                np.cumsum(_mp), rng.random(_n_fin), side="right"
                            ).clip(0, 2)
                        # Interpolate quantile vectors at each draw's (t, method)
                        _fh_q_vecs = np.zeros((_n_fin, n_q))
                        for _mj in range(3):
                            _sel = _fin_m == _mj
                            if not _sel.any():
                                continue
                            _tbl = _finish_head_tm_preds[i, _mj]  # (n_t_grid, n_q)
                            _ts = _t_finish[_sel]
                            for _qi in range(n_q):
                                _fh_q_vecs[_sel, _qi] = np.interp(
                                    _ts, _T_GRID_SEC, _tbl[:, _qi]
                                )
                        _fh_q_vecs = np.maximum.accumulate(
                            np.maximum(_fh_q_vecs, 0.0), axis=1
                        )
                        u_finish = rng.random(_n_fin)
                        _fi_idx_lo = np.clip(
                            np.searchsorted(qs, u_finish, side="right") - 1, 0, n_q - 2
                        )
                        _fi_alpha = (u_finish - qs[_fi_idx_lo]) / np.maximum(
                            qs[_fi_idx_lo + 1] - qs[_fi_idx_lo], 1e-9
                        )
                        counts[is_finish_draw] = np.maximum(
                            _fh_q_vecs[np.arange(_n_fin), _fi_idx_lo]
                            + _fi_alpha * (
                                _fh_q_vecs[np.arange(_n_fin), _fi_idx_lo + 1]
                                - _fh_q_vecs[np.arange(_n_fin), _fi_idx_lo]
                            ),
                            0.0,
                        )
                elif _finish_head_t_preds is not None and force_r1_end and ceil_sec <= 300.0:
                    is_finish_draw = dur_sec < (ceil_sec - 1.0)
                    if is_finish_draw.any():
                        _t_finish = dur_sec[is_finish_draw]  # seconds
                        _fh_t_tbl_i = _finish_head_t_preds[i]  # (n_t_grid, n_q)
                        _n_fin = int(is_finish_draw.sum())
                        # Interpolate quantile vectors at each draw's finish time
                        _fh_q_vecs = np.zeros((_n_fin, n_q))
                        for _qi in range(n_q):
                            _fh_q_vecs[:, _qi] = np.interp(
                                _t_finish, _T_GRID_SEC, _fh_t_tbl_i[:, _qi]
                            )
                        _fh_q_vecs = np.maximum.accumulate(
                            np.maximum(_fh_q_vecs, 0.0), axis=1
                        )
                        # Vectorized sampling: bilinear in (t, u)
                        u_finish = rng.random(_n_fin)
                        _fi_idx_lo = np.clip(
                            np.searchsorted(qs, u_finish, side="right") - 1, 0, n_q - 2
                        )
                        _fi_alpha = (u_finish - qs[_fi_idx_lo]) / np.maximum(
                            qs[_fi_idx_lo + 1] - qs[_fi_idx_lo], 1e-9
                        )
                        counts[is_finish_draw] = np.maximum(
                            _fh_q_vecs[np.arange(_n_fin), _fi_idx_lo]
                            + _fi_alpha * (
                                _fh_q_vecs[np.arange(_n_fin), _fi_idx_lo + 1]
                                - _fh_q_vecs[np.arange(_n_fin), _fi_idx_lo]
                            ),
                            0.0,
                        )
                elif use_finish_head and force_r1_end and _finish_head_preds is not None and ceil_sec <= 300.0:
                    # Legacy t-independent head (v8.6): backward compat for pickled models
                    # without _finish_head_has_t.  Guard kept for same reason as above.
                    is_finish_draw = dur_sec < (ceil_sec - 1.0)
                    if is_finish_draw.any():
                        _fh_qv = _finish_head_preds[i]  # (n_q,) monotone quantile values
                        u_finish = rng.random(int(is_finish_draw.sum()))
                        counts[is_finish_draw] = np.maximum(
                            np.interp(u_finish, qs, _fh_qv), 0.0
                        )
                elif apply_burst and getattr(self, "r1_finish_fitted", False) and ceil_sec <= 300.0:
                    # v8.5 OLS-rescale burst (kept for backward compat / apply_burst=True path)
                    is_finish_draw = dur_sec < (ceil_sec - 1.0)
                    if is_finish_draw.any():
                        _t_fin = dur_sec[is_finish_draw]
                        _a = float(getattr(self, "r1_finish_intercept", 0.0))
                        _b = float(getattr(self, "r1_finish_slope", 0.0))
                        _burst_mean = np.maximum(_a + _b * _t_fin, 0.1)
                        _rate_samp_mean = float(np.mean(rate_samp))
                        _rate_rel = rate_samp[is_finish_draw] / max(_rate_samp_mean, 1e-6)
                        counts[is_finish_draw] = np.maximum(_rate_rel * _burst_mean, 0.0)
                elif apply_burst:
                    # Legacy burst path (backward compat for old pickled models)
                    _r1_intercept = getattr(self, "r1_burst_intercept", 0.0)
                    _r1_slope = getattr(self, "r1_burst_slope", 0.0)
                    if _r1_intercept > 0.0 and ceil_sec <= 300.0:
                        is_finish_draw = dur_sec < (ceil_sec - 1.0)
                        if is_finish_draw.any():
                            _t_finish = dur_sec[is_finish_draw]
                            _correction = _r1_intercept + _r1_slope * _t_finish
                            counts[is_finish_draw] += np.maximum(_correction, 0.0)

                samples[active_mask] = np.maximum(counts, 0.0)

            cdfs.append(RateXDurationCDF(samples, p_zero=p_zero))

        return cdfs

    def fit_method_adjustments(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        active_minutes_train: np.ndarray,
        method_train: pd.Series,
        sample_weight: np.ndarray | None = None,
        event_dates_train: pd.Series | None = None,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        active_minutes_val: np.ndarray | None = None,
        method_val: pd.Series | None = None,
        temporal_oof: bool = False,
    ) -> "RateHurdleCountModel":
        """Fit per-method log-rate adjustments from training residuals.

        Computes mean residual log(rate_obs) - log(rate_pred) per method class.
        Stores result in self.method_log_rate_adj to replace hard-coded simulator
        multipliers at inference time.

        Parameters
        ----------
        sample_weight : np.ndarray | None
            Per-row sample weights forwarded to the finish-count head (r1) and
            SUB count head (takedowns).  Pass the same recency/censoring weights
            used in cm.fit() so the heads track the same distribution as the
            base rate models.
        temporal_oof : bool
            Prod tier. X_val/y_val/etc are in-sample (val ⊂ train) in prod mode
            — the finish-head dispersion-factor KS sweep below would be scored
            on data used to fit the finish-count quantile models. When True,
            replace X_val/y_val/active_minutes_val/method_val with a temporal
            holdout carved from the tail of train (event_dates_train required).
            Not a fully disjoint OOF split (the quantile models above still fit
            on all of X_train, so there is some overlap with this holdout's
            recent window) — a strict improvement over 100%-in-sample val, not
            a complete leak fix.
        """
        _METHOD_NORM = {
            "KO/TKO": "KO/TKO", "SUB": "SUB",
            "U-DEC": "DEC", "S-DEC": "DEC", "M-DEC": "DEC", "DEC": "DEC",
        }
        if temporal_oof and event_dates_train is not None:
            _dates_tr = pd.to_datetime(event_dates_train).values
            _cutoff = (_dates_tr.max() - pd.DateOffset(months=6)).to_datetime64()
            _mask = _dates_tr >= _cutoff
            if _mask.sum() < 30:
                _mask = np.zeros(len(_dates_tr), dtype=bool)
                _mask[np.argsort(_dates_tr)[-30:]] = True
            X_val = X_train.iloc[_mask].reset_index(drop=True)
            y_val = pd.Series(y_train).reset_index(drop=True).iloc[_mask].reset_index(drop=True)
            active_minutes_val = np.asarray(active_minutes_train, dtype=float)[_mask]
            method_val = pd.Series(method_train).reset_index(drop=True).iloc[_mask].reset_index(drop=True)
        eps_min = 5.0 / 60.0
        y_tr = y_train.fillna(0).clip(lower=0).values.astype(float)
        Xf = X_train.reindex(columns=self.feature_cols, fill_value=0).fillna(0)
        act_tr = np.maximum(np.asarray(active_minutes_train, dtype=float), eps_min)

        pos_mask = y_tr > 0
        if pos_mask.sum() < 30:
            return self

        y_pos = y_tr[pos_mask]
        act_pos = act_tr[pos_mask]
        Xa_pos = Xf.values[pos_mask]
        method_raw = pd.Series(
            method_train.values if hasattr(method_train, "values") else list(method_train)
        )
        method_norm = method_raw.map(_METHOD_NORM).fillna("DEC").values[pos_mask]

        obs_log_rate = np.log(np.maximum(y_pos / act_pos, 1e-6))

        # Predict using the median quantile model (closest to q=0.50)
        qs = np.array(QUANTILE_GRID)
        median_idx = int(np.argmin(np.abs(qs - 0.5)))
        pred_log_rate = self.quantile_models[median_idx].predict(Xa_pos)

        residuals = obs_log_rate - pred_log_rate

        adj: dict[str, float] = {}
        for m in ["KO/TKO", "SUB", "DEC"]:
            mask = method_norm == m
            adj[m] = float(np.mean(residuals[mask])) if mask.sum() >= 10 else 0.0

        self.method_log_rate_adj = adj
        print(
            f"  {self.target} method log-rate adj: "
            f"KO={adj['KO/TKO']:+.3f}, SUB={adj['SUB']:+.3f}, DEC={adj['DEC']:+.3f}"
        )

        # ── Per-method hurdle log-odds adjustments (v8.2) ────────────────────
        # For ALL training rows: compare observed P(count>0) per method to
        # the model's predicted hurdle. Store log-odds residual per method.
        # This lets predict_cdf create bimodal distributions for finish fights
        # (near-zero TDs for KO draws, near-certain TDs for sub draws).
        y_all = y_train.fillna(0).clip(lower=0).values.astype(float)
        Xf_all = X_train.reindex(columns=self.feature_cols, fill_value=0).fillna(0)
        method_raw_all = pd.Series(
            method_train.values if hasattr(method_train, "values") else list(method_train)
        )
        method_norm_all = method_raw_all.map(_METHOD_NORM).fillna("DEC").values
        obs_hurdle_all = (y_all > 0).astype(float)
        pred_hurdle_all = self.pos_clf.predict_proba(Xf_all.values)[:, 1]

        logodds_adj: dict[str, float] = {}
        for m in ["KO/TKO", "SUB", "DEC"]:
            mask = method_norm_all == m
            if mask.sum() >= 10:
                obs_r = float(np.clip(obs_hurdle_all[mask].mean(), 1e-6, 1 - 1e-6))
                pred_r = float(np.clip(pred_hurdle_all[mask].mean(), 1e-6, 1 - 1e-6))
                logodds_adj[m] = float(
                    np.log(obs_r / (1 - obs_r)) - np.log(pred_r / (1 - pred_r))
                )
            else:
                logodds_adj[m] = 0.0
        self.method_logodds_hurdle_adj = logodds_adj
        print(
            f"  {self.target} method logodds-hurdle adj: "
            f"KO={logodds_adj['KO/TKO']:+.3f}, "
            f"SUB={logodds_adj['SUB']:+.3f}, "
            f"DEC={logodds_adj['DEC']:+.3f}"
        )

        # ── R1 finish-regime component (v8.5, ceiling-bounded models only) ─────
        # The DGP for R1-finish fights is a burst: y ≈ a + b·t with fixed-width
        # spread (0–6 strikes), not rate×t accumulation. OLS is fit on ALL R1-finish
        # training rows (including zeros) so the location estimate is unbiased.
        # The empirical residual pool ε = y − (a + b·t) reproduces the true
        # dispersion when resampled at inference. Supersedes the v8.3 affine burst
        # correction which patched location but left dispersion wrong.
        if self.active_minutes_ceiling is not None and self.active_minutes_ceiling <= 5.0:
            _eps_r1 = 5.0 / 60.0
            act_all_r1 = np.maximum(np.asarray(active_minutes_train, dtype=float), _eps_r1)
            is_r1_finish = act_all_r1 < (self.active_minutes_ceiling - 0.05)
            n_r1 = int(is_r1_finish.sum())
            if n_r1 >= 20:
                t_r1 = act_all_r1[is_r1_finish] * 60.0  # seconds
                y_r1 = y_all[is_r1_finish]
                T_mat = np.column_stack([np.ones(n_r1), t_r1])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(T_mat, y_r1, rcond=None)
                    r1_a = float(coeffs[0])
                    r1_b = float(coeffs[1])
                except Exception:
                    r1_a = float(np.mean(y_r1))
                    r1_b = 0.0
                self.r1_finish_intercept = r1_a
                self.r1_finish_slope = r1_b
                self.r1_finish_resid_pool = None  # retired; Poisson used at inference
                self.r1_finish_fitted = True
                self.r1_burst_intercept = 0.0  # retired
                self.r1_burst_slope = 0.0
                # Diagnostic: report residual spread for reference
                r1_resid = y_r1 - (r1_a + r1_b * t_r1)
                print(
                    f"  {self.target} R1 finish component (Poisson): "
                    f"a={r1_a:.2f}  b={r1_b:.4f}  n={n_r1}  "
                    f"resid_range=[{r1_resid.min():.1f}, {r1_resid.max():.1f}]"
                )
            else:
                self.r1_finish_intercept = 0.0
                self.r1_finish_slope = 0.0
                self.r1_finish_resid_pool = None
                self.r1_finish_fitted = False
                self.r1_burst_intercept = 0.0
                self.r1_burst_slope = 0.0

            # ── Finish-count head (v8.7 t-conditional + v8.9 method-conditional) ──
            # 25-quantile LGBM on raw R1 count conditioned on finish-time t (seconds)
            # AND method one-hot (KO/TKO, SUB, DEC).  Trained on POSITIVE r1_end rows.
            # Feature order: [features, t_finish_sec, ko, sub, dec] — inference MUST
            # build the table in the same order.
            # v8.9 (Fix B): method conditioning captures the bimodality the t-only head
            # missed — SUB finishes are low-strike (mean ~8.8, 43% land 1-5) while
            # KO/TKO finishes are higher and wider (mean ~14.8, tail to 79).  Without
            # it the head produced a too-narrow unimodal distribution (cond-null PIT
            # U-shaped: both tails 2-3x over-weight).
            # Recency-weighted (same schedule as base rate models) for the +15% drift.
            is_r1_finish_pos = is_r1_finish & (y_all > 0)
            n_r1_all = int(is_r1_finish_pos.sum())
            if n_r1_all >= 20:
                # Append finish-time in seconds, then a 3-way method one-hot.
                _t_finish_sec = act_all_r1[is_r1_finish_pos] * 60.0  # active_min * 60
                _m_r1 = method_norm_all[is_r1_finish_pos]
                _onehot_r1 = np.zeros((n_r1_all, 3), dtype=float)
                for _j, _mn in enumerate(["KO/TKO", "SUB", "DEC"]):
                    _onehot_r1[:, _j] = (_m_r1 == _mn).astype(float)
                Xa_r1_fh = np.column_stack(
                    [Xf_all.values[is_r1_finish_pos], _t_finish_sec[:, None], _onehot_r1]
                )
                y_r1_fh = y_all[is_r1_finish_pos]
                # Forward recency sample weight to the head fit
                _sw_r1 = (
                    sample_weight[is_r1_finish_pos]
                    if sample_weight is not None
                    else None
                )
                _fh_cfg = _cfg()["lgbm_quantile"]
                # Cap trees at 300: r1_end has ~1400 train rows so 800 is heavy overfit
                _fh_n_estimators = min(_fh_cfg["n_estimators"], 300)
                _min_child_fh = max(5, min(_fh_cfg["min_child_samples"], n_r1_all // 5))
                self.finish_count_quantile_models = []
                for _fhq in QUANTILE_GRID:
                    _fhm = lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=_fhq,
                        n_estimators=_fh_n_estimators,
                        learning_rate=_fh_cfg["learning_rate"],
                        num_leaves=_fh_cfg["num_leaves"],
                        min_child_samples=_min_child_fh,
                        verbosity=-1,
                        random_state=SEED,
                        deterministic=True,
                        force_row_wise=True,
                        num_threads=1,
                        feature_fraction_seed=SEED,
                        bagging_seed=SEED,
                        data_random_seed=SEED,
                        extra_seed=SEED,
                        objective_seed=SEED,
                    )
                    _fhm.fit(Xa_r1_fh, y_r1_fh, sample_weight=_sw_r1)
                    self.finish_count_quantile_models.append(_fhm)
                self.finish_head_fitted = True
                self._finish_head_has_t = True  # signal inference to use t-conditional path
                self._finish_head_has_method = True  # v8.9: method-conditional table
                # Training method mix among r1 finishes — fallback when method_proba
                # is absent at inference so the head can still fire.
                self._finish_head_method_prior = _onehot_r1.mean(axis=0)
                _m_means = {
                    _mn: float(y_r1_fh[_onehot_r1[:, _j] == 1.0].mean())
                    for _j, _mn in enumerate(["KO/TKO", "SUB", "DEC"])
                    if (_onehot_r1[:, _j] == 1.0).sum() > 0
                }
                print(
                    f"  {self.target} finish-count head (v8.9, t+method): "
                    f"n={n_r1_all}  mean={y_r1_fh.mean():.2f}  std={y_r1_fh.std():.2f}  "
                    f"by-method means={ {k: round(v, 1) for k, v in _m_means.items()} }"
                )
                # v8.11: tune dispersion factor on val r1-finish rows.
                # Sweep f in [1.0, 1.2, ..., 2.5]; pick the smallest f that is within
                # tolerance of the minimum val KS (prefer conservatism over extremity).
                # Applied in predict_cdf's finish-head table build to widen quantiles.
                _fhd_sweep = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]
                _fhd_default = 2.0
                _val_ok = (X_val is not None and y_val is not None
                           and active_minutes_val is not None and method_val is not None)
                if _val_ok:
                    _eps_r1v = 5.0 / 60.0
                    _act_val_r1 = np.maximum(np.asarray(active_minutes_val, dtype=float), _eps_r1v)
                    _is_r1f_val = _act_val_r1 < (self.active_minutes_ceiling - 0.05)
                    _n_r1_val = int(_is_r1f_val.sum())
                    if _n_r1_val >= 10:
                        _y_val_r1 = (
                            y_val.fillna(0).clip(lower=0).values.astype(float)[_is_r1f_val]
                        )
                        _t_val_r1 = _act_val_r1[_is_r1f_val] * 60.0
                        _mv_raw = pd.Series(
                            method_val.values if hasattr(method_val, "values") else list(method_val)
                        )
                        _meth_val_r1 = _mv_raw.map(_METHOD_NORM).fillna("DEC").values[_is_r1f_val]
                        _Xf_val = X_val.reindex(columns=self.feature_cols, fill_value=0).fillna(0)
                        _Xr_val = _Xf_val.values[_is_r1f_val]
                        _oh_val = np.zeros((_n_r1_val, 3), dtype=float)
                        for _j2, _mn2 in enumerate(["KO/TKO", "SUB", "DEC"]):
                            _oh_val[:, _j2] = (_meth_val_r1 == _mn2).astype(float)
                        _Xh_val = np.column_stack([_Xr_val, _t_val_r1[:, None], _oh_val])
                        _qp_val = np.column_stack(
                            [m.predict(_Xh_val) for m in self.finish_count_quantile_models]
                        )
                        _qp_val = np.maximum.accumulate(np.maximum(_qp_val, 0.0), axis=1)
                        _ppos_val = self.pos_clf.predict_proba(_Xr_val)[:, 1]
                        _qs_fhd = np.array(QUANTILE_GRID)
                        _med_idx_v = int(np.argmin(np.abs(_qs_fhd - 0.5)))

                        from scipy import stats as _sc_stats_fhd  # noqa: PLC0415

                        def _pit_ks_fhd(qp, f, seed=99):
                            _med = qp[:, _med_idx_v:_med_idx_v + 1]
                            _qpw = np.maximum.accumulate(
                                np.maximum(_med + f * (qp - _med), 0.0), axis=1
                            )
                            _rng2 = np.random.default_rng(seed)
                            _pit2 = np.empty(len(_y_val_r1))
                            for _i2, _v2 in enumerate(_y_val_r1):
                                _pz2 = 1.0 - _ppos_val[_i2]
                                _pit2[_i2] = (
                                    _rng2.uniform(0, _pz2) if _v2 <= 0
                                    else _pz2 + (1 - _pz2) * np.interp(_v2, _qpw[_i2], _qs_fhd)
                                )
                            return _sc_stats_fhd.kstest(np.clip(_pit2, 0, 1), "uniform")[0]

                        _fhd_ks = [_pit_ks_fhd(_qp_val, f) for f in _fhd_sweep]
                        _best_ks = min(_fhd_ks)
                        _tol_fhd = 0.015
                        _plateau = [
                            f for f, k in zip(_fhd_sweep, _fhd_ks)
                            if k <= _best_ks + _tol_fhd
                        ]
                        _fhd_default = min(_plateau)  # smallest factor in the plateau
                        print(
                            f"  {self.target} finish-head disp sweep (val n={_n_r1_val}): "
                            f"best_KS={_best_ks:.3f}  plateau_factor={_fhd_default:.1f}  "
                            f"KS@2.0={_fhd_ks[_fhd_sweep.index(2.0)]:.3f}"
                        )
                    else:
                        print(
                            f"  {self.target} finish-head disp factor: "
                            f"val r1-finish n={_n_r1_val} < 10, using fallback f={_fhd_default:.1f}"
                        )
                else:
                    print(
                        f"  {self.target} finish-head disp factor: "
                        f"no val data provided, using fallback f={_fhd_default:.1f}"
                    )
                self.finish_head_disp_factor = _fhd_default
            else:
                self.finish_count_quantile_models = []
                self.finish_head_fitted = False
                self._finish_head_has_t = False
                self._finish_head_has_method = False

        # ── Duration-binned method rate adjustments (v8.5, full-fight models) ──
        # For takedowns the flat per-method rate adj was a duration confound (zeroed
        # at eval). Stratifying by duration quartile disentangles the confound: the
        # per-bin residuals reflect genuine method effects at matched exposure.
        # Applied per-draw at inference when use_binned_rate_adj=True.
        if self.active_minutes_ceiling is None:
            _eps_bin = 5.0 / 60.0
            act_for_bins = np.maximum(np.asarray(active_minutes_train, dtype=float), _eps_bin)
            try:
                _, bin_edges = pd.qcut(act_for_bins, q=4, retbins=True, duplicates="drop")
            except Exception:
                bin_edges = None
            if bin_edges is not None and len(bin_edges) >= 3:
                bin_edges_stored = bin_edges.copy().astype(float)
                bin_edges_stored[0] = 0.0    # open left edge
                bin_edges_stored[-1] = np.inf  # open right edge
                self._dur_bin_edges = bin_edges_stored
                n_bins = len(bin_edges_stored) - 1
                dur_bins_all = np.digitize(act_for_bins, bin_edges_stored[1:-1])  # 0-indexed
                qs_arr2 = np.array(QUANTILE_GRID)
                median_idx2 = int(np.argmin(np.abs(qs_arr2 - 0.5)))
                binned_rate_adj: dict[str, list[float]] = {}
                for _bm in ["KO/TKO", "SUB", "DEC"]:
                    _bm_mask_all = method_norm_all == _bm
                    adjs_for_m: list[float] = []
                    for _b in range(n_bins):
                        _mb_pos = _bm_mask_all & (dur_bins_all == _b) & (y_all > 0)
                        if int(_mb_pos.sum()) >= 5:
                            _obs_lr = np.log(np.maximum(
                                y_all[_mb_pos] / act_for_bins[_mb_pos], 1e-6
                            ))
                            _pred_lr = self.quantile_models[median_idx2].predict(
                                Xf_all.values[_mb_pos]
                            )
                            adjs_for_m.append(float(np.mean(_obs_lr - _pred_lr)))
                        else:
                            adjs_for_m.append(adj.get(_bm, 0.0))
                    binned_rate_adj[_bm] = adjs_for_m
                self.method_log_rate_adj_binned = binned_rate_adj
                print(f"  {self.target} duration-binned rate adj ({n_bins} bins):")
                for _bm2, _adjs in binned_rate_adj.items():
                    print(f"    {_bm2}: {[f'{_a:+.3f}' for _a in _adjs]}")
            else:
                self.method_log_rate_adj_binned = None
                self._dur_bin_edges = None

            # v8.11: compute recency weight for cond-hurdle and SUB head so they track
            # the modern higher-TD SUB era (train E[TD|>0,SUB] 1.89→2.04 by era, test 2.19).
            # Anchored at train_end from the active split config, so prod split uses 2024-12-31.
            # Independent of the censoring weight used for the main rate model.
            _recency_sw_all = None
            if event_dates_train is not None:
                import os, yaml as _yaml  # noqa: E401
                _sc_fname = os.environ.get("UFC_SPLIT_CONFIG", "split.yaml")
                from ufc.io import paths as _paths
                _rc_anchor = pd.to_datetime(
                    _yaml.safe_load((_paths.root() / "configs" / _sc_fname).read_text())["train_end"]
                )
                _rc_days = (
                    _rc_anchor - pd.to_datetime(event_dates_train)
                ).dt.days.clip(lower=0).astype(float)
                _recency_sw_all = np.power(0.5, _rc_days / 730.0).clip(lower=0.05).values

            # ── Conditional hurdle (v8.6, full-fight models) ──────────────────────
            # Train P(>0 | features, method_onehot, log_active_min). Because it is a
            # probability (not a rate), there is no denominator confound. Corrects the
            # KO-finish zero-mass deficit: P(>0 TD | KO, Q1)≈0.133 vs. the global
            # hurdle that is blind to realized method and duration.
            _eps_ch = 5.0 / 60.0
            act_for_ch = np.maximum(
                np.asarray(active_minutes_train, dtype=float), _eps_ch
            )
            log_act_all_ch = np.log(act_for_ch)
            # Method one-hot: [KO/TKO, SUB, DEC]
            _ch_onehot = np.zeros((len(y_all), 3), dtype=float)
            for _chmi, _chmn in enumerate(["KO/TKO", "SUB", "DEC"]):
                _ch_onehot[:, _chmi] = (method_norm_all == _chmn).astype(float)
            Xa_aug_ch = np.column_stack(
                [Xf_all.values, _ch_onehot, log_act_all_ch[:, None]]
            )
            y_pos_ch = (y_all > 0).astype(int)
            _ch_lgbm_cfg = _cfg()["lgbm_quantile"]
            # Cap trees at 150: 800 trees on ~2000 rows overfit badly, producing
            # miscalibrated predictions at extreme durations (long DEC, short KO).
            _ch_n_est = min(_ch_lgbm_cfg["n_estimators"], 150)
            from sklearn.calibration import CalibratedClassifierCV as _CalCV  # noqa: PLC0415
            _lgbm_cond_base = lgb.LGBMClassifier(
                n_estimators=_ch_n_est,
                num_leaves=_ch_lgbm_cfg["num_leaves"],
                learning_rate=_ch_lgbm_cfg["learning_rate"],
                min_child_samples=_ch_lgbm_cfg["min_child_samples"],
                verbosity=-1,
                random_state=SEED,
                deterministic=True,
                force_row_wise=True,
                num_threads=1,
                feature_fraction_seed=SEED,
                bagging_seed=SEED,
                data_random_seed=SEED,
                extra_seed=SEED,
                objective_seed=SEED,
            )
            # cv=5 Platt calibration: out-of-fold predictions prevent over-fitting
            # the calibration layer to the same data as the main model.
            self.pos_clf_cond = _CalCV(
                _lgbm_cond_base, method="sigmoid", cv=5
            )
            # No sample_weight on the cond-hurdle: recency weighting shifted the
            # forced-DEC conditional CDF and caused a decision-segment regression.
            # The unweighted fit is well-calibrated for the conditional null.
            self.pos_clf_cond.fit(Xa_aug_ch, y_pos_ch)
            self._cond_hurdle_aug_cols = (
                list(self.feature_cols) + ["_ch_ko", "_ch_sub", "_ch_dec", "_ch_log_act_min"]
            )
            self._has_cond_hurdle = True
            print(
                f"  {self.target} conditional hurdle (v8.6, LGBM+Platt cv=5): "
                f"n={len(y_all)}  pos_rate={y_pos_ch.mean():.3f}  n_est={_ch_n_est}"
            )

            # ── SUB count head (v8.7, full-fight models only) ──────────────────
            # 25-quantile LGBM on raw td_landed for SUB-finish rows, conditioned on
            # log_active_min.  Captures the front-loaded TD distribution where
            # td/min falls with fight duration (the TD sets up the submission).
            # rate×duration cannot represent this; only the positive-count shape
            # differs — P(>0 TD|SUB)≈global so the hurdle has no leverage.
            # Trained on all positive SUB rows (not limited to a finish regime).
            is_sub_pos = (method_norm_all == "SUB") & (y_all > 0)
            n_sub_pos = int(is_sub_pos.sum())
            if n_sub_pos >= 50:
                _log_act_sub = np.log(np.maximum(act_for_ch[is_sub_pos], _eps_ch))
                Xa_sub = np.column_stack(
                    [Xf_all.values[is_sub_pos], _log_act_sub[:, None]]
                )
                y_sub = y_all[is_sub_pos]
                # v8.11: prefer recency weight over censoring weight for the SUB head
                # so it tracks the modern higher-TD era.
                _sw_sub = (
                    _recency_sw_all[is_sub_pos]
                    if _recency_sw_all is not None
                    else (sample_weight[is_sub_pos] if sample_weight is not None else None)
                )
                _sub_cfg = _cfg()["lgbm_quantile"]
                _sub_n_est = min(_sub_cfg["n_estimators"], 300)
                _min_child_sub = max(5, min(_sub_cfg["min_child_samples"], n_sub_pos // 5))
                self.sub_count_quantile_models = []
                for _subq in QUANTILE_GRID:
                    _subm = lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=_subq,
                        n_estimators=_sub_n_est,
                        learning_rate=_sub_cfg["learning_rate"],
                        num_leaves=_sub_cfg["num_leaves"],
                        min_child_samples=_min_child_sub,
                        verbosity=-1,
                        random_state=SEED,
                        deterministic=True,
                        force_row_wise=True,
                        num_threads=1,
                        feature_fraction_seed=SEED,
                        bagging_seed=SEED,
                        data_random_seed=SEED,
                        extra_seed=SEED,
                        objective_seed=SEED,
                    )
                    _subm.fit(Xa_sub, y_sub, sample_weight=_sw_sub)
                    self.sub_count_quantile_models.append(_subm)
                self.sub_head_fitted = True
                print(
                    f"  {self.target} SUB count head (v8.7): "
                    f"n={n_sub_pos}  mean={y_sub.mean():.2f}  std={y_sub.std():.2f}"
                )
            else:
                self.sub_count_quantile_models = []
                self.sub_head_fitted = False

        return self

    def save(self, path: Path, gitsha: str = "latest") -> Path:
        out = path / f"props_{self.target}_{gitsha}.joblib"
        joblib.dump(self, out, compress=3)
        return out

    @staticmethod
    def load(path: Path) -> "RateHurdleCountModel":
        return joblib.load(path)


class ControlShareModel:
    """Two-stage hurdle on ctrl_time share: phi = ctrl_sec / total_fight_sec.

    Stage 1: P(phi > 0) via LGBMClassifier.
    Stage 2: quantile regression on logit(clip(phi, eps, 1-eps)) for positive rows.

    At predict_cdf time: per MC draw, sample duration d from DurationCDF,
    sample logit_phi from Stage-2 quantiles, ctrl_sec = expit(logit_phi) * d.
    Guarantees ctrl_sec in [0, fight_duration] by construction — no hard clip needed.
    """

    _EPS = 1e-4

    def __init__(self, target: str = "ctrl_time"):
        self.target = target
        self.pos_clf: lgb.LGBMClassifier | None = None
        self.quantile_models: list[lgb.LGBMRegressor] = []
        self.feature_cols: list[str] = []
        # Per-method log-odds adjustment to logit(phi), fit on train residuals.
        # Applied at inference identically to how RateHurdleCountModel.method_log_rate_adj
        # is applied to log-rate draws. None = no adjustment.
        self.method_logit_share_adj: dict[str, float] | None = None
        # Scalar logit shift applied to pos_clf probability at inference so that
        # mean predicted P(ctrl>0) matches the empirical rate on the val fold.
        # Positive = reduces P(ctrl>0) = raises P(ctrl==0).
        self.hurdle_logit_adj: float = 0.0
        # Dispersion factor applied to logit(phi) quantile draws around the per-row
        # median. >1 widens the positive-share distribution. Corrects IQR under-dispersion
        # (low-ctrl PIT=0.31, high-ctrl PIT=0.94). 1.0 = no-op (backward compat).
        self.share_disp: float = 1.0

    def fit(self, X_train: pd.DataFrame, y_ctrl_train: pd.Series,
            total_sec_train: np.ndarray,
            X_val: pd.DataFrame, y_ctrl_val: pd.Series,
            total_sec_val: np.ndarray,
            rate_prop_cols: list[str],
            sample_weight: np.ndarray | None = None,
            temporal_oof: bool = False,
            train_dates: "pd.Series | None" = None) -> "ControlShareModel":
        cfg = _cfg()
        lgbm_cfg = cfg["lgbm_quantile"]
        self.feature_cols = rate_prop_cols

        y_tr = y_ctrl_train.fillna(0).clip(0).values.astype(float)
        y_vl = y_ctrl_val.fillna(0).clip(0).values.astype(float)
        tot_tr = np.maximum(total_sec_train, 1.0)
        tot_vl = np.maximum(total_sec_val, 1.0)

        X_tr = X_train[rate_prop_cols].fillna(0).values
        X_vl = X_val[rate_prop_cols].fillna(0).values

        phi_tr = np.clip(y_tr / tot_tr, 0.0, 1.0)
        phi_vl = np.clip(y_vl / tot_vl, 0.0, 1.0)

        print(f"  Fitting hurdle binary classifier for {self.target} (ctrl-share)...")
        y_pos_tr = (y_tr > 0).astype(int)
        y_pos_vl = (y_vl > 0).astype(int)
        self.pos_clf = lgb.LGBMClassifier(
            n_estimators=lgbm_cfg["n_estimators"],
            num_leaves=lgbm_cfg["num_leaves"],
            learning_rate=lgbm_cfg["learning_rate"],
            min_child_samples=lgbm_cfg["min_child_samples"],
            verbosity=-1, random_state=SEED,
            deterministic=True, force_row_wise=True, num_threads=1,
            feature_fraction_seed=SEED, bagging_seed=SEED,
            data_random_seed=SEED, extra_seed=SEED, objective_seed=SEED,
        )
        sw_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        if temporal_oof and train_dates is not None:
            _h_best_n = _oof_best_n(self.pos_clf, X_tr, y_pos_tr, train_dates, sample_weight)
            if _h_best_n is not None:
                print(f"    [temporal_oof] {self.target} hurdle best_n={_h_best_n}")
                self.pos_clf.set_params(n_estimators=_h_best_n)
                self.pos_clf.fit(X_tr, y_pos_tr, **sw_kw)
            else:
                self.pos_clf.fit(X_tr, y_pos_tr, eval_set=[(X_vl, y_pos_vl)],
                                 callbacks=[lgb.early_stopping(50, verbose=False),
                                            lgb.log_evaluation(period=0)], **sw_kw)
        else:
            self.pos_clf.fit(X_tr, y_pos_tr, eval_set=[(X_vl, y_pos_vl)],
                             callbacks=[lgb.early_stopping(50, verbose=False),
                                        lgb.log_evaluation(period=0)], **sw_kw)

        print(f"  Fitting LGBM quantiles on logit(share) for {self.target}...")
        pos_mask = phi_tr > 0
        phi_pos_tr = np.clip(phi_tr[pos_mask], self._EPS, 1 - self._EPS)
        logit_tr = scipy.special.logit(phi_pos_tr)
        sw_pos = sample_weight[pos_mask] if sample_weight is not None else None

        pos_mask_vl = phi_vl > 0
        phi_pos_vl = np.clip(phi_vl[pos_mask_vl], self._EPS, 1 - self._EPS)
        logit_vl = scipy.special.logit(phi_pos_vl)
        X_vl_pos = X_vl[pos_mask_vl]

        print(f"    n_pos={pos_mask.sum()}  logit_phi mean={logit_tr.mean():.2f}  std={logit_tr.std():.2f}")

        # Temporal-OOF: probe best tree count once (median q), reuse for all quantiles.
        X_tr_pos = X_tr[pos_mask]
        _cq_best_n: "int | None" = None
        if temporal_oof and train_dates is not None and len(phi_pos_tr) > 20:
            _dates_pos_c = pd.Series(
                pd.to_datetime(
                    train_dates.values if hasattr(train_dates, "values") else list(train_dates)
                )[pos_mask]
            )
            _probe_cq = lgb.LGBMRegressor(
                objective="quantile", alpha=0.5,
                n_estimators=lgbm_cfg["n_estimators"],
                learning_rate=lgbm_cfg["learning_rate"],
                num_leaves=lgbm_cfg["num_leaves"],
                min_child_samples=lgbm_cfg["min_child_samples"],
                verbosity=-1, random_state=SEED,
                deterministic=True, force_row_wise=True, num_threads=1,
                feature_fraction_seed=SEED, bagging_seed=SEED,
                data_random_seed=SEED, extra_seed=SEED, objective_seed=SEED,
            )
            _cq_best_n = _oof_best_n(_probe_cq, X_tr_pos, logit_tr, _dates_pos_c, sw_pos,
                                     min_hold=50, frac=0.2)
            if _cq_best_n is not None:
                print(f"    [temporal_oof] {self.target} quantile best_n={_cq_best_n}")

        self.quantile_models = []
        for q in QUANTILE_GRID:
            m = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                n_estimators=lgbm_cfg["n_estimators"],
                learning_rate=lgbm_cfg["learning_rate"],
                num_leaves=lgbm_cfg["num_leaves"],
                min_child_samples=lgbm_cfg["min_child_samples"],
                verbosity=-1, random_state=SEED,
                deterministic=True, force_row_wise=True, num_threads=1,
                feature_fraction_seed=SEED, bagging_seed=SEED,
                data_random_seed=SEED, extra_seed=SEED, objective_seed=SEED,
            )
            fit_kw = {"sample_weight": sw_pos} if sw_pos is not None else {}
            if temporal_oof and _cq_best_n is not None and len(phi_pos_tr) > 20:
                m.set_params(n_estimators=_cq_best_n)
                m.fit(X_tr_pos, logit_tr, **fit_kw)
            elif len(phi_pos_tr) > 20:
                m.fit(X_tr_pos, logit_tr,
                      eval_set=[(X_vl_pos, logit_vl)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=0)], **fit_kw)
            self.quantile_models.append(m)

        # Hurdle calibration: scalar logit adjustment on val so mean predicted
        # P(ctrl>0) matches the empirical val rate (standard, self-contained hurdle
        # zero-rate match). This is the ONLY honest, reproducible calibration target.
        #
        # NOTE (drift, verified 2026-06-14): ctrl_time is a documented drift-limited
        # Gate-B FAIL. Faithful val/test PIT-KS grids (scripts/_ctrl_val_grid.py) show
        # the zero-control rate drifted 2024(val)=0.176 -> 2025+(test)=0.195, so the
        # val-KS-optimal and test-KS-optimal adj are DISJOINT (val wants adj<=0, test
        # wants adj~0.20). No honest val-based calibration passes test; the only fix is
        # retraining with 2024+ exposure. We deliberately do NOT tune adj toward the
        # test KS (that would be gate-gaming). zero-rate match keeps it principled.
        # Prod mode: X_vl/y_vl above are in-sample (val ⊂ train) — pos_clf was
        # fit on X_tr, and scoring the hurdle-calib target on in-sample val
        # would tune the adjustment against data the classifier can already
        # see. Score against a temporal holdout carved from the tail of train
        # instead (does not affect pos_clf's own fit or the quantile models
        # above, only this calibration target).
        if temporal_oof and train_dates is not None:
            _dates_hc = pd.to_datetime(train_dates).values
            _cutoff_hc = (_dates_hc.max() - pd.DateOffset(months=6)).to_datetime64()
            _mask_hc = _dates_hc >= _cutoff_hc
            if _mask_hc.sum() < 30:
                _mask_hc = np.zeros(len(_dates_hc), dtype=bool)
                _mask_hc[np.argsort(_dates_hc)[-30:]] = True
            X_hurdle_calib, y_hurdle_calib = X_tr[_mask_hc], y_tr[_mask_hc]
        else:
            X_hurdle_calib, y_hurdle_calib = X_vl, y_vl
        p_pos_pred_vl = self.pos_clf.predict_proba(X_hurdle_calib)[:, 1]
        emp_p_pos_vl = float((y_hurdle_calib > 0).mean())
        print(f"  {self.target} hurdle calib: pred_p_pos(val)={p_pos_pred_vl.mean():.4f}  "
              f"emp_p_pos(val)={emp_p_pos_vl:.4f}")
        if 0.01 < emp_p_pos_vl < 0.99:
            def _obj(adj: float) -> float:
                return float(scipy.special.expit(
                    scipy.special.logit(p_pos_pred_vl) - adj
                ).mean()) - emp_p_pos_vl
            try:
                val_fit_adj = float(scipy.optimize.brentq(_obj, -5.0, 5.0))
            except Exception:
                val_fit_adj = 0.0
            self.hurdle_logit_adj = val_fit_adj
        else:
            self.hurdle_logit_adj = 0.0
        print(f"  {self.target} hurdle_logit_adj={self.hurdle_logit_adj:.4f}  "
              f"(val zero-rate match)")

        # Positive-share dispersion: kept NEUTRAL (1.0 = native quantile dispersion).
        # The IQR moment-match (-> ~0.81, narrow) and the faithful PIT-KS (-> widen)
        # disagree on direction, so neither is a trustworthy lever; adding either would
        # be unvalidated. We trust the LGBM quantile spread as-is.
        self.share_disp = 1.0
        print(f"  {self.target} share_disp={self.share_disp:.4f} (neutral)")

        return self

    def predict_cdf(self, X: pd.DataFrame,
                    duration_cdfs: list | None = None,
                    method_proba: np.ndarray | None = None,
                    duration_cdfs_by_method: dict | None = None,
                    **_kwargs) -> list[RateXDurationCDF]:
        """Return ctrl_sec CDF for each row via MC over (logit_phi, duration)."""
        from ufc.models.props_duration import _build_dur_cdf_grid, duration_inverse_cdf  # noqa: PLC0415

        Xf = X.reindex(columns=self.feature_cols, fill_value=0).fillna(0)
        Xa = Xf.values
        n_rows = len(Xf)

        p_pos_all = self.pos_clf.predict_proba(Xa)[:, 1]
        _hadj = getattr(self, "hurdle_logit_adj", 0.0)
        if _hadj != 0.0:
            p_pos_all = scipy.special.expit(scipy.special.logit(p_pos_all) - _hadj)
        qs = np.array(QUANTILE_GRID)
        logit_preds = np.column_stack([m.predict(Xa) for m in self.quantile_models])
        logit_preds = np.maximum.accumulate(logit_preds, axis=1)

        _method_names = ["KO/TKO", "SUB", "DEC"]
        rng = np.random.default_rng(SEED)

        _adj = getattr(self, "method_logit_share_adj", None)
        _adj_arr = np.zeros(3)
        if _adj:
            for _k, _mn in enumerate(_method_names):
                _adj_arr[_k] = _adj.get(_mn, 0.0)

        _method_grids: list[list] | None = None
        if duration_cdfs_by_method is not None:
            _method_grids = [[], [], []]
            for _mc, _mn in enumerate(_method_names):
                _cdfs_m = duration_cdfs_by_method.get(_mn, [])
                for _mi in range(n_rows):
                    _dc = _cdfs_m[_mi] if _mi < len(_cdfs_m) else None
                    if _dc is not None:
                        _method_grids[_mc].append(
                            _build_dur_cdf_grid(_dc, float(_dc._scheduled_sec))
                        )
                    else:
                        _method_grids[_mc].append(None)

        prebuilt_grids = []
        for i in range(n_rows):
            dc = duration_cdfs[i] if duration_cdfs and i < len(duration_cdfs) else None
            if dc is not None:
                prebuilt_grids.append(_build_dur_cdf_grid(dc, float(dc._scheduled_sec)))
            else:
                prebuilt_grids.append(None)

        cdfs_out = []
        for i in range(n_rows):
            p_pos = p_pos_all[i]
            lp = logit_preds[i]
            mp = method_proba[i] if method_proba is not None else np.full(3, 1.0 / 3)

            method_codes = rng.choice(3, size=_N_MC, p=mp / mp.sum())

            dur_sec_arr = np.empty(_N_MC)
            for mc_k, mn in enumerate(_method_names):
                mask = method_codes == mc_k
                if not mask.any():
                    continue
                n_m = int(mask.sum())
                g_p = None
                if _method_grids is not None and _method_grids[mc_k][i] is not None:
                    g_p = _method_grids[mc_k][i]
                elif prebuilt_grids[i] is not None:
                    g_p = prebuilt_grids[i]
                if g_p is not None:
                    g, p = g_p
                    u = rng.random(n_m)
                    dur_sec_arr[mask] = duration_inverse_cdf(None, u, _prebuilt_grid=(g, p))
                else:
                    dur_sec_arr[mask] = 900.0

            hurdle = rng.random(_N_MC) < p_pos
            u_q = rng.random(_N_MC)
            logit_phi_s = np.interp(u_q, qs, lp)
            _sdisp = getattr(self, "share_disp", 1.0)
            if _sdisp != 1.0:
                _med = np.interp(0.5, qs, lp)
                logit_phi_s = _med + _sdisp * (logit_phi_s - _med)
            if _adj:
                logit_phi_s = logit_phi_s + _adj_arr[method_codes]
            phi_s = scipy.special.expit(logit_phi_s)

            ctrl_s = np.where(hurdle, phi_s * dur_sec_arr, 0.0).clip(0.0)
            cdfs_out.append(RateXDurationCDF(ctrl_s, 1.0 - p_pos))

        return cdfs_out

    def fit_method_adjustments(self,
                               X_train: pd.DataFrame, y_ctrl_train: pd.Series,
                               total_sec_train: np.ndarray,
                               method_train: pd.Series,
                               sample_weight: np.ndarray | None = None) -> None:
        """Fit per-method logit(phi) residuals on train, store as method_logit_share_adj."""
        y = y_ctrl_train.fillna(0).clip(0).values.astype(float)
        tot = np.maximum(total_sec_train, 1.0)
        phi = np.clip(y / tot, self._EPS, 1 - self._EPS)
        pos_mask = y > 0

        Xf = X_train[self.feature_cols].fillna(0).values
        logit_preds = np.column_stack([m.predict(Xf) for m in self.quantile_models])
        logit_preds = np.maximum.accumulate(logit_preds, axis=1)
        qs = np.array(QUANTILE_GRID)
        pred_median_logit = logit_preds[:, len(qs) // 2]

        actual_logit = scipy.special.logit(phi)
        residual = actual_logit - pred_median_logit

        adj: dict[str, float] = {}
        _method_names = ["KO/TKO", "SUB", "DEC"]
        method_arr = method_train.fillna("").values
        for mn in _method_names:
            if "DEC" in mn:
                mask = np.array(["DEC" in str(m).upper() for m in method_arr]) & pos_mask
            elif "KO" in mn:
                mask = np.array(["KO" in str(m).upper() for m in method_arr]) & pos_mask
            else:
                mask = np.array(["SUB" in str(m).upper() for m in method_arr]) & pos_mask
            if mask.sum() < 10:
                adj[mn] = 0.0
                continue
            w = sample_weight[mask] if sample_weight is not None else None
            if w is not None:
                adj[mn] = float(np.average(residual[mask], weights=w))
            else:
                adj[mn] = float(residual[mask].mean())

        self.method_logit_share_adj = adj
        print(f"  {self.target} method logit-share adj: KO={adj.get('KO/TKO',0):.3f}  SUB={adj.get('SUB',0):.3f}  DEC={adj.get('DEC',0):.3f}")

    def save(self, path: Path, gitsha: str = "latest") -> Path:
        out = path / f"props_{self.target}_{gitsha}.joblib"
        joblib.dump(self, out, compress=3)
        return out

    @staticmethod
    def load(path: Path) -> "ControlShareModel":
        return joblib.load(path)


def predict_combined_count_cdf(
    model: RateHurdleCountModel,
    X_a: pd.DataFrame,
    X_b: pd.DataFrame,
    duration_cdfs: list | None = None,
    method_proba: np.ndarray | None = None,
    duration_cdfs_by_method: dict | None = None,
    n_mc: int = _N_MC,
    seed: int = SEED,
) -> RateXDurationCDF:
    """Combined both-fighter count CDF sharing duration draws across fighters.

    Used for "Sig Strikes Combo" (A + B total) — both fighters experience the
    same fight, so duration is sampled once per draw and shared. Rate draws for
    each fighter are independent conditional on duration/method. v1 captures the
    duration/method-induced correlation only; residual cross-fighter rate
    correlation is unmodeled (known approximation).

    Parameters
    ----------
    model : RateHurdleCountModel
        The trained sig_strikes model (or any full-fight count model).
    X_a, X_b : pd.DataFrame
        Single-row feature DataFrames for fighter A and fighter B respectively.
    duration_cdfs : list[DurationCDF] | None
        One DurationCDF per row (single-element list for single matchup).
    method_proba : np.ndarray | None
        Shape (1, 3) [P(KO), P(SUB), P(DEC)].
    duration_cdfs_by_method : dict | None
        Keys "KO/TKO", "SUB", "DEC"; values are single-element lists of DurationCDF.
    n_mc : int
        Number of MC draws. Matches _N_MC default for consistency.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    RateXDurationCDF
        CDF of (count_A + count_B) totals across n_mc draws.
    """
    from ufc.models.props_duration import _build_dur_cdf_grid, duration_inverse_cdf  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    qs = np.array(QUANTILE_GRID)
    _method_names = ["KO/TKO", "SUB", "DEC"]

    def _get_fighter_preds(X: pd.DataFrame) -> tuple[float, np.ndarray]:
        """Return (p_pos, log_rate_preds_row)."""
        Xf = X.reindex(columns=model.feature_cols, fill_value=0).fillna(0)
        Xa = Xf.values
        p_pos = float(model.pos_clf.predict_proba(Xa)[0, 1])
        log_rate_row = np.column_stack([m.predict(Xa) for m in model.quantile_models])
        log_rate_row = np.maximum.accumulate(log_rate_row, axis=1)
        _rcf = getattr(model, "rate_calib_factor", 1.0)
        if _rcf != 1.0:
            log_rate_row = log_rate_row + np.log(_rcf)
        return p_pos, log_rate_row[0]

    p_pos_a, lrp_a = _get_fighter_preds(X_a)
    p_pos_b, lrp_b = _get_fighter_preds(X_b)

    # Build duration grid for row 0
    dur_cdf = duration_cdfs[0] if duration_cdfs else None

    if dur_cdf is not None:
        # Scheduled ceiling in seconds
        if "scheduled_rounds" in X_a.columns:
            sr = float(X_a["scheduled_rounds"].iloc[0])
        else:
            sr = 3.0
        ceil_sec = sr * 300.0
        effective_ceil = min(ceil_sec, float(getattr(dur_cdf, "_scheduled_sec", ceil_sec)))
        grid, probs = _build_dur_cdf_grid(dur_cdf, effective_ceil)
    else:
        grid = probs = None

    # Method-conditional duration grids
    method_grids: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    if duration_cdfs_by_method:
        for mn in _method_names:
            m_cdfs = duration_cdfs_by_method.get(mn)
            if m_cdfs and len(m_cdfs) > 0 and m_cdfs[0] is not None:
                mc = m_cdfs[0]
                mc_ceil = float(getattr(mc, "_scheduled_sec", 900.0))
                method_grids[mn] = _build_dur_cdf_grid(mc, mc_ceil)
            else:
                method_grids[mn] = None

    # MC draws
    # Sample method per draw
    if method_proba is not None and method_proba.shape[0] >= 1:
        mp = method_proba[0]
    else:
        mp = np.array([1 / 3, 1 / 3, 1 / 3])

    method_codes = rng.choice(len(_method_names), size=n_mc, p=mp / mp.sum())

    # Sample duration per draw (shared by both fighters)
    dur_sec_arr = np.empty(n_mc)
    for mc_i, mn in enumerate(_method_names):
        mask = method_codes == mc_i
        if not mask.any():
            continue
        n_m = int(mask.sum())
        mg = method_grids.get(mn) if duration_cdfs_by_method else None
        if mg is None:
            mg = (grid, probs) if grid is not None else None
        if mg is not None:
            g, p = mg
            u = rng.random(n_m)
            dur_sec_arr[mask] = duration_inverse_cdf(None, u, _prebuilt_grid=(g, p))
        else:
            dur_sec_arr[mask] = 900.0  # fallback: full 3 rounds

    act_min_arr = np.maximum(dur_sec_arr / 60.0, 0.5 / 60.0)

    _rate_ceiling = getattr(model, "rate_ceiling", None)
    _combo_mla = getattr(model, "method_log_rate_adj", None)
    _combo_adj_arr: np.ndarray | None = None
    if _combo_mla:
        _combo_adj_arr = np.array([_combo_mla.get(m, 0.0) for m in _method_names])

    def _sample_counts(p_pos: float, lrp: np.ndarray) -> np.ndarray:
        hurdle = rng.random(n_mc) < p_pos
        u_r = rng.random(n_mc)
        log_r = np.interp(u_r, qs, lrp)
        if _combo_adj_arr is not None:
            log_r = log_r + _combo_adj_arr[method_codes]
        rate = np.exp(log_r)
        if _rate_ceiling is not None:
            rate = np.minimum(rate, _rate_ceiling)
        counts = np.where(hurdle, rate * act_min_arr, 0.0)
        return counts

    counts_a = _sample_counts(p_pos_a, lrp_a)
    counts_b = _sample_counts(p_pos_b, lrp_b)
    total = counts_a + counts_b

    p_zero = float((total == 0).mean())
    return RateXDurationCDF(total, p_zero)
