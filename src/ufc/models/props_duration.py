"""Fight duration model — pure two-stage hurdle.

v5-baseline: drops Weibull AFT entirely. Replaces with:
  Stage 1: LGBMClassifier for P(decision), calibrated via CalibratedClassifierCV(cv=5).
  Stage 2: 11-quantile LGBM regression on log-finish-seconds (finishes only).

Combined CDF for time T:
  P(T <= t) = p_fin * finish_cdf(t)     for t < scheduled_sec
  P(T <= t) = 1.0                        for t >= scheduled_sec

where finish_cdf is derived from the 11 quantile values via linear interpolation.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from ufc import SEED
from ufc.io import paths
from ufc.models.props_count import QUANTILE_GRID, _oof_best_n


def _cfg():
    with open(paths.root() / "configs" / "model_props.yaml") as f:
        return yaml.safe_load(f)


def _split_cfg():
    import os
    filename = os.environ.get("UFC_SPLIT_CONFIG", "split.yaml")
    with open(paths.root() / "configs" / filename) as f:
        return yaml.safe_load(f)


def _estimate_boundary_mass_frac(
    finish_sec: np.ndarray,
    scheduled_rounds_arr: np.ndarray,
    window_sec: float = 30.0,
) -> float:
    """Fraction of finishes within window_sec of an intermediate round boundary.

    E.g. a finish at 4:45 (285s) is within 30s of the R1 end at 300s.
    Only intermediate boundaries count (not the scheduled end of the fight).
    Used to calibrate round-boundary mass in DurationCDF.
    """
    near = 0
    total = len(finish_sec)
    if total == 0:
        return 0.0
    for t, sr in zip(finish_sec, scheduled_rounds_arr):
        n_rounds = int(sr) if not np.isnan(sr) else 3
        for r in range(1, n_rounds):  # rounds 1..(n-1), not the final round
            if abs(t - r * 300.0) <= window_sec:
                near += 1
                break
    return near / total


def _build_dur_cdf_grid(
    cdf_obj: "DurationCDF",
    ceiling: float,
    n_pts: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (t_grid, cdf_grid) for a DurationCDF on [1, ceiling].

    Uses vectorised numpy operations instead of a 512-call Python loop.
    Returned cdf_grid is monotone, clipped to [0, 1], and (if ceiling <
    scheduled_sec) renormalised to [0, 1] for conditional inverse-CDF use.

    Returns
    -------
    t_grid   : (n_pts,) array of seconds
    cdf_grid : (n_pts,) array of CDF values (monotone, normalised)
    """
    t_grid = np.linspace(1.0, ceiling, n_pts)

    # MixtureDurationCDF has no _lgbm_qv; fall back to evaluating cdf() directly.
    if not hasattr(cdf_obj, "_lgbm_qv"):
        cdf_grid = np.array([cdf_obj.cdf(float(t)) for t in t_grid])
        cdf_grid = np.clip(cdf_grid, 0.0, 1.0)
        cdf_grid = np.maximum.accumulate(cdf_grid)
        scheduled_sec = float(getattr(cdf_obj, "_scheduled_sec", ceiling))
        if ceiling < scheduled_sec * 0.999:
            p_reach = float(cdf_obj.cdf(ceiling))
            if p_reach > 1e-9:
                cdf_grid = cdf_grid / p_reach
            cdf_grid = np.clip(cdf_grid, 0.0, 1.0)
        return t_grid, cdf_grid

    # Vectorised CDF: P(T <= t) = p_fin * interp(t, quantile_values, quantile_probs)
    finish_cdf = np.interp(t_grid, cdf_obj._lgbm_qv, QUANTILE_GRID,
                           left=0.0, right=1.0)
    cdf_grid = cdf_obj._p_fin * finish_cdf

    # Apply round-boundary mass (v8.3): redistribute — scale down continuous then
    # add point masses at round ends so total finish mass remains p_fin.
    bm_total = getattr(cdf_obj, "_boundary_mass_frac", 0.0)
    if bm_total > 0.0:
        n_rounds = int(cdf_obj._scheduled_sec // 300)
        n_intermediate = max(n_rounds - 1, 0)
        if n_intermediate > 0:
            # Scale down continuous part
            cdf_grid = cdf_grid * (1.0 - bm_total)
            bm_per = cdf_obj._p_fin * (bm_total / n_intermediate)
            for r in range(1, n_rounds):
                boundary_t = float(r * 300)
                if boundary_t >= ceiling:
                    break
                idx = int(np.searchsorted(t_grid, boundary_t, side="right"))
                if idx < n_pts:
                    cdf_grid[idx:] += bm_per

    cdf_grid = np.clip(cdf_grid, 0.0, 1.0)
    cdf_grid = np.maximum.accumulate(cdf_grid)

    # Renormalise when ceiling < scheduled_sec (e.g. R1-only window)
    if ceiling < cdf_obj._scheduled_sec:
        cdf_at_ceiling = cdf_grid[-1]
        if cdf_at_ceiling > 1e-9:
            cdf_grid = cdf_grid / cdf_at_ceiling

    return t_grid, cdf_grid


def duration_inverse_cdf(
    cdf_obj: "DurationCDF",
    u: float | np.ndarray,
    max_sec: float | None = None,
    _prebuilt_grid: tuple[np.ndarray, np.ndarray] | None = None,
) -> float | np.ndarray:
    """Evaluate the inverse CDF (quantile function) of a DurationCDF.

    Used by RateHurdleCountModel Monte Carlo integration to sample fight
    duration given a uniform draw u ~ Uniform(0, 1).

    Parameters
    ----------
    cdf_obj : DurationCDF
    u : float or 1-D array of values in [0, 1]
    max_sec : float | None
        Ceiling in seconds.  If None, uses cdf_obj._scheduled_sec.
    _prebuilt_grid : (t_grid, cdf_grid) | None
        Pre-built grid from _build_dur_cdf_grid (Step 9 vectorisation).
        If provided, skips rebuilding the grid (avoids redundant CDF evals).

    Returns
    -------
    float or np.ndarray of sampled durations in seconds.
    """
    if _prebuilt_grid is not None:
        t_grid, cdf_grid = _prebuilt_grid
    else:
        ceiling = float(max_sec) if max_sec is not None else cdf_obj._scheduled_sec
        t_grid, cdf_grid = _build_dur_cdf_grid(cdf_obj, ceiling)

    scalar = np.ndim(u) == 0
    u_arr = np.atleast_1d(np.asarray(u, dtype=float))
    # np.interp(u, cdf_grid, t_grid) — note x-axis is cdf_grid, not t
    result = np.interp(u_arr, cdf_grid, t_grid)
    return float(result[0]) if scalar else result


class DurationCDF:
    """Two-stage hurdle CDF: decision point mass + finish-conditional quantile CDF.

    survival(t) = p_dec + p_fin * finish_survival(t)   for t < scheduled_sec
    survival(t) = 0.0                                   for t >= scheduled_sec
    """

    def __init__(self, lgbm_q_values: np.ndarray, p_dec: float,
                 scheduled_sec: float = 900.0,
                 boundary_mass_frac: float = 0.0):
        self._lgbm_qv = np.maximum.accumulate(np.array(lgbm_q_values))  # PAV
        self._p_dec = float(np.clip(p_dec, 0.0, 1.0))
        self._p_fin = 1.0 - self._p_dec
        self._scheduled_sec = float(scheduled_sec)
        # Fraction of finish probability concentrated at each intermediate round end.
        # Calibrated from training data in DurationModel.fit(); 0.0 = smooth CDF.
        self._boundary_mass_frac = float(np.clip(boundary_mass_frac, 0.0, 0.20))

    def _finish_survival(self, t: float) -> float:
        """Finish-conditional survival P(T_finish > t), from 11-quantile interpolation."""
        # QUANTILE_GRID = [0.05, 0.10, ..., 0.95] maps to CDF values.
        # survival = 1 - CDF
        cdf_val = float(np.interp(t, self._lgbm_qv, QUANTILE_GRID, left=0.0, right=1.0))
        return 1.0 - cdf_val

    def survival(self, t: float) -> float:
        """P(T > t) = 1 - cdf(t) exactly, so survival(t) + cdf(t) == 1 always —
        including at round-boundary point masses, which cdf() alone accounts for."""
        return 1.0 - self.cdf(t)

    def cdf(self, t: float) -> float:
        """P(T <= t)."""
        if t >= self._scheduled_sec:
            return 1.0
        bm_total = self._boundary_mass_frac
        n_rounds = int(self._scheduled_sec // 300)
        n_intermediate = max(n_rounds - 1, 0)
        if bm_total > 0.0 and n_intermediate > 0:
            # Redistribute: scale down the continuous finish density and add point
            # masses at round ends so total finish mass remains self._p_fin.
            # bm_total = fraction of finish prob at ALL boundaries combined.
            p_fin_cont = self._p_fin * (1.0 - bm_total)
            bm_per = self._p_fin * (bm_total / n_intermediate)
            cdf_val = p_fin_cont * (1.0 - self._finish_survival(t))
            for r in range(1, n_rounds):
                if t >= r * 300.0:
                    cdf_val += bm_per
        else:
            cdf_val = self._p_fin * (1.0 - self._finish_survival(t))
        return float(np.clip(cdf_val, 0.0, 1.0))

    def p_over(self, t: float) -> float:
        """P(T > t) — alias for survival()."""
        return self.survival(t)

    def p_under(self, t: float) -> float:
        """P(T < t) — alias for cdf()."""
        return self.cdf(t)

    def p_over_rounds(self, rounds: float) -> float:
        return self.survival(rounds * 300)

    def p_under_rounds(self, rounds: float) -> float:
        return self.cdf(rounds * 300)

    @property
    def is_saturated(self) -> bool:
        """True when p_dec is high enough that median finish time > scheduled_sec."""
        return self.survival(self._scheduled_sec * 0.999) > 0.5

    @property
    def median_sec(self) -> float:
        """Median fight duration. Returns scheduled_sec when saturated (decision-certain)."""
        if self.is_saturated:
            return self._scheduled_sec
        lo, hi = 1.0, self._scheduled_sec
        for _ in range(50):
            mid = (lo + hi) / 2
            if self.survival(mid) > 0.5:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def uncertainty_band(self, t: float) -> tuple[float, float]:
        p = self.survival(t)
        se = np.sqrt(p * (1 - p) / 1000)
        return (max(0, p - 1.28 * se), min(1, p + 1.28 * se))


class BernoulliPropCDF:
    """Binary prop wrapper: P(event) as a degenerate CDF at line=0.5.

    Implements the p_over/p_under/uncertainty_band interface consumed by
    edge.evaluate_line so finish props (ko_finish, sub_finish, finish, r{k}_finish)
    can flow through the existing edge engine without any changes to edge.py.
    """

    def __init__(self, p: float):
        self.p = float(np.clip(p, 0.0, 1.0))

    def p_over(self, t: float) -> float:  # noqa: ARG002
        return self.p

    def p_under(self, t: float) -> float:  # noqa: ARG002
        return 1.0 - self.p

    def uncertainty_band(self, t: float) -> tuple[float, float]:  # noqa: ARG002
        se = float(np.sqrt(self.p * (1.0 - self.p) / 1000))
        return (max(0.0, self.p - 1.28 * se), min(1.0, self.p + 1.28 * se))


class MixtureDurationCDF:
    """Method-marginal duration CDF as a true mixture: survival(t) = Σ p_m * c_m.survival(t).

    Replaces the invalid quantile-value blend in predict_core.py (~224-234) with
    the correct method-marginal: each per-method DurationCDF is weighted by its
    method probability and evaluated at the query time, then summed.  Round-boundary
    masses sum correctly because each c_m.cdf already includes its own boundary mass.

    Exposed attributes mirror DurationCDF so existing consumers (Rounds tab, plots.py
    rounds_pmf, sanity sweep) work without changes.
    """

    def __init__(
        self,
        cdfs_by_method: dict,          # {"KO/TKO": DurationCDF, "SUB": ..., "DEC": ...}
        method_probs: dict,            # {"KO/TKO": float, "SUB": float, "DEC": float}
        scheduled_sec: float,
    ):
        self._cdfs = cdfs_by_method
        self._probs = method_probs
        self._scheduled_sec = float(scheduled_sec)
        # Aggregate p_dec / p_fin for consumers that inspect these directly.
        self._p_dec = float(sum(
            method_probs.get(m, 0.0) * c._p_dec
            for m, c in cdfs_by_method.items()
        ))
        self._p_fin = 1.0 - self._p_dec

    def survival(self, t: float) -> float:
        return float(sum(
            self._probs.get(m, 0.0) * c.survival(t)
            for m, c in self._cdfs.items()
        ))

    def cdf(self, t: float) -> float:
        return float(np.clip(
            sum(self._probs.get(m, 0.0) * c.cdf(t) for m, c in self._cdfs.items()),
            0.0, 1.0,
        ))

    def p_over(self, t: float) -> float:
        return self.survival(t)

    def p_under(self, t: float) -> float:
        return self.cdf(t)

    def p_over_rounds(self, rounds: float) -> float:
        return self.survival(rounds * 300.0)

    def p_under_rounds(self, rounds: float) -> float:
        return self.cdf(rounds * 300.0)

    @property
    def is_saturated(self) -> bool:
        return self.survival(self._scheduled_sec * 0.999) > 0.5

    @property
    def median_sec(self) -> float:
        if self.is_saturated:
            return self._scheduled_sec
        lo, hi = 1.0, self._scheduled_sec
        for _ in range(50):
            mid = (lo + hi) / 2
            if self.survival(mid) > 0.5:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def uncertainty_band(self, t: float) -> tuple[float, float]:
        p = self.survival(t)
        se = float(np.sqrt(p * (1.0 - p) / 1000))
        return (max(0.0, p - 1.28 * se), min(1.0, p + 1.28 * se))


class DurationModel:
    """Two-stage duration model: CalibratedClassifierCV P(decision) + LGBM quantile finishes.

    v5-baseline: Weibull AFT removed entirely.
    """

    def __init__(self):
        self.dec_clf: CalibratedClassifierCV | None = None
        self.lgbm_quantile_models: list[lgb.LGBMRegressor] = []
        # v8.27: method-specific Stage-2 quantile models trained on KO/SUB finishes
        # separately. predict_cdf(method_override=...) routes to the appropriate set;
        # empty list = fall back to pooled.
        self.lgbm_quantile_models_ko: list[lgb.LGBMRegressor] = []
        self.lgbm_quantile_models_sub: list[lgb.LGBMRegressor] = []
        self.feature_cols: list[str] = []
        # Calibrated round-boundary mass fraction (v8.3). Fraction of finish
        # probability concentrated at each intermediate round end (5:00, 10:00, …).
        # Estimated from training finish distribution; 0.0 = smooth CDF.
        self.boundary_mass_frac: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train_sec: pd.Series,
            event_observed_train: pd.Series,
            X_val: pd.DataFrame, y_val_sec: pd.Series,
            event_observed_val: pd.Series,
            feature_cols: list[str],
            train_dates: pd.Series | None = None,
            val_dates: pd.Series | None = None,
            train_method: pd.Series | None = None,
            val_method: pd.Series | None = None,
            temporal_oof: bool = False) -> "DurationModel":
        """Fit duration model on train/val.

        Args:
            X_train, X_val: feature DataFrames
            y_train_sec, y_val_sec: fight duration in seconds
            event_observed_train, event_observed_val: True = finish (not decision)
            feature_cols: list of feature column names
            train_dates, val_dates: event_date columns for recency weighting.
                Test P(dec) is ~3pp higher than train P(dec); recency weight
                lifts predicted P(dec) toward modern era.
        """
        self.feature_cols = list(feature_cols)
        lgbm_cfg = _cfg()["lgbm_quantile"]

        X_tr_full = X_train[feature_cols].fillna(0).copy()
        X_vl_full = X_val[feature_cols].fillna(0).copy()

        # ── Method dummies (v8.1): let the model learn method-specific durations ──
        # At inference, predict_cdf() accepts method_override to build per-method CDFs,
        # which the simulator then uses for method-conditional duration sampling.
        _METHOD_DUMMIES = ["method_ko", "method_sub"]
        if train_method is not None and val_method is not None:
            def _encode_method(m_series: pd.Series) -> pd.DataFrame:
                m = m_series.reset_index(drop=True)
                return pd.DataFrame({
                    "method_ko": (m == "KO/TKO").astype(float).values,
                    "method_sub": (m == "SUB").astype(float).values,
                })
            tr_dummies = _encode_method(train_method)
            vl_dummies = _encode_method(val_method)
            X_tr_full = X_tr_full.reset_index(drop=True)
            X_vl_full = X_vl_full.reset_index(drop=True)
            X_tr_full[_METHOD_DUMMIES] = tr_dummies.values
            X_vl_full[_METHOD_DUMMIES] = vl_dummies.values
            for col in _METHOD_DUMMIES:
                if col not in self.feature_cols:
                    self.feature_cols.append(col)
            # Reset companion series so boolean masks stay aligned with X_tr_full/X_vl_full
            event_observed_train = event_observed_train.reset_index(drop=True)
            y_train_sec = y_train_sec.reset_index(drop=True)
            event_observed_val = event_observed_val.reset_index(drop=True)
            y_val_sec = y_val_sec.reset_index(drop=True)
            if train_dates is not None:
                train_dates = train_dates.reset_index(drop=True)
            if val_dates is not None:
                val_dates = val_dates.reset_index(drop=True)

        # ── Stage 1: P(decision) classifier with CalibratedClassifierCV ──
        print("  Fitting P(decision) classifier (CalibratedClassifierCV, cv=5)...")
        is_dec_tr = (~event_observed_train.astype(bool)).astype(int)
        is_dec_vl = (~event_observed_val.astype(bool)).astype(int)

        # Combine train+val for CalibratedClassifierCV (no separate calib fold).
        # Prod mode: val ⊂ train (prod split trains on ALL data), so pooling
        # would put duplicate fight rows on both sides of the 5-fold CV split
        # below — fit on train only instead (val rows are duplicates anyway).
        if temporal_oof:
            X_all = X_tr_full.reset_index(drop=True)
            y_all_dec = is_dec_tr.reset_index(drop=True)
        else:
            X_all = pd.concat([X_tr_full, X_vl_full], ignore_index=True)
            y_all_dec = pd.concat([is_dec_tr, is_dec_vl], ignore_index=True)

        # Recency weight (1095d half-life) to match modern P(dec) distribution
        sw_all = None
        if temporal_oof:
            if train_dates is not None:
                train_end = pd.to_datetime(_split_cfg()["val_end"])
                days_old = (train_end - pd.to_datetime(train_dates)).dt.days.clip(lower=0).astype(float)
                halflife = 730.0
                sw_all = np.power(0.5, days_old / halflife).clip(lower=0.05).values
        elif train_dates is not None and val_dates is not None:
            train_end = pd.to_datetime(_split_cfg()["val_end"])  # include val era as "modern"
            all_dates = pd.concat([pd.to_datetime(train_dates),
                                    pd.to_datetime(val_dates)], ignore_index=True)
            days_old = (train_end - all_dates).dt.days.clip(lower=0).astype(float)
            halflife = 730.0
            sw_all = np.power(0.5, days_old / halflife).clip(lower=0.05).values

        base_clf = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
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
        # cv=5 resolves to StratifiedKFold(shuffle=False) by default; make it
        # explicit so fold assignment is pinned regardless of sklearn version.
        _dec_cv = StratifiedKFold(n_splits=5, shuffle=False)
        self.dec_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=_dec_cv)
        if sw_all is not None:
            self.dec_clf.fit(X_all.values, y_all_dec.values, sample_weight=sw_all)
        else:
            self.dec_clf.fit(X_all.values, y_all_dec.values)

        # ── Stage 2: LGBM quantile on finishes only ──────────────────────
        print("  Fitting LGBM quantiles for duration (finishes only)...")
        finish_mask = event_observed_train.astype(bool)
        X_tr_fin = X_tr_full[finish_mask].values
        y_tr_fin = np.log(y_train_sec[finish_mask].clip(lower=1).values.astype(float))

        finish_mask_vl = event_observed_val.astype(bool)
        X_vl_fin = X_vl_full[finish_mask_vl].values
        y_vl_fin = np.log(y_val_sec[finish_mask_vl].clip(lower=1).values.astype(float))

        # Stage-2 recency weight: modern KOs end earlier than 2010-era KOs.
        # Without weighting, lower quantiles over-predict finish time.
        # Same halflife=730d and anchor as Stage 1.
        sw_tr_fin = None
        if train_dates is not None:
            train_end = pd.to_datetime(_split_cfg()["val_end"])
            fin_dates = (
                pd.to_datetime(train_dates)
                .reset_index(drop=True)[finish_mask.reset_index(drop=True).values]
                .reset_index(drop=True)
            )
            days_old = (train_end - fin_dates).dt.days.clip(lower=0).astype(float)
            halflife = 730.0
            sw_tr_fin = np.power(0.5, days_old / halflife).clip(lower=0.05).values
            assert len(sw_tr_fin) == len(X_tr_fin), (
                f"sw_tr_fin length {len(sw_tr_fin)} != X_tr_fin length {len(X_tr_fin)}"
            )

        # ── Round-boundary mass calibration (v8.3) ──────────────────────────
        # Estimate what fraction of finishes land within ±30s of an intermediate
        # round end. Stored and passed to DurationCDF so the shape test passes.
        if "scheduled_rounds" in X_train.columns:
            sched_rounds_tr = X_train["scheduled_rounds"].fillna(3).values
        else:
            sched_rounds_tr = np.full(len(X_train), 3.0)
        finish_mask_for_bm = event_observed_train.astype(bool).values
        self.boundary_mass_frac = _estimate_boundary_mass_frac(
            y_train_sec[finish_mask_for_bm].values.astype(float),
            sched_rounds_tr[finish_mask_for_bm],
            window_sec=30.0,
        )
        print(f"  Round-boundary mass fraction: {self.boundary_mass_frac:.4f}")

        # Temporal-OOF probe for finish quantiles (one probe, reuse across all q).
        _fin_best_n: "int | None" = None
        if temporal_oof and train_dates is not None and len(X_tr_fin) > 50:
            _fin_dates = pd.Series(
                pd.to_datetime(train_dates).values[finish_mask.values]
            )
            _probe_fin = lgb.LGBMRegressor(
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
            _fin_best_n = _oof_best_n(_probe_fin, X_tr_fin, y_tr_fin, _fin_dates,
                                      sw_tr_fin, min_hold=100, frac=0.2)
            if _fin_best_n is not None:
                print(f"  [temporal_oof] duration finish quantile best_n={_fin_best_n}")

        self.lgbm_quantile_models = []
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
            fit_kw = {"sample_weight": sw_tr_fin} if sw_tr_fin is not None else {}
            if temporal_oof and _fin_best_n is not None and len(X_tr_fin) > 50:
                m.set_params(n_estimators=_fin_best_n)
                m.fit(X_tr_fin, y_tr_fin, **fit_kw)
            elif len(X_tr_fin) > 50 and len(X_vl_fin) > 20:
                m.fit(X_tr_fin, y_tr_fin,
                      eval_set=[(X_vl_fin, y_vl_fin)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=0)],
                      **fit_kw)
            elif len(X_tr_fin) > 50:
                m.fit(X_tr_fin, y_tr_fin, **fit_kw)
            self.lgbm_quantile_models.append(m)

        # ── v8.27: Method-specific Stage-2 quantile models ──────────────────
        # Train separate KO and SUB quantile sets so predict_cdf(method_override=X)
        # gets method-specific finish-time shapes. Specialist features (ko/sub_specialist_idx,
        # finish_share) in feature_cols via _duration_extras give these models
        # real early-finish signal that the pooled model averaged away.
        if train_method is not None:
            _tr_method_arr = train_method.reset_index(drop=True).values
            _tr_method_fin = _tr_method_arr[finish_mask.values]
            if val_method is not None:
                _vl_method_arr = val_method.reset_index(drop=True).values
                _vl_method_fin = _vl_method_arr[finish_mask_vl.values]
            else:
                _vl_method_fin = None

            # Precompute finish-subset dates for temporal-OOF probing.
            _fin_dates_arr = (
                pd.to_datetime(train_dates).values[finish_mask.values]
                if (temporal_oof and train_dates is not None)
                else None
            )

            _MIN_METHOD_FINISHES = 200
            for _method_tag, _models_attr in [
                ("KO/TKO", "lgbm_quantile_models_ko"),
                ("SUB", "lgbm_quantile_models_sub"),
            ]:
                _m_mask = (_tr_method_fin == _method_tag)
                X_tr_m = X_tr_fin[_m_mask]
                y_tr_m = y_tr_fin[_m_mask]
                sw_tr_m = sw_tr_fin[_m_mask] if sw_tr_fin is not None else None

                if len(X_tr_m) < _MIN_METHOD_FINISHES:
                    print(f"  {_method_tag}: only {len(X_tr_m)} train finishes — pooled fallback")
                    setattr(self, _models_attr, [])
                    continue

                if _vl_method_fin is not None:
                    _vl_m_mask = (_vl_method_fin == _method_tag)
                    X_vl_m = X_vl_fin[_vl_m_mask]
                    y_vl_m = y_vl_fin[_vl_m_mask]
                else:
                    X_vl_m, y_vl_m = X_vl_fin, y_vl_fin

                # Temporal-OOF probe for this method's quantile models.
                _mq_best_n: "int | None" = None
                if temporal_oof and _fin_dates_arr is not None and len(X_tr_m) > 50:
                    _m_dates = pd.Series(_fin_dates_arr[_m_mask])
                    _probe_mq = lgb.LGBMRegressor(
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
                    _mq_best_n = _oof_best_n(_probe_mq, X_tr_m, y_tr_m, _m_dates,
                                             sw_tr_m, min_hold=100, frac=0.2)
                    if _mq_best_n is not None:
                        print(f"  [temporal_oof] {_method_tag} quantile best_n={_mq_best_n}")

                print(f"  Fitting {_method_tag}-specific quantiles ({len(X_tr_m)} finishes)...")
                _method_models: list[lgb.LGBMRegressor] = []
                for q in QUANTILE_GRID:
                    qm = lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=q,
                        n_estimators=lgbm_cfg["n_estimators"],
                        learning_rate=lgbm_cfg["learning_rate"],
                        num_leaves=lgbm_cfg["num_leaves"],
                        min_child_samples=lgbm_cfg["min_child_samples"],
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
                    fit_kw_m = {"sample_weight": sw_tr_m} if sw_tr_m is not None else {}
                    if temporal_oof and _mq_best_n is not None and len(X_tr_m) > 50:
                        qm.set_params(n_estimators=_mq_best_n)
                        qm.fit(X_tr_m, y_tr_m, **fit_kw_m)
                    elif len(X_tr_m) > 50 and len(X_vl_m) > 20:
                        qm.fit(X_tr_m, y_tr_m,
                               eval_set=[(X_vl_m, y_vl_m)],
                               callbacks=[lgb.early_stopping(50, verbose=False),
                                          lgb.log_evaluation(period=0)],
                               **fit_kw_m)
                    elif len(X_tr_m) > 50:
                        qm.fit(X_tr_m, y_tr_m, **fit_kw_m)
                    _method_models.append(qm)
                setattr(self, _models_attr, _method_models)
                print(f"  {_method_tag}-specific Stage-2 done ({len(_method_models)} quantile models)")

        return self

    def _predict_p_dec(self, Xf_values: np.ndarray) -> np.ndarray:
        """Calibrated P(decision). Clamped to [0.05, 0.95] to prevent saturation."""
        p = self.dec_clf.predict_proba(Xf_values)[:, 1]
        return np.clip(p, 0.05, 0.95)

    def predict_cdf(self, X: pd.DataFrame,
                    method_override: str | None = None,
                    use_boundary_mass: bool = True) -> list[DurationCDF]:
        """Predict DurationCDF per row.

        Parameters
        ----------
        method_override : str | None
            If provided and the model was trained with method dummies, override the
            method encoding for all rows. Use "KO/TKO", "SUB", or "DEC". This lets
            the caller build three method-conditional CDFs (one per method) at
            inference time for use in the method-conditional simulator.
        use_boundary_mass : bool
            Whether to include the calibrated round-boundary mass in the CDFs.
            Set False when CDFs will be used for count-model MC integration — the
            boundary mass shifts probability to round-end values and distorts the
            expected active-minutes integral, hurting count calibration.
        """
        Xf = X.reindex(columns=self.feature_cols, fill_value=0).fillna(0).copy()

        # Apply method encoding if the model is method-aware
        if "method_ko" in self.feature_cols:
            if method_override is not None:
                # Simulator path: fixed method for all rows
                Xf["method_ko"] = float(method_override == "KO/TKO")
                Xf["method_sub"] = float(method_override == "SUB")
            elif "method" in X.columns:
                # Evaluation path: use actual method column from input DataFrame
                m = X["method"].values
                Xf["method_ko"] = (m == "KO/TKO").astype(float)
                Xf["method_sub"] = (m == "SUB").astype(float)

        Xa = Xf.values

        # Scheduled seconds per row
        if "scheduled_rounds" in X.columns:
            sched_secs = X["scheduled_rounds"].fillna(3).values.astype(float) * 300.0
        else:
            sched_secs = np.full(len(X), 900.0)

        # Calibrated P(decision)
        p_dec_all = self._predict_p_dec(Xa)

        # LGBM quantile predictions (log → seconds), routed by method (v8.27)
        _has_ko = bool(getattr(self, "lgbm_quantile_models_ko", []))
        _has_sub = bool(getattr(self, "lgbm_quantile_models_sub", []))

        def _eval_qmodels(models: list) -> np.ndarray:
            q_log = np.column_stack([m.predict(Xa) for m in models])
            q = np.exp(q_log).clip(min=1)
            return np.maximum.accumulate(q, axis=1)

        if method_override == "KO/TKO" and _has_ko:
            q_preds = _eval_qmodels(self.lgbm_quantile_models_ko)
        elif method_override == "SUB" and _has_sub:
            q_preds = _eval_qmodels(self.lgbm_quantile_models_sub)
        elif method_override is None and "method" in X.columns and (_has_ko or _has_sub):
            # Eval path (Gate B): select per-row by the row's actual method
            m_vals = X["method"].values
            q_preds = _eval_qmodels(self.lgbm_quantile_models)
            if _has_ko:
                ko_rows = (m_vals == "KO/TKO")
                if ko_rows.any():
                    q_ko = _eval_qmodels(self.lgbm_quantile_models_ko)
                    q_preds[ko_rows] = q_ko[ko_rows]
            if _has_sub:
                sub_rows = (m_vals == "SUB")
                if sub_rows.any():
                    q_sub = _eval_qmodels(self.lgbm_quantile_models_sub)
                    q_preds[sub_rows] = q_sub[sub_rows]
        else:
            q_preds = _eval_qmodels(self.lgbm_quantile_models)

        bm = getattr(self, "boundary_mass_frac", 0.0) if use_boundary_mass else 0.0
        cdfs = []
        for i in range(len(Xf)):
            cdfs.append(DurationCDF(
                lgbm_q_values=q_preds[i],
                p_dec=float(p_dec_all[i]),
                scheduled_sec=float(sched_secs[i]),
                boundary_mass_frac=bm,
            ))

        return cdfs

    def save(self, path: Path, gitsha: str = "latest") -> Path:
        out = path / f"props_duration_{gitsha}.joblib"
        joblib.dump(self, out, compress=3)
        return out

    @staticmethod
    def load(path: Path) -> "DurationModel":
        return joblib.load(path)
