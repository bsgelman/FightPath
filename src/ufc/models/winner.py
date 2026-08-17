"""Winner classification model — diverse ensemble with OOF blend weights.

v9: 3×LGBM + 2×CatBoost + 2×XGB + 1×Logistic (ratings-only).
  - OOF Nelder-Mead log-loss blend weights
  - val-A for halflife search + early stopping probe
  - val-B for Platt refinement
  - pre-UFC record priors, TrueSkill, Glicko-z monotone constraints
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from scipy.optimize import minimize as scipy_minimize

from ufc import SEED
from ufc.models.base import get_feature_cols
from ufc.training.symmetrize import inference_average
from ufc.training.feature_pruning import prune_features


_LGBM_SEEDS = [42, 43, 44]  # 3 seeds for v9 diverse ensemble

# Soft-cap steepness. Replaces the old hard np.clip ceiling/floor at ±max_prob.
# Higher k => closer to a hard clip (top values cram harder against the cap);
# lower k => more visible spread but more nibble on legit sub-cap values.
_SOFTCAP_K = 60.0


def _soft_cap(p: np.ndarray, cap: float, k: float = _SOFTCAP_K) -> np.ndarray:
    """Smooth, monotone saturation toward the band [1-cap, cap].

    Drop-in for ``np.clip(p, 1-cap, cap)``. Near-identity for p comfortably
    inside the band; in the cap region it bends so probabilities approach — but
    never reach — the cap while preserving their order. So a raw 0.86 still reads
    higher than a raw 0.76 instead of both pinning to exactly the cap, yet all
    values stay <= cap (strictly no more confident than the old hard clip).

    Built from softplus_k(x) = log(1+e^{kx})/k (stable via logaddexp):
      upper soft-min toward  cap:    cap   - softplus_k(cap   - p)
      lower soft-max toward 1-cap:  (1-cap) + softplus_k(p - (1-cap))
    The two boundaries (e.g. 0.25 and 0.75) are far apart, so each operation only
    bends its own tail and is identity for the other — net symmetric about 0.5.
    Monotone => AUROC and the 0.5-threshold accuracy are unchanged.
    """
    p = np.asarray(p, dtype=float)
    cap = float(cap)
    if cap >= 1.0:
        return p
    floor = 1.0 - cap
    upper = cap - np.logaddexp(0.0, k * (cap - p)) / k          # soft-min -> cap
    return floor + np.logaddexp(0.0, k * (upper - floor)) / k   # soft-max -> floor

_CATBOOST_PARAMS = {
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "bagging_temperature": 1.0,
    "random_strength": 1.0,
    "border_count": 128,
    "verbose": 0,
    "allow_writing_files": False,
    "thread_count": 1,
}

_XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_weight": 5,
    "tree_method": "hist",
    "n_jobs": 1,
    "verbosity": 0,
}

# Rating/prior feature prefixes for the logistic-only member
_RATING_FEATURE_PREFIXES = (
    "elo_diff", "glicko_mu_pre", "glicko_z", "ts_z", "ts_mu_pre",
    "pre_ufc_win_rate_shrunk", "age_diff",
)


# Fixed hyperparams — conservative defaults with early stopping.
# n_estimators=1000 acts as a ceiling; early stopping on val determines actual count.
_LGBM_PARAMS = {
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.03,
    "n_estimators": 1000,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    # ── Determinism flags (Step 1) ───────────────────────────────────────
    # Without these, multithreaded histogram building and un-pinned bagging /
    # feature-fraction seeds cause best_iteration_ to drift across re-runs,
    # which produces 8–9 pp swings for extreme archetypes like Oliveira.
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
    "extra_seed": SEED,
    "objective_seed": SEED,
}


# ── Step 2: Monotone constraints for specialist & rating features ──────────────
# Higher specialist / rating for A → higher P(win A) (+1).
# Higher specialist / rating for B → lower P(win A) (−1).
# All other features unconstrained (0).
# Using method="intermediate" which is the LightGBM recommended setting for
# non-trivial constraints (less aggressive than "basic", avoids slow "advanced").
_MONO_POSITIVE = frozenset({
    # Specialist amplifiers (more dangerous finish path for A)
    "sub_specialist_idx_a", "sub_specialist_x_weakness_a",
    "ko_specialist_idx_a",  "ko_specialist_x_weakness_a",
    "grappling_control_threat_a", "finish_share_a",
    "r1_sub_threat_a", "r1_ko_threat_a",
    "sub_threat_a",    "ko_threat_a",
    # Ratings (higher = more skilled A)
    "elo_diff_a", "glicko_mu_pre_a",
    # v9: uncertainty-scaled rating diffs + pre-UFC record
    "glicko_z_a", "ts_z_a", "ts_mu_pre_a", "pre_ufc_win_rate_shrunk_a",
})

_MONO_NEGATIVE = frozenset({
    # B-perspective specialist amplifiers (higher = more dangerous for A to face)
    "sub_specialist_idx_b", "sub_specialist_x_weakness_b",
    "ko_specialist_idx_b",  "ko_specialist_x_weakness_b",
    "grappling_control_threat_b", "finish_share_b",
    "r1_sub_threat_b", "r1_ko_threat_b",
    "sub_threat_b",    "ko_threat_b",
    # B rating (higher = B more skilled)
    "elo_diff_b", "glicko_mu_pre_b",
    # Age difference: positive = A older relative to B = disadvantage for A
    "age_diff",
    # v9: B-side uncertainty-scaled rating diffs + pre-UFC record
    "glicko_z_b", "ts_z_b", "ts_mu_pre_b", "pre_ufc_win_rate_shrunk_b",
})


def _build_monotone_constraints(feature_cols: list[str]) -> list[int] | None:
    """Build LightGBM monotone_constraints list aligned to feature_cols.

    Returns None if no constrained feature is present (avoids unnecessary overhead).
    """
    constraints = []
    n_constrained = 0
    for col in feature_cols:
        if col in _MONO_POSITIVE:
            constraints.append(1)
            n_constrained += 1
        elif col in _MONO_NEGATIVE:
            constraints.append(-1)
            n_constrained += 1
        else:
            constraints.append(0)
    if n_constrained == 0:
        return None
    return constraints


def get_winner_feature_cols(df: pd.DataFrame) -> list[str]:
    """Select numeric feature columns safe for winner model."""
    exclude = [
        "sig_str_landed_a", "sig_str_landed_b", "td_landed_a", "td_landed_b",
        "ctrl_sec_a", "ctrl_sec_b",  # post-fight stats
        # Era baselines are fight-level symmetric signals (one value per
        # event_date or event_date×weight_class); they cannot discriminate A vs B
        # and only add noise to a binary classifier.  They are already excluded
        # from prop count/duration models via tune_props.py; only the winner
        # model was accidentally leaking them in through fight_cols in assemble.py.
        "era_avg_sig_str_l12mo", "wc_finish_share_l2y", "wc_5rd_dec_rate",
        # V7.2: era method baselines — fight-symmetric, assembled with _a/_b suffixes.
        "era_ko_share_l24mo_a", "era_ko_share_l24mo_b",
        "era_sub_share_l24mo_a", "era_sub_share_l24mo_b",
    ]
    cols = get_feature_cols(df, exclude_patterns=exclude)
    cols = [c for c in cols if c in df.columns]
    return prune_features(cols, model_name="winner")


class WinnerModel:
    """Diverse ensemble winner model: 3xLGBM + 2xCatBoost + 2xXGB + 1xLogistic.

    v9 strategy:
    1. Probe LGBM(seed=42) + early stopping on val-A -> best_n (floored at 200)
    2. Train all members on full training data
    3. 5-fold OOF per member -> SLSQP blend weight optimization (log-loss simplex)
    4. Isotonic calibration on blended OOF predictions
    5. Platt refinement on val-B (separate from val-A used for early stopping)
    Inference: blend -> isotonic -> Platt
    """

    def __init__(self, lgbm_params: dict | None = None):
        self.lgbm_params = {**_LGBM_PARAMS, **(lgbm_params or {})}
        self.lgbms: list = []
        self.catboosts: list = []
        self.xgbs: list = []
        self.logistic: LogisticRegression | None = None
        self.blend_weights: np.ndarray | None = None
        self.calibrator: IsotonicRegression | None = None
        self.platt_a: float = 1.0
        self.platt_b: float = 0.0
        self.temperature: float = 1.0
        self.max_prob: float = 1.0
        self.feature_cols: list[str] = []
        self.rating_cols: list[str] = []
        self._rating_idx: list[int] = []
        self._best_n: int = 500

    @property
    def lgbm(self):
        return self.lgbms[0] if self.lgbms else None

    def _make_lgbm(self, seed: int, n_estimators: int, fit_params: dict):
        params = {
            **fit_params,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "extra_seed": seed,
            "objective_seed": seed,
        }
        return lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            verbosity=-1,
            n_estimators=n_estimators,
            random_state=seed,
            **params,
        )

    def _make_catboost(self, seed: int, n_iterations: int):
        try:
            from catboost import CatBoostClassifier  # noqa: PLC0415
        except ImportError:
            return None
        params = {**_CATBOOST_PARAMS, "random_seed": seed, "iterations": n_iterations}
        return CatBoostClassifier(**params)

    def _make_xgb(self, seed: int, n_estimators: int):
        try:
            from xgboost import XGBClassifier  # noqa: PLC0415
        except ImportError:
            return None
        params = {**_XGB_PARAMS, "random_state": seed, "n_estimators": n_estimators}
        return XGBClassifier(objective="binary:logistic", **params)

    def _predict_raw_mean(self, X_arr: np.ndarray) -> np.ndarray:
        return self._blend_predict(X_arr)

    def _blend_predict(self, X_arr: np.ndarray,
                       X_rating: np.ndarray | None = None) -> np.ndarray:
        preds = []
        for m in self.lgbms:
            preds.append(m.predict_proba(X_arr)[:, 1])
        for m in self.catboosts:
            preds.append(m.predict_proba(X_arr)[:, 1])
        for m in self.xgbs:
            preds.append(m.predict_proba(X_arr)[:, 1])
        if self.logistic is not None:
            Xr = X_arr[:, self._rating_idx] if X_rating is None and self._rating_idx else (
                X_rating if X_rating is not None else X_arr
            )
            preds.append(self.logistic.predict_proba(Xr)[:, 1])
        preds_mat = np.column_stack(preds)
        if self.blend_weights is not None and len(self.blend_weights) == preds_mat.shape[1]:
            return preds_mat @ self.blend_weights
        return preds_mat.mean(axis=1)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series,
            feature_cols: list[str],
            sample_weight: np.ndarray | None = None,
            X_val_platt: pd.DataFrame | None = None,
            y_val_platt: pd.Series | None = None,
            temporal_oof: bool = False,
            train_dates: "pd.Series | None" = None) -> "WinnerModel":
        """Fit diverse ensemble.

        X_val / y_val     : val-A window (halflife search + early stopping probe)
        X_val_platt / ... : val-B window (Platt refinement); falls back to val-A if None
        """
        self.feature_cols = feature_cols
        self.rating_cols = [c for c in feature_cols
                            if any(c.startswith(p) for p in _RATING_FEATURE_PREFIXES)]
        self._rating_idx = [feature_cols.index(c) for c in self.rating_cols]

        X_tr = X_train.reindex(columns=feature_cols, fill_value=0).fillna(0)
        X_vl = X_val.reindex(columns=feature_cols, fill_value=0).fillna(0)
        X_tr_arr = X_tr.values
        y_tr_arr = y_train.values

        n_est = self.lgbm_params["n_estimators"]
        fit_params = {k: v for k, v in self.lgbm_params.items() if k != "n_estimators"}

        mc = _build_monotone_constraints(feature_cols)
        if mc is not None:
            fit_params["monotone_constraints"] = mc
            fit_params["monotone_constraints_method"] = "intermediate"
            n_constrained = sum(1 for x in mc if x != 0)
            print(f"  Monotone constraints: {n_constrained}/{len(feature_cols)} constrained")

        sw_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}

        # Step 1: probe LGBM to find best_n
        # Prod mode (temporal_oof): X_val/y_val are in-sample (val ⊂ train) — early
        # stopping against them would inflate best_iteration_ on data the model can
        # already see. Carve a temporal holdout from the tail of train instead,
        # mirroring the Step 5 "recent 18-month" pattern below.
        if temporal_oof and train_dates is not None:
            dates_vec = pd.to_datetime(train_dates).values
            max_date = dates_vec.max()
            cutoff = (max_date - pd.DateOffset(months=6)).to_datetime64()
            probe_val_mask = dates_vec >= cutoff
            if probe_val_mask.sum() < 30:
                probe_val_mask = np.zeros(len(dates_vec), dtype=bool)
                probe_val_mask[np.argsort(dates_vec)[-30:]] = True
            probe_tr_mask = ~probe_val_mask
            probe_X_vl, probe_y_vl = X_tr[probe_val_mask], y_train.values[probe_val_mask]
            probe_X_tr, probe_y_tr = X_tr[probe_tr_mask], y_train.values[probe_tr_mask]
            probe_sw_kw = ({"sample_weight": sample_weight[probe_tr_mask]}
                           if sample_weight is not None else {})
            print(f"  Step 1: Probe fit (seed=42, n_est={n_est}) with early stopping on "
                  f"temporal holdout (n={probe_val_mask.sum()}, prod mode)...")
        else:
            probe_X_tr, probe_y_tr, probe_X_vl, probe_y_vl = X_tr, y_train, X_vl, y_val
            probe_sw_kw = sw_kw
            print(f"  Step 1: Probe fit (seed=42, n_est={n_est}) with early stopping on val-A...")
        probe = self._make_lgbm(SEED, n_est, fit_params)
        probe.fit(
            probe_X_tr, probe_y_tr,
            eval_set=[(probe_X_vl, probe_y_vl)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
            **probe_sw_kw,
        )
        raw_best = int(probe.best_iteration_) if probe.best_iteration_ else n_est
        self._best_n = max(200, raw_best)
        # CB/XGB at lr=0.05 need fewer iters than LGBM at lr=0.03 for same coverage
        cb_n = max(200, int(self._best_n * _LGBM_PARAMS["learning_rate"]
                            / _CATBOOST_PARAMS["learning_rate"]))
        print(f"  Early stopping: raw={raw_best} -> floored={self._best_n}, cb_n={cb_n}")

        # Step 2: Train all members on full training data
        print(f"  Step 2: Training {len(_LGBM_SEEDS)} LGBM + 2 CB + 2 XGB + 1 LR...")
        self.lgbms = [self._make_lgbm(s, self._best_n, fit_params) for s in _LGBM_SEEDS]
        for m in self.lgbms:
            m.fit(X_tr_arr, y_tr_arr, **sw_kw)

        self.catboosts = []
        for seed in [42, 77]:
            m = self._make_catboost(seed, cb_n)
            if m is not None:
                m.fit(X_tr_arr, y_tr_arr, **sw_kw)
                self.catboosts.append(m)

        self.xgbs = []
        for seed in [42, 77]:
            m = self._make_xgb(seed, cb_n)
            if m is not None:
                m.fit(X_tr_arr, y_tr_arr, **sw_kw)
                self.xgbs.append(m)

        if self.rating_cols:
            X_tr_rating = X_tr_arr[:, self._rating_idx]
            self.logistic = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
            self.logistic.fit(X_tr_rating, y_tr_arr, **sw_kw)

        n_members = (len(self.lgbms) + len(self.catboosts) + len(self.xgbs)
                     + (1 if self.logistic else 0))
        print(f"  Ensemble: {len(self.lgbms)} LGBM + {len(self.catboosts)} CB"
              f" + {len(self.xgbs)} XGB + {1 if self.logistic else 0} LR = {n_members} members")

        # Step 3: 5-fold OOF per member -> blend weight optimization
        print(f"  Step 3: 5-fold OOF per member -> optimize blend weights...")
        oof_matrix = np.zeros((len(X_tr_arr), n_members))

        if temporal_oof and train_dates is not None:
            # Prod mode: expanding-window temporal folds — no future data leaks into OOF preds
            dates_arr = pd.to_datetime(train_dates).values
            sort_idx = np.argsort(dates_arr, kind="stable")
            covered = np.zeros(len(X_tr_arr), dtype=bool)
            tss = TimeSeriesSplit(n_splits=5)
            _fold_iter = (
                (sort_idx[tr], sort_idx[te])
                for tr, te in tss.split(X_tr_arr[sort_idx])
            )
            print("  OOF mode: temporal (TimeSeriesSplit, expanding window)")
        else:
            covered = np.ones(len(X_tr_arr), dtype=bool)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            _fold_iter = skf.split(X_tr_arr, y_tr_arr)

        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(_fold_iter, start=1):
            if temporal_oof and train_dates is not None:
                covered[fold_val_idx] = True
            fold_sw_kw = ({"sample_weight": sample_weight[fold_train_idx]}
                          if sample_weight is not None else {})
            col = 0
            for s in _LGBM_SEEDS:
                fm = self._make_lgbm(s, self._best_n, fit_params)
                fm.fit(X_tr_arr[fold_train_idx], y_tr_arr[fold_train_idx], **fold_sw_kw)
                oof_matrix[fold_val_idx, col] = fm.predict_proba(X_tr_arr[fold_val_idx])[:, 1]
                col += 1
            for seed in [42, 77]:
                fm = self._make_catboost(seed, cb_n)
                if fm is not None:
                    fm.fit(X_tr_arr[fold_train_idx], y_tr_arr[fold_train_idx], **fold_sw_kw)
                    oof_matrix[fold_val_idx, col] = fm.predict_proba(X_tr_arr[fold_val_idx])[:, 1]
                col += 1
            for seed in [42, 77]:
                fm = self._make_xgb(seed, cb_n)
                if fm is not None:
                    fm.fit(X_tr_arr[fold_train_idx], y_tr_arr[fold_train_idx], **fold_sw_kw)
                    oof_matrix[fold_val_idx, col] = fm.predict_proba(X_tr_arr[fold_val_idx])[:, 1]
                col += 1
            if self.logistic is not None:
                X_rat_fold = X_tr_arr[:, self._rating_idx]
                flr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
                flr.fit(X_rat_fold[fold_train_idx], y_tr_arr[fold_train_idx], **fold_sw_kw)
                oof_matrix[fold_val_idx, col] = flr.predict_proba(
                    X_rat_fold[fold_val_idx])[:, 1]
            print(f"    Fold {fold_idx}/5 complete.")

        # SLSQP simplex blend weight optimization (covered rows only in temporal mode)
        def _blend_ll(w: np.ndarray) -> float:
            p = np.clip(oof_matrix[covered] @ w, 1e-7, 1 - 1e-7)
            return -float(np.mean(y_tr_arr[covered] * np.log(p) + (1 - y_tr_arr[covered]) * np.log(1 - p)))

        x0 = np.ones(n_members) / n_members
        bounds = [(0.0, 1.0)] * n_members
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        opt = scipy_minimize(_blend_ll, x0, method="SLSQP", bounds=bounds,
                             constraints=constraints, options={"maxiter": 500, "ftol": 1e-8})
        raw_w = np.clip(opt.x, 0.0, 1.0)
        self.blend_weights = raw_w / raw_w.sum()
        print(f"  Blend weights: {np.round(self.blend_weights, 3)}")

        # Step 4: isotonic on blended OOF (covered rows only in temporal mode)
        blended_oof = oof_matrix @ self.blend_weights
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(blended_oof[covered], y_tr_arr[covered])
        print(f"  OOF isotonic calibration complete (n={covered.sum()}/{len(covered)}).")

        # Step 5: Platt refinement + ECE-opt prob cap — tier-aware
        # Prod mode (temporal_oof): use recent 18-month temporal-OOF slice → no in-sample overlap
        # Eval mode (default):      use val-B (or val-A fallback) → unchanged from before
        if temporal_oof and train_dates is not None:
            max_date = pd.to_datetime(train_dates).max()
            cutoff = (max_date - pd.DateOffset(months=18)).to_datetime64()
            dates_vec = pd.to_datetime(train_dates).values
            recent_mask = covered & (dates_vec >= cutoff)
            if recent_mask.sum() < 30:
                recent_mask = covered
            val_raw_oof = blended_oof[recent_mask]
            val_iso_oof = np.clip(self.calibrator.predict(val_raw_oof), 1e-6, 1 - 1e-6)
            val_logit = np.log(val_iso_oof / (1 - val_iso_oof))
            y_pl_arr = y_tr_arr[recent_mask]
            label = f"temporal-OOF(18mo, n={recent_mask.sum()})"
        else:
            X_platt_df = X_val_platt if X_val_platt is not None else X_val
            y_platt_s = y_val_platt if y_val_platt is not None else y_val
            label = "val-B" if X_val_platt is not None else "val-A"
            X_pl = X_platt_df.reindex(columns=feature_cols, fill_value=0).fillna(0)
            X_pl_rating = X_pl.values[:, self._rating_idx] if self._rating_idx else None
            val_raw = self._blend_predict(X_pl.values, X_pl_rating)
            val_iso = np.clip(self.calibrator.predict(val_raw), 1e-6, 1 - 1e-6)
            val_logit = np.log(val_iso / (1 - val_iso))
            y_pl_arr = y_platt_s.values

        def _neg_ll(params: np.ndarray) -> float:
            a, b = params
            p = np.clip(1.0 / (1.0 + np.exp(-(a * val_logit + b))), 1e-7, 1 - 1e-7)
            return -float(np.mean(y_pl_arr * np.log(p) + (1 - y_pl_arr) * np.log(1 - p)))

        from scipy.optimize import minimize as _sp_min
        res = _sp_min(_neg_ll, x0=[1.0, 0.0], method="Nelder-Mead")
        self.platt_a, self.platt_b = float(res.x[0]), float(res.x[1])
        print(f"  Platt ({label}): a={self.platt_a:.4f}, b={self.platt_b:.4f}")

        # Step 5b: ECE-optimal symmetric probability cap
        val_platt = 1.0 / (1.0 + np.exp(-(self.platt_a * val_logit + self.platt_b)))

        def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
            err = 0.0
            for lo, hi in zip(edges[:-1], edges[1:]):
                m_bin = (probs >= lo) & (probs < hi)
                if m_bin.sum() == 0:
                    continue
                err += m_bin.sum() * abs(probs[m_bin].mean() - y[m_bin].mean())
            return err / len(y)

        best_cap, best_ece = 1.0, _ece(val_platt, y_pl_arr)
        for cap in np.arange(0.65, 1.01, 0.05):
            pp = np.clip(val_platt, 1.0 - cap, cap)
            e = _ece(pp, y_pl_arr)
            if e < best_ece:
                best_ece, best_cap = e, float(cap)
        self.max_prob = float(best_cap)
        print(f"  Prob-cap (ECE-opt on {label}): max_prob={self.max_prob:.2f}  ECE={best_ece:.4f}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X.reindex(columns=self.feature_cols, fill_value=0).fillna(0)
        X_rating = Xf.values[:, self._rating_idx] if self._rating_idx else None
        raw = self._blend_predict(Xf.values, X_rating)
        iso = np.clip(self.calibrator.predict(raw), 1e-6, 1 - 1e-6)
        logit = np.log(iso / (1 - iso))
        t = getattr(self, "temperature", 1.0)
        p = 1.0 / (1.0 + np.exp(-(t * (self.platt_a * logit + self.platt_b))))
        cap = getattr(self, "max_prob", 1.0)
        return _soft_cap(p, cap)

    def predict_symmetric(self, X_a: pd.DataFrame, X_b: pd.DataFrame) -> np.ndarray:
        p_a = self.predict_proba(X_a)
        p_b = self.predict_proba(X_b)
        return np.array([inference_average(pa, pb) for pa, pb in zip(p_a, p_b)])

    def save(self, path: Path, gitsha: str = "latest") -> Path:
        out_path = path / f"winner_ensemble_{gitsha}.joblib"
        joblib.dump(self, out_path, compress=3)
        return out_path

    @staticmethod
    def load(path: Path) -> "WinnerModel":
        return joblib.load(path)


# Backward-compatibility alias
WinnerEnsemble = WinnerModel
