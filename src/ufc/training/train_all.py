"""End-to-end training orchestrator — v5-baseline.

Trains: WinnerModel (single LGBM + CalibratedClassifierCV),
        MethodClassifier (temperature scaling + prior shrinkage),
        HurdleCountModel × 3 (sig_strikes, takedowns, r1_sig_strikes),
        DurationModel (hurdle: CalibratedClassifierCV P(dec) + LGBM quantile finishes).

No Optuna. No CV folds. No CatBoost/XGBoost. No Weibull AFT.
No method symmetrization (method is fight-level, not fighter-specific).
"""
from __future__ import annotations

import subprocess
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.training.symmetrize import symmetrize
from ufc.training.tune_props import get_prop_feature_cols
from ufc.models.winner import WinnerModel, get_winner_feature_cols
from ufc.models.method import MethodClassifier, METHOD_CLASSES
from ufc.models.props_count import HurdleCountModel, RateHurdleCountModel, ControlShareModel
from ufc.training.prop_targets import PROP_TARGET_SPECS
from ufc.models.props_duration import DurationModel
from ufc.training.feature_pruning import (
    compute_dead_features_from_importances, save_dead_features,
)
from ufc.training.recency import search_halflife_method, search_halflife_winner


def _fit_rate_calib_factor(count_model: "RateHurdleCountModel",
                           duration_model: "DurationModel",
                           X_val: "pd.DataFrame",
                           y_val: "pd.Series",
                           method_proba: "np.ndarray") -> float:
    """Val-anchored multiplicative rate factor on the METHOD-MARGINAL forecast.

    Integrates the count against method-conditional durations weighted by predicted
    method probs, with the per-method flat rate-adj neutralised. This matches the
    production forecast path (predict.py uses method-conditional durations, not the
    realized method), so the factor corrects exactly what production over-predicts.

    Previously passed X_val[prop_cols] which stripped `method`, causing the duration
    model to default every fight to DEC-mode → ~40% inflated durations → factor 0.684.
    Now passes the full frame and explicitly marginalises over predicted method probs.
    """
    dur_cdfs = duration_model.predict_cdf(X_val, use_boundary_mass=False)
    dur_by_method = {
        m: duration_model.predict_cdf(X_val, method_override=m, use_boundary_mass=False)
        for m in ("KO/TKO", "SUB", "DEC")
    }
    saved_adj = count_model.method_log_rate_adj
    saved_f = count_model.rate_calib_factor
    count_model.method_log_rate_adj = None
    count_model.rate_calib_factor = 1.0
    try:
        cdfs = count_model.predict_cdf(
            X_val, duration_cdfs=dur_cdfs,
            method_proba=method_proba, duration_cdfs_by_method=dur_by_method,
        )
        pred_mean = float(np.mean([c._samples.mean() for c in cdfs]))
    finally:
        count_model.method_log_rate_adj = saved_adj
        count_model.rate_calib_factor = saved_f
    actual_mean = float(np.asarray(y_val.fillna(0), dtype=float).mean())
    return actual_mean / pred_mean if pred_mean > 0 else 1.0


def _gitsha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.root(), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d")


def train(model_dir: "Path | None" = None, prod_mode: bool = False) -> dict:
    """Full training pipeline. Returns dict of saved artifact paths."""
    import os
    import yaml
    gitsha = _gitsha()
    if model_dir is None:
        model_dir = paths.outputs_models()
    model_dir.mkdir(parents=True, exist_ok=True)

    _split_filename = os.environ.get("UFC_SPLIT_CONFIG", "split.yaml")
    _split_cfg = yaml.safe_load((paths.root() / "configs" / _split_filename).read_text())
    _train_end_anchor = _split_cfg["train_end"]

    print("=== v5-baseline Training ===")
    print(f"  gitsha: {gitsha}")

    # ── Load features ─────────────────────────────────────────────────────
    print("\n[1/5] Loading features...")
    winner_df = parquet.read(paths.processed("features_winner"))
    props_df = parquet.read(paths.processed("features_props"))

    winner_df["event_date"] = pd.to_datetime(winner_df["event_date"])
    props_df["event_date"] = pd.to_datetime(props_df["event_date"])

    # Re-apply interaction features so new features added to interactions.py
    # since last feature assembly are included without requiring a full rebuild.
    from ufc.features.interactions import compute_interactions  # noqa: PLC0415
    print("  Re-applying interaction features (picks up v8.2 KO discrimination features)...")
    winner_df = compute_interactions(winner_df)
    props_df = compute_interactions(props_df)

    # ── Winner model ──────────────────────────────────────────────────────
    print("\n[2/5] Training winner model...")
    splits = get_splits(winner_df)
    train_mask = splits["train"]
    # v9: val-A for halflife search + early stopping; val-B for Platt refinement
    val_a_mask = splits.get("val_a", splits["val"])
    val_b_mask = splits.get("val_b", splits["val"])

    feature_cols = get_winner_feature_cols(winner_df)
    print(f"  Feature count: {len(feature_cols)}")

    # Symmetrize training data (winner only — method is NOT symmetrized)
    train_df = winner_df[train_mask].copy()
    train_sym = symmetrize(train_df)
    y_train_sym = train_sym["won_a"].astype(float)
    valid_sym = y_train_sym.notna()
    train_sym = train_sym[valid_sym]
    y_train_sym = y_train_sym[valid_sym]

    val_a_df = winner_df[val_a_mask].copy()
    y_val_a = val_a_df["won_a"].astype(float)
    val_a_df = val_a_df[y_val_a.notna()]
    y_val_a = y_val_a[y_val_a.notna()]

    val_b_df = winner_df[val_b_mask].copy()
    y_val_b = val_b_df["won_a"].astype(float)
    val_b_df = val_b_df[y_val_b.notna()]
    y_val_b = y_val_b[y_val_b.notna()]

    print(f"  Train (sym): {len(train_sym)}  Val-A: {len(val_a_df)}  Val-B: {len(val_b_df)}")

    # v9: halflife search on val-A only (not val-B, which is reserved for Platt)
    best_h_winner, sw_winner = search_halflife_winner(
        train_sym, y_train_sym, val_a_df, y_val_a,
        feature_cols=feature_cols,
        train_dates=train_sym["event_date"],
        grid=[730, 1095, 1460, 1825, None],
        anchor=_train_end_anchor,
        ece_cap=0.04,
        brier_floor_margin=0.001,
        verbose=True,
        temporal_oof=prod_mode,
    )
    print(f"  Winner halflife selected: {best_h_winner}")

    winner_model = WinnerModel()
    winner_model.fit(train_sym, y_train_sym, val_a_df, y_val_a, feature_cols,
                     sample_weight=sw_winner,
                     X_val_platt=val_b_df, y_val_platt=y_val_b,
                     temporal_oof=prod_mode,
                     train_dates=train_sym["event_date"])
    winner_path = winner_model.save(model_dir, gitsha)
    print(f"  Saved winner model: {winner_path.name}")

    # Export zero-importance features for next training run (from LGBM)
    try:
        lgbm_model = getattr(winner_model, "lgbm", None)
        if lgbm_model is not None and hasattr(lgbm_model, "feature_importances_"):
            imp = np.array(lgbm_model.feature_importances_, dtype=float)
            dead = {f for f, v in zip(feature_cols, imp) if abs(v) <= 1e-6}
            if dead:
                dead_path = save_dead_features(dead, "winner")
                print(f"  Exported {len(dead)} winner dead features -> {dead_path.name}")
    except Exception as e:
        print(f"  [warn] dead feature export: {e}")

    # ── Method classifier ─────────────────────────────────────────────────
    print("\n[3/5] Training method classifier...")
    prop_feature_cols = get_prop_feature_cols(props_df, model_name="method")
    props_splits = get_splits(props_df)
    p_train = props_df[props_splits["train"]].dropna(subset=["method"])
    p_val = props_df[props_splits["val"]].dropna(subset=["method"])

    # DQ/NC are corrupt labels for method prediction — drop them
    valid_methods = ["KO/TKO", "SUB", "U-DEC", "S-DEC", "M-DEC"]
    p_train = p_train[p_train["method"].isin(valid_methods)].copy()
    p_val = p_val[p_val["method"].isin(valid_methods)].copy()
    # Freak-injury outcomes are corrupt method labels too (a 26s arm dislocation
    # is not evidence of KO ability) — exclude from fit AND calibration folds.
    if "injury_freak" in p_train.columns:
        n0 = len(p_train) + len(p_val)
        p_train = p_train[~p_train["injury_freak"].fillna(False).astype(bool)].copy()
        p_val = p_val[~p_val["injury_freak"].fillna(False).astype(bool)].copy()
        print(f"  Dropped {n0 - len(p_train) - len(p_val)} freak-injury method rows")

    # NOTE: method is NOT symmetrized — method is a fight-level outcome.
    # Swapping A/B perspective doesn't change the fight method label.

    # V7.1: val-based halflife search with per-class ECE guardrail (≤0.05).
    # Step 8 reverted because uniform halflife=1095d degraded ECE; the search
    # finds the best halflife subject to the ECE cap, or falls back to uniform.
    # v8.25: re-add specialist/KO-matchup features excluded from count/duration models
    # but designed for winner/method use (tune_props.py:31-44 comments say "method/winner only").
    # v8.26: removed era_ko/sub_share features — they dominated importances (45 vs 5-15 for
    # fighter-specific signals), anchoring every fight to the era KO/SUB mean and suppressing
    # individual specialist discrimination. Fighter rolling rates already capture era drift implicitly.
    _method_extras = [c for c in [
        # Specialist finish amplifiers — strongest finish discriminators (interactions.py:116-140)
        "ko_specialist_idx_a", "ko_specialist_idx_b",
        "ko_specialist_x_weakness_a", "ko_specialist_x_weakness_b",
        "ko_specialist_x_chin_a", "ko_specialist_x_chin_b",
        "sub_specialist_idx_a", "sub_specialist_idx_b",
        "sub_specialist_x_weakness_a", "sub_specialist_x_weakness_b",
        "finish_share_a", "finish_share_b",
        "grappling_control_threat_a", "grappling_control_threat_b",
        # KO matchup discrimination features — method/winner only (interactions.py:175-188)
        "ko_matchup_a", "ko_matchup_b",
        "chin_decay_threat_a", "chin_decay_threat_b",
        "layoff_chin_a", "layoff_chin_b",
    ] if c in p_train.columns]
    method_feature_cols = [c for c in prop_feature_cols if c in p_train.columns]
    for _c in _method_extras:
        if _c not in method_feature_cols:
            method_feature_cols.append(_c)
    best_h_method, sw_method = search_halflife_method(
        p_train, p_train["method"],
        p_val,   p_val["method"],
        method_feature_cols=method_feature_cols,
        train_dates=p_train["event_date"],
        grid=[365, 730, 1095, 1460, None],
        anchor=_train_end_anchor,
        verbose=True,
        temporal_oof=prod_mode,
    )
    print(f"  Method halflife selected: {best_h_method}")

    method_clf = MethodClassifier()
    method_clf.fit(
        p_train, p_train["method"],
        p_val, p_val["method"],
        method_feature_cols,
        sample_weight=sw_method,
        train_dates=p_train["event_date"],
        temporal_oof=prod_mode,
    )
    method_path = method_clf.save(model_dir, gitsha)
    print(f"  Saved method model: {method_path.name}")

    # Export method dead features
    try:
        m_dead = compute_dead_features_from_importances(
            {"lgbm": method_clf.model.feature_importances_},
            method_clf.feature_cols,
        )
        if m_dead:
            m_dead_path = save_dead_features(m_dead, "method")
            print(f"  Exported {len(m_dead)} method dead features -> {m_dead_path.name}")
    except Exception as e:
        print(f"  [warn] method dead feature export: {e}")

    # ── Prop count models ─────────────────────────────────────────────────
    print("\n[4/5] Training prop count models...")
    prop_cols_props = get_prop_feature_cols(props_df, model_name="props")
    prop_cols = [c for c in prop_cols_props if c in props_df.columns]

    # v8.22: exclude referee_stoppage_threshold from the rate-head feature set.
    # The rate head's job is "how active is this fighter per minute", not "when
    # does the fight end".  referee_stoppage_threshold (a ref-level stoppage-tendency
    # feature) contaminated the rate model (ranked #1 in TOP30 vs SLpM at #21),
    # suppressing projected sig-strike counts for high-volume fighters.
    # Keep full prop_cols for duration/hurdle-stage (predicting whether count > 0
    # is legitimately influenced by fight-ending context).
    _RATE_EXCLUDE = {"referee_stoppage_threshold"}
    rate_prop_cols = [c for c in prop_cols if c not in _RATE_EXCLUDE]
    print(f"  Rate feature set: {len(rate_prop_cols)} cols "
          f"(excluded {len(prop_cols) - len(rate_prop_cols)}: {_RATE_EXCLUDE & set(prop_cols)})")

    _train_end_dt = pd.to_datetime(_train_end_anchor)

    def _compute_active_minutes(rows: pd.DataFrame, ceiling: float | None) -> np.ndarray:
        if ceiling is not None:
            # R1 ceiling: use actual end_time if fight ended in R1, else full ceiling
            end_round = rows["end_round"].fillna(99).astype(float)
            end_time = rows["end_time_sec"].fillna(ceiling * 60.0).astype(float)
            r1_sec = np.where(end_round == 1, np.clip(end_time, 5.0, ceiling * 60.0), ceiling * 60.0)
            return (r1_sec / 60.0)
        else:
            sched_sec = (rows["scheduled_rounds"].fillna(3).astype(float) * 300.0).clip(lower=1)
            return (rows["total_fight_sec"].fillna(sched_sec) / 60.0).clip(lower=5.0 / 60.0).values

    def _compute_sample_weight(rows: pd.DataFrame, scheme: str | None) -> np.ndarray | None:
        if scheme == "censor" and "total_fight_sec" in rows.columns and "scheduled_rounds" in rows.columns:
            sched_sec = (rows["scheduled_rounds"].fillna(3).astype(float) * 300.0).clip(lower=1)
            return (rows["total_fight_sec"].fillna(sched_sec) / sched_sec).clip(0.1, 1.0).values
        elif scheme == "recency" and "event_date" in rows.columns:
            days_old = (_train_end_dt - pd.to_datetime(rows["event_date"])).dt.days.clip(lower=0).astype(float)
            return np.power(0.5, days_old / 730.0).clip(lower=0.05).values
        return None

    trained_count_models: dict = {}
    for spec in PROP_TARGET_SPECS:
        target, raw_col_a = spec.target, spec.raw_col_a
        if raw_col_a not in props_df.columns:
            print(f"  Skipping {target} — column {raw_col_a} not found")
            continue

        rows_a = props_df[props_splits["train"]].copy().reset_index(drop=True)
        rows_a["_y"] = rows_a[raw_col_a]
        X_tr = rows_a[prop_cols].fillna(0)
        y_tr = rows_a["_y"].fillna(0)

        rows_a_val = props_df[props_splits["val"]].copy().reset_index(drop=True)
        rows_a_val["_y"] = rows_a_val[raw_col_a]
        X_vl = rows_a_val[prop_cols].fillna(0)
        y_vl = rows_a_val["_y"].fillna(0)

        act_min_tr = _compute_active_minutes(rows_a, spec.ceiling)
        act_min_vl = _compute_active_minutes(rows_a_val, spec.ceiling)
        sample_weight = _compute_sample_weight(rows_a, spec.weight)

        if getattr(spec, "model_kind", "rate_hurdle") == "control_share":
            print(f"  Training {target} (ControlShareModel)...")
            total_sec_tr = rows_a["total_fight_sec"].fillna(
                rows_a["scheduled_rounds"].fillna(3) * 300.0
            ).clip(lower=1.0).values
            total_sec_vl = rows_a_val["total_fight_sec"].fillna(
                rows_a_val["scheduled_rounds"].fillna(3) * 300.0
            ).clip(lower=1.0).values
            cm = ControlShareModel(target=target)
            cm.fit(
                rows_a[prop_cols].fillna(0), y_tr, total_sec_tr,
                rows_a_val[prop_cols].fillna(0), y_vl, total_sec_vl,
                rate_prop_cols, sample_weight=sample_weight,
                temporal_oof=prod_mode,
                train_dates=rows_a["event_date"] if "event_date" in rows_a.columns else None,
            )
            if "method" in rows_a.columns:
                cm.fit_method_adjustments(
                    rows_a[prop_cols].fillna(0), y_tr, total_sec_tr,
                    rows_a["method"], sample_weight=sample_weight,
                )
        else:
            print(f"  Training {target} (RateHurdleCountModel, ceiling={spec.ceiling})...")
            cm = RateHurdleCountModel(
                target=target,
                active_minutes_ceiling=spec.ceiling,
                rate_ceiling=spec.rate_ceiling,
            )
            # v8.22: use rate_prop_cols (referee_stoppage_threshold excluded) so the
            # rate head learns fighter pace, not fight-ending timing.
            cm.fit(X_tr, y_tr, act_min_tr,
                   X_vl, y_vl, act_min_vl,
                   rate_prop_cols, sample_weight=sample_weight,
                   temporal_oof=prod_mode,
                   train_dates=rows_a["event_date"] if "event_date" in rows_a.columns else None)
            # Fit learned method-conditional rate adjustments (v8.1).
            if "method" in rows_a.columns:
                cm.fit_method_adjustments(
                    X_tr, y_tr, act_min_tr, rows_a["method"],
                    sample_weight=sample_weight,
                    event_dates_train=rows_a["event_date"] if "event_date" in rows_a.columns else None,
                    X_val=X_vl, y_val=y_vl,
                    active_minutes_val=act_min_vl,
                    method_val=rows_a_val["method"] if "method" in rows_a_val.columns else None,
                    temporal_oof=prod_mode,
                )

        cp = cm.save(model_dir, gitsha)
        print(f"  Saved: {cp.name}")
        trained_count_models[target] = cm

    # ── Duration model ────────────────────────────────────────────────────
    print("\n[5/5] Training fight duration model (hurdle: no Weibull AFT)...")
    dur_train = props_df[props_splits["train"]].dropna(subset=["total_fight_sec"]).copy()
    dur_val = props_df[props_splits["val"]].dropna(subset=["total_fight_sec"]).copy()

    dur_train = dur_train.drop_duplicates(subset=["fight_id"])
    dur_val = dur_val.drop_duplicates(subset=["fight_id"])

    event_obs_train = ~dur_train["method"].isin(["U-DEC", "S-DEC", "M-DEC"])
    event_obs_val = ~dur_val["method"].isin(["U-DEC", "S-DEC", "M-DEC"])

    dur_model = DurationModel()
    dur_cols = [c for c in prop_cols if c in dur_train.columns]
    # v8.27: re-add finish-propensity specialist features for method-specific timing.
    # Excluded from count props (distort KS) but valid for duration where they help
    # KO/SUB quantile models learn early-finish tendencies specific to each method.
    _duration_extras = [c for c in [
        "ko_specialist_idx_a", "ko_specialist_idx_b",
        "sub_specialist_idx_a", "sub_specialist_idx_b",
        "finish_share_a", "finish_share_b",
    ] if c in dur_train.columns and c not in dur_cols]
    dur_cols = dur_cols + _duration_extras
    if _duration_extras:
        print(f"  Duration extras added ({len(_duration_extras)}): {_duration_extras}")
    dur_model.fit(
        dur_train, dur_train["total_fight_sec"], event_obs_train,
        dur_val, dur_val["total_fight_sec"], event_obs_val,
        dur_cols,
        train_dates=dur_train.get("event_date"),
        val_dates=dur_val.get("event_date"),
        train_method=dur_train.get("method"),
        val_method=dur_val.get("method"),
        temporal_oof=prod_mode,
    )
    dp = dur_model.save(model_dir, gitsha)
    print(f"  Saved: {dp.name}")

    # ── v8.13: val-anchored rate calibration factor (all rate_calib targets) ─
    # Computed on the METHOD-MARGINAL forecast (production path) so the factor
    # corrects exactly what predict.py over-predicts, not the realized-method gate.
    # Expected range [0.90, 1.10]; outside range means something changed.
    if prod_mode:
        # Prod tier: val ⊂ train, so the in-sample val mask would score the rate
        # factor on data the models already trained on. Use a temporal holdout
        # carved from the tail of train instead (same pattern as the halflife
        # search / winner-method probe guards above).
        _train_rows = props_df[props_splits["train"]].copy().reset_index(drop=True)
        _calib_dates = pd.to_datetime(_train_rows["event_date"])
        _calib_cutoff = (_calib_dates.max() - pd.DateOffset(months=6)).to_datetime64()
        _calib_mask = (_calib_dates >= _calib_cutoff).values
        if _calib_mask.sum() < 30:
            _calib_mask = np.zeros(len(_train_rows), dtype=bool)
            _calib_mask[np.argsort(_calib_dates.values)[-30:]] = True
        _val_rows_calib = _train_rows[_calib_mask].reset_index(drop=True)
    else:
        _val_rows_calib = props_df[props_splits["val"]].copy().reset_index(drop=True)
    _mp_calib = method_clf.predict_proba_dict(_val_rows_calib)
    _method_proba_calib = np.column_stack([_mp_calib[c] for c in METHOD_CLASSES])
    for spec in PROP_TARGET_SPECS:
        if not spec.rate_calib:
            continue
        if spec.target not in trained_count_models or spec.raw_col_a not in props_df.columns:
            continue
        _cm_rc = trained_count_models[spec.target]
        _y_val_rc = _val_rows_calib[spec.raw_col_a]
        print(f"\n  [rate_calib] Computing {spec.target} rate_calib_factor on val set...")
        _factor = _fit_rate_calib_factor(_cm_rc, dur_model, _val_rows_calib, _y_val_rc, _method_proba_calib)
        print(f"  [rate_calib] {spec.target} rate_calib_factor = {_factor:.4f}  "
              f"(expected [0.90, 1.10])")
        if not (0.90 <= _factor <= 1.10):
            print(f"  [rate_calib] WARNING: factor {_factor:.4f} outside [0.90, 1.10] -- "
                  f"check val set or rate model before shipping")
        _cm_rc.rate_calib_factor = _factor
        _rc_path = _cm_rc.save(model_dir, gitsha)
        print(f"  [rate_calib] Re-saved {spec.target} with factor: {_rc_path.name}")

    # ── Feature-importance diagnostics ────────────────────────────────────
    print("\n[Diagnostics] Generating feature-importance charts...")
    fi_dir = paths.outputs_reports() / "feature_importance"
    try:
        from ufc.evaluation.feature_importance import (
            plot_winner_importance, plot_method_importance,
            plot_count_model_importance, plot_duration_importance,
        )
        plot_winner_importance(winner_model, fi_dir, gitsha)
        plot_method_importance(method_clf, fi_dir, gitsha)
        for _target, _cm in trained_count_models.items():
            plot_count_model_importance(_cm, fi_dir, gitsha, target=_target)
        plot_duration_importance(dur_model, fi_dir, gitsha)
        print(f"  Saved feature-importance charts -> {fi_dir}")
    except Exception as e:
        print(f"  [warn] feature-importance generation failed: {e}")

    print(f"\n=== Training complete. Artifacts in {model_dir} ===")
    return {"gitsha": gitsha, "model_dir": str(model_dir)}
