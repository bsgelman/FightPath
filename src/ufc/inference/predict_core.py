"""Core prediction logic: pure computation returning FightPrediction dataclasses.

predict.py imports from here so the UI and CLI share the exact same
method-conditional prop-CDF path.

Winner model: v9 diverse ensemble (3×LGBM + 2×CatBoost + 2×XGB + LogisticRegression),
SLSQP OOF blend, isotonic + Platt calibration, max_prob clip. Pre-UFC priors seed
Elo/Glicko/TrueSkill for debut fighters. Test accuracy 65.7% (2025–2026, n=737).
"""
from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model / data loading helpers
# ---------------------------------------------------------------------------

def _find_latest_model(pattern: str) -> Path | None:
    """Look in outputs/models/prod/ first, fall back to outputs/models/."""
    from ufc.io import paths
    prod_dir = paths.outputs_models_prod()
    prod_files = sorted(prod_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if prod_files:
        return prod_files[-1]
    eval_dir = paths.outputs_models()
    eval_files = sorted(eval_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return eval_files[-1] if eval_files else None


def load_models(verbose: bool = True) -> dict:
    """Load all available trained models. Returns dict keyed by model name."""
    from ufc.models.winner import WinnerModel
    from ufc.models.method import MethodClassifier
    from ufc.models.props_count import HurdleCountModel
    from ufc.models.props_duration import DurationModel

    models = {}

    winner_path = _find_latest_model("winner_ensemble_*.joblib")
    if winner_path:
        if verbose:
            print(f"  Loading winner model: {winner_path.name}")
        models["winner"] = WinnerModel.load(winner_path)

    method_path = _find_latest_model("method_clf_*.joblib")
    if method_path:
        if verbose:
            print(f"  Loading method model: {method_path.name}")
        models["method"] = MethodClassifier.load(method_path)

    ss_path = _find_latest_model("props_sig_strikes_*.joblib")
    if ss_path:
        if verbose:
            print(f"  Loading sig strikes model: {ss_path.name}")
        models["sig_strikes"] = HurdleCountModel.load(ss_path)

    td_path = _find_latest_model("props_takedowns_*.joblib")
    if td_path:
        if verbose:
            print(f"  Loading takedowns model: {td_path.name}")
        models["takedowns"] = HurdleCountModel.load(td_path)

    r1_path = _find_latest_model("props_r1_sig_strikes_*.joblib")
    if r1_path:
        if verbose:
            print(f"  Loading R1 sig strikes model: {r1_path.name}")
        models["r1_sig_strikes"] = HurdleCountModel.load(r1_path)

    for _key, _pattern in [
        ("knockdowns",       "props_knockdowns_*.joblib"),
        ("sub_attempts",     "props_sub_attempts_*.joblib"),
        ("r1_takedowns",     "props_r1_takedowns_*.joblib"),
        ("body_sig_strikes", "props_body_sig_strikes_*.joblib"),
        ("leg_sig_strikes",  "props_leg_sig_strikes_*.joblib"),
        ("ctrl_time",        "props_ctrl_time_*.joblib"),
    ]:
        _p = _find_latest_model(_pattern)
        if _p:
            if verbose:
                print(f"  Loading {_key}: {_p.name}")
            models[_key] = HurdleCountModel.load(_p)

    dur_path = _find_latest_model("props_duration_*.joblib")
    if dur_path:
        if verbose:
            print(f"  Loading duration model: {dur_path.name}")
        models["duration"] = DurationModel.load(dur_path)

    return models


def load_reference_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (fighters_df, pre_fight_state, ref_history_df) from processed parquets."""
    from ufc.io import paths, parquet
    from ufc.inference.ref_history import build_ref_history
    fighters_df = parquet.read(paths.interim("fighters"))
    pre_fight_state = parquet.read(paths.processed("pre_fight_state"))
    ref_history_df = build_ref_history()
    return fighters_df, pre_fight_state, ref_history_df


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FightPrediction:
    red_name: str           # canonical fighter name
    blue_name: str          # canonical fighter name
    red_id: str
    blue_id: str
    rounds: int
    is_title: bool
    event_date: date
    prob_red: float
    prob_blue: float        # always 1 - prob_red
    method_probs: dict      # {"KO/TKO": float, "SUB": float, "DEC": float}
    # Duration CDFs
    dur_cdf: object | None              # raw DEC-mode CDF (used by count-prop MC)
    dur_cdfs_by_method: dict | None     # {"KO/TKO", "SUB", "DEC" -> DurationCDF}
    display_dur_cdf: object | None      # method-marginal blend — use this in the UI
    # Prop count CDFs (RateXDurationCDF)
    ss_cdf_red: object | None
    ss_cdf_blue: object | None
    td_cdf_red: object | None
    td_cdf_blue: object | None
    r1_cdf_red: object | None
    r1_cdf_blue: object | None
    # New count CDFs (all default None so old callers don't break)
    kd_cdf_red: object | None = None
    kd_cdf_blue: object | None = None
    sub_att_cdf_red: object | None = None
    sub_att_cdf_blue: object | None = None
    r1_td_cdf_red: object | None = None
    r1_td_cdf_blue: object | None = None
    body_cdf_red: object | None = None
    body_cdf_blue: object | None = None
    leg_cdf_red: object | None = None
    leg_cdf_blue: object | None = None
    ctrl_cdf_red: object | None = None
    ctrl_cdf_blue: object | None = None
    ss_combo_cdf: object | None = None
    # Joint MC simulation (None when run_simulation=False)
    sim_samples: dict | None = None
    # Data-volume metadata (Fix B — thin-data flag)
    n_fights_red: int = 0   # UFC fights in pre_fight_state for red corner
    n_fights_blue: int = 0  # UFC fights in pre_fight_state for blue corner
    low_data: bool = False  # True when either fighter has < LOW_DATA_THRESHOLD fights
    # UFC career record (wins, losses, draws) — attached by service after predict_fight
    record_red:  tuple | None = None
    record_blue: tuple | None = None
    # Weight class actually used for the bout (explicit override or inferred from history)
    weight_class: str = ""
    # Per-fight method resolution score: |P(finish) - era_finish_rate|, 0=base-rate, 1=max
    method_edge_score: float = 0.0
    has_method_edge: bool = False
    # Top per-fight P(win) drivers (grouped feature attribution, red perspective).
    # [{key, label, accent, delta, magnitude, favors}], sorted by magnitude.
    winner_drivers: list | None = None


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict_fight(
    red_name: str,
    blue_name: str,
    rounds: int,
    is_title: bool,
    event_date: date,
    models: dict,
    fighters_df: pd.DataFrame,
    pre_fight_state: pd.DataFrame,
    n_simulate: int = 50000,
    location: str = "",
    referee: str = "",
    ref_history_df: pd.DataFrame | None = None,
    run_simulation: bool = True,
    verbose: bool = False,
    weight_class: str | None = None,
) -> FightPrediction:
    """Run the full prediction pipeline for one matchup.

    Returns a FightPrediction dataclass with all CDF objects populated.
    Raises ValueError on unknown fighter names (caller decides how to handle).

    Prop-count CDFs use the method-conditional path. Do not "simplify" the
    RateHurdleCountModel branches — they override method_log_rate_adj to None
    per-call (via predict_cdf's method_log_rate_adj_override, never by mutating
    the shared model instance) and pass duration_cdfs_by_method to prevent
    DEC-mode count inflation (PIT-KS 0.243).
    """
    from ufc.inference.matchup import find_fighter, build_matchup_features
    from ufc.inference.simulator import simulate

    # ── Resolve fighter IDs ────────────────────────────────────────────────
    red_id, red_canonical = find_fighter(red_name, fighters_df)
    blue_id, blue_canonical = find_fighter(blue_name, fighters_df)

    # ── Build feature vectors for both perspectives ────────────────────────
    feat = build_matchup_features(
        red_id, blue_id, event_date, rounds, is_title,
        pre_fight_state, fighters_df,
        location=location, referee=referee,
        ref_history_df=ref_history_df,
        weight_class=weight_class,
    )
    feat_flip = build_matchup_features(
        blue_id, red_id, event_date, rounds, is_title,
        pre_fight_state, fighters_df,
        location=location, referee=referee,
        ref_history_df=ref_history_df,
        weight_class=weight_class,
    )

    # ── Winner prediction ──────────────────────────────────────────────────
    prob_red = 0.5
    if "winner" in models:
        from ufc.training.symmetrize import inference_average
        # Single batched call (2 rows) avoids double CatBoost/XGB Pool overhead
        feat_both = pd.concat([feat, feat_flip], ignore_index=True)
        probs = models["winner"].predict_proba(feat_both)
        prob_red = inference_average(float(probs[0]), float(probs[1]))

    # ── Per-fight P(win) driver attribution (read-only, never fatal) ───────
    winner_drivers_list = None
    if "winner" in models:
        try:
            from ufc.inference.attribution import winner_drivers
            winner_drivers_list = winner_drivers(models["winner"], feat, feat_flip, top_n=5)
        except Exception:
            winner_drivers_list = None

    from ufc.inference.wc_temperature import apply_wc_temperature
    _red_pfs = pre_fight_state[pre_fight_state["fighter_id"] == red_id]
    _blue_pfs = pre_fight_state[pre_fight_state["fighter_id"] == blue_id]
    inferred_wc = str(_red_pfs.iloc[0]["weight_class"]) if (
        len(_red_pfs) > 0 and "weight_class" in _red_pfs.columns
    ) else ""
    wc = weight_class or inferred_wc
    prob_red = apply_wc_temperature(prob_red, wc)

    # ── Data-volume shrinkage (Fix B) ─────────────────────────────────────
    # Fighters with few UFC fights → high feature uncertainty → shrink toward 0.5.
    # Threshold: 4 fights for full confidence; debutants (0 fights) → 0.5 exactly.
    # fights_career column in pre_fight_state counts completed UFC bouts.
    _LOW_DATA_THRESHOLD = 4
    def _n_fights(pfs_rows: "pd.DataFrame") -> int:
        if len(pfs_rows) == 0:
            return 0
        # Sentinel state row: fights_career already counts ALL of the fighter's
        # completed UFC bouts (the row is not tied to a specific upcoming fight),
        # so no +1 correction is needed here.
        v = pfs_rows.iloc[0].get("fights_career")
        base = int(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else 0
        return base
    n_fights_red  = _n_fights(_red_pfs)
    n_fights_blue = _n_fights(_blue_pfs)
    _factor_red  = min(1.0, n_fights_red  / _LOW_DATA_THRESHOLD)
    _factor_blue = min(1.0, n_fights_blue / _LOW_DATA_THRESHOLD)
    _shrink_factor = min(_factor_red, _factor_blue)
    if _shrink_factor < 1.0:
        prob_red = prob_red * _shrink_factor + 0.5 * (1.0 - _shrink_factor)
    _low_data = (n_fights_red < _LOW_DATA_THRESHOLD) or (n_fights_blue < _LOW_DATA_THRESHOLD)

    # ── Method prediction (symmetrized) ───────────────────────────────────
    method_probs = {"KO/TKO": 0.33, "SUB": 0.17, "DEC": 0.50}
    if "method" in models:
        mp_a = models["method"].predict_proba_dict(feat)
        mp_b = models["method"].predict_proba_dict(feat_flip)
        method_probs = {k: float((mp_a[k][0] + mp_b[k][0]) / 2.0) for k in mp_a}

    # ── Method edge score (distance from era base-rate) ───────────────────
    _era_finish = (
        float(models["method"].class_priors[0] + models["method"].class_priors[1])
        if "method" in models else 0.505
    )
    _finish_prob = method_probs.get("KO/TKO", 0.0) + method_probs.get("SUB", 0.0)
    _method_edge_score = round(abs(_finish_prob - _era_finish), 3)
    _has_method_edge = _method_edge_score > 0.08

    # ── Duration CDFs ──────────────────────────────────────────────────────
    ss_cdf_red = ss_cdf_blue = td_cdf_red = td_cdf_blue = None
    r1_cdf_red = r1_cdf_blue = dur_cdf = None
    dur_cdfs_by_method = None
    display_dur_cdf = None

    if "duration" in models:
        dur_model = models["duration"]
        dur_cdf = dur_model.predict_cdf(feat)[0]
        if "method_ko" in getattr(dur_model, "feature_cols", []):
            dur_cdfs_by_method = {
                "KO/TKO": dur_model.predict_cdf(feat, method_override="KO/TKO")[0],
                "SUB":    dur_model.predict_cdf(feat, method_override="SUB")[0],
                "DEC":    dur_model.predict_cdf(feat, method_override="DEC")[0],
            }

        # v8.24: method-marginal duration CDF for display — true mixture, not blend.
        # survival(t) = Σ p_m * c_m.survival(t) (each method CDF evaluated separately,
        # then weighted).  The old quantile-value average was mathematically invalid.
        display_dur_cdf = dur_cdf  # fallback when method CDFs unavailable
        if dur_cdfs_by_method is not None:
            from ufc.models.props_duration import MixtureDurationCDF as _MixCDF
            display_dur_cdf = _MixCDF(
                cdfs_by_method=dur_cdfs_by_method,
                method_probs=method_probs,
                scheduled_sec=dur_cdf._scheduled_sec,
            )

    # ── Sig strikes CDFs ───────────────────────────────────────────────────
    if "sig_strikes" in models:
        from ufc.models.props_count import RateHurdleCountModel as _RHCM
        ss_model = models["sig_strikes"]
        if isinstance(ss_model, _RHCM):
            # v8.13: method-conditional path (PIT-KS 0.027, matches Gate B).
            # Nulling method_log_rate_adj is required — durations carry the method signal.
            _ss_mp = np.array([[
                method_probs.get("KO/TKO", 0.33),
                method_probs.get("SUB",    0.17),
                method_probs.get("DEC",    0.50),
            ]])
            _ss_dur = [dur_cdf] if dur_cdf is not None else None
            _ss_method_cdfs = (
                {mn: [cdf] for mn, cdf in dur_cdfs_by_method.items()}
                if dur_cdfs_by_method is not None else None
            )
            ss_cdf_red  = ss_model.predict_cdf(
                    feat,      duration_cdfs=_ss_dur,
                    method_proba=_ss_mp, duration_cdfs_by_method=_ss_method_cdfs,
                )[0]
            ss_cdf_blue = ss_model.predict_cdf(
                    feat_flip, duration_cdfs=_ss_dur,
                    method_proba=_ss_mp, duration_cdfs_by_method=_ss_method_cdfs,
                )[0]
        else:
            ss_cdf_red  = ss_model.predict_cdf(feat)[0]
            ss_cdf_blue = ss_model.predict_cdf(feat_flip)[0]

    # ── Takedowns CDFs ─────────────────────────────────────────────────────
    if "takedowns" in models:
        from ufc.models.props_count import RateHurdleCountModel as _RHCM
        td_model = models["takedowns"]
        if isinstance(td_model, _RHCM):
            # v8.11/v8.13: method-conditional path.
            # Never use cond_hurdle on the takedowns marginal — it breaks the 5rd gate.
            _td_mp = np.array([[
                method_probs.get("KO/TKO", 0.33),
                method_probs.get("SUB",    0.17),
                method_probs.get("DEC",    0.50),
            ]])
            _td_dur_cdfs = [dur_cdf] if dur_cdf is not None else None
            _td_method_cdfs = (
                {mn: [cdf] for mn, cdf in dur_cdfs_by_method.items()}
                if dur_cdfs_by_method is not None else None
            )
            td_cdf_red  = td_model.predict_cdf(
                feat,
                duration_cdfs=_td_dur_cdfs,
                method_proba=_td_mp,
                duration_cdfs_by_method=_td_method_cdfs,
                use_sub_count_head=True,
                use_cond_hurdle=False,
                method_log_rate_adj_override=None,
            )[0]
            td_cdf_blue = td_model.predict_cdf(
                feat_flip,
                duration_cdfs=_td_dur_cdfs,
                method_proba=_td_mp,
                duration_cdfs_by_method=_td_method_cdfs,
                use_sub_count_head=True,
                use_cond_hurdle=False,
                method_log_rate_adj_override=None,
            )[0]
        else:
            td_cdf_red  = td_model.predict_cdf(feat)[0]
            td_cdf_blue = td_model.predict_cdf(feat_flip)[0]

    # ── R1 sig strikes CDFs ────────────────────────────────────────────────
    if "r1_sig_strikes" in models:
        from ufc.models.props_count import RateHurdleCountModel as _RHCM
        r1_model = models["r1_sig_strikes"]
        if isinstance(r1_model, _RHCM):
            # v8.13: method-conditional path; 5-min ceiling dominates for r1.
            _r1_mp = np.array([[
                method_probs.get("KO/TKO", 0.33),
                method_probs.get("SUB",    0.17),
                method_probs.get("DEC",    0.50),
            ]])
            _r1_method_cdfs = (
                {mn: [cdf] for mn, cdf in dur_cdfs_by_method.items()}
                if dur_cdfs_by_method is not None else None
            )
            # hurdle_floor: thin-data fighters with no round-level history get
            # r1_sig_str_ctd=0, which the hurdle reads as "lands 0 R1 sig strikes"
            # → degenerate P(under)=100%. Floor P(>0) to a near-certainty (empirical
            # marginal is 0.965). Serving-only — Gate B calls predict_cdf without it.
            r1_cdf_red  = r1_model.predict_cdf(
                feat,      duration_cdfs=[dur_cdf],
                duration_cdfs_by_method=_r1_method_cdfs,
                active_minutes_ceiling=5.0, apply_burst=False,
                method_proba=_r1_mp, use_finish_head=True,
                hurdle_floor=0.90,
            )[0]
            r1_cdf_blue = r1_model.predict_cdf(
                feat_flip, duration_cdfs=[dur_cdf],
                duration_cdfs_by_method=_r1_method_cdfs,
                active_minutes_ceiling=5.0, apply_burst=False,
                method_proba=_r1_mp, use_finish_head=True,
                hurdle_floor=0.90,
            )[0]
        else:
            r1_cdf_red  = r1_model.predict_cdf(feat)[0]
            r1_cdf_blue = r1_model.predict_cdf(feat_flip)[0]

    # ── New count prop CDFs ────────────────────────────────────────────────
    kd_cdf_red = kd_cdf_blue = None
    sub_att_cdf_red = sub_att_cdf_blue = None
    r1_td_cdf_red = r1_td_cdf_blue = None
    body_cdf_red = body_cdf_blue = None
    leg_cdf_red = leg_cdf_blue = None
    ctrl_cdf_red = ctrl_cdf_blue = None
    ss_combo_cdf = None

    _new_mp = np.array([[
        method_probs.get("KO/TKO", 0.33),
        method_probs.get("SUB",    0.17),
        method_probs.get("DEC",    0.50),
    ]])
    _new_dur = [dur_cdf] if dur_cdf is not None else None
    _new_method_cdfs = (
        {mn: [cdf] for mn, cdf in dur_cdfs_by_method.items()}
        if dur_cdfs_by_method is not None else None
    )

    def _predict_pair_td_style(model_key: str, use_sub_head: bool = False, use_adj: bool = False):
        """Takedowns-style pair prediction (method-conditional). use_adj=True keeps method_log_rate_adj."""
        from ufc.models.props_count import RateHurdleCountModel as _R
        m = models.get(model_key)
        if m is None or not isinstance(m, _R):
            return None, None
        # use_adj=True: omit the override kwarg entirely so predict_cdf falls back
        # to its own _UNSET default (i.e. self.method_log_rate_adj). use_adj=False:
        # explicitly override to None for this call only — no instance mutation.
        _adj_kw = {} if use_adj else {"method_log_rate_adj_override": None}
        cdf_r = m.predict_cdf(
            feat, duration_cdfs=_new_dur, method_proba=_new_mp,
            duration_cdfs_by_method=_new_method_cdfs,
            use_sub_count_head=use_sub_head, use_cond_hurdle=False,
            **_adj_kw,
        )[0]
        cdf_b = m.predict_cdf(
            feat_flip, duration_cdfs=_new_dur, method_proba=_new_mp,
            duration_cdfs_by_method=_new_method_cdfs,
            use_sub_count_head=use_sub_head, use_cond_hurdle=False,
            **_adj_kw,
        )[0]
        return cdf_r, cdf_b

    def _predict_pair_ss_style(model_key: str, use_adj: bool = False):
        """Sig-strikes-style pair (method-conditional). use_adj=True keeps method_log_rate_adj."""
        from ufc.models.props_count import RateHurdleCountModel as _R
        m = models.get(model_key)
        if m is None or not isinstance(m, _R):
            return None, None
        _adj_kw = {} if use_adj else {"method_log_rate_adj_override": None}
        cdf_r = m.predict_cdf(
            feat, duration_cdfs=_new_dur, method_proba=_new_mp,
            duration_cdfs_by_method=_new_method_cdfs,
            **_adj_kw,
        )[0]
        cdf_b = m.predict_cdf(
            feat_flip, duration_cdfs=_new_dur, method_proba=_new_mp,
            duration_cdfs_by_method=_new_method_cdfs,
            **_adj_kw,
        )[0]
        return cdf_r, cdf_b

    def _predict_pair_share_style(model_key: str):
        """ControlShareModel pair — falls back to td-style if model not yet retrained."""
        from ufc.models.props_count import ControlShareModel as _CS, RateHurdleCountModel as _R
        m = models.get(model_key)
        if m is None:
            return None, None
        if isinstance(m, _CS):
            cdf_r = m.predict_cdf(
                feat, duration_cdfs=_new_dur, method_proba=_new_mp,
                duration_cdfs_by_method=_new_method_cdfs,
            )[0]
            cdf_b = m.predict_cdf(
                feat_flip, duration_cdfs=_new_dur, method_proba=_new_mp,
                duration_cdfs_by_method=_new_method_cdfs,
            )[0]
        elif isinstance(m, _R):
            cdf_r = m.predict_cdf(
                feat, duration_cdfs=_new_dur, method_proba=_new_mp,
                duration_cdfs_by_method=_new_method_cdfs, use_cond_hurdle=False,
                method_log_rate_adj_override=None,
            )[0]
            cdf_b = m.predict_cdf(
                feat_flip, duration_cdfs=_new_dur, method_proba=_new_mp,
                duration_cdfs_by_method=_new_method_cdfs, use_cond_hurdle=False,
                method_log_rate_adj_override=None,
            )[0]
        else:
            return None, None
        return cdf_r, cdf_b

    kd_cdf_red,      kd_cdf_blue      = _predict_pair_td_style("knockdowns",       use_sub_head=False)
    sub_att_cdf_red, sub_att_cdf_blue = _predict_pair_td_style("sub_attempts",      use_sub_head=True)
    body_cdf_red,    body_cdf_blue    = _predict_pair_ss_style("body_sig_strikes", use_adj=True)
    leg_cdf_red,     leg_cdf_blue     = _predict_pair_ss_style("leg_sig_strikes")
    ctrl_cdf_red,    ctrl_cdf_blue    = _predict_pair_share_style("ctrl_time")

    # KO/TKO almost always involves a knockdown; floor P(KD≥1) per fighter to
    # P(KO/TKO) × P(fighter wins) × 0.85.  0.85 accounts for stoppages without
    # a scored knockdown (ground-and-pound TKOs, doctor/corner stoppages).
    # The prop and method models are independent — this post-hoc constraint
    # prevents coherence violations like P(KD)=19% vs P(KO)=32%.
    _p_ko = method_probs.get("KO/TKO", 0.0)
    if kd_cdf_red is not None and _p_ko > 0:
        _floor_r = _p_ko * float(prob_red) * 0.85
        kd_cdf_red._p_zero = min(kd_cdf_red._p_zero, 1.0 - _floor_r)
    if kd_cdf_blue is not None and _p_ko > 0:
        _floor_b = _p_ko * (1.0 - float(prob_red)) * 0.85
        kd_cdf_blue._p_zero = min(kd_cdf_blue._p_zero, 1.0 - _floor_b)

    if "r1_takedowns" in models:
        from ufc.models.props_count import RateHurdleCountModel as _R
        _r1td_model = models["r1_takedowns"]
        if isinstance(_r1td_model, _R):
            r1_td_cdf_red  = _r1td_model.predict_cdf(
                feat,      duration_cdfs=_new_dur, duration_cdfs_by_method=_new_method_cdfs,
                active_minutes_ceiling=5.0, apply_burst=False,
                method_proba=_new_mp, use_finish_head=True,
            )[0]
            r1_td_cdf_blue = _r1td_model.predict_cdf(
                feat_flip, duration_cdfs=_new_dur, duration_cdfs_by_method=_new_method_cdfs,
                active_minutes_ceiling=5.0, apply_burst=False,
                method_proba=_new_mp, use_finish_head=True,
            )[0]

    if "sig_strikes" in models and dur_cdf is not None:
        from ufc.models.props_count import predict_combined_count_cdf as _pcc
        from ufc.models.props_count import RateHurdleCountModel as _R
        _ss_m = models["sig_strikes"]
        if isinstance(_ss_m, _R):
            ss_combo_cdf = _pcc(
                _ss_m, feat, feat_flip,
                duration_cdfs=_new_dur,
                method_proba=_new_mp,
                duration_cdfs_by_method=_new_method_cdfs,
            )

    # ── Monte Carlo joint simulation ───────────────────────────────────────
    sim_samples = None
    if run_simulation:
        sig_str_method_adj = getattr(models.get("sig_strikes"), "method_log_rate_adj", None)
        td_method_adj = getattr(models.get("takedowns"), "method_log_rate_adj", None)
        if verbose:
            print(f"\n  Running {n_simulate:,} Monte Carlo simulations...")
        sim_samples = simulate(
            winner_prob=prob_red,
            method_probs=method_probs,
            duration_cdf=dur_cdf,
            sig_str_cdf_a=ss_cdf_red,
            sig_str_cdf_b=ss_cdf_blue,
            td_cdf_a=td_cdf_red,
            td_cdf_b=td_cdf_blue,
            scheduled_rounds=rounds,
            n_samples=n_simulate,
            fighter_id_a=red_id,
            fighter_id_b=blue_id,
            duration_cdfs_by_method=dur_cdfs_by_method,
            sig_str_method_adj=sig_str_method_adj,
            td_method_adj=td_method_adj,
        )

    return FightPrediction(
        red_name=red_canonical,
        blue_name=blue_canonical,
        red_id=red_id,
        blue_id=blue_id,
        rounds=rounds,
        is_title=is_title,
        event_date=event_date,
        prob_red=prob_red,
        prob_blue=1.0 - prob_red,
        method_probs=method_probs,
        n_fights_red=n_fights_red,
        n_fights_blue=n_fights_blue,
        low_data=_low_data,
        dur_cdf=dur_cdf,
        dur_cdfs_by_method=dur_cdfs_by_method,
        display_dur_cdf=display_dur_cdf,
        ss_cdf_red=ss_cdf_red,
        ss_cdf_blue=ss_cdf_blue,
        td_cdf_red=td_cdf_red,
        td_cdf_blue=td_cdf_blue,
        r1_cdf_red=r1_cdf_red,
        r1_cdf_blue=r1_cdf_blue,
        kd_cdf_red=kd_cdf_red,
        kd_cdf_blue=kd_cdf_blue,
        sub_att_cdf_red=sub_att_cdf_red,
        sub_att_cdf_blue=sub_att_cdf_blue,
        r1_td_cdf_red=r1_td_cdf_red,
        r1_td_cdf_blue=r1_td_cdf_blue,
        body_cdf_red=body_cdf_red,
        body_cdf_blue=body_cdf_blue,
        leg_cdf_red=leg_cdf_red,
        leg_cdf_blue=leg_cdf_blue,
        ctrl_cdf_red=ctrl_cdf_red,
        ctrl_cdf_blue=ctrl_cdf_blue,
        ss_combo_cdf=ss_combo_cdf,
        sim_samples=sim_samples,
        weight_class=str(wc) if wc else "",
        method_edge_score=_method_edge_score,
        has_method_edge=_has_method_edge,
        winner_drivers=winner_drivers_list,
    )
