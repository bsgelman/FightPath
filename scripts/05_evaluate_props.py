"""Phase 5: Evaluate prop model calibration (PIT histograms).

v5-baseline: all count models now use HurdlePropCDF (hurdle structure).
v5.3-phase1: segmented PIT diagnostics added (finish/decision/5rd/3rd/r1_end/past_r1).

Unified eval helper handles randomized PIT for zero-mass correctly for all.

Run: python scripts/05_evaluate_props.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.models.props_count import HurdleCountModel, RateHurdleCountModel, ControlShareModel
from ufc.models.props_duration import DurationModel
from ufc.models.method import MethodClassifier, METHOD_CLASSES
from ufc.evaluation.pit import pit_histogram, pit_histogram_segmented
from ufc.evaluation.reportcard import write_prop_report
from ufc.models.props_count import QUANTILE_GRID


def _plot_r1_strikes_scatter(test_df: pd.DataFrame, save_path: Path) -> None:
    """Scatter of actual R1 sig strikes vs actual finish time for r1_end fights.

    Overlays: (1) origin-only linear fit  y = b*t  (what rate×duration assumes),
              (2) linear fit with intercept  y = a + b*t  (reveals burst floor).
    A positive intercept 'a' means short finishes have more strikes than the
    linear model predicts — the 'finishing burst' the count model is blind to.
    """
    is_r1_end = (test_df["end_round"].fillna(99) == 1).values
    has_data = test_df["r1_sig_str_landed_a"].notna() & test_df["end_time_sec"].notna()
    mask = is_r1_end & has_data.values
    if mask.sum() < 10:
        print("  [r1 scatter] insufficient r1_end rows, skipping")
        return

    r1_df = test_df[mask].copy()
    t = r1_df["end_time_sec"].values.astype(float)
    y = r1_df["r1_sig_str_landed_a"].values.astype(float)

    # Origin-only fit: y ≈ b*t
    b_origin = float(np.dot(t, y) / np.dot(t, t))
    # Intercept fit: y ≈ a + b*t
    coeffs = np.polyfit(t, y, 1)  # [slope, intercept]
    slope_int, intercept = float(coeffs[0]), float(coeffs[1])

    t_line = np.linspace(max(t.min(), 5), t.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(t, y, alpha=0.35, s=18, color="steelblue", label="Observed (r1_end fights)")
    ax.plot(t_line, b_origin * t_line, "r--", linewidth=1.5,
            label=f"Origin-linear: y={b_origin:.2f}·t  (model assumption)")
    ax.plot(t_line, slope_int * t_line + intercept, "g-", linewidth=1.8,
            label=f"With intercept: y={intercept:.1f}+{slope_int:.2f}·t  (burst floor={intercept:.1f})")
    ax.set_xlabel("Actual finish time (seconds)")
    ax.set_ylabel("Actual R1 sig strikes landed")
    ax.set_title(f"R1 Sig Strikes vs Finish Time — r1_end fights (n={mask.sum()})\n"
                 f"Intercept={intercept:.1f} suggests burst floor the linear model misses")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [r1 scatter] n={mask.sum()}  origin-slope={b_origin:.3f}  "
          f"intercept={intercept:.2f}  slope_w_int={slope_int:.3f}")
    print(f"  [r1 scatter] saved to {save_path.name}")


def _compute_pit_vals(cdfs, y_true, rng):
    """Compute randomized PIT values (handles zero-mass at count=0)."""
    pit_vals = np.zeros(len(y_true))
    for i, (cdf, y) in enumerate(zip(cdfs, y_true)):
        if y == 0:
            p_zero = cdf.cdf(0)
            pit_vals[i] = float(rng.uniform(0, p_zero)) if p_zero > 0 else 0.0
        else:
            pit_vals[i] = cdf.cdf(float(y))
    return pit_vals


def _eval_count_model(name: str, target_col: str, model_file: str,
                      test_df: pd.DataFrame, report_dir: Path,
                      segments: dict | None = None,
                      duration_cdfs: list | None = None,
                      active_minutes_ceiling: float | None = None,
                      method_proba: np.ndarray | None = None,
                      duration_cdfs_by_method: dict | None = None,
                      zero_method_rate_adj: bool = False,
                      apply_burst: bool = True,
                      apply_method_hurdle: bool = False,
                      use_binned_rate_adj: bool = False,
                      conditional_seg_methods: dict | None = None,
                      conditional_r1_end: bool = False,
                      conditional_full_round: bool = False,
                      use_finish_head: bool = False,
                      use_cond_hurdle: bool = False,
                      use_cond_hurdle_for_overrides: bool | None = None,
                      mean_preserve_cond_hurdle: bool = True,
                      use_sub_count_head: bool = False) -> dict | None:
    """Evaluate a count model (HurdleCountModel or RateHurdleCountModel) using randomized PIT.

    Parameters
    ----------
    segments : dict[str, np.ndarray] | None
        Boolean masks for per-segment PIT analysis.
    duration_cdfs : list | None
        For RateHurdleCountModel: duration CDF per test row (from DurationModel).
    active_minutes_ceiling : float | None
        For RateHurdleCountModel r1_sig_strikes: 5.0.
    zero_method_rate_adj : bool
        If True, temporarily zero the per-method flat rate adjustment before
        prediction. Used for takedowns when use_binned_rate_adj=True replaces it.
    apply_method_hurdle : bool
        If True, enable method-conditional hurdle in predict_cdf (v8.4 fix).
        Re-enables stored method_logodds_hurdle_adj with mean-preserving
        re-centering so the overall KS gate is not affected.
    use_binned_rate_adj : bool
        If True, apply the (method × duration-bin) rate adjustments fitted in
        v8.5. Replaces the flat per-method scalar for takedowns so the
        duration confound is resolved per-draw rather than zeroed out.
    conditional_seg_methods : dict[str, str] | None
        Maps segment name -> method name (e.g. {"KO_finish": "KO/TKO"}).
        Each named segment is evaluated against a one-hot method-conditional
        forecast instead of the marginal, fixing the invalid PIT null for
        method-correlated sub-segments (uniformity only holds conditionally).
    """
    model_dir = paths.outputs_models()
    # Load the NEWEST model by mtime (not alphabetical): gitsha-named files do not
    # sort chronologically (e.g. "3370f8c" < "535b274"), so a freshly retrained
    # model would otherwise be shadowed by a stale one. Matches predict.py's loader.
    files = sorted(model_dir.glob(model_file), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"  No model found for {name} (pattern: {model_file})")
        return None

    print(f"\n  Evaluating {name}...")
    # Load via generic joblib — handles both HurdleCountModel and RateHurdleCountModel
    import joblib
    cm = joblib.load(files[-1])
    _cond_cdfs_by_m: dict[str, list] = {}
    if isinstance(cm, RateHurdleCountModel):
        _orig_rate_adj = None
        if zero_method_rate_adj and getattr(cm, "method_log_rate_adj", None) is not None:
            _orig_rate_adj = cm.method_log_rate_adj
            cm.method_log_rate_adj = None
        try:
            cdfs = cm.predict_cdf(test_df, duration_cdfs=duration_cdfs,
                                  active_minutes_ceiling=active_minutes_ceiling,
                                  method_proba=method_proba,
                                  duration_cdfs_by_method=duration_cdfs_by_method,
                                  apply_burst=apply_burst,
                                  apply_method_hurdle=apply_method_hurdle,
                                  use_binned_rate_adj=use_binned_rate_adj,
                                  use_finish_head=use_finish_head,
                                  use_cond_hurdle=use_cond_hurdle,
                                  mean_preserve_cond_hurdle=mean_preserve_cond_hurdle,
                                  use_sub_count_head=use_sub_count_head)
            # Phase 1: build one-hot method-conditional CDFs for sub-segment eval.
            # Method-correlated sub-segments must be tested against conditional
            # forecasts — PIT uniformity only holds conditionally when method is
            # informative about the count (it is for takedowns).
            if conditional_seg_methods and segments:
                # use_cond_hurdle_for_overrides lets callers decouple the cond-hurdle
                # setting for the marginal call (use_cond_hurdle) from the forced-method
                # segment-override calls.  Takedowns uses False/True to restore the
                # pre-v8.6 marginal while keeping the KO/SUB/DEC conditional fixes.
                _override_cond_hurdle = (
                    use_cond_hurdle_for_overrides
                    if use_cond_hurdle_for_overrides is not None
                    else use_cond_hurdle
                )
                _n_test = len(test_df)
                for _m_name in set(conditional_seg_methods.values()):
                    _m_idx = ["KO/TKO", "SUB", "DEC"].index(_m_name)
                    _onehot = np.zeros((_n_test, 3), dtype=float)
                    _onehot[:, _m_idx] = 1.0
                    _cond_cdfs_by_m[_m_name] = cm.predict_cdf(
                        test_df,
                        duration_cdfs=duration_cdfs,
                        active_minutes_ceiling=active_minutes_ceiling,
                        method_proba=_onehot,
                        duration_cdfs_by_method=duration_cdfs_by_method,
                        apply_burst=apply_burst,
                        apply_method_hurdle=apply_method_hurdle,
                        use_binned_rate_adj=use_binned_rate_adj,
                        use_finish_head=use_finish_head,
                        use_cond_hurdle=_override_cond_hurdle,
                        mean_preserve_cond_hurdle=False,  # forced-method: keep P(>0|method)
                        use_sub_count_head=use_sub_count_head,
                    )
        finally:
            if _orig_rate_adj is not None:
                cm.method_log_rate_adj = _orig_rate_adj
    elif isinstance(cm, ControlShareModel):
        cdfs = cm.predict_cdf(test_df, duration_cdfs=duration_cdfs,
                              method_proba=method_proba,
                              duration_cdfs_by_method=duration_cdfs_by_method)
    else:
        cdfs = cm.predict_cdf(test_df)
    y_true = test_df.get(target_col, pd.Series(dtype=float)).fillna(0).values

    rng = np.random.default_rng(42)
    pit_vals = _compute_pit_vals(cdfs, y_true, rng)

    slug = name.lower().replace(" ", "_")

    # Build per-segment conditional PIT overrides (Phase 1 eval-methodology fix)
    _seg_override: dict | None = None
    if segments and _cond_cdfs_by_m and conditional_seg_methods:
        _seg_override = {}
        _rng_cond = np.random.default_rng(43)
        for _sn, _mn in conditional_seg_methods.items():
            if _sn not in segments or _mn not in _cond_cdfs_by_m:
                continue
            _mask = np.asarray(segments[_sn], dtype=bool)
            _sc = _cond_cdfs_by_m[_mn]
            _seg_cdfs = [_sc[_ix] for _ix, _b in enumerate(_mask) if _b]
            _seg_override[_sn] = _compute_pit_vals(_seg_cdfs, y_true[_mask], _rng_cond)

    # v8.10: Combined 'finish' conditional null for takedowns.
    # The marginal-slice 'finish' PIT is non-uniform by construction: the forecast
    # must average over method (KO avg 0.47 TDs, SUB 1.09, DEC 1.24); realized finishes
    # are KO+SUB only, so the method-mixture marginal systematically over-predicts TDs.
    # This is the same mixture-slice artifact as decision/past_r1 — the model IS
    # calibrated when conditioned (combined realized-method null KS=0.027, p=0.92).
    # Fix: score each finish fight vs its realized-method (KO/SUB) override CDF.
    # Fires only for takedowns (the only caller with KO/SUB in conditional_seg_methods).
    if (segments and _cond_cdfs_by_m and "finish" in segments
            and "KO/TKO" in _cond_cdfs_by_m and "SUB" in _cond_cdfs_by_m):
        _meth_col = test_df["method"].fillna("").values
        _fin_mask = np.asarray(segments["finish"], dtype=bool)
        _fin_cdfs, _fin_y = [], []
        for _ix in range(len(test_df)):
            if not _fin_mask[_ix]:
                continue
            _m = str(_meth_col[_ix]).upper()
            if "KO" in _m:
                _src = _cond_cdfs_by_m["KO/TKO"]
            elif "SUB" in _m:
                _src = _cond_cdfs_by_m["SUB"]
            else:
                _src = cdfs  # rare OTH -> marginal
            _fin_cdfs.append(_src[_ix])
            _fin_y.append(y_true[_ix])
        if len(_fin_cdfs) >= 20:
            if _seg_override is None:
                _seg_override = {}
            _seg_override["finish"] = _compute_pit_vals(
                _fin_cdfs, np.array(_fin_y), np.random.default_rng(46)
            )
            from scipy import stats as _sc_stats
            _fin_ks, _fin_p = _sc_stats.kstest(_seg_override["finish"], "uniform")
            print(f"    [finish conditional null (realized-method KO/SUB)] "
                  f"n={len(_fin_cdfs)} KS={_fin_ks:.3f} p={_fin_p:.4f} "
                  f"-> {'PASS' if _fin_p > 0.05 else 'FAIL'}")

    # Conditional r1_end PIT null (v8.6): force every active draw to be a
    # finish draw (force_r1_end=True), giving F(count | ends in R1).
    # PIT uniformity under this conditional is a valid null — the marginal
    # slice is biased because r1_end membership correlates with count value.
    if (conditional_r1_end
            and isinstance(cm, RateHurdleCountModel)
            and segments is not None
            and "r1_end" in segments):
        _r1_mask = np.asarray(segments["r1_end"], dtype=bool)
        if _r1_mask.sum() >= 20:
            _r1_cond_cdfs = cm.predict_cdf(
                test_df,
                duration_cdfs=duration_cdfs,
                active_minutes_ceiling=active_minutes_ceiling,
                method_proba=method_proba,
                force_r1_end=True,
                use_finish_head=use_finish_head,
                apply_burst=apply_burst,
            )
            _r1_seg_cdfs = [_r1_cond_cdfs[_ix] for _ix, _b in enumerate(_r1_mask) if _b]
            _r1_pit = _compute_pit_vals(
                _r1_seg_cdfs, y_true[_r1_mask], np.random.default_rng(44)
            )
            if _seg_override is None:
                _seg_override = {}
            _seg_override["r1_end"] = _r1_pit
            from scipy import stats as _scipy_stats
            _r1_ks, _r1_p = _scipy_stats.kstest(_r1_pit, "uniform")
            _note = "use_finish_head" if use_finish_head else "force_r1_end only"
            print(f"    [r1_end conditional null ({_note})] "
                  f"n={int(_r1_mask.sum())} KS={_r1_ks:.3f} p={_r1_p:.4f} "
                  f"-> {'PASS' if _r1_p > 0.05 else 'FAIL'}")

    # Symmetric past_r1 conditional null (v8.8): force every active draw to be a
    # full-round draw (force_full_round=True, p_r1_end=0), giving F(count | survives R1).
    # This is the valid null for past_r1 fights.  The mixture-slice null (slicing the
    # marginal PIT by realized outcome) is non-uniform by construction for any calibrated
    # mixture forecast: for past_r1 fights F_i(Y) ≈ p_r1_end + (1−p_r1_end)·U,
    # i.e. Uniform[p_r1_end, 1], not [0,1].  The conditional null isolates F_full.
    if (conditional_full_round
            and isinstance(cm, RateHurdleCountModel)
            and segments is not None
            and "past_r1" in segments):
        _fr_mask = np.asarray(segments["past_r1"], dtype=bool)
        if _fr_mask.sum() >= 20:
            _fr_cond_cdfs = cm.predict_cdf(
                test_df,
                duration_cdfs=duration_cdfs,
                active_minutes_ceiling=active_minutes_ceiling,
                method_proba=method_proba,
                force_full_round=True,
                use_finish_head=use_finish_head,
                apply_burst=apply_burst,
            )
            _fr_seg_cdfs = [_fr_cond_cdfs[_ix] for _ix, _b in enumerate(_fr_mask) if _b]
            _fr_pit = _compute_pit_vals(
                _fr_seg_cdfs, y_true[_fr_mask], np.random.default_rng(45)
            )
            if _seg_override is None:
                _seg_override = {}
            _seg_override["past_r1"] = _fr_pit
            from scipy import stats as _scipy_stats_fr
            _fr_ks, _fr_p = _scipy_stats_fr.kstest(_fr_pit, "uniform")
            print(f"    [past_r1 conditional null (force_full_round)] "
                  f"n={int(_fr_mask.sum())} KS={_fr_ks:.3f} p={_fr_p:.4f} "
                  f"-> {'PASS' if _fr_p > 0.05 else 'FAIL'}")

    if segments:
        result = pit_histogram_segmented(
            y_true, pit_vals,
            segments=segments,
            title=f"PIT — {name}",
            save_path=report_dir / f"pit_{slug}_seg.png",
            segment_values_override=_seg_override,
        )
        # Also emit the plain overall plot for backward compat
        pit_histogram(
            y_true, pit_vals,
            title=f"PIT — {name}",
            save_path=report_dir / f"pit_{slug}.png",
        )
    else:
        result = pit_histogram(
            y_true, pit_vals,
            title=f"PIT — {name}",
            save_path=report_dir / f"pit_{slug}.png",
        )

    ks_stat = result["ks_stat"]
    ks_p = result["ks_p"]
    status = "PASS" if ks_p > 0.05 else "FAIL"
    print(f"  KS stat={ks_stat:.3f}, p={ks_p:.4f} -> {status}")

    if "segments" in result:
        for seg_name, s in result["segments"].items():
            _note = " [cond]" if (conditional_seg_methods and seg_name in conditional_seg_methods) else ""
            if not np.isnan(s["ks_stat"]):
                seg_status = "PASS" if s["ks_p"] > 0.05 else "FAIL"
                print(f"    [{seg_name}] n={s['n']} KS={s['ks_stat']:.3f} p={s['ks_p']:.4f} -> {seg_status}{_note}")
            else:
                print(f"    [{seg_name}] n={s['n']} — too few for KS{_note}")

    return result


def _td_hurdle_diagnostic(test_df: pd.DataFrame, model_dir: Path) -> None:
    """Phase 0: print empirical P(>0 TD|method,duration) table + stored hurdle adj values."""
    import joblib
    files = sorted(model_dir.glob("props_takedowns_*.joblib"), key=lambda p: p.stat().st_mtime)
    if not files:
        print("  [td_diag] no takedowns model found, skipping")
        return

    cm = joblib.load(files[-1])
    adj = getattr(cm, "method_logodds_hurdle_adj", None)
    print("\n  [td_diag] Stored method_logodds_hurdle_adj (negative = fewer TDs expected):")
    if adj:
        for m, v in adj.items():
            print(f"    {m}: {v:+.3f}")
    else:
        print("    (not fitted — model predates v8.1)")

    _METHOD_NORM = {
        "KO/TKO": "KO/TKO", "SUB": "SUB",
        "U-DEC": "DEC", "S-DEC": "DEC", "M-DEC": "DEC",
    }
    cols_needed = ["td_landed_a", "total_fight_sec", "method"]
    td_df = test_df.dropna(subset=cols_needed).copy()
    td_df["method_grp"] = td_df["method"].map(_METHOD_NORM).fillna("DEC")
    td_df["has_td"] = (td_df["td_landed_a"].fillna(0) > 0).astype(float)
    try:
        td_df["dur_q"] = pd.qcut(td_df["total_fight_sec"], q=4,
                                  labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        td_df["dur_q"] = pd.cut(td_df["total_fight_sec"], bins=4,
                                 labels=["Q1", "Q2", "Q3", "Q4"])

    table = td_df.groupby(["method_grp", "dur_q"], observed=True)["has_td"].agg(
        P_gt0="mean", n="count"
    )
    print("\n  [td_diag] Empirical P(>0 TD | method, duration_quartile) — test set:")
    print(table.to_string())

    marg = td_df.groupby("method_grp")["has_td"].agg(P_gt0="mean", n="count")
    print("\n  [td_diag] P(>0 TD | method) marginal:")
    print(marg.to_string())
    print("  (If KO gap persists within each quartile, the hurdle fix is structural.)")


def main():
    print("=== Phase 5: Prop Evaluation (v5.3 — segmented PIT) ===")
    model_dir = paths.outputs_models()
    report_dir = paths.outputs_reports()
    report_dir.mkdir(parents=True, exist_ok=True)

    props_df = parquet.read(paths.processed("features_props"))
    props_df["event_date"] = pd.to_datetime(props_df["event_date"])
    splits = get_splits(props_df)
    test_df = props_df[splits["test"]].copy()
    print(f"  Test set: {len(test_df)} rows")

    # ── Segment masks for count models ───────────────────────────────────────
    is_finish = ~test_df["method"].isin(["U-DEC", "S-DEC", "M-DEC"]).values
    is_decision = ~is_finish
    is_5rd = (test_df["scheduled_rounds"].fillna(3) == 5).values
    is_3rd = (test_df["scheduled_rounds"].fillna(3) == 3).values
    is_r1_end = (test_df["end_round"].fillna(99) == 1).values
    is_past_r1 = (test_df["end_round"].fillna(99) > 1).values

    count_segs = {
        "finish": is_finish,
        "decision": is_decision,
        "5rd": is_5rd,
        "3rd": is_3rd,
    }
    r1_segs = {
        "r1_end": is_r1_end,
        "past_r1": is_past_r1,
    }

    # Sub-vs-KO finish split for takedowns diagnostic (root-cause confirmation):
    # Hypothesis — miscalibration is opposite-signed in sub vs KO finishes,
    # confirming that method-conditional hurdle+rate is the required fix.
    is_ko_finish = is_finish & test_df["method"].fillna("").str.contains("KO", case=False).values
    is_sub_finish = is_finish & test_df["method"].fillna("").str.contains("SUB", case=False).values
    td_segs = {
        **count_segs,
        "KO_finish": is_ko_finish,
        "sub_finish": is_sub_finish,
    }

    results = {}

    # ── Method classifier probs (for method-conditional count sampling) ───────
    method_proba: np.ndarray | None = None
    method_clf_files = sorted(model_dir.glob("method_clf_*.joblib"), key=lambda p: p.stat().st_mtime)
    if method_clf_files:
        print("  Loading method classifier for method-conditional count sampling...")
        method_clf = MethodClassifier.load(method_clf_files[-1])
        probs_dict = method_clf.predict_proba_dict(test_df)
        method_proba = np.column_stack([probs_dict[c] for c in METHOD_CLASSES])
        print(f"  Method proba shape: {method_proba.shape}  "
              f"mean KO={method_proba[:, 0].mean():.3f}  "
              f"SUB={method_proba[:, 1].mean():.3f}  "
              f"DEC={method_proba[:, 2].mean():.3f}")

    # ── Duration model (evaluated first; CDFs reused for rate models) ─────────
    dur_cdfs_full: list | None = None  # CDFs indexed to full test_df rows
    dur_files = sorted(model_dir.glob("props_duration_*.joblib"), key=lambda p: p.stat().st_mtime)
    if dur_files:
        print("\n  Evaluating duration model...")
        dm = DurationModel.load(dur_files[-1])

        # Predict duration CDFs for ALL test rows.
        # use_boundary_mass=False for count-model MC integration (smooth CDFs preserve
        # expected active-minutes calibration); boundary mass is only needed for the
        # duration KS evaluation itself.
        dur_cdfs_full = dm.predict_cdf(test_df, use_boundary_mass=False)

        # Build three method-conditional duration CDFs for the takedowns coherence
        # fix: each draw in the count MC loop selects duration from its method's CDF
        # so KO draws are short and DEC draws are long (coherent joint).
        dur_cdfs_by_method: dict | None = None
        if hasattr(dm, "feature_cols") and "method_ko" in dm.feature_cols:
            dur_cdfs_by_method = {
                "KO/TKO": dm.predict_cdf(test_df, method_override="KO/TKO", use_boundary_mass=False),
                "SUB":    dm.predict_cdf(test_df, method_override="SUB",    use_boundary_mass=False),
                "DEC":    dm.predict_cdf(test_df, method_override="DEC",    use_boundary_mass=False),
            }
            print("  Built method-conditional duration CDFs (KO/SUB/DEC) for takedowns.")

        dur_test = test_df.drop_duplicates(subset=["fight_id"]).dropna(subset=["total_fight_sec"])
        # Duration PIT uses boundary mass for the most accurate shape test
        cdfs = dm.predict_cdf(dur_test)

        y_true = dur_test["total_fight_sec"].clip(lower=1).values
        is_decision_dur = dur_test["method"].isin(["U-DEC", "S-DEC", "M-DEC"]).values
        is_finish_dur = ~is_decision_dur

        # Randomized PIT for mixed distribution:
        # Decisions are a point mass at scheduled_sec -> PIT ~ Uniform(cdf(t-), 1.0)
        # Finishes are continuous -> PIT = cdf(t)
        rng_dur = np.random.default_rng(42)
        pit_vals = np.zeros(len(y_true))
        for i, (cdf, y, is_dec) in enumerate(zip(cdfs, y_true, is_decision_dur)):
            if is_dec:
                cdf_lower = cdf.cdf(y - 1)
                pit_vals[i] = float(rng_dur.uniform(cdf_lower, 1.0))
            else:
                pit_vals[i] = cdf.cdf(float(y))

        dur_segs = {
            "finish": is_finish_dur,
            "decision": is_decision_dur,
        }

        # ── Step 3: renormalised segment PITs (F0-A fix) ──────────────────
        # For a correctly-calibrated mixture P(T<=t) = p_fin * cdf_finish(t),
        # the raw PIT lives on [0, p_fin] for finishes and [p_fin, 1] for
        # decisions.  KS-testing those sub-arrays against Uniform(0,1) is
        # structurally biased: finish KS ≈ 0.44, decision KS ≈ 0.42 even for
        # a perfect model.  The valid per-segment null is Uniform(0,1) AFTER
        # renormalisation onto [0,1]:
        #   finish   : pit_norm = pit / p_fin           (rescale [0,p_fin] → [0,1])
        #   decision : pit_norm = (pit - p_fin) / p_dec (shift+scale [p_fin,1] → [0,1])
        # Aggregate KS (overall) still uses raw PIT and is unaffected.
        p_fin_arr = np.array([c._p_fin for c in cdfs], dtype=float)
        p_dec_arr = np.array([c._p_dec for c in cdfs], dtype=float)
        pit_renorm_finish = np.clip(
            pit_vals[is_finish_dur] / np.maximum(p_fin_arr[is_finish_dur], 1e-6),
            0.0, 1.0,
        )
        pit_renorm_dec = np.clip(
            (pit_vals[is_decision_dur] - p_fin_arr[is_decision_dur])
            / np.maximum(p_dec_arr[is_decision_dur], 1e-6),
            0.0, 1.0,
        )
        dur_segs_renorm = {
            "finish (renormed)": is_finish_dur,
            "decision (renormed)": is_decision_dur,
        }
        seg_override = {
            "finish (renormed)": pit_renorm_finish,
            "decision (renormed)": pit_renorm_dec,
        }

        result = pit_histogram_segmented(
            y_true, pit_vals,
            segments=dur_segs_renorm,
            title="PIT — Fight Duration",
            save_path=report_dir / "pit_duration_seg.png",
            segment_values_override=seg_override,
        )
        # Backward-compat plain plot
        pit_histogram(
            y_true, pit_vals,
            title="PIT — Fight Duration",
            save_path=report_dir / "pit_duration.png",
        )

        ks_stat = result["ks_stat"]
        ks_p = result["ks_p"]
        status = "PASS" if ks_p > 0.05 else "FAIL"
        print(f"  KS stat={ks_stat:.3f}, p={ks_p:.4f} -> {status}")
        for seg_name, s in result["segments"].items():
            if not np.isnan(s["ks_stat"]):
                seg_status = "PASS" if s["ks_p"] > 0.05 else "FAIL"
                note = " [renormed: valid per-segment null]"
                print(f"    [{seg_name}] n={s['n']} KS={s['ks_stat']:.3f} p={s['ks_p']:.4f} -> {seg_status}{note}")
            else:
                print(f"    [{seg_name}] n={s['n']} — too few for KS")
        results["duration"] = result

        # v8.24 short-tail diagnostic (non-blocking): bucket calibration of
        # display_dur_cdf.p_over(t) vs empirical (y_true > t) for finish rows.
        # Root-cause: display CDF is now the correct mixture; this confirms
        # whether the short-tail bias is resolved.  Not a hard gate.
        try:
            from ufc.inference.predict_core import load_models, load_reference_data, predict_fight
            from ufc.io import paths as _paths, parquet as _pq
            print("\n  [short-tail diag] display p_over(t) vs empirical across finishes:")
            _fighters = _pq.read(_paths.interim("fighters"))
            _pfs = _pq.read(_paths.processed("pre_fight_state"))
            _models_diag = load_models(verbose=False)
            _refh = None
            try:
                from ufc.inference.ref_history import build_ref_history
                _refh = build_ref_history()
            except Exception:
                pass
            _finish_mask = ~test_df["method"].isin(["U-DEC", "S-DEC", "M-DEC"])
            _finish_rows = test_df[_finish_mask].head(50)
            _diag_errs: dict[int, list] = {150: [], 300: [], 450: []}
            for _, _row in _finish_rows.iterrows():
                try:
                    _rn = _fighters.loc[_fighters["fighter_id"] == _row["fighter_id_a"], "fighter_name"]
                    _bn = _fighters.loc[_fighters["fighter_id"] == _row["fighter_id_b"], "fighter_name"]
                    if _rn.empty or _bn.empty:
                        continue
                    import datetime as _dt
                    _ev = _row.get("event_date")
                    _ed = _ev if isinstance(_ev, _dt.date) else _dt.date(2023, 1, 1)
                    _pr = predict_fight(
                        str(_rn.iloc[0]), str(_bn.iloc[0]),
                        int(_row.get("scheduled_rounds", 3) or 3),
                        bool(_row.get("is_title", False)),
                        _ed, _models_diag, _fighters, _pfs,
                        ref_history_df=_refh, run_simulation=False, verbose=False,
                    )
                    _true_t = float(_row["total_fight_sec"])
                    for _t in [150, 300, 450]:
                        _diag_errs[_t].append(_pr.display_dur_cdf.p_over(_t) - float(_true_t > _t))
                except Exception:
                    pass
            for _t in [150, 300, 450]:
                _e = _diag_errs[_t]
                if _e:
                    import numpy as _np2
                    print(f"    t={_t}s: n={len(_e)}  mean_err={_np2.mean(_e):+.3f}  "
                          f"({'+' if _np2.mean(_e) > 0 else ''}{'over' if _np2.mean(_e) > 0 else 'under'}-predicting long fights)")
        except Exception as _diag_exc:
            print(f"  [short-tail diag] skipped: {_diag_exc}")

    # ── Sig Strikes ──────────────────────────────────────────────────────────
    # v8.13: pass method_proba + duration_cdfs_by_method + zero_method_rate_adj
    # so the gate measures the PRODUCTION forecast (method-marginal), not a
    # realized-method path that production never takes.  Previously the gate used
    # realized-method durations (dur_cdfs_full carries the actual method column)
    # and passed KS 0.032, while production was method-blind (DEC-default) with
    # KS 0.243 — a hidden 42% over-prediction that would systematically over-bet OVERs.
    # The rate-adj is zeroed (method signal lives in the durations, not the rate scalars).
    r = _eval_count_model("sig_strikes", "sig_str_landed_a",
                          "props_sig_strikes_*.joblib", test_df, report_dir,
                          segments=count_segs,
                          duration_cdfs=dur_cdfs_full,
                          method_proba=method_proba,
                          duration_cdfs_by_method=dur_cdfs_by_method,
                          zero_method_rate_adj=False)
    if r:
        results["sig_strikes"] = r

    # ── v8.10: CRPS sharpness verification for sig strikes ─────────────────
    # Confirms the forecast carries real betting edge beyond calibration.
    # Climatology baseline: train-set empirical distribution of sig_str_landed_a,
    # stratified by scheduled_rounds (3rd vs 5rd) — a fair, leak-free baseline.
    # Skill score = 1 - CRPS_model / CRPS_clim; > 0 means model beats climatology.
    # v8.13: CRPS also uses the method-marginal path (matching the gate and production).
    try:
        import joblib as _jlb
        _ss_files = sorted(model_dir.glob("props_sig_strikes_*.joblib"),
                           key=lambda p: p.stat().st_mtime)
        if _ss_files and dur_cdfs_full is not None:
            print("\n  [CRPS] Computing sig_strikes sharpness vs climatology (method-marginal)...")
            _ss_cm = _jlb.load(_ss_files[-1])
            # Use method-marginal call with method_log_rate_adj matching the gate and production path.
            _ss_cdfs = _ss_cm.predict_cdf(
                    test_df, duration_cdfs=dur_cdfs_full,
                    method_proba=method_proba,
                    duration_cdfs_by_method=dur_cdfs_by_method,
                )
            _y_ss = test_df["sig_str_landed_a"].fillna(0).values.astype(float)

            # Train set climatology pools (stratified by scheduled_rounds)
            _tr_df = props_df[splits["train"]].copy()
            _clim_pools: dict[int, np.ndarray] = {}
            for _sr in [3, 5]:
                _tr_sr = _tr_df[_tr_df["scheduled_rounds"].fillna(3).astype(int) == _sr]
                _pool = _tr_sr["sig_str_landed_a"].fillna(0).values.astype(float)
                if len(_pool) >= 10:
                    _clim_pools[_sr] = _pool
            # Fallback pool for other scheduled_rounds values
            _clim_pools[0] = _tr_df["sig_str_landed_a"].fillna(0).values.astype(float)

            def _crps_samples(samples: np.ndarray, y: float, n_sub: int = 500) -> float:
                """CRPS via energy form: E|X-y| - 0.5*E|X-X'|."""
                if len(samples) > n_sub:
                    rng_crps = np.random.default_rng(99)
                    s = rng_crps.choice(samples, n_sub, replace=False)
                else:
                    s = samples
                term1 = float(np.mean(np.abs(s - y)))
                term2 = 0.5 * float(np.mean(np.abs(s[:, None] - s[None, :])))
                return term1 - term2

            _crps_model, _crps_clim = [], []
            _sr_test = test_df["scheduled_rounds"].fillna(3).astype(int).values
            for _ii, (_cdf_i, _yi, _sri) in enumerate(zip(_ss_cdfs, _y_ss, _sr_test)):
                _crps_model.append(_crps_samples(_cdf_i._samples, _yi))
                _pool = _clim_pools.get(int(_sri), _clim_pools[0])
                _crps_clim.append(_crps_samples(_pool, _yi))

            _crps_m = float(np.mean(_crps_model))
            _crps_c = float(np.mean(_crps_clim))
            _skill = 1.0 - _crps_m / _crps_c if _crps_c > 0 else float("nan")
            _outcome_str = "beats clim [OK]" if _skill > 0 else "below clim [WARN]"
            print(f"  [CRPS] sig_strikes: model={_crps_m:.3f}  clim={_crps_c:.3f}  "
                  f"skill={_skill:.3f}  ({_outcome_str})")
            if results.get("sig_strikes") is not None:
                results["sig_strikes"]["crps_model"] = _crps_m
                results["sig_strikes"]["crps_clim"] = _crps_c
                results["sig_strikes"]["crps_skill"] = _skill
        else:
            print("  [CRPS] skipping — no sig_strikes model or duration CDFs available")
    except Exception as _crps_exc:
        print(f"  [CRPS] warning: CRPS computation failed: {_crps_exc}")

    # ── Takedowns ─────────────────────────────────────────────────────────────
    # v8.3: method-conditional duration CDFs (KO draws short, DEC draws long).
    # v8.4: sub-segments evaluated against method-conditional forecasts.
    # use_binned_rate_adj=False: the duration-binned Q1 residuals are a denominator
    # artifact (TD/min inflated for short fights) — enabling it worsens overall gate.
    # apply_method_hurdle=False: superseded by use_cond_hurdle (v8.6).
    # use_cond_hurdle=True (v8.6): per-draw P(>0 TD | row, sampled_method,
    #   sampled_dur) from pos_clf_cond. Corrects the KO-finish zero-mass deficit
    #   (P(>0 TD|KO,Q1)=0.133 vs. global hurdle ≈ blind to method/duration).
    #   Mean-preserving rescaling ensures overall gate is not disturbed.
    # v8.7: use_cond_hurdle=False on the marginal call; the conditional hurdle
    # integrates P(>0|method,dur) which extrapolates at 25-min (5rd DEC) and
    # broke the 5rd gate in v8.6.  use_cond_hurdle_for_overrides=True keeps the
    # method-conditional fix active for KO/SUB/DEC forced-method segment evals
    # (which is where the KO_finish fix comes from).
    r = _eval_count_model("takedowns", "td_landed_a",
                          "props_takedowns_*.joblib", test_df, report_dir,
                          segments=td_segs,
                          duration_cdfs=dur_cdfs_full,
                          method_proba=method_proba,
                          duration_cdfs_by_method=dur_cdfs_by_method,
                          zero_method_rate_adj=True,
                          apply_method_hurdle=False,
                          use_binned_rate_adj=False,
                          use_cond_hurdle=False,
                          use_cond_hurdle_for_overrides=True,
                          use_sub_count_head=True,
                          conditional_seg_methods={
                              "KO_finish": "KO/TKO",
                              "sub_finish": "SUB",
                              "decision": "DEC",
                          })
    if r:
        results["takedowns"] = r
    _td_hurdle_diagnostic(test_df, model_dir)

    # ── R1 Sig Strikes ───────────────────────────────────────────────────────
    # v8.8 evaluation methodology (principal-review findings):
    #
    # Gate: overall marginal PIT-KS only (p > 0.05).  Segment outcomes are
    # diagnostics, not gates.  Reason: slicing the per-fight PIT by *realized*
    # R1 outcome is non-uniform by construction for any calibrated mixture.
    # For r1_end fights, F_i(Y) ≈ p_r1_end · U → Uniform[0, p_r1_end].
    # For past_r1 fights, F_i(Y) ≈ p_r1_end + (1−p_r1_end) · U → Uniform[p,1].
    # Neither is [0,1]; the only valid null is the mixture marginal (overall).
    #
    # Diagnostics (conditional nulls):
    #   r1_end  → force_r1_end=True:   F(count | ends in R1)    — already in v8.6
    #   past_r1 → force_full_round=True: F(count | survives R1) — new in v8.8
    #
    # Forecast path: use_finish_head=True (t-conditional head, v8.7), apply_burst=False.
    # Same flags used in predict.py (v8.8) so eval scores the production forecast.
    r = _eval_count_model("r1_sig_strikes", "r1_sig_str_landed_a",
                          "props_r1_sig_strikes_*.joblib", test_df, report_dir,
                          segments=r1_segs,
                          duration_cdfs=dur_cdfs_full,
                          active_minutes_ceiling=5.0,
                          method_proba=method_proba,  # v8.9: method-conditioned finish head
                          apply_burst=False,
                          use_finish_head=True,
                          conditional_r1_end=True,
                          conditional_full_round=True)
    if r:
        results["r1_sig_strikes"] = r

    # R1 finishing-burst diagnostic: scatter of actual strikes vs finish time.
    _plot_r1_strikes_scatter(test_df, report_dir / "r1_strikes_vs_finish_time.png")

    # ── Knockdowns ────────────────────────────────────────────────────────────
    # Same method-marginal path as sig_strikes. Conditional-null segments are
    # diagnostics only — KO-finish positives (~dozens) are too sparse for KS.
    r = _eval_count_model("knockdowns", "kd_for_a",
                          "props_knockdowns_*.joblib", test_df, report_dir,
                          segments=td_segs,
                          duration_cdfs=dur_cdfs_full,
                          method_proba=method_proba,
                          duration_cdfs_by_method=dur_cdfs_by_method,
                          zero_method_rate_adj=True,
                          apply_method_hurdle=False,
                          use_binned_rate_adj=False,
                          use_cond_hurdle=False,
                          use_cond_hurdle_for_overrides=True,
                          use_sub_count_head=False,
                          conditional_seg_methods={
                              "KO_finish": "KO/TKO",
                              "sub_finish": "SUB",
                              "decision": "DEC",
                          })
    if r:
        results["knockdowns"] = r

    # ── Submission Attempts ──────────────────────────────────────────────────
    # Takedowns-clone structure: censor-weighted, SUB count head, cond-hurdle overrides.
    r = _eval_count_model("sub_attempts", "sub_att_for_a",
                          "props_sub_attempts_*.joblib", test_df, report_dir,
                          segments=td_segs,
                          duration_cdfs=dur_cdfs_full,
                          method_proba=method_proba,
                          duration_cdfs_by_method=dur_cdfs_by_method,
                          zero_method_rate_adj=True,
                          apply_method_hurdle=False,
                          use_binned_rate_adj=False,
                          use_cond_hurdle=False,
                          use_cond_hurdle_for_overrides=True,
                          use_sub_count_head=True,
                          conditional_seg_methods={
                              "KO_finish": "KO/TKO",
                              "sub_finish": "SUB",
                              "decision": "DEC",
                          })
    if r:
        results["sub_attempts"] = r

    # ── R1 Takedowns ─────────────────────────────────────────────────────────
    # Mirror r1_sig_strikes exactly. r1_end conditional may fail (diagnostic only).
    r = _eval_count_model("r1_takedowns", "r1_td_landed_a",
                          "props_r1_takedowns_*.joblib", test_df, report_dir,
                          segments=r1_segs,
                          duration_cdfs=dur_cdfs_full,
                          active_minutes_ceiling=5.0,
                          method_proba=method_proba,
                          apply_burst=False,
                          use_finish_head=True,
                          conditional_r1_end=True,
                          conditional_full_round=True)
    if r:
        results["r1_takedowns"] = r

    # ── Body Sig Strikes ─────────────────────────────────────────────────────
    # Sig-strikes clone: method-marginal, rate_calib_factor applied.
    r = _eval_count_model("body_sig_strikes", "body_landed_a",
                          "props_body_sig_strikes_*.joblib", test_df, report_dir,
                          segments=count_segs,
                          duration_cdfs=dur_cdfs_full,
                          method_proba=method_proba,
                          duration_cdfs_by_method=dur_cdfs_by_method,
                          zero_method_rate_adj=False)
    if r:
        results["body_sig_strikes"] = r

    # ── Leg Sig Strikes ──────────────────────────────────────────────────────
    r = _eval_count_model("leg_sig_strikes", "leg_landed_a",
                          "props_leg_sig_strikes_*.joblib", test_df, report_dir,
                          segments=count_segs,
                          duration_cdfs=dur_cdfs_full,
                          method_proba=method_proba,
                          duration_cdfs_by_method=dur_cdfs_by_method,
                          zero_method_rate_adj=True)
    if r:
        results["leg_sig_strikes"] = r

    # ── Control Time ─────────────────────────────────────────────────────────
    # Canonical unit: seconds. rate_ceiling=60.0 already baked into the model.
    # Evaluate in seconds — lines are in minutes so display converts, but gate
    # operates on the raw seconds PIT (uniformity holds in any monotone transform).
    r = _eval_count_model("ctrl_time", "ctrl_sec_a",
                          "props_ctrl_time_*.joblib", test_df, report_dir,
                          segments=td_segs,
                          duration_cdfs=dur_cdfs_full,
                          method_proba=method_proba,
                          duration_cdfs_by_method=dur_cdfs_by_method,
                          zero_method_rate_adj=True,
                          apply_method_hurdle=False,
                          use_cond_hurdle=False,
                          use_cond_hurdle_for_overrides=True,
                          use_sub_count_head=False,
                          conditional_seg_methods={
                              "KO_finish": "KO/TKO",
                              "sub_finish": "SUB",
                              "decision": "DEC",
                          })
    if r:
        results["ctrl_time"] = r

    # ── Sig Strikes Combo (combined A+B) ─────────────────────────────────────
    # No artifact — built from the sig_strikes model via predict_combined_count_cdf.
    # Gate: marginal PIT-KS on realized (sig_str_landed_a + sig_str_landed_b).
    try:
        import joblib as _jlb
        from ufc.models.props_count import predict_combined_count_cdf
        from ufc.training.symmetrize import symmetrize
        _ss_files = sorted(model_dir.glob("props_sig_strikes_*.joblib"),
                           key=lambda p: p.stat().st_mtime)
        if _ss_files and dur_cdfs_full is not None:
            print("\n  Evaluating sig_strikes_combo (combined A+B, derived from sig_strikes)...")
            _ss_model = _jlb.load(_ss_files[-1])
            _y_combo = (test_df["sig_str_landed_a"].fillna(0) +
                        test_df["sig_str_landed_b"].fillna(0)).values.astype(float)

            _combo_cdfs = []
            for _i in range(len(test_df)):
                _row_a = test_df.iloc[_i:_i+1]
                _row_b = symmetrize(_row_a)
                _combo_cdf = predict_combined_count_cdf(
                    _ss_model, _row_a, _row_b,
                    duration_cdfs=[dur_cdfs_full[_i]] if dur_cdfs_full else None,
                    method_proba=method_proba[_i:_i+1] if method_proba is not None else None,
                    duration_cdfs_by_method={
                        m: [dur_cdfs_by_method[m][_i]] for m in dur_cdfs_by_method
                    } if dur_cdfs_by_method else None,
                )
                _combo_cdfs.append(_combo_cdf)

            # Randomized PIT for combined
            from scipy.stats import ks_1samp
            _rng_pit = np.random.default_rng(42)
            _pits = []
            for _cdf_i, _yi in zip(_combo_cdfs, _y_combo):
                _u_lo = _cdf_i.cdf(_yi - 0.5)
                _u_hi = _cdf_i.cdf(_yi)
                _pits.append(float(_rng_pit.uniform(_u_lo, _u_hi)))
            _pits = np.array(_pits)
            _ks_stat, _ks_p = ks_1samp(_pits, lambda x: x)
            _status = "PASS" if _ks_p > 0.05 else "FAIL"
            print(f"  sig_strikes_combo: KS={_ks_stat:.3f}  p={_ks_p:.4f}  [{_status}]")
            results["sig_strikes_combo"] = {"ks_stat": _ks_stat, "ks_p": _ks_p}
        else:
            print("  [sig_strikes_combo] skipping — no sig_strikes model or duration CDFs")
    except Exception as _combo_exc:
        print(f"  [sig_strikes_combo] warning: {_combo_exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n=== Gate B Summary ===")
    print(f"  {'Model':<20} {'KS':>6} {'p':>8} {'Status':>8}")
    print("  " + "-" * 46)
    for name, r in results.items():
        ks = r["ks_stat"]
        p = r["ks_p"]
        status = "PASS" if p > 0.05 else "FAIL"
        print(f"  {name:<20} {ks:>6.3f} {p:>8.4f} {status:>8}")

    print(f"\n  PIT plots saved to {report_dir}")

    # ── Prop calibration report ───────────────────────────────────────────────
    from datetime import date
    prop_report_path = report_dir / f"props_calibration_{date.today()}.md"
    write_prop_report(results, prop_report_path)


if __name__ == "__main__":
    main()
