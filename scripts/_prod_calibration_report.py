"""Prod-tier calibration sanity check.

Loads prod artifacts from outputs/models/prod/ and scores the val window
(the most recent 6 months of data, also used for calibration). There is no
separate holdout — the prod model trains on ALL data; the val window overlaps
with training by design. Metrics here are in-distribution sanity checks, not
OOS evaluations. Eval-tier Gates A-D (locked split) remain the correctness proof.

Run:
    python scripts/_prod_calibration_report.py
"""
import os
import sys
from datetime import date
from pathlib import Path

# Must set env var BEFORE any ufc imports so every _cfg() call picks it up.
os.environ["UFC_SPLIT_CONFIG"] = "split_prod.yaml"

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.models.winner import WinnerModel
from ufc.models.method import MethodClassifier, METHOD_CLASSES
from ufc.models.props_duration import DurationModel
from ufc.models.props_count import HurdleCountModel, RateHurdleCountModel, ControlShareModel
from ufc.training.symmetrize import symmetrize
from ufc.evaluation.metrics import winner_metrics


TOLERANCE = {
    "accuracy": 0.62,
    "brier": 0.235,
}


def _find_prod_model(pattern: str) -> Path | None:
    prod_dir = paths.outputs_models_prod()
    files = sorted(prod_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _winner_metrics(model_dir_label: str) -> dict:
    import joblib
    wp = _find_prod_model("winner_ensemble_*.joblib")
    if not wp:
        print("  [prod] No winner model found in prod/")
        return {}
    winner_model = WinnerModel.load(wp)
    print(f"  Winner: {wp.name}")

    features = parquet.read(paths.processed("features_winner"))
    features["event_date"] = pd.to_datetime(features["event_date"])
    splits = get_splits(features)
    holdout = features[splits["val"]].dropna(subset=["won_a"]).copy()
    if len(holdout) == 0:
        print("  [prod] No val rows for winner eval")
        return {}

    sym = symmetrize(holdout)
    n = len(holdout)
    p_full = winner_model.predict_proba(sym)
    probs = (p_full[:n] + (1.0 - p_full[n:])) / 2.0
    y_true = holdout["won_a"].astype(float).values
    return winner_metrics(y_true, probs)


def _prop_ks(props_df: pd.DataFrame, method_clf, dur_model,
             prop_specs: list[tuple]) -> dict[str, dict]:
    import joblib
    splits = get_splits(props_df)
    holdout = props_df[splits["val"]].copy().reset_index(drop=True)
    if len(holdout) == 0:
        print("  [prod] No val rows for prop eval")
        return {}

    mp = method_clf.predict_proba_dict(holdout)
    method_proba = np.column_stack([mp[c] for c in METHOD_CLASSES])
    dur_cdfs = dur_model.predict_cdf(holdout, use_boundary_mass=False)
    dur_by_m = {m: dur_model.predict_cdf(holdout, method_override=m, use_boundary_mass=False)
                for m in ("KO/TKO", "SUB", "DEC")}

    results = {}
    rng = np.random.default_rng(42)

    for name, pattern, target_col, kwargs in prop_specs:
        mf = _find_prod_model(pattern)
        if not mf:
            results[name] = {"status": "no_model"}
            continue
        print(f"  Prop {name}: {mf.name}")
        import joblib as jl
        cm = jl.load(mf)

        try:
            if isinstance(cm, RateHurdleCountModel):
                cdfs = cm.predict_cdf(holdout, duration_cdfs=dur_cdfs,
                                      method_proba=method_proba,
                                      duration_cdfs_by_method=dur_by_m, **kwargs)
            elif isinstance(cm, ControlShareModel):
                cdfs = cm.predict_cdf(holdout, duration_cdfs=dur_cdfs,
                                      method_proba=method_proba,
                                      duration_cdfs_by_method=dur_by_m)
            else:
                cdfs = cm.predict_cdf(holdout)
        except Exception as e:
            results[name] = {"status": f"error: {e}"}
            continue

        y = holdout.get(target_col, pd.Series(dtype=float)).fillna(0).values
        pit = np.zeros(len(y))
        for i, (cdf, yi) in enumerate(zip(cdfs, y)):
            if yi == 0:
                p0 = cdf.cdf(0)
                pit[i] = float(rng.uniform(0, p0)) if p0 > 0 else 0.0
            else:
                pit[i] = cdf.cdf(float(yi))

        ks, p = stats.kstest(pit, "uniform")
        results[name] = {"ks": float(ks), "p": float(p),
                         "pass": p > 0.05, "n": int(len(y))}

    return results


def main():
    print("=== Prod Calibration Report ===")
    print(f"  split: {os.environ['UFC_SPLIT_CONFIG']}")

    # ── Winner ──────────────────────────────────────────────────────────────
    print("\n[1] Winner val-window metrics (in-dist sanity)...")
    wm = _winner_metrics("prod")
    if wm:
        print(f"  acc={wm.get('accuracy', '?'):.4f}  brier={wm.get('brier', '?'):.4f}  "
              f"ece={wm.get('ece', '?'):.4f}")

    # ── Props ────────────────────────────────────────────────────────────────
    print("\n[2] Prop PIT-KS on val window (in-dist sanity)...")
    wp = _find_prod_model("method_clf_*.joblib")
    dp = _find_prod_model("props_duration_*.joblib")
    if not wp or not dp:
        print("  [prod] Missing method or duration model — skipping props")
        prop_results = {}
    else:
        import joblib
        method_clf = MethodClassifier.load(wp)
        dur_model = DurationModel.load(dp)
        props_df = parquet.read(paths.processed("features_props"))
        props_df["event_date"] = pd.to_datetime(props_df["event_date"])

        prop_specs = [
            ("sig_strikes",    "props_sig_strikes_*.joblib",    "sig_str_landed_a",     {"apply_burst": False}),
            ("takedowns",      "props_takedowns_*.joblib",      "td_landed_a",          {"apply_burst": False}),
            ("r1_sig_strikes", "props_r1_sig_strikes_*.joblib", "r1_sig_str_landed_a",  {"active_minutes_ceiling": 5.0, "use_finish_head": True, "apply_burst": False}),
            ("knockdowns",     "props_knockdowns_*.joblib",     "kd_for_a",             {}),
            ("sub_attempts",   "props_sub_attempts_*.joblib",   "sub_att_for_a",        {}),
            ("r1_takedowns",   "props_r1_takedowns_*.joblib",   "r1_td_landed_a",       {"active_minutes_ceiling": 5.0}),
            ("body_strikes",   "props_body_sig_strikes_*.joblib","body_landed_a",       {"apply_burst": False}),
            ("leg_strikes",    "props_leg_sig_strikes_*.joblib","leg_landed_a",         {"apply_burst": False}),
            ("ctrl_time",      "props_ctrl_time_*.joblib",      "ctrl_sec_a",           {}),
        ]
        prop_results = _prop_ks(props_df, method_clf, dur_model, prop_specs)

    # ── Duration ─────────────────────────────────────────────────────────────
    dur_ks_result = {}
    if dp:
        import joblib
        dur_model2 = DurationModel.load(dp)
        props_df2 = parquet.read(paths.processed("features_props"))
        props_df2["event_date"] = pd.to_datetime(props_df2["event_date"])
        splits2 = get_splits(props_df2)
        holdout2 = props_df2[splits2["val"]].dropna(subset=["total_fight_sec"]).copy()
        holdout2 = holdout2.drop_duplicates(subset=["fight_id"]).reset_index(drop=True)
        if len(holdout2) > 0:
            mp2 = MethodClassifier.load(wp).predict_proba_dict(holdout2)
            mpa2 = np.column_stack([mp2[c] for c in METHOD_CLASSES])
            dur_by_m2 = {m: dur_model2.predict_cdf(holdout2, method_override=m, use_boundary_mass=False)
                         for m in ("KO/TKO", "SUB", "DEC")}
            dur_cdfs2 = dur_model2.predict_cdf(holdout2, use_boundary_mass=False)
            sched = (holdout2["scheduled_rounds"].fillna(3).astype(float) * 300.0).values
            actual = holdout2["total_fight_sec"].values.astype(float)
            rng2 = np.random.default_rng(42)
            pit_dur = np.array([float(rng2.uniform(0, cdf.cdf(t))) if t == s else cdf.cdf(t)
                                for cdf, t, s in zip(dur_cdfs2, actual, sched)])
            ks_d, p_d = stats.kstest(pit_dur, "uniform")
            dur_ks_result = {"ks": float(ks_d), "p": float(p_d),
                             "pass": p_d > 0.05, "n": len(holdout2)}
            print(f"  duration  KS={ks_d:.3f}  p={p_d:.4f}  {'PASS' if p_d > 0.05 else 'FAIL'}  n={len(holdout2)}")

    # ── Write report ─────────────────────────────────────────────────────────
    report_dir = paths.outputs_reports()
    report_path = report_dir / f"prod_calibration_{date.today()}.md"

    import yaml
    split_cfg = yaml.safe_load((paths.root() / "configs" / "split_prod.yaml").read_text())

    lines = [
        f"# Prod Model Calibration Report — {date.today()}",
        "",
        f"Split: `split_prod.yaml`  train≤{split_cfg['train_end']}  "
        f"val(calib+eval)={split_cfg['val_start']}→{split_cfg['val_end']}  "
        f"(no holdout — prod trains on ALL data; test window = future fights)",
        "",
        "**In-distribution sanity check, not a Gate.**  Val window also used for calibration "
        "(Platt/rate_calib) — metrics here are optimistic; eval-tier Gates A–D remain the correctness proof. "
        "Winner ECE here is structurally inflated (val⊂train memorization vs the 0.75 prob-cap) and is informational only.",
        "",
        "## Winner (val window, in-dist)",
        "",
        "| Metric | Value | Tolerance |",
        "|--------|-------|-----------|",
    ]
    for k, tol in TOLERANCE.items():
        v = wm.get(k, float("nan"))
        cmp = "<=" if k != "accuracy" else ">="
        ok = (v >= tol) if k == "accuracy" else (v <= tol)
        lines.append(f"| {k} | {v:.4f} | {cmp}{tol} {'✓' if ok else '⚠'} |")
    if wm:
        lines.append(f"| ece (info only) | {wm.get('ece', float('nan')):.4f} | "
                     "n/a — val⊂train + prob-cap 0.75 → structurally ≥0.20; not a health signal |")

    lines += [
        "",
        "## Props (val-window PIT-KS, in-dist; p > 0.05 = well-calibrated)",
        "",
        "| Prop | KS | p | n | Status |",
        "|------|----|---|---|--------|",
    ]
    if dur_ks_result:
        r = dur_ks_result
        lines.append(f"| duration | {r['ks']:.3f} | {r['p']:.4f} | {r['n']} | "
                     f"{'✓ PASS' if r['pass'] else '⚠ FAIL'} |")
    for pname, r in prop_results.items():
        if "ks" not in r:
            lines.append(f"| {pname} | — | — | — | {r.get('status', '?')} |")
        else:
            lines.append(f"| {pname} | {r['ks']:.3f} | {r['p']:.4f} | {r['n']} | "
                         f"{'✓ PASS' if r['pass'] else '⚠ FAIL'} |")

    lines += [
        "",
        "## Ship rule",
        "",
        "Prod artifacts should be shipped when:",
        "1. Eval-tier Gates A–D PASS (run `loop_engine\\run_gates.py`)",
        "2. Winner val-window acc ≥ 0.62 AND Brier ≤ 0.235 (in-dist sanity floors; in-dist ECE is "
        "structurally ≥0.20 — memorized acc vs 0.75 prob-cap — and is reported for information only)",
        "3. Props look reasonable (in-dist; no catastrophic miscalibration)",
        "",
        f"_Generated by _prod_calibration_report.py on {date.today()}_",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[prod] Report: {report_path}")


if __name__ == "__main__":
    main()
