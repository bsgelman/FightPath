"""Advisory prop-edge scorecard — resolution, calibration, edge-pick backtest.

NOT a gate. Does not replace or modify Gate B (05_evaluate_props.py).
Uses the identical CDF build path as 05_evaluate_props.py so betting probabilities
match what production serves.

Outputs:
  outputs/reports/prop_edge_scorecard_<date>.md   — human-readable triage table
  outputs/reports/prop_edge_scorecard_<date>.parquet — machine-readable for trend tracking

Run: python scripts/05c_evaluate_prop_edge.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
from datetime import date

import joblib

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.models.props_count import (
    HurdleCountModel, RateHurdleCountModel, ControlShareModel,
)
from ufc.models.props_duration import DurationModel
from ufc.models.method import MethodClassifier, METHOD_CLASSES
from ufc.valuation.payouts import implied_prob_per_leg
from ufc.valuation.prop_menu import SCORECARD_LINES


# ── Breakeven reference ────────────────────────────────────────────────────────
_PP2_BE = implied_prob_per_leg("powerplay_power_2pick", n_legs=2)   # ≈ 0.577
_PP3_BE = implied_prob_per_leg("powerplay_power_3pick", n_legs=3)   # ≈ 0.585
_EDGE_THRESH = 0.05   # model_prob > breakeven + 5% to flag as "edge pick"
_N_BOOT = 2000


# ── CDF helpers ───────────────────────────────────────────────────────────────

def _load_model(model_dir: Path, pattern: str):
    files = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    return joblib.load(files[-1])


def _count_cdfs(
    model_dir: Path,
    pattern: str,
    test_df: pd.DataFrame,
    zero_method_rate_adj: bool = False,
    **predict_kwargs,
) -> list | None:
    """Load count model → CDFs for all test rows. Mirrors gate call in 05_evaluate_props.py."""
    cm = _load_model(model_dir, pattern)
    if cm is None:
        return None

    _orig_adj = None
    if zero_method_rate_adj and isinstance(cm, RateHurdleCountModel):
        _orig_adj = getattr(cm, "method_log_rate_adj", None)
        if _orig_adj is not None:
            cm.method_log_rate_adj = None
    try:
        if isinstance(cm, RateHurdleCountModel):
            cdfs = cm.predict_cdf(test_df, **predict_kwargs)
        elif isinstance(cm, ControlShareModel):
            _ctrl_kw = {k: v for k, v in predict_kwargs.items()
                        if k in ("duration_cdfs", "method_proba", "duration_cdfs_by_method")}
            cdfs = cm.predict_cdf(test_df, **_ctrl_kw)
        else:
            cdfs = cm.predict_cdf(test_df)
    finally:
        if _orig_adj is not None:
            cm.method_log_rate_adj = _orig_adj
    return cdfs


# ── Metrics ───────────────────────────────────────────────────────────────────

def _bss(p: np.ndarray, y: np.ndarray) -> float:
    """Brier skill score vs always-predict-base-rate baseline."""
    base_rate = float(y.mean())
    brier_model = float(np.mean((p - y) ** 2))
    brier_base  = float(np.mean((base_rate - y) ** 2))
    return float(1.0 - brier_model / brier_base) if brier_base > 1e-9 else float("nan")


def _ece_line(p: np.ndarray, y: np.ndarray, n_bins: int = 8) -> float:
    """ECE of p_over(line) vs realized y_over binary."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        frac = mask.sum() / n
        ece += frac * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def _edge_backtest(
    p: np.ndarray, y: np.ndarray, breakeven: float, thresh: float, n_boot: int
) -> dict:
    """Edge-pick backtest: n_picks, hit_rate, 90% CI, flat_roi."""
    mask = p > (breakeven + thresh)
    n_picks = int(mask.sum())
    if n_picks == 0:
        return {
            "n_picks": 0, "hit_rate": float("nan"),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "flat_roi": float("nan"), "edge_proven": False,
        }
    y_sel = y[mask].astype(float)
    hit_rate = float(y_sel.mean())
    flat_roi = float(2.0 * hit_rate - 1.0)   # +1 for win, -1 for loss at 1:1

    rng = np.random.default_rng(42)
    boot = [float(y_sel[rng.integers(n_picks, size=n_picks)].mean()) for _ in range(n_boot)]
    ci_lo, ci_hi = float(np.percentile(boot, 5)), float(np.percentile(boot, 95))
    edge_proven = ci_lo >= breakeven
    return {
        "n_picks": n_picks, "hit_rate": hit_rate,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "flat_roi": flat_roi, "edge_proven": edge_proven,
    }


def _scorecard_per_prop(
    name: str,
    cdfs: list,
    y_true: np.ndarray,
    lines: list[float],
    breakeven: float,
    n_boot: int = _N_BOOT,
) -> list[dict]:
    """Compute scorecard metrics at each characteristic line for one prop."""
    rows = []
    for line in lines:
        p_over = np.array([c.p_over(float(line)) for c in cdfs], dtype=float)
        y_over = (y_true > line).astype(float)
        n = len(y_over)
        base_rate = float(y_over.mean())

        auc     = _auc(p_over, y_over)
        ece     = _ece_line(p_over, y_over)
        bss_val = _bss(p_over, y_over)
        bt      = _edge_backtest(p_over, y_over, breakeven, _EDGE_THRESH, n_boot)

        rows.append({
            "prop":       name,
            "line":       line,
            "n":          n,
            "base_rate":  round(base_rate, 4),
            "auc":        round(auc, 4) if not np.isnan(auc) else float("nan"),
            "ece":        round(ece, 4),
            "brier_skill":round(bss_val, 4) if not np.isnan(bss_val) else float("nan"),
            "n_picks":    bt["n_picks"],
            "hit_rate":   round(bt["hit_rate"], 4) if not np.isnan(bt["hit_rate"]) else float("nan"),
            "ci_lo":      round(bt["ci_lo"], 4) if not np.isnan(bt["ci_lo"]) else float("nan"),
            "ci_hi":      round(bt["ci_hi"], 4) if not np.isnan(bt["ci_hi"]) else float("nan"),
            "flat_roi":   round(bt["flat_roi"], 4) if not np.isnan(bt["flat_roi"]) else float("nan"),
            "breakeven":  round(breakeven, 4),
            "edge_proven":bt["edge_proven"],
        })
    return rows


# ── Summary aggregation ───────────────────────────────────────────────────────

def _prop_summary(rows: list[dict]) -> dict:
    """Aggregate per-line rows into a single per-prop summary (mean across lines)."""
    if not rows:
        return {}
    aucs = [r["auc"] for r in rows if not np.isnan(r["auc"])]
    bss  = [r["brier_skill"] for r in rows if not np.isnan(r["brier_skill"])]
    n_picks_total = sum(r["n_picks"] for r in rows)
    edge_proven   = any(r["edge_proven"] for r in rows)
    mean_hit      = np.mean([r["hit_rate"] for r in rows if not np.isnan(r.get("hit_rate", float("nan")))])

    return {
        "prop":           rows[0]["prop"],
        "mean_auc":       round(float(np.mean(aucs)), 4) if aucs else float("nan"),
        "mean_brier_skill": round(float(np.mean(bss)), 4) if bss else float("nan"),
        "n_picks_total":  n_picks_total,
        "mean_hit_rate":  round(float(mean_hit), 4) if not np.isnan(mean_hit) else float("nan"),
        "breakeven":      rows[0]["breakeven"],
        "edge_proven_any_line": edge_proven,
    }


# ── Report writer ─────────────────────────────────────────────────────────────

def _write_report(all_rows: list[dict], summaries: list[dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Prop Edge Scorecard — {date.today()}",
        "",
        "Advisory diagnostic. NOT a gate — Gate B (PIT-KS) unchanged.",
        "Breakeven = Power Play 2-pick power implied (~0.577). Edge picks = model_prob > BE + 5%.",
        "AUC ≈ 0.50 = no resolution (unpredictable). AUC ≳ 0.55 = real signal.",
        "Edge proven = bootstrap 90% CI lower bound ≥ breakeven.",
        "",
        "## Summary (averaged across characteristic lines)",
        "",
        "| Prop | Mean AUC | Brier Skill | N edge-picks | Mean hit-rate | Breakeven | Edge proven? | Suggested tier |",
        "|------|----------|-------------|--------------|---------------|-----------|--------------|----------------|",
    ]

    for s in summaries:
        auc_str  = f"{s['mean_auc']:.3f}" if not np.isnan(s.get("mean_auc", float("nan"))) else "n/a"
        bss_str  = f"{s['mean_brier_skill']:+.3f}" if not np.isnan(s.get("mean_brier_skill", float("nan"))) else "n/a"
        hr_str   = f"{s['mean_hit_rate']:.3f}" if not np.isnan(s.get("mean_hit_rate", float("nan"))) else "n/a"
        ep       = "✓" if s["edge_proven_any_line"] else "✗"
        auc_val  = s.get("mean_auc", float("nan"))
        bss_val  = s.get("mean_brier_skill", float("nan"))
        # Heuristic tier suggestion (human must confirm and update prop_trust.yaml)
        if not np.isnan(auc_val) and auc_val >= 0.545 and s["edge_proven_any_line"]:
            tier = "TRUST"
        elif not np.isnan(auc_val) and auc_val <= 0.510 and not s["edge_proven_any_line"]:
            tier = "CUT"
        else:
            tier = "WATCH"
        lines.append(
            f"| {s['prop']} | {auc_str} | {bss_str} | {s['n_picks_total']} | {hr_str} | {s['breakeven']:.3f} | {ep} | {tier} |"
        )

    lines += [
        "",
        "> **Action:** Update `configs/prop_trust.yaml` with these suggested tiers after reviewing.",
        "",
        "## Per-line detail",
        "",
        "| Prop | Line | N | Base rate | AUC | ECE | Brier skill | N picks | Hit rate | 90% CI | Edge proven? |",
        "|------|------|---|-----------|-----|-----|-------------|---------|----------|--------|--------------|",
    ]

    for r in all_rows:
        auc_s  = f"{r['auc']:.3f}" if not np.isnan(r["auc"]) else "n/a"
        bss_s  = f"{r['brier_skill']:+.3f}" if not np.isnan(r["brier_skill"]) else "n/a"
        hr_s   = f"{r['hit_rate']:.3f}" if not np.isnan(r.get("hit_rate", float("nan"))) else "n/a"
        ci_s   = (f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                  if not np.isnan(r.get("ci_lo", float("nan"))) else "n/a")
        ep_s   = "✓" if r["edge_proven"] else "✗"
        lines.append(
            f"| {r['prop']} | {r['line']} | {r['n']} | {r['base_rate']:.3f} | "
            f"{auc_s} | {r['ece']:.3f} | {bss_s} | {r['n_picks']} | {hr_s} | {ci_s} | {ep_s} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **AUC**: Does higher model p_over(line) rank actual overs above actual unders?",
        "  - 0.50 = coin flip (no signal at this line).",
        "  - 0.55 = weak but real signal.",
        "  - 0.60+ = meaningful discrimination.",
        "- **Brier skill**: Positive = model beats always-predict-base-rate. Negative = worse than naive.",
        "- **Edge picks**: Rows where model_prob > breakeven + 5%. Hit-rate CI spans whether",
        "  these picks would be profitable at the structural DFS breakeven.",
        "- **Edge proven**: CI lower bound ≥ breakeven → statistically confirmed edge on test set.",
        "",
        f"_Generated by scripts/05c_evaluate_prop_edge.py on {date.today()}_",
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Scorecard written to {save_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Prop Edge Scorecard (advisory, not a gate) ===")
    model_dir  = paths.outputs_models()
    report_dir = paths.outputs_reports()
    report_dir.mkdir(parents=True, exist_ok=True)

    props_df = parquet.read(paths.processed("features_props"))
    props_df["event_date"] = pd.to_datetime(props_df["event_date"])
    splits   = get_splits(props_df)
    test_df  = props_df[splits["test"]].copy().reset_index(drop=True)
    print(f"  Test set: {len(test_df)} rows")

    # ── Method classifier ─────────────────────────────────────────────────────
    method_proba = None
    mc_files = sorted(model_dir.glob("method_clf_*.joblib"), key=lambda p: p.stat().st_mtime)
    if mc_files:
        mc = MethodClassifier.load(mc_files[-1])
        probs = mc.predict_proba_dict(test_df)
        method_proba = np.column_stack([probs[c] for c in METHOD_CLASSES])
        print(f"  Method proba loaded (mean KO={method_proba[:,0].mean():.3f})")

    # ── Duration CDFs ─────────────────────────────────────────────────────────
    dur_cdfs_full      = None
    dur_cdfs_by_method = None
    dur_files = sorted(model_dir.glob("props_duration_*.joblib"), key=lambda p: p.stat().st_mtime)
    if dur_files:
        dm = DurationModel.load(dur_files[-1])
        dur_cdfs_full = dm.predict_cdf(test_df, use_boundary_mass=False)
        if hasattr(dm, "feature_cols") and "method_ko" in dm.feature_cols:
            dur_cdfs_by_method = {
                m: dm.predict_cdf(test_df, method_override=m, use_boundary_mass=False)
                for m in ("KO/TKO", "SUB", "DEC")
            }
        print("  Duration CDFs built.")

    breakeven = _PP2_BE
    all_rows: list[dict] = []

    # ── Per-prop evaluation ───────────────────────────────────────────────────
    # Call signatures match 05_evaluate_props.py exactly for production parity.

    _base_kw = dict(
        duration_cdfs=dur_cdfs_full,
        method_proba=method_proba,
        duration_cdfs_by_method=dur_cdfs_by_method,
    )

    prop_configs: list[dict] = [
        # (name, pattern, raw_col, zero_method_rate_adj, extra_predict_kwargs)
        dict(name="sig_strikes",
             pattern="props_sig_strikes_*.joblib",
             col="sig_str_landed_a",
             zero_adj=False,
             kw=dict(**_base_kw)),

        dict(name="takedowns",
             pattern="props_takedowns_*.joblib",
             col="td_landed_a",
             zero_adj=True,
             kw=dict(**_base_kw,
                     apply_method_hurdle=False,
                     use_binned_rate_adj=False,
                     use_cond_hurdle=False,
                     use_sub_count_head=True)),

        dict(name="r1_sig_strikes",
             pattern="props_r1_sig_strikes_*.joblib",
             col="r1_sig_str_landed_a",
             zero_adj=False,
             kw=dict(duration_cdfs=dur_cdfs_full,
                     active_minutes_ceiling=5.0,
                     method_proba=method_proba,
                     apply_burst=False,
                     use_finish_head=True)),

        dict(name="knockdowns",
             pattern="props_knockdowns_*.joblib",
             col="kd_for_a",
             zero_adj=True,
             kw=dict(**_base_kw,
                     apply_method_hurdle=False,
                     use_binned_rate_adj=False,
                     use_cond_hurdle=False,
                     use_sub_count_head=False)),

        dict(name="sub_attempts",
             pattern="props_sub_attempts_*.joblib",
             col="sub_att_for_a",
             zero_adj=True,
             kw=dict(**_base_kw,
                     apply_method_hurdle=False,
                     use_binned_rate_adj=False,
                     use_cond_hurdle=False,
                     use_sub_count_head=True)),

        dict(name="r1_takedowns",
             pattern="props_r1_takedowns_*.joblib",
             col="r1_td_landed_a",
             zero_adj=False,
             kw=dict(duration_cdfs=dur_cdfs_full,
                     active_minutes_ceiling=5.0,
                     method_proba=method_proba,
                     apply_burst=False,
                     use_finish_head=True)),

        dict(name="body_sig_strikes",
             pattern="props_body_sig_strikes_*.joblib",
             col="body_landed_a",
             zero_adj=False,
             kw=dict(**_base_kw)),

        dict(name="leg_sig_strikes",
             pattern="props_leg_sig_strikes_*.joblib",
             col="leg_landed_a",
             zero_adj=True,
             kw=dict(**_base_kw)),

        dict(name="ctrl_time",
             pattern="props_ctrl_time_*.joblib",
             col="ctrl_sec_a",
             zero_adj=True,
             kw=dict(**_base_kw,
                     apply_method_hurdle=False,
                     use_cond_hurdle=False)),
    ]

    for cfg in prop_configs:
        name    = cfg["name"]
        lines   = SCORECARD_LINES.get(name, [])
        if not lines:
            print(f"\n  [{name}] no scorecard lines defined — skip")
            continue

        print(f"\n  [{name}] building CDFs...")
        cdfs = _count_cdfs(
            model_dir, cfg["pattern"], test_df,
            zero_method_rate_adj=cfg["zero_adj"],
            **cfg["kw"],
        )
        if cdfs is None:
            print(f"  [{name}] model not found — skip")
            continue

        y_true = test_df[cfg["col"]].fillna(0).values.astype(float)
        rows   = _scorecard_per_prop(name, cdfs, y_true, lines, breakeven)
        all_rows.extend(rows)
        for r in rows:
            hr = f"{r['hit_rate']:.3f}" if not np.isnan(r.get("hit_rate", float("nan"))) else "n/a"
            ep = "+edge" if r["edge_proven"] else "  -"
            auc_s = f"{r['auc']:.3f}" if not np.isnan(r["auc"]) else "n/a"
            print(f"    line={r['line']:6.1f}  n={r['n']:4d}  base={r['base_rate']:.3f}  "
                  f"AUC={auc_s}  BSS={r['brier_skill']:+.3f}  "
                  f"n_picks={r['n_picks']:3d}  hit={hr}  {ep}")

    # ── Duration (deduped per fight) ──────────────────────────────────────────
    dur_lines = SCORECARD_LINES.get("duration", [])
    if dur_files and dur_cdfs_full is not None and dur_lines:
        print("\n  [duration] building CDFs...")
        dur_test = test_df.drop_duplicates(subset=["fight_id"]).dropna(
            subset=["total_fight_sec"]
        ).reset_index(drop=True)
        dur_cdfs_eval = dm.predict_cdf(dur_test, use_boundary_mass=False)
        y_dur = dur_test["total_fight_sec"].clip(lower=1).values.astype(float)
        rows  = _scorecard_per_prop("duration", dur_cdfs_eval, y_dur, dur_lines, breakeven)
        all_rows.extend(rows)
        for r in rows:
            hr    = f"{r['hit_rate']:.3f}" if not np.isnan(r.get("hit_rate", float("nan"))) else "n/a"
            ep    = "+edge" if r["edge_proven"] else "  -"
            auc_s = f"{r['auc']:.3f}" if not np.isnan(r["auc"]) else "n/a"
            print(f"    line={r['line']:6.0f}s  n={r['n']:4d}  base={r['base_rate']:.3f}  "
                  f"AUC={auc_s}  BSS={r['brier_skill']:+.3f}  "
                  f"n_picks={r['n_picks']:3d}  hit={hr}  {ep}")

    # ── Summaries ─────────────────────────────────────────────────────────────
    prop_names = list(dict.fromkeys(r["prop"] for r in all_rows))
    summaries  = []
    for pn in prop_names:
        prows = [r for r in all_rows if r["prop"] == pn]
        summaries.append(_prop_summary(prows))

    print("\n\n=== Scorecard Summary ===")
    print(f"  {'Prop':<22} {'AUC':>6} {'BSS':>7} {'NPicksTotal':>11} {'EdgeProven':>10}")
    print("  " + "-" * 60)
    for s in summaries:
        auc_s = f"{s['mean_auc']:.3f}" if not np.isnan(s.get("mean_auc", float("nan"))) else " n/a"
        bss_s = f"{s['mean_brier_skill']:+.3f}" if not np.isnan(s.get("mean_brier_skill", float("nan"))) else "  n/a"
        ep_s  = "YES" if s["edge_proven_any_line"] else "no"
        print(f"  {s['prop']:<22} {auc_s:>6} {bss_s:>7} {s['n_picks_total']:>11} {ep_s:>10}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    today = date.today().isoformat()
    md_path = report_dir / f"prop_edge_scorecard_{today}.md"
    _write_report(all_rows, summaries, md_path)

    pq_path = report_dir / f"prop_edge_scorecard_{today}.parquet"
    pd.DataFrame(all_rows).to_parquet(pq_path, index=False)
    print(f"  Parquet written to {pq_path.name}")

    print("\n  Update configs/prop_trust.yaml with the suggested tiers above.")


if __name__ == "__main__":
    main()
