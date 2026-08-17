"""Honest walk-forward backtest of the SERVED (prod) winner methodology.

The prod tier (outputs/models/prod/, configs/split_prod.yaml) trains on ALL data
through the latest card, so it has NO held-out test set — scoring it on recent
fights is in-sample/memorized (see _prod_calibration_report.py). The eval tier is
the honest yardstick the Gates evaluate, but it is NOT the served model.

This script gives the served *methodology* an honest out-of-sample check: it
replays the rolling prod split at several historical cutoffs, retraining the
winner head "as-of" each cutoff (prod-style: temporal-OOF calibration, val = the
6 months before the cutoff) and scoring ONLY strictly-future fights in the next
window. Those predictions are genuinely out-of-sample, so the pooled
accuracy / Brier / ECE is an honest estimate of what the prod winner head does
on fights it has never seen.

Design constraints (keeps the eval tier and Gates A-D byte-identical):
  * never sets UFC_SPLIT_CONFIG and never reads/writes configs/split*.yaml
  * builds split masks INLINE (does not call get_splits)
  * never writes into outputs/models/ or outputs/models/prod/ (scratch joblibs
    live under outputs/reports/walkforward/_models/, a non-LFS dir)
  * only reads the precomputed features_winner parquet and reuses eval utilities

Leakage note (verified before building): every fitted feature transform in
assemble.py (style scores, opponent-quality, sparse-history fill) is fit on the
<=2023 train fold, which is strictly in the past relative to every cutoff here
(>= 2024-06); rolling features are causal (shift(1).expanding); there are no
pca_style_* columns in the winner frame. So no future information leaks into any
cutoff's training data — the result is honest (mildly conservative).

Run:
    python scripts/_prod_walkforward_backtest.py
    python scripts/_prod_walkforward_backtest.py --refit   # ignore cached models

This is a diagnostic, NOT a Gate. Pooled floors mirror the prod calibration
tolerances (_prod_calibration_report.py): Brier <= 0.235, ECE <= 0.07,
accuracy >= 0.60. Exits non-zero if a pooled floor fails (advisory ship-gate).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.io import paths, parquet
from ufc.features.interactions import compute_interactions
from ufc.training.symmetrize import symmetrize
from ufc.training.recency import search_halflife_winner
from ufc.models.winner import WinnerModel, get_winner_feature_cols
from ufc.evaluation.metrics import winner_metrics
from ufc.evaluation.calibration import reliability_curve

# Cutoffs kept <= 2025-12 so each 6-month window is fully resolved (won_a present).
CUTOFFS = ["2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31"]
HORIZON_MONTHS = 6
TRAIN_START = "2010-01-01"  # matches prod resolve_prod_split_dates
HALFLIFE_GRID = [730, 1095, 1460, 1825, None]

# Pooled-OOS floors — mirror _prod_calibration_report.py TOLERANCE (prod tier,
# NOT the stricter eval Gate-A floors of Brier<=0.225 / ECE<=0.05).
FLOORS = {"accuracy": 0.60, "brier": 0.235, "ece": 0.07}
# Per-window diagnostic flags (noisier; advisory only).
WINDOW_FLAG_BRIER = 0.25
WINDOW_FLAG_ECE = 0.10


def _val_windows(cutoff: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Mirror scripts/03_train_prod.py:resolve_prod_split_dates for the 6mo val window."""
    val_start = cutoff - pd.DateOffset(months=HORIZON_MONTHS)
    val_mid = val_start + (cutoff - val_start) / 2
    return val_start, val_mid, cutoff


def _fit_winner_as_of(df: pd.DataFrame, feature_cols: list[str],
                      cutoff: pd.Timestamp, scratch_dir: Path,
                      refit: bool) -> WinnerModel:
    """Replicate the winner block of train_all.py:116-166 at an arbitrary cutoff.

    Uses prod-mode temporal-OOF calibration (self-derives its window from the
    training frame's own max date, so it is honestly 'as-of cutoff').
    """
    tag = cutoff.strftime("%Y-%m-%d")
    cached = scratch_dir / f"winner_ensemble_{tag}.joblib"
    if cached.exists() and not refit:
        print(f"  [cache] loading {cached.name}")
        return WinnerModel.load(cached)

    val_start, val_mid, _ = _val_windows(cutoff)
    dates = df["event_date"]

    train_mask = (dates >= TRAIN_START) & (dates <= cutoff)
    val_a_mask = (dates >= val_start) & (dates < val_mid)
    val_b_mask = (dates >= val_mid) & (dates <= cutoff)

    # Train (symmetrized, drop unresolved labels)
    train_df = df[train_mask].copy()
    train_sym = symmetrize(train_df)
    y_train_sym = train_sym["won_a"].astype(float)
    keep = y_train_sym.notna()
    train_sym, y_train_sym = train_sym[keep], y_train_sym[keep]

    val_a_df = df[val_a_mask].copy()
    y_val_a = val_a_df["won_a"].astype(float)
    val_a_df, y_val_a = val_a_df[y_val_a.notna()], y_val_a[y_val_a.notna()]

    val_b_df = df[val_b_mask].copy()
    y_val_b = val_b_df["won_a"].astype(float)
    val_b_df, y_val_b = val_b_df[y_val_b.notna()], y_val_b[y_val_b.notna()]

    print(f"  Train(sym)={len(train_sym)}  Val-A={len(val_a_df)}  Val-B={len(val_b_df)}")

    best_h, sw = search_halflife_winner(
        train_sym, y_train_sym, val_a_df, y_val_a,
        feature_cols=feature_cols,
        train_dates=train_sym["event_date"],
        grid=HALFLIFE_GRID,
        anchor=tag,
        ece_cap=0.04,
        brier_floor_margin=0.001,
        verbose=False,
    )
    print(f"  halflife={best_h}")

    model = WinnerModel()
    model.fit(train_sym, y_train_sym, val_a_df, y_val_a, feature_cols,
              sample_weight=sw,
              X_val_platt=val_b_df, y_val_platt=y_val_b,
              temporal_oof=True,
              train_dates=train_sym["event_date"])
    scratch_dir.mkdir(parents=True, exist_ok=True)
    model.save(scratch_dir, tag)
    return model


def _score_window(model: WinnerModel, future_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrize -> predict both orderings -> average (matches backtest.py:31-37)."""
    future_sym = symmetrize(future_df)
    n = len(future_df)
    p_full = model.predict_proba(future_sym)
    probs = (p_full[:n] + (1.0 - p_full[n:])) / 2.0
    y_true = future_df["won_a"].astype(float).values
    return y_true, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true",
                    help="ignore cached per-cutoff models and retrain")
    ap.add_argument("--cutoffs", default=None,
                    help="comma-separated cutoff override (default: all of CUTOFFS). "
                         "Each prod-style winner retrain is heavy, so a partial run "
                         "(e.g. the cached cutoffs) gives a fast honest read.")
    args = ap.parse_args()

    cutoffs = [c.strip() for c in args.cutoffs.split(",")] if args.cutoffs else CUTOFFS

    print("=== Prod Walk-Forward Backtest (winner, honest OOS) ===")

    df = parquet.read(paths.processed("features_winner"))
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = compute_interactions(df)  # parity with train_all.py:110-113
    feature_cols = get_winner_feature_cols(df)
    print(f"  features_winner rows={len(df)}  feature_cols={len(feature_cols)}")

    scratch_dir = paths.outputs_reports() / "walkforward" / "_models"

    rows = []
    pooled_y, pooled_p = [], []
    for c_str in cutoffs:
        cutoff = pd.Timestamp(c_str)
        horizon_end = cutoff + pd.DateOffset(months=HORIZON_MONTHS)
        future_df = df[(df["event_date"] > cutoff) & (df["event_date"] <= horizon_end)] \
            .dropna(subset=["won_a"]).copy()
        print(f"\n[cutoff {c_str}]  future=({c_str}, {horizon_end.date()}]  n={len(future_df)}")
        if len(future_df) == 0:
            print("  (no future fights — skipping)")
            continue

        model = _fit_winner_as_of(df, feature_cols, cutoff, scratch_dir, args.refit)
        y_true, probs = _score_window(model, future_df)
        m = winner_metrics(y_true, probs)
        m["n"] = int(len(y_true))
        m["cutoff"] = c_str
        flagged = m["brier"] > WINDOW_FLAG_BRIER or m["ece"] > WINDOW_FLAG_ECE
        m["flag"] = flagged
        rows.append(m)
        pooled_y.append(y_true)
        pooled_p.append(probs)
        print(f"  acc={m['accuracy']:.4f}  brier={m['brier']:.4f}  ece={m['ece']:.4f}  "
              f"logloss={m['log_loss']:.4f}{'  [FLAG]' if flagged else ''}")

    if not pooled_y:
        print("No scored windows — aborting.")
        sys.exit(1)

    pooled_y = np.concatenate(pooled_y)
    pooled_p = np.concatenate(pooled_p)
    pooled = winner_metrics(pooled_y, pooled_p)
    pooled["n"] = int(len(pooled_y))

    report_dir = paths.outputs_reports() / "walkforward"
    report_dir.mkdir(parents=True, exist_ok=True)
    reliability_curve(
        pooled_y, pooled_p,
        title=f"Prod Walk-Forward Winner Calibration (pooled OOS, n={pooled['n']})",
        save_path=report_dir / "reliability.png",
    )

    passes = {
        "accuracy": pooled["accuracy"] >= FLOORS["accuracy"],
        "brier": pooled["brier"] <= FLOORS["brier"],
        "ece": pooled["ece"] <= FLOORS["ece"],
    }
    all_pass = all(passes.values())

    print(f"\n[pooled OOS] n={pooled['n']}  acc={pooled['accuracy']:.4f}  "
          f"brier={pooled['brier']:.4f}  ece={pooled['ece']:.4f}  "
          f"-> {'PASS' if all_pass else 'FAIL'}")

    _write_report(rows, pooled, passes, all_pass)

    if not all_pass:
        sys.exit(1)


def _write_report(rows: list[dict], pooled: dict, passes: dict, all_pass: bool) -> None:
    report_path = paths.outputs_reports() / f"prod_walkforward_{date.today()}.md"
    lines = [
        f"# Prod Walk-Forward Backtest (winner) — {date.today()}",
        "",
        "**Honest out-of-sample check of the SERVED winner methodology.** Each cutoff "
        "retrains the winner head prod-style (temporal-OOF calibration, val = the 6 months "
        "before the cutoff, train = all data through the cutoff) and scores ONLY the next "
        f"{HORIZON_MONTHS}-month window of strictly-future fights. Pooling those windows gives "
        "genuinely out-of-sample metrics — unlike `_prod_calibration_report.py`, whose val "
        "window overlaps training.",
        "",
        "Diagnostic, not a Gate. Eval-tier Gates A-D remain the correctness proof "
        "(this script never touches eval training or split configs).",
        "",
        "## Per-cutoff windows (out-of-sample)",
        "",
        "| Cutoff | Window n | Accuracy | Brier | ECE | Log-loss | AUROC | Flag |",
        "|--------|----------|----------|-------|-----|----------|-------|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['cutoff']} | {r['n']} | {r['accuracy']:.4f} | {r['brier']:.4f} | "
            f"{r['ece']:.4f} | {r['log_loss']:.4f} | {r['auroc']:.4f} | "
            f"{'⚠' if r['flag'] else ''} |"
        )
    lines += [
        "",
        "## Pooled OOS (the headline number)",
        "",
        "| Metric | Pooled | Floor | Status |",
        "|--------|--------|-------|--------|",
        f"| accuracy | {pooled['accuracy']:.4f} | >={FLOORS['accuracy']} | "
        f"{'✓' if passes['accuracy'] else '⚠'} |",
        f"| brier | {pooled['brier']:.4f} | <={FLOORS['brier']} | "
        f"{'✓' if passes['brier'] else '⚠'} |",
        f"| ece | {pooled['ece']:.4f} | <={FLOORS['ece']} | "
        f"{'✓' if passes['ece'] else '⚠'} |",
        f"| log_loss | {pooled['log_loss']:.4f} | — | — |",
        f"| auroc | {pooled['auroc']:.4f} | — | — |",
        f"| n | {pooled['n']} | — | — |",
        "",
        f"**Verdict: {'PASS' if all_pass else 'FAIL'}** "
        f"(pooled floors mirror `_prod_calibration_report.py`; per-window rows are "
        f"diagnostics — flag = Brier>{WINDOW_FLAG_BRIER} or ECE>{WINDOW_FLAG_ECE}).",
        "",
        "Pooled OOS is expected to be modestly worse than the in-sample prod val numbers; "
        "that gap is the previously-missing honest signal.",
        "",
        "![reliability](walkforward/reliability.png)",
        "",
        f"_Generated by scripts/_prod_walkforward_backtest.py on {date.today()}_",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
