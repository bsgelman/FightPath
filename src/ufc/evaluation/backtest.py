"""Walk-forward backtest on the test set."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.evaluation.metrics import winner_metrics, roi_vs_line, single_leg_hit_rate
from ufc.evaluation.parlay_backtest import walk_forward_parlay
from ufc.evaluation.calibration import reliability_curve


def run_backtest(winner_model, features_winner: pd.DataFrame) -> dict:
    """Run backtest on test split. Returns metrics dict."""
    splits = get_splits(features_winner)
    test_mask = splits["test"]
    test_df = features_winner[test_mask].dropna(subset=["won_a"]).copy()

    if len(test_df) == 0:
        return {"error": "No test data"}

    # Step 1 fix (F0-B): backtest must match inference, which averages both
    # orderings via predict_symmetric.  Calling predict_proba on A-perspective
    # only produces metrics for a model the user never sees at inference time.
    from ufc.training.symmetrize import symmetrize  # noqa: PLC0415
    test_sym = symmetrize(test_df)
    n = len(test_df)
    p_full = winner_model.predict_proba(test_sym)
    probs = (p_full[:n] + (1.0 - p_full[n:])) / 2.0
    y_true = test_df["won_a"].astype(float).values

    metrics = winner_metrics(y_true, probs)

    # Step 11: correct single-leg diagnostic (no multiplier inflation)
    slhr = single_leg_hit_rate(y_true, probs, payout_type="powerplay_power_3pick",
                               threshold=0.05)
    metrics.update(slhr)
    metrics["n_test"] = int(len(test_df))

    # Step 11: walk-forward parlay simulation (honest ROI by parlay size)
    if "event_date" in test_df.columns:
        parlay_results = walk_forward_parlay(
            test_df.reset_index(drop=True), probs, y_true, threshold=0.05
        )
        metrics["parlay_roi"] = parlay_results

    # Reliability diagram
    report_dir = paths.outputs_reports()
    report_dir.mkdir(parents=True, exist_ok=True)
    reliability_curve(
        y_true, probs,
        title=f"Winner Calibration (Test Set, n={len(test_df)})",
        save_path=report_dir / "calibration_winner.png",
    )

    # ROI curve
    _plot_roi_curve(y_true, probs, report_dir / "roi_curve.png")

    # Step 6: stability slices — also pass WC-temperature-applied probabilities
    # so per-WC Brier reflects the deployed inference pipeline (not raw model).
    probs_wc_temp: np.ndarray | None = None
    if "weight_class" in test_df.columns:
        from ufc.inference.wc_temperature import apply_wc_temperature  # noqa: PLC0415
        wc_series = test_df["weight_class"].reset_index(drop=True)
        probs_wc_temp = np.array([
            apply_wc_temperature(float(p), str(wc))
            for p, wc in zip(probs, wc_series)
        ])

    metrics["stability_slices"] = compute_stability_slices(
        test_df, probs, y_true,
        flag_brier_threshold=0.255,
        probs_wc_temp=probs_wc_temp,
    )

    return metrics


def compute_stability_slices(
    test_df: pd.DataFrame,
    probs: np.ndarray,
    y_true: np.ndarray,
    flag_brier_threshold: float = 0.255,
    probs_wc_temp: np.ndarray | None = None,
) -> dict:
    """Compute Brier/ECE/accuracy slices for stability audit.

    Slices: quarterly, weight_class, title/non-title/5rd, was_debutant.
    Any slice with n >= 50 and Brier > flag_brier_threshold is flagged.
    Bootstrap 95% CI is computed for each slice to distinguish real failures
    from sampling noise (especially important for small WC slices like LHW n=56).

    Returns dict with keys 'slices' (list of row dicts) and 'flags' (list of flag strings).
    """
    from ufc.evaluation.metrics import expected_calibration_error  # noqa: PLC0415
    from sklearn.metrics import brier_score_loss, accuracy_score  # noqa: PLC0415

    df = test_df.reset_index(drop=True).copy()
    probs_arr = np.asarray(probs, dtype=float)
    y_arr = np.asarray(y_true, dtype=float)
    probs_wc_arr = np.asarray(probs_wc_temp, dtype=float) if probs_wc_temp is not None else None

    _bootstrap_rng = np.random.default_rng(42)

    def _brier_bootstrap_ci(yt: np.ndarray, yp: np.ndarray,
                             n_boot: int = 1000) -> tuple[float, float]:
        n = len(yt)
        boots = np.empty(n_boot)
        for i in range(n_boot):
            idx = _bootstrap_rng.integers(0, n, size=n)
            boots[i] = float(brier_score_loss(yt[idx], yp[idx]))
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    def _slice_metrics(bool_mask: np.ndarray,
                       alt_probs: np.ndarray | None = None) -> dict | None:
        n = int(bool_mask.sum())
        if n == 0:
            return None
        yp = (alt_probs if alt_probs is not None else probs_arr)[bool_mask]
        yt = y_arr[bool_mask]
        brier = float(brier_score_loss(yt, yp))
        ci_lo, ci_hi = _brier_bootstrap_ci(yt, yp)
        return {
            "n": n,
            "acc": float(accuracy_score(yt, (yp > 0.5).astype(int))),
            "brier": brier,
            "brier_ci_lo": ci_lo,
            "brier_ci_hi": ci_hi,
            "ece": float(expected_calibration_error(yt, yp)),
        }

    slices: list[dict] = []
    flags: list[str] = []

    def _add(label: str, bool_mask: np.ndarray,
             alt_probs: np.ndarray | None = None) -> None:
        m = _slice_metrics(bool_mask, alt_probs=alt_probs)
        if m is None:
            return
        row = {"slice": label, **m}
        slices.append(row)
        if m["n"] >= 50 and m["brier"] > flag_brier_threshold:
            ci_lo, ci_hi = m["brier_ci_lo"], m["brier_ci_hi"]
            within_noise = ci_lo <= flag_brier_threshold
            noise_note = " [within CI of threshold — likely noise]" if within_noise else ""
            flags.append(
                f"BRIER FLAG: {label} (n={m['n']}) "
                f"Brier={m['brier']:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] "
                f"> {flag_brier_threshold}{noise_note}"
            )

    # ── 1. Quarterly ─────────────────────────────────────────────────────────
    if "event_date" in df.columns:
        quarters = pd.to_datetime(df["event_date"]).dt.to_period("Q").astype(str)
        for q in sorted(quarters.unique()):
            _add(f"Q:{q}", (quarters == q).values)

    # ── 2. Weight class (small classes → "Other") ────────────────────────────
    # Uses temperature-applied probs when available so per-WC Brier reflects
    # the deployed inference pipeline, not raw model output.
    if "weight_class" in df.columns:
        wc_counts = df["weight_class"].value_counts()
        wc_col = df["weight_class"].where(
            df["weight_class"].isin(wc_counts[wc_counts >= 30].index), other="Other"
        )
        for wc in sorted(wc_col.unique()):
            _add(f"WC:{wc}", (wc_col == wc).values,
                 alt_probs=probs_wc_arr)

    # ── 3. Title / 5-round / normal ──────────────────────────────────────────
    if "is_title" in df.columns:
        is_title = df["is_title"].fillna(0).astype(bool).values
        _add("title", is_title)
        _add("non-title", ~is_title)
    if "scheduled_rounds" in df.columns:
        is_5rd = (df["scheduled_rounds"].fillna(3) == 5).values
        _add("5rd", is_5rd)
        _add("3rd", ~is_5rd)

    # ── 4. Debutant fights ───────────────────────────────────────────────────
    # Either fighter has < 2 prior UFC fights at event_date.
    if "fights_career_a" in df.columns and "fights_career_b" in df.columns:
        is_debutant = (
            (df["fights_career_a"].fillna(0) < 2)
            | (df["fights_career_b"].fillna(0) < 2)
        ).values
        _add("debutant", is_debutant)
        _add("veteran", ~is_debutant)

    return {"slices": slices, "flags": flags}


def _plot_roi_curve(y_true, y_prob, save_path: Path,
                     payout_type: str = "powerplay_power_3pick"):
    thresholds = np.linspace(0, 0.15, 30)
    rois, n_bets = [], []
    for thresh in thresholds:
        r = roi_vs_line(y_true, y_prob, payout_type=payout_type, threshold=thresh)
        rois.append(r["roi"])
        n_bets.append(r["n_bets"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(thresholds, rois, "b-o")
    ax1.axhline(0, color="red", linestyle="--")
    ax1.set_xlabel("Edge threshold")
    ax1.set_ylabel("ROI")
    ax1.set_title(f"ROI vs Edge Threshold ({payout_type})")

    ax2.plot(thresholds, n_bets, "g-o")
    ax2.set_xlabel("Edge threshold")
    ax2.set_ylabel("# Bets")
    ax2.set_title("Number of Bets vs Threshold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
