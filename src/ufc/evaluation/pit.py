"""PIT (Probability Integral Transform) histogram for CDF calibration."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def pit_histogram(
    y_true: np.ndarray,
    cdf_values: np.ndarray,
    title: str = "PIT Histogram",
    save_path: Path | None = None,
) -> dict:
    """Compute PIT and plot histogram.

    cdf_values[i] = P(X <= y_true[i]) from the model's CDF for observation i.
    Under a well-calibrated model, PIT values should be uniform on [0, 1].
    """
    pit = np.clip(cdf_values, 0, 1)

    ks_stat, ks_p = stats.kstest(pit, "uniform")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pit, bins=20, density=True, alpha=0.7, label="PIT")
    ax.axhline(1.0, color="red", linestyle="--", label="Uniform")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\nKS stat={ks_stat:.3f}, p={ks_p:.3f}")
    ax.legend()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {"ks_stat": float(ks_stat), "ks_p": float(ks_p), "pit": pit.tolist()}


def pit_histogram_segmented(
    y_true: np.ndarray,
    cdf_values: np.ndarray,
    segments: dict[str, np.ndarray],
    title: str = "PIT Histogram",
    save_path: Path | None = None,
    segment_values_override: dict[str, np.ndarray] | None = None,
) -> dict:
    """Compute PIT segmented by fight-outcome subgroups, plot multi-panel histogram.

    Parameters
    ----------
    y_true : array-like
        True observed values (unused in KS computation but kept for API symmetry).
    cdf_values : array-like
        Randomized PIT values: CDF evaluated at y_true[i], already randomized
        for discrete zero mass by the caller. Should be ~Uniform(0,1) if calibrated.
    segments : dict[str, np.ndarray]
        Mapping of segment name -> boolean array of length len(cdf_values).
        Each True entry selects observations belonging to that segment.
        Segments may overlap (e.g. "finish" and "5rd" share some rows).
        Segments with fewer than 20 observations get KS=NaN.
    title : str
        Main title displayed on the overall PIT panel.
    save_path : Path, optional
        If provided, saves the figure here (parent dirs created automatically).

    Returns
    -------
    dict with keys:
        ks_stat  : float — overall KS statistic
        ks_p     : float — overall KS p-value
        pit      : list[float] — clipped PIT values
        segments : dict[str, dict] — per-segment {ks_stat, ks_p, n}

    Notes
    -----
    ``segment_values_override`` lets callers supply per-segment PIT arrays that
    differ from the slice of ``cdf_values``.  This is used by the duration model
    diagnostic (Step 3) to renormalise finish/decision segments onto [0,1] before
    KS-testing them.  The overall aggregate KS always uses raw ``cdf_values``.
    """
    pit = np.clip(cdf_values, 0, 1)
    ks_stat, ks_p = stats.kstest(pit, "uniform")

    # Per-segment KS — use override values if supplied, else slice from pit
    seg_results: dict[str, dict] = {}
    for seg_name, mask in segments.items():
        mask = np.asarray(mask, dtype=bool)
        if segment_values_override and seg_name in segment_values_override:
            seg_pit = np.clip(segment_values_override[seg_name], 0.0, 1.0)
        else:
            seg_pit = pit[mask]
        n = int(mask.sum())
        if n >= 20:
            seg_ks, seg_p = stats.kstest(seg_pit, "uniform")
        else:
            seg_ks, seg_p = float("nan"), float("nan")
        seg_results[seg_name] = {
            "ks_stat": float(seg_ks),
            "ks_p": float(seg_p),
            "n": n,
        }

    # ── Figure layout ────────────────────────────────────────────────────────
    # Row 0: single overall PIT panel (spans all columns via colspan trick).
    # Row 1: one panel per segment.
    n_segs = max(1, len(segments))
    fig = plt.figure(figsize=(4 * n_segs, 8))
    gs = fig.add_gridspec(2, n_segs, hspace=0.45, wspace=0.35)

    # Overall PIT in first cell of row 0; hide the rest of row 0
    ax_overall = fig.add_subplot(gs[0, :])  # span all columns
    ax_overall.hist(pit, bins=20, density=True, alpha=0.7, color="steelblue", label="PIT")
    ax_overall.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Uniform")
    ax_overall.set_xlabel("PIT value")
    ax_overall.set_ylabel("Density")
    ax_overall.set_title(
        f"{title} — overall\nKS={ks_stat:.3f}, p={ks_p:.4f}, n={len(pit)}"
    )
    ax_overall.legend(fontsize=8)

    # Per-segment row
    for j, (seg_name, mask) in enumerate(segments.items()):
        mask = np.asarray(mask, dtype=bool)
        if segment_values_override and seg_name in segment_values_override:
            seg_pit = np.clip(segment_values_override[seg_name], 0.0, 1.0)
        else:
            seg_pit = pit[mask]
        ax = fig.add_subplot(gs[1, j])
        sr = seg_results[seg_name]
        if sr["n"] >= 5:
            ax.hist(seg_pit, bins=15, density=True, alpha=0.75, color="darkorange")
            ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0)
        else:
            ax.text(0.5, 0.5, "n < 5", ha="center", va="center", transform=ax.transAxes)
        ks_str = f"{sr['ks_stat']:.3f}" if not np.isnan(sr["ks_stat"]) else "n/a"
        p_str = f"{sr['ks_p']:.3f}" if not np.isnan(sr["ks_p"]) else "n/a"
        ax.set_title(f"{seg_name} (n={sr['n']})\nKS={ks_str}, p={p_str}", fontsize=9)
        ax.set_xlabel("PIT value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "pit": pit.tolist(),
        "segments": seg_results,
    }
