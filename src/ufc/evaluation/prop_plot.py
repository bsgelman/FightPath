"""Per-prop probability distribution plots for predict-time inference."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _slug(s: str) -> str:
    """Lowercase, replace non-alphanumeric runs with '_'."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _is_duration_cdf(cdf) -> bool:
    return hasattr(cdf, "survival") and hasattr(cdf, "median_sec")


def _is_hurdle_cdf(cdf) -> bool:
    return hasattr(cdf, "_p_zero")


def _get_median(cdf) -> float:
    if hasattr(cdf, "median"):
        return float(cdf.median)
    if hasattr(cdf, "median_sec"):
        return float(cdf.median_sec)
    return 0.0


def plot_prop_distribution(
    cdf,
    line: float,
    market: str,
    save_path: Path,
    fighter_label: str = "",
    edge=None,          # Edge dataclass from ufc.valuation.edge; optional
    x_unit: str | None = None,
    title_suffix: str = "",
) -> Path:
    """Plot CDF-derived distribution with shaded P(under)/P(over) and market-line overlay.

    Works for PropCDF, HurdlePropCDF (discrete count), and DurationCDF (continuous).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    if _is_duration_cdf(cdf):
        p_under, p_over = _plot_duration(ax, cdf, line)
        y_label = "Density"
        unit = x_unit or "seconds"
    else:
        p_under, p_over = _plot_count(ax, cdf, line)
        y_label = "Probability mass"
        unit = x_unit or market.replace("_", " ")

    # Market line
    ax.axvline(line, color="black", linestyle="--", linewidth=1.4,
               label=f"Market line: {line}")

    # Median reference line
    med = _get_median(cdf)
    if med > 0:
        ax.axvline(med, color="gray", linestyle=":", linewidth=1.0,
                   label=f"Model median: {med:.1f}")

    ax.set_xlabel(f"{market.replace('_', ' ').title()} ({unit})")
    ax.set_ylabel(y_label)
    title = f"{fighter_label} — {market}  (line {line})"
    if title_suffix:
        title += f"  {title_suffix}"
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8)

    # Edge annotation text box (top-right corner)
    if edge is not None:
        lines_txt = [
            f"P(Under): {p_under:.3f}",
            f"P(Over):  {p_over:.3f}",
            f"Implied:  {edge.implied_prob:.3f}",
            f"Net edge: {edge.edge_pct:+.3f}",
            f"Kelly:    {edge.kelly_fraction:.3f}",
        ]
        ax.text(0.98, 0.97, "\n".join(lines_txt),
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.85, edgecolor="lightgray"))

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _plot_count(ax, cdf, line: float) -> tuple[float, float]:
    """Render discrete PMF for PropCDF / HurdlePropCDF. Returns (p_under, p_over)."""
    try:
        p99_val = float(cdf.quantile(0.99))
    except Exception:
        p99_val = max(line * 2, 10.0)
    x_max = int(max(line * 2.0, p99_val) + 6)
    x = np.arange(0, x_max + 1, dtype=float)

    # PMF via finite difference at integer support (exact for discrete distributions)
    pmf = np.array([cdf.cdf(xi + 0.5) - cdf.cdf(xi - 0.5) for xi in x])
    pmf = np.clip(pmf, 0, None)

    p_under = float(cdf.p_under(line))
    p_over = float(cdf.p_over(line))

    if _is_hurdle_cdf(cdf):
        # Draw a distinct bar for the zero-inflation spike so it's visually clear
        p_zero = float(cdf._p_zero)  # private attr; no public accessor for this value
        pmf[0] = p_zero
        spike_color = "#C0392B" if line > 0 else "#2E86C1"
        ax.bar([0.0], [p_zero], width=0.5, color=spike_color, alpha=0.7,
               label=f"P(zero) = {p_zero:.3f}", zorder=3)
        x_plot = x[1:]
        pmf_plot = pmf[1:]
    else:
        x_plot = x
        pmf_plot = pmf

    mask_under = x_plot < line
    mask_over = x_plot >= line

    ax.fill_between(x_plot, 0, pmf_plot, where=mask_under,
                    color="#C0392B", alpha=0.45,
                    label=f"P(Under) = {p_under:.3f}", step="mid")
    ax.fill_between(x_plot, 0, pmf_plot, where=mask_over,
                    color="#2E86C1", alpha=0.45,
                    label=f"P(Over)  = {p_over:.3f}", step="mid")

    # Thin line + dots to make discrete nature explicit
    ax.step(x_plot, pmf_plot, where="mid", color="dimgray", linewidth=0.9, alpha=0.55)
    ax.plot(x_plot, pmf_plot, "o", color="dimgray", markersize=2.0, alpha=0.45)

    return p_under, p_over


def _plot_duration(ax, cdf, line: float) -> tuple[float, float]:
    """Render continuous PDF for DurationCDF. Returns (p_under, p_over)."""
    sched = float(getattr(cdf, "_scheduled_sec", 900.0))
    x = np.linspace(0.5, sched * 1.02, 800)

    cdf_vals = np.array([cdf.cdf(xi) for xi in x])
    pdf = np.gradient(cdf_vals, x)
    pdf = np.clip(pdf, 0, None)
    area = np.trapezoid(pdf, x)
    if area > 0:
        pdf /= area

    p_under = float(cdf.cdf(line))
    p_over = float(cdf.survival(line))

    mask_under = x < line
    mask_over = x >= line

    ax.fill_between(x, 0, pdf, where=mask_under,
                    color="#C0392B", alpha=0.45,
                    label=f"P(Under) = {p_under:.3f}")
    ax.fill_between(x, 0, pdf, where=mask_over,
                    color="#2E86C1", alpha=0.45,
                    label=f"P(Over)  = {p_over:.3f}")
    ax.plot(x, pdf, color="dimgray", linewidth=1.0, alpha=0.65)

    return p_under, p_over
