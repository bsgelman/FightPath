"""Global feature-importance diagnostic charts for all trained models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_top_n(
    importances: np.ndarray,
    feature_cols: list[str],
    title: str,
    save_path: Path,
    top_n: int = 30,
    x_label: str = "Importance",
    csv_path: Path | None = None,
) -> None:
    """Sort by importance, horizontal bar chart, savefig + optional full CSV."""
    fi = np.array(importances, dtype=float)
    fc = list(feature_cols)
    if len(fi) != len(fc):
        raise ValueError(f"importance length {len(fi)} != feature_cols length {len(fc)}")

    order = np.argsort(fi)[::-1]
    sorted_cols = [fc[i] for i in order]
    sorted_fi = fi[order]

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"feature": sorted_cols, "importance": sorted_fi}).to_csv(csv_path, index=False)

    n = min(top_n, len(sorted_cols))
    top_cols = sorted_cols[:n][::-1]   # reverse so highest is at top of chart
    top_fi = sorted_fi[:n][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * n + 1)))
    y_pos = np.arange(n)
    ax.barh(y_pos, top_fi, color="#2E86C1", alpha=0.82)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_cols, fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_title(title, fontsize=10, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_winner_importance(winner_model, out_dir: Path, gitsha: str, top_n: int = 30) -> list[Path]:
    """WinnerModel — importance from the final LGBM estimator."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat = list(winner_model.feature_cols)
    saved: list[Path] = []

    try:
        # WinnerModel stores .lgbm directly
        lgbm_model = getattr(winner_model, "lgbm", None)
        if lgbm_model is None or not hasattr(lgbm_model, "feature_importances_"):
            return saved
        fi = np.array(lgbm_model.feature_importances_, dtype=float)
        stem = out_dir / f"winner_lgbm_TOP{top_n}_{gitsha}"
        _plot_top_n(fi, feat, f"Winner — LGBM Top {top_n} ({gitsha})",
                    stem.with_suffix(".png"), top_n=top_n,
                    x_label="Importance (gain)",
                    csv_path=out_dir / f"winner_lgbm_TOP{top_n}_{gitsha}_full.csv")
        saved.append(stem.with_suffix(".png"))
    except Exception as e:
        print(f"  [warn] winner importance: {e}")

    return saved


def plot_method_importance(method_clf, out_dir: Path, gitsha: str, top_n: int = 30) -> Path | None:
    """Single chart from the LGBM method classifier."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(method_clf, "model", None) is None:
        print("  [warn] method_clf.model is None, skipping importance plot")
        return None
    try:
        fi = np.array(method_clf.model.feature_importances_, dtype=float)
        feat = list(method_clf.feature_cols)
        stem = out_dir / f"method_TOP{top_n}_{gitsha}"
        _plot_top_n(fi, feat, f"Method Classifier Top {top_n} ({gitsha})",
                    stem.with_suffix(".png"), top_n=top_n,
                    x_label="Importance (gain)",
                    csv_path=out_dir / f"method_TOP{top_n}_{gitsha}_full.csv")
        return stem.with_suffix(".png")
    except Exception as e:
        print(f"  [warn] method importance: {e}")
        return None


def plot_count_model_importance(
    model,
    out_dir: Path,
    gitsha: str,
    top_n: int = 30,
    target: str | None = None,
) -> list[Path]:
    """CountModel → 1 chart.  HurdleCountModel → 2 charts (quantile + hurdle stages)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = target or getattr(model, "target", "unknown")
    feat = list(model.feature_cols)
    saved: list[Path] = []

    # Quantile regressors — mean importance across all quantile regressors
    q_models = (getattr(model, "lgbm_quantile_models", None)
                or getattr(model, "quantile_models", None))
    if q_models:
        try:
            all_fi = np.stack([np.array(m.feature_importances_, dtype=float)
                               for m in q_models], axis=0)
            avg_fi = all_fi.mean(axis=0)
            stem = out_dir / f"props_{name}_quantile_TOP{top_n}_{gitsha}"
            _plot_top_n(avg_fi, feat,
                        f"{name} — Quantile Regressors (mean across grid) Top {top_n} ({gitsha})",
                        stem.with_suffix(".png"), top_n=top_n,
                        x_label="Mean importance across quantiles",
                        csv_path=out_dir / f"props_{name}_quantile_TOP{top_n}_{gitsha}_full.csv")
            saved.append(stem.with_suffix(".png"))
        except Exception as e:
            print(f"  [warn] {name} quantile importance: {e}")

    # Hurdle binary stage (HurdleCountModel only)
    pos_clf = getattr(model, "pos_clf", None)
    if pos_clf is not None:
        try:
            fi = np.array(pos_clf.feature_importances_, dtype=float)
            stem = out_dir / f"props_{name}_hurdle_TOP{top_n}_{gitsha}"
            _plot_top_n(fi, feat,
                        f"{name} — Hurdle Classifier (P>0) Top {top_n} ({gitsha})",
                        stem.with_suffix(".png"), top_n=top_n,
                        x_label="Importance (gain)",
                        csv_path=out_dir / f"props_{name}_hurdle_TOP{top_n}_{gitsha}_full.csv")
            saved.append(stem.with_suffix(".png"))
        except Exception as e:
            print(f"  [warn] {name} hurdle importance: {e}")

    return saved


def plot_duration_importance(dur_model, out_dir: Path, gitsha: str, top_n: int = 30) -> list[Path]:
    """Two charts: decision classifier + finish quantile regressors. Weibull AFT skipped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat = list(dur_model.feature_cols)
    saved: list[Path] = []

    if getattr(dur_model, "dec_clf", None) is not None:
        try:
            fi = np.array(dur_model.dec_clf.feature_importances_, dtype=float)
            stem = out_dir / f"duration_dec_clf_TOP{top_n}_{gitsha}"
            _plot_top_n(fi, feat,
                        f"Duration — P(Decision) Classifier Top {top_n} ({gitsha})",
                        stem.with_suffix(".png"), top_n=top_n,
                        x_label="Importance (gain)",
                        csv_path=out_dir / f"duration_dec_clf_TOP{top_n}_{gitsha}_full.csv")
            saved.append(stem.with_suffix(".png"))
        except Exception as e:
            print(f"  [warn] duration dec_clf importance: {e}")

    if getattr(dur_model, "lgbm_quantile_models", None):
        try:
            all_fi = np.stack([np.array(m.feature_importances_, dtype=float)
                               for m in dur_model.lgbm_quantile_models], axis=0)
            avg_fi = all_fi.mean(axis=0)
            stem = out_dir / f"duration_finish_quantile_TOP{top_n}_{gitsha}"
            _plot_top_n(avg_fi, feat,
                        f"Duration — Finish Quantile Regressors (mean) Top {top_n} ({gitsha})",
                        stem.with_suffix(".png"), top_n=top_n,
                        x_label="Mean importance across quantiles",
                        csv_path=out_dir / f"duration_finish_quantile_TOP{top_n}_{gitsha}_full.csv")
            saved.append(stem.with_suffix(".png"))
        except Exception as e:
            print(f"  [warn] duration quantile importance: {e}")

    # v8.27: method-specific quantile models (KO / SUB)
    for _method_tag, _attr in [("KO", "lgbm_quantile_models_ko"), ("SUB", "lgbm_quantile_models_sub")]:
        _models = getattr(dur_model, _attr, None)
        if _models:
            try:
                all_fi = np.stack([np.array(m.feature_importances_, dtype=float)
                                   for m in _models], axis=0)
                avg_fi = all_fi.mean(axis=0)
                stem = out_dir / f"duration_finish_{_method_tag}_quantile_TOP{top_n}_{gitsha}"
                _plot_top_n(avg_fi, feat,
                            f"Duration — {_method_tag} Finish Quantile (mean) Top {top_n} ({gitsha})",
                            stem.with_suffix(".png"), top_n=top_n,
                            x_label="Mean importance across quantiles",
                            csv_path=out_dir / f"duration_finish_{_method_tag}_quantile_TOP{top_n}_{gitsha}_full.csv")
                saved.append(stem.with_suffix(".png"))
            except Exception as e:
                print(f"  [warn] duration {_method_tag} quantile importance: {e}")

    return saved
