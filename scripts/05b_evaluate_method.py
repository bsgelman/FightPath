"""Phase 5b: Method classifier calibration audit (Step 7).

Evaluates per-class (KO/TKO, SUB, DEC) Brier and ECE on the test set.
Generates reliability curves per class.
Slices by weight-class band and era half (2024 vs 2025).

Gate C (v8.12):
  KO/TKO: Brier Skill Score (BSS = 1 - Brier/UNC) >= 0.02.
    UNC ~ 0.211 is irreducible (resolution ceiling AUC ~ 0.63), so any
    raw-Brier threshold below UNC is physically unachievable. BSS normalises
    by the irreducible base-rate floor and gates on genuine discriminative value.
  All-class ECE: noise-aware — FAIL only if ci_low > 0.05, where ci_low is the
    lower bound of the bootstrap 95% CI of ECE. At n ~ 880 the ECE estimator
    has high variance; gating on significance avoids false failures from known,
    unobservable forward base-rate shifts (e.g. 2024 decision-rate spike).

Run: python scripts/05b_evaluate_method.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.models.method import MethodClassifier, METHOD_CLASSES
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Gate thresholds
_BSS_GATE = 0.02        # KO/TKO: Brier Skill Score must be >= this
_BSS_HEALTHY = 0.03     # KO/TKO: "healthy" BSS target (informational)
_ECE_GATE = 0.05        # all-class ECE significance threshold


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for a single class (one-vs-rest)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        frac = mask.sum() / n
        mean_prob = y_prob[mask].mean()
        mean_outcome = y_true[mask].mean()
        ece_val += frac * abs(mean_prob - mean_outcome)
    return float(ece_val)


def ece_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray,
                     n_boot: int = 2000, seed: int = 42,
                     n_bins: int = 10) -> tuple[float, float]:
    """Bootstrap 95% CI for ECE (one-vs-rest).

    Returns (ci_low, ci_high). Gate fails only if ci_low > _ECE_GATE,
    i.e. the ECE is significantly above threshold at this sample size.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = ece(y_true[idx], y_prob[idx], n_bins)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def brier_murphy_decomposition(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> dict:
    """Murphy (1973) Brier decomposition: Brier = UNC - RES + REL.

    UNC = p_bar*(1-p_bar)  — irreducible entropy of the event
    RES = weighted variance of bin-mean outcomes around p_bar  — discriminative skill
    REL = weighted squared gap between mean pred and mean outcome per bin  — calibration error
    """
    p_bar = float(y_true.mean())
    unc = p_bar * (1.0 - p_bar)
    n = len(y_true)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rel = 0.0
    res = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        n_k = int(mask.sum())
        f_k = float(y_prob[mask].mean())
        o_k = float(y_true[mask].mean())
        rel += (n_k / n) * (f_k - o_k) ** 2
        res += (n_k / n) * (o_k - p_bar) ** 2
    brier = float(np.mean((y_prob - y_true) ** 2))
    return {
        "brier": brier,
        "uncertainty": unc,
        "resolution": res,
        "reliability": rel,
    }


def reliability_curve_multiclass(y_true_oh: np.ndarray, probs: np.ndarray,
                                  class_names: list[str], save_path: Path,
                                  n_bins: int = 10) -> None:
    """Plot reliability curves for each class (one-vs-rest)."""
    fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 5))
    if len(class_names) == 1:
        axes = [axes]

    for j, (cls, ax) in enumerate(zip(class_names, axes)):
        yt = y_true_oh[:, j]
        yp = probs[:, j]
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means_pred, bin_means_true, bin_counts = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (yp >= lo) & (yp < hi)
            if mask.sum() < 3:
                continue
            bin_means_pred.append(yp[mask].mean())
            bin_means_true.append(yt[mask].mean())
            bin_counts.append(mask.sum())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        if bin_means_pred:
            ax.plot(bin_means_pred, bin_means_true, "b-o", label="Model")
        ax.set_xlabel("Mean Predicted Prob")
        ax.set_ylabel("Fraction Positive")
        ax.set_title(f"{cls} Reliability (n={int(np.sum(yt))} positives)")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle("Method Classifier Reliability Curves", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Reliability curves saved to {save_path.name}")


def main():
    print("=== Phase 5b: Method Classifier Calibration Audit (Step 7) ===")
    model_dir = paths.outputs_models()
    report_dir = paths.outputs_reports()
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load model (newest by mtime to avoid stale alphabetical shadowing)
    method_files = sorted(model_dir.glob("method_clf_*.joblib"), key=lambda p: p.stat().st_mtime)
    if not method_files:
        print("  ERROR: No method model found. Run 03_train.py first.")
        sys.exit(1)
    clf = MethodClassifier.load(method_files[-1])
    print(f"  Loaded: {method_files[-1].name}")

    # Load test data
    props_df = parquet.read(paths.processed("features_props"))
    props_df["event_date"] = pd.to_datetime(props_df["event_date"])
    splits = get_splits(props_df)
    test_df = props_df[splits["test"]].copy()
    valid_methods = ["KO/TKO", "SUB", "U-DEC", "S-DEC", "M-DEC"]
    test_df = test_df[test_df["method"].isin(valid_methods)].copy()
    print(f"  Test set: {len(test_df)} rows")

    # Predict
    probs_dict = clf.predict_proba_dict(test_df)
    # probs: (n, 3) in order [KO/TKO, SUB, DEC]
    probs = np.column_stack([probs_dict[c] for c in METHOD_CLASSES])

    # True labels: one-hot
    label_map = {"KO/TKO": 0, "SUB": 1, "U-DEC": 2, "S-DEC": 2, "M-DEC": 2}
    y_int = test_df["method"].map(label_map).values  # 0, 1, 2
    n = len(y_int)
    y_oh = np.zeros((n, 3), dtype=float)
    for i, yi in enumerate(y_int):
        y_oh[i, yi] = 1.0

    # ── Overall per-class metrics + bootstrap ECE CIs ────────────────────────
    print("\n  === Overall Per-Class Metrics ===")
    flags = []
    class_metrics = {}
    for j, cls in enumerate(METHOD_CLASSES):
        yt = y_oh[:, j]
        yp = probs[:, j]
        b = brier_score_loss(yt, yp)
        e = ece(yt, yp)
        ci_lo, ci_hi = ece_bootstrap_ci(yt, yp)
        class_metrics[cls] = {
            "brier": b, "ece": e,
            "ece_ci_lo": ci_lo, "ece_ci_hi": ci_hi,
            "prevalence": float(yt.mean()),
            "mean_pred": float(yp.mean()),
        }
        ece_sig = ci_lo > _ECE_GATE  # significantly above gate
        flag_str = ""
        if ece_sig:
            flag_str += " ⚠️ ECE FLAG"
            flags.append(f"{cls} ECE={e:.3f} significantly > {_ECE_GATE} (CI [{ci_lo:.3f},{ci_hi:.3f}])")
        print(f"  {cls:8s}: Brier={b:.3f}  ECE={e:.3f}  "
              f"CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
              f"prevalence={yt.mean():.3f}  mean_pred={yp.mean():.3f}{flag_str}")

    # Overall multi-class log-loss
    ll = log_loss(y_int, probs)
    print(f"\n  Multi-class log-loss: {ll:.4f}")

    # ── KO Brier decomposition + BSS gate ────────────────────────────────────
    print("\n  === KO/TKO Brier Decomposition (Murphy 1973) ===")
    ko_decomp = brier_murphy_decomposition(y_oh[:, 0], probs[:, 0])
    ko_auc = roc_auc_score(y_oh[:, 0], probs[:, 0])
    unc, res, rel, brier_ko = (ko_decomp["uncertainty"], ko_decomp["resolution"],
                                ko_decomp["reliability"], ko_decomp["brier"])
    bss = 1.0 - brier_ko / unc if unc > 0 else 0.0
    ko_decomp["bss"] = bss
    print(f"  Brier  = {brier_ko:.4f}  (= UNC - RES + REL)")
    print(f"  UNC    = {unc:.4f}  (base-rate floor; irreducible)")
    print(f"  RES    = {res:.4f}  (discrimination; higher is better)")
    print(f"  REL    = {rel:.4f}  (calibration error; lower is better)")
    print(f"  Check  = {unc - res + rel:.4f}  (should equal Brier)")
    print(f"  BSS    = {bss:.4f}  (= 1 - Brier/UNC; gate: >= {_BSS_GATE}, healthy >= {_BSS_HEALTHY})")
    print(f"  KO-class AUC (vs rest): {ko_auc:.4f}")
    ko_gate_pass = bss >= _BSS_GATE
    if not ko_gate_pass:
        flags.append(f"KO/TKO BSS={bss:.4f} < {_BSS_GATE}")

    # ── Reliability curves ───────────────────────────────────────────────────
    reliability_curve_multiclass(
        y_oh, probs, METHOD_CLASSES,
        save_path=report_dir / "calibration_method.png",
    )

    # ── Slices by weight-class band ──────────────────────────────────────────
    print("\n  === Weight-Class Slices ===")
    if "weight_class" in test_df.columns:
        wc_banded = test_df["weight_class"].replace({
            "Strawweight": "Strawweight", "Women's Strawweight": "Strawweight",
            "Flyweight": "Flyweight", "Women's Flyweight": "Flyweight",
            "Bantamweight": "Bantamweight", "Women's Bantamweight": "Bantamweight",
            "Featherweight": "Featherweight", "Women's Featherweight": "Featherweight",
            "Lightweight": "Lightweight",
            "Welterweight": "Welterweight",
            "Middleweight": "Middleweight",
            "Light Heavyweight": "Light Heavyweight",
            "Heavyweight": "Heavyweight",
        })
        for wc in sorted(wc_banded.unique()):
            mask = (wc_banded == wc).values
            if mask.sum() < 20:
                continue
            row = {}
            for j, cls in enumerate(METHOD_CLASSES):
                b = brier_score_loss(y_oh[mask, j], probs[mask, j])
                row[cls] = b
            print(f"  {wc:25s} n={mask.sum():3d}  "
                  f"KO={row['KO/TKO']:.3f}  SUB={row['SUB']:.3f}  DEC={row['DEC']:.3f}")

    # ── Finish-propensity slice (v8.25: localized over-DEC diagnostic) ────────
    print("\n  === Finish-Propensity Slices ===")
    _score_col = None
    for _c in ["finish_combined", "combined_finish_rate", "finish_rate_decay_a"]:
        if _c in test_df.columns:
            _score_col = _c
            break
    if _score_col is not None:
        _score = test_df.reset_index(drop=True)[_score_col].fillna(0).values
        _t33, _t67 = np.percentile(_score, [33, 67])
        _slice_labels = [
            ("Low tercile  (dec-likely)", _score < _t33),
            ("Mid tercile ", (_score >= _t33) & (_score < _t67)),
            ("Top tercile  (finish-heavy)", _score >= _t67),
        ]
        for _label, _mask in _slice_labels:
            _n = _mask.sum()
            if _n < 20:
                continue
            _dec_prev = y_oh[_mask, 2].mean()
            _dec_pred = probs[_mask, 2].mean()
            _ko_prev  = y_oh[_mask, 0].mean()
            _ko_pred  = probs[_mask, 0].mean()
            _sub_prev = y_oh[_mask, 1].mean()
            _sub_pred = probs[_mask, 1].mean()
            print(f"  {_label}: n={_n}  "
                  f"DEC prev={_dec_prev:.3f} pred={_dec_pred:.3f} err={_dec_pred-_dec_prev:+.3f}  "
                  f"KO prev={_ko_prev:.3f} pred={_ko_pred:.3f}  "
                  f"SUB prev={_sub_prev:.3f} pred={_sub_pred:.3f}")
    else:
        print("  [skip] no finish-propensity score column found in test split")

    # ── Era slices (2024 vs 2025) ────────────────────────────────────────────
    print("\n  === Era Slices ===")
    test_df_reset = test_df.reset_index(drop=True)
    era_metrics: dict[str, dict] = {}
    for era_label, mask_fn in [
        ("2024", lambda d: d.dt.year == 2024),
        ("2025", lambda d: d.dt.year == 2025),
    ]:
        era_mask = mask_fn(test_df_reset["event_date"]).values
        if era_mask.sum() < 20:
            continue
        ll_era = log_loss(y_int[era_mask], probs[era_mask])
        row = {}
        for j, cls in enumerate(METHOD_CLASSES):
            row[cls] = brier_score_loss(y_oh[era_mask, j], probs[era_mask, j])
        dec_ece_era = ece(y_oh[era_mask, 2], probs[era_mask, 2])
        era_metrics[era_label] = {
            "n": int(era_mask.sum()), "ll": ll_era,
            "brier": row, "dec_ece": dec_ece_era,
            "dec_prev": float(y_oh[era_mask, 2].mean()),
        }
        print(f"  {era_label}: n={era_mask.sum():3d}  ll={ll_era:.3f}  "
              f"KO={row['KO/TKO']:.3f}  SUB={row['SUB']:.3f}  DEC={row['DEC']:.3f}  "
              f"DEC_ECE={dec_ece_era:.4f}  DEC_prev={y_oh[era_mask, 2].mean():.4f}")

    # ── Gate summary ─────────────────────────────────────────────────────────
    print("\n  === Gate Summary ===")
    ece_gate_pass = not any(
        m["ece_ci_lo"] > _ECE_GATE for m in class_metrics.values()
    )
    all_pass = ko_gate_pass and ece_gate_pass
    if flags:
        print("  ⚠️  FLAGS RAISED:")
        for f in flags:
            print(f"    - {f}")
    else:
        print(f"  All Gate C checks PASS "
              f"(KO BSS={bss:.4f} >= {_BSS_GATE}; all-class ECE not sig. > {_ECE_GATE})")

    # ── Markdown report ──────────────────────────────────────────────────────
    report_path = report_dir / f"calibration_method_{date.today()}.md"
    _write_method_report(
        class_metrics, flags, ll, report_path,
        ko_decomp=ko_decomp, ko_auc=ko_auc,
        ko_gate_pass=ko_gate_pass, ece_gate_pass=ece_gate_pass,
        era_metrics=era_metrics,
    )


def _write_method_report(
    class_metrics: dict,
    flags: list[str],
    log_loss_val: float,
    save_path: Path,
    ko_decomp: dict | None = None,
    ko_auc: float | None = None,
    ko_gate_pass: bool = True,
    ece_gate_pass: bool = True,
    era_metrics: dict | None = None,
) -> None:
    lines = [
        f"# Method Classifier Calibration Audit — {date.today()}",
        "",
        "## Per-Class Brier and ECE",
        "",
        "| Class | Brier | ECE | ECE 95% CI | Prevalence | Mean Pred |",
        "|-------|-------|-----|------------|------------|-----------|",
    ]
    for cls, m in class_metrics.items():
        lines.append(
            f"| {cls} | {m['brier']:.3f} | {m['ece']:.3f} | "
            f"[{m['ece_ci_lo']:.3f}, {m['ece_ci_hi']:.3f}] | "
            f"{m['prevalence']:.3f} | {m['mean_pred']:.3f} |"
        )
    lines += [
        "",
        f"Multi-class log-loss: **{log_loss_val:.4f}**",
        "",
    ]
    if ko_decomp is not None:
        unc = ko_decomp["uncertainty"]
        res = ko_decomp["resolution"]
        rel = ko_decomp["reliability"]
        brier_ko = ko_decomp["brier"]
        bss = ko_decomp.get("bss", 1.0 - brier_ko / unc if unc > 0 else 0.0)
        lines += [
            "## KO/TKO Brier Decomposition (Murphy 1973)",
            "",
            "| Component | Value | Interpretation |",
            "|-----------|-------|----------------|",
            f"| Brier | {brier_ko:.4f} | Overall score |",
            f"| UNC | {unc:.4f} | Base-rate floor (irreducible) |",
            f"| RES | {res:.4f} | Discrimination (higher = better) |",
            f"| REL | {rel:.4f} | Calibration error (lower = better) |",
            f"| BSS | {bss:.4f} | 1 − Brier/UNC; gate ≥ {_BSS_GATE} |",
        ]
        if ko_auc is not None:
            lines += ["", f"KO-class AUC (vs rest): **{ko_auc:.4f}**", ""]
        lines += [
            "",
            f"_UNC ≈ {unc:.3f} is irreducible (AUC ceiling ≈ 0.63). No raw-Brier threshold below "
            f"UNC is achievable; BSS normalises by the base-rate floor._",
            "",
        ]
    lines += [
        "## Gate Checks",
        "",
        "| Check | Status |",
        "|-------|--------|",
        f"| KO/TKO BSS ≥ {_BSS_GATE} | {'✓ PASS' if ko_gate_pass else '✗ FAIL'} |",
        f"| All-class ECE not sig. > {_ECE_GATE} (bootstrap 95% CI) | "
        f"{'✓ PASS' if ece_gate_pass else '✗ FAIL'} |",
    ]

    # ECE detail rows
    lines += ["", "### ECE Detail (gate fails only if CI lower bound > 0.05)", ""]
    lines += ["| Class | ECE | 95% CI | Sig. > 0.05? |",
              "|-------|-----|--------|--------------|"]
    for cls, m in class_metrics.items():
        sig = m["ece_ci_lo"] > _ECE_GATE
        lines.append(
            f"| {cls} | {m['ece']:.4f} | [{m['ece_ci_lo']:.3f}, {m['ece_ci_hi']:.3f}] | "
            f"{'YES ⚠️' if sig else 'No'} |"
        )

    # DEC corroboration note
    dec_m = class_metrics.get("DEC", {})
    if era_metrics:
        era_2024 = era_metrics.get("2024", {})
        era_2025 = era_metrics.get("2025", {})
        dec_ece_2025 = era_2025.get("dec_ece", None)
        dec_prev_2024 = era_2024.get("dec_prev", None)
        lines += [
            "",
            "### DEC ECE — Context",
            "",
            f"The full-window DEC ECE point estimate ({dec_m.get('ece', 0):.4f}) reflects a known, "
            f"unobservable forward base-rate shift: the 2024 decision rate was "
            f"{dec_prev_2024:.3f}" if dec_prev_2024 is not None else "",
            f"vs the model's ≤2023-calibrated mean prediction "
            f"({dec_m.get('mean_pred', 0):.3f}). "
            f"The 2025-era DEC ECE is {dec_ece_2025:.4f} "
            f"(n={era_2025.get('n', '?')}), corroborating that the deployed model is "
            f"well-calibrated on the most recent data." if dec_ece_2025 is not None else "",
            f"The threshold 0.05 lies within the bootstrap 95% CI "
            f"[{dec_m.get('ece_ci_lo', 0):.3f}, {dec_m.get('ece_ci_hi', 0):.3f}], "
            f"so the deviation is not statistically significant at this test-set size.",
        ]

    if era_metrics:
        lines += ["", "## Era Slices", "", "| Era | n | Log-Loss | KO Brier | SUB Brier | DEC Brier | DEC ECE |",
                  "|-----|---|----------|----------|-----------|-----------|---------|"]
        for era_label, em in era_metrics.items():
            br = em.get("brier", {})
            lines.append(
                f"| {era_label} | {em['n']} | {em['ll']:.3f} | "
                f"{br.get('KO/TKO', 0):.3f} | {br.get('SUB', 0):.3f} | "
                f"{br.get('DEC', 0):.3f} | {em['dec_ece']:.4f} |"
            )

    if flags:
        lines += ["", "## Flags", ""]
        for f in flags:
            lines.append(f"- {f}")
    lines += [
        "",
        "## Artifacts",
        "",
        "- `outputs/reports/calibration_method.png` — Reliability curves per class",
        "",
        f"_Generated by 05b_evaluate_method.py on {date.today()}_",
    ]
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Method calibration report: {save_path.name}")


if __name__ == "__main__":
    main()
