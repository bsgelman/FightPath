"""Step 2 gate: verify monotone constraint sign directions.

Perturb each constrained feature ±20% and confirm ΔP(win A) has correct sign.
Expected:
  +1 features (specialist_a, elo_a, etc.): increase → ΔP > 0
  -1 features (specialist_b, age_diff, etc.): increase → ΔP < 0
"""
import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import joblib
from ufc.io import paths
from ufc.training.splits import get_splits
from ufc.training.symmetrize import symmetrize
from ufc.models.winner import _MONO_POSITIVE, _MONO_NEGATIVE

models_dir = paths.outputs_models()
wm = joblib.load(models_dir / "winner_ensemble_20260526.joblib")

feat = pd.read_parquet("data/processed/features_winner.parquet")
splits = get_splits(feat)
# Use test set for evaluation; take first 100 rows for speed
test_df = feat[splits["test"]].dropna(subset=["won_a"]).head(100).copy()

def sym_predict(df):
    sym = symmetrize(df)
    n = len(df)
    p = wm.predict_proba(sym)
    return (p[:n] + (1.0 - p[n:])) / 2.0

baseline_probs = sym_predict(test_df)

results = []
feature_cols = wm.feature_cols

for col in feature_cols:
    if col not in (_MONO_POSITIVE | _MONO_NEGATIVE):
        continue
    if col not in test_df.columns:
        continue

    perturbed = test_df.copy()
    col_vals = test_df[col].fillna(0)
    # Increase by 20% of the feature's std (or 0.2 if std is tiny)
    delta = max(float(col_vals.std()), 0.01) * 0.2
    perturbed[col] = col_vals + delta

    new_probs = sym_predict(perturbed)
    avg_delta = float((new_probs - baseline_probs).mean())

    expected_sign = "+" if col in _MONO_POSITIVE else "-"
    actual_sign = "+" if avg_delta > 0 else ("-" if avg_delta < 0 else "0")
    ok = (expected_sign == actual_sign) or abs(avg_delta) < 1e-6
    results.append((col, expected_sign, actual_sign, avg_delta, ok))

print(f"{'Feature':<40} {'Exp':>5} {'Act':>5} {'dP_avg':>10} {'Status'}")
print("-" * 70)
n_pass = n_fail = 0
for col, exp, act, dp, ok in sorted(results, key=lambda x: x[0]):
    status = "PASS" if ok else "FAIL"
    if ok: n_pass += 1
    else: n_fail += 1
    print(f"{col:<40} {exp:>5} {act:>5} {dp:>10.5f}  {status}")

print(f"\nMonotone sign check: {n_pass} PASS, {n_fail} FAIL")
