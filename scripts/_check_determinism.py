"""Step 1 gate: verify determinism by running training twice from identical state.

Usage:
  python scripts/_check_determinism.py --save-ref    # run 1: save reference state + probs
  python scripts/_check_determinism.py --compare     # run 2: compare to reference
  python scripts/_check_determinism.py --infer-only  # just test inference determinism

Between --save-ref and --compare, the caller must restore dead_features_winner.txt
to the snapshot so both training runs start from the same state.
"""
import sys; sys.path.insert(0, "src")
import argparse
import shutil
import numpy as np
import pandas as pd
import joblib
from ufc.io import paths
from ufc.training.splits import get_splits
from ufc.training.symmetrize import symmetrize

models_dir = paths.outputs_models()
SNAP = models_dir / "dead_features_winner.SNAP"
REF_PROBS = models_dir / "det_probs_ref.npy"

parser = argparse.ArgumentParser()
parser.add_argument("--save-ref",   action="store_true")
parser.add_argument("--compare",    action="store_true")
parser.add_argument("--infer-only", action="store_true")
args = parser.parse_args()

feat = pd.read_parquet("data/processed/features_winner.parquet")
splits = get_splits(feat)
test_df = feat[splits["test"]].dropna(subset=["won_a"]).copy()
test_sym = symmetrize(test_df)
n = len(test_df)

wm = joblib.load(models_dir / "winner_ensemble_20260526.joblib")

# Always test inference determinism
pa = wm.predict_proba(test_sym)
pb = wm.predict_proba(test_sym)
diff_infer = float(np.abs(pa - pb).max())
print("=== Inference determinism (same model × 2) ===")
print(f"  max_abs_diff : {diff_infer:.2e}  -> {'PASS' if diff_infer == 0 else 'FAIL'}")

probs = (pa[:n] + (1.0 - pa[n:])) / 2.0
print(f"  n={n}  mean={probs.mean():.6f}  std={probs.std():.6f}")

dead_path = models_dir / "dead_features_winner.txt"
print(f"  Dead features: {dead_path.read_text().count(chr(10)) + 1 if dead_path.exists() else 0}")

if args.save_ref:
    # Snapshot the dead features file BEFORE this run's update
    if dead_path.exists():
        shutil.copy2(dead_path, SNAP)
        print(f"\nSnapshot saved -> {SNAP.name}")
    np.save(REF_PROBS, probs)
    print(f"Reference probs saved -> {REF_PROBS.name}")

elif args.compare:
    if not REF_PROBS.exists():
        print("\nERROR: no reference probs found; run with --save-ref first.")
        sys.exit(1)
    ref = np.load(REF_PROBS)
    max_diff = float(np.abs(ref - probs).max())
    n_diff = int((ref != probs).sum())
    print("\n=== Training determinism (ref run vs current run) ===")
    print(f"  n_different  : {n_diff}")
    print(f"  max_abs_diff : {max_diff:.2e}")
    status = "PASS" if max_diff == 0.0 else "FAIL"
    print(f"  TRAINING     : {status}")
    if status == "PASS":
        # Clean up snapshot + ref
        REF_PROBS.unlink(missing_ok=True)
        SNAP.unlink(missing_ok=True)
        print("  (cleaned up snapshot files)")
