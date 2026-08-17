"""One-time: val-fit r1_sig_strikes finish_draw_scale on the EXISTING model (no weight
retrain) and re-save as newest. Reproducible via _fit_finish_draw_scale (same as retrain).
"""
import sys, warnings, subprocess
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.models.props_duration import DurationModel
from ufc.models.method import MethodClassifier, METHOD_CLASSES
from ufc.features.interactions import compute_interactions
import joblib
import importlib.util
_s = importlib.util.spec_from_file_location("_rco", str(ROOT / "scripts" / "_retrain_count_only.py"))
_rco = importlib.util.module_from_spec(_s); _s.loader.exec_module(_rco)

gitsha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                 cwd=str(ROOT)).decode().strip()
model_dir = paths.outputs_models()
props_df = parquet.read(paths.processed("features_props"))
props_df["event_date"] = pd.to_datetime(props_df["event_date"])
props_df = compute_interactions(props_df)
splits = get_splits(props_df)
val_rows = props_df[splits["val"]].copy().reset_index(drop=True)

dm = DurationModel.load(sorted(model_dir.glob("props_duration_*.joblib"),
                               key=lambda p: p.stat().st_mtime)[-1])
mclf = MethodClassifier.load(sorted(model_dir.glob("method_clf_*.joblib"),
                                    key=lambda p: p.stat().st_mtime)[-1])
mp = np.column_stack([mclf.predict_proba_dict(val_rows)[c] for c in METHOD_CLASSES])

r1_path = sorted(model_dir.glob("props_r1_sig_strikes_*.joblib"),
                 key=lambda p: p.stat().st_mtime)[-1]
cm = joblib.load(r1_path)
print("loaded", r1_path.name, "current finish_draw_scale", getattr(cm, "finish_draw_scale", "MISSING"))

scale = _rco._fit_finish_draw_scale(cm, dm, val_rows, val_rows["r1_sig_str_landed_a"],
                                    mp, ceiling=5.0)
print("val-fit finish_draw_scale =", round(scale, 4))
cm.finish_draw_scale = scale
out = cm.save(model_dir, gitsha)
print("re-saved:", out.name)
