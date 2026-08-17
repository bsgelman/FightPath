"""Train prop count and duration models only (skips winner/method/Optuna)."""
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

import pandas as pd
from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.training.tune_props import get_prop_feature_cols
from ufc.models.props_count import CountModel, HurdleCountModel
from ufc.models.props_duration import DurationModel
from ufc.training.train_all import _gitsha

gitsha = _gitsha()
model_dir = paths.outputs_models()
model_dir.mkdir(parents=True, exist_ok=True)

print(f"=== Prop + Duration Training ===")
print(f"  gitsha: {gitsha}")

props_df = parquet.read(paths.processed("features_props"))
props_df["event_date"] = pd.to_datetime(props_df["event_date"])
props_splits = get_splits(props_df)
prop_feature_cols = get_prop_feature_cols(props_df)
prop_cols = [c for c in prop_feature_cols if c in props_df.columns]
print(f"  prop_cols: {len(prop_cols)}")

for target, raw_col_a in [
    ("sig_strikes", "sig_str_landed_a"),
    ("takedowns", "td_landed_a"),
    ("r1_sig_strikes", "r1_sig_str_landed_a"),
]:
    if raw_col_a not in props_df.columns:
        print(f"  Skipping {target} — column {raw_col_a} not found")
        continue

    rows_a = props_df[props_splits["train"]].copy()
    rows_a["_y"] = rows_a[raw_col_a]
    X_tr = rows_a[prop_cols].fillna(0)
    y_tr = rows_a["_y"].fillna(0)

    rows_a_val = props_df[props_splits["val"]].copy()
    rows_a_val["_y"] = rows_a_val[raw_col_a]
    X_vl = rows_a_val[prop_cols].fillna(0)
    y_vl = rows_a_val["_y"].fillna(0)

    if (target != "r1_sig_strikes"
            and "total_fight_sec" in rows_a.columns
            and "scheduled_rounds" in rows_a.columns):
        sched_sec = (rows_a["scheduled_rounds"].fillna(3).astype(float) * 300.0).clip(lower=1)
        sample_weight = (rows_a["total_fight_sec"].fillna(sched_sec) / sched_sec).clip(0.1, 1.0).values
    else:
        sample_weight = None

    print(f"\nTraining {target}...")
    cm = HurdleCountModel(target=target) if target == "takedowns" else CountModel(target=target)
    cm.fit(X_tr, y_tr, X_vl, y_vl, prop_cols, sample_weight=sample_weight)
    cp = cm.save(model_dir, gitsha)
    print(f"  Saved: {cp.name}")

print("\nTraining duration model...")
dur_train = (props_df[props_splits["train"]]
             .dropna(subset=["total_fight_sec"])
             .drop_duplicates(subset=["fight_id"]).copy())
dur_val = (props_df[props_splits["val"]]
           .dropna(subset=["total_fight_sec"])
           .drop_duplicates(subset=["fight_id"]).copy())
event_obs_train = ~dur_train["method"].isin(["U-DEC", "S-DEC", "M-DEC"])
event_obs_val = ~dur_val["method"].isin(["U-DEC", "S-DEC", "M-DEC"])
dur_cols = [c for c in prop_cols if c in dur_train.columns]

dm = DurationModel()
dm.fit(dur_train, dur_train["total_fight_sec"], event_obs_train,
       dur_val, dur_val["total_fight_sec"], event_obs_val, dur_cols)
dp = dm.save(model_dir, gitsha)
print(f"  Saved: {dp.name}")
print("\n=== Done ===")
