"""Grade pending live predictions against resolved results.

Run AFTER results are scraped + features rebuilt (and it's safe to run before or
after the prod retrain — grading reads the resolved outcomes, not the model).
Matches by (event_date, sorted normalized name pair) so corner order doesn't
matter. Fills actual_winner / correct / resolved_at for any pending row whose
fight has now happened.

    python scripts/08_grade_predictions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from ufc.io import paths, parquet
from ufc.inference import prediction_log as plog


def _build_result_map() -> dict:
    """key -> winner display name, from resolved features_winner rows."""
    fw = parquet.read(paths.processed("features_winner"))
    fw = fw.dropna(subset=["won_a"]).copy()
    fw["event_date"] = pd.to_datetime(fw["event_date"])

    # Resolve fighter names (prefer explicit name cols, else id -> name map)
    id_to_name = {}
    try:
        fdf = parquet.read(paths.interim("fighters"))
        ncol = next((c for c in ("fighter_name", "name") if c in fdf.columns), None)
        if "fighter_id" in fdf.columns and ncol:
            id_to_name = dict(zip(fdf["fighter_id"].astype(str), fdf[ncol].astype(str)))
    except Exception:
        pass

    def nm(row, side):
        v = row.get(f"fighter_{side}_name")
        if isinstance(v, str) and v:
            return v
        fid = row.get(f"fighter_id_{side}") or row.get(f"fighter_{side}_id")
        return id_to_name.get(str(fid), "")

    out = {}
    for _, r in fw.iterrows():
        a, b = nm(r, "a"), nm(r, "b")
        if not a or not b:
            continue
        key = plog.matchup_key(r["event_date"].strftime("%Y-%m-%d"), a, b)
        out[key] = a if bool(r["won_a"]) else b
    return out


def main():
    log = plog.load_log()
    pending = log[log["status"] == "pending"]
    if len(pending) == 0:
        print("[grade] No pending predictions.")
        return

    results = _build_result_map()
    graded = 0
    for idx, row in pending.iterrows():
        winner = results.get(row["key"])
        if not winner:
            continue
        correct = plog.norm_name(row["pred_winner"]) == plog.norm_name(winner)
        log.at[idx, "actual_winner"] = winner
        log.at[idx, "correct"] = bool(correct)
        log.at[idx, "status"] = "resolved"
        log.at[idx, "resolved_at"] = plog.now_iso()
        graded += 1

    if graded:
        plog.save_log(log)
    resolved_total = int((log["status"] == "resolved").sum())
    correct_total = int(log.loc[log["status"] == "resolved", "correct"].astype(bool).sum())
    still_pending = int((log["status"] == "pending").sum())
    rate = correct_total / resolved_total if resolved_total else 0.0
    print(f"[grade] graded {graded} this run · live record now "
          f"{correct_total}/{resolved_total} ({rate*100:.1f}%) · {still_pending} pending")


if __name__ == "__main__":
    main()
