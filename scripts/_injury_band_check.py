"""Check A (spec 2026-07-15): frozen-prod-model band probe for the injury flag.

Scores Van vs Pantoja (2026-09-19 rematch) under the CURRENT prod winner
weights against the freshly rebuilt pre_fight_state. Hard gate after the
Step-2 (ratings) rebuild: P(Van) strictly inside (LO, HI).

Band amended 2026-07-16 (RUNS.md INJ-2): the spec's original (0.669, 0.701)
used the full-fight-removal counterfactual as its floor, but full removal also
deletes stat channels that HURT Van (sapm/TDD-volume pollution from the 26s
fight), which label hygiene correctly keeps — so partial hygiene can sit below
it. Measured true endpoints via in-memory K-sweep under frozen prod weights:
K=1 -> 0.7110, K=0.25 -> 0.6592, K=0 -> 0.6280 (monotone). Band = (K=0, K=1).
Run: PYTHONPATH=src python scripts/_injury_band_check.py
"""
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.inference.predict_core import _find_latest_model, load_reference_data, predict_fight
from ufc.models.winner import WinnerModel

LO, HI = 0.628, 0.711

def main() -> int:
    wp = _find_latest_model("winner_ensemble_*.joblib")
    print(f"winner model: {wp}")
    models = {"winner": WinnerModel.load(wp)}
    fighters_df, state, ref_history_df = load_reference_data()
    pred = predict_fight(
        red_name="Joshua Van", blue_name="Alexandre Pantoja",
        rounds=5, is_title=True, event_date=date(2026, 9, 19),
        models=models, fighters_df=fighters_df, pre_fight_state=state,
        location="Los Angeles, California, USA",
        ref_history_df=ref_history_df, run_simulation=False,
    )
    p = pred.prob_red
    inside = LO < p < HI
    print(f"P(Van) = {p:.4f}  band ({LO}, {HI})  ->  {'PASS' if inside else 'FAIL'}")
    if not inside:
        print("  at/above HI: flag not reaching features | at/below LO: dampening too strong")
    return 0 if inside else 1

if __name__ == "__main__":
    sys.exit(main())
