"""Log the SERVED (prod) model's winner predictions for upcoming cards.

Run this AFTER scraping upcoming cards (06_scrape_upcoming.py) and BEFORE the
next prod retrain — that ordering is what makes the record honest: the prod model
has not yet trained on these fights. Re-running never overwrites an existing
prediction (the first pre-fight call is locked in).

    python scripts/07_log_predictions.py

Pairs with scripts/08_grade_predictions.py (grades these once results land).
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from ufc.io import paths
from ufc.inference.predict_core import (
    load_models, load_reference_data, predict_fight, _find_latest_model,
)
from ufc.inference import prediction_log as plog


def _prod_sha() -> str:
    p = _find_latest_model("winner_ensemble_*.joblib")
    if not p:
        return "unknown"
    # winner_ensemble_<sha>.joblib
    return p.stem.replace("winner_ensemble_", "")


def main():
    upcoming_dir = paths.root() / "cards" / "upcoming"
    cards = sorted(upcoming_dir.glob("*.json"))
    if not cards:
        print("[log-preds] No upcoming cards found — nothing to log.")
        return

    print("[log-preds] Loading prod models + reference data...")
    models = load_models(verbose=False)
    if "winner" not in models:
        print("[log-preds] No winner model available — abort.")
        return
    fighters_df, pre_fight_state, ref_history_df = load_reference_data()
    model_sha = _prod_sha()
    print(f"[log-preds] prod winner sha={model_sha}")

    log = plog.load_log()
    existing = set(log["key"].tolist())
    new_rows = []
    skipped = unfound = 0

    for cf in cards:
        card = json.loads(cf.read_text(encoding="utf-8"))
        ev_name = card.get("event_name", cf.stem)
        ev_date = str(card.get("event_date", ""))[:10]
        card_id = cf.stem
        for m in card.get("matchups", []):
            red, blue = m.get("red", ""), m.get("blue", "")
            if not red or not blue or not ev_date:
                continue
            key = plog.matchup_key(ev_date, red, blue)
            if key in existing:
                skipped += 1
                continue
            try:
                pred = predict_fight(
                    red_name=red, blue_name=blue,
                    rounds=int(m.get("scheduled_rounds", 3)),
                    is_title=bool(m.get("is_title", False)),
                    event_date=date.fromisoformat(ev_date),
                    models=models, fighters_df=fighters_df,
                    pre_fight_state=pre_fight_state, ref_history_df=ref_history_df,
                    run_simulation=False, verbose=False,
                    weight_class=m.get("weight_class") or None,
                    referee=m.get("referee", ""),
                )
            except Exception as e:
                unfound += 1
                print(f"  skip {red} vs {blue}: {e}")
                continue

            p_red = float(pred.prob_red)
            new_rows.append({
                "key": key, "event_date": ev_date, "event_name": ev_name,
                "card_id": card_id, "red": red, "blue": blue,
                "p_red": round(p_red, 4),
                "pred_winner": red if p_red >= 0.5 else blue,
                "model_sha": model_sha, "logged_at": plog.now_iso(),
                "status": "pending", "actual_winner": None,
                "correct": None, "resolved_at": None,
            })
            existing.add(key)

    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
        plog.save_log(log)
    print(f"[log-preds] logged {len(new_rows)} new · skipped {skipped} already-logged "
          f"· {unfound} unpredictable · total log size {len(log)}")


if __name__ == "__main__":
    main()
