"""Test ELO pre-fight ratings are computed from strict chronological prefix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from ufc.features.ratings import compute_elo


def _make_ledger(n_fights=6) -> pd.DataFrame:
    """Synthetic 2-fighter ledger with known fight sequence."""
    rows = []
    f_a, f_b = "fighter_A", "fighter_B"
    base = pd.Timestamp("2015-01-01")

    # Alternating wins
    for i in range(n_fights):
        event_date = base + pd.Timedelta(days=i * 120)
        winner = f_a if i % 2 == 0 else f_b
        for (this, opp, won) in [(f_a, f_b, 1 if winner == f_a else 0),
                                  (f_b, f_a, 1 if winner == f_b else 0)]:
            rows.append({
                "fight_id": f"fight_{i}",
                "event_id": f"event_{i}",
                "event_date": event_date,
                "event_rank": i,
                "fighter_id": this,
                "opponent_id": opp,
                "won": won,
                "method": "U-DEC",
                "total_fight_sec": 900,
                "end_round": 3,
                "sig_str_landed": 50,
                "sig_str_absorbed_landed": 40,
                "td_landed": 2,
                "td_attempted": 4,
                "td_absorbed_landed": 1,
                "td_absorbed_attempted": 3,
                "ctrl_sec": 60,
                "ctrl_sec_absorbed": 30,
                "kd_for": 0, "kd_against": 0,
                "sub_att_for": 0, "sub_att_against": 0,
                "rev_for": 0, "rev_against": 0,
                "head_landed": 30, "body_landed": 10, "leg_landed": 10,
                "distance_landed": 40, "clinch_landed": 5, "ground_landed": 5,
                "head_absorbed": 25, "body_absorbed": 8, "leg_absorbed": 7,
                "distance_absorbed": 35, "clinch_absorbed": 3, "ground_absorbed": 2,
                "head_attempted": 60, "body_attempted": 20, "leg_attempted": 20,
                "distance_attempted": 80, "clinch_attempted": 10, "ground_attempted": 10,
                "sig_str_attempted": 100,
                "sig_str_absorbed_attempted": 90,
                "td_absorbed_attempted": 3,
                "age_years": 28.0, "reach_in": 72.0, "height_in": 71.0,
                "stance": "ORTHO", "weight_lbs": 170.0,
                "referee": "ref_A", "location": "Las Vegas",
                "is_title": False, "is_main_event": False,
                "scheduled_rounds": 3,
                "weight_class": "Welterweight",
            })
    return pd.DataFrame(rows)


class TestEloChronology:
    def test_first_fight_is_initial_rating(self):
        ledger = _make_ledger()
        result = compute_elo(ledger)

        f_a = result[result["fighter_id"] == "fighter_A"].sort_values("event_rank")
        first_elo = f_a.iloc[0]["elo_pre"]
        assert abs(first_elo - 1500.0) < 1e-6, \
            f"First fight ELO should be initial 1500, got {first_elo}"

    def test_elo_pre_not_equal_post(self):
        ledger = _make_ledger(n_fights=3)
        result = compute_elo(ledger)

        f_a = result[result["fighter_id"] == "fighter_A"].sort_values("event_rank")
        # After first fight, ELO should have changed
        elo_0 = f_a.iloc[0]["elo_pre"]
        elo_1 = f_a.iloc[1]["elo_pre"]
        assert elo_0 != elo_1, "ELO should update after a fight"

    def test_elo_pre_at_fight_n_uses_only_prefix(self):
        """Re-compute ELO manually for fight 2 and verify it matches pipeline."""
        ledger = _make_ledger(n_fights=4)
        result = compute_elo(ledger)

        # Manually compute ELO for fighter_A after 2 fights (fights 0 and 1)
        # Fight 0: A wins (U-DEC, mult=1.0)
        # Fight 1: B wins (U-DEC, mult=1.0)
        K_BASE = 24
        MULT = 1.0  # U-DEC
        INIT = 1500.0

        r_a, r_b = INIT, INIT
        # Fight 0: A wins
        e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
        q = 1.0 + 0.4 * (lambda x: x / (1 + abs(x)))((r_b - INIT) / 200.0)  # approx tanh
        import math
        q_mult = 1.0 + 0.4 * math.tanh((r_b - INIT) / 200.0)
        K = K_BASE * MULT * q_mult
        r_a_after_0 = r_a + K * (1 - e_a)
        r_b_after_0 = r_b + K * (0 - (1 - e_a))

        # Fight 1: B wins — K for fighter A uses opponent B's ELO (r_b), not A's
        r_a, r_b = r_a_after_0, r_b_after_0
        e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
        q_mult_a = 1.0 + 0.4 * math.tanh((r_b - INIT) / 200.0)
        K_a = K_BASE * MULT * q_mult_a
        r_a_after_1 = r_a + K_a * (0 - e_a)

        # ELO at fight 2 should be r_a_after_1
        f_a = result[result["fighter_id"] == "fighter_A"].sort_values("event_rank")
        elo_at_fight_2 = f_a.iloc[2]["elo_pre"]
        assert abs(elo_at_fight_2 - r_a_after_1) < 0.1, \
            f"ELO at fight 2: expected {r_a_after_1:.2f}, got {elo_at_fight_2:.2f}"

    def test_winner_elo_increases(self):
        ledger = _make_ledger(n_fights=4)
        result = compute_elo(ledger)

        f_a = result[result["fighter_id"] == "fighter_A"].sort_values("event_rank")
        # Fight 0: A wins — elo_pre at fight 1 should be > fight 0
        elo_0 = f_a.iloc[0]["elo_pre"]
        elo_1 = f_a.iloc[1]["elo_pre"]
        assert elo_1 > elo_0, "ELO should increase after a win"
