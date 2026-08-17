import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.inference.attribution import _assign_group


class TestAssignGroup:
    def test_reach_and_height_and_weight_are_reach_group(self):
        assert _assign_group("reach_diff") == "reach"
        assert _assign_group("height_diff") == "reach"
        assert _assign_group("weight_diff") == "reach"

    def test_opp_hittability_is_striking_defense_not_reach(self):
        # opp_hittability_a/b = opponent's sapm x (1 - str_def) — a chin/defense
        # proxy, not a size measure. Must not land in "Reach & size".
        assert _assign_group("opp_hittability_a") == "striking_def"
        assert _assign_group("opp_hittability_b") == "striking_def"

    def test_sapm_and_str_def_are_striking_defense(self):
        assert _assign_group("sapm_decay_a") == "striking_def"
        assert _assign_group("str_def_decay_b") == "striking_def"

    def test_accumulated_damage_is_its_own_group_not_age(self):
        # total_sig_str_absorbed is a CUMULATIVE count, so it scales with how much
        # a fighter has fought, not with durability — the opposite direction to
        # both age (younger is better) and tenure (more is better). Keep it out of
        # "Age & freshness" so it cannot net against the layoff signal there.
        assert _assign_group("total_sig_str_absorbed_career_a") == "mileage"
        assert _assign_group("total_sig_str_absorbed_career_b") == "mileage"

    def test_age_group_keeps_age_and_layoff_only(self):
        assert _assign_group("age_diff") == "age"
        assert _assign_group("age_years_a") == "age"
        assert _assign_group("layoff_days_b") == "age"
        assert _assign_group("layoff_age_interaction_a") == "age"

    def test_mileage_is_not_folded_into_experience(self):
        # Opposite directions: more tenure is good, more mileage is bad — folding
        # them together would re-create the same netting-out bug.
        assert _assign_group("fights_career_a") == "experience"
        assert _assign_group("total_rounds_career_b") == "experience"
        assert _assign_group("total_sig_str_absorbed_career_a") != "experience"
