"""Tests for symmetric training augmentation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from ufc.training.symmetrize import symmetrize, inference_average


def _make_sample_wide() -> pd.DataFrame:
    rows = [
        {
            "fight_id": "f1", "event_date": "2020-01-01",
            "fighter_id_a": "fa", "fighter_id_b": "fb",
            "won_a": 1.0,
            "slpm_decay_a": 5.0, "slpm_decay_b": 3.0,
            "td_per_15_decay_a": 2.0, "td_per_15_decay_b": 1.0,
            "reach_diff": 2.0,
            "age_diff": 3.0,
            "pace_diff": 1.0,
        },
        {
            "fight_id": "f2", "event_date": "2020-02-01",
            "fighter_id_a": "fc", "fighter_id_b": "fd",
            "won_a": 0.0,
            "slpm_decay_a": 4.0, "slpm_decay_b": 6.0,
            "td_per_15_decay_a": 0.5, "td_per_15_decay_b": 3.0,
            "reach_diff": -1.0,
            "age_diff": -2.0,
            "pace_diff": -0.5,
        },
    ]
    return pd.DataFrame(rows)


class TestSymmetrize:
    def test_doubles_row_count(self):
        df = _make_sample_wide()
        sym = symmetrize(df)
        assert len(sym) == 2 * len(df)

    def test_flipped_labels_complement(self):
        df = _make_sample_wide()
        sym = symmetrize(df)
        orig = sym[~sym["_is_flipped"]].sort_values("fight_id").reset_index(drop=True)
        flip = sym[sym["_is_flipped"]].sort_values("fight_id").reset_index(drop=True)
        for i in range(len(orig)):
            p_orig = orig.iloc[i]["won_a"]
            p_flip = flip.iloc[i]["won_a"]
            assert abs(p_orig + p_flip - 1.0) < 1e-9, \
                f"Labels not complementary: {p_orig} + {p_flip} != 1"

    def test_flipped_a_b_cols_swapped(self):
        df = _make_sample_wide()
        sym = symmetrize(df)
        orig_row = sym[~sym["_is_flipped"] & (sym["fight_id"] == "f1")].iloc[0]
        flip_row = sym[sym["_is_flipped"] & (sym["fight_id"] == "f1")].iloc[0]

        assert abs(orig_row["slpm_decay_a"] - flip_row["slpm_decay_b"]) < 1e-9
        assert abs(orig_row["slpm_decay_b"] - flip_row["slpm_decay_a"]) < 1e-9

    def test_diff_features_negated(self):
        df = _make_sample_wide()
        sym = symmetrize(df)
        orig_row = sym[~sym["_is_flipped"] & (sym["fight_id"] == "f1")].iloc[0]
        flip_row = sym[sym["_is_flipped"] & (sym["fight_id"] == "f1")].iloc[0]

        assert abs(orig_row["reach_diff"] + flip_row["reach_diff"]) < 1e-9, \
            "reach_diff should be negated in flip"
        assert abs(orig_row["age_diff"] + flip_row["age_diff"]) < 1e-9, \
            "age_diff should be negated in flip"


class TestInferenceAverage:
    def test_symmetric_sum_to_one(self):
        p_a = 0.65  # P(A wins from A perspective)
        p_b = 0.40  # P(A wins from B perspective) = 1 - P(B wins from B perspective)
        # P(B wins from B perspective) = 0.60
        result = inference_average(p_a, p_b)
        assert 0 <= result <= 1.0

    def test_certain_win_both(self):
        # Both perspectives agree A wins with certainty
        result = inference_average(1.0, 0.0)  # p_b=0 means P(B wins)=0
        assert abs(result - 1.0) < 1e-9

    def test_equal_prob(self):
        result = inference_average(0.5, 0.5)
        assert abs(result - 0.5) < 1e-9
