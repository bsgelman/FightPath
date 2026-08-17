"""Basic training for prop models (no heavy Optuna search — NGBoost is slow)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ufc.training.feature_pruning import prune_features


def get_prop_feature_cols(df: pd.DataFrame, model_name: str = "props") -> list[str]:
    """Feature columns for prop models (same as winner + scheduled_rounds).

    Specialist interaction features (sub_specialist_idx, ko_specialist_idx,
    finish_share, grappling_control_threat) and era/weight-class baseline
    features are excluded from count/duration prop models — they are
    winner/method-specific signals that add noise and distributional distortion
    to raw strike/takedown count predictions.
    """
    exclude = {"fight_id", "event_id", "fighter_id_a", "fighter_id_b",
               "event_date", "event_rank", "won_a", "method",
               "end_round", "end_time_sec", "total_fight_sec",
               "sig_str_landed_a", "sig_str_landed_b",
               "td_landed_a", "td_landed_b", "weight_class",
               "stance_pair", "referee", "location",
               # Per-fight raw outcome stats — unknown at inference time
               "ctrl_sec_a", "ctrl_sec_b",
               "ctrl_sec_absorbed_a", "ctrl_sec_absorbed_b",
               "sub_att_for_a", "sub_att_for_b",
               # R1 raw outcomes — unknown at inference time
               "r1_sig_str_landed_a", "r1_sig_str_landed_b",
               "r1_td_landed_a", "r1_td_landed_b",
               # New prop target labels — unknown at inference time
               "kd_for_a", "kd_for_b",
               "body_landed_a", "body_landed_b",
               "leg_landed_a", "leg_landed_b",
               # Specialist finish amplifiers — winner/method model features only.
               # High-variance interaction products; hurt count model KS calibration.
               "sub_specialist_idx_a", "sub_specialist_idx_b",
               "sub_specialist_x_weakness_a", "sub_specialist_x_weakness_b",
               "ko_specialist_idx_a", "ko_specialist_idx_b",
               "ko_specialist_x_weakness_a", "ko_specialist_x_weakness_b",
               "finish_share_a", "finish_share_b",
               "grappling_control_threat_a", "grappling_control_threat_b",
               # v8.2 KO matchup discrimination features — method/winner only.
               # High-variance KO-specific products; confound count model calibration.
               "ko_matchup_a", "ko_matchup_b",
               "chin_decay_threat_a", "chin_decay_threat_b",
               "layoff_chin_a", "layoff_chin_b",
               "ko_specialist_x_chin_a", "ko_specialist_x_chin_b",
               # Era/weight-class baselines — cause double-counting with fighter rolling
               # rates in count models; kept only in features_winner/features_props for
               # winner+method use.
               "era_avg_sig_str_l12mo", "wc_finish_share_l2y", "wc_5rd_dec_rate",
               # V7.2: era KO/SUB share features (assembled with _a/_b suffixes).
               "era_ko_share_l24mo_a", "era_ko_share_l24mo_b",
               "era_sub_share_l24mo_a", "era_sub_share_l24mo_b"}
    cols = [c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, "Int8", "Int16"]]
    return prune_features(cols, model_name=model_name)
