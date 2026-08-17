"""Time-series train/val/test splits — no shuffling."""
from __future__ import annotations

import os

import pandas as pd
import yaml

from ufc.io import paths


def _cfg():
    filename = os.environ.get("UFC_SPLIT_CONFIG", "split.yaml")
    with open(paths.root() / "configs" / filename) as f:
        return yaml.safe_load(f)


def get_splits(df: pd.DataFrame, date_col: str = "event_date") -> dict[str, pd.Series]:
    """Return boolean masks for train/val/test splits (3-way; no calib fold)."""
    cfg = _cfg()
    dates = pd.to_datetime(df[date_col])
    masks = {
        "train": (dates >= cfg["train_start"]) & (dates <= cfg["train_end"]),
        "val":   (dates >= cfg["val_start"])   & (dates <= cfg["val_end"]),
        "test":  (dates >= cfg["test_start"])  & (dates <= cfg["test_end"]),
    }
    if "val_a_start" in cfg:
        masks["val_a"] = (dates >= cfg["val_a_start"]) & (dates <= cfg["val_a_end"])
        masks["val_b"] = (dates >= cfg["val_b_start"]) & (dates <= cfg["val_b_end"])
    return masks
