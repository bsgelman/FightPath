"""Typed read/write helpers for parquet files."""
from pathlib import Path
import pandas as pd


def read(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


def write(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
