"""Leakage-safe rolling/expanding/decay utilities.

ALL functions guarantee: feature at row i uses only data from rows {0..i-1}
within each fighter group (sorted by sort_col).

The mechanism: shift(1) before every expanding/rolling operation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def causal_expanding(
    df: pd.DataFrame,
    by: str,
    sort_col: str,
    value_col: str,
    agg: str = "mean",
    min_periods: int = 1,
) -> pd.Series:
    """Per-group expanding aggregation, EXCLUDING the current row.

    Returns a Series aligned with df's original index.
    """
    sorted_df = df.sort_values([by, sort_col])
    result = (
        sorted_df.groupby(by, sort=False)[value_col]
        .apply(lambda s: s.shift(1).expanding(min_periods=min_periods).agg(agg))
    )
    # Reset the inner group level so index aligns with sorted_df
    if isinstance(result.index, pd.MultiIndex):
        result = result.reset_index(level=0, drop=True)
    # Re-index to match original df index
    return result.reindex(df.index)


def causal_rolling(
    df: pd.DataFrame,
    by: str,
    sort_col: str,
    value_col: str,
    window: int,
    agg: str = "mean",
    min_periods: int = 1,
) -> pd.Series:
    """Per-group rolling aggregation over last `window` rows, EXCLUDING current."""
    sorted_df = df.sort_values([by, sort_col])
    result = (
        sorted_df.groupby(by, sort=False)[value_col]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=min_periods).agg(agg))
    )
    if isinstance(result.index, pd.MultiIndex):
        result = result.reset_index(level=0, drop=True)
    return result.reindex(df.index)


def causal_date_window(
    df: pd.DataFrame,
    by: str,
    date_col: str,
    value_col: str,
    window_days: int,
    agg: str = "mean",
) -> pd.Series:
    """Per-group rolling aggregation within a date window, EXCLUDING current row.

    Vectorized via np.searchsorted on sorted dates + cumulative sums.
    """
    sorted_df = df.sort_values([by, date_col])
    out = np.full(len(sorted_df), np.nan)
    pos = 0
    for fighter, grp in sorted_df.groupby(by, sort=False):
        n = len(grp)
        dates = grp[date_col].values.astype("datetime64[D]")
        vals = grp[value_col].values.astype(float)
        valid = ~np.isnan(vals)
        # Cumulative sum / count of valid values (length n+1)
        cs = np.concatenate([[0.0], np.cumsum(np.where(valid, vals, 0.0))])
        cc = np.concatenate([[0],   np.cumsum(valid.astype(int))])
        cutoffs = dates - np.timedelta64(window_days, "D")
        lefts = np.searchsorted(dates, cutoffs, side="left")
        for i in range(1, n):                          # i=0 has no prior rows
            l = lefts[i]
            r = i                                       # strictly prior (exclude self)
            cnt = cc[r] - cc[l]
            if cnt == 0:
                continue
            s = cs[r] - cs[l]
            if agg == "mean":
                out[pos + i] = s / cnt
            elif agg == "sum":
                out[pos + i] = s
            elif agg == "max":
                out[pos + i] = vals[l:r][~np.isnan(vals[l:r])].max() if cnt > 0 else np.nan
            elif agg == "min":
                out[pos + i] = vals[l:r][~np.isnan(vals[l:r])].min() if cnt > 0 else np.nan
        pos += n
    return pd.Series(out, index=sorted_df.index).reindex(df.index)


def causal_decay(
    df: pd.DataFrame,
    by: str,
    date_col: str,
    value_col: str,
    halflife_days: float,
) -> pd.Series:
    """Time-decayed weighted mean. weight = 0.5 ** (days_since / halflife_days)."""
    sorted_df = df.sort_values([by, date_col])
    out = np.full(len(sorted_df), np.nan)
    pos = 0
    for fighter, grp in sorted_df.groupby(by, sort=False):
        n = len(grp)
        dates = grp[date_col].values.astype("datetime64[D]")
        vals = grp[value_col].values.astype(float)
        valid = ~np.isnan(vals)
        if not valid.any() or n < 2:
            pos += n
            continue
        for i in range(1, n):
            mask = valid[:i]
            if not mask.any():
                continue
            d_prior = dates[:i][mask].astype("datetime64[D]")
            v_prior = vals[:i][mask]
            days_since = (dates[i] - d_prior).astype("timedelta64[D]").astype(float)
            w = np.power(0.5, days_since / halflife_days)
            wsum = w.sum()
            if wsum > 0:
                out[pos + i] = float((w * v_prior).sum() / wsum)
        pos += n
    return pd.Series(out, index=sorted_df.index).reindex(df.index)


def causal_count(
    df: pd.DataFrame,
    by: str,
    sort_col: str,
    value_col: str | None = None,
) -> pd.Series:
    """Count of prior rows per group (career fight count excluding current)."""
    sorted_df = df.sort_values([by, sort_col])
    if value_col is None:
        result = (
            sorted_df.groupby(by, sort=False).cumcount()
        )
    else:
        result = (
            sorted_df.groupby(by, sort=False)[value_col]
            .apply(lambda s: s.shift(1).expanding().count())
        )
        if isinstance(result.index, pd.MultiIndex):
            result = result.reset_index(level=0, drop=True)
    return result.reindex(df.index)


def causal_sum_expanding(
    df: pd.DataFrame,
    by: str,
    sort_col: str,
    value_col: str,
) -> pd.Series:
    """Expanding cumulative sum, EXCLUDING current row."""
    return causal_expanding(df, by, sort_col, value_col, agg="sum", min_periods=1)


def all_window_flavors(
    df: pd.DataFrame,
    by: str,
    sort_col: str,
    date_col: str,
    value_col: str,
    halflife_days: float = 548,
    window_24mo_days: int = 730,
    prefix: str | None = None,
) -> dict[str, pd.Series]:
    """Compute all 5 window flavors for a numeric column.

    Returns dict with keys: _ctd, _l3, _l5, _2y, _decay.
    prefix overrides value_col as the key prefix.
    """
    p = prefix or value_col
    return {
        f"{p}_ctd": causal_expanding(df, by, sort_col, value_col),
        f"{p}_l3": causal_rolling(df, by, sort_col, value_col, window=3),
        f"{p}_l5": causal_rolling(df, by, sort_col, value_col, window=5),
        f"{p}_2y": causal_date_window(df, by, date_col, value_col, window_days=window_24mo_days),
        f"{p}_decay": causal_decay(df, by, date_col, value_col, halflife_days=halflife_days),
    }
