"""Target-only factor calculation helpers."""
from __future__ import annotations

import pandas as pd


def _target_series(series: pd.Series, target_index: pd.Index, name: str | None = None) -> pd.Series:
    result = pd.Series(series, index=series.index, name=name)
    return result.loc[target_index]


def select_target_series(
    series: pd.Series,
    *,
    target_index: pd.Index,
    name: str | None = None,
) -> pd.Series:
    return _target_series(series, target_index, name=name)


def groupby_rolling_target(
    data: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    target_index: pd.Index,
    window: int,
    min_periods: int,
    agg: str,
    sort: bool = False,
) -> pd.Series:
    grouped = data.groupby(group_col, sort=sort)[value_col]
    rolling_obj = grouped.rolling(window, min_periods=min_periods)
    if agg == "mean":
        series = rolling_obj.mean().reset_index(level=0, drop=True)
    elif agg == "max":
        series = rolling_obj.max().reset_index(level=0, drop=True)
    elif agg == "min":
        series = rolling_obj.min().reset_index(level=0, drop=True)
    elif agg == "std":
        series = rolling_obj.std().reset_index(level=0, drop=True)
    elif agg == "sum":
        series = rolling_obj.sum().reset_index(level=0, drop=True)
    else:
        raise ValueError(f"Unsupported rolling agg: {agg}")
    return _target_series(series, target_index)


def groupby_shift_target(
    data: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    target_index: pd.Index,
    periods: int,
    sort: bool = False,
) -> pd.Series:
    grouped = data.groupby(group_col, sort=sort)[value_col]
    shifted = grouped.shift(periods)
    return _target_series(shifted, target_index)


def groupby_diff_target(
    data: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    target_index: pd.Index,
    periods: int,
    sort: bool = False,
) -> pd.Series:
    grouped = data.groupby(group_col, sort=sort)[value_col]
    diffed = grouped.diff(periods)
    return _target_series(diffed, target_index)


def groupby_pct_change_target(
    data: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    target_index: pd.Index,
    periods: int,
    sort: bool = False,
) -> pd.Series:
    grouped = data.groupby(group_col, sort=sort)[value_col]
    changed = grouped.pct_change(periods)
    return _target_series(changed, target_index)
