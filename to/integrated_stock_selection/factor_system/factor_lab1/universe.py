"""Universe eligibility helpers for factor calculation."""
from __future__ import annotations

import pandas as pd


def non_st_mask(data: pd.DataFrame) -> pd.Series:
    """Rows eligible for normal cross-sectional/sector/market statistics.

    If ``is_st`` is unavailable, keep the historical behavior by treating every
    row as eligible. The mask uses only the row's own same-day ``is_st`` value.
    """
    if "is_st" not in data.columns:
        return pd.Series(True, index=data.index)
    is_st = pd.to_numeric(data["is_st"], errors="coerce").fillna(0)
    return is_st.ne(1)


def eligible_values(series: pd.Series, data: pd.DataFrame) -> pd.Series:
    """Return a copy of ``series`` where ST rows are excluded as NaN."""
    mask = non_st_mask(data).reindex(series.index).fillna(True)
    return series.where(mask)


def eligible_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return rows eligible for normal aggregate statistics."""
    return data.loc[non_st_mask(data)].copy()


__all__ = [
    "non_st_mask",
    "eligible_values",
    "eligible_frame",
]
