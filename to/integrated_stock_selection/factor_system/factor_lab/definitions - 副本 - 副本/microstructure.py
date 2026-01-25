"""微结构类因子。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


def _price_impact(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    sorted_data = data.sort_values(['stock_code', 'date']).reset_index()
    results = np.ones(len(sorted_data), dtype=float)

    start_indices = sorted_data.groupby('stock_code').indices
    for stock_code, indices in start_indices.items():
        vol_array = sorted_data.loc[indices, 'volume'].to_numpy(dtype=float, copy=False)
        pct_array = sorted_data.loc[indices, 'pct_chg'].to_numpy(dtype=float, copy=False)
        impact = _compute_price_impact_series(vol_array, pct_array)
        results[indices] = impact

    series = pd.Series(results, index=sorted_data['index'])
    series = series.reindex(data.index)
    return series.astype('float64')


def _compute_price_impact_series(volume: np.ndarray, pct_chg: np.ndarray) -> np.ndarray:
    window = 20
    output = np.ones(len(volume), dtype=float)
    for idx in range(len(volume)):
        start = max(0, idx - window + 1)
        vol_window = volume[start:idx + 1]
        if vol_window.size < 5:
            output[idx] = 1.0
            continue
        hist_mean = vol_window[:-1].mean() if vol_window[:-1].size else 0.0
        volume_ratio = vol_window[-1] / hist_mean if hist_mean > 0 else 1.0
        price_change = abs(pct_chg[idx])
        impact = price_change / (volume_ratio + 0.1)
        output[idx] = float(np.clip(impact, 0.0, 10.0))
    return output


price_impact = register_factor(
    FactorDefinition(
        name='price_impact',
        required_columns={'stock_code', 'date', 'pct_chg', 'volume'},
        lookback_days=30,
        calculator=_price_impact,
        description='基于成交量放大与价格变化的价格冲击指标',
    )
)


__all__ = ['price_impact']
