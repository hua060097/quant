"""
统计矩因子（偏度、峰度、高阶风险交互）- 大师级优化版。
核心关注：收益率分布的非正态性（肥尾、偏斜）以及与市场的尾部相关性。

优化日志：
1. 全局移除 .fillna(0)，保留 NaN (未知) 状态。
2. 移除冗余的 Sector 逻辑（因为使用全局 Z-Score，不需要行业数据）。
3. 增加 Inf 清洗。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

def _get_market_data(ctx: FactorContext) -> pd.DataFrame:
    """获取对齐后的市场数据"""
    market = ctx.require_market()
    return market.sort_values('date').reset_index(drop=True)


# ==============================================================================
# 0. 新增核心工具：全局按日稳健 Z-Score (Top-1 专用版)
# ==============================================================================
def _global_robust_zscore(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """
    [大师级] 全局按日稳健 Z-Score (Robust Z-Score based on MAD)
    
    逻辑升级：
    1. 使用 Median 和 MAD 代替 Mean/Std。
    2. [重要修复] 不再 fillna(0.0)，保留 NaN。让 LightGBM 处理缺失数据。
    """
    df = pd.DataFrame({'val': series.values, 'date': date_series.values}, index=series.index)
    medians = df.groupby('date')['val'].transform('median')
    abs_diff = (df['val'] - medians).abs()
    temp_diff = pd.DataFrame({'diff': abs_diff.values, 'date': date_series.values}, index=series.index)
    mads = temp_diff.groupby('date')['diff'].transform('median')
    robust_std = mads * 1.4826
    z_score = (df['val'] - medians) / (robust_std + 1e-9)
    mask_zero_mad = (mads < 1e-6)
    if mask_zero_mad.any():
        means = df.groupby('date')['val'].transform('mean')
        stds = df.groupby('date')['val'].transform('std')
        z_fallback = (df['val'] - means) / (stds + 1e-9)
        z_score = z_score.where(~mask_zero_mad, z_fallback)
    return z_score.clip(-6.0, 6.0).astype('float64')


# ==============================================================================
# 1. 收益偏度 (Skewness)
# ==============================================================================
def _skewness_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    20日收益率偏度
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(np.nan, index=data.index)
    
    grouped = data.groupby('stock_code')['pct_chg']
    raw = grouped.transform(lambda x: x.rolling(20, min_periods=15).skew())
    return _global_robust_zscore(raw, data['date'])


# ==============================================================================
# 2. 收益峰度 (Kurtosis)
# ==============================================================================
def _kurtosis_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    20日收益率峰度
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(np.nan, index=data.index)

    grouped = data.groupby('stock_code')['pct_chg']
    raw = grouped.transform(lambda x: x.rolling(20, min_periods=15).kurt())
    return _global_robust_zscore(raw, data['date'])


# ==============================================================================
# 3. 协同偏度 (Coskewness)
# ==============================================================================
def _coskewness(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    协同偏度 (Coskewness)
    """
    market = _get_market_data(ctx)
    if 'daily_change_pct' not in market.columns:
        return pd.Series(np.nan, index=data.index)

    df = pd.DataFrame({
        'date': data['date'],
        'stock': data['stock_code'],
        'r_i': data['pct_chg']
    }, index=data.index)
    
    market_ret = market.set_index('date')['daily_change_pct']
    if market_ret.index.duplicated().any():
         market_ret = market_ret[~market_ret.index.duplicated(keep='first')]

    m_centered = market_ret - market_ret.mean()
    df['m_sq'] = df['date'].map(m_centered ** 2)

    df_sorted = df.sort_values(['stock', 'date'])
    grouped = df_sorted.groupby('stock')
    
    i_mean = grouped['r_i'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df_sorted['i_centered'] = df_sorted['r_i'] - i_mean
    
    num = (df_sorted['i_centered'] * df_sorted['m_sq']).rolling(20, min_periods=10).mean()
    i_std = grouped['i_centered'].transform(lambda x: x.rolling(20, min_periods=10).std())
    m_sq_std = grouped['m_sq'].transform(lambda x: x.rolling(20, min_periods=10).std())
    
    raw_val = num / (i_std * m_sq_std + 1e-9)
    raw_val = raw_val.replace([np.inf, -np.inf], np.nan)
    raw_series = raw_val.reindex(data.index)
    return _global_robust_zscore(raw_series, data['date'])


# ==============================================================================
# 4. 下行贝塔 (Downside Beta)
# ==============================================================================
def _downside_beta(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    下行贝塔 (Downside Beta)
    """
    market = _get_market_data(ctx)
    mkt_down = market[market['daily_change_pct'] < 0].set_index('date')['daily_change_pct']
    if mkt_down.index.duplicated().any():
        mkt_down = mkt_down[~mkt_down.index.duplicated(keep='first')]
    
    df = data[['date', 'pct_chg', 'stock_code']].copy()
    df['mkt_ret'] = df['date'].map(mkt_down)
    grouped = df.groupby('stock_code', sort=False)
    cov = grouped[['pct_chg', 'mkt_ret']].rolling(30, min_periods=10).cov().unstack().iloc[:, 1].reset_index(level=0, drop=True)
    return _global_robust_zscore(cov, data['date'])


skewness_20d = register_factor(FactorDefinition(name='skewness_20d', required_columns={'stock_code', 'date', 'pct_chg'}, lookback_days=30, calculator=_skewness_20d, description='20日收益率偏度（暴涨暴跌的不对称性）'))
kurtosis_20d = register_factor(FactorDefinition(name='kurtosis_20d', required_columns={'stock_code', 'date', 'pct_chg'}, lookback_days=30, calculator=_kurtosis_20d, description='20日收益率峰度（肥尾风险/极端波动概率）'))
coskewness = register_factor(FactorDefinition(name='coskewness', required_columns={'stock_code', 'date', 'pct_chg'}, requires_market_data=True, lookback_days=30, calculator=_coskewness, description='协同偏度（集成去极值与日内Z-score版）'))
downside_beta = register_factor(FactorDefinition(name='downside_beta', required_columns={'stock_code', 'date', 'pct_chg'}, requires_market_data=True, lookback_days=40, calculator=_downside_beta, description='下行贝塔（下跌市场中的协方差，衡量抗跌性）'))

__all__ = [
    'skewness_20d',
    'kurtosis_20d',
    'coskewness',
    'downside_beta'
]
