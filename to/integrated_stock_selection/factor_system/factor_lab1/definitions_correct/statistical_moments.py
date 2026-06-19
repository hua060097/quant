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

# 移除未使用的 _get_sector_map_safe


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
    # 1. 构造临时 DF
    df = pd.DataFrame({'val': series.values, 'date': date_series.values}, index=series.index)
    
    # 2. 向量化计算每日中位数 (Median)
    medians = df.groupby('date')['val'].transform('median')
    
    # 3. 向量化计算每日绝对中位差 (MAD)
    abs_diff = (df['val'] - medians).abs()
    
    # 复用索引构造临时 DF
    temp_diff = pd.DataFrame({'diff': abs_diff.values, 'date': date_series.values}, index=series.index)
    mads = temp_diff.groupby('date')['diff'].transform('median')
    
    # 4. 计算稳健 Z-Score
    robust_std = mads * 1.4826
    z_score = (df['val'] - medians) / (robust_std + 1e-9)
    
    # 5. 极值处理：特殊处理 MAD 为 0 的情况
    mask_zero_mad = (mads < 1e-6)
    if mask_zero_mad.any():
        means = df.groupby('date')['val'].transform('mean')
        stds = df.groupby('date')['val'].transform('std')
        z_fallback = (df['val'] - means) / (stds + 1e-9)
        z_score = z_score.where(~mask_zero_mad, z_fallback)
    
    # 6. 宽阈值去极值 (Clip 6.0)
    # [修正] 删除 fillna(0.0)
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
    
    # 直接分组计算，无需行业映射
    grouped = data.groupby('stock_code')['pct_chg']
    
    # min_periods=15: 保证数据足够才计算偏度
    # 移除 ffill: 不要用旧数据填充，没有就是没有
    raw = grouped.transform(lambda x: x.rolling(20, min_periods=15).skew())
    
    # [修正] 删除 fillna(0)
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
    
    # 计算 rolling kurtosis
    raw = grouped.transform(lambda x: x.rolling(20, min_periods=15).kurt())

    # [修正] 删除 fillna(0)
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

    # 1. 准备数据 (移除冗余的 sector)
    df = pd.DataFrame({
        'date': data['date'],
        'stock': data['stock_code'],
        'r_i': data['pct_chg']
    }, index=data.index)
    
    # 对齐市场数据
    market_ret = market.set_index('date')['daily_change_pct']
    # 确保市场数据索引是唯一的
    if market_ret.index.duplicated().any():
         market_ret = market_ret[~market_ret.index.duplicated(keep='first')]

    m_centered = market_ret - market_ret.mean()
    df['m_sq'] = df['date'].map(m_centered ** 2)

    # 2. 向量化计算
    df_sorted = df.sort_values(['stock', 'date'])
    grouped = df_sorted.groupby('stock')
    
    i_mean = grouped['r_i'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df_sorted['i_centered'] = df_sorted['r_i'] - i_mean
    
    # 分子：E[(r_i - mean_i) * (r_m - mean_m)^2]
    num = (df_sorted['i_centered'] * df_sorted['m_sq']).rolling(20, min_periods=10).mean()
    
    # 分母：Std(r_i) * Std(r_m^2)
    i_std = grouped['i_centered'].transform(lambda x: x.rolling(20, min_periods=10).std())
    m_sq_std = grouped['m_sq'].transform(lambda x: x.rolling(20, min_periods=10).std())
    
    # [修正] 处理除零 Inf -> NaN
    raw_val = num / (i_std * m_sq_std + 1e-9)
    raw_val = raw_val.replace([np.inf, -np.inf], np.nan)
    
    # 3. 索引恢复
    raw_series = raw_val.reindex(data.index)
    
    # [修正] 删除 fillna(0)
    return _global_robust_zscore(raw_series, data['date'])

# ==============================================================================
# 4. 下行贝塔 (Downside Beta)
# ==============================================================================
def _downside_beta(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    下行贝塔 (Downside Beta)
    """
    market = _get_market_data(ctx)
    
    # 1. 筛选市场下跌日
    mkt_down = market[market['daily_change_pct'] < 0].set_index('date')['daily_change_pct']
    # 去重
    if mkt_down.index.duplicated().any():
        mkt_down = mkt_down[~mkt_down.index.duplicated(keep='first')]
    
    # 2. 映射市场收益
    df = data[['date', 'pct_chg', 'stock_code']].copy()
    df['mkt_ret'] = df['date'].map(mkt_down) # 上涨日会变成 NaN
    
    # 3. 计算 Beta 代理变量 (Covariance)
    # 理由：在截面 Z-Score 中，分母 Var(r_m) 是常数，不影响排序，直接用 Cov 效率更高
    grouped = df.groupby('stock_code', sort=False)
    
    # 协方差 (rolling cov 会自动处理 mkt_ret 为 NaN 的行)
    cov = grouped[['pct_chg', 'mkt_ret']].rolling(30, min_periods=10).cov().unstack().iloc[:, 1].reset_index(level=0, drop=True)
    
    # [修正] 删除 fillna(0)
    return _global_robust_zscore(cov, data['date'])


# 注册因子
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