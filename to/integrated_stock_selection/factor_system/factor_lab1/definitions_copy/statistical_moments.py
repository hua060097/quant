"""
统计矩因子（偏度、峰度、高阶风险交互）。
核心关注：收益率分布的非正态性（肥尾、偏斜）以及与市场的尾部相关性。
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

def _get_sector_map_safe(ctx: FactorContext) -> pd.Series:
    if ctx.extra and 'stock_to_sw_sector' in ctx.extra:
        return ctx.extra['stock_to_sw_sector']
    raise ValueError("Sector data missing in context")



# ==============================================================================
# 1. 收益偏度 (Skewness) - 修复对齐与标准化版
# ==============================================================================
def _skewness_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'stock': data['stock_code'],
        'pct': data['pct_chg'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown')
    }, index=data.index)
    
    df_sorted = df.sort_values(['stock', 'date'])
    raw = df_sorted.groupby('stock')['pct'].transform(lambda x: x.rolling(20, min_periods=15).skew().fillna(method='ffill'))
    
    return _apply_sector_robust_zscore(raw.reindex(data.index), df)

# ==============================================================================
# 2. 收益峰度 (Kurtosis) - 修复对齐与标准化版
# ==============================================================================
def _kurtosis_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if 'pct_chg' not in data.columns:
        return pd.Series(0.0, index=data.index)

    # 1. 准备基础数据并注入行业映射
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'stock': data['stock_code'],
        'pct': data['pct_chg'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown')
    }, index=data.index)

    # 2. 时序安全计算：先排序，再分组计算
    df_sorted = df.sort_values(['stock', 'date'])

    # 使用 lambda 实现（与 skewness 保持风格统一，且使用了更现代的 .ffill()）
    df_sorted['kurt_raw'] = df_sorted.groupby('stock')['pct'].transform(
        lambda x: x.rolling(20, min_periods=15).kurt().ffill()
    )

    # 3. 恢复原始索引对齐，确保数据不位移
    raw_series = df_sorted['kurt_raw'].reindex(data.index)

    # 4. 执行行业内按日健壮标准化
    # 注意：这里必须传入 df，而不是 data['date']，否则函数找不到 'sector' 列
    return _apply_sector_robust_zscore(raw_series, df)

# ==============================================================================
# 通用辅助函数：截面健壮标准化
# ==============================================================================
def _apply_sector_robust_zscore(series: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    确保 df 中包含 'date' 和 'sector' 列。
    利用 MAD 进行去极值，并在行业内进行 Z-score 标准化。
    """
    def robust_zscore(s):
        valid = s.dropna()
        if len(valid) < 5: 
            return s * 0
        
        # 1.4826 是 MAD 转标准差的缩放系数，3 代表稳健版的 3-sigma 原则
        med = valid.median()
        mad = (valid - med).abs().median()
        limit = 3 * 1.4826 * mad
        
        clipped = valid.clip(med - limit, med + limit)
        # 减去均值除以标准差，实现量纲统一
        return (clipped - clipped.mean()) / (clipped.std() + 1e-8)

    # 核心：必须按日期和行业进行双重分组，彻底消除行业偏见
    return series.groupby([df['date'], df['sector']]).transform(robust_zscore).fillna(0.0)

skewness_20d = register_factor(
    FactorDefinition(
        name='skewness_20d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=30,
        calculator=_skewness_20d,
        description='20日收益率偏度（暴涨暴跌的不对称性）',
    )
)


kurtosis_20d = register_factor(
    FactorDefinition(
        name='kurtosis_20d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=30,
        calculator=_kurtosis_20d,
        description='20日收益率峰度（肥尾风险/极端波动概率）',
    )
)



# ==============================================================================
# 3. 协同偏度 (Coskewness) - 替代 Idiosyncratic Volatility
# ==============================================================================
def _coskewness(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    market = _get_market_data(ctx)
    if 'daily_change_pct' not in market.columns:
        return pd.Series(0.0, index=data.index)

    # 1. 准备数据并注入行业信息
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'stock': data['stock_code'],
        'r_i': data['pct_chg'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown')
    }, index=data.index)
    
    # 对齐市场数据
    market_ret = market.set_index('date')['daily_change_pct']
    m_centered = market_ret - market_ret.mean()
    df['m_sq'] = df['date'].map(m_centered ** 2)

    # 2. 向量化计算原始协同偏度 (Coskewness Raw)
    df_sorted = df.sort_values(['stock', 'date'])
    i_mean = df_sorted.groupby('stock')['r_i'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df_sorted['i_centered'] = df_sorted['r_i'] - i_mean
    
    num = (df_sorted['i_centered'] * df_sorted['m_sq']).rolling(20, min_periods=10).mean()
    i_std = df_sorted.groupby('stock')['i_centered'].transform(lambda x: x.rolling(20, min_periods=10).std())
    m_sq_std = df_sorted.groupby('stock')['m_sq'].transform(lambda x: x.rolling(20, min_periods=10).std())
    
    df_sorted['raw_val'] = num / (i_std * m_sq_std + 1e-8)
    
    # 3. 索引恢复并执行行业内标准化
    raw_series = df_sorted['raw_val'].reindex(data.index)
    return _apply_sector_robust_zscore(raw_series, df)

# 注册因子保持不变
coskewness = register_factor(
    FactorDefinition(
        name='coskewness',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_market_data=True,
        lookback_days=30,
        calculator=_coskewness,
        description='协同偏度（集成去极值与日内Z-score版）',
    )
)
# ==============================================================================
# 4. 下行贝塔 (Downside Beta) - 真正的抗跌指标
# ==============================================================================
def _downside_beta(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    下行贝塔 (Downside Beta)
    
    逻辑：
    只使用市场收益 < 0 (或 < 均值) 的日子来计算 Beta。
    
    意义：
    投资者不关心上涨时的 Beta（那是好事），只关心下跌时的 Beta（那是亏损）。
    高 Downside Beta 的股票在熊市中跌得比大盘还惨。
    """
    market = _get_market_data(ctx)
    
    # 1. 筛选市场下跌日
    mkt_down = market[market['daily_change_pct'] < 0].set_index('date')['daily_change_pct']
    
    # 2. 映射市场收益
    df = data[['date', 'pct_chg', 'stock_code']].copy()
    df['mkt_ret'] = df['date'].map(mkt_down) # 上涨日会变成 NaN
    
    # 3. 计算 Beta = Cov(r_i, r_m) / Var(r_m)
    # 由于只保留了下跌日，这就是 Downside Beta
    grouped = df.groupby('stock_code', sort=False)
    
    # 协方差
    cov = grouped[['pct_chg', 'mkt_ret']].rolling(30, min_periods=10).cov().unstack().iloc[:, 1].reset_index(level=0, drop=True)
    
    # 市场方差 (下跌日)
    # 注意：这里的 Var 应该对应同期的市场下跌日方差
    # 简单起见，我们假设分母是个常数缩放，主要看分子 Cov 的变化
    # 或者为了严谨，我们计算 Rolling Var of Market Down
    # 为了效率，我们直接用 Cov 作为代理指标 (Downside Covariance)
    # 因为在截面上，同一天的 Var(r_m) 是相同的，对排序无影响！
    
    return cov.fillna(0.0).astype('float64')

downside_beta = register_factor(
    FactorDefinition(
        name='downside_beta',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_market_data=True,
        lookback_days=40,
        calculator=_downside_beta,
        description='下行贝塔（下跌市场中的协方差，衡量抗跌性）',
    )
)

__all__ = [
    'skewness_20d',
    'kurtosis_20d',
    'coskewness',
    'downside_beta'
]

