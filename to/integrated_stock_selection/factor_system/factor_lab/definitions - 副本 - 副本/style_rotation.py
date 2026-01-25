"""
风格轮动与市场结构因子（深度优化版）- 大师级修正。
核心关注：
1. 板块内部的微观结构（龙头效应、广度）。
2. 全市场的风格偏好（大小盘、高低位）。

优化日志：
1. 全局移除 .fillna(0) 和 .fillna(0.5)，保留 NaN (未知) 状态。
2. 确保所有因子在数据缺失时诚实地返回 NaN，不伪造中性信号。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


def _get_sector_map_safe(ctx: FactorContext) -> pd.Series:
    """获取行业映射"""
    if ctx.extra and 'stock_to_sw_sector' in ctx.extra:
        return ctx.extra['stock_to_sw_sector']
    raise ValueError("Missing 'stock_to_sw_sector' in context")


# ==============================================================================
# 1. 板块排名变化 (Sector Rank Change)
# ==============================================================================
def _sector_rank_change_5d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    SW行业5日排名变化
    """
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown'),
        'pct': data['pct_chg']
    }, index=data.index)

    # 1. 行业收益率排名
    sector_ret = df.groupby(['date', 'sector'])['pct'].mean()
    # 转为宽表：行是日期，列是行业
    sector_rank_df = sector_ret.unstack('sector').rank(axis=1, pct=True)
    
    # 2. 计算 5 日排名变化 (时序)
    rank_change_df = sector_rank_df - sector_rank_df.shift(5)
    
    # 3. 映射回个股
    rank_change_s = rank_change_df.stack().rename('rank_change')
    res = df.join(rank_change_s, on=['date', 'sector'], how='left')
    
    # [修正] 删除 fillna(0.0)，保留 NaN (clip 处理 NaN 不会报错，会保留 NaN)
    return res['rank_change'].clip(-1, 1).astype('float64')

sector_rank_change_5d = register_factor(
    FactorDefinition(
        name='sector_rank_change_5d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=10,
        calculator=_sector_rank_change_5d,
        description='SW行业5日排名变化（行业轮动核心因子）',
    )
)


# ==============================================================================
# 2. 板块动量加速度 (Sector Momentum Acceleration)
# ==============================================================================
def _sector_momentum_accel_rank(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    SW行业动量加速度分位排名
    """
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown'),
        'pct': data['pct_chg']
    }, index=data.index)

    # 计算行业收益
    sector_ret = df.groupby(['date', 'sector'])['pct'].mean()
    sector_ret_df = sector_ret.unstack('sector').sort_index()
    
    mom_5d = sector_ret_df.rolling(5, min_periods=3).mean()
    mom_20d = sector_ret_df.rolling(20, min_periods=10).mean()
    accel = mom_5d - mom_20d

    # 截面排名
    accel_rank = accel.rank(axis=1, pct=True)

    # 映射
    accel_rank_s = accel_rank.stack().rename('factor_value')
    df_result = df.join(accel_rank_s, on=['date', 'sector'], how='left')

    # [修正] 删除 fillna(0.5)，保留 NaN
    return df_result['factor_value'].astype('float64')

sector_momentum_accel_rank = register_factor(
    FactorDefinition(
        name='sector_momentum_accel_rank',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=25,
        calculator=_sector_momentum_accel_rank,
        description='SW行业动量加速度（5d-20d）分位排名',
    )
)


# ==============================================================================
# 3. 板块换手率突变 (Sector Turnover Change)
# ==============================================================================
def _sector_turnover_change_5d_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    SW行业换手率5d/20d变化
    """
    if 'turnover_rate' not in data.columns:
        return pd.Series(np.nan, index=data.index)

    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown'),
        'turnover': data['turnover_rate']
    }, index=data.index)

    # 行业平均换手
    sector_to = df.groupby(['date', 'sector'])['turnover'].mean().unstack('sector')
    
    # 计算 5d/20d 比率
    to_5 = sector_to.rolling(5, min_periods=3).mean()
    to_20 = sector_to.rolling(20, min_periods=10).mean()
    
    # [优化] 移除 Clip，保留原始比率的爆发力，排名会处理极端值
    ratio = to_5 / (to_20 + 1e-9)
    
    # 截面排名
    ratio_rank_df = ratio.rank(axis=1, pct=True)
    
    # 广播
    ratio_rank_s = ratio_rank_df.stack().rename('ratio_rank')
    res = df.join(ratio_rank_s, on=['date', 'sector'], how='left')
    
    # [修正] 删除 fillna(0.5)
    return res['ratio_rank'].astype('float64')

sector_turnover_change_5d_20d = register_factor(
    FactorDefinition(
        name='sector_turnover_change_5d_20d',
        required_columns={'stock_code', 'date', 'turnover_rate'},
        requires_sector_data=True,
        lookback_days=25,
        calculator=_sector_turnover_change_5d_20d,
        description='SW行业换手率5d/20d变化（资金流向）',
    )
)


# ==============================================================================
# 4. 板块 HHI 集中度变化 (Sector HHI Change)
# ==============================================================================
def _sector_hhi_change(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    SW行业成交量集中度变化
    """
    if 'volume' not in data.columns:
        return pd.Series(np.nan, index=data.index)

    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'stock': data['stock_code'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown'),
        'volume': data['volume']
    }, index=data.index)

    # 计算 HHI
    sector_total_vol = df.groupby(['date', 'sector'])['volume'].transform('sum')
    df['share'] = df['volume'] / (sector_total_vol + 1e-9)
    df['share_sq'] = df['share'] ** 2
    hhi_series = df.groupby(['date', 'sector'])['share_sq'].sum()

    # 时序计算
    hhi_wide = hhi_series.unstack('sector').sort_index()
    hhi_5 = hhi_wide.rolling(5, min_periods=3).mean()
    hhi_20 = hhi_wide.rolling(20, min_periods=10).mean()
    hhi_diff = hhi_5 - hhi_20 # 这里不 fillna，如果数据不够就算不出 diff

    # 截面排名
    hhi_rank_wide = hhi_diff.rank(axis=1, pct=True)

    # 广播
    hhi_final_s = hhi_rank_wide.stack().rename('hhi_rank')
    df = df.join(hhi_final_s, on=['date', 'sector'], how='left')

    # [修正] 删除 fillna(0.5)
    return df['hhi_rank'].astype('float64')

sector_hhi_change = register_factor(
    FactorDefinition(
        name='sector_hhi_change',
        required_columns={'stock_code', 'date', 'volume'},
        requires_sector_data=True,
        lookback_days=25,
        calculator=_sector_hhi_change,
        description='SW行业成交量集中度变化（龙头效应切换）',
    )
)


# ==============================================================================
# 5. 板块相对强度 (Relative Strength vs Market)
# ==============================================================================
def _sector_relative_strength_vs_market(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    SW行业相对全市场强弱
    """
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown'),
        'pct': data['pct_chg']
    }, index=data.index)

    # 行业平均收益
    sector_ret = df.groupby(['date', 'sector'])['pct'].mean().unstack('sector')
    
    # 截面排名
    rel_rank = sector_ret.rank(axis=1, pct=True)
    
    # 广播
    rel_rank_s = rel_rank.stack().rename('rel_rank_factor')
    res = df.join(rel_rank_s, on=['date', 'sector'], how='left')
    
    # [修正] 删除 fillna(0.5)
    return res['rel_rank_factor'].astype('float64')

sector_relative_strength_vs_market = register_factor(
    FactorDefinition(
        name='sector_relative_strength_vs_market',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=10,
        calculator=_sector_relative_strength_vs_market,
        description='SW行业相对全市场强弱（行业超额收益）',
    )
)


# ==============================================================================
# 6. 赫芬达尔集中度 (HHI Concentration)
# ==============================================================================
def _sector_concentration_hhi(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    板块成交量集中度 (HHI Index)
    """
    if 'volume' not in data.columns:
        return pd.Series(np.nan, index=data.index)

    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'vol': data['volume'].values,
        'date': data['date'].values,
        'sector': data['stock_code'].map(sector_map).fillna('Unknown').values
    }, index=data.index)
    
    sector_total_vol = df.groupby(['date', 'sector'])['vol'].transform('sum')
    share = df['vol'] / (sector_total_vol + 1e-9)
    df['share_sq'] = share ** 2
    hhi = df.groupby(['date', 'sector'])['share_sq'].transform('sum')
    
    # [修正] 删除 fillna(0.0)
    return hhi.astype('float64')

sector_concentration = register_factor(
    FactorDefinition(
        name='sector_concentration',
        required_columns={'stock_code', 'date', 'volume'},
        requires_sector_data=True,
        lookback_days=1,
        calculator=_sector_concentration_hhi,
        description='板块成交量赫芬达尔集中度(HHI)',
    )
)


# ==============================================================================
# 7. 板块广度推力 (Sector Breadth Thrust)
# ==============================================================================
def _sector_breadth_thrust(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    板块净上涨家数占比
    """
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'pct': data['pct_chg'].values,
        'date': data['date'].values,
        'sector': data['stock_code'].map(sector_map).fillna('Unknown').values
    }, index=data.index)
    
    # 上涨记1，下跌记-1
    df['status'] = np.where(df['pct'] > 0.2, 1.0, np.where(df['pct'] < -0.2, -1.0, 0.0))
    
    breadth = df.groupby(['date', 'sector'])['status'].transform('mean')
    
    # [修正] 删除 fillna(0.0)
    return breadth.astype('float64')

sector_breadth = register_factor(
    FactorDefinition(
        name='sector_breadth',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=1,
        calculator=_sector_breadth_thrust,
        description='板块净上涨家数占比(广度推力)',
    )
)


# ==============================================================================
# 8. 大小盘风格轮动 (Size Style Rotation)
# ==============================================================================
def _size_rotation_strength(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    全市场小盘股相对于大盘股的超额收益
    """
    if 'circ_mv' not in data.columns or 'pct_chg' not in data.columns:
        return pd.Series(np.nan, index=data.index)
    
    df = data[['date', 'circ_mv', 'pct_chg']].copy()
    
    groups = df.groupby('date')['circ_mv']
    q80 = groups.transform(lambda x: x.quantile(0.8))
    q20 = groups.transform(lambda x: x.quantile(0.2))
    
    is_large = df['circ_mv'] >= q80
    is_small = df['circ_mv'] <= q20
    
    # 利用 boolean indexing 计算每日风格收益
    # 这种写法可能会产生 NaN (如果某天没有小盘股数据)，保留 NaN 是对的
    large_ret = df[is_large].groupby('date')['pct_chg'].mean()
    small_ret = df[is_small].groupby('date')['pct_chg'].mean()
    
    size_premium = small_ret - large_ret
    
    # [修正] 删除 fillna(0.0)
    return data['date'].map(size_premium).astype('float64')

size_rotation_strength = register_factor(
    FactorDefinition(
        name='size_rotation_strength',
        required_columns={'stock_code', 'date', 'circ_mv', 'pct_chg'},
        lookback_days=1,
        calculator=_size_rotation_strength,
        description='全市场小盘股相对于大盘股的超额收益(风格轮动)',
    )
)


# ==============================================================================
# 9. 高低位切换强度 (Value/Growth Rotation)
# ==============================================================================
def _value_growth_rotation(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    低位股相对于高位股的超额收益
    """
    if 'close' not in data.columns or 'pct_chg' not in data.columns:
        return pd.Series(np.nan, index=data.index)
        
    grouped = data.groupby('stock_code', sort=False)['close']
    prev_close = grouped.shift(1)
    prev_20 = grouped.shift(21)
    
    # 计算过去20日动量 (因子本身会有 NaN)
    mom_20 = (prev_close - prev_20) / (prev_20 + 1e-9)
    
    df = pd.DataFrame({
        'mom': mom_20.values,
        'pct': data['pct_chg'].values,
        'date': data['date'].values
    }, index=data.index).dropna()
    
    if df.empty:
        return pd.Series(np.nan, index=data.index)

    groups = df.groupby('date')['mom']
    q80 = groups.transform(lambda x: x.quantile(0.8))
    q20 = groups.transform(lambda x: x.quantile(0.2))
    
    is_high = df['mom'] >= q80
    is_low = df['mom'] <= q20
    
    high_ret = df[is_high].groupby('date')['pct'].mean()
    low_ret = df[is_low].groupby('date')['pct'].mean()
    
    # 低位 - 高位
    rev_premium = low_ret - high_ret
    
    # 平滑 (3日均值)
    # rev_premium 是按日期索引的 Series，rolling 会按时间顺序计算
    rev_premium_smooth = rev_premium.rolling(3).mean()
    
    # [修正] 删除 fillna(0.0)
    return data['date'].map(rev_premium_smooth).astype('float64')

value_growth_rotation = register_factor(
    FactorDefinition(
        name='value_growth_rotation',
        required_columns={'stock_code', 'date', 'close', 'pct_chg'},
        lookback_days=30,
        calculator=_value_growth_rotation,
        description='低位股相对于高位股的超额收益(高低切换强度)',
    )
)

__all__ = [
    'sector_concentration',
    'sector_breadth',
    'size_rotation_strength',
    'value_growth_rotation',
    'sector_rank_change_5d',
    'sector_momentum_accel_rank',
    'sector_turnover_change_5d_20d',
    'sector_hhi_change',
    'sector_relative_strength_vs_market'
]