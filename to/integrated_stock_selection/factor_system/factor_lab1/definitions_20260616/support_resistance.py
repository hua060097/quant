"""
交易行为支撑/压力因子 (Trading Support/Resistance Alpha) - 大师级优化版。
核心思想：位置(Position) + 量能(Volume) = 强度(Strength)

优化日志：
1. 消除隐性填充：Numba 初始化由 np.zeros 改为 np.full(nan)，防止新股被误判为支撑崩塌。
2. 引入全局稳健 Z-Score：确保因子值在截面上可比，适配 Top-1 策略。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series
from ..universe import non_st_mask
from numba import njit

EPS = 1e-9

# ==============================================================================
# 0. 核心工具：全局按日稳健 Z-Score
# ==============================================================================
def _global_robust_zscore(
    series: pd.Series,
    date_series: pd.Series,
    eligible: pd.Series | None = None,
) -> pd.Series:
    """全局按日稳健 Z-Score (基于 Median/MAD)"""
    if eligible is None:
        eligible = pd.Series(True, index=series.index)
    eligible = eligible.reindex(series.index).fillna(True)
    df = pd.DataFrame({'val': series.where(eligible).values, 'date': date_series.values}, index=series.index)
    
    medians = df.groupby('date')['val'].transform('median')
    abs_diff = (df['val'] - medians).abs()
    
    temp_diff = pd.DataFrame({'diff': abs_diff.values, 'date': date_series.values}, index=series.index)
    mads = temp_diff.groupby('date')['diff'].transform('median')
    
    robust_std = mads * 1.4826
    z_score = (df['val'] - medians) / (robust_std + 1e-9)
    
    # 极值处理 (MAD=0)
    mask_zero_mad = (mads < 1e-6)
    if mask_zero_mad.any():
        means = df.groupby('date')['val'].transform('mean')
        stds = df.groupby('date')['val'].transform('std')
        z_fallback = (df['val'] - means) / (stds + 1e-9)
        z_score = z_score.where(~mask_zero_mad, z_fallback)
    
    # [修正] 不填 0，保留 NaN
    return z_score.where(eligible).clip(-6.0, 6.0).astype('float64')


@njit
def _sigmoid(x):
    """将任意数值平滑映射到 (0, 1)"""
    return 1.0 / (1.0 + np.exp(-x))

@njit
def _calc_sr_dynamics_numba(
    close, volume,
    lookback=20,      # 价格区间窗口
    vol_window=5      # 成交量短期窗口
):
    n = len(close)
    
    # [修正] 初始化为 NaN，而不是 0
    # 0.0 代表"支撑崩塌"，NaN 代表"数据不足/未知"
    support_score = np.full(n, np.nan)
    resistance_score = np.full(n, np.nan)
    
    # 必须有足够的数据才开始计算
    if n < lookback:
        return support_score, resistance_score
    
    for i in range(lookback, n):
        # 1. 获取过去的价格区间 (不含今天)
        past_close = close[i - lookback : i]
        p_min = np.min(past_close)
        p_max = np.max(past_close)
        range_span = p_max - p_min + EPS
        
        curr_price = close[i]
        
        # 2. 计算相对位置
        # < 0: 跌破下沿 (Breakdown)
        # > 1: 突破上沿 (Breakout)
        pos = (curr_price - p_min) / range_span
        
        # 3. 计算量能动态
        # 近期量能 vs 长期量能
        vol_recent = np.mean(volume[i - vol_window : i + 1]) 
        vol_base = np.mean(volume[i - lookback : i]) + EPS
        vol_ratio = vol_recent / vol_base
        
        # 量能因子化 (Sigmoid压缩)
        vol_factor = _sigmoid((vol_ratio - 1.0) * 2.0) 
        
        # ---------------------------------------
        # 4. 构建因子逻辑
        # ---------------------------------------
        
        # === 支撑强度 (Support Strength) ===
        dist_to_low = pos 
        
        if dist_to_low < 0:
            # 跌破 -> 支撑失效 (0分)
            support_score[i] = 0.0 
        elif dist_to_low > 1:
            # 突破 -> 远离支撑 (0分)
            support_score[i] = 0.0
        else:
            # 在区间内：越接近底部且缩量 -> 分数越高
            proximity_score = np.exp(-4.0 * dist_to_low)
            volume_score = 1.0 - vol_factor
            support_score[i] = proximity_score * volume_score

        # === 压力强度 (Resistance Pressure) ===
        dist_to_high = 1.0 - pos
        
        if pos > 1:
            # 突破 -> 压力消失 (0分)
            resistance_score[i] = 0.0
        elif pos < 0:
            # 跌破 -> 远离顶部 (0分)
            resistance_score[i] = 0.0
        else:
            # 在区间内：越接近顶部且放量 -> 压力越大
            proximity_score = np.exp(-4.0 * dist_to_high)
            volume_score = vol_factor
            resistance_score[i] = proximity_score * volume_score
            
    return support_score, resistance_score

def _support_resistance_alpha_calc(data: pd.DataFrame, _: FactorContext) -> tuple[pd.Series, pd.Series]:
    """
    计算支撑和压力Alpha因子
    """
    if 'close' not in data.columns or 'volume' not in data.columns:
        return pd.Series(np.nan, index=data.index), pd.Series(np.nan, index=data.index)

    # 准备 numpy 数组
    # 使用 groupby().apply() 保持时序逻辑
    def compute(g):
        c = g['close'].values.astype(np.float64)
        v = g['volume'].values.astype(np.float64)
        s, r = _calc_sr_dynamics_numba(c, v)
        return pd.DataFrame({'support': s, 'resistance': r}, index=g.index)

    res = data.groupby('stock_code', group_keys=False).apply(compute)
    
    # 确保索引对齐 (groupby apply 可能乱序，但在新版 pandas 中 group_keys=False 通常保留原索引)
    if not res.index.equals(data.index):
        res = res.reindex(data.index)
        
    return res['support'], res['resistance']

def _support_strength_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    s, _ = _support_resistance_alpha_calc(data, ctx)
    # [优化] 引入全局稳健标准化，与其他因子对齐
    return _global_robust_zscore(s, data['date'], eligible=non_st_mask(data))

def _resistance_pressure_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    _, r = _support_resistance_alpha_calc(data, ctx)
    # [优化] 引入全局稳健标准化
    return _global_robust_zscore(r, data['date'], eligible=non_st_mask(data))


def _rolling_prior_high(data: pd.DataFrame, window: int = 20, min_periods: int = 10) -> pd.Series:
    high = pd.to_numeric(data['high'], errors='coerce')
    prior_high = (
        high.groupby(data['stock_code'], sort=False)
        .rolling(window, min_periods=min_periods)
        .max()
        .reset_index(level=0, drop=True)
        .groupby(data['stock_code'], sort=False)
        .shift(1)
    )
    return prior_high.astype('float64')


def _volume_ratio_to_prior_base(data: pd.DataFrame, window: int = 20, min_periods: int = 10) -> pd.Series:
    if 'volume' in data.columns:
        volume = pd.to_numeric(data['volume'], errors='coerce')
    elif 'vol' in data.columns:
        volume = pd.to_numeric(data['vol'], errors='coerce')
    elif 'amount' in data.columns:
        volume = pd.to_numeric(data['amount'], errors='coerce')
    else:
        return pd.Series(np.nan, index=data.index, dtype='float64')

    base = (
        volume.groupby(data['stock_code'], sort=False)
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
        .groupby(data['stock_code'], sort=False)
        .shift(1)
    )
    return (volume / (base + EPS)).replace([np.inf, -np.inf], np.nan).clip(0.0, 30.0).astype('float64')


def _upper_shadow_ratio(data: pd.DataFrame) -> pd.Series:
    upper = data['high'] - np.maximum(data['open'], data['close'])
    ratio = upper / (data['close'] + EPS)
    return pd.Series(ratio, index=data.index).clip(0.0, 0.5).astype('float64')


def _prior_high_distance_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    当前收盘价相对“前20日最高价”的距离。

    负值：仍在前高下方；0附近：正在挑战前高；正值：已突破前高。
    """
    if not {'stock_code', 'high', 'close'}.issubset(data.columns):
        return pd.Series(np.nan, index=data.index, dtype='float64')
    prior_high = _rolling_prior_high(data, window=20, min_periods=10)
    distance = pd.to_numeric(data['close'], errors='coerce') / (prior_high + EPS) - 1.0
    return distance.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0).astype('float64')


def _prior_high_proximity_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    当前价是否处于前高博弈带。

    以距前高3%左右为核心带宽，越接近前高越接近1，远离前高趋近0。
    """
    distance = _prior_high_distance_20d(data, ctx).abs()
    proximity = np.exp(-distance / 0.03)
    return pd.Series(proximity, index=data.index).clip(0.0, 1.0).astype('float64')


def _prior_high_volume_contest_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    带量挑战前高的博弈强度。

    核心逻辑：越接近前高、越放量、越有上影/冲高回落，说明前高位置分歧越激烈。
    输出做按日 robust z-score，便于与其他横截面因子共同训练。
    """
    if not {'stock_code', 'date', 'open', 'high', 'close'}.issubset(data.columns):
        return pd.Series(np.nan, index=data.index, dtype='float64')

    distance = _prior_high_distance_20d(data, ctx)
    proximity = _prior_high_proximity_20d(data, ctx)
    volume_ratio = _volume_ratio_to_prior_base(data, window=20, min_periods=10).fillna(1.0)
    upper_shadow = _upper_shadow_ratio(data).fillna(0.0)

    # 仍在前高下方或刚突破不远时，分歧最有解释力；远离前高后不再当作前高博弈。
    contest_zone = ((distance >= -0.06) & (distance <= 0.04)).astype('float64')
    vol_excess = np.maximum(0.0, volume_ratio - 1.0)
    raw = proximity * contest_zone * vol_excess * (1.0 + upper_shadow * 10.0)
    raw = pd.Series(raw, index=data.index).replace([np.inf, -np.inf], np.nan).clip(0.0, 20.0)
    return _global_robust_zscore(raw, data['date'], eligible=non_st_mask(data))

# ==========================
# 注册因子
# ==========================

support_distance = register_factor(
    FactorDefinition(
        name='support_distance',
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=40,
        calculator=_support_strength_alpha,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _support_strength_alpha(d, c), target_index=ti
        ),
        description='支撑强度Alpha：缩量回踩底部的程度 (全局标准化)',
    )
)

resistance_distance = register_factor(
    FactorDefinition(
        name='resistance_distance', 
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=40,
        calculator=_resistance_pressure_alpha,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _resistance_pressure_alpha(d, c), target_index=ti
        ),
        description='压力强度Alpha：放量接近顶部的程度 (全局标准化)',
    )
)

prior_high_distance_20d = register_factor(
    FactorDefinition(
        name='prior_high_distance_20d',
        required_columns={'stock_code', 'date', 'high', 'close'},
        lookback_days=35,
        calculator=_prior_high_distance_20d,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _prior_high_distance_20d(d, c), target_index=ti
        ),
        description='当前收盘价相对前20日最高价的距离，负值为未突破，正值为突破',
    )
)

prior_high_proximity_20d = register_factor(
    FactorDefinition(
        name='prior_high_proximity_20d',
        required_columns={'stock_code', 'date', 'high', 'close'},
        lookback_days=35,
        calculator=_prior_high_proximity_20d,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _prior_high_proximity_20d(d, c), target_index=ti
        ),
        description='[0,1] 当前股价接近前20日高点的程度，越接近前高越高',
    )
)

prior_high_volume_contest_20d = register_factor(
    FactorDefinition(
        name='prior_high_volume_contest_20d',
        required_columns={'stock_code', 'date', 'open', 'high', 'close'},
        lookback_days=35,
        calculator=_prior_high_volume_contest_20d,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _prior_high_volume_contest_20d(d, c), target_index=ti
        ),
        description='[-6,6] 带量挑战前高的博弈强度：接近前高×超额量能×上影分歧',
    )
)

__all__ = [
    'support_distance',
    'resistance_distance',
    'prior_high_distance_20d',
    'prior_high_proximity_20d',
    'prior_high_volume_contest_20d',
]
