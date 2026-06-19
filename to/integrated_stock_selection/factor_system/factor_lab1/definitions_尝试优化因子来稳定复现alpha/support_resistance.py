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
from numba import njit

EPS = 1e-9

# ==============================================================================
# 0. 核心工具：全局按日稳健 Z-Score
# ==============================================================================
def _global_robust_zscore(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """全局按日稳健 Z-Score (基于 Median/MAD)"""
    df = pd.DataFrame({'val': series.values, 'date': date_series.values}, index=series.index)
    
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
    return z_score.clip(-6.0, 6.0).astype('float64')


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
    return _global_robust_zscore(s, data['date'])

def _resistance_pressure_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    _, r = _support_resistance_alpha_calc(data, ctx)
    # [优化] 引入全局稳健标准化
    return _global_robust_zscore(r, data['date'])

# ==========================
# 注册因子
# ==========================

support_distance = register_factor(
    FactorDefinition(
        name='support_distance',
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=40,
        calculator=_support_strength_alpha,
        description='支撑强度Alpha：缩量回踩底部的程度 (全局标准化)',
    )
)

resistance_distance = register_factor(
    FactorDefinition(
        name='resistance_distance', 
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=40,
        calculator=_resistance_pressure_alpha,
        description='压力强度Alpha：放量接近顶部的程度 (全局标准化)',
    )
)

__all__ = ['support_distance', 'resistance_distance']