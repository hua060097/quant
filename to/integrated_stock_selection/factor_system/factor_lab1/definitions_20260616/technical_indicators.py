"""
技术指标因子（深度优化版）- 大师级修正。
重点解决：量纲统一(PPO)、引入成交量权重(MFI)、正交化趋势检测(Vortex)。

优化日志：
1. MFI: 删除冗余的 prev_tp 计算代码，移除 fillna(0)。
2. Vortex: 删除重复计算的逻辑，仅保留安全的 transform 写法，移除 fillna(0)。
3. Mass Index: 移除 fillna(0)。
4. 全局: 确保 NaN 透传，不伪造技术指标信号。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series

# ==============================================================================
# 1. PPO (Percentage Price Oscillator) - MACD 的专业替代版
# ==============================================================================
def _ppo_strength(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    PPO 趋势强度因子 (标准化 MACD)
    """
    if 'close' not in data.columns:
        return pd.Series(np.nan, index=data.index)

    # 1. 计算 PPO
    grouped = data.groupby('stock_code', sort=False)['close']
    ema12 = grouped.ewm(span=12, adjust=False).mean().reset_index(level=0, drop=True)
    ema26 = grouped.ewm(span=26, adjust=False).mean().reset_index(level=0, drop=True)
    
    # 避免分母为 0
    ppo_line = ((ema12 - ema26) / (ema26 + 1e-9)) * 100
    
    # 2. 计算 Signal Line (9日 EMA of PPO)
    # 构造临时 DF 以便再次 groupby
    temp = pd.DataFrame({'ppo': ppo_line.values, 'stock_code': data['stock_code'].values}, index=data.index)
    ppo_signal = temp.groupby('stock_code', sort=False)['ppo'].ewm(span=9, adjust=False).mean().reset_index(level=0, drop=True)
    
    # 3. 计算 Histogram (动能)
    ppo_hist = ppo_line - ppo_signal
    
    # 4. 因子构造：趋势 + 加速
    score = ppo_line * 0.5 + ppo_hist * 1.5
    
    # clip 不会填充 NaN，它会保留 NaN，这是对的
    return score.clip(-10, 10).astype('float64')


def _ppo_strength_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return select_target_series(_ppo_strength(data, _), target_index=target_index)

ppo_strength = register_factor(
    FactorDefinition(
        name='ppo_strength',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=40,
        calculator=_ppo_strength,
        calculation_scope='stateful_history',
        target_calculator=_ppo_strength_target,
        description='PPO价格百分比震荡指标（解决MACD量纲问题）',
    )
)

# ==============================================================================
# 2. MFI (Money Flow Index) - 带量的 RSI
# ==============================================================================
def _mfi_divergence(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    资金流量指标 (Money Flow Index)
    """
    cols = ['high', 'low', 'close', 'volume']
    if not all(c in data.columns for c in cols):
        return pd.Series(np.nan, index=data.index)

    # 1. 计算典型价格 (Typical Price)
    tp = (data['high'] + data['low'] + data['close']) / 3.0
    
    # 2. 计算原始资金流 (Raw Money Flow)
    rmf = tp * data['volume']
    
    # 3. 区分正负流向
    # [优化] 删除冗余的 grouped block，直接构建 efficient calculation
    
    # 构造 shift 后的 TP
    tp_series = pd.Series(tp.values, index=data.index)
    tp_prev = tp_series.groupby(data['stock_code'], sort=False).shift(1)
    
    pos_flow = np.where(tp > tp_prev, rmf, 0.0)
    neg_flow = np.where(tp < tp_prev, rmf, 0.0)
    
    # 4. 滚动求和 (14日)
    flow_df = pd.DataFrame({
        'pos': pos_flow, 
        'neg': neg_flow, 
        'stock_code': data['stock_code']
    }, index=data.index)
    
    grouped_flow = flow_df.groupby('stock_code', sort=False)
    pos_sum = grouped_flow['pos'].rolling(14, min_periods=10).sum().reset_index(level=0, drop=True)
    neg_sum = grouped_flow['neg'].rolling(14, min_periods=10).sum().reset_index(level=0, drop=True)
    
    # 5. 计算 MFI
    mfi = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-9)))
    
    # 6. 因子化处理
    # MFI < 20 超卖，MFI > 80 超买
    mfi_score = (mfi - 50) / 20.0  # 映射到大致 [-2.5, 2.5]
    
    # [修正] 删除 fillna(0.0)，保留 NaN
    return mfi_score.astype('float64')


def _mfi_divergence_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return select_target_series(_mfi_divergence(data, _), target_index=target_index)

mfi_strength = register_factor(
    FactorDefinition(
        name='mfi_strength',
        required_columns={'stock_code', 'date', 'high', 'low', 'close', 'volume'},
        lookback_days=30,
        calculator=_mfi_divergence,
        calculation_scope='target_with_history_window',
        target_calculator=_mfi_divergence_target,
        description='资金流量指标（成交量加权的RSI）',
    )
)

# ==============================================================================
# 3. 优化版：旋涡爆发强度 (Vortex Burst)
# ==============================================================================
def _vortex_burst(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    [优化] 旋涡爆发强度 (Vortex Burst)
    """
    cols = ['high', 'low', 'close']
    if not all(c in data.columns for c in cols):
        return pd.Series(np.nan, index=data.index)

    # 预计算 shift
    shifted = data[['high', 'low', 'close']].groupby(data['stock_code'], sort=False).shift(1)
    
    high = data['high']
    low = data['low']
    close_prev = shifted['close']
    # high_prev = shifted['high'] # 未使用，可注释
    low_prev = shifted['low']
    high_prev_actual = shifted['high'] # 实际上 vm_minus 需要 high_prev
    
    # 1. True Range (TR)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # 2. Vortex Movement (VM)
    vm_plus = (high - low_prev).abs()
    vm_minus = (low - high_prev_actual).abs() # 修正：这里应该是 low - high_prev
    
    # 3. Rolling Sum (10日)
    # [优化] 直接使用 transform 构造对齐的 Series，删除冗余的 unsafe rolling 代码
    temp_df = pd.DataFrame({
        'tr': tr,
        'vm_plus': vm_plus,
        'vm_minus': vm_minus,
        'stock_code': data['stock_code']
    }, index=data.index)
    
    window = 10
    g = temp_df.groupby('stock_code', sort=False)
    
    tr_s = g['tr'].transform(lambda x: x.rolling(window, min_periods=5).sum())
    vp_s = g['vm_plus'].transform(lambda x: x.rolling(window, min_periods=5).sum())
    vm_s = g['vm_minus'].transform(lambda x: x.rolling(window, min_periods=5).sum())
    
    # 4. 计算归一化剪刀差
    vortex_burst = (vp_s - vm_s) / (vp_s + vm_s + 1e-9)
    
    # [修正] 删除 fillna(0.0)
    return vortex_burst.astype('float64')


def _vortex_burst_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return select_target_series(_vortex_burst(data, _), target_index=target_index)

vortex_trend = register_factor(
    FactorDefinition(
        name='vortex_trend',
        required_columns={'stock_code', 'date', 'high', 'low', 'close'},
        lookback_days=20,
        calculator=_vortex_burst,
        calculation_scope='target_with_history_window',
        target_calculator=_vortex_burst_target,
        description='[优化] 旋涡爆发强度 (Vortex Normalized Burst)',
    )
)

# ==============================================================================
# 4. Mass Index (梅斯线) - 专门捕捉反转
# ==============================================================================
def _mass_index_reversal(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    梅斯线 (Mass Index)
    """
    cols = ['high', 'low']
    if not all(c in data.columns for c in cols):
        return pd.Series(np.nan, index=data.index)
        
    # 1. High - Low Range
    hl_range = data['high'] - data['low']
    
    grouped_range = hl_range.groupby(data['stock_code'], sort=False)
    
    # 2. EMA(Range, 9)
    ema9 = grouped_range.ewm(span=9, adjust=False).mean().reset_index(level=0, drop=True)
    
    # 3. EMA(EMA(Range, 9), 9)
    ema9_df = pd.DataFrame({'val': ema9.values, 'stock_code': data['stock_code'].values}, index=data.index)
    ema_ema9 = ema9_df.groupby('stock_code', sort=False)['val'].ewm(span=9, adjust=False).mean().reset_index(level=0, drop=True)
    
    # 4. Ratio
    ratio = ema9 / (ema_ema9 + 1e-9)
    
    # 5. Mass Index = Sum(Ratio, 25)
    ratio_df = pd.DataFrame({'val': ratio.values, 'stock_code': data['stock_code'].values}, index=data.index)
    mass_index = ratio_df.groupby('stock_code', sort=False)['val'].rolling(25, min_periods=20).sum().reset_index(level=0, drop=True)
    
    # 6. 因子化
    mass_score = (mass_index - 25.0) / 2.0
    
    # [修正] 删除 fillna(0.0)
    return mass_score.astype('float64')


def _mass_index_reversal_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return select_target_series(_mass_index_reversal(data, _), target_index=target_index)

mass_index_reversal = register_factor(
    FactorDefinition(
        name='mass_index_reversal',
        required_columns={'stock_code', 'date', 'high', 'low'},
        lookback_days=60,
        calculator=_mass_index_reversal,
        calculation_scope='stateful_history',
        target_calculator=_mass_index_reversal_target,
        description='梅斯线（基于波动区间的反转预警）',
    )
)


__all__ = [
    'ppo_strength',
    'mfi_strength',
    'vortex_trend',
    'mass_index_reversal',
]
