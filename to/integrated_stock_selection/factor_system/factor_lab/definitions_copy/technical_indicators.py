"""
技术指标因子（深度优化版）。
重点解决：量纲统一(PPO)、引入成交量权重(MFI)、正交化趋势检测(Vortex)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 1. PPO (Percentage Price Oscillator) - MACD 的专业替代版
# ==============================================================================
def _ppo_strength(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    PPO 趋势强度因子 (标准化 MACD)
    
    逻辑：
    ((EMA12 - EMA26) / EMA26) * 100
    解决传统 MACD 在高价股和低价股之间不可比的问题。
    
    输出：
    - PPO Histogram (动能柱): 反映加速趋势
    - PPO Line: 反映绝对趋势
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)

    # 1. 计算 PPO
    # 为了速度，直接用 pandas ewm
    ema12 = data.groupby('stock_code', sort=False)['close'].ewm(span=12, adjust=False).mean().reset_index(level=0, drop=True)
    ema26 = data.groupby('stock_code', sort=False)['close'].ewm(span=26, adjust=False).mean().reset_index(level=0, drop=True)
    
    ppo_line = ((ema12 - ema26) / (ema26 + 1e-9)) * 100
    
    # 2. 计算 Signal Line (9日 EMA of PPO)
    # 注意：这里需要再次 groupby 计算 ewm
    # 为了避免复杂的索引对齐，我们构造临时 DF
    temp = pd.DataFrame({'ppo': ppo_line.values, 'stock_code': data['stock_code'].values}, index=data.index)
    ppo_signal = temp.groupby('stock_code', sort=False)['ppo'].ewm(span=9, adjust=False).mean().reset_index(level=0, drop=True)
    
    # 3. 计算 Histogram (动能)
    ppo_hist = ppo_line - ppo_signal
    
    # 4. 因子构造：趋势 + 加速
    # 仅仅知道 PPO > 0 (趋势向上) 不够，我们更关心 PPO_Hist > 0 (趋势加速)
    # 综合评分：PPO绝对位置 + 2倍的动能柱
    score = ppo_line * 0.5 + ppo_hist * 1.5
    
    # 简单的去极值
    return score.clip(-10, 10).astype('float64')

ppo_strength = register_factor(
    FactorDefinition(
        name='ppo_strength',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=40,
        calculator=_ppo_strength,
        description='PPO价格百分比震荡指标（解决MACD量纲问题）',
    )
)

# ==============================================================================
# 2. MFI (Money Flow Index) - 带量的 RSI
# ==============================================================================
def _mfi_divergence(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    资金流量指标 (Money Flow Index)
    
    逻辑：
    RSI 只看价格涨跌，MFI 引入成交量权重。
    MFI = 100 - 100 / (1 + Money_Ratio)
    
    核心价值：
    当价格创新高但 MFI 没有创新高（背离），或者 MFI 在低位放量回升，是极佳信号。
    此因子与单纯的 Price Momentum 相关性较低。
    """
    cols = ['high', 'low', 'close', 'volume']
    if not all(c in data.columns for c in cols):
        return pd.Series(50.0, index=data.index)

    # 1. 计算典型价格 (Typical Price)
    tp = (data['high'] + data['low'] + data['close']) / 3.0
    
    # 2. 计算原始资金流 (Raw Money Flow)
    rmf = tp * data['volume']
    
    # 3. 区分正负流向
    # 需要 shift 比较
    grouped = data.groupby('stock_code', sort=False)
    prev_tp = grouped['high'].shift(1).fillna(tp) # 暂用high shift做占位，实际应该用tp shift
    # 重新正确计算 shift
    # 为了效率，我们用 diff 逻辑
    # 注意：这里需要确保 groupby 正确
    
    # 更高效的写法：
    # 构造 shift 后的 TP
    tp_series = pd.Series(tp.values, index=data.index)
    tp_prev = tp_series.groupby(data['stock_code'], sort=False).shift(1)
    
    pos_flow = np.where(tp > tp_prev, rmf, 0.0)
    neg_flow = np.where(tp < tp_prev, rmf, 0.0)
    
    # 4. 滚动求和 (14日)
    # 将 numpy array 转回 series 以利用 rolling
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
    
    # 6. 因子化处理：我们要找的是“超卖反转”和“资金推动”
    # MFI < 20 是超卖，MFI > 80 是超买
    # 但线性模型不喜欢 0-100 的区间，我们将其中心化并缩放
    mfi_score = (mfi - 50) / 20.0  # 映射到大致 [-2.5, 2.5]
    
    return mfi_score.fillna(0.0).astype('float64')

mfi_strength = register_factor(
    FactorDefinition(
        name='mfi_strength',
        required_columns={'stock_code', 'date', 'high', 'low', 'close', 'volume'},
        lookback_days=30,
        calculator=_mfi_divergence,
        description='资金流量指标（成交量加权的RSI）',
    )
)

# ==============================================================================
# 3. Vortex Indicator (旋涡指标) - 纯粹的趋势发现
# ==============================================================================
def _vortex_trend(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    旋涡指标 (VI)
    
    逻辑：
    利用 High-Low 的波幅和 High-PrevLow 的距离来判断趋势。
    它不依赖均线，因此与 MA 类因子互补性极强。
    
    计算：
    VM+ = Abs(CurrentHigh - PrevLow)
    VM- = Abs(CurrentLow - PrevHigh)
    """
    cols = ['high', 'low', 'close']
    if not all(c in data.columns for c in cols):
        return pd.Series(0.0, index=data.index)

    grouped = data.groupby('stock_code', sort=False)
    high = data['high']
    low = data['low']
    close = data['close']
    
    high_prev = grouped['high'].shift(1)
    low_prev = grouped['low'].shift(1)
    close_prev = grouped['close'].shift(1)
    
    # 1. True Range (TR)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # 2. Vortex Movement (VM)
    vm_plus = (high - low_prev).abs()
    vm_minus = (low - high_prev).abs()
    
    # 3. Rolling Sum (14日)
    # 为了方便，构造 DF
    calc_df = pd.DataFrame({
        'tr': tr, 'vm_plus': vm_plus, 'vm_minus': vm_minus, 
        'stock_code': data['stock_code']
    }, index=data.index)
    
    g = calc_df.groupby('stock_code', sort=False)
    tr_sum = g['tr'].rolling(14, min_periods=10).sum().reset_index(level=0, drop=True)
    vm_plus_sum = g['vm_plus'].rolling(14, min_periods=10).sum().reset_index(level=0, drop=True)
    vm_minus_sum = g['vm_minus'].rolling(14, min_periods=10).sum().reset_index(level=0, drop=True)
    
    # 4. VI Lines
    vi_plus = vm_plus_sum / (tr_sum + 1e-9)
    vi_minus = vm_minus_sum / (tr_sum + 1e-9)
    
    # 5. 因子化：净趋势强度
    # VI+ > VI- 表示上升趋势
    vortex_diff = vi_plus - vi_minus
    
    return vortex_diff.fillna(0.0).astype('float64')

vortex_trend = register_factor(
    FactorDefinition(
        name='vortex_trend',
        required_columns={'stock_code', 'date', 'high', 'low', 'close'},
        lookback_days=30,
        calculator=_vortex_trend,
        description='旋涡指标趋势强度（基于高低点波幅的趋势判断）',
    )
)

# ==============================================================================
# 4. Mass Index (梅斯线) - 专门捕捉反转
# ==============================================================================
def _mass_index_reversal(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    梅斯线 (Mass Index)
    
    逻辑：
    基于高低价差幅（Range）的扩大和收缩来预测反转。
    当 Mass Index 变得非常高（>27）然后跌破 26.5 时，通常是趋势反转信号（Reversal Bulge）。
    
    因子化处理：
    我们不输出原始 Mass Index，而是输出它的变化率和极值状态，帮助模型识别“变盘点”。
    """
    cols = ['high', 'low']
    if not all(c in data.columns for c in cols):
        return pd.Series(0.0, index=data.index)
        
    # 1. High - Low Range
    hl_range = data['high'] - data['low']
    
    grouped_range = hl_range.groupby(data['stock_code'], sort=False)
    
    # 2. EMA(Range, 9)
    ema9 = grouped_range.ewm(span=9, adjust=False).mean().reset_index(level=0, drop=True)
    
    # 3. EMA(EMA(Range, 9), 9)
    # 需要再次 groupby
    ema9_df = pd.DataFrame({'val': ema9.values, 'stock_code': data['stock_code'].values}, index=data.index)
    ema_ema9 = ema9_df.groupby('stock_code', sort=False)['val'].ewm(span=9, adjust=False).mean().reset_index(level=0, drop=True)
    
    # 4. Ratio
    ratio = ema9 / (ema_ema9 + 1e-9)
    
    # 5. Mass Index = Sum(Ratio, 25)
    ratio_df = pd.DataFrame({'val': ratio.values, 'stock_code': data['stock_code'].values}, index=data.index)
    mass_index = ratio_df.groupby('stock_code', sort=False)['val'].rolling(25, min_periods=20).sum().reset_index(level=0, drop=True)
    
    # 6. 因子化：
    # Mass Index 越高，波动范围扩大的越厉害，变盘概率越大。
    # 我们将其标准化，均值通常在 25 左右。
    mass_score = (mass_index - 25.0) / 2.0
    
    return mass_score.fillna(0.0).astype('float64')

mass_index_reversal = register_factor(
    FactorDefinition(
        name='mass_index_reversal',
        required_columns={'stock_code', 'date', 'high', 'low'},
        lookback_days=60,
        calculator=_mass_index_reversal,
        description='梅斯线（基于波动区间的反转预警）',
    )
)


__all__ = [
    'ppo_strength',
    'mfi_strength',
    'vortex_trend',
    'mass_index_reversal',
]

