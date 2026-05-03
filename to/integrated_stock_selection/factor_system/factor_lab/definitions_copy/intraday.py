"""
日内与隔夜微观结构因子（Intraday & Overnight）。
核心关注：隔夜信息的消化程度、日内博弈的胜负、开盘后的聪明钱流向。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 1. 智能隔夜动能 (Smart Overnight Momentum)
# ==============================================================================
def _smart_overnight_return(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    标准化隔夜收益 (Normalized Overnight Return)
    
    逻辑：
    (Open_T - Close_T-1) / Volatility_T-1
    
    意义：
    衡量隔夜信息的“意外程度”。
    如果波动率平时很低，突然大幅高开，这是一个极其强烈的信号（重大利好）。
    """
    if 'open' not in data.columns:
        return pd.Series(0.0, index=data.index)
        
    grouped = data.groupby('stock_code', sort=False)
    close_prev = grouped['close'].shift(1)
    
    # 隔夜绝对涨幅
    overnight = (data['open'] - close_prev) / (close_prev + 1e-9)
    
    # 用过去5天的波动率来标准化
    # 这样能区分：是“平时就这德行”还是“突发利好”
    vol = grouped['close'].pct_change().rolling(5).std()
    
    score = overnight / (vol + 1e-9)
    
    return score.fillna(0.0).clip(-5, 5).astype('float64')

overnight_return = register_factor(
    FactorDefinition(
        name='overnight_return',
        required_columns={'stock_code', 'date', 'open', 'close'},
        lookback_days=10,
        calculator=_smart_overnight_return,
        description='标准化隔夜收益（衡量隔夜信息的意外程度）',
    )
)

# ==============================================================================
# 2. 日内反转效应 (Intraday Reversal) - 聪明钱 vs 傻钱
# ==============================================================================
def _intraday_reversal(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    日内走势与隔夜走势的对抗 (Gap vs Intraday)
    
    逻辑：
    (Close - Open) * (Open - Prev_Close)
    
    含义：
    - 负值：高开低走（套人），或者低开高走（洗盘）。这是主力最喜欢的手法。
    - 正值：高开高走（逼空），或者低开低走（崩盘）。
    
    优化：
    我们计算一个“反包意愿”指标：
    Intraday_Return / Abs(Overnight_Return)
    """
    grouped = data.groupby('stock_code', sort=False)
    close_prev = grouped['close'].shift(1)
    
    overnight_ret = (data['open'] - close_prev) / (close_prev + 1e-9)
    intraday_ret = (data['close'] - data['open']) / (data['open'] + 1e-9)
    
    # 如果隔夜幅度很小，这个因子没意义
    mask = overnight_ret.abs() > 0.005 # 0.5% 的缺口才有博弈价值
    
    # 强力回补因子
    # 例如：低开 -3%，盘中涨 +4%，结果是 +4 / 3 = 1.33 (强力反包)
    # 例如：高开 +3%，盘中跌 -4%，结果是 -4 / 3 = -1.33 (乌云盖顶)
    reversal_force = intraday_ret / (overnight_ret.abs() + 1e-9)
    
    result = np.where(mask, reversal_force, 0.0)
    
    return pd.Series(result, index=data.index).clip(-3, 3).astype('float64')

intraday_reversal = register_factor(
    FactorDefinition(
        name='intraday_reversal',
        required_columns={'stock_code', 'date', 'open', 'close'},
        lookback_days=5,
        calculator=_intraday_reversal,
        description='日内走势对隔夜缺口的回补/反包力度',
    )
)

# ==============================================================================
# 3. 开盘攻击波 (Open to High) - 早盘买气
# ==============================================================================
def _open_to_high_strength(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    开盘攻击强度
    
    逻辑：
    (High - Open) / (High - Low)
    
    意义：
    早盘开盘后，多头能把价格推多高？
    - 1.0: 收盘在最高，或者全天在Open上方运行（极强）。
    - 接近0: 开盘即巅峰（极弱）。
    """
    range_hl = data['high'] - data['low']
    push_up = data['high'] - data['open']
    
    strength = push_up / (range_hl + 1e-9)
    
    return strength.fillna(0.5).astype('float64')

open_to_high_ratio = register_factor(
    FactorDefinition(
        name='open_to_high_ratio',
        required_columns={'stock_code', 'date', 'open', 'high', 'low'},
        lookback_days=0,
        calculator=_open_to_high_strength,
        description='开盘后多头向上拓展空间的能力',
    )
)


__all__ = [
    'overnight_return',
    'intraday_reversal',
    'open_to_high_ratio',
]

