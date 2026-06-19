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
    
    # return score.fillna(0.0).clip(-5, 5).astype('float64')
    return score.clip(-5, 5).astype('float64')

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
# 1. 优化版日内反转：日内强度 (Intraday Intensity)
# ==============================================================================
def _intraday_intensity(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    [优化] 日内强度因子 (Intraday Intensity)
    
    原逻辑问题：
    1. mask > 0.5% 导致大部分样本为 0 (稀疏)。
    2. 除以隔夜收益导致数值不稳定 (小分母爆炸)。
    
    优化逻辑：
    计算“实体率” —— (收盘 - 开盘) / (最高 - 最低)。
    这直接衡量了日内多空博弈的胜负，不需要依赖隔夜缺口的大小。
    
    含义：
    - 接近 1：光头光脚大阳线 (日内买盘完胜)
    - 接近 -1：光头光脚大阴线 (日内卖盘完胜)
    - 接近 0：十字星 (多空平衡)
    """
    # 1. 计算日内振幅 (Range)
    high_low_range = data['high'] - data['low']
    
    # 2. 计算日内涨跌 (Body)
    close_open = data['close'] - data['open']
    
    # 3. 计算强度 (Intensity)
    # 分母加 1e-9 防止除零
    # 这一步已经归一化到了 [-1, 1]，无需 clip
    intensity = close_open / (high_low_range + 1e-9)
    
    # 4. 引入成交量加权 (Volume Confirmation)
    # 如果是放量的大阳线/大阴线，权重更高；缩量则权重降低
    # 计算相对成交量：Vol / MA5(Vol)
    grouped_vol = data.groupby('stock_code')['volume']
    vol_ma5 = grouped_vol.transform(lambda x: x.rolling(5, min_periods=3).mean())
    vol_ratio = data['volume'] / (vol_ma5 + 1e-9)
    
    # 限制放量倍数在 [0.5, 3] 之间，防止极端数据干扰
    vol_weight = vol_ratio.clip(0.5, 3.0)
    
    # 最终因子：量能加持的日内强度
    factor = intensity * vol_weight
    
    # return factor.fillna(0.0)
    return factor

intraday_reversal = register_factor(
    FactorDefinition(
        name='intraday_reversal', # 保持原名以便替换，实际是强度因子
        required_columns={'stock_code', 'date', 'open', 'close', 'high', 'low', 'volume'},
        lookback_days=10,
        calculator=_intraday_intensity,
        description='[优化] 量能加权的日内K线实体强度',
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
    
    return strength.astype('float64')

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

