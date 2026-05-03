"""
行为金融因子（Behavioral Finance）。
核心关注：心理锚点（52周高低点）与趋势加速。
已剔除高噪的成交量二阶导数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 1. 距离52周高点 (Anchoring Bias - High)
# ==============================================================================
def _distance_from_52w_high(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    距离52周高点
    
    逻辑：
    (Close / 52_Week_High) - 1.0
    
    大师点评：
    这是动量策略的核心锚点。
    值越接近0，说明上方无套牢盘，突破概率大。
    值越小（如 -0.5），说明深套，解套压力大。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    # 250日 ≈ 52周
    grouped = data.groupby('stock_code', sort=False)['close']
    high_52w = grouped.rolling(250, min_periods=60).max().reset_index(level=0, drop=True)
    
    distance = (data['close'] / (high_52w + 1e-9)) - 1.0
    
    return distance.fillna(-0.5).astype('float64')

distance_from_52w_high = register_factor(
    FactorDefinition(
        name='distance_from_52w_high',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=260,
        calculator=_distance_from_52w_high,
        description='距离52周高点的幅度（动量锚点）',
    )
)

# ==============================================================================
# 2. 距离52周低点 (Anchoring Bias - Low)
# ==============================================================================
def _distance_from_52w_low(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    距离52周低点
    
    逻辑：
    (Close / 52_Week_Low) - 1.0
    
    大师点评：
    这是反转和抄底策略的核心锚点。
    - 如果该值很小（接近0）：说明股价在底部徘徊，具有安全边际，或者正在阴跌。
    - 如果该值很大：说明已经脱离底部，趋势形成。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['close']
    low_52w = grouped.rolling(250, min_periods=60).min().reset_index(level=0, drop=True)
    
    distance = (data['close'] / (low_52w + 1e-9)) - 1.0
    
    return distance.fillna(0.5).astype('float64')

distance_from_52w_low = register_factor(
    FactorDefinition(
        name='distance_from_52w_low',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=260,
        calculator=_distance_from_52w_low,
        description='距离52周低点的幅度（安全边际锚点）',
    )
)

# ==============================================================================
# 3. 价格加速度 (Price Acceleration)
# ==============================================================================
def _price_acceleration(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    价格加速度 (二阶差分)
    
    逻辑：
    Velocity = Close_T - Close_T-5
    Acceleration = Velocity_T - Velocity_T-5
    
    标准化：
    Accel / Close
    
    意义：
    趋势不仅要涨，还要“加速涨”。
    """
    grouped = data.groupby('stock_code', sort=False)['close']
    
    # 速度 (5日变动)
    velocity = grouped.diff(5)
    
    # 加速度 (速度的5日变动)
    acceleration = velocity.diff(5)
    
    # 标准化
    acc_norm = acceleration / (data['close'] + 1e-9)
    
    return acc_norm.fillna(0.0).clip(-0.2, 0.2).astype('float64')

price_acceleration = register_factor(
    FactorDefinition(
        name='price_acceleration',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=20,
        calculator=_price_acceleration,
        description='价格变化的加速度（二阶导数，衡量趋势力度变化）',
    )
)


__all__ = [
    'distance_from_52w_high',
    'distance_from_52w_low',
    'price_acceleration',
]

