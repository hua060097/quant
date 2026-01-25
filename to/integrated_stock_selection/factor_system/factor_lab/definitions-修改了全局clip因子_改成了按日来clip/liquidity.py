"""
流动性与资金流向因子（深度优化版）。
核心关注：非流动性冲击(Amihud)、换手率稳定性(CV)、异常放量(Shock)与资金流压力(Flow)。

修改日志：
1. 移除所有 fillna 操作，保留 NaN 状态以防止数据污染。
2. 优化 _abnormal_turnover 的 clip 范围，放宽至 30 以捕捉极端爆发。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 1. 增强版 Amihud 非流动性因子 (Illiquidity)
# ==============================================================================
def _amihud_illiquidity(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    Amihud 非流动性指标 (Price Impact)
    
    逻辑：
    |Ret| / Amount
    """
    if 'amount' not in data.columns or 'pct_chg' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    # 1. 计算单日 Amihud
    # 避免除以 0，加一个小 epsilon
    daily_illio = data['pct_chg'].abs() / (data['amount'] + 1e-9)
    
    # 2. 滚动平均 (20日)
    temp = pd.DataFrame({'val': daily_illio.values, 'stock_code': data['stock_code'].values}, index=data.index)
    
    rolling_illio = temp.groupby('stock_code', sort=False)['val'].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    
    # 3. 对数标准化 (Log Transform)
    log_illio = np.log1p(rolling_illio * 1e9) 
    
    # [修改] 删除 fillna(0.0)，保留 NaN
    return log_illio.astype('float64')

amihud_illiquidity = register_factor(
    FactorDefinition(
        name='amihud_illiquidity',
        required_columns={'stock_code', 'date', 'amount', 'pct_chg'},
        lookback_days=30,
        calculator=_amihud_illiquidity,
        description='Amihud非流动性因子（单位成交额的价格冲击，对数化）',
    )
)

# ==============================================================================
# 2. 换手率稳定性 (Turnover Stability) - 机构控盘度
# ==============================================================================
def _turnover_stability(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    换手率变异系数 (CV of Turnover)
    
    逻辑：
    Std(Turnover) / Mean(Turnover)
    """
    if 'turnover_rate' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['turnover_rate']
    
    # 20日窗口
    rolling_mean = grouped.rolling(20, min_periods=10).mean()
    rolling_std = grouped.rolling(20, min_periods=10).std()
    
    # 计算 CV (Coefficient of Variation)
    cv = rolling_std / (rolling_mean + 1e-9)
    
    # 因子化：数值越大 = 越稳定
    stability = 1.0 / (cv + 0.1) 
    
    # [修改] 删除 fillna(0.0)，保留 NaN
    return stability.reset_index(level=0, drop=True).astype('float64')

turnover_stability = register_factor(
    FactorDefinition(
        name='turnover_stability',
        required_columns={'stock_code', 'date', 'turnover_rate'},
        lookback_days=30,
        calculator=_turnover_stability,
        description='换手率稳定性（变异系数倒数，衡量筹码结构稳定性）',
    )
)

# ==============================================================================
# 3. 异常放量 (Abnormal Turnover) - 事件驱动/变盘信号
# ==============================================================================
def _abnormal_turnover(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    异常换手率 (Turnover Shock)
    
    逻辑：
    Today_Turnover / Mean_Turnover_60d
    
    含义：
    - > 3.0: 极端放量
    - < 0.5: 极致缩量
    """
    if 'turnover_rate' not in data.columns:
        return pd.Series(1.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['turnover_rate']
    
    # 长周期基准 (60日均量)
    base_turnover = grouped.rolling(60, min_periods=20).mean().reset_index(level=0, drop=True)
    
    # 异常度
    abnormal_ratio = data['turnover_rate'] / (base_turnover + 1e-9)
    
    # [修改] 
    # 1. 处理 Inf (除以0的情况)，将其设为 NaN (让模型处理或后续填充)
    abnormal_ratio = abnormal_ratio.replace([np.inf, -np.inf], np.nan)
    
    # 2. 优化 Clip：放宽上限到 30.0，为了捕捉 Top-1 级别的极端爆发 (如 0.5% -> 15% 换手)
    #    同时保留 0 下限
    # 3. 删除 fillna(1.0)，保留 NaN
    return abnormal_ratio.clip(0, 30.0).astype('float64')

abnormal_turnover = register_factor(
    FactorDefinition(
        name='abnormal_turnover',
        required_columns={'stock_code', 'date', 'turnover_rate'},
        lookback_days=80,
        calculator=_abnormal_turnover,
        description='异常换手率（当日换手/60日均换手，捕捉热度突变）',
    )
)

# ==============================================================================
# 4. 资金流压力 (Money Flow Pressure) - 智能资金方向
# ==============================================================================
def _money_flow_pressure(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    资金流压力 (Smart Money Flow Proxy)
    """
    cols = ['high', 'low', 'close', 'turnover_rate']
    if not all(c in data.columns for c in cols):
        return pd.Series(0.0, index=data.index)
    
    # 1. 计算 MFM (Money Flow Multiplier) [-1, 1]
    high_low = data['high'] - data['low']
    mfm = ((2 * data['close'] - data['high'] - data['low']) / (high_low + 1e-9))
    
    # 处理一字板
    is_flat = high_low < 1e-4
    mfm = np.where(is_flat, 0.0, mfm) 
    
    # 2. 结合换手率
    pressure = mfm * data['turnover_rate']
    
    # 3. 滚动平滑 (5日)
    temp = pd.DataFrame({'p': pressure.values, 'stock_code': data['stock_code'].values}, index=data.index)
    flow_5d = temp.groupby('stock_code', sort=False)['p'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # [修改] 删除 fillna(0.0)，保留 NaN
    return flow_5d.astype('float64')

money_flow_pressure = register_factor(
    FactorDefinition(
        name='money_flow_pressure',
        required_columns={'stock_code', 'date', 'high', 'low', 'close', 'turnover_rate'},
        lookback_days=10,
        calculator=_money_flow_pressure,
        description='资金流压力（基于收盘位置加权的换手率，识别出货/吸筹）',
    )
)


__all__ = [
    'turnover_stability',
    'abnormal_turnover',
    'money_flow_pressure',
    'amihud_illiquidity',
]