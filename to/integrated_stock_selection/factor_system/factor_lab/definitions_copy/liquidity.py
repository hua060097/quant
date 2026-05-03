"""
流动性与资金流向因子（深度优化版）。
核心关注：非流动性冲击(Amihud)、换手率稳定性(CV)、异常放量(Shock)与资金流压力(Flow)。
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
    衡量：单位成交额能带来多大的价格变动。
    - 数值越大：非流动性越高（一点点钱就能砸盘/拉涨停），通常是小票或锁仓股。
    - 数值越小：流动性越好（大资金进出不影响价格），通常是蓝筹。
    
    优化：
    1. 取对数：因为 Amihud 分布极度长尾，必须 Log 处理。
    2. 滚动平均：取 20 日均值，消除单日噪音。
    """
    if 'amount' not in data.columns or 'pct_chg' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    # 1. 计算单日 Amihud
    # 避免除以 0，加一个小 epsilon
    # amount 单位通常很大，乘以 1e6 或保持原样均可，这里取 Log 后无所谓
    daily_illio = data['pct_chg'].abs() / (data['amount'] + 1e-9)
    
    # 2. 滚动平均 (20日)
    # 构造临时 DF 提速
    temp = pd.DataFrame({'val': daily_illio.values, 'stock_code': data['stock_code'].values}, index=data.index)
    
    # 使用 rolling mean
    rolling_illio = temp.groupby('stock_code', sort=False)['val'].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    
    # 3. 对数标准化 (Log Transform)
    # Amihud 值差异巨大，Log 是必须的
    log_illio = np.log1p(rolling_illio * 1e9) # 乘 1e9 是为了让数值不那么小
    
    return log_illio.fillna(0.0).astype('float64')

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
    
    含义：
    - 高 CV：换手率忽高忽低。这是典型的**“脉冲式交易”**，通常是游资或散户行为（一日游）。
    - 低 CV：换手率非常稳定。这是典型的**“机构控盘”**或**“合力锁仓”**。
    
    大师点评：
    LightGBM 非常喜欢这个因子。在同样的涨幅下，CV 越低，趋势持续性越强。
    """
    if 'turnover_rate' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['turnover_rate']
    
    # 20日窗口
    rolling_mean = grouped.rolling(20, min_periods=10).mean()
    rolling_std = grouped.rolling(20, min_periods=10).std()
    
    # 计算 CV (Coefficient of Variation)
    cv = rolling_std / (rolling_mean + 1e-9)
    
    # 因子化：我们通常认为“稳定”是好事，所以取负数，或者倒数。
    # 这里取倒数：数值越大 = 越稳定 = 越好
    stability = 1.0 / (cv + 0.1) 
    
    return stability.reset_index(level=0, drop=True).fillna(0.0).astype('float64')

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
    - > 3.0: 极端放量（可能是出货，也可能是主升浪启动，取决于位置）。
    - < 0.5: 极致缩量（惜售）。
    
    这个因子捕捉的是**“关注度突变”**。
    """
    if 'turnover_rate' not in data.columns:
        return pd.Series(1.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['turnover_rate']
    
    # 长周期基准 (60日均量)
    base_turnover = grouped.rolling(60, min_periods=20).mean().reset_index(level=0, drop=True)
    
    # 异常度
    abnormal_ratio = data['turnover_rate'] / (base_turnover + 1e-9)
    
    # 限制极值，防止新股数据导致的无限大
    return abnormal_ratio.clip(0, 10).fillna(1.0).astype('float64')

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
    
    逻辑：
    利用日内收盘位置 (CLV) 加权换手率。
    Pressure = Turnover * ((Close - Low) - (High - Close)) / (High - Low)
             = Turnover * (2*Close - High - Low) / (High - Low)
             
    原理 (MFM - Money Flow Multiplier)：
    - 如果收盘在最高价：资金全量买入，Pressure = +Turnover
    - 如果收盘在最低价：资金全量卖出，Pressure = -Turnover
    - 如果收盘在中间：多空平衡，Pressure = 0
    
    优势：
    比单纯的 `Pct_Chg * Turnover` 更准。例如：
    某天高开低走收大阴线，Pct_Chg可能是正的（比昨天涨），但 MFM 是负的（主力在出货）。
    这个因子能识别这种**“假涨真跌”**。
    """
    cols = ['high', 'low', 'close', 'turnover_rate']
    if not all(c in data.columns for c in cols):
        return pd.Series(0.0, index=data.index)
    
    # 1. 计算 MFM (Money Flow Multiplier) [-1, 1]
    # (2C - H - L) / (H - L)
    high_low = data['high'] - data['low']
    mfm = ((2 * data['close'] - data['high'] - data['low']) / (high_low + 1e-9))
    
    # 处理一字板
    is_flat = high_low < 1e-4
    # 涨停一字 MFM=1，跌停一字 MFM=-1
    # 简单起见，如果一字且涨幅>0设为1，涨幅<0设为-1
    # 这里为了速度，简单 clip 即可，或者保持 0
    mfm = np.where(is_flat, 0.0, mfm) 
    
    # 2. 结合换手率
    pressure = mfm * data['turnover_rate']
    
    # 3. 滚动平滑 (5日)
    # 单日的资金流噪音大，看5日累计资金流
    temp = pd.DataFrame({'p': pressure.values, 'stock_code': data['stock_code'].values}, index=data.index)
    flow_5d = temp.groupby('stock_code', sort=False)['p'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    return flow_5d.fillna(0.0).astype('float64')

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

