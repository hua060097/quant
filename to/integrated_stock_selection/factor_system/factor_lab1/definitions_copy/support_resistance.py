"""
交易行为支撑/压力因子 (Trading Support/Resistance Alpha)
- 核心思想：位置(Position) + 量能(Volume) = 强度(Strength)
- 改进点：数值有界化(Sigmoid)，处理突破/跌破逻辑，增加信号密度
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from numba import njit

EPS = 1e-9

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
    
    # 输出：支撑强度，压力强度
    support_score = np.zeros(n)
    resistance_score = np.zeros(n)
    
    # 预计算：为了效率，可以在循环外处理部分逻辑，但为了逻辑清晰，放在循环内
    # 在 T 时刻，只能看 T 及以前的数据
    
    for i in range(lookback, n):
        # 1. 获取过去的价格区间 (不包含今天，避免high-low lookahead? 
        # 不，对于支撑压力，通常看过去N天作为参考系。
        # 今天的Close是用来和参考系对比的。
        
        # 过去N天的极值 (不含今天)
        # 理由：如果含今天，close 永远在 [min, max] 之间，无法识别“突破”
        past_close = close[i - lookback : i]
        p_min = np.min(past_close)
        p_max = np.max(past_close)
        range_span = p_max - p_min + EPS
        
        curr_price = close[i]
        
        # 2. 计算相对位置 (Position)
        # < 0: 跌破下沿 (Breakdown)
        # > 1: 突破上沿 (Breakout)
        # 0~1: 区间震荡
        pos = (curr_price - p_min) / range_span
        
        # 3. 计算量能动态 (Volume Dynamics)
        # 近期量能 vs 长期量能
        vol_recent = np.mean(volume[i - vol_window : i + 1]) # 含今天
        vol_base = np.mean(volume[i - lookback : i]) + EPS
        vol_ratio = vol_recent / vol_base
        
        # 量能因子化 (Sigmoid压缩，中心在1.0)
        # vol_factor > 0.5 表示放量，< 0.5 表示缩量
        # 这里的 scaling factor 3.0 是为了让变化更敏感
        vol_factor = _sigmoid((vol_ratio - 1.0) * 2.0) 
        
        # ---------------------------------------
        # 4. 构建因子逻辑
        # ---------------------------------------
        
        # === 支撑强度 (Support Strength) ===
        # 逻辑：价格接近下沿 (pos小) + 缩量 (vol_factor小) = 强支撑 (惜售)
        # 如果跌破 (pos < 0)，支撑失效，甚至变成恐慌
        
        dist_to_low = pos # 越小越接近底部
        
        if dist_to_low < 0:
            # 已经跌破过去N天最低价 -> 支撑崩塌
            # 此时不仅没有支撑，反而是恐慌。给一个负分或者0分。
            support_score[i] = 0.0 
        elif dist_to_low > 1:
            # 突破新高了，离支撑很远，支撑意义不大
            support_score[i] = 0.0
        else:
            # 在区间内
            # 距离分：越接近底部(0)，得分越高。使用高斯衰减。
            # exp(-5 * x) -> x=0时1, x=0.5时0.08 (只关注底部附近的)
            proximity_score = np.exp(-4.0 * dist_to_low)
            
            # 量能分：越缩量(vol_factor小)，得分越高 (1 - vol_factor)
            # 缩量下跌是洗盘，放量下跌是出货
            volume_score = 1.0 - vol_factor
            
            support_score[i] = proximity_score * volume_score

        # === 压力强度 (Resistance Pressure) ===
        # 逻辑：价格接近上沿 (pos大) + 滞涨/放量滞涨 = 强压力
        # 如果突破 (pos > 1) -> 压力转化为动力 (Breakout)，压力归零
        
        dist_to_high = 1.0 - pos # 越小越接近顶部
        
        if pos > 1:
            # 有效突破 -> 压力消失
            resistance_score[i] = 0.0
        elif pos < 0:
            # 跌破下沿 -> 距离顶部太远，无压力讨论意义
            resistance_score[i] = 0.0
        else:
            # 在区间内
            # 距离分：越接近顶部，压力越大
            proximity_score = np.exp(-4.0 * dist_to_high)
            
            # 量能分：
            # 这里的逻辑比较微妙。
            # 高位放量 (vol_factor大) -> 分歧大，压力大 (有人在卖)
            # 高位缩量 (vol_factor小) -> 没人卖，可能轻松突破
            # 所以压力强度与量能成正比
            volume_score = vol_factor
            
            resistance_score[i] = proximity_score * volume_score
            
    return support_score, resistance_score

def _support_resistance_alpha_calc(data: pd.DataFrame, _: FactorContext) -> tuple[pd.Series, pd.Series]:
    """
    计算支撑和压力Alpha因子
    """
    if 'close' not in data.columns or 'volume' not in data.columns:
        return pd.Series(0.0, index=data.index), pd.Series(0.0, index=data.index)

    # Groupby Stock
    # 必须按股票分组，因为涉及到时序窗口
    def compute(g):
        c = g['close'].values.astype(np.float64)
        v = g['volume'].values.astype(np.float64)
        s, r = _calc_sr_dynamics_numba(c, v)
        return pd.DataFrame({'support': s, 'resistance': r}, index=g.index)

    # 这里的 groupby apply 可能会慢，但对于日线数据通常可接受
    # 如果追求极致性能，可以像之前那样把所有股票拼在一起处理（需处理边界），
    # 但最安全的方式还是 groupby
    res = data.groupby('stock_code', group_keys=False).apply(compute)
    
    return res['support'], res['resistance']

def _support_strength_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    s, _ = _support_resistance_alpha_calc(data, ctx)
    return s

def _resistance_pressure_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    _, r = _support_resistance_alpha_calc(data, ctx)
    return r

# ==========================
# 注册因子
# ==========================

support_distance = register_factor(
    FactorDefinition(
        name='support_distance', # 沿用旧名字，方便你不用改 main.py
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=40,
        calculator=_support_strength_alpha,
        description='支撑强度Alpha：缩量回踩底部的程度 (Position + Volume)',
    )
)

resistance_distance = register_factor(
    FactorDefinition(
        name='resistance_distance', # 沿用旧名字
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=40,
        calculator=_resistance_pressure_alpha,
        description='压力强度Alpha：放量接近顶部的程度 (Position + Volume)',
    )
)

__all__ = ['support_distance', 'resistance_distance']