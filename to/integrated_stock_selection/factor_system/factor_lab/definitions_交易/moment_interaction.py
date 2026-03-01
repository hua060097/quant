"""
动量交互与爆发因子 (Momentum Interaction & Breakout Factors) - 大师级策略专用
文件名: momentum_interaction.py

策略逻辑：
基于 "板块地位" + "短期爆发" + "资金确认" 的黄金三角交互。
旨在捕捉 Top 1 级别的龙头加速形态，同时引入非线性衍生特征以强化模型对"起爆点"和"力竭点"的识别。

包含因子：
1. leader_acceleration_score: 龙头加速因子 (板块动量 x 乖离率)
2. vol_price_explosion_score: 量价爆发因子 (异常换手 x 乖离率)
3. dragon_resonance_score: 真龙共振因子 (板块动量 x 乖离率 x 异常换手)
4. ma5_ratio_acceleration: 乖离率加速度 (乖离率一阶差分)
5. sector_rank_stability_3d: 龙头地位稳定性 (3日动量排名均值)
6. ma5_ratio_risk_penalty: 超买风险惩罚 (乖离率平方)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 0. 基础工具 (复用自 Sector/Liquidity/MA 模块，确保独立性)
# ==============================================================================
def _global_robust_zscore(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """
    [复用] 全局按日稳健 Z-Score (Robust Z-Score based on MAD)
    """
    df = pd.DataFrame({'val': series.values, 'date': date_series.values}, index=series.index)
    
    medians = df.groupby('date')['val'].transform('median')
    abs_diff = (df['val'] - medians).abs()
    
    temp_diff = pd.DataFrame({'diff': abs_diff.values, 'date': date_series.values}, index=series.index)
    mads = temp_diff.groupby('date')['diff'].transform('median')
    
    robust_std = mads * 1.4826
    z_score = (df['val'] - medians) / (robust_std + 1e-9)
    
    # 极值处理
    mask_zero_mad = (mads < 1e-6)
    if mask_zero_mad.any():
        means = df.groupby('date')['val'].transform('mean')
        stds = df.groupby('date')['val'].transform('std')
        z_fallback = (df['val'] - means) / (stds + 1e-9)
        z_score = z_score.where(~mask_zero_mad, z_fallback)
    
    return z_score.clip(-6.0, 6.0).astype('float64')

# ==============================================================================
# 1. 中间指标计算器 (计算核心三角组件)
# ==============================================================================
def _calc_components(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算黄金三角的三个基础组件：
    1. Rank: 板块/个股动量排名 (Z-Score)
    2. Ratio: MA5 乖离率
    3. Vol: 异常换手率
    """
    # --- A. 动量排名 (intra_sector_momentum_rank 复刻) ---
    # 逻辑：5日涨幅求和 -> 全局 Z-Score
    mom_5d = data.groupby('stock_code')['pct_chg'].rolling(5, min_periods=3).sum().reset_index(level=0, drop=True)
    rank_z = _global_robust_zscore(mom_5d, data['date'])
    
    # --- B. MA5 乖离率 (price_ma5_ratio 复刻) ---
    # 逻辑：(Close - MA5) / MA5 * 100
    ma5 = data.groupby('stock_code')['close'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    ma5_ratio = ((data['close'] - ma5) / (ma5 + 1e-9)) * 100
    
    # --- C. 异常换手 (abnormal_turnover 复刻) ---
    # 逻辑：Turnover / MA60_Turnover
    # 注意：这里需要处理 turnover_rate 缺失的情况
    if 'turnover_rate' in data.columns:
        to = data['turnover_rate']
        ma60_to = data.groupby('stock_code')['turnover_rate'].rolling(60, min_periods=20).mean().reset_index(level=0, drop=True)
        abnormal_vol = to / (ma60_to + 1e-9)
        # 极值处理 (同 liquidity.py)
        abnormal_vol = abnormal_vol.replace([np.inf, -np.inf], np.nan).clip(0, 30.0)
    else:
        abnormal_vol = pd.Series(1.0, index=data.index)
        
    return rank_z, ma5_ratio, abnormal_vol

# ==============================================================================
# 2. 交互因子实现
# ==============================================================================

def _leader_acceleration_score(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    [黄金交互 A] 龙头加速因子
    逻辑：板块动量排名 * MA5乖离率
    含义：强者(Rank高) + 加速(Ratio大) = 确定性溢价
    """
    if 'pct_chg' not in data.columns or 'close' not in data.columns:
        return pd.Series(np.nan, index=data.index)
        
    rank, ratio, _ = _calc_components(data)
    
    # 交互
    score = rank * ratio
    return score.astype('float64')

def _vol_price_explosion_score(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    [黄金交互 B] 量价爆发因子
    逻辑：异常换手 * MA5乖离率
    含义：资金进场(Vol大) + 价格突破(Ratio大) = 爆发力
    """
    if 'turnover_rate' not in data.columns:
        return pd.Series(np.nan, index=data.index)
        
    _, ratio, vol = _calc_components(data)
    
    score = vol * ratio
    return score.astype('float64')

def _dragon_resonance_score(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    [黄金交互 C] 真龙共振因子
    逻辑：板块动量 * MA5乖离率 * 异常换手
    含义：黄金三角共振，Top 1 核心特征
    """
    rank, ratio, vol = _calc_components(data)
    
    score = rank * ratio * vol
    return score.astype('float64')

# ==============================================================================
# 3. 短期异动衍生因子
# ==============================================================================

def _ma5_ratio_acceleration(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    [衍生 A] 乖离率加速度
    逻辑：MA5乖离率 - 昨日MA5乖离率
    含义：识别"加速初期" vs "减速滞涨"
    """
    _, ratio, _ = _calc_components(data)
    
    # 计算一阶差分
    # 注意：需要按股票分组 shift
    prev_ratio = ratio.groupby(data['stock_code']).shift(1)
    acceleration = ratio - prev_ratio
    
    return acceleration.astype('float64')

def _sector_rank_stability_3d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    [衍生 B] 龙头地位稳定性
    逻辑：动量排名的3日均值
    含义：过滤掉"一日游"的伪龙头，寻找持续强势股
    """
    rank, _, _ = _calc_components(data)
    
    stability = rank.groupby(data['stock_code']).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    return stability.astype('float64')

def _ma5_ratio_risk_penalty(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    [衍生 C] 超买风险惩罚
    逻辑：MA5乖离率的平方 (保留符号或仅关注幅度)
    这里实现为：Signed Squared (保持方向，但放大极端值) -> sign(x) * x^2
    或者直接 x^2 作为"偏离度能量"。
    
    根据策略需求：我们希望模型警惕过大的乖离。
    直接提供 x^2，树模型可以切出 > Threshold 的部分作为风险区。
    """
    _, ratio, _ = _calc_components(data)
    
    # 简单的平方，表示"偏离能量"
    penalty = ratio ** 2
    return penalty.astype('float64')

# ==============================================================================
# 4. 注册因子
# ==============================================================================

leader_acceleration_score = register_factor(FactorDefinition(
    name='leader_acceleration_score',
    required_columns={'stock_code', 'date', 'pct_chg', 'close'},
    lookback_days=10,
    calculator=_leader_acceleration_score,
    description='龙头加速因子(板块排名xMA5乖离)'
))

vol_price_explosion_score = register_factor(FactorDefinition(
    name='vol_price_explosion_score',
    required_columns={'stock_code', 'date', 'turnover_rate', 'close'},
    lookback_days=65, # 需要60日均量
    calculator=_vol_price_explosion_score,
    description='量价爆发因子(异常换手xMA5乖离)'
))

dragon_resonance_score = register_factor(FactorDefinition(
    name='dragon_resonance_score',
    required_columns={'stock_code', 'date', 'pct_chg', 'close', 'turnover_rate'},
    lookback_days=65,
    calculator=_dragon_resonance_score,
    description='真龙共振因子(排名x乖离x换手)'
))

ma5_ratio_acceleration = register_factor(FactorDefinition(
    name='ma5_ratio_acceleration',
    required_columns={'stock_code', 'date', 'close'},
    lookback_days=10,
    calculator=_ma5_ratio_acceleration,
    description='乖离率加速度(MA5乖离率变动)'
))

sector_rank_stability_3d = register_factor(FactorDefinition(
    name='sector_rank_stability_3d',
    required_columns={'stock_code', 'date', 'pct_chg'},
    lookback_days=10,
    calculator=_sector_rank_stability_3d,
    description='龙头地位稳定性(3日排名均值)'
))

ma5_ratio_risk_penalty = register_factor(FactorDefinition(
    name='ma5_ratio_risk_penalty',
    required_columns={'stock_code', 'date', 'close'},
    lookback_days=10,
    calculator=_ma5_ratio_risk_penalty,
    description='超买风险惩罚(MA5乖离率平方)'
))

__all__ = [
    'leader_acceleration_score',
    'vol_price_explosion_score',
    'dragon_resonance_score',
    'ma5_ratio_acceleration',
    'sector_rank_stability_3d',
    'ma5_ratio_risk_penalty'
]