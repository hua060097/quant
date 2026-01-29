"""KDJ技术指标因子。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor



def _calc_kdj_core(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    n: int = 9, 
    m1: int = 3, 
    m2: int = 3
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    标准KDJ计算核心逻辑
    """
    # 1. 计算RSV
    # lowest_low = low.rolling(n, min_periods=n//2).min() # 使用较小的min_periods避免前期NaN
    # highest_high = high.rolling(n, min_periods=n//2).max()
    
    # 优化：Groupby内部调用，传入的应该是单只股票的Series
    # 如果传入的是全量Series，必须确保已经GroupBy过或者在外部Loop
    # 这里我们假设在calculator内部处理GroupBy
    
    lowest_low = low.rolling(window=n, min_periods=n).min()
    highest_high = high.rolling(window=n, min_periods=n).max()
    
    rsv = ((close - lowest_low) / (highest_high - lowest_low + 1e-9)) * 100
    # rsv = rsv.fillna(50.0)
    rsv = rsv
    
    # 2. 计算K, D, J
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j


def _kdj_features_calc(data: pd.DataFrame, _: FactorContext) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    计算基础KDJ因子 (N=9)
    返回: K, D, J, Diff(K-D)
    """
    if not {'high', 'low', 'close'}.issubset(data.columns):
        idx = data.index
        return (pd.Series(50, idx), pd.Series(50, idx), pd.Series(50, idx), pd.Series(0, idx))

    # 按股票分组计算
    def compute(g):
        k, d, j = _calc_kdj_core(g['high'], g['low'], g['close'], n=9, m1=3, m2=3)
        return pd.DataFrame({'k': k, 'd': d, 'j': j, 'diff': k - d}, index=g.index)

    # 性能优化：直接对Groupby对象apply可能会慢，但对于KDJ这种依赖Rolling的指标是必须的
    # 为了加速，可以考虑先sort_values再进行运算（如果数据量极大）
    # 这里保持简单逻辑
    results = data.groupby('stock_code', sort=False, group_keys=False).apply(compute)
    
    return results['k'], results['d'], results['j'], results['diff']


def _kdj_long_term_calc(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    计算长周期KDJ因子 (N=45, 模拟周线级别趋势)
    返回: Diff_Long (K_45 - D_45)
    """
    if not {'high', 'low', 'close'}.issubset(data.columns):
        return pd.Series(0, index=data.index)

    def compute(g):
        # 使用 N=45 近似 9周 (9*5) 的窗口，或者 N=27 (9*3)
        # 这里选用 N=36 (约8周) 平滑
        k, d, _ = _calc_kdj_core(g['high'], g['low'], g['close'], n=36, m1=3, m2=3)
        return k - d

    # return data.groupby('stock_code', sort=False, group_keys=False).apply(compute).fillna(0.0)
    return data.groupby('stock_code', sort=False, group_keys=False).apply(compute)


def _kdj_golden_cross_strength_opt(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    优化版 KDJ 金叉/趋势强度因子
    
    改进点：
    1. 移除周线重采样逻辑（避免未来函数和信号滞后）。
    2. 引入 N=36 的长周期 KDJ 替代周线，捕捉中线趋势。
    3. 输出连续值而非离散的分数，保留信息密度。
    4. 结合位置（低位金叉更优）和动量（斜率）。
    
    计算公式：
    Score = Norm(K9 - D9) + 0.5 * Norm(K36 - D36) + Low_Position_Bonus
    """
    # 1. 计算短周期 (9) 和 长周期 (36)
    # 为了效率，我们在一个groupby里算完
    def compute_all(g):
        # 短周期
        k9, d9, j9 = _calc_kdj_core(g['high'], g['low'], g['close'], n=9)
        diff9 = k9 - d9
        
        # 长周期
        k36, d36, _ = _calc_kdj_core(g['high'], g['low'], g['close'], n=36)
        diff36 = k36 - d36
        
        # 3. 计算得分
        # 基础分：短周期乖离 + 长周期乖离 (趋势共振)
        # diff9 > 0 表示金叉状态，diff36 > 0 表示多头趋势
        base_score = diff9 + 0.5 * diff36
        
        # 4. 低位加成 (Low Position Bonus)
        # 如果在低位 (K < 30) 出现金叉趋势 (diff9 > 0)，给予额外奖励
        # 使用高斯函数：当 K 在 20 附近时，权重最大
        low_pos_weight = np.exp(-((k9 - 20) ** 2) / (2 * 15 ** 2))
        # 只有在多头倾向时才给低位奖励
        position_bonus = np.where(diff9 > 0, low_pos_weight * 20.0, 0.0)
        
        # 5. J值加速 (J > K > D)
        # j_boost = np.where((j9 > k9) & (k9 > d9), 10.0, 0.0)
        
        total_score = base_score + position_bonus # + j_boost
        
        return total_score

    if not {'high', 'low', 'close'}.issubset(data.columns):
        return pd.Series(0.5, index=data.index)

    score = data.groupby('stock_code', sort=False, group_keys=False).apply(compute_all)
    
    # 简单的全局缩放，映射到 0~1 (假设 score 分布在 -100 到 +100 之间)
    # KDJ diff max is 100. Base score max ~ 150.
    # Sigmoid 压缩
    norm_score = 1 / (1 + np.exp(-score / 20.0))
    
    # return norm_score.fillna(0.5).astype('float64')
    return norm_score.astype('float64')


# ==========================
# 注册因子
# ==========================

# 1. 核心复合因子 (替代原来的 kdj_golden_cross_strength)
kdj_golden_cross_strength = register_factor(
    FactorDefinition(
        name='kdj_golden_cross_strength',
        required_columns={'stock_code', 'date', 'high', 'low', 'close'},
        lookback_days=60,
        calculator=_kdj_golden_cross_strength_opt,
        description='KDJ趋势共振强度（短周期+长周期+低位加成），无未来函数',
    )
)

# 2. 基础 KDJ 差值 (K - D)，最纯粹的信号
kdj_diff_9 = register_factor(
    FactorDefinition(
        name='kdj_diff_9',
        required_columns={'stock_code', 'date', 'high', 'low', 'close'},
        lookback_days=20,
        calculator=lambda d, c: _kdj_features_calc(d, c)[3], # 返回 diff
        description='9日KDJ指标 K-D 差值',
    )
)

# 3. K值 (反映绝对位置)
kdj_k_9 = register_factor(
    FactorDefinition(
        name='kdj_k_9',
        required_columns={'stock_code', 'date', 'high', 'low', 'close'},
        lookback_days=20,
        calculator=lambda d, c: _kdj_features_calc(d, c)[0], # 返回 k
        description='9日KDJ指标 K值',
    )
)

# 4. J值 (反映超买超卖，最灵敏)
kdj_j_9 = register_factor(
    FactorDefinition(
        name='kdj_j_9',
        required_columns={'stock_code', 'date', 'high', 'low', 'close'},
        lookback_days=20,
        calculator=lambda d, c: _kdj_features_calc(d, c)[2], # 返回 j
        description='9日KDJ指标 J值',
    )
)

__all__ = ['kdj_golden_cross_strength','kdj_diff_9','kdj_k_9','kdj_j_9']

