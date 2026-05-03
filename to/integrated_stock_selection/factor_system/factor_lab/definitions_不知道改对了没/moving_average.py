"""
移动均线形态与趋势因子（深度优化版）。
涵盖：乖离率(Bias)、均线纠缠(Convergence)、多头排列(Alignment)、趋势斜率(Slope)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

def _daily_robust_zscore(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """按日稳健 Z-Score (Cross-Sectional)"""
    df = pd.DataFrame({'val': series.values, 'date': date_series.values}, index=series.index)
    
    medians = df.groupby('date')['val'].transform('median')
    abs_diff = (df['val'] - medians).abs()
    
    # 按日计算 MAD
    temp_mad = pd.DataFrame({'diff': abs_diff.values, 'date': date_series.values}, index=series.index)
    mads = temp_mad.groupby('date')['diff'].transform('median')
    
    # 计算 Z-Score
    z_score = (df['val'] - medians) / (mads * 1.4826 + 1e-9)
    
    # 极值处理
    mask_zero_mad = (mads < 1e-6)
    if mask_zero_mad.any():
        means = df.groupby('date')['val'].transform('mean')
        stds = df.groupby('date')['val'].transform('std')
        z_fallback = (df['val'] - means) / (stds + 1e-9)
        z_score = z_score.where(~mask_zero_mad, z_fallback)
        
    return z_score.clip(-6.0, 6.0) # 这里的硬截断是为了防止计算过程中的极端异常

# ==============================================================================
# 1. 均线纠缠度 (MA Entanglement / Convergence) - 变盘预警
# ==============================================================================
def _ma_entanglement(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    均线纠缠度 (Convergence Score)
    
    逻辑：
    衡量短期、中期、长期均线是否“粘合”在一起。
    粘合意味着筹码成本一致，波动率压缩，即将选择方向（变盘）。
    
    计算：
    1. 计算 MA5, MA10, MA20, MA60
    2. 计算这4个数值在当天的 标准差 / 均值 (变异系数, CV)
    3. 取负数或倒数，使得 数值越大 = 越纠缠 = 越容易变盘
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['close']
    
    # 1. 计算各周期均线
    ma5 = grouped.rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    ma10 = grouped.rolling(10, min_periods=10).mean().reset_index(level=0, drop=True)
    ma20 = grouped.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    ma60 = grouped.rolling(60, min_periods=60).mean().reset_index(level=0, drop=True)
    
    # 2. 构造临时DF计算横截面统计量
    # 我们关心的是“同一天”这4根线的离散程度
    mas = pd.concat([ma5, ma10, ma20, ma60], axis=1)
    
    # 计算均值和标准差 (按行)
    ma_mean = mas.mean(axis=1)
    ma_std = mas.std(axis=1)
    
    # 3. 变异系数 (CV) = Std / Mean
    # CV 越小，说明均线越粘合
    cv = ma_std / (ma_mean + 1e-9)
    
    # 4. 因子化：取倒数，或者用 exp(-CV) 映射到 0~1
    # 越接近 1，说明极致粘合
    entanglement = np.exp(-10.0 * cv) 
    
    # 这里的系数 10.0 是经验值，为了让分布更均匀
    return entanglement.astype('float64')

ma_convergence_score = register_factor(
    FactorDefinition(
        name='ma_convergence_score',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_ma_entanglement,
        description='均线粘合度（数值越大，均线越密集，变盘概率越大）',
    )
)

# ==============================================================================
# 2. 均线多头排列得分 (MA Alignment Score) - 趋势质量
# ==============================================================================
def _ma_alignment_score(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    均线多头排列得分
    
    逻辑：
    不仅仅判断是否多头排列，而是给出一个“完美程度”的打分。
    
    打分项 (Total 1.0):
    - Close > MA5: +0.2
    - MA5 > MA10: +0.2
    - MA10 > MA20: +0.2
    - MA20 > MA60: +0.2
    - MA60 上行: +0.2
    
    空头排列则为负分。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['close']
    
    ma5 = grouped.rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    ma10 = grouped.rolling(10, min_periods=10).mean().reset_index(level=0, drop=True)
    ma20 = grouped.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    ma60 = grouped.rolling(60, min_periods=60).mean().reset_index(level=0, drop=True)
    
    # MA60 趋势 (当前 > 5天前)
    ma60_trend = ma60 > ma60.groupby(data['stock_code']).shift(5)
    
    score = np.zeros(len(data))
    
    # 向量化打分
    # 1. 价格强势
    score += np.where(data['close'] > ma5, 0.2, -0.2)
    # 2. 短期趋势
    score += np.where(ma5 > ma10, 0.2, -0.2)
    # 3. 中期趋势
    score += np.where(ma10 > ma20, 0.2, -0.2)
    # 4. 长期趋势
    score += np.where(ma20 > ma60, 0.2, -0.2)
    # 5. 长期均线方向
    score += np.where(ma60_trend, 0.2, -0.2)
    
    # 结果范围 [-1.0, 1.0]
    # 1.0 = 完美多头排列
    # -1.0 = 完美空头排列
    return pd.Series(score, index=data.index).astype('float64')

ma_alignment_score = register_factor(
    FactorDefinition(
        name='ma_alignment_score',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_ma_alignment_score,
        description='均线多头排列得分（趋势的结构化评分）',
    )
)

# ==============================================================================
# 3. 均线乖离率 (Bias) - 均值回归核心
# ==============================================================================
def _price_ma60_ratio(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    60日均线乖离率
    
    逻辑：
    (Close - MA60) / MA60
    这是衡量中期趋势强弱和超买超卖的核心指标。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    # 计算 MA60
    ma60 = data.groupby('stock_code', sort=False)['close'].rolling(60, min_periods=30).mean().reset_index(level=0, drop=True)
    
    # 计算乖离 (百分比)
    ratio = (data['close'] - ma60) / (ma60 + 1e-9)
    
    return ratio.astype('float64')

price_ma60_ratio = register_factor(
    FactorDefinition(
        name='price_ma60_ratio',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_price_ma60_ratio,
        description='价格相对60日均线的乖离率',
    )
)

def _price_ma60_ratio_z(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    60日均线乖离率
    
    逻辑：
    (Close - MA60) / MA60
    这是衡量中期趋势强弱和超买超卖的核心指标。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    # 计算 MA60
    ma60 = data.groupby('stock_code', sort=False)['close'].rolling(60, min_periods=30).mean().reset_index(level=0, drop=True)
    
    # 计算乖离 (百分比)
    ratio = (data['close'] - ma60) / (ma60 + 1e-9)
    return _daily_robust_zscore(ratio, data['date']).astype('float64')


price_ma60_ratio_z = register_factor(
    FactorDefinition(
        name='price_ma60_ratio_z',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_price_ma60_ratio_z,
        description='价格相对60日均线的乖离率',
    )
)

# ==============================================================================
# 4. 均线斜率 (Normalized Slope) - 趋势速度
# ==============================================================================
def _ma60_slope_norm(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    60日均线标准化斜率
    
    逻辑优化：
    原逻辑可能只是差分，受股价绝对值影响。
    新逻辑：(MA60_T - MA60_T-5) / MA60_T-5
    即：MA60 的 5日变化率。
    
    意义：
    衡量中期趋势的“加速度”或“陡峭程度”。
    陡峭向上的均线支撑力更强。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['close']
    ma60 = grouped.rolling(60, min_periods=30).mean().reset_index(level=0, drop=True)
    
    # 计算 MA60 的 5日变化率 (ROC)
    # 这里需要再次 groupby shift
    # 为了效率，构造临时Series
    ma60_s = pd.Series(ma60.values, index=data.index)
    ma60_prev = ma60_s.groupby(data['stock_code'], sort=False).shift(5)
    
    slope = (ma60 - ma60_prev) / (ma60_prev + 1e-9)
    
    # 放大一点数值方便观察，并去极值
    return (slope * 100).astype('float64')

ma60_slope = register_factor(
    FactorDefinition(
        name='ma60_slope',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_ma60_slope_norm,
        description='60日均线的标准化斜率（趋势陡峭度）',
    )
)

def _price_ma5_ratio(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    价格相对MA5的比率（短期乖离）
    
    大师点评：
    保留此因子，配合 range_compression 使用。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['close']
    ma5 = grouped.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    ratio = ((data['close'] - ma5) / (ma5 + 1e-9)) * 100
    return ratio.astype('float64')

price_ma5_ratio = register_factor(
    FactorDefinition(
        name='price_ma5_ratio',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_price_ma5_ratio,
        description='5日均线乖离率（短期情绪极限）',
    )
)

price_ma5_ratio_raw = register_factor(
    FactorDefinition(
        name='price_ma5_ratio_raw',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_price_ma5_ratio,
        description='5日均线乖离率原始值（仅用于权重/分析，不做归一化）',
    )
)

def _price_ma10_ratio(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    价格相对MA5的比率（短期乖离）
    
    大师点评：
    保留此因子，配合 range_compression 使用。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['close']
    ma10 = grouped.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    
    ratio = ((data['close'] - ma10) / (ma10 + 1e-9)) * 100
    return ratio.astype('float64')

price_ma10_ratio = register_factor(
    FactorDefinition(
        name='price_ma10_ratio',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_price_ma10_ratio,
        description='5日均线乖离率（短期情绪极限）',
    )
)

def _price_ma10_ratio_z(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    价格相对MA5的比率（短期乖离）
    
    大师点评：
    保留此因子，配合 range_compression 使用。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['close']
    ma10 = grouped.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    
    ratio = ((data['close'] - ma10) / (ma10 + 1e-9)) * 100
    return _daily_robust_zscore(ratio, data['date']).astype('float64')
    
price_ma10_ratio_z = register_factor(
    FactorDefinition(
        name='price_ma10_ratio_z',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_price_ma10_ratio_z,
        description='5日均线乖离率（短期情绪极限）',
    )
)

def _price_ma5_ratio_z(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    价格相对MA5的比率（短期乖离）
    
    大师点评：
    保留此因子，配合 range_compression 使用。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['close']
    ma5 = grouped.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    ratio = ((data['close'] - ma5) / (ma5 + 1e-9)) * 100
    return _daily_robust_zscore(ratio, data['date']).astype('float64')
    
price_ma5_ratio_z = register_factor(
    FactorDefinition(
        name='price_ma5_ratio_z',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_price_ma5_ratio_z,
        description='5日均线乖离率（短期情绪极限）',
    )
)



__all__ = [
    'ma_convergence_score',
    'ma60_slope',
    'price_ma60_ratio',
    'price_ma60_ratio_z',
    'ma_alignment_score',
    'price_ma5_ratio',
    'price_ma5_ratio_raw',
    'price_ma10_ratio',
    'price_ma10_ratio_z',
    'price_ma5_ratio_z',
]
