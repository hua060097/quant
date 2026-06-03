"""
移动均线形态与趋势因子（深度优化版）。
涵盖：乖离率(Bias)、均线纠缠(Convergence)、多头排列(Alignment)、趋势斜率(Slope)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import groupby_rolling_target, groupby_shift_target

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


def _target_frame(data: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    return data.loc[target_index].copy().reset_index(drop=True)


def _ma_target(
    data: pd.DataFrame,
    target_index: pd.Index,
    window: int,
    min_periods: int,
) -> pd.Series:
    return groupby_rolling_target(
        data,
        group_col='stock_code',
        value_col='close',
        target_index=target_index,
        window=window,
        min_periods=min_periods,
        agg='mean',
    ).reset_index(drop=True).astype('float64')


def _build_centered_time_axis(window: int) -> tuple[np.ndarray, float]:
    x = np.arange(window, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered)) + 1e-12
    return x_centered, denominator


def _rolling_log_price_slope(
    data: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index, dtype='float64')

    x_centered, denominator = _build_centered_time_axis(window)

    def _ols_slope(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) != window or not np.isfinite(arr).all():
            return np.nan
        y = np.log(np.clip(arr, 1e-6, None))
        return float(np.dot(x_centered, y - y.mean()) / denominator)

    slope = (
        data.groupby('stock_code', sort=False)['close']
        .rolling(window, min_periods=min_periods)
        .apply(_ols_slope, raw=True)
        .reset_index(level=0, drop=True)
    )
    return (slope * 100.0).astype('float64')


def _rolling_log_price_slope_target(
    data: pd.DataFrame,
    target_index: pd.Index,
    window: int,
    min_periods: int,
) -> pd.Series:
    full_series = _rolling_log_price_slope(data, window=window, min_periods=min_periods)
    return full_series.loc[target_index].reset_index(drop=True).astype('float64')

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


def _ma_entanglement_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)))
    ma5 = _ma_target(data, target_index, 5, 5)
    ma10 = _ma_target(data, target_index, 10, 10)
    ma20 = _ma_target(data, target_index, 20, 20)
    ma60 = _ma_target(data, target_index, 60, 60)
    mas = pd.concat([ma5, ma10, ma20, ma60], axis=1)
    ma_mean = mas.mean(axis=1)
    ma_std = mas.std(axis=1)
    cv = ma_std / (ma_mean + 1e-9)
    entanglement = np.exp(-10.0 * cv)
    return entanglement.astype('float64')

ma_convergence_score = register_factor(
    FactorDefinition(
        name='ma_convergence_score',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_ma_entanglement,
        calculation_scope='target_with_history_window',
        target_calculator=_ma_entanglement_target,
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


def _ma_alignment_score_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)))
    target_data = _target_frame(data, target_index)
    ma5 = _ma_target(data, target_index, 5, 5)
    ma10 = _ma_target(data, target_index, 10, 10)
    ma20 = _ma_target(data, target_index, 20, 20)
    ma60 = _ma_target(data, target_index, 60, 60)
    full_ma60 = data.groupby('stock_code', sort=False)['close'].rolling(60, min_periods=60).mean().reset_index(level=0, drop=True)
    full_ma60_s = pd.Series(full_ma60.values, index=data.index)
    ma60_prev = full_ma60_s.groupby(data['stock_code'], sort=False).shift(5).loc[target_index].reset_index(drop=True)
    ma60_trend = ma60 > ma60_prev
    score = np.zeros(len(target_data))
    score += np.where(target_data['close'].astype('float64') > ma5, 0.2, -0.2)
    score += np.where(ma5 > ma10, 0.2, -0.2)
    score += np.where(ma10 > ma20, 0.2, -0.2)
    score += np.where(ma20 > ma60, 0.2, -0.2)
    score += np.where(ma60_trend, 0.2, -0.2)
    return pd.Series(score, index=range(len(target_index))).astype('float64')

ma_alignment_score = register_factor(
    FactorDefinition(
        name='ma_alignment_score',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_ma_alignment_score,
        calculation_scope='target_with_history_window',
        target_calculator=_ma_alignment_score_target,
        description='均线多头排列得分（趋势的结构化评分）',
    )
)

# ==============================================================================
# 3. 均线排列持续性与均线带扩张 - 更专业的结构确认
# ==============================================================================
def _ma_alignment_persistence_5d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    均线多头排列持续性（5日）

    逻辑：
    - 先判断当天是否满足 MA5 > MA10 > MA20 > MA60
    - 再统计过去 5 个交易日里，这种完整多头排列维持了多久

    含义：
    - 越接近 1，说明趋势结构不是“今天刚对”，而是已经持续稳定
    - 比单点的 ma_alignment_score 更偏“趋势延续质量”
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)

    grouped = data.groupby('stock_code', sort=False)['close']
    ma5 = grouped.rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    ma10 = grouped.rolling(10, min_periods=10).mean().reset_index(level=0, drop=True)
    ma20 = grouped.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    ma60 = grouped.rolling(60, min_periods=60).mean().reset_index(level=0, drop=True)
    aligned = ((ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)).astype('float64')
    persistence = aligned.groupby(data['stock_code'], sort=False).rolling(5, min_periods=2).mean().reset_index(level=0, drop=True)
    return persistence.astype('float64')


def _ma_alignment_persistence_5d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)))
    full = _ma_alignment_persistence_5d(data, _)
    return full.loc[target_index].reset_index(drop=True).astype('float64')


ma_alignment_persistence_5d = register_factor(
    FactorDefinition(
        name='ma_alignment_persistence_5d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_ma_alignment_persistence_5d,
        calculation_scope='target_with_history_window',
        target_calculator=_ma_alignment_persistence_5d_target,
        description='过去5日完整多头均线排列的持续度，越高说明趋势结构越稳定',
    )
)


def _ma_ribbon_expansion_3d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    短周期均线带 3 日扩张速度

    逻辑：
    - 计算 MA5/10/20 的带宽：(max - min) / close
    - 再减去 3 天前的带宽

    含义：
    - > 0：均线带在展开，趋势推进更有“展开感”
    - < 0：均线带在收缩，趋势可能失速或重新纠缠
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)

    grouped = data.groupby('stock_code', sort=False)['close']
    ma5 = grouped.rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    ma10 = grouped.rolling(10, min_periods=10).mean().reset_index(level=0, drop=True)
    ma20 = grouped.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    ribbon = pd.concat([ma5, ma10, ma20], axis=1)
    width = (ribbon.max(axis=1) - ribbon.min(axis=1)) / (data['close'].astype('float64') + 1e-9)
    expansion = width.groupby(data['stock_code'], sort=False).diff(3)
    return expansion.astype('float64')


def _ma_ribbon_expansion_3d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)))
    full = _ma_ribbon_expansion_3d(data, _)
    return full.loc[target_index].reset_index(drop=True).astype('float64')


ma_ribbon_expansion_3d = register_factor(
    FactorDefinition(
        name='ma_ribbon_expansion_3d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_ma_ribbon_expansion_3d,
        calculation_scope='target_with_history_window',
        target_calculator=_ma_ribbon_expansion_3d_target,
        description='MA5/10/20 均线带近3日扩张速度，越低说明趋势带收缩失速',
    )
)

# ==============================================================================
# 4. 均线乖离率 (Bias) - 均值回归核心
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


def _price_ma60_ratio_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)))
    ma60 = _ma_target(data, target_index, 60, 30)
    target_close = data.loc[target_index, 'close'].reset_index(drop=True).astype('float64')
    ratio = (target_close - ma60) / (ma60 + 1e-9)
    return ratio.astype('float64')

price_ma60_ratio = register_factor(
    FactorDefinition(
        name='price_ma60_ratio',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_price_ma60_ratio,
        calculation_scope='target_with_history_window',
        target_calculator=_price_ma60_ratio_target,
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


def _price_ma60_ratio_z_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    target_data = _target_frame(data, target_index)
    ratio = _price_ma60_ratio_target(data, target_index, _).reset_index(drop=True)
    return _daily_robust_zscore(ratio, target_data['date']).astype('float64')


price_ma60_ratio_z = register_factor(
    FactorDefinition(
        name='price_ma60_ratio_z',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_price_ma60_ratio_z,
        calculation_scope='target_with_history_window',
        target_calculator=_price_ma60_ratio_z_target,
        description='价格相对60日均线的乖离率',
    )
)

# ==============================================================================
# 4. 对数价格斜率 (Log-Price Slope) - 趋势推进速度
# ==============================================================================
def _log_price_slope_5d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    5日对数价格回归斜率。

    金融含义：
    - 刻画极短周期价格推进速度
    - 使用对数价格，减少价格绝对水平对斜率的污染
    """
    return _rolling_log_price_slope(data, window=5, min_periods=5)


def _log_price_slope_5d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _rolling_log_price_slope_target(data, target_index, window=5, min_periods=5)


log_price_slope_5d = register_factor(
    FactorDefinition(
        name='log_price_slope_5d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_log_price_slope_5d,
        calculation_scope='target_with_history_window',
        target_calculator=_log_price_slope_5d_target,
        description='5日对数价格回归斜率，衡量极短周期价格推进速度',
    )
)


def _log_price_slope_10d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    10日对数价格回归斜率。

    金融含义：
    - 刻画短周期主升推进速度
    - 比单日收益更稳，比长周期均线斜率更贴近交易节奏
    """
    return _rolling_log_price_slope(data, window=10, min_periods=8)


def _log_price_slope_10d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _rolling_log_price_slope_target(data, target_index, window=10, min_periods=8)


log_price_slope_10d = register_factor(
    FactorDefinition(
        name='log_price_slope_10d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=20,
        calculator=_log_price_slope_10d,
        calculation_scope='target_with_history_window',
        target_calculator=_log_price_slope_10d_target,
        description='10日对数价格回归斜率，衡量短周期趋势推进速度',
    )
)


def _log_price_slope_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    20日对数价格回归斜率。

    金融含义：
    - 刻画中短周期趋势推进速度
    - 用于区分持续趋势推进与单纯短期脉冲
    """
    return _rolling_log_price_slope(data, window=20, min_periods=15)


def _log_price_slope_20d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _rolling_log_price_slope_target(data, target_index, window=20, min_periods=15)


log_price_slope_20d = register_factor(
    FactorDefinition(
        name='log_price_slope_20d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=30,
        calculator=_log_price_slope_20d,
        calculation_scope='target_with_history_window',
        target_calculator=_log_price_slope_20d_target,
        description='20日对数价格回归斜率，衡量中短周期趋势推进速度',
    )
)


def _log_price_slope_10d_x_log_circ_mv(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    10日斜率与流通市值对数的交叉因子。

    金融含义：
    - 相同斜率在不同市值股票上的含义并不一致
    - 该交叉项让模型自行学习“斜率 × 市值结构”的差异
    """
    if 'circ_mv' not in data.columns:
        return pd.Series(0.0, index=data.index, dtype='float64')

    slope = _rolling_log_price_slope(data, window=10, min_periods=8)
    log_circ_mv = np.log(np.clip(data['circ_mv'].astype(np.float64), 1.0, None))
    return (slope * log_circ_mv).astype('float64')


def _log_price_slope_10d_x_log_circ_mv_target(
    data: pd.DataFrame,
    target_index: pd.Index,
    _: FactorContext,
) -> pd.Series:
    if 'circ_mv' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)), dtype='float64')

    slope = _rolling_log_price_slope_target(data, target_index, window=10, min_periods=8)
    log_circ_mv = np.log(np.clip(data.loc[target_index, 'circ_mv'].astype(np.float64), 1.0, None)).reset_index(drop=True)
    return (slope * log_circ_mv).astype('float64')


log_price_slope_10d_x_log_circ_mv = register_factor(
    FactorDefinition(
        name='log_price_slope_10d_x_log_circ_mv',
        required_columns={'stock_code', 'date', 'close', 'circ_mv'},
        lookback_days=20,
        calculator=_log_price_slope_10d_x_log_circ_mv,
        calculation_scope='target_with_history_window',
        target_calculator=_log_price_slope_10d_x_log_circ_mv_target,
        description='10日对数价格斜率与流通市值对数的交叉项',
    )
)


# ==============================================================================
# 5. 均线斜率 (Normalized Slope) - 趋势速度
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


def _ma60_slope_norm_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)))
    full_ma60 = data.groupby('stock_code', sort=False)['close'].rolling(60, min_periods=30).mean().reset_index(level=0, drop=True)
    ma60_target = full_ma60.loc[target_index].reset_index(drop=True)
    ma60_prev = pd.Series(full_ma60.values, index=data.index).groupby(data['stock_code'], sort=False).shift(5).loc[target_index].reset_index(drop=True)
    slope = (ma60_target - ma60_prev) / (ma60_prev + 1e-9)
    return (slope * 100).astype('float64')

ma60_slope = register_factor(
    FactorDefinition(
        name='ma60_slope',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=80,
        calculator=_ma60_slope_norm,
        calculation_scope='target_with_history_window',
        target_calculator=_ma60_slope_norm_target,
        description='60日均线的标准化斜率（趋势陡峭度）',
    )
)

def _ma5_slope_norm(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    5日均线标准化斜率

    逻辑：
    - 先计算 MA5
    - 再计算 MA5 相对 1 日前的变化率

    含义：
    - > 0：短期均线仍在抬升
    - < 0：短期均线开始走平或转弱
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)

    grouped = data.groupby('stock_code', sort=False)['close']
    ma5 = grouped.rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    ma5_s = pd.Series(ma5.values, index=data.index)
    ma5_prev = ma5_s.groupby(data['stock_code'], sort=False).shift(1)
    slope = (ma5 - ma5_prev) / (ma5_prev + 1e-9)
    return (slope * 100).astype('float64')


def _ma5_slope_norm_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)))
    full_ma5 = data.groupby('stock_code', sort=False)['close'].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    ma5_target = full_ma5.loc[target_index].reset_index(drop=True)
    ma5_prev = pd.Series(full_ma5.values, index=data.index).groupby(data['stock_code'], sort=False).shift(1).loc[target_index].reset_index(drop=True)
    slope = (ma5_target - ma5_prev) / (ma5_prev + 1e-9)
    return (slope * 100).astype('float64')


ma5_slope = register_factor(
    FactorDefinition(
        name='ma5_slope',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_ma5_slope_norm,
        calculation_scope='target_with_history_window',
        target_calculator=_ma5_slope_norm_target,
        description='5日均线的标准化斜率（短期趋势方向与速度）',
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


def _price_ma5_ratio_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)))
    ma5 = _ma_target(data, target_index, 5, 1)
    target_close = data.loc[target_index, 'close'].reset_index(drop=True).astype('float64')
    ratio = ((target_close - ma5) / (ma5 + 1e-9)) * 100
    return ratio.astype('float64')

price_ma5_ratio = register_factor(
    FactorDefinition(
        name='price_ma5_ratio',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_price_ma5_ratio,
        calculation_scope='target_with_history_window',
        target_calculator=_price_ma5_ratio_target,
        description='5日均线乖离率（短期情绪极限）',
    )
)

price_ma5_ratio_raw = register_factor(
    FactorDefinition(
        name='price_ma5_ratio_raw',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_price_ma5_ratio,
        calculation_scope='target_with_history_window',
        target_calculator=_price_ma5_ratio_target,
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


def _price_ma10_ratio_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)))
    ma10 = _ma_target(data, target_index, 10, 1)
    target_close = data.loc[target_index, 'close'].reset_index(drop=True).astype('float64')
    ratio = ((target_close - ma10) / (ma10 + 1e-9)) * 100
    return ratio.astype('float64')

price_ma10_ratio = register_factor(
    FactorDefinition(
        name='price_ma10_ratio',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_price_ma10_ratio,
        calculation_scope='target_with_history_window',
        target_calculator=_price_ma10_ratio_target,
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


def _price_ma10_ratio_z_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    target_data = _target_frame(data, target_index)
    ratio = _price_ma10_ratio_target(data, target_index, _).reset_index(drop=True)
    return _daily_robust_zscore(ratio, target_data['date']).astype('float64')
    
price_ma10_ratio_z = register_factor(
    FactorDefinition(
        name='price_ma10_ratio_z',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_price_ma10_ratio_z,
        calculation_scope='target_with_history_window',
        target_calculator=_price_ma10_ratio_z_target,
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


def _price_ma5_ratio_z_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    target_data = _target_frame(data, target_index)
    ratio = _price_ma5_ratio_target(data, target_index, _).reset_index(drop=True)
    return _daily_robust_zscore(ratio, target_data['date']).astype('float64')
    
price_ma5_ratio_z = register_factor(
    FactorDefinition(
        name='price_ma5_ratio_z',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_price_ma5_ratio_z,
        calculation_scope='target_with_history_window',
        target_calculator=_price_ma5_ratio_z_target,
        description='5日均线乖离率（短期情绪极限）',
    )
)



__all__ = [
    'ma_convergence_score',
    'ma_alignment_persistence_5d',
    'ma_ribbon_expansion_3d',
    'log_price_slope_5d',
    'log_price_slope_10d',
    'log_price_slope_20d',
    'log_price_slope_10d_x_log_circ_mv',
    'ma5_slope',
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
