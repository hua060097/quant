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
from ..target_utils import (
    groupby_diff_target,
    groupby_rolling_target,
)
from ..universe import non_st_mask

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

    return distance.astype('float64')


def _distance_from_52w_high_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)), dtype='float64')
    high_52w = groupby_rolling_target(
        data,
        group_col='stock_code',
        value_col='close',
        target_index=target_index,
        window=250,
        min_periods=60,
        agg='max',
    )
    target_close = data.loc[target_index, 'close'].reset_index(drop=True).astype('float64')
    distance = (target_close / (high_52w.reset_index(drop=True) + 1e-9)) - 1.0
    return distance.astype('float64')

distance_from_52w_high = register_factor(
    FactorDefinition(
        name='distance_from_52w_high',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=260,
        calculator=_distance_from_52w_high,
        calculation_scope='target_with_history_window',
        target_calculator=_distance_from_52w_high_target,
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

    return distance.astype('float64')


def _distance_from_52w_low_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)), dtype='float64')
    low_52w = groupby_rolling_target(
        data,
        group_col='stock_code',
        value_col='close',
        target_index=target_index,
        window=250,
        min_periods=60,
        agg='min',
    )
    target_close = data.loc[target_index, 'close'].reset_index(drop=True).astype('float64')
    distance = (target_close / (low_52w.reset_index(drop=True) + 1e-9)) - 1.0
    return distance.astype('float64')

distance_from_52w_low = register_factor(
    FactorDefinition(
        name='distance_from_52w_low',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=260,
        calculator=_distance_from_52w_low,
        calculation_scope='target_with_history_window',
        target_calculator=_distance_from_52w_low_target,
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
    趋势不仅要涨，还要"加速涨"。
    """
    grouped = data.groupby('stock_code', sort=False)['close']

    # 速度 (5日变动)
    velocity = grouped.diff(5)

    # 加速度 (速度的5日变动)
    acceleration = velocity.diff(5)

    # 标准化
    acc_norm = acceleration / (data['close'] + 1e-9)

    return acc_norm.astype('float64')


def _price_acceleration_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(0.0, index=range(len(target_index)), dtype='float64')
    # diff(5) 需要中间序列的全历史，直接在全量上算后取目标日期
    full_velocity = data.groupby('stock_code', sort=False)['close'].diff(5)
    full_acceleration = full_velocity.groupby(data['stock_code'], sort=False).diff(5)
    target_acc = full_acceleration.loc[target_index].reset_index(drop=True)
    target_close = data.loc[target_index, 'close'].reset_index(drop=True).astype('float64')
    acc_norm = target_acc / (target_close + 1e-9)
    return acc_norm.astype('float64')

price_acceleration = register_factor(
    FactorDefinition(
        name='price_acceleration',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=20,
        calculator=_price_acceleration,
        calculation_scope='target_with_history_window',
        target_calculator=_price_acceleration_target,
        description='价格变化的加速度（二阶导数，衡量趋势力度变化）',
    )
)


# ==============================================================================
# 4a. 芯片积累后供给紧张度 (Post-Accumulation Supply Tightness)
#     = 趋势 x 缩量 x 波缩 的等权几何均值
#     无手工权重、无涨停惩罚、无高位保持（避免与trend共线）
# ==============================================================================
def _post_accumulation_supply_tightness(
    data: pd.DataFrame,
    ctx: FactorContext | None = None,
) -> pd.Series:
    """
    芯片积累后供给紧张度。

    3维度等权几何均值，表达"趋势 + 量缩 + 波缩是否同时成立"：
    1. 趋势强度：10日涨幅截面分位
    2. 缩量强度：近3日均量 vs 前7日均量（无重叠对比窗口）
    3. 波动压缩：近3日绝对收益 vs 前7日绝对收益（量缩+波缩=真供给紧张）

    无手工权重、无涨停惩罚、无高位保持。
    每只股票每天输出连续值 [0, 1]。
    越接近1 = 趋势延续 + 量波双缩 -> 供给紧张，偏多头。
    """
    req = {'stock_code', 'date', 'close', 'volume'}
    if not req.issubset(data.columns):
        return pd.Series(np.nan, index=data.index, dtype='float64')

    stock = data['stock_code']
    dates = pd.to_datetime(data['date'])
    close = pd.to_numeric(data['close'], errors='coerce')
    volume = pd.to_numeric(data['volume'], errors='coerce')

    g_close = close.groupby(stock, sort=False)
    g_vol = volume.groupby(stock, sort=False)

    # ---- 1. 趋势强度（10日收益率） ----
    trend_strength = close / g_close.shift(10) - 1.0

    # ---- 2. 缩量强度（近3日均量 vs 前7日均量，无重叠窗口） ----
    recent_vol = g_vol.transform(lambda x: x.rolling(3, min_periods=2).mean())
    base_vol = g_vol.transform(lambda x: x.shift(3).rolling(7, min_periods=4).mean())
    vol_contraction = np.log1p(base_vol / (recent_vol + 1e-9))

    # ---- 3. 波动压缩（近3日绝对收益 vs 前7日绝对收益） ----
    log_ret = np.log(close / (g_close.shift(1) + 1e-9))
    abs_ret = log_ret.abs()
    g_abs = abs_ret.groupby(stock, sort=False)
    recent_abs = g_abs.transform(lambda x: x.rolling(3, min_periods=2).mean())
    base_abs = g_abs.transform(lambda x: x.shift(3).rolling(7, min_periods=4).mean())
    vola_contraction = np.log1p(base_abs / (recent_abs + 1e-9))

    # ---- 每日截面分位排名 ----
    rank_df = pd.DataFrame({'date': dates})
    eligible = non_st_mask(data).reindex(data.index).fillna(True)

    def _cs_rank(values):
        rank_df['v'] = pd.Series(values, index=data.index).where(eligible)
        return rank_df.groupby('date')['v'].rank(pct=True, na_option='keep')

    r_trend = _cs_rank(trend_strength)
    r_vol = _cs_rank(vol_contraction)
    r_vola = _cs_rank(vola_contraction)

    # ---- 等权几何均值：fuzzy AND，无手工权重 ----
    raw = (r_trend * r_vol * r_vola) ** (1.0 / 3.0)

    return (
        raw
        .fillna(0.5)
        .clip(0.0, 1.0)
        .astype('float64')
    )


# ==============================================================================
# 4b. 真实换手积累 (True Turnover Accumulation)
#     = 10日累计换手率的每日截面分位
#     独立于主因子，捕捉芯片真实交换深度
# ==============================================================================
def _true_turnover_10d(
    data: pd.DataFrame,
    ctx: FactorContext | None = None,
) -> pd.Series:
    """
    10日累计换手率的每日截面分位。

    高值 = 过去10日大量芯片换手（真金白银推动，非无量拉升）；
    低值 = 换手不充分（可能是冷门股或无量一字板）。
    独立于供给紧张主因子，提供"芯片交换深度"的独立维度。
    """
    req = {'stock_code', 'date', 'close', 'volume'}
    if not req.issubset(data.columns):
        return pd.Series(np.nan, index=data.index, dtype='float64')

    dates = pd.to_datetime(data['date'])
    stock = data['stock_code']
    volume = pd.to_numeric(data['volume'], errors='coerce')

    if 'turnover_rate' in data.columns:
        turnover = pd.to_numeric(data['turnover_rate'], errors='coerce')
    else:
        circ_mv = pd.to_numeric(data.get('circ_mv', 1e10), errors='coerce')
        turnover = volume / (circ_mv + 1e-9)

    cum_to_10d = turnover.groupby(stock, sort=False).transform(
        lambda x: x.rolling(10, min_periods=5).sum()
    )

    rank_df = pd.DataFrame({'date': dates})
    eligible = non_st_mask(data).reindex(data.index).fillna(True)
    rank_df['v'] = cum_to_10d.where(eligible)
    result = rank_df.groupby('date')['v'].rank(pct=True, na_option='keep')

    return (
        result
        .fillna(0.5)
        .clip(0.0, 1.0)
        .astype('float64')
    )


# ==============================================================================
# 4c. 近期供给紧张度 (Recent Tightness, ATR-normalized)
#     = exp(-drawdown/ATR)，相对自身波动水平的回撤，截面可比
# ==============================================================================
def _tightness_3d(
    data: pd.DataFrame,
    ctx: FactorContext | None = None,
) -> pd.Series:
    """
    3日微观供给紧张度（ATR归一化版）。

    度量相对自身波动水平的近期抛压：
    drawdown = (前3日峰值 - 当前价) / ATR_5d
    tightness = exp(-drawdown)

    ATR归一化使高波股和低波股截面可比：
    - 高波股跌0.5%但ATR=1% → drawdown=0.5 → tightness≈0.61（还算紧）
    - 低波股跌0.1%但ATR=0.05% → drawdown=2 → tightness≈0.14（其实松）
    而非原来低波股天然 tight、高波股天然 loose 的偏置。

    完全独立于长期趋势，是纯粹的微观结构信号。
    """
    req = {'stock_code', 'date', 'close'}
    if not req.issubset(data.columns):
        return pd.Series(np.nan, index=data.index, dtype='float64')

    stock = data['stock_code']
    close = pd.to_numeric(data['close'], errors='coerce')
    g_close = close.groupby(stock, sort=False)

    # 前3日峰值（shift(1) 避免当天收盘=当天峰值）
    recent_peak = g_close.transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).max()
    )

    # ATR归一化：使截面可比
    tr = close.diff().abs()
    atr_5d = tr.groupby(stock, sort=False).transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )

    # 相对回撤 = 距离峰值 / 正常波动水平
    drawdown = (recent_peak - close) / (atr_5d + 1e-9)
    # exp(-drawdown): drawdown=0→1, 1ATR→0.37, 2ATR→0.14
    tightness = np.exp(-drawdown.clip(0.0, 5.0))

    return (
        tightness
        .fillna(0.5)
        .clip(0.0, 1.0)
        .astype('float64')
    )




post_accumulation_supply_tightness = register_factor(
    FactorDefinition(
        name='post_accumulation_supply_tightness',
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=30,
        calculator=_post_accumulation_supply_tightness,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: _post_accumulation_supply_tightness(d, c)
            .loc[ti].reset_index(drop=True).astype('float64'),
        description='供给紧张结构：趋势x缩量x波缩的等权几何均值(fuzzy AND)，无手工权重',
    )
)

true_turnover_10d = register_factor(
    FactorDefinition(
        name='true_turnover_10d',
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=15,
        calculator=_true_turnover_10d,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: _true_turnover_10d(d, c)
            .loc[ti].reset_index(drop=True).astype('float64'),
        description='真实换手积累：10日累计换手率全市场截面分位，独立芯片深度信号',
    )
)

tightness_3d = register_factor(
    FactorDefinition(
        name='tightness_3d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=5,
        calculator=_tightness_3d,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: _tightness_3d(d, c)
            .loc[ti].reset_index(drop=True).astype('float64'),
        description='3日微观供给紧张度(ATR归一化)：exp(-回撤/ATR5)，相对自身波动的抛压度量',
    )
)


__all__ = [
    'distance_from_52w_high',
    'distance_from_52w_low',
    'price_acceleration',
    'post_accumulation_supply_tightness',
    'true_turnover_10d',
    'tightness_3d',
]
