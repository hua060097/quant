"""
紧凑版隔夜/日内收益结构因子。

目标：用一个综合因子刻画短线上涨质量，避免将隔夜跳空贡献、日内承接、
高开日内兑现压力拆成过多高度相关的特征。

因子含义：
- 值越高：过去 5 日收益更多来自日内承接，隔夜跳空依赖较低，高开后日内兑现压力较小。
- 值越低：过去 5 日更依赖隔夜跳空，或高开后日内走弱/兑现压力明显。

设计原则：
1. 使用 log return，保证隔夜收益与日内收益在窗口内可加。
2. 只注册一个主因子，降低同源特征冗余。
3. 保留 NaN，不使用 fillna(0)，避免伪造信号。
4. 所有比例项使用有效波动规模归一化，避免总收益接近 0 时比例爆炸。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series


_WINDOW = 5
_MIN_PERIODS = 5
_DENOM_FLOOR = 0.02  # 约等于 2% log-return floor，抑制低波动窗口比例爆炸。
_OUTPUT_CLIP = 3.0


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """安全计算 log(numerator / denominator)，非正价格返回 NaN。"""
    numerator = pd.to_numeric(numerator, errors='coerce')
    denominator = pd.to_numeric(denominator, errors='coerce')
    valid = (numerator > 0) & (denominator > 0)
    out = pd.Series(np.nan, index=numerator.index, dtype='float64')
    out.loc[valid] = np.log(numerator.loc[valid] / denominator.loc[valid])
    return out


def _rolling_component_sums(data: pd.DataFrame, window: int = _WINDOW) -> pd.DataFrame:
    """按股票计算隔夜/日内结构的 rolling sum，保持原 index 对齐。"""
    close_prev = data.groupby('stock_code', sort=False)['close'].shift(1)
    gap_log_ret = _safe_log_ratio(data['open'], close_prev)
    intra_log_ret = _safe_log_ratio(data['close'], data['open'])

    valid_pair = gap_log_ret.notna() & intra_log_ret.notna()

    # 正隔夜跳空规模：只统计 gap > 0 的隔夜上涨，衡量“靠高开定价”的依赖。
    positive_gap_abs = pd.Series(np.nan, index=data.index, dtype='float64')
    positive_gap_abs.loc[valid_pair] = 0.0
    positive_gap_mask = valid_pair & (gap_log_ret > 0)
    positive_gap_abs.loc[positive_gap_mask] = gap_log_ret.loc[positive_gap_mask].abs()

    # 高开后日内兑现：gap > 0 且 intra < 0 时累加 abs(intra)。
    gap_up_selloff = pd.Series(np.nan, index=data.index, dtype='float64')
    gap_up_selloff.loc[valid_pair] = 0.0
    selloff_mask = valid_pair & (gap_log_ret > 0) & (intra_log_ret < 0)
    gap_up_selloff.loc[selloff_mask] = intra_log_ret.loc[selloff_mask].abs()

    temp = pd.DataFrame(
        {
            'gap': gap_log_ret.values,
            'intra': intra_log_ret.values,
            'abs_gap': gap_log_ret.abs().values,
            'abs_intra': intra_log_ret.abs().values,
            'positive_gap_abs': positive_gap_abs.values,
            'gap_up_selloff': gap_up_selloff.values,
            'stock_code': data['stock_code'].values,
        },
        index=data.index,
    )

    rolled = (
        temp.groupby('stock_code', sort=False)[
            ['gap', 'intra', 'abs_gap', 'abs_intra', 'positive_gap_abs', 'gap_up_selloff']
        ]
        .rolling(window, min_periods=_MIN_PERIODS)
        .sum()
        .reset_index(level=0, drop=True)
        .astype('float64')
    )
    return rolled.reindex(data.index)


def _gap_intraday_quality_score(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    5日隔夜/日内综合结构质量因子。

    score = 日内承接强度 - 正隔夜跳空依赖 - 高开日内兑现压力

    其中各分量均按窗口内有效波动规模归一化：
    effective_scale = sum(abs(gap)) + sum(abs(intra)) + floor

    这个设计比单独输出 gap_contrib/intra_contrib/selloff 多个同源因子更紧凑，
    同时避免在总收益接近 0 的情况下出现极端比例。
    """
    required = {'stock_code', 'date', 'open', 'close'}
    if not required.issubset(data.columns):
        return pd.Series(np.nan, index=data.index)

    sums = _rolling_component_sums(data, _WINDOW)
    effective_scale = sums['abs_gap'] + sums['abs_intra'] + _DENOM_FLOOR

    # 日内承接强度：日内累计收益越强越好。
    intraday_support = sums['intra'] / effective_scale

    # 正跳空依赖：只惩罚正向高开贡献，避免把低开后日内修复误判为坏结构。
    positive_gap_reliance = sums['positive_gap_abs'] / effective_scale

    # 高开后日内兑现压力：高开但日内收跌，是“靠隔夜情绪续命、盘中承接不足”的核心风险。
    selloff_pressure = sums['gap_up_selloff'] / effective_scale

    score = intraday_support - positive_gap_reliance - selloff_pressure
    return score.clip(-_OUTPUT_CLIP, _OUTPUT_CLIP).astype('float64')


def _gap_intraday_quality_score_target(
    data: pd.DataFrame,
    target_index: pd.Index,
    ctx: FactorContext,
) -> pd.Series:
    return select_target_series(_gap_intraday_quality_score(data, ctx), target_index=target_index)


gap_intraday_quality_score_5d = register_factor(
    FactorDefinition(
        name='gap_intraday_quality_score_5d',
        required_columns={'stock_code', 'date', 'open', 'close'},
        lookback_days=_WINDOW + 2,
        calculator=_gap_intraday_quality_score,
        calculation_scope='target_with_history_window',
        target_calculator=_gap_intraday_quality_score_target,
        description='5日隔夜/日内综合结构质量：日内承接强、正跳空依赖低、高开兑现压力低时得分更高',
    )
)


__all__ = [
    'gap_intraday_quality_score_5d',
]
