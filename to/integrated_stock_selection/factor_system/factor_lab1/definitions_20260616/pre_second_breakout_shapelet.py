"""
二次拉升前蓄势结构因子（Shapelet 形态相似度版）。

设计目标：
1. 识别“首次上攻 → 回落洗盘 → 横住蓄势”的二次拉升前结构；
2. 不把二次主升完成后的暴涨走势纳入模板，避免高位接盘信号；
3. 使用滑动 shapelet 最大相似度，而非固定完整路径模板，允许不同股票节奏不一致；
4. 使用累计 log return 路径、去线性趋势后的相关系数，减少绝对涨幅/趋势强度主导；
5. 对最近 3 日已明显拉升的股票做连续过热惩罚；
6. 全程保留 NaN，不伪造信号。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series


# ==============================================================================
# 二次拉升前蓄势 Shapelet
# ==============================================================================
# 12 日局部形态模板：首次上攻 -> 回落 -> 横住/轻微抬升。
# 注意：模板不包含“二次暴力拉升完成”阶段，目的是预测主升前结构，而非事后识别。
_PRE_SECOND_BREAKOUT_SHAPELET_12 = np.array(
    [
        0.000, 0.010, 0.030, 0.070, 0.110,
        0.090, 0.060, 0.040,
        0.045, 0.050, 0.055, 0.060,
    ],
    dtype=np.float64,
)

_WINDOW_DAYS = 20
_SHAPELET_DAYS = 12
_MIN_VALID_CLOSES = 20
_RECENT_OVERHEAT_DAYS = 3
_OVERHEAT_PENALTY_STRENGTH = 20.0
_EPS = 1e-12


def _detrended_zscore_np(values: np.ndarray) -> np.ndarray | None:
    """
    对短路径做去线性趋势 + z-score。

    使用固定长度小数组的 numpy 实现，避免 rolling/apply 的高额开销。
    若残差方差过小，返回 None，调用方可将相似度视为 0。
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 3 or not np.isfinite(x).all():
        return None

    n = x.size
    t = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    x_mean = float(x.mean())
    t_denom = float(np.dot(t, t))
    if t_denom <= _EPS:
        return None

    slope = float(np.dot(x - x_mean, t) / t_denom)
    resid = x - (x_mean + slope * t)
    std = float(np.sqrt(np.mean(resid * resid)))
    if not np.isfinite(std) or std <= _EPS:
        return None
    return resid / std


_TEMPLATE_Z = _detrended_zscore_np(_PRE_SECOND_BREAKOUT_SHAPELET_12)
if _TEMPLATE_Z is None:  # 理论上不应发生，保留硬保护。
    raise RuntimeError("pre-second-breakout shapelet template has near-zero residual variance")


def _max_sliding_shapelet_corr_from_close(close_window: np.ndarray) -> float:
    """
    在 20 日 close 窗口内滑动匹配 12 日 shapelet，返回最大去趋势相关系数。

    输入为连续 20 日收盘价，要求全为有限正数。
    """
    close = np.asarray(close_window, dtype=np.float64)
    if close.size != _WINDOW_DAYS or not np.isfinite(close).all() or np.any(close <= 0.0):
        return np.nan

    # 累计 log return 路径，消除价格量纲。
    path = np.log(close / close[0])
    if not np.isfinite(path).all():
        return np.nan

    best_corr = -1.0
    last_start = _WINDOW_DAYS - _SHAPELET_DAYS
    for start in range(last_start + 1):
        sub_path = path[start:start + _SHAPELET_DAYS]
        sub_z = _detrended_zscore_np(sub_path)
        if sub_z is None:
            corr = 0.0
        else:
            # 两边均为 population z-score，均值约0，std约1；相关系数用 mean(dot) 即可。
            corr = float(np.mean(sub_z * _TEMPLATE_Z))
            if not np.isfinite(corr):
                corr = 0.0
            else:
                corr = float(np.clip(corr, -1.0, 1.0))
        if corr > best_corr:
            best_corr = corr

    return best_corr


def _recent_overheat_penalty(close_values: np.ndarray, end_pos: int) -> float:
    """
    最近 3 日已经明显上涨时，对正向形态相似度做平滑惩罚。

    该惩罚没有硬阈值：上涨越大，惩罚越强；最近 3 日不涨或下跌则不惩罚。
    """
    start_pos = end_pos - _RECENT_OVERHEAT_DAYS
    if start_pos < 0:
        return 1.0

    c0 = close_values[start_pos]
    c1 = close_values[end_pos]
    if not (np.isfinite(c0) and np.isfinite(c1)) or c0 <= 0.0 or c1 <= 0.0:
        return 1.0

    recent_log_ret = float(np.log(c1 / c0))
    if not np.isfinite(recent_log_ret) or recent_log_ret <= 0.0:
        return 1.0
    return float(np.exp(-_OVERHEAT_PENALTY_STRENGTH * recent_log_ret * recent_log_ret))


def _compute_pre_second_breakout_shapelet_for_group(close_values: np.ndarray) -> np.ndarray:
    """
    对单只股票按时间顺序计算二次拉升前蓄势 shapelet 因子。
    """
    close = np.asarray(close_values, dtype=np.float64)
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < _MIN_VALID_CLOSES:
        return out

    for end_pos in range(_WINDOW_DAYS - 1, n):
        window = close[end_pos - _WINDOW_DAYS + 1:end_pos + 1]
        if not np.isfinite(window).all() or np.any(window <= 0.0):
            continue

        best_corr = _max_sliding_shapelet_corr_from_close(window)
        if not np.isfinite(best_corr):
            continue

        # 仅对正向相似结构做“已过热”惩罚；负相关结构保持负值，避免被错误抬高。
        if best_corr > 0.0:
            best_corr *= _recent_overheat_penalty(close, end_pos)

        out[end_pos] = float(np.clip(best_corr, -1.0, 1.0))

    return out


def _pre_second_breakout_shapelet_corr_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    二次拉升前蓄势结构因子。

    因子定义：过去20日内，任意连续12日子路径与“首次上攻-回落-横住蓄势”shapelet
    的最大去趋势相关系数，并对最近3日已明显拉升的正向相似度做连续惩罚。

    数值含义：
    - 越接近 1：越像“首次上攻后洗盘蓄势，尚未完成二次主升”；
    - 接近 0：无明显结构；
    - 越接近 -1：走势结构与目标形态相反。
    """
    required_cols = {'stock_code', 'date', 'close'}
    if not required_cols.issubset(data.columns):
        return pd.Series(np.nan, index=data.index, dtype='float64')

    close_all = pd.to_numeric(data['close'], errors='coerce').to_numpy(dtype=np.float64)
    result = np.full(len(data), np.nan, dtype=np.float64)

    # 按股票独立计算。每组内部显式按 date 排序，再按原始行位置写回，避免依赖外部排序。
    stock_groups = data.groupby('stock_code', sort=False).indices
    date_values = pd.to_datetime(data['date'], errors='coerce').to_numpy()

    for _, positions in stock_groups.items():
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size < _MIN_VALID_CLOSES:
            continue

        group_dates = date_values[pos]
        # NaT 会排到末尾；正常因子数据不应有 NaT，这里只做安全处理。
        order = np.argsort(group_dates, kind='mergesort')
        sorted_pos = pos[order]
        close_group = close_all[sorted_pos]

        score_group = _compute_pre_second_breakout_shapelet_for_group(close_group)
        result[sorted_pos] = score_group

    return pd.Series(result, index=data.index, dtype='float64')


def _pre_second_breakout_shapelet_corr_20d_target(
    data: pd.DataFrame,
    target_index: pd.Index,
    _: FactorContext,
) -> pd.Series:
    return select_target_series(
        _pre_second_breakout_shapelet_corr_20d(data, _),
        target_index=target_index,
    )


pre_second_breakout_shapelet_corr_20d = register_factor(
    FactorDefinition(
        name='pre_second_breakout_shapelet_corr_20d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=30,
        calculator=_pre_second_breakout_shapelet_corr_20d,
        calculation_scope='target_with_history_window',
        target_calculator=_pre_second_breakout_shapelet_corr_20d_target,
        description='二次拉升前蓄势结构Shapelet相似度：首次上攻-回落-横住蓄势，惩罚最近3日已过热拉升',
    )
)


__all__ = [
    'pre_second_breakout_shapelet_corr_20d',
]
