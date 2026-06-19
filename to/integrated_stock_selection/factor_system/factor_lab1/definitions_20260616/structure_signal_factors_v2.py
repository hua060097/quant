"""
结构形态与多周期位置升级因子（v2，高性能版）。

设计目标：
1. 升级 pre_second_breakout_shapelet_corr_20d：
   - 保留已验证有效的“首次上攻 -> 回落 -> 横住蓄势”预测语义；
   - 不包含二次主升完成后的暴涨段，避免事后形态识别/高位接盘；
   - 用 NumPy sliding_window_view 向量化 20日窗口内的 12日 shapelet 滑动匹配，显著减少 Python 循环；
   - 使用多 shapelet max pooling，覆盖不同节奏的试盘-回落-蓄势形态；
   - 对较早出现的 shapelet 做连续 recency 衰减，减少“很久以前像、当前已走完”的 stale pattern；
   - 对最近 3 日已明显拉升的正向信号做连续过热惩罚。

2. 升级 position_20_60_spread：
   - 因子测试显示单独静态 position 弱，但 20/60 相对位置 spread 有信息；
   - 使用原始 high/low extreme position，而不是 robust quantile position，因为测试结果暗示极端高低点本身含有短线形态信息；
   - 新增 3日 spread 修复速度，构成“相对位置水平 + 相对位置改善”的单一连续因子；
   - 输出有界、可解释、NaN 透传。

注意：
- 本文件使用 v2 新因子名，避免与旧因子重名冲突，便于 A/B 测试。
- 若验证通过，可将注册 name 改回原因子名用于替换。
- 所有 rolling/shift 均按 stock_code 历史序列计算，不使用未来数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series

_EPS = 1e-12

# ==============================================================================
# 1. Fast multi-shapelet pre-second-breakout factor
# ==============================================================================

_WINDOW_DAYS = 20
_SHAPELET_DAYS = 12
_MIN_VALID_CLOSES = 20
_RECENT_OVERHEAT_DAYS = 3
_OVERHEAT_PENALTY_STRENGTH = 20.0
_RECENCY_DECAY = 0.10  # 连续衰减，越早结束的 shapelet 权重越低；非阈值规则。

# 三个 12日局部模板：都只覆盖“试盘/首次上攻 -> 回落 -> 横住蓄势”，不包含二次暴力主升。
_SHAPELETS_12 = np.array(
    [
        # Balanced: 首次上攻较清晰，回落后缓慢抬升。
        [0.000, 0.010, 0.030, 0.070, 0.110, 0.090, 0.060, 0.040, 0.045, 0.050, 0.055, 0.060],
        # Fast impulse: 前段启动更快，随后回落横住。
        [0.000, 0.020, 0.060, 0.110, 0.130, 0.100, 0.070, 0.045, 0.040, 0.045, 0.052, 0.058],
        # Slow reaccumulation: 上攻略慢，回落较温和，尾部蓄势更平滑。
        [0.000, 0.004, 0.012, 0.030, 0.060, 0.090, 0.105, 0.080, 0.055, 0.048, 0.052, 0.058],
    ],
    dtype=np.float64,
)


def _detrend_zscore_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    对最后一维为时间轴的矩阵做去线性趋势 + population z-score。

    Parameters
    ----------
    x : ndarray, shape (..., L)

    Returns
    -------
    z : ndarray, shape (..., L)
        去趋势标准化后的路径。低方差/非法路径处填 0。
    valid : ndarray, shape (...)
        该路径是否有效。
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[-1] < 3:
        valid = np.zeros(arr.shape[:-1], dtype=bool)
        return np.zeros_like(arr, dtype=np.float64), valid

    finite = np.isfinite(arr).all(axis=-1)
    l = arr.shape[-1]
    t = np.linspace(-1.0, 1.0, l, dtype=np.float64)
    t_denom = float(np.dot(t, t))

    mean = np.nanmean(arr, axis=-1, keepdims=True)
    centered = arr - mean
    slope = np.sum(centered * t, axis=-1, keepdims=True) / (t_denom + _EPS)
    resid = centered - slope * t
    std = np.sqrt(np.nanmean(resid * resid, axis=-1, keepdims=True))
    valid = finite & np.isfinite(std[..., 0]) & (std[..., 0] > _EPS)

    z = np.divide(resid, std, out=np.zeros_like(resid, dtype=np.float64), where=std > _EPS)
    z = np.where(valid[..., None], z, 0.0)
    return z.astype(np.float64), valid


def _prepare_template_z() -> np.ndarray:
    z, valid = _detrend_zscore_matrix(_SHAPELETS_12)
    if not bool(valid.all()):
        raise RuntimeError("One or more pre-second-breakout v2 shapelets have near-zero residual variance")
    return z.astype(np.float64)


_TEMPLATE_Z = _prepare_template_z()  # shape: (K, 12)
_START_COUNT = _WINDOW_DAYS - _SHAPELET_DAYS + 1
# start 越接近最后一个可用窗口，说明该局部结构越靠近当前预测日，权重越高。
_RECENCY_WEIGHTS = np.exp(-_RECENCY_DECAY * (_START_COUNT - 1 - np.arange(_START_COUNT, dtype=np.float64)))


def _fast_shapelet_score_for_sorted_close(close_values: np.ndarray) -> np.ndarray:
    """对单只股票已按日期排序的 close 序列，向量化计算 v2 shapelet 分数。"""
    close = np.asarray(close_values, dtype=np.float64)
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < _MIN_VALID_CLOSES:
        return out

    finite_positive = np.isfinite(close) & (close > 0.0)
    # 20日 close 窗口: shape (n-19, 20)
    windows = sliding_window_view(close, _WINDOW_DAYS)
    valid_windows = sliding_window_view(finite_positive, _WINDOW_DAYS).all(axis=1)

    if windows.size == 0:
        return out

    # 累计 log return 路径，消除价格量纲。
    with np.errstate(divide="ignore", invalid="ignore"):
        paths = np.log(windows / windows[:, :1])

    # 从每个 20日路径中滑动截取 12日子路径: shape (m, 9, 12)
    sub_paths = sliding_window_view(paths, _SHAPELET_DAYS, axis=1)
    # numpy 对 axis=1 的输出通常为 (m, 9, 12)，这里显式确保。
    if sub_paths.shape[1] != _START_COUNT:
        sub_paths = np.swapaxes(sub_paths, 1, 2)

    sub_z, valid_sub = _detrend_zscore_matrix(sub_paths)
    # corr: shape (m, 9, K)
    corr = np.einsum("msl,kl->msk", sub_z, _TEMPLATE_Z, optimize=True) / float(_SHAPELET_DAYS)
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.where(valid_sub[..., None], corr, 0.0)

    # 旧形态衰减：避免很早出现过形态、当前已进入主升后仍给高分。
    corr = corr * _RECENCY_WEIGHTS[None, :, None]
    best = np.max(corr, axis=(1, 2))

    # 窗口本身非法则 NaN。
    best = np.where(valid_windows & np.isfinite(best), best, np.nan)

    # 最近3日过热惩罚，只惩罚正向结构相似度，负值不被抬高。
    end_positions = np.arange(_WINDOW_DAYS - 1, n, dtype=np.int64)
    start_3d = end_positions - _RECENT_OVERHEAT_DAYS
    valid_recent = start_3d >= 0
    recent_log_ret = np.zeros(best.shape[0], dtype=np.float64)
    if valid_recent.any():
        c0 = close[start_3d[valid_recent]]
        c1 = close[end_positions[valid_recent]]
        ok = np.isfinite(c0) & np.isfinite(c1) & (c0 > 0.0) & (c1 > 0.0)
        tmp = np.zeros(ok.shape[0], dtype=np.float64)
        tmp[ok] = np.log(c1[ok] / c0[ok])
        recent_log_ret[valid_recent] = tmp

    pos_recent = np.maximum(recent_log_ret, 0.0)
    penalty = np.exp(-_OVERHEAT_PENALTY_STRENGTH * pos_recent * pos_recent)
    best = np.where(best > 0.0, best * penalty, best)

    out[_WINDOW_DAYS - 1:] = np.clip(best, -1.0, 1.0)
    return out.astype(np.float64)


def _pre_second_breakout_shapelet_corr_20d_v2(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    required = {"stock_code", "date", "close"}
    if not required.issubset(data.columns):
        return pd.Series(np.nan, index=data.index, dtype="float64")

    close_all = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=np.float64)
    date_values = pd.to_datetime(data["date"], errors="coerce").to_numpy()
    result = np.full(len(data), np.nan, dtype=np.float64)

    for _, positions in data.groupby("stock_code", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size < _MIN_VALID_CLOSES:
            continue
        # 显式按日期排序后计算，再写回原始位置，避免依赖上游排序。
        order = np.argsort(date_values[pos], kind="mergesort")
        sorted_pos = pos[order]
        close_sorted = close_all[sorted_pos]
        result[sorted_pos] = _fast_shapelet_score_for_sorted_close(close_sorted)

    return pd.Series(result, index=data.index, dtype="float64")


def _pre_second_breakout_shapelet_corr_20d_v2_target(
    data: pd.DataFrame,
    target_index: pd.Index,
    _: FactorContext,
) -> pd.Series:
    return select_target_series(
        _pre_second_breakout_shapelet_corr_20d_v2(data, _),
        target_index=target_index,
    )


pre_second_breakout_shapelet_corr_20d_v2 = register_factor(
    FactorDefinition(
        name="pre_second_breakout_shapelet_corr_20d_v2",
        required_columns={"stock_code", "date", "close"},
        lookback_days=30,
        calculator=_pre_second_breakout_shapelet_corr_20d_v2,
        calculation_scope="target_with_history_window",
        target_calculator=_pre_second_breakout_shapelet_corr_20d_v2_target,
        description="v2高性能多Shapelet：20日内匹配首次上攻-回落-横住蓄势结构，带recency衰减与最近3日过热惩罚",
    )
)


# ==============================================================================
# 2. Position 20/60 spread v2: level + reclaim speed
# ==============================================================================

def _nan(index: pd.Index | range) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="float64")


def _safe_num(s: pd.Series | np.ndarray | float, index: pd.Index | None = None) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    return pd.Series(s, index=index, dtype="float64").replace([np.inf, -np.inf], np.nan)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = _safe_num(numerator)
    den = _safe_num(denominator).reindex(num.index)
    out = num / den.where(den.abs() > 1e-9)
    return out.replace([np.inf, -np.inf], np.nan).astype("float64")


def _rolling_col_by_stock(
    data: pd.DataFrame,
    value: pd.Series,
    window: int,
    min_periods: int,
    agg: str,
) -> pd.Series:
    val = _safe_num(value).reindex(data.index)
    rolled = val.groupby(data["stock_code"], sort=False).rolling(window=window, min_periods=min_periods)
    if agg == "max":
        out = rolled.max()
    elif agg == "min":
        out = rolled.min()
    else:
        raise ValueError(f"Unsupported rolling agg: {agg}")
    return out.reset_index(level=0, drop=True).astype("float64")


def _close_position_extreme_raw(data: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    """
    原始高低点区间位置，不填 0.5。

    这类 extreme anchor 在本轮测试中没有被 robust quantile 版本击败，说明极端高/低点
    本身可能包含短线冲高失败、箱体修复等有效结构信息，因此 v2 spread 使用该定义。
    """
    required = {"stock_code", "high", "low", "close"}
    if not required.issubset(data.columns):
        return _nan(data.index)

    high = _safe_num(data["high"])
    low = _safe_num(data["low"])
    close = _safe_num(data["close"])
    high_n = _rolling_col_by_stock(data, high, window, min_periods, "max")
    low_n = _rolling_col_by_stock(data, low, window, min_periods, "min")
    pos = _safe_div(close - low_n, high_n - low_n)
    return pos.clip(0.0, 1.0).astype("float64")


def _shift_by_stock(data: pd.DataFrame, series: pd.Series, periods: int) -> pd.Series:
    return series.groupby(data["stock_code"], sort=False).shift(periods).astype("float64")


def _position_20_60_spread_v2(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    多周期位置修复因子 v2。

    基础：spread = extreme_pos20 - extreme_pos60。
    测试显示 spread 相比单独 position 更有信息，因此保留多周期相对位置语义。

    升级：加入 3日 spread 改善速度：delta3 = spread - spread.shift(3)。
    为避免 delta3 的天然范围约为 spread 的两倍，使用 delta3/2 做等尺度归一。

    score = 0.5 * spread + 0.5 * (delta3 / 2)

    含义：
    - 高分：短期位置相对中期位置较强，且最近在改善；
    - 低分：短期相对中期走弱，或高位回落结构更明显。
    """
    required = {"stock_code", "date", "high", "low", "close"}
    if not required.issubset(data.columns):
        return _nan(data.index)

    pos20 = _close_position_extreme_raw(data, window=20, min_periods=10)
    pos60 = _close_position_extreme_raw(data, window=60, min_periods=30)
    spread = (pos20 - pos60).clip(-1.0, 1.0)
    delta3 = spread - _shift_by_stock(data, spread, periods=3)

    # spread 和 delta3/2 都被映射到大致 [-1,1]，再等权平均；不是回测调参权重。
    score = 0.5 * spread + 0.5 * (delta3 / 2.0)
    return score.clip(-1.0, 1.0).astype("float64")


def _position_20_60_spread_v2_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return select_target_series(_position_20_60_spread_v2(data, _), target_index=target_index)


position_20_60_spread_v2 = register_factor(
    FactorDefinition(
        name="position_20_60_spread_v2",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=85,
        calculator=_position_20_60_spread_v2,
        calculation_scope="target_with_history_window",
        target_calculator=_position_20_60_spread_v2_target,
        description="v2多周期位置修复：extreme_pos20-extreme_pos60，并融合3日spread改善速度，刻画短期相对中期位置上移",
    )
)


__all__ = [
    "pre_second_breakout_shapelet_corr_20d_v2",
    "position_20_60_spread_v2",
]
