"""
中短周期结构位置因子模块（robust position family）。

设计目标：
- 替代/增强传统 close_position_Nd = (close - rolling_low) / (rolling_high - rolling_low)。
- 降低单根极端 high/low、长上影、恐慌下影对 position 因子的污染。
- 同时提供收盘价确认的 rolling percentile position，以及 20/60 日多周期结构因子。
- 输出保持可解释、有界、NaN 透传；不在因子内部伪造中性值。
- 所有 rolling 仅按 stock_code 的历史窗口计算，不使用未来数据。

因子说明：
1. close_position_20d_robust / close_position_60d_robust
   使用 rolling 10% low quantile 与 90% high quantile 作为稳健区间锚点，输出 [0,1]。

2. close_rank_position_20d / close_rank_position_60d
   当前 close 在自身过去 N 日 close 序列中的百分位排名，输出 [0,1]。
   该因子只使用收盘价，避免日内 high/low 极端影线干扰。

3. position_20_60_spread
   close_position_20d_robust - close_position_60d_robust，输出 [-1,1]。
   刻画短期位置相对中期位置的抬升/回落。

4. position_pullback_setup_20_60
   close_position_60d_robust * (1 - close_position_20d_robust)，输出 [0,1]。
   刻画“中期结构不弱，但短期已回踩/不过热”的蓄势位置。

5. position_20_60_alignment
   sqrt(close_position_20d_robust * close_position_60d_robust)，输出 [0,1]。
   刻画 20/60 日位置一致处于较高区域的趋势结构。

建议：
- 第一轮重训可优先验证 close_position_20d_robust、close_position_60d_robust、
  position_pullback_setup_20_60、position_20_60_spread。
- 原始 close_position_20d/60d 建议先保留对照，不直接覆盖历史因子。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_EPS = 1e-9


def _nan(index: pd.Index | range) -> pd.Series:
    """返回 float64 NaN 序列。缺必要字段时使用 NaN，而不是伪造 0/0.5 信号。"""
    return pd.Series(np.nan, index=index, dtype="float64")


def _safe_num(s: pd.Series | np.ndarray | float, index: pd.Index | None = None) -> pd.Series:
    if isinstance(s, pd.Series):
        out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return out.astype("float64")
    return pd.Series(s, index=index, dtype="float64").replace([np.inf, -np.inf], np.nan)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """安全除法。分母过小处返回 NaN，避免把窄幅/异常窗口硬填成中性位置。"""
    num = _safe_num(numerator)
    den = _safe_num(denominator).reindex(num.index)
    out = num / den.where(den.abs() > _EPS)
    return out.replace([np.inf, -np.inf], np.nan).astype("float64")


def _target_from_full(full: pd.Series, target_index: pd.Index) -> pd.Series:
    return full.loc[target_index].reset_index(drop=True).astype("float64")


def _rolling_quantile_by_stock(
    data: pd.DataFrame,
    value: pd.Series,
    window: int,
    min_periods: int,
    quantile: float,
) -> pd.Series:
    """按 stock_code 做 rolling quantile，保持原 index 对齐。"""
    val = _safe_num(value).reindex(data.index)
    out = (
        val.groupby(data["stock_code"], sort=False)
        .rolling(window=window, min_periods=min_periods)
        .quantile(quantile)
        .reset_index(level=0, drop=True)
    )
    return out.astype("float64")


def _rolling_rank_pct_by_stock(
    data: pd.DataFrame,
    value: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """
    当前值在自身历史 rolling window 内的百分位排名。

    pandas Rolling.rank(pct=True) 直接给出当前观测在窗口内的百分位，
    比 rolling.apply 自定义 Python 函数更快、更稳定。
    """
    val = _safe_num(value).reindex(data.index)
    grouped = val.groupby(data["stock_code"], sort=False)
    try:
        out = grouped.rolling(window=window, min_periods=min_periods).rank(pct=True)
        out = out.reset_index(level=0, drop=True)
    except AttributeError:
        # 兼容极老 pandas：窗口较小，只作为兜底路径。
        def _last_rank_pct(arr: np.ndarray) -> float:
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return np.nan
            last = arr[-1]
            if not np.isfinite(last):
                return np.nan
            less = np.sum(arr < last)
            equal = np.sum(arr == last)
            return float((less + 0.5 * equal) / arr.size)

        out = (
            grouped.rolling(window=window, min_periods=min_periods)
            .apply(_last_rank_pct, raw=True)
            .reset_index(level=0, drop=True)
        )
    return out.clip(0.0, 1.0).astype("float64")


def _close_position_robust_raw(
    data: pd.DataFrame,
    window: int,
    min_periods: int,
    low_q: float = 0.10,
    high_q: float = 0.90,
) -> pd.Series:
    """
    稳健区间位置：使用 low 的低分位与 high 的高分位作为区间边界。

    相比 rolling max/min：
    - 避免单根长上影把高点锚点拉得过高；
    - 避免单根恐慌下影把低点锚点拉得过低；
    - 仍保留 close 在近期价格区间中所处位置的经济含义。
    """
    required = {"stock_code", "high", "low", "close"}
    if not required.issubset(data.columns):
        return _nan(data.index)

    low = _safe_num(data["low"]).astype("float64")
    high = _safe_num(data["high"]).astype("float64")
    close = _safe_num(data["close"]).astype("float64")

    low_anchor = _rolling_quantile_by_stock(data, low, window, min_periods, low_q)
    high_anchor = _rolling_quantile_by_stock(data, high, window, min_periods, high_q)
    pos = _safe_div(close - low_anchor, high_anchor - low_anchor)
    return pos.clip(0.0, 1.0).astype("float64")


def _close_rank_position_raw(data: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    """当前 close 在过去 N 日 close 窗口内的百分位位置。"""
    if not {"stock_code", "close"}.issubset(data.columns):
        return _nan(data.index)
    close = _safe_num(data["close"]).astype("float64")
    return _rolling_rank_pct_by_stock(data, close, window, min_periods).astype("float64")


def _pos20_robust_full(data: pd.DataFrame) -> pd.Series:
    return _close_position_robust_raw(data, window=20, min_periods=10)


def _pos60_robust_full(data: pd.DataFrame) -> pd.Series:
    return _close_position_robust_raw(data, window=60, min_periods=30)


# ==============================================================================
# 1. Robust range position
# ==============================================================================
def _close_position_20d_robust(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _pos20_robust_full(data)


def _close_position_20d_robust_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_close_position_20d_robust(data, _), target_index)


close_position_20d_robust = register_factor(
    FactorDefinition(
        name="close_position_20d_robust",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=30,
        calculator=_close_position_20d_robust,
        calculation_scope="target_with_history_window",
        target_calculator=_close_position_20d_robust_target,
        description="[0,1] 稳健20日位置：close相对rolling low 10%分位与high 90%分位的位置，降低极端影线污染",
    )
)


def _close_position_60d_robust(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _pos60_robust_full(data)


def _close_position_60d_robust_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_close_position_60d_robust(data, _), target_index)


close_position_60d_robust = register_factor(
    FactorDefinition(
        name="close_position_60d_robust",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=80,
        calculator=_close_position_60d_robust,
        calculation_scope="target_with_history_window",
        target_calculator=_close_position_60d_robust_target,
        description="[0,1] 稳健60日位置：close相对rolling low 10%分位与high 90%分位的位置，刻画中期结构位置",
    )
)


# ==============================================================================
# 2. Close-only percentile position
# ==============================================================================
def _close_rank_position_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _close_rank_position_raw(data, window=20, min_periods=10)


def _close_rank_position_20d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_close_rank_position_20d(data, _), target_index)


close_rank_position_20d = register_factor(
    FactorDefinition(
        name="close_rank_position_20d",
        required_columns={"stock_code", "date", "close"},
        lookback_days=30,
        calculator=_close_rank_position_20d,
        calculation_scope="target_with_history_window",
        target_calculator=_close_rank_position_20d_target,
        description="[0,1] 当前close在过去20日close序列中的百分位排名，只看收盘确认位置，不受日内影线污染",
    )
)


def _close_rank_position_60d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _close_rank_position_raw(data, window=60, min_periods=30)


def _close_rank_position_60d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_close_rank_position_60d(data, _), target_index)


close_rank_position_60d = register_factor(
    FactorDefinition(
        name="close_rank_position_60d",
        required_columns={"stock_code", "date", "close"},
        lookback_days=80,
        calculator=_close_rank_position_60d,
        calculation_scope="target_with_history_window",
        target_calculator=_close_rank_position_60d_target,
        description="[0,1] 当前close在过去60日close序列中的百分位排名，只看收盘确认的中期位置",
    )
)


# ==============================================================================
# 3. 20/60 multi-horizon position structure
# ==============================================================================
def _position_20_60_spread(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    pos20 = _pos20_robust_full(data)
    pos60 = _pos60_robust_full(data)
    return (pos20 - pos60).clip(-1.0, 1.0).astype("float64")


def _position_20_60_spread_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_position_20_60_spread(data, _), target_index)


position_20_60_spread = register_factor(
    FactorDefinition(
        name="position_20_60_spread",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=80,
        calculator=_position_20_60_spread,
        calculation_scope="target_with_history_window",
        target_calculator=_position_20_60_spread_target,
        description="[-1,1] 稳健20日位置-稳健60日位置；越高表示短期相对中期快速抬升，越低表示中期较强但短期回踩",
    )
)


def _position_pullback_setup_20_60(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    pos20 = _pos20_robust_full(data)
    pos60 = _pos60_robust_full(data)
    score = pos60 * (1.0 - pos20)
    return score.clip(0.0, 1.0).astype("float64")


def _position_pullback_setup_20_60_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_position_pullback_setup_20_60(data, _), target_index)


position_pullback_setup_20_60 = register_factor(
    FactorDefinition(
        name="position_pullback_setup_20_60",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=80,
        calculator=_position_pullback_setup_20_60,
        calculation_scope="target_with_history_window",
        target_calculator=_position_pullback_setup_20_60_target,
        description="[0,1] 中期位置较强但短期不过热/已回踩的蓄势结构：pos60_robust*(1-pos20_robust)",
    )
)


def _position_20_60_alignment(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    pos20 = _pos20_robust_full(data)
    pos60 = _pos60_robust_full(data)
    product = (pos20 * pos60).clip(lower=0.0)
    return np.sqrt(product).clip(0.0, 1.0).astype("float64")


def _position_20_60_alignment_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_position_20_60_alignment(data, _), target_index)


position_20_60_alignment = register_factor(
    FactorDefinition(
        name="position_20_60_alignment",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=80,
        calculator=_position_20_60_alignment,
        calculation_scope="target_with_history_window",
        target_calculator=_position_20_60_alignment_target,
        description="[0,1] 20日和60日稳健位置一致性：sqrt(pos20_robust*pos60_robust)，刻画多周期高位/强趋势结构",
    )
)


__all__ = [
    "close_position_20d_robust",
    "close_position_60d_robust",
    "close_rank_position_20d",
    "close_rank_position_60d",
    "position_20_60_spread",
    "position_pullback_setup_20_60",
    "position_20_60_alignment",
]
