"""
Run-up turbulence factors.

These factors distinguish a smooth advance from a late-stage advance that
requires increasingly large intraday ranges to make limited directional
progress.  All calculations are causal and use only each stock's own history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series

_EPS = 1e-12


def _ordered_frame(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values(["stock_code", "date"], kind="mergesort")


def _rolling(
    values: pd.Series,
    stocks: pd.Series,
    window: int,
    min_periods: int,
    agg: str,
) -> pd.Series:
    rolling = values.groupby(stocks, sort=False).rolling(window, min_periods=min_periods)
    if agg == "mean":
        result = rolling.mean()
    elif agg == "sum":
        result = rolling.sum()
    elif agg == "max":
        result = rolling.max()
    elif agg == "min":
        result = rolling.min()
    elif agg == "median":
        result = rolling.median()
    else:
        raise ValueError(f"Unsupported rolling aggregation: {agg}")
    return result.reset_index(level=0, drop=True)


def _has_meaningful_coverage(values: pd.Series, length: int) -> bool:
    min_coverage = max(20, int(length * 0.05))
    return values.notna().sum() >= min_coverage


def _activity_proxy(ordered: pd.DataFrame, close: pd.Series) -> pd.Series:
    """
    Select a price-neutral trading-activity proxy.

    Turnover rate is preferred because it directly measures ownership churn.
    Amount is divided by close to remove the mechanical increase in traded
    value caused only by a rising share price.  Raw volume is the last fallback.
    """
    if "turnover_rate" in ordered.columns:
        turnover = pd.to_numeric(ordered["turnover_rate"], errors="coerce").astype("float64")
        turnover = turnover.where(turnover > 0.0)
        if _has_meaningful_coverage(turnover, len(ordered)):
            return turnover

    if "amount" in ordered.columns:
        amount = pd.to_numeric(ordered["amount"], errors="coerce").astype("float64")
        implied_volume = amount / close.replace(0.0, np.nan)
        implied_volume = implied_volume.where(implied_volume > 0.0)
        if _has_meaningful_coverage(implied_volume, len(ordered)):
            return implied_volume

    if "volume" in ordered.columns:
        volume = pd.to_numeric(ordered["volume"], errors="coerce").astype("float64")
        volume = volume.where(volume > 0.0)
        if _has_meaningful_coverage(volume, len(ordered)):
            return volume

    return pd.Series(np.nan, index=ordered.index, dtype="float64")


def _compress(raw: pd.Series) -> pd.Series:
    """Lightly compress a non-negative risk score while preserving NaN."""
    return np.log1p(raw.clip(lower=0.0)).replace([np.inf, -np.inf], np.nan).astype("float64")


def _components(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate reusable causal components in stock/date order."""
    ordered = _ordered_frame(data)
    stocks = ordered["stock_code"]
    close = pd.to_numeric(ordered["close"], errors="coerce").astype("float64")
    high = pd.to_numeric(ordered["high"], errors="coerce").astype("float64")
    low = pd.to_numeric(ordered["low"], errors="coerce").astype("float64")
    prev_close = close.groupby(stocks, sort=False).shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    normalized_true_range = (true_range / prev_close.replace(0.0, np.nan)).clip(0.0, 1.0)

    recent_range_5d = _rolling(normalized_true_range, stocks, 5, 4, "mean")
    # The baseline excludes the recent five sessions, preventing the expansion
    # itself from immediately raising its own comparison base.
    prior_range = normalized_true_range.groupby(stocks, sort=False).shift(5)
    baseline_range_20d = _rolling(prior_range, stocks, 20, 12, "median")
    range_expansion = np.log(
        (recent_range_5d + _EPS) / (baseline_range_20d + _EPS)
    ).clip(-3.0, 3.0)

    close_5d_ago = close.groupby(stocks, sort=False).shift(5)
    close_10d_ago = close.groupby(stocks, sort=False).shift(10)
    return_5d = close / close_5d_ago - 1.0
    return_10d = close / close_10d_ago - 1.0
    recent_runup_peak = _rolling(return_10d.clip(lower=0.0), stocks, 10, 3, "max")

    range_budget_5d = _rolling(normalized_true_range, stocks, 5, 4, "sum")
    directional_efficiency_5d = (
        return_5d.abs() / (range_budget_5d + _EPS)
    ).clip(0.0, 1.0)
    positive_efficiency_5d = (
        return_5d.clip(lower=0.0) / (range_budget_5d + _EPS)
    ).clip(0.0, 1.0)
    relative_range_level_5d = (
        recent_range_5d / (baseline_range_20d + _EPS)
    ).clip(0.0, 10.0)
    # Quiet consolidation can also have low directional efficiency. Multiplying
    # by relative range level reserves high turbulence values for genuinely
    # wide, inefficient travel rather than any sideways price action.
    path_turbulence_5d = (1.0 - directional_efficiency_5d) * relative_range_level_5d
    signed_path_turbulence_5d = (1.0 - positive_efficiency_5d) * relative_range_level_5d

    activity = _activity_proxy(ordered, close)
    recent_activity_5d = _rolling(activity, stocks, 5, 4, "median")
    prior_activity = activity.groupby(stocks, sort=False).shift(5)
    baseline_activity_20d = _rolling(prior_activity, stocks, 20, 12, "median")
    activity_expansion = np.log(
        (recent_activity_5d + _EPS) / (baseline_activity_20d + _EPS)
    ).clip(-3.0, 3.0)

    intraday_range = (high - low).where((high - low) > _EPS)
    close_location = ((close - low) / intraday_range).clip(0.0, 1.0)
    high_to_close_drawdown = ((high - close) / close.replace(0.0, np.nan)).clip(0.0, 1.0)
    high_to_close_tr_share = ((high - close) / (true_range + _EPS)).clip(0.0, 1.0)
    rejection_parts = [1.0 - close_location, high_to_close_tr_share]
    if "open" in ordered.columns:
        open_price = pd.to_numeric(ordered["open"], errors="coerce").astype("float64")
        upper_shadow_ratio = (
            (high - pd.concat([open_price, close], axis=1).max(axis=1)) / intraday_range
        ).clip(0.0, 1.0)
        rejection_parts.append(upper_shadow_ratio)
    rejection_pressure = pd.concat(rejection_parts, axis=1).mean(axis=1, skipna=True)
    # Retain a price-scaled drawdown contribution so large absolute rejection
    # remains visible even on exceptionally wide-range sessions.
    rejection_pressure = (
        0.75 * rejection_pressure + 0.25 * (high_to_close_drawdown * 10.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    recent_rejection_5d = _rolling(rejection_pressure, stocks, 5, 4, "mean")
    prior_rejection = rejection_pressure.groupby(stocks, sort=False).shift(5)
    baseline_rejection_20d = _rolling(prior_rejection, stocks, 20, 12, "median")
    rejection_increase = (recent_rejection_5d - baseline_rejection_20d).clip(lower=0.0)
    rejection_ratio_increase = np.log(
        (recent_rejection_5d + 0.05) / (baseline_rejection_20d + 0.05)
    ).clip(lower=0.0, upper=3.0)
    rejection_deterioration = np.sqrt(
        (rejection_increase / 0.10).clip(0.0, 3.0) * rejection_ratio_increase
    ).clip(0.0, 3.0)
    # The factor should react mainly to a recent deterioration in rejection,
    # not to a stock whose normal trading character always includes shadows.
    rejection_confirmation = (
        0.90 * rejection_deterioration + 0.10 * recent_rejection_5d
    ).clip(0.0, 3.0)

    high_20d = _rolling(high, stocks, 20, 12, "max")
    low_20d = _rolling(low, stocks, 20, 12, "min")
    low_close_20d = _rolling(close, stocks, 20, 12, "min")
    high_level_position = (
        (close - low_20d) / (high_20d - low_20d + _EPS)
    ).clip(0.0, 1.0)
    retained_runup_20d = (close / (low_close_20d + _EPS) - 1.0).clip(lower=0.0)
    # Retained run-up prevents the signal from disappearing immediately when a
    # strong advance transitions into several days of high-level sideways churn.
    runup_strength = pd.concat(
        [
            return_10d.clip(lower=0.0),
            retained_runup_20d * high_level_position,
            recent_runup_peak,
        ],
        axis=1,
    ).max(axis=1).clip(0.0, 1.5)

    return pd.DataFrame(
        {
            "runup_strength": runup_strength,
            "range_expansion": range_expansion,
            "path_turbulence_5d": path_turbulence_5d,
            "signed_path_turbulence_5d": signed_path_turbulence_5d,
            "activity_expansion": activity_expansion,
            "rejection_confirmation": rejection_confirmation,
            "high_level_position": high_level_position,
        },
        index=ordered.index,
    ).reindex(data.index)


def _runup_range_expansion_10d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    Retained recent run-up multiplied by positive short-range expansion.

    A high value means the stock has risen, while its last-five-day normalized
    true range is now materially larger than its own preceding 20-day baseline.
    """
    c = _components(data)
    return _compress(c["runup_strength"] * c["range_expansion"].clip(lower=0.0))


def _runup_path_turbulence_10d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    Retained recent run-up multiplied by recent directional inefficiency.

    Unlike close-only efficiency_ratio, the path budget includes intraday true
    range, so wide daily oscillations are penalized even if closes look smooth.
    """
    c = _components(data)
    return _compress(c["runup_strength"] * c["path_turbulence_5d"])


def _runup_signed_path_turbulence_10d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    Run-up followed by non-positive or inefficient recent progress.

    Positive efficiency gives no credit to a fast decline.  Therefore both
    wide sideways churn and a high-level selloff remain visible as risk.
    """
    c = _components(data)
    return _compress(c["runup_strength"] * c["signed_path_turbulence_5d"])


def _runup_volume_friction_10d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """Run-up path friction confirmed by expanding price-neutral activity."""
    c = _components(data)
    raw = (
        c["runup_strength"]
        * c["path_turbulence_5d"]
        * c["activity_expansion"].clip(lower=0.0)
    )
    return _compress(raw)


def _runup_upper_shadow_turbulence_10d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """Run-up turbulence confirmed mainly by recently worsening rejection."""
    c = _components(data)
    raw = (
        c["runup_strength"]
        * c["signed_path_turbulence_5d"]
        * c["rejection_confirmation"]
    )
    return _compress(raw)


def _high_level_churn_risk_10d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    Joint late-stage churn risk.

    It becomes large only when a prior run-up, range expansion, inefficient
    price travel, and a still-high price location coexist.  The geometric mean
    keeps one extreme component from dominating the whole signal.
    """
    c = _components(data)
    expansion_leg = c["runup_strength"] * c["range_expansion"].clip(lower=0.0)
    turbulence_leg = c["runup_strength"] * c["signed_path_turbulence_5d"]
    result = np.sqrt(expansion_leg.clip(lower=0.0) * turbulence_leg.clip(lower=0.0))
    result = result * c["high_level_position"]
    return _compress(result)


def _target(
    calculator,
    data: pd.DataFrame,
    target_index: pd.Index,
    context: FactorContext,
) -> pd.Series:
    return select_target_series(calculator(data, context), target_index=target_index)


runup_range_expansion_10d = register_factor(
    FactorDefinition(
        name="runup_range_expansion_10d",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=40,
        calculator=_runup_range_expansion_10d,
        calculation_scope="target_with_history_window",
        target_calculator=lambda d, ti, c: _target(_runup_range_expansion_10d, d, ti, c),
        description="上涨后振幅扩张：10/20日保留涨幅×近5日真实波幅相对前置20日基线的正向扩张",
    )
)


runup_path_turbulence_10d = register_factor(
    FactorDefinition(
        name="runup_path_turbulence_10d",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=25,
        calculator=_runup_path_turbulence_10d,
        calculation_scope="target_with_history_window",
        target_calculator=lambda d, ti, c: _target(_runup_path_turbulence_10d, d, ti, c),
        description="上涨路径震荡度：10/20日保留涨幅×近5日基于真实波幅的方向低效率",
    )
)


runup_signed_path_turbulence_10d = register_factor(
    FactorDefinition(
        name="runup_signed_path_turbulence_10d",
        required_columns={"stock_code", "date", "high", "low", "close"},
        optional_columns={"open", "amount", "volume", "turnover_rate"},
        lookback_days=40,
        calculator=_runup_signed_path_turbulence_10d,
        calculation_scope="target_with_history_window",
        target_calculator=lambda d, ti, c: _target(_runup_signed_path_turbulence_10d, d, ti, c),
        description="上涨后有符号路径震荡风险：宽幅横盘和高位下跌均被视为低效风险",
    )
)


runup_volume_friction_10d = register_factor(
    FactorDefinition(
        name="runup_volume_friction_10d",
        required_columns={"stock_code", "date", "high", "low", "close"},
        optional_columns={"amount", "volume", "turnover_rate"},
        lookback_days=40,
        calculator=_runup_volume_friction_10d,
        calculation_scope="target_with_history_window",
        target_calculator=lambda d, ti, c: _target(_runup_volume_friction_10d, d, ti, c),
        description="上涨后量能摩擦：路径低效×价格中性成交活跃度扩张，优先换手率，其次amount/close、volume",
    )
)


runup_upper_shadow_turbulence_10d = register_factor(
    FactorDefinition(
        name="runup_upper_shadow_turbulence_10d",
        required_columns={"stock_code", "date", "high", "low", "close"},
        optional_columns={"open"},
        lookback_days=40,
        calculator=_runup_upper_shadow_turbulence_10d,
        calculation_scope="target_with_history_window",
        target_calculator=lambda d, ti, c: _target(_runup_upper_shadow_turbulence_10d, d, ti, c),
        description="上涨后冲高回落震荡：有符号路径低效×近5日上影及弱收盘压力相对前置20日的恶化",
    )
)


high_level_churn_risk_10d = register_factor(
    FactorDefinition(
        name="high_level_churn_risk_10d",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=40,
        calculator=_high_level_churn_risk_10d,
        calculation_scope="target_with_history_window",
        target_calculator=lambda d, ti, c: _target(_high_level_churn_risk_10d, d, ti, c),
        description="高位宽幅震荡风险：上涨、振幅扩张、路径低效与高位滞留的联合连续信号",
    )
)


__all__ = [
    "runup_range_expansion_10d",
    "runup_path_turbulence_10d",
    "runup_signed_path_turbulence_10d",
    "runup_volume_friction_10d",
    "runup_upper_shadow_turbulence_10d",
    "high_level_churn_risk_10d",
]
