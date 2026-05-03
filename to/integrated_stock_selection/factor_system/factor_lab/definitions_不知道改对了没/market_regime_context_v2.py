"""
增强市场环境因子 v2。

设计原则：
1. 仅使用当前项目中已经稳定存在的字段。
2. 因子按交易日广播给所有股票，专门服务树模型的 regime gating。
3. 不替换旧 market_context.py，作为 challenger 的增量因子集使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


def _get_market_data_safe(ctx: FactorContext) -> pd.DataFrame:
    market = ctx.require_market().copy()
    market = market.sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market["date"])
    return market


def _broadcast_daily_factor(data: pd.DataFrame, daily_series: pd.Series, fill_value=np.nan) -> pd.Series:
    aligned = daily_series.copy()
    aligned.index = pd.to_datetime(aligned.index)
    out = pd.to_datetime(data["date"]).map(aligned)
    if fill_value is not np.nan:
        out = out.fillna(fill_value)
    return out.astype("float64")


def _mkt_regime_trend_pressure_v2(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    market = _get_market_data_safe(ctx)
    close = pd.to_numeric(market["close"], errors="coerce")
    ret = pd.to_numeric(market["daily_change_pct"], errors="coerce")

    ma10 = close.rolling(10, min_periods=5).mean()
    ma30 = close.rolling(30, min_periods=15).mean()
    vol20 = ret.rolling(20, min_periods=10).std()
    trend_pressure = ((ma10 - ma30) / (ma30 + 1e-9)) / (vol20 + 1e-6)
    trend_pressure.index = market["date"]
    return _broadcast_daily_factor(data, trend_pressure)


def _mkt_breadth_thrust_v2(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    daily_pos = data.groupby("date")["pct_chg"].apply(lambda x: (x > 0).sum() / max(len(x), 1))
    breadth_5 = daily_pos.rolling(5, min_periods=3).mean()
    breadth_20 = daily_pos.rolling(20, min_periods=10).mean()
    thrust = breadth_5 - breadth_20
    return _broadcast_daily_factor(data, thrust)


def _mkt_emotion_stability_v2(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    work = data[["date"]].copy()
    for col in ["is_limit_up", "is_limit_down", "is_zhaban"]:
        if col not in data.columns:
            work[col] = 0
        else:
            work[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    daily = work.groupby("date").agg(
        limit_up=("is_limit_up", "sum"),
        limit_down=("is_limit_down", "sum"),
        zhaban=("is_zhaban", "sum"),
        total=("date", "count"),
    )
    heat = (daily["limit_up"] - daily["limit_down"]) / (daily["total"] + 1e-9)
    stability = 1.0 - daily["zhaban"] / (daily["limit_up"] + daily["zhaban"] + 1e-9)
    emotion = (heat.rolling(3, min_periods=1).mean() * stability.rolling(3, min_periods=1).mean())
    return _broadcast_daily_factor(data, emotion)


def _mkt_dispersion_shock_v2(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    daily_std = data.groupby("date")["pct_chg"].std()
    shock = daily_std / (daily_std.rolling(20, min_periods=10).mean() + 1e-9)
    return _broadcast_daily_factor(data, shock)


def _mkt_liquidity_impulse_v2(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    market = _get_market_data_safe(ctx)
    if "amount" in market.columns:
        amt = pd.to_numeric(market["amount"], errors="coerce")
    elif "volume" in market.columns:
        amt = pd.to_numeric(market["volume"], errors="coerce")
    else:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    impulse = amt / (amt.rolling(20, min_periods=10).mean() + 1e-9)
    impulse.index = market["date"]
    return _broadcast_daily_factor(data, impulse)


mkt_regime_trend_pressure_v2 = register_factor(
    FactorDefinition(
        name="mkt_regime_trend_pressure_v2",
        required_columns={"date"},
        requires_market_data=True,
        lookback_days=40,
        calculator=_mkt_regime_trend_pressure_v2,
        description="市场趋势压力 v2（短中期均线趋势 / 波动率）",
    )
)

mkt_breadth_thrust_v2 = register_factor(
    FactorDefinition(
        name="mkt_breadth_thrust_v2",
        required_columns={"date", "pct_chg"},
        lookback_days=25,
        calculator=_mkt_breadth_thrust_v2,
        description="市场广度脉冲 v2（5日广度减20日广度）",
    )
)

mkt_emotion_stability_v2 = register_factor(
    FactorDefinition(
        name="mkt_emotion_stability_v2",
        required_columns={"date"},
        optional_columns={"is_limit_up", "is_limit_down", "is_zhaban"},
        lookback_days=5,
        calculator=_mkt_emotion_stability_v2,
        description="市场情绪稳定度 v2（涨跌停热度与炸板稳定性）",
    )
)

mkt_dispersion_shock_v2 = register_factor(
    FactorDefinition(
        name="mkt_dispersion_shock_v2",
        required_columns={"date", "pct_chg"},
        lookback_days=25,
        calculator=_mkt_dispersion_shock_v2,
        description="市场离散度冲击 v2（横截面波动相对20日均值）",
    )
)

mkt_liquidity_impulse_v2 = register_factor(
    FactorDefinition(
        name="mkt_liquidity_impulse_v2",
        required_columns={"date"},
        requires_market_data=True,
        lookback_days=25,
        calculator=_mkt_liquidity_impulse_v2,
        description="市场流动性脉冲 v2（成交额相对20日均值）",
    )
)


__all__ = [
    "mkt_regime_trend_pressure_v2",
    "mkt_breadth_thrust_v2",
    "mkt_emotion_stability_v2",
    "mkt_dispersion_shock_v2",
    "mkt_liquidity_impulse_v2",
]
