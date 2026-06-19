"""
监管异动风险因子。

核心思想：
1. 用个股涨跌幅相对市场基准涨跌幅近似“偏离值”。
2. 按股票代码前缀区分主板 / 创业板 / 科创板阈值。
3. 输出尽量保持原始风险比例，供模型或规则直接使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


def _get_market_change_map(ctx: FactorContext) -> pd.Series:
    market = ctx.require_market().sort_values("date").reset_index(drop=True)
    if "daily_change_pct" in market.columns:
        series = pd.to_numeric(market["daily_change_pct"], errors="coerce")
    elif "pct_chg" in market.columns:
        series = pd.to_numeric(market["pct_chg"], errors="coerce")
    else:
        raise ValueError("market_data 缺少 daily_change_pct/pct_chg，无法计算异动偏离值")
    return pd.Series(series.values, index=pd.to_datetime(market["date"]), dtype="float64")


def _board_threshold_3d(stock_code: pd.Series) -> pd.Series:
    code = stock_code.fillna("").astype(str)
    growth_mask = code.str.startswith(("30", "68"))
    threshold = pd.Series(20.0, index=stock_code.index, dtype="float64")
    threshold.loc[growth_mask] = 30.0
    return threshold


def _prepare_deviation_df(data: pd.DataFrame, ctx: FactorContext) -> pd.DataFrame:
    market_map = _get_market_change_map(ctx)
    df = data[["date", "stock_code", "pct_chg"]].copy()
    df["__orig_idx__"] = data.index
    df["date"] = pd.to_datetime(df["date"])
    df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
    df["market_pct_chg"] = df["date"].map(market_map)
    df["deviation"] = df["pct_chg"] - df["market_pct_chg"]
    df["board_thr_3d"] = _board_threshold_3d(df["stock_code"])
    return df.sort_values(["stock_code", "date", "__orig_idx__"]).reset_index(drop=True)


def _rolling_sum_by_stock(series: pd.Series, stock_code: pd.Series, window: int, min_periods: int) -> pd.Series:
    temp = pd.DataFrame({"v": series.values, "stock_code": stock_code.values}, index=series.index)
    return (
        temp.groupby("stock_code", sort=False)["v"]
        .rolling(window, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
        .astype("float64")
    )


def _abnormal_deviation_3d_ratio(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    df = _prepare_deviation_df(data, ctx)
    cum3 = _rolling_sum_by_stock(df["deviation"], df["stock_code"], window=3, min_periods=3)
    ratio = cum3.abs() / (df["board_thr_3d"] + 1e-9)
    out = pd.Series(ratio.values, index=df["__orig_idx__"].values, dtype="float64")
    return out.reindex(data.index).astype("float64")


def _abnormal_event_count_10d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    df = _prepare_deviation_df(data, ctx)
    cum3 = _rolling_sum_by_stock(df["deviation"], df["stock_code"], window=3, min_periods=3)
    trigger = (cum3.abs() >= (df["board_thr_3d"] - 1e-9)).astype("float64")
    count_10d = _rolling_sum_by_stock(trigger, df["stock_code"], window=10, min_periods=1)
    out = pd.Series(count_10d.values, index=df["__orig_idx__"].values, dtype="float64")
    return out.reindex(data.index).astype("float64")


def _severe_abnormal_pressure_10d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    df = _prepare_deviation_df(data, ctx)
    cum10 = _rolling_sum_by_stock(df["deviation"], df["stock_code"], window=10, min_periods=5)
    up_ratio = cum10 / 100.0
    down_ratio = (-cum10) / 50.0
    pressure = np.maximum(up_ratio, down_ratio)
    pressure = np.maximum(pressure, 0.0)
    out = pd.Series(pressure, index=df["__orig_idx__"].values, dtype="float64")
    return out.reindex(data.index).astype("float64")


def _severe_abnormal_pressure_30d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    df = _prepare_deviation_df(data, ctx)
    cum30 = _rolling_sum_by_stock(df["deviation"], df["stock_code"], window=30, min_periods=10)
    up_ratio = cum30 / 200.0
    down_ratio = (-cum30) / 70.0
    pressure = np.maximum(up_ratio, down_ratio)
    pressure = np.maximum(pressure, 0.0)
    out = pd.Series(pressure, index=df["__orig_idx__"].values, dtype="float64")
    return out.reindex(data.index).astype("float64")


abnormal_deviation_3d_ratio = register_factor(
    FactorDefinition(
        name="abnormal_deviation_3d_ratio",
        required_columns={"date", "stock_code", "pct_chg"},
        requires_market_data=True,
        lookback_days=5,
        calculator=_abnormal_deviation_3d_ratio,
        description="近3日偏离值累计相对异动阈值的比例，主板20%/创业板科创板30%",
    )
)


abnormal_event_count_10d = register_factor(
    FactorDefinition(
        name="abnormal_event_count_10d",
        required_columns={"date", "stock_code", "pct_chg"},
        requires_market_data=True,
        lookback_days=15,
        calculator=_abnormal_event_count_10d,
        description="近10日内触发3日异常波动条件的次数",
    )
)


severe_abnormal_pressure_10d = register_factor(
    FactorDefinition(
        name="severe_abnormal_pressure_10d",
        required_columns={"date", "stock_code", "pct_chg"},
        requires_market_data=True,
        lookback_days=15,
        calculator=_severe_abnormal_pressure_10d,
        description="近10日偏离值累计相对严重异常波动阈值的压力比例",
    )
)


severe_abnormal_pressure_30d = register_factor(
    FactorDefinition(
        name="severe_abnormal_pressure_30d",
        required_columns={"date", "stock_code", "pct_chg"},
        requires_market_data=True,
        lookback_days=35,
        calculator=_severe_abnormal_pressure_30d,
        description="近30日偏离值累计相对严重异常波动阈值的压力比例",
    )
)


__all__ = [
    "abnormal_deviation_3d_ratio",
    "abnormal_event_count_10d",
    "severe_abnormal_pressure_10d",
    "severe_abnormal_pressure_30d",
]
