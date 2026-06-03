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
from ..target_utils import select_target_series


_REG_BENCHMARK_CODES = {
    "sh_main": "000002.SH",
    "sz_main": "399107.SZ",
    "chinext": "399102.SZ",
    "star": "000688.SH",
}


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


def _normalize_index_code(code: object) -> str:
    value = str(code).strip().upper()
    if "." in value:
        return value
    pure_code = value.zfill(6) if value.isdigit() else value
    if pure_code.startswith("399"):
        return f"{pure_code}.SZ"
    if pure_code.startswith(("000", "880")):
        return f"{pure_code}.SH"
    return pure_code


def _reg_benchmark_code(stock_code: pd.Series) -> pd.Series:
    code = stock_code.fillna("").astype(str).str.split(".", n=1).str[0]
    benchmark = pd.Series(np.nan, index=stock_code.index, dtype="object")
    benchmark.loc[code.str.startswith(("600", "601", "603", "605"))] = _REG_BENCHMARK_CODES["sh_main"]
    benchmark.loc[code.str.startswith(("000", "001", "002", "003"))] = _REG_BENCHMARK_CODES["sz_main"]
    benchmark.loc[code.str.startswith(("300", "301"))] = _REG_BENCHMARK_CODES["chinext"]
    benchmark.loc[code.str.startswith(("688", "689"))] = _REG_BENCHMARK_CODES["star"]
    return benchmark


def _get_regulatory_index_data(ctx: FactorContext) -> pd.DataFrame:
    index_data = ctx.extra.get("regulatory_index_data") if ctx.extra else None
    if index_data is None or not isinstance(index_data, pd.DataFrame) or index_data.empty:
        raise ValueError(
            "reg_severe_up_10d_ratio 需要 context_extra['regulatory_index_data']，"
            "请先运行 scripts/download_regulatory_indices.py 下载监管基准指数。"
        )

    df = index_data.copy()
    rename_map = {}
    if "trade_date" in df.columns and "date" not in df.columns:
        rename_map["trade_date"] = "date"
    if "date_str" in df.columns and "date" not in df.columns:
        rename_map["date_str"] = "date"
    if "ts_code" in df.columns and "index_code" not in df.columns:
        rename_map["ts_code"] = "index_code"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"date", "index_code", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"监管基准指数数据缺少必要列: {sorted(missing)}")

    df = df.loc[:, ["date", "index_code", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["index_code"] = df["index_code"].map(_normalize_index_code)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "index_code", "close"])
    df = df.sort_values(["index_code", "date"]).drop_duplicates(["index_code", "date"], keep="last")
    return df.reset_index(drop=True)


def _abnormal_deviation_3d_ratio(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    df = _prepare_deviation_df(data, ctx)
    cum3 = _rolling_sum_by_stock(df["deviation"], df["stock_code"], window=3, min_periods=3)
    ratio = cum3.abs() / (df["board_thr_3d"] + 1e-9)
    out = pd.Series(ratio.values, index=df["__orig_idx__"].values, dtype="float64")
    return out.reindex(data.index).astype("float64")


def _abnormal_deviation_3d_ratio_target(data: pd.DataFrame, target_index: pd.Index, ctx: FactorContext) -> pd.Series:
    return select_target_series(_abnormal_deviation_3d_ratio(data, ctx), target_index=target_index)


def _abnormal_event_count_10d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    df = _prepare_deviation_df(data, ctx)
    cum3 = _rolling_sum_by_stock(df["deviation"], df["stock_code"], window=3, min_periods=3)
    trigger = (cum3.abs() >= (df["board_thr_3d"] - 1e-9)).astype("float64")
    count_10d = _rolling_sum_by_stock(trigger, df["stock_code"], window=10, min_periods=1)
    out = pd.Series(count_10d.values, index=df["__orig_idx__"].values, dtype="float64")
    return out.reindex(data.index).astype("float64")


def _abnormal_event_count_10d_target(data: pd.DataFrame, target_index: pd.Index, ctx: FactorContext) -> pd.Series:
    return select_target_series(_abnormal_event_count_10d(data, ctx), target_index=target_index)


def _severe_abnormal_pressure_10d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    df = _prepare_deviation_df(data, ctx)
    cum10 = _rolling_sum_by_stock(df["deviation"], df["stock_code"], window=10, min_periods=5)
    up_ratio = cum10 / 100.0
    down_ratio = (-cum10) / 50.0
    pressure = np.maximum(up_ratio, down_ratio)
    pressure = np.maximum(pressure, 0.0)
    out = pd.Series(pressure, index=df["__orig_idx__"].values, dtype="float64")
    return out.reindex(data.index).astype("float64")


def _severe_abnormal_pressure_10d_target(data: pd.DataFrame, target_index: pd.Index, ctx: FactorContext) -> pd.Series:
    return select_target_series(_severe_abnormal_pressure_10d(data, ctx), target_index=target_index)


def _reg_severe_up_10d_ratio(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    index_df = _get_regulatory_index_data(ctx)

    df = data[["date", "stock_code", "close"]].copy()
    df["__orig_idx__"] = data.index
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values(["stock_code", "date", "__orig_idx__"]).reset_index(drop=True)
    df["benchmark_code"] = _reg_benchmark_code(df["stock_code"])
    df["stock_ret_10d_pct"] = (
        df.groupby("stock_code", sort=False)["close"]
        .transform(lambda close: (close / close.shift(10) - 1.0) * 100.0)
    )

    index_df["index_ret_10d_pct"] = (
        index_df.groupby("index_code", sort=False)["close"]
        .transform(lambda close: (close / close.shift(10) - 1.0) * 100.0)
    )
    index_ret = index_df.loc[:, ["date", "index_code", "index_ret_10d_pct"]]
    merged = df.merge(
        index_ret,
        left_on=["date", "benchmark_code"],
        right_on=["date", "index_code"],
        how="left",
        sort=False,
    )

    deviation = merged["stock_ret_10d_pct"] - merged["index_ret_10d_pct"]
    # pressure = (deviation / 100.0).clip(lower=0.0)
    pressure = (deviation / 100.0)
    out = pd.Series(pressure.values, index=merged["__orig_idx__"].values, dtype="float64")
    return out.reindex(data.index).astype("float64")


def _reg_severe_up_10d_ratio_target(data: pd.DataFrame, target_index: pd.Index, ctx: FactorContext) -> pd.Series:
    return select_target_series(_reg_severe_up_10d_ratio(data, ctx), target_index=target_index)


def _severe_abnormal_pressure_30d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    df = _prepare_deviation_df(data, ctx)
    cum30 = _rolling_sum_by_stock(df["deviation"], df["stock_code"], window=30, min_periods=10)
    up_ratio = cum30 / 200.0
    down_ratio = (-cum30) / 70.0
    pressure = np.maximum(up_ratio, down_ratio)
    pressure = np.maximum(pressure, 0.0)
    out = pd.Series(pressure, index=df["__orig_idx__"].values, dtype="float64")
    return out.reindex(data.index).astype("float64")


def _severe_abnormal_pressure_30d_target(data: pd.DataFrame, target_index: pd.Index, ctx: FactorContext) -> pd.Series:
    return select_target_series(_severe_abnormal_pressure_30d(data, ctx), target_index=target_index)


abnormal_deviation_3d_ratio = register_factor(
    FactorDefinition(
        name="abnormal_deviation_3d_ratio",
        required_columns={"date", "stock_code", "pct_chg"},
        requires_market_data=True,
        lookback_days=5,
        calculator=_abnormal_deviation_3d_ratio,
        calculation_scope="target_with_history_window",
        target_calculator=_abnormal_deviation_3d_ratio_target,
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
        calculation_scope="target_with_history_window",
        target_calculator=_abnormal_event_count_10d_target,
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
        calculation_scope="target_with_history_window",
        target_calculator=_severe_abnormal_pressure_10d_target,
        description="近10日偏离值累计相对严重异常波动阈值的压力比例",
    )
)


reg_severe_up_10d_ratio = register_factor(
    FactorDefinition(
        name="reg_severe_up_10d_ratio",
        required_columns={"date", "stock_code", "close"},
        requires_market_data=False,
        lookback_days=15,
        calculator=_reg_severe_up_10d_ratio,
        calculation_scope="target_with_history_window",
        target_calculator=_reg_severe_up_10d_ratio_target,
        description="近10个交易日上涨偏离值相对+100%严重异常波动阈值的比例，按交易所基准指数复合收益计算",
    )
)


severe_abnormal_pressure_30d = register_factor(
    FactorDefinition(
        name="severe_abnormal_pressure_30d",
        required_columns={"date", "stock_code", "pct_chg"},
        requires_market_data=True,
        lookback_days=35,
        calculator=_severe_abnormal_pressure_30d,
        calculation_scope="target_with_history_window",
        target_calculator=_severe_abnormal_pressure_30d_target,
        description="近30日偏离值累计相对严重异常波动阈值的压力比例",
    )
)


__all__ = [
    "abnormal_deviation_3d_ratio",
    "abnormal_event_count_10d",
    "severe_abnormal_pressure_10d",
    "reg_severe_up_10d_ratio",
    "severe_abnormal_pressure_30d",
]
