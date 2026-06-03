"""
个股资金流因子。

数据来源：
- local_data/moneyflow_ths/YYYYMMDD.parquet

接入方式：
- 主流程按日期范围加载资金流 parquet
- 通过 FactorContext.extra['moneyflow_data'] 注入

设计原则：
- 只保留少量有明确金融含义的连续数值因子
- 不在这里写死任何阈值或规则
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


def _prepare_moneyflow_df(ctx: FactorContext) -> pd.DataFrame:
    moneyflow = (ctx.extra or {}).get("moneyflow_data")
    if moneyflow is None:
        raise ValueError("当前因子需要 moneyflow_data，但未在 context_extra 中提供。")

    df = moneyflow.copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "stock_code"])

    rename_map = {}
    if "trade_date" in df.columns:
        rename_map["trade_date"] = "date"
    if "ts_code" in df.columns:
        rename_map["ts_code"] = "stock_code"
    df = df.rename(columns=rename_map)

    required = {"date", "stock_code"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"moneyflow_data 缺少必要列: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"].astype(str).str.replace("-", "", regex=False), format="%Y%m%d", errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str)

    numeric_cols = [
        "buy_lg_amount_rate",
        "buy_sm_amount_rate",
        "net_amount",
        "net_d5_amount",
        "buy_lg_amount",
        "buy_sm_amount",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values(["stock_code", "date"])
        .drop_duplicates(subset=["stock_code", "date"], keep="last")
        .reset_index(drop=True)
    )
    return df


def _align_moneyflow_series(data: pd.DataFrame, ctx: FactorContext, column: str) -> pd.Series:
    mf = _prepare_moneyflow_df(ctx)
    base = data[["date", "stock_code"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base["stock_code"] = base["stock_code"].astype(str)
    base["__idx__"] = data.index

    if mf.empty or column not in mf.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    merged = base.merge(
        mf[["date", "stock_code", column]],
        on=["date", "stock_code"],
        how="left",
        sort=False,
    )
    series = pd.Series(merged[column].values, index=merged["__idx__"].values, dtype="float64")
    return series.reindex(data.index).astype("float64")


def _buy_lg_amount_rate(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _align_moneyflow_series(data, ctx, "buy_lg_amount_rate")


def _buy_sm_amount_rate(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _align_moneyflow_series(data, ctx, "buy_sm_amount_rate")


def _lg_pos_sm_neg_10d_days(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    近10日大单正、小单负的出现次数。

    数值越大，表示最近一段时间内越多天出现“大资金承接、小资金撤离”的结构。
    """
    mf = _prepare_moneyflow_df(ctx)
    if mf.empty or "buy_lg_amount_rate" not in mf.columns or "buy_sm_amount_rate" not in mf.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    mf = mf.copy()
    mf["lg_pos_sm_neg_flag"] = (
        pd.to_numeric(mf["buy_lg_amount_rate"], errors="coerce").gt(0)
        & pd.to_numeric(mf["buy_sm_amount_rate"], errors="coerce").lt(0)
    ).astype("float64")

    base = data[["date", "stock_code"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base["stock_code"] = base["stock_code"].astype(str)
    base["__idx__"] = data.index

    merged = base.merge(
        mf[["date", "stock_code", "lg_pos_sm_neg_flag"]],
        on=["date", "stock_code"],
        how="left",
        sort=False,
    )
    merged = merged.sort_values(["stock_code", "date"])
    merged["lg_pos_sm_neg_10d_days"] = (
        merged.groupby("stock_code", sort=False)["lg_pos_sm_neg_flag"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    )
    series = pd.Series(merged["lg_pos_sm_neg_10d_days"].values, index=merged["__idx__"].values, dtype="float64")
    return series.reindex(data.index).astype("float64")


def _net_d5_amount_to_turnover(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    net_d5 = _align_moneyflow_series(data, ctx, "net_d5_amount")
    amount = pd.to_numeric(data["amount"], errors="coerce").astype("float64")
    ratio = net_d5 / (amount + 1e-9)
    return ratio.replace([np.inf, -np.inf], np.nan).astype("float64")


def _main_outflow_retail_takeover(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    主力流出-散户接盘强度。

    定义为：
    buy_sm_amount_rate - buy_lg_amount_rate

    数值越大：
    - 小单买入占比越高
    - 大单买入占比越低
    更接近“主力退出、散户承接”的结构。
    """
    small_rate = _align_moneyflow_series(data, ctx, "buy_sm_amount_rate")
    large_rate = _align_moneyflow_series(data, ctx, "buy_lg_amount_rate")
    return (small_rate - large_rate).astype("float64")


buy_lg_amount_rate = register_factor(
    FactorDefinition(
        name="buy_lg_amount_rate",
        required_columns={"date", "stock_code"},
        lookback_days=1,
        calculator=_buy_lg_amount_rate,
        calculation_scope='target_cross_section',
        target_calculator=lambda d, ti, c: _buy_lg_amount_rate(d.loc[ti], c),
        description="资金流大单买入占比，越低越偏主力流出。",
    )
)


buy_sm_amount_rate = register_factor(
    FactorDefinition(
        name="buy_sm_amount_rate",
        required_columns={"date", "stock_code"},
        lookback_days=1,
        calculator=_buy_sm_amount_rate,
        calculation_scope='target_cross_section',
        target_calculator=lambda d, ti, c: _buy_sm_amount_rate(d.loc[ti], c),
        description="资金流小单买入占比，越高越偏散户接盘。",
    )
)


lg_pos_sm_neg_10d_days = register_factor(
    FactorDefinition(
        name="lg_pos_sm_neg_10d_days",
        required_columns={"date", "stock_code"},
        lookback_days=10,
        calculator=_lg_pos_sm_neg_10d_days,
        calculation_scope='target_cross_section',
        target_calculator=lambda d, ti, c: _lg_pos_sm_neg_10d_days(d, c).loc[ti],
        description="近10日大单买入占比为正且小单买入占比为负的天数，衡量资金结构分歧。",
    )
)


net_d5_amount_to_turnover = register_factor(
    FactorDefinition(
        name="net_d5_amount_to_turnover",
        required_columns={"date", "stock_code", "amount"},
        lookback_days=1,
        calculator=_net_d5_amount_to_turnover,
        calculation_scope='target_cross_section',
        target_calculator=lambda d, ti, c: _net_d5_amount_to_turnover(d.loc[ti], c),
        description="5日净流入金额相对当日成交额的比例，衡量持续资金方向。",
    )
)


main_outflow_retail_takeover = register_factor(
    FactorDefinition(
        name="main_outflow_retail_takeover",
        required_columns={"date", "stock_code"},
        lookback_days=1,
        calculator=_main_outflow_retail_takeover,
        calculation_scope='target_cross_section',
        target_calculator=lambda d, ti, c: _main_outflow_retail_takeover(d.loc[ti], c),
        description="小单买入占比减大单买入占比，数值越大越接近主力流出、散户接盘。",
    )
)


__all__ = [
    "buy_lg_amount_rate",
    "buy_sm_amount_rate",
    "lg_pos_sm_neg_10d_days",
    "net_d5_amount_to_turnover",
    "main_outflow_retail_takeover",
]
