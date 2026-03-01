"""
统计矩因子（偏度、峰度、高阶风险交互）- FrozenMean 双路版

核心关注：收益率分布的非正态性（肥尾、偏斜）以及与市场尾部的协同风险。

本次关键修复：
1) coskewness：原实现使用 market_ret.mean()（全局均值），若你全量生成因子 => 含 test 期信息（弱泄露/漂移源）。
   => 改为 FrozenMean：只用训练期日期范围拟合 market_mean，并冻结用于全量计算。
2) coskewness：原实现分子 num 的 rolling 未按 stock 分组，会串线（严重不稳定源）。
   => 分子/分母全部改为按 stock_code 分组 rolling。
3) 输出双路：
   - coskewness_raw   : 全样本均值 center（对照/复现旧版）
   - coskewness_gated : FrozenMean center（推荐用于现在的“按日生成”一致性）
   - coskewness       : 默认指向 coskewness_gated
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


# ==============================================================================
# 0. 工具：市场数据 & 稳健 Z
# ==============================================================================
def _get_market_data(ctx: FactorContext) -> pd.DataFrame:
    """获取对齐后的市场数据"""
    market = ctx.require_market()
    return market.sort_values("date").reset_index(drop=True)


def _to_ts(date_series: pd.Series) -> pd.Series:
    """稳健把 date 转为 Timestamp（支持 'YYYY-MM-DD' / 'YYYYMMDD' / int）"""
    if date_series.dtype == "datetime64[ns]":
        return date_series
    ds = date_series.astype(str)
    ds = ds.str.replace("-", "", regex=False)
    return pd.to_datetime(ds, format="%Y%m%d", errors="coerce")


def _global_robust_zscore(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """
    全局按日稳健 Z-Score (MAD)，不 fillna(0)，保留 NaN
    """
    df = pd.DataFrame({"val": series.values, "date": date_series.values}, index=series.index)

    medians = df.groupby("date")["val"].transform("median")
    abs_diff = (df["val"] - medians).abs()

    temp_diff = pd.DataFrame({"diff": abs_diff.values, "date": date_series.values}, index=series.index)
    mads = temp_diff.groupby("date")["diff"].transform("median")

    robust_std = mads * 1.4826
    z = (df["val"] - medians) / (robust_std + 1e-9)

    mask_zero_mad = (mads < 1e-6)
    if mask_zero_mad.any():
        means = df.groupby("date")["val"].transform("mean")
        stds = df.groupby("date")["val"].transform("std")
        z_fallback = (df["val"] - means) / (stds + 1e-9)
        z = z.where(~mask_zero_mad, z_fallback)

    return z.clip(-6.0, 6.0).astype("float64")


# ==============================================================================
# 1. Skewness / Kurtosis（保持你原逻辑）
# ==============================================================================
def _skewness_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    grouped = data.groupby("stock_code")["pct_chg"]
    raw = grouped.transform(lambda x: x.rolling(20, min_periods=15).skew())
    return _global_robust_zscore(raw, data["date"])


def _kurtosis_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    grouped = data.groupby("stock_code")["pct_chg"]
    raw = grouped.transform(lambda x: x.rolling(20, min_periods=15).kurt())
    return _global_robust_zscore(raw, data["date"])


# ==============================================================================
# 2. Coskewness（双路：raw 全局均值 / gated 冻结均值）
# ==============================================================================
class _FrozenMarketMean:
    """
    只用训练期拟合市场均值，并冻结。
    这样你即使全量一次性生成因子，也不会把 val/test 的信息注入到 market_centering 中。
    """
    TRAIN_START = pd.Timestamp("2023-06-01")
    TRAIN_END = pd.Timestamp("2025-03-01")

    def __init__(self, market_df: pd.DataFrame):
        self.market_df = market_df.copy()
        self.market_df["_ts"] = _to_ts(self.market_df["date"])

        train = self.market_df[
            (self.market_df["_ts"] >= self.TRAIN_START) & (self.market_df["_ts"] <= self.TRAIN_END)
        ]
        s = pd.to_numeric(train.get("daily_change_pct"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        self.mean_ = float(s.mean()) if not s.empty else 0.0

    def centered_sq_series(self) -> pd.Series:
        m = self.market_df.set_index("date")["daily_change_pct"]
        if m.index.duplicated().any():
            m = m[~m.index.duplicated(keep="first")]
        m = pd.to_numeric(m, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return (m - self.mean_) ** 2


def _coskewness_core(data: pd.DataFrame, ctx: FactorContext, mode: str) -> pd.Series:
    """
    mode:
      - 'raw'   : 使用全样本市场均值
      - 'gated' : 使用训练期冻结市场均值（推荐）
    """
    market = _get_market_data(ctx)
    if "daily_change_pct" not in market.columns or "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    # --- 构造 market 的 m_sq(date) ---
    if mode == "raw":
        m = market.set_index("date")["daily_change_pct"]
        if m.index.duplicated().any():
            m = m[~m.index.duplicated(keep="first")]
        m = pd.to_numeric(m, errors="coerce").replace([np.inf, -np.inf], np.nan)
        m_sq = (m - float(m.mean(skipna=True))) ** 2
    elif mode == "gated":
        m_sq = _FrozenMarketMean(market).centered_sq_series()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- 对齐到个股样本 ---
    df = pd.DataFrame(
        {
            "date": data["date"].values,
            "stock": data["stock_code"].values,
            "r_i": pd.to_numeric(data["pct_chg"], errors="coerce").values,
        },
        index=data.index,
    )

    # 映射 m_sq 到每条样本（市场平方偏离）
    df["m_sq"] = df["date"].map(m_sq)

    # 排序保证 rolling 因果
    df_sorted = df.sort_values(["stock", "date"])
    g = df_sorted.groupby("stock", sort=False)

    # 个股去均值（rolling mean）
    i_mean = g["r_i"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    i_centered = df_sorted["r_i"] - i_mean

    # 分子：E[(r_i - mean_i) * (m_sq)]  —— 必须按 stock 分组 rolling
    num = g.apply(
        lambda sub: (sub["r_i"] - sub["r_i"].rolling(20, min_periods=10).mean()) * sub["m_sq"]
    ).reset_index(level=0, drop=True)
    num = num.groupby(df_sorted["stock"]).transform(lambda x: x.rolling(20, min_periods=10).mean())

    # 分母：Std(r_i_centered) * Std(m_sq) —— 均按 stock 分组 rolling（m_sq 是按 date 映射，但 rolling 要在 stock 序列上做）
    i_std = g.apply(
        lambda sub: (sub["r_i"] - sub["r_i"].rolling(20, min_periods=10).mean()).rolling(20, min_periods=10).std()
    ).reset_index(level=0, drop=True)

    m_sq_std = g["m_sq"].transform(lambda x: x.rolling(20, min_periods=10).std())

    raw_val = num / (i_std * m_sq_std + 1e-9)
    raw_val = raw_val.replace([np.inf, -np.inf], np.nan)

    # 恢复原 index 顺序
    raw_series = raw_val.reindex(data.index)

    # 截面稳健标准化
    return _global_robust_zscore(raw_series, data["date"])


def _coskewness_raw(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _coskewness_core(data, ctx, mode="raw")


def _coskewness_gated(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _coskewness_core(data, ctx, mode="gated")


# ==============================================================================
# 3. Downside Beta（保持你原逻辑，但做一点稳健清洗）
# ==============================================================================
def _downside_beta(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    market = _get_market_data(ctx)
    if "daily_change_pct" not in market.columns or "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    mkt_down = market[market["daily_change_pct"] < 0].set_index("date")["daily_change_pct"]
    if mkt_down.index.duplicated().any():
        mkt_down = mkt_down[~mkt_down.index.duplicated(keep="first")]
    mkt_down = pd.to_numeric(mkt_down, errors="coerce").replace([np.inf, -np.inf], np.nan)

    df = data[["date", "pct_chg", "stock_code"]].copy()
    df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
    df["mkt_ret"] = df["date"].map(mkt_down)

    g = df.groupby("stock_code", sort=False)
    cov = (
        g[["pct_chg", "mkt_ret"]]
        .rolling(30, min_periods=10)
        .cov()
        .unstack()
        .iloc[:, 1]
        .reset_index(level=0, drop=True)
    )
    cov = cov.replace([np.inf, -np.inf], np.nan)

    return _global_robust_zscore(cov, data["date"])


# ==============================================================================
# 4. 注册因子
# ==============================================================================
skewness_20d = register_factor(
    FactorDefinition(
        name="skewness_20d",
        required_columns={"stock_code", "date", "pct_chg"},
        lookback_days=30,
        calculator=_skewness_20d,
        description="20日收益率偏度（暴涨暴跌的不对称性）",
    )
)

kurtosis_20d = register_factor(
    FactorDefinition(
        name="kurtosis_20d",
        required_columns={"stock_code", "date", "pct_chg"},
        lookback_days=30,
        calculator=_kurtosis_20d,
        description="20日收益率峰度（肥尾风险/极端波动概率）",
    )
)

coskewness_raw = register_factor(
    FactorDefinition(
        name="coskewness_raw",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_market_data=True,
        lookback_days=30,
        calculator=_coskewness_raw,
        description="协同偏度 raw：全样本市场均值中心化（对照用）",
    )
)

coskewness_gated = register_factor(
    FactorDefinition(
        name="coskewness_gated",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_market_data=True,
        lookback_days=30,
        calculator=_coskewness_gated,
        description="协同偏度 gated：训练期冻结市场均值中心化（推荐/更稳）",
    )
)

coskewness = register_factor(
    FactorDefinition(
        name="coskewness",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_market_data=True,
        lookback_days=30,
        calculator=_coskewness_gated,
        description="协同偏度 gated：训练期冻结市场均值中心化（推荐/更稳）",
    )
)

# 默认 coskewness 指向更稳的版本
# coskewness = coskewness_gated

downside_beta = register_factor(
    FactorDefinition(
        name="downside_beta",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_market_data=True,
        lookback_days=40,
        calculator=_downside_beta,
        description="下行贝塔（下跌市场中的协方差，衡量抗跌性）",
    )
)

__all__ = [
    "skewness_20d",
    "kurtosis_20d",
    "coskewness",
    "coskewness_raw",
    "coskewness_gated",
    "downside_beta",
]
