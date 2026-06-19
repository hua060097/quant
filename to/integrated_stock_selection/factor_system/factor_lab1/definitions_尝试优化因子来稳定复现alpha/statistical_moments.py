from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


# ==============================================================================
# 配置：coskewness 默认指向 raw 还是 gated
# ==============================================================================
COSKEWNESS_ALIAS: str = "raw"  # "raw" or "gated"


# ==============================================================================
# 0) 基础工具
# ==============================================================================

def _get_market_data(ctx: FactorContext) -> pd.DataFrame:
    """获取市场基准数据（按日期排序）"""
    market = ctx.require_market()
    return market.sort_values("date").reset_index(drop=True)


def _global_robust_zscore(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """
    全局按日稳健 Z-Score (MAD)
    - MAD=0 时 fallback 到 mean/std
    - 常数 clip(-6,6)，无未来泄露
    """
    if series.empty:
        return series.astype("float64")

    df = pd.DataFrame({"val": series.values, "date": date_series.values}, index=series.index)

    med = df.groupby("date")["val"].transform("median")
    abs_dev = (df["val"] - med).abs()
    mad = abs_dev.groupby(df["date"]).transform("median")

    robust_std = mad * 1.4826
    z = (df["val"] - med) / (robust_std + 1e-9)

    mask_zero = mad < 1e-6
    if mask_zero.any():
        means = df.groupby("date")["val"].transform("mean")
        stds = df.groupby("date")["val"].transform("std")
        z_fb = (df["val"] - means) / (stds + 1e-9)
        z = z.where(~mask_zero, z_fb)

    return z.clip(-6.0, 6.0).astype("float64")


def _cs_winsorize_by_date(series: pd.Series, date_series: pd.Series,
                          lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """
    按日横截面 winsor（只用当日截面分布，无未来泄露）
    """
    if series.empty:
        return series.astype("float64")

    df = pd.DataFrame({"v": series.values, "d": date_series.values}, index=series.index)
    lo = df.groupby("d")["v"].transform(lambda x: x.quantile(lower_q))
    hi = df.groupby("d")["v"].transform(lambda x: x.quantile(upper_q))
    out = df["v"].clip(lo, hi)
    return out.astype("float64")


def _market_signal_weak_gate(mkt_ret: pd.Series, vol_window: int = 20) -> pd.Series:
    """
    弱市场信号 gate（(0,1]）
    gate = 1 / (1 + vol_20 / vol_ref)
    - 只用过去 rolling vol 构造，不泄露未来
    - vol_ref 用更长窗中位数，避免 gate 自己抖
    """
    m = mkt_ret.sort_index().astype("float64")
    vol = m.rolling(vol_window, min_periods=10).std()
    vol_ref = vol.rolling(120, min_periods=60).median()
    ratio = vol / (vol_ref + 1e-8)
    gate = 1.0 / (1.0 + ratio)
    return gate.clip(0.1, 1.0).astype("float64")


# ==============================================================================
# 1) Skewness / Kurtosis
# ==============================================================================

def _skewness_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    g = data.groupby("stock_code", sort=False)["pct_chg"]
    raw = g.transform(lambda x: x.rolling(20, min_periods=15).skew())
    raw = raw.replace([np.inf, -np.inf], np.nan)
    return _global_robust_zscore(raw, data["date"])


def _kurtosis_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    g = data.groupby("stock_code", sort=False)["pct_chg"]
    raw = g.transform(lambda x: x.rolling(20, min_periods=15).kurt())
    raw = raw.replace([np.inf, -np.inf], np.nan)
    return _global_robust_zscore(raw, data["date"])


# ==============================================================================
# 2) Coskewness (raw + gated)
# ==============================================================================

def _coskewness_raw(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    Coskewness ~ E[(r_i-μ_i) * (r_m-μ_m)^2] / (σ_i * σ_m^2)

    关键工程点：
    - 市场侧：只在 market 序列上 rolling（不在股票行上 rolling）
    - 个股侧：严格 stock 分组 rolling（修复“跨股票串窗” bug）
    - 处理：先按日 robust z，再按日 winsor（无未来泄露，保尾部但不爆炸）
    """
    if "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    market = _get_market_data(ctx)
    if "daily_change_pct" not in market.columns:
        return pd.Series(np.nan, index=data.index)

    mkt = (
        market.drop_duplicates("date")
        .set_index("date")["daily_change_pct"]
        .astype("float64")
        .sort_index()
    )

    # market centered
    m_mean = mkt.rolling(60, min_periods=30).mean()
    m_c = mkt - m_mean

    m_sq = (m_c ** 2)
    m_std = m_c.rolling(60, min_periods=30).std()
    m_std2 = (m_std ** 2).replace(0, np.nan)

    # map to stock rows
    df = pd.DataFrame(
        {
            "date": data["date"].values,
            "stock": data["stock_code"].values,
            "r_i": data["pct_chg"].astype("float64").values,
        },
        index=data.index,
    )

    df["m_sq"] = df["date"].map(m_sq)
    df["m_std2"] = df["date"].map(m_std2)

    # sort & group
    df = df.sort_values(["stock", "date"])
    g = df.groupby("stock", sort=False)

    # stock centered (rolling mean)
    i_mean = g["r_i"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    i_c = df["r_i"] - i_mean

    # numerator: E[i_c * m_sq] (MUST be stock-group rolling)
    prod = i_c * df["m_sq"]
    num = prod.groupby(df["stock"], sort=False).transform(lambda x: x.rolling(20, min_periods=10).mean())

    # denom: σ_i * σ_m^2
    i_std = g["r_i"].transform(lambda x: x.rolling(20, min_periods=10).std())
    denom = (i_std * df["m_std2"]).clip(lower=1e-6)

    raw = (num / denom).replace([np.inf, -np.inf], np.nan)

    z = _global_robust_zscore(raw, data["date"])
    z = _cs_winsorize_by_date(z, data["date"], 0.01, 0.99)

    # 恢复到原 index 顺序
    z = z.reindex(data.index)
    return z.astype("float64")


def _coskewness_gated(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    gated = raw * weak_market_gate
    gate 越大表示市场越安静/弱信号（更像当年 mkt_ret=0 的 regime）
    - 不改数据，仅在因子层引入 regime 门控，帮助在正确数据上复现旧触发结构
    """
    raw = _coskewness_raw(data, ctx)

    market = _get_market_data(ctx)
    if "daily_change_pct" not in market.columns:
        return raw

    mkt = (
        market.drop_duplicates("date")
        .set_index("date")["daily_change_pct"]
        .astype("float64")
        .sort_index()
    )
    gate = _market_signal_weak_gate(mkt, vol_window=20)

    # 映射到股票行
    g_series = data["date"].map(gate).astype("float64")

    out = raw * g_series

    # 输出仍按日做一次轻 winsor，避免 gate 乘法放大偶发极值
    out = _cs_winsorize_by_date(out, data["date"], 0.01, 0.99)
    return out.astype("float64")


def _coskewness_alias(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if COSKEWNESS_ALIAS.lower() == "gated":
        return _coskewness_gated(data, ctx)
    return _coskewness_raw(data, ctx)


# ==============================================================================
# 3) Downside Beta（显式 rolling cov，更稳）
# ==============================================================================

def _downside_beta(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    下行贝塔 proxy：rolling Cov(r_i, r_m | r_m<0)
    - market_ret=0 的坏点：在条件 r_m<0 下通常不会参与（影响相对次要）
    - 用显式 rolling mean：cov = E[x*y] - E[x]E[y]，避免 cov().unstack() 结构不稳
    """
    if "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index)

    market = _get_market_data(ctx)
    if "daily_change_pct" not in market.columns:
        return pd.Series(np.nan, index=data.index)

    mkt_down = market.loc[market["daily_change_pct"] < 0, ["date", "daily_change_pct"]]
    mkt_down = (
        mkt_down.drop_duplicates("date")
        .set_index("date")["daily_change_pct"]
        .astype("float64")
        .sort_index()
    )

    df = data[["date", "pct_chg", "stock_code"]].copy()
    df["pct_chg"] = df["pct_chg"].astype("float64")
    df["mkt_ret"] = df["date"].map(mkt_down)  # 非下跌日为 NaN

    df = df.sort_values(["stock_code", "date"])
    g = df.groupby("stock_code", sort=False)

    xy = df["pct_chg"] * df["mkt_ret"]
    exy = xy.groupby(df["stock_code"], sort=False).transform(lambda x: x.rolling(30, min_periods=10).mean())
    ex = g["pct_chg"].transform(lambda x: x.rolling(30, min_periods=10).mean())
    ey = g["mkt_ret"].transform(lambda x: x.rolling(30, min_periods=10).mean())

    cov = (exy - ex * ey).replace([np.inf, -np.inf], np.nan)
    return _global_robust_zscore(cov, data["date"])


# ==============================================================================
# 4) Register
# ==============================================================================

skewness_20d = register_factor(FactorDefinition(
    name="skewness_20d",
    required_columns={"stock_code", "date", "pct_chg"},
    lookback_days=30,
    calculator=_skewness_20d,
    description="20日收益率偏度（按日MAD稳健标准化）"
))

kurtosis_20d = register_factor(FactorDefinition(
    name="kurtosis_20d",
    required_columns={"stock_code", "date", "pct_chg"},
    lookback_days=30,
    calculator=_kurtosis_20d,
    description="20日收益率峰度（按日MAD稳健标准化）"
))

# ✅ 新增：raw
coskewness_raw = register_factor(FactorDefinition(
    name="coskewness_raw",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=70,  # market 60d + stock 20d
    calculator=_coskewness_raw,
    description="协同偏度 raw（修复分组rolling + 按日稳健 + 无泄露winsor）"
))

# ✅ 新增：gated
coskewness_gated = register_factor(FactorDefinition(
    name="coskewness_gated",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=70,
    calculator=_coskewness_gated,
    description="协同偏度 gated（raw * 弱市场门控，用于复现弱信号regime）"
))

# ✅ 保留：coskewness（别名指向 raw/gated，可切换 COSKEWNESS_ALIAS）
coskewness = register_factor(FactorDefinition(
    name="coskewness",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=70,
    calculator=_coskewness_alias,
    description="协同偏度（alias -> raw/gated，受 COSKEWNESS_ALIAS 控制）"
))

downside_beta = register_factor(FactorDefinition(
    name="downside_beta",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=40,
    calculator=_downside_beta,
    description="下行贝塔（下跌日 rolling cov proxy，按日MAD稳健标准化）"
))

__all__ = [
    "skewness_20d",
    "kurtosis_20d",
    "coskewness_raw",
    "coskewness_gated",
    "coskewness",
    "downside_beta",
]
"""训练时把 coskewness_raw 和 coskewness_gated 都加入特征

coskewness 这个旧名你可以先保持 raw（默认）避免特征表改动太大

如果你想直接替换旧行为：把文件顶部 COSKEWNESS_ALIAS = "gated" 即可"""