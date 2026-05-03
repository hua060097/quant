"""
Market Corr 因子 - valid_cs_mask 适配版（输出 float32）

在 market_corr_optimized_noswitch.py 基础上：
- 增加 valid_cs_mask：按日横截面 robust z 的 median/MAD 仅使用 mask=True 样本统计；
  mask=False（ST/新股期等）不参与统计，且最终输出置 0（中性）。
- 输出统一为 float32（内部计算保持 float64 稳定）。

因子（名称保持不变）：
- beta_60d
- relative_strength_20d
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_EPS = 1e-12
_CLIP_Z = 6.0
_OUT_DTYPE = "float32"


# ==============================================================================
# Utils
# ==============================================================================
def _to_ts(date_series: pd.Series) -> pd.Series:
    """date -> Timestamp normalize（00:00:00）"""
    if pd.api.types.is_datetime64_any_dtype(date_series):
        ts = pd.to_datetime(date_series, errors="coerce")
    else:
        ds = date_series.astype(str).str.replace("-", "", regex=False)
        ts = pd.to_datetime(ds, format="%Y%m%d", errors="coerce")
    return ts.dt.normalize()


def _num64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def _clean_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(1.0) > 0.5


def _cs_robust_z(
    x: pd.Series,
    date_keys: pd.Series,
    *,
    valid_cs_mask: pd.Series | None = None,
    clip_z: float = _CLIP_Z,
) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")

    if valid_cs_mask is not None:
        m = _clean_bool(valid_cs_mask)
        s_stat = s.where(m, np.nan)
    else:
        m = None
        s_stat = s

    g = s_stat.groupby(date_keys, sort=False)
    med = g.transform("median")
    mad = g.transform(lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))))
    z = (s - med) / (1.4826 * mad + _EPS)
    z = z.clip(-clip_z, clip_z)

    if m is not None:
        z = z.where(m, np.nan)

    return z.fillna(0.0).astype(_OUT_DTYPE)


def _get_market_map(ctx: FactorContext) -> pd.Series:
    """
    返回 market_ret Series: index=_ts, value=mkt_ret
    使用 daily_change_pct 作为市场日收益
    """
    mkt = ctx.require_market().copy()
    if "date" not in mkt.columns or "daily_change_pct" not in mkt.columns:
        raise ValueError("market_data 需要包含列：date, daily_change_pct")
    mkt["_ts"] = _to_ts(mkt["date"])
    mkt["mkt_ret"] = _num64(mkt["daily_change_pct"])
    mkt = mkt.dropna(subset=["_ts"]).sort_values("_ts")
    # 同日如果重复，取均值
    mkt_ret = mkt.groupby("_ts", sort=True)["mkt_ret"].mean()
    return mkt_ret


# ==============================================================================
# 1) Rolling Beta (60d)
# ==============================================================================
def _rolling_beta_60d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if not {"stock_code", "date", "pct_chg"}.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    mkt_ret = _get_market_map(ctx)

    df = data[["stock_code", "date", "pct_chg"]].copy()
    df["_ts"] = _to_ts(df["date"])
    df["stock_code"] = df["stock_code"].astype(str)
    df["ret"] = _num64(df["pct_chg"])
    df["mkt_ret"] = df["_ts"].map(mkt_ret)

    df.sort_values(["stock_code", "_ts"], inplace=True)

    g = df.groupby("stock_code", sort=False)

    x = df["ret"]
    y = df["mkt_ret"]
    xy = x * y
    y2 = y * y

    mean_x = g["ret"].transform(lambda s: s.rolling(60, min_periods=30).mean())
    mean_y = g["mkt_ret"].transform(lambda s: s.rolling(60, min_periods=30).mean())
    mean_xy = xy.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(60, min_periods=30).mean())
    mean_y2 = y2.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(60, min_periods=30).mean())

    cov = mean_xy - mean_x * mean_y
    var = mean_y2 - mean_y * mean_y
    beta = cov / (var + _EPS)

    vmask = data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None
    out = _cs_robust_z(beta, df["_ts"], valid_cs_mask=vmask).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# ==============================================================================
# 2) Relative Strength (20d cumulative excess return)
# ==============================================================================
def _relative_strength_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if not {"stock_code", "date", "pct_chg"}.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    mkt_ret = _get_market_map(ctx)

    df = data[["stock_code", "date", "pct_chg"]].copy()
    df["_ts"] = _to_ts(df["date"])
    df["stock_code"] = df["stock_code"].astype(str)
    df["ret"] = _num64(df["pct_chg"])
    df["mkt_ret"] = df["_ts"].map(mkt_ret)

    df.sort_values(["stock_code", "_ts"], inplace=True)

    alpha = df["ret"] - df["mkt_ret"]
    rs = alpha.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=10).sum())

    vmask = data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None
    out = _cs_robust_z(rs, df["_ts"], valid_cs_mask=vmask).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# ==============================================================================
# Register factors (names unchanged)
# ==============================================================================
beta_60d = register_factor(
    FactorDefinition(
        name="beta_60d",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_market_data=True,
        lookback_days=80,
        calculator=_rolling_beta_60d,
        description="60日市场贝塔系数（Cov/Var），按日robust z（valid_cs_mask；float32）",
    )
)

relative_strength_20d = register_factor(
    FactorDefinition(
        name="relative_strength_20d",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_market_data=True,
        lookback_days=30,
        calculator=_relative_strength_20d,
        description="20日累计超额收益（stock - market），按日robust z（valid_cs_mask；float32）",
    )
)

__all__ = [
    "relative_strength_20d",
    "beta_60d",
]
