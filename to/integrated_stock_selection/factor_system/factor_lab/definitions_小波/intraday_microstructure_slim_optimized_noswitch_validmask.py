"""
Intraday & Microstructure 因子（slim）- valid_cs_mask 适配版（输出 float32）

要点：
- valid_cs_mask：按日横截面 robust z 的 median/MAD 仅在 mask=True 样本上统计；
  mask=False（ST/新股期等）不参与统计，且最终输出置 0（中性），避免污染分布。
- 输出统一为 float32（内部计算 float64 以保证稳定）。
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
    """
    按日横截面 robust z + clip。
    若提供 valid_cs_mask：仅 mask=True 样本参与统计；mask=False 输出置 0（中性）。
    """
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


def _attach_sorted(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["_ts"] = _to_ts(x["date"])
    x["stock_code"] = x["stock_code"].astype(str)
    x.sort_values(["stock_code", "_ts"], inplace=True)
    return x


# ==============================================================================
# 1) overnight_return: 标准化隔夜缺口（gap surprise）
# ==============================================================================
def _overnight_return(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "open", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df = _attach_sorted(data[["stock_code", "date"]].copy())

    open_ = _num64(data["open"]).reindex(df.index)
    close = _num64(data["close"]).reindex(df.index)
    g = df["stock_code"]

    close_prev = close.groupby(g, sort=False).shift(1)
    gap = (open_ - close_prev) / (close_prev + _EPS)

    # 过去 5 天日收益波动标准化（组内 rolling）
    ret1 = close.groupby(g, sort=False).pct_change(1)
    vol5 = ret1.groupby(g, sort=False).transform(lambda s: s.rolling(5, min_periods=3).std())

    score = (gap / (vol5 + _EPS)).clip(-8.0, 8.0)

    vmask = data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None
    out = _cs_robust_z(score, df["_ts"], valid_cs_mask=vmask).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# ==============================================================================
# 2) intraday_reversal: 量能确认的日内实体强度
# ==============================================================================
def _intraday_reversal(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "open", "close", "high", "low", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df = _attach_sorted(data[["stock_code", "date"]].copy())

    open_ = _num64(data["open"]).reindex(df.index)
    close = _num64(data["close"]).reindex(df.index)
    high = _num64(data["high"]).reindex(df.index)
    low = _num64(data["low"]).reindex(df.index)
    vol = _num64(data["volume"]).reindex(df.index)
    g = df["stock_code"]

    rng = (high - low)
    body = (close - open_)
    intensity = (body / (rng + _EPS)).clip(-1.5, 1.5)

    vol_ma5 = vol.groupby(g, sort=False).transform(lambda s: s.rolling(5, min_periods=3).mean())
    vol_ratio = vol / (vol_ma5 + _EPS)
    vol_weight = vol_ratio.clip(0.5, 3.0)

    raw = intensity * vol_weight

    vmask = data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=vmask).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# ==============================================================================
# 3) price_impact: 价格冲击
# ==============================================================================
def _price_impact(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "pct_chg", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df = _attach_sorted(data[["stock_code", "date"]].copy())

    pct = _num64(data["pct_chg"]).reindex(df.index)
    vol = _num64(data["volume"]).reindex(df.index)
    g = df["stock_code"]

    win, minp = 20, 8
    mean_all = vol.groupby(g, sort=False).transform(lambda s: s.rolling(win, min_periods=minp).mean())
    mean_hist = mean_all.groupby(g, sort=False).shift(1)
    volume_ratio = vol / (mean_hist + _EPS)

    impact = (pct.abs() / (volume_ratio + 0.1)).clip(0.0, 10.0)

    vmask = data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None
    out = _cs_robust_z(impact, df["_ts"], valid_cs_mask=vmask).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# ==============================================================================
# Register factors (names unchanged)
# ==============================================================================
overnight_return = register_factor(FactorDefinition(
    name="overnight_return",
    required_columns={"stock_code", "date", "open", "close"},
    lookback_days=10,
    calculator=_overnight_return,
    description="标准化隔夜缺口（gap surprise），按日CS robust z（valid_cs_mask；float32）",
))

intraday_reversal = register_factor(FactorDefinition(
    name="intraday_reversal",
    required_columns={"stock_code", "date", "open", "close", "high", "low", "volume"},
    lookback_days=10,
    calculator=_intraday_reversal,
    description="量能确认的日内实体强度（按日CS robust z；valid_cs_mask；float32）",
))

price_impact = register_factor(FactorDefinition(
    name="price_impact",
    required_columns={"stock_code", "date", "pct_chg", "volume"},
    lookback_days=30,
    calculator=_price_impact,
    description="价格冲击：abs(pct_chg)/(volume_ratio+0.1)，按日CS robust z（valid_cs_mask；float32）",
))

__all__ = [
    "overnight_return",
    "intraday_reversal",
    "price_impact",
]
