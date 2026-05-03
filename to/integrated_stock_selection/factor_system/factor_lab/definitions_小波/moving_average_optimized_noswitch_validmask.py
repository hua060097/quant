"""
Moving Average 因子 - valid_cs_mask 适配 + float32

基于 moving_average_optimized_noswitch.py：
- 时序纯净：rolling/shift 严格因果
- 横截面统一：入模版本（*_z + ma_* score）按日 CS robust z（median/MAD）+ clip + fillna(0)
- valid_cs_mask：CS 统计仅用 mask=True；mask=False 行输出 0
- 输出 dtype：float32（内部 float64）

保留因子名（与原文件一致）：
- ma_convergence_score
- ma_alignment_score
- price_ma60_ratio (raw)
- price_ma60_ratio_z
- ma60_slope
- price_ma5_ratio (raw)
- price_ma10_ratio (raw)
- price_ma10_ratio_z
- price_ma5_ratio_z
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_EPS = 1e-12
_CLIP_Z = 6.0
_OUT_DTYPE = "float32"


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


def _prep(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["_ts"] = _to_ts(df["date"])
    df["stock_code"] = df["stock_code"].astype(str)
    if "valid_cs_mask" in df.columns:
        df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"])
    else:
        df["valid_cs_mask"] = True
    if "close" in df.columns:
        df["close"] = _num64(df["close"])
    df.sort_values(["stock_code", "_ts"], inplace=True)
    return df


def _cs_robust_z(x: pd.Series, date_keys: pd.Series, *, valid_cs_mask: pd.Series | None = None, clip_z: float = _CLIP_Z) -> pd.Series:
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


def _mask_to_zero(raw: pd.Series, valid_cs_mask: pd.Series) -> pd.Series:
    m = _clean_bool(valid_cs_mask)
    return raw.where(m, 0.0).fillna(0.0)


# 1) ma_convergence_score
def _ma_convergence_score(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if not {"stock_code", "date", "close"}.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _prep(data[cols])

    g = df.groupby("stock_code", sort=False)["close"]
    ma5 = g.transform(lambda s: s.rolling(5, min_periods=5).mean())
    ma10 = g.transform(lambda s: s.rolling(10, min_periods=10).mean())
    ma20 = g.transform(lambda s: s.rolling(20, min_periods=20).mean())
    ma60 = g.transform(lambda s: s.rolling(60, min_periods=60).mean())

    mas = pd.concat([ma5, ma10, ma20, ma60], axis=1)
    ma_mean = mas.mean(axis=1)
    ma_std = mas.std(axis=1)

    cv = ma_std / (ma_mean.abs() + _EPS)
    ent = np.exp(-10.0 * cv)
    out = _cs_robust_z(ent, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# 2) ma_alignment_score
def _ma_alignment_score(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if not {"stock_code", "date", "close"}.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _prep(data[cols])
    g = df.groupby("stock_code", sort=False)["close"]

    ma5 = g.transform(lambda s: s.rolling(5, min_periods=5).mean())
    ma10 = g.transform(lambda s: s.rolling(10, min_periods=10).mean())
    ma20 = g.transform(lambda s: s.rolling(20, min_periods=20).mean())
    ma60 = g.transform(lambda s: s.rolling(60, min_periods=60).mean())

    ma60_prev5 = ma60.groupby(df["stock_code"], sort=False).shift(5)
    ma60_trend_up = ma60 > ma60_prev5

    close = df["close"]
    score = (
        np.where(close > ma5, 0.2, -0.2)
        + np.where(ma5 > ma10, 0.2, -0.2)
        + np.where(ma10 > ma20, 0.2, -0.2)
        + np.where(ma20 > ma60, 0.2, -0.2)
        + np.where(ma60_trend_up, 0.2, -0.2)
    )
    score_s = pd.Series(score, index=df.index, dtype="float64")
    out = _cs_robust_z(score_s, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# raw ratios
def _price_ma_ratio(data: pd.DataFrame, window: int, minp: int, mult: float = 1.0) -> pd.Series:
    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _prep(data[cols])
    close = df["close"]
    ma = close.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(window, min_periods=minp).mean())
    ratio = (close - ma) / (ma + _EPS) * mult
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio = _mask_to_zero(ratio, df["valid_cs_mask"])
    return ratio.reindex(data.index).astype(_OUT_DTYPE)


def _price_ma_ratio_z(data: pd.DataFrame, window: int, minp: int, mult: float = 1.0) -> pd.Series:
    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _prep(data[cols])
    close = df["close"]
    ma = close.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(window, min_periods=minp).mean())
    ratio = (close - ma) / (ma + _EPS) * mult
    out = _cs_robust_z(ratio, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# ma60_slope
def _ma60_slope(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if not {"stock_code", "date", "close"}.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _prep(data[cols])
    g = df.groupby("stock_code", sort=False)["close"]
    ma60 = g.transform(lambda s: s.rolling(60, min_periods=30).mean())
    ma60_prev5 = ma60.groupby(df["stock_code"], sort=False).shift(5)
    slope = (ma60 - ma60_prev5) / (ma60_prev5 + _EPS) * 100.0
    out = _cs_robust_z(slope, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# Register
ma_convergence_score = register_factor(FactorDefinition(
    name="ma_convergence_score",
    required_columns={"stock_code", "date", "close"},
    lookback_days=80,
    calculator=_ma_convergence_score,
))

ma_alignment_score = register_factor(FactorDefinition(
    name="ma_alignment_score",
    required_columns={"stock_code", "date", "close"},
    lookback_days=80,
    calculator=_ma_alignment_score,
))

price_ma60_ratio = register_factor(FactorDefinition(
    name="price_ma60_ratio",
    required_columns={"stock_code", "date", "close"},
    lookback_days=80,
    calculator=lambda d, c: _price_ma_ratio(d, 60, 30, 1.0),
))

price_ma60_ratio_z = register_factor(FactorDefinition(
    name="price_ma60_ratio_z",
    required_columns={"stock_code", "date", "close"},
    lookback_days=80,
    calculator=lambda d, c: _price_ma_ratio_z(d, 60, 30, 1.0),
))

ma60_slope = register_factor(FactorDefinition(
    name="ma60_slope",
    required_columns={"stock_code", "date", "close"},
    lookback_days=80,
    calculator=_ma60_slope,
))

price_ma5_ratio = register_factor(FactorDefinition(
    name="price_ma5_ratio",
    required_columns={"stock_code", "date", "close"},
    lookback_days=10,
    calculator=lambda d, c: _price_ma_ratio(d, 5, 1, 100.0),
))

price_ma10_ratio = register_factor(FactorDefinition(
    name="price_ma10_ratio",
    required_columns={"stock_code", "date", "close"},
    lookback_days=10,
    calculator=lambda d, c: _price_ma_ratio(d, 10, 1, 100.0),
))

price_ma10_ratio_z = register_factor(FactorDefinition(
    name="price_ma10_ratio_z",
    required_columns={"stock_code", "date", "close"},
    lookback_days=10,
    calculator=lambda d, c: _price_ma_ratio_z(d, 10, 1, 100.0),
))

price_ma5_ratio_z = register_factor(FactorDefinition(
    name="price_ma5_ratio_z",
    required_columns={"stock_code", "date", "close"},
    lookback_days=10,
    calculator=lambda d, c: _price_ma_ratio_z(d, 5, 1, 100.0),
))

__all__ = [
    "ma_convergence_score",
    "ma_alignment_score",
    "price_ma60_ratio",
    "price_ma60_ratio_z",
    "ma60_slope",
    "price_ma5_ratio",
    "price_ma10_ratio",
    "price_ma10_ratio_z",
    "price_ma5_ratio_z",
]
