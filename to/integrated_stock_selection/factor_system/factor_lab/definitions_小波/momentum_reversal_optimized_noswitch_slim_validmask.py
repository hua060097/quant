"""
Momentum & Reversal 因子（精简版） - valid_cs_mask 适配 + float32

基于 momentum_reversal_optimized_noswitch_slim.py：
- 时序纯净：按 stock_code,_ts 排序 rolling/shift
- 横截面统一：最终输出按日 CS robust z（median/MAD）+ clip + fillna(0)
- valid_cs_mask：CS 统计仅用 mask=True；mask=False 行输出 0
- 输出 dtype：float32（内部 float64）

保留因子（14个，名称不变）：
- high_to_close_drawdown
- intraday_clv
- pct_chg_1d
- pct_chg_2d
- return_3d
- return_10d
- return_20d
- momentum_decay
- trend_acceleration
- trend_consistency
- cpv_factor
- stable_momentum
- trend_pullback_composite_alpha
- volume_confirmed_pullback_alpha
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


def _attach_sorted(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["_ts"] = _to_ts(df["date"])
    df["stock_code"] = df["stock_code"].astype(str)
    if "valid_cs_mask" in df.columns:
        df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"])
    else:
        df["valid_cs_mask"] = True
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


# 1) High->Close drawdown
def _high_to_close_drawdown(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if not {"date", "high", "close"}.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    df = data[["date", "high", "close"]].copy()
    df["_ts"] = _to_ts(df["date"])
    high = _num64(df["high"])
    close = _num64(df["close"])
    raw = (close - high) / (high + _EPS)
    vmask = data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None
    return _cs_robust_z(raw, df["_ts"], valid_cs_mask=vmask).reindex(data.index).astype(_OUT_DTYPE)


# 2) CLV
def _intraday_clv(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if not {"date", "high", "low", "close"}.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    df = data[["date", "high", "low", "close"]].copy()
    df["_ts"] = _to_ts(df["date"])
    high = _num64(df["high"])
    low = _num64(df["low"])
    close = _num64(df["close"])
    rng = high - low
    clv = (close - low) / (rng + _EPS)
    clv = clv.where(rng.abs() >= 1e-6, 1.0).clip(0.0, 1.0)
    vmask = data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None
    return _cs_robust_z(clv, df["_ts"], valid_cs_mask=vmask).reindex(data.index).astype(_OUT_DTYPE)


# 3) pct_chg 1d/2d raw (cs z)
def _pct_chg_kd(k: int):
    def calc(data: pd.DataFrame, _: FactorContext) -> pd.Series:
        if not {"stock_code", "date", "pct_chg"}.issubset(set(data.columns)):
            return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
        df = _attach_sorted(data[["stock_code", "date", "pct_chg"] + (["valid_cs_mask"] if "valid_cs_mask" in data.columns else [])])
        pct = _num64(df["pct_chg"])
        raw = pct.groupby(df["stock_code"], sort=False).shift(k-1) if k>1 else pct
        out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
        return out.astype(_OUT_DTYPE)
    return calc


# 4) return_kd from close
def _return_kd(k: int):
    def calc(data: pd.DataFrame, _: FactorContext) -> pd.Series:
        if not {"stock_code", "date", "close"}.issubset(set(data.columns)):
            return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
        cols = ["stock_code", "date", "close"]
        if "valid_cs_mask" in data.columns:
            cols.append("valid_cs_mask")
        df = _attach_sorted(data[cols])
        close = _num64(df["close"])
        prev = close.groupby(df["stock_code"], sort=False).shift(k)
        raw = (close - prev) / (prev + _EPS) * 100.0
        out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
        return out.astype(_OUT_DTYPE)
    return calc


# 5) momentum_decay
def _momentum_decay(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "close", "pct_chg"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    cols = ["stock_code", "date", "close", "pct_chg"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _attach_sorted(data[cols])
    close = _num64(df["close"])
    pct = _num64(df["pct_chg"])
    g = df["stock_code"]
    prev5 = close.groupby(g, sort=False).shift(5)
    prev20 = close.groupby(g, sort=False).shift(20)
    ret5 = (close - prev5) / (prev5 + _EPS)
    ret20 = (close - prev20) / (prev20 + _EPS)
    vol20 = pct.groupby(g, sort=False).transform(lambda s: s.rolling(20, min_periods=10).std())
    raw = (ret5 - (ret20 / 4.0)) / (vol20 + _EPS)
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# 6) trend_acceleration
def _trend_acceleration(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _attach_sorted(data[cols])
    close = _num64(df["close"])
    g = df["stock_code"]
    ma5 = close.groupby(g, sort=False).transform(lambda s: s.rolling(5, min_periods=3).mean())
    ma20 = close.groupby(g, sort=False).transform(lambda s: s.rolling(20, min_periods=10).mean())
    slope5 = ma5.groupby(g, sort=False).diff(3)
    slope20 = ma20.groupby(g, sort=False).diff(10)
    raw = (slope5 - slope20).replace([np.inf, -np.inf], np.nan)
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# 7) trend_consistency
def _trend_consistency(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if not {"stock_code", "date", "pct_chg"}.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    cols = ["stock_code", "date", "pct_chg"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _attach_sorted(data[cols])
    pct = _num64(df["pct_chg"])
    up = (pct > 0).astype("float64")
    raw = up.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(5, min_periods=1).mean())
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# 8) cpv_factor
def _cpv_factor(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "close", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    cols = ["stock_code", "date", "close", "volume"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _attach_sorted(data[cols])
    g = df["stock_code"]
    close = _num64(df["close"])
    vol = _num64(df["volume"]).clip(lower=0.0)

    prev_close = close.groupby(g, sort=False).shift(1)
    a = np.log(close + _EPS) - np.log(prev_close + _EPS)
    prev_vol = vol.groupby(g, sort=False).shift(1)
    b = np.log1p(vol) - np.log1p(prev_vol.clip(lower=0.0))

    a = pd.Series(a, index=df.index).replace([np.inf, -np.inf], np.nan).astype("float64")
    b = pd.Series(b, index=df.index).replace([np.inf, -np.inf], np.nan).astype("float64")

    win, minp = 20, 15
    mean_a = a.groupby(g, sort=False).transform(lambda s: s.rolling(win, min_periods=minp).mean())
    mean_b = b.groupby(g, sort=False).transform(lambda s: s.rolling(win, min_periods=minp).mean())
    mean_a2 = (a*a).groupby(g, sort=False).transform(lambda s: s.rolling(win, min_periods=minp).mean())
    mean_b2 = (b*b).groupby(g, sort=False).transform(lambda s: s.rolling(win, min_periods=minp).mean())
    mean_ab = (a*b).groupby(g, sort=False).transform(lambda s: s.rolling(win, min_periods=minp).mean())
    var_a = mean_a2 - mean_a*mean_a
    var_b = mean_b2 - mean_b*mean_b
    cov = mean_ab - mean_a*mean_b
    corr = cov / (np.sqrt(var_a.clip(lower=0.0)*var_b.clip(lower=0.0)) + _EPS)
    corr = corr.clip(-1.0, 1.0)
    out = _cs_robust_z(corr, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# 9) stable_momentum (Kaufman ER signed)
def _stable_momentum(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if not {"stock_code", "date", "close"}.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _attach_sorted(data[cols])
    g = df["stock_code"]
    close = _num64(df["close"])
    lookback = 10
    net = close.groupby(g, sort=False).diff(lookback).abs()
    path = close.groupby(g, sort=False).diff(1).abs()
    total = path.groupby(g, sort=False).transform(lambda s: s.rolling(lookback, min_periods=max(2, lookback-2)).sum())
    er = net / (total + _EPS)
    direction = np.sign(close.groupby(g, sort=False).diff(lookback))
    raw = er * direction
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# 10) trend_pullback_composite_alpha
def _trend_pullback_composite_alpha(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "close", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    cols = ["stock_code", "date", "close", "volume"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _attach_sorted(data[cols])
    g = df["stock_code"]
    close = _num64(df["close"])
    vol = _num64(df["volume"]).clip(lower=0.0)

    ma20 = close.groupby(g, sort=False).transform(lambda s: s.rolling(20, min_periods=10).mean())
    ma60 = close.groupby(g, sort=False).transform(lambda s: s.rolling(60, min_periods=30).mean())
    trend_gate = ((ma20 > ma60) & (close > ma20)).astype("float64")

    prev20 = close.groupby(g, sort=False).shift(20)
    ret20 = (close - prev20) / (prev20 + _EPS)
    trend_gate *= (ret20 > 0).astype("float64")

    prev5 = close.groupby(g, sort=False).shift(5)
    ret5 = (close - prev5) / (prev5 + _EPS)
    pullback = -ret5.clip(upper=0.0)

    vol3 = vol.groupby(g, sort=False).transform(lambda s: s.rolling(3, min_periods=2).mean())
    vol10 = vol.groupby(g, sort=False).transform(lambda s: s.rolling(10, min_periods=5).mean())
    contraction = 1.0 - vol3 / (vol10 + _EPS)

    ma5 = close.groupby(g, sort=False).transform(lambda s: s.rolling(5, min_periods=3).mean())
    support = (close > ma5).astype("float64")
    quality = (-pullback + contraction + support)

    raw = trend_gate * quality
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# 11) volume_confirmed_pullback_alpha
def _volume_confirmed_pullback_alpha(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "close", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    cols = ["stock_code", "date", "close", "volume"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _attach_sorted(data[cols])
    g = df["stock_code"]
    close = _num64(df["close"])
    vol = _num64(df["volume"]).clip(lower=0.0)

    vol5 = vol.groupby(g, sort=False).transform(lambda s: s.rolling(5, min_periods=3).mean())
    vol20 = vol.groupby(g, sort=False).transform(lambda s: s.rolling(20, min_periods=10).mean())
    expansion = vol5 / (vol20 + _EPS)

    prev5 = close.groupby(g, sort=False).shift(5)
    ret5 = (close - prev5) / (prev5 + _EPS)

    prev3 = close.groupby(g, sort=False).shift(3)
    ret3 = (close - prev3) / (prev3 + _EPS)
    pullback_depth = -ret3.clip(upper=0.0) / (ret5.abs() + _EPS)

    pullback_vol = vol.groupby(g, sort=False).transform(lambda s: s.rolling(3, min_periods=2).mean())
    pullback_vol_ratio = pullback_vol / (vol5 + _EPS)

    raw = ret5 - pullback_depth - pullback_vol_ratio + 0.2 * (expansion - 1.0)
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# ==============================================================================
# Register (names unchanged)
# ==============================================================================
high_to_close_drawdown = register_factor(FactorDefinition(
    name="high_to_close_drawdown",
    required_columns={"stock_code", "date", "high", "close"},
    lookback_days=0,
    calculator=_high_to_close_drawdown,
))

intraday_clv = register_factor(FactorDefinition(
    name="intraday_clv",
    required_columns={"stock_code", "date", "high", "low", "close"},
    lookback_days=5,
    calculator=_intraday_clv,
))

pct_chg_1d = register_factor(FactorDefinition(
    name="pct_chg_1d",
    required_columns={"stock_code", "date", "pct_chg"},
    lookback_days=2,
    calculator=_pct_chg_kd(1),
))

pct_chg_2d = register_factor(FactorDefinition(
    name="pct_chg_2d",
    required_columns={"stock_code", "date", "pct_chg"},
    lookback_days=3,
    calculator=_pct_chg_kd(2),
))

return_3d = register_factor(FactorDefinition(
    name="return_3d",
    required_columns={"stock_code", "date", "close"},
    lookback_days=6,
    calculator=_return_kd(3),
))

return_10d = register_factor(FactorDefinition(
    name="return_10d",
    required_columns={"stock_code", "date", "close"},
    lookback_days=15,
    calculator=_return_kd(10),
))

return_20d = register_factor(FactorDefinition(
    name="return_20d",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=_return_kd(20),
))

momentum_decay = register_factor(FactorDefinition(
    name="momentum_decay",
    required_columns={"stock_code", "date", "close", "pct_chg"},
    lookback_days=25,
    calculator=_momentum_decay,
))

trend_acceleration = register_factor(FactorDefinition(
    name="trend_acceleration",
    required_columns={"stock_code", "date", "close"},
    lookback_days=30,
    calculator=_trend_acceleration,
))

trend_consistency = register_factor(FactorDefinition(
    name="trend_consistency",
    required_columns={"stock_code", "date", "pct_chg"},
    lookback_days=10,
    calculator=_trend_consistency,
))

cpv_factor = register_factor(FactorDefinition(
    name="cpv_factor",
    required_columns={"stock_code", "date", "close", "volume"},
    lookback_days=30,
    calculator=_cpv_factor,
))

stable_momentum = register_factor(FactorDefinition(
    name="stable_momentum",
    required_columns={"stock_code", "date", "close"},
    lookback_days=20,
    calculator=_stable_momentum,
))

trend_pullback_composite_alpha = register_factor(FactorDefinition(
    name="trend_pullback_composite_alpha",
    required_columns={"stock_code", "date", "close", "volume"},
    lookback_days=60,
    calculator=_trend_pullback_composite_alpha,
))

volume_confirmed_pullback_alpha = register_factor(FactorDefinition(
    name="volume_confirmed_pullback_alpha",
    required_columns={"stock_code", "date", "close", "volume"},
    lookback_days=30,
    calculator=_volume_confirmed_pullback_alpha,
))

__all__ = [
    "high_to_close_drawdown",
    "intraday_clv",
    "pct_chg_1d",
    "pct_chg_2d",
    "return_3d",
    "return_10d",
    "return_20d",
    "momentum_decay",
    "trend_acceleration",
    "trend_consistency",
    "cpv_factor",
    "stable_momentum",
    "trend_pullback_composite_alpha",
    "volume_confirmed_pullback_alpha",
]
