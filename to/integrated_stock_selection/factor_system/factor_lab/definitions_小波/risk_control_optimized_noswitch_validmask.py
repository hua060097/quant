"""
Risk Control 因子 - valid_cs_mask 适配 + float32

基于 risk_control_optimized_noswitch.py：
- 时序纯净：按 stock_code,_ts 排序 rolling/shift
- 横截面统一：按日 CS robust z（median/MAD）+ clip + fillna(0)
- valid_cs_mask：CS 统计仅用 mask=True；mask=False 行输出 0
- 输出 dtype：float32（内部 float64）

因子名保持不变：
- efficiency_ratio
- ulcer_index
- max_rally_20d
- max_drawdown_20d
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


def _prep_close(data: pd.DataFrame) -> pd.DataFrame:
    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = data[cols].copy()
    df["_ts"] = _to_ts(df["date"])
    df["stock_code"] = df["stock_code"].astype(str)
    df["close"] = _num64(df["close"])
    if "valid_cs_mask" in df.columns:
        df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"])
    else:
        df["valid_cs_mask"] = True
    df.sort_values(["stock_code", "_ts"], inplace=True)
    return df


def _efficiency_ratio(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if "close" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    N, minp = 20, 15
    df = _prep_close(data)
    g = df.groupby("stock_code", sort=False)["close"]
    change_n = g.diff(N).abs()
    daily_path = g.diff(1).abs()
    total_path = daily_path.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(N, min_periods=minp).sum())
    er = change_n / (total_path + _EPS)
    out = _cs_robust_z(er, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


def _ulcer_index(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if "close" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    N, minp_max, minp_ui = 20, 5, 10
    df = _prep_close(data)
    g = df.groupby("stock_code", sort=False)["close"]
    rolling_max = g.transform(lambda s: s.rolling(N, min_periods=minp_max).max())
    dd_pct = (df["close"] / (rolling_max + _EPS) - 1.0) * 100.0
    dd_sq = dd_pct * dd_pct
    mean_dd_sq = dd_sq.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(N, min_periods=minp_ui).mean())
    ulcer = np.sqrt(mean_dd_sq)
    raw = -ulcer
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# def _max_rally_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
#     if "close" not in data.columns:
#         return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
#     N, minp = 20, 5
#     df = _prep_close(data)
#     g = df.groupby("stock_code", sort=False)["close"]
#     rolling_min = g.transform(lambda s: s.rolling(N, min_periods=minp).min())
#     rally = (df["close"] - rolling_min) / (rolling_min + _EPS)
#     out = _cs_robust_z(rally, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
#     return out.astype(_OUT_DTYPE)


def _max_rally_20d_vC2(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if "close" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    N, minp = 20, 5
    df = _prep_close(data).copy()

    g = df.groupby("stock_code", sort=False)["close"]
    df["_smooth_close"] = g.transform(lambda s: s.rolling(3, min_periods=1).mean())

    rolling_min = df.groupby("stock_code", sort=False)["_smooth_close"].transform(
        lambda s: s.rolling(N, min_periods=minp).min()
    )
    raw = (df["close"] - rolling_min) / (rolling_min + _EPS)

    _ts = df["_ts"] if "_ts" in df.columns else df.index.get_level_values("date")
    valid_mask = df.get("valid_cs_mask", None)

    # ---- 横截面 winsorize：分位线只用有效样本算 ----
    q_lo, q_hi = 0.005, 0.995  # 建议先温和一点；你也可以改回 0.01/0.99

    if valid_mask is not None:
        m = pd.to_numeric(valid_mask, errors="coerce").fillna(1.0) > 0.5
        raw_stat = raw.where(m, np.nan)
    else:
        m = None
        raw_stat = raw

    grp = raw_stat.groupby(_ts, sort=False)
    lo = grp.transform(lambda s: s.quantile(q_lo))
    hi = grp.transform(lambda s: s.quantile(q_hi))

    wins = raw.clip(lower=lo, upper=hi)
    if m is not None:
        wins = wins.where(m, np.nan)

    out = _cs_robust_z(wins, _ts, valid_cs_mask=valid_mask).reindex(data.index)
    return out.astype(_OUT_DTYPE)


def _max_drawdown_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if "close" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)
    N, minp = 20, 5
    df = _prep_close(data)
    g = df.groupby("stock_code", sort=False)["close"]
    rolling_max = g.transform(lambda s: s.rolling(N, min_periods=minp).max())
    drawdown = (df["close"] - rolling_max) / (rolling_max + _EPS)
    out = _cs_robust_z(drawdown, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


efficiency_ratio = register_factor(FactorDefinition(
    name="efficiency_ratio",
    required_columns={"stock_code", "date", "close"},
    lookback_days=30,
    calculator=_efficiency_ratio,
))

ulcer_index = register_factor(FactorDefinition(
    name="ulcer_index",
    required_columns={"stock_code", "date", "close"},
    lookback_days=30,
    calculator=_ulcer_index,
))

max_rally_20d = register_factor(FactorDefinition(
    name="max_rally_20d",
    required_columns={"stock_code", "date", "close"},
    lookback_days=30,
    calculator=_max_rally_20d_vC2,
))

max_drawdown_20d = register_factor(FactorDefinition(
    name="max_drawdown_20d",
    required_columns={"stock_code", "date", "close"},
    lookback_days=30,
    calculator=_max_drawdown_20d,
))

__all__ = [
    "efficiency_ratio",
    "ulcer_index",
    "max_drawdown_20d",
    "max_rally_20d",
]
