"""
sector_optimized_noswitch.py

对 sector.py 做“optimized_noswitch”体系的统一重写与修复：
- 时序纯净：所有 rolling/shift 按 stock_code,_ts 或 sector_code,_ts 排序后进行，严格因果；rolling 阶段不 fillna
- 横截面统一：所有输出统一为按日 CS robust z（median/MAD）+ clip + fillna(0)
- 性能与稳定性：避免 groupby.apply；相关性/波动用滚动矩法或 transform；行业日度表做 ctx.extra 缓存

与原 sector.py 的主要差异：
- 原文件刻意“保留 NaN 不 fill”fileciteturn20file0；本版本按你的系统口径统一做 CS robust z + fillna(0)
- 修复 sector_momentum_interaction 原逻辑中以“df.groupby('date')['mom']”计算 sector Z 的口径不一致问题（那是按股票横截面而非行业横截面）fileciteturn20file0
- 修复/替换 sector_correlation_20d 的 groupby.apply rolling.corr 为滚动矩法（更快、更稳定）fileciteturn20file0

依赖：
- ctx.extra["stock_to_sw_sector"] : stock_code -> sector_code

输入列：
- 必需：stock_code, date, pct_chg
- 部分因子需要：close（用于20d动量），circ_mv（用于市值排名）

输出因子（name 不变，便于直接替换）：
- sector_relative_volatility
- sector_volatility_5d
- intra_sector_momentum_rank
- sector_momentum_dispersion_rank
- sector_dispersion
- sector_momentum_consistency
- sector_correlation_20d
- sector_momentum_interaction
- sw_circ_mv_rank_intra_sector
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


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

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
    按日横截面 robust z（median/MAD），并 clip。
    - 统计 median/MAD 时仅使用 valid_cs_mask==True 的样本
    - valid_cs_mask==False 的行输出置 0（中性）
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

def _center_rank01(r01: pd.Series) -> pd.Series:
    return (r01 - 0.5) * 2.0


def _sector_map(ctx: FactorContext) -> dict:
    mp = (ctx.extra or {}).get("stock_to_sw_sector")
    if mp is None:
        raise ValueError("SW行业映射必须通过 ctx.extra['stock_to_sw_sector'] 提供")
    return mp


def _prep_base(data: pd.DataFrame, ctx: FactorContext) -> pd.DataFrame:
    df = data.copy()
    # valid cross-section mask (ST/new/etc). Keep rows but exclude from stats/rolling.
    if "valid_cs_mask" in df.columns:
        df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"]).astype(bool)
    else:
        df["valid_cs_mask"] = True
    df["_ts"] = _to_ts(df["date"])
    df["stock_code"] = df["stock_code"].astype(str)
    df["sector_code"] = df["stock_code"].map(_sector_map(ctx)).fillna("Unknown")
    if "pct_chg" in df.columns:
        df["pct_chg"] = _num(df["pct_chg"]).where(df["valid_cs_mask"], np.nan)
    if "close" in df.columns:
        df["close"] = _num(df["close"]).where(df["valid_cs_mask"], np.nan)
    if "circ_mv" in df.columns:
        df["circ_mv"] = _num(df["circ_mv"]).where(df["valid_cs_mask"], np.nan)
    return df


# ==============================================================================
# Sector daily table (cached)
# ==============================================================================
def _sector_daily_table(data: pd.DataFrame, ctx: FactorContext) -> pd.DataFrame:
    """
    返回行业日度表：_ts, sector_code, sector_avg_ret, sector_vol_5d, sector_dispersion
    以及用于其它派生：sector_mom_3/5/10/20, sector_mom_std_ranks, etc.
    """
    key = "_sector_daily_table_v2"
    cached = (ctx.extra or {}).get(key)
    if cached is not None:
        return cached

    df = _prep_base(data, ctx)
    df_stat = df[df["valid_cs_mask"]].copy()
    if "pct_chg" not in df.columns:
        # 最小返回
        out = pd.DataFrame({"_ts": df["_ts"], "sector_code": df["sector_code"], "sector_avg_ret": np.nan})
        out = out.drop_duplicates(["_ts", "sector_code"]).sort_values(["sector_code", "_ts"]).reset_index(drop=True)
        ctx.extra[key] = out
        return out

    g = df_stat.groupby(["_ts", "sector_code"], sort=True)

    out = pd.DataFrame({
        "_ts": g.size().index.get_level_values(0),
        "sector_code": g.size().index.get_level_values(1),
        "sector_avg_ret": g["pct_chg"].mean().astype("float64").values,
        "sector_dispersion": g["pct_chg"].std().astype("float64").values,
        "sector_count": g.size().astype("float64").values,
    }).sort_values(["sector_code", "_ts"]).reset_index(drop=True)

    # 5d vol (annualized optional, but scale doesn't matter after CS robust z)
    parts = []
    for _, grp in out.groupby("sector_code", sort=False):
        grp = grp.sort_values("_ts").reset_index(drop=True)
        r = _num(grp["sector_avg_ret"])
        vol5 = r.rolling(5, min_periods=3).std() * np.sqrt(252.0)
        grp["sector_volatility_5d_raw"] = vol5

        # multi-horizon mean returns for consistency
        for p in (3, 5, 10, 20):
            grp[f"sector_meanret_{p}d"] = r.rolling(p, min_periods=max(2, p // 2)).mean()
        parts.append(grp)

    out = pd.concat(parts, axis=0, ignore_index=True)

    ctx.extra[key] = out
    return out


def _map_sector_col_to_stock(data: pd.DataFrame, ctx: FactorContext, col: str) -> pd.Series:
    df = _prep_base(data, ctx)
    sec = _sector_daily_table(data, ctx)[["_ts", "sector_code", col]]
    merged = df.merge(sec, on=["_ts", "sector_code"], how="left")
    return merged[col].reindex(data.index)


# ==============================================================================
# 1) sector_relative_volatility: std of active return (20d)
# ==============================================================================
def _sector_relative_volatility(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["stock_code", "_ts", "sector_code", "pct_chg"]].copy()
    df.sort_values(["stock_code", "_ts"], inplace=True)

    # sector mean per day
    sec_mean = df.groupby(["_ts", "sector_code"], sort=False)["pct_chg"].transform("mean")
    active = df["pct_chg"] - sec_mean

    idio = active.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=10).std())

    out = _cs_robust_z(idio, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")


# ==============================================================================
# 2) sector_volatility_5d: sector 5d vol percentile then CS robust z
# ==============================================================================
def _sector_volatility_5d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    sec = _sector_daily_table(data, ctx)[["_ts", "sector_code", "sector_volatility_5d_raw"]].copy()
    # rank across sectors per day
    sec["vol_rank01"] = sec.groupby("_ts", sort=False)["sector_volatility_5d_raw"].rank(pct=True, method="average")
    sec["vol_rank01"] = sec["vol_rank01"].astype("float64")

    df = _prep_base(data, ctx)[["_ts", "sector_code"]].copy()
    merged = df.merge(sec[["_ts", "sector_code", "vol_rank01"]], on=["_ts", "sector_code"], how="left")

    raw = _center_rank01(merged["vol_rank01"])
    out = _cs_robust_z(raw, merged["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")


# ==============================================================================
# 3) intra_sector_momentum_rank: stock 5d mom rank within sector per day
# ==============================================================================
def _intra_sector_momentum_rank(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["stock_code", "_ts", "sector_code", "pct_chg"]].copy()
    df.sort_values(["stock_code", "_ts"], inplace=True)

    mom5 = df.groupby("stock_code", sort=False)["pct_chg"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    df["mom5"] = mom5

    # rank within sector/day
    r01 = df.groupby(["_ts", "sector_code"], sort=False)["mom5"].rank(pct=True, method="average")
    raw = _center_rank01(r01)

    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")


# ==============================================================================
# 4) sector_momentum_dispersion_rank: std of stock mom5 within sector; rank across sectors per day
# ==============================================================================
def _sector_momentum_dispersion_rank(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["stock_code", "_ts", "sector_code", "pct_chg"]].copy()
    df.sort_values(["stock_code", "_ts"], inplace=True)
    mom5 = df.groupby("stock_code", sort=False)["pct_chg"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    df["mom5"] = mom5

    sec_disp = df.groupby(["_ts", "sector_code"], sort=False)["mom5"].agg(["count", "std"]).reset_index()
    sec_disp = sec_disp[sec_disp["count"] >= 3]
    sec_disp["rank01"] = sec_disp.groupby("_ts", sort=False)["std"].rank(pct=True, method="average")

    base = df[["_ts", "sector_code"]].copy()
    merged = base.merge(sec_disp[["_ts", "sector_code", "rank01"]], on=["_ts", "sector_code"], how="left")
    raw = _center_rank01(merged["rank01"])

    out = _cs_robust_z(raw, merged["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")


# ==============================================================================
# 5) sector_dispersion: std of pct_chg within sector/day
# ==============================================================================
def _sector_dispersion(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["_ts", "sector_code", "pct_chg"]].copy()
    sec_std = df.groupby(["_ts", "sector_code"], sort=False)["pct_chg"].transform("std")

    out = _cs_robust_z(sec_std, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")


# ==============================================================================
# 6) sector_momentum_consistency: multi-horizon ranks std (lower std => more consistent)
# ==============================================================================
def _sector_momentum_consistency(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    sec = _sector_daily_table(data, ctx).copy()

    # ranks across sectors per day for each horizon
    rank_cols = []
    for p in (3, 5, 10, 20):
        col = f"sector_meanret_{p}d"
        rk = f"rk_{p}"
        sec[rk] = sec.groupby("_ts", sort=False)[col].rank(pct=True, method="average")
        rank_cols.append(rk)

    # consistency: 1 - normalized std across ranks
    rstd = sec[rank_cols].std(axis=1, skipna=True)
    # normalize per day (optional). Use cross-sectional robust z later anyway.
    raw = 1.0 - (rstd / (rstd.groupby(sec["_ts"]).transform("max") + _EPS))
    sec["consistency_raw"] = raw

    df = _prep_base(data, ctx)[["_ts", "sector_code"]].copy()
    merged = df.merge(sec[["_ts", "sector_code", "consistency_raw"]], on=["_ts", "sector_code"], how="left")

    out = _cs_robust_z(merged["consistency_raw"], merged["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")


# ==============================================================================
# 7) sector_correlation_20d: corr(stock_ret, sector_avg_ret) rolling 20d via moments
# ==============================================================================
def _sector_correlation_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["stock_code", "_ts", "sector_code", "pct_chg"]].copy()
    df.sort_values(["stock_code", "_ts"], inplace=True)

    # sector avg ret map to each stock-day
    sec_avg = df.groupby(["_ts", "sector_code"], sort=False)["pct_chg"].transform("mean")
    x = df["pct_chg"]
    y = sec_avg
    xy = x * y
    x2 = x * x
    y2 = y * y

    win = 20
    minp = 10
    g = df.groupby("stock_code", sort=False)

    mean_x = g["pct_chg"].transform(lambda s: s.rolling(win, min_periods=minp).mean())
    mean_y = y.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(win, min_periods=minp).mean())
    mean_xy = xy.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(win, min_periods=minp).mean())
    mean_x2 = x2.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(win, min_periods=minp).mean())
    mean_y2 = y2.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(win, min_periods=minp).mean())

    cov = mean_xy - mean_x * mean_y
    varx = mean_x2 - mean_x * mean_x
    vary = mean_y2 - mean_y * mean_y

    corr = cov / (np.sqrt(varx.clip(lower=0.0) * vary.clip(lower=0.0)) + _EPS)
    corr = corr.clip(-1.0, 1.0)

    out = _cs_robust_z(corr, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")


# ==============================================================================
# 8) sector_momentum_interaction: sector 20d mom × stock relative strength (20d)
# ==============================================================================
def _sector_momentum_interaction(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "close" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["stock_code", "_ts", "sector_code", "close"]].copy()
    df.sort_values(["stock_code", "_ts"], inplace=True)

    mom20 = df.groupby("stock_code", sort=False)["close"].transform(lambda s: s.pct_change(20))
    df["mom20"] = mom20

    sec_mom = df.groupby(["_ts", "sector_code"], sort=False)["mom20"].transform("mean")
    rel = mom20 - sec_mom

    # 组件都按日 CS robust z，再相加（避免量纲/分布差异）
    z_sec = _cs_robust_z(sec_mom, df["_ts"], valid_cs_mask=_vmask(data))
    z_rel = _cs_robust_z(rel, df["_ts"], valid_cs_mask=_vmask(data))

    raw = z_sec + z_rel
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")


# ==============================================================================
# 9) sw_circ_mv_rank_intra_sector: circ_mv rank within sector/day
# ==============================================================================
def _sw_circ_mv_rank_intra_sector(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "circ_mv" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["_ts", "sector_code", "circ_mv"]].copy()
    r01 = df.groupby(["_ts", "sector_code"], sort=False)["circ_mv"].rank(pct=True, method="average")
    raw = _center_rank01(r01)

    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")


# ==============================================================================
# Register factors (names unchanged)
# ==============================================================================
sector_relative_volatility = register_factor(FactorDefinition(
    name="sector_relative_volatility",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=30,
    calculator=_sector_relative_volatility,
    description="个股相对行业的特质波动率：active_ret 20d std（按日CS robust z）",
))

sector_volatility_5d = register_factor(FactorDefinition(
    name="sector_volatility_5d",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_sector_volatility_5d,
    description="行业5日波动率在行业间的分位（再按日CS robust z）",
))

intra_sector_momentum_rank = register_factor(FactorDefinition(
    name="intra_sector_momentum_rank",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_intra_sector_momentum_rank,
    description="个股5日动量在行业内的分位排名（再按日CS robust z）",
))

sector_momentum_dispersion_rank = register_factor(FactorDefinition(
    name="sector_momentum_dispersion_rank",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_sector_momentum_dispersion_rank,
    description="行业内个股5日动量离散度（std）在行业间的分位（再按日CS robust z）",
))

sector_dispersion = register_factor(FactorDefinition(
    name="sector_dispersion",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=2,
    calculator=_sector_dispersion,
    description="行业内当日收益离散度（std），映射到股票（按日CS robust z）",
))

sector_momentum_consistency = register_factor(FactorDefinition(
    name="sector_momentum_consistency",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=30,
    calculator=_sector_momentum_consistency,
    description="行业多周期动量一致性：3/5/10/20日行业均值收益排名的稳定性（按日CS robust z）",
))

sector_correlation_20d = register_factor(FactorDefinition(
    name="sector_correlation_20d",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=40,
    calculator=_sector_correlation_20d,
    description="个股收益与所属行业均值收益的20日滚动相关性（矩法；按日CS robust z）",
))

sector_momentum_interaction = register_factor(FactorDefinition(
    name="sector_momentum_interaction",
    required_columns={"stock_code", "date", "close"},
    requires_sector_data=True,
    lookback_days=40,
    calculator=_sector_momentum_interaction,
    description="行业20日动量 × 个股相对行业强度（组件CS robust z 后相加，再按日CS robust z）",
))

sw_circ_mv_rank_intra_sector = register_factor(FactorDefinition(
    name="sw_circ_mv_rank_intra_sector",
    required_columns={"stock_code", "date", "circ_mv"},
    requires_sector_data=True,
    lookback_days=2,
    calculator=_sw_circ_mv_rank_intra_sector,
    description="行业内流通市值分位排名（再按日CS robust z）",
))



def _vmask(data: pd.DataFrame) -> pd.Series:
    """Return boolean valid cross-section mask aligned to `data.index`."""
    if isinstance(data, pd.DataFrame) and ("valid_cs_mask" in data.columns):
        return _clean_bool(data["valid_cs_mask"]).astype(bool)
    return pd.Series(True, index=data.index)

__all__ = [
    "intra_sector_momentum_rank",
    "sector_momentum_dispersion_rank",
    "sector_dispersion",
    "sector_relative_volatility",
    "sector_momentum_consistency",
    "sector_correlation_20d",
    "sector_volatility_5d",
    "sector_momentum_interaction",
    "sw_circ_mv_rank_intra_sector",
]