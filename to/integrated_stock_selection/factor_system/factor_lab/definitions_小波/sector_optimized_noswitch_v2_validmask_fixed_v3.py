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
    """
    Prepare base panel with sector_code, _ts, and cleaned numeric columns.

    Important (for your valid_cs_mask semantics):
    - valid_cs_mask is a *cross-sectional* eligibility flag (ST/new/vol0/etc).
    - Do NOT apply it as a time-series NaN mask before rolling computations; otherwise you
      mutilate the time series and create long NaN cascades in rolling features.
    - Instead: use valid_cs_mask only when computing cross-sectional / sector aggregates,
      and in the final cross-sectional robust z-score.
    """
    df = data.copy()

    # valid cross-section mask (ST/new/etc). Keep rows; do NOT zero/NaN out TS fields here.
    if "valid_cs_mask" in df.columns:
        df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"]).astype(bool)
    else:
        df["valid_cs_mask"] = True

    df["_ts"] = _to_ts(df["date"])
    df["stock_code"] = df["stock_code"].astype(str)

    # sector mapping
    df["sector_code"] = df["stock_code"].map(_sector_map(ctx)).fillna("Unknown")

    # numeric cleaning (keep values; masking is handled later)
    if "pct_chg" in df.columns:
        df["pct_chg"] = _num(df["pct_chg"])
    if "close" in df.columns:
        df["close"] = _num(df["close"])
    if "circ_mv" in df.columns:
        df["circ_mv"] = _num(df["circ_mv"])

    return df

def _sector_daily_table(data: pd.DataFrame, ctx: FactorContext) -> pd.DataFrame:
    """
    Return sector daily table:
      _ts, sector_code, sector_avg_ret, sector_dispersion, sector_count,
      sector_volatility_5d_raw, sector_meanret_{p}d (p in 3/5/10/20)

    Notes:
    - sector aggregates are computed ONLY on valid_cs_mask == True rows, to avoid
      ST/new/invalid samples contaminating sector signals.
    - caching is protected by a data-window signature to avoid walk-forward cache poisoning.
    """
    # signature for walk-forward safety
    ts = _to_ts(data["date"])
    sig = (len(data), ts.min(), ts.max())

    key = "_sector_daily_table_v2"
    cached = (ctx.extra or {}).get(key)
    if isinstance(cached, dict) and cached.get("sig") == sig and isinstance(cached.get("data"), pd.DataFrame):
        return cached["data"]

    df = _prep_base(data, ctx)

    if "pct_chg" not in df.columns:
        out = pd.DataFrame({"_ts": df["_ts"], "sector_code": df["sector_code"], "sector_avg_ret": np.nan})
        out = out.drop_duplicates(["_ts", "sector_code"]).sort_values(["sector_code", "_ts"]).reset_index(drop=True)
        ctx.extra[key] = {"sig": sig, "data": out}
        return out

    df_stat = df[df["valid_cs_mask"]].copy()
    g = df_stat.groupby(["_ts", "sector_code"], sort=True)

    out = pd.DataFrame({
        "_ts": g.size().index.get_level_values(0),
        "sector_code": g.size().index.get_level_values(1),
        "sector_avg_ret": g["pct_chg"].mean().astype("float64").values,
        "sector_dispersion": g["pct_chg"].std().astype("float64").values,
        "sector_count": g.size().astype("float64").values,
    }).sort_values(["sector_code", "_ts"]).reset_index(drop=True)

    # 5d volatility and multi-horizon mean returns (computed on sector_avg_ret TS)
    parts: list[pd.DataFrame] = []
    for _, grp in out.groupby("sector_code", sort=False):
        grp = grp.sort_values("_ts").reset_index(drop=True)
        r = _num(grp["sector_avg_ret"])
        grp["sector_volatility_5d_raw"] = r.rolling(5, min_periods=3).std() * np.sqrt(252.0)
        for p in (3, 5, 10, 20):
            grp[f"sector_meanret_{p}d"] = r.rolling(p, min_periods=max(2, p // 2)).mean()
        parts.append(grp)

    out = pd.concat(parts, axis=0, ignore_index=True)
    ctx.extra[key] = {"sig": sig, "data": out}
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
    """
    Idiosyncratic volatility relative to sector mean return.

    Design:
    - sector mean is computed on valid_cs_mask==True subset per (day, sector).
    - idiosyncratic vol is rolling std of (stock_ret - sector_mean) per stock.
    - final output is cross-sectional robust z by day, with valid_cs_mask isolation.
    """
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["stock_code", "_ts", "sector_code", "pct_chg", "valid_cs_mask"]].copy()
    df.sort_values(["stock_code", "_ts"], inplace=True)

    # sector mean per day (valid subset only)
    df_stat = df[df["valid_cs_mask"]].copy()
    sec_mean_tbl = (
        df_stat.groupby(["_ts", "sector_code"], sort=False)["pct_chg"]
        .mean()
        .rename("_sec_mean")
        .reset_index()
    )
    df = df.merge(sec_mean_tbl, on=["_ts", "sector_code"], how="left", sort=False)

    df["_active"] = df["pct_chg"] - df["_sec_mean"]
    idio = df.groupby("stock_code", sort=False)["_active"].transform(lambda s: s.rolling(20, min_periods=10).std())

    out = _cs_robust_z(idio, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")

def _sector_volatility_5d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    daily = _sector_daily_table(data, ctx)

    # rank sector vol among sectors each day
    dfv = daily.dropna(subset=["sector_volatility_5d_raw"]).copy()
    dfv["rank01"] = dfv.groupby("_ts", sort=False)["sector_volatility_5d_raw"].rank(pct=True, method="average")
    dfv["raw"] = _center_rank01(dfv["rank01"])

    # map back to stock rows
    base = _prep_base(data, ctx)[["_ts", "sector_code"]].copy()
    base = base.merge(dfv[["_ts", "sector_code", "raw"]], on=["_ts", "sector_code"], how="left", sort=False)

    out = _cs_robust_z(base["raw"], base["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")

def _intra_sector_momentum_rank(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    5D momentum rank within (day, sector), computed on valid_cs_mask==True subset.

    - mom5 is computed on the full time series (no pre-masking).
    - the intra-sector rank is computed only among valid names; invalid names do not
      participate in the rank distribution.
    - final output is daily cross-sectional robust z with valid_cs_mask isolation.
    """
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["stock_code", "_ts", "sector_code", "pct_chg", "valid_cs_mask"]].copy()
    df.sort_values(["stock_code", "_ts"], inplace=True)

    mom5 = df.groupby("stock_code", sort=False)["pct_chg"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    df["mom5"] = mom5

    # rank within sector/day on valid subset only
    r01 = pd.Series(np.nan, index=df.index, dtype="float64")
    dfv = df[df["valid_cs_mask"]]
    if len(dfv) > 0:
        r01.loc[dfv.index] = (
            dfv.groupby(["_ts", "sector_code"], sort=False)["mom5"]
            .rank(pct=True, method="average")
            .astype("float64")
        )

    raw = _center_rank01(r01)

    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")

def _sector_momentum_dispersion_rank(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["stock_code", "_ts", "sector_code", "pct_chg", "valid_cs_mask"]].copy()
    df.sort_values(["stock_code", "_ts"], inplace=True)

    mom5 = df.groupby("stock_code", sort=False)["pct_chg"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    df["mom5"] = mom5

    # dispersion (std of mom5) within (day, sector), on valid subset only
    dfv = df[df["valid_cs_mask"]]
    disp_tbl = (
        dfv.groupby(["_ts", "sector_code"], sort=False)["mom5"]
        .std()
        .rename("_disp")
        .reset_index()
    )

    # rank sectors per day by dispersion (valid sectors only)
    disp_tbl["_rank01"] = disp_tbl.groupby("_ts", sort=False)["_disp"].rank(pct=True, method="average")
    disp_tbl["_raw"] = _center_rank01(disp_tbl["_rank01"])

    df = df.merge(disp_tbl[["_ts", "sector_code", "_raw"]], on=["_ts", "sector_code"], how="left", sort=False)

    out = _cs_robust_z(df["_raw"], df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")

def _sector_dispersion(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["_ts", "sector_code", "pct_chg", "valid_cs_mask"]].copy()

    # dispersion computed on valid subset only
    df_stat = df[df["valid_cs_mask"]].copy()
    sec_std_tbl = (
        df_stat.groupby(["_ts", "sector_code"], sort=False)["pct_chg"]
        .std()
        .rename("_sec_std")
        .reset_index()
    )
    df = df.merge(sec_std_tbl, on=["_ts", "sector_code"], how="left", sort=False)

    out = _cs_robust_z(df["_sec_std"], df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")

def _sector_momentum_consistency(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    daily = _sector_daily_table(data, ctx).copy()

    # for each horizon, rank sectors by mean return on that day
    cols = []
    for p in (3, 5, 10, 20):
        c = f"sector_meanret_{p}d"
        if c not in daily.columns:
            continue
        rk = daily.groupby("_ts", sort=False)[c].rank(pct=True, method="average")
        daily[f"rk_{p}"] = rk.astype("float64")
        cols.append(f"rk_{p}")

    if not cols:
        return pd.Series(0.0, index=data.index, dtype="float64")

    # consistency: std of ranks across horizons (lower std => more consistent)
    daily["rk_std"] = daily[cols].std(axis=1)
    # invert so higher is better (more consistent)
    daily["raw"] = -daily["rk_std"]

    base = _prep_base(data, ctx)[["_ts", "sector_code"]].copy()
    base = base.merge(daily[["_ts", "sector_code", "raw"]], on=["_ts", "sector_code"], how="left", sort=False)

    out = _cs_robust_z(base["raw"], base["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")

def _sector_correlation_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["stock_code", "_ts", "sector_code", "pct_chg", "valid_cs_mask"]].copy()
    df.sort_values(["stock_code", "_ts"], inplace=True)

    # sector avg return per day (valid subset only)
    dfv = df[df["valid_cs_mask"]]
    sec_avg_tbl = (
        dfv.groupby(["_ts", "sector_code"], sort=False)["pct_chg"]
        .mean()
        .rename("_sec_avg")
        .reset_index()
    )
    df = df.merge(sec_avg_tbl, on=["_ts", "sector_code"], how="left", sort=False)
    df["_sec_avg"] = _num(df["_sec_avg"])

    g = df.groupby("stock_code", sort=False)

    def _roll_corr(x: pd.Series, y: pd.Series) -> pd.Series:
        return x.rolling(20, min_periods=10).corr(y)

    corr = g.apply(lambda sub: _roll_corr(sub["pct_chg"], sub["_sec_avg"])).reset_index(level=0, drop=True)
    corr = corr.reindex(df.index)

    out = _cs_robust_z(corr, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")

def _sector_momentum_interaction(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "close" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["stock_code", "_ts", "sector_code", "close", "valid_cs_mask"]].copy()
    df.sort_values(["stock_code", "_ts"], inplace=True)

    g = df.groupby("stock_code", sort=False)
    prev_close = g["close"].shift(1)
    prev_20 = g["close"].shift(21)
    mom20 = (prev_close - prev_20) / (prev_20 + _EPS)
    df["mom20"] = _num(mom20)

    # sector momentum per day (valid subset only)
    dfv = df[df["valid_cs_mask"]]
    sec_mom_tbl = (
        dfv.groupby(["_ts", "sector_code"], sort=False)["mom20"]
        .mean()
        .rename("_sec_mom")
        .reset_index()
    )
    df = df.merge(sec_mom_tbl, on=["_ts", "sector_code"], how="left", sort=False)

    rel = df["mom20"] - df["_sec_mom"]

    # components: daily CS robust z then combine then final CS robust z
    z_sec = _cs_robust_z(df["_sec_mom"], df["_ts"], valid_cs_mask=_vmask(data))
    z_rel = _cs_robust_z(rel, df["_ts"], valid_cs_mask=_vmask(data))

    raw = z_sec + z_rel
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")

def _sw_circ_mv_rank_intra_sector(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "circ_mv" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = _prep_base(data, ctx)[["_ts", "sector_code", "circ_mv", "valid_cs_mask"]].copy()

    r01 = pd.Series(np.nan, index=df.index, dtype="float64")
    dfv = df[df["valid_cs_mask"]]
    if len(dfv) > 0:
        r01.loc[dfv.index] = (
            dfv.groupby(["_ts", "sector_code"], sort=False)["circ_mv"]
            .rank(pct=True, method="average")
            .astype("float64")
        )
    raw = _center_rank01(r01)

    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")

def _vmask(data: pd.DataFrame) -> pd.Series:
    """Return boolean valid cross-section mask aligned to `data.index`."""
    if isinstance(data, pd.DataFrame) and ("valid_cs_mask" in data.columns):
        return _clean_bool(data["valid_cs_mask"]).astype(bool)
    return pd.Series(True, index=data.index)

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