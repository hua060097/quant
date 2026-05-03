"""
style_rotation.py - 优化版（无开关）

你当前体系：optimized_noswitch（统一工程口径）
- 横截面：按日 robust z（median/MAD）+ clip + fillna(0)
- 时序：严格因果 rolling（按 stock_code/_ts 或 sector_code/_ts 排序）
- 缓存：仅使用 ctx.extra（你的 FactorContext 没有 cache/_ctx_cache）
- 避免 unstack/stack 的宽表操作，改为 long 表 + groupby/transform（更省内存、更快）

关于“sector level 常数”：
- 这类特征在同日跨板块有差异，因此在 lambdarank（按日 group）里是可用的。
- 但在板块内部是常数，更多决定“选哪个板块”，对“板块内选谁”需要配合交互（你已有 sector_interaction / sector_wavelet）。

本文件保留原有 9 个因子名：
1) sector_rank_change_5d
2) sector_momentum_accel_rank
3) sector_turnover_change_5d_20d
4) sector_hhi_change
5) sector_relative_strength_vs_market
6) sector_concentration
7) sector_breadth
8) size_rotation_strength   (market-level daily)
9) value_growth_rotation    (market-level daily)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


_EPS = 1e-12


def _vmask(data: pd.DataFrame) -> pd.Series:
    """Return boolean valid cross-section mask aligned to `data.index`."""
    if isinstance(data, pd.DataFrame) and ("valid_cs_mask" in data.columns):
        return _clean_bool(data["valid_cs_mask"]).astype(bool)
    return pd.Series(True, index=data.index)

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

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = _num(a)
    b = _num(b)
    return a / (b.replace(0.0, np.nan) + _EPS)


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
    """[0,1] -> [-1,1]"""
    return (r01 - 0.5) * 2.0


def _get_sector_map(ctx: FactorContext) -> dict:
    mp = (ctx.extra or {}).get("stock_to_sw_sector")
    if mp is None:
        raise ValueError("SW行业映射必须通过 ctx.extra['stock_to_sw_sector'] 提供")
    return mp


# ==============================================================================
# Base builders (cached)
# ==============================================================================
def _attach_sector_code(df: pd.DataFrame, ctx: FactorContext) -> pd.DataFrame:
    x = df.copy()
    x["_ts"] = _to_ts(x["date"])
    mp = _get_sector_map(ctx)
    x["sector_code"] = x["stock_code"].map(mp).fillna("Unknown")
    return x


def _sector_daily_base(data: pd.DataFrame, ctx: FactorContext) -> pd.DataFrame:
    """
    Per (ts, sector) base table:
      - sector_avg_ret
      - sector_avg_turnover (if turnover_rate exists)
      - sector_vol_hhi (volume HHI, if volume exists)
      - sector_breadth (adv-dec ratio proxy)
    """
    key = "_style_rotation_sector_daily_base_v2"
    cached = ctx.extra.get(key)
    if cached is not None:
        return cached

    df = _attach_sector_code(data, ctx)
    if "valid_cs_mask" in df.columns:
        df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"]).astype(bool)
    else:
        df["valid_cs_mask"] = True
    df["pct_chg"] = _num(df.get("pct_chg", np.nan)).where(df["valid_cs_mask"], np.nan)
    df_stat = df[df["valid_cs_mask"]].copy()

    g = df_stat.groupby(["_ts", "sector_code"], sort=True)

    out = g["pct_chg"].mean().rename("sector_avg_ret").reset_index()

    # turnover (avg)
    if "turnover_rate" in df.columns:
        df_stat["turnover_rate"] = _num(df_stat.get("turnover_rate", np.nan))
        df["turnover_rate"] = _num(df["turnover_rate"])
        to = g["turnover_rate"].mean().rename("sector_avg_turnover").reset_index()
        out = out.merge(to, on=["_ts", "sector_code"], how="left")
    else:
        out["sector_avg_turnover"] = np.nan

    # volume HHI (concentration)
    if "volume" in df.columns:
        df_stat["volume"] = _num(df_stat.get("volume", np.nan)).fillna(0.0)
        df["volume"] = _num(df["volume"]).fillna(0.0)
        # share within sector/day
        total = g["volume"].transform("sum")
        share = df["volume"] / (total + _EPS)
        share_sq = share * share
        hhi = share_sq.groupby([df["_ts"], df["sector_code"]], sort=False).sum()
        hhi = hhi.rename("sector_vol_hhi").reset_index()
        out = out.merge(hhi, on=["_ts", "sector_code"], how="left")
    else:
        out["sector_vol_hhi"] = np.nan

    # breadth thrust: net adv ratio within sector
    status = np.where(df_stat["pct_chg"] > 0.2, 1.0, np.where(df["pct_chg"] < -0.2, -1.0, 0.0))
    df_stat["_status"] = status
    # mean status in [-1,1]
    breadth = df_stat.groupby(["_ts","sector_code"], sort=True)["_status"].mean().rename("sector_breadth").reset_index()
    out = out.merge(breadth, on=["_ts", "sector_code"], how="left")

    # cleanup
    for c in ["sector_avg_ret", "sector_avg_turnover", "sector_vol_hhi", "sector_breadth"]:
        out[c] = _num(out[c])

    ctx.extra[key] = out.sort_values(["sector_code", "_ts"]).reset_index(drop=True)
    return ctx.extra[key]


def _market_daily_ret(data: pd.DataFrame) -> pd.Series:
    df = data.copy()
    df["_ts"] = _to_ts(df["date"])
    pct = _num(df.get("pct_chg", np.nan))
    if "valid_cs_mask" in df.columns:
        pct = pct.where(_clean_bool(df["valid_cs_mask"]), np.nan)
    return pct.groupby(df["_ts"], sort=True).mean()


# ==============================================================================
# Factor calculators (sector-level -> mapped to stocks)
# ==============================================================================
def _map_sector_metric_to_stock(data: pd.DataFrame, ctx: FactorContext, metric: pd.DataFrame, col: str) -> pd.Series:
    """metric: columns ['_ts','sector_code', col]"""
    df = _attach_sector_code(data, ctx)
    merged = df.merge(metric[["_ts", "sector_code", col]], on=["_ts", "sector_code"], how="left")
    # standardize across all stocks each day
    return _cs_robust_z(merged[col], merged["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)


# 1) sector_rank_change_5d
def _sector_rank_change_5d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    base = _sector_daily_base(data, ctx)[["_ts", "sector_code", "sector_avg_ret"]].copy()
    base = base.sort_values(["sector_code", "_ts"])

    # cross-sector rank per day
    base["rk"] = base.groupby("_ts", sort=False)["sector_avg_ret"].rank(pct=True, method="average")
    # time series change per sector
    base["rk_chg_5"] = base.groupby("sector_code", sort=False)["rk"].diff(5)
    base["rk_chg_5"] = _num(base["rk_chg_5"]).clip(-1.0, 1.0)

    return _map_sector_metric_to_stock(data, ctx, base, "rk_chg_5")


# 2) sector_momentum_accel_rank (mom5 - mom20, then cross-sector rank)
def _sector_momentum_accel_rank(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    base = _sector_daily_base(data, ctx)[["_ts", "sector_code", "sector_avg_ret"]].copy()
    base = base.sort_values(["sector_code", "_ts"])

    base["mom5"] = base.groupby("sector_code", sort=False)["sector_avg_ret"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    base["mom20"] = base.groupby("sector_code", sort=False)["sector_avg_ret"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    base["accel"] = base["mom5"] - base["mom20"]

    # cross-sector rank per day, center to [-1,1] for symmetry
    base["accel_rank01"] = base.groupby("_ts", sort=False)["accel"].rank(pct=True, method="average")
    base["accel_center"] = _center_rank01(base["accel_rank01"])

    return _map_sector_metric_to_stock(data, ctx, base, "accel_center")


# 3) sector_turnover_change_5d_20d (avg turnover ratio, then rank)
def _sector_turnover_change_5d_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "turnover_rate" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    base = _sector_daily_base(data, ctx)[["_ts", "sector_code", "sector_avg_turnover"]].copy()
    base = base.sort_values(["sector_code", "_ts"])

    base["to5"] = base.groupby("sector_code", sort=False)["sector_avg_turnover"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    base["to20"] = base.groupby("sector_code", sort=False)["sector_avg_turnover"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    base["ratio"] = base["to5"] / (base["to20"] + _EPS)

    base["ratio_rank01"] = base.groupby("_ts", sort=False)["ratio"].rank(pct=True, method="average")
    base["ratio_center"] = _center_rank01(base["ratio_rank01"])

    return _map_sector_metric_to_stock(data, ctx, base, "ratio_center")


# 4) sector_hhi_change (HHI 5d-20d, then rank)
def _sector_hhi_change(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "volume" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    base = _sector_daily_base(data, ctx)[["_ts", "sector_code", "sector_vol_hhi"]].copy()
    base = base.sort_values(["sector_code", "_ts"])

    base["hhi5"] = base.groupby("sector_code", sort=False)["sector_vol_hhi"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    base["hhi20"] = base.groupby("sector_code", sort=False)["sector_vol_hhi"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    base["diff"] = base["hhi5"] - base["hhi20"]

    base["diff_rank01"] = base.groupby("_ts", sort=False)["diff"].rank(pct=True, method="average")
    base["diff_center"] = _center_rank01(base["diff_rank01"])

    return _map_sector_metric_to_stock(data, ctx, base, "diff_center")


# 5) sector_relative_strength_vs_market (sector avg ret rank vs market)
def _sector_relative_strength_vs_market(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    base = _sector_daily_base(data, ctx)[["_ts", "sector_code", "sector_avg_ret"]].copy()
    mkt = _market_daily_ret(data)
    base["mkt_ret"] = base["_ts"].map(mkt)

    base["rel"] = base["sector_avg_ret"] - base["mkt_ret"]
    base["rel_rank01"] = base.groupby("_ts", sort=False)["rel"].rank(pct=True, method="average")
    base["rel_center"] = _center_rank01(base["rel_rank01"])

    return _map_sector_metric_to_stock(data, ctx, base, "rel_center")


# 6) sector_concentration (level HHI, rank across sectors)
def _sector_concentration(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "volume" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    base = _sector_daily_base(data, ctx)[["_ts", "sector_code", "sector_vol_hhi"]].copy()
    base["hhi_rank01"] = base.groupby("_ts", sort=False)["sector_vol_hhi"].rank(pct=True, method="average")
    base["hhi_center"] = _center_rank01(base["hhi_rank01"])

    return _map_sector_metric_to_stock(data, ctx, base, "hhi_center")


# 7) sector_breadth (net adv ratio within sector, rank across sectors)
def _sector_breadth(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    base = _sector_daily_base(data, ctx)[["_ts", "sector_code", "sector_breadth"]].copy()
    base["br_rank01"] = base.groupby("_ts", sort=False)["sector_breadth"].rank(pct=True, method="average")
    base["br_center"] = _center_rank01(base["br_rank01"])

    return _map_sector_metric_to_stock(data, ctx, base, "br_center")


# ==============================================================================
# Market-level daily style rotations (broadcast to all stocks)
# ==============================================================================
def _size_rotation_strength(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    小盘 vs 大盘超额收益（按日计算，market-level 常数）
    - 用 circ_mv 分位：小盘=<=20%，大盘=>=80%
    - 输出为日度序列映射到所有股票；缺失填 0
    """
    if "circ_mv" not in data.columns or "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = data[["date", "circ_mv", "pct_chg"]].copy()
    df["_ts"] = _to_ts(df["date"])
    df["circ_mv"] = _num(df["circ_mv"])
    df["pct_chg"] = _num(df["pct_chg"])

    g = df.groupby("_ts", sort=True)
    q80 = g["circ_mv"].transform(lambda x: x.quantile(0.8))
    q20 = g["circ_mv"].transform(lambda x: x.quantile(0.2))

    is_large = df["circ_mv"] >= q80
    is_small = df["circ_mv"] <= q20

    # daily means
    small_ret = df.loc[is_small].groupby("_ts")["pct_chg"].mean()
    large_ret = df.loc[is_large].groupby("_ts")["pct_chg"].mean()

    spread = (small_ret - large_ret).reindex(g.size().index)
    spread = _num(spread).fillna(0.0)

    out = df["_ts"].map(spread).fillna(0.0).astype("float64")
    return out.reindex(data.index)


def _value_growth_rotation(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    低位 vs 高位超额收益（按日计算，market-level 常数）
    - 用过去20日动量(价格位置)分位：低位=<=20%，高位=>=80%
    - 输出为日度序列映射到所有股票；缺失填 0
    """
    if "close" not in data.columns or "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = data[["stock_code", "date", "close", "pct_chg"]].copy()
    df["_ts"] = _to_ts(df["date"])
    df["close"] = _num(df["close"])
    df["pct_chg"] = _num(df["pct_chg"])
    df.sort_values(["stock_code", "_ts"], inplace=True)

    # 20d momentum (causal)
    prev_close = df.groupby("stock_code", sort=False)["close"].shift(1)
    prev_20 = df.groupby("stock_code", sort=False)["close"].shift(21)
    mom20 = (prev_close - prev_20) / (prev_20 + _EPS)

    df["mom20"] = _num(mom20)

    g = df.groupby("_ts", sort=True)
    q80 = g["mom20"].transform(lambda x: x.quantile(0.8))
    q20 = g["mom20"].transform(lambda x: x.quantile(0.2))

    is_high = df["mom20"] >= q80
    is_low = df["mom20"] <= q20

    low_ret = df.loc[is_low].groupby("_ts")["pct_chg"].mean()
    high_ret = df.loc[is_high].groupby("_ts")["pct_chg"].mean()

    spread = (low_ret - high_ret).reindex(g.size().index)
    spread = _num(spread).fillna(0.0)

    out = df["_ts"].map(spread).fillna(0.0).astype("float64")
    return out.reindex(data.index)


# ==============================================================================
# Register factors
# ==============================================================================
sector_rank_change_5d = register_factor(FactorDefinition(
    name="sector_rank_change_5d",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_sector_rank_change_5d,
    description="SW行业5日排名变化（跨行业rank变化，映射到股票；按日robust z）",
))

sector_momentum_accel_rank = register_factor(FactorDefinition(
    name="sector_momentum_accel_rank",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_sector_momentum_accel_rank,
    description="SW行业动量加速度（5d-20d）跨行业rank，映射到股票；按日robust z",
))

sector_turnover_change_5d_20d = register_factor(FactorDefinition(
    name="sector_turnover_change_5d_20d",
    required_columns={"stock_code", "date", "turnover_rate"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_sector_turnover_change_5d_20d,
    description="SW行业换手率5d/20d变化（跨行业rank）映射到股票；按日robust z",
))

sector_hhi_change = register_factor(FactorDefinition(
    name="sector_hhi_change",
    required_columns={"stock_code", "date", "volume"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_sector_hhi_change,
    description="SW行业成交量集中度变化（HHI 5d-20d，跨行业rank）映射到股票；按日robust z",
))

sector_relative_strength_vs_market = register_factor(FactorDefinition(
    name="sector_relative_strength_vs_market",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=5,
    calculator=_sector_relative_strength_vs_market,
    description="SW行业相对全市场强弱（行业均值-市场均值，跨行业rank）映射到股票；按日robust z",
))

sector_concentration = register_factor(FactorDefinition(
    name="sector_concentration",
    required_columns={"stock_code", "date", "volume"},
    requires_sector_data=True,
    lookback_days=2,
    calculator=_sector_concentration,
    description="SW行业成交量集中度水平（HHI 跨行业rank）映射到股票；按日robust z",
))

sector_breadth = register_factor(FactorDefinition(
    name="sector_breadth",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=2,
    calculator=_sector_breadth,
    description="SW行业广度推力（净上涨占比，跨行业rank）映射到股票；按日robust z",
))

size_rotation_strength = register_factor(FactorDefinition(
    name="size_rotation_strength",
    required_columns={"date", "circ_mv", "pct_chg"},
    lookback_days=30,
    calculator=_size_rotation_strength,
    description="大小盘风格轮动：小盘-大盘超额收益（market-level 日度，广播到股票）",
))

value_growth_rotation = register_factor(FactorDefinition(
    name="value_growth_rotation",
    required_columns={"stock_code", "date", "close", "pct_chg"},
    lookback_days=60,
    calculator=_value_growth_rotation,
    description="高低位切换：低位-高位超额收益（market-level 日度，广播到股票）",
))

__all__ = [
    "sector_rank_change_5d",
    "sector_momentum_accel_rank",
    "sector_turnover_change_5d_20d",
    "sector_hhi_change",
    "sector_relative_strength_vs_market",
    "sector_concentration",
    "sector_breadth",
    "size_rotation_strength",
    "value_growth_rotation",
]