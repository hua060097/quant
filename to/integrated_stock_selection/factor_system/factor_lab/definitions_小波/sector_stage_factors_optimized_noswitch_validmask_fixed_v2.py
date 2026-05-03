"""
sector_stage_factors_optimized_noswitch.py

目的：补齐“板块启动初期 vs 启动末期（高潮/分歧/衰竭）”的阶段识别因子，并额外提供 breakout_z。

设计原则（与你现有 optimized_noswitch 体系一致）：
- 仅使用 ctx.extra 做行业映射/缓存（FactorContext 无 cache/_ctx_cache）
- 所有 rolling 严格因果（按 sector_code/_ts 或 stock_code/_ts 排序后 rolling）
- 输出统一：按日横截面 robust z（median/MAD）+ clip + fillna(0)
- 尽量避免宽表 unstack，使用 long 表 groupby/merge

依赖：
- 行业映射：ctx.extra["stock_to_sw_sector"] : stock_code -> sector_code
- 输入列：stock_code,date,pct_chg,is_limit_up,is_limit_down,is_zhaban
- 可选列：volume, turnover_rate, close, high（用于更好的集中度/突破）

本文件提供的因子（名称可直接喂给模型）：
A. 板块阶段（按行业日度统计，再映射到股票）：
1) sector_emotion_heat_level_20d     : 行业情绪热度的 20日历史分位（0~1 映射到 [-1,1]）
2) sector_emotion_heat_accel_3d      : 情绪热度 3日变化（再做跨股票按日 robust z）
3) sector_divergence_risk            : 高热度 + 炸板占比上升（分歧/末期风险）
4) sector_breadth_level_20d          : 行业广度的 20日历史分位（扩散/末期常见）
5) sector_concentration_level_20d    : 行业成交集中度(HHI)的 20日历史分位（过度集中/抱团）
6) sector_streak_strength_10d        : 行业“强势连贯性”强度（近10日强势天数占比）
7) sector_drawdown_from_peak_10d     : 行业“指数代理”的 10日回撤（衰竭/见顶后加速下跌）

B. 个股结构突破：
8) breakout_z                        : 个股突破强度（close / rolling_max(high,20) - 1），按日 robust z

说明：
- 这些“sector level 常数”在同日跨板块有差异，因此在按日 lambdarank 里可用于跨板块排序；
  其中 sector_divergence_risk / streak / drawdown 更专注“末期风险”，通常更有利于降回撤。
- breakout_z 与 mom5_intra_z/intra_rank 可以共存：breakout 更偏“位置/结构”，动量偏“速度”，rank偏“相对领先”。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


_EPS = 1e-12
_CLIP_Z = 6.0
_OUT_DTYPE = "float32"

_REQ_BASE = {"stock_code", "date", "pct_chg", "is_limit_up", "is_limit_down", "is_zhaban"}


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
    return (r01 - 0.5) * 2.0


def _get_sector_map(ctx: FactorContext) -> dict:
    mp = (ctx.extra or {}).get("stock_to_sw_sector")
    if mp is None:
        raise ValueError("SW行业映射必须通过 ctx.extra['stock_to_sw_sector'] 提供")
    return mp


def _rolling_last_percentile(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    Rolling 历史分位：取窗口内“最后一个值”在窗口内的百分位（0~1），严格因果。
    计算量 O(n*window)（window=20，按行业序列一般可接受）。
    """
    arr = x.values.astype(float)
    out = np.full_like(arr, np.nan, dtype=float)
    n = len(arr)
    for i in range(n):
        lo = max(0, i - window + 1)
        w = arr[lo:i+1]
        w = w[np.isfinite(w)]
        if w.size < min_periods or not np.isfinite(arr[i]):
            continue
        out[i] = float((w <= arr[i]).mean())
    return pd.Series(out, index=x.index)


# ==============================================================================
# Sector daily stats builder (cached)
# ==============================================================================
def _sector_daily_stage_table(data: pd.DataFrame, ctx: FactorContext) -> pd.DataFrame:
    """
    行业日度“阶段/情绪”表：主键 (_ts, sector_code) 唯一。
    注意：
    - valid_cs_mask 只用于“行业横截面统计”排除 ST/新股/不交易样本，**不**在个股时序上挖空。
    - ctx.extra 缓存必须带签名，避免 walk-forward 跨窗口复用陈旧表（cache poisoning）。
    """
    # ---- safe cache with signature ----
    key = "_sector_stage_daily_table_v1"
    df_ts = _to_ts(data["date"])
    sig = (len(data), df_ts.min(), df_ts.max())
    cached_obj = ctx.extra.get(key)
    if isinstance(cached_obj, dict) and cached_obj.get("sig") == sig:
        return cached_obj["data"]

    df = data.copy()
    df["_ts"] = df_ts
    if "valid_cs_mask" in df.columns:
        df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"]).astype(bool)
    else:
        df["valid_cs_mask"] = True

    mp = _get_sector_map(ctx)
    df["sector_code"] = df["stock_code"].astype(str).map(mp).fillna("Unknown")

    # 不在这里 where(mask, nan) 挖空个股序列；统计时用 df_stat 控制即可
    df["pct_chg"] = _num(df.get("pct_chg", np.nan))

    for c in ["is_limit_up", "is_limit_down", "is_zhaban"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype("int64")

    # 只用有效样本做行业统计
    df_stat = df[df["valid_cs_mask"]].copy()
    g = df_stat.groupby(["_ts", "sector_code"], sort=True)

    out = pd.DataFrame({
        "_ts": g.size().index.get_level_values(0),
        "sector_code": g.size().index.get_level_values(1),
        "sector_count": g.size().astype("float64").values,
        "limit_up_cnt": g["is_limit_up"].sum().astype("float64").values,
        "limit_down_cnt": g["is_limit_down"].sum().astype("float64").values,
        "zhaban_cnt": g["is_zhaban"].sum().astype("float64").values,
        "sector_avg_ret": g["pct_chg"].mean().astype("float64").values,
    })

    out["up_ratio"] = _safe_div(out["limit_up_cnt"], out["sector_count"]).fillna(0.0)
    out["dn_ratio"] = _safe_div(out["limit_down_cnt"], out["sector_count"]).fillna(0.0)
    out["zb_ratio"] = _safe_div(out["zhaban_cnt"], out["sector_count"]).fillna(0.0)

    out["sector_emotion_heat"] = out["up_ratio"] - 1.2 * out["dn_ratio"] - 0.6 * out["zb_ratio"]

    # breadth：行业内涨跌结构（阈值 0.2/-0.2 保持你的原语义）
    status = np.where(df_stat["pct_chg"] > 0.2, 1.0, np.where(df_stat["pct_chg"] < -0.2, -1.0, 0.0))
    df_stat["_status"] = status
    breadth = (
        df_stat.groupby(["_ts", "sector_code"], sort=True)["_status"]
        .mean()
        .rename("sector_breadth")
        .reset_index()
    )
    out = out.merge(breadth, on=["_ts", "sector_code"], how="left")

    # HHI（可选）：用 volume 计算行业集中度（只对 valid_cs_mask=True 的样本）
    if "volume" in df_stat.columns:
        df_stat["volume"] = _num(df_stat.get("volume", np.nan)).fillna(0.0)
        total = df_stat.groupby(["_ts", "sector_code"], sort=False)["volume"].transform("sum")
        share = df_stat["volume"] / (total + _EPS)
        hhi = (
            (share * share)
            .groupby([df_stat["_ts"], df_stat["sector_code"]], sort=False)
            .sum()
            .rename("sector_hhi")
            .reset_index()
        )
        out = out.merge(hhi, on=["_ts", "sector_code"], how="left")
    else:
        out["sector_hhi"] = np.nan

    out = out.sort_values(["sector_code", "_ts"]).reset_index(drop=True)

    parts = []
    for _, grp in out.groupby("sector_code", sort=False):
        grp = grp.sort_values("_ts").reset_index(drop=True)

        heat_p = _rolling_last_percentile(grp["sector_emotion_heat"], window=20, min_periods=10)
        breadth_p = _rolling_last_percentile(grp["sector_breadth"], window=20, min_periods=10)
        hhi_p = _rolling_last_percentile(grp["sector_hhi"], window=20, min_periods=10)

        grp["sector_emotion_heat_level_20d"] = _center_rank01(heat_p)
        grp["sector_breadth_level_20d"] = _center_rank01(breadth_p)
        grp["sector_concentration_level_20d"] = _center_rank01(hhi_p)

        grp["sector_emotion_heat_accel_3d"] = grp["sector_emotion_heat"] - grp["sector_emotion_heat"].shift(3)

        zb_up = grp["zb_ratio"] - grp["zb_ratio"].shift(3)
        grp["sector_divergence_risk"] = grp["sector_emotion_heat_level_20d"].clip(0, 1) * _num(zb_up)

        up_med60 = grp["up_ratio"].rolling(60, min_periods=20).median()
        strong = ((grp["sector_emotion_heat"] > 0.0) & (grp["up_ratio"] > up_med60)).astype(float)
        grp["sector_streak_strength_10d"] = strong.rolling(10, min_periods=5).mean()

        # drawdown from peak (10d) on sector index proxy
        r = _num(grp["sector_avg_ret"])
        scale = 0.01 if np.nanmedian(np.abs(r.values)) > 1.5 else 1.0
        rp = r.fillna(0.0) * scale
        idx = (1.0 + rp).cumprod()
        peak = idx.rolling(10, min_periods=5).max()
        grp["sector_drawdown_from_peak_10d"] = idx / (peak + _EPS) - 1.0

        for c in [
            "sector_emotion_heat_level_20d",
            "sector_emotion_heat_accel_3d",
            "sector_divergence_risk",
            "sector_breadth_level_20d",
            "sector_concentration_level_20d",
            "sector_streak_strength_10d",
            "sector_drawdown_from_peak_10d",
        ]:
            grp[c] = _num(grp[c]).fillna(0.0)

        parts.append(grp)

    stage = pd.concat(parts, axis=0, ignore_index=True)

    keep = [
        "_ts", "sector_code",
        "sector_emotion_heat_level_20d",
        "sector_emotion_heat_accel_3d",
        "sector_divergence_risk",
        "sector_breadth_level_20d",
        "sector_concentration_level_20d",
        "sector_streak_strength_10d",
        "sector_drawdown_from_peak_10d",
    ]
    stage = stage[keep].sort_values(["sector_code", "_ts"]).reset_index(drop=True)

    ctx.extra[key] = {"sig": sig, "data": stage}
    return stage

def _map_sector_stage_to_stock(data: pd.DataFrame, ctx: FactorContext, col: str) -> pd.Series:
    """
    将行业日度表按 (_ts, sector_code) 映射回个股行，保持与 data.index 一一对应。
    这里 **不能** 直接 df.merge(...).reindex(data.index)，否则会因 index label 对齐导致静默全 NaN。
    """
    stage = _sector_daily_stage_table(data, ctx)
    df = data.copy()
    df["_ts"] = _to_ts(df["date"])
    mp = _get_sector_map(ctx)
    df["sector_code"] = df["stock_code"].astype(str).map(mp).fillna("Unknown")

    stage_s = stage.set_index(["_ts", "sector_code"])[col]

    key = pd.MultiIndex.from_arrays([df["_ts"].values, df["sector_code"].values])
    mapped = stage_s.reindex(key).to_numpy()

    raw = pd.Series(mapped, index=data.index, dtype="float64")
    return _cs_robust_z(raw, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)

# ==============================================================================
# Stock breakout (structure)
# ==============================================================================
def _breakout_z(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if "close" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype="float64")

    df = data[["stock_code", "date", "close"]].copy()
    df["_ts"] = _to_ts(df["date"])
    df["close"] = _num(df["close"])

    if "high" in data.columns:
        df["high"] = _num(data["high"])
    else:
        df["high"] = df["close"]

    df.sort_values(["stock_code", "_ts"], inplace=True)

    roll_max = df.groupby("stock_code", sort=False)["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    raw = df["close"] / (roll_max + _EPS) - 1.0

    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=_vmask(data)).reindex(data.index)
    return out.astype("float64")


# ==============================================================================
# Factor calculators
# ==============================================================================
def _sector_emotion_heat_level_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _map_sector_stage_to_stock(data, ctx, "sector_emotion_heat_level_20d")


def _sector_emotion_heat_accel_3d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _map_sector_stage_to_stock(data, ctx, "sector_emotion_heat_accel_3d")


def _sector_divergence_risk(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _map_sector_stage_to_stock(data, ctx, "sector_divergence_risk")


def _sector_breadth_level_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _map_sector_stage_to_stock(data, ctx, "sector_breadth_level_20d")


def _sector_concentration_level_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _map_sector_stage_to_stock(data, ctx, "sector_concentration_level_20d")


def _sector_streak_strength_10d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _map_sector_stage_to_stock(data, ctx, "sector_streak_strength_10d")


def _sector_drawdown_from_peak_10d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _map_sector_stage_to_stock(data, ctx, "sector_drawdown_from_peak_10d")


# ==============================================================================
# Register factors
# ==============================================================================
sector_emotion_heat_level_20d = register_factor(FactorDefinition(
    name="sector_emotion_heat_level_20d",
    required_columns=_REQ_BASE,
    requires_sector_data=True,
    lookback_days=40,
    calculator=_sector_emotion_heat_level_20d,
    description="行业情绪热度 20日历史分位（[-1,1]），映射到股票；按日robust z",
))

sector_emotion_heat_accel_3d = register_factor(FactorDefinition(
    name="sector_emotion_heat_accel_3d",
    required_columns=_REQ_BASE,
    requires_sector_data=True,
    lookback_days=10,
    calculator=_sector_emotion_heat_accel_3d,
    description="行业情绪热度 3日加速度，映射到股票；按日robust z",
))

sector_divergence_risk = register_factor(FactorDefinition(
    name="sector_divergence_risk",
    required_columns=_REQ_BASE,
    requires_sector_data=True,
    lookback_days=40,
    calculator=_sector_divergence_risk,
    description="行业末期分歧风险：高热度(分位) × 炸板占比上升（3日）",
))

sector_breadth_level_20d = register_factor(FactorDefinition(
    name="sector_breadth_level_20d",
    required_columns=_REQ_BASE,
    requires_sector_data=True,
    lookback_days=40,
    calculator=_sector_breadth_level_20d,
    description="行业广度 20日历史分位（扩散程度），映射到股票；按日robust z",
))

sector_concentration_level_20d = register_factor(FactorDefinition(
    name="sector_concentration_level_20d",
    required_columns=_REQ_BASE.union({"volume"}),
    requires_sector_data=True,
    lookback_days=40,
    calculator=_sector_concentration_level_20d,
    description="行业成交集中度(HHI) 20日历史分位（抱团/过热），映射到股票；按日robust z",
))

sector_streak_strength_10d = register_factor(FactorDefinition(
    name="sector_streak_strength_10d",
    required_columns=_REQ_BASE,
    requires_sector_data=True,
    lookback_days=80,
    calculator=_sector_streak_strength_10d,
    description="行业强势连贯性：近10日强势天数占比（强势=heat>0 且 up_ratio 高于60日中位）",
))

sector_drawdown_from_peak_10d = register_factor(FactorDefinition(
    name="sector_drawdown_from_peak_10d",
    required_columns=_REQ_BASE,
    requires_sector_data=True,
    lookback_days=30,
    calculator=_sector_drawdown_from_peak_10d,
    description="行业指数代理的10日回撤（衰竭/见顶后回撤），映射到股票；按日robust z",
))

breakout_z = register_factor(FactorDefinition(
    name="breakout_z",
    required_columns={"stock_code", "date", "close"},
    lookback_days=30,
    calculator=_breakout_z,
    description="个股突破强度：close / rolling_max(high,20) - 1；按日robust z",
))



def _vmask(data: pd.DataFrame) -> pd.Series:
    """Return boolean valid cross-section mask aligned to `data.index`."""
    if isinstance(data, pd.DataFrame) and ("valid_cs_mask" in data.columns):
        return _clean_bool(data["valid_cs_mask"]).astype(bool)
    return pd.Series(True, index=data.index)

__all__ = [
    "sector_emotion_heat_level_20d",
    "sector_emotion_heat_accel_3d",
    "sector_divergence_risk",
    "sector_breadth_level_20d",
    "sector_concentration_level_20d",
    "sector_streak_strength_10d",
    "sector_drawdown_from_peak_10d",
    "breakout_z",
]