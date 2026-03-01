"""
动量交互与爆发因子（valid_cs_mask 适配 + float32）

对 moment_interaction_updated_short.py 的工程化优化：
- 时序纯净：所有 rolling/shift 按 (stock_code, _ts) 排序，严格因果
- 横截面统一：需要入模的因子最终按日 CS robust z（median/MAD）+ clip + fillna(0)
- valid_cs_mask：
  * 参与 CS robust z 的统计仅使用 mask=True 样本（ST/新股等不污染截面）
  * mask=False 的行最终输出置 0（中性）
- 输出 dtype：统一 float32（内部仍用 float64）

包含因子（名称保持不变）：
- leader_acceleration_score
- vol_price_explosion_score
- dragon_resonance_score
- ma5_ratio_acceleration
- sector_rank_stability_3d
- ma5_ratio_risk_penalty
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


def _prep(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["_ts"] = _to_ts(df["date"])
    df["stock_code"] = df["stock_code"].astype(str)
    if "valid_cs_mask" in df.columns:
        df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"])
    else:
        df["valid_cs_mask"] = True
    df.sort_values(["stock_code", "_ts"], inplace=True)
    return df


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


# ==============================================================================
# Components
# ==============================================================================
def _calc_components(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算三角组件：
    - rank_z：5日动量（sum pct_chg）按日 robust z
    - ma5_ratio：MA5 乖离率（百分比）
    - abnormal_turnover：换手/自身20日均值（clip）
    """
    g = df.groupby("stock_code", sort=False)

    # A) 5d momentum
    mom_5d = g["pct_chg"].transform(lambda s: _num64(s).rolling(5, min_periods=3).sum())
    rank_z = _cs_robust_z(mom_5d, df["_ts"], valid_cs_mask=df["valid_cs_mask"])

    # B) MA5 ratio
    close = _num64(df["close"])
    ma5 = close.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(5, min_periods=2).mean())
    ma5_ratio = (close - ma5) / (ma5 + _EPS) * 100.0
    ma5_ratio = ma5_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # C) abnormal turnover
    if "turnover_rate" in df.columns:
        to = _num64(df["turnover_rate"]).clip(lower=0.0)
        ma20 = to.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=10).mean())
        abnormal = to / (ma20 + _EPS)
        abnormal = abnormal.replace([np.inf, -np.inf], np.nan).clip(0.0, 15.0).fillna(1.0)
    else:
        abnormal = pd.Series(1.0, index=df.index, dtype="float64")

    # mask=False 不参与统计，且最终置 0（中性）
    ma5_ratio = ma5_ratio.where(df["valid_cs_mask"], 0.0)
    abnormal = abnormal.where(df["valid_cs_mask"], 1.0)  # 作为乘子中性

    return rank_z.astype("float64"), ma5_ratio.astype("float64"), abnormal.astype("float64")


# ==============================================================================
# Factor calculators
# ==============================================================================
def _leader_acceleration_score(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "pct_chg", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df = _prep(data[list(need) | {"valid_cs_mask"} if "valid_cs_mask" in data.columns else list(need)].copy())
    rank_z, ratio, _ab = _calc_components(df)
    raw = rank_z.astype("float64") * ratio
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


def _vol_price_explosion_score(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "turnover_rate", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    cols = ["stock_code", "date", "turnover_rate", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _prep(data[cols].copy())
    # pct_chg is not required here; use ratio/abnormal only
    # build ratio/abnormal with close + turnover_rate
    # ratio:
    close = _num64(df["close"])
    ma5 = close.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(5, min_periods=2).mean())
    ratio = (close - ma5) / (ma5 + _EPS) * 100.0
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0).where(df["valid_cs_mask"], 0.0)

    to = _num64(df["turnover_rate"]).clip(lower=0.0)
    ma20 = to.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=10).mean())
    abnormal = (to / (ma20 + _EPS)).replace([np.inf, -np.inf], np.nan).clip(0.0, 15.0).fillna(1.0)
    abnormal = abnormal.where(df["valid_cs_mask"], 1.0)

    raw = abnormal * ratio
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


def _dragon_resonance_score(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "pct_chg", "close", "turnover_rate"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    cols = ["stock_code", "date", "pct_chg", "close", "turnover_rate"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _prep(data[cols].copy())
    rank_z, ratio, abnormal = _calc_components(df)
    raw = rank_z.astype("float64") * ratio * abnormal
    out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


def _ma5_ratio_acceleration(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _prep(data[cols].copy())

    close = _num64(df["close"])
    ma5 = close.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(5, min_periods=2).mean())
    ratio = (close - ma5) / (ma5 + _EPS) * 100.0
    prev = ratio.groupby(df["stock_code"], sort=False).shift(1)
    acc = (ratio - prev).replace([np.inf, -np.inf], np.nan)

    out = _cs_robust_z(acc, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


def _sector_rank_stability_3d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    这里的“rank”仍采用全市场 5d 动量 robust z（不是行业内），
    稳定性用 3 日均值来过滤一日游。
    """
    need = {"stock_code", "date", "pct_chg"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    cols = ["stock_code", "date", "pct_chg"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _prep(data[cols].copy())

    g = df.groupby("stock_code", sort=False)
    mom_5d = g["pct_chg"].transform(lambda s: _num64(s).rolling(5, min_periods=3).sum())
    rank_z = _cs_robust_z(mom_5d, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).astype("float64")

    stab = rank_z.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(3, min_periods=1).mean())
    out = _cs_robust_z(stab, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


def _ma5_ratio_risk_penalty(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df = _prep(data[cols].copy())

    close = _num64(df["close"])
    ma5 = close.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(5, min_periods=2).mean())
    ratio = (close - ma5) / (ma5 + _EPS) * 100.0
    penalty = (ratio * ratio).replace([np.inf, -np.inf], np.nan)

    out = _cs_robust_z(penalty, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(data.index)
    return out.astype(_OUT_DTYPE)


# ==============================================================================
# Register factors
# ==============================================================================
leader_acceleration_score = register_factor(FactorDefinition(
    name="leader_acceleration_score",
    required_columns={"stock_code", "date", "pct_chg", "close"},
    lookback_days=10,
    calculator=_leader_acceleration_score,
    description="龙头加速（动量robust z × MA5乖离；valid_cs_mask；float32）",
))

vol_price_explosion_score = register_factor(FactorDefinition(
    name="vol_price_explosion_score",
    required_columns={"stock_code", "date", "turnover_rate", "close"},
    lookback_days=25,
    calculator=_vol_price_explosion_score,
    description="量价爆发（异常换手 × MA5乖离；valid_cs_mask；float32）",
))

dragon_resonance_score = register_factor(FactorDefinition(
    name="dragon_resonance_score",
    required_columns={"stock_code", "date", "pct_chg", "close", "turnover_rate"},
    lookback_days=25,
    calculator=_dragon_resonance_score,
    description="真龙共振（三角共振；valid_cs_mask；float32）",
))

ma5_ratio_acceleration = register_factor(FactorDefinition(
    name="ma5_ratio_acceleration",
    required_columns={"stock_code", "date", "close"},
    lookback_days=10,
    calculator=_ma5_ratio_acceleration,
    description="MA5乖离加速度（valid_cs_mask；float32）",
))

sector_rank_stability_3d = register_factor(FactorDefinition(
    name="sector_rank_stability_3d",
    required_columns={"stock_code", "date", "pct_chg"},
    lookback_days=10,
    calculator=_sector_rank_stability_3d,
    description="龙头地位稳定性（3日均值；valid_cs_mask；float32）",
))

ma5_ratio_risk_penalty = register_factor(FactorDefinition(
    name="ma5_ratio_risk_penalty",
    required_columns={"stock_code", "date", "close"},
    lookback_days=10,
    calculator=_ma5_ratio_risk_penalty,
    description="超买风险惩罚（乖离平方；valid_cs_mask；float32）",
))

__all__ = [
    "leader_acceleration_score",
    "vol_price_explosion_score",
    "dragon_resonance_score",
    "ma5_ratio_acceleration",
    "sector_rank_stability_3d",
    "ma5_ratio_risk_penalty",
]
