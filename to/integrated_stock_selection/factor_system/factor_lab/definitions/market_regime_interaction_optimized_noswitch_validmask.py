"""
Market Regime × Stock Signal 交互因子 - valid_cs_mask 适配版（输出 float32）

在 market_regime_interaction_optimized_noswitch.py 基础上：
- 增加 valid_cs_mask：
  1) 市场 regime 统计口径（adv/dec、涨跌停、炸板、成交额等）只用 mask=True 样本聚合
  2) 因子最终按日 CS robust z 的 median/MAD 只用 mask=True 样本统计
  3) mask=False 行输出置 0（中性）
- 输出统一 float32（内部 float64）

因子名保持不变（8个）：
- mrk_emotion_x_pullback_quality
- mrk_limit_x_breakout20_quality
- mrk_breadth_x_relstrength_5d
- mrk_volreg_x_lowvol_trend_20d
- mrk_liqreg_x_volume_surge
- mrk_emotion_x_intraday_failure
- mrk_breadth_x_gap_surprise
- mrk_limit_x_reversal_risk
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
# Common utils
# ==============================================================================
def _to_ts(date_series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(date_series):
        ts = pd.to_datetime(date_series, errors="coerce")
    else:
        ds = date_series.astype(str).str.replace("-", "", regex=False)
        ts = pd.to_datetime(ds, format="%Y%m%d", errors="coerce")
    return ts.dt.normalize()


def _attach_sorted(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["_ts"] = _to_ts(x["date"])
    x["stock_code"] = x["stock_code"].astype(str)
    x.sort_values(["stock_code", "_ts"], inplace=True)
    return x


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


# ==============================================================================
# Market regime builders (computed from stock-level panel per date)
# ==============================================================================
def _safe_bool_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if s.dtype == bool:
                return s
            try:
                return pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.5
            except Exception:
                continue
    return None


def _approx_limit_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pct = _num64(df["pct_chg"]) if "pct_chg" in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")
    up = (pct >= 9.5) | (pct >= 19.0)
    dn = (pct <= -9.5) | (pct <= -19.0)
    return up.fillna(False), dn.fillna(False)


def _market_daily_series(data: pd.DataFrame) -> pd.DataFrame:
    """
    返回每个交易日的 market regime raw series（index 为 _ts）。
    注意：聚合仅使用 valid_cs_mask=True 的样本，避免 ST/新股污染市场统计。
    """
    tmp = data.copy()
    tmp["_ts"] = _to_ts(tmp["date"])
    if "valid_cs_mask" in tmp.columns:
        m = _clean_bool(tmp["valid_cs_mask"])
        tmp = tmp[m].copy()
    # 若过滤后为空，直接返回空表
    if tmp.empty:
        return pd.DataFrame(columns=["adv_dec", "limit_balance", "emotion_heat", "mkt_ret_mean", "vol_reg", "liq_reg"])

    if "pct_chg" in tmp.columns:
        pct = _num64(tmp["pct_chg"])
        ret = pct / 100.0
    else:
        ret = pd.Series(np.nan, index=tmp.index, dtype="float64")

    adv = (ret > 0).astype("float64")
    dec = (ret < 0).astype("float64")
    n = tmp.groupby("_ts", sort=False).size().astype("float64")
    adv_dec = (adv.groupby(tmp["_ts"], sort=False).sum() - dec.groupby(tmp["_ts"], sort=False).sum()) / (n + _EPS)

    is_up = _safe_bool_col(tmp, ["is_limit_up", "limit_up", "limit_up_flag"])
    is_dn = _safe_bool_col(tmp, ["is_limit_down", "limit_down", "limit_down_flag"])
    is_zb = _safe_bool_col(tmp, ["is_zhaban", "zhaban", "open_board", "is_open_board"])

    if is_up is None or is_dn is None:
        up2, dn2 = _approx_limit_flags(tmp)
        is_up = up2 if is_up is None else is_up
        is_dn = dn2 if is_dn is None else is_dn

    if is_zb is None:
        if {"high", "close", "pct_chg"}.issubset(set(tmp.columns)):
            pct2 = _num64(tmp["pct_chg"])
            up_touch = (pct2 >= 8.5) | (pct2 >= 18.0)
            not_up = ~is_up
            is_zb = (up_touch & not_up).fillna(False)
        else:
            is_zb = pd.Series(False, index=tmp.index)

    up_ratio = is_up.groupby(tmp["_ts"], sort=False).mean().astype("float64")
    dn_ratio = is_dn.groupby(tmp["_ts"], sort=False).mean().astype("float64")
    zb_ratio = is_zb.groupby(tmp["_ts"], sort=False).mean().astype("float64")
    limit_balance = (up_ratio - dn_ratio).astype("float64")

    emotion_heat = (2.0 * limit_balance + 0.6 * adv_dec - 1.2 * zb_ratio).astype("float64")

    mkt_ret_mean = ret.groupby(tmp["_ts"], sort=False).mean().astype("float64")
    vol_reg = mkt_ret_mean.rolling(10, min_periods=8).std()

    if "volume" in tmp.columns:
        vol = _num64(tmp["volume"]).clip(lower=0.0)
        mkt_vol_sum = vol.groupby(tmp["_ts"], sort=False).sum().astype("float64")
        mu = mkt_vol_sum.rolling(10, min_periods=8).mean()
        sd = mkt_vol_sum.rolling(10, min_periods=8).std()
        liq_reg = (mkt_vol_sum - mu) / (sd + _EPS)
    else:
        liq_reg = pd.Series(np.nan, index=mkt_ret_mean.index, dtype="float64")

    return pd.DataFrame(
        {
            "adv_dec": adv_dec,
            "limit_balance": limit_balance,
            "emotion_heat": emotion_heat,
            "mkt_ret_mean": mkt_ret_mean,
            "vol_reg": vol_reg,
            "liq_reg": liq_reg,
        }
    )


def _map_daily_to_rows(data: pd.DataFrame, daily: pd.DataFrame, col: str) -> pd.Series:
    ts = _to_ts(data["date"])
    s = ts.map(daily[col])
    return pd.to_numeric(s, errors="coerce").astype("float64")


# ==============================================================================
# Stock-side primitives (computed from OHLCV only)
# ==============================================================================
def _stock_pullback_quality(data: pd.DataFrame) -> pd.Series:
    need = {"stock_code", "date", "close", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = _attach_sorted(data[["stock_code", "date"]].copy())
    close = _num64(data["close"]).reindex(df.index)
    vol = _num64(data["volume"]).reindex(df.index)
    g = df["stock_code"]

    prev5 = close.groupby(g, sort=False).shift(5)
    ret5 = (close - prev5) / (prev5 + _EPS)
    pullback = -ret5.clip(upper=0.0)

    vol3 = vol.groupby(g, sort=False).transform(lambda s: s.rolling(3, min_periods=2).mean())
    vol10 = vol.groupby(g, sort=False).transform(lambda s: s.rolling(10, min_periods=6).mean())
    contraction = 1.0 - vol3 / (vol10 + _EPS)

    ma5 = close.groupby(g, sort=False).transform(lambda s: s.rolling(5, min_periods=3).mean())
    support = (close > ma5).astype("float64")

    return (-pullback + contraction + support).reindex(data.index)


def _stock_breakout20(data: pd.DataFrame) -> pd.Series:
    need = {"stock_code", "date", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = _attach_sorted(data[["stock_code", "date"]].copy())
    close = _num64(data["close"]).reindex(df.index)
    g = df["stock_code"]

    high20 = close.groupby(g, sort=False).transform(lambda s: s.rolling(20, min_periods=12).max())
    raw = close / (high20 + _EPS) - 1.0

    std20 = raw.groupby(g, sort=False).transform(lambda s: s.rolling(20, min_periods=12).std())
    z = raw / (std20 + _EPS)
    return z.reindex(data.index)


def _stock_relstrength_5d(data: pd.DataFrame, mkt_ret_mean: pd.Series) -> pd.Series:
    need = {"stock_code", "date", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = _attach_sorted(data[["stock_code", "date"]].copy())
    close = _num64(data["close"]).reindex(df.index)
    g = df["stock_code"]

    prev5 = close.groupby(g, sort=False).shift(5)
    ret5 = (close - prev5) / (prev5 + _EPS)

    ts = _to_ts(data["date"])
    mkt = ts.map(mkt_ret_mean).astype("float64")
    raw = ret5.reindex(data.index) - mkt.rolling(5, min_periods=3).mean().reindex(data.index)
    return raw


def _stock_lowvol_trend_20d(data: pd.DataFrame) -> pd.Series:
    need = {"stock_code", "date", "close", "pct_chg"}
    if not need.issubset(set(data.columns)):
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = _attach_sorted(data[["stock_code", "date"]].copy())
    close = _num64(data["close"]).reindex(df.index)
    pct = _num64(data["pct_chg"]).reindex(df.index)
    g = df["stock_code"]

    prev20 = close.groupby(g, sort=False).shift(20)
    ret20 = (close - prev20) / (prev20 + _EPS)
    vol20 = pct.groupby(g, sort=False).transform(lambda s: s.rolling(20, min_periods=12).std()) / 100.0
    return (ret20 / (vol20 + _EPS)).reindex(data.index)


def _stock_volume_surge(data: pd.DataFrame) -> pd.Series:
    need = {"stock_code", "date", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = _attach_sorted(data[["stock_code", "date"]].copy())
    vol = _num64(data["volume"]).reindex(df.index).clip(lower=0.0)
    g = df["stock_code"]

    ma20 = vol.groupby(g, sort=False).transform(lambda s: s.rolling(20, min_periods=12).mean())
    return (vol / (ma20 + _EPS) - 1.0).reindex(data.index)


def _stock_intraday_failure(data: pd.DataFrame) -> pd.Series:
    need = {"date", "high", "low", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(np.nan, index=data.index, dtype="float64")

    high = _num64(data["high"])
    low = _num64(data["low"])
    close = _num64(data["close"])
    drawdown = (close - high) / (high + _EPS)
    rng = (high - low)
    clv = (close - low) / (rng + _EPS)
    clv = clv.where(rng.abs() >= 1e-6, 1.0).clip(0.0, 1.0)
    return (-drawdown + (1.0 - clv)).astype("float64")


def _stock_gap_surprise(data: pd.DataFrame) -> pd.Series:
    need = {"stock_code", "date", "open", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = _attach_sorted(data[["stock_code", "date"]].copy())
    open_ = _num64(data["open"]).reindex(df.index)
    close = _num64(data["close"]).reindex(df.index)
    g = df["stock_code"]

    prev_close = close.groupby(g, sort=False).shift(1)
    gap = (open_ - prev_close) / (prev_close + _EPS)
    ret1 = close.groupby(g, sort=False).pct_change(1)
    vol5 = ret1.groupby(g, sort=False).transform(lambda s: s.rolling(5, min_periods=3).std())
    return (gap / (vol5 + _EPS)).clip(-8.0, 8.0).reindex(data.index)


def _stock_reversal_risk(data: pd.DataFrame) -> pd.Series:
    need = {"stock_code", "date", "close", "pct_chg", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = _attach_sorted(data[["stock_code", "date"]].copy())
    close = _num64(data["close"]).reindex(df.index)
    pct = _num64(data["pct_chg"]).reindex(df.index)
    vol = _num64(data["volume"]).reindex(df.index).clip(lower=0.0)
    g = df["stock_code"]

    prev3 = close.groupby(g, sort=False).shift(3)
    ret3 = (close - prev3) / (prev3 + _EPS)
    vol3 = pct.groupby(g, sort=False).transform(lambda s: s.rolling(3, min_periods=2).std()) / 100.0
    surge = vol / (vol.groupby(g, sort=False).transform(lambda s: s.rolling(10, min_periods=6).mean()) + _EPS) - 1.0
    return (ret3 / (vol3 + _EPS) + 0.3 * surge).reindex(data.index)


# ==============================================================================
# Factor calculators
# ==============================================================================
def _daily_cache(data: pd.DataFrame) -> pd.DataFrame:
    return _market_daily_series(data)


def _vmask(data: pd.DataFrame) -> pd.Series | None:
    return data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None


def _mrk_emotion_x_pullback_quality(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    daily = _daily_cache(data)
    raw = _map_daily_to_rows(data, daily, "emotion_heat") * _stock_pullback_quality(data)
    return _cs_robust_z(raw, _to_ts(data["date"]), valid_cs_mask=_vmask(data))


def _mrk_limit_x_breakout20_quality(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    daily = _daily_cache(data)
    lb = _map_daily_to_rows(data, daily, "limit_balance")
    breakout = _stock_breakout20(data)
    pullq = _stock_pullback_quality(data)
    raw = lb * breakout * (1.0 + 0.5 * pullq)
    return _cs_robust_z(raw, _to_ts(data["date"]), valid_cs_mask=_vmask(data))


def _mrk_breadth_x_relstrength_5d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    daily = _daily_cache(data)
    adv_dec = _map_daily_to_rows(data, daily, "adv_dec")
    rels = _stock_relstrength_5d(data, daily["mkt_ret_mean"])
    raw = adv_dec * rels
    return _cs_robust_z(raw, _to_ts(data["date"]), valid_cs_mask=_vmask(data))


def _mrk_volreg_x_lowvol_trend_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    daily = _daily_cache(data)
    volreg = _map_daily_to_rows(data, daily, "vol_reg")
    lvtrend = _stock_lowvol_trend_20d(data)
    raw = (-volreg) * lvtrend
    return _cs_robust_z(raw, _to_ts(data["date"]), valid_cs_mask=_vmask(data))


def _mrk_liqreg_x_volume_surge(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    daily = _daily_cache(data)
    liq = _map_daily_to_rows(data, daily, "liq_reg")
    surge = _stock_volume_surge(data)
    raw = liq * surge
    return _cs_robust_z(raw, _to_ts(data["date"]), valid_cs_mask=_vmask(data))


def _mrk_emotion_x_intraday_failure(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    daily = _daily_cache(data)
    emotion = _map_daily_to_rows(data, daily, "emotion_heat")
    fail = _stock_intraday_failure(data)
    raw = (-emotion) * fail
    return _cs_robust_z(raw, _to_ts(data["date"]), valid_cs_mask=_vmask(data))


def _mrk_breadth_x_gap_surprise(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    daily = _daily_cache(data)
    adv_dec = _map_daily_to_rows(data, daily, "adv_dec")
    gap = _stock_gap_surprise(data)
    raw = adv_dec * gap
    return _cs_robust_z(raw, _to_ts(data["date"]), valid_cs_mask=_vmask(data))


def _mrk_limit_x_reversal_risk(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    daily = _daily_cache(data)
    lb = _map_daily_to_rows(data, daily, "limit_balance")
    risk = _stock_reversal_risk(data)
    raw = (-lb) * risk
    return _cs_robust_z(raw, _to_ts(data["date"]), valid_cs_mask=_vmask(data))


# ==============================================================================
# Register
# ==============================================================================
mrk_emotion_x_pullback_quality = register_factor(FactorDefinition(
    name="mrk_emotion_x_pullback_quality",
    required_columns={"stock_code", "date", "close", "volume"},
    lookback_days=70,
    calculator=_mrk_emotion_x_pullback_quality,
    description="市场情绪 × 回撤质量（valid_cs_mask；float32）",
))

mrk_limit_x_breakout20_quality = register_factor(FactorDefinition(
    name="mrk_limit_x_breakout20_quality",
    required_columns={"stock_code", "date", "close", "volume", "pct_chg"},
    lookback_days=70,
    calculator=_mrk_limit_x_breakout20_quality,
    description="涨跌停平衡 × 20日突破 × 回撤门控（valid_cs_mask；float32）",
))

mrk_breadth_x_relstrength_5d = register_factor(FactorDefinition(
    name="mrk_breadth_x_relstrength_5d",
    required_columns={"stock_code", "date", "close", "pct_chg"},
    lookback_days=40,
    calculator=_mrk_breadth_x_relstrength_5d,
    description="adv-dec × 个股5日相对强弱（valid_cs_mask；float32）",
))

mrk_volreg_x_lowvol_trend_20d = register_factor(FactorDefinition(
    name="mrk_volreg_x_lowvol_trend_20d",
    required_columns={"stock_code", "date", "close", "pct_chg"},
    lookback_days=70,
    calculator=_mrk_volreg_x_lowvol_trend_20d,
    description="市场波动regime × 低波趋势（valid_cs_mask；float32）",
))

mrk_liqreg_x_volume_surge = register_factor(FactorDefinition(
    name="mrk_liqreg_x_volume_surge",
    required_columns={"stock_code", "date", "volume"},
    lookback_days=40,
    calculator=_mrk_liqreg_x_volume_surge,
    description="市场流动性regime × 量能突变（valid_cs_mask；float32）",
))

mrk_emotion_x_intraday_failure = register_factor(FactorDefinition(
    name="mrk_emotion_x_intraday_failure",
    required_columns={"stock_code", "date", "high", "low", "close"},
    lookback_days=10,
    calculator=_mrk_emotion_x_intraday_failure,
    description="情绪 × 日内冲高回落失败（valid_cs_mask；float32）",
))

mrk_breadth_x_gap_surprise = register_factor(FactorDefinition(
    name="mrk_breadth_x_gap_surprise",
    required_columns={"stock_code", "date", "open", "close"},
    lookback_days=20,
    calculator=_mrk_breadth_x_gap_surprise,
    description="adv-dec × 隔夜缺口 surprise（valid_cs_mask；float32）",
))

mrk_limit_x_reversal_risk = register_factor(FactorDefinition(
    name="mrk_limit_x_reversal_risk",
    required_columns={"stock_code", "date", "close", "pct_chg", "volume"},
    lookback_days=30,
    calculator=_mrk_limit_x_reversal_risk,
    description="涨跌停平衡 × 反转风险惩罚（valid_cs_mask；float32）",
))

__all__ = [
    "mrk_emotion_x_pullback_quality",
    "mrk_limit_x_breakout20_quality",
    "mrk_breadth_x_relstrength_5d",
    "mrk_volreg_x_lowvol_trend_20d",
    "mrk_liqreg_x_volume_surge",
    "mrk_emotion_x_intraday_failure",
    "mrk_breadth_x_gap_surprise",
    "mrk_limit_x_reversal_risk",
]
