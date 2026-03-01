"""Market breadth + limit/zhaban cycle summary factors (light version).

This module builds daily market-wide series from per-stock daily fields:
- adv/dec breadth (based on pct_chg)
- limit-up / limit-down / zhaban ratios
- a compact "emotion heat" proxy

Then it derives a small set of short-horizon transforms (MA3, Z10, Delta3)
plus one 32-day spectral summary for emotion_heat.

Design goal: keep factor count low while still providing
(1) contemporaneous breadth/limit regime
(2) short-horizon regime change
(3) a compact cycle proxy.

Notes
-----
- These are *market-level* features: within a given date they are identical
  for all stocks. Tree models can still learn interactions with stock factors.
- Lookbacks are chosen to align with a 3-day forward-weighted label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


# -----------------------------
# helpers
# -----------------------------

_REQUIRED_COLS_LIST = {"date", "stock_code", "pct_chg", "is_limit_up", "is_limit_down", "is_zhaban"}
_REQUIRED_COLS = _REQUIRED_COLS_LIST


def _as_date_series(x: pd.Series) -> pd.Series:
    if np.issubdtype(x.dtype, np.datetime64):
        return pd.to_datetime(x).dt.date
    return pd.to_datetime(x, errors="coerce").dt.date


def _ctx_cache(ctx: FactorContext) -> Dict:
    """获取一个可写的缓存字典（兼容不同 FactorContext 实现）。

    优先使用 ctx.cache / ctx.extra；若不存在则挂在 ctx._tmp_cache 上。
    """
    if hasattr(ctx, "cache") and isinstance(getattr(ctx, "cache"), dict):
        return getattr(ctx, "cache")
    if hasattr(ctx, "extra") and isinstance(getattr(ctx, "extra"), dict):
        return getattr(ctx, "extra")
    if not hasattr(ctx, "_tmp_cache") or getattr(ctx, "_tmp_cache") is None:
        setattr(ctx, "_tmp_cache", {})
    return getattr(ctx, "_tmp_cache")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.replace(0, np.nan))


def _build_market_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Build daily market aggregates from per-stock daily data."""
    x = df[list(_REQUIRED_COLS_LIST)].copy()
    x["date"] = _as_date_series(x["date"])

    # Ensure numeric
    x["pct_chg"] = pd.to_numeric(x["pct_chg"], errors="coerce")
    for c in ["is_limit_up", "is_limit_down", "is_zhaban"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype(int)

    g = x.groupby("date", sort=True)
    cnt = g.size().rename("mkt_count")

    adv = g.apply(lambda d: (d["pct_chg"] > 0).sum()).rename("mkt_adv")
    dec = g.apply(lambda d: (d["pct_chg"] < 0).sum()).rename("mkt_dec")

    limit_up = g["is_limit_up"].sum().rename("mkt_limit_up")
    limit_down = g["is_limit_down"].sum().rename("mkt_limit_down")
    zhaban = g["is_zhaban"].sum().rename("mkt_zhaban")

    out = pd.concat([cnt, adv, dec, limit_up, limit_down, zhaban], axis=1).reset_index()

    out["mkt_adv_dec"] = _safe_div(out["mkt_adv"] - out["mkt_dec"], out["mkt_count"]).fillna(0.0)
    out["mkt_limit_balance"] = _safe_div(out["mkt_limit_up"] - out["mkt_limit_down"], out["mkt_count"]).fillna(0.0)
    out["mkt_limit_up_ratio"] = _safe_div(out["mkt_limit_up"], out["mkt_count"]).fillna(0.0)
    out["mkt_limit_down_ratio"] = _safe_div(out["mkt_limit_down"], out["mkt_count"]).fillna(0.0)
    out["mkt_zhaban_ratio"] = _safe_div(out["mkt_zhaban"], out["mkt_count"]).fillna(0.0)

    # A compact "emotion heat" proxy:
    #   - reward limit-up dominance
    #   - penalize limit-down and zhaban
    out["mkt_emotion_heat"] = (
        out["mkt_limit_up_ratio"]
        - 1.2 * out["mkt_limit_down_ratio"]
        - 0.6 * out["mkt_zhaban_ratio"]
    )

    keep = [
        "date",
        "mkt_adv_dec",
        "mkt_limit_balance",
        "mkt_limit_up_ratio",
        "mkt_limit_down_ratio",
        "mkt_zhaban_ratio",
        "mkt_emotion_heat",
    ]
    return out[keep].sort_values("date").reset_index(drop=True)


def _rolling_spectral_entropy_and_hf_ratio(x: np.ndarray) -> tuple[float, float, float, float]:
    """Return (dom_period, dom_strength, spec_entropy, hf_ratio) for a 1D window."""
    if x.size < 8 or np.all(~np.isfinite(x)):
        return (np.nan, np.nan, np.nan, np.nan)

    s = np.asarray(x, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 8:
        return (np.nan, np.nan, np.nan, np.nan)

    # Detrend + window
    s = s - np.mean(s)
    win = np.hanning(s.size)
    s = s * win

    # Real FFT
    fft = np.fft.rfft(s)
    psd = (np.abs(fft) ** 2)

    # Remove DC
    if psd.size <= 2:
        return (np.nan, np.nan, np.nan, np.nan)
    psd0 = psd.copy()
    psd0[0] = 0.0

    total = psd0.sum()
    if total <= 1e-18:
        return (np.nan, np.nan, np.nan, np.nan)

    p = psd0 / total
    ent = -np.sum(p * np.log(p + 1e-18)) / np.log(p.size)

    # dominant frequency index (excluding DC)
    k = int(np.argmax(psd0[1:]) + 1)
    # Convert to pseudo-period (in days) for this discrete window
    dom_period = float(s.size) / float(k)
    dom_strength = float(psd0[k] / total)

    # high-frequency energy ratio: top 1/3 frequencies vs all (excluding DC)
    start = int(np.floor((p.size - 1) * 2 / 3)) + 1  # keep away from DC
    hf = float(psd0[start:].sum() / total)

    return (dom_period, dom_strength, float(ent), hf)


def _rolling_spectral(df_daily: pd.DataFrame, col: str, window: int = 32) -> pd.DataFrame:
    s = pd.to_numeric(df_daily[col], errors="coerce").values
    dom_p = np.full(len(s), np.nan)
    dom_s = np.full(len(s), np.nan)
    ent = np.full(len(s), np.nan)
    hf = np.full(len(s), np.nan)

    for i in range(window - 1, len(s)):
        w = s[i - window + 1 : i + 1]
        dp, ds, e, h = _rolling_spectral_entropy_and_hf_ratio(w)
        dom_p[i] = dp
        dom_s[i] = ds
        ent[i] = e
        hf[i] = h

    return pd.DataFrame(
        {
            f"{col}_dom_period_{window}d": dom_p,
            f"{col}_dom_strength_{window}d": dom_s,
            f"{col}_spec_entropy_{window}d": ent,
            f"{col}_hf_energy_ratio_{window}d": hf,
        }
    )


def _attach_market_features(df: pd.DataFrame, ctx: FactorContext) -> pd.DataFrame:
    """Attach compact market features to df by date."""
    cache = _ctx_cache(ctx)
    key = "_market_breadth_limit_features_light_v1"
    if key in cache:
        feats_daily = cache[key]
    else:
        daily = _build_market_daily(df)

        # short-horizon transforms (align with 3-day label horizon)
        for col in ["mkt_adv_dec", "mkt_limit_balance", "mkt_emotion_heat"]:
            daily[f"{col}_ma3"] = daily[col].rolling(3, min_periods=3).mean()
            daily[f"{col}_delta3"] = daily[col] - daily[col].shift(3)
            daily[f"{col}_z10"] = (daily[col] - daily[col].rolling(10, min_periods=10).mean()) / (
                daily[col].rolling(10, min_periods=10).std(ddof=0) + 1e-12
            )

        # one spectral summary (emotion heat) as a compact cycle proxy
        daily = pd.concat([daily, _rolling_spectral(daily, "mkt_emotion_heat", window=32)], axis=1)

        feats_daily = daily
        cache[key] = feats_daily

    x = df.copy()
    x["date"] = _as_date_series(x["date"])

    keep_cols = [c for c in feats_daily.columns if c != "date"]
    return x.merge(feats_daily[["date", *keep_cols]], on="date", how="left")


def _calc_market_col(col: str):
    def _f(df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
        m = _attach_market_features(df, ctx)
        return pd.to_numeric(m[col], errors="coerce")

    return _f


# -----------------------------
# Factor registrations
# -----------------------------

# Base (same-day) market series
# Base (same-day) market series
mkt_adv_dec = register_factor(
    FactorDefinition(
        name="mkt_adv_dec",
        required_columns=_REQUIRED_COLS,
        lookback_days=1,
        calculator=_calc_market_col("mkt_adv_dec"),
    )
)
mkt_limit_balance = register_factor(
    FactorDefinition(
        name="mkt_limit_balance",
        required_columns=_REQUIRED_COLS,
        lookback_days=1,
        calculator=_calc_market_col("mkt_limit_balance"),
    )
)
mkt_limit_up_ratio =register_factor(
    FactorDefinition(
        name="mkt_limit_up_ratio",
        required_columns=_REQUIRED_COLS,
        lookback_days=1,
        calculator=_calc_market_col("mkt_limit_up_ratio"),
    )
)
mkt_limit_down_ratio =register_factor(
    FactorDefinition(
        name="mkt_limit_down_ratio",
        required_columns=_REQUIRED_COLS,
        lookback_days=1,
        calculator=_calc_market_col("mkt_limit_down_ratio"),
    )
)
mkt_zhaban_ratio = register_factor(
    FactorDefinition(
        name="mkt_zhaban_ratio",
        required_columns=_REQUIRED_COLS,
        lookback_days=1,
        calculator=_calc_market_col("mkt_zhaban_ratio"),
    )
)
mkt_emotion_heat = register_factor(
    FactorDefinition(
        name="mkt_emotion_heat",
        required_columns=_REQUIRED_COLS,
        lookback_days=1,
        calculator=_calc_market_col("mkt_emotion_heat"),
    )
)

# Short-horizon transforms (3d/10d) on 3 core series
mkt_adv_dec_ma3 = register_factor(
    FactorDefinition(
        name="mkt_adv_dec_ma3",
        required_columns=_REQUIRED_COLS,
        lookback_days=3,
        calculator=_calc_market_col("mkt_adv_dec_ma3"),
    )
)
mkt_adv_dec_delta3 = register_factor(
    FactorDefinition(
        name="mkt_adv_dec_delta3",
        required_columns=_REQUIRED_COLS,
        lookback_days=4,
        calculator=_calc_market_col("mkt_adv_dec_delta3"),
    )
)
mkt_adv_dec_z10 = register_factor(
    FactorDefinition(
        name="mkt_adv_dec_z10",
        required_columns=_REQUIRED_COLS,
        lookback_days=10,
        calculator=_calc_market_col("mkt_adv_dec_z10"),
    )
)

mkt_limit_balance_ma3 =register_factor(
    FactorDefinition(
        name="mkt_limit_balance_ma3",
        required_columns=_REQUIRED_COLS,
        lookback_days=3,
        calculator=_calc_market_col("mkt_limit_balance_ma3"),
    )
)
mkt_limit_balance_delta3 = register_factor(
    FactorDefinition(
        name="mkt_limit_balance_delta3",
        required_columns=_REQUIRED_COLS,
        lookback_days=4,
        calculator=_calc_market_col("mkt_limit_balance_delta3"),
    )
)
mkt_limit_balance_z10 = register_factor(
    FactorDefinition(
        name="mkt_limit_balance_z10",
        required_columns=_REQUIRED_COLS,
        lookback_days=10,
        calculator=_calc_market_col("mkt_limit_balance_z10"),
    )
)

mkt_emotion_heat_ma3 = register_factor(
    FactorDefinition(
        name="mkt_emotion_heat_ma3",
        required_columns=_REQUIRED_COLS,
        lookback_days=3,
        calculator=_calc_market_col("mkt_emotion_heat_ma3"),
    )
)
mkt_emotion_heat_delta3 = register_factor(
    FactorDefinition(
        name="mkt_emotion_heat_delta3",
        required_columns=_REQUIRED_COLS,
        lookback_days=4,
        calculator=_calc_market_col("mkt_emotion_heat_delta3"),
    )
)
mkt_emotion_heat_z10 = register_factor(
    FactorDefinition(
        name="mkt_emotion_heat_z10",
        required_columns=_REQUIRED_COLS,
        lookback_days=10,
        calculator=_calc_market_col("mkt_emotion_heat_z10"),
    )
)

# Spectral summary (32d) on emotion heat only
mkt_emotion_heat_dom_period_32d = register_factor(
    FactorDefinition(
        name="mkt_emotion_heat_dom_period_32d",
        required_columns=_REQUIRED_COLS,
        lookback_days=32,
        calculator=_calc_market_col("mkt_emotion_heat_dom_period_32d"),
    )
)
mkt_emotion_heat_dom_strength_32d = register_factor(
    FactorDefinition(
        name="mkt_emotion_heat_dom_strength_32d",
        required_columns=_REQUIRED_COLS,
        lookback_days=32,
        calculator=_calc_market_col("mkt_emotion_heat_dom_strength_32d"),
    )
)
mkt_emotion_heat_spec_entropy_32d = register_factor(
    FactorDefinition(
        name="mkt_emotion_heat_spec_entropy_32d",
        required_columns=_REQUIRED_COLS,
        lookback_days=32,
        calculator=_calc_market_col("mkt_emotion_heat_spec_entropy_32d"),
    )
)
mkt_emotion_heat_hf_energy_ratio_32d =register_factor(
    FactorDefinition(
        name="mkt_emotion_heat_hf_energy_ratio_32d",
        required_columns=_REQUIRED_COLS,
        lookback_days=32,
        calculator=_calc_market_col("mkt_emotion_heat_hf_energy_ratio_32d"),
    )
)