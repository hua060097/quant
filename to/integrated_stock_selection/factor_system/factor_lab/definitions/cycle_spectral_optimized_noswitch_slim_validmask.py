"""
cycle_spectral_optimized_noswitch_slim.py (SLIM)

基于你上传的 cycle_spectral.py fileciteturn23file0 做了两件事：
1) 工程化修复：时序纯净（按 stock_code,_ts）、组内 rolling、统一按日 CS robust z + clip + fillna(0)、输出 float32。
2) 因子精简：频域/小波因子非常容易“漂移 + 高噪声 + 计算重”，在 Top1/Top3 + 2~3日标签下，
   通常只保留“强度/噪声度/高频占比”这类更稳健的统计量，舍弃“主周期/相位”这类不稳定/强参数化特征。

保留（4个）：
- cycle_strength_20d         : FFT 主频强度（主频功率/总功率）
- spectral_entropy_20d       : FFT 谱熵（越高越无周期/噪声化）
- wavelet_highfreq_energy_20d: 小波高频能量占比（越高越偏高频/更易冲击）
- volume_cycle_strength_20d  : 成交量主频强度（log(volume)差分的 FFT 强度）

建议下线（理由：不稳定/冗余/对标签边际贡献低）：
- cycle_dominant_period_20d, cycle_phase_sin_20d, cycle_phase_cos_20d
- wavelet_dominant_level_20d, wavelet_cycle_period_20d

依赖：
- 价格类：close
- 成交量类：volume（若你的数据列名是 vol，可在 pipeline 里映射到 volume 后再算）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pywt  # type: ignore
except Exception:  # pragma: no cover
    pywt = None

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_EPS = 1e-12
_CLIP_Z = 6.0
_OUT_DTYPE = "float32"


# ==============================================================================
# Core utils (same style as your optimized_noswitch)
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


def _cs_robust_z(x: pd.Series, date_keys: pd.Series, *, valid_cs_mask: pd.Series | None = None, clip_z: float = _CLIP_Z) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    if valid_cs_mask is not None:
        m = pd.to_numeric(valid_cs_mask, errors="coerce").fillna(1.0) > 0.5
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
    z = z.fillna(0.0)
    return z.astype(_OUT_DTYPE)


# ==============================================================================
# Frequency helpers (compute on numpy window)
# ==============================================================================
def _linear_detrend(arr: np.ndarray) -> np.ndarray:
    y = np.asarray(arr, dtype=float)
    n = y.shape[0]
    if n <= 2:
        return y - np.nanmean(y)
    t = np.arange(n, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < max(3, n // 2):
        yy = np.nan_to_num(y, nan=np.nanmean(y))
        return yy - np.mean(yy)
    tt = t[mask]
    yy = y[mask]
    a, b = np.polyfit(tt, yy, 1)
    trend = a * t + b
    out = np.nan_to_num(y, nan=np.nanmean(y)) - trend
    out = out - np.mean(out)
    return out


def _fft_power_no_dc(x: np.ndarray) -> np.ndarray:
    y = _linear_detrend(x)
    spec = np.fft.rfft(y)
    power = (spec.real ** 2 + spec.imag ** 2).astype(float)
    if power.shape[0] <= 1:
        return np.array([], dtype=float)
    return power[1:]


def _fft_cycle_strength(x: np.ndarray) -> float:
    p = _fft_power_no_dc(np.asarray(x, dtype=float))
    if p.size == 0:
        return np.nan
    total = float(np.nansum(p))
    if not np.isfinite(total) or total <= 0:
        return np.nan
    dom = float(np.nanmax(p))
    return dom / total


def _fft_spectral_entropy(x: np.ndarray) -> float:
    p = _fft_power_no_dc(np.asarray(x, dtype=float))
    if p.size <= 1:
        return np.nan
    p = np.clip(p, 0.0, np.inf)
    s = float(np.sum(p))
    if s <= 0 or not np.isfinite(s):
        return np.nan
    w = p / s
    w = np.clip(w, 1e-12, 1.0)
    ent = -float(np.sum(w * np.log(w)))
    return float(ent / np.log(len(w)))


def _wavelet_highfreq_energy_ratio(x: np.ndarray, wavelet: str = "db4", level: int = 3) -> float:
    if pywt is None:
        return np.nan
    y = _linear_detrend(np.asarray(x, dtype=float))
    n = y.shape[0]
    if n < 2 ** (level + 1):
        return np.nan
    coeffs = pywt.wavedec(y, wavelet=wavelet, level=level, mode="periodization")
    energies = [float(np.sum(np.square(c))) for c in coeffs if c is not None]
    if len(energies) <= 1:
        return np.nan
    total = float(np.sum(energies))
    if total <= 0:
        return np.nan
    high = float(np.sum(energies[1:]))
    return high / total


# ==============================================================================
# Rolling computation (group-safe)
# ==============================================================================
def _group_rolling_apply(series: pd.Series, group: pd.Series, window: int, func, min_periods: int) -> pd.Series:
    # raw=True gives ndarray; faster than raw=False with Series
    return series.groupby(group, sort=False).transform(
        lambda s: s.rolling(window=window, min_periods=min_periods).apply(lambda v: func(v), raw=True)
    )


# ==============================================================================
# Factor calculators (raw -> CS robust z)
# ==============================================================================
def _cycle_strength_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df = _attach_sorted(data[list(need)].copy())
    close = _num64(df["close"])
    g = df["stock_code"]

    ret1 = close.groupby(g, sort=False).pct_change(1)
    raw = _group_rolling_apply(ret1, g, 20, _fft_cycle_strength, 20)

    out = _cs_robust_z(raw.reindex(data.index), _to_ts(data["date"]), valid_cs_mask=(data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None))
    return out.astype(_OUT_DTYPE)


def _spectral_entropy_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df = _attach_sorted(data[list(need)].copy())
    close = _num64(df["close"])
    g = df["stock_code"]

    ret1 = close.groupby(g, sort=False).pct_change(1)
    raw = _group_rolling_apply(ret1, g, 20, _fft_spectral_entropy, 20)

    out = _cs_robust_z(raw.reindex(data.index), _to_ts(data["date"]), valid_cs_mask=(data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None))
    return out.astype(_OUT_DTYPE)


def _wavelet_highfreq_energy_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "close"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df = _attach_sorted(data[list(need)].copy())
    close = _num64(df["close"])
    g = df["stock_code"]

    ret1 = close.groupby(g, sort=False).pct_change(1)
    raw = _group_rolling_apply(ret1, g, 20, lambda v: _wavelet_highfreq_energy_ratio(v, level=3), 20)

    out = _cs_robust_z(raw.reindex(data.index), _to_ts(data["date"]), valid_cs_mask=(data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None))
    return out.astype(_OUT_DTYPE)


def _volume_cycle_strength_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df = _attach_sorted(data[list(need)].copy())
    vol = _num64(df["volume"]).clip(lower=0.0)
    g = df["stock_code"]

    lv = np.log1p(vol)
    dv = lv.groupby(g, sort=False).diff(1)
    raw = _group_rolling_apply(dv, g, 20, _fft_cycle_strength, 20)

    out = _cs_robust_z(raw.reindex(data.index), _to_ts(data["date"]), valid_cs_mask=(data["valid_cs_mask"] if "valid_cs_mask" in data.columns else None))
    return out.astype(_OUT_DTYPE)


# ==============================================================================
# Register
# ==============================================================================
cycle_strength_20d = register_factor(FactorDefinition(
    name="cycle_strength_20d",
    description="FFT 主频强度（主频功率/总功率，rolling=20），基于 close 的日收益率；CS robust z",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=_cycle_strength_20d,
))

spectral_entropy_20d = register_factor(FactorDefinition(
    name="spectral_entropy_20d",
    description="FFT 谱熵（rolling=20），越高越噪声化/无明显周期；CS robust z",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=_spectral_entropy_20d,
))

wavelet_highfreq_energy_20d = register_factor(FactorDefinition(
    name="wavelet_highfreq_energy_20d",
    description="Wavelet 高频能量占比（rolling=20, level=3, db4），越高越偏高频/更易冲击；CS robust z",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=_wavelet_highfreq_energy_20d,
))

volume_cycle_strength_20d = register_factor(FactorDefinition(
    name="volume_cycle_strength_20d",
    description="成交量主频强度（rolling=20），基于 log1p(volume) 差分的 FFT；CS robust z",
    required_columns={"stock_code", "date", "volume"},
    lookback_days=25,
    calculator=_volume_cycle_strength_20d,
))

__all__ = [
    "cycle_strength_20d",
    "spectral_entropy_20d",
    "wavelet_highfreq_energy_20d",
    "volume_cycle_strength_20d",
]
