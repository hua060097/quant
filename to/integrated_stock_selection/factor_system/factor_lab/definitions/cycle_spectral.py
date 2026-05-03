
"""cycle_spectral.py

周期/频域因子（FFT + Wavelet），偏短周期，适配“未来3日加权收益”标签。

设计原则：
- 输入为日频OHLCV（至少需要 close；部分因子可用 volume）。
- 以 rolling window 的频域特征刻画“当下市场/个股处于什么节奏”，
  供排序模型学习“在不同节奏下，哪些形态更容易赢”。

实现注意：
- 频域计算对样本长度敏感：默认 window=20（约一个月交易日）更贴近 3 日标签的可交易节奏；
  你也可以在 factor 参数里调整到 16/24/30 等更短/更长窗口。
- 对每个窗口先做去均值/去线性趋势，避免低频趋势主导 FFT/Wavelet。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pywt  # type: ignore
except Exception:  # pragma: no cover
    pywt = None  # noqa: N816

from factors.factor_registry import register_factor, FactorDefinition


# =========================
# helpers
# =========================

def _as_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)

def _linear_detrend(arr: np.ndarray) -> np.ndarray:
    """对窗口序列做去线性趋势（最小二乘一阶）并去均值。"""
    y = np.asarray(arr, dtype=float)
    n = y.shape[0]
    if n <= 2:
        return y - np.nanmean(y)
    t = np.arange(n, dtype=float)
    # 处理 nan：用有效点拟合，剩余用 0 填充
    mask = np.isfinite(y)
    if mask.sum() < max(3, n // 2):
        yy = np.nan_to_num(y, nan=np.nanmean(y))
        return yy - np.mean(yy)
    tt = t[mask]
    yy = y[mask]
    # y = a*t + b
    a, b = np.polyfit(tt, yy, 1)
    trend = a * t + b
    out = np.nan_to_num(y, nan=np.nanmean(y)) - trend
    out = out - np.mean(out)
    return out

def _rolling_apply_1d(s: pd.Series, window: int, func, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    return s.rolling(window=window, min_periods=min_periods).apply(func, raw=False)

def _fft_power(x: np.ndarray) -> np.ndarray:
    """返回 rfft 的功率谱（不含 DC）。"""
    y = _linear_detrend(x)
    # rfft: length n -> n//2+1 bins, bin0 为 DC
    spec = np.fft.rfft(y)
    power = (spec.real ** 2 + spec.imag ** 2).astype(float)
    if power.shape[0] <= 1:
        return np.array([], dtype=float)
    return power[1:]  # drop DC

def _fft_dominant_k(x: np.ndarray) -> int | None:
    p = _fft_power(x)
    if p.size == 0 or not np.isfinite(p).any():
        return None
    k = int(np.nanargmax(p)) + 1  # +1 because DC removed
    return k

def _fft_dominant_period_days(x: np.ndarray) -> float:
    """dominant period in days for window length n."""
    y = np.asarray(x, dtype=float)
    n = y.shape[0]
    k = _fft_dominant_k(y)
    if k is None or k <= 0:
        return np.nan
    # freq = k / n cycles/day -> period = n / k days
    return float(n) / float(k)

def _fft_cycle_strength(x: np.ndarray) -> float:
    """主频强度：主频功率 / 总功率（去 DC 后）。"""
    p = _fft_power(np.asarray(x, dtype=float))
    if p.size == 0:
        return np.nan
    total = np.nansum(p)
    if total <= 0 or not np.isfinite(total):
        return np.nan
    dom = float(np.nanmax(p))
    return dom / total

def _fft_spectral_entropy(x: np.ndarray) -> float:
    """谱熵：功率归一化后 entropy / log(K)，范围约 [0,1]，越高越“无周期”。"""
    p = _fft_power(np.asarray(x, dtype=float))
    if p.size <= 1:
        return np.nan
    p = np.clip(p, 0.0, np.inf)
    s = np.sum(p)
    if s <= 0:
        return np.nan
    w = p / s
    w = np.clip(w, 1e-12, 1.0)
    ent = -np.sum(w * np.log(w))
    return float(ent / np.log(len(w)))

def _fft_phase_sin_cos(x: np.ndarray) -> tuple[float, float]:
    """主频相位的 sin/cos（用于模型捕捉“周期位置”）。"""
    y = _linear_detrend(np.asarray(x, dtype=float))
    n = y.shape[0]
    if n < 6:
        return (np.nan, np.nan)
    k = _fft_dominant_k(y)
    if k is None:
        return (np.nan, np.nan)
    spec = np.fft.rfft(y)
    if k >= spec.shape[0]:
        return (np.nan, np.nan)
    phase = np.angle(spec[k])
    return (float(np.sin(phase)), float(np.cos(phase)))


def _wavelet_energy_ratio_highfreq(x: np.ndarray, wavelet: str = "db4", level: int = 3) -> float:
    """高频能量占比：sum(detail energies) / total energy。"""
    if pywt is None:
        return np.nan
    y = _linear_detrend(np.asarray(x, dtype=float))
    n = y.shape[0]
    if n < 2 ** (level + 1):
        return np.nan
    coeffs = pywt.wavedec(y, wavelet=wavelet, level=level, mode="periodization")
    # coeffs: [cA_L, cD_L, cD_{L-1}, ..., cD_1]
    energies = [float(np.sum(np.square(c))) for c in coeffs if c is not None]
    if len(energies) <= 1:
        return np.nan
    total = float(np.sum(energies))
    if total <= 0:
        return np.nan
    # 高频：所有 detail 部分（排除近似 cA）
    high = float(np.sum(energies[1:]))
    return high / total

def _wavelet_dominant_level(x: np.ndarray, wavelet: str = "db4", level: int = 3) -> float:
    """主导尺度 level（能量最大的 detail level）。输出 1..level，越大越偏低频。"""
    if pywt is None:
        return np.nan
    y = _linear_detrend(np.asarray(x, dtype=float))
    n = y.shape[0]
    if n < 2 ** (level + 1):
        return np.nan
    coeffs = pywt.wavedec(y, wavelet=wavelet, level=level, mode="periodization")
    # details energies
    detail_coeffs = coeffs[1:]  # cD_L ... cD_1
    detail_energies = np.array([np.sum(np.square(c)) for c in detail_coeffs], dtype=float)
    if detail_energies.size == 0 or not np.isfinite(detail_energies).any():
        return np.nan
    # idx 0 corresponds to level L, idx -1 to level 1
    idx = int(np.nanargmax(detail_energies))
    dom_level = int(level - idx)  # map to 1..level
    return float(dom_level)

def _wavelet_cycle_period_proxy(x: np.ndarray, wavelet: str = "db4", level: int = 3) -> float:
    """用主导小波尺度近似周期：period ≈ 2^level（交易日）。"""
    dom_level = _wavelet_dominant_level(x, wavelet=wavelet, level=level)
    if not np.isfinite(dom_level):
        return np.nan
    return float(2.0 ** dom_level)


# =========================
# factor calculators
# =========================

def cycle_dominant_period_20d(df: pd.DataFrame, context=None) -> pd.Series:
    """FFT 主导周期（交易日，rolling=20）。"""
    close = df["close"].astype(float)
    ret = close.pct_change()
    return _rolling_apply_1d(ret, 20, lambda s: _fft_dominant_period_days(s.values), min_periods=20)

def cycle_strength_20d(df: pd.DataFrame, context=None) -> pd.Series:
    """主导周期强度（主频功率/总功率，rolling=20）。"""
    close = df["close"].astype(float)
    ret = close.pct_change()
    return _rolling_apply_1d(ret, 20, lambda s: _fft_cycle_strength(s.values), min_periods=20)

def spectral_entropy_20d(df: pd.DataFrame, context=None) -> pd.Series:
    """谱熵（rolling=20），越高越“无周期/噪声化”。"""
    close = df["close"].astype(float)
    ret = close.pct_change()
    return _rolling_apply_1d(ret, 20, lambda s: _fft_spectral_entropy(s.values), min_periods=20)

def cycle_phase_sin_20d(df: pd.DataFrame, context=None) -> pd.Series:
    """主频相位 sin（rolling=20）。"""
    close = df["close"].astype(float)
    ret = close.pct_change()
    return _rolling_apply_1d(ret, 20, lambda s: _fft_phase_sin_cos(s.values)[0], min_periods=20)

def cycle_phase_cos_20d(df: pd.DataFrame, context=None) -> pd.Series:
    """主频相位 cos（rolling=20）。"""
    close = df["close"].astype(float)
    ret = close.pct_change()
    return _rolling_apply_1d(ret, 20, lambda s: _fft_phase_sin_cos(s.values)[1], min_periods=20)

def wavelet_highfreq_energy_20d(df: pd.DataFrame, context=None) -> pd.Series:
    """Wavelet 高频能量占比（rolling=20, level=3）。"""
    close = df["close"].astype(float)
    ret = close.pct_change()
    return _rolling_apply_1d(ret, 20, lambda s: _wavelet_energy_ratio_highfreq(s.values, level=3), min_periods=20)

def wavelet_dominant_level_20d(df: pd.DataFrame, context=None) -> pd.Series:
    """Wavelet 主导尺度 level（rolling=20, level=3）。"""
    close = df["close"].astype(float)
    ret = close.pct_change()
    return _rolling_apply_1d(ret, 20, lambda s: _wavelet_dominant_level(s.values, level=3), min_periods=20)

def wavelet_cycle_period_20d(df: pd.DataFrame, context=None) -> pd.Series:
    """Wavelet 周期近似（2^dom_level，rolling=20, level=3）。"""
    close = df["close"].astype(float)
    ret = close.pct_change()
    return _rolling_apply_1d(ret, 20, lambda s: _wavelet_cycle_period_proxy(s.values, level=3), min_periods=20)

def volume_cycle_strength_20d(df: pd.DataFrame, context=None) -> pd.Series:
    """成交量的主频强度（rolling=20）。需要 df['volume'] 或 df['vol']。"""
    if "volume" in df.columns:
        vol = df["volume"].astype(float)
    elif "vol" in df.columns:
        vol = df["vol"].astype(float)
    else:
        return pd.Series(np.nan, index=df.index)
    # log-vol 差分
    lv = np.log(vol.replace(0, np.nan))
    dv = lv.diff()
    return _rolling_apply_1d(dv, 20, lambda s: _fft_cycle_strength(s.values), min_periods=20)


# =========================
# registrations (match moment_interaction/kdj style)
# =========================

cycle_dominant_period_20d = register_factor(FactorDefinition(
    name="cycle_dominant_period_20d",
    description="FFT 主导周期（交易日，rolling=20），基于 close 的日收益率",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=cycle_dominant_period_20d
))

cycle_strength_20d = register_factor(FactorDefinition(
    name="cycle_strength_20d",
    description="FFT 主频强度（主频功率/总功率，rolling=20），越高越呈现“节奏一致”",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=cycle_strength_20d
))

spectral_entropy_20d = register_factor(FactorDefinition(
    name="spectral_entropy_20d",
    description="谱熵（rolling=20），越高越噪声化/无明显周期",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=spectral_entropy_20d
))

cycle_phase_sin_20d = register_factor(FactorDefinition(
    name="cycle_phase_sin_20d",
    description="主频相位 sin（rolling=20），用于刻画周期位置（与 cos 配对）",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=cycle_phase_sin_20d
))

cycle_phase_cos_20d = register_factor(FactorDefinition(
    name="cycle_phase_cos_20d",
    description="主频相位 cos（rolling=20），用于刻画周期位置（与 sin 配对）",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=cycle_phase_cos_20d
))

wavelet_highfreq_energy_20d = register_factor(FactorDefinition(
    name="wavelet_highfreq_energy_20d",
    description="Wavelet 高频能量占比（rolling=20, level=3, db4），越高越偏高频/更易冲击",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=wavelet_highfreq_energy_20d
))

wavelet_dominant_level_20d = register_factor(FactorDefinition(
    name="wavelet_dominant_level_20d",
    description="Wavelet 主导尺度 level（rolling=20, level=3, db4），1=偏高频，3=偏低频",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=wavelet_dominant_level_20d
))

wavelet_cycle_period_20d = register_factor(FactorDefinition(
    name="wavelet_cycle_period_20d",
    description="Wavelet 周期近似（2^dom_level，交易日，rolling=20, level=3, db4）",
    required_columns={"stock_code", "date", "close"},
    lookback_days=25,
    calculator=wavelet_cycle_period_20d
))

volume_cycle_strength_20d = register_factor(FactorDefinition(
    name="volume_cycle_strength_20d",
    description="成交量主频强度（rolling=20），基于 log(volume) 差分的 FFT",
    required_columns={"stock_code", "date", "volume"},
    lookback_days=25,
    calculator=volume_cycle_strength_20d
))
