"""
tech_sr_worldquant_slim_optimized_noswitch.py

对你上传的三个因子文件：
- support_resistance.py fileciteturn22file0
- technical_indicators.py fileciteturn22file1
- worldquant.py fileciteturn22file2

做“是否值得保留”的专业取舍，并将保留下来的统一到一个新文件里，保持你当前
optimized_noswitch 的统一风格：

- 时序纯净：按 stock_code,_ts 排序，rolling/shift 严格在组内执行
- 横截面统一：按日 CS robust z（median/MAD）+ clip + fillna(0)
- 内存友好：最终输出 float32（计算过程用 float64 保稳定）

为什么要重写：
1) 你这三份文件里普遍“保留 NaN、不 fill 0”，且使用自定义 global robust z。对你当前统一 pipeline
   （CS robust z + fillna(0)）来说会造成口径不一致，训练时缺失结构也更不稳定。
2) worldquant.py 里的 alpha_001（PV corr）与你系统里已有的 cpv/corr 类（如 cpv_factor、price_impact）
   信息高度重叠，常常贡献不大，因此在精简版里直接下线。
3) technical_indicators.py 中 vortex / mass index 在你已有的趋势/波动结构体系里也常常是“弱增量、强噪声”；
   在 Top1/Top3 场景里更容易诱发追尾/反复。精简版只保留 PPO（量纲一致的趋势）与 MFI（资金流强弱）。

最终保留的因子（6 个）：
- support_distance      ：区间内“靠近底部且缩量”的支撑强度（向量化 + numba）
- resistance_distance   ：区间内“靠近顶部且放量”的压力强度（向量化 + numba）
- ppo_strength          ：PPO 趋势强度（MACD 的量纲统一版）
- mfi_strength          ：MFI 资金流强弱（带量 RSI）
- alpha_002             ：成交量变化与日内涨幅的负相关（反转/过热回吐）
- alpha_017             ：缩量上涨/放量滞涨（收益rank - 量变rank）

下线（建议你删除原文件里的这些项或整个文件）：
- worldquant.alpha_001：与 cpv/corr 重叠度高
- technical_indicators.vortex_trend / mass_index_reversal：通常增量弱且噪声大（建议先下线验证）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from numba import njit

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

def _attach_sorted(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["_ts"] = _to_ts(x["date"])
    if "valid_cs_mask" in x.columns:
        x["valid_cs_mask"] = _clean_bool(x["valid_cs_mask"]).astype(bool)
    else:
        x["valid_cs_mask"] = True
    x["stock_code"] = x["stock_code"].astype(str)
    x.sort_values(["stock_code", "_ts"], inplace=True)
    return x


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

@njit(cache=True)
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@njit
def _calc_sr_dynamics_numba(close: np.ndarray, volume: np.ndarray, lookback: int, vol_window: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    support_score = np.full(n, np.nan)
    resistance_score = np.full(n, np.nan)
    if n <= lookback:
        return support_score, resistance_score

    for i in range(lookback, n):
        past_close = close[i - lookback:i]  # exclude today
        p_min = np.min(past_close)
        p_max = np.max(past_close)
        span = p_max - p_min + 1e-9
        curr = close[i]
        pos = (curr - p_min) / span

        # volume dynamics
        v_recent = np.mean(volume[i - vol_window:i + 1])
        v_base = np.mean(volume[i - lookback:i]) + 1e-9
        v_ratio = v_recent / v_base
        x = (v_ratio - 1.0) * 2.0
        v_factor = 1.0 / (1.0 + np.exp(-x))

        # support
        dist_low = pos
        if dist_low < 0.0 or dist_low > 1.0:
            support_score[i] = 0.0
        else:
            proximity = np.exp(-4.0 * dist_low)
            volume_score = 1.0 - v_factor
            support_score[i] = proximity * volume_score

        # resistance
        dist_high = 1.0 - pos
        if pos > 1.0 or pos < 0.0:
            resistance_score[i] = 0.0
        else:
            proximity = np.exp(-4.0 * dist_high)
            volume_score = v_factor
            resistance_score[i] = proximity * volume_score

    return support_score, resistance_score


def _support_resistance_raw(data: pd.DataFrame, lookback: int = 20, vol_window: int = 5) -> pd.DataFrame:
    """
    返回两个 raw 序列：support_raw / resistance_raw
    保证组内顺序正确（时序纯净）。
    """
    need = {"stock_code", "date", "close", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.DataFrame({"support_raw": np.nan, "resistance_raw": np.nan}, index=data.index)

    df = _attach_sorted(data[["stock_code", "date"] + (["valid_cs_mask"] if "valid_cs_mask" in data.columns else [])].copy())
    close = _num64(data["close"]).reindex(df.index)
    vol = _num64(data["volume"]).reindex(df.index)
    g = df["stock_code"]

    def _calc_one(group_idx: pd.Index) -> pd.DataFrame:
        c = close.loc[group_idx].to_numpy(dtype=np.float64, copy=False)
        v = vol.loc[group_idx].to_numpy(dtype=np.float64, copy=False)
        s, r = _calc_sr_dynamics_numba(c, v, lookback, vol_window)
        return pd.DataFrame({"support_raw": s, "resistance_raw": r}, index=group_idx)

    out = df.groupby("stock_code", sort=False, group_keys=False).apply(lambda sub: _calc_one(sub.index))
    out = out.reindex(df.index)

    # restore original order
    out = out.reindex(data.index)
    return out


def _support_distance(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    raw = _support_resistance_raw(data)["support_raw"]
    ts = _to_ts(data["date"])
    return _cs_robust_z(raw, ts, valid_cs_mask=_vmask(data))


def _resistance_distance(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    raw = _support_resistance_raw(data)["resistance_raw"]
    ts = _to_ts(data["date"])
    return _cs_robust_z(raw, ts, valid_cs_mask=_vmask(data))


# ==============================================================================
# Technical Indicators (keep PPO & MFI only)
# ==============================================================================
def _ppo_strength(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if "close" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df = _attach_sorted(data[["stock_code", "date"] + (["valid_cs_mask"] if "valid_cs_mask" in data.columns else [])].copy())
    close = _num64(data["close"]).reindex(df.index)
    g = df["stock_code"]

    ema12 = close.groupby(g, sort=False).ewm(span=12, adjust=False).mean()
    ema26 = close.groupby(g, sort=False).ewm(span=26, adjust=False).mean()
    ppo_line = (ema12 - ema26) / (ema26 + _EPS) * 100.0

    # signal: ema of ppo in-group
    ppo_signal = ppo_line.groupby(g, sort=False).ewm(span=9, adjust=False).mean()
    ppo_hist = ppo_line - ppo_signal

    raw = (0.5 * ppo_line + 1.5 * ppo_hist).clip(-10.0, 10.0)

    out = _cs_robust_z(raw.reindex(data.index), _to_ts(data["date"]), valid_cs_mask=_vmask(data))
    return out.astype(_OUT_DTYPE)


def _mfi_strength(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    cols = {"stock_code", "date", "high", "low", "close", "volume"}
    if not cols.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df = _attach_sorted(data[["stock_code", "date"] + (["valid_cs_mask"] if "valid_cs_mask" in data.columns else [])].copy())
    high = _num64(data["high"]).reindex(df.index)
    low = _num64(data["low"]).reindex(df.index)
    close = _num64(data["close"]).reindex(df.index)
    vol = _num64(data["volume"]).reindex(df.index)
    g = df["stock_code"]

    tp = (high + low + close) / 3.0
    rmf = tp * vol

    tp_prev = tp.groupby(g, sort=False).shift(1)
    pos_flow = rmf.where(tp > tp_prev, 0.0)
    neg_flow = rmf.where(tp < tp_prev, 0.0)

    pos_sum = pos_flow.groupby(g, sort=False).transform(lambda s: s.rolling(14, min_periods=10).sum())
    neg_sum = neg_flow.groupby(g, sort=False).transform(lambda s: s.rolling(14, min_periods=10).sum())

    mfi = 100.0 - (100.0 / (1.0 + pos_sum / (neg_sum + _EPS)))
    raw = (mfi - 50.0) / 20.0  # ~[-2.5, 2.5]
    raw = raw.where(df["valid_cs_mask"], np.nan)

    out = _cs_robust_z(raw.reindex(data.index), _to_ts(data["date"]), valid_cs_mask=_vmask(data))
    return out.astype(_OUT_DTYPE)


# ==============================================================================
# WorldQuant (keep 002 & 017; drop 001)
# ==============================================================================
def _alpha_002(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "open", "close", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df = _attach_sorted(data[["stock_code", "date"] + (["valid_cs_mask"] if "valid_cs_mask" in data.columns else [])].copy())
    open_ = _num64(data["open"]).reindex(df.index)
    close = _num64(data["close"]).reindex(df.index)
    vol = _num64(data["volume"]).reindex(df.index)
    g = df["stock_code"]

    delta_vol = np.log1p(vol.clip(lower=0.0)).groupby(g, sort=False).diff(2)
    intra_ret = (close - open_) / (open_ + _EPS)

    temp = pd.DataFrame({"dv": delta_vol, "ir": intra_ret, "stock_code": g.values}, index=df.index)

    def _corr6(sub: pd.DataFrame) -> pd.Series:
        return sub["dv"].rolling(6, min_periods=5).corr(sub["ir"])

    raw_corr = temp.groupby("stock_code", sort=False, group_keys=False).apply(_corr6)
    raw = (-1.0 * raw_corr).reindex(data.index)

    return _cs_robust_z(raw, _to_ts(data["date"]), valid_cs_mask=_vmask(data))


def _alpha_017(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"stock_code", "date", "pct_chg", "volume"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    # 这里按“date”做 CS rank，本身已归一化，再做 robust z 以对齐全体分布
    df = data[["stock_code", "date", "pct_chg", "volume"]].copy()
    df["stock_code"] = df["stock_code"].astype(str)
    df["_ts"] = _to_ts(df["date"])

    vol_chg = _num64(df["volume"]).groupby(df["stock_code"], sort=False).pct_change(1)
    tmp = pd.DataFrame({"pct": _num64(df["pct_chg"]), "volchg": vol_chg, "date": df["_ts"]}, index=df.index)

    ranks = tmp.groupby("date", sort=False)[["pct", "volchg"]].rank(pct=True)
    raw = (ranks["pct"] - ranks["volchg"]).astype("float64")

    return _cs_robust_z(raw, df["_ts"], valid_cs_mask=_vmask(data)).astype(_OUT_DTYPE)


# ==============================================================================
# Register
# ==============================================================================
support_distance = register_factor(FactorDefinition(
    name="support_distance",
    required_columns={"stock_code", "date", "close", "volume"},
    lookback_days=40,
    calculator=_support_distance,
    description="支撑强度：区间内越靠近底部且缩量越强（按日CS robust z，float32）",
))

resistance_distance = register_factor(FactorDefinition(
    name="resistance_distance",
    required_columns={"stock_code", "date", "close", "volume"},
    lookback_days=40,
    calculator=_resistance_distance,
    description="压力强度：区间内越靠近顶部且放量越强（按日CS robust z，float32）",
))

ppo_strength = register_factor(FactorDefinition(
    name="ppo_strength",
    required_columns={"stock_code", "date", "close"},
    lookback_days=40,
    calculator=_ppo_strength,
    description="PPO 趋势强度（量纲统一的 MACD 替代；按日CS robust z，float32）",
))

mfi_strength = register_factor(FactorDefinition(
    name="mfi_strength",
    required_columns={"stock_code", "date", "high", "low", "close", "volume"},
    lookback_days=30,
    calculator=_mfi_strength,
    description="MFI 资金流强弱（带量 RSI；按日CS robust z，float32）",
))

alpha_002 = register_factor(FactorDefinition(
    name="alpha_002",
    required_columns={"stock_code", "date", "open", "close", "volume"},
    lookback_days=15,
    calculator=_alpha_002,
    description="WQ Alpha002：-corr(delta(log(vol),2), intraday_ret,6)，按日CS robust z（float32）",
))

alpha_017 = register_factor(FactorDefinition(
    name="alpha_017",
    required_columns={"stock_code", "date", "pct_chg", "volume"},
    lookback_days=10,
    calculator=_alpha_017,
    description="WQ Alpha017：rank(ret)-rank(vol_chg)（缩量上涨/放量滞涨），按日CS robust z（float32）",
))



def _vmask(data: pd.DataFrame) -> pd.Series:
    """Return boolean valid cross-section mask aligned to `data.index`."""
    if isinstance(data, pd.DataFrame) and ("valid_cs_mask" in data.columns):
        return _clean_bool(data["valid_cs_mask"]).astype(bool)
    return pd.Series(True, index=data.index)

__all__ = [
    "support_distance",
    "resistance_distance",
    "ppo_strength",
    "mfi_strength",
    "alpha_002",
    "alpha_017",
]
