"""
Wavelet / Regime Interaction Factors - 优化版（无开关）

定位：
- 你当前是按日 group 的 LightGBM lambdarank（6桶标签来自组内 rank）。
- 纯“市场级常数特征”无法直接影响组内排序；要进入排序，必须变成“gate × alpha”的交互项。
- 你现有交互特征已覆盖：市场趋势×beta、市场波动×低波、行业强弱/轮动×个股动量/突破/换手/价量相关、
  涨跌停情绪×个股活跃/韧性/接力。 (见 cross_level_interaction / sector_interaction / limit_interaction)
- 明显缺口：缺少“局部时频（多尺度）”的市场状态门控（wavelet / multi-resolution），用于识别：
  (a) 趋势(低频能量主导) vs 震荡(高频能量主导)
  (b) 情绪突变/拐点（detail 1/2 的能量与尖峰）
  (c) 波动/情绪的“加速度”（能量变化率）

本文件提供：
- 以“市场情绪热度 mkt_emotion_heat（由涨停/跌停/炸板比率构造）”为基础序列
- 进行 rolling Haar wavelet 多尺度分解（默认窗口=64天，levels=4）
- 构造多尺度能量比例、突变强度、能量变化率等 gate
- 再与个股 alpha（动量/反转/波动/流动性/beta 等）做交互，并做按日横截面 robust z 输出

工程要求：
- 仅使用 ctx.extra 作为缓存（你的 FactorContext 没有 _ctx_cache）
- 所有时序 rolling 严格因果（只用历史窗口）
- 输出统一：按日横截面 robust z（median/MAD）+ clip + fillna(0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_CLIP_Z = 6.0
_EPS = 1e-12
_OUT_DTYPE = "float32"
_REQ_BASE = {"stock_code", "date", "pct_chg", "is_limit_up", "is_limit_down", "is_zhaban"}


# ==============================================================================
# Utils
# ==============================================================================
def _to_ts(date_series: pd.Series) -> pd.Series:
    """date -> Timestamp normalize（00:00:00），避免 dtype 漂移与 merge 慢。"""
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

def _safe_div(a, b, eps: float = 1e-12):
    a = _num(pd.Series(a)) if not isinstance(a, pd.Series) else _num(a)
    b = _num(pd.Series(b)) if not isinstance(b, pd.Series) else _num(b)
    return a / (b.replace(0.0, np.nan) + eps)


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

def _rolling_robust_z1d(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """1D 时序 rolling robust z（median/MAD）"""
    x = _num(s)

    def _med(v):
        v = v[np.isfinite(v)]
        return np.nan if v.size == 0 else float(np.median(v))

    def _mad(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan
        m = float(np.median(v))
        return float(np.median(np.abs(v - m)))

    med = x.rolling(window, min_periods=min_periods).apply(_med, raw=True)
    mad = x.rolling(window, min_periods=min_periods).apply(_mad, raw=True)
    z = (x - med) / (1.4826 * mad + _EPS)
    return z.replace([np.inf, -np.inf], np.nan)


# ==============================================================================
# Haar wavelet (multi-resolution) on a rolling window
# ==============================================================================
def _haar_dwt_energies(w: np.ndarray, levels: int = 4) -> tuple[list[float], float, float]:
    """
    Haar DWT energies on window w.
    Returns:
      - detail_energies: [E_d1, E_d2, ...]
      - approx_energy: E_aL
      - last_detail_abs: |last d1 coefficient| (burst proxy)
    """
    x = np.asarray(w, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < (2 ** (levels + 1)):
        return ([np.nan] * levels, np.nan, np.nan)

    # detrend for stability
    x = x - np.mean(x)

    detail_E = []
    last_d1_abs = np.nan
    a = x.copy()

    for lv in range(1, levels + 1):
        n = (a.size // 2) * 2
        a = a[:n]
        # Haar low/high
        lo = (a[0::2] + a[1::2]) / np.sqrt(2.0)
        hi = (a[0::2] - a[1::2]) / np.sqrt(2.0)

        e = float(np.sum(hi * hi))
        detail_E.append(e)

        if lv == 1 and hi.size > 0:
            last_d1_abs = float(np.abs(hi[-1]))

        a = lo

    approx_E = float(np.sum(a * a))
    return (detail_E, approx_E, last_d1_abs)


def _rolling_wavelet_features(s: pd.Series, window: int = 64, levels: int = 4) -> pd.DataFrame:
    """
    For each t, compute energies from s[t-window+1:t] (inclusive).
    Output aligned with s.index.
    """
    x = _num(s).values
    n = len(x)

    cols = {}
    for lv in range(1, levels + 1):
        cols[f"wv_E_d{lv}"] = np.full(n, np.nan, dtype=float)
    cols["wv_E_aL"] = np.full(n, np.nan, dtype=float)
    cols["wv_burst_d1_abs"] = np.full(n, np.nan, dtype=float)
    cols["wv_E_total"] = np.full(n, np.nan, dtype=float)
    cols["wv_hf_ratio"] = np.full(n, np.nan, dtype=float)   # (d1+d2)/total
    cols["wv_lf_ratio"] = np.full(n, np.nan, dtype=float)   # (d3+d4+aL)/total

    for i in range(window - 1, n):
        w = x[i - window + 1 : i + 1]
        dE, aE, burst = _haar_dwt_energies(w, levels=levels)
        if not np.isfinite(aE):
            continue

        total = aE + float(np.nansum(dE))
        if total <= 1e-18:
            continue

        for lv, e in enumerate(dE, start=1):
            cols[f"wv_E_d{lv}"][i] = e
        cols["wv_E_aL"][i] = aE
        cols["wv_burst_d1_abs"][i] = burst
        cols["wv_E_total"][i] = total

        hf = float(np.nansum(dE[:2]))  # d1+d2
        lf = float(np.nansum(dE[2:])) + aE
        cols["wv_hf_ratio"][i] = hf / total
        cols["wv_lf_ratio"][i] = lf / total

    return pd.DataFrame(cols, index=s.index)


# ==============================================================================
# Helper
# ==============================================================================
class _WaveletInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.ctx = ctx
        self.original_index = data.index

        df = data.copy()
        df["_ts"] = _to_ts(df["date"])
        self._sig = (len(df), df["_ts"].min(), df["_ts"].max())
        if "valid_cs_mask" in df.columns:
            df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"]).astype(bool)
        else:
            df["valid_cs_mask"] = True
        df.sort_values(["stock_code", "_ts"], inplace=True)

        # ensure required flags
        for c in ["is_limit_up", "is_limit_down", "is_zhaban"]:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

        df["pct_chg"] = _num(df["pct_chg"])

        # optional columns
        for c in ["volume", "circ_mv", "high", "low", "close", "open"]:
            if c in df.columns:
                df[c] = _num(df[c])

        self.df = df
        self._precompute()

    # --------------------------------------------------------------------------
    # Market daily series & wavelet gates
    # --------------------------------------------------------------------------
    def _build_market_daily(self) -> pd.DataFrame:
        df_stat = self.df[self.df["valid_cs_mask"]].copy()
        g = df_stat.groupby("_ts", sort=True)

        cnt = g.size().astype("float64").rename("mkt_count")
        limit_up = g["is_limit_up"].sum().astype("float64").rename("mkt_limit_up")
        limit_down = g["is_limit_down"].sum().astype("float64").rename("mkt_limit_down")
        zhaban = g["is_zhaban"].sum().astype("float64").rename("mkt_zhaban")

        out = pd.concat([cnt, limit_up, limit_down, zhaban], axis=1).reset_index()
        out["mkt_limit_up_ratio"] = _safe_div(out["mkt_limit_up"], out["mkt_count"]).fillna(0.0)
        out["mkt_limit_down_ratio"] = _safe_div(out["mkt_limit_down"], out["mkt_count"]).fillna(0.0)
        out["mkt_zhaban_ratio"] = _safe_div(out["mkt_zhaban"], out["mkt_count"]).fillna(0.0)

        # emotion heat proxy（与你 breadth 文件同一语义）
        out["mkt_emotion_heat"] = (
            out["mkt_limit_up_ratio"]
            - 1.2 * out["mkt_limit_down_ratio"]
            - 0.6 * out["mkt_zhaban_ratio"]
        )

        out = out.sort_values("_ts").reset_index(drop=True)
        return out[["_ts", "mkt_emotion_heat"]]

    def _precompute(self):
        df = self.df

        # --- market daily + wavelet features (cached) ---
        key = "_wavelet_mkt_daily_v1"
        daily = self.ctx.extra.get(key)
        if daily is None:
            daily = self._build_market_daily()

            # wavelet
            wv = _rolling_wavelet_features(daily["mkt_emotion_heat"], window=64, levels=4)
            daily = pd.concat([daily, wv], axis=1)

            # add time-series normalizations (robust z) for gates
            for col in ["wv_hf_ratio", "wv_lf_ratio", "wv_E_total", "wv_burst_d1_abs"]:
                daily[f"{col}_rz20"] = _rolling_robust_z1d(daily[col], window=20, min_periods=20)

            # energy change rate (vol-of-vol proxy)
            daily["wv_E_total_delta5"] = daily["wv_E_total"] - daily["wv_E_total"].shift(5)
            daily["wv_E_total_delta5_rz20"] = _rolling_robust_z1d(daily["wv_E_total_delta5"], window=20, min_periods=20)

            # clean
            for c in daily.columns:
                if c == "_ts":
                    continue
                daily[c] = _num(daily[c]).fillna(0.0)

            self.ctx.extra[key] = daily

        self.daily = daily.set_index("_ts")

        # --- stock alphas ---
        # mom / reversal
        mom5 = df.groupby("stock_code", sort=False)["pct_chg"].transform(lambda s: s.rolling(5, min_periods=3).sum())
        rev3 = -df.groupby("stock_code", sort=False)["pct_chg"].transform(lambda s: s.rolling(3, min_periods=2).sum())

        df["mom5_z"] = _cs_robust_z(mom5, df["_ts"], valid_cs_mask=df["valid_cs_mask"])
        df["rev3_z"] = _cs_robust_z(rev3, df["_ts"], valid_cs_mask=df["valid_cs_mask"])

        # vol20 (defensive alpha uses -vol)
        vol20 = df.groupby("stock_code", sort=False)["pct_chg"].transform(lambda s: s.rolling(20, min_periods=10).std())
        df["vol20_z"] = _cs_robust_z(vol20, df["_ts"], valid_cs_mask=df["valid_cs_mask"])

        # turnover z (liquidity/attention)
        # Prefer exchange-reported turnover_rate if available; otherwise fall back to volume/circ_mv proxy.
        if "turnover_rate" in df.columns:
            to = pd.to_numeric(df["turnover_rate"], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
        elif "volume" in df.columns and "circ_mv" in df.columns:
            to = _safe_div(df["volume"], df["circ_mv"]).replace([np.inf, -np.inf], np.nan).astype("float64")
        else:
            to = np.nan
        if isinstance(to, pd.Series):
            df["turnover_z"] = _cs_robust_z(to, df["_ts"], valid_cs_mask=df["valid_cs_mask"])
        else:
            df["turnover_z"] = df["mom5_z"] * 0.0


        # beta60 (optional; requires market_data)
        df["beta60_z"] = 0.0
        if getattr(self.ctx, "market_data", None) is not None:
            mkt = self.ctx.require_market().copy()
            mkt["_ts"] = _to_ts(mkt["date"])
            mkt = mkt.sort_values("_ts")
            if "daily_change_pct" in mkt.columns:
                m = mkt.set_index("_ts")["daily_change_pct"]
                tmp = df[["_ts", "stock_code", "pct_chg"]].copy()
                tmp["m_ret"] = tmp["_ts"].map(m)
                tmp["pct_chg"] = _num(tmp["pct_chg"])
                tmp["m_ret"] = _num(tmp["m_ret"])

                def _beta(grp: pd.DataFrame) -> pd.Series:
                    r = grp["pct_chg"]
                    mr = grp["m_ret"]
                    cov = r.rolling(60, min_periods=30).cov(mr)
                    var = mr.rolling(60, min_periods=30).var()
                    return cov / (var + 1e-9)

                beta = tmp.groupby("stock_code", sort=False).apply(_beta).reset_index(level=0, drop=True)
                beta = beta.reindex(df.index)
                df["beta60_z"] = _cs_robust_z(beta, df["_ts"], valid_cs_mask=df["valid_cs_mask"])

        self.df = df

    # --------------------------------------------------------------------------
    # factor outputs
    # --------------------------------------------------------------------------
    def _gate(self, name: str) -> pd.Series:
        # map daily gate to each row
        return self.df["_ts"].map(self.daily[name]).astype("float64")

    def _final(self, raw: pd.Series) -> pd.Series:
        out = _cs_robust_z(raw, self.df["_ts"], clip_z=6.0, fillna=0.0, valid_cs_mask=self.df["valid_cs_mask"])
        return out.reindex(self.original_index).astype("float64")

    def factor(self, name: str) -> pd.Series:
        df = self.df

        # ----------- Regime: choppy vs trend (hf vs lf) -----------
        if name == "wv_chop_gate_reversal":
            # 高频占比高 => 更震荡/噪声 => 偏好反转（rev3）
            gate = self._gate("wv_hf_ratio_rz20")
            alpha = df["rev3_z"]
            return self._final(gate * alpha)

        if name == "wv_trend_gate_momentum":
            # 低频占比高 => 趋势主导 => 偏好动量（mom5）
            gate = self._gate("wv_lf_ratio_rz20")
            alpha = df["mom5_z"]
            return self._final(gate * alpha)

        # ----------- Burst / turning points -----------
        if name == "wv_burst_gate_liquidity":
            # 情绪突变（d1 burst）=> 更依赖流动性/注意力 => 偏好高换手
            gate = self._gate("wv_burst_d1_abs_rz20")
            alpha = df["turnover_z"]
            return self._final(gate * alpha)

        if name == "wv_burst_gate_lowvol_defensive":
            # 突变期更容易尾部 => 偏好低波（-vol20）
            gate = self._gate("wv_burst_d1_abs_rz20")
            alpha = -df["vol20_z"]
            return self._final(gate * alpha)

        # ----------- Energy / vol-of-vol -----------
        if name == "wv_energy_gate_lowvol":
            # 总能量高（情绪/波动强）=> 防御倾向：低波
            gate = self._gate("wv_E_total_rz20")
            alpha = -df["vol20_z"]
            return self._final(gate * alpha)

        if name == "wv_energy_delta_gate_momentum":
            # 能量上升（波动/情绪加速）=> “顺势加强”型：动量更有效（但风险更大）
            gate = self._gate("wv_E_total_delta5_rz20")
            alpha = df["mom5_z"]
            return self._final(gate * alpha)

        # ----------- Optional: beta exposure in trend regimes -----------
        if name == "wv_trend_gate_beta":
            # 趋势主导时，系统性暴露更重要：高beta更占优（需要 market_data 才有意义）
            gate = self._gate("wv_lf_ratio_rz20")
            alpha = df["beta60_z"]
            return self._final(gate * alpha)

        return pd.Series(0.0, index=self.original_index, dtype="float64")


def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _WaveletInteractionHelper:
    """
    Safe helper cache for walk-forward:
    reuse only when (len, min_ts, max_ts) signature matches current data window.
    """
    key = "_wavelet_interaction_helper_v2"
    # compute signature from current data
    _ts = _to_ts(data["date"])
    sig = (len(data), _ts.min(), _ts.max())

    h = None
    if hasattr(ctx, "extra"):
        h = ctx.extra.get(key)

    if h is None or getattr(h, "_sig", None) != sig:
        h = _WaveletInteractionHelper(data, ctx)
        h._sig = sig
        if hasattr(ctx, "extra"):
            ctx.extra[key] = h
    return h


# ==============================================================================
# Registrations
# ==============================================================================
def _c(name: str):
    def _f(d: pd.DataFrame, c: FactorContext) -> pd.Series:
        return _get_helper(d, c).factor(name)
    return _f


wv_chop_gate_reversal = register_factor(FactorDefinition(
    name="wv_chop_gate_reversal",
    required_columns=_REQ_BASE,
    lookback_days=90,  # 64d window + 20d rz warmup buffer
    calculator=_c("wv_chop_gate_reversal"),
    description="Wavelet: 高频占比(震荡)门控 × 短期反转（rev3）"
))

wv_trend_gate_momentum = register_factor(FactorDefinition(
    name="wv_trend_gate_momentum",
    required_columns=_REQ_BASE,
    lookback_days=90,
    calculator=_c("wv_trend_gate_momentum"),
    description="Wavelet: 低频占比(趋势)门控 × 短期动量（mom5）"
))

wv_burst_gate_liquidity = register_factor(FactorDefinition(
    name="wv_burst_gate_liquidity",
    required_columns=_REQ_BASE.union({"volume", "circ_mv"}),
    lookback_days=90,
    calculator=_c("wv_burst_gate_liquidity"),
    description="Wavelet: 情绪突变(burst)门控 × 流动性/注意力（turnover_z）"
))

wv_burst_gate_lowvol_defensive = register_factor(FactorDefinition(
    name="wv_burst_gate_lowvol_defensive",
    required_columns=_REQ_BASE,
    lookback_days=90,
    calculator=_c("wv_burst_gate_lowvol_defensive"),
    description="Wavelet: 情绪突变(burst)门控 × 低波防御（-vol20）"
))

wv_energy_gate_lowvol = register_factor(FactorDefinition(
    name="wv_energy_gate_lowvol",
    required_columns=_REQ_BASE,
    lookback_days=90,
    calculator=_c("wv_energy_gate_lowvol"),
    description="Wavelet: 总能量(情绪/波动强度)门控 × 低波（-vol20）"
))

wv_energy_delta_gate_momentum = register_factor(FactorDefinition(
    name="wv_energy_delta_gate_momentum",
    required_columns=_REQ_BASE,
    lookback_days=100,
    calculator=_c("wv_energy_delta_gate_momentum"),
    description="Wavelet: 能量上升(加速)门控 × 动量（mom5）"
))

wv_trend_gate_beta = register_factor(FactorDefinition(
    name="wv_trend_gate_beta",
    required_columns=_REQ_BASE,   # beta 需要 market_data，但列需求仍是本表（market 在 ctx）
    requires_market_data=True,
    lookback_days=140,
    calculator=_c("wv_trend_gate_beta"),
    description="Wavelet: 趋势(低频占比)门控 × Beta暴露（需要 market_data）"
))



def _vmask(data: pd.DataFrame) -> pd.Series:
    """Return boolean valid cross-section mask aligned to `data.index`."""
    if isinstance(data, pd.DataFrame) and ("valid_cs_mask" in data.columns):
        return _clean_bool(data["valid_cs_mask"]).astype(bool)
    return pd.Series(True, index=data.index)

__all__ = [
    "wv_chop_gate_reversal",
    "wv_trend_gate_momentum",
    "wv_burst_gate_liquidity",
    "wv_burst_gate_lowvol_defensive",
    "wv_energy_gate_lowvol",
    "wv_energy_delta_gate_momentum",
    "wv_trend_gate_beta",
]