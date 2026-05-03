"""
Sector Wavelet Gate × Intra-Sector Leadership Interaction Factors - 优化版（无开关）

你当前训练设定：按日 group 的 LightGBM lambdarank（6桶标签来自组内rank）。
要让“行业状态”进入组内排序，必须构造成 (sector_state_t,sector × stock_alpha_i,t) 的交互项，
否则行业状态若在同一行业内对所有股票相同，就无法提升该行业内部的排序。

本文件做的事：
1) 用“行业内股票的涨停/跌停/炸板”统计构造行业情绪序列 sector_emotion_heat(ts, sector)
   - limit_up_ratio - 1.2*limit_down_ratio - 0.6*zhaban_ratio
2) 对每个行业序列做 rolling Haar wavelet（窗口=64天，levels=4），提取：
   - 高频能量占比（chop/震荡）
   - 低频能量占比（trend/趋势）
   - d1 突变强度（burst）
   - 总能量与能量变化率（energy / acceleration）
   并对 gate 做 rolling robust z（20天）以稳健化
3) 构造“行业门控 × 行业内 alpha”的交互因子：
   - 震荡门控 × 行业内短期反转
   - 趋势门控 × 行业内短期动量
   - 突变门控 × 行业内领先（动量领先）
   - 能量门控 × 行业内低波防御
   - 能量加速门控 × 行业内动量

输出：
- 统一按日横截面 robust z（median/MAD）+ clip + fillna(0)
- 仅使用 ctx.extra 做缓存（你的 FactorContext 无 cache/_ctx_cache）

依赖：
- 行业映射：ctx.extra["stock_to_sw_sector"] (同你 sector_interaction_optimized_noswitch.py)
- 输入数据列：stock_code,date,pct_chg,is_limit_up,is_limit_down,is_zhaban
- 若无映射，则退化为 "All" 一个行业（仍可运行，但行业门控意义弱）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_CLIP_Z = 6.0
_EPS = 1e-12
_OUT_DTYPE = "float32"
_REQ = {"stock_code", "date", "pct_chg", "is_limit_up", "is_limit_down", "is_zhaban"}


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


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    a = _num(a)
    b = _num(b)
    return a / (b.replace(0.0, np.nan) + eps)

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


def _cs_robust_z_within(
    x: pd.Series,
    group_keys,
    *,
    valid_cs_mask: pd.Series | None = None,
    clip_z: float = _CLIP_Z,
) -> pd.Series:
    """在任意分组内做 robust z（median/MAD）并 clip。

    用于“行业内横截面”标准化，例如 group_keys=[ts, sector_code]。
    统计 median/MAD 时仅使用 valid_cs_mask==True 的样本；mask=False 行输出置 0。
    """
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")

    # normalize group keys
    if isinstance(group_keys, pd.Series):
        keys = group_keys
    elif isinstance(group_keys, (list, tuple)) and len(group_keys) > 0 and all(isinstance(k, pd.Series) for k in group_keys):
        keys = list(group_keys)
    else:
        raise TypeError("group_keys must be a pandas Series or a list/tuple of pandas Series")

    if valid_cs_mask is not None:
        m = _clean_bool(valid_cs_mask)
        s_stat = s.where(m, np.nan)
    else:
        m = None
        s_stat = s

    g = s_stat.groupby(keys, sort=False)
    med = g.transform("median")
    mad = g.transform(lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))))
    z = (s - med) / (1.4826 * mad + _EPS)
    z = z.clip(-clip_z, clip_z)

    if m is not None:
        z = z.where(m, np.nan)

    return z.fillna(0.0).astype(_OUT_DTYPE)


def _rolling_robust_z1d(s: pd.Series, window: int, min_periods: int) -> pd.Series:
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
# Haar wavelet
# ==============================================================================
def _haar_dwt_energies(w: np.ndarray, levels: int = 4) -> tuple[list[float], float, float]:
    x = np.asarray(w, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < (2 ** (levels + 1)):
        return ([np.nan] * levels, np.nan, np.nan)

    x = x - np.mean(x)
    a = x.copy()
    detail_E = []
    last_d1_abs = np.nan

    for lv in range(1, levels + 1):
        n = (a.size // 2) * 2
        a = a[:n]
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
    x = _num(s).values
    n = len(x)

    cols = {}
    for lv in range(1, levels + 1):
        cols[f"wv_E_d{lv}"] = np.full(n, np.nan, dtype=float)
    cols["wv_E_aL"] = np.full(n, np.nan, dtype=float)
    cols["wv_burst_d1_abs"] = np.full(n, np.nan, dtype=float)
    cols["wv_E_total"] = np.full(n, np.nan, dtype=float)
    cols["wv_hf_ratio"] = np.full(n, np.nan, dtype=float)
    cols["wv_lf_ratio"] = np.full(n, np.nan, dtype=float)

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
        hf = float(np.nansum(dE[:2]))
        lf = float(np.nansum(dE[2:])) + aE
        cols["wv_hf_ratio"][i] = hf / total
        cols["wv_lf_ratio"][i] = lf / total

    return pd.DataFrame(cols, index=s.index)


# ==============================================================================
# Helper
# ==============================================================================
class _SectorWaveletInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.ctx = ctx
        self.original_index = data.index

        df = data.copy()
        df["_ts"] = _to_ts(df["date"])
        if "valid_cs_mask" in df.columns:
            df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"]).astype(bool)
        else:
            df["valid_cs_mask"] = True

        # sector mapping
        sector_map = ctx.extra.get("stock_to_sw_sector", {})
        if sector_map:
            df["sector_code"] = df["stock_code"].map(sector_map).fillna("Unknown")
        else:
            df["sector_code"] = "All"

        df.sort_values(["stock_code", "_ts"], inplace=True)

        # signature for cache validation
        self._sig = (len(df), df["_ts"].min(), df["_ts"].max())

        df["pct_chg"] = _num(df["pct_chg"])  # keep full TS; mask only in CS stats
        for c in ["is_limit_up", "is_limit_down", "is_zhaban"]:
            df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype("int64")

        self.df = df
        self._precompute()

    # --------------------------------------------------------------------------
    # sector daily emotion + wavelet gates (cached)
    # --------------------------------------------------------------------------
    def _build_sector_daily(self) -> pd.DataFrame:
        x = self.df[["_ts", "sector_code", "is_limit_up", "is_limit_down", "is_zhaban", "valid_cs_mask"]].copy()
        x = x[x["valid_cs_mask"]].copy()
        g = x.groupby(["_ts", "sector_code"], sort=True)

        cnt = g.size().astype("float64").rename("cnt")
        up = g["is_limit_up"].sum().astype("float64").rename("up")
        dn = g["is_limit_down"].sum().astype("float64").rename("dn")
        zb = g["is_zhaban"].sum().astype("float64").rename("zb")

        out = pd.concat([cnt, up, dn, zb], axis=1).reset_index()
        out["up_ratio"] = _safe_div(out["up"], out["cnt"]).fillna(0.0)
        out["dn_ratio"] = _safe_div(out["dn"], out["cnt"]).fillna(0.0)
        out["zb_ratio"] = _safe_div(out["zb"], out["cnt"]).fillna(0.0)

        out["sector_emotion_heat"] = out["up_ratio"] - 1.2 * out["dn_ratio"] - 0.6 * out["zb_ratio"]
        return out[["_ts", "sector_code", "sector_emotion_heat"]].sort_values(["sector_code", "_ts"]).reset_index(drop=True)

    def _precompute(self):
        # NOTE: sector_daily depends on the provided `data` slice (dates/universe),
        # so caching it globally in ctx.extra can create subtle mismatches when you
        # call factor generation on different windows. We compute it per-helper.
        base = self._build_sector_daily()

        # wavelet features on sector time-series
        feats = []
        for col in ["sector_ret", "sector_vol", "sector_limit_up_ratio", "sector_turnover_z"]:
            w = _rolling_wavelet_features(base, col=col, window=64, levels=3, prefix=col)

            # robust z within sector time-series, then lag 1 to be causal
            for c in [x for x in w.columns if x.startswith(col + "_wv_")]:
                w[c] = _rolling_robust_z1d(w[c], window=120, minp=40).shift(1)

            feats.append(w)

        # merge all features
        out = feats[0]
        for t in feats[1:]:
            out = out.merge(t, on=["sector_code", "_ts"], how="left")

        # keep only wavelet feature columns + keys
        keep_cols = ["sector_code", "_ts"] + [c for c in out.columns if c.endswith("_z")]
        daily_long = out[keep_cols].copy()

        self.sector_daily = daily_long  # long table

        # map gates onto each stock row by (ts, sector)
        df = self.df
        # map to stocks (by date, sector)
        # NOTE: mapping also depends on current `data` slice; avoid caching huge mapped tables in ctx.extra.
        tmp = df[["_ts", "sector_code"]].copy()
        tmp["_row_id"] = np.arange(len(tmp), dtype=np.int64)

        # de-dup keys to shrink join, then merge back by _row_id (stable, avoids index/order pitfalls)
        merged = tmp.drop_duplicates(subset=["_ts", "sector_code"]).merge(
            self.sector_daily,
            on=["_ts", "sector_code"],
            how="left",
            sort=False,
        )
        mapped = tmp.merge(merged, on=["_row_id", "_ts", "sector_code"], how="left", sort=False)
        mapped = mapped.drop(columns=["_row_id"])
        for c in ["wv_hf_ratio_rz20","wv_lf_ratio_rz20","wv_burst_d1_abs_rz20","wv_E_total_rz20","wv_E_total_delta5_rz20"]:
            df[c] = _num(mapped[c]).fillna(0.0)

        # -------- stock alphas (within sector) --------
        g = df.groupby("stock_code", sort=False)

        mom5 = g["pct_chg"].transform(lambda s: s.rolling(5, min_periods=3).sum())
        rev3 = -g["pct_chg"].transform(lambda s: s.rolling(3, min_periods=2).sum())
        vol20 = g["pct_chg"].transform(lambda s: s.rolling(20, min_periods=10).std())

        # within-sector cross-sectional standardization
        df["mom5_intra_z"] = _cs_robust_z_within(mom5, [df["_ts"], df["sector_code"]], valid_cs_mask=df["valid_cs_mask"])
        df["rev3_intra_z"] = _cs_robust_z_within(rev3, [df["_ts"], df["sector_code"]], valid_cs_mask=df["valid_cs_mask"])
        df["vol20_intra_z"] = _cs_robust_z_within(vol20, [df["_ts"], df["sector_code"]], valid_cs_mask=df["valid_cs_mask"])

        self.df = df

    # --------------------------------------------------------------------------
    def _final(self, raw: pd.Series) -> pd.Series:
        out = _cs_robust_z(raw, self.df["_ts"], valid_cs_mask=self.df["valid_cs_mask"])
        return out.reindex(self.original_index)

    def factor(self, name: str) -> pd.Series:
        df = self.df

        if name == "sw_wv_chop_gate_intra_reversal":
            return self._final(df["wv_hf_ratio_rz20"] * df["rev3_intra_z"])

        if name == "sw_wv_trend_gate_intra_momentum":
            return self._final(df["wv_lf_ratio_rz20"] * df["mom5_intra_z"])

        if name == "sw_wv_burst_gate_intra_leader":
            # burst × (momentum leader) — use intra mom5 z as leader proxy
            return self._final(df["wv_burst_d1_abs_rz20"] * df["mom5_intra_z"])

        if name == "sw_wv_energy_gate_intra_lowvol":
            return self._final(df["wv_E_total_rz20"] * (-df["vol20_intra_z"]))

        if name == "sw_wv_energy_delta_gate_intra_momentum":
            return self._final(df["wv_E_total_delta5_rz20"] * df["mom5_intra_z"])

        return pd.Series(0.0, index=self.original_index, dtype="float64")


def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _SectorWaveletInteractionHelper:
    """Per-call helper with safe caching.

    We cache only when the incoming `data` slice matches (len, min_ts, max_ts).
    This avoids subtle bugs when you generate factors for different windows but reuse ctx.extra.
    """
    key = "_sector_wavelet_helper_v1"
    df_ts = _to_ts(data["date"])
    sig = (len(data), df_ts.min(), df_ts.max())

    h = ctx.extra.get(key) if hasattr(ctx, "extra") else None
    if h is None or getattr(h, "_sig", None) != sig:
        h = _SectorWaveletInteractionHelper(data, ctx)
        if hasattr(ctx, "extra"):
            ctx.extra[key] = h
    return h


def _c(name: str):
    def _f(d: pd.DataFrame, c: FactorContext) -> pd.Series:
        return _get_helper(d, c).factor(name)
    return _f


# ==============================================================================
# Factors
# ==============================================================================
sw_wv_chop_gate_intra_reversal = register_factor(FactorDefinition(
    name="sw_wv_chop_gate_intra_reversal",
    required_columns=_REQ,
    lookback_days=120,  # 64 + 20 + buffer
    calculator=_c("sw_wv_chop_gate_intra_reversal"),
    description="行业Wavelet：高频占比(震荡)门控 × 行业内短期反转（rev3_intra_z）",
))

sw_wv_trend_gate_intra_momentum = register_factor(FactorDefinition(
    name="sw_wv_trend_gate_intra_momentum",
    required_columns=_REQ,
    lookback_days=120,
    calculator=_c("sw_wv_trend_gate_intra_momentum"),
    description="行业Wavelet：低频占比(趋势)门控 × 行业内短期动量（mom5_intra_z）",
))

sw_wv_burst_gate_intra_leader = register_factor(FactorDefinition(
    name="sw_wv_burst_gate_intra_leader",
    required_columns=_REQ,
    lookback_days=120,
    calculator=_c("sw_wv_burst_gate_intra_leader"),
    description="行业Wavelet：突变(burst)门控 × 行业内领先（动量领先 proxy）",
))

sw_wv_energy_gate_intra_lowvol = register_factor(FactorDefinition(
    name="sw_wv_energy_gate_intra_lowvol",
    required_columns=_REQ,
    lookback_days=120,
    calculator=_c("sw_wv_energy_gate_intra_lowvol"),
    description="行业Wavelet：总能量(情绪/波动强度)门控 × 行业内低波（-vol20_intra_z）",
))

sw_wv_energy_delta_gate_intra_momentum = register_factor(FactorDefinition(
    name="sw_wv_energy_delta_gate_intra_momentum",
    required_columns=_REQ,
    lookback_days=140,
    calculator=_c("sw_wv_energy_delta_gate_intra_momentum"),
    description="行业Wavelet：能量上升(加速)门控 × 行业内动量（mom5_intra_z）",
))



def _vmask(data: pd.DataFrame) -> pd.Series:
    """Return boolean valid cross-section mask aligned to `data.index`."""
    if isinstance(data, pd.DataFrame) and ("valid_cs_mask" in data.columns):
        return _clean_bool(data["valid_cs_mask"]).astype(bool)
    return pd.Series(True, index=data.index)

__all__ = [
    "sw_wv_chop_gate_intra_reversal",
    "sw_wv_trend_gate_intra_momentum",
    "sw_wv_burst_gate_intra_leader",
    "sw_wv_energy_gate_intra_lowvol",
    "sw_wv_energy_delta_gate_intra_momentum",
]