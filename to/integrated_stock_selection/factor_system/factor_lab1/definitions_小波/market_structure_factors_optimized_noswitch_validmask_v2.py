"""
市场结构与行业宽度因子 - valid_cs_mask 适配版（输出 float32）

在 market_structure_factors_optimized_noswitch.py 基础上：
- 增加 valid_cs_mask：
  1) 行业收益/行业breadth 等行业聚合统计仅使用 mask=True 样本
  2) 按日横截面 robust z 的 median/MAD 仅使用 mask=True 样本统计
  3) mask=False 行输出置 0（中性）
- 输出统一 float32（内部 float64）

因子名保持不变（4个）：
- mkt_sector_dispersion
- sector_mkt_corr_20d
- intra_sector_breadth
- stock_sector_pos_align
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_EPS = 1e-12
_CLIP_Z = 6.0
_OUT_DTYPE = "float32"


def _to_ts(date_series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(date_series):
        return pd.to_datetime(date_series, errors="coerce")
    ds = date_series.astype(str).str.replace("-", "", regex=False)
    return pd.to_datetime(ds, format="%Y%m%d", errors="coerce")


def _clean_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(1.0) > 0.5


def _cs_robust_z(
    x: pd.Series,
    group_keys,
    *,
    valid_cs_mask: pd.Series | None = None,
    mad_scale: float = 1.4826,
    eps: float = 1e-12,
    clip_z: float = _CLIP_Z,
) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    if valid_cs_mask is not None:
        m = _clean_bool(valid_cs_mask)
        s_stat = s.where(m, np.nan)
    else:
        m = None
        s_stat = s

    g = s_stat.groupby(group_keys, sort=False)
    med = g.transform("median")
    mad = g.transform(lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))))
    z = (s - med) / (mad_scale * mad + eps)
    z = z.clip(-clip_z, clip_z)

    if m is not None:
        z = z.where(m, np.nan)

    return z.fillna(0.0).astype(_OUT_DTYPE)


def _rolling_robust_z_1d(
    s: pd.Series,
    *,
    window: int = 252,
    min_periods: int = 60,
    mad_scale: float = 1.4826,
    eps: float = 1e-12,
    clip_z: float = _CLIP_Z,
) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")

    def _med(v):
        v = v[~np.isnan(v)]
        return np.nan if v.size == 0 else float(np.median(v))

    def _mad(v):
        v = v[~np.isnan(v)]
        if v.size == 0:
            return np.nan
        m = float(np.median(v))
        return float(np.median(np.abs(v - m)))

    roll_med = x.rolling(window, min_periods=min_periods).apply(_med, raw=True)
    roll_mad = x.rolling(window, min_periods=min_periods).apply(_mad, raw=True)
    z = (x - roll_med) / (mad_scale * roll_mad + eps)
    z = z.clip(-clip_z, clip_z)
    return z.fillna(0.0).astype("float64")


def _get_market_data(ctx: FactorContext) -> pd.DataFrame:
    market = ctx.require_market().copy()
    market["_ts"] = _to_ts(market["date"]).dt.normalize()
    return market.sort_values("_ts").reset_index(drop=True)


class _MarketStructureHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.ctx = ctx
        self.original_index = data.index

        df = data.copy()
        df["_ts"] = _to_ts(df["date"]).dt.normalize()
        df["stock_code"] = df["stock_code"].astype(str)

        if "valid_cs_mask" in df.columns:
            df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"])
        else:
            df["valid_cs_mask"] = True

        sector_map = ctx.extra.get("stock_to_sw_sector", {})
        df["sector_code"] = df["stock_code"].map(sector_map).fillna("Unknown").astype(str)

        df.sort_values(["stock_code", "_ts"], inplace=True)
        self.data = df
        self._precompute()

    def _precompute(self) -> None:
        df = self.data

        # 只用 valid_cs_mask=True 的样本做行业聚合统计
        base = df[df["valid_cs_mask"]].copy()

        s_ret = pd.to_numeric(base["pct_chg"], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
        s_ret_wide = (
            base.assign(_r=s_ret)
            .groupby(["_ts", "sector_code"])["_r"]
            .mean()
            .unstack("sector_code")
            .sort_index()
        )

        # market return series (optional; only needed for sector_mkt_corr_20d)
        try:
            mkt_df = _get_market_data(self.ctx).copy()
            mkt_df["_ts"] = _to_ts(mkt_df["date"])
            mkt_df = mkt_df.sort_values("_ts")
            mkt_ret = pd.to_numeric(mkt_df["daily_change_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
            mkt_ret.index = mkt_df["_ts"].values
            mkt_ret = mkt_ret[~mkt_ret.index.duplicated(keep="first")]
            mkt_ret = mkt_ret.reindex(s_ret_wide.index)
        except Exception:
            # if market_data not provided, we can still compute sector dispersion/breadth
            mkt_ret = pd.Series(np.nan, index=s_ret_wide.index, dtype="float64")

        # 市场级：行业收益离散度（同日常数，做时间序列 rolling robust z）
        mkt_sector_disp = s_ret_wide.std(axis=1, skipna=True)
        self.mkt_sector_dispersion_z = _rolling_robust_z_1d(mkt_sector_disp, window=252, min_periods=60)

        # 行业-市场相关性 rolling corr（20）
        self.s_mkt_corr = s_ret_wide.rolling(20, min_periods=10).corr(mkt_ret)

        # 行业 breadth：行业内上涨占比
        is_pos = (s_ret > 0).astype("float64")
        s_breadth = (
            base.assign(is_pos=is_pos)
            .groupby(["_ts", "sector_code"])["is_pos"]
            .mean()
            .unstack("sector_code")
            .sort_index()
        )
        self.s_breadth = s_breadth

        # 个股位置锚点（120d，严格因果，使用全样本本身）
        window_size = 120
        min_p = 40
        high_anchor = df.groupby("stock_code", sort=False)["high"].transform(lambda x: x.rolling(window_size, min_periods=min_p).max())
        low_anchor = df.groupby("stock_code", sort=False)["low"].transform(lambda x: x.rolling(window_size, min_periods=min_p).min())

        close = pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
        denom = (high_anchor - low_anchor).astype("float64") + 1e-8
        df["stock_pos"] = ((close - low_anchor) / denom).clip(0.0, 1.0)
        self.data = df

        s_pos_mean = (
            df[df["valid_cs_mask"]]
            .groupby(["_ts", "sector_code"], sort=False)["stock_pos"]
            .mean()
            .unstack("sector_code")
            .sort_index()
        )
        self.s_pos_mean = s_pos_mean

    def compute(self, factor_name: str) -> pd.Series:
        df = self.data
        m = df["valid_cs_mask"]

        if factor_name == "mkt_sector_dispersion":
            v = df["_ts"].map(self.mkt_sector_dispersion_z)
            v = pd.to_numeric(v, errors="coerce").fillna(0.0).astype("float64")
            # mask=False 置 0，保持一致
            v = pd.Series(np.where(m.values, v.values, 0.0), index=df.index, dtype="float64")
            return v.reindex(self.original_index).astype(_OUT_DTYPE)

        if factor_name == "sector_mkt_corr_20d":
            s_val = self.s_mkt_corr.stack(dropna=False).rename("v")
            joined = df.join(s_val, on=["_ts", "sector_code"], how="left")["v"]
            out = _cs_robust_z(joined, df["_ts"], valid_cs_mask=m)
            return out.reindex(self.original_index).astype(_OUT_DTYPE)

        if factor_name == "intra_sector_breadth":
            s_val = self.s_breadth.stack(dropna=False).rename("v")
            joined = df.join(s_val, on=["_ts", "sector_code"], how="left")["v"]
            out = _cs_robust_z(joined, df["_ts"], valid_cs_mask=m)
            return out.reindex(self.original_index).astype(_OUT_DTYPE)

        if factor_name == "stock_sector_pos_align":
            s_pos = self.s_pos_mean.stack(dropna=False).rename("sp")
            joined = df.join(s_pos, on=["_ts", "sector_code"], how="left")
            diff = (joined["stock_pos"] - joined["sp"]).replace([np.inf, -np.inf], np.nan)
            out = _cs_robust_z(diff, df["_ts"], valid_cs_mask=m)
            return out.reindex(self.original_index).astype(_OUT_DTYPE)

        return pd.Series(0.0, index=self.original_index, dtype=_OUT_DTYPE)


def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _MarketStructureHelper:
    """Safe helper cache with (len, min_ts, max_ts) signature to avoid walk-forward cache poisoning."""
    key = "_market_structure_helper_validmask_v1"

    df_ts = _to_ts(data["date"])
    sig = (len(data), df_ts.min(), df_ts.max())

    h = ctx.extra.get(key)
    if h is None or getattr(h, "_sig", None) != sig:
        h = _MarketStructureHelper(data, ctx)
        ctx.extra[key] = h
    return h



def _calc_disp(d, c): return _get_helper(d, c).compute("mkt_sector_dispersion")
def _calc_corr(d, c): return _get_helper(d, c).compute("sector_mkt_corr_20d")
def _calc_breadth(d, c): return _get_helper(d, c).compute("intra_sector_breadth")
def _calc_align(d, c): return _get_helper(d, c).compute("stock_sector_pos_align")


mkt_sector_dispersion = register_factor(FactorDefinition(
    name="mkt_sector_dispersion",
    required_columns={"date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=30,
    calculator=_calc_disp,
))

sector_mkt_corr_20d = register_factor(FactorDefinition(
    name="sector_mkt_corr_20d",
    required_columns={"date", "pct_chg"},
    requires_market_data=True,
    requires_sector_data=True,
    lookback_days=40,
    calculator=_calc_corr,
))

intra_sector_breadth = register_factor(FactorDefinition(
    name="intra_sector_breadth",
    required_columns={"date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_calc_breadth,
))

stock_sector_pos_align = register_factor(FactorDefinition(
    name="stock_sector_pos_align",
    required_columns={"stock_code", "date", "high", "low", "close"},
    requires_sector_data=True,
    lookback_days=120,
    calculator=_calc_align,
))

__all__ = [
    "mkt_sector_dispersion",
    "sector_mkt_corr_20d",
    "intra_sector_breadth",
    "stock_sector_pos_align",
]
