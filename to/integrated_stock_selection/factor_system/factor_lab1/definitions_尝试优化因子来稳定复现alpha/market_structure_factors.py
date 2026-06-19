"""
市场结构与行业宽度因子 (Market Structure & Sector Breadth Factors)

【无未来泄露 · 不修复数据 · corr 双路输出（raw + gated）· 保锋利】
新增：
- sector_mkt_corr_20d_raw    : 原始行业-市场 rolling corr
- sector_mkt_corr_20d_gated  : 弱市场信号门控后的 rolling corr（用于复现旧 bug regime 的“弱市场信号”结构）

其余：
- mkt_sector_dispersion      : 市场行业离散度（time-series rolling 因果 clip）
- intra_sector_breadth       : 行业内上涨广度（按日横截面 MAD clip）
- stock_sector_pos_align     : 个股 120 日位置 - 行业均值位置（按日横截面 MAD clip）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


# ==============================================================================
# 1) Robust helpers (no lookahead)
# ==============================================================================

def _cs_mad_clip_by_date(values: pd.Series, date_series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    """
    按日横截面 MAD 截尾（只用当日截面，不可能泄露未来）。
    """
    if values.empty:
        return values.astype("float64")

    df = pd.DataFrame({"v": values.values, "d": date_series.values}, index=values.index)

    med = df.groupby("d")["v"].transform("median")
    abs_dev = (df["v"] - med).abs()
    mad = abs_dev.groupby(df["d"]).transform("median")

    scale = mad * 1.4826
    width = n_mad * scale

    out = df["v"].astype("float64")
    mask = mad > 1e-12
    out = out.where(~mask, out.clip(med - width, med + width))
    return out


def _rolling_adaptive_clip_timeseries(
    ts: pd.Series,
    n_mad: float = 3.0,
    window: int = 120,
    min_periods: int = 20,
    shift_threshold_by_1: bool = True,
) -> pd.Series:
    """
    市场级 time-series 的因果截尾：
    - rolling median/MAD（时间维）
    - 默认 shift(1)：今天用昨天之前的阈值（严格因果）
    """
    if ts.empty:
        return ts.astype("float64")

    ts = ts.sort_index().astype("float64")

    med = ts.rolling(window, min_periods=min_periods).median()
    mad = (ts - med).abs().rolling(window, min_periods=min_periods).median()

    if shift_threshold_by_1:
        med = med.shift(1)
        mad = mad.shift(1)

    width = n_mad * mad * 1.4826
    width = width.where(width > 1e-12, np.inf)

    upper = med + width
    lower = med - width
    return ts.clip(lower, upper).astype("float64")


def _get_market_data(ctx: FactorContext) -> pd.DataFrame:
    market = ctx.require_market()
    return market.sort_values("date").reset_index(drop=True)


def _market_signal_weak_gate(mkt_ret: pd.Series, vol_window: int = 20) -> pd.Series:
    """
    弱市场信号 gate（(0,1]，越大表示越“弱市场信号/更安静”）：

    gate = 1 / (1 + (vol_20 / vol_ref))

    - vol_20: 市场20日波动
    - vol_ref: 更长窗(120)的中位数尺度，保证 gate 本身不抖
    """
    v = mkt_ret.rolling(vol_window, min_periods=10).std()
    v_ref = v.rolling(120, min_periods=60).median()

    ratio = v / (v_ref + 1e-8)
    gate = 1.0 / (1.0 + ratio)

    # 温和限幅，避免 gate 自己极端化
    return gate.clip(0.1, 1.0).astype("float64")


# ==============================================================================
# 2) Helper class
# ==============================================================================

class _MarketStructureHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.data = data.copy()
        self.ctx = ctx

        sector_map = ctx.extra.get("stock_to_sw_sector", {})
        self.data["sector_code"] = self.data["stock_code"].map(sector_map).fillna("Unknown")

        # 排序保证 rolling/corr 可靠
        self.data.sort_values(["date", "stock_code"], inplace=True)

        self._precompute()

    def _precompute(self):
        # --- sector returns wide (date x sector) ---
        self.s_ret_wide = (
            self.data.groupby(["date", "sector_code"], sort=False)["pct_chg"]
            .mean()
            .unstack("sector_code")
            .sort_index()
        )

        # --- market returns (no fixing, your data is already corrected) ---
        mkt_df = _get_market_data(self.ctx).set_index("date")
        self.mkt_ret = mkt_df["daily_change_pct"].sort_index().astype("float64")

        # --- weak-market gate ---
        self.mkt_weak_gate = _market_signal_weak_gate(self.mkt_ret, vol_window=20)

        # --- market structure: dispersion (time-series) ---
        self.mkt_sector_dispersion = self.s_ret_wide.std(axis=1)

        # --- sector-market corr (rolling causal) ---
        # raw
        self.s_mkt_corr_raw = self.s_ret_wide.rolling(20, min_periods=10).corr(self.mkt_ret)

        # gated
        self.s_mkt_corr_gated = self.s_mkt_corr_raw.mul(self.mkt_weak_gate, axis=0)

        # --- sector breadth ---
        is_positive = (self.data["pct_chg"] > 0).astype(int)
        self.s_breadth = (
            self.data.assign(is_pos=is_positive)
            .groupby(["date", "sector_code"], sort=False)["is_pos"]
            .mean()
            .unstack("sector_code")
            .sort_index()
        )

        # --- stock position in 120d range ---
        window_size = 120
        min_p = 40

        high_anchor = (
            self.data.groupby("stock_code", sort=False)["high"]
            .transform(lambda x: x.rolling(window_size, min_periods=min_p).max())
        )
        low_anchor = (
            self.data.groupby("stock_code", sort=False)["low"]
            .transform(lambda x: x.rolling(window_size, min_periods=min_p).min())
        )

        denom = (high_anchor - low_anchor).replace(0, np.nan)
        self.data["stock_pos"] = (self.data["close"] - low_anchor) / (denom + 1e-8)
        self.data["stock_pos"] = self.data["stock_pos"].clip(0.0, 1.0)

        self.s_pos_mean = (
            self.data.groupby(["date", "sector_code"], sort=False)["stock_pos"]
            .mean()
            .unstack("sector_code")
            .sort_index()
        )

    def _map_sector_series_to_stock(self, s: pd.Series, colname: str) -> pd.Series:
        """
        s: MultiIndex Series (date, sector_code) -> value
        映射到 self.data 的每一行股票
        """
        s = s.rename(colname)
        out = self.data.join(s, on=["date", "sector_code"], how="left")[colname]
        return out

    def compute(self, factor_name: str) -> pd.Series:
        if factor_name == "mkt_sector_dispersion":
            ts = _rolling_adaptive_clip_timeseries(
                self.mkt_sector_dispersion,
                n_mad=3.0,
                window=120,
                min_periods=20,
                shift_threshold_by_1=True,
            )
            return self.data["date"].map(ts).astype("float64")

        if factor_name == "sector_mkt_corr_20d_raw":
            raw = self.s_mkt_corr_raw.stack()
            v = self._map_sector_series_to_stock(raw, "v")
            return _cs_mad_clip_by_date(v, self.data["date"], n_mad=3.0)

        if factor_name == "sector_mkt_corr_20d_gated":
            gated = self.s_mkt_corr_gated.stack()
            v = self._map_sector_series_to_stock(gated, "v")
            return _cs_mad_clip_by_date(v, self.data["date"], n_mad=3.0)

        if factor_name == "intra_sector_breadth":
            s_val = self.s_breadth.stack().rename("v")
            v = self.data.join(s_val, on=["date", "sector_code"], how="left")["v"]
            return _cs_mad_clip_by_date(v, self.data["date"], n_mad=3.0)

        if factor_name == "stock_sector_pos_align":
            s_pos = self.s_pos_mean.stack().rename("sp")
            joined = self.data.join(s_pos, on=["date", "sector_code"], how="left")
            raw = joined["stock_pos"] - joined["sp"]
            return _cs_mad_clip_by_date(raw, self.data["date"], n_mad=3.0)

        return pd.Series(0.0, index=self.data.index).astype("float64")


# ==============================================================================
# 3) Register
# ==============================================================================

def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _MarketStructureHelper:
    key = "_market_structure_helper_v4"
    if key not in ctx.extra:
        ctx.extra[key] = _MarketStructureHelper(data, ctx)
    return ctx.extra[key]


def _calc_disp(d, c): return _get_helper(d, c).compute("mkt_sector_dispersion")
def _calc_corr_raw(d, c): return _get_helper(d, c).compute("sector_mkt_corr_20d_raw")
def _calc_corr_gated(d, c): return _get_helper(d, c).compute("sector_mkt_corr_20d_gated")
def _calc_breadth(d, c): return _get_helper(d, c).compute("intra_sector_breadth")
def _calc_align(d, c): return _get_helper(d, c).compute("stock_sector_pos_align")


mkt_sector_dispersion = register_factor(FactorDefinition(
    name="mkt_sector_dispersion",
    required_columns={"date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=30,
    calculator=_calc_disp,
    description="市场行业离散度（time-series rolling 因果截尾）"
))

# ✅ 新增：raw
sector_mkt_corr_20d_raw = register_factor(FactorDefinition(
    name="sector_mkt_corr_20d_raw",
    required_columns={"date", "pct_chg"},
    requires_market_data=True,
    requires_sector_data=True,
    lookback_days=40,
    calculator=_calc_corr_raw,
    description="行业-市场 20日相关性（raw，无门控）"
))

# ✅ 新增：gated
sector_mkt_corr_20d_gated = register_factor(FactorDefinition(
    name="sector_mkt_corr_20d_gated",
    required_columns={"date", "pct_chg"},
    requires_market_data=True,
    requires_sector_data=True,
    lookback_days=40,
    calculator=_calc_corr_gated,
    description="行业-市场 20日相关性（弱市场信号门控版）"
))

intra_sector_breadth = register_factor(FactorDefinition(
    name="intra_sector_breadth",
    required_columns={"date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_calc_breadth,
    description="行业内上涨广度（按日横截面MAD截尾）"
))

stock_sector_pos_align = register_factor(FactorDefinition(
    name="stock_sector_pos_align",
    required_columns={"stock_code", "date", "high", "low", "close"},
    requires_sector_data=True,
    lookback_days=120,
    calculator=_calc_align,
    description="个股位置 - 行业均值位置（按日横截面MAD截尾，保结构锋利）"
))

__all__ = [
    "mkt_sector_dispersion",
    "sector_mkt_corr_20d_raw",
    "sector_mkt_corr_20d_gated",
    "intra_sector_breadth",
    "stock_sector_pos_align",
]
