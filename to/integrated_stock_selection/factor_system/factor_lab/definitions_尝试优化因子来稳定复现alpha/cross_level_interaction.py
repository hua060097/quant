from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


# ==============================================================================
# 工具函数
# ==============================================================================

def _rolling_zscore(series: pd.Series, window: int = 120) -> pd.Series:
    """
    时间序列 rolling z-score（仅使用过去窗口；用于 market gate 的尺度控制）
    """
    s = series.astype("float64").sort_index()
    mu = s.rolling(window, min_periods=30).mean()
    sd = s.rolling(window, min_periods=30).std()
    return (s - mu) / (sd + 1e-8)


def _cs_mad_winsorize_by_date(values: pd.Series, date_series: pd.Series, n_mad: float = 3.5) -> pd.Series:
    """
    按日横截面 MAD winsor（不做 expanding、不看未来）
    - 这是“结果层”保护：防止乘法交互在极端日炸穿，但尽量保留尾部锋利
    """
    if values.empty:
        return values.astype("float64")

    df = pd.DataFrame({"v": values.values, "d": date_series.values}, index=values.index)

    med = df.groupby("d")["v"].transform("median")
    mad = (df["v"] - med).abs().groupby(df["d"]).transform("median")

    width = n_mad * mad * 1.4826
    out = df["v"].astype("float64")

    mask = mad > 1e-12
    out = out.where(~mask, out.clip(med - width, med + width))
    out = out.replace([np.inf, -np.inf], np.nan)

    return out.astype("float64")


# ==============================================================================
# 核心计算器
# ==============================================================================

class _CrossInteractionHelper:
    """
    生成三类跨层因子：
      1) market_vols_adaptive_iv:  市场波动 gate * (-个股IV proxy)
      2) market_trend_adaptive_beta: 市场趋势 gate * beta_60d
      3) sector_excess_intra_rank: (行业收益-市场收益) * 个股行业内当日rank

    每类输出三路：
      - *_raw   : 最锋利（不 winsor、不 tanh）
      - *       : stable（tanh gate + 当日MAD winsor）
      - *_delta : raw - stable（触发边界信号）
    """

    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.ctx = ctx

        # 保留原 index，最后输出要对齐
        self.original_index = data.index

        # 拷贝 + 排序（rolling 依赖顺序一致）
        self.data = data.copy()
        self.data.sort_values(["stock_code", "date"], inplace=True)

        # SW 二级行业映射：ctx.extra['stock_to_sw_sector']（你已提供）
        sector_map = ctx.extra.get("stock_to_sw_sector", {})
        self.data["sector_code"] = self.data["stock_code"].map(sector_map).fillna("Unknown")

        # 市场数据
        self.market = ctx.require_market().copy()
        self.market = self.market.sort_values("date").drop_duplicates("date").set_index("date")

        self._precompute()

    def _precompute(self):
        # === Market ===
        mkt_ret = self.market["daily_change_pct"].astype("float64")
        self.mkt_ret = mkt_ret

        # 原始市场状态量（只用过去）
        self.mkt_vol_raw = mkt_ret.rolling(20, min_periods=10).std()
        self.mkt_trend_raw = mkt_ret.rolling(20, min_periods=10).sum()

        # gate 两路：
        # raw_gate: rolling z（不压缩，保锋利）
        self.mkt_vol_gate_raw = _rolling_zscore(self.mkt_vol_raw).clip(-6.0, 6.0)
        self.mkt_trend_gate_raw = _rolling_zscore(self.mkt_trend_raw).clip(-6.0, 6.0)

        # stable_gate: tanh 压缩（更稳）
        self.mkt_vol_gate = np.tanh(self.mkt_vol_gate_raw)
        self.mkt_trend_gate = np.tanh(self.mkt_trend_gate_raw)

        # === Stock IV proxy ===
        # 用“个股 20 日收益波动率”做 IV proxy，再做全市场当日截面 z
        stk_vol_20d = (
            self.data.groupby("stock_code", sort=False)["pct_chg"]
            .rolling(20, min_periods=10)
            .std()
            .reset_index(level=0, drop=True)
        ).replace([np.inf, -np.inf], np.nan)

        self.data["stk_iv_z"] = (
            stk_vol_20d.groupby(self.data["date"])
            .transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        ).clip(-8.0, 8.0)

        # === Beta 60d（沿用你原思路：cov/var；尽量不改变旧 alpha 的“手感”）===
        df = self.data.merge(
            self.market[["daily_change_pct"]],
            left_on="date",
            right_index=True,
            how="left",
        )

        cov = (
            df.groupby("stock_code", sort=False)[["pct_chg", "daily_change_pct"]]
            .rolling(60, min_periods=30)
            .cov()
            .unstack()
            .iloc[:, 1]
            .reset_index(level=0, drop=True)
        )

        var = (
            df.groupby("stock_code", sort=False)["daily_change_pct"]
            .transform(lambda x: x.rolling(60, min_periods=30).var())
        )

        beta = (cov / (var + 1e-8)).replace([np.inf, -np.inf], np.nan).clip(-8.0, 8.0)
        self.data["beta_60d"] = beta

        # === Sector daily return & intra rank ===
        # 行业当日收益（简单均值；与你其它模块一致）
        self.sec_ret_daily = self.data.groupby(["date", "sector_code"], sort=False)["pct_chg"].mean()

        # 行业内当日 rank（[-1, 1]）
        self.data["intra_rank"] = (
            self.data.groupby(["date", "sector_code"], sort=False)["pct_chg"]
            .rank(pct=True)
            .mul(2.0)
            .sub(1.0)
        )

    # --------------------------------------------------------------------------
    # 三类 raw / stable / delta 计算
    # --------------------------------------------------------------------------

    def _market_vols_adaptive_iv_raw(self) -> pd.Series:
        # raw：raw gate * (-IV z)，不 winsor
        gate = self.data["date"].map(self.mkt_vol_gate_raw).astype("float64")
        alpha = (-self.data["stk_iv_z"]).astype("float64")
        out = (gate * alpha).replace([np.inf, -np.inf], np.nan)
        return out.reindex(self.original_index)

    def _market_vols_adaptive_iv(self) -> pd.Series:
        # stable：tanh gate * (-IV z) + 当日MAD winsor
        gate = self.data["date"].map(self.mkt_vol_gate).astype("float64")
        alpha = (-self.data["stk_iv_z"]).astype("float64")
        out = (gate * alpha).replace([np.inf, -np.inf], np.nan)
        out = _cs_mad_winsorize_by_date(out, self.data["date"], n_mad=3.5)
        return out.reindex(self.original_index)

    def _market_vols_adaptive_iv_delta(self) -> pd.Series:
        raw = self._market_vols_adaptive_iv_raw()
        stable = self._market_vols_adaptive_iv()
        out = (raw - stable).replace([np.inf, -np.inf], np.nan)
        # delta 也做轻 winsor，防止差分爆炸（仍不泄露未来）
        out = _cs_mad_winsorize_by_date(out, self.data.loc[out.index, "date"], n_mad=4.0)
        return out.astype("float64")

    def _market_trend_adaptive_beta_raw(self) -> pd.Series:
        # raw：raw trend gate * beta
        gate = self.data["date"].map(self.mkt_trend_gate_raw).astype("float64")
        alpha = self.data["beta_60d"].astype("float64")
        out = (gate * alpha).replace([np.inf, -np.inf], np.nan)
        return out.reindex(self.original_index)

    def _market_trend_adaptive_beta(self) -> pd.Series:
        # stable：tanh gate * beta + 当日MAD winsor
        gate = self.data["date"].map(self.mkt_trend_gate).astype("float64")
        alpha = self.data["beta_60d"].astype("float64")
        out = (gate * alpha).replace([np.inf, -np.inf], np.nan)
        out = _cs_mad_winsorize_by_date(out, self.data["date"], n_mad=3.5)
        return out.reindex(self.original_index)

    def _market_trend_adaptive_beta_delta(self) -> pd.Series:
        raw = self._market_trend_adaptive_beta_raw()
        stable = self._market_trend_adaptive_beta()
        out = (raw - stable).replace([np.inf, -np.inf], np.nan)
        out = _cs_mad_winsorize_by_date(out, self.data.loc[out.index, "date"], n_mad=4.0)
        return out.astype("float64")

    def _sector_excess_intra_rank_raw(self) -> pd.Series:
        # raw：行业超额收益 * 行业内 rank（不 winsor）
        s_ret = self.data.set_index(["date", "sector_code"]).index.map(self.sec_ret_daily)
        m_ret = self.data["date"].map(self.mkt_ret)
        s_excess = (s_ret - m_ret).astype("float64")

        alpha = self.data["intra_rank"].astype("float64")
        out = (s_excess * alpha).replace([np.inf, -np.inf], np.nan)
        out = pd.Series(out, index=self.data.index).astype("float64")
        return out.reindex(self.original_index)

    def _sector_excess_intra_rank(self) -> pd.Series:
        # stable：当日MAD winsor（这里不需要 gate，但对极端日乘法仍会炸）
        out = self._sector_excess_intra_rank_raw().copy()
        # 注意：winsor 需要 date_series 对齐 out.index
        date_s = self.data.loc[out.index, "date"]
        out = _cs_mad_winsorize_by_date(out, date_s, n_mad=3.5)
        return out.astype("float64")

    def _sector_excess_intra_rank_delta(self) -> pd.Series:
        raw = self._sector_excess_intra_rank_raw()
        stable = self._sector_excess_intra_rank()
        out = (raw - stable).replace([np.inf, -np.inf], np.nan)
        out = _cs_mad_winsorize_by_date(out, self.data.loc[out.index, "date"], n_mad=4.0)
        return out.astype("float64")

    # --------------------------------------------------------------------------
    # 外部接口
    # --------------------------------------------------------------------------

    def compute(self, name: str) -> pd.Series:
        if name == "market_vols_adaptive_iv_raw":
            return self._market_vols_adaptive_iv_raw()
        if name == "market_vols_adaptive_iv":
            return self._market_vols_adaptive_iv()
        if name == "market_vols_adaptive_iv_delta":
            return self._market_vols_adaptive_iv_delta()

        if name == "market_trend_adaptive_beta_raw":
            return self._market_trend_adaptive_beta_raw()
        if name == "market_trend_adaptive_beta":
            return self._market_trend_adaptive_beta()
        if name == "market_trend_adaptive_beta_delta":
            return self._market_trend_adaptive_beta_delta()

        if name == "sector_excess_intra_rank_raw":
            return self._sector_excess_intra_rank_raw()
        if name == "sector_excess_intra_rank":
            return self._sector_excess_intra_rank()
        if name == "sector_excess_intra_rank_delta":
            return self._sector_excess_intra_rank_delta()

        return pd.Series(0.0, index=self.original_index, dtype="float64")


# ==============================================================================
# 缓存 helper（避免重复计算）
# ==============================================================================

def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _CrossInteractionHelper:
    key = "_cross_interaction_helper_v3_raw_stable_delta"
    if key not in ctx.extra:
        ctx.extra[key] = _CrossInteractionHelper(data, ctx)
    return ctx.extra[key]


# ==============================================================================
# 注册：三路输出
# ==============================================================================

# --- market_vols_adaptive_iv ---
market_vols_adaptive_iv_raw = register_factor(FactorDefinition(
    name="market_vols_adaptive_iv_raw",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=160,  # 20d vol + 120d z + buffer
    calculator=lambda d, c: _get_helper(d, c).compute("market_vols_adaptive_iv_raw"),
    description="RAW：market vol gate(z) * (-stock IV z)，不截尾，最锋利"
))

market_vols_adaptive_iv = register_factor(FactorDefinition(
    name="market_vols_adaptive_iv",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=160,
    calculator=lambda d, c: _get_helper(d, c).compute("market_vols_adaptive_iv"),
    description="STABLE：tanh(vol gate) * (-IV z) + 按日MAD winsor（不泄露）"
))

market_vols_adaptive_iv_delta = register_factor(FactorDefinition(
    name="market_vols_adaptive_iv_delta",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=160,
    calculator=lambda d, c: _get_helper(d, c).compute("market_vols_adaptive_iv_delta"),
    description="DELTA：raw - stable（regime/触发边界信号）"
))

# --- market_trend_adaptive_beta ---
market_trend_adaptive_beta_raw = register_factor(FactorDefinition(
    name="market_trend_adaptive_beta_raw",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=200,  # beta 60 + gate 140
    calculator=lambda d, c: _get_helper(d, c).compute("market_trend_adaptive_beta_raw"),
    description="RAW：market trend gate(z) * beta_60d，不截尾，最锋利"
))

market_trend_adaptive_beta = register_factor(FactorDefinition(
    name="market_trend_adaptive_beta",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=200,
    calculator=lambda d, c: _get_helper(d, c).compute("market_trend_adaptive_beta"),
    description="STABLE：tanh(trend gate) * beta_60d + 按日MAD winsor（不泄露）"
))

market_trend_adaptive_beta_delta = register_factor(FactorDefinition(
    name="market_trend_adaptive_beta_delta",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=200,
    calculator=lambda d, c: _get_helper(d, c).compute("market_trend_adaptive_beta_delta"),
    description="DELTA：raw - stable（regime/触发边界信号）"
))

# --- sector_excess_intra_rank ---
sector_excess_intra_rank_raw = register_factor(FactorDefinition(
    name="sector_excess_intra_rank_raw",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    requires_sector_data=True,
    lookback_days=30,
    calculator=lambda d, c: _get_helper(d, c).compute("sector_excess_intra_rank_raw"),
    description="RAW：(sector_ret - mkt_ret) * intra_rank，不截尾，最锋利"
))

sector_excess_intra_rank = register_factor(FactorDefinition(
    name="sector_excess_intra_rank",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    requires_sector_data=True,
    lookback_days=30,
    calculator=lambda d, c: _get_helper(d, c).compute("sector_excess_intra_rank"),
    description="STABLE：RAW + 按日MAD winsor（不泄露）"
))

sector_excess_intra_rank_delta = register_factor(FactorDefinition(
    name="sector_excess_intra_rank_delta",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    requires_sector_data=True,
    lookback_days=30,
    calculator=lambda d, c: _get_helper(d, c).compute("sector_excess_intra_rank_delta"),
    description="DELTA：raw - stable（触发边界信号）"
))

__all__ = [
    "market_vols_adaptive_iv_raw",
    "market_vols_adaptive_iv",
    "market_vols_adaptive_iv_delta",
    "market_trend_adaptive_beta_raw",
    "market_trend_adaptive_beta",
    "market_trend_adaptive_beta_delta",
    "sector_excess_intra_rank_raw",
    "sector_excess_intra_rank",
    "sector_excess_intra_rank_delta",
]
