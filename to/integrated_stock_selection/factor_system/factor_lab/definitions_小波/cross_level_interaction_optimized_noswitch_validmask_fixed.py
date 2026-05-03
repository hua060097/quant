"""
跨层级交互因子 (Cross-Level Interaction Factors) - valid_cs_mask 适配版（输出 float32）

在原 cross_level_interaction_optimized_noswitch.py 基础上：
- 增加 valid_cs_mask 支持：当日横截面 robust z 的 median/MAD 仅在 mask=True 的样本上统计，
  mask=False（ST/新股期等）不参与统计，避免污染分布。
- 输出统一为 float32（内部预计算保持 float64 稳定）。

注意：
- 本文件输出已经是“按日截面 robust z + clip”的终态特征，pipeline 不应再二次 zscore。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_EPS = 1e-12
_CLIP_Z = 5.0
_OUT_DTYPE = "float32"


# ==============================================================================
# 1) 工具函数
# ==============================================================================
def _to_ts(date_series: pd.Series) -> pd.Series:
    """将 date 列稳健转换为 pandas Timestamp（支持 'YYYY-MM-DD' / 'YYYYMMDD' / int）"""
    if pd.api.types.is_datetime64_any_dtype(date_series):
        return pd.to_datetime(date_series, errors="coerce").dt.normalize()
    ds = date_series.astype(str).str.replace("-", "", regex=False)
    return pd.to_datetime(ds, format="%Y%m%d", errors="coerce").dt.normalize()


def _clean_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(1.0) > 0.5


def _robust_z_clip_cs(
    values: pd.Series,
    group_key: pd.Series,
    *,
    valid_cs_mask: pd.Series | None = None,
    clip_z: float = _CLIP_Z,
    eps: float = _EPS,
) -> pd.Series:
    """
    按 group_key（通常是按日）做 robust z: (x - median) / (MAD*1.4826)，并对 z 做 clip。

    若提供 valid_cs_mask：
    - 仅 mask=True 的样本参与 median/MAD 统计
    - mask=False 不参与统计（等价于统计时为 NaN）
    - 最终对 mask=False 的样本输出 0（中性），避免其影响训练/排序
    """
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")

    if valid_cs_mask is not None:
        m = _clean_bool(valid_cs_mask)
        x_stat = x.where(m, np.nan)
    else:
        m = None
        x_stat = x

    g = x_stat.groupby(group_key, sort=False)
    med = g.transform("median")
    mad = g.transform(lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))))
    denom = 1.4826 * mad + eps

    z = (x - med) / denom
    z = z.clip(-clip_z, clip_z)

    if m is not None:
        z = z.where(m, np.nan).fillna(0.0)
    else:
        z = z.fillna(0.0)

    return z.astype(_OUT_DTYPE)


def _pct_rank_scaled(values: pd.Series, group_keys) -> pd.Series:
    """
    按 group_keys 做 pct-rank，并线性映射到 [-1, 1]

    group_keys 支持：
    - 单个 pd.Series
    - list[pd.Series]（推荐）
    注意：不要传入 ["_ts","sector_code"] 这种字符串列表给 Series.groupby，否则会触发 KeyError。
    """
    s = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")

    if isinstance(group_keys, pd.Series):
        keys = group_keys
    elif isinstance(group_keys, (list, tuple)) and len(group_keys) > 0 and all(isinstance(k, pd.Series) for k in group_keys):
        keys = list(group_keys)
    else:
        raise TypeError("group_keys must be a pandas Series or a list/tuple of pandas Series")

    r = s.groupby(keys, sort=False).rank(pct=True)
    return (r * 2.0 - 1.0).astype("float64")


# ==============================================================================
# 2) 核心交互计算类
# ==============================================================================
class _CrossInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.ctx = ctx
        self.original_index = data.index

        self.data = data.copy()
        self.data["_ts"] = _to_ts(self.data["date"])
        self.data["stock_code"] = self.data["stock_code"].astype(str)

        # valid_cs_mask（缺失则全部 True）
        if "valid_cs_mask" in self.data.columns:
            self.data["valid_cs_mask"] = _clean_bool(self.data["valid_cs_mask"])
        else:
            self.data["valid_cs_mask"] = True

        # 行业映射
        sector_map = ctx.extra.get("stock_to_sw_sector", {})
        self.data["sector_code"] = self.data["stock_code"].map(sector_map).fillna("Unknown").astype(str)

        # 市场数据（必须）
        market = ctx.require_market().copy()
        market["_ts"] = _to_ts(market["date"])
        market = market.sort_values("_ts").reset_index(drop=True)
        self.market_df = market

        self._precompute()

    def _precompute(self) -> None:
        # === A) 市场维度 ===
        mkt = self.market_df.set_index("_ts")

        self.mkt_ret_today = pd.to_numeric(mkt["daily_change_pct"], errors="coerce").astype("float64")

        self.mkt_vol_20d = self.mkt_ret_today.rolling(20, min_periods=10).std()

        m_mean = self.mkt_ret_today.rolling(60, min_periods=30).mean()
        m_std = self.mkt_ret_today.rolling(60, min_periods=30).std()
        self.mkt_trend_60d = (m_mean / (m_std + 1e-8)).astype("float64")

        # === B) 行业维度 ===
        g = self.data.groupby(["_ts", "sector_code"], sort=False)["pct_chg"]
        sec_sum = g.transform("sum").astype("float64")
        sec_cnt = g.transform("count").astype("float64")
        x = pd.to_numeric(self.data["pct_chg"], errors="coerce").astype("float64")

        denom = (sec_cnt - 1.0)
        loo_mean = (sec_sum - x) / denom.replace(0.0, np.nan)
        sec_mean = sec_sum / sec_cnt.replace(0.0, np.nan)
        self.data["sector_ret_loo"] = loo_mean.fillna(sec_mean).astype("float64")

        # === C) 个股维度 ===
        self.data["stk_vol_20d"] = (
            self.data.groupby("stock_code", sort=False)["pct_chg"]
            .transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(20, min_periods=10).std())
            .astype("float64")
        )

        tmp = self.data[["_ts", "stock_code", "pct_chg"]].copy()
        tmp = tmp.merge(
            self.market_df[["_ts", "daily_change_pct"]].rename(columns={"daily_change_pct": "m_ret"}),
            on="_ts",
            how="left",
        )
        tmp["pct_chg"] = pd.to_numeric(tmp["pct_chg"], errors="coerce").astype("float64")
        tmp["m_ret"] = pd.to_numeric(tmp["m_ret"], errors="coerce").astype("float64")

        def _rolling_beta(df: pd.DataFrame) -> pd.Series:
            r = df["pct_chg"]
            m = df["m_ret"]
            cov = r.rolling(60, min_periods=30).cov(m)
            var = m.rolling(60, min_periods=30).var()
            return cov / (var + 1e-8)

        beta = tmp.groupby("stock_code", sort=False).apply(_rolling_beta).reset_index(level=0, drop=True)
        self.data["beta_60d"] = beta.reindex(self.data.index).astype("float64")

        self.data["intra_rank"] = _pct_rank_scaled(self.data["pct_chg"], [self.data["_ts"], self.data["sector_code"]])

    def _finalize(self, raw: pd.Series) -> pd.Series:
        out = _robust_z_clip_cs(raw, self.data["_ts"], valid_cs_mask=self.data["valid_cs_mask"], clip_z=_CLIP_Z)
        return out.reindex(self.original_index).astype(_OUT_DTYPE)

    def compute_factor(self, factor_name: str) -> pd.Series:
        if factor_name == "market_vols_adaptive_iv":
            m_gate = self.data["_ts"].map(self.mkt_vol_20d)
            alpha = -self.data["stk_vol_20d"]
            raw = (pd.to_numeric(m_gate, errors="coerce") * pd.to_numeric(alpha, errors="coerce")).astype("float64")
            return self._finalize(raw)

        if factor_name == "market_trend_adaptive_beta":
            m_gate = self.data["_ts"].map(self.mkt_trend_60d)
            alpha = self.data["beta_60d"]
            raw = (pd.to_numeric(m_gate, errors="coerce") * pd.to_numeric(alpha, errors="coerce")).astype("float64")
            return self._finalize(raw)

        if factor_name == "sector_excess_intra_rank":
            s_ret = self.data["sector_ret_loo"]
            m_ret = self.data["_ts"].map(self.mkt_ret_today)
            s_excess = (pd.to_numeric(s_ret, errors="coerce") - pd.to_numeric(m_ret, errors="coerce")).astype("float64")
            alpha = self.data["intra_rank"].astype("float64")
            raw = (s_excess * alpha).astype("float64")
            return self._finalize(raw)

        return pd.Series(0.0, index=self.original_index, dtype=_OUT_DTYPE)


# ==============================================================================
# 3) 注册与外部接口
# ==============================================================================
def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _CrossInteractionHelper:
    key = "_cross_level_interaction_helper_validmask_v1"
    helper = ctx.extra.get(key)
    if helper is None:
        helper = _CrossInteractionHelper(data, ctx)
        ctx.extra[key] = helper
    return helper


def _calc_mkt_iv(d, c): return _get_helper(d, c).compute_factor("market_vols_adaptive_iv")
def _calc_mkt_beta(d, c): return _get_helper(d, c).compute_factor("market_trend_adaptive_beta")
def _calc_sec_excess(d, c): return _get_helper(d, c).compute_factor("sector_excess_intra_rank")


market_vols_adaptive_iv = register_factor(FactorDefinition(
    name="market_vols_adaptive_iv",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=80,
    calculator=_calc_mkt_iv,
    description="环境自适应波动（CS robust z，valid_cs_mask 过滤统计；float32）：市场高波动期偏好低波动个股"
))

market_trend_adaptive_beta = register_factor(FactorDefinition(
    name="market_trend_adaptive_beta",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=120,
    calculator=_calc_mkt_beta,
    description="趋势自适应Beta（CS robust z，valid_cs_mask 过滤统计；float32）：市场趋势强时偏好高Beta暴露"
))

sector_excess_intra_rank = register_factor(FactorDefinition(
    name="sector_excess_intra_rank",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    requires_sector_data=True,
    lookback_days=10,
    calculator=_calc_sec_excess,
    description="行业超额×行业内领先（CS robust z，valid_cs_mask 过滤统计；float32）"
))

__all__ = [
    "market_vols_adaptive_iv",
    "market_trend_adaptive_beta",
    "sector_excess_intra_rank",
]
