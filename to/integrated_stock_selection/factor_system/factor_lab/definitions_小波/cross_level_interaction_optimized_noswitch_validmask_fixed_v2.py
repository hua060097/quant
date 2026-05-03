"""
跨层级交互因子 (Cross-Level Interaction Factors) - valid_cs_mask 适配版（输出 float32）

审计&修复要点（v2）：
1) valid_cs_mask 的语义：用于“横截面统计/训练/打分”的有效样本隔离（ST/新股/成交量0等）。
   因此：凡是“当日横截面聚合/排名”的统计口径，都应仅使用 mask=True 的样本，以免无效样本污染有效样本的特征值。
2) 严格因果 rolling：时间序列 rolling 不前置使用 valid_cs_mask（避免时间序列被挖空）。
3) Helper 缓存安全：ctx.extra 的缓存 key 引入 data 的时间窗口签名，避免不同 lookback 窗口复用导致错算。
4) lookback_days：因为 helper 内部会预计算 60d/20d rolling（beta/vol/trend），三个因子的 lookback 统一给到足够长度，
   否则引擎若按各自 lookback 切片，会导致 sector 因子用到“短窗 helper”从而缺失/错算。
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


def _clean_num64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


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
    x = _clean_num64(values)

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


def _pct_rank_scaled(values: pd.Series, group_keys, *, valid_cs_mask: pd.Series | None = None) -> pd.Series:
    """
    按 group_keys 做 pct-rank，并线性映射到 [-1, 1]。

    关键：若给 valid_cs_mask，则当日/同组的 rank 只在 mask=True 样本上计算，
    mask=False 的输出直接为 NaN（后续会在 _robust_z_clip_cs 中置 0）。
    """
    s = _clean_num64(values)

    if isinstance(group_keys, pd.Series):
        keys = group_keys
    elif isinstance(group_keys, (list, tuple)) and len(group_keys) > 0 and all(isinstance(k, pd.Series) for k in group_keys):
        keys = list(group_keys)
    else:
        raise TypeError("group_keys must be a pandas Series or a list/tuple of pandas Series")

    if valid_cs_mask is not None:
        m = _clean_bool(valid_cs_mask)
        s_stat = s.where(m, np.nan)
        r = s_stat.groupby(keys, sort=False).rank(pct=True)
        r = r.where(m, np.nan)
    else:
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

        # 个股收益清洗（仅做数值清洗；不在这里用 mask 去挖空时间序列）
        self.data["pct_chg"] = _clean_num64(self.data.get("pct_chg", np.nan))

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
        self.mkt_ret_today = _clean_num64(mkt["daily_change_pct"])

        self.mkt_vol_20d = self.mkt_ret_today.rolling(20, min_periods=10).std()

        m_mean = self.mkt_ret_today.rolling(60, min_periods=30).mean()
        m_std = self.mkt_ret_today.rolling(60, min_periods=30).std()
        self.mkt_trend_60d = (m_mean / (m_std + 1e-8)).astype("float64")

        # === B) 行业维度（横截面聚合必须遵守 valid_cs_mask 口径）===
        # 用 mask=True 的样本计算行业收益；避免 ST/新股/成交量0 等污染行业均值与行业内排序。
        m = self.data["valid_cs_mask"].astype(bool)
        pct = self.data["pct_chg"].where(m, np.nan)

        g = pct.groupby([self.data["_ts"], self.data["sector_code"]], sort=False)
        sec_sum = g.transform("sum").astype("float64")
        sec_cnt = g.transform("count").astype("float64")

        x = self.data["pct_chg"].astype("float64")

        denom = (sec_cnt - 1.0)
        loo_mean = (sec_sum - x.where(m, np.nan)) / denom.replace(0.0, np.nan)
        sec_mean = sec_sum / sec_cnt.replace(0.0, np.nan)
        # 对有效样本：优先用 LOO；不足则回退行业均值；再回退 0（例如行业当日有效样本为 0 的极端情况）
        self.data["sector_ret_loo"] = loo_mean.fillna(sec_mean).fillna(0.0).astype("float64")

        # === C) 个股维度（rolling 严格因果，不前置 mask）===
        # rolling 波动
        self.data.sort_values(["stock_code", "_ts"], inplace=True)
        self.data["stk_vol_20d"] = (
            self.data.groupby("stock_code", sort=False)["pct_chg"]
            .transform(lambda s: s.rolling(20, min_periods=10).std())
            .astype("float64")
        )

        # rolling beta（60d）
        tmp = self.data[["_ts", "stock_code", "pct_chg"]].copy()
        tmp = tmp.merge(
            self.market_df[["_ts", "daily_change_pct"]].rename(columns={"daily_change_pct": "m_ret"}),
            on="_ts",
            how="left",
        )
        tmp["m_ret"] = _clean_num64(tmp["m_ret"])

        def _rolling_beta(df: pd.DataFrame) -> pd.Series:
            r = df["pct_chg"]
            mret = df["m_ret"]
            cov = r.rolling(60, min_periods=30).cov(mret)
            var = mret.rolling(60, min_periods=30).var()
            return cov / (var + 1e-8)

        beta = tmp.groupby("stock_code", sort=False).apply(_rolling_beta).reset_index(level=0, drop=True)
        self.data["beta_60d"] = beta.reindex(self.data.index).astype("float64")

        # 行业内当日收益排名（横截面排名必须遵守 mask 口径）
        self.data["intra_rank"] = _pct_rank_scaled(
            self.data["pct_chg"],
            [self.data["_ts"], self.data["sector_code"]],
            valid_cs_mask=self.data["valid_cs_mask"],
        )

    def _finalize(self, raw: pd.Series) -> pd.Series:
        out = _robust_z_clip_cs(raw, self.data["_ts"], valid_cs_mask=self.data["valid_cs_mask"], clip_z=_CLIP_Z)
        return out.reindex(self.original_index).astype(_OUT_DTYPE)

    def compute_factor(self, factor_name: str) -> pd.Series:
        if factor_name == "market_vols_adaptive_iv":
            m_gate = self.data["_ts"].map(self.mkt_vol_20d)
            alpha = -self.data["stk_vol_20d"]
            raw = (m_gate.astype("float64") * alpha.astype("float64")).astype("float64")
            return self._finalize(raw)

        if factor_name == "market_trend_adaptive_beta":
            m_gate = self.data["_ts"].map(self.mkt_trend_60d)
            alpha = self.data["beta_60d"]
            raw = (m_gate.astype("float64") * alpha.astype("float64")).astype("float64")
            return self._finalize(raw)

        if factor_name == "sector_excess_intra_rank":
            s_ret = self.data["sector_ret_loo"]
            m_ret = self.data["_ts"].map(self.mkt_ret_today)
            s_excess = (s_ret.astype("float64") - m_ret.astype("float64")).astype("float64")
            alpha = self.data["intra_rank"].astype("float64")
            raw = (s_excess * alpha).astype("float64")
            return self._finalize(raw)

        return pd.Series(0.0, index=self.original_index, dtype=_OUT_DTYPE)


# ==============================================================================
# 3) 注册与外部接口
# ==============================================================================
def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _CrossInteractionHelper:
    # 缓存 key 引入数据窗口签名，避免不同 lookback 切片复用
    ts = _to_ts(data["date"])
    sig = (str(ts.min()), str(ts.max()), int(len(data)))
    key = f"_cross_level_interaction_helper_validmask_v2_{sig[0]}_{sig[1]}_{sig[2]}"
    helper = ctx.extra.get(key)
    if helper is None:
        helper = _CrossInteractionHelper(data, ctx)
        ctx.extra[key] = helper
    return helper


def _calc_mkt_iv(d, c): 
    return _get_helper(d, c).compute_factor("market_vols_adaptive_iv")


def _calc_mkt_beta(d, c): 
    return _get_helper(d, c).compute_factor("market_trend_adaptive_beta")


def _calc_sec_excess(d, c): 
    return _get_helper(d, c).compute_factor("sector_excess_intra_rank")


# helper 内部需要 60d/20d rolling，因此三个因子统一给足 lookback
_LOOKBACK = 140

market_vols_adaptive_iv = register_factor(FactorDefinition(
    name="market_vols_adaptive_iv",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=_LOOKBACK,
    calculator=_calc_mkt_iv,
    description="环境自适应波动（CS robust z，valid_cs_mask 口径；float32）：市场高波动期偏好低波动个股"
))

market_trend_adaptive_beta = register_factor(FactorDefinition(
    name="market_trend_adaptive_beta",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=_LOOKBACK,
    calculator=_calc_mkt_beta,
    description="趋势自适应Beta（CS robust z，valid_cs_mask 口径；float32）：市场趋势强时偏好高Beta暴露"
))

sector_excess_intra_rank = register_factor(FactorDefinition(
    name="sector_excess_intra_rank",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    requires_sector_data=True,
    lookback_days=_LOOKBACK,
    calculator=_calc_sec_excess,
    description="行业超额×行业内领先（CS robust z，valid_cs_mask 口径；float32）"
))

__all__ = [
    "market_vols_adaptive_iv",
    "market_trend_adaptive_beta",
    "sector_excess_intra_rank",
]
