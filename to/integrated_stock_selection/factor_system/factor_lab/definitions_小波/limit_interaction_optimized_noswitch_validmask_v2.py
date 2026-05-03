"""
涨跌停与市场情绪交互因子 - valid_cs_mask 适配版（输出 float32）

要点：
- valid_cs_mask：用于板块/市场聚合统计与 robust z 的统计口径过滤
  * 聚合统计、median/MAD 仅在 mask=True 样本上计算
  * mask=False 样本最终输出置 0（中性），避免 ST/新股异常波动污染分布
- 输出 float32（内部计算 float64）

本文件输出已经是终态特征，pipeline 不应再二次 zscore。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_EPS = 1e-8
_OUT_DTYPE = "float32"


def _to_ts(date_series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(date_series):
        return pd.to_datetime(date_series, errors="coerce").dt.normalize()
    ds = date_series.astype(str).str.replace("-", "", regex=False)
    return pd.to_datetime(ds, format="%Y%m%d", errors="coerce").dt.normalize()


def _clean_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(1.0) > 0.5


def _robust_z_by_groups(
    df: pd.DataFrame,
    values: pd.Series,
    group_cols: list[str],
    *,
    valid_cs_mask: pd.Series | None = None,
    z_clip: float = 6.0,
    eps: float = 1e-12,
) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")

    if valid_cs_mask is not None:
        m = _clean_bool(valid_cs_mask)
        s_stat = s.where(m, np.nan)
    else:
        m = None
        s_stat = s

    tmp = df[group_cols].copy()
    tmp["_v"] = s_stat.values
    g = tmp.groupby(group_cols, sort=False)["_v"]
    med = g.transform("median")
    mad = g.transform(lambda x: (x - x.median()).abs().median())

    z = (s - med) / (1.4826 * mad + eps)
    z = z.replace([np.inf, -np.inf], np.nan)

    if z_clip and z_clip > 0:
        z = z.clip(-float(z_clip), float(z_clip))

    if m is not None:
        z = z.where(m, np.nan)

    return z.fillna(0.0).astype("float64")


class _LimitInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.data = data.copy()
        self.original_index = data.index
        self.ctx = ctx

        self.data["_ts"] = _to_ts(self.data["date"])
        # signature for safe caching under walk-forward windows
        self._sig = (len(self.data), self.data["_ts"].min(), self.data["_ts"].max())
        self.data["stock_code"] = self.data["stock_code"].astype(str)
        self.data.sort_values(["stock_code", "_ts"], inplace=True)

        if "valid_cs_mask" in self.data.columns:
            self.data["valid_cs_mask"] = _clean_bool(self.data["valid_cs_mask"])
        else:
            self.data["valid_cs_mask"] = True

        sector_map = ctx.extra.get("stock_to_sw_sector", {})
        self.data["sector_code"] = self.data["stock_code"].map(sector_map).fillna("Unknown").astype(str) if sector_map else "All"

        for col in ["is_limit_up", "is_limit_down", "is_zhaban"]:
            if col not in self.data.columns:
                self.data[col] = 0
            self.data[col] = self.data[col].fillna(0).astype(int)

        self._precompute()

    def _precompute(self) -> None:
        # 只用 valid_cs_mask=True 的样本做板块/市场统计
        base = self.data[self.data["valid_cs_mask"]].copy()

        g_sec = base.groupby(["_ts", "sector_code"], sort=False)
        sector_stats = g_sec[["is_limit_up", "is_limit_down", "is_zhaban"]].sum()
        sector_counts = g_sec["stock_code"].count()

        s_limit_up_ratio = sector_stats["is_limit_up"] / (sector_counts + _EPS)
        self.s_heat_rank = (
            s_limit_up_ratio.unstack("sector_code")
            .rolling(3, min_periods=1).mean()
            .rank(axis=1, pct=True)
            .stack()
        )

        total_attempts = sector_stats["is_limit_up"] + sector_stats["is_zhaban"]
        s_zhaban_ratio = sector_stats["is_zhaban"] / (total_attempts + _EPS)
        self.s_divergence_rank = (
            s_zhaban_ratio.unstack("sector_code")
            .rolling(3, min_periods=1).mean()
            .rank(axis=1, pct=True)
            .stack()
        )

        self.data["is_prev_limit_up"] = (
            self.data.groupby("stock_code", sort=False)["is_limit_up"].shift(1).fillna(0).astype(int)
        )

        tmp = self.data[self.data["valid_cs_mask"]].copy()
        tmp["_r"] = pd.to_numeric(tmp.get("pct_chg"), errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
        limit_up_premium = tmp.loc[tmp["is_prev_limit_up"] == 1].groupby("_ts", sort=False)["_r"].mean()
        self.market_mood = self.data["_ts"].map(limit_up_premium)

        # stock-level proxies（行业内robust_z）
        self.data["active_z"] = _robust_z_by_groups(
            self.data, self.data.get("pct_chg"), ["_ts", "sector_code"], valid_cs_mask=self.data["valid_cs_mask"], z_clip=6.0
        )

        # Prefer exchange-reported turnover_rate if available; otherwise fall back to volume/circ_mv proxy.
        if "turnover_rate" in self.data.columns:
            to = pd.to_numeric(self.data["turnover_rate"], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
        elif "volume" in self.data.columns and "circ_mv" in self.data.columns:
            to = (
                pd.to_numeric(self.data["volume"], errors="coerce")
                / (pd.to_numeric(self.data["circ_mv"], errors="coerce") + _EPS)
            ).replace([np.inf, -np.inf], np.nan).astype("float64")
        else:
            to = None

        if isinstance(to, pd.Series):
            self.data["turnover_z"] = _robust_z_by_groups(
                self.data, to, ["_ts", "sector_code"], valid_cs_mask=self.data["valid_cs_mask"], z_clip=6.0
            )
        else:
            self.data["turnover_z"] = self.data["active_z"]

        prev_close = self.data.groupby("stock_code", sort=False)["close"].shift(1)
        self.data["intraday_strength"] = (
            (pd.to_numeric(self.data["close"], errors="coerce") - pd.to_numeric(self.data["open"], errors="coerce"))
            / (prev_close + _EPS)
        )

        high = pd.to_numeric(self.data.get("high"), errors="coerce")
        low = pd.to_numeric(self.data.get("low"), errors="coerce")
        close = pd.to_numeric(self.data.get("close"), errors="coerce")
        range_len = (high - low)
        clv = (close - low) / (range_len + _EPS)
        is_flat = range_len.abs() < 1e-5
        clv = np.where(is_flat & (self.data["is_limit_up"] == 1), 1.0, clv)
        clv = np.where(is_flat & (self.data["is_limit_down"] == 1), 0.0, clv)
        self.data["clv"] = pd.to_numeric(clv, errors="coerce")
        self.data["resilience_z"] = _robust_z_by_groups(
            self.data, self.data["clv"], ["_ts", "sector_code"], valid_cs_mask=self.data["valid_cs_mask"], z_clip=6.0
        )

    def get_factor(self, name: str) -> pd.Series:
        m = self.data["valid_cs_mask"]

        if name == "sector_limit_overflow_alpha":
            gate = self.data.set_index(["_ts", "sector_code"]).index.map(self.s_heat_rank)
            alpha = self.data["turnover_z"]
            tradable = (1 - self.data["is_limit_up"]) * (1 - self.data["is_limit_down"])
            raw = pd.to_numeric(gate, errors="coerce") * pd.to_numeric(alpha, errors="coerce") * tradable
            out = _robust_z_by_groups(self.data, raw, ["_ts", "sector_code"], valid_cs_mask=m, z_clip=6.0)

        elif name == "sector_broken_resilience_alpha":
            gate = self.data.set_index(["_ts", "sector_code"]).index.map(self.s_divergence_rank)
            raw = pd.to_numeric(gate, errors="coerce") * pd.to_numeric(self.data["resilience_z"], errors="coerce")
            out = _robust_z_by_groups(self.data, raw, ["_ts", "sector_code"], valid_cs_mask=m, z_clip=6.0)

        elif name == "limit_sustain_alpha":
            raw = self.data["is_prev_limit_up"].astype("float64") * pd.to_numeric(self.data["intraday_strength"], errors="coerce")
            out = _robust_z_by_groups(self.data, raw, ["_ts"], valid_cs_mask=m, z_clip=6.0)

        elif name == "market_emotion_resonance_alpha":
            env = pd.to_numeric(self.market_mood, errors="coerce").fillna(0.0)
            raw = env * pd.to_numeric(self.data["active_z"], errors="coerce")
            out = _robust_z_by_groups(self.data, raw, ["_ts"], valid_cs_mask=m, z_clip=6.0)

        else:
            raise ValueError(f"Unknown factor: {name}")

        return out.reindex(self.original_index).fillna(0.0).astype(_OUT_DTYPE)


def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _LimitInteractionHelper:
    """Safe helper cache for walk-forward: reuse only when signature matches current data window."""
    key = "_limit_interaction_helper_validmask_v2"
    _ts = _to_ts(data["date"])
    sig = (len(data), _ts.min(), _ts.max())

    h = None
    if hasattr(ctx, "extra"):
        h = ctx.extra.get(key)

    if h is None or getattr(h, "_sig", None) != sig:
        h = _LimitInteractionHelper(data, ctx)
        h._sig = sig
        if hasattr(ctx, "extra"):
            ctx.extra[key] = h
    return h


def _c_overflow(d, c): return _get_helper(d, c).get_factor("sector_limit_overflow_alpha")
def _c_resilience(d, c): return _get_helper(d, c).get_factor("sector_broken_resilience_alpha")
def _c_sustain(d, c): return _get_helper(d, c).get_factor("limit_sustain_alpha")
def _c_resonance(d, c): return _get_helper(d, c).get_factor("market_emotion_resonance_alpha")


sector_limit_overflow_alpha = register_factor(FactorDefinition(
    name="sector_limit_overflow_alpha",
    required_columns={"stock_code", "date", "is_limit_up", "is_limit_down", "volume", "circ_mv"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_c_overflow,
    description="板块高潮溢出因子(补涨) - RobustCS（valid_cs_mask；float32）"
))

sector_broken_resilience_alpha = register_factor(FactorDefinition(
    name="sector_broken_resilience_alpha",
    required_columns={"stock_code", "date", "is_zhaban", "high", "low", "close"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_c_resilience,
    description="板块分歧抵抗因子(抗分歧) - RobustCS（valid_cs_mask；float32）"
))

limit_sustain_alpha = register_factor(FactorDefinition(
    name="limit_sustain_alpha",
    required_columns={"stock_code", "date", "is_limit_up", "open", "close"},
    requires_sector_data=False,
    lookback_days=10,
    calculator=_c_sustain,
    description="涨停次日接力/核按钮识别因子 - RobustCS（valid_cs_mask；float32）"
))

market_emotion_resonance_alpha = register_factor(FactorDefinition(
    name="market_emotion_resonance_alpha",
    required_columns={"stock_code", "date", "is_limit_up", "pct_chg"},
    requires_sector_data=False,
    lookback_days=10,
    calculator=_c_resonance,
    description="市场情绪共振因子(昨日涨停溢价x个股活跃) - RobustCS（valid_cs_mask；float32）"
))

__all__ = [
    "sector_limit_overflow_alpha",
    "sector_broken_resilience_alpha",
    "limit_sustain_alpha",
    "market_emotion_resonance_alpha",
]
