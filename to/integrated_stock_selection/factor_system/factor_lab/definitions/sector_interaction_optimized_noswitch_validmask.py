"""
板块-个股交互因子 - valid_cs_mask 适配 + float32

基于 sector_interaction_optimized_noswitch.py：
- 行业/市场聚合统计仅使用 valid_cs_mask=True 样本（避免 ST/新股污染板块 gate）
- 个股侧预计算与最终输出 CS robust z 的 median/MAD 仅使用 mask=True
- mask=False 行输出 0（中性）
- 输出 dtype：float32（内部 float64）

包含因子名保持不变（9个）：
- sector_vols_divergence_alpha
- sector_turnover_surge_pv_alpha
- sector_dispersion_leader_alpha
- sector_trend_gated_momentum_intra_z
- sector_rotation_breakout_intra_z
- sector_leader_capture_rank
- sector_strong_leader_alpha
- sector_liquidity_gated_turnover_z
- sector_weak_reversal_alpha
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


def _num64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


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


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / (b + eps)


class _SectorInteractionHelper:
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

        # 行业映射
        sector_map = ctx.extra.get("stock_to_sw_sector", {})
        df["sector_code"] = df["stock_code"].map(sector_map).fillna("Unknown") if sector_map else "All"
        df["sector_code"] = df["sector_code"].astype(str)

        # 数值化
        for col in ["pct_chg", "high", "close", "volume", "circ_mv"]:
            if col in df.columns:
                df[col] = _num64(df[col])

        df.sort_values(["stock_code", "_ts"], inplace=True)
        self.data = df
        self._precompute()

    def _precompute(self) -> None:
        df = self.data
        eps = 1e-9

        # 仅 mask=True 参与板块聚合
        base = df[df["valid_cs_mask"]].copy()

        # A) 板块收益宽表
        s_ret_wide = (
            base.groupby(["_ts", "sector_code"])["pct_chg"]
            .mean()
            .unstack("sector_code")
            .sort_index()
        )

        # 市场收益（mask=True 样本的均值）
        mkt_ret = base.groupby("_ts")["pct_chg"].mean().reindex(s_ret_wide.index)

        # A1 行业动量加速度 rank
        s_mom_5 = s_ret_wide.rolling(5, min_periods=3).mean()
        s_mom_20 = s_ret_wide.rolling(20, min_periods=10).mean()
        s_accel = s_mom_5 - s_mom_20
        self.s_accel_rank = s_accel.rank(axis=1, pct=True)

        # A2 行业排名变化
        s_rank = s_ret_wide.rank(axis=1, pct=True)
        self.s_rank_change = s_rank - s_rank.shift(5)

        # A3 行业相对强弱 robust z（跨板块）
        s_excess = s_ret_wide.sub(mkt_ret, axis=0).rolling(5, min_periods=3).mean()
        med = s_excess.median(axis=1)
        mad = (s_excess.sub(med, axis=0)).abs().median(axis=1)
        denom = 1.4826 * mad + 1e-12
        self.s_rel_strength_z = s_excess.sub(med, axis=0).div(denom, axis=0).clip(-_CLIP_Z, _CLIP_Z)

        # A4 HHI change rank（成交额份额平方）
        close = base["close"] if "close" in base.columns else pd.Series(1.0, index=base.index, dtype="float64")
        vol = base["volume"] if "volume" in base.columns else pd.Series(np.nan, index=base.index, dtype="float64")
        base["_amount"] = (vol * close).astype("float64")
        sector_total_amt = base.groupby(["_ts", "sector_code"])["_amount"].transform("sum")
        share_sq = (_safe_div(base["_amount"], sector_total_amt, eps=eps) ** 2).fillna(0.0)
        hhi_wide = (
            base.assign(_share_sq=share_sq)
            .groupby(["_ts", "sector_code"])["_share_sq"]
            .sum()
            .unstack("sector_code")
            .sort_index()
        )
        self.s_hhi_change_rank = hhi_wide.diff(5).rank(axis=1, pct=True)

        # A5 行业换手变化 rank
        if {"volume", "circ_mv"}.issubset(set(base.columns)):
            turnover = _safe_div(base["volume"], base["circ_mv"], eps=eps)
            s_to = (
                base.assign(_to=turnover)
                .groupby(["_ts", "sector_code"])["_to"]
                .mean()
                .unstack("sector_code")
                .sort_index()
            )
            v5 = s_to.rolling(5, min_periods=3).mean()
            v20 = s_to.rolling(20, min_periods=10).mean()
            self.s_turnover_change_rank = ((v5 - v20) / (v20 + eps)).rank(axis=1, pct=True)
        else:
            self.s_turnover_change_rank = s_ret_wide * np.nan

        # A6 行业波动
        self.s_vol = s_ret_wide.rolling(20, min_periods=10).std()

        # A7 行业分歧：行业内个股收益 std（mask=True）
        self.s_dispersion = (
            base.groupby(["_ts", "sector_code"])["pct_chg"]
            .std()
            .unstack("sector_code")
            .sort_index()
        )

        # B) 个股侧预计算（全样本，但 CS 标准化仅用 mask=True）
        mom_5 = df.groupby("stock_code", sort=False)["pct_chg"].transform(lambda x: x.rolling(5, min_periods=3).sum())
        df["stock_mom_5d_z"] = _cs_robust_z(mom_5, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).astype("float64")

        mom_20 = df.groupby("stock_code", sort=False)["pct_chg"].transform(lambda x: x.rolling(20, min_periods=15).sum())
        df["intra_rank"] = df.assign(_m=mom_20).groupby("_ts")["_m"].rank(pct=True) * 2 - 1
        df["intra_rank"] = df["intra_rank"].where(df["valid_cs_mask"], 0.0).astype("float64")

        if {"high", "close"}.issubset(set(df.columns)):
            high20 = df.groupby("stock_code", sort=False)["high"].transform(lambda x: x.rolling(20, min_periods=10).max())
            breakout = (df["close"] / (high20 + eps)) - 1.0
            df["breakout_z"] = _cs_robust_z(breakout, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).astype("float64")
        else:
            df["breakout_z"] = 0.0

        if {"volume", "circ_mv"}.issubset(set(df.columns)):
            turnover = _safe_div(df["volume"], df["circ_mv"], eps=eps)
            to_ma20_vol = df.groupby("stock_code", sort=False)["volume"].transform(lambda x: x.rolling(20, min_periods=10).mean())
            to_ma20 = _safe_div(to_ma20_vol, df["circ_mv"], eps=eps)
            abnormal = (turnover / (to_ma20 + eps)) - 1.0
            df["abnormal_to_z"] = _cs_robust_z(abnormal, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).astype("float64")
        else:
            df["abnormal_to_z"] = 0.0

        if "volume" in df.columns:
            lv = np.log1p(df["volume"].fillna(0.0)).astype("float64")
            tmp = pd.DataFrame({"pct_chg": df["pct_chg"].astype("float64"), "_lv": lv, "stock_code": df["stock_code"]}, index=df.index)
            pv_corr = tmp.groupby("stock_code", sort=False).apply(lambda g: g["pct_chg"].rolling(20, min_periods=15).corr(g["_lv"]))
            pv_corr = pv_corr.reset_index(level=0, drop=True).reindex(df.index)
            df["pv_corr_z"] = _cs_robust_z(pv_corr, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).astype("float64")
        else:
            df["pv_corr_z"] = 0.0

        self.data = df

    def get_factor(self, factor_name: str) -> pd.Series:
        df = self.data

        if factor_name == "sector_vols_divergence_alpha":
            gate_wide = -self.s_vol
            alpha = df["breakout_z"]
        elif factor_name == "sector_turnover_surge_pv_alpha":
            gate_wide = self.s_turnover_change_rank
            alpha = df["pv_corr_z"]
        elif factor_name == "sector_dispersion_leader_alpha":
            gate_wide = self.s_dispersion
            alpha = df["intra_rank"]
        elif factor_name == "sector_trend_gated_momentum_intra_z":
            gate_wide = self.s_accel_rank
            alpha = df["stock_mom_5d_z"]
        elif factor_name == "sector_rotation_breakout_intra_z":
            gate_wide = self.s_rank_change
            alpha = df["breakout_z"]
        elif factor_name == "sector_leader_capture_rank":
            gate_wide = self.s_hhi_change_rank
            alpha = df["intra_rank"]
        elif factor_name == "sector_strong_leader_alpha":
            gate_wide = self.s_rel_strength_z
            alpha = df["intra_rank"]
        elif factor_name == "sector_liquidity_gated_turnover_z":
            gate_wide = self.s_turnover_change_rank
            alpha = df["abnormal_to_z"]
        elif factor_name == "sector_weak_reversal_alpha":
            gate_wide = -self.s_rel_strength_z
            alpha = -df["stock_mom_5d_z"]
        else:
            raise ValueError(f"Unknown factor: {factor_name}")

        gate_long = gate_wide.stack(dropna=False).rename("gate")
        gate = df.join(gate_long, on=["_ts", "sector_code"], how="left")["gate"].astype("float64")

        raw = (gate * pd.to_numeric(alpha, errors="coerce").astype("float64")).replace([np.inf, -np.inf], np.nan)

        out = _cs_robust_z(raw, df["_ts"], valid_cs_mask=df["valid_cs_mask"])
        return out.reindex(self.original_index).astype(_OUT_DTYPE)


def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _SectorInteractionHelper:
    key = "_sector_interaction_helper_validmask_v1"
    h = ctx.extra.get(key)
    if h is None:
        h = _SectorInteractionHelper(data, ctx)
        ctx.extra[key] = h
    return h


def _c(name: str):
    def calc(d: pd.DataFrame, c: FactorContext) -> pd.Series:
        return _get_helper(d, c).get_factor(name)
    return calc


sector_vols_divergence_alpha = register_factor(FactorDefinition(
    name="sector_vols_divergence_alpha",
    required_columns={"stock_code", "date", "pct_chg", "high", "close"},
    requires_sector_data=True,
    lookback_days=40,
    calculator=_c("sector_vols_divergence_alpha"),
))

sector_turnover_surge_pv_alpha = register_factor(FactorDefinition(
    name="sector_turnover_surge_pv_alpha",
    required_columns={"stock_code", "date", "pct_chg", "volume", "close"},
    requires_sector_data=True,
    lookback_days=40,
    calculator=_c("sector_turnover_surge_pv_alpha"),
))

sector_dispersion_leader_alpha = register_factor(FactorDefinition(
    name="sector_dispersion_leader_alpha",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=30,
    calculator=_c("sector_dispersion_leader_alpha"),
))

sector_trend_gated_momentum_intra_z = register_factor(FactorDefinition(
    name="sector_trend_gated_momentum_intra_z",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_c("sector_trend_gated_momentum_intra_z"),
))

sector_rotation_breakout_intra_z = register_factor(FactorDefinition(
    name="sector_rotation_breakout_intra_z",
    required_columns={"stock_code", "date", "pct_chg", "high", "close"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_c("sector_rotation_breakout_intra_z"),
))

sector_leader_capture_rank = register_factor(FactorDefinition(
    name="sector_leader_capture_rank",
    required_columns={"stock_code", "date", "pct_chg", "volume", "close"},
    requires_sector_data=True,
    lookback_days=30,
    calculator=_c("sector_leader_capture_rank"),
))

sector_strong_leader_alpha = register_factor(FactorDefinition(
    name="sector_strong_leader_alpha",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_c("sector_strong_leader_alpha"),
))

sector_liquidity_gated_turnover_z = register_factor(FactorDefinition(
    name="sector_liquidity_gated_turnover_z",
    required_columns={"stock_code", "date", "pct_chg", "volume", "circ_mv"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_c("sector_liquidity_gated_turnover_z"),
))

sector_weak_reversal_alpha = register_factor(FactorDefinition(
    name="sector_weak_reversal_alpha",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_c("sector_weak_reversal_alpha"),
))

__all__ = [
    "sector_vols_divergence_alpha",
    "sector_turnover_surge_pv_alpha",
    "sector_dispersion_leader_alpha",
    "sector_trend_gated_momentum_intra_z",
    "sector_rotation_breakout_intra_z",
    "sector_leader_capture_rank",
    "sector_strong_leader_alpha",
    "sector_liquidity_gated_turnover_z",
    "sector_weak_reversal_alpha",
]
