"""
板块-个股交互因子 (Sector-Stock Interaction Factors) - FrozenClip版
文件名: sector_interaction.py

包含因子 (共9个):
1. sector_vols_divergence_alpha: 低波动板块中的突破个股
2. sector_turnover_surge_pv_alpha: 换手激增板块中的价量配合个股
3. sector_dispersion_leader_alpha: 高分歧板块中的领涨个股
4. sector_trend_gated_momentum_intra_z: 动量加速板块中的强势个股
5. sector_rotation_breakout_intra_z: 排名飙升板块中的突破个股
6. sector_leader_capture_rank: 筹码集中板块中的领涨个股
7. sector_strong_leader_alpha: 强势板块中的领涨个股
8. sector_liquidity_gated_turnover_z: 资金流入板块中的异常放量个股
9. sector_weak_reversal_alpha: 弱势板块中的超跌反转个股

本次改动（关键）：
- 原 _adaptive_clip() 使用全样本 median/MAD -> 存在未来信息泄露（你“全量一起算因子”的 pipeline 会触发）
- 改为 FrozenMADClipper：
  只用训练集日期范围拟合阈值（train_start~train_end），并冻结用于全量 clip
- 训练集日期范围（用户指定）：
  train_start = 2023-06-01
  train_end   = 2025-03-01
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..frozen_store import load_frozen_clipper_params, save_frozen_clipper_params
_LOGGER = logging.getLogger(__name__)


# ==============================================================================
# 0. Frozen（训练期冻结）MAD Clipper
# ==============================================================================
class FrozenMADClipper:
    """
    Frozen MAD Clipper:
    - fit(): 仅用训练期样本拟合 median 与 MAD（全局阈值）
    - transform(): 对全样本使用冻结阈值 clip
    """

    def __init__(self, n_mad: float = 3.0, eps: float = 1e-12):
        self.n_mad = float(n_mad)
        self.eps = float(eps)
        self._fitted = False

        self.median_ = 0.0
        self.mad_ = 0.0
        self.lower_ = -np.inf
        self.upper_ = np.inf

    def fit(self, series: pd.Series) -> "FrozenMADClipper":
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            self.median_, self.mad_ = 0.0, 0.0
            self.lower_, self.upper_ = -np.inf, np.inf
            self._fitted = True
            return self

        med = float(s.median())
        mad = float((s - med).abs().median())
        thr = self.n_mad * mad * 1.4826

        # mad 极小 -> 不裁剪（避免门槛过小导致输出跳/常数化）
        if mad < self.eps or thr < self.eps:
            self.median_, self.mad_ = med, mad
            self.lower_, self.upper_ = -np.inf, np.inf
        else:
            self.median_, self.mad_ = med, mad
            self.lower_, self.upper_ = med - thr, med + thr

        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("FrozenMADClipper must be fitted before transform().")
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return s.clip(self.lower_, self.upper_)


def _to_ts(date_series: pd.Series) -> pd.Series:
    """稳健把 date 转为 Timestamp（支持 'YYYY-MM-DD' / 'YYYYMMDD' / int）"""
    if date_series.dtype == "datetime64[ns]":
        return date_series
    ds = date_series.astype(str)
    ds = ds.str.replace("-", "", regex=False)
    return pd.to_datetime(ds, format="%Y%m%d", errors="coerce")


def _resolve_frozen_window(extra: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = extra.get("frozen_clip_train_start")
    end = extra.get("frozen_clip_train_end")
    default_start = pd.Timestamp("2023-06-01")
    default_end = pd.Timestamp("2025-03-01")
    try:
        start_ts = pd.Timestamp(start) if start is not None else default_start
        end_ts = pd.Timestamp(end) if end is not None else default_end
    except Exception:
        return default_start, default_end
    return start_ts, end_ts


# ==============================================================================
# 1. 全局稳健 Z-Score（保留你原逻辑）
# ==============================================================================
def _global_robust_zscore(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """
    全局按日稳健 Z-Score (MAD) —— 截面标准化，不引入未来信息
    """
    df = pd.DataFrame({"val": series.values, "date": date_series.values}, index=series.index)

    medians = df.groupby("date")["val"].transform("median")
    abs_diff = (df["val"] - medians).abs()
    temp_diff = pd.DataFrame({"diff": abs_diff.values, "date": date_series.values}, index=series.index)
    mads = temp_diff.groupby("date")["diff"].transform("median")

    robust_std = mads * 1.4826
    z = (df["val"] - medians) / (robust_std + 1e-9)

    mask_zero_mad = (mads < 1e-6)
    if mask_zero_mad.any():
        means = df.groupby("date")["val"].transform("mean")
        stds = df.groupby("date")["val"].transform("std")
        z_fallback = (df["val"] - means) / (stds + 1e-9)
        z = z.where(~mask_zero_mad, z_fallback)

    return z.clip(-6.0, 6.0).astype("float64")


# ==============================================================================
# 2. 核心计算辅助类
# ==============================================================================
class _SectorInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        # 必须先拷贝并排序，确保 rolling/shift 正确
        self.data = data.copy()
        self.data.sort_values(["stock_code", "date"], inplace=True)

        self.ctx = ctx
        self.original_index = data.index

        # 时间戳用于训练期 mask
        self.data["_ts"] = _to_ts(self.data["date"])
        self.train_start_, self.train_end_ = _resolve_frozen_window(ctx.extra or {})

        # 行业映射（SW 二级行业直接当成 sector_code 类别即可）
        sector_map = ctx.extra.get("stock_to_sw_sector", {})
        if not sector_map:
            self.data["sector_code"] = "All"
        else:
            self.data["sector_code"] = self.data["stock_code"].map(sector_map).fillna("Unknown")

        # 冻结 clipper cache：每个 factor_name 一套
        self._clipper_cache: dict[str, FrozenMADClipper] = {}
        self._module_name = "sector_interaction"

        # 执行预计算
        self._precompute_factors()

    def _train_mask(self) -> pd.Series:
        ts = self.data["_ts"]
        return (ts >= self.train_start_) & (ts <= self.train_end_)

    def _frozen_clip(self, factor_name: str, series: pd.Series, n_mad: float = 3.0) -> pd.Series:
        clipper = self._clipper_cache.get(factor_name)
        if clipper is None:
            clipper = FrozenMADClipper(n_mad=n_mad)
            cached = load_frozen_clipper_params(
                module_name=self._module_name,
                factor_name=factor_name,
                train_start=self.train_start_,
                train_end=self.train_end_,
                n_mad=n_mad,
                module_file=__file__,
            )
            if cached:
                clipper.median_ = float(cached["median"])
                clipper.mad_ = float(cached["mad"])
                clipper.lower_ = float(cached["lower"])
                clipper.upper_ = float(cached["upper"])
                clipper._fitted = True
                _LOGGER.info(
                    "[FrozenClip] %s/%s 使用本地冻结参数 train=%s~%s lower=%.6f upper=%.6f",
                    self._module_name,
                    factor_name,
                    self.train_start_,
                    self.train_end_,
                    clipper.lower_,
                    clipper.upper_,
                )
            else:
                train_series = series[self._train_mask()]
                _LOGGER.info(
                    "[FrozenClip] %s/%s 现场拟合 train=%s~%s samples=%s series_non_na=%s",
                    self._module_name,
                    factor_name,
                    self.train_start_,
                    self.train_end_,
                    int(train_series.dropna().shape[0]),
                    int(series.dropna().shape[0]),
                )
                clipper.fit(train_series)
                save_frozen_clipper_params(
                    module_name=self._module_name,
                    factor_name=factor_name,
                    train_start=self.train_start_,
                    train_end=self.train_end_,
                    n_mad=n_mad,
                    module_file=__file__,
                    median=clipper.median_,
                    mad=clipper.mad_,
                    lower=clipper.lower_,
                    upper=clipper.upper_,
                    fit_sample_count=int(train_series.dropna().shape[0]),
                )
            self._clipper_cache[factor_name] = clipper
        return clipper.transform(series)

    def _precompute_factors(self):
        """预计算行业 Gate 与个股 Alpha"""

        # --- A. 行业层面指标 (Sector Gates) ---
        s_ret_wide = (
            self.data.groupby(["date", "sector_code"])["pct_chg"]
            .mean()
            .unstack("sector_code")
            .sort_index()
        )

        # 1) 行业动量加速度 Gate
        s_mom_5 = s_ret_wide.rolling(5, min_periods=3).mean()
        s_mom_20 = s_ret_wide.rolling(20, min_periods=10).mean()
        self.s_accel_rank = (s_mom_5 - s_mom_20).rank(axis=1, pct=True)

        # 2) 行业排名变化 Gate（保留 NaN）
        s_rank_wide = s_ret_wide.rank(axis=1, pct=True)
        self.s_rank_change = (s_rank_wide - s_rank_wide.shift(5))

        # 3) 行业相对强弱 Gate（板块间 Z）
        mkt_ret = self.data.groupby("date")["pct_chg"].mean()
        s_rel_strength = s_ret_wide.sub(mkt_ret, axis=0).rolling(5, min_periods=3).mean()
        self.s_rel_strength_z = s_rel_strength.apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=1
        )

        # 4) 行业集中度与成交 Gate
        self._compute_hhi_metrics()
        self._compute_sector_turnover_metrics()

        # 5) 行业波动与离散 Gate
        self.s_vol = s_ret_wide.rolling(20, min_periods=10).std()
        self.s_dispersion = self.data.groupby(["date", "sector_code"])["pct_chg"].std().unstack("sector_code")

        # --- B. 个股层面指标 (Stock Alphas) ---

        # 1) 动量 Z (全局截面稳健)
        stock_mom_5 = self.data.groupby("stock_code")["pct_chg"].transform(
            lambda x: x.rolling(5, min_periods=3).sum()
        )
        self.data["stock_mom_5d_z"] = _global_robust_zscore(stock_mom_5, self.data["date"])

        # 2) 动量 rank（映射到 -1~1）
        stock_mom_20 = self.data.groupby("stock_code")["pct_chg"].transform(
            lambda x: x.rolling(20, min_periods=15).sum()
        )
        self.data["intra_rank"] = self.data.assign(m=stock_mom_20).groupby("date")["m"].rank(pct=True) * 2 - 1

        # 3) 突破强度 Z
        high_20d = self.data.groupby("stock_code")["high"].transform(
            lambda x: x.rolling(20, min_periods=10).max()
        )
        breakout = (self.data["close"] / (high_20d + 1e-9)) - 1
        self.data["breakout_z"] = _global_robust_zscore(breakout, self.data["date"])

        # 4) 异常换手 Z（你的字段：volume + circ_mv）
        if "volume" in self.data.columns and "circ_mv" in self.data.columns:
            turnover = self.data["volume"] / (self.data["circ_mv"] + 1e-9)
            turnover_ma20 = (
                self.data.groupby("stock_code")["volume"].transform(lambda x: x.rolling(20).mean())
                / (self.data["circ_mv"] + 1e-9)
            )
            abnormal_to = (turnover / (turnover_ma20 + 1e-9)) - 1
            self.data["abnormal_to_z"] = _global_robust_zscore(abnormal_to, self.data["date"])
        else:
            self.data["abnormal_to_z"] = pd.Series(np.nan, index=self.data.index)

        # 5) 价量相关性 Z（rolling corr）
        def _calc_corr(df: pd.DataFrame) -> pd.Series:
            # 确保 df['volume'] 存在，否则 corr 全 NaN
            if "volume" not in df.columns:
                return pd.Series(np.nan, index=df.index)
            return df["pct_chg"].rolling(20, min_periods=15).corr(df["volume"])

        pv_corr = self.data.groupby("stock_code", group_keys=False).apply(_calc_corr)
        if not pv_corr.index.equals(self.data.index):
            pv_corr = pv_corr.reindex(self.data.index)

        self.data["pv_corr_z"] = _global_robust_zscore(pv_corr, self.data["date"])

    def _compute_hhi_metrics(self):
        """集中度 HHI 计算"""
        close = self.data.get("close", pd.Series(1.0, index=self.data.index))
        self.data["amount"] = self.data["volume"] * close

        sector_total = self.data.groupby(["date", "sector_code"])["amount"].transform("sum")
        self.data["share_sq"] = (self.data["amount"] / (sector_total + 1e-9)) ** 2

        hhi_wide = self.data.groupby(["date", "sector_code"])["share_sq"].sum().unstack("sector_code")
        self.s_hhi_change_rank = (hhi_wide.diff(5)).rank(axis=1, pct=True)

    def _compute_sector_turnover_metrics(self):
        """行业成交变化（volume proxy）"""
        s_vol = self.data.groupby(["date", "sector_code"])["volume"].mean().unstack("sector_code")
        v_5 = s_vol.rolling(5, min_periods=3).mean()
        v_20 = s_vol.rolling(20, min_periods=10).mean()
        self.s_turnover_change_rank = ((v_5 - v_20) / (v_20 + 1e-9)).rank(axis=1, pct=True)

    def get_interaction_factor(self, factor_name: str) -> pd.Series:
        """映射 Gate 与 Alpha 的乘积逻辑 + Frozen clip"""
        raw = self._build_raw_interaction(factor_name)

        # 4) Frozen clip（训练期拟合阈值，后续全量一致）
        clipped = self._frozen_clip(factor_name, raw, n_mad=3.0)

        # 5) 索引回到原始输入
        return clipped.reindex(self.original_index).astype("float64")

    def _build_raw_interaction(self, factor_name: str, target_data: pd.DataFrame | None = None) -> pd.Series:
        rows = self.data if target_data is None else target_data.copy()

        if factor_name == "sector_vols_divergence_alpha":
            gate_wide = -self.s_vol
            alpha = rows["breakout_z"]
        elif factor_name == "sector_turnover_surge_pv_alpha":
            gate_wide = self.s_turnover_change_rank
            alpha = rows["pv_corr_z"]
        elif factor_name == "sector_dispersion_leader_alpha":
            gate_wide = self.s_dispersion
            alpha = rows["intra_rank"]
        elif factor_name == "sector_trend_gated_momentum_intra_z":
            gate_wide = self.s_accel_rank
            alpha = rows["stock_mom_5d_z"]
        elif factor_name == "sector_rotation_breakout_intra_z":
            gate_wide = self.s_rank_change
            alpha = rows["breakout_z"]
        elif factor_name == "sector_leader_capture_rank":
            gate_wide = self.s_hhi_change_rank
            alpha = rows["intra_rank"]
        elif factor_name == "sector_strong_leader_alpha":
            gate_wide = self.s_rel_strength_z
            alpha = rows["intra_rank"]
        elif factor_name == "sector_liquidity_gated_turnover_z":
            gate_wide = self.s_turnover_change_rank
            alpha = rows["abnormal_to_z"]
        elif factor_name == "sector_weak_reversal_alpha":
            gate_wide = -self.s_rel_strength_z
            alpha = -rows["stock_mom_5d_z"]
        else:
            raise ValueError(f"Unknown factor: {factor_name}")

        gate_long = gate_wide.stack().rename("gate")
        gate = rows.join(gate_long, on=["date", "sector_code"], how="left")["gate"]
        return (gate * alpha).astype("float64")

    def compute_target(self, factor_name: str, target_index: pd.Index) -> pd.Series:
        raw_target = self._build_raw_interaction(factor_name, self.data.loc[target_index])
        raw_full = self._build_raw_interaction(factor_name)
        _ = self._frozen_clip(factor_name, raw_full, n_mad=3.0)
        clipper = self._clipper_cache[factor_name]
        return clipper.transform(raw_target).reindex(target_index).astype("float64")


# ==============================================================================
# 3. 注册因子
# ==============================================================================
def _get_sector_interaction_helper(data: pd.DataFrame, ctx: FactorContext) -> _SectorInteractionHelper:
    helper_key = "_sector_interaction_helper_frozenclip_v1"
    sig = (len(data), data.index[0] if len(data) else None, data.index[-1] if len(data) else None)
    sig_key = f"{helper_key}__sig"
    if helper_key not in ctx.extra or ctx.extra.get(sig_key) != sig:
        ctx.extra[helper_key] = _SectorInteractionHelper(data, ctx)
        ctx.extra[sig_key] = sig
    return ctx.extra[helper_key]


def _c_vols_div(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor("sector_vols_divergence_alpha")
def _c_pv_surge(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor("sector_turnover_surge_pv_alpha")
def _c_disp_leader(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor("sector_dispersion_leader_alpha")
def _c_trend_gate(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor("sector_trend_gated_momentum_intra_z")
def _c_rot_break(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor("sector_rotation_breakout_intra_z")
def _c_hhi_cap(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor("sector_leader_capture_rank")
def _c_strong_alpha(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor("sector_strong_leader_alpha")
def _c_liq_gate(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor("sector_liquidity_gated_turnover_z")
def _c_weak_rev(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor("sector_weak_reversal_alpha")


def _target_factor(name):
    return lambda d, ti, c: _get_sector_interaction_helper(d, c).compute_target(name, ti)


sector_vols_divergence_alpha = register_factor(FactorDefinition(
    name="sector_vols_divergence_alpha",
    required_columns={"stock_code", "date", "pct_chg", "high", "close"},
    requires_sector_data=True,
    lookback_days=40,
    calculator=_c_vols_div,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_vols_divergence_alpha")
))

sector_turnover_surge_pv_alpha = register_factor(FactorDefinition(
    name="sector_turnover_surge_pv_alpha",
    required_columns={"stock_code", "date", "pct_chg", "volume"},
    requires_sector_data=True,
    lookback_days=40,
    calculator=_c_pv_surge,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_turnover_surge_pv_alpha")
))

sector_dispersion_leader_alpha = register_factor(FactorDefinition(
    name="sector_dispersion_leader_alpha",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=30,
    calculator=_c_disp_leader,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_dispersion_leader_alpha")
))

sector_trend_gated_momentum_intra_z = register_factor(FactorDefinition(
    name="sector_trend_gated_momentum_intra_z",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_c_trend_gate,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_trend_gated_momentum_intra_z")
))

sector_rotation_breakout_intra_z = register_factor(FactorDefinition(
    name="sector_rotation_breakout_intra_z",
    required_columns={"stock_code", "date", "pct_chg", "high", "close"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_c_rot_break,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_rotation_breakout_intra_z")
))

sector_leader_capture_rank = register_factor(FactorDefinition(
    name="sector_leader_capture_rank",
    required_columns={"stock_code", "date", "pct_chg", "volume"},
    requires_sector_data=True,
    lookback_days=30,
    calculator=_c_hhi_cap,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_leader_capture_rank")
))

sector_strong_leader_alpha = register_factor(FactorDefinition(
    name="sector_strong_leader_alpha",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_c_strong_alpha,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_strong_leader_alpha")
))

sector_liquidity_gated_turnover_z = register_factor(FactorDefinition(
    name="sector_liquidity_gated_turnover_z",
    required_columns={"stock_code", "date", "pct_chg", "volume", "circ_mv"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_c_liq_gate,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_liquidity_gated_turnover_z")
))

sector_weak_reversal_alpha = register_factor(FactorDefinition(
    name="sector_weak_reversal_alpha",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=25,
    calculator=_c_weak_rev,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_weak_reversal_alpha")
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
