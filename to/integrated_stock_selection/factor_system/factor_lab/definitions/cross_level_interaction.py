"""
跨层级交互因子 (Cross-Level Interaction Factors)

核心逻辑：
1) Stock * Market：引入市场波动环境与趋势强度，动态调整风险暴露。
2) Stock * Sector * Market：识别处于超额收益板块中的头部个股。

本版本改动（关键）：
- 将原先 _adaptive_clip（全样本 MAD 去极值）改为 Frozen MAD Clipper：
  只使用训练集日期范围拟合（train_start~train_end）的 median / MAD 阈值，
  并冻结用于全体日期（train/val/test）clip，避免未来信息泄露。

训练集日期范围（由用户指定）：
- train_start: 2023-06-01
- train_end:   2025-03-01

日期：2026-01-31
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
# 0. Frozen（训练期阈值冻结）MAD Clipper
# ==============================================================================
class FrozenMADClipper:
    """
    Frozen MAD Clipper:
    - fit(): 仅用训练期样本拟合 median 与 MAD（全局阈值）
    - transform(): 对全样本使用冻结阈值 clip

    说明：
    - 这是“最小侵入”的替代方案：尽量保持你旧 alpha 的“锋利度”和分布，
      同时彻底去除“用未来数据确定 clip 边界”的泄露。
    - 阈值是全局的（不是按日/按行业），以最大程度贴近你旧版 _adaptive_clip 的风格。
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
            # 极端情况下训练期全 NaN：退化为不裁剪
            self.median_, self.mad_ = 0.0, 0.0
            self.lower_, self.upper_ = -np.inf, np.inf
            self._fitted = True
            return self

        med = float(s.median())
        mad = float((s - med).abs().median())
        # MAD -> robust std
        thr = self.n_mad * mad * 1.4826

        # 稀疏保护：mad 极小就不裁剪
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

    def fit_transform(self, series: pd.Series) -> pd.Series:
        return self.fit(series).transform(series)


# ==============================================================================
# 1. 辅助工具函数
# ==============================================================================
def _get_market_data(ctx: FactorContext) -> pd.DataFrame:
    """获取对齐后的市场数据"""
    market = ctx.require_market()
    return market.sort_values("date").reset_index(drop=True)


def _to_ts(date_series: pd.Series) -> pd.Series:
    """
    将 date 列稳健转换为 pandas Timestamp（支持 'YYYY-MM-DD' / 'YYYYMMDD' / int）
    """
    if date_series.dtype == "datetime64[ns]":
        return date_series
    # 尽量处理 int(YYYYMMDD)、str(YYYY-MM-DD/YYYMMDD)
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
# 2. 核心交互计算类
# ==============================================================================
class _CrossInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.data = data.copy()
        self.data.sort_values(["stock_code", "date"], inplace=True)
        self.ctx = ctx
        self.original_index = data.index

        # 行业映射（SW二级行业等均可，只要 sector_code 一致）
        sector_map = ctx.extra.get("stock_to_sw_sector", {})
        self.data["sector_code"] = self.data["stock_code"].map(sector_map).fillna("Unknown")

        # 市场数据
        self.market_df = _get_market_data(ctx)

        # 统一时间戳列（用于训练期 mask）
        self.data["_ts"] = _to_ts(self.data["date"])
        self.market_df["_ts"] = _to_ts(self.market_df["date"])
        self.train_start_, self.train_end_ = _resolve_frozen_window(ctx.extra or {})

        # Frozen clippers（按因子名缓存一次，避免重复拟合）
        self._clipper_cache: dict[str, FrozenMADClipper] = {}
        self._module_name = "cross_level_interaction"

        # 预计算跨层级指标
        self._precompute()

    def _train_mask(self, ts: pd.Series) -> pd.Series:
        """训练期日期 mask（包含端点）"""
        return (ts >= self.train_start_) & (ts <= self.train_end_)

    def _intra_sector_zscore(self, series: pd.Series) -> pd.Series:
        """行业内按日 Z-Score 标准化算子（对齐你原实现）"""
        return self.data.assign(tmp=series).groupby(["date", "sector_code"])["tmp"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    def _precompute(self):
        """预计算市场背景、行业趋势与个股敏感度指标"""
        # --- A. 市场维度 ---
        mkt = self.market_df.set_index("date")
        self.mkt_vol = mkt["daily_change_pct"].rolling(20, min_periods=10).std()
        self.mkt_trend = mkt["daily_change_pct"].rolling(20, min_periods=10).sum()
        self.mkt_ret_today = mkt["daily_change_pct"]

        # --- B. 行业维度 ---
        self.s_ret_daily = self.data.groupby(["date", "sector_code"])["pct_chg"].mean()

        # --- C. 个股维度 ---
        # 1) 20日个股波动率（IV 代理） -> 行业内 z
        stk_vol = self.data.groupby("stock_code")["pct_chg"].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        self.data["stk_vol_intra_z"] = self._intra_sector_zscore(stk_vol)

        # 2) 60日 beta（相对大盘）
        df_with_mkt = self.data.merge(
            self.market_df[["date", "daily_change_pct"]], on="date", how="left"
        ).rename(columns={"daily_change_pct": "m_ret"})

        grouped = df_with_mkt.groupby("stock_code", sort=False)
        cov_60d = (
            grouped[["pct_chg", "m_ret"]]
            .rolling(60, min_periods=30)
            .cov()
            .unstack()
            .iloc[:, 1]
            .reset_index(level=0, drop=True)
        )
        var_m_60d = grouped["m_ret"].transform(lambda x: x.rolling(60, min_periods=30).var())
        self.data["beta_60d"] = cov_60d / (var_m_60d + 1e-8)

        # 3) 行业内当日排名（[-1, 1]）
        self.data["intra_rank"] = (
            self.data.groupby(["date", "sector_code"])["pct_chg"].rank(pct=True) * 2 - 1
        )

    # ------------------------------
    # Frozen clip 应用
    # ------------------------------
    def _frozen_clip(self, name: str, raw: pd.Series, ts: pd.Series, n_mad: float = 3.0) -> pd.Series:
        """
        对 raw 应用“训练期拟合、全期使用”的 Frozen MAD clip。
        - name: 因子名，用于 cache
        - ts: 时间戳序列（用于训练期 mask）
        """
        clipper = self._clipper_cache.get(name)
        if clipper is None:
            clipper = FrozenMADClipper(n_mad=n_mad)
            cached = load_frozen_clipper_params(
                module_name=self._module_name,
                factor_name=name,
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
                    name,
                    self.train_start_,
                    self.train_end_,
                    clipper.lower_,
                    clipper.upper_,
                )
            else:
                mask = self._train_mask(ts)
                train_series = raw[mask]
                _LOGGER.info(
                    "[FrozenClip] %s/%s 现场拟合 train=%s~%s samples=%s series_non_na=%s",
                    self._module_name,
                    name,
                    self.train_start_,
                    self.train_end_,
                    int(train_series.dropna().shape[0]),
                    int(raw.dropna().shape[0]),
                )
                clipper.fit(train_series)
                save_frozen_clipper_params(
                    module_name=self._module_name,
                    factor_name=name,
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
            self._clipper_cache[name] = clipper
        return clipper.transform(raw)

    def compute_factor(self, factor_name: str) -> pd.Series:
        """根据名称执行具体的交互逻辑（Frozen clip 版）"""
        raw = self._build_raw_factor(factor_name)
        out = self._frozen_clip(factor_name, raw, self.data["_ts"], n_mad=3.0)
        return out.reindex(self.original_index)

    def _build_raw_factor(self, factor_name: str, target_data: pd.DataFrame | None = None) -> pd.Series:
        rows = self.data if target_data is None else target_data.copy()

        if factor_name == "market_vols_adaptive_iv":
            m_gate = rows["date"].map(self.mkt_vol)
            alpha = -rows["stk_vol_intra_z"]
            raw = (m_gate * alpha).astype("float64")
        elif factor_name == "market_trend_adaptive_beta":
            m_gate = rows["date"].map(self.mkt_trend)
            alpha = rows["beta_60d"]
            raw = (m_gate * alpha).astype("float64")
        elif factor_name == "sector_excess_intra_rank":
            s_ret = rows.set_index(["date", "sector_code"]).index.map(self.s_ret_daily)
            m_ret = rows["date"].map(self.mkt_ret_today)
            s_excess = (s_ret - m_ret).astype("float64")
            alpha = rows["intra_rank"].astype("float64")
            raw = (s_excess * alpha).astype("float64")
        else:
            raw = pd.Series(0.0, index=rows.index, dtype="float64")

        return raw

    def compute_target(self, factor_name: str, target_index: pd.Index) -> pd.Series:
        raw_full = self._build_raw_factor(factor_name)
        _ = self._frozen_clip(factor_name, raw_full, self.data["_ts"], n_mad=3.0)
        clipper = self._clipper_cache[factor_name]
        raw_target = self._build_raw_factor(factor_name, self.data.loc[target_index])
        target_ts = self.data.loc[target_index, "_ts"]
        # 保持接口一致，clip 只需要阈值，这里不再重拟合
        _ = target_ts
        return clipper.transform(raw_target).reindex(target_index)


# ==============================================================================
# 3. 注册与外部接口
# ==============================================================================
def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _CrossInteractionHelper:
    key = "_cross_interaction_helper"
    sig = (len(data), data.index[0] if len(data) else None, data.index[-1] if len(data) else None)
    sig_key = f"{key}__sig"
    if key not in ctx.extra or ctx.extra.get(sig_key) != sig:
        ctx.extra[key] = _CrossInteractionHelper(data, ctx)
        ctx.extra[sig_key] = sig
    return ctx.extra[key]


def _calc_mkt_iv(d, c): return _get_helper(d, c).compute_factor("market_vols_adaptive_iv")
def _calc_mkt_beta(d, c): return _get_helper(d, c).compute_factor("market_trend_adaptive_beta")
def _calc_sec_excess(d, c): return _get_helper(d, c).compute_factor("sector_excess_intra_rank")
def _target_factor(name): return lambda d, ti, c: _get_helper(d, c).compute_target(name, ti)


market_vols_adaptive_iv = register_factor(FactorDefinition(
    name="market_vols_adaptive_iv",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=40,
    calculator=_calc_mkt_iv,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("market_vols_adaptive_iv"),
    description="环境自适应波动因子(FrozenClip): 市场高波动时倾向低波股"
))

market_trend_adaptive_beta = register_factor(FactorDefinition(
    name="market_trend_adaptive_beta",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    lookback_days=80,
    calculator=_calc_mkt_beta,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("market_trend_adaptive_beta"),
    description="趋势自适应贝塔(FrozenClip): 市场走强时倾向高Beta股"
))

sector_excess_intra_rank = register_factor(FactorDefinition(
    name="sector_excess_intra_rank",
    required_columns={"stock_code", "date", "pct_chg"},
    requires_market_data=True,
    requires_sector_data=True,
    lookback_days=10,
    calculator=_calc_sec_excess,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_excess_intra_rank"),
    description="三位一体领先因子(FrozenClip): 行业跑赢大盘且个股领涨行业"
))

__all__ = [
    "market_vols_adaptive_iv",
    "market_trend_adaptive_beta",
    "sector_excess_intra_rank",
]
