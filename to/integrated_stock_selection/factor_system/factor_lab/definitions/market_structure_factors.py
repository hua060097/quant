"""
市场结构与行业宽度因子 (Market Structure & Sector Breadth Factors) - FrozenClip版

改动要点（关键）：
- 原 _adaptive_clip() 在全样本上算 median/MAD 阈值 -> 存在未来信息泄露（尤其你“全量一起生成因子”的 pipeline）
- 改为 Frozen MAD Clipper：
  只用训练集日期范围拟合阈值（train_start~train_end），并冻结用于全期 clip。

训练集日期范围（用户指定）：
- train_start: 2023-06-01
- train_end:   2025-03-01

其余逻辑尽量保持原版（包括 120 日位置锚点等），避免对你已有 alpha 形态造成额外扰动。
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
# 1. Frozen（训练期阈值冻结）MAD Clipper
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

        # mad 极小 -> 不裁剪（避免“门槛趋近 0 导致输出跳变/常数化”）
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


# ==============================================================================
# 2. 工具函数
# ==============================================================================
def _to_ts(date_series: pd.Series) -> pd.Series:
    """将 date 列稳健转为 Timestamp（支持 'YYYY-MM-DD' / 'YYYYMMDD' / int）"""
    if date_series.dtype == "datetime64[ns]":
        return date_series
    ds = date_series.astype(str)
    ds = ds.str.replace("-", "", regex=False)
    return pd.to_datetime(ds, format="%Y%m%d", errors="coerce")


def _get_market_data(ctx: FactorContext) -> pd.DataFrame:
    """获取市场基准数据（按 date 排序）"""
    market = ctx.require_market()
    m = market.sort_values("date").reset_index(drop=True).copy()
    # 确保 market 的 date 可与 data['date'] 对齐（统一 Timestamp）
    m["_ts"] = _to_ts(m["date"])
    return m


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
# 3. 市场结构计算辅助类
# ==============================================================================
class _MarketStructureHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.data = data.copy()
        self.ctx = ctx
        self.original_index = data.index

        # 行业映射注入（你是 SW 二级行业，映射表保持二级即可）
        sector_map = ctx.extra.get("stock_to_sw_sector", {})
        self.data["sector_code"] = self.data["stock_code"].map(sector_map).fillna("Unknown")

        # 时间戳（用于训练期 mask）
        self.data["_ts"] = _to_ts(self.data["date"])
        self.train_start_, self.train_end_ = _resolve_frozen_window(ctx.extra or {})

        # 冻结 clipper 缓存
        self._clipper_cache: dict[str, FrozenMADClipper] = {}
        self._module_name = "market_structure_factors"

        # 预计算
        self._precompute()

    def _train_mask(self) -> pd.Series:
        ts = self.data["_ts"]
        return (ts >= self.train_start_) & (ts <= self.train_end_)

    def _frozen_clip(self, factor_name: str, series: pd.Series, n_mad: float = 3.0) -> pd.Series:
        """
        Frozen clip：仅用训练期拟合阈值，冻结用于全量。
        """
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

    def _precompute(self):
        """向量化预计算市场与行业层级指标"""

        # --- A. 行业收益宽表 ---
        s_ret_wide = (
            self.data.groupby(["date", "sector_code"])["pct_chg"]
            .mean()
            .unstack("sector_code")
            .sort_index()
        )
        # 统一 index 为 Timestamp，保证和 market 对齐
        s_ret_wide.index = _to_ts(s_ret_wide.index.to_series()).values

        # --- B. 市场收益序列 ---
        mkt_df = _get_market_data(self.ctx).set_index("_ts")
        # 这里假设 market 中 daily_change_pct 是“市场基准日涨跌幅”（你之前就是这么用）
        mkt_ret = pd.to_numeric(mkt_df["daily_change_pct"], errors="coerce")

        # --- C. 市场/行业层级 ---
        # 1) 市场级：全市场行业离散度
        self.mkt_sector_dispersion = s_ret_wide.std(axis=1)

        # 2) 行业-市场：行业与大盘相关性（rolling corr）
        # 结果是 MultiIndex (date, sector) 结构
        self.s_mkt_corr = s_ret_wide.rolling(20, min_periods=10).corr(mkt_ret)

        # 3) 行业级：行业内上涨股票占比 (Breadth)
        is_positive = (pd.to_numeric(self.data["pct_chg"], errors="coerce") > 0).astype(int)
        self.s_breadth = (
            self.data.assign(is_pos=is_positive)
            .groupby(["date", "sector_code"])["is_pos"]
            .mean()
            .unstack("sector_code")
        )
        self.s_breadth.index = _to_ts(self.s_breadth.index.to_series()).values

        # --- D. 个股-行业层级（120 日锚点） ---
        window_size = 120
        min_p = 40

        # 必须保证按 stock_code/date 排序，rolling 才严格因果
        self.data.sort_values(["stock_code", "date"], inplace=True)

        high_anchor = self.data.groupby("stock_code")["high"].transform(
            lambda x: x.rolling(window_size, min_periods=min_p).max()
        )
        low_anchor = self.data.groupby("stock_code")["low"].transform(
            lambda x: x.rolling(window_size, min_periods=min_p).min()
        )

        # 个股在过去 window 内的位置 (0-1)
        self.data["stock_pos"] = (self.data["close"] - low_anchor) / (high_anchor - low_anchor + 1e-8)

        # 行业平均位置
        s_pos_mean = self.data.groupby(["date", "sector_code"])["stock_pos"].mean().unstack("sector_code")
        s_pos_mean.index = _to_ts(s_pos_mean.index.to_series()).values
        self.s_pos_mean = s_pos_mean

    def compute(self, factor_name: str) -> pd.Series:
        """执行因子映射逻辑（FrozenClip）"""
        raw = self._build_raw_factor(factor_name)
        return self._frozen_clip(factor_name, raw, n_mad=3.0).reindex(self.original_index)

    def _build_raw_factor(self, factor_name: str, target_data: pd.DataFrame | None = None) -> pd.Series:
        rows = self.data if target_data is None else target_data.copy()

        if factor_name == "mkt_sector_dispersion":
            v = rows["_ts"].map(self.mkt_sector_dispersion)
            return v.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")

        if factor_name == "sector_mkt_corr_20d":
            s_val = self.s_mkt_corr.stack().rename("v")
            joined = rows.join(s_val, on=["_ts", "sector_code"], how="left")["v"]
            return joined.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")

        if factor_name == "intra_sector_breadth":
            s_val = self.s_breadth.stack().rename("v")
            joined = rows.join(s_val, on=["_ts", "sector_code"], how="left")["v"]
            return joined.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")

        if factor_name == "stock_sector_pos_align":
            s_pos = self.s_pos_mean.stack().rename("sp")
            joined = rows.join(s_pos, on=["_ts", "sector_code"], how="left")
            diff = (joined["stock_pos"] - joined["sp"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return diff.astype("float64")

        return pd.Series(0.0, index=rows.index, dtype="float64")

    def compute_target(self, factor_name: str, target_index: pd.Index) -> pd.Series:
        raw_full = self._build_raw_factor(factor_name)
        _ = self._frozen_clip(factor_name, raw_full, n_mad=3.0)
        clipper = self._clipper_cache[factor_name]
        raw_target = self._build_raw_factor(factor_name, self.data.loc[target_index])
        return clipper.transform(raw_target).reindex(target_index)


# ==============================================================================
# 4. 注册逻辑
# ==============================================================================
def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _MarketStructureHelper:
    key = "_market_structure_helper_frozenclip_v1"
    sig = (len(data), data.index[0] if len(data) else None, data.index[-1] if len(data) else None)
    sig_key = f"{key}__sig"
    if key not in ctx.extra or ctx.extra.get(sig_key) != sig:
        ctx.extra[key] = _MarketStructureHelper(data, ctx)
        ctx.extra[sig_key] = sig
    return ctx.extra[key]


def _calc_disp(d, c): return _get_helper(d, c).compute("mkt_sector_dispersion")
def _calc_corr(d, c): return _get_helper(d, c).compute("sector_mkt_corr_20d")
def _calc_breadth(d, c): return _get_helper(d, c).compute("intra_sector_breadth")
def _calc_align(d, c): return _get_helper(d, c).compute("stock_sector_pos_align")
def _target_factor(name): return lambda d, ti, c: _get_helper(d, c).compute_target(name, ti)


# 因子注册（lookback_days 保持你原文件设置）
mkt_sector_dispersion = register_factor(FactorDefinition(
    name="mkt_sector_dispersion",
    required_columns={"date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=30,
    calculator=_calc_disp,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("mkt_sector_dispersion")
))

sector_mkt_corr_20d = register_factor(FactorDefinition(
    name="sector_mkt_corr_20d",
    required_columns={"date", "pct_chg"},
    requires_market_data=True,
    requires_sector_data=True,
    lookback_days=40,
    calculator=_calc_corr,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_mkt_corr_20d")
))

intra_sector_breadth = register_factor(FactorDefinition(
    name="intra_sector_breadth",
    required_columns={"date", "pct_chg"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_calc_breadth,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("intra_sector_breadth")
))

stock_sector_pos_align = register_factor(FactorDefinition(
    name="stock_sector_pos_align",
    required_columns={"stock_code", "date", "high", "low", "close"},
    requires_sector_data=True,
    lookback_days=120,
    calculator=_calc_align,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("stock_sector_pos_align")
))

__all__ = [
    "mkt_sector_dispersion",
    "sector_mkt_corr_20d",
    "intra_sector_breadth",
    "stock_sector_pos_align",
]
