"""
涨跌停与市场情绪交互因子 (Limit-Up/Down & Sentiment Interaction Factors) - FrozenClip版
文件名: limit_interaction.py

本版本改动（关键）：
- 将 _adaptive_clip（全样本 MAD 去极值，存在未来信息泄露）
  替换为 Frozen MAD Clipper：
  只用训练集日期范围拟合阈值（train_start~train_end），并冻结用于全期 clip。

训练集日期范围（用户指定）：
- train_start: 2023-06-01
- train_end:   2025-03-01

其余逻辑保持与原版本一致，尽量保留原 alpha 的“锋利度”。
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

    这是对原 _adaptive_clip 的“最小侵入替代”，避免未来信息泄露。
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
            # 训练期全为空：退化为不裁剪
            self.median_, self.mad_ = 0.0, 0.0
            self.lower_, self.upper_ = -np.inf, np.inf
            self._fitted = True
            return self

        med = float(s.median())
        mad = float((s - med).abs().median())
        thr = self.n_mad * mad * 1.4826

        # 稀疏保护：mad 极小就不裁剪（与你原 _adaptive_clip 的“mad==0不裁剪”一致）
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
# 2. 基础统计学工具
# ==============================================================================
def _safe_zscore(series: pd.Series) -> pd.Series:
    """安全的 Z-Score"""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    std = s.std()
    if s.empty or std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (s - s.mean()) / (std + 1e-8)




def _cross_sectional_robust_zscore(series: pd.Series, dates: pd.Series) -> pd.Series:
    """
    横截面 Robust Z-Score（无未来信息泄露）

    使用 median + MAD：
        z = (x - median) / (1.4826 * MAD)

    相比全局 std：
    - 不使用未来 regime 波动结构
    - 对 A 股涨停/炸板极值更稳定
    - 更适合横截面排序模型
    """

    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def _robust(x: pd.Series) -> pd.Series:
        med = x.median()
        mad = (x - med).abs().median()

        if pd.isna(mad) or mad < 1e-12:
            return pd.Series(0.0, index=x.index)

        return (x - med) / (1.4826 * mad + 1e-9)

    return s.groupby(dates).transform(_robust)


def _rolling_percentile_self(series: pd.Series, window: int = 20, min_periods: int = 10) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    values: list[float] = []
    for i, cur in enumerate(s):
        if pd.isna(cur):
            values.append(np.nan)
            continue
        hist = s.iloc[max(0, i - window + 1): i + 1].dropna()
        if len(hist) < min_periods:
            values.append(np.nan)
            continue
        values.append(float((hist <= cur).mean()))
    return pd.Series(values, index=series.index, dtype="float64")


def _to_ts(date_series: pd.Series) -> pd.Series:
    """
    将 date 列稳健转换为 pandas Timestamp（支持 'YYYY-MM-DD' / 'YYYYMMDD' / int）
    """
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
# 3. 核心计算引擎
# ==============================================================================
class _LimitInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        # [关键] 必须先拷贝并排序，否则 groupby().shift() 可能会乱序
        self.data = data.copy()
        self.data.sort_values(["stock_code", "date"], inplace=True)

        self.ctx = ctx
        self.original_index = data.index

        # 时间戳列（用于训练期 mask）
        self.data["_ts"] = _to_ts(self.data["date"])
        self.train_start_, self.train_end_ = _resolve_frozen_window(ctx.extra or {})

        # 1. 行业映射（你是 SW 二级行业，只要 stock_to_sw_sector 映射的是二级即可）
        sector_map = ctx.extra.get("stock_to_sw_sector", {})
        if not sector_map:
            self.data["sector_code"] = "All"
        else:
            self.data["sector_code"] = self.data["stock_code"].map(sector_map).fillna("Unknown")

        # 2. 关键列预处理 (确保是 1/0 且无 NaN)
        cols_check = ["is_limit_up", "is_limit_down", "is_zhaban"]
        for col in cols_check:
            if col not in self.data.columns:
                self.data[col] = 0
            else:
                self.data[col] = self.data[col].fillna(0).astype(int)

        # Frozen clippers（按因子名缓存一次，避免重复拟合）
        self._clipper_cache: dict[str, FrozenMADClipper] = {}
        self._module_name = "limit_interaction"

        # 3. 启动预计算
        self._precompute_metrics()

    def _train_mask(self, ts: pd.Series) -> pd.Series:
        return (ts >= self.train_start_) & (ts <= self.train_end_)

    def _intra_sector_standardize(self, series: pd.Series) -> pd.Series:
        """行业内标准化"""
        return self.data.assign(tmp=series).groupby(["date", "sector_code"])["tmp"].transform(_safe_zscore)

    def _frozen_clip(self, factor_name: str, series: pd.Series, n_mad: float = 3.0) -> pd.Series:
        """
        Frozen clip：仅用训练期拟合阈值，冻结用于全量。
        注意：这里对“已填充 NaN 后”的序列做 clip，保持你原流程的语义一致。
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
                mask = self._train_mask(self.data["_ts"])
                train_series = series[mask]
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

    def _build_raw_factor(self, factor_name: str, target_data: pd.DataFrame | None = None) -> tuple[pd.Series, bool]:
        rows = self.data if target_data is None else target_data.copy()

        if factor_name == "sector_limit_overflow_alpha":
            gate = rows.join(self.s_heat_rank.rename("gate"), on=["date", "sector_code"])["gate"]
            alpha = rows["turnover_z"]
            filter_mask = (1 - rows["is_limit_up"]) * (1 - rows["is_limit_down"])
            raw_factor = gate * alpha * filter_mask
            need_sector_neutral = True

        elif factor_name == "sector_broken_resilience_alpha":
            gate = rows.join(self.s_divergence_rank.rename("gate"), on=["date", "sector_code"])["gate"]
            alpha = rows["resilience_z"]
            raw_factor = gate * alpha
            need_sector_neutral = True

        elif factor_name == "limit_sustain_alpha":
            gate = rows["is_prev_limit_up"]
            alpha = rows["intraday_strength"]
            raw_factor = gate * alpha
            need_sector_neutral = False

        elif factor_name == "market_emotion_resonance_alpha":
            env = self.market_mood.reindex(rows.index)
            alpha = rows["active_z"]
            raw_factor = env * alpha
            need_sector_neutral = False

        elif factor_name == "prev_first_limit_up_performance":
            # 市场环境变量：昨日首板涨停今日平均表现，直接广播
            raw_factor = self.first_limit_up_perf_broadcast.reindex(rows.index).fillna(0)
            need_sector_neutral = False

        elif factor_name == "prev_first_limit_up_performance_percentile_20d":
            # 市场环境变量：昨日首板涨停今日平均表现的20日滚动分位数，直接广播
            raw_factor = self.first_limit_up_perf_percentile_20d_broadcast.reindex(rows.index).fillna(0)
            need_sector_neutral = False

        elif factor_name == "prev_non_first_limit_up_performance":
            # 市场环境变量：昨日非首板涨停今日平均表现，直接广播
            raw_factor = self.non_first_limit_up_perf_broadcast.reindex(rows.index).fillna(0)
            need_sector_neutral = False

        elif factor_name == "prev_non_first_limit_up_performance_percentile_20d":
            # 市场环境变量：昨日非首板涨停今日平均表现的20日滚动分位数，直接广播
            raw_factor = self.non_first_limit_up_perf_percentile_20d_broadcast.reindex(rows.index).fillna(0)
            need_sector_neutral = False

        else:
            raise ValueError(f"Unknown factor: {factor_name}")

        return raw_factor.astype("float64"), need_sector_neutral

    def _precompute_metrics(self):
        """预计算指标"""

        # --- A. 板块级情绪 (Sector Level) ---
        sector_stats = self.data.groupby(["date", "sector_code"])[["is_limit_up", "is_limit_down", "is_zhaban"]].sum()
        sector_counts = self.data.groupby(["date", "sector_code"])["stock_code"].count()

        # A1. 板块热度 (3日均值 rank)
        s_limit_up_ratio = sector_stats["is_limit_up"] / (sector_counts + 1e-8)
        self.s_heat_rank = (
            s_limit_up_ratio.unstack("sector_code")
            .rolling(3, min_periods=1).mean()
            .rank(axis=1, pct=True)
            .stack()
        )

        # A2. 板块分歧 (3日均值 rank)
        total_attempts = sector_stats["is_limit_up"] + sector_stats["is_zhaban"]
        s_zhaban_ratio = sector_stats["is_zhaban"] / (total_attempts + 1e-8)
        self.s_divergence_rank = (
            s_zhaban_ratio.unstack("sector_code")
            .rolling(3, min_periods=1).mean()
            .rank(axis=1, pct=True)
            .stack()
        )

        # --- B. 全市场级情绪 (Market Level) ---
        prev_limit_up = self.data.groupby("stock_code")["is_limit_up"].shift(1).fillna(0)
        self.data["is_prev_limit_up"] = prev_limit_up

        # 昨日首板涨停标识：昨日涨停且前日非涨停（区分首板与连板）
        prev_prev_limit_up = self.data.groupby("stock_code")["is_limit_up"].shift(2).fillna(0)
        self.data["is_prev_first_limit_up"] = ((prev_limit_up == 1) & (prev_prev_limit_up == 0)).astype(int)
        self.data["is_prev_non_first_limit_up"] = ((prev_limit_up == 1) & (prev_prev_limit_up == 1)).astype(int)

        # 昨日首板涨停今日平均表现（市场环境变量，按日聚合后广播给所有股票）
        first_limit_up_perf = (
            self.data[self.data["is_prev_first_limit_up"] == 1]
            .groupby("date")["pct_chg"]
            .mean()
        )
        # 无首板涨停股的日期填0（市场冰点无信号）
        self.first_limit_up_perf_broadcast = self.data["date"].map(first_limit_up_perf).fillna(0)

        daily_perf = self.data[["date"]].drop_duplicates().sort_values("date")
        daily_perf["first_limit_up_perf"] = daily_perf["date"].map(first_limit_up_perf).fillna(0.0)
        daily_perf["first_limit_up_perf_percentile_20d"] = _rolling_percentile_self(
            daily_perf["first_limit_up_perf"],
            window=20,
            min_periods=10,
        )
        pct_map = pd.Series(
            daily_perf["first_limit_up_perf_percentile_20d"].values,
            index=daily_perf["date"],
        )
        self.first_limit_up_perf_percentile_20d_broadcast = self.data["date"].map(pct_map).fillna(0)

        # 昨日非首板涨停今日平均表现（连板股次日表现，按日聚合后广播给所有股票）
        non_first_limit_up_perf = (
            self.data[self.data["is_prev_non_first_limit_up"] == 1]
            .groupby("date")["pct_chg"]
            .mean()
        )
        self.non_first_limit_up_perf_broadcast = self.data["date"].map(non_first_limit_up_perf).fillna(0)

        daily_non_first_perf = self.data[["date"]].drop_duplicates().sort_values("date")
        daily_non_first_perf["non_first_limit_up_perf"] = daily_non_first_perf["date"].map(non_first_limit_up_perf).fillna(0.0)
        daily_non_first_perf["non_first_limit_up_perf_percentile_20d"] = _rolling_percentile_self(
            daily_non_first_perf["non_first_limit_up_perf"],
            window=20,
            min_periods=10,
        )
        non_first_pct_map = pd.Series(
            daily_non_first_perf["non_first_limit_up_perf_percentile_20d"].values,
            index=daily_non_first_perf["date"],
        )
        self.non_first_limit_up_perf_percentile_20d_broadcast = self.data["date"].map(non_first_pct_map).fillna(0)

        # 昨日涨停溢价：只取 is_prev_limit_up == 1 的样本
        limit_up_premium = self.data[self.data["is_prev_limit_up"] == 1].groupby("date")["pct_chg"].mean()

        # [保持原语义] 无涨停股 -> NaN（市场冰点），不填 0
        self.market_mood = self.data["date"].map(limit_up_premium)

        # --- C. 个股级特征 (Stock Level) ---
        # C1. 活跃度 Z-Score（行业内）
        self.data["active_z"] = self._intra_sector_standardize(self.data["pct_chg"])

        # C2. 换手活跃度（行业内）
        if "volume" in self.data.columns and "circ_mv" in self.data.columns:
            to = self.data["volume"] / (self.data["circ_mv"] + 1e-8)
            self.data["turnover_z"] = self._intra_sector_standardize(to)
        else:
            self.data["turnover_z"] = self.data["active_z"]

        # C3. 日内强度
        if {"open", "close"}.issubset(self.data.columns):
            prev_close = self.data.groupby("stock_code")["close"].shift(1)
            self.data["intraday_strength"] = (self.data["close"] - self.data["open"]) / (prev_close + 1e-8)
        else:
            self.data["intraday_strength"] = 0.0

        # C4. CLV / Resilience（修复一字板）
        if {"high", "low", "close"}.issubset(self.data.columns):
            range_len = self.data["high"] - self.data["low"]
            clv = (self.data["close"] - self.data["low"]) / (range_len + 1e-8)

            is_flat = range_len < 1e-5
            mask_one_bar_limit_up = is_flat & (self.data["is_limit_up"] == 1)
            self.data["clv"] = np.where(mask_one_bar_limit_up, 1.0, clv)
        else:
            self.data["clv"] = 0.0

        self.data["resilience_z"] = self._intra_sector_standardize(self.data["clv"])

    def get_factor(self, factor_name: str) -> pd.Series:
        """因子计算路由"""
        raw_factor, need_sector_neutral = self._build_raw_factor(factor_name)

        # 2) 后处理（保持原语义）
        # A. 填充空值：共振因子 NaN -> 0 意味“无共振”
        joined_res = raw_factor.fillna(0)

        if factor_name in {
            "prev_first_limit_up_performance",
            "prev_first_limit_up_performance_percentile_20d",
            "prev_non_first_limit_up_performance",
            "prev_non_first_limit_up_performance_percentile_20d",
        }:
            # 市场环境变量：同一天所有股票取同一个广播值。
            # 截面 robust z-score 会因为日内截面无离散度而退化为全 0；
            # FrozenClip 也会削弱“首板次日平均涨跌幅”的原始量纲含义。
            return joined_res.reindex(self.original_index).fillna(0).astype("float64")

        # B. Frozen MAD Clip（训练期拟合阈值）
        clipped_res = self._frozen_clip(factor_name, joined_res, n_mad=3.0)

        # C. 标准化分流
        if need_sector_neutral:
            final_res = self.data.assign(res=clipped_res).groupby(["date", "sector_code"])["res"].transform(_safe_zscore)
        elif factor_name == "limit_sustain_alpha":
            # 方案A: 仅"昨日涨停股"子集内做截面z-score
            # 非涨停股天然为0（中性，不参与排序竞争），涨停股之间比较日内接力强度
            mask = self.data["is_prev_limit_up"] == 1
            final_res = pd.Series(0.0, index=self.data.index, dtype="float64")
            if mask.sum() >= 3:  # 最少3只涨停股截面才稳定
                subset_series = clipped_res[mask]
                subset_dates = self.data.loc[mask, "date"]
                final_res.loc[subset_series.index] = _cross_sectional_robust_zscore(
                    subset_series, subset_dates
                ).astype("float64")
        else:
            final_res = _cross_sectional_robust_zscore(
                clipped_res,
                self.data["date"]
            )

        return final_res.reindex(self.original_index).fillna(0).astype("float64")

    def compute_target(self, factor_name: str, target_index: pd.Index) -> pd.Series:
        target_data = self.data.loc[target_index].copy()
        raw_target, need_sector_neutral = self._build_raw_factor(factor_name, target_data)

        if factor_name in {
            "prev_first_limit_up_performance",
            "prev_first_limit_up_performance_percentile_20d",
            "prev_non_first_limit_up_performance",
            "prev_non_first_limit_up_performance_percentile_20d",
        }:
            # 市场环境变量：同一天所有股票取同一个广播值，直接保留原始广播值。
            return raw_target.reindex(target_index).fillna(0).astype("float64")

        # 拟合/读取 frozen 参数时，仍以全历史 raw 为背景；输出只保留目标日期
        raw_full, _ = self._build_raw_factor(factor_name)
        _ = self._frozen_clip(factor_name, raw_full.fillna(0), n_mad=3.0)
        clipper = self._clipper_cache[factor_name]
        clipped_target = clipper.transform(raw_target.fillna(0))

        if need_sector_neutral:
            final_res = (
                target_data.assign(res=clipped_target)
                .groupby(["date", "sector_code"])["res"]
                .transform(_safe_zscore)
            )
        elif factor_name == "limit_sustain_alpha":
            mask = target_data["is_prev_limit_up"] == 1
            final_res = pd.Series(0.0, index=target_data.index, dtype="float64")
            if mask.sum() >= 3:
                subset_series = clipped_target[mask]
                subset_dates = target_data.loc[mask, "date"]
                final_res.loc[subset_series.index] = _cross_sectional_robust_zscore(
                    subset_series, subset_dates
                ).astype("float64")
        else:
            final_res = _cross_sectional_robust_zscore(
                clipped_target,
                target_data["date"]
            )

        return final_res.reindex(target_index).fillna(0).astype("float64")


# ==============================================================================
# 4. 注册因子
# ==============================================================================
def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _LimitInteractionHelper:
    key = "_limit_interaction_helper_frozenclip_v1"
    sig = (len(data), data.index[0] if len(data) else None, data.index[-1] if len(data) else None)
    sig_key = f"{key}__sig"
    if key not in ctx.extra or ctx.extra.get(sig_key) != sig:
        ctx.extra[key] = _LimitInteractionHelper(data, ctx)
        ctx.extra[sig_key] = sig
    return ctx.extra[key]


def _c_overflow(d, c): return _get_helper(d, c).get_factor("sector_limit_overflow_alpha")
def _c_resilience(d, c): return _get_helper(d, c).get_factor("sector_broken_resilience_alpha")
def _c_sustain(d, c): return _get_helper(d, c).get_factor("limit_sustain_alpha")
def _c_resonance(d, c): return _get_helper(d, c).get_factor("market_emotion_resonance_alpha")
def _c_prev_first_limit_up_perf(d, c): return _get_helper(d, c).get_factor("prev_first_limit_up_performance")
def _c_prev_first_limit_up_perf_pct20(d, c): return _get_helper(d, c).get_factor("prev_first_limit_up_performance_percentile_20d")
def _c_prev_non_first_limit_up_perf(d, c): return _get_helper(d, c).get_factor("prev_non_first_limit_up_performance")
def _c_prev_non_first_limit_up_perf_pct20(d, c): return _get_helper(d, c).get_factor("prev_non_first_limit_up_performance_percentile_20d")
def _target_factor(name):
    return lambda d, ti, c: _get_helper(d, c).compute_target(name, ti)


sector_limit_overflow_alpha = register_factor(FactorDefinition(
    name="sector_limit_overflow_alpha",
    required_columns={"stock_code", "date", "is_limit_up", "is_limit_down", "volume", "circ_mv"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_c_overflow,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_limit_overflow_alpha"),
    description="板块高潮溢出因子(补涨) - FrozenClip"
))

sector_broken_resilience_alpha = register_factor(FactorDefinition(
    name="sector_broken_resilience_alpha",
    required_columns={"stock_code", "date", "is_zhaban", "high", "low", "close"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_c_resilience,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("sector_broken_resilience_alpha"),
    description="板块分歧抵抗因子(抗分歧) - FrozenClip"
))

limit_sustain_alpha = register_factor(FactorDefinition(
    name="limit_sustain_alpha",
    required_columns={"stock_code", "date", "is_limit_up", "open", "close"},
    requires_sector_data=False,
    lookback_days=10,
    calculator=_c_sustain,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("limit_sustain_alpha"),
    description="涨停次日接力/核按钮识别因子 - FrozenClip"
))

market_emotion_resonance_alpha = register_factor(FactorDefinition(
    name="market_emotion_resonance_alpha",
    required_columns={"stock_code", "date", "is_limit_up", "pct_chg"},
    requires_sector_data=False,
    lookback_days=10,
    calculator=_c_resonance,
    calculation_scope='stateful_history',
    target_calculator=_target_factor("market_emotion_resonance_alpha"),
    description="市场情绪共振因子(昨日涨停溢价x个股活跃) - FrozenClip"
))

prev_first_limit_up_performance = register_factor(FactorDefinition(
    name="prev_first_limit_up_performance",
    required_columns={"stock_code", "date", "is_limit_up", "pct_chg"},
    requires_sector_data=False,
    lookback_days=10,
    calculator=_c_prev_first_limit_up_perf,
    calculation_scope='daily_broadcast',
    target_calculator=_target_factor("prev_first_limit_up_performance"),
    description="昨日首板涨停今日平均表现(市场环境变量,按日广播,保留原始涨跌幅均值)"
))

prev_first_limit_up_performance_percentile_20d = register_factor(FactorDefinition(
    name="prev_first_limit_up_performance_percentile_20d",
    required_columns={"stock_code", "date", "is_limit_up", "pct_chg"},
    requires_sector_data=False,
    lookback_days=25,
    calculator=_c_prev_first_limit_up_perf_pct20,
    calculation_scope='daily_broadcast',
    target_calculator=_target_factor("prev_first_limit_up_performance_percentile_20d"),
    description="昨日首板涨停今日平均表现位于过去20日窗口的滚动分位数(按日广播)"
))

prev_non_first_limit_up_performance = register_factor(FactorDefinition(
    name="prev_non_first_limit_up_performance",
    required_columns={"stock_code", "date", "is_limit_up", "pct_chg"},
    requires_sector_data=False,
    lookback_days=10,
    calculator=_c_prev_non_first_limit_up_perf,
    calculation_scope='daily_broadcast',
    target_calculator=_target_factor("prev_non_first_limit_up_performance"),
    description="昨日非首板涨停今日平均表现(连板股市场环境变量,按日广播,保留原始涨跌幅均值)"
))

prev_non_first_limit_up_performance_percentile_20d = register_factor(FactorDefinition(
    name="prev_non_first_limit_up_performance_percentile_20d",
    required_columns={"stock_code", "date", "is_limit_up", "pct_chg"},
    requires_sector_data=False,
    lookback_days=25,
    calculator=_c_prev_non_first_limit_up_perf_pct20,
    calculation_scope='daily_broadcast',
    target_calculator=_target_factor("prev_non_first_limit_up_performance_percentile_20d"),
    description="昨日非首板涨停今日平均表现位于过去20日窗口的滚动分位数(按日广播)"
))

__all__ = [
    "sector_limit_overflow_alpha",
    "sector_broken_resilience_alpha",
    "limit_sustain_alpha",
    "market_emotion_resonance_alpha",
    "prev_first_limit_up_performance",
    "prev_first_limit_up_performance_percentile_20d",
    "prev_non_first_limit_up_performance",
    "prev_non_first_limit_up_performance_percentile_20d",
]
