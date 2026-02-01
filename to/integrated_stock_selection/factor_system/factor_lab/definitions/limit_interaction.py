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
import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


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


def _to_ts(date_series: pd.Series) -> pd.Series:
    """
    将 date 列稳健转换为 pandas Timestamp（支持 'YYYY-MM-DD' / 'YYYYMMDD' / int）
    """
    if date_series.dtype == "datetime64[ns]":
        return date_series
    ds = date_series.astype(str)
    ds = ds.str.replace("-", "", regex=False)
    return pd.to_datetime(ds, format="%Y%m%d", errors="coerce")


# ==============================================================================
# 3. 核心计算引擎
# ==============================================================================
class _LimitInteractionHelper:
    # 训练期阈值拟合范围（用户指定）
    TRAIN_START = pd.Timestamp("2023-06-01")
    TRAIN_END = pd.Timestamp("2025-03-01")

    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        # [关键] 必须先拷贝并排序，否则 groupby().shift() 可能会乱序
        self.data = data.copy()
        self.data.sort_values(["stock_code", "date"], inplace=True)

        self.ctx = ctx
        self.original_index = data.index

        # 时间戳列（用于训练期 mask）
        self.data["_ts"] = _to_ts(self.data["date"])

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

        # 3. 启动预计算
        self._precompute_metrics()

    def _train_mask(self, ts: pd.Series) -> pd.Series:
        return (ts >= self.TRAIN_START) & (ts <= self.TRAIN_END)

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
            mask = self._train_mask(self.data["_ts"])
            clipper.fit(series[mask])
            self._clipper_cache[factor_name] = clipper
        return clipper.transform(series)

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
        prev_close = self.data.groupby("stock_code")["close"].shift(1)
        self.data["intraday_strength"] = (self.data["close"] - self.data["open"]) / (prev_close + 1e-8)

        # C4. CLV / Resilience（修复一字板）
        range_len = self.data["high"] - self.data["low"]
        clv = (self.data["close"] - self.data["low"]) / (range_len + 1e-8)

        is_flat = range_len < 1e-5
        mask_one_bar_limit_up = is_flat & (self.data["is_limit_up"] == 1)
        self.data["clv"] = np.where(mask_one_bar_limit_up, 1.0, clv)

        self.data["resilience_z"] = self._intra_sector_standardize(self.data["clv"])

    def get_factor(self, factor_name: str) -> pd.Series:
        """因子计算路由"""

        # 1) Raw
        if factor_name == "sector_limit_overflow_alpha":
            gate = self.data.join(self.s_heat_rank.rename("gate"), on=["date", "sector_code"])["gate"]
            alpha = self.data["turnover_z"]
            filter_mask = (1 - self.data["is_limit_up"]) * (1 - self.data["is_limit_down"])
            raw_factor = gate * alpha * filter_mask
            need_sector_neutral = True

        elif factor_name == "sector_broken_resilience_alpha":
            gate = self.data.join(self.s_divergence_rank.rename("gate"), on=["date", "sector_code"])["gate"]
            alpha = self.data["resilience_z"]
            raw_factor = gate * alpha
            need_sector_neutral = True

        elif factor_name == "limit_sustain_alpha":
            gate = self.data["is_prev_limit_up"]
            alpha = self.data["intraday_strength"]
            raw_factor = gate * alpha
            need_sector_neutral = False

        elif factor_name == "market_emotion_resonance_alpha":
            env = self.market_mood  # 可能为 NaN（冰点）
            alpha = self.data["active_z"]
            raw_factor = env * alpha
            need_sector_neutral = False

        else:
            raise ValueError(f"Unknown factor: {factor_name}")

        # 2) 后处理（保持原语义）
        # A. 填充空值：共振因子 NaN -> 0 意味“无共振”
        joined_res = raw_factor.fillna(0)

        # B. Frozen MAD Clip（训练期拟合阈值）
        clipped_res = self._frozen_clip(factor_name, joined_res, n_mad=3.0)

        # C. 标准化分流
        if need_sector_neutral:
            final_res = self.data.assign(res=clipped_res).groupby(["date", "sector_code"])["res"].transform(_safe_zscore)
        else:
            global_std = clipped_res.std()
            if global_std == 0 or np.isnan(global_std):
                final_res = clipped_res
            else:
                final_res = clipped_res / global_std

        return final_res.reindex(self.original_index).fillna(0).astype("float64")


# ==============================================================================
# 4. 注册因子
# ==============================================================================
def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _LimitInteractionHelper:
    key = "_limit_interaction_helper_frozenclip_v1"
    if key not in ctx.extra:
        ctx.extra[key] = _LimitInteractionHelper(data, ctx)
    return ctx.extra[key]


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
    description="板块高潮溢出因子(补涨) - FrozenClip"
))

sector_broken_resilience_alpha = register_factor(FactorDefinition(
    name="sector_broken_resilience_alpha",
    required_columns={"stock_code", "date", "is_zhaban", "high", "low", "close"},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_c_resilience,
    description="板块分歧抵抗因子(抗分歧) - FrozenClip"
))

limit_sustain_alpha = register_factor(FactorDefinition(
    name="limit_sustain_alpha",
    required_columns={"stock_code", "date", "is_limit_up", "open", "close"},
    requires_sector_data=False,
    lookback_days=10,
    calculator=_c_sustain,
    description="涨停次日接力/核按钮识别因子 - FrozenClip"
))

market_emotion_resonance_alpha = register_factor(FactorDefinition(
    name="market_emotion_resonance_alpha",
    required_columns={"stock_code", "date", "is_limit_up", "pct_chg"},
    requires_sector_data=False,
    lookback_days=10,
    calculator=_c_resonance,
    description="市场情绪共振因子(昨日涨停溢价x个股活跃) - FrozenClip"
))

__all__ = [
    "sector_limit_overflow_alpha",
    "sector_broken_resilience_alpha",
    "limit_sustain_alpha",
    "market_emotion_resonance_alpha",
]
