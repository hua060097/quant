"""
K线与成交量相关因子 - 优化版（无开关，支持 valid_cs_mask，输出 float32）

在原 optimized_noswitch 的基础上增加：
- valid_cs_mask：若 data 中包含该列，则 mask=False（ST/新股等）样本不会参与任何“按日截面统计”（本文件本身多为日内形态，
  无需做 robust z），同时这些样本的输出会被置为 0（中性），避免污染训练/排序。
- 输出统一为 float32（内部计算保持 float64 以稳定数值）

注意：本文件的因子多为 bounded / ratio 类，不做额外 CS zscore，避免二次标准化改变形态。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_EPS = 1e-12
_OUT_DTYPE = "float32"


# ==============================================================================
# 工具函数
# ==============================================================================
def _to_ts(date_series: pd.Series) -> pd.Series:
    """稳健把 date 转为 Timestamp（支持 'YYYY-MM-DD' / 'YYYYMMDD' / int）"""
    if pd.api.types.is_datetime64_any_dtype(date_series):
        return pd.to_datetime(date_series, errors="coerce")
    ds = date_series.astype(str).str.replace("-", "", regex=False)
    return pd.to_datetime(ds, format="%Y%m%d", errors="coerce")


def _clean_num(s: pd.Series, fill: float = 0.0) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s2.fillna(fill).astype("float64")


def _clean_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(1.0) > 0.5


def _apply_valid_mask_to_output(data: pd.DataFrame, out: pd.Series) -> pd.Series:
    """
    若存在 valid_cs_mask：把无效样本输出置为 0（中性），避免污染训练/排序。
    注意：这里不改变有效样本的分布形态。
    """
    if "valid_cs_mask" not in data.columns:
        return out.astype(_OUT_DTYPE, copy=False)
    m = _clean_bool(data["valid_cs_mask"]).reindex(out.index)
    return out.where(m, np.nan).fillna(0.0).astype(_OUT_DTYPE)


def _sorted_view(data: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Index]:
    """
    返回按 (stock_code, _ts) 排序后的视图与原 index，方便 rolling 严格因果。
    """
    df = data[cols].copy()
    df["_ts"] = _to_ts(df["date"])
    df.sort_values(["stock_code", "_ts"], inplace=True)
    return df, data.index


# ==============================================================================
# 1. 实体力度 (Body Efficiency)
# ==============================================================================
def _body_efficiency(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    efficiency = (close-open) / (high-low)
    - 一字涨停：+1
    - 一字跌停：-1
    - 一字平盘/停牌：0
    """
    need = {"open", "close", "high", "low", "pct_chg"}
    if not need.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    o = _clean_num(data["open"])
    c = _clean_num(data["close"])
    h = _clean_num(data["high"])
    l = _clean_num(data["low"])
    pct = _clean_num(data["pct_chg"])

    hl_range = (h - l)
    body = (c - o)

    eff = body / (hl_range + 1e-9)

    is_flat = hl_range.abs() < 1e-6
    eff = np.where(is_flat & (pct > 0), 1.0, eff)
    eff = np.where(is_flat & (pct < 0), -1.0, eff)
    eff = np.where(is_flat & (pct == 0), 0.0, eff)

    out = pd.Series(eff, index=data.index, dtype="float64").clip(-1.0, 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _apply_valid_mask_to_output(data, out)


body_efficiency = register_factor(
    FactorDefinition(
        name="body_efficiency",
        required_columns={"stock_code", "date", "open", "close", "high", "low", "pct_chg"},
        lookback_days=0,
        calculator=_body_efficiency,
        description="K线实体效率（(收-开)/(高-低)，衡量日内趋势强度；一字板对称修复；valid_cs_mask 无效样本置0；float32）",
    )
)


# ==============================================================================
# 2. 影线主导权 (Shadow Dominance)
# ==============================================================================
def _shadow_dominance(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    dominance = (upper_shadow - lower_shadow) / (high-low)
    值域 [-1,1]
    """
    need = {"open", "close", "high", "low"}
    if not need.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    o = _clean_num(data["open"])
    c = _clean_num(data["close"])
    h = _clean_num(data["high"])
    l = _clean_num(data["low"])

    upper = (h - np.maximum(o, c))
    lower = (np.minimum(o, c) - l)
    rng = (h - l)

    dom = (upper - lower) / (rng + 1e-9)
    dom = np.where(rng.abs() < 1e-6, 0.0, dom)

    out = pd.Series(dom, index=data.index, dtype="float64").clip(-1.0, 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _apply_valid_mask_to_output(data, out)


shadow_dominance = register_factor(
    FactorDefinition(
        name="shadow_dominance",
        required_columns={"stock_code", "date", "open", "close", "high", "low"},
        lookback_days=0,
        calculator=_shadow_dominance,
        description="影线主导权（(上影-下影)/振幅，正值代表抛压，负值代表支撑；valid_cs_mask 无效样本置0；float32）",
    )
)


# ==============================================================================
# 3. 缺口动能 (Gap Power)
# ==============================================================================
def _gap_power(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    gap_power = (open - prev_close) / ATR20_proxy
    ATR20_proxy 用 True Range 的 20日均值，并整体 shift(1) 保证因果。
    """
    need = {"stock_code", "date", "open", "close", "high", "low"}
    if not need.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    cols = ["stock_code", "date", "open", "close", "high", "low"]
    df, orig_idx = _sorted_view(data, cols)
    for col in ["open", "close", "high", "low"]:
        df[col] = _clean_num(df[col], fill=np.nan)

    g = df.groupby("stock_code", sort=False)
    prev_close = g["close"].shift(1)

    hl = (df["high"] - df["low"])
    h_pc = (df["high"] - prev_close).abs()
    l_pc = (df["low"] - prev_close).abs()
    tr = np.maximum(hl, np.maximum(h_pc, l_pc))

    atr20 = tr.groupby(df["stock_code"], sort=False).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    atr20 = atr20.shift(1)

    gap = (df["open"] - prev_close)
    score = gap / (atr20 + 1e-9)

    out = pd.Series(score, index=df.index, dtype="float64").clip(-5.0, 5.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out.reindex(orig_idx)
    return _apply_valid_mask_to_output(data, out)


gap_power = register_factor(
    FactorDefinition(
        name="gap_power",
        required_columns={"stock_code", "date", "open", "close", "high", "low"},
        lookback_days=25,
        calculator=_gap_power,
        description="标准化跳空缺口强度（缺口幅度/历史TR均值，分母严格使用过去数据；valid_cs_mask 无效样本置0；float32）",
    )
)


# ==============================================================================
# 4. 波动压缩 (Range Compression)
# ==============================================================================
def _range_compression(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    compression = (high-low) / mean(high-low, 20d)
    <1：压缩；>1：扩张
    """
    need = {"stock_code", "date", "high", "low"}
    if not need.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    df, orig_idx = _sorted_view(data, ["stock_code", "date", "high", "low"])
    df["high"] = _clean_num(df["high"], fill=np.nan)
    df["low"] = _clean_num(df["low"], fill=np.nan)

    rng = (df["high"] - df["low"])
    avg_rng = rng.groupby(df["stock_code"], sort=False).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    comp = rng / (avg_rng + 1e-9)

    out = pd.Series(comp, index=df.index, dtype="float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out.reindex(orig_idx)
    return _apply_valid_mask_to_output(data, out)


range_compression = register_factor(
    FactorDefinition(
        name="range_compression",
        required_columns={"stock_code", "date", "high", "low"},
        lookback_days=30,
        calculator=_range_compression,
        description="波动压缩比（当日振幅/20日平均振幅）；valid_cs_mask 无效样本置0；float32",
    )
)


__all__ = [
    "body_efficiency",
    "shadow_dominance",
    "gap_power",
    "range_compression",
]
