"""
行为金融因子（Behavioral Finance）- 优化版（无开关，支持 valid_cs_mask）

在你现有“时序纯净 + 横截面统一”的因子体系中，本文件增加了对 valid_cs_mask 的支持：
- valid_cs_mask=True 的样本参与当日横截面 median/MAD 统计；
- valid_cs_mask=False（例如 ST/新股期）不会污染横截面分布；
- 输出仍对全体样本给值：无效样本会在最终 fillna(0) 后趋于“中性”，便于你在训练/打分阶段用 mask 过滤。

原文件的核心思想（52周高低点锚定 + 趋势加速度）是合理的，但存在几类“量化工程/可复现性”隐患：
1) rolling/diff 未强制按时间排序：如果输入 data 未按 date 排序，会出现非因果或漂移结果。
2) 数值清洗不足：inf/字符串/缺失会在 rolling 中传播，导致隐性 NaN 与训练不稳定。
3) price_acceleration 用“价格差分”尺度依赖强：不同价位股票不可比，且对复权/跳变更敏感。
4) 输出未做按日横截面稳健标准化：Top-K/LTR 下更容易被极端值/分布漂移影响。

本优化版的统一策略：
- 所有时序计算严格因果：先按 (stock_code, _ts) 排序，再 rolling/diff，再 reindex 回原 index。
- 价格类量改为 log-price 的差分（近似收益）以获得尺度不变性。
- 输出统一做“按日横截面 Robust Z（median/MAD）并 clip z”，并支持 valid_cs_mask。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor


import numpy as np
import pandas as pd

_EPS = 1e-9
_CLIP_Z = 6.0
_OUT_DTYPE = "float32"


# ==============================================================================
def _to_ts(date_series: pd.Series) -> pd.Series:
    """稳健把 date 转为 Timestamp（支持 'YYYY-MM-DD' / 'YYYYMMDD' / int）"""
    if pd.api.types.is_datetime64_any_dtype(date_series):
        return pd.to_datetime(date_series, errors="coerce").dt.normalize()
    ds = date_series.astype(str).str.replace("-", "", regex=False)
    return pd.to_datetime(ds, format="%Y%m%d", errors="coerce").dt.normalize()


def _clean_num(s: pd.Series, fill: float = np.nan) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s2.fillna(fill).astype("float64")


def _clean_bool(s: pd.Series) -> pd.Series:
    """把 valid_cs_mask 这类列转为 bool（缺失视为 True，以免意外把所有样本排除）。"""
    if s.dtype == bool:
        return s
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(1.0) > 0.5


def _cs_robust_z(
    x: pd.Series,
    date_keys: pd.Series,
    *,
    valid_cs_mask: pd.Series | None = None,
    mad_scale: float = 1.4826,
    eps: float = _EPS,
    clip_z: float = _CLIP_Z,
    fillna: float = 0.0,
) -> pd.Series:
    """
    按日横截面 Robust Z：median/MAD 标准化并 clip。

    valid_cs_mask 的专业用法（与你的含义一致）：
    - mask=True：参与当日 median/MAD 统计，且输出保留
    - mask=False：不参与统计；最终输出置为 fillna（默认 0），避免污染训练/打分
    """
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")

    if valid_cs_mask is not None:
        m = _clean_bool(valid_cs_mask)
        s_stat = s.where(m, np.nan)  # 仅统计用：无效样本不参与当日分布
    else:
        m = None
        s_stat = s

    g = s_stat.groupby(date_keys, sort=False)
    med = g.transform("median")
    mad = g.transform(lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))))

    z = (s - med) / (mad_scale * mad + eps)
    z = z.replace([np.inf, -np.inf], np.nan).clip(-clip_z, clip_z)

    # 关键：无效样本最终输出置为 fillna（通常 0）
    if m is not None:
        z = z.where(m, np.nan)

    return z.fillna(fillna).astype(_OUT_DTYPE)


def _sorted_view(data: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Index]:
    """按 (stock_code, _ts) 排序视图 + 原 index，用于严格因果 rolling。"""
    df = data[cols].copy()
    df["_ts"] = _to_ts(df["date"])
    df["stock_code"] = df["stock_code"].astype(str)
    df.sort_values(["stock_code", "_ts"], inplace=True)
    return df, data.index


# ==============================================================================
# 1) 距离N日高点 (Anchoring Bias - High)
# ==============================================================================
def _distance_from_high_N(data: pd.DataFrame, _: FactorContext, N: int = 250, minp: int = 60) -> pd.Series:
    """
    距离 N 日高点（锚定偏差：High）
    raw = close / rolling_max(close, N) - 1
    说明：
    - rolling 使用个股时间序列（严格因果），不在 rolling 前对 valid_cs_mask 挖空。
    - valid_cs_mask 仅用于横截面 robust z 的统计隔离（ST/新股/成交量0等不污染当日分布）。
    """
    need = {"stock_code", "date", "close"}
    if not need.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df, orig_idx = _sorted_view(data, cols)

    # 清洗 close，并写回 df，确保 rolling 使用数值列
    df["close"] = _clean_num(df["close"], fill=np.nan)

    g = df.groupby("stock_code", sort=False)
    highN = g["close"].transform(lambda x: x.rolling(N, min_periods=minp).max())
    raw = (df["close"] / (highN + 1e-9)) - 1.0

    vmask = df["valid_cs_mask"] if "valid_cs_mask" in df.columns else None
    out = _cs_robust_z(raw, date_keys=df["_ts"], valid_cs_mask=vmask)
    return out.reindex(orig_idx)

def _distance_from_low_N(data: pd.DataFrame, _: FactorContext, N: int = 250, minp: int = 60) -> pd.Series:
    """
    距离 N 日低点（锚定偏差：Low）
    raw = close / rolling_min(close, N) - 1
    说明同 _distance_from_high_N。
    """
    need = {"stock_code", "date", "close"}
    if not need.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df, orig_idx = _sorted_view(data, cols)

    df["close"] = _clean_num(df["close"], fill=np.nan)

    g = df.groupby("stock_code", sort=False)
    lowN = g["close"].transform(lambda x: x.rolling(N, min_periods=minp).min())
    raw = (df["close"] / (lowN + 1e-9)) - 1.0

    vmask = df["valid_cs_mask"] if "valid_cs_mask" in df.columns else None
    out = _cs_robust_z(raw, date_keys=df["_ts"], valid_cs_mask=vmask)
    return out.reindex(orig_idx)

def _price_acceleration(data: pd.DataFrame, _: FactorContext, L: int = 5) -> pd.Series:
    """
    价格加速度（二阶差分，log-price）
      v_t = log(close)_t - log(close)_{t-L}
      a_t = v_t - v_{t-L}  = diff(L).diff(L)

    设计要点：
    - rolling/diff 不在 valid_cs_mask 挖空后进行（保持时间序列连续性）。
    - valid_cs_mask 仅用于最终按日横截面 robust z 的统计隔离。
    """
    need = {"stock_code", "date", "close"}
    if not need.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    cols = ["stock_code", "date", "close"]
    if "valid_cs_mask" in data.columns:
        cols.append("valid_cs_mask")
    df, orig_idx = _sorted_view(data, cols)

    df["close"] = _clean_num(df["close"], fill=np.nan)
    df["_lp"] = np.log(np.clip(df["close"].astype("float64"), 1e-9, np.inf))

    g = df.groupby("stock_code", sort=False)
    a = g["_lp"].transform(lambda x: x.diff(L).diff(L))

    vmask = df["valid_cs_mask"] if "valid_cs_mask" in df.columns else None
    out = _cs_robust_z(a, date_keys=df["_ts"], valid_cs_mask=vmask)
    return out.reindex(orig_idx)


distance_from_52w_high = register_factor(
    FactorDefinition(
        name="distance_from_52w_high",
        required_columns={"stock_code", "date", "close"},
        lookback_days=260,
        calculator=_distance_from_high_N,
        description="距离52周高点幅度（按日截面robust z，动量锚点；支持 valid_cs_mask）",
    )
)





distance_from_52w_low = register_factor(
    FactorDefinition(
        name="distance_from_52w_low",
        required_columns={"stock_code", "date", "close"},
        lookback_days=260,
        calculator=_distance_from_low_N,
        description="距离52周低点幅度（按日截面robust z，安全边际锚点；支持 valid_cs_mask）",
    )
)



price_acceleration = register_factor(
    FactorDefinition(
        name="price_acceleration",
        required_columns={"stock_code", "date", "close"},
        lookback_days=20,
        calculator=_price_acceleration,
        description="价格加速度（二阶差分，log-price 口径；按日截面robust z；支持 valid_cs_mask）",
    )
)


__all__ = [
    "distance_from_52w_high",
    "distance_from_52w_low",
    "price_acceleration",
]
