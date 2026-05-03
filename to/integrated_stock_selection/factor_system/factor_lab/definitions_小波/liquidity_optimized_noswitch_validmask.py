"""
Liquidity 因子 - valid_cs_mask 适配版（输出 float32）

要点：
- 时序纯净：按 stock_code,_ts 排序后 rolling/shift，严格因果。
- 横截面统一：输出按日横截面 robust z（median/MAD）+ clip + fillna(0)。
- valid_cs_mask：median/MAD 仅用 mask=True 样本统计；mask=False 输出置 0（中性），避免污染分布。
- 输出统一 float32（内部计算 float64）。
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
        ts = pd.to_datetime(date_series, errors="coerce")
    else:
        ds = date_series.astype(str).str.replace("-", "", regex=False)
        ts = pd.to_datetime(ds, format="%Y%m%d", errors="coerce")
    return ts.dt.normalize()


def _num64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def _clean_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(1.0) > 0.5


def _cs_robust_z(
    x: pd.Series,
    date_keys: pd.Series,
    *,
    valid_cs_mask: pd.Series | None = None,
    clip_z: float = _CLIP_Z,
) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")

    if valid_cs_mask is not None:
        m = _clean_bool(valid_cs_mask)
        s_stat = s.where(m, np.nan)
    else:
        m = None
        s_stat = s

    g = s_stat.groupby(date_keys, sort=False)
    med = g.transform("median")
    mad = g.transform(lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))))
    z = (s - med) / (1.4826 * mad + _EPS)
    z = z.clip(-clip_z, clip_z)

    if m is not None:
        z = z.where(m, np.nan)

    return z.fillna(0.0).astype(_OUT_DTYPE)


class _LiqHelper:
    def __init__(self, data: pd.DataFrame):
        self.original_index = data.index
        df = data.copy()
        df["_ts"] = _to_ts(df["date"])
        df["stock_code"] = df["stock_code"].astype(str)

        if "valid_cs_mask" in df.columns:
            df["valid_cs_mask"] = _clean_bool(df["valid_cs_mask"])
        else:
            df["valid_cs_mask"] = True

        for c in ["pct_chg", "amount", "turnover_rate", "high", "low", "close"]:
            if c in df.columns:
                df[c] = _num64(df[c])

        df.sort_values(["stock_code", "_ts"], inplace=True)
        self.df = df


def _amihud_illiquidity(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if "amount" not in data.columns or "pct_chg" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    h = _LiqHelper(data)
    df = h.df

    daily = df["pct_chg"].abs() / (df["amount"] + _EPS)
    roll = daily.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=10).mean())
    logv = np.log1p(roll * 1e9)

    out = _cs_robust_z(logv, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(h.original_index)
    return out.astype(_OUT_DTYPE)




def _turnover_stability(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if "turnover_rate" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    h = _LiqHelper(data)
    df = h.df

    tr = df["turnover_rate"]
    g = tr.groupby(df["stock_code"], sort=False)
    mean20 = g.transform(lambda s: s.rolling(20, min_periods=10).mean())
    std20 = g.transform(lambda s: s.rolling(20, min_periods=10).std())

    cv = std20 / (mean20 + _EPS)
    stability = 1.0 / (cv + 0.1)

    out = _cs_robust_z(stability, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(h.original_index)
    return out.astype(_OUT_DTYPE)


# def _abnormal_turnover(data: pd.DataFrame, _: FactorContext) -> pd.Series:
#     if "turnover_rate" not in data.columns:
#         return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

#     h = _LiqHelper(data)
#     df = h.df

#     tr = df["turnover_rate"]
#     base60 = tr.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(60, min_periods=20).mean())
#     ratio = tr / (base60 + _EPS)
#     ratio = ratio.replace([np.inf, -np.inf], np.nan).clip(0.0, 30.0)

#     out = _cs_robust_z(ratio, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(h.original_index)
#     return out.astype(_OUT_DTYPE)

def _abnormal_turnover(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    稳健异常换手（推荐最终版）
    目标：
      - 避免极端换手日导致特征重要性“断层第一”与不稳定；
      - 保留“相对自身历史的异常流动性”信号；
      - 与你的体系兼容：时间序列处理在 stock 内完成，最后用 _cs_robust_z 做截面稳健标准化 + mask 隔离。
    """
    if "turnover_rate" not in data.columns:
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    h = _LiqHelper(data)
    df = h.df  # 已排序: ["stock_code","_ts"]

    tr = pd.to_numeric(df["turnover_rate"], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    # <=0 通常代表缺失/停牌/异常口径：不在 rolling 前做 mask，只把它当缺失
    tr = tr.where(tr > 0, np.nan)

    # 1) 压缩长尾（换手天然长尾）
    x = np.log1p(tr)

    N, minp = 60, 20
    g = x.groupby(df["stock_code"], sort=False)

    # 2) rolling median
    med = g.transform(lambda s: s.rolling(N, min_periods=minp).quantile(0.5))

    # 3) rolling MAD（稳健尺度）
    def _roll_mad(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        m = np.nanmedian(arr)
        return float(np.nanmedian(np.abs(arr - m)))

    mad = g.transform(lambda s: s.rolling(N, min_periods=minp).apply(_roll_mad, raw=True))

    # 4) 个股时间序列稳健 z
    ts_z = (x - med) / (1.4826 * mad + _EPS)
    ts_z = ts_z.replace([np.inf, -np.inf], np.nan).clip(-8.0, 8.0)

    # 5) 截面稳健标准化 + mask 隔离（你体系的关键）
    out = _cs_robust_z(ts_z, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(h.original_index)

    return out.fillna(0.0).astype(_OUT_DTYPE)


def _money_flow_pressure(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    need = {"high", "low", "close", "turnover_rate"}
    if not need.issubset(set(data.columns)):
        return pd.Series(0.0, index=data.index, dtype=_OUT_DTYPE)

    h = _LiqHelper(data)
    df = h.df

    high_low = df["high"] - df["low"]
    mfm = (2 * df["close"] - df["high"] - df["low"]) / (high_low + _EPS)
    mfm = pd.Series(np.where(high_low.abs() < 1e-6, 0.0, mfm), index=df.index, dtype="float64")

    pressure = mfm * df["turnover_rate"]
    flow5 = pressure.groupby(df["stock_code"], sort=False).transform(lambda s: s.rolling(5, min_periods=1).mean())

    out = _cs_robust_z(flow5, df["_ts"], valid_cs_mask=df["valid_cs_mask"]).reindex(h.original_index)
    return out.astype(_OUT_DTYPE)


amihud_illiquidity = register_factor(FactorDefinition(
    name="amihud_illiquidity",
    required_columns={"stock_code", "date", "amount", "pct_chg"},
    lookback_days=30,
    calculator=_amihud_illiquidity,
    description="Amihud非流动性：|ret|/amount 20d均值，对数化后按日robust z（valid_cs_mask；float32）",
))

turnover_stability = register_factor(FactorDefinition(
    name="turnover_stability",
    required_columns={"stock_code", "date", "turnover_rate"},
    lookback_days=30,
    calculator=_turnover_stability,
    description="换手稳定性：1/(CV(turnover)+0.1)，按日robust z（valid_cs_mask；float32）",
))

abnormal_turnover = register_factor(FactorDefinition(
    name="abnormal_turnover",
    required_columns={"stock_code", "date", "turnover_rate"},
    lookback_days=80,
    calculator=_abnormal_turnover,
    description="异常换手：turnover/60d mean(turnover)，clip[0,30]后按日robust z（valid_cs_mask；float32）",
))

money_flow_pressure = register_factor(FactorDefinition(
    name="money_flow_pressure",
    required_columns={"stock_code", "date", "high", "low", "close", "turnover_rate"},
    lookback_days=10,
    calculator=_money_flow_pressure,
    description="资金流压力：MFM*turnover 的 5d均值，按日robust z（valid_cs_mask；float32）",
))

__all__ = [
    "turnover_stability",
    "abnormal_turnover",
    "money_flow_pressure",
    "amihud_illiquidity",
]
