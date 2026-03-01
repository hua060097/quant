
"""
统计矩因子（偏度、峰度、协同偏度、下行Beta）- 优化版（无开关）

面向 A 股“横截面选股（Top-K）+ LightGBM/LTR”的专业化改造要点：
1) 移除 FrozenMean / FrozenClip：金融分布非平稳，冻结统计量在 regime shift 下易失配。
2) 统一输出使用“按日横截面 Robust Z（median/MAD）并 clip z”，减少极值支配与超参敏感。
3) 时间对齐统一使用 `_ts`（Timestamp），避免 date dtype 不一致导致的静默 NaN / 对齐漂移。
4) coskewness 采用严格因果 rolling centering：
   - market_center = mkt_ret - rolling_mean(mkt_ret, 252)
   - m_sq = market_center^2
   - stock_center = r_i - rolling_mean(r_i, 20)
   - coskewness = rolling_mean(stock_center * m_sq, 20) / (rolling_std(stock_center,20) * rolling_std(m_sq,20))
5) downside_beta 采用“窗口内仅使用市场下跌日”计算协方差/方差（严格因果）：
   beta_down = Cov(r_i, mkt_ret | mkt_ret<0) / Var(mkt_ret | mkt_ret<0)
   用 numpy 在每只股票内部滚动计算，避免 pandas cov/unstack 的列位脆弱性。

注意：
- 若你的训练是“按日 group 的 LTR 排序”，这些特征在同日不同股间存在差异，可直接用于排序。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_CLIP_Z = 6.0
_EPS = 1e-12
_OUT_DTYPE = "float32"

def _vmask(data: pd.DataFrame) -> pd.Series:
    """Return boolean valid cross-section mask aligned to `data.index`."""
    if isinstance(data, pd.DataFrame) and ("valid_cs_mask" in data.columns):
        return _clean_bool(data["valid_cs_mask"]).astype(bool)
    # fallback: all valid
    return pd.Series(True, index=data.index)

# ==============================================================================
# 工具函数
# ==============================================================================
def _to_ts(date_series: pd.Series) -> pd.Series:
    """稳健把 date 转为 Timestamp（支持 'YYYY-MM-DD' / 'YYYYMMDD' / int）"""
    if pd.api.types.is_datetime64_any_dtype(date_series):
        return pd.to_datetime(date_series, errors="coerce")
    ds = date_series.astype(str).str.replace("-", "", regex=False)
    return pd.to_datetime(ds, format="%Y%m%d", errors="coerce")


def _get_market_series(ctx: FactorContext) -> pd.Series:
    """获取市场日收益序列（index=_ts，值=daily_change_pct），去重并排序"""
    m = ctx.require_market().copy()
    m["_ts"] = _to_ts(m["date"])
    m = m.sort_values("_ts")
    s = pd.to_numeric(m.get("daily_change_pct"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    s.index = m["_ts"].values
    if getattr(s.index, "duplicated", None) is not None and s.index.duplicated().any():
        s = s[~s.index.duplicated(keep="first")]
    return s.astype("float64")

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
    """
    按日横截面 robust z（median/MAD），并 clip。
    - 统计 median/MAD 时仅使用 valid_cs_mask==True 的样本
    - valid_cs_mask==False 的行输出置 0（中性）
    """
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

def _skewness_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = data[["stock_code", "date", "pct_chg"]].copy()
    df["_ts"] = _to_ts(df["date"])
    if "valid_cs_mask" in data.columns:
        df["valid_cs_mask"] = _clean_bool(data["valid_cs_mask"]).values
    else:
        df["valid_cs_mask"] = True
    df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df.sort_values(["stock_code", "_ts"], inplace=True)

    raw = df.groupby("stock_code", sort=False)["pct_chg"].transform(
        lambda x: x.rolling(20, min_periods=15).skew()
    )

    raw = raw.where(df["valid_cs_mask"].values, np.nan)

    out = _cs_robust_z(raw, date_keys=df["_ts"], valid_cs_mask=_vmask(data))
    return out.reindex(data.index)


def _kurtosis_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = data[["stock_code", "date", "pct_chg"]].copy()
    df["_ts"] = _to_ts(df["date"])
    if "valid_cs_mask" in data.columns:
        df["valid_cs_mask"] = _clean_bool(data["valid_cs_mask"]).values
    else:
        df["valid_cs_mask"] = True
    df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df.sort_values(["stock_code", "_ts"], inplace=True)

    raw = df.groupby("stock_code", sort=False)["pct_chg"].transform(
        lambda x: x.rolling(20, min_periods=15).kurt()
    )

    raw = raw.where(df["valid_cs_mask"].values, np.nan)

    out = _cs_robust_z(raw, date_keys=df["_ts"], valid_cs_mask=_vmask(data))
    return out.reindex(data.index)


# ==============================================================================
# 2) Coskewness（严格因果 rolling centering，无冻结统计）
# ==============================================================================
# def _coskewness(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
#     if "pct_chg" not in data.columns:
#         return pd.Series(np.nan, index=data.index, dtype="float64")

#     mkt = _get_market_series(ctx)  # index=_ts
#     if mkt.empty:
#         return pd.Series(np.nan, index=data.index, dtype="float64")

#     # 市场中心化：用过去 252 日滚动均值（避免全样本均值引入跨期耦合/弱泄漏）
#     mkt_mu = mkt.rolling(252, min_periods=60).mean()
#     m_center = (mkt - mkt_mu).astype("float64")
#     m_sq = (m_center ** 2).astype("float64")  # index=_ts

#     df = data[["stock_code", "date", "pct_chg"]].copy()
#     df["_ts"] = _to_ts(df["date"])
#     if "valid_cs_mask" in data.columns:
#         df["valid_cs_mask"] = _clean_bool(data["valid_cs_mask"]).values
#     else:
#         df["valid_cs_mask"] = True
#     df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce").replace([np.inf, -np.inf], np.nan)
#     df.loc[~df["valid_cs_mask"], "pct_chg"] = np.nan
#     df["m_sq"] = df["_ts"].map(m_sq)

#     df.sort_values(["stock_code", "_ts"], inplace=True)
#     g = df.groupby("stock_code", sort=False)

#     # 个股中心化：过去 20 日滚动均值
#     i_mu = g["pct_chg"].transform(lambda x: x.rolling(20, min_periods=10).mean())
#     i_center = df["pct_chg"] - i_mu

#     df["_i_center"] = i_center
#     df["_prod"] = (i_center * df["m_sq"]).astype("float64")

#     # 分子：rolling mean(i_center * m_sq)
#     num = g["_prod"].transform(lambda x: x.rolling(20, min_periods=10).mean())

#     # 分母：std(i_center) * std(m_sq)，都在 stock 序列内 rolling（严格因果）
#     i_std = g["_i_center"].transform(lambda x: x.rolling(20, min_periods=10).std())
#     m_std = g["m_sq"].transform(lambda x: x.rolling(20, min_periods=10).std())

#     val = num / (i_std * m_std + 1e-9)
#     val = val.replace([np.inf, -np.inf], np.nan)

#     out = _cs_robust_z(val, date_keys=df["_ts"], valid_cs_mask=_vmask(data))
#     return out.reindex(data.index)


def _coskewness(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    稳健协偏度/尾部暴露因子（推荐版本）
    - 市场尾部：只看下行（downside）偏离的平方 m_tail = min(m_center, 0)^2
    - 中心化：EWMA（严格因果，适应regime漂移）
    - 个股极端：轻度时序clip，避免方差爆炸
    - 不在rolling前使用 valid_cs_mask（避免时序挖空）
    - 输出：_cs_robust_z + valid_cs_mask 做截面稳健标准化
    """
    import numpy as np
    import pandas as pd

    if "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=_OUT_DTYPE)

    mkt = _get_market_series(ctx)  # index = _ts (Timestamp)
    if mkt is None or len(mkt) == 0:
        return pd.Series(np.nan, index=data.index, dtype=_OUT_DTYPE)

    # ---------- 超参（你后面可以纳入搜索，但先给一个稳健默认） ----------
    N = 60        # rolling窗口：尾部类统计不建议太短
    minp = 20     # 最小样本
    hl_m = 63     # 市场EWMA半衰期：约1个季度
    hl_i = 20     # 个股EWMA半衰期：约1个月

    # ---------- 市场尾部构造：EWMA中心化 + downside tail ----------
    m = pd.to_numeric(mkt, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    m_mu = m.ewm(halflife=hl_m, min_periods=minp, adjust=False).mean()
    m_center = m - m_mu
    m_down = np.minimum(m_center, 0.0)
    m_tail = (m_down ** 2).astype("float64")  # index=_ts

    # ---------- 个股数据 ----------
    df = data[["stock_code", "date", "pct_chg"]].copy()
    df["_ts"] = _to_ts(df["date"])

    # pct_chg：保留时序连续性，不要用 valid_cs_mask 先挖空
    r = pd.to_numeric(df["pct_chg"], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    # 温和clip防极端：你可按制度改成 [-0.11, 0.11] 或 [-0.22, 0.22]，但先用统一上限
    r = r.clip(lower=-0.20, upper=0.20)
    df["pct_chg"] = r

    # 映射市场尾部指标到每条样本
    df["m_tail"] = df["_ts"].map(m_tail)

    # 排序保证严格因果 rolling
    df.sort_values(["stock_code", "_ts"], inplace=True)
    g = df.groupby("stock_code", sort=False)

    # ---------- 个股中心化：EWMA mean ----------
    i_mu = g["pct_chg"].transform(lambda x: x.ewm(halflife=hl_i, min_periods=minp, adjust=False).mean())
    i_center = df["pct_chg"] - i_mu
    df["_i_center"] = i_center

    # ---------- 相关系数形式（天然边界[-1,1]，更适合树模型） ----------
    # 分子：E[i_center * m_tail] rolling mean
    df["_prod"] = (df["_i_center"] * df["m_tail"]).astype("float64")
    num = g["_prod"].transform(lambda x: x.rolling(N, min_periods=minp).mean())

    # 分母：std(i_center) * std(m_tail) rolling std（都在 stock 序列上做rolling，严格因果）
    i_std = g["_i_center"].transform(lambda x: x.rolling(N, min_periods=minp).std())
    m_std = g["m_tail"].transform(lambda x: x.rolling(N, min_periods=minp).std())

    val = num / (i_std * m_std + 1e-12)
    val = val.replace([np.inf, -np.inf], np.nan)

    # ---------- 截面稳健标准化 + 掩码隔离（只在这里用 valid_cs_mask） ----------
    out = _cs_robust_z(val, date_keys=df["_ts"], valid_cs_mask=_vmask(data))
    return out.reindex(data.index).astype(_OUT_DTYPE)


# ==============================================================================
# 3) Downside Beta（窗口内仅使用市场下跌日，严格因果）
# ==============================================================================
def _downside_beta(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    mkt = _get_market_series(ctx)
    if mkt.empty:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = data[["stock_code", "date", "pct_chg"]].copy()
    df["_ts"] = _to_ts(df["date"])
    if "valid_cs_mask" in data.columns:
        df["valid_cs_mask"] = _clean_bool(data["valid_cs_mask"]).values
    else:
        df["valid_cs_mask"] = True
    df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["mkt_ret"] = df["_ts"].map(mkt)

    df.sort_values(["stock_code", "_ts"], inplace=True)

    window = 30
    min_periods = 10

    out = np.full(len(df), np.nan, dtype=np.float64)

    codes = df["stock_code"].values
    r = df["pct_chg"].values.astype(np.float64)
    m = df["mkt_ret"].values.astype(np.float64)

    n = len(df)
    start = 0
    while start < n:
        code = codes[start]
        end = start + 1
        while end < n and codes[end] == code:
            end += 1

        rr = r[start:end]
        mm = m[start:end]
        T = end - start

        for t in range(T):
            l = 0 if t < window - 1 else (t - window + 1)

            rr_w = rr[l:t+1]
            mm_w = mm[l:t+1]

            mask = np.isfinite(rr_w) & np.isfinite(mm_w) & (mm_w < 0)
            if mask.sum() < min_periods:
                continue

            x = mm_w[mask]
            y = rr_w[mask]

            x_mean = float(x.mean())
            y_mean = float(y.mean())
            x_var = float(((x - x_mean) ** 2).mean())
            if x_var < 1e-12:
                continue
            cov = float(((x - x_mean) * (y - y_mean)).mean())
            out[start + t] = cov / x_var

        start = end

    beta = pd.Series(out, index=df.index, dtype="float64").replace([np.inf, -np.inf], np.nan)
    out_cs = _cs_robust_z(beta, date_keys=df["_ts"], valid_cs_mask=_vmask(data))
    return out_cs.reindex(data.index)


# ==============================================================================
# 注册因子（保持原名称；不再提供“对照/开关”差异，coskewness_raw/gated 兼容同一实现）
# ==============================================================================
skewness_20d = register_factor(
    FactorDefinition(
        name="skewness_20d",
        required_columns={"stock_code", "date", "pct_chg"},
        lookback_days=30,
        calculator=_skewness_20d,
        description="20日收益率偏度（按日截面robust z）",
    )
)

kurtosis_20d = register_factor(
    FactorDefinition(
        name="kurtosis_20d",
        required_columns={"stock_code", "date", "pct_chg"},
        lookback_days=30,
        calculator=_kurtosis_20d,
        description="20日收益率峰度（按日截面robust z）",
    )
)

coskewness = register_factor(
    FactorDefinition(
        name="coskewness",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_market_data=True,
        lookback_days=300,
        calculator=_coskewness,
        description="协同偏度：rolling中心化市场平方偏离与个股中心化收益的协同（按日截面robust z）",
    )
)

coskewness_raw = register_factor(
    FactorDefinition(
        name="coskewness_raw",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_market_data=True,
        lookback_days=300,
        calculator=_coskewness,
        description="（兼容）协同偏度：同 coskewness",
    )
)

coskewness_gated = register_factor(
    FactorDefinition(
        name="coskewness_gated",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_market_data=True,
        lookback_days=300,
        calculator=_coskewness,
        description="（兼容）协同偏度：同 coskewness",
    )
)

downside_beta = register_factor(
    FactorDefinition(
        name="downside_beta",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_market_data=True,
        lookback_days=60,
        calculator=_downside_beta,
        description="下行Beta：窗口内仅用市场下跌日计算（按日截面robust z）",
    )
)





__all__ = [
    "skewness_20d",
    "kurtosis_20d",
    "coskewness",
    "coskewness_raw",
    "coskewness_gated",
    "downside_beta",
]
