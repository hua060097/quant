"""
Wavelet / Regime Interaction Factors - Master V10
目标：针对 3 日 top 2% 爆发预测的极度不平衡样本 (2.26%)
整合说明：
1. 采用 V9 的稳健 safe_div 和带 mask 的 cs_robust_z。
2. 修复 V9 中丢失的 Beta 计算逻辑（矢量化且索引对齐）。
3. 增强缓存签名（Cache Signature），包含列名与样本数，适配 Walk-forward。
4. 恢复 32 日窗口的高分辨率小波分解。
5. 完整保留 7 个因子定义。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_CLIP = 8.0
_EPS = 1e-12
_REQ_BASE = {
    "stock_code", "date", "pct_chg",
    "is_limit_up", "is_limit_down", "is_zhaban",
    "valid_cs_mask"
}

# ==============================================================================
# 核心工具链 (融合最优逻辑)
# ==============================================================================

def _to_ts(s):
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def _num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _safe_div(a, b):
    # ChatGPT 建议的最优写法，防止分母为 0 导致的溢出
    return _num(a) / (_num(b) + _EPS)

def _cs_robust_z(x, date_keys, valid_mask):
    """横截面 Robust Z，仅对有效样本统计，无效样本归 0"""
    s = _num(x).astype("float64")
    mask = valid_mask.astype(bool)
    s_valid = s.where(mask)
    
    g = s_valid.groupby(date_keys, sort=False)
    med = g.transform("median")
    mad = g.transform(lambda v: np.nanmedian(np.abs(v - np.nanmedian(v))))
    
    z = (s - med) / (1.4826 * mad + _EPS)
    z = z.clip(-_CLIP, _CLIP)
    return z.where(mask, 0.0)

def _rolling_robust_z1d(s, window):
    x = _num(s)
    roll = x.rolling(window, min_periods=max(5, window//2))
    med = roll.median()
    mad = roll.apply(lambda v: np.median(np.abs(v - np.median(v))), raw=True)
    return (x - med) / (1.4826 * mad + _EPS)

# ==============================================================================
# 小波多尺度 (32日窗口高分辨率)
# ==============================================================================

def _haar_wavelet_features(w: np.ndarray):
    """32日窗口 Haar 小波特征提取"""
    x = np.asarray(w, float)
    x = x[np.isfinite(x)]
    if len(x) < 32: return [np.nan]*5
    
    x = x - np.mean(x)
    # 第一层 (D1: 最快突变)
    hi1 = (x[0::2] - x[1::2]) / np.sqrt(2.0)
    lo1 = (x[0::2] + x[1::2]) / np.sqrt(2.0)
    # 第二层 (D2: 次快波动)
    hi2 = (lo1[0::2] - lo1[1::2]) / np.sqrt(2.0)
    lo2 = (lo1[0::2] + lo1[1::2]) / np.sqrt(2.0)
    
    e_d1, e_d2 = np.sum(hi1**2), np.sum(hi2**2)
    e_approx = np.sum(lo2**2)
    total_e = e_d1 + e_d2 + e_approx
    
    if total_e < 1e-15: return [np.nan]*5
    
    # 特征工程
    hf_ratio = (e_d1 + e_d2) / total_e
    lf_ratio = e_approx / total_e
    burst = np.abs(hi1[-1]) # 瞬时突变 proxy
    
    return [hf_ratio, lf_ratio, total_e, burst, e_approx]

def _rolling_wavelet_32(s: pd.Series):
    x = _num(s).values
    n = len(x)
    res = np.full((n, 5), np.nan)
    for i in range(31, n):
        res[i] = _haar_wavelet_features(x[i-31:i+1])
    return pd.DataFrame(res, columns=["hf","lf","total","burst","approx"], index=s.index)

# ==============================================================================
# 因子辅助器 (修正缓存与 Beta)
# ==============================================================================

class _WaveletV10Master:
    def __init__(self, df: pd.DataFrame, ctx: FactorContext):
        self.ctx = ctx
        self.original_index = df.index
        # 缓存签名：必须包含数据长度、时间边界、列名、个股数，防止回测污染
        ts = _to_ts(df["date"])
        self._sig = (len(df), ts.min(), ts.max(), tuple(sorted(df.columns)), df["stock_code"].nunique())
        
        df = df.copy()
        df["_ts"] = ts
        df["vmask"] = df.get("valid_cs_mask", True).astype(bool)
        df.sort_values(["stock_code", "_ts"], inplace=True)
        self.df = df
        self._precompute()

    def _precompute(self):
        df = self.df
        mask = df["vmask"]
        
        # 1. 市场情绪门控 (保持 V9 完整 heat 逻辑)
        mkt_g = df[mask].groupby("_ts")
        cnt = mkt_g.size()
        heat = _safe_div(mkt_g["is_limit_up"].sum() - 1.2*mkt_g["is_limit_down"].sum() - 0.6*mkt_g["is_zhaban"].sum(), cnt)
        
        daily = heat.to_frame("heat")
        wv = _rolling_wavelet_32(daily["heat"])
        daily = pd.concat([daily, wv], axis=1)
        
        for c in ["hf", "lf", "total", "burst"]:
            daily[f"{c}_z"] = _rolling_robust_z1d(daily[c], 20)
        daily["total_delta_z"] = _rolling_robust_z1d(daily["total"].diff(3), 20)
        self.daily = daily.fillna(0.0)

        # 2. 个股 Alpha
        gb = df.groupby("stock_code", sort=False)
        df["pct_chg"] = _num(df["pct_chg"])
        df["mom5_z"] = _cs_robust_z(gb["pct_chg"].transform(lambda s: s.rolling(5,3).sum()), df["_ts"], mask)
        df["rev3_z"] = _cs_robust_z(gb["pct_chg"].transform(lambda s: -s.rolling(3,2).sum()), df["_ts"], mask)
        df["vol20_z"] = _cs_robust_z(gb["pct_chg"].transform(lambda s: s.rolling(20,10).std()), df["_ts"], mask)
        
        to = _num(df.get("turnover_rate", _safe_div(df.get("volume",0), df.get("circ_mv",1))))
        df["turnover_z"] = _cs_robust_z(to, df["_ts"], mask)

        # 3. 恢复 Beta 计算 (修复 V9 占位符问题)
        df["beta60_z"] = 0.0
        if getattr(self.ctx, "market_data", None) is not None:
            m_ret = self.ctx.require_market().copy()
            m_ret["_ts"] = _to_ts(m_ret["date"])
            m_map = m_ret.set_index("_ts")["daily_change_pct"]
            df["m_ret"] = df["_ts"].map(m_map)
            
            # 严格索引对齐的 Beta 计算
            def _calc_beta(g):
                cov = g["pct_chg"].rolling(60, 30).cov(g["m_ret"])
                var = g["m_ret"].rolling(60, 30).var()
                return cov / (var + 1e-9)
            
            beta = gb.apply(_calc_beta).reset_index(level=0, drop=True)
            df["beta60_z"] = _cs_robust_z(beta.reindex(df.index), df["_ts"], mask)
        
        self.df = df

    def factor(self, name):
        df = self.df
        gate_map = {
            "wv_chop_gate_reversal": ("hf_z", "rev3_z"),
            "wv_trend_gate_momentum": ("lf_z", "mom5_z"),
            "wv_burst_gate_liquidity": ("burst_z", "turnover_z"),
            "wv_burst_gate_lowvol_defensive": ("burst_z", "-vol20_z"),
            "wv_energy_gate_lowvol": ("total_z", "-vol20_z"),
            "wv_energy_delta_gate_momentum": ("total_delta_z", "mom5_z"),
            "wv_trend_gate_beta": ("lf_z", "beta60_z")
        }
        
        if name not in gate_map: return pd.Series(0.0, index=self.original_index)
        
        g_col, a_col = gate_map[name]
        gate_vals = self.daily[g_col].reindex(df["_ts"]).values
        
        if a_col.startswith("-"):
            alpha_vals = -df[a_col[1:]]
        else:
            alpha_vals = df[a_col]
            
        res = gate_vals * alpha_vals
        return pd.Series(res, index=df.index).clip(-_CLIP, _CLIP).reindex(self.original_index).fillna(0.0)

# ==============================================================================
# Getter & Registry
# ==============================================================================

def _get_helper(data, ctx):
    key = "_wv_v10_master"
    ts = _to_ts(data["date"])
    # 极度安全的签名，确保回测准确性
    sig = (len(data), ts.min(), ts.max(), tuple(sorted(data.columns)), data["stock_code"].nunique())
    
    h = getattr(ctx, "extra", {}).get(key)
    if h is None or getattr(h, "_sig", None) != sig:
        h = _WaveletV10Master(data, ctx)
        if hasattr(ctx, "extra"): ctx.extra[key] = h
    return h

def _c(name):
    return lambda d, c: _get_helper(d, c).factor(name)


# ============================================
# Wavelet Gate 系列因子注册（无 for 循环版本）
# ============================================

wv_chop_gate_reversal = register_factor(FactorDefinition(
    name="wv_chop_gate_reversal",
    required_columns=_REQ_BASE,
    requires_market_data=False,
    lookback_days=120,
    calculator=_c("wv_chop_gate_reversal"),
    description="Wavelet: 震荡(高频占比)门控 × 反转因子"
))

wv_trend_gate_momentum = register_factor(FactorDefinition(
    name="wv_trend_gate_momentum",
    required_columns=_REQ_BASE,
    requires_market_data=False,
    lookback_days=140,
    calculator=_c("wv_trend_gate_momentum"),
    description="Wavelet: 趋势(低频占比)门控 × 动量因子"
))

wv_burst_gate_liquidity = register_factor(FactorDefinition(
    name="wv_burst_gate_liquidity",
    required_columns=_REQ_BASE,
    requires_market_data=False,
    lookback_days=100,
    calculator=_c("wv_burst_gate_liquidity"),
    description="Wavelet: 波动爆发门控 × 流动性因子"
))

wv_burst_gate_lowvol_defensive = register_factor(FactorDefinition(
    name="wv_burst_gate_lowvol_defensive",
    required_columns=_REQ_BASE,
    requires_market_data=False,
    lookback_days=100,
    calculator=_c("wv_burst_gate_lowvol_defensive"),
    description="Wavelet: 波动爆发门控 × 低波防御因子"
))

wv_energy_gate_lowvol = register_factor(FactorDefinition(
    name="wv_energy_gate_lowvol",
    required_columns=_REQ_BASE,
    requires_market_data=False,
    lookback_days=120,
    calculator=_c("wv_energy_gate_lowvol"),
    description="Wavelet: 能量强度门控 × 低波因子"
))

wv_energy_delta_gate_momentum = register_factor(FactorDefinition(
    name="wv_energy_delta_gate_momentum",
    required_columns=_REQ_BASE,
    requires_market_data=False,
    lookback_days=120,
    calculator=_c("wv_energy_delta_gate_momentum"),
    description="Wavelet: 能量变化率门控 × 动量因子"
))

wv_trend_gate_beta = register_factor(FactorDefinition(
    name="wv_trend_gate_beta",
    required_columns=_REQ_BASE,
    requires_market_data=True,
    lookback_days=140,
    calculator=_c("wv_trend_gate_beta"),
    description="Wavelet: 趋势(低频占比)门控 × Beta暴露（需要 market_data）"
))