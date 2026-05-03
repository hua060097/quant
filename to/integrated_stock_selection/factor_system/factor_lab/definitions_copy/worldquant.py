"""
WorldQuant 101 Alphas 精选版（深度优化）。
向量化复刻最有效的几个 Alpha。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# 辅助函数：标准化排名 (0~1)
def _rank(df):
    return df.rank(axis=1, pct=True)

# 辅助函数：滚动相关性
def _ts_corr(x, y, window):
    return x.rolling(window).corr(y)

# 辅助函数：滚动排名
def _ts_rank(x, window):
    return x.rolling(window).apply(lambda s: s.rank(pct=True).iloc[-1])

# ==============================================================================
# Alpha 001: Rank(Ts_ArgMax(SignedPower(Returns < 0 ? Std(Returns, 20) : Close, 2.), 5)) - 0.5
# 简化版：Rank(Close) 与 Rank(Volume) 的相关性反转？
# 不，WQ原文公式极其复杂。
# 工业界通用的简化版 Alpha 001 是：Rank(Ts_ArgMax(Power(If(Ret<0, Std(Ret,20), Close), 2), 5))
# 这个太复杂且过拟合。
# 我们通常使用的是另一个版本的 Alpha 001 (Kakushadze论文版):
# Rank(Ts_ArgMax(SignedPower(((Returns < 0) ? Stddev(Returns, 20) : Close), 2.), 5))
# 
# 但你之前的代码写的是 Rank(Close) vs Rank(Vol) 的相关性，这其实是 Alpha 006 的变体。
# 我这里给你一个最经典的【量价背离】Alpha，这也是你之前代码想表达的意思。
# ==============================================================================
def _alpha_001_pv_correlation(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    Alpha 001 (PV Correlation)
    
    逻辑：
    -Corr(Rank(Close), Rank(Volume), 5)
    
    含义：
    量价背离因子。
    - 如果价格涨且量涨 (Corr>0) -> 负号后 <0 -> 并不是最好的？
    - 其实 WQ 很多 Alpha 是反转逻辑。
    - 这里我们计算原始相关性，让 LightGBM 自己去学正负关系。
    """
    if 'close' not in data.columns or 'volume' not in data.columns:
        return pd.Series(0.0, index=data.index)
        
    # 由于 Rank 是横截面操作，计算量巨大。
    # 我们可以简化为 TS_Corr(Close, Volume)
    # 因为在短周期内，Rank的相关性 与 原始值的相关性 符号通常一致。
    
    grouped = data.groupby('stock_code', sort=False)
    # 计算量价的滚动相关性 (5日)
    corr = grouped[['close', 'volume']].rolling(10).corr().unstack().iloc[:, 1]
    
    # 填充NaN
    return corr.fillna(0.0).reset_index(level=0, drop=True).astype('float64')

alpha_001 = register_factor(
    FactorDefinition(
        name='alpha_001',
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=20,
        calculator=_alpha_001_pv_correlation,
        description='量价相关性（10日滚动相关系数）',
    )
)

# ==============================================================================
# Alpha 002: -1 * Correlation(Rank(Delta(Log(Volume), 2)), Rank(((Close - Open) / Open)), 6)
# 简化版：成交量变化 与 日内涨幅 的相关性
# ==============================================================================
def _alpha_002_volatility_reversal(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    Alpha 002 (Volatility Reversal)
    
    逻辑：
    -1 * Correlation(Delta(Log(Volume), 2), (Close-Open)/Open, 6)
    
    含义：
    如果你在放量的时候涨（买盘竭尽），或者缩量的时候跌。
    负相关通常意味着：放量上涨是不好的（诱多），放量下跌是好的（恐慌盘出清）。
    """
    grouped = data.groupby('stock_code', sort=False)
    
    # 1. Delta(Log(Volume), 2)
    # 避免log(0)
    log_vol = np.log1p(data['volume'])
    delta_vol = log_vol.groupby(data['stock_code'], sort=False).diff(2)
    
    # 2. Intraday Return
    intra_ret = (data['close'] - data['open']) / (data['open'] + 1e-9)
    
    # 3. Correlation (6日)
    # 构造临时DF
    temp = pd.DataFrame({'dv': delta_vol, 'ir': intra_ret, 'stock_code': data['stock_code']}, index=data.index)
    corr = temp.groupby('stock_code', sort=False)[['dv', 'ir']].rolling(6).corr().unstack().iloc[:, 1]
    
    return (-1 * corr).fillna(0.0).reset_index(level=0, drop=True).astype('float64')

alpha_002 = register_factor(
    FactorDefinition(
        name='alpha_002',
        required_columns={'stock_code', 'date', 'close', 'open', 'volume'},
        lookback_days=10,
        calculator=_alpha_002_volatility_reversal,
        description='成交量变化与日内涨幅的负相关性（反转因子）',
    )
)

# ==============================================================================
# Alpha 017: -1 * Rank(Ts_Rank(Close, 10)) * Rank(Ts_Rank(Ts_ArgMax(High, 30), 5)) ...
# 简化版：Rank(Twisted_Close) vs Rank(Vol)
# 这是一个非常复杂的公式。
# 我们用一个经典的“量价背离”替代版：Price_Rank - Volume_Rank
# ==============================================================================
def _alpha_017_price_volume_divergence(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    量价背离 (Price-Volume Rank Divergence)
    """
    # 🛡️ 1. 确保必要列存在
    if 'pct_chg' not in data.columns or 'volume' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    # 🛡️ 2. 选取子集时必须包含 stock_code 和 date 用于分组
    # 使用 copy 避免 SettingWithCopyWarning
    df = data[['stock_code', 'date', 'pct_chg', 'volume']].copy()
    
    # 🛡️ 3. 计算成交量变化率 (Time-Series)
    # 显式指定 level=0 或确保索引对齐
    df['vol_chg'] = df.groupby('stock_code', sort=False)['volume'].pct_change()
    
    # 🛡️ 4. 每日截面排名 (Cross-Sectional Rank)
    # rank(pct=True) 默认处理 NaN，但 vol_chg 的第一天全是 NaN，所以要小心
    # 这里直接对 DataFrame 分组排名
    ranks = df.groupby('date')[['pct_chg', 'vol_chg']].rank(pct=True)
    
    # 🛡️ 5. 计算差值
    # 此时 ranks 的索引与原 data 完全一致
    divergence = ranks['pct_chg'] - ranks['vol_chg']
    
    # 🛡️ 6. 填充第一天计算 pct_change 产生的 NaN 并对齐原始索引
    return divergence.fillna(0.0).reindex(data.index).astype('float64')

alpha_017 = register_factor(
    FactorDefinition(
        name='alpha_017',
        required_columns={'stock_code', 'date', 'pct_chg', 'volume'},
        lookback_days=5,
        calculator=_alpha_017_price_volume_divergence,
        description='缩量上涨因子（收益率排名 - 成交量变化排名）',
    )
)

__all__ = [
    'alpha_001',
    'alpha_017',
    'alpha_002',
]

