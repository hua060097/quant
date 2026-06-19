"""
WorldQuant 101 Alphas 精选版（深度优化）。
向量化复刻最有效的几个 Alpha。

优化日志：
1. 全局移除 .fillna(0)，保留 NaN (未知) 状态。
2. 引入 _global_robust_zscore (Median/MAD)，确保因子在截面上具有可比性。
3. 重写滚动相关性计算逻辑，使用 groupby().apply() 替代脆弱的 unstack()。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series
from ..universe import non_st_mask

# ==============================================================================
# 0. 核心工具：全局按日稳健 Z-Score (Top-1 专用版)
# ==============================================================================
def _global_robust_zscore(
    series: pd.Series,
    date_series: pd.Series,
    eligible: pd.Series | None = None,
) -> pd.Series:
    """
    [大师级] 全局按日稳健 Z-Score (Robust Z-Score based on MAD)
    
    逻辑：
    使用 Median 和 MAD 代替 Mean/Std，防止龙头股的极端 Alpha 被普通 Z-Score 压缩。
    保留 NaN，不填 0。
    """
    if eligible is None:
        eligible = pd.Series(True, index=series.index)
    eligible = eligible.reindex(series.index).fillna(True)
    # 1. 构造临时 DF
    df = pd.DataFrame({'val': series.where(eligible).values, 'date': date_series.values}, index=series.index)
    
    # 2. 向量化计算每日中位数 (Median)
    medians = df.groupby('date')['val'].transform('median')
    
    # 3. 向量化计算每日绝对中位差 (MAD)
    abs_diff = (df['val'] - medians).abs()
    
    # 复用索引构造临时 DF
    temp_diff = pd.DataFrame({'diff': abs_diff.values, 'date': date_series.values}, index=series.index)
    mads = temp_diff.groupby('date')['diff'].transform('median')
    
    # 4. 计算稳健 Z-Score
    robust_std = mads * 1.4826
    z_score = (df['val'] - medians) / (robust_std + 1e-9)
    
    # 5. 极值处理：特殊处理 MAD 为 0 的情况
    mask_zero_mad = (mads < 1e-6)
    if mask_zero_mad.any():
        means = df.groupby('date')['val'].transform('mean')
        stds = df.groupby('date')['val'].transform('std')
        z_fallback = (df['val'] - means) / (stds + 1e-9)
        z_score = z_score.where(~mask_zero_mad, z_fallback)
    
    # 6. 宽阈值去极值 (Clip 6.0)
    # [修正] 删除 fillna(0.0)
    return z_score.where(eligible).clip(-6.0, 6.0).astype('float64')


# ==============================================================================
# Alpha 001: 量价相关性 (PV Correlation)
# ==============================================================================
def _alpha_001_pv_correlation(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    Alpha 001 (PV Correlation)
    
    逻辑：
    Corr(Close, Volume, 10)
    
    含义：
    量价配合度。
    Z-Score 后：正值代表量价齐升/齐跌（趋势强），负值代表背离。
    """
    if 'close' not in data.columns or 'volume' not in data.columns:
        return pd.Series(np.nan, index=data.index)
        
    # 构造临时 DataFrame 以便 apply
    temp = pd.DataFrame({
        'c': data['close'],
        'v': data['volume'],
        'stock_code': data['stock_code']
    }, index=data.index)
    
    def calc_corr(df):
        # min_periods=8: 保证至少有8天数据才计算
        return df['c'].rolling(10, min_periods=8).corr(df['v'])

    # 计算原始相关系数
    raw_corr = temp.groupby('stock_code', sort=False, group_keys=False).apply(calc_corr)
    
    # [优化] 进行全局稳健标准化
    # 这样每天的因子值分布都是 N(0, 1)，方便横向比较谁的量价配合最完美
    return _global_robust_zscore(raw_corr, data['date'], eligible=non_st_mask(data))

alpha_001 = register_factor(
    FactorDefinition(
        name='alpha_001',
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=20,
        calculator=_alpha_001_pv_correlation,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _alpha_001_pv_correlation(d, c), target_index=ti
        ),
        description='量价相关性（10日滚动相关系数，全局标准化）',
    )
)

# ==============================================================================
# Alpha 002: 成交量变化与日内涨幅的负相关性
# ==============================================================================
def _alpha_002_volatility_reversal(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    Alpha 002 (Volatility Reversal)
    
    逻辑：
    -1 * Correlation(Delta(Log(Volume), 2), (Close-Open)/Open, 6)
    """
    if 'close' not in data.columns or 'volume' not in data.columns:
        return pd.Series(np.nan, index=data.index)

    grouped = data.groupby('stock_code', sort=False)
    
    # 1. Delta(Log(Volume), 2)
    # [优化] 使用 log1p 防止 volume=0 报错
    log_vol = np.log1p(data['volume'])
    delta_vol = log_vol.groupby(data['stock_code'], sort=False).diff(2)
    
    # 2. Intraday Return
    intra_ret = (data['close'] - data['open']) / (data['open'] + 1e-9)
    
    # 3. Correlation (6日)
    temp = pd.DataFrame({
        'dv': delta_vol, 
        'ir': intra_ret, 
        'stock_code': data['stock_code']
    }, index=data.index)
    
    def calc_corr(df):
        return df['dv'].rolling(6, min_periods=5).corr(df['ir'])

    raw_corr = temp.groupby('stock_code', sort=False, group_keys=False).apply(calc_corr)
    
    # 取负号
    alpha_raw = -1.0 * raw_corr
    
    # [优化] 全局稳健标准化，移除 fillna
    return _global_robust_zscore(alpha_raw, data['date'], eligible=non_st_mask(data))

alpha_002 = register_factor(
    FactorDefinition(
        name='alpha_002',
        required_columns={'stock_code', 'date', 'close', 'open', 'volume'},
        lookback_days=15,
        calculator=_alpha_002_volatility_reversal,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _alpha_002_volatility_reversal(d, c), target_index=ti
        ),
        description='成交量变化与日内涨幅的负相关性（反转因子，全局标准化）',
    )
)

# ==============================================================================
# Alpha 017: 量价背离 (Price-Volume Rank Divergence)
# ==============================================================================
def _alpha_017_price_volume_divergence(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    量价背离 (Price-Volume Rank Divergence)
    
    逻辑：Rank(Pct_Chg) - Rank(Vol_Chg)
    含义：缩量上涨(分高) vs 放量滞涨(分低)
    """
    # 🛡️ 1. 确保必要列存在
    if 'pct_chg' not in data.columns or 'volume' not in data.columns:
        return pd.Series(np.nan, index=data.index)
    
    # 🛡️ 2. 选取子集
    # copy 避免 SettingWithCopyWarning
    df = data[['stock_code', 'date', 'pct_chg', 'volume']].copy()
    eligible = non_st_mask(data).reindex(df.index).fillna(True)
    
    # 🛡️ 3. 计算成交量变化率 (Time-Series)
    # 显式指定 level=0 或确保索引对齐
    df['vol_chg'] = df.groupby('stock_code', sort=False)['volume'].pct_change()
    
    # 🛡️ 4. 每日截面排名 (Cross-Sectional Rank)
    # rank(pct=True) 默认处理 NaN (结果为 NaN)，这很好
    rank_input = df[['pct_chg', 'vol_chg']].copy()
    rank_input.loc[~eligible, ['pct_chg', 'vol_chg']] = np.nan
    ranks = pd.concat([df[['date']], rank_input], axis=1).groupby('date')[['pct_chg', 'vol_chg']].rank(pct=True)
    
    # 🛡️ 5. 计算差值
    # NaN - Value = NaN，这是我们想要的
    divergence = ranks['pct_chg'] - ranks['vol_chg']
    
    # 🛡️ 6. 标准化
    # 虽然 Rank 差值本身已经归一化（-1到1），但为了与其他因子保持分布一致性，
    # 我们也对其进行 Z-Score 处理。
    # 且不 fillna(0)。
    return _global_robust_zscore(divergence, data['date'], eligible=eligible)

alpha_017 = register_factor(
    FactorDefinition(
        name='alpha_017',
        required_columns={'stock_code', 'date', 'pct_chg', 'volume'},
        lookback_days=5,
        calculator=_alpha_017_price_volume_divergence,
        calculation_scope='target_cross_section',
        target_calculator=lambda d, ti, c: select_target_series(
            _alpha_017_price_volume_divergence(d, c), target_index=ti
        ),
        description='缩量上涨因子（收益率排名 - 成交量变化排名，全局标准化）',
    )
)

__all__ = [
    'alpha_001',
    'alpha_017',
    'alpha_002',
]
