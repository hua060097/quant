"""
板块-个股交互因子 (Sector-Stock Interaction Factors) - 时序去极值版
文件名: sector_interaction.py

优化日志：
1. 核心修复：使用 _rolling_adaptive_clip (60日) 替代全局 Clip。
2. 基础修复：使用 _daily_robust_zscore 替代全局 Z-Score，确保计算逻辑无未来函数。
3. 适配：保留数值幅度，不强制 Rank，以匹配旧模型对波动率的敏感性。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 1. 基础辅助函数 (Rolling Clip & Daily Z-Score)
# ==============================================================================

def _rolling_adaptive_clip(series: pd.Series, date_series: pd.Series, n_mad: float = 3.0, window: int = 120) -> pd.Series:
    """时序滚动去极值 (Rolling Adaptive Clip)"""
    if series.empty: return series
    
    # 1. 构造 DF 并按日期分组
    df = pd.DataFrame({'val': series.values, 'date': date_series.values}, index=series.index)
    
    # 2. 计算每日截面统计量
    daily_stats = df.groupby('date')['val'].agg(
        median='median',
        mad=lambda x: (x - x.median()).abs().median()
    ).sort_index()
    
    # 3. 计算历史平滑阈值 (Rolling 60)
    rolling_stats = daily_stats.rolling(window=window, min_periods=1).median()
    
    # 4. 映射回个股数据
    df['thresh_median'] = df['date'].map(rolling_stats['median'])
    df['thresh_mad'] = df['date'].map(rolling_stats['mad'])
    
    # 5. 稀疏保护
    mask_sparse = (df['thresh_mad'] < 1e-6)
    
    # 计算最终上下界
    half_width = n_mad * df['thresh_mad'] * 1.4826
    final_half_width = half_width.where(~mask_sparse, np.inf)
    
    upper = df['thresh_median'] + final_half_width
    lower = df['thresh_median'] - final_half_width
    
    upper = upper.fillna(np.inf)
    lower = lower.fillna(-np.inf)
    
    return series.clip(lower, upper)

def _daily_robust_zscore(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """按日稳健 Z-Score (Cross-Sectional)"""
    df = pd.DataFrame({'val': series.values, 'date': date_series.values}, index=series.index)
    
    medians = df.groupby('date')['val'].transform('median')
    abs_diff = (df['val'] - medians).abs()
    
    # 按日计算 MAD
    temp_mad = pd.DataFrame({'diff': abs_diff.values, 'date': date_series.values}, index=series.index)
    mads = temp_mad.groupby('date')['diff'].transform('median')
    
    # 计算 Z-Score
    z_score = (df['val'] - medians) / (mads * 1.4826 + 1e-9)
    
    # 极值处理
    mask_zero_mad = (mads < 1e-6)
    if mask_zero_mad.any():
        means = df.groupby('date')['val'].transform('mean')
        stds = df.groupby('date')['val'].transform('std')
        z_fallback = (df['val'] - means) / (stds + 1e-9)
        z_score = z_score.where(~mask_zero_mad, z_fallback)
        
    return z_score.clip(-6.0, 6.0) # 这里的硬截断是为了防止计算过程中的极端异常

# ==============================================================================
# 2. 核心计算辅助类
# ==============================================================================
class _SectorInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        # 必须先拷贝并排序，确保 rolling/shift 正确
        self.data = data.copy()
        self.data.sort_values(['stock_code', 'date'], inplace=True)
        
        self.ctx = ctx
        self.original_index = data.index

        # 获取行业映射
        sector_map = ctx.extra.get('stock_to_sw_sector', {})
        if not sector_map:
            self.data['sector_code'] = 'All'
        else:
            self.data['sector_code'] = self.data['stock_code'].map(sector_map).fillna('Unknown')
        
        # 执行预计算
        self._precompute_factors()

    def _precompute_factors(self):
        """预计算行业 Gate 与个股 Alpha"""
        
        # --- A. 行业层面指标 (Sector Gates) ---
        s_ret_wide = self.data.groupby(['date', 'sector_code'])['pct_chg'].mean().unstack('sector_code').sort_index()
        
        s_mom_5 = s_ret_wide.rolling(5, min_periods=3).mean()
        s_mom_20 = s_ret_wide.rolling(20, min_periods=10).mean()
        self.s_accel_rank = (s_mom_5 - s_mom_20).rank(axis=1, pct=True)
        
        s_rank_wide = s_ret_wide.rank(axis=1, pct=True)
        self.s_rank_change = (s_rank_wide - s_rank_wide.shift(5))
        
        mkt_ret = self.data.groupby('date')['pct_chg'].mean()
        s_rel_strength = s_ret_wide.sub(mkt_ret, axis=0).rolling(5, min_periods=3).mean()
        self.s_rel_strength_z = s_rel_strength.apply(lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=1)

        self._compute_hhi_metrics()
        self._compute_sector_turnover_metrics()

        self.s_vol = s_ret_wide.rolling(20, min_periods=10).std()
        self.s_dispersion = self.data.groupby(['date', 'sector_code'])['pct_chg'].std().unstack('sector_code')

        # --- B. 个股层面指标 (Stock Alphas) - 按日 Z-Score ---
        # 使用 _daily_robust_zscore 替代 _global_robust_zscore
        
        stock_mom_5 = self.data.groupby('stock_code')['pct_chg'].transform(lambda x: x.rolling(5, min_periods=3).sum())
        self.data['stock_mom_5d_z'] = _daily_robust_zscore(stock_mom_5, self.data['date'])

        stock_mom_20 = self.data.groupby('stock_code')['pct_chg'].transform(lambda x: x.rolling(20, min_periods=15).sum())
        self.data['intra_rank'] = self.data.assign(m=stock_mom_20).groupby('date')['m'].rank(pct=True) * 2 - 1

        high_20d = self.data.groupby('stock_code')['high'].transform(lambda x: x.rolling(20, min_periods=10).max())
        breakout = (self.data['close'] / (high_20d + 1e-9)) - 1
        self.data['breakout_z'] = _daily_robust_zscore(breakout, self.data['date'])

        if 'volume' in self.data.columns and 'circ_mv' in self.data.columns:
            turnover = self.data['volume'] / (self.data['circ_mv'] + 1e-9)
            turnover_ma20 = self.data.groupby('stock_code')['volume'].transform(lambda x: x.rolling(20).mean()) / (self.data['circ_mv'] + 1e-9)
            abnormal_to = (turnover / (turnover_ma20 + 1e-9)) - 1
            self.data['abnormal_to_z'] = _daily_robust_zscore(abnormal_to, self.data['date'])
        else:
            self.data['abnormal_to_z'] = pd.Series(np.nan, index=self.data.index)

        def calc_corr(df):
            return df['pct_chg'].rolling(20, min_periods=15).corr(df['volume'])
        
        pv_corr = self.data.groupby('stock_code', group_keys=False).apply(calc_corr)
        if not pv_corr.index.equals(self.data.index):
             pv_corr = pv_corr.reindex(self.data.index)
             
        self.data['pv_corr_z'] = _daily_robust_zscore(pv_corr, self.data['date'])

    def _compute_hhi_metrics(self):
        self.data['amount'] = self.data['volume'] * self.data.get('close', 1.0)
        sector_total = self.data.groupby(['date', 'sector_code'])['amount'].transform('sum')
        self.data['share_sq'] = (self.data['amount'] / (sector_total + 1e-9)) ** 2
        hhi_wide = self.data.groupby(['date', 'sector_code'])['share_sq'].sum().unstack('sector_code')
        self.s_hhi_change_rank = (hhi_wide.diff(5)).rank(axis=1, pct=True)

    def _compute_sector_turnover_metrics(self):
        s_vol = self.data.groupby(['date', 'sector_code'])['volume'].mean().unstack('sector_code')
        v_5 = s_vol.rolling(5).mean()
        v_20 = s_vol.rolling(20).mean()
        self.s_turnover_change_rank = ((v_5 - v_20) / (v_20 + 1e-9)).rank(axis=1, pct=True)

    def get_interaction_factor(self, factor_name: str) -> pd.Series:
        """映射 Gate 与 Alpha 的乘积逻辑"""
        
        # 1. 门控项映射 (Gate)
        gate_series = None
        if factor_name == 'sector_vols_divergence_alpha': gate_series = -self.s_vol
        elif factor_name == 'sector_turnover_surge_pv_alpha': gate_series = self.s_turnover_change_rank
        elif factor_name == 'sector_dispersion_leader_alpha': gate_series = self.s_dispersion
        elif factor_name == 'sector_trend_gated_momentum_intra_z': gate_series = self.s_accel_rank
        elif factor_name == 'sector_rotation_breakout_intra_z': gate_series = self.s_rank_change
        elif factor_name == 'sector_leader_capture_rank': gate_series = self.s_hhi_change_rank
        elif factor_name == 'sector_strong_leader_alpha': gate_series = self.s_rel_strength_z
        elif factor_name == 'sector_liquidity_gated_turnover_z': gate_series = self.s_turnover_change_rank
        elif factor_name == 'sector_weak_reversal_alpha': gate_series = -self.s_rel_strength_z
        else: raise ValueError(f"Unknown factor: {factor_name}")

        s_metric_long = gate_series.stack().rename('gate')
        joined = self.data.join(s_metric_long, on=['date', 'sector_code'], how='left')
        gate = joined['gate']

        # 2. 个股 Alpha 项映射
        alpha = None
        if factor_name == 'sector_vols_divergence_alpha': alpha = self.data['breakout_z']
        elif factor_name == 'sector_turnover_surge_pv_alpha': alpha = self.data['pv_corr_z']
        elif factor_name == 'sector_dispersion_leader_alpha': alpha = self.data['intra_rank']
        elif factor_name == 'sector_trend_gated_momentum_intra_z': alpha = self.data['stock_mom_5d_z']
        elif factor_name == 'sector_rotation_breakout_intra_z': alpha = self.data['breakout_z']
        elif factor_name == 'sector_leader_capture_rank': alpha = self.data['intra_rank']
        elif factor_name == 'sector_strong_leader_alpha': alpha = self.data['intra_rank']
        elif factor_name == 'sector_liquidity_gated_turnover_z': alpha = self.data['abnormal_to_z']
        elif factor_name == 'sector_weak_reversal_alpha': alpha = -self.data['stock_mom_5d_z']
        
        # 3. 计算交互
        raw_factor = gate * alpha
        
        # 4. 后处理：使用 Rolling Adaptive Clip
        # 60日窗口，n_mad=3.0   这里多改了一个window从60改到120 202618:15
        return _rolling_adaptive_clip(raw_factor, self.data['date'], n_mad=3.1, window=120).reindex(self.original_index).astype('float64')

# ==============================================================================
# 3. 注册因子
# ==============================================================================
def _get_sector_interaction_helper(data: pd.DataFrame, ctx: FactorContext) -> _SectorInteractionHelper:
    helper_key = '_sector_interaction_helper_v_final_rolling'
    if helper_key not in ctx.extra:
        ctx.extra[helper_key] = _SectorInteractionHelper(data, ctx)
    return ctx.extra[helper_key]

def _c_vols_div(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_vols_divergence_alpha')
def _c_pv_surge(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_turnover_surge_pv_alpha')
def _c_disp_leader(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_dispersion_leader_alpha')
def _c_trend_gate(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_trend_gated_momentum_intra_z')
def _c_rot_break(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_rotation_breakout_intra_z')
def _c_hhi_cap(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_leader_capture_rank')
def _c_strong_alpha(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_strong_leader_alpha')
def _c_liq_gate(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_liquidity_gated_turnover_z')
def _c_weak_rev(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_weak_reversal_alpha')

sector_vols_divergence_alpha = register_factor(FactorDefinition(name='sector_vols_divergence_alpha', required_columns={'stock_code', 'date', 'pct_chg', 'high', 'close'}, requires_sector_data=True, lookback_days=40, calculator=_c_vols_div))
sector_turnover_surge_pv_alpha = register_factor(FactorDefinition(name='sector_turnover_surge_pv_alpha', required_columns={'stock_code', 'date', 'pct_chg', 'volume'}, requires_sector_data=True, lookback_days=40, calculator=_c_pv_surge))
sector_dispersion_leader_alpha = register_factor(FactorDefinition(name='sector_dispersion_leader_alpha', required_columns={'stock_code', 'date', 'pct_chg'}, requires_sector_data=True, lookback_days=30, calculator=_c_disp_leader))
sector_trend_gated_momentum_intra_z = register_factor(FactorDefinition(name='sector_trend_gated_momentum_intra_z', required_columns={'stock_code', 'date', 'pct_chg'}, requires_sector_data=True, lookback_days=25, calculator=_c_trend_gate))
sector_rotation_breakout_intra_z = register_factor(FactorDefinition(name='sector_rotation_breakout_intra_z', required_columns={'stock_code', 'date', 'pct_chg', 'high', 'close'}, requires_sector_data=True, lookback_days=25, calculator=_c_rot_break))
sector_leader_capture_rank = register_factor(FactorDefinition(name='sector_leader_capture_rank', required_columns={'stock_code', 'date', 'pct_chg', 'volume'}, requires_sector_data=True, lookback_days=30, calculator=_c_hhi_cap))
sector_strong_leader_alpha = register_factor(FactorDefinition(name='sector_strong_leader_alpha', required_columns={'stock_code', 'date', 'pct_chg'}, requires_sector_data=True, lookback_days=25, calculator=_c_strong_alpha))
sector_liquidity_gated_turnover_z = register_factor(FactorDefinition(name='sector_liquidity_gated_turnover_z', required_columns={'stock_code', 'date', 'pct_chg', 'volume', 'circ_mv'}, requires_sector_data=True, lookback_days=25, calculator=_c_liq_gate))
sector_weak_reversal_alpha = register_factor(FactorDefinition(name='sector_weak_reversal_alpha', required_columns={'stock_code', 'date', 'pct_chg'}, requires_sector_data=True, lookback_days=25, calculator=_c_weak_rev))

__all__ = [
    'sector_vols_divergence_alpha', 'sector_turnover_surge_pv_alpha', 'sector_dispersion_leader_alpha',
    'sector_trend_gated_momentum_intra_z', 'sector_rotation_breakout_intra_z', 'sector_leader_capture_rank',
    'sector_strong_leader_alpha', 'sector_liquidity_gated_turnover_z', 'sector_weak_reversal_alpha'
]
