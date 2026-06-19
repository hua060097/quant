"""
板块-个股交互因子 (Sector-Stock Interaction Factors) - 大师级完整版
文件名: sector_interaction.py

包含因子 (共9个):
1. sector_vols_divergence_alpha: 低波动板块中的突破个股
2. sector_turnover_surge_pv_alpha: 换手激增板块中的价量配合个股
3. sector_dispersion_leader_alpha: 高分歧板块中的领涨个股
4. sector_trend_gated_momentum_intra_z: 动量加速板块中的强势个股
5. sector_rotation_breakout_intra_z: 排名飙升板块中的突破个股
6. sector_leader_capture_rank: 筹码集中板块中的领涨个股
7. sector_strong_leader_alpha: 强势板块中的领涨个股
8. sector_liquidity_gated_turnover_z: 资金流入板块中的异常放量个股
9. sector_weak_reversal_alpha: 弱势板块中的超跌反转个股

优化日志：
1. 完整保留 9 个交互因子。
2. 移除所有 fillna(0)，保留 NaN 状态。
3. 个股 Alpha 指标统一使用 _global_robust_zscore (Median/MAD)。
4. 修复索引对齐问题。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 1. 基础辅助函数
# ==============================================================================
def _daily_adaptive_clip(series: pd.Series, date_series: pd.Series, n_mad: float = 2.0) -> pd.Series:
    """
    按日截面去极值。
    参数建议：n_mad=2.0 (比默认3.0更严，模拟旧模型的强去噪效果)
    """
    df = pd.DataFrame({'val': series.values, 'date': date_series.values}, index=series.index)
    medians = df.groupby('date')['val'].transform('median')
    abs_diff = (df['val'] - medians).abs()
    
    # 必须重新构造DF来算MAD
    temp_mad = pd.DataFrame({'diff': abs_diff.values, 'date': date_series.values})
    mads = temp_mad.groupby('date')['diff'].transform('median')
    
    threshold = n_mad * mads * 1.4826
    
    return series.clip(medians - threshold, medians + threshold)

def _global_robust_zscore(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """
    [大师级] 全局按日稳健 Z-Score (Robust Z-Score based on MAD)
    """
    # 构造临时 DF
    df = pd.DataFrame({'val': series.values, 'date': date_series.values}, index=series.index)
    
    # 向量化计算每日中位数 (自动忽略 NaN)
    medians = df.groupby('date')['val'].transform('median')
    
    # 计算 MAD
    abs_diff = (df['val'] - medians).abs()
    temp_diff = pd.DataFrame({'diff': abs_diff.values, 'date': date_series.values}, index=series.index)
    mads = temp_diff.groupby('date')['diff'].transform('median')
    
    # 计算 Z-Score
    robust_std = mads * 1.4826
    z_score = (df['val'] - medians) / (robust_std + 1e-9)
    
    # 极值处理 (MAD=0)
    mask_zero_mad = (mads < 1e-6)
    if mask_zero_mad.any():
        means = df.groupby('date')['val'].transform('mean')
        stds = df.groupby('date')['val'].transform('std')
        z_fallback = (df['val'] - means) / (stds + 1e-9)
        z_score = z_score.where(~mask_zero_mad, z_fallback)
        
    # 宽阈值去极值，保留 NaN
    return z_score.clip(-6.0, 6.0).astype('float64')

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
        # 计算行业日均收益 (Index: date, Columns: sector_code)
        s_ret_wide = self.data.groupby(['date', 'sector_code'])['pct_chg'].mean().unstack('sector_code').sort_index()
        
        # 1. 行业动量加速度 Gate
        s_mom_5 = s_ret_wide.rolling(5, min_periods=3).mean()
        s_mom_20 = s_ret_wide.rolling(20, min_periods=10).mean()
        self.s_accel_rank = (s_mom_5 - s_mom_20).rank(axis=1, pct=True)
        
        # 2. 行业排名变化 Gate
        s_rank_wide = s_ret_wide.rank(axis=1, pct=True)
        # 不填 0，保留 NaN
        self.s_rank_change = (s_rank_wide - s_rank_wide.shift(5))
        
        # 3. 行业相对强弱 Gate (板块间 Z-Score)
        mkt_ret = self.data.groupby('date')['pct_chg'].mean()
        s_rel_strength = s_ret_wide.sub(mkt_ret, axis=0).rolling(5, min_periods=3).mean()
        self.s_rel_strength_z = s_rel_strength.apply(lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=1)

        # 4. 行业集中度与成交 Gate
        self._compute_hhi_metrics()
        self._compute_sector_turnover_metrics()

        # 5. 行业波动与离散 Gate
        self.s_vol = s_ret_wide.rolling(20, min_periods=10).std()
        self.s_dispersion = self.data.groupby(['date', 'sector_code'])['pct_chg'].std().unstack('sector_code')

        # --- B. 个股层面指标 (Stock Alphas) - 全局模式 ---
        
        # 1. 动量 Z-Score (全局)
        stock_mom_5 = self.data.groupby('stock_code')['pct_chg'].transform(lambda x: x.rolling(5, min_periods=3).sum())
        self.data['stock_mom_5d_z'] = _global_robust_zscore(stock_mom_5, self.data['date'])

        # 2. 动量排名 (全局 Rank 映射到 -1~1)
        stock_mom_20 = self.data.groupby('stock_code')['pct_chg'].transform(lambda x: x.rolling(20, min_periods=15).sum())
        self.data['intra_rank'] = self.data.assign(m=stock_mom_20).groupby('date')['m'].rank(pct=True) * 2 - 1

        # 3. 突破强度 Z-Score (全局)
        high_20d = self.data.groupby('stock_code')['high'].transform(lambda x: x.rolling(20, min_periods=10).max())
        breakout = (self.data['close'] / (high_20d + 1e-9)) - 1
        self.data['breakout_z'] = _global_robust_zscore(breakout, self.data['date'])

        # 4. 异常换手 Z-Score (全局)
        if 'volume' in self.data.columns and 'circ_mv' in self.data.columns:
            turnover = self.data['volume'] / (self.data['circ_mv'] + 1e-9)
            turnover_ma20 = self.data.groupby('stock_code')['volume'].transform(lambda x: x.rolling(20).mean()) / (self.data['circ_mv'] + 1e-9)
            abnormal_to = (turnover / (turnover_ma20 + 1e-9)) - 1
            self.data['abnormal_to_z'] = _global_robust_zscore(abnormal_to, self.data['date'])
        else:
            self.data['abnormal_to_z'] = pd.Series(np.nan, index=self.data.index)

        # 5. 价量相关性 Z-Score (全局)
        def calc_corr(df):
            return df['pct_chg'].rolling(20, min_periods=15).corr(df['volume'])
        
        pv_corr = self.data.groupby('stock_code', group_keys=False).apply(calc_corr)
        # 索引对齐
        if not pv_corr.index.equals(self.data.index):
             pv_corr = pv_corr.reindex(self.data.index)
             
        self.data['pv_corr_z'] = _global_robust_zscore(pv_corr, self.data['date'])

    def _compute_hhi_metrics(self):
        """集中度 HHI 计算"""
        self.data['amount'] = self.data['volume'] * self.data.get('close', 1.0)
        # 按行业求和
        sector_total = self.data.groupby(['date', 'sector_code'])['amount'].transform('sum')
        self.data['share_sq'] = (self.data['amount'] / (sector_total + 1e-9)) ** 2
        
        # 聚合成宽表
        hhi_wide = self.data.groupby(['date', 'sector_code'])['share_sq'].sum().unstack('sector_code')
        # 变化率 Rank
        self.s_hhi_change_rank = (hhi_wide.diff(5)).rank(axis=1, pct=True)

    def _compute_sector_turnover_metrics(self):
        """行业换手变化"""
        s_vol = self.data.groupby(['date', 'sector_code'])['volume'].mean().unstack('sector_code')
        v_5 = s_vol.rolling(5).mean()
        v_20 = s_vol.rolling(20).mean()
        self.s_turnover_change_rank = ((v_5 - v_20) / (v_20 + 1e-9)).rank(axis=1, pct=True)

    def get_interaction_factor(self, factor_name: str) -> pd.Series:
        """映射 Gate 与 Alpha 的乘积逻辑"""
        
        # 1. 门控项映射 (Gate) - 宽表 Series
        gate_series = None
        
        if factor_name == 'sector_vols_divergence_alpha':
            gate_series = -self.s_vol
        elif factor_name == 'sector_turnover_surge_pv_alpha':
            gate_series = self.s_turnover_change_rank
        elif factor_name == 'sector_dispersion_leader_alpha':
            gate_series = self.s_dispersion
        elif factor_name == 'sector_trend_gated_momentum_intra_z':
            gate_series = self.s_accel_rank
        elif factor_name == 'sector_rotation_breakout_intra_z':
            gate_series = self.s_rank_change
        elif factor_name == 'sector_leader_capture_rank':
            gate_series = self.s_hhi_change_rank
        elif factor_name == 'sector_strong_leader_alpha':
            gate_series = self.s_rel_strength_z
        elif factor_name == 'sector_liquidity_gated_turnover_z':
            gate_series = self.s_turnover_change_rank
        elif factor_name == 'sector_weak_reversal_alpha':
            gate_series = -self.s_rel_strength_z
        else:
            raise ValueError(f"Unknown factor: {factor_name}")

        # 广播回个股
        s_metric_long = gate_series.stack().rename('gate')
        joined = self.data.join(s_metric_long, on=['date', 'sector_code'], how='left')
        gate = joined['gate'] # 保留 NaN

        # 2. 个股 Alpha 项映射 (Alpha)
        alpha = None
        if factor_name == 'sector_vols_divergence_alpha':
            alpha = self.data['breakout_z']
        elif factor_name == 'sector_turnover_surge_pv_alpha':
            alpha = self.data['pv_corr_z']
        elif factor_name == 'sector_dispersion_leader_alpha':
            alpha = self.data['intra_rank']
        elif factor_name == 'sector_trend_gated_momentum_intra_z':
            alpha = self.data['stock_mom_5d_z']
        elif factor_name == 'sector_rotation_breakout_intra_z':
            alpha = self.data['breakout_z']
        elif factor_name == 'sector_leader_capture_rank':
            alpha = self.data['intra_rank']
        elif factor_name == 'sector_strong_leader_alpha':
            alpha = self.data['intra_rank']
        elif factor_name == 'sector_liquidity_gated_turnover_z':
            alpha = self.data['abnormal_to_z']
        elif factor_name == 'sector_weak_reversal_alpha':
            alpha = -self.data['stock_mom_5d_z']
        
        # 3. 计算交互 (Gate * Alpha)
        raw_factor = gate * alpha
        
        # 4. 后处理：自适应去极值，并确保索引与原始数据一致
        clipped = _daily_adaptive_clip(raw_factor, self.data['date'], n_mad=2.0).reindex(self.original_index).astype('float64')
        # 2. [关键] 强制转为每日百分比排名
        final_rank = clipped.groupby(self.data['date']).rank(pct=True)
        
        return final_rank.reindex(self.original_index).fillna(0.5)
        # return _adaptive_clip(raw_factor).reindex(self.original_index).astype('float64')

# ==============================================================================
# 3. 注册因子
# ==============================================================================
def _get_sector_interaction_helper(data: pd.DataFrame, ctx: FactorContext) -> _SectorInteractionHelper:
    helper_key = '_sector_interaction_helper_v_final'
    if helper_key not in ctx.extra:
        ctx.extra[helper_key] = _SectorInteractionHelper(data, ctx)
    return ctx.extra[helper_key]

# 包装器
def _c_vols_div(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_vols_divergence_alpha')
def _c_pv_surge(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_turnover_surge_pv_alpha')
def _c_disp_leader(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_dispersion_leader_alpha')
def _c_trend_gate(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_trend_gated_momentum_intra_z')
def _c_rot_break(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_rotation_breakout_intra_z')
def _c_hhi_cap(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_leader_capture_rank')
def _c_strong_alpha(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_strong_leader_alpha')
def _c_liq_gate(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_liquidity_gated_turnover_z')
def _c_weak_rev(d, c): return _get_sector_interaction_helper(d, c).get_interaction_factor('sector_weak_reversal_alpha')

# 注册所有 9 个因子
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
    'sector_vols_divergence_alpha',
    'sector_turnover_surge_pv_alpha',
    'sector_dispersion_leader_alpha',
    'sector_trend_gated_momentum_intra_z',
    'sector_rotation_breakout_intra_z',
    'sector_leader_capture_rank',
    'sector_strong_leader_alpha',
    'sector_liquidity_gated_turnover_z',
    'sector_weak_reversal_alpha'
]