"""
板块-个股交互因子 (Sector-Stock Interaction Factors) - 完整生产版

集成列表：
1-6 (原始): 动量门控、旋转突破、集中度捕获、强势领先、流动性门控、弱势反转
7-9 (新增): 波动分歧、成交放量、分化领先龙头
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 1. 基础辅助函数
# ==============================================================================
def _adaptive_clip(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    """自适应去极值（基于 1.4826 * MAD 缩放）"""
    if series.empty:
        return series
    median = series.median()
    mad = (series - median).abs().median()
    threshold = n_mad * mad * 1.4826
    return series.clip(median - threshold, median + threshold)

# ==============================================================================
# 2. 核心计算辅助类
# ==============================================================================
class _SectorInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.data = data.copy()
        self.ctx = ctx
        self.original_index = data.index

        # 获取行业映射
        sector_map = ctx.extra.get('stock_to_sw_sector', {})
        if not sector_map:
            raise ValueError("Missing 'stock_to_sw_sector' in context.extra")

        self.data['sector_code'] = self.data['stock_code'].map(sector_map).fillna('Unknown')
        
        # 执行向量化预计算，为所有9个因子准备底层指标
        self._precompute_factors()

    def _intra_sector_standardize(self, series: pd.Series):
        """通用行业内标准化算子"""
        return self.data.assign(tmp=series).groupby(['date', 'sector_code'])['tmp'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    def _precompute_factors(self):
        """预计算行业 Gate 与个股 Alpha"""
        # --- 行业层面指标 (Sector Level) ---
        s_ret_wide = self.data.groupby(['date', 'sector_code'])['pct_chg'].mean().unstack('sector_code').sort_index()
        
        # 行业动量加速度 (原始因子1)
        s_mom_5 = s_ret_wide.rolling(5, min_periods=3).mean()
        s_mom_20 = s_ret_wide.rolling(20, min_periods=10).mean()
        self.s_accel_rank = (s_mom_5 - s_mom_20).rank(axis=1, pct=True)
        
        # 行业排名变化 (原始因子2)
        s_rank_wide = s_ret_wide.rank(axis=1, pct=True)
        self.s_rank_change = (s_rank_wide - s_rank_wide.shift(5)).fillna(0)
        
        # 行业相对强弱 (原始因子4, 6)
        mkt_ret = self.data.groupby('date')['pct_chg'].mean()
        s_rel_strength = s_ret_wide.sub(mkt_ret, axis=0).rolling(5, min_periods=3).mean()
        self.s_rel_strength_z = s_rel_strength.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1)

        # 行业集中度 HHI 与 成交变化 (原始因子3, 5)
        self._compute_hhi_metrics()
        self._compute_sector_turnover_metrics()

        # 新增：行业波动率 (新增因子7)
        self.s_vol = s_ret_wide.rolling(20, min_periods=10).std()
        
        # 新增：行业离散度 (新增因子9)
        self.s_dispersion = self.data.groupby(['date', 'sector_code'])['pct_chg'].std().unstack('sector_code')

        # --- 个股层面指标 (Stock Alpha) ---
        # 5日动量 Z-Score (用于原始因子1, 6)
        stock_mom_5 = self.data.groupby('stock_code')['pct_chg'].transform(lambda x: x.rolling(5, min_periods=3).sum())
        self.data['stock_mom_5d_z'] = self._intra_sector_standardize(stock_mom_5)

        # 20日动量排名 (用于原始因子3, 4, 新增因子9)
        stock_mom_20 = self.data.groupby('stock_code')['pct_chg'].transform(lambda x: x.rolling(20, min_periods=15).sum())
        self.data['intra_rank'] = self.data.assign(m=stock_mom_20).groupby(['date', 'sector_code'])['m'].rank(pct=True) * 2 - 1

        # 突破强度 (用于原始因子2, 新增因子7)
        high_20d = self.data.groupby('stock_code')['high'].transform(lambda x: x.rolling(20, min_periods=10).max())
        self.data['breakout_z'] = self._intra_sector_standardize((self.data['close'] / (high_20d + 1e-8)) - 1)

        # 异常换手率 (用于原始因子5)
        if 'volume' in self.data.columns and 'circ_mv' in self.data.columns:
            turnover = self.data['volume'] / (self.data['circ_mv'] + 1e-8)
            turnover_ma20 = self.data.groupby('stock_code')['volume'].transform(lambda x: x.rolling(20).mean()) / (self.data['circ_mv'] + 1e-8)
            self.data['abnormal_to_z'] = self._intra_sector_standardize((turnover / (turnover_ma20 + 1e-8)) - 1)

        # 新增：价量相关性 (用于新增因子8)
        pv_corr = self.data.groupby('stock_code').apply(
            lambda x: x['pct_chg'].rolling(20, min_periods=15).corr(x['volume'])
        ).reset_index(0, drop=True)
        self.data['pv_corr_z'] = self._intra_sector_standardize(pv_corr)

    def _compute_hhi_metrics(self):
        """集中度 HHI 计算"""
        self.data['amount'] = self.data['volume'] * self.data.get('close', 1.0)
        sector_total = self.data.groupby(['date', 'sector_code'])['amount'].transform('sum')
        self.data['share_sq'] = (self.data['amount'] / (sector_total + 1e-9)) ** 2
        hhi_wide = self.data.groupby(['date', 'sector_code'])['share_sq'].sum().unstack('sector_code').fillna(0)
        self.s_hhi_change_rank = (hhi_wide.diff(5)).rank(axis=1, pct=True)

    def _compute_sector_turnover_metrics(self):
        """行业换手变化"""
        s_vol = self.data.groupby(['date', 'sector_code'])['volume'].mean().unstack('sector_code')
        v_5, v_20 = s_vol.rolling(5).mean(), s_vol.rolling(20).mean()
        self.s_turnover_change_rank = ((v_5 - v_20) / (v_20 + 1e-8)).rank(axis=1, pct=True)

    def get_interaction_factor(self, factor_name: str) -> pd.Series:
        """映射 Gate 与 Alpha 的乘积逻辑"""
        
        # 门控项映射
        metric_map = {
            # 新增 3 因子
            'sector_vols_divergence_alpha': -self.s_vol,          # 行业静默期
            'sector_turnover_surge_pv_alpha': self.s_turnover_change_rank, # 行业放量
            'sector_dispersion_leader_alpha': self.s_dispersion,  # 行业分化
            # 原始 6 因子
            'sector_trend_gated_momentum_intra_z': self.s_accel_rank,
            'sector_rotation_breakout_intra_z': self.s_rank_change,
            'sector_leader_capture_rank': self.s_hhi_change_rank,
            'sector_strong_leader_alpha': self.s_rel_strength_z,
            'sector_liquidity_gated_turnover_z': self.s_turnover_change_rank,
            'sector_weak_reversal_alpha': -self.s_rel_strength_z
        }
        
        if factor_name not in metric_map:
            raise ValueError(f"Unknown factor: {factor_name}")

        s_metric_long = metric_map[factor_name].stack().rename('gate')
        joined = self.data.join(s_metric_long, on=['date', 'sector_code'], how='left')
        gate = _adaptive_clip(joined['gate'].fillna(joined['gate'].median()))

        # 个股 Alpha 项映射
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
            alpha = self.data.get('abnormal_to_z', 0)
        elif factor_name == 'sector_weak_reversal_alpha':
            alpha = -self.data['stock_mom_5d_z']
        else:
            alpha = 0

        return _adaptive_clip(gate * alpha).reindex(self.original_index).fillna(0).astype('float64')

# ==============================================================================
# 3. 注册逻辑
# ==============================================================================
def _get_sector_interaction_helper(data: pd.DataFrame, ctx: FactorContext) -> _SectorInteractionHelper:
    helper_key = '_sector_interaction_helper'
    if helper_key not in ctx.extra:
        ctx.extra[helper_key] = _SectorInteractionHelper(data, ctx)
    return ctx.extra[helper_key]

# 计算包装器
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
    'sector_vols_divergence_alpha', 'sector_turnover_surge_pv_alpha', 'sector_dispersion_leader_alpha',
    'sector_trend_gated_momentum_intra_z', 'sector_rotation_breakout_intra_z', 'sector_leader_capture_rank',
    'sector_strong_leader_alpha', 'sector_liquidity_gated_turnover_z', 'sector_weak_reversal_alpha'
]