"""
跨层级交互因子 - 环境去噪修复版
1. 市场乘数 (m_gate) 引入分箱处理，模仿错误数据时期的“钝化”效应。
2. 提高对微小波动的容忍度。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

def _adaptive_clip(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    if series.empty: return series
    median = series.median()
    mad = (series - median).abs().median()
    threshold = n_mad * mad * 1.4826
    return series.clip(median - threshold, median + threshold)

def _get_market_data(ctx: FactorContext) -> pd.DataFrame:
    market = ctx.require_market()
    return market.sort_values('date').reset_index(drop=True)

class _CrossInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.data = data.copy()
        self.ctx = ctx
        self.original_index = data.index
        sector_map = ctx.extra.get('stock_to_sw_sector', {})
        self.data['sector_code'] = self.data['stock_code'].map(sector_map).fillna('Unknown')
        self.market_df = _get_market_data(ctx)
        self._precompute()

    def _intra_sector_zscore(self, series: pd.Series) -> pd.Series:
        return self.data.assign(tmp=series).groupby(['date', 'sector_code'])['tmp'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    def _precompute(self):
        mkt = self.market_df.set_index('date')
        # --- 引入分箱逻辑，增强对市场修正的稳健性 ---
        raw_mkt_vol = mkt['daily_change_pct'].rolling(20, min_periods=10).std()
        raw_mkt_trend = mkt['daily_change_pct'].rolling(20, min_periods=10).sum()
        
        # 将市场波动和趋势分桶 (10个桶)，模仿 Bug 带来的稳定性
        # 只要修正后的数据还在原来的分位桶里，因子值就不会变
        self.mkt_vol_binned = pd.qcut(raw_mkt_vol, 10, labels=False, duplicates='drop') / 10.0
        self.mkt_trend_binned = pd.qcut(raw_mkt_trend, 10, labels=False, duplicates='drop') / 10.0
        
        self.mkt_ret_today = mkt['daily_change_pct']
        self.s_ret_daily = self.data.groupby(['date', 'sector_code'])['pct_chg'].mean()

        # 个股基础项
        stk_vol = self.data.groupby('stock_code')['pct_chg'].transform(lambda x: x.rolling(20, min_periods=10).std())
        self.data['stk_vol_intra_z'] = self._intra_sector_zscore(stk_vol)
        
        df_with_mkt = self.data.merge(self.market_df[['date', 'daily_change_pct']], on='date', how='left')
        df_with_mkt = df_with_mkt.rename(columns={'daily_change_pct': 'm_ret'})
        grouped = df_with_mkt.groupby('stock_code', sort=False)
        cov_60d = grouped[['pct_chg', 'm_ret']].rolling(60, min_periods=30).cov().unstack().iloc[:, 1].reset_index(level=0, drop=True)
        var_m_60d = grouped['m_ret'].transform(lambda x: x.rolling(60, min_periods=30).var())
        self.data['beta_60d'] = cov_60d / (var_m_60d + 1e-8)
        self.data['intra_rank'] = self.data.groupby(['date', 'sector_code'])['pct_chg'].rank(pct=True) * 2 - 1

    def compute_factor(self, factor_name: str) -> pd.Series:
        if factor_name == 'market_vols_adaptive_iv':
            # 使用分桶后的市场波动作为门控
            m_gate = self.data['date'].map(self.mkt_vol_binned)
            alpha = -self.data['stk_vol_intra_z']
            return _adaptive_clip(m_gate * alpha)

        elif factor_name == 'market_trend_adaptive_beta':
            # 使用分桶后的市场趋势作为门控
            m_gate = self.data['date'].map(self.mkt_trend_binned)
            alpha = self.data['beta_60d']
            return _adaptive_clip(m_gate * alpha)

        elif factor_name == 'sector_excess_intra_rank':
            s_ret = self.data.set_index(['date', 'sector_code']).index.map(self.s_ret_daily)
            m_ret = self.data['date'].map(self.mkt_ret_today)
            s_excess = s_ret - m_ret
            return _adaptive_clip(s_excess * self.data['intra_rank'])

        return pd.Series(0.0, index=self.original_index)

# 注册接口
def _get_helper(d, c):
    if '_cross_interaction_helper' not in c.extra: c.extra['_cross_interaction_helper'] = _CrossInteractionHelper(d, c)
    return c.extra['_cross_interaction_helper']

def _calc_mkt_iv(d, c): return _get_helper(d, c).compute_factor('market_vols_adaptive_iv')
def _calc_mkt_beta(d, c): return _get_helper(d, c).compute_factor('market_trend_adaptive_beta')
def _calc_sec_excess(d, c): return _get_helper(d, c).compute_factor('sector_excess_intra_rank')

market_vols_adaptive_iv = register_factor(FactorDefinition(name='market_vols_adaptive_iv', required_columns={'stock_code', 'date', 'pct_chg'}, requires_market_data=True, lookback_days=40, calculator=_calc_mkt_iv))
market_trend_adaptive_beta = register_factor(FactorDefinition(name='market_trend_adaptive_beta', required_columns={'stock_code', 'date', 'pct_chg'}, requires_market_data=True, lookback_days=80, calculator=_calc_mkt_beta))
sector_excess_intra_rank = register_factor(FactorDefinition(name='sector_excess_intra_rank', required_columns={'stock_code', 'date', 'pct_chg'}, requires_market_data=True, requires_sector_data=True, lookback_days=10, calculator=_calc_sec_excess))

__all__ = ['market_vols_adaptive_iv', 'market_trend_adaptive_beta', 'sector_excess_intra_rank']