"""
市场结构与行业宽度因子 (Market Structure & Sector Breadth Factors) - 120日数据优化版

优化说明：
1. 将 stock_sector_pos_align 的 250 日锚点调整为 120 日（半年线）或 60 日（季线）。
2. 调低 min_periods 门槛，确保在历史数据有限时能更早产出因子。
3. 调整 lookback_days 注册参数，适配 120 日数据限制。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 1. 辅助函数
# ==============================================================================
def _adaptive_clip(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    """自适应去极值"""
    if series.empty: return series
    median = series.median()
    mad = (series - median).abs().median()
    threshold = n_mad * mad * 1.4826
    return series.clip(median - threshold, median + threshold)

def _get_market_data(ctx: FactorContext) -> pd.DataFrame:
    """获取市场基准数据"""
    market = ctx.require_market()
    return market.sort_values('date').reset_index(drop=True)

# ==============================================================================
# 2. 市场结构计算辅助类
# ==============================================================================
class _MarketStructureHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.data = data.copy()
        self.ctx = ctx
        self.original_index = data.index

        # 行业映射注入
        sector_map = ctx.extra.get('stock_to_sw_sector', {})
        self.data['sector_code'] = self.data['stock_code'].map(sector_map).fillna('Unknown')
        
        # 预计算
        self._precompute()

    def _precompute(self):
        """向量化预计算市场与行业层级指标"""
        # --- A. 基础数据准备 ---
        s_ret_wide = self.data.groupby(['date', 'sector_code'])['pct_chg'].mean().unstack('sector_code').sort_index()
        mkt_df = _get_market_data(self.ctx).set_index('date')
        mkt_ret = mkt_df['daily_change_pct']

        # --- B. 市场/行业层级计算 ---
        # 1. 市场级：全市场行业离散度
        self.mkt_sector_dispersion = s_ret_wide.std(axis=1)
        
        # 2. 行业-市场：行业与大盘相关性
        self.s_mkt_corr = s_ret_wide.rolling(20, min_periods=10).corr(mkt_ret)

        # 3. 行业级：行业内上涨股票占比 (Breadth)
        is_positive = (self.data['pct_chg'] > 0).astype(int)
        self.s_breadth = self.data.assign(is_pos=is_positive).groupby(['date', 'sector_code'])['is_pos'].mean().unstack('sector_code')

        # --- C. 个股-行业层级计算 (针对 120 日数据优化) ---
        # 4. 价格位置共振 (由 250 日改为 120 日锚点)
        # 120日约等于半年线位置，是 A 股中线逻辑的重要参考
        window_size = 120
        min_p = 40  # 降低准入门槛，有 2 个月数据即可计算位置
        
        high_anchor = self.data.groupby('stock_code')['high'].transform(lambda x: x.rolling(window_size, min_periods=min_p).max())
        low_anchor = self.data.groupby('stock_code')['low'].transform(lambda x: x.rolling(window_size, min_periods=min_p).min())
        
        # 个股在过去 120 日内的相对位置 (0-1)
        # 优化：加入极小值防止除以 0，同时处理新股数据不足的情况
        self.data['stock_pos'] = (self.data['close'] - low_anchor) / (high_anchor - low_anchor + 1e-8)
        
        # 行业平均位置 (反映行业处于周期的什么阶段)
        self.s_pos_mean = self.data.groupby(['date', 'sector_code'])['stock_pos'].mean().unstack('sector_code')

    def compute(self, factor_name: str) -> pd.Series:
        """执行因子映射逻辑"""
        
        if factor_name == 'mkt_sector_dispersion':
            return _adaptive_clip(self.data['date'].map(self.mkt_sector_dispersion).fillna(0))

        elif factor_name == 'sector_mkt_corr_20d':
            s_val = self.s_mkt_corr.stack().rename('v')
            return _adaptive_clip(self.data.join(s_val, on=['date', 'sector_code'], how='left')['v'].fillna(0))

        elif factor_name == 'intra_sector_breadth':
            s_val = self.s_breadth.stack().rename('v')
            return _adaptive_clip(self.data.join(s_val, on=['date', 'sector_code'], how='left')['v'].fillna(0))

        elif factor_name == 'stock_sector_pos_align':
            # 个股 120 日位置 vs 行业 120 日平均位置
            s_pos = self.s_pos_mean.stack().rename('sp')
            joined = self.data.join(s_pos, on=['date', 'sector_code'], how='left')
            # 这里的逻辑是：即便行业整体在低位，先行反弹的个股会获得极高分
            return _adaptive_clip(joined['stock_pos'] - joined['sp'].fillna(0))

        return pd.Series(0.0, index=self.original_index)

# ==============================================================================
# 3. 注册逻辑
# ==============================================================================
def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _MarketStructureHelper:
    key = '_market_structure_helper'
    if key not in ctx.extra:
        ctx.extra[key] = _MarketStructureHelper(data, ctx)
    return ctx.extra[key]

def _calc_disp(d, c): return _get_helper(d, c).compute('mkt_sector_dispersion')
def _calc_corr(d, c): return _get_helper(d, c).compute('sector_mkt_corr_20d')
def _calc_breadth(d, c): return _get_helper(d, c).compute('intra_sector_breadth')
def _calc_align(d, c): return _get_helper(d, c).compute('stock_sector_pos_align')

# 因子注册
# 🔧 优化：lookback_days 由 260 降至 130，确保 120 天数据环境下能运行
mkt_sector_dispersion = register_factor(FactorDefinition(name='mkt_sector_dispersion', required_columns={'date', 'pct_chg'}, requires_sector_data=True, lookback_days=30, calculator=_calc_disp))
sector_mkt_corr_20d = register_factor(FactorDefinition(name='sector_mkt_corr_20d', required_columns={'date', 'pct_chg'}, requires_market_data=True, requires_sector_data=True, lookback_days=40, calculator=_calc_corr))
intra_sector_breadth = register_factor(FactorDefinition(name='intra_sector_breadth', required_columns={'date', 'pct_chg'}, requires_sector_data=True, lookback_days=10, calculator=_calc_breadth))
stock_sector_pos_align = register_factor(FactorDefinition(name='stock_sector_pos_align', required_columns={'stock_code', 'date', 'high', 'low', 'close'}, requires_sector_data=True, lookback_days=120, calculator=_calc_align))

__all__ = ['mkt_sector_dispersion', 'sector_mkt_corr_20d', 'intra_sector_breadth', 'stock_sector_pos_align']