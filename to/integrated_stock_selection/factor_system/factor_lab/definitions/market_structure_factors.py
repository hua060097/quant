"""
市场结构与行业宽度因子 (Market Structure & Sector Breadth Factors) - 时序去极值版

优化说明：
1. 核心修复：引入 _rolling_adaptive_clip (60日) 替代全局 Clip，消除未来函数并保留幅度。
2. mkt_sector_dispersion: 豁免 Clip 操作，保留全市场状态的原始波动性。
3. 适配 120 日数据窗口，降低 min_periods 门槛。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 1. 基础统计学工具 (时序滚动去极值)
# ==============================================================================

def _rolling_adaptive_clip(series: pd.Series, date_series: pd.Series, n_mad: float = 3.0, window: int = 120) -> pd.Series:
    """
    时序滚动去极值 (Rolling Adaptive Clip)
    逻辑：利用过去 N 天(60天)的数据分布来决定今天的截断阈值。
    """
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
    
    # 6. 执行截断
    upper = upper.fillna(np.inf)
    lower = lower.fillna(-np.inf)
    
    return series.clip(lower, upper)

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
        # 4. 价格位置共振
        window_size = 120
        min_p = 40
        
        high_anchor = self.data.groupby('stock_code')['high'].transform(lambda x: x.rolling(window_size, min_periods=min_p).max())
        low_anchor = self.data.groupby('stock_code')['low'].transform(lambda x: x.rolling(window_size, min_periods=min_p).min())
        
        # 个股位置
        self.data['stock_pos'] = (self.data['close'] - low_anchor) / (high_anchor - low_anchor + 1e-8)
        
        # 行业平均位置
        self.s_pos_mean = self.data.groupby(['date', 'sector_code'])['stock_pos'].mean().unstack('sector_code')

    def compute(self, factor_name: str) -> pd.Series:
        """执行因子映射逻辑"""
        x = 3.1
        if factor_name == 'mkt_sector_dispersion':
            # [特殊处理] 这是一个全市场广播因子，每天只有一个值。
            # 不要进行 Clip，直接保留原始数值，反映市场状态。
            return self.data['date'].map(self.mkt_sector_dispersion)

        elif factor_name == 'sector_mkt_corr_20d':
            s_val = self.s_mkt_corr.stack().rename('v')
            raw = self.data.join(s_val, on=['date', 'sector_code'], how='left')['v']
            # 使用 Rolling Clip  多改了一个window 60到120,下面三个都是   18:17
            return _rolling_adaptive_clip(raw, self.data['date'], n_mad=x, window=120)

        elif factor_name == 'intra_sector_breadth':
            s_val = self.s_breadth.stack().rename('v')
            raw = self.data.join(s_val, on=['date', 'sector_code'], how='left')['v']
            # 使用 Rolling Clip
            return _rolling_adaptive_clip(raw, self.data['date'], n_mad=x, window=120)

        elif factor_name == 'stock_sector_pos_align':
            s_pos = self.s_pos_mean.stack().rename('sp')
            joined = self.data.join(s_pos, on=['date', 'sector_code'], how='left')
            raw = joined['stock_pos'] - joined['sp']
            # 使用 Rolling Clip
            return _rolling_adaptive_clip(raw, self.data['date'], n_mad=x, window=120)

        return pd.Series(0.0, index=self.original_index)

# ==============================================================================
# 3. 注册逻辑
# ==============================================================================
def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _MarketStructureHelper:
    key = '_market_structure_helper_rolling'
    if key not in ctx.extra:
        ctx.extra[key] = _MarketStructureHelper(data, ctx)
    return ctx.extra[key]

def _calc_disp(d, c): return _get_helper(d, c).compute('mkt_sector_dispersion')
def _calc_corr(d, c): return _get_helper(d, c).compute('sector_mkt_corr_20d')
def _calc_breadth(d, c): return _get_helper(d, c).compute('intra_sector_breadth')
def _calc_align(d, c): return _get_helper(d, c).compute('stock_sector_pos_align')

# 因子注册
mkt_sector_dispersion = register_factor(FactorDefinition(name='mkt_sector_dispersion', required_columns={'date', 'pct_chg'}, requires_sector_data=True, lookback_days=30, calculator=_calc_disp))
sector_mkt_corr_20d = register_factor(FactorDefinition(name='sector_mkt_corr_20d', required_columns={'date', 'pct_chg'}, requires_market_data=True, requires_sector_data=True, lookback_days=40, calculator=_calc_corr))
intra_sector_breadth = register_factor(FactorDefinition(name='intra_sector_breadth', required_columns={'date', 'pct_chg'}, requires_sector_data=True, lookback_days=10, calculator=_calc_breadth))
stock_sector_pos_align = register_factor(FactorDefinition(name='stock_sector_pos_align', required_columns={'stock_code', 'date', 'high', 'low', 'close'}, requires_sector_data=True, lookback_days=120, calculator=_calc_align))

__all__ = ['mkt_sector_dispersion', 'sector_mkt_corr_20d', 'intra_sector_breadth', 'stock_sector_pos_align']
