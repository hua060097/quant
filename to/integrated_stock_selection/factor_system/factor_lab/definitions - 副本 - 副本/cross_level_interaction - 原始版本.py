"""
跨层级交互因子 (Cross-Level Interaction Factors)

核心逻辑：
1. 方向二：个股 * 市场 (Stock-Market Interaction) —— 引入市场波动环境与趋势强度，动态调整风险暴露。
2. 方向三：三位一体 (Stock-Sector-Market) —— 识别处于超额收益板块中的头部个股。

日期：2025-01-25
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 1. 辅助工具函数
# ==============================================================================
def _adaptive_clip(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    """自适应去极值（稳健统计学 MAD 法）"""
    if series.empty:
        return series
    median = series.median()
    mad = (series - median).abs().median()
    threshold = n_mad * mad * 1.4826
    return series.clip(median - threshold, median + threshold)

def _get_market_data(ctx: FactorContext) -> pd.DataFrame:
    """获取对齐后的市场数据"""
    market = ctx.require_market()
    return market.sort_values('date').reset_index(drop=True)

# ==============================================================================
# 2. 核心交互计算类
# ==============================================================================
class _CrossInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.data = data.copy()
        self.ctx = ctx
        self.original_index = data.index

        # 获取行业映射
        sector_map = ctx.extra.get('stock_to_sw_sector', {})
        self.data['sector_code'] = self.data['stock_code'].map(sector_map).fillna('Unknown')
        
        # 获取市场数据
        self.market_df = _get_market_data(ctx)
        
        # 预计算所有跨层级指标
        self._precompute()

    def _intra_sector_zscore(self, series: pd.Series) -> pd.Series:
        """行业内按日 Z-Score 标准化算子"""
        return self.data.assign(tmp=series).groupby(['date', 'sector_code'])['tmp'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    def _precompute(self):
        """预计算市场背景、行业趋势与个股敏感度指标"""
        # --- A. 市场维度 (Market Level) ---
        mkt = self.market_df.set_index('date')
        # 市场波动环境 (20日年化波动率代理值)
        self.mkt_vol = mkt['daily_change_pct'].rolling(20, min_periods=10).std()
        # 市场趋势强度 (20日累计收益)
        self.mkt_trend = mkt['daily_change_pct'].rolling(20, min_periods=10).sum()
        # 市场当日收益 (用于计算超额)
        self.mkt_ret_today = mkt['daily_change_pct']

        # --- B. 行业维度 (Sector Level) ---
        # 行业日均收益率
        self.s_ret_daily = self.data.groupby(['date', 'sector_code'])['pct_chg'].mean()

        # --- C. 个股与交互基础维度 (Stock Level) ---
        # 1. 20日个股波动率 (IV代理项)
        stk_vol = self.data.groupby('stock_code')['pct_chg'].transform(lambda x: x.rolling(20, min_periods=10).std())
        self.data['stk_vol_intra_z'] = self._intra_sector_zscore(stk_vol)
        
        # 2. 60日个股 Beta (相对于大盘)
        # 合并市场收益以计算协方差
        df_with_mkt = self.data.merge(self.market_df[['date', 'daily_change_pct']], on='date', how='left')
        df_with_mkt = df_with_mkt.rename(columns={'daily_change_pct': 'm_ret'})
        
        grouped = df_with_mkt.groupby('stock_code', sort=False)
        cov_60d = grouped[['pct_chg', 'm_ret']].rolling(60, min_periods=30).cov().unstack().iloc[:, 1].reset_index(level=0, drop=True)
        var_m_60d = grouped['m_ret'].transform(lambda x: x.rolling(60, min_periods=30).var())
        self.data['beta_60d'] = cov_60d / (var_m_60d + 1e-8)
        
        # 3. 行业内当日排名 (Intra-Sector Rank)
        self.data['intra_rank'] = self.data.groupby(['date', 'sector_code'])['pct_chg'].rank(pct=True) * 2 - 1

    def compute_factor(self, factor_name: str) -> pd.Series:
        """根据名称执行具体的交互逻辑"""
        
        if factor_name == 'market_vols_adaptive_iv':
            # 特征 3: 市场波动 * (-个股特质波动)
            # 逻辑: 大波动环境惩罚高波股
            m_gate = self.data['date'].map(self.mkt_vol)
            alpha = -self.data['stk_vol_intra_z']
            return _adaptive_clip(m_gate * alpha)

        elif factor_name == 'market_trend_adaptive_beta':
            # 特征 4: 市场趋势强度 * 个股 Beta
            # 逻辑: 顺境买高Beta, 逆境买低Beta
            m_gate = self.data['date'].map(self.mkt_trend)
            alpha = self.data['beta_60d']
            return _adaptive_clip(m_gate * alpha)

        elif factor_name == 'sector_excess_intra_rank':
            # 特征 5: 行业超额收益 * 个股行业内排名
            # 逻辑: 寻找强板块中的领涨股 (三位一体)
            s_ret = self.data.set_index(['date', 'sector_code']).index.map(self.s_ret_daily)
            m_ret = self.data['date'].map(self.mkt_ret_today)
            s_excess = s_ret - m_ret
            alpha = self.data['intra_rank']
            return _adaptive_clip(s_excess * alpha)

        return pd.Series(0.0, index=self.original_index)

# ==============================================================================
# 3. 注册与外部接口
# ==============================================================================
def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _CrossInteractionHelper:
    key = '_cross_interaction_helper'
    if key not in ctx.extra:
        ctx.extra[key] = _CrossInteractionHelper(data, ctx)
    return ctx.extra[key]

# 包装器函数
def _calc_mkt_iv(d, c): return _get_helper(d, c).compute_factor('market_vols_adaptive_iv')
def _calc_mkt_beta(d, c): return _get_helper(d, c).compute_factor('market_trend_adaptive_beta')
def _calc_sec_excess(d, c): return _get_helper(d, c).compute_factor('sector_excess_intra_rank')

# 因子注册
market_vols_adaptive_iv = register_factor(FactorDefinition(
    name='market_vols_adaptive_iv',
    required_columns={'stock_code', 'date', 'pct_chg'},
    requires_market_data=True,
    lookback_days=40,
    calculator=_calc_mkt_iv,
    description='环境自适应波动因子: 市场高波动时倾向低波股'
))

market_trend_adaptive_beta = register_factor(FactorDefinition(
    name='market_trend_adaptive_beta',
    required_columns={'stock_code', 'date', 'pct_chg'},
    requires_market_data=True,
    lookback_days=80, # Beta需要更长的回看窗口
    calculator=_calc_mkt_beta,
    description='趋势自适应贝塔: 市场走强时倾向高Beta股'
))

sector_excess_intra_rank = register_factor(FactorDefinition(
    name='sector_excess_intra_rank',
    required_columns={'stock_code', 'date', 'pct_chg'},
    requires_market_data=True,
    requires_sector_data=True,
    lookback_days=10,
    calculator=_calc_sec_excess,
    description='三位一体领先因子: 行业跑赢大盘且个股领涨行业'
))