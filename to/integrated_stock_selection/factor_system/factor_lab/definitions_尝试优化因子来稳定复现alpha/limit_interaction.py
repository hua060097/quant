"""
涨跌停与情绪结构交互因子
【SW 二级行业 · 稳定版 · Trigger 友好】
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 工具函数
# ==============================================================================

def _safe_log1p(x):
    return np.log1p(np.clip(x, 0, None))

def _safe_tanh(x, scale=1.0):
    return np.tanh(x / scale)

def _cs_rank_pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)

# ==============================================================================
# 主计算器
# ==============================================================================

class _LimitInteractionHelper:

    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.data = data.copy()
        self.ctx = ctx

        sector_map = ctx.extra.get('stock_to_sw_sector', {})
        self.data['sector_code'] = self.data['stock_code'].map(sector_map).fillna('ALL')

        for c in ['is_limit_up', 'is_limit_down', 'is_zhaban']:
            if c not in self.data:
                self.data[c] = 0
            self.data[c] = self.data[c].fillna(0).astype(int)

        self.data.sort_values(['date', 'stock_code'], inplace=True)
        self._prepare()

    def _prepare(self):
        # --- sector ---
        sec_limit = self.data.groupby(['date', 'sector_code'])['is_limit_up'].transform('sum')
        sec_cnt = self.data.groupby(['date', 'sector_code'])['stock_code'].transform('count')

        self.data['sector_limit_strength'] = _safe_log1p(sec_limit)
        self.data['sector_cnt'] = sec_cnt.clip(lower=1)

        # --- market ---
        mkt_limit = self.data.groupby('date')['is_limit_up'].transform('sum')
        self.data['market_limit_strength'] = _safe_log1p(mkt_limit)

    # --------------------------------------------------------------------------
    # 1. 板块涨停溢出
    # --------------------------------------------------------------------------

    def sector_limit_overflow_alpha(self):
        raw = (
            self.data['is_limit_up']
            * self.data['sector_limit_strength']
            / self.data['sector_cnt']
        )
        alpha = _safe_tanh(raw, scale=1.5)
        return self.data.groupby('date').apply(
            lambda x: _cs_rank_pct(alpha.loc[x.index])
        ).reset_index(level=0, drop=True)

    # --------------------------------------------------------------------------
    # 2. 市场情绪共振
    # --------------------------------------------------------------------------

    def market_emotion_resonance_alpha(self):
        raw = self.data['is_limit_up'] * self.data['market_limit_strength']
        alpha = _safe_tanh(raw, scale=2.0)
        return self.data.groupby('date').apply(
            lambda x: _cs_rank_pct(alpha.loc[x.index])
        ).reset_index(level=0, drop=True)

    # --------------------------------------------------------------------------
    # 3. 炸板韧性（SW 二级行业稳定版）
    # --------------------------------------------------------------------------

    def sector_broken_resilience_alpha(self):
        # 炸板后“仍有涨停”的概率近似
        broken = self.data['is_zhaban']
        limit = self.data['is_limit_up']

        sec_broken = self.data.groupby(['date', 'sector_code'])['is_zhaban'].transform('sum')
        sec_limit = self.data.groupby(['date', 'sector_code'])['is_limit_up'].transform('sum')

        # 使用 log 平滑 + 避免比例爆炸
        resilience = (
            broken
            * _safe_log1p(sec_limit + 1)
            / _safe_log1p(sec_broken + 2)
        )

        alpha = _safe_tanh(resilience, scale=1.2)
        return self.data.groupby('date').apply(
            lambda x: _cs_rank_pct(alpha.loc[x.index])
        ).reset_index(level=0, drop=True)

    # --------------------------------------------------------------------------
    # 4. 连板持续度（饱和型）
    # --------------------------------------------------------------------------

    def limit_sustain_alpha(self):
        # 使用 rolling 连续涨停次数，但做饱和
        cont_limit = (
            self.data.groupby('stock_code')['is_limit_up']
            .rolling(5, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        sustain = _safe_tanh(cont_limit, scale=2.0)
        return self.data.groupby('date').apply(
            lambda x: _cs_rank_pct(sustain.loc[x.index])
        ).reset_index(level=0, drop=True)

# ==============================================================================
# 对外因子接口
# ==============================================================================

def _sector_limit_overflow_alpha(data, ctx):
    return _LimitInteractionHelper(data, ctx).sector_limit_overflow_alpha()

def _market_emotion_resonance_alpha(data, ctx):
    return _LimitInteractionHelper(data, ctx).market_emotion_resonance_alpha()

def _sector_broken_resilience_alpha(data, ctx):
    return _LimitInteractionHelper(data, ctx).sector_broken_resilience_alpha()

def _limit_sustain_alpha(data, ctx):
    return _LimitInteractionHelper(data, ctx).limit_sustain_alpha()

# ==============================================================================
# 注册
# ==============================================================================

sector_limit_overflow_alpha = register_factor(FactorDefinition(
    name='sector_limit_overflow_alpha',
    required_columns={'stock_code', 'date', 'is_limit_up'},
    lookback_days=5,
    calculator=_sector_limit_overflow_alpha,
    description='板块涨停溢出强度（SW二级·稳定版）'
))

market_emotion_resonance_alpha = register_factor(FactorDefinition(
    name='market_emotion_resonance_alpha',
    required_columns={'stock_code', 'date', 'is_limit_up'},
    lookback_days=5,
    calculator=_market_emotion_resonance_alpha,
    description='市场情绪共振（稳定饱和）'
))

sector_broken_resilience_alpha = register_factor(FactorDefinition(
    name='sector_broken_resilience_alpha',
    required_columns={'stock_code', 'date', 'is_limit_up', 'is_zhaban'},
    lookback_days=5,
    calculator=_sector_broken_resilience_alpha,
    description='板块炸板韧性（SW二级·稳定版）'
))

limit_sustain_alpha = register_factor(FactorDefinition(
    name='limit_sustain_alpha',
    required_columns={'stock_code', 'date', 'is_limit_up'},
    lookback_days=6,
    calculator=_limit_sustain_alpha,
    description='连板持续度（饱和型）'
))

__all__ = [
    'sector_limit_overflow_alpha',
    'market_emotion_resonance_alpha',
    'sector_broken_resilience_alpha',
    'limit_sustain_alpha'
]
