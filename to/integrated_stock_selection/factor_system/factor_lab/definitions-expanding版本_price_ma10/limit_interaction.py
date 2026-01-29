"""
涨跌停与市场情绪交互因子 (Limit-Up/Down & Sentiment Interaction Factors) - 时序去极值版
文件名: limit_interaction.py

优化日志：
1. 核心修复：使用 _rolling_adaptive_clip (60日窗口) 替代全局 Clip。
   - 目的：规避未来函数，同时保留时序波动率特征（复现旧模型对极端行情的敏感度）。
2. CLV (Resilience) 修复：修正一字涨停问题。
3. Market Mood 修复：保留 NaN 状态。
4. 稀疏保护：防止涨停接力因子被误杀。
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
    
    # 2. 计算每日截面统计量 (Daily Cross-Sectional Stats)
    # 我们需要知道每一天：中位数是多少？波动幅度(MAD)是多少？
    daily_stats = df.groupby('date')['val'].agg(
        median='median',
        mad=lambda x: (x - x.median()).abs().median()
    ).sort_index() # 按日期排序
    
    # 3. 计算历史平滑阈值 (Rolling Statistics)
    # 使用 Expanding + Rolling 结合，或纯 Rolling。这里用 Rolling(60)。
    # min_periods=1 保证初期数据也能计算（类似 Expanding 效果）
    # shift(1) 是为了严格防未来：今天的阈值只能由“昨天及以前”的数据决定。
    # 但考虑到截面统计量本身不包含个股未来，用当日截面统计量作为基准也是合规的截面处理。
    # 为了最强稳健性，模拟旧模型的"积累"效应，我们使用包含当日的 Rolling 平滑。
    
    # rolling_stats = daily_stats.rolling(window=window, min_periods=1).median()
    rolling_stats = daily_stats.expanding(min_periods=1).median()
    
    # 4. 映射回个股数据
    # 将计算好的"历史平滑基准"映射回每一行
    # 注意：map 会自动对齐日期索引
    df['thresh_median'] = df['date'].map(rolling_stats['median'])
    df['thresh_mad'] = df['date'].map(rolling_stats['mad'])
    
    # 5. 稀疏保护 (Limit Sustain 专用)
    # 如果历史波动率极低 (MAD ~ 0)，说明大部分时候是0，此时不应截断非0值
    mask_sparse = (df['thresh_mad'] < 1e-6)
    
    # 计算最终上下界
    # 阈值 = 平滑中位数 +/- n * 平滑MAD
    half_width = n_mad * df['thresh_mad'] * 1.4826
    
    # 稀疏日阈值设为无穷大
    final_half_width = half_width.where(~mask_sparse, np.inf)
    
    upper = df['thresh_median'] + final_half_width
    lower = df['thresh_median'] - final_half_width
    
    # 6. 执行截断
    # 填充 NaN (比如前几天可能没有 rolling 结果，虽然 min_periods=1 解决了这个问题)
    # 使用 ffill 处理极少数边缘情况
    upper = upper.fillna(np.inf)
    lower = lower.fillna(-np.inf)
    
    return series.clip(lower, upper)

def _safe_zscore(series: pd.Series) -> pd.Series:
    """安全的 Z-Score"""
    if series.empty or series.std() == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / (series.std() + 1e-8)

# ==============================================================================
# 2. 核心计算引擎
# ==============================================================================
class _LimitInteractionHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext):
        self.data = data.copy()
        self.data.sort_values(['stock_code', 'date'], inplace=True)
        self.ctx = ctx
        self.original_index = data.index

        sector_map = ctx.extra.get('stock_to_sw_sector', {})
        if not sector_map:
            self.data['sector_code'] = 'All'
        else:
            self.data['sector_code'] = self.data['stock_code'].map(sector_map).fillna('Unknown')
        
        cols_check = ['is_limit_up', 'is_limit_down', 'is_zhaban']
        for col in cols_check:
            if col not in self.data.columns:
                self.data[col] = 0
            else:
                self.data[col] = self.data[col].fillna(0).astype(int)

        self._precompute_metrics()

    def _intra_sector_standardize(self, series: pd.Series):
        return self.data.assign(tmp=series).groupby(['date', 'sector_code'])['tmp'].transform(_safe_zscore)

    def _precompute_metrics(self):
        sector_stats = self.data.groupby(['date', 'sector_code'])[['is_limit_up', 'is_limit_down', 'is_zhaban']].sum()
        sector_counts = self.data.groupby(['date', 'sector_code'])['stock_code'].count()
        
        s_limit_up_ratio = (sector_stats['is_limit_up'] / (sector_counts + 1e-8))
        self.s_heat_rank = s_limit_up_ratio.unstack('sector_code').rolling(3, min_periods=1).mean().rank(axis=1, pct=True).stack()

        total_attempts = sector_stats['is_limit_up'] + sector_stats['is_zhaban']
        s_zhaban_ratio = sector_stats['is_zhaban'] / (total_attempts + 1e-8)
        self.s_divergence_rank = s_zhaban_ratio.unstack('sector_code').rolling(3, min_periods=1).mean().rank(axis=1, pct=True).stack()

        prev_limit_up = self.data.groupby('stock_code')['is_limit_up'].shift(1).fillna(0)
        self.data['is_prev_limit_up'] = prev_limit_up
        limit_up_premium = self.data[self.data['is_prev_limit_up'] == 1].groupby('date')['pct_chg'].mean()
        self.market_mood = self.data['date'].map(limit_up_premium)

        self.data['active_z'] = self._intra_sector_standardize(self.data['pct_chg'])
        
        if 'volume' in self.data.columns and 'circ_mv' in self.data.columns:
            to = self.data['volume'] / (self.data['circ_mv'] + 1e-8)
            self.data['turnover_z'] = self._intra_sector_standardize(to)
        else:
            self.data['turnover_z'] = self.data['active_z']

        prev_close = self.data.groupby('stock_code')['close'].shift(1)
        self.data['intraday_strength'] = (self.data['close'] - self.data['open']) / (prev_close + 1e-8)
        
        range_len = self.data['high'] - self.data['low']
        clv = (self.data['close'] - self.data['low']) / (range_len + 1e-8)
        is_flat = range_len < 1e-5
        mask_one_bar_limit_up = is_flat & (self.data['is_limit_up'] == 1)
        self.data['clv'] = np.where(mask_one_bar_limit_up, 1.0, clv)
        self.data['resilience_z'] = self._intra_sector_standardize(self.data['clv'])

    def get_factor(self, factor_name: str) -> pd.Series:
        if factor_name == 'sector_limit_overflow_alpha':
            gate = self.data.join(self.s_heat_rank.rename('gate'), on=['date', 'sector_code'])['gate']
            alpha = self.data['turnover_z']
            filter_mask = (1 - self.data['is_limit_up']) * (1 - self.data['is_limit_down'])
            raw_factor = gate * alpha * filter_mask
            need_sector_neutral = True

        elif factor_name == 'sector_broken_resilience_alpha':
            gate = self.data.join(self.s_divergence_rank.rename('gate'), on=['date', 'sector_code'])['gate']
            alpha = self.data['resilience_z']
            raw_factor = gate * alpha
            need_sector_neutral = True

        elif factor_name == 'limit_sustain_alpha':
            gate = self.data['is_prev_limit_up']
            alpha = self.data['intraday_strength']
            raw_factor = gate * alpha
            need_sector_neutral = False

        elif factor_name == 'market_emotion_resonance_alpha':
            env = self.market_mood
            alpha = self.data['active_z']
            raw_factor = env * alpha
            need_sector_neutral = False

        else:
            raise ValueError(f"Unknown factor: {factor_name}")

        # === 2. 后处理 (Post-processing) ===
        
        # A. 填充空值
        joined_res = raw_factor.fillna(0)
        
        # B. [核心修改] 时序滚动去极值 (Rolling Adaptive Clip)
        # 窗口 60 天，n_mad=3.0 (保留更多幅度信息) 多改了一个window 60到120 18:16
        clipped_res = _rolling_adaptive_clip(joined_res, self.data['date'], n_mad=3.0, window=120)
        
        # C. 标准化分流
        # 注意：不再强制 Rank，而是保留数值幅度
        if need_sector_neutral:
            # 密集因子：做行业内 Z-Score
            final_res = self.data.assign(res=clipped_res).groupby(['date', 'sector_code'])['res'].transform(_safe_zscore)
        else:
            # 稀疏/绝对因子：不做处理，直接返回 Clip 后的值
            # 这里的 global_std 逻辑被移除，因为 rolling_clip 已经控制了幅度
            final_res = clipped_res

        return final_res.reindex(self.original_index).fillna(0)

# ==============================================================================
# 3. 注册因子
# ==============================================================================
def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _LimitInteractionHelper:
    key = '_limit_interaction_helper_v3_rolling' 
    if key not in ctx.extra:
        ctx.extra[key] = _LimitInteractionHelper(data, ctx)
    return ctx.extra[key]

def _c_overflow(d, c): return _get_helper(d, c).get_factor('sector_limit_overflow_alpha')
def _c_resilience(d, c): return _get_helper(d, c).get_factor('sector_broken_resilience_alpha')
def _c_sustain(d, c): return _get_helper(d, c).get_factor('limit_sustain_alpha')
def _c_resonance(d, c): return _get_helper(d, c).get_factor('market_emotion_resonance_alpha')

sector_limit_overflow_alpha = register_factor(FactorDefinition(name='sector_limit_overflow_alpha', required_columns={'stock_code', 'date', 'is_limit_up', 'is_limit_down', 'volume', 'circ_mv'}, requires_sector_data=True, lookback_days=10, calculator=_c_overflow))
sector_broken_resilience_alpha = register_factor(FactorDefinition(name='sector_broken_resilience_alpha', required_columns={'stock_code', 'date', 'is_zhaban', 'high', 'low', 'close'}, requires_sector_data=True, lookback_days=10, calculator=_c_resilience))
limit_sustain_alpha = register_factor(FactorDefinition(name='limit_sustain_alpha', required_columns={'stock_code', 'date', 'is_limit_up', 'open', 'close'}, requires_sector_data=False, lookback_days=10, calculator=_c_sustain))
market_emotion_resonance_alpha = register_factor(FactorDefinition(name='market_emotion_resonance_alpha', required_columns={'stock_code', 'date', 'is_limit_up', 'pct_chg'}, requires_sector_data=False, lookback_days=10, calculator=_c_resonance))

__all__ = ['sector_limit_overflow_alpha', 'sector_broken_resilience_alpha', 'limit_sustain_alpha', 'market_emotion_resonance_alpha']
