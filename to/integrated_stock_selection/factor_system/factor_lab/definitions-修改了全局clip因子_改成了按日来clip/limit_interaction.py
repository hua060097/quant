"""
涨跌停与市场情绪交互因子 (Limit-Up/Down & Sentiment Interaction Factors) - 大师优化版
文件名: limit_interaction.py

优化日志：
1. CLV (Resilience) 修复：修正一字涨停导致的计算为0的问题，强制设为 1.0。
2. Market Mood 修复：保留 NaN 状态（市场冰点），区分于 0 溢价。
3. 因子计算流：标准化逻辑增强，确保稀疏因子和密集因子的处理路径分离。
4. [Critical] 修复 Clip 逻辑：使用按日 Clip + 稀疏保护，并对截面因子强制 Rank。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 1. 基础统计学工具
# ==============================================================================

# [修改 1] 替换全局 _adaptive_clip 为按日处理 + 稀疏保护
def _daily_adaptive_clip(series: pd.Series, date_series: pd.Series, n_mad: float = 2.0) -> pd.Series:
    """
    按日截面去极值 (含稀疏保护)
    n_mad 建议设为 2.0 以模拟旧模型的高信噪比截断效果
    """
    if series.empty: return series
    
    # 构造 DF
    df = pd.DataFrame({'val': series.values, 'date': date_series.values}, index=series.index)
    
    # 按日计算统计量
    medians = df.groupby('date')['val'].transform('median')
    # 注意：transform会自动广播回原索引
    
    # 计算绝对偏差
    abs_diff = (df['val'] - medians).abs()
    
    # 再次构造 DF 计算 MAD (偏差的中位数)
    temp_mad = pd.DataFrame({'diff': abs_diff.values, 'date': date_series.values}, index=series.index)
    mads = temp_mad.groupby('date')['diff'].transform('median')
    
    # 计算阈值
    threshold = n_mad * mads * 1.4826
    
    # [关键修复] 稀疏保护: 如果 MAD 极小，说明大部分是 0，不应截断
    # 设定一个极小阈值判定是否稀疏
    mask_sparse = (mads < 1e-6)
    
    # 对于稀疏日，阈值设为无穷大（即不截断）；对于正常日，保持计算值
    final_threshold = threshold.where(~mask_sparse, np.inf)
    
    return series.clip(medians - final_threshold, medians + final_threshold)

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
        # [关键修复] 必须先拷贝并排序，否则 groupby().shift() 可能会乱序
        self.data = data.copy()
        self.data.sort_values(['stock_code', 'date'], inplace=True)
        
        self.ctx = ctx
        self.original_index = data.index

        # 1. 行业映射
        sector_map = ctx.extra.get('stock_to_sw_sector', {})
        if not sector_map:
            self.data['sector_code'] = 'All'
        else:
            self.data['sector_code'] = self.data['stock_code'].map(sector_map).fillna('Unknown')
        
        # 2. 关键列预处理 (确保是 1/0 且无 NaN)
        cols_check = ['is_limit_up', 'is_limit_down', 'is_zhaban']
        for col in cols_check:
            if col not in self.data.columns:
                self.data[col] = 0
            else:
                self.data[col] = self.data[col].fillna(0).astype(int)

        # 3. 启动预计算
        self._precompute_metrics()

    def _intra_sector_standardize(self, series: pd.Series):
        """行业内标准化"""
        return self.data.assign(tmp=series).groupby(['date', 'sector_code'])['tmp'].transform(_safe_zscore)

    def _precompute_metrics(self):
        """预计算指标"""
        
        # --- A. 板块级情绪 (Sector Level) ---
        sector_stats = self.data.groupby(['date', 'sector_code'])[['is_limit_up', 'is_limit_down', 'is_zhaban']].sum()
        sector_counts = self.data.groupby(['date', 'sector_code'])['stock_code'].count()
        
        # A1. 板块热度 (3日均值)
        s_limit_up_ratio = (sector_stats['is_limit_up'] / (sector_counts + 1e-8))
        self.s_heat_rank = s_limit_up_ratio.unstack('sector_code').rolling(3, min_periods=1).mean().rank(axis=1, pct=True).stack()

        # A2. 板块分歧
        total_attempts = sector_stats['is_limit_up'] + sector_stats['is_zhaban']
        s_zhaban_ratio = sector_stats['is_zhaban'] / (total_attempts + 1e-8)
        self.s_divergence_rank = s_zhaban_ratio.unstack('sector_code').rolling(3, min_periods=1).mean().rank(axis=1, pct=True).stack()

        # --- B. 全市场级情绪 (Market Level) ---
        
        # [状态保留] 显式计算昨日涨停
        # 必须 fillna(0)，因为无数据意味着"非涨停"
        prev_limit_up = self.data.groupby('stock_code')['is_limit_up'].shift(1).fillna(0)
        self.data['is_prev_limit_up'] = prev_limit_up

        # 计算昨日涨停溢价 (只取 is_prev_limit_up == 1 的样本)
        limit_up_premium = self.data[self.data['is_prev_limit_up'] == 1].groupby('date')['pct_chg'].mean()
        
        # [大师级优化] 删除 fillna(0)
        # 如果某天无涨停股，这就是 NaN (市场冰点)，保留这个状态，不要伪造为 0
        self.market_mood = self.data['date'].map(limit_up_premium)

        # --- C. 个股级特征 (Stock Level) ---
        
        # C1. 活跃度 Z-Score
        self.data['active_z'] = self._intra_sector_standardize(self.data['pct_chg'])
        
        # C2. 换手活跃度
        if 'volume' in self.data.columns and 'circ_mv' in self.data.columns:
            to = self.data['volume'] / (self.data['circ_mv'] + 1e-8)
            self.data['turnover_z'] = self._intra_sector_standardize(to)
        else:
            self.data['turnover_z'] = self.data['active_z']

        # C3. 实体强度 (Intraday Strength)
        prev_close = self.data.groupby('stock_code')['close'].shift(1)
        # 用 Close - Open 衡量日内多空博弈
        self.data['intraday_strength'] = (self.data['close'] - self.data['open']) / (prev_close + 1e-8)
        
        # C4. 抵抗程度 (CLV - Close Location Value)
        # [大师级优化] 修复一字板问题
        range_len = self.data['high'] - self.data['low']
        
        # 基础计算 (Close - Low) / (High - Low)
        # 范围 0.0 ~ 1.0
        clv = (self.data['close'] - self.data['low']) / (range_len + 1e-8)
        
        # 修正：当 High == Low (range_len ~ 0) 时
        is_flat = range_len < 1e-5
        
        # 如果是一字涨停：应为最强 (1.0)
        # 如果是一字跌停：应为最弱 (0.0) -> 原公式分子也是0，结果为0，无需修正
        # 我们使用 numpy 的 where 进行向量化修正
        mask_one_bar_limit_up = is_flat & (self.data['is_limit_up'] == 1)
        
        # 应用修正：只有一字涨停才强制为 1.0，否则保持原值 (一字跌停或停牌保持0)
        self.data['clv'] = np.where(mask_one_bar_limit_up, 1.0, clv)
        
        # 标准化 CLV
        self.data['resilience_z'] = self._intra_sector_standardize(self.data['clv'])


    def get_factor(self, factor_name: str) -> pd.Series:
        """因子计算路由"""
        
        # 1. 计算原始因子值 (Raw Factor)
        if factor_name == 'sector_limit_overflow_alpha':
            # [密集因子] 溢出效应
            gate = self.data.join(self.s_heat_rank.rename('gate'), on=['date', 'sector_code'])['gate']
            alpha = self.data['turnover_z']
            filter_mask = (1 - self.data['is_limit_up']) * (1 - self.data['is_limit_down'])
            raw_factor = gate * alpha * filter_mask
            
            is_broadcast_factor = False # 是截面因子

        elif factor_name == 'sector_broken_resilience_alpha':
            # [密集因子] 抵抗效应
            gate = self.data.join(self.s_divergence_rank.rename('gate'), on=['date', 'sector_code'])['gate']
            alpha = self.data['resilience_z']
            raw_factor = gate * alpha
            
            is_broadcast_factor = False

        elif factor_name == 'limit_sustain_alpha':
            # [稀疏因子] 接力/核按钮
            gate = self.data['is_prev_limit_up']
            alpha = self.data['intraday_strength']
            raw_factor = gate * alpha
            
            is_broadcast_factor = False

        elif factor_name == 'market_emotion_resonance_alpha':
            # [广播因子] 情绪共振
            # 这是一个全市场信号，所有股票值相同（或接近）
            env = self.market_mood
            alpha = self.data['active_z']
            raw_factor = env * alpha
            
            # ⚠️ 注意：虽然名字像广播，但其实带了个股 alpha (active_z)。
            # 所以它其实是一个"随大盘情绪放大的个股动量"。
            # 依然可以做截面 Rank，因为它有个股差异。
            # 只有纯广播（如 market_trend）才不能做 Rank。
            is_broadcast_factor = False 

        else:
            raise ValueError(f"Unknown factor: {factor_name}")

        # === 2. 后处理 (Post-processing) ===
        
        # A. 填充空值
        joined_res = raw_factor.fillna(0)
        
        # B. 自适应去极值 (按日处理 + 稀疏保护)
        clipped_res = _daily_adaptive_clip(joined_res, self.data['date'], n_mad=2.0)
        
        # C. [最终步骤] 强制转为每日百分比排名 (0.0 ~ 1.0)
        # 逻辑：对于截面因子，Rank化是消除异常值影响、复现旧模型"只看相对强弱"特性的最佳手段。
        if not is_broadcast_factor:
            final_res = clipped_res.groupby(self.data['date']).rank(pct=True)
            # Rank 后的空值填 0.5 (中位数)
            return final_res.reindex(self.original_index).fillna(0.5)
        else:
            # 如果是纯广播因子（这里实际上没有），则保持原值
            return clipped_res.reindex(self.original_index).fillna(0)

# ==============================================================================
# 3. 注册因子
# ==============================================================================
def _get_helper(data: pd.DataFrame, ctx: FactorContext) -> _LimitInteractionHelper:
    key = '_limit_interaction_helper_v3_fixed' 
    if key not in ctx.extra:
        ctx.extra[key] = _LimitInteractionHelper(data, ctx)
    return ctx.extra[key]

# 包装器
def _c_overflow(d, c): return _get_helper(d, c).get_factor('sector_limit_overflow_alpha')
def _c_resilience(d, c): return _get_helper(d, c).get_factor('sector_broken_resilience_alpha')
def _c_sustain(d, c): return _get_helper(d, c).get_factor('limit_sustain_alpha')
def _c_resonance(d, c): return _get_helper(d, c).get_factor('market_emotion_resonance_alpha')

# 注册
sector_limit_overflow_alpha = register_factor(FactorDefinition(
    name='sector_limit_overflow_alpha',
    required_columns={'stock_code', 'date', 'is_limit_up', 'is_limit_down', 'volume', 'circ_mv'},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_c_overflow,
    description='板块高潮溢出因子(补涨)'
))

sector_broken_resilience_alpha = register_factor(FactorDefinition(
    name='sector_broken_resilience_alpha',
    required_columns={'stock_code', 'date', 'is_zhaban', 'high', 'low', 'close'},
    requires_sector_data=True,
    lookback_days=10,
    calculator=_c_resilience,
    description='板块分歧抵抗因子(抗分歧)'
))

limit_sustain_alpha = register_factor(FactorDefinition(
    name='limit_sustain_alpha',
    required_columns={'stock_code', 'date', 'is_limit_up', 'open', 'close'},
    requires_sector_data=False,
    lookback_days=10,
    calculator=_c_sustain,
    description='涨停次日接力/核按钮识别因子'
))

market_emotion_resonance_alpha = register_factor(FactorDefinition(
    name='market_emotion_resonance_alpha',
    required_columns={'stock_code', 'date', 'is_limit_up', 'pct_chg'},
    requires_sector_data=False,
    lookback_days=10,
    calculator=_c_resonance,
    description='市场情绪共振因子(昨日涨停溢价x个股活跃)'
))

__all__ = [
    'sector_limit_overflow_alpha',
    'sector_broken_resilience_alpha',
    'limit_sustain_alpha',
    'market_emotion_resonance_alpha'
]
