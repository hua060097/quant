"""K线与成交量相关因子。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series


# ==============================================================================
# 1. 实体力度 (Body Efficiency) - 趋势的坚决程度
# ==============================================================================
def _body_efficiency(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    K线实体效率 (优化版) - 彻底解决一字板识别与十字星冲突
    """
    if 'close' not in data.columns or 'pct_chg' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    # 1. 计算基础几何属性
    hl_range = data['high'] - data['low']
    body_range = data['close'] - data['open']
    
    # 2. 正常情况下的效率计算
    # 逻辑：位移 / 振幅
    efficiency = body_range / (hl_range + 1e-9)
    
    # 3. 处理一字板 (High == Low)
    # 判断条件：振幅极小
    is_flat = hl_range < 1e-4
    
    # [修正点]：利用涨跌幅判断一字板的性质
    # - 一字涨停：效率设为 1.0 (纯买盘)
    # - 一字跌停：效率设为 -1.0 (纯卖盘)
    # - 停牌或一字平盘：效率设为 0.0 (无波动)
    efficiency = np.where(is_flat & (data['pct_chg'] > 0), 1.0, efficiency)
    efficiency = np.where(is_flat & (data['pct_chg'] < 0), -1.0, efficiency)
    efficiency = np.where(is_flat & (data['pct_chg'] == 0), 0.0, efficiency)
    
    return pd.Series(efficiency, index=data.index).clip(-1, 1).astype('float64')

body_efficiency = register_factor(
    FactorDefinition(
        name='body_efficiency',
        required_columns={'stock_code', 'date', 'open', 'close', 'high', 'low'},
        lookback_days=0,
        calculator=_body_efficiency,
        calculation_scope='target_cross_section',
        target_calculator=lambda d, ti, c: _body_efficiency(
            d.loc[ti, ['stock_code', 'date', 'open', 'close', 'high', 'low', 'pct_chg']]
            if 'pct_chg' in d.columns else d.loc[ti],
            c,
        ),
        description='K线实体效率（(收-开)/(高-低)，衡量日内趋势强度）',
    )
)

# ==============================================================================
# 2. 影线主导权 (Shadow Dominance) - 抛压 vs 承接
# ==============================================================================
def _shadow_dominance(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    影线主导权
    
    逻辑：
    (上影线长度 - 下影线长度) / (High - Low)
    
    值域 [-1, 1]:
    - > 0: 上影线长 -> 抛压重 (Shooting Star/墓碑)
    - < 0: 下影线长 -> 承接强 (Hammer/金针探底)
    - 0: 上下影线相等 -> 多空平衡
    
    大师点评：
    LightGBM 非常喜欢这个特征，能有效识别“冲高回落”的骗线。
    """
    # 上影线：High - Max(Open, Close)
    upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
    # 下影线：Min(Open, Close) - Low
    lower_shadow = np.minimum(data['open'], data['close']) - data['low']
    
    rng = data['high'] - data['low']
    
    dominance = (upper_shadow - lower_shadow) / (rng + 1e-9)
    
    return pd.Series(dominance, index=data.index).clip(-1, 1).astype('float64')

shadow_dominance = register_factor(
    FactorDefinition(
        name='shadow_dominance',
        required_columns={'stock_code', 'date', 'open', 'close', 'high', 'low'},
        lookback_days=0,
        calculator=_shadow_dominance,
        calculation_scope='target_cross_section',
        target_calculator=lambda d, ti, c: _shadow_dominance(
            d.loc[ti, ['stock_code', 'date', 'open', 'close', 'high', 'low']],
            c,
        ),
        description='影线主导权（(上影-下影)/振幅，正值代表抛压，负值代表支撑）',
    )
)

# ==============================================================================
# 3. 缺口动能 (Gap Power) - 主力意志的体现
# ==============================================================================
def _gap_power(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    标准化跳空缺口强度 (Gap Power) - 工业级优化版
    """
    if 'open' not in data.columns or 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
        
    # 1. 预计算核心序列
    grouped = data.groupby('stock_code', sort=False)
    prev_close = grouped['close'].shift(1)
    
    # 2. 计算 True Range (TR)
    hl = data['high'] - data['low']
    h_pc = (data['high'] - prev_close).abs()
    l_pc = (data['low'] - prev_close).abs()
    tr = np.maximum(hl, np.maximum(h_pc, l_pc))
    
    # 3. 计算波动率基准 (ATR 代理)
    # 直接在 transform 中使用链式操作，避免 lambda 产生的额外开销
    # 逻辑：取20日均值后整体后移一位（确保计算今天缺口时，分母只包含昨天及以前的信息）
    avg_tr = (
        tr.groupby(data['stock_code'], sort=False)
        .rolling(20, min_periods=10).mean()
        .reset_index(level=0, drop=True)
        .groupby(data['stock_code'], sort=False)
        .shift(1)
    )
    
    # 4. 计算缺口并标准化
    gap = data['open'] - prev_close
    gap_score = gap / (avg_tr + 1e-9)
    
    return gap_score.clip(-5, 5).astype('float64')

gap_power = register_factor(
    FactorDefinition(
        name='gap_power',
        required_columns={'stock_code', 'date', 'open', 'close', 'high', 'low'},
        lookback_days=10,
        calculator=_gap_power,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _gap_power(d, c), target_index=ti
        ),
        description='标准化跳空缺口强度（缺口幅度/历史波动率）',
    )
)

# ==============================================================================
# 4. 波动压缩 (Range Compression) - 捕捉“横盘等线”
# ==============================================================================
def _range_compression(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    波动率压缩指标 (NR7 / Compression)
    
    逻辑：
    Today_Range / Avg_Range_20d
    
    值域：
    < 1.0: 波动压缩（盘整、蓄势、等线）
    > 1.0: 波动放大（突破、加速）
    
    大师点评：
    这就是你要找的因子！
    当 price_ma5_ratio 接近 0 且 range_compression < 0.5 时，
    意味着“价格紧贴均线且极致缩量横盘”，这是变盘暴涨的前夜。
    """
    high_low = data['high'] - data['low']
    
    # 20日平均振幅
    grouped = data.groupby('stock_code', sort=False)['high'] # 借用索引
    # 必须构造 Series 才能 rolling
    hl_series = pd.Series(high_low.values, index=data.index)
    avg_range = hl_series.groupby(data['stock_code'], sort=False).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    
    compression = high_low / (avg_range + 1e-9)
    
    return compression.astype('float64')

range_compression = register_factor(
    FactorDefinition(
        name='range_compression',
        required_columns={'stock_code', 'date', 'high', 'low'},
        lookback_days=30,
        calculator=_range_compression,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: _range_compression(d, c).reindex(ti),
        description='波动率压缩比（当日振幅/20日平均振幅），识别横盘蓄势',
    )
)


__all__ = [
    'body_efficiency',
    'shadow_dominance',
    'gap_power',
    'range_compression',
]
