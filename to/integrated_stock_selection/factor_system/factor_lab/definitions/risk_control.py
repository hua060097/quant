"""
风险控制与路径形态因子（终极精简版）。
只保留逻辑最硬核、计算最快、互补性最强的四大金刚。
剔除 VaR、连续天数、恢复速度等冗余噪音。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series




# ==============================================================================
# 1. 效率比率 (Kaufman Efficiency Ratio) - 趋势纯度之王
# ==============================================================================
def _efficiency_ratio(data: pd.DataFrame, _: FactorContext) -> pd.Series:

    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)

    N = 20

    grouped = data.groupby('stock_code', sort=False)['close']

    # 位移
    change_n = grouped.diff(N).abs()

    # 每日路径
    daily_path = grouped.diff(1).abs()

    # 正确 rolling
    total_path = (
        daily_path
        .groupby(data['stock_code'])
        .rolling(N, min_periods=15)
        .sum()
        .reset_index(level=0, drop=True)
    )

    er = change_n / (total_path + 1e-9)

    return er.astype('float64')


efficiency_ratio = register_factor(
    FactorDefinition(
        name='efficiency_ratio',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=30,
        calculator=_efficiency_ratio,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _efficiency_ratio(d, c), target_index=ti
        ),
        description='考夫曼效率比率（趋势纯净度，1=直线，0=噪音）',
    )
)

# ==============================================================================
# 2. 溃疡指数 (Ulcer Index) - 比 VaR 更实战的风险指标
# ==============================================================================
def _ulcer_index(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    溃疡指数 (Ulcer Index)
    
    逻辑：
    UI = Sqrt(Mean(Percentage_Drawdown^2))
    
    大师点评：
    VaR 只看概率分布，Ulcer Index 看的是“回撤的深度”和“在水下憋气的时间”。
    它会严厉惩罚那些“阴跌不止”或者“深套不反弹”的股票。
    我们取负值，值越大（越接近0）越好。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
        
    N = 20
    
    grouped = data.groupby('stock_code', sort=False)['close']
    
    # 1. 计算滚动最高价 (水位线)
    rolling_max = grouped.rolling(N, min_periods=5).max().reset_index(level=0, drop=True)
    
    # 2. 计算回撤百分比 (Drawdown %)
    # dd 都是负数或0
    dd = (data['close'] / (rolling_max + 1e-9) - 1.0) * 100
    
    # 3. 计算平方均值的开方 (RMS of Drawdown)
    # 必须手动构建 DataFrame 来做滚动 group 计算，因为 dd 是 Series
    # 这种写法比 apply 快 100 倍
    temp = pd.DataFrame({'dd_sq': dd.values**2, 'stock_code': data['stock_code'].values}, index=data.index)
    mean_dd_sq = temp.groupby('stock_code', sort=False)['dd_sq'].rolling(N, min_periods=10).mean().reset_index(level=0, drop=True)
    
    ulcer = np.sqrt(mean_dd_sq)
    
    # 4. 取负数：让因子方向变成“越大越安全”
    return -ulcer.astype('float64')

ulcer_index = register_factor(
    FactorDefinition(
        name='ulcer_index',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=30,
        calculator=_ulcer_index,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _ulcer_index(d, c), target_index=ti
        ),
        description='溃疡指数（衡量回撤痛苦程度，取负值）',
    )
)

# ==============================================================================
# 3. 20日最大涨幅 (Max Rally) - 进攻核心
# ==============================================================================
def _max_rally_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    20日最大涨幅 (相对于区间最低点)
    
    逻辑：
    (Close - RollingMin) / RollingMin
    注意：这里计算的是“当前收盘价相对于过去20天最低价的涨幅”，即 Current Rally。
    它反映了当前的势能。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    grouped = data.groupby('stock_code', sort=False)['close']
    
    # 滚动最低价
    rolling_min = grouped.rolling(20, min_periods=5).min().reset_index(level=0, drop=True)
    
    # 计算涨幅
    rally = (data['close'] - rolling_min) / (rolling_min + 1e-9)
    
    return rally.astype('float64')


def _max_rally_20d_raw(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _max_rally_20d(data, ctx)

max_rally_20d = register_factor(
    FactorDefinition(
        name='max_rally_20d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=30,
        calculator=_max_rally_20d,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _max_rally_20d(d, c), target_index=ti
        ),
        description='当前价格相对于过去20日最低价的涨幅（进攻爆发力）',
    )
)

max_rally_20d_raw = register_factor(
    FactorDefinition(
        name='max_rally_20d_raw',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=30,
        calculator=_max_rally_20d_raw,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _max_rally_20d_raw(d, c), target_index=ti
        ),
        description='当前价格相对于过去20日最低价的原始涨幅（不参与z-score）',
    )
)


def _max_rally_10d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    10日最大涨幅 (相对于区间最低点)

    逻辑：
    (Close - RollingMin10) / RollingMin10
    相比 max_rally_20d 更偏短周期、更敏感。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)

    grouped = data.groupby('stock_code', sort=False)['close']
    rolling_min = grouped.rolling(10, min_periods=5).min().reset_index(level=0, drop=True)
    rally = (data['close'] - rolling_min) / (rolling_min + 1e-9)
    return rally.astype('float64')


def _max_rally_10d_raw(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _max_rally_10d(data, ctx)


max_rally_10d = register_factor(
    FactorDefinition(
        name='max_rally_10d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=15,
        calculator=_max_rally_10d,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _max_rally_10d(d, c), target_index=ti
        ),
        description='当前价格相对于过去10日最低价的涨幅（短周期进攻爆发力）',
    )
)

max_rally_10d_raw = register_factor(
    FactorDefinition(
        name='max_rally_10d_raw',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=15,
        calculator=_max_rally_10d_raw,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _max_rally_10d_raw(d, c), target_index=ti
        ),
        description='当前价格相对于过去10日最低价的原始涨幅（不参与z-score）',
    )
)

# ==============================================================================
# 4. 20日最大回撤 (Max Drawdown) - 防守核心
# ==============================================================================
def _max_drawdown_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    20日最大回撤 (相对于区间最高点)
    
    逻辑：
    (Close - RollingMax) / RollingMax
    反映当前是否处于“深坑”中。
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index)
        
    grouped = data.groupby('stock_code', sort=False)['close']
    rolling_max = grouped.rolling(20, min_periods=5).max().reset_index(level=0, drop=True)
    
    drawdown = (data['close'] - rolling_max) / (rolling_max + 1e-9)
    
    return drawdown.astype('float64')

max_drawdown_20d = register_factor(
    FactorDefinition(
        name='max_drawdown_20d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=30,
        calculator=_max_drawdown_20d,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _max_drawdown_20d(d, c), target_index=ti
        ),
        description='当前价格相对于过去20日最高价的跌幅（回撤深度）',
    )
)

# ==============================================================================
# 5. 近3日跌停标记：最近3个交易日是否有过跌停
# ==============================================================================
def _limit_down_3d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    if 'is_limit_down' not in data.columns:
        return pd.Series(0, index=data.index)
    grouped = data.groupby('stock_code', sort=False)['is_limit_down']
    result = grouped.transform(lambda x: x.rolling(3, min_periods=1).max())
    return result.fillna(0).astype(int)


limit_down_3d = register_factor(
    FactorDefinition(
        name='limit_down_3d',
        required_columns={'stock_code', 'date', 'is_limit_down'},
        lookback_days=5,
        calculator=_limit_down_3d,
        calculation_scope='target_with_history_window',
        target_calculator=lambda d, ti, c: select_target_series(
            _limit_down_3d(d, c), target_index=ti
        ),
        description='近3个交易日是否出现过跌停（1=有，0=无）',
    )
)

__all__ = [
    'efficiency_ratio',
    'ulcer_index',
    'max_drawdown_20d',
    'max_rally_20d',
    'max_rally_10d',
    'max_rally_20d_raw',
    'max_rally_10d_raw',
    'limit_down_3d',
]
