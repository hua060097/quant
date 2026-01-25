"""
市场环境因子 (Market Context Factors)。
将全市场的状态（大盘趋势、情绪、风险）广播给每一只股票。
让模型学会“看天吃饭”。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

def _get_market_data_safe(ctx: FactorContext) -> pd.DataFrame:
    """获取对齐后的市场指数数据"""
    market = ctx.require_market()
    # 确保按日期排序
    return market.sort_values('date').reset_index(drop=True)

# ==============================================================================
# 1. 市场趋势因子 (Market Trend)
# ==============================================================================
def _market_trend_strength(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    大盘趋势强度
    逻辑：(Index_Close - Index_MA20) / Index_MA20
    广播给所有股票：告诉模型现在大盘是在20日线上方（牛）还是下方（熊）。
    """
    market = _get_market_data_safe(ctx)

    # 调试信息：打印market数据
    print("\n[DEBUG] _market_trend_strength - market数据检查:")
    print(f"  market shape: {market.shape}")
    print(f"  market columns: {list(market.columns)}")
    print(f"  market前5行:")
    print(market.head())
    print(f"  market后5行:")
    print(market.tail())
    print(f"  market['close']统计信息:")
    print(f"    count: {len(market['close'])}, min: {market['close'].min():.2f}, max: {market['close'].max():.2f}, mean: {market['close'].mean():.2f}")

    # 计算大盘均线乖离
    ma20 = market['close'].rolling(20).mean()
    trend = (market['close'] - ma20) / (ma20 + 1e-9)

    print(f"  计算后trend统计信息:")
    print(f"    count: {len(trend)}, min: {trend.min():.4f}, max: {trend.max():.4f}, mean: {trend.mean():.4f}")
    print(f"    trend前10个值:")
    print(trend.head(10))
    print(f"    trend后10个值:")
    print(trend.tail(10))

    # 映射回个股数据 (Merge by date)
    # 1. 构建日期映射表
    trend_map = pd.Series(trend.values, index=market['date'])

    print(f"  trend_map size: {len(trend_map)}")
    print(f"  trend_map前5个:")
    print(trend_map.head())

    # 2. 映射
    # return data['date'].map(trend_map).fillna(0.0).astype('float64')
    result = data['date'].map(trend_map).fillna(0.0).astype('float64')

    print(f"  最终result统计信息:")
    print(f"    count: {len(result)}, min: {result.min():.4f}, max: {result.max():.4f}, mean: {result.mean():.4f}")
    print(f"    唯一值数量: {result.nunique()}")
    print(f"    唯一值样本: {list(result.unique())[:10]}")
    print("[DEBUG] _market_trend_strength - 调试信息结束\n")

    return result

# ==============================================================================
# 2. 市场情绪因子：赚钱效应 (Market Sentiment)
# ==============================================================================
def _market_sentiment_advance_decline(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    全市场涨跌比 (A/D Ratio) 的平滑版本
    逻辑：(涨家数 - 跌家数) / 总家数
    由于我们在这就只有个股数据，我们需要先聚合计算全市场状态，再广播回来。
    """
    # 1. 在当前数据集中计算每日涨跌统计
    # 注意：这假设data包含了全市场大部分股票，如果是全样本训练这很准
    daily_stats = data.groupby('date')['pct_chg'].apply(
        lambda x: (x > 0).sum() - (x < 0).sum()
    ) / data.groupby('date')['stock_code'].count()
    
    # 2. 平滑处理 (3日均值)，避免单日噪音
    sentiment = daily_stats.rolling(3).mean()
    
    # 3. 映射回个股
    # return data['date'].map(sentiment).fillna(0.0).astype('float64')
    return data['date'].map(sentiment).astype('float64')

# ==============================================================================
# 3. 市场恐慌因子 (Market Risk)
# ==============================================================================
def _market_volatility_context(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    大盘波动率环境
    逻辑：大盘指数的20日波动率
    当波动率极高时，模型可能会倾向于防御性股票。
    """
    market = _get_market_data_safe(ctx)
    
    # 计算大盘波动率
    mkt_vol = market['daily_change_pct'].rolling(20).std()
    
    vol_map = pd.Series(mkt_vol.values, index=market['date'])
    # return data['date'].map(vol_map).fillna(0.0).astype('float64')
    return data['date'].map(vol_map).astype('float64')

# ==============================================================================
# 4. 市场成交量萎缩/放大 (Market Liquidity)
# ==============================================================================
def _market_liquidity_change(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    全市场成交额变化率
    逻辑：今日全市场成交额 / 5日均值
    """
    market = _get_market_data_safe(ctx)
    
    # 如果市场数据里有 volume 或 amount
    if 'amount' in market.columns:
        mkt_amt = market['amount']
    elif 'volume' in market.columns:
        mkt_amt = market['volume']
    else:
        return pd.Series(1.0, index=data.index)
        
    amt_ma5 = mkt_amt.rolling(5).mean()
    liq_change = mkt_amt / (amt_ma5 + 1e-9)
    
    liq_map = pd.Series(liq_change.values, index=market['date'])
    # return data['date'].map(liq_map).fillna(1.0).astype('float64')
    return data['date'].map(liq_map).astype('float64')

# 注册因子
market_trend_strength = register_factor(
    FactorDefinition(
        name='market_trend_strength',
        required_columns={'date'},
        requires_market_data=True,
        lookback_days=30,
        calculator=_market_trend_strength,
        description='大盘趋势强度（广播因子）',
    )
)

market_sentiment_ad = register_factor(
    FactorDefinition(
        name='market_sentiment_ad',
        required_columns={'date', 'pct_chg', 'stock_code'},
        lookback_days=0,
        calculator=_market_sentiment_advance_decline,
        description='全市场赚钱效应（涨跌家数差占比）',
    )
)

market_volatility_context = register_factor(
    FactorDefinition(
        name='market_volatility_context',
        required_columns={'date'},
        requires_market_data=True,
        lookback_days=30,
        calculator=_market_volatility_context,
        description='大盘波动率环境（广播因子）',
    )
)


market_liquidity_change = register_factor(
    FactorDefinition(
        name='market_liquidity_change',
        required_columns={'date'},
        requires_market_data=True,
        lookback_days=30,
        calculator=_market_liquidity_change,
        description='大盘波动率环境（广播因子）',
    )
)

__all__ = [
    'market_trend_strength',
    'market_sentiment_ad',
    'market_volatility_context',
    'market_liquidity_change'
]