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
from ..target_utils import select_target_series


def _target_rows_only(
    data: pd.DataFrame,
    target_index: pd.Index,
    ctx: FactorContext,
    calculator,
) -> pd.Series:
    target_data = data.loc[target_index].copy()
    return calculator(target_data, ctx)

def _get_market_data_safe(ctx: FactorContext) -> pd.DataFrame:
    """获取对齐后的市场指数数据"""
    market = ctx.require_market()
    # 确保按日期排序
    return market.sort_values('date').reset_index(drop=True)


def _prepare_stock_timeseries(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cols = ['date', 'stock_code'] + columns
    df = data[cols].copy()
    return df.sort_values(['stock_code', 'date']).reset_index(drop=True)


def _rolling_percentile_self(series: pd.Series, window: int = 20, min_periods: int = 10) -> pd.Series:
    values: list[float] = []
    for i in range(len(series)):
        cur = series.iloc[i]
        if pd.isna(cur):
            values.append(np.nan)
            continue
        hist = series.iloc[max(0, i - window + 1): i + 1].dropna()
        if len(hist) < min_periods:
            values.append(np.nan)
            continue
        values.append(float((hist <= cur).mean()))
    return pd.Series(values, index=series.index, dtype='float64')


def _rolling_percentile_shift1(series: pd.Series, window: int = 20, min_periods: int = 10) -> pd.Series:
    values: list[float] = []
    for i in range(len(series)):
        cur = series.iloc[i]
        if pd.isna(cur):
            values.append(np.nan)
            continue
        hist = series.iloc[max(0, i - window):i].dropna()
        if len(hist) < min_periods:
            values.append(np.nan)
            continue
        values.append(float((hist <= cur).mean()))
    return pd.Series(values, index=series.index, dtype='float64')


def _train_percentile_frozen(series: pd.Series, date_index: pd.Series, ctx: FactorContext) -> pd.Series:
    extra = ctx.extra or {}
    start = extra.get("frozen_clip_train_start")
    end = extra.get("frozen_clip_train_end")
    default_start = pd.Timestamp("2023-06-01")
    default_end = pd.Timestamp("2025-03-01")
    try:
        start_ts = pd.Timestamp(start) if start is not None else default_start
        end_ts = pd.Timestamp(end) if end is not None else default_end
    except Exception:
        start_ts, end_ts = default_start, default_end

    ts = pd.to_datetime(pd.Series(date_index).reset_index(drop=True), errors='coerce')
    series_values = pd.Series(series, dtype='float64').reset_index(drop=True)
    train_mask = (ts >= start_ts) & (ts <= end_ts)
    train_hist = series_values.loc[train_mask].dropna()
    if train_hist.empty:
        return pd.Series(np.nan, index=series.index, dtype='float64')
    values = []
    for cur in series_values:
        if pd.isna(cur):
            values.append(np.nan)
        else:
            values.append(float((train_hist <= cur).mean()))
    return pd.Series(values, index=series.index, dtype='float64')

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


def _market_advance_decline_ratio(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    市场上涨家数/下跌家数比例。

    与 market_sentiment_ad 的差异：
    - market_sentiment_ad 是 (上涨家数-下跌家数)/总家数，偏平滑情绪。
    - 本因子是 Advancers/Decliners，保留市场宽度的非线性强弱。
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype='float64')

    pct = pd.to_numeric(data['pct_chg'], errors='coerce')
    daily = pd.DataFrame({'date': data['date'].values, 'pct_chg': pct.values}, index=data.index)
    counts = daily.groupby('date', sort=True)['pct_chg'].agg(
        advancers=lambda x: (x > 0).sum(),
        decliners=lambda x: (x < 0).sum(),
    )
    ratio = counts['advancers'] / (counts['decliners'] + 1.0)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).clip(0.0, 20.0)
    return data['date'].map(ratio).astype('float64')


def _market_advance_decline_ratio_percentile_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    市场上涨家数/下跌家数比例在近20日中的滚动分位数。

    使用包含当日的20日窗口，表示“今天市场宽度在近期处于什么位置”。
    """
    ratio = _market_advance_decline_ratio(data, ctx)
    daily_ratio = pd.DataFrame({'date': data['date'].values, 'ratio': ratio.values}).drop_duplicates('date')
    daily_ratio = daily_ratio.sort_values('date')
    percentile = _rolling_percentile_self(daily_ratio['ratio'], window=20, min_periods=10)
    pct_map = pd.Series(percentile.values, index=daily_ratio['date'])
    return data['date'].map(pct_map).astype('float64')

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


def _market_prev_60d_new_high_avg_return(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    昨日处于60日新高股票在今日的平均涨跌幅。
    """
    if 'close' not in data.columns or 'pct_chg' not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype='float64')

    df = _prepare_stock_timeseries(data, ['close', 'pct_chg'])
    close = pd.to_numeric(df['close'], errors='coerce')
    rolling_high_60 = close.groupby(df['stock_code']).transform(
        lambda x: x.rolling(60, min_periods=60).max()
    )
    is_60d_new_high = close >= (rolling_high_60 - 1e-9)
    prev_60d_new_high = is_60d_new_high.groupby(df['stock_code']).shift(1).fillna(False)

    daily_avg = df.loc[prev_60d_new_high, ['date', 'pct_chg']].groupby('date')['pct_chg'].mean()
    return data['date'].map(daily_avg).astype('float64')


def _market_60d_new_high_ratio(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    当日股价处于60日新高的股票占全部股票数量的比例。
    """
    if 'close' not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype='float64')

    df = _prepare_stock_timeseries(data, ['close'])
    close = pd.to_numeric(df['close'], errors='coerce')
    rolling_high_60 = close.groupby(df['stock_code']).transform(
        lambda x: x.rolling(60, min_periods=60).max()
    )
    is_60d_new_high = (close >= (rolling_high_60 - 1e-9)).astype('float64')

    daily_ratio = df.assign(is_60d_new_high=is_60d_new_high).groupby('date')['is_60d_new_high'].mean()
    return data['date'].map(daily_ratio).astype('float64')


def _market_60d_new_high_ratio_train_percentile(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if 'close' not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype='float64')

    df = _prepare_stock_timeseries(data, ['close'])
    close = pd.to_numeric(df['close'], errors='coerce')
    rolling_high_60 = close.groupby(df['stock_code']).transform(
        lambda x: x.rolling(60, min_periods=60).max()
    )
    is_60d_new_high = (close >= (rolling_high_60 - 1e-9)).astype('float64')
    daily_ratio = df.assign(is_60d_new_high=is_60d_new_high).groupby('date')['is_60d_new_high'].mean().sort_index()
    train_pct = _train_percentile_frozen(daily_ratio, pd.Series(daily_ratio.index), ctx)
    return data['date'].map(pd.Series(train_pct.values, index=daily_ratio.index)).astype('float64')


def _market_event_count_percentile_20d(data: pd.DataFrame, event_col: str) -> pd.Series:
    if event_col not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype='float64')

    daily_count = (
        pd.to_numeric(data[event_col], errors='coerce')
        .fillna(0)
        .groupby(data['date'])
        .sum()
        .astype('float64')
        .sort_index()
    )
    percentile = _rolling_percentile_self(daily_count, window=20, min_periods=10)
    return data['date'].map(percentile).astype('float64')


def _limit_up_count_percentile_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _market_event_count_percentile_20d(data, 'is_limit_up')


def _zhaban_count_percentile_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _market_event_count_percentile_20d(data, 'is_zhaban')


def _limit_down_count_percentile_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _market_event_count_percentile_20d(data, 'is_limit_down')

# 注册因子
market_trend_strength = register_factor(
    FactorDefinition(
        name='market_trend_strength',
        required_columns={'date'},
        requires_market_data=True,
        lookback_days=30,
        calculator=_market_trend_strength,
        calculation_scope='target_cross_section',
        target_calculator=lambda data, target_index, ctx: _target_rows_only(
            data, target_index, ctx, _market_trend_strength
        ),
        description='大盘趋势强度（广播因子）',
    )
)

market_sentiment_ad = register_factor(
    FactorDefinition(
        name='market_sentiment_ad',
        required_columns={'date', 'pct_chg', 'stock_code'},
        lookback_days=0,
        calculator=_market_sentiment_advance_decline,
        calculation_scope='daily_broadcast',
        target_calculator=lambda d, ti, c: select_target_series(
            _market_sentiment_advance_decline(d, c), target_index=ti
        ),
        description='全市场赚钱效应（涨跌家数差占比）',
    )
)

market_advance_decline_ratio = register_factor(
    FactorDefinition(
        name='market_advance_decline_ratio',
        required_columns={'date', 'pct_chg'},
        lookback_days=0,
        calculator=_market_advance_decline_ratio,
        calculation_scope='daily_broadcast',
        target_calculator=lambda d, ti, c: select_target_series(
            _market_advance_decline_ratio(d, c), target_index=ti
        ),
        description='市场上涨家数/下跌家数比例，越高代表市场宽度越强',
    )
)

market_advance_decline_ratio_percentile_20d = register_factor(
    FactorDefinition(
        name='market_advance_decline_ratio_percentile_20d',
        required_columns={'date', 'pct_chg'},
        lookback_days=25,
        calculator=_market_advance_decline_ratio_percentile_20d,
        calculation_scope='stateful_history',
        target_calculator=lambda d, ti, c: select_target_series(
            _market_advance_decline_ratio_percentile_20d(d, c), target_index=ti
        ),
        description='市场上涨/下跌家数比例在近20日滚动窗口中的历史分位数',
    )
)

market_volatility_context = register_factor(
    FactorDefinition(
        name='market_volatility_context',
        required_columns={'date'},
        requires_market_data=True,
        lookback_days=30,
        calculator=_market_volatility_context,
        calculation_scope='target_cross_section',
        target_calculator=lambda data, target_index, ctx: _target_rows_only(
            data, target_index, ctx, _market_volatility_context
        ),
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
        calculation_scope='target_cross_section',
        target_calculator=lambda data, target_index, ctx: _target_rows_only(
            data, target_index, ctx, _market_liquidity_change
        ),
        description='大盘波动率环境（广播因子）',
    )
)

market_prev_60d_new_high_avg_return = register_factor(
    FactorDefinition(
        name='market_prev_60d_new_high_avg_return',
        required_columns={'stock_code', 'date', 'close', 'pct_chg'},
        lookback_days=70,
        calculator=_market_prev_60d_new_high_avg_return,
        calculation_scope='daily_broadcast',
        target_calculator=lambda d, ti, c: select_target_series(
            _market_prev_60d_new_high_avg_return(d, c), target_index=ti
        ),
        description='昨日60日新高股票在当日的平均涨跌幅',
    )
)

market_60d_new_high_ratio = register_factor(
    FactorDefinition(
        name='market_60d_new_high_ratio',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=70,
        calculator=_market_60d_new_high_ratio,
        calculation_scope='daily_broadcast',
        target_calculator=lambda d, ti, c: select_target_series(
            _market_60d_new_high_ratio(d, c), target_index=ti
        ),
        description='当日处于60日新高股票数量占全部股票数量的比例',
    )
)

market_60d_new_high_ratio_train_percentile = register_factor(
    FactorDefinition(
        name='market_60d_new_high_ratio_train_percentile',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=70,
        calculator=_market_60d_new_high_ratio_train_percentile,
        calculation_scope='stateful_history',
        target_calculator=lambda d, ti, c: select_target_series(
            _market_60d_new_high_ratio_train_percentile(d, c), target_index=ti
        ),
        description='当日60日新高占比在冻结训练期分布中的分位数',
    )
)

limit_up_count_percentile_20d = register_factor(
    FactorDefinition(
        name='limit_up_count_percentile_20d',
        required_columns={'date', 'is_limit_up'},
        lookback_days=25,
        calculator=_limit_up_count_percentile_20d,
        calculation_scope='stateful_history',
        target_calculator=lambda d, ti, c: select_target_series(
            _limit_up_count_percentile_20d(d, c), target_index=ti
        ),
        description='涨停股票数量在20日滚动窗口中的历史分位数',
    )
)

zhaban_count_percentile_20d = register_factor(
    FactorDefinition(
        name='zhaban_count_percentile_20d',
        required_columns={'date', 'is_zhaban'},
        lookback_days=25,
        calculator=_zhaban_count_percentile_20d,
        calculation_scope='stateful_history',
        target_calculator=lambda d, ti, c: select_target_series(
            _zhaban_count_percentile_20d(d, c), target_index=ti
        ),
        description='炸板股票数量在20日滚动窗口中的历史分位数',
    )
)

limit_down_count_percentile_20d = register_factor(
    FactorDefinition(
        name='limit_down_count_percentile_20d',
        required_columns={'date', 'is_limit_down'},
        lookback_days=25,
        calculator=_limit_down_count_percentile_20d,
        calculation_scope='stateful_history',
        target_calculator=lambda d, ti, c: select_target_series(
            _limit_down_count_percentile_20d(d, c), target_index=ti
        ),
        description='跌停股票数量在20日滚动窗口中的历史分位数',
    )
)

__all__ = [
    'market_trend_strength',
    'market_sentiment_ad',
    'market_advance_decline_ratio',
    'market_advance_decline_ratio_percentile_20d',
    'market_volatility_context',
    'market_liquidity_change',
    'market_prev_60d_new_high_avg_return',
    'market_60d_new_high_ratio',
    'market_60d_new_high_ratio_train_percentile',
    'limit_up_count_percentile_20d',
    'zhaban_count_percentile_20d',
    'limit_down_count_percentile_20d',
]
