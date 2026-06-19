"""
板块相关因子 (大师级优化版)
文件名: sector.py

优化日志：
1. 全局去除了 .fillna(0) 和 .fillna(0.5)，保留 NaN (未知) 状态。
2. _global_robust_zscore: 升级为纯净版，不污染分布。
3. _sector_momentum_interaction_opt: 逻辑增强，使用双重 Z-Score 叠加。
4. _sector_correlation_20d: 修复相关性计算中的填充问题。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series

# ==============================================================================
# 辅助工具：获取行业映射
# ==============================================================================
def _get_sector_map_safe(ctx: FactorContext) -> pd.Series:
    if ctx.extra and 'stock_to_sw_sector' in ctx.extra:
        return ctx.extra['stock_to_sw_sector']
    raise ValueError("Sector data missing in context")


def _prepare_sector_stock_frame(data: pd.DataFrame, ctx: FactorContext, columns: list[str]) -> pd.DataFrame:
    sector_map = _get_sector_map_safe(ctx)
    cols = ['date', 'stock_code'] + columns
    df = data[cols].copy()
    df['sector_code'] = df['stock_code'].map(sector_map).fillna('Unknown')
    return df


# ==============================================================================
# 0. 新增核心工具：全局按日稳健 Z-Score (Top-1 专用版)
# ==============================================================================
def _global_robust_zscore(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """
    [大师级] 全局按日稳健 Z-Score (Robust Z-Score based on MAD)
    
    逻辑升级：
    1. 使用 Median 和 MAD (Median Absolute Deviation) 代替 Mean/Std。
    2. 宽阈值截断 (Clip +/- 6.0)。
    3. [重要修复] 不再 fillna(0.0)，保留 NaN。让 LightGBM 处理缺失数据。
    """
    # 1. 构造临时 DF
    df = pd.DataFrame({'val': series.values, 'date': date_series.values}, index=series.index)
    
    # 2. 向量化计算每日中位数 (Median)
    medians = df.groupby('date')['val'].transform('median')
    
    # 3. 向量化计算每日绝对中位差 (MAD)
    abs_diff = (df['val'] - medians).abs()
    
    # 为了效率，我们复用索引构造临时 DF
    temp_diff = pd.DataFrame({'diff': abs_diff.values, 'date': date_series.values}, index=series.index)
    mads = temp_diff.groupby('date')['diff'].transform('median')
    
    # 4. 计算稳健 Z-Score
    robust_std = mads * 1.4826
    z_score = (df['val'] - medians) / (robust_std + 1e-9)
    
    # 5. 极值处理：特殊处理 MAD 为 0 的情况
    mask_zero_mad = (mads < 1e-6)
    if mask_zero_mad.any():
        means = df.groupby('date')['val'].transform('mean')
        stds = df.groupby('date')['val'].transform('std')
        z_fallback = (df['val'] - means) / (stds + 1e-9)
        z_score = z_score.where(~mask_zero_mad, z_fallback)
    
    # 6. 宽阈值去极值 (Clip 6.0)
    # [修正] 删除 fillna(0.0)，保留 NaN
    return z_score.clip(-6.0, 6.0).astype('float64')


class _SectorDataHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext) -> None:
        import logging
        logger = logging.getLogger(__name__)

        self.ctx = ctx
        self.original_index = data.index
        self.data = data.copy()
        self.data['__orig_index'] = self.data.index

        self.sw_sector_mapping = None
        if ctx.extra and 'stock_to_sw_sector' in ctx.extra:
            self.sw_sector_mapping = ctx.extra['stock_to_sw_sector']
        else:
            raise ValueError("SW行业映射必须通过ctx.extra['stock_to_sw_sector']提供")

        self.sector_data = self._build_sw_sector_data(data)
        self._attach_sector_code()
        
        # 排序优化：先按 Sector 排序方便后续 rolling
        self.sector_data = self.sector_data.sort_values(['sector_code', 'date']).reset_index(drop=True)

    def _attach_sector_code(self) -> None:
        if 'sector_code' not in self.data.columns:
            self.data['sector_code'] = self.data['stock_code'].map(self.sw_sector_mapping)

    def _build_sw_sector_data(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        sw_data = stock_data.copy()
        sw_data['sector_code'] = sw_data['stock_code'].map(self.sw_sector_mapping)

        # 聚合计算行业平均收益
        sector_avg = sw_data.groupby(['date', 'sector_code']).agg({
            'pct_chg': 'mean'
        }).reset_index()

        sector_avg = sector_avg.rename(columns={'pct_chg': 'pct_chg_avg'})
        return sector_avg

    def attach_sector_features(self, columns: list[str]) -> pd.DataFrame:
        if not isinstance(columns, list):
            columns = [columns]
            
        merged = self.data.merge(
            self.sector_data[['date', 'sector_code'] + columns],
            on=['date', 'sector_code'],
            how='left',
            suffixes=('', '_sector'),
        )
        return merged


# ==============================================================================
# 1. 行业特质波动率 (Idiosyncratic Volatility)
# ==============================================================================
def _sector_relative_volatility(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    行业特质波动率 (Idiosyncratic Volatility)
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(np.nan, index=data.index)

    # 1. 构造临时表
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'stock': data['stock_code'],
        'pct': data['pct_chg']
    }, index=data.index)
    df['sector'] = df['stock'].map(sector_map).fillna('Unknown')

    # 2. 计算行业超额收益
    s_ret = df.groupby(['date', 'sector'])['pct'].transform('mean')
    df['active_ret'] = df['pct'] - s_ret

    # 3. 计算 20 日滚动特质波动率
    df_sorted = df.sort_values(['stock', 'date'])
    df['idio_vol'] = df_sorted.groupby('stock')['active_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )

    # [修正] 删除 fillna(0)，保留 NaN
    return _global_robust_zscore(df['idio_vol'], data['date'])


def _sector_volatility_5d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    SW行业5日波动率的全市场分位排名
    """
    helper = _SectorDataHelper(data, ctx)
    sector_df = helper.sector_data.copy()

    # 确保时间序列排序
    sector_df['date_dt'] = pd.to_datetime(sector_df['date'])
    sector_df = sector_df.sort_values(['sector_code', 'date_dt'])

    # 计算波动率
    sector_df['sector_volatility_5d'] = sector_df.groupby('sector_code')['pct_chg_avg'].transform(
        lambda x: x.rolling(window=5, min_periods=3).std() * np.sqrt(252)
    )

    # 横向排名
    # [修正] 删除 fillna(0.5)
    sector_df['volatility_rank'] = sector_df.groupby('date_dt')['sector_volatility_5d'].rank(
        pct=True, method='min'
    )

    helper.sector_data = sector_df
    merged = helper.attach_sector_features(['volatility_rank'])
    merged = merged.set_index('__orig_index')

    return merged['volatility_rank'].reindex(helper.original_index).astype('float64')


def _intra_sector_momentum_rank(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """个股 5 日动量 全局 Z-Score"""
    mom_5d = data.groupby('stock_code')['pct_chg'].rolling(5, min_periods=3).sum().reset_index(level=0, drop=True)
    # [修正] 删除 fillna(0)
    return _global_robust_zscore(mom_5d, data['date'])


def _sector_momentum_dispersion_rank(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    计算行业动量离散度排名
    """
    sw_sector_mapping = ctx.extra.get('stock_to_sw_sector')
    
    data_work = data.copy()
    data_work['sector_code'] = data_work['stock_code'].map(sw_sector_mapping)
    data_work = data_work.sort_values(['stock_code', 'date'])
    
    data_work['momentum_5d'] = data_work.groupby('stock_code')['pct_chg'].rolling(
        window=5, min_periods=3
    ).sum().reset_index(level=0, drop=True)

    data_work['date_dt'] = pd.to_datetime(data_work['date'])
    
    # 计算行业离散度
    sector_dispersion = data_work.groupby(['date', 'sector_code'])['momentum_5d'].agg(['count', 'std']).reset_index()
    
    # 至少3只股票才算离散度
    sector_dispersion = sector_dispersion[sector_dispersion['count'] >= 3]
    
    # 排名
    sector_dispersion['date_dt'] = pd.to_datetime(sector_dispersion['date'])
    sector_dispersion['dispersion_rank'] = sector_dispersion.groupby('date_dt')['std'].rank(pct=True)

    data_with_rank = data_work.merge(
        sector_dispersion[['date', 'sector_code', 'dispersion_rank']],
        on=['date', 'sector_code'],
        how='left'
    )

    # [修正] 删除 fillna(0.5)
    return pd.Series(data_with_rank['dispersion_rank'].values, index=data.index, dtype='float64')


def _sector_dispersion(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """板块内部收益离散度"""
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'pct': data['pct_chg'].values,
        'date': data['date'].values,
        'sector': data['stock_code'].map(sector_map).fillna('Unknown').values
    }, index=data.index)
    
    dispersion = df.groupby(['date', 'sector'])['pct'].transform('std')
    
    # [修正] 删除 fillna(0.0)
    return dispersion.astype('float64')


def _sector_momentum_consistency(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """计算SW行业动量一致性"""
    helper = _SectorDataHelper(data, ctx)
    sector_df = helper.sector_data.copy()
    sector_df['date_dt'] = pd.to_datetime(sector_df['date'])
    sector_df = sector_df.sort_values(['sector_code', 'date_dt'])

    periods = [3, 5, 10, 20]
    rank_cols = []

    for period in periods:
        sector_df[f'avg_return_{period}d'] = sector_df.groupby('sector_code')['pct_chg_avg'].rolling(
            window=period, min_periods=max(1, period // 2)
        ).mean().reset_index(level=0, drop=True)

        rank_col = f'sector_return_rank_{period}d'
        sector_df[rank_col] = sector_df.groupby('date_dt')[f'avg_return_{period}d'].rank(pct=True)
        rank_cols.append(rank_col)

    helper.sector_data = sector_df
    merged = helper.attach_sector_features(rank_cols).set_index('__orig_index')
    ranks_df = merged[rank_cols].reindex(helper.original_index)

    rank_std = ranks_df.std(axis=1, skipna=True)
    max_std = rank_std.max()
    if pd.isna(max_std) or max_std <= 0:
        max_std = 1.0

    consistency = 1 - (rank_std / max_std)
    # [修正] 删除 fillna(0.5)
    return consistency.astype('float64')


def _sector_correlation_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """计算股票与SW行业的20日滚动相关性"""
    sw_sector_mapping = ctx.extra.get('stock_to_sw_sector')
    
    data_work = data.copy()
    data_work['date_dt'] = pd.to_datetime(data_work['date'])
    data_work['stock_return_dec'] = data_work['pct_chg'].astype(float)
    data_work['sector_code'] = data_work['stock_code'].map(sw_sector_mapping)

    sector_avg_return = data_work.groupby(['date', 'sector_code'])['pct_chg'].mean().reset_index()
    sector_avg_return.rename(columns={'pct_chg': 'sector_avg_pct_chg'}, inplace=True)

    data_with_sector = data_work.merge(sector_avg_return, on=['date', 'sector_code'], how='left')
    data_with_sector['sector_return_dec'] = data_with_sector['sector_avg_pct_chg'].astype(float)

    data_with_sector = data_with_sector.sort_values(['stock_code', 'date_dt'])

    def calc_rolling_corr(group):
        return group['stock_return_dec'].rolling(20, min_periods=10).corr(group['sector_return_dec'])

    # 使用 groupby().apply() 
    corr = data_with_sector.groupby('stock_code').apply(calc_rolling_corr).reset_index(level=0, drop=True)

    # [修正] 仅处理 Inf，不 fillna(0.0)
    corr = corr.replace([np.inf, -np.inf], np.nan)

    result = pd.Series(corr.values, index=data.index, dtype='float64')
    return _global_robust_zscore(result, data['date'])


# ==============================================================================
# 3. 板块动量与个股强度的交互 (Sector Momentum Interaction) - 优化版
# ==============================================================================
def _sector_momentum_interaction_opt(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    板块动量 * 个股相对强度
    
    逻辑优化：
    1. 使用 Z-Score 标准化后再相加，避免量纲不同带来的权重偏差。
    2. 删除 fillna，保留 NaN。
    """
    sector_map = _get_sector_map_safe(ctx)
    
    # 1. 计算个股20日动量
    grouped = data.groupby('stock_code', sort=False)['close']
    mom_20d = grouped.pct_change(20) # 保留 NaN
    
    # 2. 计算板块20日动量
    df = pd.DataFrame({'mom': mom_20d.values, 'date': data['date'].values, 'sector': data['stock_code'].map(sector_map).values}, index=data.index)
    sector_mom = df.groupby(['date', 'sector'])['mom'].transform('mean')
    
    # 3. 交互计算
    relative_strength = mom_20d - sector_mom
    
    # 4. 双重 Z-Score 叠加
    # 这样可以公平地衡量"板块有多强"和"个股比板块强多少"
    
    # 板块动量 Z-Score
    def calc_z(x):
        std = x.std()
        if std == 0 or pd.isna(std): return x - x.mean()
        return (x - x.mean()) / std
        
    sec_z = df.groupby('date')['mom'].transform(calc_z)
    
    # 相对强度 Z-Score
    # 需要先构造一个 Series 才能 groupby
    rel_series = pd.Series(relative_strength, index=data.index)
    rel_z = rel_series.groupby(data['date']).transform(calc_z)
    
    interaction = sec_z + rel_z 
    
    # [修正] 删除 fillna(0.0)
    return interaction.astype('float64')


def _sw_circ_mv_rank_intra_sector(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    SW行业内流通市值的相对排名
    """
    if 'circ_mv' not in data.columns:
        return pd.Series(np.nan, index=data.index)
        
    # [修正] 删除 fillna(0)
    return _global_robust_zscore(data['circ_mv'], data['date'])


def _sector_ma5_followthrough_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    行业版 MA5 跟随顺滑度。

    逻辑模仿 mkt_ma5_followthrough_v3：
    1. 先按 SW2 聚合日均涨跌幅，构造行业伪指数
    2. 再计算行业 close 相对 MA5 / MA10 的顺滑程度，并用行业收益波动率归一化
    3. 最后将所属行业当日值广播回个股
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype='float64')

    helper = _SectorDataHelper(data, ctx)
    sector_df = helper.sector_data.copy()
    sector_df['date_dt'] = pd.to_datetime(sector_df['date'])
    sector_df = sector_df.sort_values(['sector_code', 'date_dt']).reset_index(drop=True)

    sector_ret = pd.to_numeric(sector_df['pct_chg_avg'], errors='coerce') / 100.0
    pseudo_close = sector_ret.groupby(sector_df['sector_code']).transform(
        lambda x: (1.0 + x.fillna(0.0)).cumprod()
    )
    ma5 = pseudo_close.groupby(sector_df['sector_code']).transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    ma10 = pseudo_close.groupby(sector_df['sector_code']).transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    vol10 = sector_ret.groupby(sector_df['sector_code']).transform(
        lambda x: x.rolling(10, min_periods=5).std()
    )

    follow = ((pseudo_close - ma5) / (ma5 + 1e-9)) - 0.5 * ((ma5 - ma10) / (ma10 + 1e-9))
    sector_df['sector_ma5_followthrough_v3'] = (follow / (vol10 + 1e-6)).astype('float64')

    helper.sector_data = sector_df
    merged = helper.attach_sector_features(['sector_ma5_followthrough_v3']).set_index('__orig_index')
    return merged['sector_ma5_followthrough_v3'].reindex(helper.original_index).astype('float64')


def _limit_up_sw2_concentration(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    涨停股票在SW2行业分布上的集中度，使用 HHI 衡量。
    """
    if 'is_limit_up' not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype='float64')

    df = _prepare_sector_stock_frame(data, ctx, ['is_limit_up'])
    df['is_limit_up'] = pd.to_numeric(df['is_limit_up'], errors='coerce').fillna(0)

    sector_limit_count = df.groupby(['date', 'sector_code'])['is_limit_up'].sum()
    daily_limit_total = sector_limit_count.groupby('date').sum()
    sector_share = sector_limit_count / (sector_limit_count.groupby('date').transform('sum') + 1e-9)
    daily_hhi = (sector_share ** 2).groupby('date').sum()
    daily_hhi = daily_hhi.where(daily_limit_total > 0)

    return data['date'].map(daily_hhi).astype('float64')


def _sw2_high_low_position_return_spread(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    SW2行业内高位股与低位股的当日涨幅差。

    高低位定义：
    - 个股60日价格位置 = (close - 60日低点) / (60日高点 - 60日低点)
    - 行业内按当日价格位置分位划分，高位为前30%，低位为后30%
    """
    required_cols = {'close', 'pct_chg'}
    if not required_cols.issubset(data.columns):
        return pd.Series(np.nan, index=data.index, dtype='float64')

    df = _prepare_sector_stock_frame(data, ctx, ['close', 'pct_chg']).copy()
    df['__orig_index'] = data.index
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    close = pd.to_numeric(df['close'], errors='coerce')
    rolling_high_60 = close.groupby(df['stock_code']).transform(
        lambda x: x.rolling(60, min_periods=20).max()
    )
    rolling_low_60 = close.groupby(df['stock_code']).transform(
        lambda x: x.rolling(60, min_periods=20).min()
    )
    df['price_position_60d'] = (close - rolling_low_60) / (rolling_high_60 - rolling_low_60 + 1e-9)
    df['position_rank_pct'] = df.groupby(['date', 'sector_code'])['price_position_60d'].rank(pct=True, method='average')

    high_mask = df['position_rank_pct'] >= 0.7
    low_mask = df['position_rank_pct'] <= 0.3

    high_mean = df['pct_chg'].where(high_mask).groupby([df['date'], df['sector_code']]).mean()
    low_mean = df['pct_chg'].where(low_mask).groupby([df['date'], df['sector_code']]).mean()
    spread = (high_mean - low_mean).rename('sw2_high_low_position_return_spread').reset_index()

    merged = _prepare_sector_stock_frame(data, ctx, []).merge(
        spread,
        on=['date', 'sector_code'],
        how='left',
    )
    return pd.Series(
        merged['sw2_high_low_position_return_spread'].values,
        index=data.index,
        dtype='float64',
    )


# 注册所有因子
def _target_wrap(calc):
    return lambda d, ti, c: select_target_series(calc(d, c), target_index=ti)


sector_relative_volatility = register_factor(FactorDefinition(name='sector_relative_volatility', required_columns={'stock_code', 'date', 'close', 'pct_chg'}, requires_sector_data=True, lookback_days=30, calculator=_sector_relative_volatility, calculation_scope='target_with_history_window', target_calculator=_target_wrap(_sector_relative_volatility), description='个股相对于行业的特质波动率 (Std of Active Return)'))
intra_sector_momentum_rank = register_factor(FactorDefinition(name='intra_sector_momentum_rank', required_columns={'stock_code', 'date', 'pct_chg'}, requires_sector_data=True, lookback_days=5, calculator=_intra_sector_momentum_rank, calculation_scope='target_with_history_window', target_calculator=_target_wrap(_intra_sector_momentum_rank), description='SW行业内部个股5日动量的相对排名'))
sector_momentum_dispersion_rank = register_factor(FactorDefinition(name='sector_momentum_dispersion_rank', required_columns={'stock_code', 'date', 'pct_chg'}, requires_sector_data=True, lookback_days=5, calculator=_sector_momentum_dispersion_rank, calculation_scope='target_with_history_window', target_calculator=_target_wrap(_sector_momentum_dispersion_rank), description='SW行业内部个股动量离散度的全市场排名'))
sector_dispersion = register_factor(FactorDefinition(name='sector_dispersion', required_columns={'stock_code', 'date', 'pct_chg'}, requires_sector_data=True, lookback_days=1, calculator=_sector_dispersion, calculation_scope='target_cross_section', target_calculator=_target_wrap(_sector_dispersion), description='板块内部涨跌幅离散度 (选股机会度量)'))
sector_momentum_consistency = register_factor(FactorDefinition(name='sector_momentum_consistency', required_columns={'stock_code', 'date'}, requires_sector_data=True, lookback_days=20, calculator=_sector_momentum_consistency, calculation_scope='target_with_history_window', target_calculator=_target_wrap(_sector_momentum_consistency), description='板块多时间尺度动量的一致性指标'))
sector_correlation_20d = register_factor(FactorDefinition(name='sector_correlation_20d', required_columns={'stock_code', 'date', 'pct_chg'}, requires_sector_data=True, lookback_days=40, calculator=_sector_correlation_20d, calculation_scope='target_with_history_window', target_calculator=_target_wrap(_sector_correlation_20d), description='股票收益与所属SW行业收益的20日滚动相关性'))
sector_volatility_5d = register_factor(FactorDefinition(name='sector_volatility_5d', required_columns={'stock_code', 'date'}, requires_sector_data=True, lookback_days=10, calculator=_sector_volatility_5d, calculation_scope='target_with_history_window', target_calculator=_target_wrap(_sector_volatility_5d), description='SW行业5日波动率的全市场分位排名'))
sector_momentum_interaction = register_factor(FactorDefinition(name='sector_momentum_interaction', required_columns={'stock_code', 'date', 'close'}, requires_sector_data=True, lookback_days=30, calculator=_sector_momentum_interaction_opt, calculation_scope='target_with_history_window', target_calculator=_target_wrap(_sector_momentum_interaction_opt), description='板块动量与个股超额动量的综合评分'))
sw_circ_mv_rank_intra_sector = register_factor(FactorDefinition(name='sw_circ_mv_rank_intra_sector', required_columns={'stock_code', 'date', 'circ_mv'}, requires_sector_data=True, lookback_days=1, calculator=_sw_circ_mv_rank_intra_sector, calculation_scope='target_cross_section', target_calculator=_target_wrap(_sw_circ_mv_rank_intra_sector), description='SW行业内股票流通市值的相对排名（分位数排名）'))
sector_ma5_followthrough_v3 = register_factor(FactorDefinition(name='sector_ma5_followthrough_v3', required_columns={'stock_code', 'date', 'pct_chg'}, requires_sector_data=True, lookback_days=15, calculator=_sector_ma5_followthrough_v3, calculation_scope='target_with_history_window', target_calculator=_target_wrap(_sector_ma5_followthrough_v3), description='行业版MA5跟随顺滑度（基于SW2行业伪指数）'))
limit_up_sw2_concentration = register_factor(FactorDefinition(name='limit_up_sw2_concentration', required_columns={'stock_code', 'date', 'is_limit_up'}, requires_sector_data=True, lookback_days=5, calculator=_limit_up_sw2_concentration, calculation_scope='target_with_history_window', target_calculator=_target_wrap(_limit_up_sw2_concentration), description='涨停股票SW2行业分布集中度（HHI）'))
sw2_high_low_position_return_spread = register_factor(FactorDefinition(name='sw2_high_low_position_return_spread', required_columns={'stock_code', 'date', 'close', 'pct_chg'}, requires_sector_data=True, lookback_days=70, calculator=_sw2_high_low_position_return_spread, calculation_scope='target_with_history_window', target_calculator=_target_wrap(_sw2_high_low_position_return_spread), description='SW2行业内高位股与低位股的当日涨幅差'))

__all__ = [
    'intra_sector_momentum_rank', 'sector_momentum_dispersion_rank', 'sector_dispersion',
    'sector_relative_volatility', 'sector_momentum_consistency', 'sector_correlation_20d',
    'sector_volatility_5d', 'sector_momentum_interaction', 'sw_circ_mv_rank_intra_sector',
    'sector_ma5_followthrough_v3',
    'limit_up_sw2_concentration', 'sw2_high_low_position_return_spread'
]
