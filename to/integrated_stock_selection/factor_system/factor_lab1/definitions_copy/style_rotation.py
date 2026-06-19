"""
风格轮动与市场结构因子（深度优化版）。
核心关注：
1. 板块内部的微观结构（龙头效应、广度）。
2. 全市场的风格偏好（大小盘、高低位）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor



def _get_sector_map_safe(ctx: FactorContext) -> pd.Series:
    """获取行业映射"""
    if ctx.extra and 'stock_to_sw_sector' in ctx.extra:
        return ctx.extra['stock_to_sw_sector']
    raise ValueError("Missing 'stock_to_sw_sector' in context")


def _cs_rank(series: pd.Series) -> pd.Series:
    """横截面分位排名，稳健版本"""
    return series.rank(pct=True, method='average')


def _safe_z(x: pd.Series) -> pd.Series:
    std = x.std()
    if std < 1e-6 or np.isnan(std):
        return x - x.mean()
    return (x - x.mean()) / std


def _sector_rank_change_5d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    sector_map = _get_sector_map_safe(ctx)
    # 直接在原 data 索引上操作，避免重排导致的错位
    df = pd.DataFrame({
        'date': data['date'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown'),
        'pct': data['pct_chg']
    }, index=data.index)

    # 1. 行业收益率排名
    sector_ret = df.groupby(['date', 'sector'])['pct'].mean()
    # 转为宽表：行是日期，列是行业
    sector_rank_df = sector_ret.unstack('sector').rank(axis=1, pct=True)
    
    # 2. 计算 5 日排名变化 (时序)
    # 宽表状态下，shift(5) 会在日期维度上进行，逻辑最清晰
    rank_change_df = sector_rank_df - sector_rank_df.shift(5)
    
    # 3. 映射回个股
    rank_change_s = rank_change_df.stack().rename('rank_change')
    # 使用 join 自动对齐索引，无论 data 是否有序都能正确匹配
    res = df.join(rank_change_s, on=['date', 'sector'], how='left')
    
    return res['rank_change'].fillna(0.0).clip(-1, 1).astype('float64')

sector_rank_change_5d = register_factor(
    FactorDefinition(
        name='sector_rank_change_5d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=10,
        calculator=_sector_rank_change_5d,
        description='SW行业5日排名变化（行业轮动核心因子）',
    )
)


def _sector_momentum_accel_rank(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    # 1. 准备基础数据
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown'),
        'pct': data['pct_chg']
    }, index=data.index)

    # 2. 计算行业收益 (Index: date, sector)
    # 注意：这里使用 groupby.mean() 得到的是以 date 和 sector 为索引的 Series
    sector_ret = df.groupby(['date', 'sector'])['pct'].mean()

    # 3. 计算时序动量 (注意：rolling 前必须按日期排序)
    # unstack 将 sector 变为列，date 变为行，方便按行做时序 rolling
    sector_ret_df = sector_ret.unstack('sector').sort_index()
    
    mom_5d = sector_ret_df.rolling(5, min_periods=3).mean()
    mom_20d = sector_ret_df.rolling(20, min_periods=10).mean()
    accel = mom_5d - mom_20d

    # 4. 计算截面排名 (Cross-sectional Rank)
    # axis=1 表示在每一天(行)对不同行业进行排名
    accel_rank = accel.rank(axis=1, pct=True)

    # 5. 映射回个股 (最关键的一步)
    # stack() 将数据还原回 (date, sector) 索引
    accel_rank_s = accel_rank.stack().rename('factor_value')
    
    # 使用 join 按照 ['date', 'sector'] 将行业排名贴回到原 df 的每一行
    # 这是最稳健且不会出错的对齐方式
    df_result = df.join(accel_rank_s, on=['date', 'sector'], how='left')

    return df_result['factor_value'].fillna(0.5).astype('float64')

sector_momentum_accel_rank = register_factor(
    FactorDefinition(
        name='sector_momentum_accel_rank',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=25,
        calculator=_sector_momentum_accel_rank,
        description='SW行业动量加速度（5d-20d）分位排名',
    )
)



def _sector_turnover_change_5d_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if 'turnover_rate' not in data.columns:
        return pd.Series(0.0, index=data.index)

    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown'),
        'turnover': data['turnover_rate']
    }, index=data.index)

    # 1. 行业平均换手
    sector_to = df.groupby(['date', 'sector'])['turnover'].mean().unstack('sector')
    
    # 2. 计算 5d/20d 比率
    to_5 = sector_to.rolling(5, min_periods=3).mean()
    to_20 = sector_to.rolling(20, min_periods=10).mean()
    ratio = (to_5 / (to_20 + 1e-9)).clip(0.2, 5.0)
    
    # 3. 截面排名 (行业间的换手脉冲强度对比)
    ratio_rank_df = ratio.rank(axis=1, pct=True)
    
    # 4. 广播回个股
    ratio_rank_s = ratio_rank_df.stack().rename('ratio_rank')
    res = df.join(ratio_rank_s, on=['date', 'sector'], how='left')
    
    return res['ratio_rank'].fillna(0.5).astype('float64')

sector_turnover_change_5d_20d = register_factor(
    FactorDefinition(
        name='sector_turnover_change_5d_20d',
        required_columns={'stock_code', 'date', 'turnover_rate'},
        requires_sector_data=True,
        lookback_days=25,
        calculator=_sector_turnover_change_5d_20d,
        description='SW行业换手率5d/20d变化（资金流向）',
    )
)


def _sector_hhi_change(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    if 'volume' not in data.columns:
        return pd.Series(0.0, index=data.index)

    # 1. 构造基础数据（保留原始索引以备最后对齐）
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'stock': data['stock_code'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown'),
        'volume': data['volume']
    }, index=data.index)

    # 2. 【核心步骤】计算行业级别的 HHI
    # 先算行业总成交
    sector_total_vol = df.groupby(['date', 'sector'])['volume'].transform('sum')
    # 算个股在行业内的占比
    df['share'] = df['volume'] / (sector_total_vol + 1e-9)
    # 算占比平方
    df['share_sq'] = df['share'] ** 2
    # 聚合得到行业 HHI (Index: date, sector)
    hhi_series = df.groupby(['date', 'sector'])['share_sq'].sum()

    # 3. 【时序计算】将行业转为宽表进行 rolling
    hhi_wide = hhi_series.unstack('sector').sort_index()
    hhi_5 = hhi_wide.rolling(5, min_periods=3).mean()
    hhi_20 = hhi_wide.rolling(20, min_periods=10).mean()
    hhi_diff = (hhi_5 - hhi_20).fillna(0.0)

    # 4. 【截面排名】让不同日期之间的集中度变化具有可比性
    hhi_rank_wide = hhi_diff.rank(axis=1, pct=True)

    # 5. 【对齐广播】将行业指标贴回个股
    # 将宽表还原为 (date, sector) 索引的 Series
    hhi_final_s = hhi_rank_wide.stack().rename('hhi_rank')
    
    # 通过 join 将 'hhi_rank' 按照 ['date', 'sector'] 两个 key 对应到原表每一行
    df = df.join(hhi_final_s, on=['date', 'sector'], how='left')

    # 6. 返回与输入 data 长度和索引完全一致的 Series
    return df['hhi_rank'].fillna(0.5).astype('float64')


sector_hhi_change = register_factor(
    FactorDefinition(
        name='sector_hhi_change',
        required_columns={'stock_code', 'date', 'volume'},
        requires_sector_data=True,
        lookback_days=25,
        calculator=_sector_hhi_change,
        description='SW行业成交量集中度变化（龙头效应切换）',
    )
)



def _sector_relative_strength_vs_market(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'sector': data['stock_code'].map(sector_map).fillna('Unknown'),
        'pct': data['pct_chg']
    }, index=data.index)

    # 1. 计算行业平均收益
    sector_ret = df.groupby(['date', 'sector'])['pct'].mean().unstack('sector')
    
    # 2. 截面排名 (直接排名已隐含了“相对市场强弱”的关系)
    rel_rank = sector_ret.rank(axis=1, pct=True)
    
    # 3. 映射回个股
    rel_rank_s = rel_rank.stack().rename('rel_rank_factor')
    res = df.join(rel_rank_s, on=['date', 'sector'], how='left')
    
    return res['rel_rank_factor'].fillna(0.5).astype('float64')


sector_relative_strength_vs_market = register_factor(
    FactorDefinition(
        name='sector_relative_strength_vs_market',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=10,
        calculator=_sector_relative_strength_vs_market,
        description='SW行业相对全市场强弱（行业超额收益）',
    )
)



# ==============================================================================
# 1. 赫芬达尔集中度 (HHI Concentration) - 龙头效应的精确度量
# ==============================================================================
def _sector_concentration_hhi(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    板块成交量集中度 (HHI Index)
    
    逻辑：
    HHI = Sum((Volume_i / Total_Volume)^2)
    
    值域 [0, 1]:
    - 接近 1.0: 极度集中（全板块只炒一只股，如妖股吸血）。
    - 接近 1/N: 极度分散（普涨普跌）。
    
    这对 LightGBM 很有用：
    - 高 HHI 时，模型应倾向于买 Volume_Rank 高的票。
    - 低 HHI 时，模型可挖掘补涨股。
    """
    if 'volume' not in data.columns:
        return pd.Series(0.0, index=data.index)

    sector_map = _get_sector_map_safe(ctx)
    
    # 构造工作表
    df = pd.DataFrame({
        'vol': data['volume'].values,
        'date': data['date'].values,
        'sector': data['stock_code'].map(sector_map).fillna('Unknown').values
    }, index=data.index)
    
    # 1. 计算板块总成交量
    sector_total_vol = df.groupby(['date', 'sector'])['vol'].transform('sum')
    
    # 2. 计算市占率 (Market Share)
    share = df['vol'] / (sector_total_vol + 1e-9)
    
    # 3. 计算 HHI (Share Squared Sum)
    # 这一步需要再次 groupby sum
    df['share_sq'] = share ** 2
    hhi = df.groupby(['date', 'sector'])['share_sq'].transform('sum')
    
    # 4. 标准化因子 (广播回个股，描述的是环境)
    return hhi.fillna(0.0).astype('float64')

sector_concentration = register_factor(
    FactorDefinition(
        name='sector_concentration',
        required_columns={'stock_code', 'date', 'volume'},
        requires_sector_data=True,
        lookback_days=1,
        calculator=_sector_concentration_hhi,
        description='板块成交量赫芬达尔集中度(HHI)',
    )
)

# ==============================================================================
# 2. 板块广度推力 (Sector Breadth Thrust) - 趋势健康度
# ==============================================================================
def _sector_breadth_thrust(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    板块广度推力
    
    逻辑：
    不仅仅看涨跌家数比，还要看“强势股”占比。
    定义“强势”：Close > MA20 (或者当日涨幅 > 0)
    这里为了简单高效，使用：(上涨家数 - 下跌家数) / 总家数
    
    优化点：
    我们会计算一个 5日平滑版本，避免单日噪音。
    """
    sector_map = _get_sector_map_safe(ctx)
    
    df = pd.DataFrame({
        'pct': data['pct_chg'].values,
        'date': data['date'].values,
        'sector': data['stock_code'].map(sector_map).fillna('Unknown').values
    }, index=data.index)
    
    # 1. 计算个股涨跌状态 (+1, -1)
    # 0.005 (0.5%) 作为红盘阈值，过滤微涨微跌的噪音
    df['status'] = np.where(df['pct'] > 0.2, 1.0, np.where(df['pct'] < -0.2, -1.0, 0.0))
    
    # 2. 计算板块广度 (Mean Status)
    # 结果在 [-1, 1] 之间
    breadth = df.groupby(['date', 'sector'])['status'].transform('mean')
    
    # 3. 广播回个股
    return breadth.fillna(0.0).astype('float64')

sector_breadth = register_factor(
    FactorDefinition(
        name='sector_breadth',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=1,
        calculator=_sector_breadth_thrust,
        description='板块净上涨家数占比(广度推力)',
    )
)

# ==============================================================================
# 3. 大小盘风格轮动 (Size Style Rotation) - 新增顶级因子
# ==============================================================================
def _size_rotation_strength(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    大小盘风格轮动强度
    
    逻辑：
    全市场（或分组）中：小盘股平均涨幅 - 大盘股平均涨幅。
    
    广播逻辑：
    这是一个 Context 因子。所有股票共享同一个值（或者按行业细分）。
    为了捕捉全市场风格，我们计算全市场的 Size Premium。
    
    信号含义：
    > 0: 小盘股强（炒妖、炒概念）。
    < 0: 大盘股强（护盘、蓝筹行情）。
    """
    if 'circ_mv' not in data.columns or 'pct_chg' not in data.columns:
        return pd.Series(0.0, index=data.index)
    
    # 1. 定义大盘/小盘 (Top 20% vs Bottom 20% by Market Cap)
    # 必须按日期分组计算分位数
    df = data[['date', 'circ_mv', 'pct_chg']].copy()
    
    # 计算每日的分位数阈值
    # transform 比 apply 快
    groups = df.groupby('date')['circ_mv']
    q80 = groups.transform(lambda x: x.quantile(0.8))
    q20 = groups.transform(lambda x: x.quantile(0.2))
    
    # 2. 标记风格
    is_large = df['circ_mv'] >= q80
    is_small = df['circ_mv'] <= q20
    
    # 3. 计算每日风格收益
    # 这里的技巧是：利用 boolean indexing 和 groupby
    # 为了效率，我们先构建 mask
    
    # 大盘股日均涨幅
    large_ret = df[is_large].groupby('date')['pct_chg'].mean()
    # 小盘股日均涨幅
    small_ret = df[is_small].groupby('date')['pct_chg'].mean()
    
    # 4. 计算风格差 (Size Premium)
    size_premium = small_ret - large_ret
    
    # 5. 广播回原数据
    # map 操作
    return data['date'].map(size_premium).fillna(0.0).astype('float64')

size_rotation_strength = register_factor(
    FactorDefinition(
        name='size_rotation_strength',
        required_columns={'stock_code', 'date', 'circ_mv', 'pct_chg'},
        lookback_days=1,
        calculator=_size_rotation_strength,
        description='全市场小盘股相对于大盘股的超额收益(风格轮动)',
    )
)

# ==============================================================================
# 4. 高低位切换强度 (Reversal Style Rotation) - 新增顶级因子
# ==============================================================================
def _value_growth_rotation(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    高低位切换强度 (Reversal Premium)
    
    逻辑：
    低位股（过去20日跌幅大）今天涨幅 - 高位股（过去20日涨幅大）今天涨幅。
    
    信号含义：
    > 0: 高低切（低位补涨，高位回调），市场偏谨慎/反转。
    < 0: 强者恒强（高位继续突进），市场偏动量/情绪高涨。
    """
    if 'close' not in data.columns or 'pct_chg' not in data.columns:
        return pd.Series(0.0, index=data.index)
        
    # 1. 计算过去20日涨幅 (Momentum)
    # 使用 shift(1) 避免用到今天的涨幅，我们是用昨天的状态来分组
    grouped = data.groupby('stock_code', sort=False)['close']
    prev_close = grouped.shift(1)
    prev_20 = grouped.shift(21) # T-21
    
    mom_20 = (prev_close - prev_20) / (prev_20 + 1e-9)
    
    # 2. 定义高位/低位
    df = pd.DataFrame({
        'mom': mom_20.values,
        'pct': data['pct_chg'].values,
        'date': data['date'].values
    }, index=data.index).dropna()
    
    if df.empty:
        return pd.Series(0.0, index=data.index)

    groups = df.groupby('date')['mom']
    q80 = groups.transform(lambda x: x.quantile(0.8)) # 高位阈值
    q20 = groups.transform(lambda x: x.quantile(0.2)) # 低位阈值
    
    is_high = df['mom'] >= q80
    is_low = df['mom'] <= q20
    
    # 3. 计算收益差
    high_ret = df[is_high].groupby('date')['pct'].mean()
    low_ret = df[is_low].groupby('date')['pct'].mean()
    
    # Reversal Premium = Low_Ret - High_Ret
    rev_premium = low_ret - high_ret
    
    # 4. 广播
    # 注意要对齐索引，因为 df可能有 dropna
    mapped_val = data['date'].map(rev_premium)
    
    # 平滑一下 (3日均值)，减少单日噪音，我们要的是风格趋势
    # 但 map 后不再是时序连续的 Series（对于单个股票），所以不能直接 rolling
    # 正确做法：先在 daily level rolling，再 map
    
    rev_premium_smooth = rev_premium.rolling(3).mean()
    
    return data['date'].map(rev_premium_smooth).fillna(0.0).astype('float64')

value_growth_rotation = register_factor(
    FactorDefinition(
        name='value_growth_rotation',
        required_columns={'stock_code', 'date', 'close', 'pct_chg'},
        lookback_days=30,
        calculator=_value_growth_rotation,
        description='低位股相对于高位股的超额收益(高低切换强度)',
    )
)

__all__ = [
    'sector_concentration',
    'sector_breadth',
    'size_rotation_strength',
    'value_growth_rotation',
    'sector_rank_change_5d',
    'sector_momentum_accel_rank',
    'sector_turnover_change_5d_20d',
    'sector_hhi_change',
    'sector_relative_strength_vs_market'

]

