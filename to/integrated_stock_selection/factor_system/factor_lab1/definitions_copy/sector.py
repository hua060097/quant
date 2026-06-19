"""板块相关因子。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 辅助工具：获取行业映射
# ==============================================================================
def _get_sector_map_safe(ctx: FactorContext) -> pd.Series:
    if ctx.extra and 'stock_to_sw_sector' in ctx.extra:
        return ctx.extra['stock_to_sw_sector']
    raise ValueError("Sector data missing in context")


class _SectorDataHelper:
    def __init__(self, data: pd.DataFrame, ctx: FactorContext) -> None:
        import logging
        logger = logging.getLogger(__name__)

        self.ctx = ctx
        self.original_index = data.index
        self.data = data.copy()
        self.data['__orig_index'] = self.data.index

        # [SW行业编码] 获取SW行业映射（必需）
        self.sw_sector_mapping = None
        if ctx.extra and 'stock_to_sw_sector' in ctx.extra:
            self.sw_sector_mapping = ctx.extra['stock_to_sw_sector']
            logger.info(f"[SW行业编码] 获取到SW行业映射，包含{len(self.sw_sector_mapping)}只股票")
        else:
            raise ValueError("SW行业映射必须通过ctx.extra['stock_to_sw_sector']提供")

        # [SW行业编码] 基于股票数据构建SW行业平均数据
        self.sector_data = self._build_sw_sector_data(data)
        self._attach_sector_code()
        self._validate_sector_columns()

        try:
            logger.debug(f"准备排序sector_data, shape={self.sector_data.shape}")
            self.sector_data = self.sector_data.sort_values(['date', 'sector_code']).reset_index(drop=True)
        except Exception as e:
            logger.error(f"排序sector_data失败: {e}")
            logger.error(f"sector_data列: {list(self.sector_data.columns)}")
            logger.error(f"sector_data前5行:\n{self.sector_data.head()}")
            raise

    def _attach_sector_code(self) -> None:
        if 'sector_code' in self.data.columns:
            pass
        else:
            # [SW行业编码] 使用SW行业映射
            if self.sw_sector_mapping is None:
                raise ValueError('SW行业映射必须通过ctx.extra[\"stock_to_sw_sector\"]提供')
            self.data['sector_code'] = self.data['stock_code'].map(self.sw_sector_mapping)

            # 检查是否有股票缺少SW行业映射，如果有则直接报错
            if self.data['sector_code'].isna().any():
                missing_stocks = self.data.loc[self.data['sector_code'].isna(), 'stock_code'].unique()
                raise ValueError(f'以下{len(missing_stocks)}只股票缺少SW行业映射: {missing_stocks[:10]}{"..." if len(missing_stocks) > 10 else ""}')

    def _build_sw_sector_data(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        基于股票数据计算SW行业平均值

        Args:
            stock_data: 原始股票数据，包含股票代码、日期、收益率等

        Returns:
            DataFrame: 行业平均数据，包含列 ['date', 'sector_code', 'pct_chg_avg']
        """
        import logging
        logger = logging.getLogger(__name__)

        # 复制股票数据并添加SW行业列
        sw_data = stock_data.copy()
        sw_data['sector_code'] = sw_data['stock_code'].map(self.sw_sector_mapping)

        # 检查是否有股票缺少SW行业映射
        if sw_data['sector_code'].isna().any():
            missing_stocks = sw_data.loc[sw_data['sector_code'].isna(), 'stock_code'].unique()
            raise ValueError(f'以下{len(missing_stocks)}只股票缺少SW行业映射: {missing_stocks[:10]}{"..." if len(missing_stocks) > 10 else ""}')

        # 检查必需字段
        if 'pct_chg' not in sw_data.columns:
            raise ValueError('股票数据需要pct_chg列用于计算行业平均收益率')

        # 按日期和行业计算平均值
        logger.info(f"[SW行业编码] 计算SW行业平均值，股票数: {len(sw_data)}，行业数: {sw_data['sector_code'].nunique()}")

        sector_avg = sw_data.groupby(['date', 'sector_code']).agg({
            'pct_chg': 'mean'  # 收益率平均值
        }).reset_index()

        # 重命名为pct_chg_avg以保持兼容性
        sector_avg = sector_avg.rename(columns={'pct_chg': 'pct_chg_avg'})

        logger.info(f"[SW行业编码] 行业平均数据构建完成，记录数: {len(sector_avg)}")
        return sector_avg

    def _validate_sector_columns(self) -> None:
        required = {'date', 'sector_code'}
        if not required.issubset(self.sector_data.columns):
            raise ValueError(f'sector_data 需要包含列: {sorted(required)}')

    def attach_sector_features(self, columns: list[str]) -> pd.DataFrame:
        import logging
        logger = logging.getLogger(__name__)

        # 确保columns是字符串列表，而不是嵌套列表
        if not isinstance(columns, list):
            columns = [columns]
        # 扁平化columns，防止嵌套列表
        flat_columns = []
        for col in columns:
            if isinstance(col, list):
                flat_columns.extend(col)
            else:
                flat_columns.append(col)
        columns = flat_columns

        missing = [c for c in columns if c not in self.sector_data.columns]
        if missing:
            raise ValueError(f'sector_data 缺少列: {missing}')

        try:
            logger.debug(f"准备merge，data.shape={self.data.shape}, sector_data.shape={self.sector_data.shape}")
            merged = self.data.merge(
                self.sector_data[['date', 'sector_code'] + columns],
                on=['date', 'sector_code'],
                how='left',
                suffixes=('', '_sector'),
            )
        except Exception as e:
            logger.error(f"Merge失败: {e}")
            logger.error(f"columns={columns}")
            logger.error(f"data列: {list(self.data.columns)}")
            logger.error(f"sector_data列: {list(self.sector_data.columns)}")
            raise

        return merged


# ==============================================================================
# 1. 行业特质波动率 (Idiosyncratic Volatility) - 高效替代版
# ==============================================================================
def _sector_relative_volatility(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    行业特质波动率 (Idiosyncratic Volatility) - 生产级修复版
    
    修复重点：
    1. 使用 transform 保持索引自动对齐，彻底杜绝 .values 导致的错位。
    2. 采用 MAD (Median Absolute Deviation) 动态去极值，取代硬编码 clip(3)。
    3. 只对有效值进行横截面标准化，避免“温室效应”导致的数值坍缩。
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(0.0, index=data.index)

    # 1. 构造临时表并获取行业
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'date': data['date'],
        'stock': data['stock_code'],
        'pct': data['pct_chg']
    }, index=data.index)
    df['sector'] = df['stock'].map(sector_map).fillna('Unknown')

    # 2. 计算行业超额收益 (Active Return)
    # transform 会按原始索引顺序返回，无需担心对齐
    s_ret = df.groupby(['date', 'sector'])['pct'].transform('mean')
    df['active_ret'] = df['pct'] - s_ret

    # 3. 计算 20 日滚动特质波动率
    # 必须先排好序再计算 rolling，计算完后 transform 会自动按 index 贴回原位
    df_sorted = df.sort_values(['stock', 'date'])
    df['idio_vol'] = df_sorted.groupby('stock')['active_ret'].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )

    # 4. 专业标准化流程 (去极值 + Z-Score)
    def robust_standardize(s: pd.Series):
        """对单个日期截面的 Series 进行标准化"""
        valid = s.dropna()
        if len(valid) < 10: # 样本太少不处理
            return s * 0
        
        # (1) MAD 去极值：比 clip(-3, 3) 科学得多，能自适应市场波动
        median = valid.median()
        mad = (valid - median).abs().median()
        limit = 3 * 1.4826 * mad
        # 将极端值平滑到边界，而不是直接丢弃
        valid_clipped = valid.clip(median - limit, median + limit)
        
        # (2) Z-Score：将量纲统一
        return (valid_clipped - valid_clipped.mean()) / (valid_clipped.std() + 1e-8)

    # 按日期分组应用标准化。注意：这里不 fillna(0)，只对非 NaN 进行计算
    df['final_score'] = df.groupby('date')['idio_vol'].transform(robust_standardize)

    # 5. 最终清理：填充 NaN 为 0 (中性处理)
    # 确保返回的 Series 索引与输入 data 完全一致
    return df['final_score'].fillna(0.0).astype('float64')



def _sector_volatility_5d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    计算SW行业5日波动率在全市场的分位排名
    量化工程实现：使用SW行业5日收益率波动率进行横向排名

    计算逻辑：
    1. 计算每个SW行业在每个日期的5日收益率波动率
    2. 对波动率进行横向排名（分位数排名）
    3. 返回股票所属SW行业的波动率排名值

    技术特点：
    - 使用5日窗口计算行业波动率，平衡稳定性和敏感性
    - 横截面排名便于不同行业间的比较
    - 分位数排名确保因子分布的稳定性
    - 排名范围：[0, 1]，值越大波动率越高
    """
    helper = _SectorDataHelper(data, ctx)

    # 获取包含pct_chg_avg的sector_data
    sector_df = helper.sector_data.copy()

    # 检查必需字段
    if 'pct_chg_avg' not in sector_df.columns:
        raise ValueError('sector_data 需要 pct_chg_avg 列用于计算SW行业波动率。')

    # 准备数据：确保日期格式正确并排序
    sector_df['date_dt'] = pd.to_datetime(sector_df['date'])
    # 🔧 时序安全：按sector_code和date排序，确保每个sector内部按日期升序
    sector_df = sector_df.sort_values(['sector_code', 'date_dt']).reset_index(drop=True)

    # 计算每个SW行业的5日滚动波动率
    # 使用rolling计算标准差，min_periods=3确保有足够数据
    sector_df['sector_volatility_5d'] = sector_df.groupby('sector_code')['pct_chg_avg'].transform(
        lambda x: x.rolling(window=5, min_periods=3).std() * np.sqrt(252)  # 年化波动率
    )

    # 计算每个日期横向排名（分位数排名）
    # 在每个交易日，对所有行业的波动率进行排名
    sector_df['volatility_rank'] = sector_df.groupby('date_dt')['sector_volatility_5d'].rank(
        pct=True, method='min'
    ).fillna(0.5)

    # 使用helper.attach_sector_features获取行业波动率排名
    helper.sector_data = sector_df
    merged = helper.attach_sector_features(['volatility_rank'])
    merged = merged.set_index('__orig_index')

    result = merged['volatility_rank'].fillna(0.5).reindex(helper.original_index).astype('float64')
    return result


def _intra_sector_momentum_rank(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    计算行业内个股动量排名
    量化工程实现：在SW行业内部进行个股动量相对排名

    计算逻辑：
    1. 计算个股5日动量（累计收益率）
    2. 在每个SW行业内部对个股动量进行排名
    3. 返回个股在所属行业内的相对位置

    技术特点：
    - 使用5日动量平衡短期趋势和数据稳定性
    - 行业内相对排名消除行业整体偏误
    - 分位数排名便于不同行业间的比较
    """
    import logging
    logger = logging.getLogger(__name__)

    # 获取SW行业映射
    sw_sector_mapping = ctx.extra.get('stock_to_sw_sector')
    if sw_sector_mapping is None:
        raise ValueError("SW行业映射必须通过ctx.extra['stock_to_sw_sector']提供")

    # 准备数据
    data_work = data.copy()
    data_work['sector_code'] = data_work['stock_code'].map(sw_sector_mapping)

    # 检查是否有股票缺少SW行业映射
    if data_work['sector_code'].isna().any():
        missing_stocks = data_work.loc[data_work['sector_code'].isna(), 'stock_code'].unique()
        raise ValueError(f'以下{len(missing_stocks)}只股票缺少SW行业映射: {missing_stocks[:10]}{"..." if len(missing_stocks) > 10 else ""}')

    # 计算个股5日动量
    data_work = data_work.sort_values(['stock_code', 'date'])
    data_work['momentum_5d'] = data_work.groupby('stock_code')['pct_chg'].rolling(
        window=5, min_periods=3
    ).sum().reset_index(level=0, drop=True)

    # 在每个行业内部进行排名
    data_work['date_dt'] = pd.to_datetime(data_work['date'])
    data_work = data_work.sort_values(['date_dt', 'sector_code'])

    # 行业内分位数排名
    data_work['intra_sector_rank'] = data_work.groupby(['date_dt', 'sector_code'])['momentum_5d'].rank(
        pct=True, method='min'
    ).fillna(0.5)

    logger.info(f"[SW行业编码] 行业内动量排名计算完成，股票数: {len(data_work)}")

    # 构建结果Series，索引与原始数据一致
    result = pd.Series(data_work['intra_sector_rank'].values, index=data.index, dtype='float64')
    return result


def _sector_momentum_dispersion_rank(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    计算行业动量离散度排名
    量化工程实现：衡量SW行业内部个股动量的分化程度

    计算逻辑：
    1. 计算每个SW行业内部个股动量的标准差
    2. 对行业动量离散度进行横向排名
    3. 离散度越高，行业内部分化越严重

    理论基础：
    - 高离散度可能预示行业轮动或风格切换
    - 低离散度表示行业整体趋势一致
    - 可用于行业择时和风格配置
    """
    import logging
    logger = logging.getLogger(__name__)

    # 获取SW行业映射
    sw_sector_mapping = ctx.extra.get('stock_to_sw_sector')
    if sw_sector_mapping is None:
        raise ValueError("SW行业映射必须通过ctx.extra['stock_to_sw_sector']提供")

    # 准备数据
    data_work = data.copy()
    data_work['sector_code'] = data_work['stock_code'].map(sw_sector_mapping)

    # 检查是否有股票缺少SW行业映射
    if data_work['sector_code'].isna().any():
        missing_stocks = data_work.loc[data_work['sector_code'].isna(), 'stock_code'].unique()
        raise ValueError(f'以下{len(missing_stocks)}只股票缺少SW行业映射: {missing_stocks[:10]}{"..." if len(missing_stocks) > 10 else ""}')

    # 计算个股5日动量
    data_work = data_work.sort_values(['stock_code', 'date'])
    data_work['momentum_5d'] = data_work.groupby('stock_code')['pct_chg'].rolling(
        window=5, min_periods=3
    ).sum().reset_index(level=0, drop=True)

    # 计算每个SW行业每天的动量离散度（标准差）
    data_work['date_dt'] = pd.to_datetime(data_work['date'])
    sector_dispersion = data_work.groupby(['date', 'sector_code'])['momentum_5d'].agg([
        'count',  # 行业内股票数量
        'std'     # 动量标准差
    ]).reset_index()

    # 过滤掉行业内股票数量过少的记录（确保统计有效性）
    sector_dispersion = sector_dispersion[sector_dispersion['count'] >= 3]
    sector_dispersion['momentum_dispersion'] = sector_dispersion['std'].fillna(0.0)

    # 计算行业动量离散度的排名
    sector_dispersion['date_dt'] = pd.to_datetime(sector_dispersion['date'])
    sector_dispersion['dispersion_rank'] = sector_dispersion.groupby('date_dt')['momentum_dispersion'].rank(
        pct=True, method='min'
    ).fillna(0.5)

    # 将排名映射回原始股票数据
    data_with_rank = data_work.merge(
        sector_dispersion[['date', 'sector_code', 'dispersion_rank']],
        on=['date', 'sector_code'],
        how='left'
    )

    # 对于无法匹配的记录（如行业内股票数量不足），使用中性值
    data_with_rank['dispersion_rank'] = data_with_rank['dispersion_rank'].fillna(0.5)

    logger.info(f"[SW行业编码] 行业动量离散度排名计算完成")

    # 构建结果Series，索引与原始数据一致
    result = pd.Series(data_with_rank['dispersion_rank'].values, index=data.index, dtype='float64')
    return result


# ==============================================================================
# 2. 板块内离散度 (Sector Dispersion) - 新增高价值因子
# ==============================================================================
def _sector_dispersion(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    板块内部收益离散度
    
    逻辑：
    计算当日板块内所有股票涨跌幅的标准差 (Cross-sectional Std)。
    - 高离散度：板块内分化严重，选股Alpha机会大（如：龙头涨，杂毛跌）。
    - 低离散度：板块齐涨齐跌，Beta主导，Alpha难做。
    
    注意：这是一个Context因子（同板块股票值相同），描述环境。
    """
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'pct': data['pct_chg'].values,
        'date': data['date'].values,
        'sector': data['stock_code'].map(sector_map).fillna('Unknown').values
    }, index=data.index)
    
    # 计算截面标准差
    dispersion = df.groupby(['date', 'sector'])['pct'].transform('std')
    
    return dispersion.fillna(0.0).astype('float64')




def _sector_momentum_consistency(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    计算SW行业动量一致性
    修改：使用SW行业，每个股票只属于一个SW行业

    逻辑：
    1. 先计算每个时间窗口(3d,5d,10d,20d)的SW行业排名
    2. 然后计算这4个排名之间的标准差
    3. 转换为一致性得分：1 - (std / max_std)
    """
    helper = _SectorDataHelper(data, ctx)
    sector_df = helper.sector_data.copy()

    # 检查必需字段（SW行业数据包含pct_chg_avg列）
    if 'pct_chg_avg' not in sector_df.columns:
        raise ValueError('sector_data 需要 pct_chg_avg 列用于计算SW行业平均收益率。')

    # 准备数据
    sector_df['date_dt'] = pd.to_datetime(sector_df['date'])
    sector_df = sector_df.sort_values(['sector_code', 'date_dt'])

    # 步骤1：内部计算sector_return_rank_3d/5d/10d/20d（使用平均日收益率）
    periods = [3, 5, 10, 20]
    rank_cols = []

    for period in periods:
        # 计算SW行业的period日平均日收益率
        sector_df[f'avg_return_{period}d'] = sector_df.groupby('sector_code')['pct_chg_avg'].rolling(
            window=period, min_periods=max(1, period // 2)
        ).mean().reset_index(level=0, drop=True)

        # 对平均日收益率进行排名
        rank_col = f'sector_return_rank_{period}d'
        sector_df[rank_col] = sector_df.groupby('date_dt')[f'avg_return_{period}d'].rank(
            pct=True, method='min'
        ).fillna(0.5)
        rank_cols.append(rank_col)

    # [SW行业编码] 简化逻辑：直接使用helper.attach_sector_features获取所有排名
    helper.sector_data = sector_df
    merged = helper.attach_sector_features(rank_cols).set_index('__orig_index')
    ranks_df = merged[rank_cols].fillna(0.5).reindex(helper.original_index)

    # 步骤2：计算4个排名之间的标准差
    # 使用skipna=True，只计算有效值的标准差
    rank_std = ranks_df.std(axis=1, skipna=True)

    # 步骤3：转换为一致性得分
    max_std = rank_std.max()
    if pd.isna(max_std) or max_std <= 0:
        max_std = 1.0

    consistency = 1 - (rank_std / max_std)

    # 最后才fillna，对于没有任何行业数据的股票填充0.5
    consistency = consistency.fillna(0.5)

    return consistency.astype('float64')


def _sector_correlation_20d(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    计算股票与SW行业的20日滚动相关性
    修改：使用SW行业，每个股票只属于一个SW行业

    逻辑：
    1. 计算每个SW行业所有股票的平均收益率
    2. 计算股票收益率与所属SW行业收益率的20日滚动相关性
    """
    import logging
    logger = logging.getLogger(__name__)

    # 获取SW行业映射
    sw_sector_mapping = ctx.extra.get('stock_to_sw_sector')
    if sw_sector_mapping is None:
        raise ValueError("SW行业映射必须通过ctx.extra['stock_to_sw_sector']提供")

    # 准备数据
    data_work = data.copy()
    data_work['date_dt'] = pd.to_datetime(data_work['date'])
    # 🆕 修复：去除错误的除以100操作，pct_chg已经是小数形式
    data_work['stock_return_dec'] = data_work['pct_chg'].astype(float)
    data_work['sector_code'] = data_work['stock_code'].map(sw_sector_mapping)

    # 检查是否有股票缺少SW行业映射
    if data_work['sector_code'].isna().any():
        missing_stocks = data_work.loc[data_work['sector_code'].isna(), 'stock_code'].unique()
        raise ValueError(f'以下{len(missing_stocks)}只股票缺少SW行业映射: {missing_stocks[:10]}{"..." if len(missing_stocks) > 10 else ""}')

    logger.info(f"[SW行业编码] 计算SW行业相关性，股票数: {len(data_work)}，行业数: {data_work['sector_code'].nunique()}")

    # 计算每个SW行业每天的收益率平均值
    sector_avg_return = data_work.groupby(['date', 'sector_code'])['pct_chg'].mean().reset_index()
    sector_avg_return.rename(columns={'pct_chg': 'sector_avg_pct_chg'}, inplace=True)

    # 将行业平均收益率合并回原始数据
    data_with_sector = data_work.merge(
        sector_avg_return,
        on=['date', 'sector_code'],
        how='left'
    )
    # 🆕 修复：去除错误的除以100操作，sector_avg_pct_chg已经是小数形式
    data_with_sector['sector_return_dec'] = data_with_sector['sector_avg_pct_chg'].astype(float)
    data_with_sector['sector_return_dec'] = data_with_sector['sector_return_dec'].fillna(0.0)

    # 计算每只股票的20日滚动相关性
    data_with_sector = data_with_sector.sort_values(['stock_code', 'date_dt'])

    def calc_rolling_corr(group):
        return group['stock_return_dec'].rolling(20, min_periods=10).corr(group['sector_return_dec'])

    corr = data_with_sector.groupby('stock_code').apply(calc_rolling_corr).reset_index(level=0, drop=True)

    # 处理异常值和缺失值
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 构建结果Series，索引与原始数据一致
    result = pd.Series(corr.values, index=data.index, dtype='float64')

    logger.info(f"[SW行业编码] SW行业相关性计算完成")
    return result



# ==============================================================================
# 3. 板块动量与个股强度的交互 (Sector Momentum Interaction) - 优化版
# ==============================================================================
def _sector_momentum_interaction_opt(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    板块动量 * 个股相对强度
    
    逻辑：
    (板块20日涨幅) * (个股20日涨幅 - 板块20日涨幅)
    含义：在强势板块中，寻找跑赢板块的领头羊。
    """
    sector_map = _get_sector_map_safe(ctx)
    
    # 1. 计算个股20日动量
    grouped = data.groupby('stock_code', sort=False)['close']
    mom_20d = grouped.pct_change(20).fillna(0)
    
    # 2. 计算板块20日动量 (这里用个股动量的均值近似，比指数更灵敏)
    df = pd.DataFrame({'mom': mom_20d.values, 'date': data['date'].values, 'sector': data['stock_code'].map(sector_map).values}, index=data.index)
    sector_mom = df.groupby(['date', 'sector'])['mom'].transform('mean')
    
    # 3. 交互计算
    # 个股相对强度 = 个股动量 - 板块动量
    relative_strength = mom_20d - sector_mom
    
    # 交互项：如果板块强(>0)且个股强(>0)，乘积很大 -> 双击
    # 为了避免负负得正（弱势板块里的弱势股），我们保留符号逻辑
    # 更好的逻辑：Sign(Sector) * Abs(Sector) * Relative? 
    # 简单逻辑：Sector_Mom + Relative_Strength (其实就是个股动量)
    # 高级逻辑：我们希望捕捉“板块热度”带来的溢价。
    
    # 修正公式：Sector_Rank * Intra_Sector_Rank (用Rank做乘法更稳健)
    # 但为了保持数值连续性，我们用 Z-Score
    
    # Z-Score 化
    def z_score(x): return (x - x.mean()) / (x.std() + 1e-9)
    
    sec_z = df.groupby('date')['mom'].transform(z_score) # 板块动量在全市场的Z分
    rel_z = (relative_strength - relative_strength.groupby(df['date']).transform('mean')) / relative_strength.groupby(df['date']).transform('std')
    
    # 最终因子：板块强 + 个股更强
    interaction = sec_z + rel_z 
    
    return interaction.fillna(0.0).astype('float64')



def _sw_circ_mv_rank_intra_sector(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    SW行业内流通市值的相对排名 (0~1)
    大票/小票风格在行业内的体现。
    """
    if 'circ_mv' not in data.columns:
        return pd.Series(0.5, index=data.index)
        
    sector_map = _get_sector_map_safe(ctx)
    df = pd.DataFrame({
        'mv': data['circ_mv'].values,
        'date': data['date'].values,
        'sector': data['stock_code'].map(sector_map).values
    }, index=data.index)
    
    # 板块内排名
    rank = df.groupby(['date', 'sector'])['mv'].rank(pct=True)
    return rank.fillna(0.5).astype('float64')



# 1. 替代原 idiosyncratic_volatility_sector (更名为 relative_volatility)
sector_relative_volatility = register_factor(
    FactorDefinition(
        name='sector_relative_volatility',
        required_columns={'stock_code', 'date', 'close', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=30,
        calculator=_sector_relative_volatility,
        description='个股相对于行业的特质波动率 (Std of Active Return)',
    )
)

# 行业内个股动量和离散度因子
intra_sector_momentum_rank = register_factor(
    FactorDefinition(
        name='intra_sector_momentum_rank',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=5,
        calculator=_intra_sector_momentum_rank,
        description='SW行业内部个股5日动量的相对排名',
    )
)

sector_momentum_dispersion_rank = register_factor(
    FactorDefinition(
        name='sector_momentum_dispersion_rank',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=5,
        calculator=_sector_momentum_dispersion_rank,
        description='SW行业内部个股动量离散度的全市场排名',
    )
)


# 2. 新增：板块离散度 (环境因子)
sector_dispersion = register_factor(
    FactorDefinition(
        name='sector_dispersion',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=1,
        calculator=_sector_dispersion,
        description='板块内部涨跌幅离散度 (选股机会度量)',
    )
)


sector_momentum_consistency = register_factor(
    FactorDefinition(
        name='sector_momentum_consistency',
        required_columns={'stock_code', 'date'},
        requires_sector_data=True,
        lookback_days=20,
        calculator=_sector_momentum_consistency,
        description='板块多时间尺度动量的一致性指标',
    )
)

sector_correlation_20d = register_factor(
    FactorDefinition(
        name='sector_correlation_20d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_sector_data=True,
        lookback_days=40,
        calculator=_sector_correlation_20d,
        description='股票收益与所属SW行业收益的20日滚动相关性',
    )
)

sector_volatility_5d = register_factor(
    FactorDefinition(
        name='sector_volatility_5d',
        required_columns={'stock_code', 'date'},
        requires_sector_data=True,
        lookback_days=10,
        calculator=_sector_volatility_5d,
        description='SW行业5日波动率的全市场分位排名',
    )
)


# 3. 优化：板块动量交互
sector_momentum_interaction = register_factor(
    FactorDefinition(
        name='sector_momentum_interaction',
        required_columns={'stock_code', 'date', 'close'},
        requires_sector_data=True,
        lookback_days=30,
        calculator=_sector_momentum_interaction_opt,
        description='板块动量与个股超额动量的综合评分',
    )
)



sw_circ_mv_rank_intra_sector = register_factor(
    FactorDefinition(
        name='sw_circ_mv_rank_intra_sector',
        required_columns={'stock_code', 'date', 'circ_mv'},
        requires_sector_data=True,
        lookback_days=1,
        calculator=_sw_circ_mv_rank_intra_sector,
        description='SW行业内股票流通市值的相对排名（分位数排名）',
    )
)




__all__ = [
    'intra_sector_momentum_rank',
    'sector_momentum_dispersion_rank',
    'sector_dispersion',
    'sector_relative_volatility',
    'sector_momentum_consistency',
    'sector_correlation_20d',
    'sector_volatility_5d',
    'sector_momentum_interaction',
    'sw_circ_mv_rank_intra_sector'
]
