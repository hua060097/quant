def _filter_new_stocks_batch(stock_data: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    批量过滤新股数据（重构版本 - 修复语法错误和性能问题）

    参数:
    - stock_data: 股票数据DataFrame，必须包含stock_code、date、pct_chg列
    - logger: 日志记录器

    返回:
    - pd.DataFrame: 过滤后的股票数据
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import logging

    # 处理logger参数
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # 验证输入数据
        if not _validate_stock_data_main(stock_data, logger):
            return stock_data

        # 快速检查：如果数据量很小，直接返回
        if len(stock_data) < 1000:
            logger.info(f"[FILTER] 数据量较小({len(stock_data)})，跳过新股过滤")
            return stock_data

        # 获取基本信息
        all_dates = sorted(stock_data['date'].unique())
        all_stocks = stock_data['stock_code'].unique()
        logger.info(f"[FILTER] 数据统计: {len(all_stocks)} 只股票，{len(all_dates)} 个交易日")

        # 识别新股
        new_stocks_info = _identify_new_stocks_vectorized_main(stock_data, logger)
        if not new_stocks_info:
            logger.info("[FILTER] 未发现新股特征，直接返回原始数据")
            return stock_data

        # 获取交易日历
        trading_dates = _load_trading_calendar_main(logger)
        if not trading_dates:
            return stock_data

        # 批量过滤新股数据（优化版本）
        return _filter_new_stock_data_vectorized_main(stock_data, new_stocks_info, trading_dates, logger)

    except Exception as e:
        logger.error(f"[FILTER] 批量新股过滤失败: {e}")
        import traceback
        logger.error(f"[FILTER] 错误详情: {traceback.format_exc()}")
        return stock_data


def _validate_stock_data_main(stock_data: pd.DataFrame, logger) -> bool:
    """验证股票数据的基本结构"""
    required_columns = ['stock_code', 'date']
    missing_columns = [col for col in required_columns if col not in stock_data.columns]

    if missing_columns:
        logger.error(f"[FILTER] 缺少必需的列: {missing_columns}")
        return False

    # 检查是否有pct_chg列
    if 'pct_chg' not in stock_data.columns:
        logger.warning(f"[FILTER] 数据中没有pct_chg列，无法基于极端涨跌幅识别新股")
        return False

    logger.info(f"[FILTER] 数据验证通过，数据量: {len(stock_data):,} 条")
    return True


def _identify_new_stocks_vectorized_main(stock_data: pd.DataFrame, logger) -> dict:
    """向量化识别新股"""
    extreme_threshold = 25.0
    min_data_points = 1

    # 检查缓存
    cache_key = f"new_stocks_by_extreme_pct_{extreme_threshold}"
    cached_new_stocks = _get_new_stock_cache()

    data_hash = hash(str(stock_data['date'].min().strftime('%Y%m%d')) +
                    str(stock_data['date'].max().strftime('%Y%m%d')))

    cache_is_valid = (
        cache_key in cached_new_stocks and
        'data_hash' in cached_new_stocks[cache_key] and
        cached_new_stocks[cache_key]['data_hash'] == data_hash
    )

    if cache_is_valid:
        logger.info(f"[FILTER] 使用缓存的新股识别结果")
        return cached_new_stocks[cache_key].get('new_stocks', {})

    logger.info(f"[FILTER] 缓存失效，重新识别新股...")

    # 向量化分析极端涨跌幅
    start_time = pd.Timestamp.now()

    # 筛选有效数据
    analysis_data = stock_data.dropna(subset=['pct_chg']).copy()
    logger.info(f"[FILTER] 分析数据准备完成: {len(analysis_data):,} 条有效记录")

    # 按股票分组统计极端涨跌幅
    extreme_pct_data = analysis_data.groupby('stock_code')['pct_chg'].agg([
        ('min', 'min'),
        ('max', 'max'),
        ('abs_max', lambda x: abs(x).max()),
        ('count', 'count')
    ]).reset_index()

    extreme_pct_data.columns = ['stock_code', 'min_pct', 'max_pct', 'abs_max_pct', 'data_count']
    logger.info(f"[FILTER] 极端涨跌幅统计完成，共 {len(extreme_pct_data)} 只股票")

    # 识别新股特征
    new_stock_mask = (
        (extreme_pct_data['data_count'] >= min_data_points) &
        (extreme_pct_data['abs_max_pct'].notna()) &
        (extreme_pct_data['abs_max_pct'] > extreme_threshold)
    )

    identified_new_stocks = extreme_pct_data[new_stock_mask]
    logger.info(f"[FILTER] 发现 {len(identified_new_stocks)} 只新股")

    # 构建新股信息字典
    new_stocks_info = {}
    for _, row in identified_new_stocks.iterrows():
        stock_code = row['stock_code']
        stock_data_subset = analysis_data[analysis_data['stock_code'] == stock_code]

        if len(stock_data_subset) > 0:
            extreme_date = stock_data_subset.loc[abs(stock_data_subset['pct_chg']).idxmax(), 'date']
            new_stocks_info[stock_code] = extreme_date

            # 记录关键新股信息
            abs_max_pct = row['abs_max_pct']
            if stock_code == '001279.SZ' or abs_max_pct > 100:
                logger.info(f"[FILTER] 检测到新股: {stock_code}")
                logger.info(f"[FILTER]   上市日期: {extreme_date.strftime('%Y-%m-%d')}")
                logger.info(f"[FILTER]   极端涨跌幅: {abs_max_pct:.2f}%")

    # 保存到缓存
    cached_new_stocks[cache_key] = {
        'new_stocks': new_stocks_info,
        'data_hash': data_hash,
        'last_updated': pd.Timestamp.now()
    }
    _save_new_stock_cache(cached_new_stocks)

    logger.info(f"[FILTER] 向量化新股识别完成，耗时: {(pd.Timestamp.now() - start_time).total_seconds():.2f}s")
    return new_stocks_info


def _load_trading_calendar_main(logger) -> list:
    """加载交易日历"""
    trading_dates_cache = _get_trading_dates_cache()

    if 'all_trading_dates' in trading_dates_cache:
        return trading_dates_cache['all_trading_dates']

    # 加载完整交易日历
    calendar_path = Path(__file__).parent / "local_data" / "calendar" / "trading_calendar.parquet"
    if not calendar_path.exists():
        logger.warning(f"[FILTER] 无法获取交易日历，跳过新股过滤")
        return None

    trading_calendar = pd.read_parquet(calendar_path)
    all_trading_dates = trading_calendar['date_str'].tolist()

    # 缓存交易日历
    trading_dates_cache['all_trading_dates'] = all_trading_dates
    _save_trading_dates_cache(trading_dates_cache)

    return all_trading_dates


def _filter_new_stock_data_vectorized_main(stock_data: pd.DataFrame, new_stocks_info: dict,
                                         trading_dates: list, logger) -> pd.DataFrame:
    """
    向量化过滤新股数据 - 优化版本，完全消除嵌套循环

    关键优化：
    1. 预计算所有需要过滤的日期范围
    2. 使用向量化操作而不是逐行循环
    3. 构建高效的过滤掩码
    """
    new_stock_period = 25  # 新股过滤期间：25个交易日

    # 构建所有需要过滤的日期集合（预计算优化）
    filter_date_set = set()
    for stock_code, listing_date in new_stocks_info.items():
        listing_date_str = listing_date.strftime('%Y%m%d')
        if listing_date_str not in trading_dates:
            continue

        listing_idx = trading_dates.index(listing_date_str)

        # 批量计算需要过滤的日期索引
        filter_indices = range(listing_idx,
                             min(listing_idx + new_stock_period, len(trading_dates)))

        for idx in filter_indices:
            filter_date_set.add(trading_dates[idx])

    # 转换为datetime格式
    filter_dates = [pd.to_datetime(date_str, format='%Y%m%d') for date_str in filter_date_set]

    logger.info(f"[FILTER] 预计算过滤日期完成，需要过滤 {len(filter_dates)} 个交易日")

    # 构建需要过滤的股票集合
    filter_stocks = set(new_stocks_info.keys())

    # 向量化过滤 - 一次性应用所有过滤条件
    filter_mask = (
        (stock_data['stock_code'].isin(filter_stocks)) &
        (stock_data['date'].isin(filter_dates))
    )

    # 应用过滤
    filtered_data = stock_data[~filter_mask].copy()

    # 统计结果
    original_count = len(stock_data)
    filtered_count = len(filtered_data)
    removed_count = original_count - filtered_count

    logger.info(f"[FILTER] 优化版新股过滤完成:")
    logger.info(f"  原始数据: {original_count:,} 条")
    logger.info(f"  过滤后数据: {filtered_count:,} 条")
    logger.info(f"  过滤掉新股数据: {removed_count:,} 条")
    logger.info(f"  过滤比例: {removed_count/original_count*100:.2f}%")

    # 详细统计各新股过滤情况
    for stock_code in new_stocks_info:
        original_stock_count = len(stock_data[stock_data['stock_code'] == stock_code])
        filtered_stock_count = len(filtered_data[filtered_data['stock_code'] == stock_code])
        stock_removed = original_stock_count - filtered_stock_count

        if stock_removed > 0:
            listing_date = new_stocks_info[stock_code].strftime('%Y-%m-%d')
            logger.info(f"  {stock_code}(上市:{listing_date}): 过滤 {stock_removed} 条记录")

    # 特别检查001279.SZ
    if '001279.SZ' in new_stocks_info:
        original_001279 = len(stock_data[stock_data['stock_code'] == '001279.SZ'])
        filtered_001279 = len(filtered_data[filtered_data['stock_code'] == '001279.SZ'])
        removed_001279 = original_001279 - filtered_001279

        logger.info(f"[FILTER] *** 001279.SZ处理结果: 过滤 {removed_001279} 条记录 ***")
        if removed_001279 > 0:
            logger.info(f"[FILTER] *** 成功：001279.SZ的极端值数据已被过滤，不会污染pct_chg_1d因子 ***")
        else:
            logger.warning(f"[FILTER] *** 警告：001279.SZ未被有效过滤 ***")

    return filtered_data