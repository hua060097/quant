#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块特征生成模块 - 高性能版

使用内存映射、数据压缩和增量更新技术，高效生成板块特征数据。
模块集成了三个优化模块：
1. 内存映射管理 - 高效读取大文件
2. 特征数据压缩 - 优化存储空间和内存占用
3. 特征增量更新 - 避免重复计算，提高效率

支持断点续传、多进程并行处理和自动资源优化。
"""

import os
import gc
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
import traceback

# 设置导入路径，确保能够找到模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入logger配置
from to.logger_config import configure_logger

# 将原来的logger配置替换为新的配置
logger = configure_logger(__name__, 'generate_sector_features.log')

# 导入自定义优化模块
try:
    # 尝试直接导入
    from path_utils import get_data_path, ensure_dir
    from memory_map_manager import get_memory_map_manager
    from feature_compress import optimize_dtypes, save_features, load_features
    from feature_update import (
        get_feature_update_manager,
        # update_features_incrementally,  # 已删除
        merge_feature_files,
        get_feature_info
    )
except ImportError:
    # 尝试相对导入
    try:
        from .path_utils import get_data_path, ensure_dir
        from .memory_map_manager import get_memory_map_manager
        from .feature_compress import optimize_dtypes, save_features, load_features
        from .feature_update import (
            get_feature_update_manager,
            # update_features_incrementally,  # 已删除
            merge_feature_files,
            get_feature_info
        )
    except ImportError:
        logger.error("无法导入自定义模块，请检查路径设置")
        raise


# 导入业务逻辑相关模块
try:
    # 尝试直接导入
    from data_loaders import get_daily_data, get_trading_days
    from stock_sectors_mapping import get_sector_stocks, get_tng_sectors
except ImportError:
    # 尝试相对导入
    try:
        from .data_loaders import get_daily_data, get_trading_days
        from .stock_sectors_mapping import get_sector_stocks, get_tng_sectors
    except ImportError:
        logger.error("无法导入业务逻辑模块，请检查路径设置")
        raise

# 特征生成参数
DEFAULT_START_DATE = "20230101"
DEFAULT_END_DATE = datetime.now().strftime("%Y%m%d")
DEFAULT_MAX_WORKERS = 12
DEFAULT_BATCH_SIZE = 150
DEFAULT_MIN_RECORDS_PER_FILE = 250  # 单个特征文件的最小记录数

# 数据加载器包装类，提供统一接口
class DataLoaderWrapper:
    """数据加载器包装类，为特征生成器提供统一的数据访问接口"""
    
    def get_daily_data(self, stocks, start_date, end_date):
        """获取股票日线数据"""
        return get_daily_data(stocks, start_date, end_date)
    
    def get_trading_days(self, start_date, end_date):
        """获取交易日列表"""
        return get_trading_days(start_date, end_date)

class SectorFeatureManager:
    """板块特征管理类 - 优化版

    优化重点：
    1. 简化初始化，移除冗余的内存管理器
    2. 增加增量更新功能
    3. 优化数据缓存策略
    4. 减少不必要的复杂性
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        output_format: str = 'parquet'
    ):
        """初始化板块特征管理器 - 简化版

        Args:
            output_dir: 特征输出目录
            max_workers: 最大工作进程数
            batch_size: 每批处理的日期数量
            output_format: 输出文件格式
        """
        self.output_dir = Path(output_dir) if output_dir else get_data_path('sector_features')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._output_format = output_format
        self.max_workers = max_workers
        self.batch_size = batch_size

        # 数据提供者
        self.data_provider = DataLoaderWrapper()

        # 创建板块映射
        self.sector_mapping = {}
        for sector in get_tng_sectors():
            self.sector_mapping[sector] = get_sector_stocks(sector_name=sector)

        # 全局数据缓存 - 跨板块共享
        self._global_features_cache = {}

        logger.info(f"板块特征管理器初始化完成，输出目录: {self.output_dir}")

    def _incremental_update_features(self, new_df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        """增量更新特征数据

        Args:
            new_df: 新计算的特征数据
            file_path: 目标文件路径

        Returns:
            合并后的完整数据
        """
        try:
            if not file_path.exists():
                logger.info(f"文件不存在，创建新文件: {file_path}")
                return new_df

            # 读取现有数据
            if file_path.suffix == '.parquet':
                existing_df = pd.read_parquet(file_path)
            else:
                existing_df = pd.read_csv(file_path)

            logger.info(f"读取现有数据: {len(existing_df)} 条记录")

            # 确保日期格式一致
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            new_df['date'] = pd.to_datetime(new_df['date'])

            # 获取新数据的日期范围
            new_dates = set(new_df['date'].dt.date)
            logger.info(f"新数据日期范围: {min(new_dates)} 到 {max(new_dates)}, 共{len(new_dates)}个交易日")

            # 删除现有数据中与新数据日期重复的记录
            existing_df_filtered = existing_df[~existing_df['date'].dt.date.isin(new_dates)].copy()
            removed_count = len(existing_df) - len(existing_df_filtered)

            if removed_count > 0:
                logger.info(f"覆盖现有数据: {removed_count}条（日期重复）")

            # 合并数据
            merged_df = pd.concat([existing_df_filtered, new_df], ignore_index=True)

            # 按日期排序
            merged_df = merged_df.sort_values('date').reset_index(drop=True)

            added_count = len(new_df) - removed_count
            logger.info(f"增量添加新数据: {added_count}条")
            logger.info(f"合并后总数据量: {len(merged_df)}条")

            return merged_df

        except Exception as e:
            logger.error(f"增量更新失败: {e}")
            return new_df

    def _calculate_sector_features_vectorized(self, sector: str, features_cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """向量化计算板块特征 - 高性能版本

        Args:
            sector: 板块名称
            features_cache: 日期到股票特征数据的映射

        Returns:
            板块特征DataFrame
        """
        try:
            if sector not in self.sector_mapping:
                logger.warning(f"未找到板块 {sector} 的股票映射")
                return pd.DataFrame()

            sector_stocks = self.sector_mapping[sector]
            if not sector_stocks:
                logger.warning(f"板块 {sector} 没有股票")
                return pd.DataFrame()

            all_sector_features = []

            for date_str, stock_df in features_cache.items():
                if stock_df.empty:
                    continue

                # 过滤板块内股票
                sector_df = stock_df[stock_df['stock_code'].isin(sector_stocks)].copy()
                if sector_df.empty:
                    continue

                # 向量化计算板块特征
                sector_features = self._compute_sector_metrics(sector_df, date_str, sector)
                if sector_features:
                    all_sector_features.append(sector_features)

            if not all_sector_features:
                return pd.DataFrame()

            # 合并所有日期的特征
            result_df = pd.DataFrame(all_sector_features)
            result_df['date'] = pd.to_datetime(result_df['date'])
            result_df = result_df.sort_values('date').reset_index(drop=True)

            logger.info(f"板块 {sector} 向量化计算完成，生成 {len(result_df)} 条特征记录")
            return result_df

        except Exception as e:
            logger.error(f"向量化计算板块 {sector} 特征失败: {e}")
            return pd.DataFrame()

    def _compute_sector_metrics(self, sector_df: pd.DataFrame, date_str: str, sector: str) -> Optional[Dict]:
        """计算单日板块指标 - 优化版（只计算实际保存的14个字段）

        实际保存的字段：date, sector, sector_name, open, high, low, close,
                      volume, amount, pct_chg, stock_count, up_count, down_count, flat_count

        Args:
            sector_df: 板块内股票的特征数据
            date_str: 日期字符串
            sector: 板块名称

        Returns:
            板块特征字典
        """
        try:
            if sector_df.empty:
                return None

            # 基础价格特征聚合（使用均值）
            price_features = {}
            for col in ['open', 'high', 'low', 'close']:
                if col in sector_df.columns:
                    price_features[col] = float(sector_df[col].mean())

            # 成交量和成交额聚合（使用总和）
            if 'volume' in sector_df.columns:
                price_features['volume'] = float(sector_df['volume'].sum())
            if 'amount' in sector_df.columns:
                price_features['amount'] = float(sector_df['amount'].sum())

            # 涨跌幅特征聚合（使用均值）
            if 'pct_chg' in sector_df.columns:
                price_features['pct_chg'] = float(sector_df['pct_chg'].mean())

            # 统计特征 - 优化：直接使用向量化操作
            if 'pct_chg' in sector_df.columns:
                pct_chg = sector_df['pct_chg']
                stats_features = {
                    'stock_count': len(sector_df),
                    'up_count': int((pct_chg > 0).sum()),
                    'down_count': int((pct_chg < 0).sum()),
                    'flat_count': int((pct_chg == 0).sum())
                }
            else:
                stats_features = {
                    'stock_count': len(sector_df),
                    'up_count': 0,
                    'down_count': 0,
                    'flat_count': 0
                }

            # 合并所有特征
            features = {
                'date': date_str,
                'sector': sector,
                'sector_name': sector,
                **price_features,
                **stats_features
            }

            return features

        except Exception as e:
            logger.error(f"计算板块 {sector} 在日期 {date_str} 的指标失败: {e}")
            return None

    # process_sector_incremental方法已删除 - 增量计算功能已移除

    def _load_daily_stock_features(self, date_str: str) -> Optional[pd.DataFrame]:
        """从标准路径加载每日股票特征数据 - 优化版

        Args:
            date_str: 日期字符串，格式为YYYYMMDD

        Returns:
            股票特征DataFrame，如果加载失败返回None
        """
        # 检查全局缓存
        if date_str in self._global_features_cache:
            return self._global_features_cache[date_str]

        try:
            base_dir = Path(__file__).parent
            feature_file = base_dir / "local_data/features/daily" / f"{date_str}.parquet"

            if not feature_file.exists():
                feature_file = base_dir / "local_data/features/daily" / f"{date_str}.csv"
                if not feature_file.exists():
                    return None

            # 加载数据
            if feature_file.suffix == '.parquet':
                df = pd.read_parquet(feature_file)
            else:
                df = pd.read_csv(feature_file)

            # 缓存数据
            self._global_features_cache[date_str] = df
            return df

        except Exception as e:
            logger.error(f"加载日期 {date_str} 的特征数据失败: {e}")
            return None

    def _preload_daily_features_batch(self, date_list: List[str]) -> Dict[str, pd.DataFrame]:
        """批量预加载多天的股票特征数据 - 优化版

        优化点：
        1. 使用全局缓存避免重复加载
        2. 简化逻辑，移除复杂的内存检查
        3. 提高I/O效率

        Args:
            date_list: 日期字符串列表

        Returns:
            日期到DataFrame的映射字典
        """
        features_cache = {}

        # 检查哪些日期需要加载（未在全局缓存中）
        dates_to_load = [date for date in date_list if date not in self._global_features_cache]

        if not dates_to_load:
            logger.info("所有日期数据已在缓存中")
            return {date: self._global_features_cache[date] for date in date_list}

        logger.info(f"需要加载 {len(dates_to_load)} 天的新数据")

        # 并行加载新数据
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            base_dir = Path(__file__).parent

            for date_str in dates_to_load:
                feature_file = base_dir / "local_data/features/daily" / f"{date_str}.parquet"
                if not feature_file.exists():
                    feature_file = base_dir / "local_data/features/daily" / f"{date_str}.csv"

                if feature_file.exists():
                    if feature_file.suffix == '.parquet':
                        future = executor.submit(pd.read_parquet, feature_file)
                    else:
                        future = executor.submit(pd.read_csv, feature_file)
                    futures[date_str] = future

            # 收集结果并更新全局缓存
            for date_str, future in futures.items():
                try:
                    df = future.result()
                    self._global_features_cache[date_str] = df
                    features_cache[date_str] = df
                except Exception as e:
                    logger.warning(f"加载日期 {date_str} 失败: {e}")

        # 返回所有请求日期的数据
        result = {}
        for date in date_list:
            if date in self._global_features_cache:
                result[date] = self._global_features_cache[date]

        logger.info(f"批量预加载完成，返回 {len(result)} 天的数据")
        return result

    def _preload_features_in_chunks(self, date_list: List[str], max_io_workers: int, chunk_size: int) -> Dict[str, pd.DataFrame]:
        """分块预加载特征数据，避免内存溢出

        🛡️ 内存安全策略：将大量日期分成小块处理，每块处理完后立即释放内存

        Args:
            date_list: 日期字符串列表
            max_io_workers: I/O线程池大小
            chunk_size: 每块的大小

        Returns:
            所有日期的特征数据字典
        """
        logger.info(f"🛡️ 启用分块处理模式：{len(date_list)} 天数据分成 {chunk_size} 天/块")

        all_features = {}
        total_chunks = (len(date_list) + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(date_list))
            chunk_dates = date_list[start_idx:end_idx]

            logger.info(f"处理第 {chunk_idx + 1}/{total_chunks} 块：{len(chunk_dates)} 天 ({chunk_dates[0]} ~ {chunk_dates[-1]})")

            # 记录处理前内存
            self.log_memory_usage()

            # 处理当前块
            chunk_features = self._preload_daily_features_batch(chunk_dates, max_io_workers, max_batch_size=chunk_size)
            all_features.update(chunk_features)

            # 强制垃圾回收，释放内存
            del chunk_features
            gc.collect()

            # 记录处理后内存
            self.log_memory_usage()

            logger.info(f"第 {chunk_idx + 1} 块处理完成，累计加载 {len(all_features)} 天数据")

        logger.info(f"🎉 分块处理完成，总共加载 {len(all_features)} 天的特征数据")
        return all_features

    def _batch_check_file_existence(self, date_list: List[str]) -> List[str]:
        """批量检查文件存在性，避免逐个检查

        Args:
            date_list: 日期字符串列表，格式为YYYYMMDD

        Returns:
            存在文件的日期列表
        """
        logger.debug(f"批量检查 {len(date_list)} 个日期的文件存在性")

        base_dir = Path(__file__).parent
        daily_dir = base_dir / "local_data/features/daily"

        valid_dates = []
        # 批量检查，减少系统调用
        for date_str in date_list:
            parquet_file = daily_dir / f"{date_str}.parquet"
            csv_file = daily_dir / f"{date_str}.csv"
            if parquet_file.exists() or csv_file.exists():
                valid_dates.append(date_str)

        logger.debug(f"批量检查完成，{len(valid_dates)} 个日期有对应的特征文件")
        return valid_dates


    def _calculate_sector_features_from_daily_data(self, sector: str, date_str: str, stock_features_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """基于每日股票特征数据计算板块特征

        Args:
            sector: 板块名称
            date_str: 日期字符串
            stock_features_df: 股票特征DataFrame

        Returns:
            板块特征DataFrame，如果计算失败返回None
        """
        try:
            # 获取板块股票列表（直接使用sector_mapping替代feature_generator._get_sector_stocks）
            stocks = self.sector_mapping.get(sector, [])
            if not stocks:
                logger.warning(f"板块 {sector} 没有成分股")
                return None

            # 筛选板块内的股票数据
            # 确保股票代码格式一致
            stock_codes_in_df = set(stock_features_df['stock_code'].unique()) if 'stock_code' in stock_features_df.columns else set()

            # 尝试不同的股票代码格式匹配
            matched_stocks = []
            for stock in stocks:
                if stock in stock_codes_in_df:
                    matched_stocks.append(stock)
                else:
                    # 尝试添加或移除市场前缀
                    if '.' in stock:
                        stock_without_market = stock.split('.')[0]
                        if stock_without_market in stock_codes_in_df:
                            matched_stocks.append(stock_without_market)
                    else:
                        # 尝试添加常见市场后缀
                        for suffix in ['.SZ', '.SH']:
                            stock_with_suffix = f"{stock}{suffix}"
                            if stock_with_suffix in stock_codes_in_df:
                                matched_stocks.append(stock_with_suffix)
                                break

            if not matched_stocks:
                logger.warning(f"板块 {sector} 在日期 {date_str} 没有匹配的股票数据")
                return None

            # 筛选板块内股票的数据
            sector_stock_data = stock_features_df[stock_features_df['stock_code'].isin(matched_stocks)].copy()

            if sector_stock_data.empty:
                logger.warning(f"板块 {sector} 在日期 {date_str} 没有有效的股票数据")
                return None

            # 计算板块聚合特征
            sector_features = self._aggregate_sector_features(sector_stock_data, sector, date_str)

            return sector_features

        except Exception as e:
            logger.error(f"计算板块 {sector} 在日期 {date_str} 的特征时出错: {e}")
            return None

    def _aggregate_sector_features(self, sector_stock_data: pd.DataFrame, sector: str, date_str: str) -> pd.DataFrame:
        """聚合股票数据为板块特征

        Args:
            sector_stock_data: 板块内股票的特征数据
            sector: 板块名称
            date_str: 日期字符串

        Returns:
            板块特征DataFrame
        """
        try:
            # 基础价格特征聚合（使用均值）
            price_features = {}
            price_cols = ['open', 'high', 'low', 'close', 'pre_close']
            for col in price_cols:
                if col in sector_stock_data.columns:
                    price_features[col] = sector_stock_data[col].mean()

            # 成交量和成交额聚合（使用总和）
            volume_features = {}
            volume_cols = ['volume', 'amount']
            for col in volume_cols:
                if col in sector_stock_data.columns:
                    volume_features[col] = sector_stock_data[col].sum()

            # 涨跌幅特征聚合（使用均值）
            change_features = {}
            change_cols = ['pct_chg', 'change']
            for col in change_cols:
                if col in sector_stock_data.columns:
                    change_features[col] = sector_stock_data[col].mean()

            # 统计特征
            stats_features = {
                'stock_count': len(sector_stock_data),  # 板块内股票数量
                'up_count': len(sector_stock_data[sector_stock_data.get('pct_chg', 0) > 0]),  # 上涨股票数量
                'down_count': len(sector_stock_data[sector_stock_data.get('pct_chg', 0) < 0]),  # 下跌股票数量
                'flat_count': len(sector_stock_data[sector_stock_data.get('pct_chg', 0) == 0]),  # 平盘股票数量
            }

            # 合并所有特征
            all_features = {
                'date': date_str,
                'sector': sector,
                'sector_name': sector,
                **price_features,
                **volume_features,
                **change_features,
                **stats_features
            }

            # 创建DataFrame
            features_df = pd.DataFrame([all_features])

            # 确保日期列是正确的格式
            features_df['date'] = pd.to_datetime(date_str, format='%Y%m%d')

            # 使用数据类型优化（从calculate_sector_features_for_date方法迁移）
            features_df = optimize_dtypes(features_df)

            logger.debug(f"成功聚合板块 {sector} 在日期 {date_str} 的特征，包含 {len(all_features)} 个特征")

            return features_df

        except Exception as e:
            logger.error(f"聚合板块 {sector} 在日期 {date_str} 的特征时出错: {e}")
            return pd.DataFrame()


    # calculate_sector_features_for_date方法已删除 - 功能已被_calculate_sector_features_from_daily_data替代

    def process_date(self, date: Union[str, pd.Timestamp], sector: str, preloaded_data: Optional[pd.DataFrame] = None) -> Tuple[bool, Optional[pd.DataFrame]]:
        """处理单个日期的板块特征

        Args:
            date: 日期
            sector: 板块名称
            preloaded_data: 预加载的历史数据（已不使用，保留兼容性）

        Returns:
            (成功标识, 特征DataFrame)
        """
        try:
            # 日期标准化
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y%m%d')
            else:
                date_str = str(date).replace('-', '').replace('/', '')
                
            logger.debug(f"开始处理板块 {sector} 在日期 {date_str} 的特征")
            
            # 计算特征（使用新的简化方法替代calculate_sector_features_for_date）
            # 从标准路径加载每日股票特征数据
            stock_features_df = self._load_daily_stock_features(date_str)

            if stock_features_df is None or stock_features_df.empty:
                logger.warning(f"日期 {date_str} 没有股票特征数据")
                return False, None

            # 基于股票特征数据计算板块特征
            features_df = self._calculate_sector_features_from_daily_data(sector, date_str, stock_features_df)

            if features_df is None or features_df.empty:
                logger.warning(f"计算板块 {sector} 在日期 {date_str} 的特征失败")
                return False, None
                
            # 保存特征文件
            save_success, _ = self.save_features(features_df, sector, date_str)
            
            # 更新特征元数据
            try:
                from feature_update import get_feature_update_manager
                manager = get_feature_update_manager(self.output_dir)
                manager.add_processed_date(sector, date_str)
            except (ImportError, AttributeError) as e:
                logger.debug(f"更新特征元数据失败: {e}")
                
            logger.debug(f"完成处理板块 {sector} 在日期 {date_str} 的特征")
            return save_success, features_df
            
        except Exception as e:
            logger.error(f"处理板块 {sector} 在日期 {date} 时出错: {e}")
            logger.error(traceback.format_exc())
            return False, None

    def process_sector(
        self,
        sector: str,
        start_date: str,
        end_date: str,
        batch_size: Optional[int] = None,
        output_format: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """处理单个板块特征生成 - 优化版

        Args:
            sector: 板块名称
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            batch_size: 每批处理的日期数量
            output_format: 输出格式

        Returns:
            (板块名称, 处理成功的日期列表)
        """
        if output_format is not None:
            self._output_format = output_format

        processed_dates = []

        try:
            # 获取交易日列表
            date_range = get_trading_days(start_date, end_date)

            # 简化日期处理
            if isinstance(date_range, pd.DataFrame):
                if 'date' in date_range.columns:
                    date_range = date_range['date'].tolist()
                else:
                    date_range = list(date_range.index)

            if not date_range:
                logger.warning(f"板块 {sector} 在日期范围 {start_date} - {end_date} 内没有交易日")
                return sector, []

            # 标准化日期格式
            normalized_dates = []
            for date in date_range:
                if isinstance(date, pd.Timestamp):
                    date_str = date.strftime('%Y%m%d')
                else:
                    date_str = str(date).replace('-', '')
                normalized_dates.append(date_str)

            # 检查有效日期
            valid_dates = self._batch_check_file_existence(normalized_dates)
            if not valid_dates:
                logger.warning(f"板块 {sector} 没有可用的特征数据文件")
                return sector, []

            logger.info(f"板块 {sector} 开始处理 {len(valid_dates)} 个日期")

            # 批量预加载数据
            features_cache = self._preload_daily_features_batch(valid_dates)
            if not features_cache:
                logger.warning(f"板块 {sector} 没有成功加载任何特征数据")
                return sector, []

            # 向量化计算所有日期的板块特征
            all_features = self._calculate_sector_features_vectorized(sector, features_cache)

            if all_features.empty:
                logger.warning(f"板块 {sector} 没有生成任何特征")
                return sector, []

            # 保存结果（增量更新）
            sector_dir = self.output_dir / sector
            sector_dir.mkdir(parents=True, exist_ok=True)

            # 统一使用 {sector}_features.parquet 作为主文件名
            features_parquet_path = sector_dir / f"{sector}_features.parquet"
            final_df = self._incremental_update_features(all_features, features_parquet_path)

            # 保存主特征文件
            final_df.to_parquet(features_parquet_path, index=False)
            logger.info(f"板块 {sector} 特征已保存到 {features_parquet_path}，共 {len(final_df)} 条记录")

            # 生成最新日期的CSV文件
            if not final_df.empty and 'date' in final_df.columns:
                try:
                    # 获取最新日期的数据
                    latest_date = final_df['date'].max()
                    latest_df = final_df[final_df['date'] == latest_date].copy()

                    # 保存最新日期的CSV文件
                    latest_csv_path = sector_dir / f"{sector}_latest.csv"
                    latest_df.to_csv(latest_csv_path, index=False)
                    logger.info(f"板块 {sector} 最新特征已保存到 {latest_csv_path}，日期: {latest_date}")

                except Exception as e:
                    logger.warning(f"生成板块 {sector} 最新CSV文件失败: {e}")

            processed_dates = all_features['date'].dt.strftime('%Y%m%d').tolist()
            return sector, processed_dates

        except Exception as e:
            logger.error(f"处理板块 {sector} 特征时出错: {e}")

    
    def split_sectors_by_worker(self, sectors: List[str], max_workers: Optional[int] = None) -> List[List[str]]:
        """将板块按照工作进程数量分割
        
        按照均衡负载的原则分配板块到不同工作进程。
        
        Args:
            sectors: 板块列表
            max_workers: 最大工作进程数，None则使用默认值
            
        Returns:
            分组后的板块列表
        """
        if not sectors:
            return []
        
        # 确保max_workers有效
        if max_workers is None or max_workers <= 0:
            max_workers = self.max_workers
            # 如果self.max_workers也是None，设置一个合理的默认值
            if max_workers is None:
                max_workers = 4  # 默认使用4个工作进程
        
        # 确保至少有一个工作进程
        if max_workers <= 0:
            max_workers = 1
        
        # 调整工作进程数，不超过板块数
        actual_workers = min(max_workers, len(sectors))
        
        # 初始化每个工作进程的负载
        worker_loads = [0] * actual_workers
        sector_groups = [[] for _ in range(actual_workers)]
        
        # 按照字母表顺序排序板块，使分配更确定
        sorted_sectors = sorted(sectors)
        
        # 使用贪心算法分配板块
        for sector in sorted_sectors:
            # 找到负载最小的工作进程
            min_load_worker = worker_loads.index(min(worker_loads))
            # 分配板块
            sector_groups[min_load_worker].append(sector)
            # 更新负载（简单地使用板块数量作为负载）
            worker_loads[min_load_worker] += 1
        
        # 过滤掉空组
        return [group for group in sector_groups if group]
    
    def process_date_range(
        self,
        start_date: str,
        end_date: str,
        sectors: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        output_format: str = 'parquet'  # 添加输出格式参数
    ) -> Dict[str, List[str]]:
        """处理日期范围内的所有板块特征

        新的数据处理策略：
        - 对指定日期范围进行全量计算
        - 智能合并：重叠日期用新数据覆盖，非重叠日期正常合并

        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            sectors: 板块列表，None则使用所有板块
            max_workers: 最大工作进程数，None则使用默认值
            batch_size: 每批处理的日期数量，None则使用默认值
            output_format: 输出格式，'parquet'或'csv'，默认为'parquet'

        Returns:
            处理结果字典，键为板块名称，值为处理的日期列表
        """
        # 设置默认值
        if max_workers is None:
            max_workers = self.max_workers
        if max_workers is None:
            max_workers = 4  # 默认使用4个工作进程
            
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size is None:
            batch_size = 30  # 默认每批处理30个日期
        
        # 设置输出格式
        self._output_format = output_format
        logger.info(f"设置输出格式为: {output_format}")
        
        # 获取要处理的板块列表
        if sectors is None:
            try:
                from stock_sectors_mapping import get_tng_sectors
                sectors = get_tng_sectors()
            except (ImportError, AttributeError):
                logger.warning("无法导入get_all_sectors函数，使用默认板块列表")
                # 使用默认板块列表
                sectors = get_tng_sectors()
        
        if not sectors:
            logger.warning("没有要处理的板块")
            return {}
        
        logger.info(f"开始处理日期范围 {start_date} - {end_date} 的板块特征")
        logger.info(f"将处理 {len(sectors)} 个板块的特征数据")
        logger.info(f"🛡️ 内存安全优化已启用：流式处理 + 分块加载 + 自动垃圾回收")

        # 🛡️ 内存安全检查：根据日期范围调整并发数
        from data_loaders import get_trading_days
        total_trading_days = len(get_trading_days(start_date, end_date))

        if total_trading_days > 100:
            # 大量日期时减少并发数，避免内存爆炸
            max_workers = min(max_workers, 2)
            logger.warning(f"🛡️ 检测到大量交易日 ({total_trading_days} 天)，调整并发数为 {max_workers} 以确保内存安全")

        # 记录当前内存使用
        self.log_memory_usage()
        
        # 根据进程数划分板块
        sector_groups = self.split_sectors_by_worker(sectors, max_workers)
        
        # 使用多进程处理
        results = {}

        print(f"max_workers: {max_workers}")
        
        # 多进程处理所有板块
        print(f"多进程处理 {len(sectors)} 个板块，使用 {max_workers} 个工作进程")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for sector_group in sector_groups:
                    for sector in sector_group:
                        # 提交处理任务
                        future = executor.submit(
                            self.process_sector,
                            sector,
                            start_date,
                            end_date,
                            batch_size,
                            self._output_format
                        )
                        futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        sector, processed_dates = future.result()
                        results[sector] = processed_dates
                    except Exception as e:
                        logger.error(f"处理板块特征时出错: {e}")
                        results[sector] = []
        
        # 记录最终内存使用
        self.log_memory_usage()
        
        # 处理完成后，对所有板块执行合并操作
        logger.info("处理完成，开始合并各板块的特征文件...")
        for sector in sectors:
            try:
                if sector in results and results[sector]:
                    logger.info(f"合并板块 {sector} 的特征文件")
                    self.merge_features(sector, output_format=self._output_format)
            except Exception as e:
                logger.error(f"合并板块 {sector} 特征文件时出错: {e}")
        
        # 返回处理结果
        return results
    
    def log_memory_usage(self):
        """记录内存使用情况"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        logger.info(f"内存使用: {memory_mb:.2f} MB")

        # 🚨 内存警告机制
        if memory_mb > 4000:  # 超过4GB发出警告
            logger.warning(f"⚠️ 内存使用过高: {memory_mb:.2f} MB，建议进行垃圾回收")
        elif memory_mb > 6000:  # 超过6GB发出严重警告
            logger.error(f"🚨 内存使用危险: {memory_mb:.2f} MB，强制进行垃圾回收")
            gc.collect()

        return memory_mb



    def log_io_performance(self, operation: str, file_count: int, duration: float):
        """记录I/O性能统计

        Args:
            operation: 操作类型（如'批量预加载'）
            file_count: 处理的文件数量
            duration: 耗时（秒）
        """
        if duration > 0:
            throughput = file_count / duration
            logger.info(f"I/O性能统计 - {operation}: {file_count}个文件, 耗时{duration:.2f}秒, 吞吐量{throughput:.1f}文件/秒")

    def save_features(self, df: pd.DataFrame, sector: str, date: str) -> Tuple[bool, Path]:
        """保存特征到文件
        
        Args:
            df: 特征DataFrame
            sector: 板块名称
            date: 日期字符串，格式为YYYYMMDD
            
        Returns:
            (是否成功, 文件路径)
        """
        if df.empty:
            logger.warning(f"特征DataFrame为空，不保存 板块: {sector} 日期: {date}")
            return False, None
        
        # 确保板块目录存在
        sector_dir = self.output_dir / sector
        sector_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定输出格式，默认为parquet
        output_format = getattr(self, '_output_format', 'parquet')
        
        # 根据输出格式设置文件路径和扩展名
        if output_format == 'csv':
            file_path = sector_dir / f"{sector}_{date}.csv"
            format_type = 'csv'
        else:
            # 默认使用parquet格式
            file_path = sector_dir / f"{sector}_{date}.parquet"
            format_type = 'parquet'
        
        try:
            # 使用feature_compress中的save_features，包含数据类型优化
            df_optimized = optimize_dtypes(df)
            success = save_features(df_optimized, file_path, format=format_type)
            
            if success:
                logger.info(f"成功保存特征到文件: {file_path}")
                
                # 使用新的方法添加已处理日期
                self.feature_update_manager.add_processed_date(sector, date)
                
                # 保存最新特征为CSV格式，方便查看
                latest_file_path = sector_dir / f"{sector}_latest.csv"
                save_features(df_optimized, latest_file_path, format='csv', compress=False)
                
                return True, file_path
            else:
                logger.error(f"保存特征文件失败: {file_path}")
                return False, file_path
            
        except Exception as e:
            logger.error(f"保存特征文件时出错: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False, file_path
        
    def merge_features(self, sector: str, output_format: str = None) -> bool:
        """合并板块特征文件
        
        将同一板块的多个特征文件合并为一个主文件，提高数据访问效率并支持增量更新。
        
        Args:
            sector: 板块名称
            output_format: 输出格式，默认使用self._output_format
            
        Returns:
            是否成功合并
        """
        try:
            # 使用指定的格式或默认格式
            if output_format is None:
                output_format = getattr(self, '_output_format', 'parquet')
            
            # 查找板块目录
            sector_dir = Path(self.output_dir) / sector
            if not sector_dir.exists():
                logger.warning(f"板块 {sector} 的目录不存在，无法合并文件")
                return False
            
            logger.info(f"尝试合并板块 {sector} 的特征文件，格式: {output_format}")
            
            # 合并Parquet文件
            if output_format == 'parquet':
                # 检查主Parquet文件是否存在
                main_parquet = sector_dir / f"{sector}_features.parquet"

                if main_parquet.exists():
                    try:
                        # 读取主Parquet文件以获取最新日期
                        main_df = pd.read_parquet(main_parquet)
                        if 'date' in main_df.columns and not main_df.empty:
                            # 获取最新日期的数据并保存为CSV
                            if not pd.api.types.is_datetime64_any_dtype(main_df['date']):
                                main_df['date'] = pd.to_datetime(main_df['date'])
                            latest_date = main_df['date'].max()
                            latest_df = main_df[main_df['date'] == latest_date]
                            latest_path = sector_dir / f"{sector}_latest.csv"
                            latest_df.to_csv(latest_path, index=False)
                            logger.info(f"已为板块 {sector} 创建最新特征CSV文件，日期: {latest_date}")
                            logger.info(f"板块 {sector} 特征文件已存在，共 {len(main_df)} 条记录")

                        return True

                    except Exception as e:
                        logger.warning(f"读取主Parquet文件失败: {e}")
                        return False
                else:
                    logger.info(f"板块 {sector} 的主特征文件不存在")
                    return False
            
            # 合并CSV文件（如需要）
            elif output_format == 'csv':
                # 查找所有CSV文件（排除latest文件）
                csv_files = list(sector_dir.glob(f"{sector}_20*.csv"))
                csv_files = [f for f in csv_files if not f.name.endswith('_latest.csv')]
                
                if not csv_files:
                    logger.info(f"板块 {sector} 没有需要合并的CSV特征文件")
                    return True
                
                # 加载所有CSV文件
                dfs = []
                for file_path in csv_files:
                    try:
                        df = pd.read_csv(file_path)
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                        dfs.append(df)
                    except Exception as e:
                        logger.warning(f"加载CSV文件 {file_path} 失败: {e}")
                
                if not dfs:
                    logger.warning(f"板块 {sector} 所有CSV文件加载失败")
                    return False
                
                # 合并数据
                merged_df = pd.concat(dfs, ignore_index=True)
                
                # 去重并排序
                if 'date' in merged_df.columns:
                    merged_df = merged_df.drop_duplicates(subset=['date']).sort_values('date')
                
                # 保存合并后的数据为单一CSV文件
                merged_csv = sector_dir / f"{sector}_features.csv"
                try:
                    merged_df.to_csv(merged_csv, index=False, encoding='utf-8')
                    logger.info(f"成功保存合并后的特征到CSV文件: {merged_csv}")
                    
                    # 保存最新日期的数据
                    if 'date' in merged_df.columns and not merged_df.empty:
                        latest_date = merged_df['date'].max()
                        latest_df = merged_df[merged_df['date'] == latest_date]
                        latest_path = sector_dir / f"{sector}_latest.csv"
                        latest_df.to_csv(latest_path, index=False, encoding='utf-8')
                    
                    # 成功保存后，删除原始日期CSV文件
                    for file_path in csv_files:
                        try:
                            file_path.unlink()
                        except Exception as e:
                            logger.warning(f"删除文件 {file_path} 失败: {e}")
                    
                    # 更新元数据 - 提取并添加已处理日期
                    if 'date' in merged_df.columns:
                        # 获取所有日期
                        dates = merged_df['date'].astype(str).tolist()
                        # 标准化日期格式
                        processed_dates = []
                        for d in dates:
                            try:
                                # 尝试转换为YYYYMMDD格式
                                if '-' in d:  # 如果是YYYY-MM-DD格式
                                    processed_dates.append(d.replace('-', ''))
                                elif len(d) == 8 and d.isdigit():  # 如果已经是YYYYMMDD格式
                                    processed_dates.append(d)
                                else:  # 尝试其他格式转换
                                    dt = pd.to_datetime(d)
                                    processed_dates.append(dt.strftime('%Y%m%d'))
                            except:
                                # 如果转换失败，跳过
                                logger.warning(f"无法转换日期格式: {d}")
                        
                        # 批量添加已处理日期
                        self.feature_update_manager.add_processed_dates(sector, processed_dates)
                        logger.info(f"已更新板块 {sector} 的元数据，处理了 {len(processed_dates)} 个日期 (CSV格式)")
                    
                    return True
                except Exception as e:
                    logger.error(f"保存合并后的CSV文件失败: {e}")
                    return False
            
            else:
                logger.warning(f"不支持的输出格式: {output_format}")
                return False
            
        except Exception as e:
            logger.error(f"合并特征文件时出错: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False

    def generate_features(
        self,
        sectors: List[str],
        start_date: str,
        end_date: str,
        batch_size: Optional[int] = None,
        output_format: Optional[str] = None
    ) -> Dict[str, int]:
        """生成板块特征 - 优化版

        Args:
            sectors: 板块名称列表
            start_date: 开始日期
            end_date: 结束日期
            batch_size: 每批处理的日期数
            output_format: 输出格式

        Returns:
            特征生成统计字典 {板块名称: 成功天数}
        """
        try:
            if output_format is not None:
                self._output_format = output_format
            elif not hasattr(self, '_output_format'):
                self._output_format = 'parquet'

            start_time = time.time()
            logger.info(f"开始为 {len(sectors)} 个板块生成特征，时间范围: {start_date} - {end_date}")

            results = {}

            # 并行处理每个板块
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for sector in sectors:
                    future = executor.submit(
                        self.process_sector,
                        sector,
                        start_date,
                        end_date,
                        batch_size,
                        self._output_format
                    )
                    futures.append((future, sector))

                # 收集结果
                for future, sector in futures:
                    try:
                        sector_name, processed_dates = future.result()
                        results[sector_name] = len(processed_dates)
                        logger.info(f"板块 {sector_name} 处理完成，成功生成 {len(processed_dates)} 天的特征")
                    except Exception as e:
                        logger.error(f"处理板块 {sector} 时出错: {e}")
                        results[sector] = 0

            # 计算总处理时间
            elapsed_time = time.time() - start_time
            logger.info(f"特征生成完成，耗时: {elapsed_time:.2f} 秒")

            # 清理缓存
            self._global_features_cache.clear()
            gc.collect()

            return results

        except Exception as e:
            logger.error(f"生成板块特征时出错: {e}")
            return {sector: 0 for sector in sectors}
    
    def load_features(self, sector: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """加载板块特征数据
        
        Args:
            sector: 板块名称
            start_date: 开始日期，可选
            end_date: 结束日期，可选
            
        Returns:
            板块特征DataFrame
        """
        try:
            from to.feature_loader import load_sector_features
            
            # 使用feature_loader加载特征
            df = load_sector_features(sector, start_date, end_date)
            
            if df.empty:
                logger.warning(f"未找到板块 {sector} 的特征数据")
            else:
                logger.info(f"成功加载板块 {sector} 的特征数据，共 {len(df)} 行")
            
            return df
        except Exception as e:
            logger.error(f"加载板块 {sector} 的特征数据失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return pd.DataFrame()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成板块特征数据')
    
    # 板块选项
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sector', type=str, help='指定处理的板块名称')
    group.add_argument('--all-sectors', '--all_sectors', action='store_true', help='处理所有板块')
    
    # 日期选项
    parser.add_argument('--start-date', '--start_date', dest='start_date', type=str, 
                      default=DEFAULT_START_DATE, help=f'开始日期，格式为YYYYMMDD，默认：{DEFAULT_START_DATE}')
    parser.add_argument('--end-date', '--end_date', dest='end_date', type=str, 
                      default=DEFAULT_END_DATE, help=f'结束日期，格式为YYYYMMDD，默认：{DEFAULT_END_DATE}')
    
    # 更新模式相关参数已删除 - 现在默认对指定日期范围进行全量计算
    
    # 优化选项
    parser.add_argument('--max-workers', '--max_workers', dest='max_workers', type=int, 
                       default=DEFAULT_MAX_WORKERS, help=f'最大工作进程数，默认：{DEFAULT_MAX_WORKERS}')
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, 
                       default=DEFAULT_BATCH_SIZE, help=f'每批处理的日期数量，默认：{DEFAULT_BATCH_SIZE}')
    parser.add_argument('--no-merge', '--no_merge', dest='no_merge', action='store_true', 
                       help='不合并特征文件，保留每日单独的特征文件（用于调试）')
    
    # 输出选项
    parser.add_argument('--output-format', '--output_format', dest='output_format', type=str, 
                       choices=['parquet', 'csv'], default='parquet',
                       help='输出文件格式，可选值：parquet或csv，默认：parquet')
    
    # 其他选项
    parser.add_argument('--output-dir', '--output_dir', dest='output_dir', type=str, 
                       help='输出目录，默认使用系统配置目录')
    parser.add_argument('--debug', action='store_true', help='启用调试日志')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 验证输入
    if args.sector is None and args.all_sectors is False:
        logger.error("必须指定至少一个板块名称或使用 --all_sectors")
        sys.exit(1)
        
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    # 更新模式相关检查已删除 - 现在默认对指定日期范围进行全量计算
        
    # 初始化内存映射管理器
    memory_map_manager = get_memory_map_manager()
    
    try:
        # 决定要处理的板块
        if args.all_sectors:
            sectors = get_tng_sectors()
            logger.info(f"处理所有板块 ({len(sectors)} 个)")
        else:
            sectors = [args.sector]
            logger.info(f"处理板块: {args.sector}")
        
        # 设置日期范围
        start_date = args.start_date
        end_date = args.end_date
        logger.info(f"日期范围: {start_date} - {end_date}")
        logger.info(f"输出格式: {args.output_format}")
        logger.info(f"是否合并特征: {not args.no_merge}")
        
        # 创建特征管理器
        manager = SectorFeatureManager(
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            output_format=args.output_format  # 传递输出格式参数
        )
        
        # 直接使用process_date_range处理
        if not args.no_merge:
            print("开始合并特征-------------------------------")
            # 使用process_date_range，它会在处理完所有板块后自动合并
            results = manager.process_date_range(
                start_date=start_date,
                end_date=end_date,
                sectors=sectors,
                max_workers=args.max_workers,
                batch_size=args.batch_size,
                output_format=args.output_format
            )
        else:
            # 使用generate_features而不合并
            results = manager.generate_features(
                sectors=sectors,
                start_date=start_date,
                end_date=end_date,
                batch_size=args.batch_size,
                merge_after_generation=False,  # 不合并特征文件
                output_format=args.output_format
            )
        
        # 打印结果
        total_features = sum(len(dates) for dates in results.values())
        logger.info(f"特征生成完成，总共生成 {total_features} 天的特征")
        for sector, dates in results.items():
            logger.info(f"板块 {sector}: {len(dates)} 天")
            
    except Exception as e:
        logger.error(f"生成特征时出错: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # 清理资源
        memory_map_manager.close_all()
        
if __name__ == "__main__":
    main()

"""
使用示例:

1. 从项目根目录执行（推荐方式）:
   python -m to.generate_sector_features --start-date 20230101 --end-date 20230331

2. 使用提供的启动脚本:
   python run_sector_features.py --start-date 20230101 --end-date 20230331 --sectors "TGN人工智能,TGN新能源"

3. 或者在to目录内使用相对路径执行:
   cd to
   python generate_sector_features.py --start-date 20230101 --end-date 20230331

4. 使用fiximport解决导入问题:
   ```python
   import fiximport
   from to.generate_sector_features import generate_features
   
   # 然后使用函数
   result = generate_features(start_date="20230101", end_date="20230331")
   ```

5. 在代码中导入时请记住处理路径问题:
   ```python
   import sys
   from pathlib import Path
   
   # 添加项目根目录到sys.path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   
   # 然后导入模块
   from to.generate_sector_features import generate_features
   ```

注意:
- 此模块依赖于to/目录下的其他模块
- 请确保您正确地处理了导入路径问题
- 参考README_PATH.md了解更多关于项目路径使用规范的信息
"""