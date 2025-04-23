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
logger = configure_logger(__name__)

# 导入自定义优化模块
try:
    # 尝试直接导入
    from path_utils import get_data_path, ensure_dir
    from memory_map_manager import get_memory_map_manager
    from feature_compress import optimize_dtypes, save_features, load_features
    from feature_update import (
        get_feature_update_manager, 
        update_features_incrementally,
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
            update_features_incrementally,
            merge_feature_files,
            get_feature_info
        )
    except ImportError:
        logger.error("无法导入自定义模块，请检查路径设置")
        raise

# 导入板块特征生成相关模块
try:
    from to.sector_features.technical_indicators import TechnicalIndicators
    from to.sector_features.enhanced_feature_generator import EnhancedSectorFeatureGenerator
except ImportError:
    logger.error("无法导入板块特征生成模块，请检查路径设置")
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
    """板块特征管理类
    
    负责生成和管理板块特征数据，集成了内存映射、数据压缩和增量更新功能。
    
    Attributes:
        output_dir: 输出目录路径
        memory_map_manager: 内存映射管理器
        feature_update_manager: 特征更新管理器
        max_workers: 最大工作进程数
        batch_size: 每批处理的日期数量
    """
    
    def __init__(
        self, 
        output_dir: Optional[Union[str, Path]] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        min_records_per_file: int = DEFAULT_MIN_RECORDS_PER_FILE,
        output_format: str = 'parquet'  # 添加输出格式参数
    ):
        """初始化板块特征管理器
        
        Args:
            output_dir: 特征输出目录，如果为None则使用默认目录
            max_workers: 最大工作进程数
            batch_size: 每批处理的日期数量
            min_records_per_file: 特征文件的最小记录数，用于文件合并优化
            output_format: 输出文件格式，支持'parquet'或'csv'，默认为'parquet'
        """
        self.output_dir = Path(output_dir) if output_dir else get_data_path('sector_features')
        self.output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
        
        # 设置输出格式
        self._output_format = output_format
        
        # 初始化内存映射管理器
        self.memory_map_manager = get_memory_map_manager()
        
        # 初始化特征更新管理器
        self.feature_update_manager = get_feature_update_manager(self.output_dir)
        
        # 设置处理参数
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.min_records_per_file = min_records_per_file
        
        # 缓存统计信息
        self._feature_info_cache = {}
        
        # 数据提供者
        self.data_provider = DataLoaderWrapper()
        
        # 创建EnhancedSectorFeatureGenerator实例
        from to.sector_features.enhanced_feature_generator import EnhancedSectorFeatureGenerator
        from to.sector_features.technical_indicators import TechnicalIndicators
        from to.stock_sectors_mapping import get_sector_stocks
        
        # 创建板块映射
        self.sector_mapping = {}
        for sector in get_tng_sectors():
            self.sector_mapping[sector] = get_sector_stocks(sector_name=sector)
            
        # 初始化增强特征生成器
        self.feature_generator = EnhancedSectorFeatureGenerator(
            data_provider=self.data_provider,
            sector_mapping=self.sector_mapping,
            output_dir=self.output_dir
        )
    
    def calculate_sector_features_for_date(self, date: str, sector: str, preloaded_data: Optional[pd.DataFrame] = None) -> Tuple[bool, pd.DataFrame]:
        """计算特定日期的板块特征
        
        使用EnhancedSectorFeatureGenerator生成综合特征
        
        Args:
            date: 目标日期，字符串格式，如"20230101"
            sector: 板块名称
            preloaded_data: 预加载的历史数据
            
        Returns:
            板块特征DataFrame或空DataFrame
        """
        try:
            logger.info(f"使用增强特征生成器计算板块 {sector} 在日期 {date} 的特征")
            
            # 使用EnhancedSectorFeatureGenerator生成特征
            success, features_df = self.feature_generator.generate_sector_features_for_date(sector, date, preloaded_data)
            
            if not success or features_df is None or features_df.empty:
                logger.warning(f"日期 {date} 板块 {sector} 没有特征数据")
                return False, pd.DataFrame()
            
            # 确保有date列
            if 'date' not in features_df.columns:
                features_df['date'] = date
                
            # 确保有sector_name列
            if 'sector_name' not in features_df.columns and 'sector' in features_df.columns:
                features_df['sector_name'] = features_df['sector']
            elif 'sector_name' not in features_df.columns and 'sector' not in features_df.columns:
                features_df['sector_name'] = sector
            
            # 使用数据类型优化
            features_df = optimize_dtypes(features_df)
            
            return True, features_df
        except Exception as e:
            logger.error(f"处理日期 {date} 板块 {sector} 时出错: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False, pd.DataFrame()

    def process_date(self, date: Union[str, pd.Timestamp], sector: str, preloaded_data: Optional[pd.DataFrame] = None) -> Tuple[bool, Optional[pd.DataFrame]]:
        """处理单个日期的板块特征
        
        Args:
            date: 日期
            sector: 板块名称
            preloaded_data: 预加载的历史数据
            
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
            
            # 计算特征
            success, features_df = self.calculate_sector_features_for_date(date_str, sector, preloaded_data)
            
            if not success or features_df is None or features_df.empty:
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
        force_update: bool = False,
        batch_size: Optional[int] = None,
        output_format: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """处理单个板块特征生成
        
        Args:
            sector: 板块名称
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            force_update: 是否强制重新生成
            batch_size: 每批处理的日期数量
            output_format: 输出格式，默认使用self._output_format
            
        Returns:
            (板块名称, 处理成功的日期列表)
        """
        # 设置输出格式
        if output_format is not None:
            self._output_format = output_format
            
        batch_size = batch_size or self.batch_size
        processed_dates = []
        
        try:
            # 获取日期列表
            logger.debug(f"请求交易日历: {start_date} - {end_date}")
            date_range = get_trading_days(start_date, end_date)
            logger.warning(f"获取到交易日历，类型: {type(date_range)}, 长度: {len(date_range) if hasattr(date_range, '__len__') else '未知'}")
            
            if isinstance(date_range, pd.DataFrame):
                logger.warning(f"date_range是DataFrame，尝试转换为列表")
                if 'date' in date_range.columns:
                    date_range = date_range['date'].tolist()
                else:
                    date_range = list(date_range.index) if hasattr(date_range, 'index') else []
            
            # 安全检查空列表 - 使用len()而不是直接的布尔值检查
            if not hasattr(date_range, '__len__') or len(date_range) == 0:
                logger.warning(f"板块 {sector} 在日期范围 {start_date} - {end_date} 内没有交易日")
                return sector, []
            
            # 获取最后一个交易日，用于生成CSV
            try:
                last_trading_day = date_range[-1]
                logger.debug(f"最后一个交易日: {last_trading_day}, 类型: {type(last_trading_day)}")
                
                # 确保last_trading_day是字符串类型，便于后续比较
                if isinstance(last_trading_day, pd.Timestamp):
                    last_trading_day = last_trading_day.strftime('%Y%m%d')
                elif not isinstance(last_trading_day, str):
                    last_trading_day = str(last_trading_day)
                    
            except (IndexError, TypeError) as e:
                logger.error(f"获取最后一个交易日失败: {e}, date_range类型: {type(date_range)}")
                last_trading_day = end_date
            
            # 确定需要处理的日期
            if not force_update:
                # 增量更新，只处理未生成特征的日期
                try:
                    logger.info(f"请求增量更新日期: {sector}, {start_date} - {end_date}")
                    dates_to_process = update_features_incrementally(sector, start_date, end_date, self.output_dir)
                    
                    # 输出详细日志
                    if dates_to_process:
                        logger.info(f"获取到增量更新日期，类型: {type(dates_to_process)}, 第一个日期: {dates_to_process[0] if dates_to_process else 'None'}, 日期列表长度: {len(dates_to_process)}")
                        # 检查日期类型
                        if dates_to_process and isinstance(dates_to_process[0], str):
                            logger.info(f"日期格式样例: {dates_to_process[0]}, 类型: {type(dates_to_process[0])}")
                    else:
                        logger.info(f"增量更新未返回任何日期，返回类型: {type(dates_to_process)}")
                    
                    # 确保dates_to_process是列表类型
                    if isinstance(dates_to_process, pd.DataFrame):
                        logger.warning(f"update_features_incrementally返回了DataFrame而不是列表，尝试转换")
                        if 'date' in dates_to_process.columns:
                            dates_to_process = dates_to_process['date'].tolist()
                            logger.info(f"从DataFrame['date']列提取了日期列表，长度: {len(dates_to_process)}")
                        else:
                            dates_to_process = list(dates_to_process.index) if hasattr(dates_to_process, 'index') else []
                            logger.info(f"从DataFrame.index提取了日期列表，长度: {len(dates_to_process)}")
                    
                    # 如果增量更新日期为空，我们使用所有交易日（为了演示）
                    if len(dates_to_process) == 0:
                        logger.warning(f"增量更新未返回任何日期，使用所有交易日")
                        dates_to_process = date_range
                        if date_range:
                            logger.info(f"使用全部交易日，类型: {type(date_range)}, 日期样例: {date_range[0] if date_range else 'None'}, 长度: {len(date_range)}")
                        
                except (ImportError, AttributeError) as e:
                    # 如果feature_update模块不可用，处理所有日期
                    logger.warning(f"增量更新功能不可用({e})，将处理板块 {sector} 的所有日期")
                    dates_to_process = date_range
            else:
                # 强制更新，处理所有日期
                logger.info(f"强制更新模式，处理所有日期")
                dates_to_process = date_range
                if date_range:
                    logger.info(f"使用全部交易日，类型: {type(date_range)}, 日期样例: {date_range[0] if date_range else 'None'}, 长度: {len(date_range)}")
            
            # 安全检查 - 确保dates_to_process是列表或可迭代对象，而不是DataFrame
            logger.debug(f"dates_to_process类型: {type(dates_to_process)}")
            if isinstance(dates_to_process, pd.DataFrame):
                logger.warning(f"dates_to_process仍然是DataFrame，将其转换为列表")
                try:
                    if 'date' in dates_to_process.columns:
                        dates_to_process = dates_to_process['date'].tolist()
                    else:
                        dates_to_process = dates_to_process.index.tolist() if hasattr(dates_to_process.index, 'tolist') else []
                    logger.debug(f"转换后dates_to_process类型: {type(dates_to_process)}, 长度: {len(dates_to_process)}")
                except Exception as e:
                    logger.error(f"转换dates_to_process失败: {e}")
                    dates_to_process = []
            
            # 使用len()检查是否为空而不是直接使用布尔值
            try:
                dates_count = len(dates_to_process)
                if dates_count == 0:
                    logger.info(f"板块 {sector} 在日期范围内所有日期的特征都已生成")
                    return sector, []
                logger.info(f"板块 {sector} 需要处理 {dates_count} 个日期")
            except Exception as e:
                logger.error(f"检查dates_to_process长度失败: {e}, 类型: {type(dates_to_process)}")
                return sector, []
            
            # 预加载历史数据 - 优化部分
            # 计算历史数据的开始日期(比最早处理日期提前180天，足够计算所有技术指标)
            try:
                # 将dates_to_process中的日期格式标准化
                normalized_dates = []
                for d in dates_to_process:
                    if isinstance(d, pd.Timestamp):
                        normalized_dates.append(d.strftime('%Y%m%d'))
                    else:
                        normalized_dates.append(str(d).replace('-', ''))
                
                # 找到最早需要处理的日期
                earliest_date_to_process = min(normalized_dates)
                earliest_date_obj = pd.to_datetime(earliest_date_to_process, format='%Y%m%d')
                history_start_date = (earliest_date_obj - pd.Timedelta(days=180)).strftime('%Y%m%d')
                
                logger.info(f"预加载历史数据，从 {history_start_date} 到 {end_date}")
                
                # 获取板块股票列表
                from to.sector_features.feature_generator import SectorFeatureGenerator
                feature_generator = SectorFeatureGenerator(self.data_provider, self.sector_mapping, self.output_dir)
                stocks = feature_generator._get_sector_stocks(sector)
                
                if not stocks:
                    logger.warning(f"板块 {sector} 没有成分股")
                    return sector, []
                
                # 一次性获取所有历史数据
                historical_data = self.data_provider.get_daily_data(stocks, history_start_date, end_date)
                
                if historical_data.empty:
                    logger.warning(f"无法获取板块 {sector} 的历史数据")
                    return sector, []
                
                logger.info(f"成功加载板块 {sector} 的历史数据，行数: {len(historical_data)}")
                
                # 按日期分组历史数据
                historical_data_by_date = {}
                for date_str in normalized_dates:
                    # 将日期转换为与历史数据中相同的格式进行匹配
                    date_obj = pd.to_datetime(date_str, format='%Y%m%d')
                    
                    # 获取当前日期之前的所有数据(包括当日)
                    mask = historical_data['date'] <= date_obj
                    date_historical_data = historical_data[mask].copy()
                    
                    if not date_historical_data.empty:
                        historical_data_by_date[date_str] = date_historical_data
                        logger.debug(f"为日期 {date_str} 准备了 {len(date_historical_data)} 行历史数据")
                    else:
                        logger.warning(f"日期 {date_str} 没有找到历史数据")
                
                logger.info(f"完成历史数据预处理，共 {len(historical_data_by_date)} 个日期有有效数据")
            
            except Exception as e:
                logger.error(f"预加载历史数据时出错: {e}")
                logger.error(traceback.format_exc())
                historical_data_by_date = {}
            
            # 分批处理日期
            all_features = []
            last_day_features = None
            start_time = time.time()
            
            for i in range(0, len(dates_to_process), batch_size):
                batch_dates = dates_to_process[i:i+batch_size]
                
                # 过滤无效日期值
                filtered_batch_dates = []
                for date_item in batch_dates:
                    if date_item == 'date_str':
                        logger.warning(f"从批处理中移除特殊值'date_str'，板块: {sector}")
                        continue
                    if not isinstance(date_item, (str, pd.Timestamp)):
                        logger.warning(f"从批处理中移除非字符串/Timestamp类型: {date_item}, 类型: {type(date_item)}, 板块: {sector}")
                        continue
                    filtered_batch_dates.append(date_item)
                
                batch_results = []
                
                for date in filtered_batch_dates:
                    # 标准化日期格式
                    date_str = date.strftime('%Y%m%d') if isinstance(date, pd.Timestamp) else str(date).replace('-', '')
                    
                    # 获取该日期的预加载数据
                    preloaded_data = historical_data_by_date.get(date_str)
                    
                    # 使用预加载数据处理日期
                    success, df = self.process_date(date, sector, preloaded_data)
                    if success and df is not None:
                        batch_results.append(df)
                        processed_dates.append(date)
                        
                        # 如果是最后一个交易日，保存它的特征用于生成CSV
                        # 将date转换为相同类型，以确保比较有效
                        date_str = date.strftime('%Y%m%d') if isinstance(date, pd.Timestamp) else str(date)
                        last_day_str = last_trading_day.strftime('%Y%m%d') if isinstance(last_trading_day, pd.Timestamp) else str(last_trading_day)
                        
                        # 记录详细日志
                        logger.debug(f"比较日期: '{date_str}'({type(date_str)}) 和最后交易日: '{last_day_str}'({type(last_day_str)})")
                        
                        if date_str == last_day_str:
                            logger.info(f"找到最后交易日 {date_str}，保存其特征")
                            last_day_features = df
                
                # 保存批次结果
                if len(batch_results) > 0:  # 使用len()检查是否为空
                    # 合并批次特征
                    batch_df = pd.concat(batch_results, ignore_index=True)
                    all_features.append(batch_df)
                    
                    # 输出进度
                    logger.debug(f"板块 {sector} 已处理 {len(processed_dates)}/{len(dates_to_process)} 个日期")
            
            # 合并所有特征
            if len(all_features) > 0:  # 使用len()检查是否为空
                # 合并为一个DataFrame
                all_df = pd.concat(all_features, ignore_index=True)
                
                # 排序并去重
                all_df = all_df.sort_values('date').drop_duplicates(subset=['date'],keep='last')
                
                
                # 保存完整特征文件为Parquet
                sector_dir = Path(self.output_dir) / sector
                sector_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存Parquet文件
                parquet_path = sector_dir / f"{sector}.parquet"
                from feature_compress import save_features as save_features_func
                save_features_func(
                    all_df, 
                    parquet_path, 
                    format='parquet',
                    compress=True,
                    optimize=True
                )
                
                # 只为最后一个交易日生成CSV文件
                if last_day_features is not None:
                    csv_path = sector_dir / f"{sector}_latest.csv"
                    from feature_compress import save_features as save_csv
                    save_csv(last_day_features, csv_path, format='csv', compress=False)
                    logger.debug(f"为最后一个交易日 {last_trading_day} 生成CSV文件: {csv_path}")
            
            # 记录耗时
            time_spent = time.time() - start_time
            logger.info(f"完成板块 {sector} 的特征数据处理，耗时 {time_spent:.2f} 秒，处理了 {len(processed_dates)} 个日期")
            
            return sector, processed_dates
        
        except Exception as e:
            logger.error(f"处理板块 {sector} 特征时出错: {e}")
            return sector, processed_dates
    
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
        
        # 如果只有一个进程或板块数小于进程数，直接返回
        if max_workers <= 1 or len(sectors) <= 1:
            return [sectors]
        
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
        force_update: bool = False,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        output_format: str = 'parquet'  # 添加输出格式参数
    ) -> Dict[str, List[str]]:
        """处理日期范围内的所有板块特征
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            sectors: 板块列表，None则使用所有板块
            force_update: 是否强制重新生成
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
        
        # 记录当前内存使用
        self.log_memory_usage()
        
        # 根据进程数划分板块
        sector_groups = self.split_sectors_by_worker(sectors, max_workers)
        
        # 使用多进程处理
        results = {}

        print(f"max_workers: {max_workers}")
        
        if max_workers > 1 and len(sectors) > 1:
            print(f"多进程处理 {len(sectors)} 个板块")
            # 多进程处理
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for sector_group in sector_groups:
                    for sector in sector_group:
                        future = executor.submit(
                            self.process_sector,
                            sector,
                            start_date,
                            end_date,
                            force_update,
                            batch_size,
                            self._output_format  # 传递输出格式
                        )
                        futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        sector, processed_dates = future.result()
                        results[sector] = processed_dates
                    except Exception as e:
                        logger.error(f"处理板块特征时出错: {e}")
        else:
            # 单进程处理
            print(f"单进程处理 {len(sectors)} 个板块")
            for sector in sectors:
                sector, processed_dates = self.process_sector(
                    sector,
                    start_date,
                    end_date,
                    force_update,
                    batch_size,
                    self._output_format  # 传递输出格式
                )
                results[sector] = processed_dates
        
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
        
        logger.info(f"内存使用: {memory_info.rss / (1024 * 1024):.2f} MB")

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
                # 查找所有日期Parquet文件
                date_files = list(sector_dir.glob(f"{sector}_20*.parquet"))
                
                if not date_files:
                    logger.info(f"板块 {sector} 没有需要合并的日期Parquet文件")
                    
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
                                
                                # 更新元数据 - 提取已处理日期
                                if 'date' in main_df.columns:
                                    # 获取所有日期
                                    dates = main_df['date'].astype(str).tolist()
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
                                    
                        except Exception as e:
                            logger.warning(f"读取主Parquet文件失败: {e}")
                    
                    return True
                
                # 主Parquet文件路径
                main_parquet = sector_dir / f"{sector}_features.parquet"
                
                # 加载和合并特征数据
                dfs = []
                
                # 如果存在主Parquet文件，先加载它
                if main_parquet.exists():
                    try:
                        main_df = pd.read_parquet(main_parquet)
                        if 'date' in main_df.columns and not pd.api.types.is_datetime64_any_dtype(main_df['date']):
                            main_df['date'] = pd.to_datetime(main_df['date'])
                        logger.info(f"加载主Parquet文件: {main_parquet}, {len(main_df)} 行")
                        dfs.append(main_df)
                    except Exception as e:
                        logger.warning(f"加载主Parquet文件失败: {e}")
                
                # 加载日期Parquet文件
                for file_path in date_files:
                    try:
                        df = pd.read_parquet(file_path)
                        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
                            df['date'] = pd.to_datetime(df['date'])
                        dfs.append(df)
                        logger.debug(f"加载Parquet文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"加载Parquet文件 {file_path} 失败: {e}")
                
                if not dfs:
                    logger.warning(f"板块 {sector} 所有Parquet文件加载失败")
                    return False
                
                # 合并数据
                merged_df = pd.concat(dfs, ignore_index=True)
                
                # 去重并排序
                if 'date' in merged_df.columns:
                    merged_df = merged_df.drop_duplicates(subset=['date']).sort_values('date')
                    logger.info(f"合并后的特征数据共 {len(merged_df)} 行，日期范围: {merged_df['date'].min()} 到 {merged_df['date'].max()}")
                
                # 对浮点列四舍五入到两位小数
                for col in merged_df.select_dtypes(include=['float64', 'float32']).columns:
                    merged_df[col] = merged_df[col].round(2)
                
                # 优化数据类型以减少文件大小
                for col in merged_df.columns:
                    if pd.api.types.is_float_dtype(merged_df[col]):
                        merged_df[col] = merged_df[col].astype('float32')
                    elif pd.api.types.is_integer_dtype(merged_df[col]) and col != 'date':
                        col_min, col_max = merged_df[col].min(), merged_df[col].max()
                        if col_min >= 0 and col_max <= 255:
                            merged_df[col] = merged_df[col].astype('uint8')
                        elif col_min >= 0 and col_max <= 65535:
                            merged_df[col] = merged_df[col].astype('uint16')
                
                # 保存合并后的数据到主Parquet文件
                try:
                    merged_df.to_parquet(main_parquet, index=False)
                    logger.info(f"成功保存合并后的特征到主Parquet文件: {main_parquet}")
                    
                    # 保存最新日期的数据为CSV
                    if 'date' in merged_df.columns and not merged_df.empty:
                        latest_date = merged_df['date'].max()
                        latest_df = merged_df[merged_df['date'] == latest_date]
                        latest_path = sector_dir / f"{sector}_latest.csv"
                        latest_df.to_csv(latest_path, index=False, encoding='utf-8')
                        logger.info(f"为板块 {sector} 创建最新特征CSV文件，日期: {latest_date}")
                    
                    # 成功保存后，删除原始日期Parquet文件
                    for file_path in date_files:
                        try:
                            file_path.unlink()
                            logger.debug(f"删除文件: {file_path}")
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
                        logger.info(f"已更新板块 {sector} 的元数据，处理了 {len(processed_dates)} 个日期")
                    
                    return True
                except Exception as e:
                    logger.error(f"保存合并后的主Parquet文件失败: {e}")
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
        force_update: bool = False,
        batch_size: Optional[int] = None,
        merge_after_generation: bool = True,
        output_format: Optional[str] = None
    ) -> Dict[str, int]:
        """生成板块特征
        
        Args:
            sectors: 板块名称列表
            start_date: 开始日期
            end_date: 结束日期
            force_update: 是否强制重新生成
            batch_size: 每批处理的日期数
            merge_after_generation: 是否在生成后合并特征文件
            output_format: 输出格式，"parquet"或"csv"，默认使用self._output_format
            
        Returns:
            特征生成统计字典 {板块名称: 成功天数}
        """
        try:
            # 设置输出格式
            if output_format is not None:
                self._output_format = output_format
            elif not hasattr(self, '_output_format'):
                self._output_format = 'parquet'
                
            logger.info(f"使用输出格式: {self._output_format}")
            
            # 开始计时
            start_time = time.time()
            logger.info(f"开始为 {len(sectors)} 个板块生成特征，时间范围: {start_date} - {end_date}")
            
            # 跟踪结果
            results = {}
            
            # 并行处理每个板块
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交任务
                futures = []
                for sector in sectors:
                    # 如果不强制重新生成，先检查需要处理的日期
                    if not force_update:
                        # 获取需要增量更新的日期
                        missing_dates = update_features_incrementally(sector, start_date, end_date, self.output_dir)
                        
                        # 如果没有需要更新的日期，跳过
                        if not missing_dates:
                            logger.info(f"板块 {sector} 在 {start_date} - {end_date} 已有特征数据，跳过")
                            results[sector] = 0
                            continue
                        
                        logger.info(f"板块 {sector} 需要更新 {len(missing_dates)} 个日期的特征")
                    else:
                        # 获取所有交易日
                        try:
                            missing_dates = get_trading_days(start_date, end_date)
                        except Exception as e:
                            logger.error(f"获取交易日失败: {e}")
                            missing_dates = []
                        
                        logger.info(f"强制更新板块 {sector} 的所有特征，共 {len(missing_dates)} 个日期")
                    
                    # 处理每个板块
                    future = executor.submit(
                        self.process_sector,
                        sector,
                        start_date,
                        end_date,
                        force_update,
                        batch_size,
                        self._output_format  # 传递输出格式
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
                        logger.error(f"错误详情: {traceback.format_exc()}")
                        results[sector] = 0
            
            # 在生成后合并特征文件
            if merge_after_generation:
                for sector in sectors:
                    try:
                        logger.info(f"正在合并板块 {sector} 的特征文件")
                        self.merge_features(sector, output_format=self._output_format)
                    except Exception as e:
                        logger.error(f"合并板块 {sector} 的特征文件时出错: {e}")
            
            # 计算总处理时间
            elapsed_time = time.time() - start_time
            logger.info(f"特征生成完成，耗时: {elapsed_time:.2f} 秒")
            
            # 清理缓存，释放内存
            self._feature_info_cache = {}
            gc.collect()
            
            return results
        
        except Exception as e:
            logger.error(f"生成板块特征时出错: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
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
    
    # 更新模式
    update_mode = parser.add_mutually_exclusive_group()
    update_mode.add_argument('--force', action='store_true', help='强制重新生成所有特征')
    update_mode.add_argument('--incremental', action='store_true', help='增量更新，只处理缺失的日期')
    
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
        
    # 强制更新和增量更新不能同时设置
    if args.force and args.incremental:
        logger.error("--force和--incremental不能同时设置")
        sys.exit(1)
        
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
                force_update=args.force,
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
                force_update=args.force,
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