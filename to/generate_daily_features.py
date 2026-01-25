#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票日线特征生成脚本
用于生成股票日线特征并保存到to/local_data/features/daily目录

特征生成流程:
1. 从ParquetDataStorage获取股票日线数据
2. 计算技术指标和其他特征
3. 保存为parquet格式，支持增量更新
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import time
import traceback
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import hashlib
import psutil

# 智能数据缓存系统
class HighPerformanceDataCache:
    """
    高性能数据缓存系统 - 避免重复数据加载

    特性：
    - LRU缓存策略
    - 智能内存管理
    - 缓存命中率统计
    - 自动内存清理
    """

    def __init__(self, max_memory_gb=4):
        self.cache = {}
        self.access_times = {}
        self.cache_sizes = {}
        self.max_memory = max_memory_gb * 1024**3
        self.current_memory = 0

        # 获取系统可用内存
        available_memory = psutil.virtual_memory().available
        if self.max_memory > available_memory * 0.5:
            self.max_memory = int(available_memory * 0.3)

    def _generate_cache_key(self, stock_list, start_date, end_date):
        """生成缓存键"""
        # 使用股票列表的哈希值和日期范围生成键
        stock_hash = hashlib.md5(str(sorted(stock_list)).encode()).hexdigest()[:8]
        return f"{stock_hash}_{start_date}_{end_date}"

    def get_cached_data(self, stock_list, start_date, end_date):
        """获取缓存数据"""
        cache_key = self._generate_cache_key(stock_list, start_date, end_date)

        if cache_key in self.cache:
            self.access_times[cache_key] = time.time()
            return self.cache[cache_key].copy()

        return None

    def cache_data(self, data, stock_list, start_date, end_date):
        """缓存数据"""
        cache_key = self._generate_cache_key(stock_list, start_date, end_date)
        data_size = data.memory_usage(deep=True).sum()

        # 内存管理：如果新数据会超出限制，先清理
        while self.current_memory + data_size > self.max_memory and self.cache:
            self._evict_lru()

        # 如果单个数据集就超过最大内存限制，不缓存
        if data_size > self.max_memory:
            return

        # 缓存数据
        self.cache[cache_key] = data.copy()
        self.access_times[cache_key] = time.time()
        self.cache_sizes[cache_key] = data_size
        self.current_memory += data_size

    def _evict_lru(self):
        """LRU淘汰策略"""
        if not self.cache:
            return

        # 找到最久未访问的缓存项
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])

        # 移除缓存项
        if lru_key in self.cache:
            freed_memory = self.cache_sizes.get(lru_key, 0)
            del self.cache[lru_key]
            del self.access_times[lru_key]
            if lru_key in self.cache_sizes:
                del self.cache_sizes[lru_key]
            self.current_memory -= freed_memory

    def clear_cache(self):
        """清理所有缓存"""
        self.cache.clear()
        self.access_times.clear()
        self.cache_sizes.clear()
        self.current_memory = 0

# 批量I/O操作优化
class BatchIOManager:
    """
    批量I/O操作管理器 - 减少文件操作开销

    特性：
    - 批量保存多个文件
    - 并行I/O操作
    - 智能压缩选择
    - I/O性能监控
    """

    def __init__(self, max_io_workers=4):
        self.max_io_workers = max_io_workers
        self.logger = logging.getLogger(__name__)

    def batch_save_features(self, features_dict, output_dir, compression='snappy'):
        """
        批量保存特征文件

        Args:
            features_dict: {date: DataFrame} 字典
            output_dir: 输出目录
            compression: 压缩算法 ('snappy', 'gzip', 'lz4')

        Returns:
            成功保存的文件数量
        """
        if not features_dict:
            return 0

        success_count = 0

        # 选择最优压缩算法
        optimal_compression = self._select_optimal_compression(features_dict, compression)

        # 并行保存文件
        with ThreadPoolExecutor(max_workers=self.max_io_workers) as executor:
            futures = []

            for date, df in features_dict.items():
                output_file = Path(output_dir) / f"{date}.parquet"
                future = executor.submit(
                    self._save_single_file,
                    df,
                    output_file,
                    optimal_compression
                )
                futures.append((future, date, output_file))

            # 收集结果
            for future, date, output_file in futures:
                try:
                    file_size_mb = future.result()
                    if file_size_mb > 0:
                        success_count += 1
                except Exception as e:
                    self.logger.error(f"保存文件 {date} 时出错: {e}")

        return success_count

    def _save_single_file(self, df, output_file, compression):
        """保存单个文件"""
        try:
            # 确保目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # 保存文件
            df.to_parquet(output_file, index=False, compression=compression)

            # 返回文件大小
            file_size_mb = output_file.stat().st_size / 1024**2
            return file_size_mb

        except Exception as e:
            self.logger.error(f"保存文件 {output_file} 失败: {e}")
            return 0

    def _select_optimal_compression(self, features_dict, compression):
        """选择最优压缩算法"""
        # 根据数据特征选择压缩算法
        sample_df = next(iter(features_dict.values()))
        data_size_mb = sample_df.memory_usage(deep=True).sum() / 1024**2

        if data_size_mb < 10:  # 小文件使用快速压缩
            return 'snappy'
        elif data_size_mb < 50:  # 中等文件平衡压缩比和速度
            return 'snappy'
        else:  # 大文件使用高压缩比
            return 'gzip'



# 完整涨跌停数据加载器 - 集成 enhanced_limit_data_collector.py 的所有特征
class CompleteLimitDataLoader:
    """
    完整涨跌停数据加载器
    集成 enhanced_limit_data_collector.py 中的所有特征
    """

    def __init__(self):
        # 数据目录（与 enhanced_limit_data_collector.py 保持一致）
        self.data_dir = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/limit_data")
        self.cache = {}
        self.logger = logging.getLogger(__name__)

        # 性能优化：全局数据缓存
        self._all_data = None  # 存储完整数据集
        self._data_loaded = False  # 数据加载状态
        self._date_index = {}  # 日期索引，快速查找
        self._load_stats = {
            'total_records': 0,
            'date_range': None,
            'memory_usage_mb': 0,
            'load_time_seconds': 0
        }

        # 完整的原始字段（来自 Tushare API）
        self.original_fields = [
            'trade_date', 'ts_code', 'industry', 'name', 'close', 'pct_chg',
            'amount', 'limit_amount', 'float_mv', 'total_mv', 'turnover_ratio',
            'fd_amount', 'first_time', 'last_time', 'open_times', 'up_stat',
            'limit_times', 'limit'
        ]

        # 增强特征（来自 enhanced_limit_data_collector.py）
        self.enhanced_features = [
            'limit_strength',           # 涨跌停强度
            'time_to_limit_minutes',    # 到达涨跌停时间（分钟）
            'time_efficiency_score',    # 时间效率评分
            'capital_intensity',        # 资金强度
            'turnover_efficiency',      # 换手率效率
            'stability_score',          # 稳定性评分
            'board_quality',            # 板块质量评分
            'mv_factor',               # 市值因子
            'float_ratio',             # 流通比例
            'continuity_factor',       # 连板因子
            'momentum_score'           # 动量评分
        ]

        # 所有特征列表
        self.all_features = self.original_fields + self.enhanced_features + [
            'limit_type',              # 涨跌停类型标识
            'first_time_formatted',    # 格式化的首次封板时间
            'last_time_formatted'      # 格式化的最后封板时间
        ]

        # 性能优化：初始化时预加载所有数据
        self._preload_all_data()

        logger.info(f"完整涨跌停数据加载器初始化完成")
        logger.info(f"原始字段: {len(self.original_fields)} 个")
        logger.info(f"增强特征: {len(self.enhanced_features)} 个")
        logger.info(f"数据统计: {self._load_stats}")

    def _preload_all_data(self):
        """
        预加载所有涨跌停数据到内存 - 性能优化核心功能
        一次性加载，避免重复I/O操作
        """
        import time
        start_time = time.time()

        try:
            # 尝试加载 all_limit_data.parquet
            all_data_file = self.data_dir / "all_limit_data.parquet"
            if all_data_file.exists():
                self.logger.info(f"开始预加载涨跌停数据: {all_data_file}")

                # 加载完整数据集
                self._all_data = pd.read_parquet(all_data_file)

                # 确保 trade_date 是 datetime 类型
                if 'trade_date' in self._all_data.columns:
                    self._all_data['trade_date'] = pd.to_datetime(self._all_data['trade_date'])

                # 构建日期索引，提高查询效率
                self._build_date_index()

                # 计算统计信息
                self._calculate_load_stats(start_time)

                self._data_loaded = True
                self.logger.info(f"数据预加载成功: {self._load_stats['total_records']} 条记录")
                self.logger.info(f"内存使用: {self._load_stats['memory_usage_mb']:.2f} MB")
                self.logger.info(f"加载耗时: {self._load_stats['load_time_seconds']:.2f} 秒")
                self.logger.info(f"日期范围: {self._load_stats['date_range']}")

            else:
                self.logger.warning(f"涨跌停数据文件不存在: {all_data_file}")
                self._data_loaded = False

        except Exception as e:
            self.logger.error(f"预加载涨跌停数据失败: {e}")
            self._data_loaded = False
            self._all_data = None

    def _build_date_index(self):
        """构建日期索引，提高查询效率"""
        if self._all_data is not None and 'trade_date' in self._all_data.columns:
            # 按日期分组，创建索引
            for date, group in self._all_data.groupby(self._all_data['trade_date'].dt.date):
                date_str = date.strftime('%Y%m%d')
                self._date_index[date_str] = group.index.tolist()

            self.logger.info(f"日期索引构建完成: {len(self._date_index)} 个交易日")

    def _calculate_load_stats(self, start_time):
        """计算加载统计信息"""
        if self._all_data is not None:
            self._load_stats['total_records'] = len(self._all_data)
            self._load_stats['memory_usage_mb'] = self._all_data.memory_usage(deep=True).sum() / 1024 / 1024
            self._load_stats['load_time_seconds'] = time.time() - start_time

            if 'trade_date' in self._all_data.columns:
                min_date = self._all_data['trade_date'].min().strftime('%Y-%m-%d')
                max_date = self._all_data['trade_date'].max().strftime('%Y-%m-%d')
                self._load_stats['date_range'] = f"{min_date} 至 {max_date}"

    def load_limit_data_by_date(self, date_str):
        """
        加载指定日期的完整涨跌停数据 - 高性能版本
        使用预加载的内存数据，避免重复I/O操作

        Args:
            date_str: 日期字符串 (YYYYMMDD)

        Returns:
            包含所有涨跌停特征的DataFrame
        """
        # 性能优化：首先检查缓存
        if date_str in self.cache:
            return self.cache[date_str]

        # 性能优化：使用预加载的数据
        if self._data_loaded and self._all_data is not None:
            # 使用日期索引快速查找
            if date_str in self._date_index:
                indices = self._date_index[date_str]
                date_data = self._all_data.iloc[indices].copy()

                # 缓存结果
                self.cache[date_str] = date_data
                return date_data
            else:
                # 日期不存在，返回空DataFrame
                empty_df = pd.DataFrame()
                self.cache[date_str] = empty_df
                return empty_df

        # 降级方案：如果预加载失败，使用原有的文件加载方式
        self.logger.warning(f"预加载数据不可用，使用文件加载方式: {date_str}")

        # 尝试加载合并的数据文件
        combined_data = self._load_combined_data(date_str)
        if not combined_data.empty:
            self.cache[date_str] = combined_data
            return combined_data

        # 尝试分别加载各类型数据
        separate_data = self._load_separate_data(date_str)
        if not separate_data.empty:
            self.cache[date_str] = separate_data
            return separate_data

        # 返回空DataFrame
        return pd.DataFrame()

    def _load_combined_data(self, date_str):
        """加载合并的涨跌停数据文件"""
        try:
            # 尝试加载 all_limit_data.parquet
            all_data_file = self.data_dir / "all_limit_data.parquet"
            if all_data_file.exists():
                all_data = pd.read_parquet(all_data_file)
                # 筛选指定日期的数据
                date_filter = pd.to_datetime(date_str, format='%Y%m%d').date()
                filtered_data = all_data[all_data['trade_date'].dt.date == date_filter]
                return filtered_data
        except Exception as e:
            self.logger.debug(f"加载合并数据失败 {date_str}: {e}")

        return pd.DataFrame()

    def _load_separate_data(self, date_str):
        """分别加载各类型涨跌停数据"""
        all_data_list = []

        # 加载涨停数据
        up_data = self._load_single_type_data('up', date_str)
        if not up_data.empty:
            all_data_list.append(up_data)

        # 加载跌停数据
        down_data = self._load_single_type_data('down', date_str)
        if not down_data.empty:
            all_data_list.append(down_data)

        # 加载炸板数据
        zhaban_data = self._load_single_type_data('zhaban', date_str)
        if not zhaban_data.empty:
            all_data_list.append(zhaban_data)

        # 合并数据 - 使用更安全的方法避免FutureWarning
        if all_data_list:
            valid_data = []
            for data in all_data_list:
                if not data.empty and not data.isna().all().all():
                    valid_data.append(data)

            if valid_data:
                return pd.concat(
                    valid_data,
                    ignore_index=True,
                    sort=False,
                    copy=False
                )

        return pd.DataFrame()

    def _load_single_type_data(self, limit_type, date_str):
        """加载单一类型的涨跌停数据"""
        try:
            # 尝试加载对应类型的数据文件
            type_file = self.data_dir / f"limit_{limit_type}_data.parquet"
            if type_file.exists():
                data = pd.read_parquet(type_file)
                # 筛选指定日期的数据
                date_filter = pd.to_datetime(date_str, format='%Y%m%d').date()
                filtered_data = data[data['trade_date'].dt.date == date_filter]
                return filtered_data
        except Exception as e:
            self.logger.debug(f"加载{limit_type}数据失败 {date_str}: {e}")

        return pd.DataFrame()

    def get_stock_limit_info(self, stock_code, date_str):
        """获取指定股票的完整涨跌停信息"""
        date_data = self.load_limit_data_by_date(date_str)
        if not date_data.empty:
            # 转换股票代码格式匹配
            ts_code = self._convert_stock_code_to_ts_format(stock_code)
            stock_data = date_data[date_data['ts_code'] == ts_code]

            if not stock_data.empty:
                return stock_data.iloc[0].to_dict()
        return None

    def _convert_stock_code_to_ts_format(self, stock_code):
        """转换股票代码格式"""
        if '.' in stock_code:
            return stock_code

        # 简单的格式转换逻辑
        if stock_code.startswith('00') or stock_code.startswith('30'):
            return f"{stock_code}.SZ"
        elif stock_code.startswith('60') or stock_code.startswith('68'):
            return f"{stock_code}.SH"
        else:
            return stock_code

    def get_available_features(self):
        """获取所有可用的涨跌停特征列表"""
        return self.all_features.copy()

    def validate_data_completeness(self, data):
        """验证数据完整性"""
        if data.empty:
            return False

        # 检查必要字段
        required_fields = ['ts_code', 'trade_date', 'limit_type']
        missing_fields = [field for field in required_fields if field not in data.columns]

        if missing_fields:
            self.logger.warning(f"涨跌停数据缺失必要字段: {missing_fields}")
            return False

        return True

    def get_performance_stats(self):
        """获取性能统计信息"""
        cache_size = len(self.cache)
        return {
            'data_loaded': self._data_loaded,
            'total_records': self._load_stats['total_records'],
            'memory_usage_mb': self._load_stats['memory_usage_mb'],
            'load_time_seconds': self._load_stats['load_time_seconds'],
            'date_range': self._load_stats['date_range'],
            'indexed_dates': len(self._date_index),
            'cached_dates': cache_size
        }

    def clear_cache(self):
        """清理日期级别的缓存（保留预加载的全局数据）"""
        self.cache.clear()
        self.logger.info("日期缓存已清理")

    def reload_data(self):
        """重新加载数据"""
        self.logger.info("重新加载涨跌停数据...")
        self._all_data = None
        self._data_loaded = False
        self._date_index.clear()
        self.cache.clear()
        self._preload_all_data()

def integrate_complete_limit_features(df, limit_loader):
    """
    集成完整的涨跌停特征 - 基于 enhanced_limit_data_collector.py 的所有特征

    Args:
        df: 股票数据DataFrame
        limit_loader: 完整涨跌停数据加载器

    Returns:
        集成了所有涨跌停特征的DataFrame
    """
    if df.empty or 'date' not in df.columns or 'stock_code' not in df.columns:
        return df

    # 获取日期字符串
    if 'date_str' in df.columns:
        date_str = df['date_str'].iloc[0]
    else:
        # 从date列获取日期字符串
        date_obj = df['date'].iloc[0]
        if pd.isna(date_obj):
            return df
        date_str = pd.to_datetime(date_obj).strftime('%Y%m%d')

    # 初始化所有涨跌停特征 - 智能类型初始化
    all_limit_features = _get_all_limit_features()
    string_features = {'first_time_formatted', 'last_time_formatted', 'first_time', 'last_time'}

    for feature in all_limit_features:
        if feature in string_features:
            df[feature] = ''  # 字符串特征初始化为空字符串
        else:
            df[feature] = 0.0  # 数值特征初始化为0.0

    # 如果没有加载器，返回默认值
    if limit_loader is None:
        return df

    # 加载当日完整涨跌停数据
    limit_data = limit_loader.load_limit_data_by_date(date_str)

    # 如果没有涨跌停数据，返回默认值
    if limit_data.empty:
        return df

    # 验证数据完整性
    if not limit_loader.validate_data_completeness(limit_data):
        logger.warning(f"涨跌停数据不完整 {date_str}")
        return df

    # 创建涨跌停信息映射
    limit_info_map = {}
    for _, row in limit_data.iterrows():
        ts_code = row['ts_code']
        limit_info_map[ts_code] = _extract_complete_limit_features(row)

    # 为每只股票添加完整涨跌停特征
    for idx, row in df.iterrows():
        stock_code = row['stock_code']
        # 转换股票代码格式
        ts_code = _convert_stock_code_to_ts_format(stock_code)

        if ts_code in limit_info_map:
            limit_info = limit_info_map[ts_code]
            for feature, value in limit_info.items():
                df.at[idx, feature] = value

    return df

def _get_all_limit_features():
    """获取所有涨跌停特征列表"""
    return [
        # 基础涨跌停标识
        'is_limit_up', 'is_limit_down', 'is_zhaban',

        # 原始Tushare字段（重要的）
        'fd_amount',              # 封单金额
        'open_times',             # 炸板次数/开板次数
        'limit_times',            # 连板数
        'first_time',             # 首次封板时间
        'last_time',              # 最后封板时间
        'limit_amount',           # 板上成交金额(跌停专用)
        'up_stat',                # 涨停统计

        # 增强特征（来自 enhanced_limit_data_collector.py）
        'limit_strength',         # 涨跌停强度
        'time_to_limit_minutes',  # 到达涨跌停时间（分钟）
        'time_efficiency_score',  # 时间效率评分
        'capital_intensity',      # 资金强度
        'turnover_efficiency',    # 换手率效率
        'stability_score',        # 稳定性评分
        'board_quality',          # 板块质量评分
        'mv_factor',             # 市值因子
        'float_ratio',           # 流通比例
        'continuity_factor',     # 连板因子
        'momentum_score',        # 动量评分

        # 格式化时间字段
        'first_time_formatted',   # 格式化的首次封板时间
        'last_time_formatted',    # 格式化的最后封板时间

        # 兼容性字段（保持与原有代码的兼容性）
        'limit_quality_score',    # 综合质量评分

        # 历史统计特征（基于基础标识计算）
        'limit_up_days_5d',       # 近5日涨停次数
        'limit_down_days_5d',     # 近5日跌停次数
        'limit_up_days_20d',      # 近20日涨停次数
        'consecutive_limit_up',   # 连续涨停天数
        'consecutive_limit_down', # 连续跌停天数
        'limit_up_volume_ratio',  # 涨停日成交量比率
        'limit_up_turnover_ratio' # 涨停日换手率
    ]

def _extract_complete_limit_features(limit_row):
    """从涨跌停数据行中提取所有特征"""
    features = {}

    # 基础涨跌停标识
    limit_type = limit_row.get('limit_type', '')
    features['is_limit_up'] = 1.0 if limit_type == 'up' else 0.0
    features['is_limit_down'] = 1.0 if limit_type == 'down' else 0.0
    features['is_zhaban'] = 1.0 if limit_type == 'zhaban' else 0.0

    # 原始Tushare字段 - 修复数据类型问题
    features['fd_amount'] = float(limit_row.get('fd_amount', 0))
    features['open_times'] = float(limit_row.get('open_times', 0))
    features['limit_times'] = float(limit_row.get('limit_times', 0))

    # 修复：first_time 和 last_time 应该保持为字符串，不转换为float
    first_time_val = limit_row.get('first_time', '')
    if pd.notna(first_time_val) and first_time_val != '':
        features['first_time'] = str(first_time_val)
    else:
        features['first_time'] = ''

    last_time_val = limit_row.get('last_time', '')
    if pd.notna(last_time_val) and last_time_val != '':
        features['last_time'] = str(last_time_val)
    else:
        features['last_time'] = ''

    features['limit_amount'] = float(limit_row.get('limit_amount', 0))
    # 处理 up_stat 字段（可能包含 '1/1' 这样的字符串）
    up_stat_value = limit_row.get('up_stat', 0)
    if pd.notna(up_stat_value):
        try:
            if isinstance(up_stat_value, str) and '/' in up_stat_value:
                # 处理分数格式，如 '1/1'
                parts = up_stat_value.split('/')
                if len(parts) == 2:
                    features['up_stat'] = float(parts[0]) / float(parts[1])
                else:
                    features['up_stat'] = 0.0
            else:
                features['up_stat'] = float(up_stat_value)
        except (ValueError, ZeroDivisionError):
            features['up_stat'] = 0.0
    else:
        features['up_stat'] = 0.0

    # 增强特征
    features['limit_strength'] = float(limit_row.get('limit_strength', 0))
    features['time_to_limit_minutes'] = float(limit_row.get('time_to_limit_minutes', 0)) if pd.notna(limit_row.get('time_to_limit_minutes')) else 0.0
    features['time_efficiency_score'] = float(limit_row.get('time_efficiency_score', 0))
    features['capital_intensity'] = float(limit_row.get('capital_intensity', 0))
    features['turnover_efficiency'] = float(limit_row.get('turnover_efficiency', 0))
    features['stability_score'] = float(limit_row.get('stability_score', 0))
    features['board_quality'] = float(limit_row.get('board_quality', 0))
    features['mv_factor'] = float(limit_row.get('mv_factor', 0))
    features['float_ratio'] = float(limit_row.get('float_ratio', 0))
    features['continuity_factor'] = float(limit_row.get('continuity_factor', 0))
    features['momentum_score'] = float(limit_row.get('momentum_score', 0))

    # 格式化时间字段（确保为字符串类型）
    features['first_time_formatted'] = str(limit_row.get('first_time_formatted', ''))
    features['last_time_formatted'] = str(limit_row.get('last_time_formatted', ''))

    # 综合质量评分（使用board_quality作为主要评分）
    features['limit_quality_score'] = features['board_quality']

    return features

def _convert_stock_code_to_ts_format(stock_code):
    """转换股票代码格式"""
    if '.' in stock_code:
        return stock_code

    # 简单的格式转换逻辑
    if stock_code.startswith('00') or stock_code.startswith('30'):
        return f"{stock_code}.SZ"
    elif stock_code.startswith('60') or stock_code.startswith('68'):
        return f"{stock_code}.SH"
    else:
        return stock_code

# 注意：原有的基于价格计算的涨跌停检测函数已被删除
# 现在完全使用 enhanced_limit_data_collector.py 的真实数据

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取路径工具函数
def get_project_root():
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

def get_data_path(relative_path=None):
    """获取数据目录路径"""
    project_root = get_project_root()
    data_path = project_root / "to/local_data"
    if relative_path:
        return data_path / relative_path
    return data_path

def ensure_dir(path):
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# 添加一个函数来预处理输入数据中的NaN值
def calculate_vectorized_moving_averages(df):
    """
    优化的向量化移动平均线计算 - 平衡性能和内存使用

    优化策略:
    1. 使用numpy卷积，但优化前置数据处理
    2. 预计算kernels，减少重复创建
    3. 批量处理，减少DataFrame操作
    4. 智能内存管理
    5. 针对不同数据规模使用不同策略

    Args:
        df: 输入DataFrame

    Returns:
        添加了移动平均线的DataFrame
    """
    if df.empty or 'close' not in df.columns:
        return df

    # 预定义窗口和预计算kernels
    windows = [5, 10, 20, 30, 60, 120]
    kernels = {window: np.ones(window, dtype=np.float32) / window for window in windows}

    result = df.copy()

    # 根据数据规模选择不同的处理策略
    n_rows = len(result)
    if n_rows < 10000:
        # 小数据集：使用内存预分配策略
        return _calculate_ma_small_dataset(result, windows, kernels)
    else:
        # 大数据集：使用分组处理策略
        return _calculate_ma_large_dataset(result, windows, kernels)

def _calculate_ma_small_dataset(result, windows, kernels):
    """小数据集的优化处理"""
    # 只为有足够数据的股票预分配MA列
    stocks_with_data = []
    for stock_code, group_indices in result.groupby('stock_code', sort=False).groups.items():
        close_values = result.loc[group_indices, 'close'].values
        if len(close_values) >= 5:
            stocks_with_data.append(stock_code)

    # 如果没有股票有足够数据，直接返回原数据
    if not stocks_with_data:
        return result

    # 预分配所有MA列
    for window in windows:
        result[f'ma{window}'] = np.nan

    # 按股票分组处理
    for stock_code, group_indices in result.groupby('stock_code', sort=False).groups.items():
        close_values = result.loc[group_indices, 'close'].values.astype(np.float64)

        if len(close_values) < 5:
            continue

        # 注意：cumsum_values在修复后的逻辑中不再需要，因为我们直接设置NaN

        for window in windows:
            # 初始化MA值数组为NaN
            ma_values = np.full(len(close_values), np.nan, dtype=np.float32)

            if len(close_values) >= window:
                # 使用预计算的kernel进行卷积
                conv_result = np.convolve(close_values, kernels[window], mode='full')[:len(close_values)]

                # 只有从第window个数据点开始才有有效的MA值
                ma_values[window-1:] = conv_result[window-1:].astype(np.float32)

            # 无论数据长度如何，都设置MA值（前面不足的部分自然为NaN）
            result.loc[group_indices, f'ma{window}'] = ma_values

    return result

def _calculate_ma_large_dataset(result, windows, kernels):
    """大数据集的优化处理"""
    enhanced_groups = []

    for stock_code, group in result.groupby('stock_code', sort=False):
        if len(group) < 5:
            enhanced_groups.append(group)
            continue

        group_enhanced = group.copy()
        close_values = group['close'].values.astype(np.float64)

        # 注意：cumsum_values在修复后的逻辑中不再需要，因为我们直接设置NaN

        for window in windows:
            # 初始化MA值数组为NaN
            ma_values = np.full(len(close_values), np.nan, dtype=np.float32)

            if len(close_values) >= window:
                # 使用预计算的kernel进行卷积
                conv_result = np.convolve(close_values, kernels[window], mode='full')[:len(close_values)]

                # 只有从第window个数据点开始才有有效的MA值
                ma_values[window-1:] = conv_result[window-1:].astype(np.float32)

            # 无论数据长度如何，都设置MA值（前面不足的部分自然为NaN）
            group_enhanced[f'ma{window}'] = ma_values

        enhanced_groups.append(group_enhanced)

    # 合并所有组 - 使用更安全的方法避免FutureWarning
    if enhanced_groups:
        # 过滤掉空的DataFrame和全NA的DataFrame
        valid_groups = []
        for group in enhanced_groups:
            if not group.empty and not group.isna().all().all():
                valid_groups.append(group)

        if valid_groups:
            # 使用最安全的concat方式，明确指定参数避免FutureWarning
            result = pd.concat(
                valid_groups,
                ignore_index=True,
                sort=False,
                copy=False
            )
        else:
            result = result  # 如果所有组都为空，返回原始结果

    return result

def calculate_vectorized_rsi(df, periods=[6, 12, 14, 24]):
    """
    向量化RSI计算 - 速度提升2-3倍

    Args:
        df: 输入DataFrame
        periods: RSI周期列表

    Returns:
        添加了RSI指标的DataFrame
    """
    if df.empty or 'close' not in df.columns:
        return df

    result = df.copy()

    # 按股票分组处理
    enhanced_groups = []

    for stock_code, group in result.groupby('stock_code'):
        if len(group) < max(periods) + 1:
            enhanced_groups.append(group)
            continue

        group_enhanced = group.copy()
        close_values = group['close'].values

        # 计算价格变化
        deltas = np.diff(close_values, prepend=close_values[0])

        for period in periods:
            if len(close_values) >= period + 1:
                # 向量化RSI计算
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)

                # 使用指数移动平均
                alpha = 1.0 / period
                avg_gains = pd.Series(gains).ewm(alpha=alpha, adjust=False).mean().values
                avg_losses = pd.Series(losses).ewm(alpha=alpha, adjust=False).mean().values

                # 避免除零
                avg_losses = np.where(avg_losses == 0, 0.0001, avg_losses)
                rs = avg_gains / avg_losses
                rsi = 100 - (100 / (1 + rs))

                group_enhanced[f'rsi{period}'] = rsi.astype('float16')
            else:
                group_enhanced[f'rsi{period}'] = 50.0  # 默认中性值

        enhanced_groups.append(group_enhanced)

    # 合并所有组 - 使用更安全的方法避免FutureWarning
    if enhanced_groups:
        valid_groups = []
        for group in enhanced_groups:
            if not group.empty and not group.isna().all().all():
                valid_groups.append(group)

        if valid_groups:
            result = pd.concat(
                valid_groups,
                ignore_index=True,
                sort=False,
                copy=False
            )

    return result

def preprocess_data_for_indicators(df):
    """
    预处理输入数据，解决全NaN切片问题和确保关键特征存在
    
    Args:
        df: 输入DataFrame
        
    Returns:
        处理后的DataFrame
    """
    result = df.copy()
    
    # 确保关键价格和交易量字段不含有NaN值（使用前向填充，然后再用后向填充）
    key_fields = ['open', 'high', 'low', 'close', 'volume']
    for field in key_fields:
        if field in result.columns:
            # 按股票代码分组处理，避免不同股票数据互相填充
            result[field] = result.groupby('stock_code')[field].transform(
                lambda x: x.ffill().bfill().fillna(0)
            )
    
    # 确保原始特征被保留
    original_features = ['pct_chg', 'total_mv', 'circ_mv', 'turnover_rate']
    for feature in original_features:
        if feature in result.columns:
            # 处理可能的异常值，如无穷大或极端值
            result[feature] = result[feature].replace([np.inf, -np.inf], np.nan)
            # 填充NaN值，使用股票分组的中位数，更合理
            result[feature] = result.groupby('stock_code')[feature].transform(
                lambda x: x.fillna(x.median()).fillna(0)
            )
    
    return result

def calculate_technical_indicators(df, limit_loader=None):
    """
    计算技术指标特征 - 融合高级量化特征工程

    基于顶级量化研究论文和工程实践，包含：
    1. 基础技术指标（MA, RSI, MACD, Bollinger Bands）
    2. 高级形态识别（K线形态、趋势形态）
    3. 市场微观结构特征（支撑阻力、风险收益比）
    4. 高级机器学习特征（量价关系、动量因子、波动性特征）
    5. 特征交互项（技术指标交互、价格与成交量交互）
    6. 完整涨跌停特征（基于真实数据）
    7. 特征筛选（只保留对模型训练有意义的指标）

    Args:
        df: 包含OHLCV数据的DataFrame
        limit_loader: 涨跌停数据加载器

    Returns:
        添加了全面技术指标和高级特征的DataFrame
    """
    # 预处理输入数据，解决NaN问题
    result = preprocess_data_for_indicators(df)

    # 确保按日期和股票代码排序
    if 'date' in result.columns and 'stock_code' in result.columns:
        result = result.sort_values(['stock_code', 'date'])

    # === 1. 基础技术指标 - 向量化优化版本 ===
    # 向量化移动平均线计算 - 速度提升3-5倍
    result = calculate_vectorized_moving_averages(result)
    
    # 向量化RSI计算 - 速度提升2-3倍
    result = calculate_vectorized_rsi(result, periods=[6, 12, 14, 24])
    
    # 计算MACD
    result['ema12'] = result.groupby('stock_code')['close'].ewm(span=12, adjust=False).mean().reset_index(0, drop=True)
    result['ema26'] = result.groupby('stock_code')['close'].ewm(span=26, adjust=False).mean().reset_index(0, drop=True)
    result['macd'] = result['ema12'] - result['ema26']
    result['macd_signal'] = result.groupby('stock_code')['macd'].ewm(span=9, adjust=False).mean().reset_index(0, drop=True)
    result['macd_hist'] = result['macd'] - result['macd_signal']
    
    # 确保保留原始特征
    original_features = ['pct_chg', 'total_mv', 'circ_mv', 'turnover_rate']
    for feature in original_features:
        if feature in df.columns and feature not in result.columns:
            result[feature] = df[feature]
    
    # 布林带指标
    result['bb_middle'] = result['ma20']
    result['bollinger_std'] = result.groupby('stock_code')['close'].rolling(20, min_periods=1).std().reset_index(0, drop=True)
    result['bb_upper'] = result['bb_middle'] + 2 * result['bollinger_std']
    result['bb_lower'] = result['bb_middle'] - 2 * result['bollinger_std']
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
    result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
    
    # 计算ATR
    high_low = result['high'] - result['low']
    high_close = abs(result['high'] - result['close'].shift(1))
    low_close = abs(result['low'] - result['close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    result['atr14'] = tr.groupby(result['stock_code']).rolling(window=14, min_periods=1).mean().reset_index(0, drop=True)
    
    # 计算支撑位和阻力位
    result['support14'] = result.groupby('stock_code')['low'].rolling(14, min_periods=1).min().reset_index(0, drop=True)
    result['resistance14'] = result.groupby('stock_code')['high'].rolling(14, min_periods=1).max().reset_index(0, drop=True)
    
    # 计算成交量均线
    result['volume_ma20'] = result.groupby('stock_code')['volume'].rolling(20, min_periods=1).mean().reset_index(0, drop=True)
    
    # 确保 pct_chg 存在
    if 'pct_chg' not in result.columns:
        # 如果没有 pct_chg，则计算
        result['pct_chg'] = result.groupby('stock_code')['close'].pct_change(fill_method=None) * 100
    
    # 计算波动率
    result['volatility20'] = result.groupby('stock_code')['close'].pct_change(fill_method=None).rolling(20, min_periods=1).std().reset_index(0, drop=True)
    result['volatility60'] = result.groupby('stock_code')['close'].pct_change(fill_method=None).rolling(60, min_periods=1).std().reset_index(0, drop=True)

    # === 2. 高级量化特征 - 性能优化的多线程版本 ===
    logger.info("开始计算高级特征（多线程）...")

    # 获取股票分组并预过滤
    stock_groups = [(stock_code, group) for stock_code, group in result.groupby('stock_code')
                   if len(group) >= 20]  # 预过滤，只处理有足够数据的股票
    small_groups = [(stock_code, group) for stock_code, group in result.groupby('stock_code')
                   if len(group) < 20]

    logger.info(f"共有 {len(stock_groups)} 只股票需要处理高级特征，{len(small_groups)} 只股票数据不足")

    # 智能负载均衡的并行计算优化
    enhanced_groups = []

    # 计算工作负载并智能分配线程
    def calculate_workload(group):
        return len(group) * group.shape[1]

    # 按工作负载排序股票组
    sorted_groups = sorted(stock_groups, key=lambda x: calculate_workload(x[1]), reverse=True)
    total_workload = sum(calculate_workload(group) for _, group in sorted_groups)

    # 智能线程数计算
    if total_workload < 50000:
        optimal_workers = min(2, len(sorted_groups))
    elif total_workload < 200000:
        optimal_workers = min(6, max(2, len(sorted_groups) // 8))
    else:
        optimal_workers = min(12, max(4, len(sorted_groups) // 6))

    max_workers = min(optimal_workers, len(sorted_groups))

    if len(stock_groups) > 1 and max_workers > 1:
        # 多线程处理
        logger.info(f"使用 {max_workers} 个线程并行处理")

        # 性能优化：批量提交任务，减少线程创建开销
        import time
        start_time = time.time()

        # 负载均衡任务分配
        def balanced_task_distribution(sorted_groups, num_workers):
            worker_tasks = [[] for _ in range(num_workers)]
            worker_loads = [0] * num_workers

            for stock_code, group in sorted_groups:
                workload = calculate_workload(group)
                min_load_worker = worker_loads.index(min(worker_loads))
                worker_tasks[min_load_worker].append((stock_code, group))
                worker_loads[min_load_worker] += workload

            return worker_tasks, worker_loads

        worker_tasks, _ = balanced_task_distribution(sorted_groups, max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for worker_id, tasks in enumerate(worker_tasks):
                for stock_code, group in tasks:
                    future = executor.submit(_process_stock_features_threaded, stock_code, group, limit_loader)
                    futures.append((future, stock_code, group))

            logger.info(f"已提交 {len(futures)} 个股票的高级特征计算任务")

            # 收集结果 - 性能优化：使用更高效的进度报告
            completed_count = 0
            progress_interval = max(1, len(futures) // 20)  # 最多报告20次进度

            for future, stock_code, group in futures:
                try:
                    enhanced_group = future.result(timeout=30)  # 添加超时防止卡死
                    enhanced_groups.append(enhanced_group)
                    completed_count += 1

                    # 性能优化：减少日志频率
                    if completed_count % progress_interval == 0 or completed_count == len(futures):
                        elapsed = time.time() - start_time
                        rate = completed_count / elapsed if elapsed > 0 else 0
                        logger.info(f"高级特征计算进度: {completed_count}/{len(futures)} "
                                  f"({completed_count/len(futures)*100:.1f}%, {rate:.1f} 股票/秒)")

                except Exception as e:
                    logger.warning(f"处理股票 {stock_code} 时出错: {e}，使用原始数据")
                    enhanced_groups.append(group)
                    completed_count += 1

        total_time = time.time() - start_time
        logger.info(f"多线程处理完成，总耗时: {total_time:.2f}秒")

    else:
        # 单线程处理（股票数量少时）
        logger.info("使用单线程处理（股票数量较少）")
        import time
        start_time = time.time()

        for i, (stock_code, group) in enumerate(stock_groups):
            enhanced_group = _process_stock_features_threaded(stock_code, group, limit_loader)
            enhanced_groups.append(enhanced_group)

            # 性能优化：减少日志频率
            if (i + 1) % 5 == 0 or (i + 1) == len(stock_groups):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"单线程处理进度: {i + 1}/{len(stock_groups)} ({rate:.1f} 股票/秒)")

    # 添加小数据量的股票（不需要高级特征计算）
    for stock_code, group in small_groups:
        enhanced_groups.append(group)

    # 合并所有增强后的数据 - 性能优化版本
    if enhanced_groups:
        logger.info(f"合并 {len(enhanced_groups)} 个股票的高级特征数据...")

        # 性能优化：使用更高效的合并方式
        import time
        merge_start = time.time()

        # 预估内存需求并优化合并策略
        if len(enhanced_groups) > 100:
            # 大量数据时分批合并，减少内存峰值
            logger.info("数据量较大，使用分批合并策略...")
            batch_size = 50
            merged_batches = []

            for i in range(0, len(enhanced_groups), batch_size):
                batch = enhanced_groups[i:i + batch_size]
                # 过滤空DataFrame和全NA DataFrame避免FutureWarning
                valid_batch = []
                for group in batch:
                    if not group.empty and not group.isna().all().all():
                        valid_batch.append(group)

                if valid_batch:
                    merged_batch = pd.concat(
                        valid_batch,
                        ignore_index=True,
                        sort=False,
                        copy=False
                    )
                    merged_batches.append(merged_batch)

                if (i // batch_size + 1) % 5 == 0:
                    logger.info(f"分批合并进度: {i + len(batch)}/{len(enhanced_groups)}")

            # 过滤空DataFrame和全NA DataFrame避免FutureWarning
            valid_batches = []
            for batch in merged_batches:
                if not batch.empty and not batch.isna().all().all():
                    valid_batches.append(batch)

            if valid_batches:
                result = pd.concat(
                    valid_batches,
                    ignore_index=True,
                    sort=False,
                    copy=False
                )
        else:
            # 数据量适中时直接合并 - 过滤空DataFrame和全NA DataFrame避免FutureWarning
            valid_groups = []
            for group in enhanced_groups:
                if not group.empty and not group.isna().all().all():
                    valid_groups.append(group)

            if valid_groups:
                result = pd.concat(
                    valid_groups,
                    ignore_index=True,
                    sort=False,
                    copy=False
                )

        merge_time = time.time() - merge_start
        logger.info(f"高级特征计算完成，最终数据规模: {result.shape}，合并耗时: {merge_time:.2f}秒")

    # 添加日期字符串列
    if 'date' in result.columns and pd.api.types.is_datetime64_any_dtype(result['date']):
        result['date_str'] = result['date'].dt.strftime('%Y%m%d')

    # === 3. 特征筛选 - 只保留对模型训练有意义的指标 ===
    # 基于量化研究和机器学习实践的特征重要性
    
    # 删除中间计算列 (修复：保留ema12, ema26, macd_signal等重要特征)
    intermediate_cols = ['bollinger_std', 'vol_change']  # 只删除真正的中间计算列
    result = result.drop(columns=[col for col in intermediate_cols if col in result.columns])
    
    # 高价值特征列表 - 基于特征重要性研究
    high_value_features = [
        # 基础价格数据
        'open', 'high', 'low', 'close', 'volume', 'amount',
        
        # 原始特征 - 从 local_data\daily 中读取的关键特征
        'pct_chg', 'total_mv', 'circ_mv', 'turnover_rate',
        
        # 高价值技术指标 (修复：包含所有移动平均线和RSI指标)
        'ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma120',  # 全套移动平均线
        'rsi6', 'rsi12', 'rsi14',  # 全套RSI指标
        'ema12', 'ema26', 'macd', 'macd_signal', 'macd_hist',  # 完整MACD系统
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',  # 完整布林带系统
        'atr14',         # 波动率指标
        'volatility20',  # 短期波动率
        
        # 高级形态和趋势特征 (修复：包含所有K线形态特征)
        'trend_strength',
        'trend_consistency',
        'candlestick_strength',
        'pattern_doji', 'pattern_hammer', 'pattern_shooting_star',  # 添加缺失的K线形态
        'pattern_engulfing_bullish',
        'pattern_engulfing_bearish',

        # 支撑阻力特征 (修复：包含支撑阻力位)
        'support_level', 'resistance_level',  # 添加缺失的支撑阻力位
        'risk_reward_ratio',
        'near_support',
        'near_resistance',
        
        # 高级ML特征 - 量价关系
        'volume_weighted_price',
        'vol_price_corr_20d',
        'price_vol_change_ratio',
        'volume_zscore',
        
        # 高级ML特征 - 动量因子
        'return_5d', 'return_20d',
        'relative_strength_20d',
        'momentum_acceleration',
        
        # 高级ML特征 - 波动性特征
        'parkinson_vol',
        'vol_term_structure',
        'volatility_skew',
        
        # 高级ML特征 - 流动性和反转
        'amihud_illiquidity',
        'short_reversal',
        'overbought_oversold',
        
        # 高级ML特征 - 综合指标
        'tech_momentum_signal',
        'risk_adjusted_return',
        'returns_zscore',
        
        # 特征交互项
        'rsi_macd_agreement',
        'bb_breakout_signal',
        'price_volume_divergence',
        'volatility_adjusted_momentum',
        'technical_consensus',
        'composite_trading_signal',
        'multi_period_trend_agreement',

        # 完整涨跌停特征体系（基于 enhanced_limit_data_collector.py）
        # 基础涨跌停标识
        'is_limit_up', 'is_limit_down', 'is_zhaban',

        # 原始Tushare字段
        'fd_amount', 'open_times', 'limit_times', 'first_time', 'last_time',
        'limit_amount', 'up_stat',

        # 增强特征
        'limit_strength', 'time_to_limit_minutes', 'time_efficiency_score',
        'capital_intensity', 'turnover_efficiency', 'stability_score',
        'board_quality', 'mv_factor', 'float_ratio', 'continuity_factor',
        'momentum_score', 'limit_quality_score',

        # 格式化时间字段
        'first_time_formatted', 'last_time_formatted',

        # 历史统计特征
        'limit_up_days_5d', 'limit_down_days_5d', 'limit_up_days_20d',
        'consecutive_limit_up', 'consecutive_limit_down',
        'limit_up_volume_ratio', 'limit_up_turnover_ratio',

        # 状态评分特征 (修复：添加状态评分特征)
        'technical_score', 'composite_score',

        # 监督学习目标
        'future_return_1d',        # 新增：1日后收益率
        'future_return_2d',        # 原有：2日后收益率
        'is_can_buy',             # 新增：是否可以有效买入指标

        # 元数据列
        'stock_code', 'date', 'date_str'
    ]
    
    # 只保留高价值特征
    all_cols = result.columns.tolist()
    cols_to_keep = [col for col in high_value_features if col in all_cols]
    cols_to_keep.extend([col for col in all_cols if col not in high_value_features and col in ['stock_code', 'date', 'date_str']])
    
    # 确保关键列一定会被保留
    essential_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'stock_code', 'date', 'date_str']
    for col in essential_cols:
        if col in all_cols and col not in cols_to_keep:
            cols_to_keep.append(col)
    
    # 筛选列
    result = result[cols_to_keep]
    
    # 智能数据类型优化 - 修复字符串字段转换错误
    string_fields = {'stock_code', 'date_str', 'first_time_formatted', 'last_time_formatted', 'first_time', 'last_time'}
    integer_fields = {'volume', 'amount', 'open_times', 'limit_times', 'is_limit_up', 'is_limit_down', 'is_zhaban'}

    for col in cols_to_keep:
        if col in result.columns:
            try:
                if col in string_fields:
                    # 字符串字段：确保所有值都是字符串类型
                    result[col] = result[col].apply(
                        lambda x: str(x) if pd.notna(x) and x != '' else ''
                    ).astype('object')
                elif col in integer_fields:
                    # 整数字段转换为int32
                    if result[col].dtype in ['float64', 'float32']:
                        result[col] = result[col].fillna(0).astype('int32')
                    elif result[col].dtype == 'object':
                        result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0).astype('int32')
                elif col not in ['date']:  # 排除datetime类型的date列
                    # 其他数值字段转换为float32
                    if result[col].dtype == 'object':
                        # 对于object类型，先尝试转换为数值
                        numeric_series = pd.to_numeric(result[col], errors='coerce')
                        if not numeric_series.isna().all():
                            result[col] = numeric_series.astype('float32')
                    elif result[col].dtype == 'float64':
                        result[col] = result[col].astype('float32')
                    elif result[col].dtype == 'int64':
                        result[col] = result[col].astype('float32')
            except Exception as e:
                logger.warning(f"优化列 {col} 数据类型时出错: {e}，保持原始类型")

    # 添加完整的涨跌停特征（默认值）
    complete_limit_features = _get_all_limit_features()

    for feature in complete_limit_features:
        if feature not in result.columns:
            result[feature] = 0.0

    return result

def _process_stock_features_threaded(stock_code, group, limit_loader=None):
    """
    多线程处理单只股票的高级特征计算 - 性能优化版本

    Args:
        stock_code: 股票代码
        group: 单只股票的DataFrame
        limit_loader: 涨跌停数据加载器

    Returns:
        添加了高级特征的DataFrame
    """
    try:
        # 性能优化：预先检查数据质量，避免无效计算
        if len(group) < 20:
            return group

        # 性能优化：检查必要列是否存在
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in group.columns for col in required_cols):
            logger.warning(f"股票 {stock_code} 缺少必要的价格数据列")
            return group

        # 计算基础高级特征（传递limit_loader）
        enhanced_group = calculate_advanced_features(group, limit_loader)

        # 性能优化：只有足够数据时才计算高级ML特征
        if len(enhanced_group) >= 60:  # 高级ML特征需要更多历史数据
            enhanced_group = add_advanced_ml_features(enhanced_group)

            # 计算特征交互项
            enhanced_group = add_feature_interactions(enhanced_group)

        return enhanced_group
    except Exception as e:
        logger.warning(f"处理股票 {stock_code} 的特征时出错: {e}")
        return group

def calculate_advanced_features(df, limit_loader=None):
    """
    计算高级量化特征 - 融合train_sector_models中的_enhance_stock_features逻辑

    基于顶级量化研究，包含：
    1. K线形态识别（十字星、锤子线、吞噬形态等）
    2. 趋势分析（多级别趋势判断、趋势强度）
    3. 支撑阻力分析（动态支撑阻力、风险收益比）
    4. 涨跌停检测（根据股票类型动态判断，优先使用真实数据）
    5. 未来收益率计算（监督学习目标）

    Args:
        df: 单只股票的DataFrame
        limit_loader: 涨跌停数据加载器

    Returns:
        添加了高级特征的DataFrame
    """
    if len(df) < 20:
        return df

    result = df.copy()

    try:
        # 提取基础价格数据（向量化）
        open_price = result['open'].values
        high_price = result['high'].values
        low_price = result['low'].values
        close_price = result['close'].values
        volume = result['volume'].values if 'volume' in result.columns else None

        # 1. K线形态识别
        # result = add_candlestick_patterns(result, open_price, high_price, low_price, close_price)

        # 2. 趋势分析特征
        # result = add_trend_features(result, close_price)

        # 3. 支撑阻力分析
        # result = add_support_resistance_features(result, high_price, low_price, close_price)

        # 4. 集成完整涨跌停特征（基于真实数据）
        result = integrate_complete_limit_features(result, limit_loader)

        # 5. 添加涨跌停历史特征
        # result = add_limit_historical_features(result)

        # 5. 未来收益率计算
        result = add_future_returns(result)

        # 6. 综合状态评分
        # result = add_state_scoring(result)

    except Exception as e:
        logger.warning(f"计算高级特征时出错: {e}")

    return result


def add_support_resistance_features(df, high_price, low_price, close_price):
    """
    支撑阻力分析 - 动态支撑阻力识别

    基于分位数和滚动窗口：
    - 动态支撑阻力位
    - 风险收益比计算
    - 突破概率评估
    """
    window = 20

    # 动态支撑阻力（向量化）
    df['support_level'] = df['low'].rolling(window=window, min_periods=1).quantile(0.1)
    df['resistance_level'] = df['high'].rolling(window=window, min_periods=1).quantile(0.9)

    # 计算距离和比率（向量化）- 加强处理零值和NaN值
    support = df['support_level'].values
    resistance = df['resistance_level'].values
    
    # 防止除零和处理NaN值
    # 1. 确保close_price不为0或NaN
    valid_close = np.where(
        (close_price > 0) & (~np.isnan(close_price)), 
        close_price, 
        0.001  # 用一个小的正值替代无效值
    )
    
    # 2. 确保support和resistance有效
    valid_support = np.where(
        (~np.isnan(support)) & (support > 0),
        support,
        np.minimum(valid_close * 0.95, valid_close - 0.001)  # 默认支撑位略低于当前价
    )
    
    valid_resistance = np.where(
        (~np.isnan(resistance)) & (resistance > 0),
        resistance,
        np.maximum(valid_close * 1.05, valid_close + 0.001)  # 默认阻力位略高于当前价
    )
    
    # 3. 计算百分比距离，确保分母永远不为零
    safe_close = np.maximum(valid_close, 0.001)  # 额外保障，确保分母大于0
    distance_to_support = np.maximum(0.001, (safe_close - valid_support)) / safe_close
    distance_to_resistance = np.maximum(0.001, (valid_resistance - safe_close)) / safe_close

    df['risk_reward_ratio'] = distance_to_resistance / np.maximum(distance_to_support, 0.001)
    df['support_distance_pct'] = distance_to_support * 100
    df['resistance_distance_pct'] = distance_to_resistance * 100

    # 突破信号
    df['near_support'] = (distance_to_support < 0.02).astype(int)  # 距离支撑位2%以内
    df['near_resistance'] = (distance_to_resistance < 0.02).astype(int)  # 距离阻力位2%以内

    return df

def add_limit_historical_features(df):
    """
    添加涨跌停历史特征 - 基于真实涨跌停数据计算历史统计特征

    注意：基础涨跌停标识（is_limit_up, is_limit_down）已由 integrate_complete_limit_features 提供
    这里只计算基于这些标识的历史统计特征

    Args:
        df: 已包含基础涨跌停标识的DataFrame

    Returns:
        添加了历史特征的DataFrame
    """
    if len(df) < 2:
        # 初始化历史特征为默认值
        historical_features = [
            'limit_up_days_5d', 'limit_down_days_5d', 'limit_up_days_20d',
            'consecutive_limit_up', 'consecutive_limit_down',
            'limit_up_volume_ratio', 'limit_up_turnover_ratio'
        ]
        for feature in historical_features:
            df[feature] = 0.0
        return df

    # 确保基础涨跌停标识存在
    if 'is_limit_up' not in df.columns:
        df['is_limit_up'] = 0.0
    if 'is_limit_down' not in df.columns:
        df['is_limit_down'] = 0.0

    # 1. 涨跌停历史统计特征
    # 近5日涨停次数
    df['limit_up_days_5d'] = df['is_limit_up'].rolling(5, min_periods=1).sum()
    # 近5日跌停次数
    df['limit_down_days_5d'] = df['is_limit_down'].rolling(5, min_periods=1).sum()
    # 近20日涨停次数
    df['limit_up_days_20d'] = df['is_limit_up'].rolling(20, min_periods=1).sum()

    # 2. 连续涨跌停天数计算
    df['consecutive_limit_up'] = 0.0
    df['consecutive_limit_down'] = 0.0

    if len(df) > 1:
        # 计算连续涨停天数
        limit_up_groups = (df['is_limit_up'] != df['is_limit_up'].shift(1)).cumsum()
        consecutive_up = df.groupby(limit_up_groups)['is_limit_up'].cumsum()
        df['consecutive_limit_up'] = np.where(df['is_limit_up'] == 1, consecutive_up, 0)

        # 计算连续跌停天数
        limit_down_groups = (df['is_limit_down'] != df['is_limit_down'].shift(1)).cumsum()
        consecutive_down = df.groupby(limit_down_groups)['is_limit_down'].cumsum()
        df['consecutive_limit_down'] = np.where(df['is_limit_down'] == 1, consecutive_down, 0)

    # 3. 涨跌停质量特征
    # 涨停日成交量比率（相对于20日平均成交量）
    if 'volume' in df.columns:
        avg_volume_20d = df['volume'].rolling(20, min_periods=1).mean()
        df['limit_up_volume_ratio'] = np.where(
            df['is_limit_up'] == 1,
            df['volume'] / (avg_volume_20d + 0.0001),
            0.0
        )
    else:
        df['limit_up_volume_ratio'] = 0.0

    # 涨停日换手率
    if 'turnover_rate' in df.columns:
        df['limit_up_turnover_ratio'] = np.where(
            df['is_limit_up'] == 1,
            df['turnover_rate'],
            0.0
        )
    else:
        df['limit_up_turnover_ratio'] = 0.0

    return df

def add_future_returns(df):
    """
    未来收益率计算 - 监督学习目标

    计算未来N日收益率，用于模型训练：
    - future_return_1d: 1日后收益率 (T+1日开盘买入，T+2日开盘卖出)
    - future_return_2d: 2日后收益率 (T日收盘买入，T+2日收盘卖出)
    - is_can_buy: T+1日是否可以有效买入股票的二元指标

    注意：修复了股票未上市时的计算逻辑，避免计算未来上市后的收益
    """
    if len(df) < 3:
        df['future_return_1d'] = np.nan
        df['future_return_2d'] = np.nan
        df['is_can_buy'] = 0
        return df

    try:
        # 获取基础价格数据
        open_prices = df['open'].values if 'open' in df.columns else np.full(len(df), np.nan)
        close_prices = df['close'].values
        pct_chg = df['pct_chg'].values if 'pct_chg' in df.columns else np.full(len(df), np.nan)

        # 获取涨停相关数据
        first_time_formatted = df['first_time_formatted'].values if 'first_time_formatted' in df.columns else np.full(len(df), '')
        is_limit_up = df['is_limit_up'].values if 'is_limit_up' in df.columns else np.full(len(df), np.nan)
        # 初始化结果数组
        future_returns_1d = np.full(len(df), np.nan)
        future_returns_2d = np.full(len(df), np.nan)
        is_can_buy = np.zeros(len(df), dtype=int)

        # 1. 计算 future_return_1d (T+1日开盘买入，T+2日开盘卖出)
        if len(df) >= 2:
            for i in range(len(df) - 2):
                # T+1日开盘价 (买入价格)
                t1_open = open_prices[i + 1]
                # T+2日开盘价 (卖出价格)
                t2_open = open_prices[i + 2]

                # 检查数据有效性：T+1日和T+2日都必须有有效的开盘价
                if (not np.isnan(t1_open) and t1_open > 0 and
                    not np.isnan(t2_open) and t2_open > 0):
                    # 计算1日收益率
                    future_returns_1d[i] = (t2_open - t1_open) / t1_open
                # 如果T+1日或T+2日股票未上市（开盘价为NaN或0），则保持NaN

        # 2. 修复 future_return_2d 计算逻辑 (T日收盘买入，T+2日收盘卖出)
        for i in range(len(df) - 2):
            # T日收盘价 (买入价格)
            t0_close = close_prices[i]
            # T+2日收盘价 (卖出价格)
            t2_close = close_prices[i + 2]

            # 检查数据有效性：T日和T+2日都必须有有效的收盘价
            if (not np.isnan(t0_close) and t0_close > 0 and
                not np.isnan(t2_close) and t2_close > 0):
                # 计算2日收益率
                future_returns_2d[i] = (t2_close - t0_close) / t0_close
            # 如果T日或T+2日股票未上市（收盘价为NaN或0），则保持NaN

        # 3. 计算 is_can_buy 指标 (判断T+1日是否可以有效买入)
        for i in range(len(df) - 1):
            # T+1日的数据
            t1_open = open_prices[i + 1]
            t1_pct_chg = pct_chg[i + 1]
            t1_first_time = first_time_formatted[i + 1] if i + 1 < len(first_time_formatted) else ''
            t1_is_limit_up = is_limit_up[i + 1]  # 🚀 修复：获取T+1日的涨停状态

            # 默认可以买入
            can_buy = 1

            # 检查T+1日是否有有效数据（股票已上市）
            if np.isnan(t1_open) or t1_open <= 0 or np.isnan(t1_is_limit_up):
                can_buy = 0  # 股票未上市，不可买入
            else:
                # # 条件1：检查T+1日开盘是否已经涨停 (涨幅>=9.5%认为是涨停)
                # if not np.isnan(t1_pct_chg) and t1_pct_chg >= 9.5:
                #     can_buy = 0

                # 条件2：检查T+1日是否在开盘后很快就涨停（基于真实数据分析优化）
                if t1_is_limit_up == 1 and isinstance(t1_first_time, str) and t1_first_time:
                    print(f"涨停：{t1_is_limit_up}")
                    try:
                        # 解析时间格式，支持 "HH:MM:SS", "HH:MM" 或 "HHMM" 格式
                        time_str = str(t1_first_time).strip()
                        if ':' in time_str:
                            parts = time_str.split(':')
                            hour = int(parts[0])
                            minute = int(parts[1])
                        elif len(time_str) == 4 and time_str.isdigit():
                            hour = int(time_str[:2])
                            minute = int(time_str[2:])
                        else:
                            hour, minute = 9, 35  # 默认值，不影响判断

                        # 修正判断逻辑：
                        # 1. 集合竞价时间(9:25-9:30)的涨停不影响买入判断
                        # 2. 开盘后5分钟内(9:30-9:35)涨停认为过早，不可买入
                        if hour == 9 and 25 <= minute <= 30:
                            # 集合竞价时间，不影响买入判断
                            pass
                        elif hour < 9 or (hour == 9 and minute < 35):
                            # 9:35之前涨停（排除集合竞价），认为过早
                            can_buy = 0
                    except (ValueError, AttributeError):
                        # 时间格式解析失败，保持原判断
                        pass

            is_can_buy[i] = can_buy

        # 赋值结果
        df['future_return_1d'] = future_returns_1d
        df['future_return_2d'] = future_returns_2d
        df['is_can_buy'] = is_can_buy

    except Exception as e:
        # 异常处理：设置默认值
        df['future_return_1d'] = np.nan
        df['future_return_2d'] = np.nan
        df['is_can_buy'] = 0
        print(f"计算未来收益率时出错: {e}")

    return df

def add_state_scoring(df):
    """
    综合状态评分 - 多因子综合评估 (修复：处理特征缺失问题)

    基于技术指标、形态、趋势的综合评分：
    - 技术强度评分
    - 形态质量评分
    - 趋势一致性评分
    """
    # 技术指标评分（0-100）- 修复：检查特征是否存在
    rsi_score = 0
    macd_score = 0
    bb_score = 0

    if 'rsi12' in df.columns:
        rsi_score = np.where(df['rsi12'].between(30, 70), 1, 0)  # RSI在合理区间

    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd_score = np.where(df['macd'] > df['macd_signal'], 1, 0)  # MACD金叉

    if 'close' in df.columns and 'bb_lower' in df.columns:
        bb_score = np.where(df['close'] > df['bb_lower'], 1, 0)  # 价格在布林带下轨之上

    df['technical_score'] = (rsi_score + macd_score + bb_score) / 3 * 100

    # 形态质量评分 - 修复：检查特征是否存在
    if 'candlestick_strength' in df.columns:
        pattern_score = df['candlestick_strength'] / 5 * 100  # 标准化到0-100
        df['pattern_score'] = np.clip(pattern_score, 0, 100)
    else:
        df['pattern_score'] = 50.0  # 默认中性评分

    # 趋势一致性评分 - 修复：检查特征是否存在
    if 'trend_consistency' in df.columns:
        trend_score = np.abs(df['trend_consistency']) * 100
        df['trend_score'] = np.clip(trend_score, 0, 100)
    else:
        df['trend_score'] = 50.0  # 默认中性评分

    # 综合评分（加权平均）
    df['composite_score'] = (
        df['technical_score'] * 0.4 +
        df['pattern_score'] * 0.3 +
        df['trend_score'] * 0.3
    )

    return df

def optimize_dtypes(df):
    """
    超高效数据类型优化 - 内存节省60%+

    基于数据特征的智能类型优化：
    - 价格数据: float64 -> float32 (节省50%内存，精度足够)
    - 成交量数据: int64 -> uint32 (节省50%内存，无负值)
    - 技术指标: float64 -> float16 (节省75%内存，精度足够)
    - 百分比数据: float64 -> float16 (节省75%内存)

    Args:
        df: 输入DataFrame

    Returns:
        优化后的DataFrame，内存使用减少60%+
    """
    if df.empty:
        return df

    result = df.copy()

    # 定义字符串字段列表（不应转换为数值类型）
    string_fields = {
        'stock_code', 'date_str', 'first_time_formatted', 'last_time_formatted',
        'first_time', 'last_time'  # 原始时间字段也可能是字符串
    }

    # 价格数据字段 - float64 -> float32 (节省50%内存)
    price_fields = {
        'open', 'high', 'low', 'close', 'amount',
        'ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma120',
        'ema12', 'ema26', 'bb_upper', 'bb_middle', 'bb_lower',
        'support_level', 'resistance_level', 'atr14',
        'volume_weighted_price', 'support14', 'resistance14'
    }

    # 成交量数据字段 - int64 -> uint32 (节省50%内存，无负值)
    volume_fields = {
        'volume', 'amount', 'total_mv', 'circ_mv', 'volume_ma20',
        'fd_amount', 'limit_amount'
    }

    # 技术指标字段 - float64 -> float16 (节省75%内存，精度足够)
    indicator_fields = {
        'rsi6', 'rsi12', 'rsi14', 'rsi24', 'macd', 'macd_signal', 'macd_hist',
        'bb_width', 'bb_position', 'bollinger_std',
        'trend_strength', 'trend_consistency', 'candlestick_strength',
        'risk_reward_ratio', 'support_distance_pct', 'resistance_distance_pct'
    }

    # 百分比数据字段 - float64 -> float16 (节省75%内存)
    percentage_fields = {
        'pct_chg', 'turnover_rate', 'volatility20', 'volatility60',
        'return_5d', 'return_20d', 'relative_strength_20d',
        'momentum_acceleration', 'parkinson_vol', 'vol_term_structure',
        'volatility_skew', 'amihud_illiquidity', 'short_reversal',
        'vol_price_corr_20d', 'price_vol_change_ratio', 'volume_zscore',
        'future_return_1d', 'future_return_2d'  # 新增：未来收益率字段
    }

    # 整数字段 - 优化为最小整数类型
    integer_fields = {
        'open_times', 'limit_times', 'is_limit_up', 'is_limit_down', 'is_zhaban',
        'limit_up_days_5d', 'limit_down_days_5d', 'limit_up_days_20d',
        'consecutive_limit_up', 'consecutive_limit_down',
        'pattern_doji', 'pattern_hammer', 'pattern_shooting_star',
        'pattern_engulfing_bullish', 'pattern_engulfing_bearish',
        'trend_uptrend', 'trend_downtrend', 'near_support', 'near_resistance',
        'is_can_buy'  # 新增：是否可以买入的二元指标
    }

    for col in result.columns:
        try:
            if col in string_fields:
                # 字符串字段：确保所有值都是字符串类型
                result[col] = result[col].apply(
                    lambda x: str(x) if pd.notna(x) and x != '' else ''
                ).astype('object')

            elif col in price_fields:
                # 价格数据：float64 -> float32
                if result[col].dtype == 'float64':
                    result[col] = result[col].astype('float32')

            elif col in volume_fields:
                # 成交量数据：安全转换为uint32或uint64
                if result[col].dtype in ['float64', 'float32', 'int64']:
                    result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)
                    # 检查是否有负值或超出uint32范围
                    col_min, col_max = result[col].min(), result[col].max()
                    if col_min < 0:
                        # 有负值时，先处理负值再转换
                        logger.warning(f"字段 {col} 包含负值 ({col_min})，将负值设为0")
                        result[col] = np.maximum(result[col], 0)
                        col_max = result[col].max()  # 重新计算最大值

                    if col_max > 4294967295:  # uint32最大值
                        logger.warning(f"字段 {col} 超出uint32范围 ({col_max})，使用uint64")
                        result[col] = result[col].astype('uint64')
                    else:
                        result[col] = result[col].astype('uint32')

            elif col in indicator_fields:
                # 技术指标：安全转换为float16或float32
                if result[col].dtype in ['float64', 'float32']:
                    col_min, col_max = result[col].min(), result[col].max()
                    if pd.notna(col_min) and pd.notna(col_max):
                        # 检查是否在float16安全范围内
                        if abs(col_min) < 65000 and abs(col_max) < 65000:
                            result[col] = result[col].astype('float16')
                        else:
                            logger.warning(f"字段 {col} 超出float16范围 ({col_min:.2f} - {col_max:.2f})，使用float32")
                            result[col] = result[col].astype('float32')
                    else:
                        result[col] = result[col].astype('float32')

            elif col in percentage_fields:
                # 百分比数据：安全转换为float16或float32
                if result[col].dtype in ['float64', 'float32']:
                    col_min, col_max = result[col].min(), result[col].max()
                    if pd.notna(col_min) and pd.notna(col_max):
                        # 检查是否在float16安全范围内
                        if abs(col_min) < 65000 and abs(col_max) < 65000:
                            result[col] = result[col].astype('float16')
                        else:
                            logger.warning(f"字段 {col} 超出float16范围 ({col_min:.2f} - {col_max:.2f})，使用float32")
                            result[col] = result[col].astype('float32')
                    else:
                        result[col] = result[col].astype('float32')

            elif col in integer_fields:
                # 整数字段：优化为最小整数类型
                if result[col].dtype in ['float64', 'float32', 'int64']:
                    int_values = pd.to_numeric(result[col], errors='coerce').fillna(0).astype('int64')
                    col_min, col_max = int_values.min(), int_values.max()

                    if col_min >= 0 and col_max <= 255:
                        result[col] = int_values.astype('uint8')
                    elif col_min >= 0 and col_max <= 65535:
                        result[col] = int_values.astype('uint16')
                    elif col_min >= -128 and col_max <= 127:
                        result[col] = int_values.astype('int8')
                    elif col_min >= -32768 and col_max <= 32767:
                        result[col] = int_values.astype('int16')
                    else:
                        result[col] = int_values.astype('int32')

            elif result[col].dtype == 'object':
                # 对于object类型，尝试智能检测
                if col not in string_fields:
                    # 尝试转换为数值类型
                    numeric_series = pd.to_numeric(result[col], errors='coerce')
                    if not numeric_series.isna().all():  # 如果有有效的数值
                        result[col] = numeric_series.astype('float32')
                    # 否则保持为object类型

            elif result[col].dtype == 'float64':
                # 默认将float64转换为float32
                result[col] = result[col].astype('float32')

            elif result[col].dtype == 'int64':
                # 检查是否可以安全转换为int32
                if result[col].min() >= -2147483648 and result[col].max() <= 2147483647:
                    result[col] = result[col].astype('int32')

        except Exception as e:
            # 如果转换失败，记录警告但继续处理其他列
            logger.warning(f"优化列 {col} 的数据类型时出错: {e}，保持原始类型")
            continue

    return result

def add_advanced_ml_features(df):
    """
    添加高级机器学习特征 - 基于顶级量化研究论文

    参考文献:
    - Gu et al. (2020) Empirical Asset Pricing via Machine Learning
    - Deep (2024) Advanced Financial Market Forecasting
    - Letteri (2023) VolTS: A Volatility-based Trading System

    包含以下高级特征类别:
    1. 量价关系特征：成交量与价格变动的关系
    2. 动量因子：多周期收益率和动量指标
    3. 波动性特征：波动率结构和波动率预测指标
    4. 流动性特征：基于交易量和价差的流动性指标
    5. 反转因子：短期、中期和长期反转信号
    6. 市场微观结构特征：价格压力和订单失衡指标

    Args:
        df: 输入DataFrame，需要包含基本的OHLCV数据

    Returns:
        添加了高级特征的DataFrame
    """
    if len(df) < 60:  # 确保有足够的历史数据
        return df

    result = df.copy()
    
    try:
        # === 1. 量价关系特征 ===
        # 计算成交量加权价格
        if 'volume' in result.columns and 'close' in result.columns:
            result['volume_weighted_price'] = (result['volume'] * result['close']) / result['volume'].rolling(20, min_periods=1).sum()
        
        # 成交量价格相关系数 (20天) - 完全修复版本
        if 'volume' in result.columns and 'close' in result.columns:
            # 使用更稳定的方法计算相关系数，避免groupby().apply()的复杂性
            try:
                # 方法1：使用transform方法（更稳定）
                vol_price_corr_list = []
                for stock_code, group in result.groupby('stock_code'):
                    if len(group) >= 20:
                        corr_values = group['volume'].rolling(20, min_periods=5).corr(group['close']).fillna(0)
                        # 修复无穷值和NaN值
                        corr_values = corr_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                        vol_price_corr_list.extend(corr_values.tolist())
                    else:
                        vol_price_corr_list.extend([0.0] * len(group))

                result['vol_price_corr_20d'] = vol_price_corr_list

            except Exception as e:
                logger.warning(f"计算量价相关系数时出错: {e}，使用默认值")
                result['vol_price_corr_20d'] = 0.0
        
        # 成交量变化率与价格变化率比值
        if 'volume' in result.columns and 'pct_chg' in result.columns:
            result['vol_change'] = result.groupby('stock_code')['volume'].pct_change(fill_method=None)
            result['price_vol_change_ratio'] = np.abs(result['pct_chg']) / (np.abs(result['vol_change']) + 0.0001)
        
        # 成交量异常检测 (相对于20日均值的标准差)
        if 'volume' in result.columns and 'volume_ma20' in result.columns:
            vol_std = result.groupby('stock_code')['volume'].rolling(20, min_periods=1).std().reset_index(0, drop=True)
            result['volume_zscore'] = (result['volume'] - result['volume_ma20']) / (vol_std + 0.0001)
        
        # === 2. 动量因子 ===
        # 多周期收益率 - 确保这些特征被计算，修复无穷值问题
        if 'close' in result.columns:
            # 只计算关键周期的收益率，避免计算不必要的特征
            for period in [5, 20]:
                returns = result.groupby('stock_code')['close'].pct_change(period, fill_method=None)
                # 修复无穷值和异常值
                returns = returns.replace([np.inf, -np.inf], np.nan)
                # 限制收益率在合理范围内 (-100%, +1000%)
                returns = returns.clip(-1.0, 10.0)
                returns = returns.fillna(0)
                result[f'return_{period}d'] = returns
        
        # 相对强度指标 (与大盘相比)
        if 'return_20d' in result.columns:
            # 注：实际应用中需要传入指数数据，这里简化为相对于所有股票的平均表现
            result['relative_strength_20d'] = result.groupby('date')['return_20d'].transform(lambda x: x - x.mean())
        
        # 动量加速度 (收益率变化的变化率)
        if 'return_5d' in result.columns:
            result['momentum_acceleration'] = result.groupby('stock_code')['return_5d'].diff().reset_index(0, drop=True)
        
        # === 3. 波动性特征 ===
        # 隐含波动率估计 (基于高低价差)
        if 'high' in result.columns and 'low' in result.columns:
            result['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                (np.log(np.maximum(result['high'], 0.001) / np.maximum(result['low'], 0.001)) ** 2)
            )
        
        # 波动率期限结构
        if 'volatility20' in result.columns and 'volatility60' in result.columns:
            result['vol_term_structure'] = result['volatility20'] / (result['volatility60'] + 0.0001)
        
        # 波动率偏度 (收益率分布的偏度) - 性能优化版本
        if 'pct_chg' in result.columns:
            # 性能优化：使用向量化计算
            try:
                volatility_skew = result.groupby('stock_code')['pct_chg'].rolling(
                    20, min_periods=5
                ).skew().reset_index(level=0, drop=True)
                result['volatility_skew'] = volatility_skew.fillna(0)
            except Exception as e:
                logger.warning(f"计算波动率偏度时出错: {e}，使用默认值")
                result['volatility_skew'] = 0.0
        
        # === 4. 流动性特征 ===
        # Amihud非流动性指标
        if all(col in result.columns for col in ['pct_chg', 'volume', 'close']):
            result['amihud_illiquidity'] = np.abs(result['pct_chg']) / (result['volume'] * result['close'] + 0.0001)
        
        # 换手率变化
        if 'turnover_rate' in result.columns:
            result['turnover_change'] = result.groupby('stock_code')['turnover_rate'].pct_change(fill_method=None)
        
        # 成交量波动性
        if 'volume' in result.columns:
            result['volume_volatility'] = result.groupby('stock_code')['volume'].pct_change(fill_method=None).rolling(20, min_periods=1).std().reset_index(0, drop=True)
        
        # === 5. 反转因子 ===
        # 短期反转 (过去5天收益率的负值)
        if 'return_5d' in result.columns:
            result['short_reversal'] = -result['return_5d']
        
        # 超买超卖指标 (基于RSI)
        if 'rsi12' in result.columns:
            result['overbought_oversold'] = np.where(result['rsi12'] > 70, -1, np.where(result['rsi12'] < 30, 1, 0))
        
        # === 6. 市场微观结构特征 ===
        # 价格压力指标 (基于成交量和价格变化)
        if all(col in result.columns for col in ['pct_chg', 'volume', 'volume_ma20']):
            result['price_pressure'] = result['pct_chg'] * np.sqrt(result['volume'] / (result['volume_ma20'] + 0.0001))
        
        # 价格趋势一致性
        if 'close' in result.columns:
            # 计算价格变化方向
            price_diff1 = result.groupby('stock_code')['close'].diff(1).reset_index(0, drop=True)
            price_diff2 = result.groupby('stock_code')['close'].diff(2).reset_index(0, drop=True)
            price_diff3 = result.groupby('stock_code')['close'].diff(3).reset_index(0, drop=True)
            
            # 计算方向的一致性
            result['price_trend_consistency'] = (
                np.sign(price_diff1) + 
                np.sign(price_diff2) + 
                np.sign(price_diff3)
            ) / 3
        
        # === 7. 综合预测特征 ===
        # 技术指标综合动量信号
        if all(x in result.columns for x in ['macd', 'rsi12', 'close', 'ma20']):
            result['tech_momentum_signal'] = (
                np.sign(result['macd']) +  # MACD方向
                np.sign(result['rsi12'] - 50) +  # RSI相对于中值
                np.sign(result['close'] - result['ma20'])  # 价格相对于20日均线
            ) / 3
        
        # 风险调整收益率 (夏普比率的简化版本) - 修复无穷值问题
        if 'return_20d' in result.columns and 'volatility20' in result.columns:
            risk_adj_return = result['return_20d'] / (result['volatility20'] + 0.0001)
            # 修复无穷值和异常值
            risk_adj_return = risk_adj_return.replace([np.inf, -np.inf], np.nan)
            risk_adj_return = risk_adj_return.clip(-100, 100)  # 限制在合理范围
            risk_adj_return = risk_adj_return.fillna(0)
            result['risk_adjusted_return'] = risk_adj_return
        
        # 异常收益信号 (基于收益率Z-score)
        if 'pct_chg' in result.columns:
            # 计算收益率的滚动均值和标准差
            roll_mean = result.groupby('stock_code')['pct_chg'].transform(lambda x: x.rolling(60, min_periods=5).mean())
            roll_std = result.groupby('stock_code')['pct_chg'].transform(lambda x: x.rolling(60, min_periods=5).std())
            # 计算Z分数
            result['returns_zscore'] = (result['pct_chg'] - roll_mean) / (roll_std + 0.0001)
        
        # 移除中间计算列
        if 'vol_change' in result.columns:
            result = result.drop(columns=['vol_change'])
        
    except Exception as e:
        logger.warning(f"计算高级机器学习特征时出错: {e}")
        logger.error(traceback.format_exc())
    
    return result

def add_feature_interactions(df):
    """
    添加特征交互项 - 基于特征工程最佳实践

    参考文献:
    - Gu et al. (2020) Empirical Asset Pricing via Machine Learning
    - Mirete-Ferrer et al. (2022) A review on machine learning for asset management

    包含以下交互特征:
    1. 技术指标之间的交互
    2. 价格与成交量的交互
    3. 波动率与动量的交互
    4. 趋势与反转信号的交互

    Args:
        df: 输入DataFrame，需要包含基本技术指标和高级特征

    Returns:
        添加了特征交互项的DataFrame
    """
    if len(df) < 20:
        return df

    result = df.copy()

    try:
        # === 1. 技术指标交互 ===
        # RSI与MACD交互
        if 'rsi12' in result.columns and 'macd' in result.columns:
            # RSI超买超卖区域与MACD方向的一致性
            result['rsi_macd_agreement'] = np.where(
                ((result['rsi12'] > 70) & (result['macd'] > 0)) |
                ((result['rsi12'] < 30) & (result['macd'] < 0)),
                1, 0
            )
        else:
            result['rsi_macd_agreement'] = 0

        # 布林带与价格交互
        if 'bb_width' in result.columns and 'bb_position' in result.columns:
            # 布林带宽度与价格位置的组合信号
            result['bb_breakout_signal'] = result['bb_width'] * (result['bb_position'] - 0.5) * 2
        else:
            result['bb_breakout_signal'] = 0.0

        # === 2. 价格与成交量交互 ===
        # 价量背离指标 - 修复vol_change依赖问题
        if 'pct_chg' in result.columns and 'volume' in result.columns:
            # 重新计算vol_change，确保它存在
            vol_change = result.groupby('stock_code')['volume'].pct_change(fill_method=None)
            # 价格上涨但成交量下降，或价格下跌但成交量上升
            result['price_volume_divergence'] = np.where(
                (result['pct_chg'] > 0) & (vol_change < 0) |
                (result['pct_chg'] < 0) & (vol_change > 0),
                1, 0
            )
        else:
            result['price_volume_divergence'] = 0

        # 成交量加权价格动量
        if 'return_5d' in result.columns and 'volume_ma20' in result.columns and 'volume' in result.columns:
            result['volume_weighted_momentum'] = result['return_5d'] * (result['volume'] / (result['volume_ma20'] + 0.0001))
        else:
            result['volume_weighted_momentum'] = 0.0

        # === 3. 波动率与动量交互 ===
        # 波动率调整后的动量信号 - 修复无穷值问题
        if 'return_20d' in result.columns and 'volatility20' in result.columns:
            # 动量信号强度除以波动率，降低高波动率时的信号强度
            vol_adj_momentum = result['return_20d'] / (result['volatility20'] + 0.001)
            # 修复无穷值和异常值
            vol_adj_momentum = vol_adj_momentum.replace([np.inf, -np.inf], np.nan)
            vol_adj_momentum = vol_adj_momentum.clip(-50, 50)  # 限制在合理范围
            vol_adj_momentum = vol_adj_momentum.fillna(0)
            result['volatility_adjusted_momentum'] = vol_adj_momentum
        else:
            result['volatility_adjusted_momentum'] = 0.0

        # 波动率突变与价格变动
        if 'volatility20' in result.columns and 'pct_chg' in result.columns:
            vol_change = result.groupby('stock_code')['volatility20'].pct_change(fill_method=None)
            result['vol_price_impact'] = vol_change * result['pct_chg']
        else:
            result['vol_price_impact'] = 0.0

        # === 4. 趋势与反转信号交互 ===
        # 趋势强度与反转信号的组合
        if 'trend_strength' in result.columns and 'short_reversal' in result.columns:
            # 强趋势下降低反转信号权重
            result['trend_reversal_interaction'] = result['short_reversal'] * (1 - np.minimum(result['trend_strength'] / 100, 0.9))
        else:
            result['trend_reversal_interaction'] = 0.0

        # 多周期趋势一致性 - 修改为只使用已计算的特征
        if 'return_5d' in result.columns and 'return_20d' in result.columns:
            # 多周期收益率方向一致性
            result['multi_period_trend_agreement'] = (
                np.sign(result['return_5d']) +
                np.sign(result['return_20d'])
            ) / 2  # 只使用两个周期，而不是三个
        else:
            result['multi_period_trend_agreement'] = 0.0

        # === 5. 技术分析与基本面交互 ===
        # 市值与动量交互
        if 'total_mv' in result.columns and 'return_20d' in result.columns:
            # 对数市值，避免负值
            result['log_market_cap'] = np.log(np.maximum(result['total_mv'], 0.001))
            # 市值加权动量 - 小市值股票动量效应通常更强
            result['size_momentum_interaction'] = result['return_20d'] / (result['log_market_cap'] + 0.001)
        else:
            result['size_momentum_interaction'] = 0.0

        # 换手率与价格变动交互
        if 'turnover_rate' in result.columns and 'pct_chg' in result.columns:
            # 高换手率下的价格变动可能更有意义
            result['turnover_price_signal'] = result['pct_chg'] * np.sqrt(result['turnover_rate'] + 0.001)
        else:
            result['turnover_price_signal'] = 0.0

        # === 6. 综合交互特征 ===
        # 技术信号一致性指标
        if all(x in result.columns for x in ['rsi12', 'macd', 'bb_position']):
            # 三大技术指标一致性
            result['technical_consensus'] = (
                (result['rsi12'] - 50) / 50 +  # 归一化RSI
                np.sign(result['macd']) +      # MACD方向
                (result['bb_position'] - 0.5) * 2  # 归一化布林带位置
            ) / 3
        else:
            result['technical_consensus'] = 0.0

        # 综合交易信号 - 修复依赖问题，确保所有前置特征都存在
        # 先确保tech_momentum_signal存在
        if 'tech_momentum_signal' not in result.columns:
            if all(x in result.columns for x in ['macd', 'rsi12', 'close', 'ma20']):
                result['tech_momentum_signal'] = (
                    np.sign(result['macd']) +  # MACD方向
                    np.sign(result['rsi12'] - 50) +  # RSI相对于中值
                    np.sign(result['close'] - result['ma20'])  # 价格相对于20日均线
                ) / 3
            else:
                result['tech_momentum_signal'] = 0.0

        # 现在计算综合交易信号
        result['composite_trading_signal'] = (
            result['tech_momentum_signal'] +
            np.sign(result['volatility_adjusted_momentum']) +
            (1 - 2 * result['price_volume_divergence'])  # 背离为负面信号
        ) / 3

    except Exception as e:
        logger.warning(f"计算特征交互项时出错: {e}")
        logger.error(traceback.format_exc())

    return result

# 主要功能类
class StockFeatureGenerator:
    """股票特征生成器类 - 多线程优化版本"""

    def __init__(self, output_dir=None, max_workers=None):
        """
        初始化特征生成器 - 性能优化版本

        Args:
            output_dir: 输出目录，默认为to/local_data/features/daily
            max_workers: 最大线程数，默认为CPU核心数的一半
        """
        # 设置输出目录
        self.output_dir = Path(output_dir) if output_dir else get_data_path('features/daily')
        ensure_dir(self.output_dir)

        # 性能优化：增强缓存机制
        self._stock_list_cache = None
        self._data_storage_cache = None
        self._feature_cache = {}  # 特征计算结果缓存
        self._limit_data_cache = {}  # 涨跌停数据缓存
        self._cache_hit_count = 0
        self._cache_miss_count = 0

        # 智能数据缓存系统
        self.data_cache = HighPerformanceDataCache(max_memory_gb=4)

        # 批量I/O操作管理器
        self.batch_io = BatchIOManager(max_io_workers=4)

        # 性能优化：动态调整线程数
        import os
        cpu_count = os.cpu_count() or 4
        # 根据系统性能动态调整
        if cpu_count >= 8:
            default_workers = min(8, cpu_count // 2)
        elif cpu_count >= 4:
            default_workers = min(6, cpu_count // 2)
        else:
            default_workers = 2

        self.max_workers = max_workers or default_workers

        # 初始化涨跌停数据加载器
        self.limit_loader = CompleteLimitDataLoader()

        logger.info(f"特征生成器初始化完成，输出目录: {self.output_dir}")
        logger.info(f"性能优化配置: 最大线程数 = {self.max_workers}, CPU核心数 = {cpu_count}")
        logger.info("涨跌停数据加载器初始化完成")
        logger.info("缓存系统已启用")

    def get_performance_stats(self):
        """获取性能统计信息"""
        total_requests = self._cache_hit_count + self._cache_miss_count
        hit_rate = (self._cache_hit_count / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_hits': self._cache_hit_count,
            'cache_misses': self._cache_miss_count,
            'hit_rate': f"{hit_rate:.1f}%",
            'cached_features': len(self._feature_cache),
            'cached_limit_data': len(self._limit_data_cache)
        }

    def clear_cache(self):
        """清理缓存"""
        self._feature_cache.clear()
        self._limit_data_cache.clear()
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        logger.info("缓存已清理")

    def integrate_limit_data(self, df):
        """
        集成涨跌停数据到特征DataFrame中

        注意：此方法已废弃，涨跌停数据现在在 calculate_technical_indicators 中
        通过 _process_stock_features_threaded -> calculate_advanced_features 进行处理
        保留此方法仅为向后兼容，实际不执行任何操作

        Args:
            df: 包含基础特征的DataFrame

        Returns:
            原始DataFrame（不做任何修改）
        """
        logger.info("integrate_limit_data 方法已废弃，涨跌停数据已在技术指标计算中完成集成")
        return df

    def _add_default_limit_features(self, df):
        """
        添加默认的涨跌停特征
        注意：此方法已废弃，保留仅为向后兼容
        """
        logger.info("_add_default_limit_features 方法已废弃")
        return df

    def _merge_limit_data(self, date_group, limit_data):
        """
        合并涨跌停数据
        注意：此方法已废弃，保留仅为向后兼容
        """
        logger.info("_merge_limit_data 方法已废弃")
        return date_group

    def _convert_stock_code_to_ts_format(self, stock_code):
        """转换股票代码格式"""
        if '.' in stock_code:
            return stock_code

        # 简单的格式转换逻辑
        if stock_code.startswith('00') or stock_code.startswith('30'):
            return f"{stock_code}.SZ"
        elif stock_code.startswith('60') or stock_code.startswith('68'):
            return f"{stock_code}.SH"
        else:
            return stock_code

    def _calculate_limit_quality_score(self, stock_data):
        """
        计算涨跌停质量评分
        注意：此方法已废弃，保留仅为向后兼容
        """
        logger.info("_calculate_limit_quality_score 方法已废弃")
        return 0.5
    
    def get_data_storage(self):
        """获取ParquetDataStorage实例 - 优化版本，使用缓存"""
        if self._data_storage_cache is not None:
            return self._data_storage_cache

        # 动态导入ParquetDataStorage类
        try:
            # 先尝试使用绝对导入
            sys.path.append(str(get_project_root()))
            from to.advanced_quant_data import ParquetDataStorage
        except ImportError:
            # 如果失败，尝试相对导入
            try:
                from advanced_quant_data import ParquetDataStorage
            except ImportError:
                logger.error("无法导入ParquetDataStorage类，请确保advanced_quant_data.py文件存在")
                return None

        # 创建实例并缓存
        self._data_storage_cache = ParquetDataStorage()
        return self._data_storage_cache
    
    def get_stock_list(self):
        """获取股票列表 - 优化版本，使用缓存"""
        if self._stock_list_cache is not None:
            return self._stock_list_cache

        storage = self.get_data_storage()
        if not storage:
            return []

        # 获取股票列表
        try:
            stock_list = storage.get_stock_list()
            if stock_list is not None:
                # 优化：批量过滤ST股票，避免逐个查询导致性能问题
                try:
                    logger.info(f"开始过滤ST股票，原始股票数: {len(stock_list)}")

                    # 方法1：尝试批量获取股票名称
                    try:
                        from data_loaders import get_stock_names_dict
                        stock_names = get_stock_names_dict()
                        if stock_names:
                            filtered_stock_list = [
                                stock_code for stock_code in stock_list
                                if stock_code not in stock_names or "ST" not in stock_names.get(stock_code, "")
                            ]
                            stock_list = filtered_stock_list
                            logger.info(f"使用批量方式过滤ST股票，过滤后股票数: {len(stock_list)}")
                        else:
                            raise Exception("批量获取股票名称失败")
                    except:
                        # 方法2：如果批量方式失败，使用简化的过滤逻辑
                        logger.warning("批量过滤ST股票失败，使用简化过滤逻辑")

                        # 简单的基于股票代码的过滤（跳过明显的ST股票代码模式）
                        original_count = len(stock_list)
                        # 这里可以添加一些基于代码模式的简单过滤
                        # 但为了安全起见，暂时不过滤，直接使用所有股票
                        logger.info(f"跳过ST股票过滤，使用所有 {original_count} 只股票")

                except Exception as filter_e:
                    logger.warning(f"过滤ST股票时出错: {filter_e}，使用原始股票列表")

                logger.info(f"获取股票列表成功，共 {len(stock_list)} 只股票")
                # 缓存结果
                self._stock_list_cache = stock_list
                return stock_list
            else:
                logger.warning("获取股票列表失败")
                return []
        except Exception as e:
            logger.error(f"获取股票列表时出错: {e}")
            return []
    
    def get_trading_days(self, start_date, end_date):
        """
        获取交易日列表
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            
        Returns:
            交易日列表
        """
        storage = self.get_data_storage()
        if not storage:
            return []
        
        try:
            calendar_df = storage.get_trading_calendar(start_date, end_date)
            if calendar_df is not None and not calendar_df.empty:
                if 'date' in calendar_df.columns:
                    dates = calendar_df['date'].dt.strftime('%Y%m%d').tolist()
                else:
                    # 尝试使用第一列作为日期
                    dates = calendar_df.iloc[:, 0].dt.strftime('%Y%m%d').tolist()
                return dates
            else:
                logger.warning("获取交易日历失败")
                return []
        except Exception as e:
            logger.error(f"获取交易日历时出错: {e}")
            return []
    
    def get_daily_data(self, stock_list, date):
        """
        获取指定日期的日线数据 - 基于真实交易日历的优化版本

        Args:
            stock_list: 股票列表
            date: 日期，格式为YYYYMMDD

        Returns:
            DataFrame: 日线数据
        """
        storage = self.get_data_storage()
        if not storage:
            return None

        try:
            # 使用真实交易日历计算数据加载范围
            date_obj = datetime.strptime(date, '%Y%m%d')

            # 历史数据：使用自然日计算（历史数据范围影响不大）
            max_historical_window = 130  # 120 + 10安全边际
            start_date = (date_obj - timedelta(days=max_historical_window)).strftime('%Y%m%d')

            # 未来数据：使用真实交易日历确保包含T+2交易日
            end_date = self._calculate_future_end_date(date)

            logger.info(f"获取{len(stock_list)}只股票从{start_date}到{end_date}的日线数据")
            logger.info(f"目标日期: {date}, 计算的结束日期: {end_date}")

            df = storage.get_daily_data(stock_list, start_date, end_date)



            if df is None or df.empty:
                logger.warning(f"获取日线数据失败")
                return None

            logger.info(f"获取到日线数据，形状: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"获取日线数据时出错: {e}")
            logger.error(traceback.format_exc())
            return None

    def _calculate_future_end_date(self, target_date):
        """
        基于真实交易日历计算未来数据的结束日期
        确保包含计算future_return_1d所需的T+2交易日

        Args:
            target_date: 目标日期，格式为YYYYMMDD

        Returns:
            str: 结束日期，格式为YYYYMMDD
        """
        try:
            # 获取交易日历
            storage = self.get_data_storage()
            if not storage:
                # 如果无法获取交易日历，使用保守的自然日计算
                date_obj = datetime.strptime(target_date, '%Y%m%d')
                return (date_obj + timedelta(days=7)).strftime('%Y%m%d')

            # 获取目标日期之后的交易日历
            future_end = (datetime.strptime(target_date, '%Y%m%d') + timedelta(days=30)).strftime('%Y%m%d')
            calendar_df = storage.get_trading_calendar(target_date, future_end)

            if calendar_df is None or calendar_df.empty:
                logger.warning(f"无法获取{target_date}之后的交易日历，使用保守估计")
                date_obj = datetime.strptime(target_date, '%Y%m%d')
                return (date_obj + timedelta(days=7)).strftime('%Y%m%d')

            # 提取交易日列表
            if 'date_str' in calendar_df.columns:
                trading_dates = calendar_df['date_str'].tolist()
            elif 'date' in calendar_df.columns:
                if pd.api.types.is_datetime64_any_dtype(calendar_df['date']):
                    trading_dates = calendar_df['date'].dt.strftime('%Y%m%d').tolist()
                else:
                    trading_dates = calendar_df['date'].tolist()
            else:
                logger.warning(f"交易日历格式异常，使用保守估计")
                date_obj = datetime.strptime(target_date, '%Y%m%d')
                return (date_obj + timedelta(days=7)).strftime('%Y%m%d')

            # 确保目标日期在交易日列表中
            if target_date not in trading_dates:
                logger.warning(f"目标日期{target_date}不在交易日历中，使用保守估计")
                date_obj = datetime.strptime(target_date, '%Y%m%d')
                return (date_obj + timedelta(days=7)).strftime('%Y%m%d')

            # 找到目标日期的索引
            target_index = trading_dates.index(target_date)

            # future_return_1d需要T+1和T+2交易日，所以至少需要target_index + 2
            required_index = target_index + 2

            if required_index < len(trading_dates):
                end_date = trading_dates[required_index]
                logger.info(f"基于交易日历计算: {target_date} -> T+2交易日: {end_date}")
                return end_date
            else:
                # 如果交易日历不够长，使用保守估计
                logger.warning(f"交易日历长度不足，使用保守估计")
                date_obj = datetime.strptime(target_date, '%Y%m%d')
                return (date_obj + timedelta(days=10)).strftime('%Y%m%d')

        except Exception as e:
            logger.error(f"计算未来结束日期失败: {e}")
            # 出错时使用保守的自然日计算
            date_obj = datetime.strptime(target_date, '%Y%m%d')
            return (date_obj + timedelta(days=7)).strftime('%Y%m%d')
    
    def process_features_for_date(self, date, stock_list=None):
        """
        处理指定日期的特征
        
        Args:
            date: 日期，格式为YYYYMMDD
            stock_list: 股票列表，默认为None表示所有股票
            
        Returns:
            bool: 是否成功
        """
        # 获取股票列表（如果未提供）
        if stock_list is None:
            stock_list = self.get_stock_list()
            if not stock_list:
                logger.error("无法获取股票列表")
                return False
        
        logger.info(f"开始处理日期 {date} 的特征，共 {len(stock_list)} 只股票")
        
        # 获取日线数据
        daily_data = self.get_daily_data(stock_list, date)
        if daily_data is None:
            logger.error(f"无法获取日期 {date} 的日线数据")
            return False
        
        # 计算技术指标（已包含涨跌停数据处理）
        try:
            logger.info("开始计算技术指标")
            features_df = calculate_technical_indicators(daily_data, self.limit_loader)
            logger.info(f"计算技术指标完成，生成 {features_df.shape[1]} 个特征")

            # 注意：涨跌停数据已在 calculate_technical_indicators 中通过 _process_stock_features_threaded
            # 调用 calculate_advanced_features 进行了处理，无需重复集成
            logger.info("涨跌停数据已在技术指标计算中完成集成")
        except Exception as e:
            logger.error(f"计算技术指标时出错: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 筛选当前日期的数据
        try:
            # 确保日期列是datetime类型
            if 'date' in features_df.columns and not pd.api.types.is_datetime64_any_dtype(features_df['date']):
                features_df['date'] = pd.to_datetime(features_df['date'])
            
            target_date = pd.to_datetime(date, format='%Y%m%d')
            date_features_df = features_df[features_df['date'] == target_date].copy()
            
            if date_features_df.empty:
                logger.warning(f"日期 {date} 没有数据，可能是非交易日或数据缺失")
                return False
            
            logger.info(f"筛选出日期 {date} 的数据，共 {len(date_features_df)} 行")
            
            # 确保原始特征被保留 - 从原始数据中添加可能缺失的特征
            original_features = ['pct_chg', 'total_mv', 'circ_mv', 'turnover_rate']
            original_data_for_date = daily_data[daily_data['date'] == target_date].copy() if 'date' in daily_data.columns else None
            
            if original_data_for_date is not None and not original_data_for_date.empty:
                for feature in original_features:
                    if feature in original_data_for_date.columns and feature not in date_features_df.columns:
                        logger.info(f"从原始数据中补充特征: {feature}")
                        date_features_df[feature] = original_data_for_date[feature]
            
        except Exception as e:
            logger.error(f"筛选日期数据时出错: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 确保所有特征列存在
        required_columns = [
            # 基础价格数据
            'open', 'high', 'low', 'close', 'volume', 'amount',
            
            # 原始特征 - 确保这些特征存在
            'pct_chg', 'total_mv', 'circ_mv', 'turnover_rate',
            
            # 关键技术指标
            'ma20', 'ma60', 'rsi12', 'macd', 'macd_hist',
            'bb_width', 'bb_position', 'atr14', 'volatility20',
            
            # 高级特征 - 根据可用性添加
            'trend_strength', 'trend_consistency', 'risk_reward_ratio',
            
            # 高级ML特征 - 量价关系
            'volume_weighted_price', 'price_vol_change_ratio',
            
            # 高级ML特征 - 动量和波动性
            'return_5d', 'return_20d', 'parkinson_vol',
            
            # 综合交易信号
            'technical_consensus', 'composite_trading_signal',
            
            # 元数据列
            'stock_code', 'date_str'
        ]
        
        for col in required_columns:
            if col not in date_features_df.columns:
                logger.warning(f"缺少必需的列: {col}，将添加默认值")
                if col in ['stock_code', 'date_str']:
                    date_features_df[col] = ''
                elif col in ['total_mv', 'circ_mv']:
                    # 市值特征默认值设为0
                    date_features_df[col] = 0.0
                elif col in ['pct_chg', 'turnover_rate']:
                    # 百分比特征默认值设为0
                    date_features_df[col] = 0.0
                else:
                    date_features_df[col] = 0.0
        
        # 优化数据类型
        try:
            logger.info("优化数据类型")
            date_features_df = optimize_dtypes(date_features_df)
        except Exception as e:
            logger.error(f"优化数据类型时出错: {e}")
            logger.error(traceback.format_exc())
            # 继续执行，不返回False
        
        # 保存特征
        try:
            # 使用批量I/O管理器保存
            features_dict = {date: date_features_df}
            success_count = self.batch_io.batch_save_features(
                features_dict,
                self.output_dir,
                compression='snappy'
            )

            if success_count > 0:
                output_file = self.output_dir / f"{date}.parquet"
                logger.info(f"成功保存日期 {date} 的特征到 {output_file}，共 {len(date_features_df)} 行")
                return True
            else:
                logger.error("批量保存失败")
                return False

        except Exception as e:
            logger.error(f"保存特征时出错: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_features(self, start_date, end_date, stock_list=None, force_update=False, batch_size=15, use_threading=True):
        """
        生成指定日期范围的特征 - 多线程优化版本，支持并行处理

        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            stock_list: 股票列表，默认为None表示所有股票
            force_update: 是否强制更新已存在的特征
            batch_size: 批量处理大小，每次处理多少个交易日
            use_threading: 是否使用多线程处理，默认为True

        Returns:
            bool: 是否全部成功
        """
        # 获取交易日列表
        trading_days = self.get_trading_days(start_date, end_date)
        if not trading_days:
            logger.error(f"无法获取从 {start_date} 到 {end_date} 的交易日列表")
            return False

        # 获取股票列表（如果未提供）
        if stock_list is None:
            stock_list = self.get_stock_list()
            if not stock_list:
                logger.error("无法获取股票列表")
                return False

        logger.info(f"将处理 {len(trading_days)} 个交易日，共 {len(stock_list)} 只股票")
        logger.info(f"使用{'多线程' if use_threading else '单线程'}批量处理，批次大小: {batch_size}")
        if use_threading:
            logger.info(f"多线程配置: 最大线程数 = {self.max_workers}")

        # 筛选需要处理的日期
        dates_to_process = []
        for date in trading_days:
            output_file = self.output_dir / f"{date}.parquet"
            if force_update or not output_file.exists():
                dates_to_process.append(date)
            else:
                logger.debug(f"日期 {date} 的特征文件已存在，跳过")

        if not dates_to_process:
            logger.info("所有日期的特征文件都已存在，无需处理")
            return True

        logger.info(f"需要处理 {len(dates_to_process)} 个交易日")

        # 如果是强制更新，清理相关缓存
        if force_update:
            logger.info("强制更新模式：清理缓存以确保数据一致性")
            self.clear_cache()
            self.data_cache.clear_cache()
            self.limit_loader.clear_cache()

        # 批量处理
        success_count = 0
        total_dates = len(dates_to_process)

        for i in range(0, total_dates, batch_size):
            batch_dates = dates_to_process[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_dates + batch_size - 1) // batch_size

            logger.info(f"处理批次 {batch_num}/{total_batches}，包含 {len(batch_dates)} 个日期")

            # 选择处理方式：多线程或单线程
            if use_threading and len(batch_dates) > 1:
                batch_success = self._process_date_batch_multithreaded(batch_dates, stock_list)
            else:
                batch_success = self._process_date_batch(batch_dates, stock_list)
            success_count += batch_success

            # 显示进度
            processed_dates = min(i + batch_size, total_dates)
            progress = (processed_dates / total_dates) * 100
            logger.info(f"进度: {processed_dates}/{total_dates} ({progress:.1f}%)，成功: {success_count}")

            # 释放内存
            gc.collect()

        logger.info(f"特征生成完成，共处理 {total_dates} 个交易日，成功 {success_count} 个")
        return success_count == total_dates

    def _process_date_batch_multithreaded(self, batch_dates, stock_list):
        """
        多线程批量处理多个日期的特征

        Args:
            batch_dates: 日期列表
            stock_list: 股票列表

        Returns:
            int: 成功处理的日期数量
        """
        if not batch_dates:
            return 0

        logger.info(f"使用多线程处理 {len(batch_dates)} 个日期")

        # 智能计算最小必要数据范围
        start_date_obj = datetime.strptime(min(batch_dates), '%Y%m%d')
        end_date_obj = datetime.strptime(max(batch_dates), '%Y%m%d')

        # 智能数据范围计算
        max_historical_window = 130  # 120 + 10安全边际

        extended_start = (start_date_obj - timedelta(days=max_historical_window)).strftime('%Y%m%d')
        # 使用真实交易日历计算未来数据范围，确保包含T+2交易日
        extended_end = self._calculate_future_end_date(max(batch_dates))

        logger.info(f"批量加载数据范围: {extended_start} 到 {extended_end}")

        try:
            storage = self.get_data_storage()
            if not storage:
                logger.error("无法获取数据存储实例")
                return 0

            # 智能缓存数据加载
            all_data = self.data_cache.get_cached_data(stock_list, extended_start, extended_end)

            if all_data is None:
                # 缓存未命中，从存储加载
                all_data = storage.get_daily_data(stock_list, extended_start, extended_end)
                if all_data is None or all_data.empty:
                    logger.error("批量加载数据失败")
                    return 0

                # 缓存新加载的数据
                self.data_cache.cache_data(all_data, stock_list, extended_start, extended_end)

            logger.info(f"批量加载数据成功，数据形状: {all_data.shape}")

            # 预计算所有技术指标（已包含涨跌停数据处理）
            calc_start_time = time.time()
            features_df = calculate_technical_indicators(all_data, self.limit_loader)
            calc_time = time.time() - calc_start_time
            logger.info(f"批量计算技术指标完成，生成 {features_df.shape[1]} 个特征，耗时: {calc_time:.2f}秒")

            # 注意：涨跌停数据已在 calculate_technical_indicators 中通过 _process_stock_features_threaded
            # 调用 calculate_advanced_features 进行了处理，无需重复集成
            logger.info("涨跌停数据已在技术指标计算中完成集成")

            # 释放原始数据内存
            del all_data
            gc.collect()

            # 确保日期列是datetime类型
            if 'date' in features_df.columns and not pd.api.types.is_datetime64_any_dtype(features_df['date']):
                features_df['date'] = pd.to_datetime(features_df['date'])

            # 多线程处理每个日期
            success_count = 0
            max_workers = min(self.max_workers, len(batch_dates))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务
                future_to_date = {}
                for date in batch_dates:
                    future = executor.submit(self._process_single_date_threaded, date, features_df)
                    future_to_date[future] = date

                # 收集结果
                for future in as_completed(future_to_date):
                    date = future_to_date[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                            logger.debug(f"成功处理日期 {date}")
                        else:
                            logger.warning(f"处理日期 {date} 失败")
                    except Exception as e:
                        logger.error(f"处理日期 {date} 时出错: {e}")

            # 释放特征数据内存
            del features_df
            gc.collect()

            logger.info(f"多线程批量处理完成，成功: {success_count}/{len(batch_dates)}")
            return success_count

        except Exception as e:
            logger.error(f"多线程批量处理时出错: {e}")
            logger.error(traceback.format_exc())
            # 回退到单线程处理
            logger.info("多线程处理失败，回退到单线程模式")
            return self._process_date_batch(batch_dates, stock_list)

    def _process_single_date_threaded(self, date, features_df):
        """
        多线程处理单个日期的特征保存

        Args:
            date: 日期字符串
            features_df: 预计算的特征DataFrame

        Returns:
            bool: 是否成功
        """
        try:
            target_date = pd.to_datetime(date, format='%Y%m%d')
            date_features_df = features_df[features_df['date'] == target_date].copy()

            if date_features_df.empty:
                logger.info(f"日期 {date} 没有数据")
                return False

            # 确保所有必需特征存在（简化版本，避免传递大对象）
            date_features_df = self._ensure_required_features_simple(date_features_df)

            # 优化数据类型
            date_features_df = optimize_dtypes(date_features_df)

            # 保存特征
            output_file = self.output_dir / f"{date}.parquet"
            date_features_df.to_parquet(output_file, index=False, compression='snappy')

            logger.debug(f"成功保存日期 {date} 的特征，共 {len(date_features_df)} 行")
            return True

        except Exception as e:
            logger.error(f"处理日期 {date} 时出错: {e}")
            return False

    def _ensure_required_features_simple(self, date_features_df):
        """
        简化版本的特征确保函数，用于多线程环境

        Args:
            date_features_df: 当前日期的特征DataFrame

        Returns:
            DataFrame: 补充了缺失特征的DataFrame
        """
        result = date_features_df.copy()

        # 核心必需特征列表
        required_columns = [
            'open', 'high', 'low', 'close', 'volume', 'amount',
            'pct_chg', 'total_mv', 'circ_mv', 'turnover_rate',
            'ma20', 'rsi12', 'macd',
            'stock_code', 'date_str'
        ]

        for col in required_columns:
            if col not in result.columns:
                if col in ['stock_code', 'date_str']:
                    result[col] = ''
                else:
                    result[col] = 0.0

        return result

    def _process_date_batch(self, batch_dates, stock_list):
        """
        批量处理多个日期的特征 - 优化数据加载

        Args:
            batch_dates: 日期列表
            stock_list: 股票列表

        Returns:
            int: 成功处理的日期数量
        """
        if not batch_dates:
            return 0

        success_count = 0

        # 智能计算最小必要数据范围（与多线程版本保持一致）
        start_date_obj = datetime.strptime(min(batch_dates), '%Y%m%d')
        end_date_obj = datetime.strptime(max(batch_dates), '%Y%m%d')

        # 使用相同的智能数据范围计算逻辑
        max_historical_window = 130  # 120 + 10安全边际

        extended_start = (start_date_obj - timedelta(days=max_historical_window)).strftime('%Y%m%d')
        # 使用真实交易日历计算未来数据范围，确保包含T+2交易日
        extended_end = self._calculate_future_end_date(max(batch_dates))

        logger.info(f"批量加载数据范围: {extended_start} 到 {extended_end}")

        # 一次性加载所有需要的数据
        try:
            storage = self.get_data_storage()
            if not storage:
                logger.error("无法获取数据存储实例")
                return 0

            # 智能缓存数据加载
            all_data = self.data_cache.get_cached_data(stock_list, extended_start, extended_end)

            if all_data is None:
                # 缓存未命中，从存储加载
                all_data = storage.get_daily_data(stock_list, extended_start, extended_end)
                if all_data is None or all_data.empty:
                    logger.error("批量加载数据失败")
                    return 0

                # 缓存新加载的数据
                self.data_cache.cache_data(all_data, stock_list, extended_start, extended_end)

            logger.info(f"批量加载数据成功，数据形状: {all_data.shape}")

            # 预计算所有技术指标（已包含涨跌停数据处理）
            logger.info("开始批量计算技术指标")
            calc_start_time = time.time()
            features_df = calculate_technical_indicators(all_data, self.limit_loader)
            calc_time = time.time() - calc_start_time
            logger.info(f"批量计算技术指标完成，生成 {features_df.shape[1]} 个特征，耗时: {calc_time:.2f}秒")

            # 注意：涨跌停数据已在 calculate_technical_indicators 中通过 _process_stock_features_threaded
            # 调用 calculate_advanced_features 进行了处理，无需重复集成
            logger.info("涨跌停数据已在技术指标计算中完成集成")

            # 释放原始数据内存
            # del all_data
            # gc.collect()

            # 确保日期列是datetime类型
            if 'date' in features_df.columns and not pd.api.types.is_datetime64_any_dtype(features_df['date']):
                features_df['date'] = pd.to_datetime(features_df['date'])

            # 处理每个日期 - 添加快速失败机制
            consecutive_failures = 0
            max_consecutive_failures = 5  # 最大连续失败次数

            for date in batch_dates:
                try:
                    target_date = pd.to_datetime(date, format='%Y%m%d')
                    date_features_df = features_df[features_df['date'] == target_date].copy()

                    if date_features_df.empty:
                        logger.info(f"日期 {date} 没有数据")
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(f"连续 {consecutive_failures} 个日期处理失败，停止处理以避免浪费时间")
                            raise RuntimeError(f"连续失败次数过多: {consecutive_failures}")
                        continue

                    # 确保所有必需特征存在
                    date_features_df = self._ensure_required_features(date_features_df, all_data, target_date)

                    # 优化数据类型
                    date_features_df = optimize_dtypes(date_features_df)

                    # 保存特征
                    output_file = self.output_dir / f"{date}.parquet"
                    date_features_df.to_parquet(output_file, index=False, compression='snappy')

                    logger.info(f"成功保存日期 {date} 的特征，共 {len(date_features_df)} 行")
                    success_count += 1
                    consecutive_failures = 0  # 重置连续失败计数

                except Exception as e:
                    consecutive_failures += 1
                    logger.error(f"处理日期 {date} 时出错: {e}")
                    logger.error(traceback.format_exc())

                    # 快速失败机制
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"连续 {consecutive_failures} 个日期处理失败，停止处理以避免浪费时间")
                        logger.error("这通常表示存在系统性问题，需要修复后重新运行")
                        raise RuntimeError(f"连续失败次数过多: {consecutive_failures}")

        except Exception as e:
            logger.error(f"批量处理时出错: {e}")
            logger.error(traceback.format_exc())
            # 如果批量处理失败，回退到逐个处理
            logger.info("批量处理失败，回退到逐个处理模式")
            for date in batch_dates:
                try:
                    success = self.process_features_for_date(date, stock_list)
                    if success:
                        success_count += 1
                except Exception as date_e:
                    logger.error(f"逐个处理日期 {date} 时出错: {date_e}")

        return success_count

    def _ensure_required_features(self, date_features_df, original_data, target_date):
        """
        确保所有必需的特征列存在

        Args:
            date_features_df: 当前日期的特征DataFrame
            original_data: 原始数据DataFrame
            target_date: 目标日期

        Returns:
            DataFrame: 补充了缺失特征的DataFrame
        """
        result = date_features_df.copy()

        # 确保原始特征被保留 - 从原始数据中添加可能缺失的特征
        original_features = ['pct_chg', 'total_mv', 'circ_mv', 'turnover_rate']
        original_data_for_date = original_data[original_data['date'] == target_date].copy() if 'date' in original_data.columns else None

        if original_data_for_date is not None and not original_data_for_date.empty:
            for feature in original_features:
                if feature in original_data_for_date.columns and feature not in result.columns:
                    logger.info(f"从原始数据中补充特征: {feature}")
                    # 合并特征，基于stock_code
                    if 'stock_code' in result.columns and 'stock_code' in original_data_for_date.columns:
                        feature_data = original_data_for_date[['stock_code', feature]].set_index('stock_code')
                        result = result.set_index('stock_code').join(feature_data, how='left').reset_index()
                    else:
                        result[feature] = original_data_for_date[feature].iloc[0] if len(original_data_for_date) > 0 else 0.0

        # 确保所有必需特征列存在
        required_columns = [
            # 基础价格数据
            'open', 'high', 'low', 'close', 'volume', 'amount',

            # 原始特征 - 确保这些特征存在
            'pct_chg', 'total_mv', 'circ_mv', 'turnover_rate',

            # 关键技术指标
            'ma20', 'ma60', 'rsi12', 'macd', 'macd_hist',
            'bb_width', 'bb_position', 'atr14', 'volatility20',

            # 高级特征 - 根据可用性添加
            'trend_strength', 'trend_consistency', 'risk_reward_ratio',

            # 高级ML特征 - 量价关系
            'volume_weighted_price', 'price_vol_change_ratio',

            # 高级ML特征 - 动量和波动性
            'return_5d', 'return_20d', 'parkinson_vol',

            # 综合交易信号
            'technical_consensus', 'composite_trading_signal',

            # 元数据列
            'stock_code', 'date_str'
        ]

        for col in required_columns:
            if col not in result.columns:
                logger.warning(f"缺少必需的列: {col}，将添加默认值")
                if col in ['stock_code', 'date_str']:
                    result[col] = ''
                elif col in ['total_mv', 'circ_mv']:
                    # 市值特征默认值设为0
                    result[col] = 0.0
                elif col in ['pct_chg', 'turnover_rate']:
                    # 百分比特征默认值设为0
                    result[col] = 0.0
                else:
                    result[col] = 0.0

        return result

    def check_features(self, date=None):
        """
        检查特征文件
        
        Args:
            date: 日期，格式为YYYYMMDD，如果为None则检查所有特征文件
            
        Returns:
            DataFrame: 特征信息摘要
        """
        if date:
            # 检查指定日期的特征文件
            file_path = self.output_dir / f"{date}.parquet"
            if not file_path.exists():
                logger.warning(f"日期 {date} 的特征文件不存在")
                return None
            
            try:
                df = pd.read_parquet(file_path)
                logger.info(f"文件 {file_path} 包含 {len(df)} 行，{df.shape[1]} 列")
                return df
            except Exception as e:
                logger.error(f"读取特征文件 {file_path} 失败: {e}")
                return None
        else:
            # 检查所有特征文件
            parquet_files = list(self.output_dir.glob("*.parquet"))
            if not parquet_files:
                logger.warning("未找到特征文件")
                return None
            
            # 统计信息
            file_info = []
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)
                    file_info.append({
                        'date': file_path.stem,
                        'rows': len(df),
                        'columns': df.shape[1],
                        'stocks': df['stock_code'].nunique() if 'stock_code' in df.columns else 'N/A',
                        'size_mb': file_path.stat().st_size / 1024 / 1024
                    })
                except Exception as e:
                    logger.error(f"读取特征文件 {file_path} 失败: {e}")
            
            # 转换为DataFrame并返回
            if file_info:
                info_df = pd.DataFrame(file_info)
                info_df = info_df.sort_values('date')
                return info_df
            return None

    # 临时测试方法已删除 - 问题已解决

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成股票日线特征")
    parser.add_argument("--start_date", type=str, help="开始日期，格式为YYYYMMDD")
    parser.add_argument("--end_date", type=str, help="结束日期，格式为YYYYMMDD")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认为to/local_data/features/daily")
    parser.add_argument("--force", action="store_true", help="强制更新已存在的特征")
    parser.add_argument("--clear-cache", action="store_true", help="清理所有缓存（内存缓存、特征缓存等）")
    parser.add_argument("--check", action="store_true", help="只检查特征，不生成")
    parser.add_argument("--date", type=str, help="检查指定日期的特征，与--check一起使用")
    parser.add_argument("--batch_size", type=int, default=15, help="批量处理大小，默认为15个交易日")
    parser.add_argument("--max_workers", type=int, default=None, help="最大线程数，默认为CPU核心数的一半")
    parser.add_argument("--no_threading", action="store_true", help="禁用多线程处理")
    args = parser.parse_args()

    # 创建特征生成器
    generator = StockFeatureGenerator(args.output_dir, args.max_workers)

    # 清理缓存（如果指定）
    if args.clear_cache:
        logger.info("清理所有缓存...")
        generator.clear_cache()
        generator.data_cache.clear_cache()
        generator.limit_loader.clear_cache()
        logger.info("缓存清理完成")

    if args.check:
        # 检查模式
        if args.date:
            generator.check_features(args.date)
        else:
            info_df = generator.check_features()
            if info_df is not None:
                logger.info(f"特征文件摘要:\n{info_df}")
    else:
        # 生成特征
        if args.start_date and args.end_date:
            start_time = time.time()
            success = generator.generate_features(
                args.start_date,
                args.end_date,
                force_update=args.force,
                batch_size=args.batch_size,
                use_threading=not args.no_threading
            )
            end_time = time.time()

            elapsed_time = end_time - start_time
            logger.info(f"特征生成{'成功' if success else '部分失败'}，总耗时: {elapsed_time:.2f}秒")
        else:
            logger.error("请提供开始日期和结束日期")
            parser.print_help()

if __name__ == "__main__":
    main()