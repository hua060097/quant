#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据缓存管理器 - 为量化回测框架提供高性能数据访问层

该模块实现了一个多级缓存系统，用于优化量化回测过程中的数据访问，
显著减少对底层数据API的调用次数，提高回测速度。系统包含内存缓存和
磁盘缓存两级结构，支持自动数据预加载、智能缓存管理和性能监控。

主要特性:
- 透明数据访问接口，与原始API保持一致
- 多级缓存机制(内存+磁盘)
- 智能内存管理，防止内存溢出
- 并行数据预加载功能
- 缓存性能监控和统计

示例用法:
    from data_cache_manager import get_data_cache_manager
    
    # 获取缓存管理器实例
    cache_mgr = get_data_cache_manager()
    
    # 使用缓存获取市场数据
    data = cache_mgr.get_market_data(stock_list, fields, period, start_time, end_time)

Authors: AI Assistant
Date:    2023-11-15
"""

import os
import sys
import time
import json
import pickle
import logging
import hashlib
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import concurrent.futures

# 导入原始数据API
try:
    from xtquant import xtdata as xtdata
except ImportError:
    print("警告: 无法导入xtdata模块，缓存系统将使用模拟模式运行")
    xtdata = None

# 导入本地Parquet数据存储
try:
    # 先尝试直接导入
    from advanced_quant_data import ParquetDataStorage
except (ImportError, SyntaxError, IndentationError) as e:
    try:
        # 尝试从备份文件导入
        import sys
        import os
        # 添加当前目录到sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        # 创建一个简单的模拟ParquetDataStorage
        class MockParquetDataStorage:
            """模拟的ParquetDataStorage类，用于在导入失败时提供基本功能"""
            def __init__(self, base_dir="local_data"):
                self.base_dir = base_dir
                self.metadata = {'stock_list': []}
                print("警告：使用模拟的ParquetDataStorage类")
                
            def get_trading_calendar(self, start_date=None, end_date=None):
                """获取交易日历（模拟版本）"""
                import pandas as pd
                from datetime import datetime, timedelta
                # 创建一个基本的交易日历
                start = datetime.strptime(start_date or '20230101', '%Y%m%d')
                end = datetime.strptime(end_date or '20250101', '%Y%m%d')
                dates = []
                current = start
                while current <= end:
                    # 跳过周末
                    if current.weekday() < 5:  # 0-4 表示周一至周五
                        dates.append(current.strftime('%Y%m%d'))
                    current += timedelta(days=1)
                return pd.DataFrame({'date_str': dates})
                
            def get_daily_data(self, stock_list=None, start_date=None, end_date=None, fields=None):
                """获取日线数据（模拟版本）"""
                import pandas as pd
                return pd.DataFrame() # 返回空DataFrame
        
        # 使用模拟类
        ParquetDataStorage = MockParquetDataStorage
        print(f"导入 ParquetDataStorage 失败: {e}，使用模拟版本")
    except Exception as mock_err:
        print(f"创建模拟 ParquetDataStorage 失败: {mock_err}")
        raise

# 设置日志
log_dir = Path("to/log")
log_dir.mkdir(exist_ok=True, parents=True)
logger = logging.getLogger("data_cache")
logger.setLevel(logging.INFO)

# 文件处理器
file_handler = logging.FileHandler(log_dir / "data_cache.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 全局缓存管理器实例
_CACHE_MANAGER_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()
# 添加全局API调用锁
_API_CALL_LOCK = threading.Lock()
# 添加API重试计数器
_API_RETRY_COUNTS = defaultdict(int)

def get_data_cache_manager(cache_dir=None, memory_limit_mb=2000):
    """
    获取全局数据缓存管理器实例（单例模式）
    
    参数:
        cache_dir (str, optional): 缓存目录路径
        memory_limit_mb (int, optional): 内存缓存大小限制(MB)
        
    返回:
        DataCacheManager: 数据缓存管理器实例
    """
    global _CACHE_MANAGER_INSTANCE
    
    with _INSTANCE_LOCK:
        if _CACHE_MANAGER_INSTANCE is None:
            _CACHE_MANAGER_INSTANCE = DataCacheManager(
                cache_dir=cache_dir,
                memory_limit_mb=memory_limit_mb
            )
        return _CACHE_MANAGER_INSTANCE


class LRUCache:
    """
    基于LRU（最近最少使用）算法的内存缓存实现
    
    用于存储和管理内存中的缓存数据，当缓存容量超过限制时，
    自动清除最近最少使用的数据
    """
    def __init__(self, capacity_mb=1000):
        """
        初始化LRU缓存
        
        参数:
            capacity_mb (int): 缓存容量上限，单位为MB
        """
        self.capacity_bytes = capacity_mb * 1024 * 1024  # 转换为字节
        self.cache = OrderedDict()
        self.size_map = {}
        self.current_size = 0
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key):
        """
        获取缓存项
        
        参数:
            key (str): 缓存键
            
        返回:
            缓存的数据，如果不存在则返回None
        """
        with self.lock:
            if key in self.cache:
                # 移动到最近使用的位置
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hit_count += 1
                return value
            self.miss_count += 1
            return None
    
    def put(self, key, value):
        """
        添加或更新缓存项
        
        参数:
            key (str): 缓存键
            value (Any): 缓存值
            
        返回:
            bool: 是否成功添加缓存
        """
        with self.lock:
            # 如果键已存在，移除旧值和大小记录
            if key in self.cache:
                self.current_size -= self.size_map[key]
                self.cache.pop(key)
                
            # 计算新值大小
            try:
                value_size = sys.getsizeof(pickle.dumps(value))
            except:
                # 如果无法序列化，估算大小
                value_size = sys.getsizeof(str(value))
            
            # 如果单个项目大于容量的80%，拒绝缓存
            if value_size > 0.8 * self.capacity_bytes:
                logger.warning(f"缓存项过大，大小为{value_size/1024/1024:.2f}MB，超过容量的80%，不进行缓存")
                return False
                
            # 确保有足够空间
            while self.current_size + value_size > self.capacity_bytes and self.cache:
                # 移除最久未使用的项
                oldest_key, _ = self.cache.popitem(last=False)
                self.current_size -= self.size_map[oldest_key]
                del self.size_map[oldest_key]
                
            # 添加新项
            self.cache[key] = value
            self.size_map[key] = value_size
            self.current_size += value_size
            return True
    
    def contains(self, key):
        """
        检查键是否在缓存中
        
        参数:
            key (str): 缓存键
            
        返回:
            bool: 是否存在
        """
        with self.lock:
            return key in self.cache
    
    def __contains__(self, key):
        """
        实现in运算符检查键是否在缓存中
        
        参数:
            key (str): 缓存键
            
        返回:
            bool: 是否存在
        """
        return self.contains(key)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.size_map.clear()
            self.current_size = 0
    
    def get_stats(self):
        """
        获取缓存统计信息
        
        返回:
            dict: 包含命中次数、缺失次数、当前大小等信息的字典
        """
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'memory_usage_bytes': self.current_size,
                'memory_usage_mb': self.current_size / 1024 / 1024,
                'capacity_mb': self.capacity_bytes / 1024 / 1024,
                'usage_percent': (self.current_size / self.capacity_bytes * 100) if self.capacity_bytes > 0 else 0,
                'items_count': len(self.cache),
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate
            }


class DataCacheManager:
    """
    数据缓存管理器 - 管理股票数据的内存和磁盘缓存
    
    提供透明的数据访问接口，自动在多级缓存中查找数据，
    仅在必要时调用原始API获取数据。
    """
    def __init__(self, cache_dir=None, memory_limit_mb=2000):
        """
        初始化数据缓存管理器
        
        参数:
            cache_dir (str, optional): 缓存目录路径
            memory_limit_mb (int, optional): 内存缓存大小限制(MB)
        """
        # 设置缓存目录
        self.cache_dir = Path(cache_dir) if cache_dir else Path("to/data_cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化内存缓存
        self.memory_cache = LRUCache(capacity_mb=memory_limit_mb)
        
        # 初始化元数据
        self.metadata_path = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # 初始化日志
        self.logger = logger.getChild('manager')
        self.logger.info(f"数据缓存管理器初始化完成，缓存目录: {self.cache_dir}，内存限制: {memory_limit_mb}MB")
        
        # 缓存统计
        self.cache_hits = {'market_data': 0, 'trading_dates': 0, 'other': 0}
        self.api_calls = {'market_data': 0, 'trading_dates': 0, 'other': 0}
        self.cache_stats = {'hit': {'memory': 0, 'disk': 0, 'parquet': 0}, 'miss': 0}
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 记录启动时间，用于计算运行时间
        self.start_time = time.time()
        
        # 初始化ParquetDataStorage实例
        self.parquet_storage = None
        if ParquetDataStorage is not None:
            try:
                self.parquet_storage = ParquetDataStorage()
                self.logger.info("成功初始化ParquetDataStorage实例")
            except Exception as e:
                self.logger.error(f"初始化ParquetDataStorage失败: {e}")
        
    def _load_metadata(self):
        """
        加载缓存元数据
        
        返回:
            dict: 缓存元数据
        """
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载元数据失败: {e}，将创建新的元数据")
        
        # 创建新的元数据结构
        return {
            'version': '1.0',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cache_items': {},
            'stats': {
                'total_api_calls': 0,
                'total_cache_hits': 0,
                'disk_cache_size_bytes': 0
            }
        }
    
    def _save_metadata(self):
        """保存缓存元数据到磁盘"""
        with self.lock:
            self.metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                with open(self.metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"保存元数据失败: {e}")
    
    def _get_cache_key(self, cache_type, **kwargs):
        """
        根据缓存类型和参数生成缓存键
        
        参数:
            cache_type (str): 缓存类型
            **kwargs: 缓存参数
            
        返回:
            str: 缓存键
        """
        # 排序参数以确保相同参数生成相同的键
        key_parts = [cache_type]
        
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if isinstance(v, (list, tuple)):
                # 确保列表是排序的
                v = sorted(v) if all(isinstance(x, str) for x in v) else v
                key_parts.append(f"{k}={','.join(map(str, v))}")
            else:
                key_parts.append(f"{k}={v}")
                
        key_str = '|'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_disk_cache_path(self, cache_key):
        """
        获取磁盘缓存文件路径
        
        参数:
            cache_key (str): 缓存键
            
        返回:
            Path: 缓存文件路径
        """
        # 使用键的前两个字符作为子目录，分散文件
        subdir = cache_key[:2]
        cache_path = self.cache_dir / subdir
        cache_path.mkdir(exist_ok=True)
        return cache_path / f"{cache_key}.pkl"
    
    def _save_to_disk(self, cache_key, data):
        """
        保存数据到磁盘缓存
        
        参数:
            cache_key (str): 缓存键
            data: 要保存的数据
            
        返回:
            bool: 是否成功保存
        """
        try:
            disk_cache_file = self._get_disk_cache_path(cache_key)
            with open(disk_cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            self.logger.error(f"保存到磁盘缓存出错: {e}")
            return False
    
    def _load_from_disk(self, cache_key):
        """
        从磁盘缓存加载数据
        
        参数:
            cache_key (str): 缓存键
            
        返回:
            缓存的数据，如果不存在或出错则返回None
        """
        try:
            disk_cache_file = self._get_disk_cache_path(cache_key)
            if disk_cache_file.exists():
                with open(disk_cache_file, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            self.logger.error(f"从磁盘缓存加载出错: {e}")
            return None
    
    def get_market_data(self, stock_list, field_list, period, start_time, end_time):
        """
        获取市场数据，优先从缓存获取
        
        参数:
            stock_list (list): 股票代码列表
            field_list (list): 字段列表
            period (str): 周期
            start_time (str): 开始时间
            end_time (str): 结束时间
            
        返回:
            dict: 包含市场数据的字典，键为字段名，值为DataFrame
        """
        if xtdata is None and self.parquet_storage is None:
            logger.warning("xtdata模块和ParquetDataStorage未加载，无法获取市场数据")
            return None
        
        # 生成缓存键
        cache_key = self._get_cache_key(
            'market_data',
            stock_list=stock_list,
            field_list=field_list,
            period=period,
            start_time=start_time,
            end_time=end_time
        )
        
        # 检查内存缓存
        data = self.memory_cache.get(cache_key)
        if data is not None:
            self.cache_hits['market_data'] += 1
            self.cache_stats['hit']['memory'] += 1
            
            # 尝试标准化缓存中的数据
            standardized_data = self._standardize_market_data(data, field_list, stock_list)
            if standardized_data is not None and self._is_valid_market_data(standardized_data, field_list):
                if data is not standardized_data:  # 如果数据被转换了
                    self.logger.info("内存缓存数据已转换为标准格式")
                    # 更新缓存
                    self.memory_cache.put(cache_key, standardized_data)
                return standardized_data
            else:
                self.logger.warning(f"内存缓存数据结构无法转换为标准格式(键: {list(data.keys() if data else [])})，将尝试其他数据源")
        
        # 检查磁盘缓存
        data = self._load_from_disk(cache_key)
        if data is not None:
            # 尝试标准化缓存中的数据
            standardized_data = self._standardize_market_data(data, field_list, stock_list)
            if standardized_data is not None and self._is_valid_market_data(standardized_data, field_list):
                # 加入内存缓存
                self.memory_cache.put(cache_key, standardized_data)
                self.cache_hits['market_data'] += 1
                self.cache_stats['hit']['disk'] += 1
                if data is not standardized_data:  # 如果数据被转换了
                    self.logger.info("磁盘缓存数据已转换为标准格式并更新到磁盘")
                    # 更新磁盘缓存
                    self._save_to_disk(cache_key, standardized_data)
                return standardized_data
            else:
                self.logger.warning(f"磁盘缓存数据结构无法转换为标准格式，将尝试Parquet数据源")
        
        # 检查本地Parquet数据源
        if self.parquet_storage is not None:
            try:
                self.logger.info(f"尝试从Parquet数据源获取数据: {len(stock_list)}支股票, {period}周期, {start_time}至{end_time}")
                
                # 转换周期格式
                parquet_period = period
                if period == '1d':
                    parquet_data = self.parquet_storage.get_market_data(
                        field_list=field_list,
                        stock_list=stock_list,
                        period=parquet_period,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if parquet_data and self._is_valid_market_data(parquet_data, field_list):
                        self.logger.info(f"成功从Parquet数据源获取数据")
                        self.cache_stats['hit']['parquet'] += 1
                        
                        # 更新内存和磁盘缓存
                        self.memory_cache.put(cache_key, parquet_data)
                        self._save_to_disk(cache_key, parquet_data)
                        
                        return parquet_data
                    else:
                        self.logger.warning("从Parquet数据源获取的数据为空或格式无效，将从API获取")
                elif period in ['5m', '5min']:
                    parquet_data = self.parquet_storage.get_market_data(
                        field_list=field_list,
                        stock_list=stock_list,
                        period='5m',
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if parquet_data and self._is_valid_market_data(parquet_data, field_list):
                        self.logger.info(f"成功从Parquet数据源获取数据")
                        self.cache_stats['hit']['parquet'] += 1
                        
                        # 更新内存和磁盘缓存
                        self.memory_cache.put(cache_key, parquet_data)
                        self._save_to_disk(cache_key, parquet_data)
                        
                        return parquet_data
                    else:
                        self.logger.warning("从Parquet数据源获取的数据为空或格式无效，将从API获取")
                else:
                    self.logger.warning(f"Parquet数据源不支持{period}周期，将从API获取")
            except Exception as e:
                self.logger.error(f"从Parquet数据源获取数据失败: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # 缓存未命中，调用API
        if xtdata is None:
            self.logger.error("xtdata模块未加载，无法从API获取数据")
            return None
        
        logger.info(f"所有缓存均未命中，从API获取市场数据: {len(stock_list)}支股票 {start_time}至{end_time}")
        self.cache_stats['miss'] += 1
        
        # 使用全局锁保护API调用，防止多线程并发访问API
        with _API_CALL_LOCK:
            try:
                self.api_calls['market_data'] += 1
                logger.info("获取API锁，开始API调用")
                
                # 首先检查是否有其他线程已经加载了相同数据并放入缓存
                data = self.memory_cache.get(cache_key)
                if data is not None:
                    logger.info("在等待锁期间，其他线程已加载数据到缓存")
                    standardized_data = self._standardize_market_data(data, field_list, stock_list)
                    if standardized_data is not None and self._is_valid_market_data(standardized_data, field_list):
                        return standardized_data
                
                # 确保连接
                if not xtdata.get_client().is_connected():
                    xtdata.get_client().connect()
                    # 添加连接后的短暂延迟
                    time.sleep(0.5)
                
                # 设置API调用最大重试次数
                max_retries = 3
                retry_count = 0
                original_data = None
                
                while retry_count < max_retries and original_data is None:
                    # 在重试时添加递增延迟
                    if retry_count > 0:
                        retry_delay = 1 * (2 ** (retry_count - 1))  # 指数退避: 1s, 2s, 4s ...
                        logger.info(f"第 {retry_count} 次重试，等待 {retry_delay} 秒...")
                        time.sleep(retry_delay)
                    
                    # 调用API获取数据
                    logger.info(f"调用API获取数据 (尝试 {retry_count+1}/{max_retries})")
                    original_data = xtdata.get_market_data(
                        stock_list=stock_list,
                        field_list=field_list,
                        period=period,
                        start_time=start_time,
                        end_time=end_time)
                    
                    # 如果数据为None，增加重试计数
                    if original_data is None:
                        retry_count += 1
                        logger.warning(f"API返回None，将进行第 {retry_count} 次重试...")
                    else:
                        logger.info(f"API成功返回数据")
                
                # 记录原始数据结构
                if original_data is None:
                    logger.error(f"多次重试后API仍返回None，操作失败")
                    return None
                    
                # 记录原始数据结构
                self.logger.info(f"API返回原始数据键: {list(original_data.keys()) if original_data else '无'}")
                self.logger.info(f"API返回原始数据类型: {type(original_data)}")
                
                # 标准化数据结构
                standardized_data = self._standardize_market_data(original_data, field_list, stock_list)
                
                # 检查处理后的数据
                if standardized_data is not None and self._is_valid_market_data(standardized_data, field_list):
                    self.logger.info(f"API数据已成功转换为标准格式: {list(standardized_data.keys())[:3]}...")
                    # 缓存标准化后的数据
                    self.memory_cache.put(cache_key, standardized_data)
                    self._save_to_disk(cache_key, standardized_data)
                    
                    # 同时将数据保存到Parquet数据源，如果可能的话
                    if self.parquet_storage is not None and period in ['1d', '5m', '5min']:
                        try:
                            success = False
                            if period == '1d':
                                # TODO: 将数据保存到Parquet日线数据库
                                self.logger.info("API数据获取后，已存入内存和磁盘缓存")
                            elif period in ['5m', '5min']:
                                # TODO: 将数据保存到Parquet分钟数据库
                                self.logger.info("API数据获取后，已存入内存和磁盘缓存")
                        except Exception as e:
                            self.logger.error(f"保存数据到Parquet数据源失败: {e}")
                    
                    return standardized_data
                else:
                    self.logger.error(f"无法将API返回的数据转换为标准格式，返回原始数据")
                    return original_data
            except Exception as e:
                logger.error(f"获取市场数据失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            finally:
                logger.info("释放API锁")
    
    def _is_valid_market_data(self, data, expected_fields=None):
        """
        验证市场数据结构是否符合预期
        
        参数:
            data: 要验证的数据
            expected_fields: 预期的字段列表
            
        返回:
            bool: 是否符合预期
        """
        if data is None or not isinstance(data, dict) or len(data) == 0:
            return False
            
        # 检查键是否为字段名而非股票代码
        keys = list(data.keys())
        if not keys:
            return False
            
        # 如果提供了预期字段，检查是否至少包含一个预期字段（更宽松的检查）
        if expected_fields and not any(field in keys for field in expected_fields):
            return False
            
        # 标准格式的键应该是字段名，如'open', 'close'等，不含点号
        # 放宽检查：至少大部分键不应包含点号
        dot_keys = [key for key in keys if '.' in str(key)]
        if len(dot_keys) > len(keys) / 2:  # 如果超过一半的键包含点号，判定无效
            return False
            
        # 检查值是否为DataFrame或能转换为DataFrame
        for field, value in list(data.items()):
            if '.' in str(field):  # 跳过股票代码键
                continue
            if not isinstance(value, pd.DataFrame) and not isinstance(value, dict):
                return False
                
        return True
        
    def _standardize_market_data(self, data, field_list, stock_list):
        """
        将市场数据标准化为统一格式
        
        参数:
            data: 原始数据
            field_list: 字段列表
            stock_list: 股票列表
            
        返回:
            dict: 标准化后的数据，键为字段名，值为DataFrame
        """
        if data is None:
            self.logger.warning("无法标准化空数据")
            return None
            
        # 如果数据已经是标准格式，直接返回
        if self._is_valid_market_data(data, field_list) and all(field in data for field in field_list):
            # 确保所有值都是DataFrame
            for field in field_list:
                if field in data and not isinstance(data[field], pd.DataFrame):
                    try:
                        data[field] = pd.DataFrame(data[field])
                    except:
                        continue
            return data
            
        try:
            self.logger.info(f"开始标准化数据结构，原始键数量: {len(data.keys()) if data else 0}")
            
            # 检测数据结构类型
            keys = list(data.keys())
            if not keys:
                self.logger.warning("数据没有键")
                return None
                
            # 根据键的类型推断数据结构
            first_key = keys[0]
            
            # 情况1: 键是股票代码（包含点号）
            if '.' in str(first_key):
                self.logger.info(f"检测到股票代码为键的数据结构, 示例键: {first_key}")
                standardized_data = {}
                
                # 收集所有可用字段
                all_fields = set()
                for stock_data in data.values():
                    if isinstance(stock_data, dict):
                        all_fields.update(stock_data.keys())
                
                # 使用请求的字段或所有可用字段
                fields_to_use = field_list if field_list else list(all_fields)
                
                for field in fields_to_use:
                    if field not in standardized_data:
                        standardized_data[field] = {}
                        
                    for stock_code in data.keys():
                        stock_data = data[stock_code]
                        
                        # 跳过不在请求股票列表中的股票
                        if stock_list and stock_code not in stock_list:
                            continue
                            
                        # 提取字段数据
                        if isinstance(stock_data, dict) and field in stock_data:
                            standardized_data[field][stock_code] = stock_data[field]
                
                # 转换为DataFrame
                for field in standardized_data:
                    if standardized_data[field]:  # 确保有数据
                        # 创建DataFrame时，不要调整索引，保持股票代码为行索引
                        standardized_data[field] = pd.DataFrame.from_dict(standardized_data[field], orient='index')
                
                return standardized_data
                
            # 情况2: 键是字段名但内部值不是DataFrame
            elif all(field in keys for field in field_list):
                self.logger.info(f"检测到字段为键的数据结构, 示例键: {first_key}")
                # 转换各字段的值为DataFrame
                for field in field_list:
                    if field in data and not isinstance(data[field], pd.DataFrame):
                        try:
                            # 如果值是字典，使用from_dict方法
                            if isinstance(data[field], dict):
                                data[field] = pd.DataFrame.from_dict(data[field], orient='index')
                            else:
                                # 其他类型尝试直接转换
                                data[field] = pd.DataFrame(data[field])
                        except Exception as e:
                            self.logger.warning(f"转换字段 {field} 为DataFrame失败: {e}")
                
                return data
                
            # 情况3: 其他未知结构
            else:
                self.logger.warning(f"未知的数据结构，不进行转换")
                return None
                
        except Exception as e:
            self.logger.error(f"标准化数据结构失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _update_cache_stats(self, result_type, cache_type=None):
        """
        更新缓存统计信息
        
        参数:
            result_type (str): 结果类型，'hit'或'miss'
            cache_type (str, optional): 缓存类型，'memory'或'disk'
        """
        # 防止递归
        thread_local = threading.local()
        if getattr(thread_local, 'in_update_stats', False):
            return
        
        thread_local.in_update_stats = True
        
        try:
            if result_type == 'hit':
                if cache_type:
                    self.cache_stats['hit'][cache_type] += 1
            elif result_type == 'miss':
                self.cache_stats['miss'] += 1
        finally:
            thread_local.in_update_stats = False
    
    def get_trading_dates(self, market='SH', start_time=None, end_time=None):
        """
        获取交易日期，优先从缓存获取
        
        参数:
            market (str): 市场代码
            start_time (str, optional): 开始时间
            end_time (str, optional): 结束时间
            
        返回:
            list: 交易日期列表
        """
        if xtdata is None and self.parquet_storage is None:
            self.logger.warning("xtdata模块和ParquetDataStorage未加载，无法获取交易日期")
            return []
        
        # 防止递归调用
        thread_local = threading.local()
        if getattr(thread_local, 'in_get_trading_dates', False):
            self.logger.warning("检测到get_trading_dates递归调用，直接调用原始API")
            try:
                # 规范化参数处理
                api_start_time = start_time if start_time else ''
                api_end_time = end_time if end_time else ''
                
                # 直接使用API
                trading_dates = xtdata.get_trading_dates(
                    market=market,
                    start_time=api_start_time,
                    end_time=api_end_time
                )
                return trading_dates
            except Exception as e:
                self.logger.error(f"递归调用中使用API获取交易日期失败: {e}")
                return []
        
        # 设置递归标记
        thread_local.in_get_trading_dates = True
        
        try:
            # 规范化参数
            market = market or 'SH'  # 默认使用上证市场
            
            # 记录调用信息
            self.logger.debug(f"get_trading_dates调用: market={market}, start_time={start_time}, end_time={end_time}")
            
            # 生成缓存键 - 确保使用关键字参数
            try:
                cache_key = self._get_cache_key(
                    cache_type='trading_dates',
                    market=market,
                    start_time=start_time,
                    end_time=end_time
                )
                self.logger.debug(f"生成缓存键: {cache_key}")
            except Exception as e:
                self.logger.error(f"生成缓存键时发生错误: {e}")
                # 遇到错误时返回空列表
                return []
            
            # 检查内存缓存
            try:
                if cache_key in self.memory_cache:
                    cached_dates = self.memory_cache.get(cache_key)
                    if cached_dates and len(cached_dates) > 0:
                        self._update_cache_stats('hit', 'memory')
                        self.logger.debug(f"从内存缓存获取到 {len(cached_dates)} 个交易日期")
                        return cached_dates
                    else:
                        self.logger.warning("内存缓存中的交易日期为空")
            except Exception as e:
                self.logger.error(f"检查内存缓存时出错: {e}")
                # 继续尝试下一级缓存或API
            
            # 检查磁盘缓存
            try:
                data = self._load_from_disk(cache_key)
                if data is not None and len(data) > 0:
                    self.logger.debug(f"交易日期命中磁盘缓存，获取到 {len(data)} 个日期")
                    # 加入内存缓存
                    self.memory_cache.put(cache_key, data)
                    self._update_cache_stats('hit', 'disk')
                    return data
                elif data is not None:
                    self.logger.warning("磁盘缓存中的交易日期为空")
            except Exception as e:
                self.logger.error(f"从磁盘加载缓存时出错: {e}")
                # 继续尝试API
            
            # 检查ParquetDataStorage数据源中的交易日历
            if self.parquet_storage is not None:
                try:
                    parquet_calendar = self.parquet_storage.get_trading_calendar(start_time, end_time)
                    if not parquet_calendar.empty:
                        trading_dates = parquet_calendar['date_str'].tolist()
                        if trading_dates and len(trading_dates) > 0:
                            self.logger.info(f"从Parquet数据源获取到交易日期: {len(trading_dates)}个")
                            
                            # 将结果缓存到内存和磁盘
                            self.memory_cache.put(cache_key, trading_dates)
                            self._save_to_disk(cache_key, trading_dates)
                            
                            return trading_dates
                except Exception as e:
                    self.logger.error(f"从Parquet数据源获取交易日期失败: {e}")
            
            # 缓存未命中，调用API
            try:
                # 有可能start_time和end_time为None，需要适当处理
                api_start_time = start_time if start_time else ''
                api_end_time = end_time if end_time else ''
                
                self.logger.info(f"缓存未命中，从API获取交易日期: {market} {api_start_time}至{api_end_time}")
                # 记录缓存未命中
                try:
                    self._update_cache_stats('miss')
                except Exception as e:
                    self.logger.error(f"更新缓存统计时出错: {e}")
                
                # 调用原始API
                self.api_calls['trading_dates'] += 1
                
                try:
                    # 尝试确保连接
                    xtdata.connect()
                    
                    # 调用API获取数据
                    trading_dates = xtdata.get_trading_dates(
                        market=market,
                        start_time=api_start_time,
                        end_time=api_end_time
                    )
                    
                    # 检查API返回结果
                    if trading_dates and len(trading_dates) > 0:
                        # 统一处理时间戳格式
                        if isinstance(trading_dates[0], int):
                            # 检查时间戳是否为毫秒级别
                            timestamp = trading_dates[0]
                            if timestamp > 10000000000:  # 大于10000000000的应该是毫秒级时间戳
                                from datetime import datetime
                                trading_dates = [datetime.fromtimestamp(ts/1000).strftime('%Y%m%d') for ts in trading_dates]
                        
                        # 存入缓存
                        self.memory_cache.put(cache_key, trading_dates)
                        self._save_to_disk(cache_key, trading_dates)
                        
                        self.logger.info(f"成功从API获取 {len(trading_dates)} 个交易日期，并已保存到缓存")
                        return trading_dates
                    else:
                        self.logger.warning("API返回空数据")
                        return []
                except Exception as e:
                    self.logger.error(f"调用API获取交易日期时出错: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    return []
            except Exception as e:
                self.logger.error(f"从API获取交易日期失败: {e}")
                # 打印堆栈跟踪
                import traceback
                self.logger.error(traceback.format_exc())
                return []
        finally:
            # 无论成功与否，都要清除递归标记
            thread_local.in_get_trading_dates = False
    
    def get_stock_price(self, stock_code, trade_date, field):
        """
        获取单个股票价格，优先从缓存获取
        
        参数:
            stock_code (str): 股票代码
            trade_date (str): 交易日期
            field (str): 字段名
            
        返回:
            float: 股票价格
        """
        # 调用get_market_data获取单日数据
        data = self.get_market_data(
            stock_list=[stock_code],
            field_list=[field],
            period='1d',
            start_time=trade_date,
            end_time=trade_date
        )
        
        if data is None or data.empty:
            return None
        
        try:
            return data.loc[(trade_date, stock_code), field]
        except:
            return None
    
    def preload_data(self, stock_list, start_date, end_date, fields, periods=['1d']):
        """
        预加载数据到缓存中
        
        参数:
            stock_list (list): 股票代码列表
            start_date (str): 开始日期
            end_date (str): 结束日期
            fields (list): 字段列表
            periods (list): 周期列表
            
        返回:
            dict: 预加载统计信息
        """
        logger.info(f"开始预加载数据: {len(stock_list)}支股票, {start_date}至{end_date}")
        start_time = time.time()
        
        stats = {
            'total_stocks': len(stock_list),
            'loaded_stocks': 0,
            'total_periods': len(periods),
            'errors': 0
        }
        
        for period in periods:
            for stock in stock_list:
                try:
                    self.get_market_data(
                        stock_list=[stock],
                        field_list=fields,
                        period=period,
                        start_time=start_date,
                        end_time=end_date
                    )
                    stats['loaded_stocks'] += 1
                except Exception as e:
                    logger.error(f"预加载数据错误 {stock}: {e}")
                    stats['errors'] += 1
        
        # 获取交易日期
        try:
            self.get_trading_dates('SH')
        except Exception as e:
            logger.error(f"预加载交易日期错误: {e}")
            stats['errors'] += 1
        
        elapsed = time.time() - start_time
        stats['elapsed_seconds'] = elapsed
        logger.info(f"数据预加载完成, 耗时: {elapsed:.2f}秒")
        
        # 保存元数据
        self._save_metadata()
        
        return stats
    
    def parallel_preload(self, stock_list, start_date, end_date, fields, 
                         periods=['1d'], batch_size=50, max_workers=None):
        """
        并行预加载数据到缓存中
        
        参数:
            stock_list (list): 股票代码列表
            start_date (str): 开始日期
            end_date (str): 结束日期
            fields (list): 字段列表
            periods (list): 周期列表
            batch_size (int): 每批处理的股票数量
            max_workers (int, optional): 最大工作线程数
            
        返回:
            dict: 预加载统计信息
        """
        logger.info(f"开始并行预加载数据: {len(stock_list)}支股票, {start_date}至{end_date}")
        start_time = time.time()
        
        stats = {
            'total_stocks': len(stock_list),
            'loaded_stocks': 0,
            'errors': 0,
            'batches': 0
        }
        
        # 获取交易日期先行缓存
        try:
            self.get_trading_dates('SH')
        except Exception as e:
            logger.error(f"预加载交易日期错误: {e}")
            stats['errors'] += 1
            
        # 将股票列表分成批次
        batches = [stock_list[i:i + batch_size] for i in range(0, len(stock_list), batch_size)]
        stats['batches'] = len(batches)
        
        def _load_batch(stock_batch, period):
            """批量加载数据的工作函数"""
            success = 0
            errors = 0
            
            for stock in stock_batch:
                try:
                    self.get_market_data(
                        stock_list=[stock],
                        field_list=fields,
                        period=period,
                        start_time=start_date,
                        end_time=end_date
                    )
                    success += 1
                except Exception as e:
                    logger.error(f"预加载数据错误 {stock}: {e}")
                    errors += 1
            
            return {'success': success, 'errors': errors}
        
        # 并行执行每个周期每个批次
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for period in periods:
                for batch in batches:
                    futures.append(executor.submit(_load_batch, batch, period))
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                stats['loaded_stocks'] += result['success']
                stats['errors'] += result['errors']
        
        elapsed = time.time() - start_time
        stats['elapsed_seconds'] = elapsed
        logger.info(f"并行数据预加载完成, 耗时: {elapsed:.2f}秒, "
                   f"成功: {stats['loaded_stocks']}, 错误: {stats['errors']}")
        
        # 保存元数据
        self._save_metadata()
        
        return stats
    
    def clear_cache(self, older_than_days=None):
        """
        清理缓存数据
        
        参数:
            older_than_days (int, optional): 清理指定天数前的缓存
            
        返回:
            dict: 清理统计信息
        """
        logger.info("开始清理缓存")
        stats = {
            'memory_items_cleared': 0,
            'disk_items_cleared': 0,
            'disk_space_freed_mb': 0
        }
        
        # 清理内存缓存
        self.memory_cache.clear()
        stats['memory_items_cleared'] = 1  # 表示已清理
        
        # 清理磁盘缓存
        if older_than_days is not None:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
            
            keys_to_remove = []
            for key, item in self.metadata['cache_items'].items():
                if item['last_accessed'] < cutoff_str:
                    keys_to_remove.append(key)
                    
                    # 删除文件
                    try:
                        file_path = Path(item['path'])
                        if file_path.exists():
                            file_size_mb = os.path.getsize(file_path) / 1024 / 1024
                            stats['disk_space_freed_mb'] += file_size_mb
                            os.remove(file_path)
                    except Exception as e:
                        logger.error(f"删除缓存文件失败: {e}")
            
            # 更新元数据
            with self.lock:
                for key in keys_to_remove:
                    if key in self.metadata['cache_items']:
                        del self.metadata['cache_items'][key]
                stats['disk_items_cleared'] = len(keys_to_remove)
        else:
            # 清理所有磁盘缓存
            with self.lock:
                for key, item in self.metadata['cache_items'].items():
                    try:
                        file_path = Path(item['path'])
                        if file_path.exists():
                            file_size_mb = os.path.getsize(file_path) / 1024 / 1024
                            stats['disk_space_freed_mb'] += file_size_mb
                            os.remove(file_path)
                    except Exception as e:
                        logger.error(f"删除缓存文件失败: {e}")
                
                stats['disk_items_cleared'] = len(self.metadata['cache_items'])
                self.metadata['cache_items'] = {}
                self.metadata['stats']['disk_cache_size_bytes'] = 0
        
        # 保存元数据
        self._save_metadata()
        
        logger.info(f"缓存清理完成，释放磁盘空间: {stats['disk_space_freed_mb']:.2f}MB")
        return stats
    
    def get_stats(self):
        """
        获取缓存性能统计信息
        
        返回:
            dict: 性能统计信息
        """
        memory_stats = self.memory_cache.get_stats()
        
        # 计算API调用总数和缓存命中总数
        total_api_calls = sum(self.api_calls.values())
        total_cache_hits = sum(self.cache_hits.values())
        total_requests = total_api_calls + total_cache_hits
        
        hit_rate = (total_cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        # 磁盘缓存统计
        disk_items = len(self.metadata['cache_items'])
        disk_size_mb = self.metadata['stats']['disk_cache_size_bytes'] / 1024 / 1024
        
        # 运行时间
        run_time_seconds = time.time() - self.start_time
        run_time_hours = run_time_seconds / 3600
        
        return {
            'memory_usage_mb': memory_stats['memory_usage_mb'],
            'memory_items': memory_stats['items_count'],
            'disk_items': disk_items,
            'disk_usage_mb': disk_size_mb,
            'api_calls': dict(self.api_calls),
            'cache_hits': dict(self.cache_hits),
            'total_api_calls': total_api_calls,
            'total_cache_hits': total_cache_hits,
            'hit_rate': hit_rate,
            'running_time_seconds': run_time_seconds,
            'running_time_hours': run_time_hours,
            'cache_stats': dict(self.cache_stats)
        }
    
    def generate_performance_report(self):
        """
        生成缓存性能报告
        
        返回:
            str: 性能报告文本
        """
        stats = self.get_stats()
        
        report = [
            "=" * 60,
            "数据缓存系统性能报告",
            "=" * 60,
            f"运行时间: {stats['running_time_hours']:.2f}小时",
            f"缓存命中率: {stats['hit_rate']:.2f}%",
            f"API调用总次数: {stats['total_api_calls']}",
            f"缓存命中总次数: {stats['total_cache_hits']}",
            "-" * 60,
            "内存缓存统计:",
            f"  使用内存: {stats['memory_usage_mb']:.2f}MB",
            f"  缓存项数: {stats['memory_items']}",
            "-" * 60,
            "磁盘缓存统计:",
            f"  使用空间: {stats['disk_usage_mb']:.2f}MB",
            f"  缓存项数: {stats['disk_items']}",
            "-" * 60,
            "数据源命中率:",
            f"  内存缓存: {self.cache_stats['hit'].get('memory', 0)}次",
            f"  磁盘缓存: {self.cache_stats['hit'].get('disk', 0)}次",
            f"  Parquet存储: {self.cache_stats['hit'].get('parquet', 0)}次",
            f"  API调用: {self.cache_stats.get('miss', 0)}次",
            "-" * 60,
            "API调用明细:",
        ]
        
        for api, count in stats['api_calls'].items():
            report.append(f"  {api}: {count}次")
        
        report.append("-" * 60)
        report.append("缓存命中明细:")
        
        for api, count in stats['cache_hits'].items():
            report.append(f"  {api}: {count}次")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        logger.info("\n" + report_text)
        
        # 保存报告到文件
        try:
            report_path = log_dir / f"cache_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"性能报告已保存到: {report_path}")
        except Exception as e:
            logger.error(f"保存性能报告失败: {e}")
        
        return report_text

    def get_stock_pool(self):
        """从缓存获取股票池
        
        返回:
            list: 股票池列表，如果缓存不存在返回None
        """
        # 构建缓存键
        cache_key = self._get_cache_key(cache_type="stock_pool")
        
        # 先检查内存缓存
        if cache_key in self.memory_cache:
            self._update_cache_stats('hit', 'memory')
            return self.memory_cache.get(cache_key)
        
        # 再检查磁盘缓存
        disk_cache_file = self._get_disk_cache_path(cache_key)
        if disk_cache_file.exists():
            try:
                with open(disk_cache_file, 'rb') as f:
                    stock_pool = pickle.load(f)
                    # 加入内存缓存
                    self.memory_cache.put(cache_key, stock_pool)
                    self._update_cache_stats('hit', 'disk')
                    return stock_pool
            except Exception as e:
                self.logger.error(f"读取股票池缓存出错: {e}")
        
        self._update_cache_stats('miss', None)
        return None

    def save_stock_pool(self, stock_pool):
        """保存股票池到缓存
        
        参数:
            stock_pool: 股票池列表
        
        返回:
            bool: 成功返回True，否则返回False
        """
        if not stock_pool:
            return False
        
        # 构建缓存键
        cache_key = self._get_cache_key(cache_type="stock_pool")
        
        try:
            # 保存到内存缓存
            self.memory_cache.put(cache_key, stock_pool)
            
            # 保存到磁盘缓存
            disk_cache_file = self._get_disk_cache_path(cache_key)
            with open(disk_cache_file, 'wb') as f:
                pickle.dump(stock_pool, f)
            
            self.logger.debug(f"股票池已保存到缓存，股票数量: {len(stock_pool)}")
            return True
        except Exception as e:
            self.logger.error(f"保存股票池到缓存出错: {e}")
            return False

    def get_sector_stocks(self, sector_name):
        """从缓存获取行业股票列表
        
        参数:
            sector_name: 行业或指数名称
        
        返回:
            list: 股票代码列表，如果缓存不存在返回None
        """
        if not sector_name:
            return None
        
        # 构建缓存键
        cache_key = self._get_cache_key(cache_type="sector_stocks", sector_name=sector_name)
        
        # 先检查内存缓存
        if cache_key in self.memory_cache:
            self._update_cache_stats('hit', 'memory')
            return self.memory_cache.get(cache_key)
        
        # 再检查磁盘缓存
        disk_cache_file = self._get_disk_cache_path(cache_key)
        if disk_cache_file.exists():
            try:
                with open(disk_cache_file, 'rb') as f:
                    sector_stocks = pickle.load(f)
                    # 加入内存缓存
                    self.memory_cache.put(cache_key, sector_stocks)
                    self._update_cache_stats('hit', 'disk')
                    return sector_stocks
            except Exception as e:
                self.logger.error(f"读取行业股票缓存出错: {e}")
        
        self._update_cache_stats('miss', None)
        return None

    def save_sector_stocks(self, sector_name, stock_list):
        """保存行业股票列表到缓存
        
        参数:
            sector_name: 行业或指数名称
            stock_list: 股票代码列表
        
        返回:
            bool: 成功返回True，否则返回False
        """
        if not sector_name or not stock_list:
            return False
        
        # 构建缓存键
        cache_key = self._get_cache_key(cache_type="sector_stocks", sector_name=sector_name)
        
        try:
            # 保存到内存缓存
            self.memory_cache.put(cache_key, stock_list)
            
            # 保存到磁盘缓存
            disk_cache_file = self._get_disk_cache_path(cache_key)
            with open(disk_cache_file, 'wb') as f:
                pickle.dump(stock_list, f)
            
            self.logger.debug(f"行业({sector_name})股票列表已保存到缓存，股票数量: {len(stock_list)}")
            return True
        except Exception as e:
            self.logger.error(f"保存行业股票列表到缓存出错: {e}")
            return False

    def save_trading_dates(self, market, trading_dates):
        """保存交易日期到缓存
        
        参数:
            market: 市场代码
            trading_dates: 交易日期列表
        
        返回:
            bool: 成功返回True，否则返回False
        """
        if not trading_dates:
            return False
        
        # 构建缓存键
        cache_key = self._get_cache_key(cache_type="trading_dates", market=market)
        
        try:
            # 保存到内存缓存
            self.memory_cache.put(cache_key, trading_dates)
            
            # 保存到磁盘缓存
            disk_cache_file = self._get_disk_cache_path(cache_key)
            with open(disk_cache_file, 'wb') as f:
                pickle.dump(trading_dates, f)
            
            self.logger.debug(f"交易日期已保存到缓存，日期数量: {len(trading_dates)}")
            return True
        except Exception as e:
            self.logger.error(f"保存交易日期到缓存出错: {e}")
            return False

    def clear_market_data_cache(self):
        """
        清理市场数据缓存，删除格式不符合要求的缓存
        
        返回:
            dict: 清理统计信息
        """
        stats = {'memory_items_cleared': 0, 'disk_items_cleared': 0}
        
        # 清理内存缓存中的市场数据
        keys_to_remove = []
        for key in list(self.memory_cache.cache.keys()):
            if 'market_data' in key:
                data = self.memory_cache.get(key)
                if data and not self._is_valid_market_data(data):
                    keys_to_remove.append(key)
        
        # 从内存缓存中移除无效数据
        for key in keys_to_remove:
            self.memory_cache.cache.pop(key, None)
        stats['memory_items_cleared'] = len(keys_to_remove)
        
        # 清理磁盘缓存中的市场数据
        disk_keys_to_remove = []
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for cache_file in subdir.glob("*.pkl"):
                    if 'market_data' in cache_file.name:
                        try:
                            with open(cache_file, 'rb') as f:
                                data = pickle.load(f)
                                if data and not self._is_valid_market_data(data):
                                    disk_keys_to_remove.append(cache_file)
                        except:
                            # 无法加载的缓存文件也应该删除
                            disk_keys_to_remove.append(cache_file)
        
        # 删除磁盘缓存文件
        for file_path in disk_keys_to_remove:
            try:
                file_path.unlink()
            except:
                pass
        stats['disk_items_cleared'] = len(disk_keys_to_remove)
        
        self.logger.info(f"清理了 {stats['memory_items_cleared']} 项内存缓存和 {stats['disk_items_cleared']} 项磁盘缓存")
        return stats


if __name__ == "__main__":
    # 测试代码，实际使用时应当导入此模块
    print("数据缓存管理器模块，请导入此模块使用。")
    
    # 基本测试
    cache_manager = get_data_cache_manager()
    print(f"缓存管理器初始化成功，缓存目录: {cache_manager.cache_dir}")
    
    # 测试获取交易日期
    if xtdata:
        dates = cache_manager.get_trading_dates('SH')
        print(f"获取到交易日期: {len(dates)}个")
    else:
        print("xtdata模块未加载，无法进行测试。") 