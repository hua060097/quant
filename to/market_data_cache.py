#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场数据缓存模块
用于优化数据获取过程，减少对xtdata API的调用频率，提高回测和模型训练性能

主要功能：
1. 多级缓存：内存缓存 + 磁盘缓存
2. 批量数据预加载
3. 智能缓存管理：LRU淘汰策略
4. 内存映射支持大数据集
"""

import os
import numpy as np
import pandas as pd
import hashlib
import pickle
import time
import logging
import warnings
import sys
from pathlib import Path
from functools import lru_cache
import mmap
import h5py
from concurrent.futures import ThreadPoolExecutor

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/data_cache.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("market_data_cache")

class MarketDataCache:
    """市场数据多级缓存系统，优化xtdata API调用频率"""
    
    def __init__(self, cache_dir="./data_cache", memory_limit_gb=4, 
                 hot_cache_days=30, enable_disk_cache=True):
        """
        初始化市场数据缓存系统
        
        参数:
            cache_dir: 缓存目录
            memory_limit_gb: 内存缓存限制(GB)
            hot_cache_days: 热数据缓存天数(保留在内存)
            enable_disk_cache: 是否启用磁盘缓存
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # 缓存配置
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024  # 转换为字节
        self.hot_cache_days = hot_cache_days
        self.enable_disk_cache = enable_disk_cache
        
        # 内存缓存和元数据
        self.memory_cache = {}  # {cache_key: dataframe}
        self.memory_usage = 0   # 当前内存使用量(字节)
        self.access_count = {}  # {cache_key: 访问次数}
        self.last_access = {}   # {cache_key: 最后访问时间}
        
        # 缓存元数据
        self.metadata = self._load_metadata()
        
        # 批处理配置
        self.batch_data = {}    # 用于批量请求结果的临时存储
        self.pending_requests = {}  # 等待批处理的请求
        
        # 指数和报表数据等静态数据
        self.static_data = {}
        
        # 导入需要的库
        self._import_xtdata()
        
        # 设置LRU缓存
        self.get_market_data_cached = lru_cache(maxsize=128)(self._get_market_data_uncached)
        
        logger.info(f"市场数据缓存系统初始化完成，缓存目录: {cache_dir}")
        logger.info(f"内存限制: {memory_limit_gb}GB, 热数据缓存天数: {hot_cache_days}")
    
    def _import_xtdata(self):
        """导入xtdata模块"""
        try:
            from xtquant import xtdata
            self.xtdata = xtdata
            # 检查xtdata连接
            if not xtdata.hello():
                xtdata.connect()
            return True
        except ImportError:
            warnings.warn("无法导入xtquant.xtdata模块")
            return False
    
    def _load_metadata(self):
        """加载缓存元数据"""
        metadata_path = self.cache_dir / "metadata.pkl"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"加载缓存元数据失败: {e}")
        return {
            'files': {},        # {cache_key: file_path}
            'last_update': {},  # {cache_key: 最后更新时间}
            'size': {},         # {cache_key: 文件大小}
            'stats': {
                'hits': 0,
                'misses': 0,
                'api_calls': 0
            }
        }
    
    def _save_metadata(self):
        """保存缓存元数据"""
        metadata_path = self.cache_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def _get_cache_key(self, stock_list, period, field_list, start_time, end_time):
        """生成缓存键"""
        # 确保列表有序，避免相同内容不同顺序生成不同键
        stock_list = sorted(stock_list) if isinstance(stock_list, list) else [stock_list]
        field_list = sorted(field_list) if isinstance(field_list, list) else [field_list]
        
        # 构建键的组成部分
        key_parts = [
            ','.join(stock_list),
            period,
            ','.join(field_list),
            start_time or '00000000',
            end_time or '99999999'
        ]
        
        # 使用MD5生成固定长度的键
        key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()
        return key
    
    def _get_disk_cache_path(self, cache_key):
        """获取磁盘缓存文件路径"""
        # 使用键的前两个字符作为子目录，减少单个目录下的文件数
        subdir = cache_key[:2]
        directory = self.cache_dir / subdir
        directory.mkdir(exist_ok=True)
        
        return directory / f"{cache_key}.h5"
    
    def _estimate_memory_size(self, data):
        """估计数据在内存中的大小(字节)"""
        if isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum()
        elif isinstance(data, dict):
            return sum(self._estimate_memory_size(v) for v in data.values())
        else:
            return sys.getsizeof(data)
    
    def _save_to_disk(self, cache_key, data):
        """将数据保存到磁盘缓存"""
        if not self.enable_disk_cache:
            return False
        
        try:
            file_path = self._get_disk_cache_path(cache_key)
            
            # 使用HDF5格式保存数据
            if isinstance(data, dict) and all(isinstance(v, pd.DataFrame) for v in data.values()):
                with h5py.File(file_path, 'w') as h5file:
                    for field, df in data.items():
                        # 创建字段组
                        group = h5file.create_group(field)
                        
                        # 保存索引 
                        group.create_dataset('index', data=np.array(df.index).astype('S20'))
                        
                        # 保存列名
                        group.create_dataset('columns', data=np.array(df.columns).astype('S20'))
                        
                        # 保存数据
                        for col in df.columns:
                            col_data = df[col].values
                            group.create_dataset(f'data/{col}', data=col_data)
            
            # 更新元数据
            self.metadata['files'][cache_key] = str(file_path)
            self.metadata['last_update'][cache_key] = time.time()
            self.metadata['size'][cache_key] = file_path.stat().st_size
            self._save_metadata()
            
            return True
        except Exception as e:
            logger.warning(f"保存到磁盘缓存失败: {str(e)}")
            return False
    
    def _load_from_disk(self, cache_key):
        """从磁盘缓存加载数据"""
        if not self.enable_disk_cache:
            return None
        
        if cache_key not in self.metadata['files']:
            return None
        
        try:
            file_path = Path(self.metadata['files'][cache_key])
            if not file_path.exists():
                return None
            
            # 使用HDF5加载数据
            result = {}
            with h5py.File(file_path, 'r') as h5file:
                for field in h5file.keys():
                    group = h5file[field]
                    
                    # 读取索引和列
                    index = [s.decode('utf-8') for s in group['index']]
                    columns = [s.decode('utf-8') for s in group['columns']]
                    
                    # 创建空DataFrame
                    df = pd.DataFrame(index=index, columns=columns)
                    
                    # 逐列读取数据
                    for col in columns:
                        df[col] = group[f'data/{col}'][:]
                    
                    result[field] = df
            
            # 更新访问统计
            self.last_access[cache_key] = time.time()
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            
            return result
        except Exception as e:
            logger.warning(f"从磁盘缓存加载失败: {str(e)}")
            return None
    
    def _evict_memory_cache(self, required_space=0):
        """清理内存缓存以释放空间"""
        if self.memory_usage <= self.memory_limit and required_space == 0:
            return
        
        # 按最后访问时间排序
        items = [(k, self.last_access.get(k, 0)) for k in self.memory_cache.keys()]
        items.sort(key=lambda x: x[1])
        
        # 计算需要释放的空间
        to_free = max(required_space, self.memory_usage - self.memory_limit + required_space)
        freed = 0
        
        for key, _ in items:
            if freed >= to_free:
                break
                
            # 估计此项大小
            size = self._estimate_memory_size(self.memory_cache[key])
            
            # 确保数据已持久化到磁盘
            if key not in self.metadata['files']:
                self._save_to_disk(key, self.memory_cache[key])
            
            # 从内存中移除
            del self.memory_cache[key]
            self.memory_usage -= size
            freed += size
    
    def clear_cache(self, older_than_days=None):
        """清理缓存
        
        参数:
            older_than_days: 清理超过指定天数的缓存，None表示清理所有缓存
        """
        now = time.time()
        cutoff = now - (older_than_days * 86400) if older_than_days else 0
        
        # 清理内存缓存
        keys_to_remove = []
        for key in self.memory_cache:
            if self.last_access.get(key, 0) < cutoff:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            size = self._estimate_memory_size(self.memory_cache[key])
            del self.memory_cache[key]
            self.memory_usage -= size
        
        # 清理磁盘缓存
        if self.enable_disk_cache:
            files_to_remove = []
            for key, update_time in self.metadata['last_update'].items():
                if update_time < cutoff:
                    files_to_remove.append(key)
            
            for key in files_to_remove:
                if key in self.metadata['files']:
                    file_path = Path(self.metadata['files'][key])
                    if file_path.exists():
                        file_path.unlink()
                    
                    del self.metadata['files'][key]
                    del self.metadata['last_update'][key]
                    del self.metadata['size'][key]
        
        self._save_metadata()
        
        logger.info(f"缓存清理完成: 移除了 {len(keys_to_remove)} 个内存缓存项和 {len(files_to_remove) if 'files_to_remove' in locals() else 0} 个磁盘缓存文件")
    
    def _update_memory_cache(self, cache_key, data):
        """更新内存缓存"""
        # 估计数据大小
        data_size = self._estimate_memory_size(data)
        
        # 检查是否需要清理缓存
        if data_size + self.memory_usage > self.memory_limit:
            self._evict_memory_cache(data_size)
        
        # 更新缓存
        self.memory_cache[cache_key] = data
        self.memory_usage += data_size
        self.last_access[cache_key] = time.time()
        self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
    
    def _get_market_data_uncached(self, stock_list, field_list, period, start_time, end_time):
        """直接从API获取市场数据（未缓存版本）"""
        try:
            # 记录API调用
            self.metadata['stats']['api_calls'] += 1
            
            # 记录开始获取时间
            start = time.time()
            
            # 调用xtdata API
            result = self.xtdata.get_market_data(
                field_list=field_list,
                stock_list=stock_list,
                period=period,
                start_time=start_time, 
                end_time=end_time
            )
            
            # 记录获取时间
            elapsed = time.time() - start
            logger.debug(f"API调用耗时: {elapsed:.4f}秒 - 股票数: {len(stock_list)}, 字段数: {len(field_list)}")
            
            return result
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            return None
    
    def get_market_data(self, stock_list, field_list, period='1d', start_time='', end_time=''):
        """获取市场数据，优先从缓存获取"""
        # 生成缓存键
        cache_key = self._get_cache_key(stock_list, period, field_list, start_time, end_time)
        
        # 检查内存缓存
        if cache_key in self.memory_cache:
            # 更新访问统计
            self.last_access[cache_key] = time.time()
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            self.metadata['stats']['hits'] += 1
            return self.memory_cache[cache_key]
        
        # 检查磁盘缓存
        disk_data = self._load_from_disk(cache_key)
        if disk_data is not None:
            # 更新内存缓存
            self._update_memory_cache(cache_key, disk_data)
            self.metadata['stats']['hits'] += 1
            return disk_data
        
        # 从API获取数据
        self.metadata['stats']['misses'] += 1
        data = self._get_market_data_uncached(stock_list, field_list, period, start_time, end_time)
        
        if data is not None:
            # 更新缓存
            self._update_memory_cache(cache_key, data)
            self._save_to_disk(cache_key, data)
        
        return data
    
    def get_stock_price(self, stock_code, date, field='close'):
        """获取单只股票的价格数据，优先从缓存获取"""
        # 如果日期为空，使用"今天"
        if not date:
            date = time.strftime('%Y%m%d')
        
        # 尝试从内存中获取数据点
        cache_key = self._get_cache_key([stock_code], '1d', [field], '', date)
        
        # 先检查是否有缓存
        if cache_key in self.memory_cache:
            data = self.memory_cache[cache_key]
            if field in data and stock_code in data[field].index:
                # 检查特定日期是否存在
                if date in data[field].columns:
                    return data[field].loc[stock_code, date]
        
        # 尝试获取该股票的历史数据
        # 获取最近30天数据以提高命中率
        end_date = date
        start_date = self._get_date_before_days(date, 30)
        
        data = self.get_market_data(
            stock_list=[stock_code],
            field_list=[field],
            period='1d',
            start_time=start_date,
            end_time=end_date
        )
        
        # 尝试获取特定日期的价格
        if data and field in data and stock_code in data[field].index:
            if date in data[field].columns:
                return data[field].loc[stock_code, date]
        
        # 如果没有找到特定日期的数据，尝试获取最近的交易日数据
        return self._get_closest_price(stock_code, date, field)
    
    def _get_closest_price(self, stock_code, date, field):
        """获取最接近指定日期的价格数据"""
        # 获取前后5天的数据
        date_int = int(date)
        before_date = str(date_int - 500).zfill(8)  # 大约前5天
        after_date = str(date_int + 500).zfill(8)   # 大约后5天
        
        data = self.get_market_data(
            stock_list=[stock_code],
            field_list=[field],
            period='1d',
            start_time=before_date,
            end_time=after_date
        )
        
        if data and field in data and stock_code in data[field].index:
            # 获取所有可用日期
            available_dates = [d for d in data[field].columns if d]
            
            if available_dates:
                # 找到最接近的日期
                closest_date = min(available_dates, key=lambda d: abs(int(d) - int(date)))
                return data[field].loc[stock_code, closest_date]
        
        return None
    
    def _get_date_before_days(self, date, days):
        """获取指定日期前N天的日期"""
        date_format = '%Y%m%d'
        from datetime import datetime, timedelta
        
        dt = datetime.strptime(date, date_format)
        before_dt = dt - timedelta(days=days)
        return before_dt.strftime(date_format)
    
    def preload_stock_data(self, stock_list, start_date, end_date, 
                           fields=['open', 'high', 'low', 'close', 'volume'], 
                           periods=['1d']):
        """预加载多只股票的历史数据
        
        参数:
            stock_list: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要的字段列表
            periods: 周期列表，如['1d', '5m']
        """
        logger.info(f"开始预加载数据: {len(stock_list)} 只股票, 期间: {start_date} 至 {end_date}")
        
        # 为了减少内存压力，分批加载
        batch_size = 50  # 每批股票数量
        total_batches = (len(stock_list) + batch_size - 1) // batch_size
        
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"预加载批次 {batch_num}/{total_batches}, 股票数量: {len(batch)}")
            
            for period in periods:
                try:
                    self.get_market_data(
                        stock_list=batch,
                        field_list=fields,
                        period=period,
                        start_time=start_date,
                        end_time=end_date
                    )
                except Exception as e:
                    logger.error(f"批次 {batch_num} 预加载失败: {e}")
    
    def preload_batch_parallel(self, stock_list, start_date, end_date, fields, periods=['1d']):
        """并行预加载数据，适用于大量股票"""
        # 将股票列表分批
        batch_size = max(10, len(stock_list) // 10)  # 动态批大小
        batches = [stock_list[i:i+batch_size] for i in range(0, len(stock_list), batch_size)]
        
        # 定义单批次加载函数
        def load_batch(batch):
            for period in periods:
                try:
                    self.get_market_data(
                        stock_list=batch,
                        field_list=fields,
                        period=period,
                        start_time=start_date,
                        end_time=end_date
                    )
                    return (batch[0], True)
                except Exception as e:
                    logger.error(f"批次 {batch[0]} 预加载失败: {e}")
                    return (batch[0], False)
        
        # 并行执行
        with ThreadPoolExecutor(max_workers=min(len(batches), 8)) as executor:
            futures = {executor.submit(load_batch, batch): batch for batch in batches}
            for future in futures:
                future.result()  # 等待完成
    
    def get_all_stock_prices(self, stock_list, date, fields=['open', 'high', 'low', 'close', 'volume']):
        """批量获取多只股票在指定日期的价格"""
        # 尝试从缓存获取
        result = {}
        data = self.get_market_data(
            stock_list=stock_list,
            field_list=fields,
            period='1d',
            start_time=date,
            end_time=date
        )
        
        if not data:
            logger.warning(f"无法获取日期 {date} 的股票价格数据")
            return {}
            
        # 处理数据
        for stock in stock_list:
            stock_data = {}
            for field in fields:
                if field in data and stock in data[field].index:
                    # 尝试不同的方式获取数据
                    try:
                        # 如果是DataFrame格式
                        if isinstance(data[field], pd.DataFrame) and not data[field].empty:
                            # 有些情况下，数据是按股票代码索引的
                            if date in data[field].columns:
                                stock_data[field] = float(data[field].loc[stock, date])
                            elif data[field].index.name == 'date' and stock in data[field].columns:
                                stock_data[field] = float(data[field].loc[date, stock])
                            else:
                                # 取第一个可用值
                                col = data[field].columns[0]
                                stock_data[field] = float(data[field].loc[stock, col])
                        # 如果是字典格式
                        elif isinstance(data[field], dict) and stock in data[field]:
                            stock_data[field] = float(data[field][stock].get(date, 0))
                    except (KeyError, IndexError, ValueError) as e:
                        logger.warning(f"处理股票 {stock} 字段 {field} 时出错: {e}")
            
            if stock_data:
                result[stock] = stock_data
                
        return result
    
    def get_trading_dates(self, market='SH', start_date='', end_date=''):
        """获取交易日期列表，优先从缓存获取"""
        # 缓存键
        cache_key = f"trading_dates_{market}_{start_date}_{end_date}"
        
        # 检查内存缓存
        if cache_key in self.static_data:
            return self.static_data[cache_key]
        
        # 从API获取
        try:
            dates = self.xtdata.get_trading_dates(market, start_date, end_date)
            
            # 保存到静态数据缓存
            self.static_data[cache_key] = dates
            
            return dates
        except Exception as e:
            logger.error(f"获取交易日期失败: {e}")
            return []
    
    def get_batch_market_data(self, field_list, stock_list, period, dates):
        """获取多个日期段的市场数据并合并
        
        针对分段回测场景，将一次大型请求拆分为多次小型请求，提高稳定性
        """
        if not dates or len(dates) < 2:
            logger.warning("日期范围无效，无法获取批量数据")
            return None
            
        # 分割日期范围为多个小段
        segments = []
        for i in range(0, len(dates), 60):  # 每段60个交易日
            segment_dates = dates[i:i+60]
            if segment_dates:
                segments.append((segment_dates[0], segment_dates[-1]))
        
        # 获取每段数据
        all_data = {}
        for start, end in segments:
            segment_data = self.get_market_data(
                stock_list=stock_list,
                field_list=field_list,
                period=period,
                start_time=start,
                end_time=end
            )
            
            # 合并数据
            if not all_data:
                all_data = segment_data
            elif segment_data:
                for field in field_list:
                    if field in segment_data and field in all_data:
                        try:
                            # 合并DataFrame
                            all_data[field] = pd.concat([all_data[field], segment_data[field]], axis=1)
                        except Exception as e:
                            logger.error(f"合并字段 {field} 数据时出错: {e}")
        
        return all_data
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        total_size_memory = self.memory_usage
        
        total_size_disk = 0
        if self.enable_disk_cache:
            total_size_disk = sum(self.metadata['size'].values())
        
        hit_rate = 0
        total_requests = self.metadata['stats']['hits'] + self.metadata['stats']['misses']
        if total_requests > 0:
            hit_rate = self.metadata['stats']['hits'] / total_requests * 100
            
        return {
            'memory_items': len(self.memory_cache),
            'memory_usage_mb': total_size_memory / (1024 * 1024),
            'disk_items': len(self.metadata['files']),
            'disk_usage_mb': total_size_disk / (1024 * 1024),
            'api_calls': self.metadata['stats']['api_calls'],
            'cache_hits': self.metadata['stats']['hits'],
            'cache_misses': self.metadata['stats']['misses'],
            'hit_rate': hit_rate
        }

# 全局缓存实例，用于共享
_GLOBAL_CACHE = None

def get_global_cache(memory_limit_gb=4):
    """获取全局缓存实例"""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = MarketDataCache(
            cache_dir="./data_cache",
            memory_limit_gb=memory_limit_gb, 
            enable_disk_cache=True
        )
    return _GLOBAL_CACHE

if __name__ == "__main__":
    # 测试缓存系统性能
    cache = MarketDataCache(
        cache_dir="./data_cache",
        memory_limit_gb=4,
        enable_disk_cache=True
    )
    
    # 测试股票列表
    test_stocks = ['000001.SZ', '600000.SH', '000002.SZ', '600036.SH', '601398.SH']
    test_fields = ['open', 'high', 'low', 'close', 'volume']
    
    # 测试数据获取
    print("测试数据获取...")
    start_time = time.time()
    data1 = cache.get_market_data(
        stock_list=test_stocks,
        field_list=test_fields,
        period='1d',
        start_time='20230101',
        end_time='20230131'
    )
    elapsed1 = time.time() - start_time
    print(f"第1次获取耗时: {elapsed1:.4f}秒")
    
    # 测试缓存命中
    start_time = time.time()
    data2 = cache.get_market_data(
        stock_list=test_stocks,
        field_list=test_fields,
        period='1d',
        start_time='20230101',
        end_time='20230131'
    )
    elapsed2 = time.time() - start_time
    print(f"第2次获取(缓存)耗时: {elapsed2:.4f}秒")
    print(f"加速比: {elapsed1/elapsed2:.1f}倍")
    
    # 显示缓存统计
    stats = cache.get_cache_stats()
    print("\n缓存统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")