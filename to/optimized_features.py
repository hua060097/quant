#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化的特征生成和存储模块

提供高性能的特征生成和存储功能，使用内存映射和压缩优化技术
"""

import os
import gc
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import time
import mmap
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Union, Optional, Any, Set
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# 导入路径工具
from to.path_utils import get_data_path, get_project_path, ensure_dir
from to.feature_slicer import (
    split_date_range_by_quarter, 
    save_features_by_quarter,
    update_quarter_features,
    get_quarter_bounds,
    get_quarter_file_path
)

# 设置日志
logger = logging.getLogger(__name__)

# 特征压缩方式
COMPRESSION_TYPE = 'snappy'  # 或 'zstd', 'lz4'，snappy在速度和压缩比之间最平衡

# 内存映射配置
USE_MEMORY_MAP = True
MEMORY_MAP_THRESHOLD_MB = 100  # 当文件大于此值时使用内存映射

# 批处理配置
DEFAULT_CHUNK_SIZE = 200  # 每个批次处理的日期数


class MemoryOptimizedDataFrame:
    """内存优化的DataFrame类
    
    使用内存映射和压缩技术优化大型DataFrame的处理
    """
    
    def __init__(self, data=None, temp_file=None):
        """初始化
        
        Args:
            data: 初始数据，可以是DataFrame或None
            temp_file: 临时文件路径，如果为None则自动生成
        """
        self.df = pd.DataFrame() if data is None else data
        self.temp_file = temp_file or f"temp_df_{int(time.time())}.parquet"
        self.in_memory = True
        self.file_path = Path(self.temp_file)
        self._check_and_offload()
    
    def _check_and_offload(self):
        """检查内存使用并决定是否将数据卸载到磁盘"""
        if self.in_memory and len(self.df) > 100000:  # 大约10万行作为阈值
            self._offload_to_disk()
    
    def _offload_to_disk(self):
        """将数据卸载到磁盘以节省内存"""
        if not self.in_memory:
            return
        
        # 将数据写入临时parquet文件
        self.df.to_parquet(self.file_path, compression=COMPRESSION_TYPE)
        # 清空内存中的DataFrame
        self.df = None
        self.in_memory = False
        # 强制垃圾回收
        gc.collect()
        logger.debug(f"将DataFrame卸载到磁盘: {self.file_path}")
    
    def _load_from_disk(self):
        """从磁盘加载数据到内存"""
        if self.in_memory:
            return
        
        # 从临时文件加载数据
        if os.path.exists(self.file_path):
            if USE_MEMORY_MAP and os.path.getsize(self.file_path) > MEMORY_MAP_THRESHOLD_MB * 1024 * 1024:
                self.df = pd.read_parquet(self.file_path, memory_map=True)
            else:
                self.df = pd.read_parquet(self.file_path)
            self.in_memory = True
            logger.debug(f"从磁盘加载DataFrame: {self.file_path}")
        else:
            logger.warning(f"尝试从不存在的文件加载DataFrame: {self.file_path}")
            self.df = pd.DataFrame()
            self.in_memory = True
    
    def append(self, other):
        """追加数据
        
        Args:
            other: 要追加的DataFrame
            
        Returns:
            self
        """
        if not self.in_memory:
            self._load_from_disk()
        
        self.df = pd.concat([self.df, other], ignore_index=True)
        self._check_and_offload()
        return self
    
    def filter(self, condition):
        """按条件过滤数据
        
        Args:
            condition: 过滤条件
            
        Returns:
            过滤后的DataFrame
        """
        if not self.in_memory:
            self._load_from_disk()
        
        return self.df[condition]
    
    def get_dataframe(self):
        """获取DataFrame
        
        Returns:
            DataFrame对象
        """
        if not self.in_memory:
            self._load_from_disk()
        
        return self.df
    
    def __del__(self):
        """析构函数，清理临时文件"""
        try:
            if not self.in_memory and os.path.exists(self.file_path):
                os.remove(self.file_path)
                logger.debug(f"删除临时DataFrame文件: {self.file_path}")
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {e}")


class OptimizedFeatureGenerator:
    """优化的特征生成器
    
    提供高性能的特征生成和存储功能
    """
    
    def __init__(self, output_dir=None, use_memory_map=True, compression=COMPRESSION_TYPE):
        """初始化
        
        Args:
            output_dir: 输出目录，如果为None则使用默认的sector_features目录
            use_memory_map: 是否使用内存映射
            compression: 压缩方式
        """
        self.output_dir = output_dir or get_data_path('sector_features')
        ensure_dir(self.output_dir)
        self.use_memory_map = use_memory_map
        self.compression = compression
        # 内部缓存
        self._sector_data_cache = {}
        self._sector_stats_cache = {}
    
    def generate_batch_features(
        self,
        sector: str,
        start_date: str,
        end_date: str,
        date_processor_func,
        sector_processor_func=None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_workers: int = None,
        force_update: bool = False
    ):
        """批量生成特征
        
        Args:
            sector: 板块名称
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            date_processor_func: 处理单个日期的函数，接收(日期, 板块名称)参数
            sector_processor_func: 处理整个板块的函数，接收(板块名称, 特征DataFrame)参数
            chunk_size: 每批处理的日期数量
            max_workers: 最大工作进程数
            force_update: 是否强制更新现有数据
        
        Returns:
            生成的特征文件路径列表
        """
        # 获取要处理的日期范围
        quarters = split_date_range_by_quarter(start_date, end_date)
        
        # 组织按季度的日期
        quarter_dates = {}
        for quarter_start, quarter_end in quarters:
            # 检查是否需要处理该季度
            file_path = get_quarter_file_path(sector, quarter_start, quarter_end, self.output_dir)
            
            if not force_update and file_path.exists():
                logger.info(f"季度特征文件已存在: {file_path}，跳过处理")
                continue
            
            try:
                from to.data_loaders import get_trading_days
                # 获取季度内的所有交易日
                dates = get_trading_days(quarter_start, quarter_end)
                quarter_dates[(quarter_start, quarter_end)] = dates
            except Exception as e:
                logger.error(f"获取交易日期失败: {e}")
                continue
        
        if not quarter_dates:
            logger.info(f"没有需要处理的季度，已完成")
            return []
        
        # 处理每个季度的特征
        results = []
        
        for quarter_key, dates in quarter_dates.items():
            quarter_start, quarter_end = quarter_key
            logger.info(f"处理季度 {quarter_start}-{quarter_end}, 共 {len(dates)} 个交易日")
            
            # 分批处理日期
            date_chunks = [dates[i:i+chunk_size] for i in range(0, len(dates), chunk_size)]
            
            all_features = MemoryOptimizedDataFrame()
            
            for i, date_chunk in enumerate(date_chunks):
                logger.info(f"处理第 {i+1}/{len(date_chunks)} 批，共 {len(date_chunk)} 个日期")
                
                # 并行处理日期
                if max_workers and max_workers > 1 and len(date_chunk) > 1:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(date_processor_func, date, sector)
                            for date in date_chunk
                        ]
                        
                        # 收集结果
                        for future in futures:
                            try:
                                result = future.result()
                                if result is not None and isinstance(result, pd.DataFrame) and not result.empty:
                                    all_features.append(result)
                            except Exception as e:
                                logger.error(f"处理日期时出错: {e}")
                else:
                    # 串行处理
                    for date in date_chunk:
                        try:
                            result = date_processor_func(date, sector)
                            if result is not None and isinstance(result, pd.DataFrame) and not result.empty:
                                all_features.append(result)
                        except Exception as e:
                            logger.error(f"处理日期 {date} 时出错: {e}")
                
                # 强制垃圾回收
                gc.collect()
            
            # 获取合并后的特征DataFrame
            quarter_features = all_features.get_dataframe()
            
            if sector_processor_func and not quarter_features.empty:
                try:
                    # 应用板块级别的处理
                    quarter_features = sector_processor_func(sector, quarter_features)
                except Exception as e:
                    logger.error(f"应用板块处理函数时出错: {e}")
            
            if not quarter_features.empty:
                # 保存季度特征
                file_path = get_quarter_file_path(sector, quarter_start, quarter_end, self.output_dir)
                
                # 使用PyArrow优化写入性能
                table = pa.Table.from_pandas(quarter_features)
                pq.write_table(
                    table, 
                    file_path,
                    compression=self.compression
                )
                
                logger.info(f"保存季度特征文件：{file_path}，数据形状：{quarter_features.shape}")
                results.append(file_path)
            else:
                logger.warning(f"季度 {quarter_start}-{quarter_end} 没有生成有效特征")
        
        return results
    
    def efficient_update_features(
        self,
        sector: str,
        start_date: str,
        end_date: str,
        date_processor_func,
        sector_processor_func=None
    ):
        """高效地增量更新特征
        
        仅处理缺失的日期，而不是整个日期范围
        
        Args:
            sector: 板块名称
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            date_processor_func: 处理单个日期的函数，接收(日期, 板块名称)参数
            sector_processor_func: 处理整个板块的函数，接收(板块名称, 特征DataFrame)参数
        
        Returns:
            更新的特征文件路径列表
        """
        from to.feature_slicer import find_missing_dates
        
        # 查找缺失的日期
        missing_dates = find_missing_dates(sector, start_date, end_date, self.output_dir)
        
        if not missing_dates:
            logger.info(f"日期范围 {start_date}-{end_date} 内没有缺失的日期，无需更新")
            return []
        
        logger.info(f"发现 {len(missing_dates)} 个缺失的日期，开始增量更新")
        
        # 按季度组织缺失的日期
        quarter_missing_dates = {}
        for date in missing_dates:
            quarter_start, quarter_end = get_quarter_bounds(date)
            quarter_key = (quarter_start, quarter_end)
            
            if quarter_key not in quarter_missing_dates:
                quarter_missing_dates[quarter_key] = []
            
            quarter_missing_dates[quarter_key].append(date)
        
        # 处理每个季度的缺失日期
        results = []
        
        for quarter_key, dates in quarter_missing_dates.items():
            quarter_start, quarter_end = quarter_key
            logger.info(f"处理季度 {quarter_start}-{quarter_end} 的 {len(dates)} 个缺失日期")
            
            new_features_list = []
            
            # 生成新的特征
            for date in dates:
                try:
                    result = date_processor_func(date, sector)
                    if result is not None and isinstance(result, pd.DataFrame) and not result.empty:
                        new_features_list.append(result)
                except Exception as e:
                    logger.error(f"处理日期 {date} 时出错: {e}")
            
            if not new_features_list:
                logger.warning(f"季度 {quarter_start}-{quarter_end} 没有生成有效特征")
                continue
            
            # 合并新特征
            new_features = pd.concat(new_features_list, ignore_index=True)
            
            if sector_processor_func:
                try:
                    # 应用板块级别的处理
                    new_features = sector_processor_func(sector, new_features)
                except Exception as e:
                    logger.error(f"应用板块处理函数时出错: {e}")
            
            # 更新季度特征文件
            file_path = get_quarter_file_path(sector, quarter_start, quarter_end, self.output_dir)
            
            # 如果文件已存在，读取并合并
            if file_path.exists():
                try:
                    # 使用内存映射读取大文件
                    old_features = pd.read_parquet(file_path, memory_map=self.use_memory_map)
                    
                    # 确保日期列为字符串格式
                    if 'date' in old_features.columns and not isinstance(old_features['date'].iloc[0], str):
                        old_features['date'] = old_features['date'].dt.strftime('%Y%m%d')
                    
                    if 'date' in new_features.columns and not isinstance(new_features['date'].iloc[0], str):
                        new_features['date'] = new_features['date'].dt.strftime('%Y%m%d')
                    
                    # 移除可能重复的日期
                    if 'date' in old_features.columns and 'date' in new_features.columns:
                        existing_dates = set(old_features['date'])
                        new_features = new_features[~new_features['date'].isin(existing_dates)]
                    
                    if not new_features.empty:
                        # 合并特征
                        merged_features = pd.concat([old_features, new_features], ignore_index=True)
                        
                        # 使用PyArrow优化写入性能
                        table = pa.Table.from_pandas(merged_features)
                        pq.write_table(table, file_path, compression=self.compression)
                        
                        logger.info(f"更新季度特征文件：{file_path}，新增 {len(new_features)} 条记录")
                        results.append(file_path)
                    else:
                        logger.info(f"季度特征文件：{file_path} 无需更新，所有日期已存在")
                except Exception as e:
                    logger.error(f"更新季度特征文件 {file_path} 时出错: {e}")
            else:
                # 文件不存在，直接写入新特征
                ensure_dir(file_path.parent)
                table = pa.Table.from_pandas(new_features)
                pq.write_table(table, file_path, compression=self.compression)
                
                logger.info(f"创建新的季度特征文件：{file_path}，共 {len(new_features)} 条记录")
                results.append(file_path)
        
        return results
    
    def create_merged_features(
        self,
        sectors: List[str],
        start_date: str,
        end_date: str,
        output_file: Optional[str] = None
    ):
        """创建多个板块多个季度的合并特征文件
        
        Args:
            sectors: 板块名称列表
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            output_file: 输出文件路径，如果为None则自动生成
        
        Returns:
            合并特征文件路径
        """
        all_features = []
        
        # 为每个板块加载特征
        for sector in sectors:
            try:
                # 使用内存映射加载特征
                from to.feature_slicer import load_sector_features_by_date_range
                
                df = load_sector_features_by_date_range(
                    sector,
                    start_date,
                    end_date,
                    self.output_dir,
                    use_memory_map=self.use_memory_map
                )
                
                if not df.empty:
                    all_features.append(df)
                    logger.info(f"加载板块 {sector} 的特征，形状: {df.shape}")
                else:
                    logger.warning(f"板块 {sector} 在日期范围 {start_date}-{end_date} 内没有特征数据")
            except Exception as e:
                logger.error(f"加载板块 {sector} 的特征时出错: {e}")
        
        if not all_features:
            logger.warning(f"没有找到任何板块特征数据，无法创建合并文件")
            return None
        
        # 合并所有特征
        merged_features = pd.concat(all_features, ignore_index=True)
        
        # 设置输出文件路径
        if output_file is None:
            merged_dir = ensure_dir(Path(self.output_dir) / "merged")
            output_file = merged_dir / f"merged_features_{start_date}_{end_date}.parquet"
        else:
            output_file = Path(output_file)
            ensure_dir(output_file.parent)
        
        # 使用最优压缩方式保存
        logger.info(f"保存合并特征文件: {output_file}，数据形状: {merged_features.shape}")
        
        # 使用PyArrow优化写入性能，设置行组大小以优化查询性能
        table = pa.Table.from_pandas(merged_features)
        pq.write_table(
            table, 
            output_file,
            compression=self.compression,
            row_group_size=100000  # 设置较大的行组大小，适合时间序列查询
        )
        
        return output_file
    
    def optimize_existing_features(self, sector: str, recompress: bool = True):
        """优化现有特征文件
        
        Args:
            sector: 板块名称
            recompress: 是否重新压缩
        
        Returns:
            优化的文件路径列表
        """
        from to.feature_slicer import list_available_quarters
        
        # 获取可用的季度
        quarters = list_available_quarters(sector, self.output_dir)
        
        if not quarters:
            logger.warning(f"板块 {sector} 没有可用的特征文件")
            return []
        
        optimized_files = []
        
        for quarter_start, quarter_end in quarters:
            file_path = get_quarter_file_path(sector, quarter_start, quarter_end, self.output_dir)
            
            if not file_path.exists():
                continue
            
            try:
                # 读取特征
                df = pd.read_parquet(file_path, memory_map=self.use_memory_map)
                
                # 优化内存使用
                for col in df.columns:
                    # 对数值列进行降精度处理
                    if pd.api.types.is_float_dtype(df[col]):
                        df[col] = df[col].astype('float32')
                    elif pd.api.types.is_integer_dtype(df[col]):
                        # 检查最大值和最小值，选择合适的整数类型
                        col_min, col_max = df[col].min(), df[col].max()
                        if col_min >= 0:
                            if col_max <= 255:
                                df[col] = df[col].astype('uint8')
                            elif col_max <= 65535:
                                df[col] = df[col].astype('uint16')
                            elif col_max <= 4294967295:
                                df[col] = df[col].astype('uint32')
                        else:
                            if col_min >= -128 and col_max <= 127:
                                df[col] = df[col].astype('int8')
                            elif col_min >= -32768 and col_max <= 32767:
                                df[col] = df[col].astype('int16')
                            elif col_min >= -2147483648 and col_max <= 2147483647:
                                df[col] = df[col].astype('int32')
                
                # 临时文件路径
                temp_file = file_path.with_name(f"temp_{file_path.name}")
                
                # 使用PyArrow优化写入性能
                table = pa.Table.from_pandas(df)
                pq.write_table(
                    table, 
                    temp_file,
                    compression=self.compression,
                    row_group_size=100000
                )
                
                # 替换原文件
                temp_file.replace(file_path)
                logger.info(f"优化特征文件: {file_path}")
                
                optimized_files.append(file_path)
            except Exception as e:
                logger.error(f"优化特征文件 {file_path} 时出错: {e}")
        
        return optimized_files


# 工具函数

def optimize_dataframe_memory(df):
    """优化DataFrame内存使用
    
    Args:
        df: 要优化的DataFrame
        
    Returns:
        优化后的DataFrame
    """
    for col in df.columns:
        # 对数值列进行降精度处理
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype('float32')
        elif pd.api.types.is_integer_dtype(df[col]):
            # 检查最大值和最小值，选择合适的整数类型
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= 0:
                if col_max <= 255:
                    df[col] = df[col].astype('uint8')
                elif col_max <= 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max <= 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')
        # 字符串列转换为分类类型
        elif pd.api.types.is_object_dtype(df[col]) and df[col].nunique() < len(df) * 0.5:
            df[col] = df[col].astype('category')
    
    return df

def log_memory_usage():
    """记录当前内存使用情况
    
    Returns:
        内存使用信息字典
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = {
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_mb': memory_info.vms / (1024 * 1024),
        'percent': process.memory_percent()
    }
    
    logger.info(f"内存使用: {memory_usage['rss_mb']:.2f} MB (RSS), {memory_usage['percent']:.2f}%")
    
    return memory_usage

def create_memory_mapped_temp_file(df, prefix="temp_df_"):
    """创建内存映射的临时文件
    
    Args:
        df: 要保存的DataFrame
        prefix: 临时文件前缀
        
    Returns:
        (文件路径, 内存映射对象)
    """
    import tempfile
    
    # 创建临时文件
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".parquet")
    os.close(fd)
    
    # 保存DataFrame到临时文件
    df.to_parquet(path, compression=COMPRESSION_TYPE)
    
    # 打开文件进行内存映射
    file_obj = open(path, "r+b")
    mapped = mmap.mmap(file_obj.fileno(), 0)
    
    return path, mapped, file_obj

def cleanup_memory_mapped_file(path, mapped, file_obj):
    """清理内存映射文件
    
    Args:
        path: 文件路径
        mapped: 内存映射对象
        file_obj: 文件对象
    """
    try:
        # 关闭内存映射
        if mapped:
            mapped.close()
        
        # 关闭文件
        if file_obj:
            file_obj.close()
        
        # 删除临时文件
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception as e:
        logger.error(f"清理内存映射文件时出错: {e}") 