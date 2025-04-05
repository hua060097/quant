#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
内存映射管理器模块

使用内存映射技术高效读取大文件，管理内存占用。
"""

import os
import mmap
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd

# 获取logger
logger = logging.getLogger(__name__)

# 全局内存映射管理器
_MEMORY_MAP_MANAGER = None


class MemoryMapManager:
    """内存映射管理器
    
    管理文件的内存映射，提供高效的数据访问方式。
    
    Attributes:
        max_maps: 最大同时打开的内存映射数量
        current_maps: 当前打开的内存映射
    """
    
    def __init__(self, max_maps: int = 10):
        """初始化内存映射管理器
        
        Args:
            max_maps: 最大同时打开的内存映射数量
        """
        self.max_maps = max_maps
        self.current_maps = {}  # 文件路径 -> 映射对象
    
    def get_map(self, file_path: Union[str, Path]) -> Optional[mmap.mmap]:
        """获取文件的内存映射
        
        如果映射已存在，直接返回；否则创建新映射。
        如果已达到最大映射数量，先关闭最早创建的映射。
        
        Args:
            file_path: 文件路径
            
        Returns:
            内存映射对象，如果文件不存在则返回None
        """
        file_path = str(file_path)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return None
        
        # 如果映射已存在，直接返回
        if file_path in self.current_maps:
            logger.debug(f"使用已有映射: {file_path}")
            return self.current_maps[file_path]['map']
        
        # 如果已达到最大映射数量，关闭最早创建的映射
        if len(self.current_maps) >= self.max_maps:
            oldest_file = min(self.current_maps.items(), key=lambda x: x[1]['created'])[0]
            self.close_map(oldest_file)
        
        # 创建新映射
        try:
            with open(file_path, 'rb') as f:
                mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.current_maps[file_path] = {
                    'map': mmapped_file,
                    'created': pd.Timestamp.now()
                }
                logger.debug(f"创建新映射: {file_path}")
                return mmapped_file
        except Exception as e:
            logger.error(f"创建内存映射失败: {file_path}, 错误: {e}")
            return None
    
    def close_map(self, file_path: Union[str, Path]) -> bool:
        """关闭文件的内存映射
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功关闭
        """
        file_path = str(file_path)
        
        if file_path in self.current_maps:
            try:
                self.current_maps[file_path]['map'].close()
                del self.current_maps[file_path]
                logger.debug(f"关闭映射: {file_path}")
                return True
            except Exception as e:
                logger.error(f"关闭内存映射失败: {file_path}, 错误: {e}")
                return False
        
        return False
    
    def close_all(self) -> None:
        """关闭所有内存映射"""
        for file_path in list(self.current_maps.keys()):
            self.close_map(file_path)
        
        logger.debug("关闭所有内存映射")
    
    def __del__(self):
        """析构函数，确保关闭所有映射"""
        self.close_all()


def get_memory_map_manager(max_maps: int = 10) -> MemoryMapManager:
    """获取内存映射管理器单例
    
    Args:
        max_maps: 最大同时打开的内存映射数量
        
    Returns:
        内存映射管理器实例
    """
    global _MEMORY_MAP_MANAGER
    
    if _MEMORY_MAP_MANAGER is None:
        _MEMORY_MAP_MANAGER = MemoryMapManager(max_maps)
    
    return _MEMORY_MAP_MANAGER


def read_with_mmap(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """使用内存映射读取文件
    
    对于大文件，使用内存映射可以显著减少内存使用。
    
    Args:
        file_path: 文件路径
        **kwargs: 传递给pandas的读取函数的参数
        
    Returns:
        DataFrame对象
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"文件不存在: {file_path}")
        return pd.DataFrame()
    
    # 根据文件扩展名选择读取方法
    suffix = file_path.suffix.lower()
    
    try:
        manager = get_memory_map_manager()
        
        if suffix == '.csv':
            # CSV文件使用内存映射读取
            mmapped_file = manager.get_map(file_path)
            if mmapped_file:
                df = pd.read_csv(mmapped_file, **kwargs)
                return df
        elif suffix == '.parquet':
            # Parquet文件直接使用pyarrow读取
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            df = table.to_pandas()
            return df
        elif suffix in ['.json', '.jsonl']:
            # JSON文件使用内存映射读取
            mmapped_file = manager.get_map(file_path)
            if mmapped_file:
                df = pd.read_json(mmapped_file, **kwargs)
                return df
        elif suffix in ['.pkl', '.pickle']:
            # Pickle文件直接读取
            df = pd.read_pickle(file_path)
            return df
        else:
            logger.warning(f"不支持的文件格式: {suffix}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"读取文件失败: {file_path}, 错误: {e}")
        
        # 如果内存映射失败，尝试直接读取
        try:
            if suffix == '.csv':
                df = pd.read_csv(file_path, **kwargs)
                return df
            elif suffix == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
                return df
            elif suffix in ['.json', '.jsonl']:
                df = pd.read_json(file_path, **kwargs)
                return df
            elif suffix in ['.pkl', '.pickle']:
                df = pd.read_pickle(file_path, **kwargs)
                return df
            else:
                return pd.DataFrame()
        except Exception as nested_e:
            logger.error(f"直接读取文件也失败: {file_path}, 错误: {nested_e}")
            return pd.DataFrame()
    
    return pd.DataFrame()


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.DEBUG)
    
    # 测试内存映射管理器
    manager = get_memory_map_manager()
    
    # 创建测试文件
    test_file = Path("test_mmap.csv")
    
    # 生成测试数据
    test_df = pd.DataFrame({
        'A': range(1000),
        'B': np.random.randn(1000),
        'C': pd.date_range('20230101', periods=1000)
    })
    
    # 保存测试文件
    test_df.to_csv(test_file, index=False)
    
    # 使用内存映射读取
    read_df = read_with_mmap(test_file)
    print(f"使用内存映射读取的数据形状: {read_df.shape}")
    
    # 清理测试文件
    if test_file.exists():
        test_file.unlink()
        print(f"删除测试文件: {test_file}") 