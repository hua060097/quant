#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征更新模块

提供特征数据的增量更新和管理功能，避免重复处理已生成的特征。
支持特征文件的元数据管理、特征合并和断点续传。
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple, Set, Any

# 导入内部模块
try:
    # 尝试直接导入
    from path_utils import get_data_path
    from feature_compress import load_features, save_features, optimize_dtypes
except ImportError:
    # 尝试相对导入
    try:
        from .path_utils import get_data_path
        from .feature_compress import load_features, save_features, optimize_dtypes
    except ImportError:
        # 如果都失败，使用绝对导入
        from to.path_utils import get_data_path
        from to.feature_compress import load_features, save_features, optimize_dtypes

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局单例特征更新管理器
_FEATURE_UPDATE_MANAGER = None


class FeatureUpdateManager:
    """特征更新管理器
    
    管理特征数据的增量更新，避免重复处理已生成的特征。
    维护特征文件的元数据，支持断点续传和特征合并。
    
    Attributes:
        base_dir: 特征数据根目录
        metadata_dir: 元数据目录
        _metadata_cache: 元数据缓存字典
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """初始化特征更新管理器
        
        Args:
            base_dir: 特征数据根目录，如果为None则使用默认目录
        """
        self.base_dir = Path(base_dir) if base_dir else get_data_path('sector_features')
        self.metadata_dir = self.base_dir / '.metadata'
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # 元数据缓存
        self._metadata_cache = {}
    
    def get_metadata_path(self, sector: str) -> Path:
        """获取板块元数据文件路径
        
        Args:
            sector: 板块名称
            
        Returns:
            元数据文件路径
        """
        return self.metadata_dir / f"{sector}.json"
    
    def load_metadata(self, sector: str) -> Dict[str, Any]:
        """加载板块的元数据
        
        Args:
            sector: 板块名称
            
        Returns:
            元数据字典，如果不存在则返回空字典
        """
        # 检查缓存
        if sector in self._metadata_cache:
            return self._metadata_cache[sector]
        
        metadata_path = self.get_metadata_path(sector)
        
        if not metadata_path.exists():
            # 创建默认元数据
            metadata = {
                'sector': sector,
                'last_updated': '',
                'feature_files': [],
                'date_ranges': [],
                'record_count': 0,
                'file_size': 0,
                'feature_version': '1.0'
            }
        else:
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"加载板块 {sector} 的元数据失败: {e}")
                metadata = {
                    'sector': sector,
                    'last_updated': '',
                    'feature_files': [],
                    'date_ranges': [],
                    'record_count': 0,
                    'file_size': 0,
                    'feature_version': '1.0'
                }
        
        # 更新缓存
        self._metadata_cache[sector] = metadata
        
        return metadata
    
    def save_metadata(self, sector: str, metadata: Dict[str, Any]) -> bool:
        """保存板块的元数据
        
        Args:
            sector: 板块名称
            metadata: 元数据字典
            
        Returns:
            是否保存成功
        """
        metadata_path = self.get_metadata_path(sector)
        
        try:
            # 确保目录存在
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 更新最后修改时间
            metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 保存元数据
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 更新缓存
            self._metadata_cache[sector] = metadata
            
            return True
        except Exception as e:
            logger.error(f"保存板块 {sector} 的元数据失败: {e}")
            return False
    
    def update_metadata(self, sector: str, **kwargs) -> bool:
        """更新板块的元数据
        
        Args:
            sector: 板块名称
            **kwargs: 要更新的元数据字段
            
        Returns:
            是否更新成功
        """
        metadata = self.load_metadata(sector)
        
        # 更新元数据字段
        for key, value in kwargs.items():
            metadata[key] = value
        
        # 保存更新后的元数据
        return self.save_metadata(sector, metadata)
    
    def scan_feature_files(self, sector: str) -> List[Dict[str, Any]]:
        """扫描板块的特征文件
        
        Args:
            sector: 板块名称
            
        Returns:
            特征文件信息列表
        """
        sector_dir = self.base_dir / sector
        
        if not sector_dir.exists():
            return []
        
        # 扫描特征文件
        feature_files = []
        
        for file_path in sector_dir.glob("*.csv"):
            try:
                # 获取文件信息
                file_stats = file_path.stat()
                file_size = file_stats.st_size
                modified_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                # 尝试读取文件头部获取日期范围和记录数
                try:
                    # 只读取文件头部
                    df_head = pd.read_csv(file_path, nrows=5)
                    min_date = df_head['date'].min() if 'date' in df_head.columns else None
                    max_date = df_head['date'].max() if 'date' in df_head.columns else None
                    
                    # 统计记录数
                    record_count = sum(1 for _ in open(file_path)) - 1  # 减去标题行
                except Exception as e:
                    logger.warning(f"读取文件 {file_path} 头部失败: {e}")
                    min_date = None
                    max_date = None
                    record_count = 0
                
                # 添加文件信息
                feature_files.append({
                    'file': str(file_path.relative_to(self.base_dir)),
                    'size': file_size,
                    'modified': modified_time,
                    'record_count': record_count,
                    'min_date': min_date,
                    'max_date': max_date
                })
            except Exception as e:
                logger.error(f"处理文件 {file_path} 信息失败: {e}")
        
        return feature_files
    
    def update_sector_metadata(self, sector: str) -> bool:
        """更新板块元数据
        
        扫描板块的特征文件并更新元数据。
        
        Args:
            sector: 板块名称
            
        Returns:
            是否更新成功
        """
        # 扫描特征文件
        feature_files = self.scan_feature_files(sector)
        
        if not feature_files:
            logger.warning(f"板块 {sector} 没有特征文件")
            return False
        
        # 提取日期范围
        date_ranges = []
        total_records = 0
        total_size = 0
        
        for file_info in feature_files:
            min_date = file_info.get('min_date')
            max_date = file_info.get('max_date')
            
            if min_date and max_date:
                date_ranges.append([min_date, max_date])
            
            total_records += file_info.get('record_count', 0)
            total_size += file_info.get('size', 0)
        
        # 合并重叠的日期范围
        merged_ranges = self._merge_date_ranges(date_ranges)
        
        # 更新元数据
        return self.update_metadata(
            sector,
            feature_files=feature_files,
            date_ranges=merged_ranges,
            record_count=total_records,
            file_size=total_size
        )
    
    def _merge_date_ranges(self, date_ranges: List[List[str]]) -> List[List[str]]:
        """合并重叠的日期范围
        
        Args:
            date_ranges: 日期范围列表，每个范围是[min_date, max_date]
            
        Returns:
            合并后的日期范围列表
        """
        if not date_ranges:
            return []
        
        # 排序日期范围
        sorted_ranges = sorted(date_ranges, key=lambda x: x[0])
        
        merged = [sorted_ranges[0]]
        
        for current in sorted_ranges[1:]:
            previous = merged[-1]
            
            # 检查是否重叠
            if current[0] <= previous[1]:
                # 合并重叠范围
                previous[1] = max(previous[1], current[1])
            else:
                # 添加新范围
                merged.append(current)
        
        return merged
    
    def get_feature_info(self, sector: str) -> Dict[str, Any]:
        """获取板块特征信息
        
        Args:
            sector: 板块名称
            
        Returns:
            特征信息字典
        """
        # 加载元数据
        metadata = self.load_metadata(sector)
        
        # 如果元数据为空或过期，重新扫描
        if not metadata.get('feature_files'):
            self.update_sector_metadata(sector)
            metadata = self.load_metadata(sector)
        
        # 提取特征信息
        feature_info = {
            'sector': sector,
            'last_updated': metadata.get('last_updated', ''),
            'date_ranges': metadata.get('date_ranges', []),
            'record_count': metadata.get('record_count', 0),
            'file_size': metadata.get('file_size', 0),
            'file_count': len(metadata.get('feature_files', [])),
            'feature_version': metadata.get('feature_version', '1.0')
        }
        
        return feature_info
    
    def get_missing_dates(
        self, 
        sector: str, 
        start_date: str, 
        end_date: str
    ) -> List[str]:
        """获取需要处理的日期列表
        
        Args:
            sector: 板块名称
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            
        Returns:
            需要处理的日期列表
        """
        # 获取板块特征信息
        feature_info = self.get_feature_info(sector)
        
        # 获取已处理的日期范围
        processed_ranges = feature_info.get('date_ranges', [])
        
        # 尝试获取交易日历 - 使用data_loaders中的方法
        try:
            # 动态导入，避免循环依赖
            import importlib
            module = importlib.import_module('to.data_loaders')
            get_trading_days = getattr(module, 'get_trading_days')
            
            # 获取所有交易日
            all_trading_days = get_trading_days(start_date, end_date)
            logger.debug(f"获取到交易日期列表，类型: {type(all_trading_days)}")
            
            # 处理不同类型的交易日期返回值
            if isinstance(all_trading_days, pd.DataFrame):
                # 如果是DataFrame，提取date_str列或date列的值
                if 'date_str' in all_trading_days.columns:
                    date_list = all_trading_days['date_str'].astype(str).tolist()
                elif 'date' in all_trading_days.columns:
                    date_list = all_trading_days['date'].astype(str).tolist()
                else:
                    # 如果没有合适的列，尝试使用索引
                    date_list = [str(d).replace('-', '') for d in all_trading_days.index]
                logger.debug(f"从DataFrame中提取日期列表，共 {len(date_list)} 个日期")
            else:
                # 否则保持原样，假设是列表形式
                date_list = list(all_trading_days)
                
            # 过滤掉非法日期格式
            date_list = [d for d in date_list if isinstance(d, str) and d.isdigit() and len(d) == 8]
            logger.debug(f"过滤后的日期列表，共 {len(date_list)} 个有效日期")
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"无法导入get_trading_days函数: {e}")
            # 如果无法导入，则生成日期序列
            date_list = []
            try:
                start = datetime.strptime(start_date, "%Y%m%d")
                end = datetime.strptime(end_date, "%Y%m%d")
                delta = (end - start).days + 1
                
                # 生成日期序列
                for i in range(delta):
                    dt = start + timedelta(days=i)
                    # 跳过周末
                    if dt.weekday() < 5:  # 0-4表示周一至周五
                        date_list.append(dt.strftime("%Y%m%d"))
            except Exception as e:
                logger.error(f"生成日期序列时出错: {e}")
                return []
        
        # 如果processed_ranges为空，所有日期都需要处理
        if not processed_ranges:
            return date_list
        
        # 检查每个交易日是否已处理
        missing_dates = []
        
        for date_str in date_list:
            # 检查日期格式
            if not date_str.isdigit() or len(date_str) != 8:
                logger.warning(f"跳过无效日期格式: {date_str}")
                continue
                
            # 转换为整数进行比较
            date_int = int(date_str)
            is_processed = False
            
            for range_start, range_end in processed_ranges:
                if range_start <= date_int <= range_end:
                    is_processed = True
                    break
            
            # 如果日期未处理，添加到缺失列表
            if not is_processed:
                missing_dates.append(date_str)
        
        logger.info(f"板块 {sector} 在日期范围 {start_date} - {end_date} 中有 {len(missing_dates)} 个日期需要处理")
        return missing_dates
    
    def merge_feature_files(self, sector: str) -> bool:
        """合并板块的特征文件
        
        将同一板块的多个特征文件合并为一个文件，优化存储和读取效率。
        
        Args:
            sector: 板块名称
            
        Returns:
            是否合并成功
        """
        # 获取元数据
        metadata = self.load_metadata(sector)
        feature_files = metadata.get('feature_files', [])
        
        if len(feature_files) <= 1:
            logger.info(f"板块 {sector} 没有需要合并的特征文件")
            return True
        
        # 读取所有特征文件
        dfs = []
        sector_dir = self.base_dir / sector
        
        for file_info in feature_files:
            file_path = self.base_dir / file_info['file']
            
            try:
                df = load_features(file_path)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.error(f"读取文件 {file_path} 失败: {e}")
        
        if not dfs:
            logger.warning(f"板块 {sector} 没有有效的特征数据")
            return False
        
        # 合并数据帧
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # 按日期排序并去重
        merged_df = merged_df.sort_values('date').drop_duplicates(subset=['date'])
        
        # 优化数据类型
        merged_df = optimize_dtypes(merged_df)
        
        # 保存合并文件
        output_file = sector_dir / f"{sector}.csv"
        success = save_features(merged_df, output_file)
        
        if success:
            # 删除原始文件
            for file_info in feature_files:
                file_path = self.base_dir / file_info['file']
                
                if file_path != output_file and file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.warning(f"删除文件 {file_path} 失败: {e}")
            
            # 更新元数据
            self.update_sector_metadata(sector)
        
        return success


def get_feature_update_manager(base_dir: Optional[Union[str, Path]] = None) -> FeatureUpdateManager:
    """获取特征更新管理器单例
    
    Args:
        base_dir: 特征数据根目录，None则使用默认目录
        
    Returns:
        特征更新管理器实例
    """
    global _FEATURE_UPDATE_MANAGER
    
    if _FEATURE_UPDATE_MANAGER is None:
        _FEATURE_UPDATE_MANAGER = FeatureUpdateManager(base_dir)
    
    return _FEATURE_UPDATE_MANAGER


def update_features_incrementally(
    sector: str,
    start_date: str,
    end_date: str,
    base_dir: Optional[Union[str, Path]] = None
) -> List[str]:
    """增量更新特征数据
    
    Args:
        sector: 板块名称
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD
        base_dir: 特征数据根目录，None则使用默认目录
        
    Returns:
        需要处理的日期列表
    """
    manager = get_feature_update_manager(base_dir)
    return manager.get_missing_dates(sector, start_date, end_date)


def merge_feature_files(sector: str, base_dir: Optional[Union[str, Path]] = None) -> bool:
    """合并板块的特征文件
    
    Args:
        sector: 板块名称
        base_dir: 特征数据根目录，None则使用默认目录
        
    Returns:
        是否合并成功
    """
    manager = get_feature_update_manager(base_dir)
    return manager.merge_feature_files(sector)


def get_feature_info(sector: str, base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """获取板块特征信息
    
    Args:
        sector: 板块名称
        base_dir: 特征数据根目录，None则使用默认目录
        
    Returns:
        特征信息字典
    """
    manager = get_feature_update_manager(base_dir)
    return manager.get_feature_info(sector)

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 测试增量更新
    # 创建测试数据
    dates = [f"2023{month:02d}{day:02d}" for month in range(1, 4) for day in range(1, 28) if day % 7 not in [5, 6]]
    data = []
    for date in dates:
        data.append({
            'date': date,
            'value1': np.random.randn(),
            'value2': np.random.rand() * 100,
            'value3': np.random.choice(['A', 'B', 'C'])
        })
    
    test_df = pd.DataFrame(data)
    
    # 增量更新特征
    manager = FeatureUpdateManager(get_data_path('test_features'))
    
    # 先添加一部分数据
    first_half = test_df.iloc[:len(test_df)//2]
    success = manager.add_features('test_feature', first_half)
    logger.info(f"添加第一批数据: {'成功' if success else '失败'}")
    
    # 尝试添加重复数据
    duplicate = first_half.iloc[:10]
    success = manager.add_features('test_feature', duplicate)
    logger.info(f"添加重复数据: {'成功' if success else '失败'}")
    
    # 添加新数据
    second_half = test_df.iloc[len(test_df)//2:]
    success = manager.add_features('test_feature', second_half)
    logger.info(f"添加第二批数据: {'成功' if success else '失败'}")
    
    # 获取特征信息
    info = manager.get_feature_info('test_feature')
    logger.info(f"特征信息: {info}")
    
    # 获取缺失日期
    missing = manager.get_missing_dates('test_feature', '20230101', '20230331')
    logger.info(f"缺失日期: {len(missing)} 个")
    
    # 合并特征文件
    success = manager.merge_feature_files('test_feature')
    logger.info(f"合并特征文件: {'成功' if success else '失败'}")
    
    # 再次获取特征信息
    info = manager.get_feature_info('test_feature')
    logger.info(f"合并后特征信息: {info}") 