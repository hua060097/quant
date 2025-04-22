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
import re

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
        _processed_dates_cache: 处理日期缓存
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
        
        # 处理日期缓存 - 提高查询效率
        self._processed_dates_cache = {}
    
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
                'processed_dates': []  # 已处理日期列表，取代之前的date_ranges等字段
            }
        else:
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                # 确保processed_dates字段存在
                if 'processed_dates' not in metadata:
                    # 尝试从旧版元数据中提取日期信息
                    if 'date_ranges' in metadata:
                        processed_dates = set()
                        for date_range in metadata.get('date_ranges', []):
                            if len(date_range) >= 2:
                                # 尝试提取日期范围内的所有日期
                                try:
                                    start_date = self._normalize_date(date_range[0])
                                    end_date = self._normalize_date(date_range[1])
                                    if start_date and end_date and start_date < end_date:
                                        # 获取日期范围内的所有交易日
                                        try:
                                            import importlib
                                            module = importlib.import_module('to.data_loaders')
                                            get_trading_days = getattr(module, 'get_trading_days')
                                            
                                            start_str = str(start_date)
                                            end_str = str(end_date)
                                            
                                            trading_days = get_trading_days(start_str, end_str)
                                            # 处理不同返回类型
                                            if isinstance(trading_days, pd.DataFrame):
                                                if 'date_str' in trading_days.columns:
                                                    days = trading_days['date_str'].astype(str).tolist()
                                                elif 'date' in trading_days.columns:
                                                    days = trading_days['date'].astype(str).tolist()
                                                else:
                                                    days = [str(d).replace('-', '') for d in trading_days.index]
                                            else:
                                                days = [str(d) for d in trading_days]
                                                
                                            processed_dates.update(days)
                                        except:
                                            # 无法获取交易日，使用简单的日期范围
                                            start_dt = datetime.strptime(str(start_date), "%Y%m%d")
                                            end_dt = datetime.strptime(str(end_date), "%Y%m%d")
                                            delta = (end_dt - start_dt).days + 1
                                            
                                            for i in range(delta):
                                                dt = start_dt + timedelta(days=i)
                                                # 跳过周末
                                                if dt.weekday() < 5:  # 0-4表示周一至周五
                                                    processed_dates.add(dt.strftime("%Y%m%d"))
                                except Exception as e:
                                    logger.warning(f"无法从日期范围提取日期: {date_range}, 错误: {e}")
                                
                        metadata['processed_dates'] = sorted(list(processed_dates))
                    else:
                        metadata['processed_dates'] = []
            except Exception as e:
                logger.error(f"加载板块 {sector} 的元数据失败: {e}")
                metadata = {
                    'sector': sector,
                    'last_updated': '',
                    'processed_dates': []
                }
        
        # 更新缓存
        self._metadata_cache[sector] = metadata
        
        # 更新处理日期缓存
        self._processed_dates_cache[sector] = set(metadata.get('processed_dates', []))
        
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
        
        # 同时扫描.csv和.parquet文件
        for file_pattern in ["*.csv", "*.parquet"]:
            for file_path in sector_dir.glob(file_pattern):
                try:
                    # 获取文件信息
                    file_stats = file_path.stat()
                    file_size = file_stats.st_size
                    modified_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 尝试读取文件头部获取日期范围和记录数
                    try:
                        # 根据文件格式选择读取方法
                        if file_path.suffix.lower() == '.csv':
                            df_head = pd.read_csv(file_path, nrows=5)
                            # 统计记录数
                            record_count = sum(1 for _ in open(file_path)) - 1  # 减去标题行
                        else:  # parquet文件
                            df_head = pd.read_parquet(file_path, columns=['date']).head(5)
                            # parquet不能用简单的行数统计，需要读取文件
                            full_df = pd.read_parquet(file_path)
                            record_count = len(full_df)
                        
                        min_date = df_head['date'].min() if 'date' in df_head.columns else None
                        max_date = df_head['date'].max() if 'date' in df_head.columns else None
                        
                    except Exception as e:
                        logger.warning(f"读取文件 {file_path} 头部失败: {e}")
                        # 尝试从文件名提取日期信息(如 sector_20230101.parquet)
                        try:
                            # 从文件名提取日期
                            date_pattern = re.search(r'_(\d{8})\.', file_path.name)
                            if date_pattern:
                                date_str = date_pattern.group(1)
                                min_date = date_str
                                max_date = date_str
                            else:
                                min_date = None
                                max_date = None
                        except:
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
    
    def update_sector_metadata(self, sector: str, processed_dates: Optional[List[str]] = None) -> bool:
        """更新板块元数据
        
        更新板块特征文件的元数据信息。
        
        Args:
            sector: 板块名称
            processed_dates: 已处理日期列表，如果为None则不更新
            
        Returns:
            是否更新成功
        """
        # 加载现有元数据
        metadata = self.load_metadata(sector)
        
        # 如果提供了已处理日期，则更新
        if processed_dates is not None:
            # 标准化所有日期
            norm_dates = [str(self._normalize_date(d) or d) for d in processed_dates]
            
            # 添加到现有处理日期
            existing_dates = set(metadata.get('processed_dates', []))
            existing_dates.update(norm_dates)
            
            # 更新元数据
            metadata['processed_dates'] = sorted(existing_dates)
            
            # 更新缓存
            self._processed_dates_cache[sector] = existing_dates
        
        # 更新最后修改时间
        metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存元数据
        return self.save_metadata(sector, metadata)
    
    def get_feature_info(self, sector: str) -> Dict[str, Any]:
        """获取板块特征信息
        
        Args:
            sector: 板块名称
            
        Returns:
            特征信息字典
        """
        # 加载元数据
        metadata = self.load_metadata(sector)
        
        # 提取特征信息
        feature_info = {
            'sector': sector,
            'last_updated': metadata.get('last_updated', ''),
            'processed_days': len(metadata.get('processed_dates', [])),
            'date_range': self._get_date_range_from_processed(metadata.get('processed_dates', []))
        }
        
        return feature_info
    
    def _get_date_range_from_processed(self, processed_dates: List[str]) -> List[str]:
        """从已处理日期列表获取日期范围
        
        Args:
            processed_dates: 已处理日期列表
            
        Returns:
            [最早日期, 最晚日期]，如果列表为空则返回[]
        """
        if not processed_dates:
            return []
        
        try:
            # 确保日期格式统一，并排序
            norm_dates = sorted([str(self._normalize_date(d) or d) for d in processed_dates])
            return [norm_dates[0], norm_dates[-1]]
        except Exception as e:
            logger.error(f"处理日期范围时出错: {e}")
            return []
            
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
        # 获取所有交易日
        all_trading_days = self._get_trading_days(start_date, end_date)
        
        # 过滤出未处理的日期
        missing_dates = []
        
        for date_str in all_trading_days:
            # 检查日期格式
            if not date_str.isdigit() or len(date_str) != 8:
                logger.warning(f"跳过无效日期格式: {date_str}")
                continue
                
            # 直接使用高效的is_date_processed方法检查
            if not self.is_date_processed(sector, date_str):
                missing_dates.append(date_str)
        
        logger.info(f"板块 {sector} 在日期范围 {start_date} - {end_date} 中有 {len(missing_dates)} 个日期需要处理")
        return missing_dates
    
    def _get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            
        Returns:
            交易日列表
        """
        try:
            # 动态导入，避免循环依赖
            import importlib
            module = importlib.import_module('to.data_loaders')
            get_trading_days = getattr(module, 'get_trading_days')
            
            # 获取所有交易日
            all_trading_days = get_trading_days(start_date, end_date)
            
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
            else:
                # 否则保持原样，假设是列表形式
                date_list = list(all_trading_days)
                
            # 过滤掉非法日期格式
            date_list = [d for d in date_list if isinstance(d, str) and d.isdigit() and len(d) == 8]
            
            return date_list
            
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
                
            return date_list
    
    def _normalize_date(self, date_value) -> int:
        """将各种格式的日期标准化为整数格式(YYYYMMDD)
        
        Args:
            date_value: 日期值，可以是字符串、整数或其他可转换格式
            
        Returns:
            整数格式的日期，如20230101
            如果无法转换则返回None
        """
        if date_value is None:
            return None
            
        try:
            # 已经是整数
            if isinstance(date_value, int):
                # 检查是否是有效的8位日期
                if 19000101 <= date_value <= 21000101 and len(str(date_value)) == 8:
                    return date_value
            
            # 字符串格式
            if isinstance(date_value, str):
                # 如果是数字字符串且长度为8，直接转换
                if date_value.isdigit() and len(date_value) == 8:
                    return int(date_value)
                
                # 尝试解析其他格式的日期字符串
                try:
                    return int(pd.to_datetime(date_value).strftime('%Y%m%d'))
                except:
                    pass
            
            # 尝试pandas转换
            return int(pd.to_datetime(date_value).strftime('%Y%m%d'))
            
        except Exception as e:
            logger.warning(f"无法将日期值 {date_value} 转换为标准格式: {e}")
            return None
            
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
    
    def scan_dates_from_files(self, sector: str) -> List[str]:
        """从单日特征文件名中扫描日期
        
        扫描板块目录下的单日特征文件，从文件名中提取日期信息。
        适用于元数据缺失但文件已存在的情况。
        
        Args:
            sector: 板块名称
            
        Returns:
            提取的日期列表
        """
        sector_dir = self.base_dir / sector
        processed_dates = set()
        
        if not sector_dir.exists():
            logger.warning(f"板块 {sector} 的目录不存在")
            return []
        
        # 步骤1: 查找并处理单日特征文件
        # 查找所有单日特征文件，格式如：TGN人工智能_20230101.parquet
        for ext in ['parquet', 'csv']:
            file_pattern = f"{sector}_*.{ext}"
            for file_path in sector_dir.glob(file_pattern):
                try:
                    # 从文件名中提取日期（如：TGN人工智能_20230101.parquet -> 20230101）
                    date_match = re.search(r'_(\d{8})\.(parquet|csv)', file_path.name)
                    if date_match:
                        date_str = date_match.group(1)
                        # 验证日期格式
                        if date_str.isdigit() and len(date_str) == 8:
                            try:
                                # 尝试解析日期以确保有效
                                year = int(date_str[0:4])
                                month = int(date_str[4:6])
                                day = int(date_str[6:8])
                                if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                                    processed_dates.add(date_str)
                                    logger.debug(f"从文件 {file_path.name} 中提取到日期: {date_str}")
                            except ValueError:
                                # 跳过无效日期
                                pass
                except Exception as e:
                    logger.warning(f"处理文件 {file_path} 时出错: {e}")
        
        # 步骤2: 尝试加载合并的主文件
        main_file_patterns = [
            f"{sector}.parquet",           # 原始合并文件
            f"{sector}_features.parquet",  # 优化后的主文件
        ]
        
        for pattern in main_file_patterns:
            main_files = list(sector_dir.glob(pattern))
            for main_file in main_files:
                try:
                    # 仅读取date列以提高效率
                    df = pd.read_parquet(main_file, columns=['date'])
                    
                    if not df.empty and 'date' in df.columns:
                        # 尝试将日期列转换为datetime格式
                        if not pd.api.types.is_datetime64_any_dtype(df['date']):
                            try:
                                df['date'] = pd.to_datetime(df['date'])
                            except:
                                pass
                        
                        # 提取所有日期并转换为标准格式
                        for date_val in df['date']:
                            try:
                                if isinstance(date_val, pd.Timestamp):
                                    # 对于Timestamp对象，直接格式化
                                    date_str = date_val.strftime('%Y%m%d')
                                    processed_dates.add(date_str)
                                else:
                                    # 对于其他格式，尝试标准化
                                    norm_date = self._normalize_date(str(date_val))
                                    if norm_date:
                                        processed_dates.add(str(norm_date))
                            except Exception as e:
                                logger.debug(f"处理日期 {date_val} 时出错: {e}")
                    
                    logger.info(f"从主文件 {main_file} 中提取到 {len(processed_dates)} 个处理日期")
                except Exception as e:
                    logger.warning(f"处理主文件 {main_file} 时出错: {e}")
        
        # 步骤3: 检查最新特征CSV文件
        latest_csv = sector_dir / f"{sector}_latest.csv"
        if latest_csv.exists():
            try:
                df = pd.read_csv(latest_csv)
                if 'date' in df.columns and not df.empty:
                    for date_val in df['date'].astype(str):
                        try:
                            norm_date = self._normalize_date(date_val)
                            if norm_date:
                                processed_dates.add(str(norm_date))
                        except:
                            pass
                logger.info(f"从最新特征CSV文件中提取到 {len(processed_dates)} 个处理日期")
            except Exception as e:
                logger.warning(f"处理最新特征CSV文件时出错: {e}")
        
        # 返回排序后的日期列表
        return sorted(list(processed_dates))
    
    def is_date_processed(self, sector: str, date_str: str) -> bool:
        """检查日期是否已处理
        
        高效检查日期是否已被处理，避免重复生成特征
        
        Args:
            sector: 板块名称
            date_str: 日期字符串，格式为YYYYMMDD
            
        Returns:
            是否已处理
        """
        # 确保date_str是标准格式
        date_str = str(self._normalize_date(date_str) or date_str)
        
        # 加载处理日期缓存
        if sector not in self._processed_dates_cache:
            metadata = self.load_metadata(sector)
            processed_dates = set(metadata.get('processed_dates', []))
            
            # 如果processed_dates为空，尝试从文件名中扫描日期
            if not processed_dates:
                logger.info(f"板块 {sector} 元数据中的处理日期为空，尝试从特征文件中扫描日期")
                scanned_dates = self.scan_dates_from_files(sector)
                
                if scanned_dates:
                    logger.info(f"从特征文件中扫描到 {len(scanned_dates)} 个处理日期")
                    processed_dates = set(scanned_dates)
                    
                    # 更新元数据
                    metadata['processed_dates'] = sorted(processed_dates)
                    self.save_metadata(sector, metadata)
                    logger.info(f"已更新板块 {sector} 的元数据，添加了 {len(scanned_dates)} 个扫描到的日期")
            
            self._processed_dates_cache[sector] = processed_dates
        
        # O(1)复杂度查询是否存在
        return date_str in self._processed_dates_cache[sector]
    
    def add_processed_date(self, sector: str, date_str: str) -> bool:
        """添加已处理日期
        
        Args:
            sector: 板块名称
            date_str: 日期字符串，格式为YYYYMMDD
            
        Returns:
            是否成功添加
        """
        # 标准化日期格式
        norm_date = str(self._normalize_date(date_str) or date_str)
        
        # 检查是否已存在
        if self.is_date_processed(sector, norm_date):
            return True
        
        # 加载元数据
        metadata = self.load_metadata(sector)
        processed_dates = list(metadata.get('processed_dates', []))
        
        # 添加新日期
        processed_dates.append(norm_date)
        processed_dates = sorted(set(processed_dates))  # 去重并排序
        
        # 更新元数据
        metadata['processed_dates'] = processed_dates
        metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 更新缓存
        self._processed_dates_cache[sector] = set(processed_dates)
        
        # 保存元数据
        return self.save_metadata(sector, metadata)
    
    def add_processed_dates(self, sector: str, date_list: List[str]) -> bool:
        """批量添加已处理日期
        
        Args:
            sector: 板块名称
            date_list: 日期字符串列表
            
        Returns:
            是否成功添加
        """
        if not date_list:
            return True
        
        # 标准化日期格式
        norm_dates = [str(self._normalize_date(d) or d) for d in date_list]
        
        # 加载元数据
        metadata = self.load_metadata(sector)
        processed_dates = set(metadata.get('processed_dates', []))
        
        # 添加新日期
        processed_dates.update(norm_dates)
        
        # 更新元数据
        metadata['processed_dates'] = sorted(processed_dates)
        metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 更新缓存
        self._processed_dates_cache[sector] = processed_dates
        
        # 保存元数据
        return self.save_metadata(sector, metadata)


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
    
    检查日期范围内哪些日期需要处理，避免重复计算。
    
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