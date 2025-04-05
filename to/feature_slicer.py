#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征切片器模块

提供高效的特征数据切片和处理功能，支持内存映射、增量更新和压缩优化
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import mmap
import joblib
import gc

# 导入路径工具
from to.path_utils import get_data_path, ensure_dir

# 导入特征加载器
from to.feature_loader import load_sector_features, load_sectors_features

# 设置日志
logger = logging.getLogger(__name__)

# 缓存
_SLICE_CACHE = {}
_MMAP_HANDLES = {}

# 特征分组定义
BASIC_FEATURES = ['open_avg', 'high_avg', 'low_avg', 'close_avg', 'volume_avg', 'amount_avg']
TECHNICAL_FEATURES = ['ma5_avg', 'ma10_avg', 'ma20_avg', 'ma60_avg', 'price_above_ma5_ratio', 'price_above_ma20_ratio']
MOMENTUM_FEATURES = ['change_avg']
ACTIVITY_FEATURES = ['active_count', 'stock_count']
ALL_FEATURE_GROUPS = {
    'basic': BASIC_FEATURES,
    'technical': TECHNICAL_FEATURES,
    'momentum': MOMENTUM_FEATURES,
    'activity': ACTIVITY_FEATURES
}


class FeatureSlicer:
    """特征切片器类，提供高效的特征数据处理功能"""
    
    def __init__(self, use_mmap: bool = False, use_compression: bool = False):
        """初始化特征切片器
        
        Args:
            use_mmap: 是否使用内存映射
            use_compression: 是否使用压缩
        """
        self.use_mmap = use_mmap
        self.use_compression = use_compression
        
        # 缓存
        self._cache = {}
        # 内存映射句柄
        self._mmap_handles = {}
        
        logger.info(f"初始化特征切片器：内存映射={use_mmap}，压缩={use_compression}")
    
    def get_time_slice(
        self, 
        sector: str,
        start_date: str,
        end_date: str,
        features: List[str] = None,
        normalize: bool = False
    ) -> pd.DataFrame:
        """获取指定时间段的特征切片
        
        Args:
            sector: 板块名称
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            features: 需要的特征列表，默认为None表示所有特征
            normalize: 是否对特征进行归一化
            
        Returns:
            包含特征的DataFrame
        """
        cache_key = f"{sector}_{start_date}_{end_date}_{'-'.join(features) if features else 'all'}"
        
        # 检查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 加载特征数据
        df = load_sector_features(
            sector=sector,
            start_date=start_date,
            end_date=end_date,
            features=features
        )
        
        # 如果需要归一化
        if normalize and not df.empty:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            # 排除不需要归一化的列
            exclude_cols = ['date', 'sector_name', 'stock_count']
            numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
            
            if numeric_columns.size > 0:
                # 对数值型特征进行归一化
                df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std().replace(0, 1)
                logger.debug(f"对 {sector} 板块的 {len(numeric_columns)} 个特征进行了归一化")
        
        # 更新缓存
        self._cache[cache_key] = df
        
        return df
    
    def get_feature_group(
        self,
        sector: str,
        group: str,
        start_date: str = None,
        end_date: str = None,
        normalize: bool = False
    ) -> pd.DataFrame:
        """获取特征分组
        
        Args:
            sector: 板块名称
            group: 特征分组名称，可选值为: 'basic', 'technical', 'momentum', 'activity', 'all'
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            normalize: 是否对特征进行归一化
            
        Returns:
            包含特征的DataFrame
        """
        if group == 'all':
            # 获取所有特征
            features = None
        elif group in ALL_FEATURE_GROUPS:
            # 获取指定分组的特征
            features = ALL_FEATURE_GROUPS[group]
        else:
            # 无效的分组名称
            logger.warning(f"无效的特征分组名称: {group}，可选值为: 'basic', 'technical', 'momentum', 'activity', 'all'")
            return pd.DataFrame()
        
        # 获取时间切片
        return self.get_time_slice(
            sector=sector,
            start_date=start_date,
            end_date=end_date,
            features=features,
            normalize=normalize
        )
    
    def get_window_slice(
        self,
        sector: str,
        target_date: str,
        window_size: int,
        features: List[str] = None,
        include_target_date: bool = True
    ) -> pd.DataFrame:
        """获取时间窗口特征切片
        
        Args:
            sector: 板块名称
            target_date: 目标日期，格式为YYYYMMDD
            window_size: 窗口大小（天数）
            features: 需要的特征列表，默认为None表示所有特征
            include_target_date: 是否包含目标日期
            
        Returns:
            包含特征的DataFrame
        """
        # 计算开始日期
        target_dt = datetime.strptime(target_date, '%Y%m%d')
        start_dt = target_dt - timedelta(days=window_size * 2)  # 获取更多数据以确保窗口大小
        start_date = start_dt.strftime('%Y%m%d')
        
        # 获取特征数据
        df = self.get_time_slice(
            sector=sector,
            start_date=start_date,
            end_date=target_date,
            features=features
        )
        
        if df.empty:
            return df
        
        # 确保日期列是字符串格式
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str)
        
        # 获取最接近目标日期的数据
        all_dates = sorted(df['date'].unique())
        
        if not include_target_date:
            # 排除目标日期
            all_dates = [d for d in all_dates if d < target_date]
        
        # 确保有足够的数据
        if len(all_dates) < window_size:
            logger.warning(f"板块 {sector} 在目标日期 {target_date} 之前只有 {len(all_dates)} 天的数据，少于请求的窗口大小 {window_size}")
            return df[df['date'].isin(all_dates)]
        
        # 获取窗口数据
        window_dates = all_dates[-window_size:]
        window_df = df[df['date'].isin(window_dates)]
        
        return window_df
    
    def get_sequence_data(
        self,
        sector: str,
        sequence_length: int,
        features: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        step: int = 1
    ) -> List[np.ndarray]:
        """生成序列数据，用于时间序列模型
        
        Args:
            sector: 板块名称
            sequence_length: 序列长度
            features: 需要的特征列表，默认为None表示所有特征
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            step: 滑动窗口步长
            
        Returns:
            序列数据列表，每个元素为一个形状为(sequence_length, feature_count)的numpy数组
        """
        # 获取特征数据
        df = self.get_time_slice(
            sector=sector,
            start_date=start_date,
            end_date=end_date,
            features=features
        )
        
        if df.empty:
            return []
        
        # 确保日期列是字符串格式
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str)
        
        # 按日期排序
        df = df.sort_values('date')
        
        # 排除非特征列
        exclude_cols = ['date', 'sector_name']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 转换为numpy数组
        data = df[feature_cols].values
        
        # 生成序列数据
        sequences = []
        for i in range(0, len(data) - sequence_length + 1, step):
            sequences.append(data[i:i+sequence_length])
        
        return sequences
    
    def save_slice(
        self,
        df: pd.DataFrame,
        output_path: str,
        compress: bool = None
    ) -> bool:
        """保存特征切片到文件
        
        Args:
            df: 特征DataFrame
            output_path: 输出路径
            compress: 是否压缩，默认为None表示使用初始化时的设置
            
        Returns:
            是否保存成功
        """
        if compress is None:
            compress = self.use_compression
        
        try:
            # 确保目录存在
            output_dir = os.path.dirname(output_path)
            ensure_dir(output_dir)
            
            if compress:
                # 使用joblib保存压缩数据
                joblib.dump(df, output_path, compress=3)
                logger.info(f"已将压缩特征切片保存到: {output_path}")
            else:
                # 直接保存CSV
                df.to_csv(output_path, index=False)
                logger.info(f"已将特征切片保存到: {output_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"保存特征切片失败: {e}")
            return False
    
    def load_slice(
        self,
        input_path: str,
        use_mmap: bool = None
    ) -> pd.DataFrame:
        """从文件加载特征切片
        
        Args:
            input_path: 输入路径
            use_mmap: 是否使用内存映射，默认为None表示使用初始化时的设置
            
        Returns:
            特征DataFrame
        """
        if use_mmap is None:
            use_mmap = self.use_mmap
        
        try:
            # 检查文件是否存在
            if not os.path.exists(input_path):
                logger.warning(f"特征切片文件不存在: {input_path}")
                return pd.DataFrame()
            
            # 检查文件扩展名
            file_ext = os.path.splitext(input_path)[1].lower()
            
            if file_ext == '.csv':
                # 加载CSV文件
                if use_mmap:
                    # 使用内存映射
                    return self._load_csv_with_mmap(input_path)
                else:
                    # 直接加载
                    return pd.read_csv(input_path)
            
            elif file_ext in ['.jl', '.joblib', '.pkl']:
                # 加载joblib压缩文件
                return joblib.load(input_path)
            
            else:
                logger.warning(f"不支持的文件格式: {file_ext}")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"加载特征切片失败: {e}")
            return pd.DataFrame()
    
    def _load_csv_with_mmap(self, file_path: str) -> pd.DataFrame:
        """使用内存映射加载CSV文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            DataFrame
        """
        try:
            # 如果已经有映射句柄，关闭它
            if file_path in self._mmap_handles:
                self._mmap_handles[file_path].close()
            
            # 打开文件
            with open(file_path, 'r+b') as f:
                # 创建内存映射
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
                # 保存句柄
                self._mmap_handles[file_path] = mm
                
                # 使用StringIO读取映射数据
                from io import StringIO
                s = StringIO(mm.read().decode('utf-8'))
                
                # 使用pandas读取
                df = pd.read_csv(s)
                
                return df
        
        except Exception as e:
            logger.error(f"使用内存映射加载CSV文件失败: {e}")
            return pd.read_csv(file_path)  # 回退到普通加载
    
    def close_mmap_handles(self):
        """关闭所有内存映射句柄"""
        for file_path, handle in self._mmap_handles.items():
            try:
                handle.close()
                logger.debug(f"关闭内存映射句柄: {file_path}")
            except Exception as e:
                logger.warning(f"关闭内存映射句柄失败: {file_path}, 错误: {e}")
        
        # 清空句柄字典
        self._mmap_handles = {}
    
    def clear_cache(self):
        """清除缓存"""
        self._cache = {}
        gc.collect()  # 执行垃圾回收
        logger.info("特征切片器缓存已清除")
    
    def __del__(self):
        """析构函数，清理资源"""
        self.close_mmap_handles()


# 全局特征切片器实例
_global_slicer = None


def get_slicer(use_mmap: bool = False, use_compression: bool = False) -> FeatureSlicer:
    """获取全局特征切片器实例
    
    Args:
        use_mmap: 是否使用内存映射
        use_compression: 是否使用压缩
        
    Returns:
        FeatureSlicer实例
    """
    global _global_slicer
    
    if _global_slicer is None:
        _global_slicer = FeatureSlicer(use_mmap=use_mmap, use_compression=use_compression)
    
    return _global_slicer


def get_time_slice(*args, **kwargs) -> pd.DataFrame:
    """获取指定时间段的特征切片，包装全局特征切片器
    
    参数同FeatureSlicer.get_time_slice()
    """
    slicer = get_slicer()
    return slicer.get_time_slice(*args, **kwargs)


def get_feature_group(*args, **kwargs) -> pd.DataFrame:
    """获取特征分组，包装全局特征切片器
    
    参数同FeatureSlicer.get_feature_group()
    """
    slicer = get_slicer()
    return slicer.get_feature_group(*args, **kwargs)


def get_window_slice(*args, **kwargs) -> pd.DataFrame:
    """获取时间窗口特征切片，包装全局特征切片器
    
    参数同FeatureSlicer.get_window_slice()
    """
    slicer = get_slicer()
    return slicer.get_window_slice(*args, **kwargs)


def get_sequence_data(*args, **kwargs) -> List[np.ndarray]:
    """生成序列数据，包装全局特征切片器
    
    参数同FeatureSlicer.get_sequence_data()
    """
    slicer = get_slicer()
    return slicer.get_sequence_data(*args, **kwargs)


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 测试特征切片器
    slicer = FeatureSlicer()
    
    # 获取特征数据
    sector = "TGN人工智能"
    start_date = "20230110"
    end_date = "20230120"
    
    print(f"加载 {sector} 板块 {start_date} 到 {end_date} 的特征数据...")
    
    # 获取基本特征
    df_basic = slicer.get_feature_group(sector, 'basic', start_date, end_date)
    print(f"\n基本特征 ({len(df_basic)} 行):")
    print(df_basic.head())
    
    # 获取技术指标
    df_tech = slicer.get_feature_group(sector, 'technical', start_date, end_date)
    print(f"\n技术指标 ({len(df_tech)} 行):")
    print(df_tech.head())
    
    # 获取窗口数据
    window_size = 5
    target_date = "20230120"
    df_window = slicer.get_window_slice(sector, target_date, window_size)
    print(f"\n窗口数据 (目标日期: {target_date}, 窗口大小: {window_size}, {len(df_window)} 行):")
    print(df_window)
    
    # 生成序列数据
    sequences = slicer.get_sequence_data(sector, 3, BASIC_FEATURES, start_date, end_date)
    print(f"\n序列数据 (序列长度: 3, 特征数: {len(BASIC_FEATURES)}, 序列数: {len(sequences)})")
    if sequences:
        print(f"第一个序列形状: {sequences[0].shape}")
        print(sequences[0]) 