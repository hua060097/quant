#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征加载器模块

提供按需加载特定日期和板块的特征的功能，支持内存缓存和惰性加载
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# 导入路径工具
from to.path_utils import get_data_path, ensure_dir

# 导入板块股票映射
from to.stock_sectors_mapping import get_stocks_in_sector, get_all_sectors

# 设置日志
logger = logging.getLogger(__name__)

# 缓存
_FEATURE_CACHE = {}


def load_sector_features(
    sector: str,
    start_date: str = None,
    end_date: str = None,
    features: List[str] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """加载板块特征数据
    
    Args:
        sector: 板块名称
        start_date: 开始日期，格式为YYYYMMDD，默认为None表示从最早日期开始
        end_date: 结束日期，格式为YYYYMMDD，默认为None表示到最新日期结束
        features: 需要加载的特征列表，默认为None表示加载所有特征
        use_cache: 是否使用缓存
        
    Returns:
        包含板块特征的DataFrame
    """
    global _FEATURE_CACHE
    
    # 检查缓存
    cache_key = f"{sector}_{start_date}_{end_date}_{'-'.join(features) if features else 'all'}"
    if use_cache and cache_key in _FEATURE_CACHE:
        df = _FEATURE_CACHE[cache_key]
        return df
    
    # 构建板块特征文件路径
    sector_path = get_data_path('sector_features') / sector
    
    # 如果板块目录不存在，返回空DataFrame
    if not sector_path.exists():
        logger.warning(f"板块 {sector} 特征目录不存在: {sector_path}")
        return pd.DataFrame()
    
    # 尝试读取Parquet特征文件
    try:
        # 优先使用Parquet文件，更高效
        parquet_file = sector_path / f"{sector}_features.parquet"
        
        # 检查文件是否存在
        if not parquet_file.exists():
            logger.warning(f"板块 {sector} Parquet特征文件不存在: {parquet_file}")
            # 回退到查找老格式的CSV文件
            return _load_legacy_csv_features(sector, sector_path, start_date, end_date, features)
        
        # 读取Parquet文件，有日期过滤时更高效
        filters = []
        if start_date:
            # 转换为日期格式
            start_dt = pd.to_datetime(start_date, format='%Y%m%d')
            filters.append(('date', '>=', start_dt))
            
        if end_date:
            # 转换为日期格式
            end_dt = pd.to_datetime(end_date, format='%Y%m%d')
            filters.append(('date', '<=', end_dt))
            
        # 根据是否有过滤条件和特征列表决定如何读取
        if filters and features:
            # 有日期过滤和特征过滤
            df = pd.read_parquet(
                parquet_file, 
                filters=filters, 
                columns=['date'] + features
            )
        elif filters:
            # 只有日期过滤
            df = pd.read_parquet(parquet_file, filters=filters)
        elif features:
            # 只有特征过滤
            df = pd.read_parquet(parquet_file, columns=['date'] + features)
        else:
            # 无过滤
            df = pd.read_parquet(parquet_file)
        
        # 确保日期列是日期类型
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # 按日期排序
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        # 更新缓存
        if use_cache:
            _FEATURE_CACHE[cache_key] = df.copy()
        
        logger.info(f"从Parquet文件加载板块 {sector} 特征数据，共 {len(df)} 行")
        return df
    
    except Exception as e:
        logger.error(f"从Parquet文件加载板块 {sector} 特征失败: {e}")
        # 回退到查找老格式的CSV文件
        return _load_legacy_csv_features(sector, sector_path, start_date, end_date, features)


def _load_legacy_csv_features(sector: str, sector_path: Path, start_date: str = None, end_date: str = None, features: List[str] = None) -> pd.DataFrame:
    """加载传统格式的CSV特征文件(向后兼容)
    
    Args:
        sector: 板块名称
        sector_path: 板块目录路径
        start_date: 开始日期
        end_date: 结束日期
        features: 特征列表
        
    Returns:
        特征DataFrame
    """
    try:
        # 查找可能的CSV特征文件
        feature_file = sector_path / f"{sector}_features.csv"
        date_range_files = list(sector_path.glob("*.csv"))
        
        if feature_file.exists():
            # 使用主特征文件
            df = pd.read_csv(feature_file)
        elif date_range_files:
            # 合并所有日期范围文件
            dfs = []
            for file in date_range_files:
                if not file.name.endswith('_latest.csv'):  # 排除latest文件
                    try:
                        temp_df = pd.read_csv(file)
                        dfs.append(temp_df)
                    except Exception as file_e:
                        logger.warning(f"读取文件 {file} 失败: {file_e}")
            
            if not dfs:
                logger.warning(f"板块 {sector} 没有有效的CSV特征文件")
                return pd.DataFrame()
                
            # 合并并去重
            df = pd.concat(dfs, ignore_index=True)
            df = df.drop_duplicates(subset=['date'], keep='last')
        else:
            logger.warning(f"板块 {sector} 没有找到特征文件")
            return pd.DataFrame()
        
        # 确保日期列是日期时间类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 按日期筛选
        if start_date:
            start_dt = pd.to_datetime(start_date, format='%Y%m%d')
            df = df[df['date'] >= start_dt]
            
        if end_date:
            end_dt = pd.to_datetime(end_date, format='%Y%m%d')
            df = df[df['date'] <= end_dt]
        
        # 筛选特征
        if features:
            required_cols = ['date'] + features
            available_cols = [col for col in required_cols if col in df.columns]
            missing_cols = set(required_cols) - set(available_cols)
            
            if missing_cols:
                logger.warning(f"特征列 {missing_cols} 在板块 {sector} 的特征数据中不存在")
                
            df = df[available_cols]
        
        # 按日期排序
        df = df.sort_values('date')
        
        logger.info(f"从CSV文件加载板块 {sector} 特征数据，共 {len(df)} 行")
        return df
        
    except Exception as e:
        logger.error(f"加载传统格式特征文件失败: {e}")
        return pd.DataFrame()


def load_sectors_features(
    sectors: List[str],
    start_date: str = None,
    end_date: str = None,
    features: List[str] = None,
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """加载多个板块特征数据
    
    Args:
        sectors: 板块名称列表
        start_date: 开始日期，格式为YYYYMMDD，默认为None表示从最早日期开始
        end_date: 结束日期，格式为YYYYMMDD，默认为None表示到最新日期结束
        features: 需要加载的特征列表，默认为None表示加载所有特征
        use_cache: 是否使用缓存
        
    Returns:
        板块名称到特征DataFrame的字典
    """
    result = {}
    
    # 加载每个板块的特征
    for sector in sectors:
        df = load_sector_features(
            sector=sector,
            start_date=start_date,
            end_date=end_date,
            features=features,
            use_cache=use_cache
        )
        
        if not df.empty:
            result[sector] = df
    
    return result


def load_all_sectors_features(
    start_date: str = None,
    end_date: str = None,
    features: List[str] = None,
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """加载所有板块特征数据
    
    Args:
        start_date: 开始日期，格式为YYYYMMDD，默认为None表示从最早日期开始
        end_date: 结束日期，格式为YYYYMMDD，默认为None表示到最新日期结束
        features: 需要加载的特征列表，默认为None表示加载所有特征
        use_cache: 是否使用缓存
        
    Returns:
        板块名称到特征DataFrame的字典
    """
    # 获取所有板块
    sectors = get_all_sectors()
    
    # 加载所有板块特征
    return load_sectors_features(
        sectors=sectors,
        start_date=start_date,
        end_date=end_date,
        features=features,
        use_cache=use_cache
    )


def get_available_features(sector: str = None) -> List[str]:
    """获取可用的特征列表
    
    Args:
        sector: 板块名称，默认为None表示从任意板块获取
        
    Returns:
        可用特征名称列表
    """
    if sector:
        # 从指定板块获取特征
        df = load_sector_features(sector)
        if not df.empty:
            # 排除日期列
            features = [col for col in df.columns if col != 'date']
            return features
        else:
            return []
    else:
        # 从所有板块获取特征
        sectors = get_all_sectors()
        
        # 尝试获取第一个有数据的板块
        for sector in sectors:
            df = load_sector_features(sector)
            if not df.empty:
                # 排除日期列
                features = [col for col in df.columns if col != 'date']
                return features
        
        return []


def get_available_dates(sector: str = None) -> List[str]:
    """获取可用的日期列表
    
    Args:
        sector: 板块名称，默认为None表示从任意板块获取
        
    Returns:
        可用日期列表，格式为YYYYMMDD
    """
    if sector:
        # 从指定板块获取日期
        df = load_sector_features(sector)
        if not df.empty and 'date' in df.columns:
            return sorted(df['date'].unique().tolist())
        else:
            return []
    else:
        # 从所有板块获取日期
        all_dates = set()
        sectors = get_all_sectors()
        
        # 收集所有板块的日期
        for sector in sectors:
            df = load_sector_features(sector)
            if not df.empty and 'date' in df.columns:
                all_dates.update(df['date'].unique().tolist())
        
        return sorted(list(all_dates))


def clear_cache() -> None:
    """清除缓存"""
    global _FEATURE_CACHE
    _FEATURE_CACHE = {}
    logger.info("特征缓存已清除")


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 测试加载数据
    sector = "TGN人工智能"
    print(f"加载板块 {sector} 特征...")
    df = load_sector_features(sector)
    
    if not df.empty:
        print(f"共加载 {len(df)} 行数据")
        print(f"可用特征: {', '.join(df.columns)}")
        print(f"日期范围: {df['date'].min()} - {df['date'].max()}")
        print("\n前5行数据:")
        print(df.head())
    else:
        print(f"板块 {sector} 无可用特征数据") 