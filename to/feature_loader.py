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
    cache_key = f"{sector}_{start_date}_{end_date}"
    if use_cache and cache_key in _FEATURE_CACHE:
        df = _FEATURE_CACHE[cache_key]
        # 如果指定了特征列表，筛选特征
        if features:
            required_cols = ['date'] + features
            df = df[required_cols]
        return df
    
    # 构建板块特征文件路径
    sector_path = get_data_path('sector_features') / sector
    
    # 如果板块目录不存在，返回空DataFrame
    if not sector_path.exists():
        logger.warning(f"板块 {sector} 特征目录不存在: {sector_path}")
        return pd.DataFrame()
    
    # 尝试读取特征文件
    try:
        feature_file = sector_path / f"{sector}_features.csv"
        
        # 检查文件是否存在
        if not feature_file.exists():
            logger.warning(f"板块 {sector} 特征文件不存在: {feature_file}")
            return pd.DataFrame()
        
        # 读取CSV文件
        df = pd.read_csv(feature_file)
        
        # 确保日期列是字符串格式
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str)
        else:
            logger.warning(f"板块 {sector} 特征文件缺少日期列")
            return pd.DataFrame()
        
        # 按日期筛选
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        # 如果指定了特征列表，筛选特征
        if features:
            available_features = list(set(df.columns) & set(features))
            if len(available_features) < len(features):
                missing = set(features) - set(available_features)
                logger.warning(f"板块 {sector} 缺少部分特征: {missing}")
            
            required_cols = ['date'] + available_features
            df = df[required_cols]
        
        # 更新缓存
        if use_cache:
            _FEATURE_CACHE[cache_key] = df.copy()
        
        return df
    
    except Exception as e:
        logger.error(f"加载板块 {sector} 特征失败: {e}")
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