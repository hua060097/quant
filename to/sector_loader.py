#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Union, List

# 配置日志
logger = logging.getLogger(__name__)

def load_sector_features(sector_name: str, 
                         data_dir: Optional[str] = None,
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> pd.DataFrame:
    """加载板块特征数据
    
    Args:
        sector_name: 板块名称
        data_dir: 数据目录路径，如果为None则使用默认路径
        start_date: 开始日期，格式YYYYMMDD，可选
        end_date: 结束日期，格式YYYYMMDD，可选
        
    Returns:
        板块特征的DataFrame，如果加载失败则返回空DataFrame
    """
    # 设置默认数据目录
    if data_dir is None:
        data_dir = Path(os.getcwd()) / "to" / "local_data" / "sector_features"
    else:
        data_dir = Path(data_dir)
    
    # 构建板块目录
    sector_dir = data_dir / sector_name
    
    # 检查目录是否存在
    if not sector_dir.exists():
        logger.warning(f"板块目录不存在: {sector_dir}")
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame()
        
        # 按优先级尝试加载不同格式的文件
        # 1. 首先尝试加载 {sector_name}_features.parquet
        features_parquet = sector_dir / f"{sector_name}_features.parquet"
        if features_parquet.exists():
            try:
                df = pd.read_parquet(features_parquet)
                logger.info(f"从 {features_parquet} 加载板块特征")
            except Exception as e:
                logger.warning(f"加载 {features_parquet} 失败: {e}")
        
        # 2. 如果features文件加载失败，尝试加载 {sector_name}.parquet
        if df.empty:
            regular_parquet = sector_dir / f"{sector_name}.parquet"
            if regular_parquet.exists():
                try:
                    df = pd.read_parquet(regular_parquet)
                    logger.info(f"从 {regular_parquet} 加载板块特征")
                except Exception as e:
                    logger.warning(f"加载 {regular_parquet} 失败: {e}")
        
        # 3. 如果还是加载失败，尝试加载 {sector_name}_latest.csv
        if df.empty:
            latest_csv = sector_dir / f"{sector_name}_latest.csv"
            if latest_csv.exists():
                try:
                    df = pd.read_csv(latest_csv)
                    logger.info(f"从 {latest_csv} 加载板块特征")
                except Exception as e:
                    logger.warning(f"加载 {latest_csv} 失败: {e}")
        
        # 如果所有加载尝试都失败
        if df.empty:
            logger.warning(f"未能加载板块 {sector_name} 的任何特征文件")
            return pd.DataFrame()
        
        # 确保date列是日期时间类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 过滤日期范围
        if start_date or end_date:
            # 转换日期为datetime对象
            start_dt = pd.to_datetime(start_date) if start_date else None
            end_dt = pd.to_datetime(end_date) if end_date else None
            
            # 过滤日期范围
            if start_dt and end_dt:
                df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            elif start_dt:
                df = df[df['date'] >= start_dt]
            elif end_dt:
                df = df[df['date'] <= end_dt]
        
        # 添加板块名称列（如果不存在）
        if 'sector_name' not in df.columns:
            df['sector_name'] = sector_name
        
        logger.info(f"成功加载板块 {sector_name} 的特征数据，共 {len(df)} 行")
        return df
    
    except Exception as e:
        logger.error(f"加载板块 {sector_name} 的特征数据时出错: {str(e)}")
        return pd.DataFrame()

def load_sector_features_for_date(sector_name: str, 
                                 date: str,
                                 data_dir: Optional[str] = None) -> Dict:
    """加载指定日期的板块特征数据
    
    Args:
        sector_name: 板块名称
        date: 日期字符串，格式YYYYMMDD
        data_dir: 数据目录路径，如果为None则使用默认路径
        
    Returns:
        包含特征的字典，如果加载失败则返回空字典
    """
    # 加载整个板块特征数据
    df = load_sector_features(sector_name, data_dir)
    
    if df.empty:
        logger.warning(f"板块 {sector_name} 没有特征数据")
        return {}
    
    # 将日期转换为pandas日期时间对象
    date_dt = pd.to_datetime(date)
    
    # 确保'date'列存在
    if 'date' not in df.columns:
        logger.warning(f"板块特征数据中没有'date'列")
        return {}
    
    # 获取指定日期的数据
    date_rows = df[df['date'] == date_dt]
    
    if not date_rows.empty:
        # 获取第一行（如果有多行匹配，只取第一行）
        row = date_rows.iloc[0]
        # 将Series转换为字典
        features = row.to_dict()
        features['date'] = date  # 确保日期字段使用原始格式
        return features
    else:
        logger.warning(f"板块 {sector_name} 在日期 {date} 没有特征数据")
        return {}

def get_available_sectors(data_dir: Optional[str] = None) -> List[str]:
    """获取可用的板块列表
    
    Args:
        data_dir: 数据目录路径，如果为None则使用默认路径
        
    Returns:
        可用的板块名称列表
    """
    # 设置默认数据目录
    if data_dir is None:
        data_dir = Path(os.getcwd()) / "to" / "local_data" / "sector_features"
    else:
        data_dir = Path(data_dir)
    
    # 检查目录是否存在
    if not data_dir.exists():
        logger.warning(f"板块特征目录不存在: {data_dir}")
        return []
    
    try:
        # 获取所有子目录（每个子目录对应一个板块）
        sector_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # 提取板块名称
        sector_names = [directory.name for directory in sector_dirs]
        
        logger.info(f"找到 {len(sector_names)} 个可用板块")
        return sector_names
    
    except Exception as e:
        logger.error(f"获取可用板块列表时出错: {str(e)}")
        return []

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试加载函数
    sector_name = "TGN人工智能"
    print(f"尝试加载板块 {sector_name} 的特征数据...")
    
    df = load_sector_features(sector_name)
    if not df.empty:
        print(f"成功加载 {len(df)} 行数据，列: {df.columns.tolist()}")
        print(f"第一行数据:")
        print(df.iloc[0]) 