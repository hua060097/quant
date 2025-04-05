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
    
    # 构建文件路径
    feature_file = data_dir / "sector" / f"{sector_name}.parquet"
    
    # 检查文件是否存在
    if not feature_file.exists():
        logger.warning(f"板块特征文件不存在: {feature_file}")
        return pd.DataFrame()
    
    try:
        # 读取Parquet文件
        df = pd.read_parquet(feature_file)
        
        # 确保索引是日期类型
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
        
        # 过滤日期范围
        if start_date or end_date:
            # 转换日期为datetime对象
            start_dt = pd.to_datetime(start_date) if start_date else None
            end_dt = pd.to_datetime(end_date) if end_date else None
            
            # 过滤日期范围
            if start_dt and end_dt:
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            elif start_dt:
                df = df[df.index >= start_dt]
            elif end_dt:
                df = df[df.index <= end_dt]
        
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
    
    # 获取指定日期的数据
    if date_dt in df.index:
        # 将Series转换为字典
        row = df.loc[date_dt]
        features = row.to_dict()
        features['date'] = date  # 添加日期字段
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
        data_dir = Path(os.getcwd()) / "to" / "local_data" / "sector_features" / "sector"
    else:
        data_dir = Path(data_dir) / "sector"
    
    # 检查目录是否存在
    if not data_dir.exists():
        logger.warning(f"板块特征目录不存在: {data_dir}")
        return []
    
    try:
        # 获取所有parquet文件
        feature_files = list(data_dir.glob("*.parquet"))
        
        # 提取板块名称
        sector_names = [file.stem for file in feature_files]
        
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