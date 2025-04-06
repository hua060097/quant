#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SectorFeatureManager:
    """板块特征数据管理器
    
    管理已生成的板块特征数据，提供加载、合并、查询功能。
    """
    
    def __init__(self, data_dir: str = './data/sector_features'):
        """初始化特征数据管理器
        
        Args:
            data_dir: 特征数据目录
        """
        self.data_dir = data_dir
        self.features_cache = {}  # 缓存已加载的特征数据
        
        # 检查数据目录
        if not os.path.exists(data_dir):
            logger.warning(f"特征数据目录 {data_dir} 不存在")
            os.makedirs(data_dir, exist_ok=True)
    
    def get_available_sectors(self) -> List[str]:
        """获取可用的板块列表
        
        Returns:
            可用板块名称列表
        """
        sector_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        return sorted(sector_dirs)
    
    def get_sector_files(self, sector_name: str) -> List[str]:
        """获取指定板块的所有特征文件
        
        Args:
            sector_name: 板块名称
            
        Returns:
            特征文件路径列表
        """
        sector_dir = os.path.join(self.data_dir, sector_name)
        if not os.path.exists(sector_dir):
            logger.warning(f"板块 {sector_name} 的数据目录不存在")
            return []
        
        # 查找所有CSV文件，但排除latest文件
        files = glob.glob(os.path.join(sector_dir, f"{sector_name}_*.csv"))
        files = [f for f in files if not f.endswith('_latest.csv')]
        return sorted(files)
    
    def load_sector_features(self, 
                          sector_name: str, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None,
                          use_cache: bool = True) -> pd.DataFrame:
        """加载指定板块的特征数据
        
        Args:
            sector_name: 板块名称
            start_date: 开始日期，格式YYYYMMDD，可选
            end_date: 结束日期，格式YYYYMMDD，可选
            use_cache: 是否使用缓存
            
        Returns:
            板块特征DataFrame
        """
        # 检查缓存
        cache_key = f"{sector_name}_{start_date or 'all'}_{end_date or 'all'}"
        if use_cache and cache_key in self.features_cache:
            logger.debug(f"使用缓存加载板块 {sector_name} 的特征数据")
            features_df = self.features_cache[cache_key]
            return features_df
        
        # 构建板块目录路径
        sector_dir = os.path.join(self.data_dir, sector_name)
        if not os.path.exists(sector_dir):
            logger.warning(f"板块 {sector_name} 的数据目录不存在")
            return pd.DataFrame()
        
        # 优先尝试读取Parquet文件
        parquet_file = os.path.join(sector_dir, f"{sector_name}_features.parquet")
        if os.path.exists(parquet_file):
            try:
                # 构建过滤条件
                filters = []
                if start_date:
                    # 转换为日期格式
                    start_dt = pd.to_datetime(start_date, format='%Y%m%d')
                    filters.append(('date', '>=', start_dt))
                
                if end_date:
                    # 转换为日期格式
                    end_dt = pd.to_datetime(end_date, format='%Y%m%d')
                    filters.append(('date', '<=', end_dt))
                
                # 根据是否有过滤条件决定如何读取
                if filters:
                    # 有日期过滤
                    features_df = pd.read_parquet(parquet_file, filters=filters)
                else:
                    # 无过滤
                    features_df = pd.read_parquet(parquet_file)
                
                # 确保日期列是日期类型
                if 'date' in features_df.columns and not pd.api.types.is_datetime64_any_dtype(features_df['date']):
                    features_df['date'] = pd.to_datetime(features_df['date'])
                
                # 设置索引
                if 'date' in features_df.columns:
                    features_df.set_index('date', inplace=True)
                
                # 存入缓存
                if use_cache:
                    self.features_cache[cache_key] = features_df
                
                logger.info(f"从Parquet文件加载板块 {sector_name} 的特征数据，共 {len(features_df)} 行")
                return features_df
            
            except Exception as e:
                logger.error(f"从Parquet文件加载板块 {sector_name} 特征失败: {e}")
                # 回退到CSV格式
        
        # 回退到CSV格式，获取特征文件
        feature_files = self.get_sector_files(sector_name)
        if not feature_files:
            logger.warning(f"板块 {sector_name} 没有可用的特征文件")
            return pd.DataFrame()
        
        # 加载并合并所有特征文件
        dfs = []
        for file_path in feature_files:
            try:
                df = pd.read_csv(file_path)
                
                # 确保日期列格式正确
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                elif df.index.name == 'date':
                    df.index = pd.to_datetime(df.index)
                
                dfs.append(df)
            except Exception as e:
                logger.error(f"加载文件 {file_path} 时出错: {str(e)}")
        
        if not dfs:
            logger.warning(f"没有成功加载板块 {sector_name} 的特征数据")
            return pd.DataFrame()
        
        # 合并所有数据
        features_df = pd.concat(dfs)
        
        # 去除重复行
        if not features_df.empty:
            features_df = features_df[~features_df.index.duplicated(keep='last')]
            features_df.sort_index(inplace=True)
        
        # 筛选日期范围
        if start_date or end_date:
            start_date_obj = pd.to_datetime(start_date, format='%Y%m%d') if start_date else None
            end_date_obj = pd.to_datetime(end_date, format='%Y%m%d') if end_date else None
            
            if start_date_obj and end_date_obj:
                features_df = features_df[(features_df.index >= start_date_obj) & 
                                       (features_df.index <= end_date_obj)]
            elif start_date_obj:
                features_df = features_df[features_df.index >= start_date_obj]
            elif end_date_obj:
                features_df = features_df[features_df.index <= end_date_obj]
        
        # 存入缓存
        if use_cache:
            self.features_cache[cache_key] = features_df
        
        logger.info(f"成功加载板块 {sector_name} 的特征数据，共 {len(features_df)} 行")
        return features_df
    
    def get_latest_features(self, sector_name: str, use_cache: bool = True) -> pd.DataFrame:
        """获取指定板块的最新特征数据
        
        Args:
            sector_name: 板块名称
            use_cache: 是否使用缓存
            
        Returns:
            板块最新特征DataFrame
        """
        # 检查缓存
        cache_key = f"{sector_name}_latest"
        if use_cache and cache_key in self.features_cache:
            logger.debug(f"使用缓存加载板块 {sector_name} 的最新特征数据")
            return self.features_cache[cache_key]
        
        # 构建板块目录路径
        sector_dir = os.path.join(self.data_dir, sector_name)
        if not os.path.exists(sector_dir):
            logger.warning(f"板块 {sector_name} 的数据目录不存在")
            return pd.DataFrame()
        
        # 首先尝试读取专门的最新特征文件
        latest_file = os.path.join(sector_dir, f"{sector_name}_latest.csv")
        if os.path.exists(latest_file):
            try:
                latest_df = pd.read_csv(latest_file)
                
                # 确保日期列格式正确
                if 'date' in latest_df.columns:
                    latest_df['date'] = pd.to_datetime(latest_df['date'])
                    latest_df.set_index('date', inplace=True)
                
                # 存入缓存
                if use_cache:
                    self.features_cache[cache_key] = latest_df
                
                logger.info(f"从最新特征文件加载板块 {sector_name} 的特征数据，日期: {latest_df.index[0] if not latest_df.empty else '未知'}")
                return latest_df
            
            except Exception as e:
                logger.error(f"从最新特征文件加载板块 {sector_name} 特征失败: {e}")
        
        # 尝试从Parquet文件获取最新日期的数据
        parquet_file = os.path.join(sector_dir, f"{sector_name}_features.parquet")
        if os.path.exists(parquet_file):
            try:
                # 读取所有日期列数据
                dates_df = pd.read_parquet(parquet_file, columns=['date'])
                
                # 确保日期列是日期类型
                if 'date' in dates_df.columns and not pd.api.types.is_datetime64_any_dtype(dates_df['date']):
                    dates_df['date'] = pd.to_datetime(dates_df['date'])
                
                # 获取最大日期
                if not dates_df.empty:
                    max_date = dates_df['date'].max()
                    
                    # 使用过滤条件读取最新日期的数据
                    latest_df = pd.read_parquet(parquet_file, filters=[('date', '==', max_date)])
                    
                    # 设置索引
                    if 'date' in latest_df.columns:
                        latest_df.set_index('date', inplace=True)
                    
                    # 存入缓存
                    if use_cache:
                        self.features_cache[cache_key] = latest_df
                    
                    logger.info(f"从Parquet文件加载板块 {sector_name} 的最新特征数据，日期: {max_date}")
                    return latest_df
            
            except Exception as e:
                logger.error(f"从Parquet文件获取板块 {sector_name} 最新特征失败: {e}")
        
        # 回退方案：加载所有特征数据并获取最新日期的数据
        features_df = self.load_sector_features(sector_name, use_cache=use_cache)
        if features_df.empty:
            logger.warning(f"板块 {sector_name} 没有可用的特征数据")
            return pd.DataFrame()
        
        # 获取最新日期的数据
        latest_date = features_df.index.max()
        latest_df = features_df.loc[latest_date:latest_date]
        
        # 存入缓存
        if use_cache:
            self.features_cache[cache_key] = latest_df
        
        logger.info(f"从所有特征数据中获取板块 {sector_name} 的最新特征数据，日期: {latest_date}")
        return latest_df
    
    def get_sector_feature_date_range(self, sector_name: str) -> Tuple[Optional[str], Optional[str]]:
        """获取指定板块特征数据的日期范围
        
        Args:
            sector_name: 板块名称
            
        Returns:
            (最早日期, 最晚日期) 元组，格式YYYYMMDD
        """
        features_df = self.load_sector_features(sector_name)
        if features_df.empty:
            return None, None
        
        min_date = features_df.index.min().strftime('%Y%m%d')
        max_date = features_df.index.max().strftime('%Y%m%d')
        
        return min_date, max_date
    
    def query_sector_features(self, 
                           sector_name: str, 
                           date_str: str) -> Dict[str, float]:
        """查询指定板块在特定日期的特征数据
        
        Args:
            sector_name: 板块名称
            date_str: 日期字符串，格式YYYYMMDD
            
        Returns:
            特征字典
        """
        features_df = self.load_sector_features(sector_name)
        if features_df.empty:
            logger.warning(f"板块 {sector_name} 没有可用的特征数据")
            return {}
        
        # 转换日期格式
        date_obj = pd.to_datetime(date_str, format='%Y%m%d')
        
        # 查找该日期的数据
        if date_obj not in features_df.index:
            logger.warning(f"板块 {sector_name} 在日期 {date_str} 没有特征数据")
            return {}
        
        # 获取该日期的所有特征
        features = features_df.loc[date_obj].to_dict()
        return features
    
    def compare_sectors(self, 
                      sectors: List[str], 
                      feature_name: str,
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """比较多个板块的指定特征
        
        Args:
            sectors: 板块名称列表
            feature_name: 要比较的特征名称
            start_date: 开始日期，格式YYYYMMDD，可选
            end_date: 结束日期，格式YYYYMMDD，可选
            
        Returns:
            比较结果DataFrame，每一列是一个板块的特征数据
        """
        result_dict = {}
        
        for sector in sectors:
            features_df = self.load_sector_features(sector, start_date, end_date)
            
            if features_df.empty:
                logger.warning(f"板块 {sector} 在指定时间段没有特征数据")
                continue
                
            if feature_name not in features_df.columns:
                logger.warning(f"板块 {sector} 没有特征 '{feature_name}'")
                continue
                
            result_dict[sector] = features_df[feature_name]
        
        if not result_dict:
            logger.warning(f"没有板块包含特征 '{feature_name}'")
            return pd.DataFrame()
        
        return pd.DataFrame(result_dict)
    
    def calculate_correlation(self, 
                           sector1: str, 
                           sector2: str,
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> Dict[str, float]:
        """计算两个板块特征间的相关性
        
        Args:
            sector1: 第一个板块名称
            sector2: 第二个板块名称
            start_date: 开始日期，格式YYYYMMDD，可选
            end_date: 结束日期，格式YYYYMMDD，可选
            
        Returns:
            特征相关性字典，{特征名: 相关系数}
        """
        # 加载两个板块的特征数据
        df1 = self.load_sector_features(sector1, start_date, end_date)
        df2 = self.load_sector_features(sector2, start_date, end_date)
        
        if df1.empty or df2.empty:
            logger.warning(f"至少有一个板块在指定时间段没有特征数据")
            return {}
        
        # 找到共同的特征列
        common_features = set(df1.columns) & set(df2.columns)
        if not common_features:
            logger.warning(f"两个板块没有共同的特征列")
            return {}
        
        # 对齐日期索引
        common_dates = df1.index.intersection(df2.index)
        if common_dates.empty:
            logger.warning(f"两个板块在指定时间段没有共同的日期")
            return {}
        
        aligned_df1 = df1.loc[common_dates, list(common_features)]
        aligned_df2 = df2.loc[common_dates, list(common_features)]
        
        # 计算相关系数
        correlations = {}
        for feature in common_features:
            if pd.api.types.is_numeric_dtype(aligned_df1[feature]) and pd.api.types.is_numeric_dtype(aligned_df2[feature]):
                corr = aligned_df1[feature].corr(aligned_df2[feature])
                if not np.isnan(corr):
                    correlations[feature] = corr
        
        return correlations
    
    def get_trending_sectors(self, 
                          date_str: str, 
                          trending_feature: str = 'ma_trend_score',
                          min_score: float = 3.0,
                          top_n: int = 10) -> List[Tuple[str, float]]:
        """获取指定日期趋势最强的板块
        
        Args:
            date_str: 日期字符串，格式YYYYMMDD
            trending_feature: 用于判断趋势的特征名
            min_score: 最小趋势分数阈值
            top_n: 返回前N个板块
            
        Returns:
            板块和趋势分数列表，按分数降序排列
        """
        date_obj = pd.to_datetime(date_str, format='%Y%m%d')
        sectors = self.get_available_sectors()
        
        trending_sectors = []
        for sector in sectors:
            features_df = self.load_sector_features(sector)
            
            if features_df.empty or date_obj not in features_df.index:
                continue
                
            if trending_feature not in features_df.columns:
                continue
                
            score = features_df.loc[date_obj, trending_feature]
            if pd.notnull(score) and score >= min_score:
                trending_sectors.append((sector, score))
        
        # 排序并返回top_n
        trending_sectors.sort(key=lambda x: x[1], reverse=True)
        return trending_sectors[:top_n]
    
    def clear_cache(self):
        """清空特征数据缓存"""
        self.features_cache.clear()
        logger.info("特征数据缓存已清空")