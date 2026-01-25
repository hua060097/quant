#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理器模块
为板块轮动模型提供统一的数据加载、缓存和基本处理功能
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import json
import traceback
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache

from to.logger_config import configure_logger

# 配置日志
logger = configure_logger(__name__, 'data_processor.log')

class DataProcessor:
    """
    数据处理器类
    负责特征数据的加载、缓存和基本处理
    """
    
    def __init__(self, data_dir=None, cache_dir=None):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据目录路径
            cache_dir: 缓存目录路径
        """
        # 获取项目根目录
        self.project_root = self._get_project_root()
        
        # 设置数据目录和缓存目录
        self.data_dir = data_dir or os.path.join(self.project_root, "to", "local_data")
        self.cache_dir = cache_dir or os.path.join(self.project_root, "data_cache")
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 日志记录器
        self.logger = logger
        
        # 初始化交易日历
        self.trading_calendar = self._load_trading_calendar()
        
        # 缓存字典
        self._cache = {}
    
    def _get_project_root(self):
        """获取项目根目录"""
        current_file = Path(__file__).resolve()
        return current_file.parent.parent
    
    def _load_trading_calendar(self):
        """
        加载交易日历
        
        Returns:
            交易日历列表（日期字符串）
        """
        try:
            calendar_path = os.path.join(self.data_dir, "calendar", "trading_days.json")
            if os.path.exists(calendar_path):
                with open(calendar_path, 'r', encoding='utf-8') as f:
                    calendar_data = json.load(f)
                    
                if isinstance(calendar_data, dict) and 'trading_days' in calendar_data:
                    trading_days = calendar_data['trading_days']
                    self.logger.info(f"成功加载交易日历，共 {len(trading_days)} 个交易日")
                    return trading_days
                elif isinstance(calendar_data, list):
                    self.logger.info(f"成功加载交易日历，共 {len(calendar_data)} 个交易日")
                    return calendar_data
                else:
                    self.logger.warning("交易日历格式不正确，将使用备用方法")
            else:
                self.logger.warning(f"交易日历文件不存在: {calendar_path}，将使用备用方法")
                
            # 备用方法：根据特征数据推断交易日历
            self.logger.info("使用备用方法生成交易日历...")
            calendar_backup_path = os.path.join(self.data_dir, "features", "daily", "calendar_backup.txt")
            
            if os.path.exists(calendar_backup_path):
                with open(calendar_backup_path, 'r') as f:
                    trading_days = [line.strip() for line in f if line.strip()]
                self.logger.info(f"从备份文件加载交易日历，共 {len(trading_days)} 个交易日")
                return trading_days
            
            # 如果备份文件也不存在，返回空列表
            self.logger.warning("无法加载交易日历，将返回空列表")
            return []
            
        except Exception as e:
            self.logger.error(f"加载交易日历时出错: {e}")
            self.logger.error(traceback.format_exc())
            return []
    
    def get_prev_trading_day(self, date_str):
        """
        获取前一个交易日
        
        Args:
            date_str: 日期字符串，格式为'YYYYMMDD'
            
        Returns:
            前一个交易日字符串，如果没有则返回None
        """
        if not self.trading_calendar:
            self.logger.warning("交易日历为空，无法获取前一个交易日")
            return None
            
        try:
            # 确保日期格式正确
            date_str = self._normalize_date_format(date_str)
            
            # 找到当前日期在交易日历中的位置
            if date_str in self.trading_calendar:
                idx = self.trading_calendar.index(date_str)
                if idx > 0:
                    return self.trading_calendar[idx - 1]
            
            # 如果当前日期不在交易日历中，找到小于当前日期的最大日期
            valid_days = [d for d in self.trading_calendar if d < date_str]
            return max(valid_days) if valid_days else None
            
        except Exception as e:
            self.logger.error(f"获取前一个交易日时出错: {e}")
            return None
    
    def get_next_trading_day(self, date_str):
        """
        获取下一个交易日
        
        Args:
            date_str: 日期字符串，格式为'YYYYMMDD'
            
        Returns:
            下一个交易日字符串，如果没有则返回None
        """
        if not self.trading_calendar:
            self.logger.warning("交易日历为空，无法获取下一个交易日")
            return None
            
        try:
            # 确保日期格式正确
            date_str = self._normalize_date_format(date_str)
            
            # 找到当前日期在交易日历中的位置
            if date_str in self.trading_calendar:
                idx = self.trading_calendar.index(date_str)
                if idx < len(self.trading_calendar) - 1:
                    return self.trading_calendar[idx + 1]
            
            # 如果当前日期不在交易日历中，找到大于当前日期的最小日期
            valid_days = [d for d in self.trading_calendar if d > date_str]
            return min(valid_days) if valid_days else None
            
        except Exception as e:
            self.logger.error(f"获取下一个交易日时出错: {e}")
            return None
    
    def _normalize_date_format(self, date_str):
        """
        标准化日期格式为'YYYYMMDD'
        
        Args:
            date_str: 日期字符串
            
        Returns:
            标准化后的日期字符串
        """
        if isinstance(date_str, (datetime, pd.Timestamp)):
            return date_str.strftime('%Y%m%d')
            
        # 如果是形如'YYYY-MM-DD'的格式，转换为'YYYYMMDD'
        if isinstance(date_str, str) and len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
            return date_str.replace('-', '')
            
        return date_str
    
    def _get_cache_key(self, prefix, *args, **kwargs):
        """
        生成缓存键
        
        Args:
            prefix: 缓存键前缀
            *args, **kwargs: 用于生成缓存键的参数
            
        Returns:
            缓存键
        """
        # 创建一个包含所有参数的字符串
        key_str = prefix + '_'
        if args:
            key_str += '_'.join(str(arg) for arg in args)
        
        if kwargs:
            key_str += '_' + '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        # 使用MD5生成哈希值
        hash_obj = hashlib.md5(key_str.encode())
        return hash_obj.hexdigest()
    
    def load_sector_features(self, sector_code, date, use_cache=True):
        """
        加载板块特征数据
        
        Args:
            sector_code: 板块代码
            date: 日期字符串或datetime对象
            use_cache: 是否使用缓存
            
        Returns:
            板块特征数据字典
        """
        try:
            # 标准化日期格式
            date_str = self._normalize_date_format(date)
            
            # 检查缓存
            cache_key = self._get_cache_key('sector_features', sector_code, date_str)
            if use_cache and cache_key in self._cache:
                return self._cache[cache_key]
            
            # 构建特征文件路径
            features_dir = os.path.join(self.data_dir, "sector_features")
            date_dir = date_str[:6]  # 取YYYYMM部分
            feature_file = f"{sector_code}_{date_str}.json"
            
            # 检查两种可能的路径
            path1 = os.path.join(features_dir, date_dir, feature_file)
            path2 = os.path.join(features_dir, feature_file)
            
            data = None
            
            # 尝试第一种路径
            if os.path.exists(path1):
                with open(path1, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.logger.debug(f"从路径1加载板块特征: {path1}")
            
            # 尝试第二种路径
            elif os.path.exists(path2):
                with open(path2, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.logger.debug(f"从路径2加载板块特征: {path2}")
            
            # 如果都找不到，查看可能的备用文件
            else:
                # 检查备用日期
                prev_day = self.get_prev_trading_day(date_str)
                if prev_day:
                    backup_file = f"{sector_code}_{prev_day}.json"
                    backup_path1 = os.path.join(features_dir, prev_day[:6], backup_file)
                    backup_path2 = os.path.join(features_dir, backup_file)
                    
                    if os.path.exists(backup_path1):
                        with open(backup_path1, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.logger.info(f"使用前一日数据: {backup_path1}")
                    elif os.path.exists(backup_path2):
                        with open(backup_path2, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.logger.info(f"使用前一日数据: {backup_path2}")
                    else:
                        self.logger.warning(f"找不到板块 {sector_code} 在 {date_str} 的特征数据")
                else:
                    self.logger.warning(f"找不到板块 {sector_code} 在 {date_str} 的特征数据，且无法获取前一日数据")
            
            # 缓存数据
            if data is not None and use_cache:
                self._cache[cache_key] = data
                
            return data
            
        except Exception as e:
            self.logger.error(f"加载板块 {sector_code} 特征数据时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def load_market_status(self, date, use_cache=True):
        """
        加载市场状态数据
        
        Args:
            date: 日期字符串或datetime对象
            use_cache: 是否使用缓存
            
        Returns:
            市场状态数据字典
        """
        try:
            # 标准化日期格式
            date_str = self._normalize_date_format(date)
            
            # 检查缓存
            cache_key = self._get_cache_key('market_status', date_str)
            if use_cache and cache_key in self._cache:
                return self._cache[cache_key]
            
            # 构建市场状态文件路径
            status_dir = os.path.join(self.data_dir, "market_status", "daily")
            status_file = f"market_status_{date_str}.json"
            status_path = os.path.join(status_dir, status_file)
            
            data = None
            
            # 尝试加载市场状态
            if os.path.exists(status_path):
                with open(status_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.logger.debug(f"加载市场状态: {status_path}")
            else:
                # 尝试查找前一日数据
                prev_day = self.get_prev_trading_day(date_str)
                if prev_day:
                    backup_file = f"market_status_{prev_day}.json"
                    backup_path = os.path.join(status_dir, backup_file)
                    
                    if os.path.exists(backup_path):
                        with open(backup_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.logger.info(f"使用前一日市场状态: {backup_path}")
                    else:
                        self.logger.warning(f"找不到 {date_str} 的市场状态数据，且无法获取前一日数据")
                else:
                    self.logger.warning(f"找不到 {date_str} 的市场状态数据，且无法获取前一日数据")
            
            # 缓存数据
            if data is not None and use_cache:
                self._cache[cache_key] = data
                
            return data
            
        except Exception as e:
            self.logger.error(f"加载市场状态数据时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def load_sector_list(self, date=None):
        """
        加载板块列表
        
        Args:
            date: 可选的日期参数，用于加载特定日期的板块列表
            
        Returns:
            板块列表
        """
        try:
            # 优先从映射文件加载
            mapping_path = os.path.join(self.data_dir, "sector_mapping", "sector_list.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'sectors' in data:
                    return data['sectors']
            
            # 如果没有映射文件，从features目录获取
            features_dir = os.path.join(self.data_dir, "sector_features")
            if os.path.exists(features_dir):
                # 获取所有子目录
                subdirs = [d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))]
                
                # 如果有YYYYMM格式的子目录，使用最新的子目录
                date_dirs = [d for d in subdirs if len(d) == 6 and d.isdigit()]
                if date_dirs:
                    latest_dir = os.path.join(features_dir, max(date_dirs))
                    files = os.listdir(latest_dir)
                    
                    # 提取板块代码
                    sector_codes = set()
                    for file in files:
                        if file.endswith('.json'):
                            parts = file.split('_')
                            if len(parts) >= 2:
                                sector_codes.add(parts[0])
                    
                    return sorted(list(sector_codes))
                
                # 如果没有YYYYMM格式的子目录，直接从根目录提取
                files = os.listdir(features_dir)
                sector_codes = set()
                for file in files:
                    if file.endswith('.json'):
                        parts = file.split('_')
                        if len(parts) >= 2:
                            sector_codes.add(parts[0])
                
                return sorted(list(sector_codes))
            
            # 如果以上方法都失败，返回空列表
            self.logger.warning("无法加载板块列表，将返回空列表")
            return []
            
        except Exception as e:
            self.logger.error(f"加载板块列表时出错: {e}")
            self.logger.error(traceback.format_exc())
            return []
    
    def batch_load_features(self, sector_list, start_date, end_date):
        """
        批量加载特征数据
        
        Args:
            sector_list: 板块列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            特征DataFrame
        """
        try:
            # 标准化日期格式
            start_date_str = self._normalize_date_format(start_date)
            end_date_str = self._normalize_date_format(end_date)
            
            # 获取日期范围内的交易日
            trading_days = [d for d in self.trading_calendar if start_date_str <= d <= end_date_str]
            
            if not trading_days:
                self.logger.warning(f"在日期范围 {start_date_str} 至 {end_date_str} 内找不到交易日")
                return pd.DataFrame()
            
            # 计算缓存键
            cache_key = self._get_cache_key('batch_features', '_'.join(sector_list), start_date_str, end_date_str)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            # 检查是否有缓存文件
            if os.path.exists(cache_file):
                try:
                    features_df = pd.read_pickle(cache_file)
                    self.logger.info(f"从缓存加载特征数据: {cache_file}")
                    return features_df
                except Exception as e:
                    self.logger.warning(f"加载缓存文件时出错: {e}，将重新加载数据")
            
            # 没有缓存，重新加载数据
            all_data = []
            
            # 跟踪已处理的数据量，便于日志输出
            processed_count = 0
            total_count = len(sector_list) * len(trading_days)
            
            # 打印加载进度的间隔（每处理5%的数据打印一次）
            progress_interval = max(1, total_count // 20)
            
            for sector_code in sector_list:
                for date in trading_days:
                    # 加载板块特征
                    features = self.load_sector_features(sector_code, date)
                    
                    if features:
                        # 提取特征字典
                        if 'features' in features:
                            feature_dict = features['features']
                        else:
                            feature_dict = features
                        
                        # 添加板块代码和日期
                        feature_dict['sector_code'] = sector_code
                        feature_dict['date'] = pd.to_datetime(date)
                        
                        all_data.append(feature_dict)
                    
                    # 更新处理计数
                    processed_count += 1
                    
                    # 打印进度
                    if processed_count % progress_interval == 0 or processed_count == total_count:
                        progress = processed_count / total_count * 100
                        self.logger.info(f"加载进度: {progress:.1f}% ({processed_count}/{total_count})")
            
            # 转换为DataFrame
            if all_data:
                features_df = pd.DataFrame(all_data)
                
                # 按日期和板块代码排序
                features_df = features_df.sort_values(['date', 'sector_code']).reset_index(drop=True)
                
                # 保存缓存
                try:
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    features_df.to_pickle(cache_file)
                    self.logger.info(f"特征数据已缓存: {cache_file}")
                except Exception as e:
                    self.logger.warning(f"保存缓存文件时出错: {e}")
                
                return features_df
            else:
                self.logger.warning("未找到任何特征数据")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"批量加载特征数据时出错: {e}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def load_seasonal_patterns(self, sector_code):
        """
        加载板块季节性模式数据
        
        Args:
            sector_code: 板块代码
            
        Returns:
            季节性模式数据字典
        """
        try:
            # 构建季节性数据文件路径
            seasonal_dir = os.path.join(self.data_dir, "sector_features_leading")
            seasonal_file = f"{sector_code}_seasonal.json"
            seasonal_path = os.path.join(seasonal_dir, seasonal_file)
            
            # 检查文件是否存在
            if os.path.exists(seasonal_path):
                with open(seasonal_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            else:
                self.logger.info(f"板块 {sector_code} 的季节性数据文件不存在: {seasonal_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"加载板块 {sector_code} 季节性模式数据时出错: {e}")
            return None
    
    def clear_cache(self):
        """清除内存缓存"""
        self._cache.clear()
        self.logger.info("已清除内存缓存") 