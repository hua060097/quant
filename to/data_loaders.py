#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载器模块

提供统一的数据访问接口，使用真实数据源
"""

import os
import logging
import pandas as pd
from typing import List, Dict, Optional, Union, Any

# 设置日志
logger = logging.getLogger(__name__)

# 全局数据加载器实例
_DATA_LOADER = None


def get_data_loader():
    """获取全局数据加载器实例
    
    如果无法加载真实数据源，直接抛出异常
    
    Returns:
        数据加载器实例
    
    Raises:
        ImportError: 无法导入真实数据源时抛出
    """
    global _DATA_LOADER
    if _DATA_LOADER is None:
        # 尝试初始化真实数据源
        try:
            import xtquant.xtdata as xtdata
            from to.advanced_quant_data import ParquetDataStorage
            _DATA_LOADER = ParquetDataStorage()
            logger.info("使用真实数据源: ParquetDataStorage")
        except ImportError as e:
            # 无法加载真实数据源，抛出异常直接中断程序
            logger.error(f"无法加载真实数据源: {e}")
            raise ImportError(f"无法加载真实数据源，请确保安装了xtquant并配置了正确的环境: {e}")
    
    return _DATA_LOADER


def get_trading_days(start_date: str, end_date: str) -> List[str]:
    """获取交易日列表
    
    Args:
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD
        
    Returns:
        交易日列表，格式为YYYYMMDD
    """
    loader = get_data_loader()
    if hasattr(loader, 'get_trading_calendar'):
        return loader.get_trading_calendar(start_date, end_date)
    elif hasattr(loader, 'get_trading_dates'):
        return loader.get_trading_dates(start_date, end_date)
    else:
        raise NotImplementedError(f"数据加载器 {type(loader).__name__} 未实现获取交易日历功能")


def get_daily_data(stock_list: List[str], start_date: str, end_date: str, fields: Optional[List[str]] = None) -> pd.DataFrame:
    """获取股票日线数据
    
    Args:
        stock_list: 股票代码列表
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD
        fields: 字段列表，如果为None则获取全部字段
        
    Returns:
        股票日线数据DataFrame
    """
    loader = get_data_loader()
    
    try:
        if hasattr(loader, 'get_daily_data'):
            data = loader.get_daily_data(stock_list, start_date, end_date, fields)
            # 标准化返回格式
            if isinstance(data, dict):
                # 如果是字典格式，每个键都是一个股票代码
                return _convert_dict_to_dataframe(data)
            elif isinstance(data, pd.DataFrame):
                # 如果已经是DataFrame，直接返回
                return data
            else:
                # 未知格式，抛出异常
                raise ValueError(f"get_daily_data返回未知格式: {type(data)}")
        else:
            raise NotImplementedError(f"数据加载器 {type(loader).__name__} 未实现获取日线数据功能")
    except Exception as e:
        logger.error(f"获取日线数据失败: {e}")
        raise


def get_market_data(field_list: Optional[List[str]] = None, stock_list: Optional[List[str]] = None, 
                    period: str = '1d', start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict:
    """获取市场数据
    
    Args:
        field_list: 字段列表，如果为None则获取全部字段
        stock_list: 股票代码列表，如果为None则获取全部股票
        period: 周期，默认为'1d'
        start_time: 开始时间，格式为YYYYMMDD
        end_time: 结束时间，格式为YYYYMMDD
        
    Returns:
        市场数据，{字段: DataFrame}格式
    """
    loader = get_data_loader()
    
    try:
        if hasattr(loader, 'get_market_data'):
            return loader.get_market_data(field_list, stock_list, period, start_time, end_time)
        else:
            raise NotImplementedError(f"数据加载器 {type(loader).__name__} 未实现获取市场数据功能")
    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
        raise


def _convert_dict_to_dataframe(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """将股票数据字典转换为单个DataFrame
    
    Args:
        data_dict: 股票数据字典，{股票代码: DataFrame}格式
        
    Returns:
        合并后的DataFrame
    """
    if not data_dict:
        return pd.DataFrame()
    
    all_dfs = []
    for stock_code, df in data_dict.items():
        # 确保DataFrame有code列
        if 'code' not in df.columns:
            df = df.copy()
            df['code'] = stock_code
        
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True) 