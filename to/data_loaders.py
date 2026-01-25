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
from pathlib import Path
import json
import sys
from xtquant import xtdata

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录

# 设置日志
logger = logging.getLogger(__name__)

# 全局数据加载器实例
_DATA_LOADER = None

# 股票名称缓存
_STOCK_NAMES_CACHE = None

# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

# 路径设置
def get_data_path(relative_path=None):
    """获取数据目录路径"""
    project_root = get_project_root()
    data_path = project_root / "to" / "local_data"
    if relative_path:
        return data_path / relative_path
    return data_path

market_status_dir = get_data_path("market_status/daily")

def get_bj_stock_list():
    """获取北交所股票代码列表"""
    bj_stock_list_path = get_data_path("bjstock.json")
    with open(bj_stock_list_path, "r") as f:
        return json.load(f)

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
            from to.advanced_quant_data import ParquetDataStorage
            _DATA_LOADER = ParquetDataStorage()
            logger.info("使用真实数据源: ParquetDataStorage")
        except ImportError as e:
            try:
                sys.path.append(str(get_project_root() / "to"))
                from advanced_quant_data import ParquetDataStorage
                _DATA_LOADER = ParquetDataStorage()
                logger.info("使用真实数据源: ParquetDataStorage")
            except ImportError as e:
                # 无法加载真实数据源，抛出异常直接中断程序
                logger.error(f"无法加载真实数据源: {e}")
                raise ImportError(f"无法加载真实数据源，请确保安装了xtquant并配置了正确的环境:")
    
    return _DATA_LOADER

def get_stock_list():
    """获取股票代码列表"""
    # bj_stock_list = get_bj_stock_list()
    # 获取沪深A股股票列表
    all_stocks = xtdata.get_stock_list_in_sector('沪深A股')
    # all_stocks = all_stocks + bj_stock_list
    return all_stocks

def get_stock_list_in_sector(sector: str):
    """获取板块成分股
        
        Args:
            sector: 板块名称
            
        Returns:
            股票列表
    """
    # 获取项目根目录路径
    project_root = get_project_root()
    
    # 添加绝对路径到系统路径
    to_path = os.path.join(project_root, "to")
    if to_path not in sys.path:
        sys.path.append(str(to_path))
    
    try:
        # 直接使用绝对路径导入
        from stock_sectors_mapping import get_sector_stocks
        
        # 使用导入的函数获取股票列表
        stocks = get_sector_stocks(sector_name=sector)
        return stocks
    except Exception as e:
        logger.error(f"获取板块 {sector} 成分股时出错: {e}")
        return []

def get_stock_sectors(stock_code: str):
    """获取股票所属的板块"""
      # 获取项目根目录路径
    project_root = get_project_root()
    
    # 添加绝对路径到系统路径
    to_path = os.path.join(project_root, "to")
    if to_path not in sys.path:
        sys.path.append(str(to_path))
    
    try:
        # 直接使用绝对路径导入
        from stock_sectors_mapping import get_stock_sectors
        
        # 使用导入的函数获取股票列表
        stocks = get_stock_sectors(stock_code=stock_code)
        return stocks
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 所属板块时出错: {e}")
        return []


def get_sector_list():
    """获取板块列表"""
     # 读取JSON文件
    project_root = get_project_root()
    json_file = os.path.join(project_root, "to", "sector_analysis", "tng_sectors_stats.json")
    with open(json_file, 'r', encoding='utf-8') as f:
        sector_stats = json.load(f)

    # 提取板块名称
    sectors = [item['sector_name'] for item in sector_stats if 'sector_name' in item]

    print(f"从JSON文件 {json_file} 加载了 {len(sectors)} 个概念板块")
    return sectors


def get_sector_list_with_stats():
    """获取板块列表及其统计信息"""
    # 读取JSON文件
    project_root = get_project_root()
    json_file = os.path.join(project_root, "to", "sector_analysis", "tng_sectors_stats.json")
    with open(json_file, 'r', encoding='utf-8') as f:
        sector_stats = json.load(f)

    # 创建板块名称到股票数量的映射
    sector_stock_count = {}
    for item in sector_stats:
        if 'sector_name' in item and 'stock_count' in item:
            sector_stock_count[item['sector_name']] = item['stock_count']

    print(f"从JSON文件 {json_file} 加载了 {len(sector_stock_count)} 个概念板块及其股票数量信息")
    return sector_stock_count


def get_trading_days(start_date: str, end_date: str) -> List[str]:
    """获取交易日列表
    
    Args:
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD
    """
    try:
        calendar_df = get_data_loader().get_trading_calendar(start_date, end_date)
        if calendar_df is not None and not calendar_df.empty:
            if 'date' in calendar_df.columns:
                dates = calendar_df['date'].dt.strftime('%Y%m%d').tolist()
            else:
                # 尝试使用第一列作为日期
                dates = calendar_df.iloc[:, 0].dt.strftime('%Y%m%d').tolist()
            return dates
        else:
            logger.warning("获取交易日历失败")
            return []
    except Exception as e:
        logger.error(f"获取交易日历时出错: {e}")
        return []
        

def _load_market_status(self, date):
    """
    加载市场状态数据
        
    Args:
        date: 日期，格式为YYYYMMDD
            
    Returns:
        dict: 市场状态数据
    """
    try:
        # 加载市场状态数据
        status_file = market_status_dir / "market_status.json"
        if not status_file.exists():
            self.logger.warning(f"市场状态文件不存在: {status_file}")
            return None
                
        with open(status_file, "r") as f:
            market_status_data = json.load(f)
                
        # 获取特定日期的市场状态
        if date in market_status_data:
            return market_status_data[date]
        else:
            self.logger.warning(f"日期 {date} 的市场状态数据不存在")
            return None
                
    except Exception as e:
            self.logger.error(f"加载日期 {date} 的市场状态数据时出错: {e}")
            self.logger.exception(e)
            return None


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
        calendar_df = loader.get_trading_calendar(start_date, end_date)
        if calendar_df is not None and not calendar_df.empty:
            if 'date_str' in calendar_df.columns:
                return calendar_df['date_str'].tolist()
            elif 'date' in calendar_df.columns:
                # 如果date列是datetime类型，转换为字符串
                if pd.api.types.is_datetime64_any_dtype(calendar_df['date']):
                    return calendar_df['date'].dt.strftime('%Y%m%d').tolist()
                else:
                    return calendar_df['date'].astype(str).tolist()
            else:
                # 使用第一列
                return calendar_df.iloc[:, 0].astype(str).tolist()
        else:
            return []
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
            print(f"stock_list: {stock_list}, start_date: {start_date}, end_date: {end_date}, fields: {fields}")
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


def get_stock_name(stock_code: str) -> str:
    """获取股票名称
    
    Args:
        stock_code: 股票代码
        
    Returns:
        股票名称，如果找不到则返回"未知"
    """
    global _STOCK_NAMES_CACHE
    
    # 如果缓存为空，加载股票名称数据
    if _STOCK_NAMES_CACHE is None:
        try:
            stock_names_file = get_data_path("stock_names.json")
            if os.path.exists(stock_names_file):
                with open(stock_names_file, "r", encoding="utf-8") as f:
                    _STOCK_NAMES_CACHE = json.load(f)
                logger.info(f"成功加载股票名称数据，共 {len(_STOCK_NAMES_CACHE)} 条")
            else:
                logger.warning(f"股票名称文件不存在: {stock_names_file}")
                _STOCK_NAMES_CACHE = {}
        except Exception as e:
            logger.error(f"加载股票名称数据时出错: {e}")
            _STOCK_NAMES_CACHE = {}
    
    # 从缓存中获取股票名称
    return _STOCK_NAMES_CACHE.get(stock_code, "未知")


def get_stock_names_dict() -> Dict[str, str]:
    """获取所有股票代码和名称的映射字典
    
    Returns:
        股票代码和名称的映射字典
    """
    global _STOCK_NAMES_CACHE
    
    # 如果缓存为空，加载股票名称数据
    if _STOCK_NAMES_CACHE is None:
        try:
            stock_names_file = get_data_path("stock_names.json")
            if os.path.exists(stock_names_file):
                with open(stock_names_file, "r", encoding="utf-8") as f:
                    _STOCK_NAMES_CACHE = json.load(f)
                logger.info(f"成功加载股票名称数据，共 {len(_STOCK_NAMES_CACHE)} 条")
            else:
                logger.warning(f"股票名称文件不存在: {stock_names_file}")
                _STOCK_NAMES_CACHE = {}
        except Exception as e:
            logger.error(f"加载股票名称数据时出错: {e}")
            _STOCK_NAMES_CACHE = {}
    
    return _STOCK_NAMES_CACHE


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


def main():
    print(get_stock_list_in_sector('TGN5G'))

if __name__ == "__main__":
    main()