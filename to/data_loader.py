import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SectorDataLoader:
    """板块数据加载器 - 模拟版
    
    用于本地测试的简化版数据加载器，不依赖xtquant，生成随机数据
    """
    
    def __init__(self):
        """初始化数据加载器"""
        self.cache = {}  # 用于缓存数据
    
    def get_trading_dates(self, start_date, end_date):
        """获取交易日期列表
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            
        Returns:
            交易日期列表
        """
        try:
            # 将字符串日期转换为datetime对象
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            
            # 生成日期范围内的所有日期
            all_dates = []
            current_dt = start_dt
            while current_dt <= end_dt:
                # 跳过周末
                if current_dt.weekday() < 5:  # 0-4为周一至周五
                    all_dates.append(current_dt.strftime('%Y%m%d'))
                current_dt += timedelta(days=1)
            
            logger.info(f"获取到 {len(all_dates)} 个交易日")
            return all_dates
        except Exception as e:
            logger.error(f"获取交易日期失败: {e}")
            return []
    
    def get_stock_list_in_sector(self, sector_name):
        """获取板块成分股列表
        
        Args:
            sector_name: 板块名称
            
        Returns:
            成分股代码列表
        """
        # 为每个板块生成固定的模拟成分股列表
        if sector_name == 'TGN人工智能':
            return ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH', '600519.SH', '002415.SZ']
        elif sector_name.startswith('TGN'):
            # 为其他TGN板块生成随机成分股列表
            stock_count = hash(sector_name) % 30 + 5  # 5-34只股票
            
            # 沪市股票
            sh_stocks = [f"60{i:04d}.SH" for i in range(10)]
            # 深市股票
            sz_stocks = [f"00{i:04d}.SZ" for i in range(10)]
            # 创业板股票
            cy_stocks = [f"30{i:04d}.SZ" for i in range(10)]
            
            all_stocks = sh_stocks + sz_stocks + cy_stocks
            # 使用固定种子确保每次相同板块生成相同的成分股
            np.random.seed(hash(sector_name))
            selected_stocks = np.random.choice(all_stocks, min(stock_count, len(all_stocks)), replace=False)
            
            return selected_stocks.tolist()
        else:
            return []
    
    def get_sector_data(self, sector_name, date_str, lookback_days=30):
        """获取板块行情数据
        
        Args:
            sector_name: 板块名称
            date_str: 日期字符串，格式为YYYYMMDD
            lookback_days: 回溯天数
            
        Returns:
            板块行情数据DataFrame
        """
        # 缓存键
        cache_key = f"sector_{sector_name}_{date_str}_{lookback_days}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 计算开始日期
            end_dt = datetime.strptime(date_str, '%Y%m%d')
            start_dt = end_dt - timedelta(days=lookback_days * 2)  # 乘以2以确保有足够的交易日
            
            # 获取交易日列表
            trading_dates = self.get_trading_dates(start_dt.strftime('%Y%m%d'), date_str)
            
            # 为方便模拟，只保留最近的lookback_days个交易日
            if len(trading_dates) > lookback_days:
                trading_dates = trading_dates[-lookback_days:]
            
            # 创建行情数据
            data = []
            close_price = 100.0  # 初始价格
            for date in trading_dates:
                # 生成随机价格变动
                change_pct = np.random.normal(0, 0.02)  # 均值0，标准差2%
                close_price *= (1 + change_pct)
                
                # 确保价格为正
                close_price = max(close_price, 1.0)
                
                # 生成当天的高低价
                high_price = close_price * (1 + abs(np.random.normal(0, 0.01)))
                low_price = close_price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = low_price + (high_price - low_price) * np.random.random()
                
                # 确保开盘价在最高价和最低价之间
                open_price = max(min(open_price, high_price), low_price)
                
                # 生成成交量和成交额
                volume = np.random.uniform(1e6, 1e7)
                amount = volume * close_price
                
                data.append({
                    'date': date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'amount': amount,
                    'change': change_pct * 100  # 涨跌幅，百分比
                })
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            # 缓存结果
            self.cache[cache_key] = df
            
            return df
        except Exception as e:
            logger.error(f"获取板块 {sector_name} 行情数据失败: {e}")
            return None
    
    def get_individual_stocks_data(self, sector_name, date_str, lookback_days=30):
        """获取板块成分股行情数据
        
        Args:
            sector_name: 板块名称
            date_str: 日期字符串，格式为YYYYMMDD
            lookback_days: 回溯天数
            
        Returns:
            成分股行情数据字典 {股票代码: DataFrame}
        """
        # 缓存键
        cache_key = f"stocks_{sector_name}_{date_str}_{lookback_days}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 获取成分股列表
            stock_list = self.get_stock_list_in_sector(sector_name)
            if not stock_list:
                logger.warning(f"板块 {sector_name} 没有成分股")
                return {}
            
            # 获取交易日列表
            end_dt = datetime.strptime(date_str, '%Y%m%d')
            start_dt = end_dt - timedelta(days=lookback_days * 2)
            trading_dates = self.get_trading_dates(start_dt.strftime('%Y%m%d'), date_str)
            
            # 为方便模拟，只保留最近的lookback_days个交易日
            if len(trading_dates) > lookback_days:
                trading_dates = trading_dates[-lookback_days:]
            
            # 生成成分股数据
            stocks_data = {}
            for stock_code in stock_list:
                # 为每只股票生成随机数据
                data = []
                close_price = np.random.uniform(10, 100)  # 初始价格
                
                for date in trading_dates:
                    # 生成随机价格变动
                    change_pct = np.random.normal(0, 0.02)
                    close_price *= (1 + change_pct)
                    
                    # 确保价格为正
                    close_price = max(close_price, 1.0)
                    
                    # 生成当天的高低价
                    high_price = close_price * (1 + abs(np.random.normal(0, 0.01)))
                    low_price = close_price * (1 - abs(np.random.normal(0, 0.01)))
                    open_price = low_price + (high_price - low_price) * np.random.random()
                    
                    # 生成成交量和成交额
                    volume = np.random.uniform(1e5, 1e6)
                    amount = volume * close_price
                    
                    data.append({
                        'date': date,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume,
                        'amount': amount,
                        'change': change_pct * 100  # 涨跌幅，百分比
                    })
                
                # 创建DataFrame
                df = pd.DataFrame(data)
                df.set_index('date', inplace=True)
                
                # 保存到结果字典
                stocks_data[stock_code] = df
            
            # 缓存结果
            self.cache[cache_key] = stocks_data
            
            return stocks_data
        except Exception as e:
            logger.error(f"获取板块 {sector_name} 成分股行情数据失败: {e}")
            return {}
    
    def get_daily_data(self, stock_list, start_date, end_date, fields=None):
        """获取日线数据
        
        Args:
            stock_list: 股票代码列表
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            fields: 字段列表，默认为None，表示获取所有字段
            
        Returns:
            日线数据字典 {股票代码: DataFrame}
        """
        # 缓存键
        fields_str = ','.join(fields) if fields else 'all'
        cache_key = f"daily_{','.join(stock_list)}_{start_date}_{end_date}_{fields_str}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 获取交易日列表
            trading_dates = self.get_trading_dates(start_date, end_date)
            
            # 生成日线数据
            daily_data = {}
            for stock_code in stock_list:
                # 为每只股票生成随机数据
                data = []
                close_price = np.random.uniform(10, 100)  # 初始价格
                
                for date in trading_dates:
                    # 生成随机价格变动
                    change_pct = np.random.normal(0, 0.02)
                    close_price *= (1 + change_pct)
                    
                    # 确保价格为正
                    close_price = max(close_price, 1.0)
                    
                    # 生成当天的高低价
                    high_price = close_price * (1 + abs(np.random.normal(0, 0.01)))
                    low_price = close_price * (1 - abs(np.random.normal(0, 0.01)))
                    open_price = low_price + (high_price - low_price) * np.random.random()
                    
                    # 生成成交量和成交额
                    volume = np.random.uniform(1e5, 1e6)
                    amount = volume * close_price
                    
                    data.append({
                        'date': date,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume,
                        'amount': amount,
                        'change': change_pct * 100  # 涨跌幅，百分比
                    })
                
                # 创建DataFrame
                df = pd.DataFrame(data)
                df.set_index('date', inplace=True)
                
                # 如果指定了字段，只保留这些字段
                if fields:
                    df = df[fields]
                
                # 保存到结果字典
                daily_data[stock_code] = df
            
            # 缓存结果
            self.cache[cache_key] = daily_data
            
            return daily_data
        except Exception as e:
            logger.error(f"获取日线数据失败: {e}")
            return {}
    
    def get_market_data(self, stock_list, field_list, period='1d', start_time=None, end_time=None):
        """获取市场数据，兼容xtquant接口
        
        Args:
            stock_list: 股票代码列表
            field_list: 字段列表
            period: 周期，默认为'1d'
            start_time: 开始时间，格式为YYYYMMDD
            end_time: 结束时间，格式为YYYYMMDD
            
        Returns:
            市场数据字典 {字段: DataFrame}
        """
        # 如果是日线数据，调用get_daily_data
        if period == '1d':
            daily_data = self.get_daily_data(stock_list, start_time, end_time, field_list)
            
            # 转换为xtquant接口返回的格式
            result = {}
            for field in field_list:
                if field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    # 创建一个多层索引的DataFrame
                    field_data = pd.DataFrame()
                    for stock_code, df in daily_data.items():
                        if field in df.columns:
                            field_data[stock_code] = df[field]
                    
                    # 转置，使stock_code成为索引，date成为列
                    field_data = field_data.transpose()
                    
                    result[field] = field_data
            
            return result
        else:
            logger.warning(f"目前只支持日线数据，不支持周期 {period}")
            return None 