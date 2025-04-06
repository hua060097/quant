#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强板块特征生成器模块

扩展基础板块特征生成器，提供更多高级特征，包括涨跌停、排名占比、趋势强度等。
"""

import os
import pandas as pd
import numpy as np
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Set

from to.sector_features.feature_generator import SectorFeatureGenerator

try:
    from to.path_utils import get_data_path
except ImportError:
    # 如果导入失败，提供简单的替代实现
    def get_data_path(relative_path=None):
        from pathlib import Path
        data_root = Path('to') / 'local_data'
        if relative_path:
            return data_root / relative_path
        return data_root

logger = logging.getLogger(__name__)

class EnhancedSectorFeatureGenerator(SectorFeatureGenerator):
    """增强版板块特征生成器
    
    在基础特征生成器的基础上，增加了更多高级特征，包括涨跌停、排名占比、趋势强度等。
    """
    
    def __init__(self, data_provider, sector_mapping, output_dir=None, 
                weighting_method='equal', ranking_data_path=None, top_n=100):
        """初始化增强特征生成器
        
        Args:
            data_provider: 数据提供者，需要实现get_daily_data和get_trading_days方法
            sector_mapping: 板块映射字典 {板块名称: [股票列表]}
            output_dir: 特征输出目录，默认使用path_utils.get_data_path('sector_features')
            weighting_method: 权重计算方法，可选值有equal(等权重)、market_cap(市值加权)等
            ranking_data_path: 排名数据路径，如果为None则不加载排名数据
            top_n: 排名前N的股票数量，用于计算排名占比
        """
        # 使用get_data_path设置默认输出目录
        if output_dir is None:
            output_dir = get_data_path('sector_features')
            
        super().__init__(data_provider, sector_mapping, output_dir, weighting_method)
        
        self.ranking_data_path = ranking_data_path
        self.top_n = top_n
        self.ranking_cache = {}  # 排名数据缓存
        
        # 基础涨跌停幅度配置（可根据股票类型动态调整）
        self.limit_up_pct = 0.1  # 默认涨停幅度10%
        self.limit_down_pct = -0.1  # 默认跌停幅度-10%
        
        # 特殊股票类型的涨跌停幅度
        self.special_limit_configs = {
            'ST': {'up': 0.05, 'down': -0.05},  # ST股票涨跌停5%
            'STAR': {'up': 0.2, 'down': -0.2},  # 科创板涨跌停20%
            'NEW': {'up': 0.2, 'down': -0.2}    # 新股涨跌停20%
        }
        
        logger.info(f"初始化增强板块特征生成器，权重方法: {weighting_method}，排名前N: {top_n}")
        
    def _load_sector_rankings(self, date):
        """加载特定日期的排名数据
        
        Args:
            date: 日期字符串
            
        Returns:
            排名DataFrame
        """
        if not self.ranking_data_path:
            return None
            
        # 检查缓存
        if date in self.ranking_cache:
            return self.ranking_cache[date]
            
        try:
            # 构建排名文件路径
            ranking_file = os.path.join(self.ranking_data_path, f"ranking_{date}.parquet")
            
            if not os.path.exists(ranking_file):
                logger.warning(f"日期 {date} 的排名文件不存在: {ranking_file}")
                return None
                
            # 读取排名数据
            ranking_df = pd.read_parquet(ranking_file)
            
            # 缓存结果
            self.ranking_cache[date] = ranking_df
            
            logger.debug(f"成功加载日期 {date} 的排名数据，共 {len(ranking_df)} 行")
            return ranking_df
            
        except Exception as e:
            logger.error(f"加载日期 {date} 的排名数据失败: {str(e)}")
            return None
            
    def _is_limit_up(self, current_price, prev_price, stock_code):
        """判断是否涨停
        
        Args:
            current_price: 当前价格
            prev_price: 前一日收盘价
            stock_code: 股票代码
            
        Returns:
            是否涨停
        """
        if prev_price <= 0:
            return False
            
        # 计算涨幅
        change_pct = (current_price - prev_price) / prev_price
        
        # 确定涨停幅度
        limit_up_pct = self.limit_up_pct
        
        # 根据股票类型调整涨停幅度
        if 'ST' in stock_code:
            limit_up_pct = self.special_limit_configs['ST']['up']
        elif stock_code.startswith('688'):  # 科创板股票
            limit_up_pct = self.special_limit_configs['STAR']['up']
            
        # 考虑精度误差，允许0.1%的误差
        return abs(change_pct - limit_up_pct) < 0.001
        
    def _is_limit_down(self, current_price, prev_price, stock_code):
        """判断是否跌停
        
        Args:
            current_price: 当前价格
            prev_price: 前一日收盘价
            stock_code: 股票代码
            
        Returns:
            是否跌停
        """
        if prev_price <= 0:
            return False
            
        # 计算跌幅
        change_pct = (current_price - prev_price) / prev_price
        
        # 确定跌停幅度
        limit_down_pct = self.limit_down_pct
        
        # 根据股票类型调整跌停幅度
        if 'ST' in stock_code:
            limit_down_pct = self.special_limit_configs['ST']['down']
        elif stock_code.startswith('688'):  # 科创板股票
            limit_down_pct = self.special_limit_configs['STAR']['down']
            
        # 考虑精度误差，允许0.1%的误差
        return abs(change_pct - limit_down_pct) < 0.001
        
    def _get_previous_trading_date(self, date):
        """获取前一个交易日
        
        Args:
            date: 日期字符串
            
        Returns:
            前一个交易日字符串
        """
        if isinstance(date, pd.Timestamp):
            date_str = date.strftime("%Y%m%d")
        elif isinstance(date, datetime):
            date_str = date.strftime("%Y%m%d")
        else:
            date_str = str(date).replace('-', '').replace('/', '')
            
        # 向前推7天，获取这段时间的交易日历
        date_obj = pd.to_datetime(date_str, format="%Y%m%d")
        start_date = (date_obj - timedelta(days=7)).strftime("%Y%m%d")
        
        try:
            # 获取交易日历
            trading_dates = self.data_provider.get_trading_days(start_date, date_str)
            
            # 处理不同格式的trading_dates
            if isinstance(trading_dates, pd.DataFrame):
                # 找到保存日期的列
                if 'date' in trading_dates.columns:
                    date_col = 'date'
                elif 'date_str' in trading_dates.columns:
                    date_col = 'date_str'
                else:
                    # 找不到日期列，使用第一列
                    date_col = trading_dates.columns[0]
                
                # 排序并获取所有交易日期
                trading_days_list = trading_dates[date_col].sort_values().tolist()
            else:
                # 列表格式
                trading_days_list = list(trading_dates)
                
            if len(trading_days_list) <= 1:
                logger.warning(f"获取日期 {date_str} 的前一个交易日失败，没有足够的交易日数据")
                return None
            
            # 标准化当前日期格式
            date_str_normalized = date_str.replace('-', '').replace('/', '')
            
            # 找到当前日期在列表中的位置
            current_date_index = -1
            
            # 遍历所有日期，尝试各种格式匹配
            for i, td in enumerate(trading_days_list):
                # 尝试多种标准化格式进行比较
                td_str = str(td)
                td_normalized = td_str.replace('-', '').replace('/', '')
                
                # 检查是否匹配(支持多种格式)
                if (td_normalized == date_str_normalized or 
                    td_str == date_str or
                    td_str.split()[0].replace('-', '') == date_str):
                    current_date_index = i
                    break
            
            # 如果找不到精确匹配，尝试查找最接近的日期
            if current_date_index == -1:
                try:
                    # 将当前日期转换为datetime对象
                    target_dt = pd.to_datetime(date_str_normalized, format='%Y%m%d')
                    
                    # 将所有交易日转换为datetime对象
                    trading_dts = []
                    for td in trading_days_list:
                        try:
                            if isinstance(td, pd.Timestamp):
                                trading_dts.append(td)
                            else:
                                trading_dts.append(pd.to_datetime(td))
                        except:
                            pass
                    
                    if trading_dts:
                        # 找到最接近但不超过目标日期的交易日
                        closest_i = -1
                        for i, dt in enumerate(trading_dts):
                            if dt <= target_dt and (closest_i == -1 or dt > trading_dts[closest_i]):
                                closest_i = i
                        
                        if closest_i != -1:
                            current_date_index = closest_i
                            logger.info(f"没有找到精确的日期匹配 {date_str}，使用最接近的交易日: {trading_days_list[current_date_index]}")
                except Exception as e:
                    logger.error(f"尝试查找最接近的交易日时出错: {e}")
            
            if current_date_index > 0:
                # 获取前一个交易日
                prev_date_value = trading_days_list[current_date_index - 1]
                # 确保前一交易日格式正确
                if isinstance(prev_date_value, pd.Timestamp):
                    prev_date = prev_date_value.strftime('%Y%m%d')
                else:
                    prev_date = str(prev_date_value).replace('-', '').replace('/', '')
                
                return prev_date
            else:
                logger.warning(f"无法在交易日历中找到日期 {date_str} 的位置，无法确定前一交易日")
                return None
                
        except Exception as e:
            logger.error(f"获取日期 {date_str} 的前一个交易日失败: {str(e)}")
            return None
        
    def _calculate_limit_features(self, sector, stocks, date):
        """计算涨跌停相关特征
        
        Args:
            sector: 板块名称
            stocks: 股票列表
            date: 日期字符串
            
        Returns:
            特征字典
        """
        features = {
            'limit_up_ratio': 0.0,
            'limit_down_ratio': 0.0,
            'near_limit_up_ratio': 0.0,
            'limit_open_ratio': 0.0,
        }
        
        try:
            # 获取前一个交易日
            prev_date = self._get_previous_trading_date(date)
            if not prev_date:
                logger.warning(f"无法获取日期 {date} 的前一个交易日，无法计算涨跌停特征")
                return features
                
            # 获取当前日期和前一日的数据
            try:
                current_data = self.data_provider.get_daily_data(stocks, date, date)
                prev_data = self.data_provider.get_daily_data(stocks, prev_date, prev_date)
                
                if current_data.empty or prev_data.empty:
                    logger.warning(f"日期 {date} 或前一日 {prev_date} 的数据为空，无法计算涨跌停特征")
                    return features
                    
            except Exception as e:
                logger.error(f"获取日期 {date} 和前一日 {prev_date} 的数据失败: {str(e)}")
                return features
                
            # 创建前一日收盘价字典
            prev_close_dict = {}
            for _, row in prev_data.iterrows():
                prev_close_dict[row['stock_code']] = row['close']
                
            # 计算涨跌停统计
            limit_up_count = 0
            limit_down_count = 0
            near_limit_up_count = 0  # 接近涨停（涨幅>7%）
            limit_open_count = 0  # 一字板开盘涨停
            
            valid_stock_count = 0
            
            for _, row in current_data.iterrows():
                stock_code = row['stock_code']
                
                if stock_code not in prev_close_dict:
                    continue
                    
                valid_stock_count += 1
                prev_close = prev_close_dict[stock_code]
                
                # 计算涨跌幅
                change_pct = (row['close'] - prev_close) / prev_close if prev_close > 0 else 0
                
                # 检查是否涨停
                if self._is_limit_up(row['close'], prev_close, stock_code):
                    limit_up_count += 1
                    
                # 检查是否跌停
                elif self._is_limit_down(row['close'], prev_close, stock_code):
                    limit_down_count += 1
                    
                # 检查是否接近涨停
                elif change_pct > 0.07:  # 涨幅超过7%
                    near_limit_up_count += 1
                    
                # 检查是否一字板开盘涨停（开盘价等于最高价等于最低价等于收盘价）
                if (row['open'] == row['high'] == row['low'] == row['close'] and
                    self._is_limit_up(row['open'], prev_close, stock_code)):
                    limit_open_count += 1
            
            # 计算比例
            if valid_stock_count > 0:
                features['limit_up_ratio'] = limit_up_count / valid_stock_count
                features['limit_down_ratio'] = limit_down_count / valid_stock_count
                features['near_limit_up_ratio'] = near_limit_up_count / valid_stock_count
                features['limit_open_ratio'] = limit_open_count / valid_stock_count
                
            logger.debug(f"板块 {sector} 在日期 {date} 的涨停比例: {features['limit_up_ratio']:.2f}, "
                        f"跌停比例: {features['limit_down_ratio']:.2f}")
                        
        except Exception as e:
            logger.error(f"计算板块 {sector} 在日期 {date} 的涨跌停特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
        return features
        
    def _calculate_ranking_features(self, sector, stocks, date):
        """计算排名相关特征
        
        Args:
            sector: 板块名称
            stocks: 股票列表
            date: 日期字符串
            
        Returns:
            特征字典
        """
        features = {
            'top_volume_ratio': 0.0,
            'top_change_ratio': 0.0,
            'top_turn_ratio': 0.0,
            'top100_count': 0
        }
        
        try:
            # 加载排名数据
            ranking_df = self._load_sector_rankings(date)
            if ranking_df is None or ranking_df.empty:
                logger.warning(f"日期 {date} 的排名数据为空，无法计算排名特征")
                return features
                
            # 创建股票集合，便于快速查找
            stocks_set = set(stocks)
            
            def calculate_weighted_ranking(ranking_df, stocks_set):
                """计算加权排名
                
                给排名靠前的股票更高的权重，计算板块内股票的加权排名比例
                
                Args:
                    ranking_df: 排名DataFrame
                    stocks_set: 板块股票集合
                    
                Returns:
                    加权排名比例
                """
                total_weight = sum(1/(i+1) for i in range(self.top_n))
                sector_weight = 0
                
                for i, row in ranking_df.iterrows():
                    if i >= self.top_n:
                        break
                        
                    if row['stock_code'] in stocks_set:
                        # 排名越靠前，权重越大
                        sector_weight += 1/(i+1)
                        
                return sector_weight / total_weight if total_weight > 0 else 0
            
            # 计算各类排名特征
            # 1. 成交量排名
            if 'volume_rank' in ranking_df.columns:
                volume_ranking = ranking_df.sort_values('volume_rank')
                features['top_volume_ratio'] = calculate_weighted_ranking(volume_ranking, stocks_set)
                
            # 2. 涨跌幅排名
            if 'change_rank' in ranking_df.columns:
                change_ranking = ranking_df.sort_values('change_rank')
                features['top_change_ratio'] = calculate_weighted_ranking(change_ranking, stocks_set)
                
            # 3. 换手率排名
            if 'turnover_rank' in ranking_df.columns:
                turn_ranking = ranking_df.sort_values('turnover_rank')
                features['top_turn_ratio'] = calculate_weighted_ranking(turn_ranking, stocks_set)
                
            # 4. Top100股票数量
            top100_stocks = set(ranking_df.iloc[:100]['stock_code']) if len(ranking_df) >= 100 else set()
            features['top100_count'] = len(top100_stocks.intersection(stocks_set))
            
            logger.debug(f"板块 {sector} 在日期 {date} 的成交量排名比例: {features['top_volume_ratio']:.2f}, "
                        f"涨跌幅排名比例: {features['top_change_ratio']:.2f}")
                        
        except Exception as e:
            logger.error(f"计算板块 {sector} 在日期 {date} 的排名特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
        return features
        
    def generate_sector_features_for_date(self, sector, date, preloaded_data=None):
        """为指定日期生成板块特征，增强版本
        
        Args:
            sector: 板块名称
            date: 日期字符串，格式为YYYYMMDD
            preloaded_data: 预加载的数据，如果为None则自动加载
            
        Returns:
            特征DataFrame
        """
        try:
            # 调用父类方法获取基础特征
            base_features_df = super().generate_sector_features_for_date(sector, date, preloaded_data)
            
            if base_features_df.empty:
                logger.warning(f"板块 {sector} 在日期 {date} 的基础特征为空")
                return base_features_df
                
            # 获取板块成分股
            stocks = self.sector_mapping.get(sector, [])
            if not stocks:
                # 尝试从stock_sectors_mapping获取
                from to.stock_sectors_mapping import get_sector_stocks
                stocks = get_sector_stocks(sector_name=sector)
                
            if not stocks:
                logger.warning(f"板块 {sector} 没有找到成分股，无法计算增强特征")
                return base_features_df
                
            # 1. 计算涨跌停相关特征
            limit_features = self._calculate_limit_features(sector, stocks, date)
            
            # 2. 计算排名相关特征
            ranking_features = self._calculate_ranking_features(sector, stocks, date)
            
            # 3. 计算资金流向特征
            # 这部分需要额外的资金流向数据，这里暂不实现
            
            # 合并所有特征
            # 将字典转换为Series
            enhanced_features = pd.Series({**limit_features, **ranking_features})
            
            # 合并到基础特征中
            for feature, value in enhanced_features.items():
                base_features_df[feature] = value
                
            logger.info(f"成功为板块 {sector} 在日期 {date} 生成增强特征")
            return base_features_df
            
        except Exception as e:
            logger.error(f"生成板块 {sector} 在日期 {date} 的增强特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 如果增强特征生成失败，至少返回基础特征
            try:
                return super().generate_sector_features_for_date(sector, date, preloaded_data)
            except:
                return pd.DataFrame()

    def generate_and_save_features(self, sector, start_date, end_date):
        """生成并保存板块特征数据
        
        Args:
            sector: 板块名称
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            
        Returns:
            成功生成特征的数量
        """
        # 直接调用父类的方法，因为我们已经修改了父类的方法来支持所有所需的存储格式
        return super().generate_and_save_features(sector, start_date, end_date)

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 导入必要的模块
    from to.sector_features.data_provider import XtDataProvider
    
    # 初始化数据提供器
    data_provider = XtDataProvider()
    
    # 获取板块映射
    sector = "TGN人工智能"
    import xtquant.xtdata as xtdata
    xtdata.connect()
    
    sector_mapping = {
        sector: xtdata.get_stock_list_in_sector(sector)
    }
    
    # 创建增强特征生成器
    enhanced_generator = EnhancedSectorFeatureGenerator(
        data_provider=data_provider,
        sector_mapping=sector_mapping,
        output_dir=get_data_path('sector_features'),
        weighting_method='equal'
    )
    
    # 生成特征
    date = '20230103'
    features = enhanced_generator.generate_sector_features_for_date(sector, date)
    
    # 打印特征
    if features.empty:
        print("未能生成特征")
    else:
        for k, v in features.items():
            print(f"{k}: {v}") 