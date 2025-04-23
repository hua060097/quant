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

# 导入logger配置
from to.logger_config import configure_logger

# 使用共享的logger配置
logger = configure_logger(__name__)

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
            'NEW': {'up': 0.2, 'down': -0.2},   # 新股涨跌停20%
            'STAR': {'up': 0.2, 'down': -0.2},  # 科创板涨跌停20%
            'GEM': {'up': 0.2, 'down': -0.2},   # 创业板涨跌停20%
            'BSE': {'up': 0.3, 'down': -0.3}    # 北交所涨跌停30%
        }
        
        logger.info(f"初始化增强板块特征生成器，权重方法: {weighting_method}，排名前N: {top_n}")
        
    def _load_sector_rankings(self, date):
        """加载特定日期的排名数据
        
        Args:
            date: 日期字符串
            
        Returns:
            排名字典，包含四种排名数据
        """
        from stock_daily_rankings import get_all_rankings_for_date
        
        # 检查缓存
        if date in self.ranking_cache:
            return self.ranking_cache[date]
            
        try:
            # 使用stock_daily_rankings.py提供的API获取排名数据
            all_rankings = get_all_rankings_for_date(date, as_dataframe=True)
            
            if all_rankings is None:
                logger.warning(f"日期 {date} 的排名数据获取失败")
                return None
                
            # 缓存结果
            self.ranking_cache[date] = all_rankings
            
            logger.debug(f"成功加载日期 {date} 的排名数据，包含十日涨幅:{len(all_rankings['ten_day_rise'])}条，"
                        f"日涨幅:{len(all_rankings['daily_rise'])}条，"
                        f"日跌幅:{len(all_rankings['daily_fall'])}条，"
                        f"成交额:{len(all_rankings['daily_amount'])}条")
            return all_rankings
            
        except Exception as e:
            logger.error(f"加载日期 {date} 的排名数据失败: {str(e)}")
            return None
            
    def _is_limit_up(self, pct_chg, stock_code, close=None, pre_close=None):
        """判断是否涨停
        
        基于前收盘价计算精确的涨停价，进行四舍五入后与当前收盘价比较
        
        Args:
            pct_chg: 涨跌幅百分比（如果没有提供close和pre_close，则使用此值判断）
            stock_code: 股票代码
            close: 当前收盘价
            pre_close: 前收盘价
            
        Returns:
            是否涨停
        """
        # 确定涨停幅度比率
        limit_ratio = self.limit_up_pct  # 默认10%
        
        # 根据股票类型调整涨停幅度
        if 'ST' in stock_code:
            limit_ratio = self.special_limit_configs['ST']['up']
        elif stock_code.startswith('688'):  # 科创板股票
            limit_ratio = self.special_limit_configs['STAR']['up']
        elif stock_code.startswith('300'):  # 创业板股票
            limit_ratio = self.special_limit_configs['GEM']['up']
        elif '.BJ' in stock_code:  # 北交所股票
            limit_ratio = self.special_limit_configs['BSE']['up']
            
        # 如果提供了收盘价和前收盘价，使用精确计算
        if close is not None and pre_close is not None and pre_close > 0:
            # 计算理论涨停价
            theoretical_limit = pre_close * (1 + limit_ratio)
            # 四舍五入到0.01元
            rounded_limit = round(theoretical_limit, 2)
            # 判断当前价格是否等于四舍五入后的涨停价
            return abs(close - rounded_limit) < 0.005
        else:
            # 如果未提供价格数据，则使用涨跌幅判断，允许一定误差
            limit_pct = limit_ratio * 100
            return abs(pct_chg - limit_pct) <= 0.15
        
    def _is_limit_down(self, pct_chg, stock_code, close=None, pre_close=None):
        """判断是否跌停
        
        基于前收盘价计算精确的跌停价，进行四舍五入后与当前收盘价比较
        
        Args:
            pct_chg: 涨跌幅百分比（如果没有提供close和pre_close，则使用此值判断）
            stock_code: 股票代码
            close: 当前收盘价
            pre_close: 前收盘价
            
        Returns:
            是否跌停
        """
        # 确定跌停幅度比率（负值）
        limit_ratio = self.limit_down_pct  # 默认-10%
        
        # 根据股票类型调整跌停幅度
        if 'ST' in stock_code:
            limit_ratio = self.special_limit_configs['ST']['down']
        elif stock_code.startswith('688'):  # 科创板股票
            limit_ratio = self.special_limit_configs['STAR']['down']
        elif stock_code.startswith('300'):  # 创业板股票
            limit_ratio = self.special_limit_configs['GEM']['down']
        elif '.BJ' in stock_code:  # 北交所股票
            limit_ratio = self.special_limit_configs['BSE']['down']
            
        # 如果提供了收盘价和前收盘价，使用精确计算
        if close is not None and pre_close is not None and pre_close > 0:
            # 计算理论跌停价
            theoretical_limit = pre_close * (1 + limit_ratio)
            # 四舍五入到0.01元
            rounded_limit = round(theoretical_limit, 2)
            # 判断当前价格是否等于四舍五入后的跌停价
            return abs(close - rounded_limit) < 0.005
        else:
            # 如果未提供价格数据，则使用涨跌幅判断，允许一定误差
            limit_pct = limit_ratio * 100
            return abs(pct_chg - limit_pct) <= 0.15
        
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
                                # 确保字符串转换为datetime时格式正确
                                td_str = str(td).replace('-', '').replace('/', '')
                                if td_str.isdigit() and len(td_str) >= 8:  # 确保是有效的日期格式
                                    trading_dts.append(pd.to_datetime(td_str[:8], format='%Y%m%d'))
                                else:
                                    trading_dts.append(pd.to_datetime(td))
                        except Exception as e:
                            logger.warning(f"转换日期 {td} 失败: {e}")
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
                prev_trading_day = trading_days_list[current_date_index - 1]
                
                # 确保返回格式统一为YYYYMMDD字符串
                if isinstance(prev_trading_day, pd.Timestamp):
                    return prev_trading_day.strftime('%Y%m%d')
                elif isinstance(prev_trading_day, datetime):
                    return prev_trading_day.strftime('%Y%m%d')
                else:
                    # 将其他格式转换为标准字符串格式
                    prev_trading_day_str = str(prev_trading_day)
                    # 移除所有分隔符并确保格式正确
                    prev_date_normalized = prev_trading_day_str.replace('-', '').replace('/', '')
                    if prev_date_normalized.isdigit() and len(prev_date_normalized) >= 8:
                        return prev_date_normalized[:8]  # 只取前8位，保证是YYYYMMDD格式
                    else:
                        try:
                            # 尝试解析日期字符串并转换为标准格式
                            dt = pd.to_datetime(prev_trading_day_str)
                            return dt.strftime('%Y%m%d')
                        except:
                            # 无法解析则原样返回
                            return prev_trading_day_str
                
            else:
                logger.warning(f"无法找到日期 {date_str} 的前一个交易日")
                return None
                
        except Exception as e:
            logger.error(f"获取日期 {date_str} 的前一个交易日失败: {e}")
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
            'avg_turnover_rate': 0.0,  # 添加平均换手率特征
        }
        
        try:
            # 直接获取当前日期的数据（不需要获取前一日数据，因为已有pct_chg）
            try:
                current_data = self.data_provider.get_daily_data(stocks, date, date)
                
                if current_data.empty:
                    logger.warning(f"日期 {date} 的数据为空，无法计算涨跌停特征")
                    return features
                    
            except Exception as e:
                logger.error(f"获取日期 {date} 的数据失败: {str(e)}")
                return features
                
            # 计算涨跌停统计
            limit_up_count = 0
            limit_down_count = 0
            near_limit_up_count = 0  # 接近涨停（涨幅>7%）
            limit_open_count = 0  # 一字板开盘涨停
            total_turnover_rate = 0.0
            
            valid_stock_count = 0
            
            for _, row in current_data.iterrows():
                stock_code = row['stock_code']
                valid_stock_count += 1
                
                # 获取涨跌幅和价格数据
                pct_chg = row['pct_chg'] if 'pct_chg' in row else 0
                close = row['close']
                pre_close = row['preClose'] if 'preClose' in row else None
                
                # 检查是否涨停（使用精确计算）
                if self._is_limit_up(pct_chg, stock_code, close, pre_close):
                    limit_up_count += 1
                    
                # 检查是否跌停（使用精确计算）
                elif self._is_limit_down(pct_chg, stock_code, close, pre_close):
                    limit_down_count += 1
                    
                # 检查是否接近涨停
                elif pct_chg > 7.0:  # 涨幅超过7%
                    near_limit_up_count += 1
                    
                # 检查是否一字板开盘涨停
                # 允许极小误差，以处理数据精度问题
                is_one_price_board = (
                    abs(row['open'] - row['high']) < 0.005 and
                    abs(row['high'] - row['low']) < 0.005 and
                    abs(row['low'] - row['close']) < 0.005
                )
                
                if is_one_price_board and self._is_limit_up(pct_chg, stock_code, close, pre_close):
                    limit_open_count += 1
                
                # 统计换手率
                turnover_rate = row.get('turnover_rate', 0)
                total_turnover_rate += turnover_rate
            
            # 计算比例
            if valid_stock_count > 0:
                features['limit_up_ratio'] = limit_up_count / valid_stock_count
                features['limit_down_ratio'] = limit_down_count / valid_stock_count
                features['near_limit_up_ratio'] = near_limit_up_count / valid_stock_count
                features['limit_open_ratio'] = limit_open_count / valid_stock_count
                features['avg_turnover_rate'] = total_turnover_rate / valid_stock_count
                
            logger.debug(f"板块 {sector} 在日期 {date} 的涨停比例: {features['limit_up_ratio']:.2f}, "
                        f"跌停比例: {features['limit_down_ratio']:.2f}, "
                        f"平均换手率: {features['avg_turnover_rate']:.2f}%")
                        
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
            # 十日涨幅排名特征
            'ten_day_rise_count': 0,         # 板块内股票在十日涨幅榜的数量
            'ten_day_rise_ratio': 0.0,       # 板块内股票在十日涨幅榜的占比
            # 当日涨幅排名特征
            'daily_rise_count': 0,           # 板块内股票在日涨幅榜的数量
            'daily_rise_ratio': 0.0,         # 板块内股票在日涨幅榜的占比
            # 当日跌幅排名特征
            'daily_fall_count': 0,           # 板块内股票在日跌幅榜的数量
            'daily_fall_ratio': 0.0,         # 板块内股票在日跌幅榜的占比
            # 成交额排名特征
            'daily_amount_count': 0,         # 板块内股票在成交额榜的数量
            'daily_amount_ratio': 0.0        # 板块内股票在成交额榜的占比
        }
        
        try:
            # 加载排名数据
            rankings = self._load_sector_rankings(date)
            if rankings is None:
                logger.warning(f"日期 {date} 的排名数据为空，无法计算排名特征")
                return features
                
            # 创建股票集合，便于快速查找
            stocks_set = set(stocks)
            
            # 计算十日涨幅排名特征
            if 'ten_day_rise' in rankings and not rankings['ten_day_rise'].empty:
                ten_day_stocks = set(rankings['ten_day_rise']['code'])
                intersection = ten_day_stocks.intersection(stocks_set)
                features['ten_day_rise_count'] = len(intersection)
                features['ten_day_rise_ratio'] = len(intersection) / len(ten_day_stocks) if len(ten_day_stocks) > 0 else 0.0
            
            # 计算日涨幅排名特征
            if 'daily_rise' in rankings and not rankings['daily_rise'].empty:
                daily_rise_stocks = set(rankings['daily_rise']['code'])
                intersection = daily_rise_stocks.intersection(stocks_set)
                features['daily_rise_count'] = len(intersection)
                features['daily_rise_ratio'] = len(intersection) / len(daily_rise_stocks) if len(daily_rise_stocks) > 0 else 0.0
            
            # 计算日跌幅排名特征
            if 'daily_fall' in rankings and not rankings['daily_fall'].empty:
                daily_fall_stocks = set(rankings['daily_fall']['code'])
                intersection = daily_fall_stocks.intersection(stocks_set)
                features['daily_fall_count'] = len(intersection)
                features['daily_fall_ratio'] = len(intersection) / len(daily_fall_stocks) if len(daily_fall_stocks) > 0 else 0.0
            
            # 计算成交额排名特征
            if 'daily_amount' in rankings and not rankings['daily_amount'].empty:
                daily_amount_stocks = set(rankings['daily_amount']['code'])
                intersection = daily_amount_stocks.intersection(stocks_set)
                features['daily_amount_count'] = len(intersection)
                features['daily_amount_ratio'] = len(intersection) / len(daily_amount_stocks) if len(daily_amount_stocks) > 0 else 0.0
            
            # 计算加权特征：根据排名位置加权
            if 'ten_day_rise' in rankings and not rankings['ten_day_rise'].empty:
                weighted_score = 0
                for i, row in rankings['ten_day_rise'].iterrows():
                    if row['code'] in stocks_set:
                        # 排名越靠前，权重越大
                        rank_weight = 1.0 / (i + 1)
                        weighted_score += rank_weight
                # 归一化
                features['ten_day_rise_weighted'] = weighted_score
            else:
                features['ten_day_rise_weighted'] = 0.0
                
            logger.debug(f"板块 {sector} 在日期 {date} 的十日涨幅榜占比: {features['ten_day_rise_ratio']:.2f}, "
                        f"日涨幅榜占比: {features['daily_rise_ratio']:.2f}")
                        
        except Exception as e:
            logger.error(f"计算板块 {sector} 在日期 {date} 的排名特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
        return features
        
    def generate_sector_features_for_date(self, sector, date, preloaded_data=None):
        """计算指定日期的增强板块特征
        
        包括基础特征和高级特征（涨跌停、排名占比、趋势强度等）
        
        Args:
            sector: 板块名称
            date: 日期
            preloaded_data: 预加载的数据，如果提供则不再重新加载
            
        Returns:
            (成功标志, 板块特征DataFrame)
        """
        try:
            # 首先获取基础特征
            success, base_features_df = super().generate_sector_features_for_date(sector, date, preloaded_data)
            
            if not success or base_features_df is None or base_features_df.empty:
                logger.warning(f"无法获取板块 {sector} 在日期 {date} 的基础特征")
                return False, None
                
            # 获取板块股票
            stocks = self._get_sector_stocks(sector)
            if not stocks:
                logger.warning(f"板块 {sector} 没有成分股")
                return False, None
            
            # 增加涨跌停特征
            limit_features = self._calculate_limit_features(sector, stocks, date)
            if limit_features:
                for feature_name, feature_value in limit_features.items():
                    base_features_df[feature_name] = feature_value
            
            # 增加排名特征
            ranking_features = self._calculate_ranking_features(sector, stocks, date)
            if ranking_features:
                for feature_name, feature_value in ranking_features.items():
                    base_features_df[feature_name] = feature_value
                    
            # 返回合并后的特征
            logger.debug(f"成功生成板块 {sector} 在日期 {date} 的增强特征，共 {len(base_features_df.columns)} 个特征")
            return True, base_features_df
            
        except Exception as e:
            logger.error(f"生成板块 {sector} 在日期 {date} 的增强特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None

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
    from to.generate_sector_features import DataLoaderWrapper
    
    # 初始化数据提供器
    data_provider = DataLoaderWrapper()
    
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
    success, features = enhanced_generator.generate_sector_features_for_date(sector, date)
    
    # 打印特征
    if not success or features is None or features.empty:
        print("未能生成特征")
    else:
        for k, v in features.items():
            print(f"{k}: {v}") 