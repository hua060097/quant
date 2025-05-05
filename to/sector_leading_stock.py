#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sector_leading_stocks.py

板块龙头股识别与特征计算模块
用于计算板块内龙头股指标、换手率分布、涨停分布等特征
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import sys
import traceback
import gc
import sys
from pathlib import Path
import math
import pickle
import threading
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录


log_file = 'to/log/sector_leading_stock.log'
file_handler = logging.FileHandler(log_file, encoding='utf-8')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler]
)
logger = logging.getLogger(__name__)

# 尝试导入所需模块
try:
    from to.data_loaders import get_daily_data, get_stock_list_in_sector, get_sector_list
except ImportError:
    # 添加项目根目录到sys.path
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    sys.path.append(str(project_root))
    try:
        from to.data_loaders import get_daily_data, get_stock_list_in_sector, get_sector_list
    except ImportError:
        logger.error("无法导入data_loaders模块，请确保该模块存在")
        
class SectorLeadingStocksAnalyzer:
    """板块龙头股分析器"""
    
    class StockDataAccessor:
        """股票数据访问器，提供类似字典的访问接口"""
        
        def __init__(self, data_dict, date_range):
            """
            初始化股票数据访问器
            
            Args:
                data_dict: 股票数据字典，格式为 {日期: 数据}
                date_range: 日期范围
            """
            self.data = data_dict
            self.date_range = date_range
            
        def __getitem__(self, date):
            """字典式访问接口"""
            return self.data.get(date)
            
        def get(self, date, default=None):
            """获取指定日期的数据"""
            return self.data.get(date, default)
            
        def get_coverage(self):
            """获取数据覆盖率"""
            return len(self.data) / len(self.date_range) if self.date_range else 0
            
        def is_valid(self):
            """检查数据是否有效"""
            return len(self.data) > 0
            
        def keys(self):
            """返回所有日期键"""
            return self.data.keys()
            
        def items(self):
            """返回所有(日期,数据)对"""
            return self.data.items()
            
        def values(self):
            """返回所有数据值"""
            return self.data.values()
    
    def __init__(self, output_dir=None, cache_dir=None):
        """
        初始化板块龙头股分析器
        
        Args:
            output_dir: 输出目录，默认为'to/local_data/sector_features'
            cache_dir: 缓存目录，默认为'to/local_data/features_cache/leading_stocks'
        """
        self.logger = logger
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # 获取项目根目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            self.output_dir = project_root / "to" / "local_data" / "sector_features_leading"
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置缓存目录
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # 获取项目根目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            self.cache_dir = project_root / "to" / "local_data" / "features_cache" / "leading_stocks"
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 涨停判断的标准
        self.limit_up_pct = 0.1  # 默认涨停幅度10%
        self.limit_down_pct = -0.1  # 默认跌停幅度-10%
        
        # 特殊股票类型的涨跌停幅度
        self.special_limit_configs = {
            'ST': {'up': 0.05, 'down': -0.05},      # ST股票涨跌停5%
            'STAR': {'up': 0.2, 'down': -0.2},      # 科创板涨跌停20%
            'GEM': {'up': 0.2, 'down': -0.2},       # 创业板涨跌停20%
            'BSE': {'up': 0.3, 'down': -0.3}        # 北交所涨跌停30%
        }
        
        # 行业温度计算权重
        self.weights = {
            'leading_stock_count': 0.3,
            'limit_up_count': 0.3,
            'multi_limit_up_count': 0.2,
            'turnover_distribution': 0.1,
            'high_turnover_ratio': 0.1
        }
        
        # 创建缓存目录结构
        (self.cache_dir / "daily").mkdir(exist_ok=True)
        (self.cache_dir / "stocks").mkdir(exist_ok=True)
        
        self.logger.info(f"板块龙头股分析器初始化完成，输出目录: {self.output_dir}，缓存目录: {self.cache_dir}")

    def _is_limit_up(self, pct_chg, stock_code, close=None, pre_close=None):
        """
        判断股票是否涨停，优化后的判断逻辑
        
        Args:
            pct_chg: 涨跌幅
            stock_code: 股票代码
            close: 收盘价(可选)
            pre_close: 前收盘价(可选)
            
        Returns:
            bool: 是否涨停
        """
        # 确定涨停幅度比率
        limit_ratio = self.limit_up_pct  # 默认10%
        
        # 根据股票类型调整涨停幅度
        if 'ST' in stock_code.upper():
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


    def _identify_leading_stocks(self, processed_stocks):
        """
        基于处理后的股票数据识别龙头股，用于单日数据处理
        
        Args:
            processed_stocks: 已处理的股票列表，每项为一只股票的数据字典
            
        Returns:
            dict: 龙头股识别结果，包含leading_stocks和high_elasticity_stocks
        """
        try:
            # 确保输入数据有效
            if not processed_stocks or not isinstance(processed_stocks, list):
                self.logger.warning(f"识别龙头股的输入数据无效: {type(processed_stocks)}")
                return {
                    'leading_stocks': [],
                    'high_elasticity_stocks': []
                }
                
            self.logger.info(f"开始识别龙头股，处理 {len(processed_stocks)} 只股票数据")
            
            # 定义得分维度
            score_weights = {
                'limit_up': 0.15,         # 涨停加分
                'consecutive_limit': 0.35, # 连续涨停天数加分(新增)
                'pct_chg': 0.15,          # 涨幅得分
                'turnover': 0.10,         # 换手率得分
                'amount': 0.15,           # 成交额得分
                'market_cap': 0.10,       # 市值得分
            }
            
            # 计算每只股票得分
            stock_scores = {}
            
            # 提取所有股票的属性值
            pct_chg_values = [s.get('pct_chg', 0) for s in processed_stocks if isinstance(s, dict)]
            turnover_values = [s.get('turnover', 0) for s in processed_stocks if isinstance(s, dict)]
            amount_values = [s.get('amount', 0) for s in processed_stocks if isinstance(s, dict)]
            total_mv_values = [s.get('total_mv', 0) for s in processed_stocks if isinstance(s, dict) and s.get('total_mv') is not None]
            circ_mv_values = [s.get('circ_mv', 0) for s in processed_stocks if isinstance(s, dict) and s.get('circ_mv') is not None]
            consecutive_limit_values = [s.get('consecutive_limit_days', 0) for s in processed_stocks if isinstance(s, dict)]
            
            # 统计有效值的数量
            self.logger.debug(f"有效数据统计: 涨幅数据 {len(pct_chg_values)}个, 换手率数据 {len(turnover_values)}个, 成交额数据 {len(amount_values)}个, 连续涨停数据 {len(consecutive_limit_values)}个")
            
            # 计算正则化得分
            for stock in processed_stocks:
                stock_code = stock['stock_code']
                scores = {}
                
                # 涨停得分
                is_limit_up = stock.get('is_limit_up', False)
                scores['limit_up'] = 1.0 if is_limit_up else 0.0
                
                # 连续涨停天数得分（新增）
                consecutive_limit_days = stock.get('consecutive_limit_days', 0)
                # 设置连续涨停天数得分，按照天数进行加权
                # 5板及以上得满分，1板得0.2分，以此类推
                if consecutive_limit_days >= 5:
                    scores['consecutive_limit'] = 1.0
                elif consecutive_limit_days > 0:
                    scores['consecutive_limit'] = consecutive_limit_days * 0.2
                else:
                    scores['consecutive_limit'] = 0.0
                
                # 涨幅得分（正则化）
                pct_chg = stock.get('pct_chg', 0)
                if pct_chg_values:
                    max_pct = max(pct_chg_values)
                    min_pct = min(pct_chg_values)
                    range_pct = max_pct - min_pct if max_pct > min_pct else 1
                    scores['pct_chg'] = (pct_chg - min_pct) / range_pct if range_pct else 0
                else:
                    scores['pct_chg'] = 0
                
                # 换手率得分（正则化）
                turnover = stock.get('turnover', 0)
                if turnover_values:
                    max_turnover = max(turnover_values)
                    scores['turnover'] = turnover / max_turnover if max_turnover else 0
                else:
                    scores['turnover'] = 0
                
                # 成交额得分（正则化）
                amount = stock.get('amount', 0)
                if amount_values:
                    max_amount = max(amount_values)
                    scores['amount'] = amount / max_amount if max_amount else 0
                else:
                    scores['amount'] = 0
                
                # 市值得分（考虑总市值和流通市值）
                # 优先使用流通市值，如果没有则使用总市值
                market_cap_score = 0
                circ_mv = stock.get('circ_mv', None)
                total_mv = stock.get('total_mv', None)
                
                if circ_mv is not None and circ_mv > 0 and circ_mv_values:
                    # 流通市值越大，得分越高，但过小时得分较低
                    # 使用对数比例，避免大市值股票得分过高
                    max_circ_mv = max(circ_mv_values)
                    min_circ_mv = min(v for v in circ_mv_values if v > 0)
                    
                    # 对数归一化，确保小市值也有基础分
                    log_mv = np.log10(circ_mv) if circ_mv > 0 else 0
                    log_max = np.log10(max_circ_mv) if max_circ_mv > 0 else 1
                    log_min = np.log10(min_circ_mv) if min_circ_mv > 0 else 0
                    
                    # 考虑市值范围的得分调整
                    if circ_mv > 100:  # 大市值 >100亿
                        market_cap_score = 0.8 + 0.2 * (log_mv - log_min) / (log_max - log_min) if log_max > log_min else 0.8
                    elif circ_mv > 20:  # 中等市值 20-100亿
                        market_cap_score = 0.6 + 0.2 * (log_mv - log_min) / (log_max - log_min) if log_max > log_min else 0.6
                    else:  # 小市值 <20亿
                        market_cap_score = 0.2 + 0.4 * (log_mv - log_min) / (log_max - log_min) if log_max > log_min else 0.2
                elif total_mv is not None and total_mv > 0 and total_mv_values:
                    # 使用总市值计算（备选方案）
                    # 逻辑同流通市值
                    max_total_mv = max(total_mv_values)
                    min_total_mv = min(v for v in total_mv_values if v > 0)
                    
                    log_mv = np.log10(total_mv) if total_mv > 0 else 0
                    log_max = np.log10(max_total_mv) if max_total_mv > 0 else 1
                    log_min = np.log10(min_total_mv) if min_total_mv > 0 else 0
                    
                    if total_mv > 200:  # 大市值 >200亿
                        market_cap_score = 0.8 + 0.2 * (log_mv - log_min) / (log_max - log_min) if log_max > log_min else 0.8
                    elif total_mv > 50:  # 中等市值 50-200亿
                        market_cap_score = 0.6 + 0.2 * (log_mv - log_min) / (log_max - log_min) if log_max > log_min else 0.6
                    else:  # 小市值 <50亿
                        market_cap_score = 0.2 + 0.4 * (log_mv - log_min) / (log_max - log_min) if log_max > log_min else 0.2
                
                scores['market_cap'] = market_cap_score
                
                # 计算加权总分
                total_score = 0
                for metric, weight in score_weights.items():
                    total_score += scores.get(metric, 0) * weight
                
                stock_scores[stock_code] = total_score
            
            # 按得分排序
            sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 找出高弹性标的（北交所涨停标的）
            high_elasticity = []
            for stock in processed_stocks:
                # 筛选北交所涨停股票
                if stock.get('is_bse_stock', False) and stock.get('is_limit_up', False):
                    high_elasticity.append((stock['stock_code'], stock.get('pct_chg', 0)))
            
            # 按涨幅排序
            high_elasticity = sorted(high_elasticity, key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"龙头股识别完成，找到 {len(sorted_stocks)} 只有效股票，其中北交所涨停标的 {len(high_elasticity)} 只")
            
            result = {
                'leading_stocks': sorted_stocks[:10],  # 取得分最高的10只
                'high_elasticity_stocks': high_elasticity[:5]  # 取最高的5只
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"识别龙头股时出错: {e}")
            self.logger.exception(e)
            # 返回空结果
            return {
                'leading_stocks': [],
                'high_elasticity_stocks': []
            }

    def _calculate_sector_heat(self, sector_features):
        """
        计算板块热度指数
        
        Args:
            sector_features: 板块特征数据
            
        Returns:
            float: 热度指数(0-100)
        """
        # 初始化热度分项
        heat_scores = {
            'leading_stock_count': 0,
            'limit_up_count': 0,
            'multi_limit_up_count': 0,
            'turnover_distribution': 0,
            'high_turnover_ratio': 0
        }
        
        # 计算龙头股得分(0-100)，龙头股越多分越高
        leading_count = len(sector_features['leading_stocks'])
        heat_scores['leading_stock_count'] = min(100, leading_count * 20)
        
        # 计算涨停数量得分(0-100)
        limit_up_count = sector_features['limit_up_counts']['total']
        stock_count = max(1, sector_features['stock_count'])
        limit_up_ratio = limit_up_count / stock_count
        heat_scores['limit_up_count'] = min(100, limit_up_ratio * 500)  # 20%涨停即满分
        
        # 计算多板涨停得分(0-100)，给予更高权重
        multi_limit_count = (
            sector_features['limit_up_counts']['second_limit'] +
            sector_features['limit_up_counts']['third_limit'] * 2 +
            sector_features['limit_up_counts']['fourth_limit'] * 3 +
            sector_features['limit_up_counts']['fifth_plus_limit'] * 5
        )
        heat_scores['multi_limit_up_count'] = min(100, multi_limit_count * 20)
        
        # 计算换手率分布得分(0-100)
        turnover_score = (
            sector_features['turnover_distribution']['high'] * 3 +
            sector_features['turnover_distribution']['medium'] * 1
        ) / max(1, stock_count) * 100
        heat_scores['turnover_distribution'] = min(100, turnover_score)
        
        # 计算高换手率股票比例得分(0-100)
        high_turnover_ratio = sector_features['turnover_distribution']['high'] / max(1, stock_count)
        heat_scores['high_turnover_ratio'] = min(100, high_turnover_ratio * 500)  # 20%高换手即满分
        
        # 如果有北交所高弹性标的，增加额外得分
        if 'high_elasticity_stocks' in sector_features and sector_features['high_elasticity_stocks']:
            # 添加高弹性标的得分，最高20分
            elasticity_score = min(20, len(sector_features['high_elasticity_stocks']) * 7)
            heat_scores['high_elasticity'] = elasticity_score
            
            # 调整权重以包含高弹性标的
            weights = self.weights.copy()
            weights['high_elasticity'] = 0.05
            # 其他权重比例相应减少
            for key in self.weights:
                weights[key] = self.weights[key] * 0.95
        else:
            heat_scores['high_elasticity'] = 0
            weights = self.weights
        
        # 涨停股次日溢价情况，奖励有溢价的板块
        if 'next_day_premium' in sector_features and sector_features['next_day_premium']['avg_open_premium'] > 0:
            avg_premium = sector_features['next_day_premium']['avg_open_premium']
            # 溢价3%以上给满分，线性计算
            premium_score = min(100, avg_premium * 33.3)
            heat_scores['premium_score'] = premium_score
            
            # 添加溢价得分权重
            if 'high_elasticity' in weights:
                weights['premium_score'] = 0.05
                # 其他权重再减少
                for key in self.weights:
                    weights[key] = weights[key] * 0.95
            else:
                weights['premium_score'] = 0.05
                # 其他权重再减少
                for key in self.weights:
                    weights[key] = self.weights[key] * 0.95
        else:
            heat_scores['premium_score'] = 0
        
        # 加权计算最终热度
        final_heat = 0
        for metric, score in heat_scores.items():
            if metric in weights:
                final_heat += score * weights[metric]
        
        return round(final_heat, 2)

    def process_sector_data(self, sector_code, date, stocks_data, prev_day_stocks_data=None):
        """
        分析板块内股票，计算龙头特征
        
        Args:
            sector_code: 板块代码
            date: 分析日期
            stocks_data: 板块内股票数据 {股票代码: 数据字典} 格式
            prev_day_stocks_data: 前一交易日的板块内股票数据，用于计算溢价
            
        Returns:
            dict: 板块龙头股特征
        """
        # 将日期转为字符串格式
        if isinstance(date, datetime.datetime):
            date_str = date.strftime('%Y%m%d')
        else:
            date_str = str(date)
        
        # 初始化特征
        sector_features = {
            'sector_code': sector_code,
            'date': date_str,
            'analyzed_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stock_count': 0,
            'limit_up_counts': {
                'total': 0,       # 涨停数量
                'first_limit': 0, # 首板数量
                'second_limit': 0, # 二板数量
                'third_limit': 0,  # 三板数量
                'fourth_limit': 0, # 四板数量
                'fifth_plus_limit': 0, # 五板及以上数量
                'broken_limit': 0, # 涨停被打开数量
            },
            'turnover_distribution': {
                'high': 0,      # 高换手(>10%)数量
                'medium': 0,    # 中等换手率(5-10%)数量
                'low': 0,       # 低换手率(<5%)数量
            },
            'market_cap_distribution': {
                'large': 0,     # 大市值股(>100亿)数量
                'medium': 0,    # 中等市值股(20-100亿)数量
                'small': 0,     # 小市值股(<20亿)数量
            },
            'leading_stocks': [],  # 龙头股列表
            'high_elasticity_stocks': [], # 高弹性标的列表(北交所涨停标的)
            'limit_up_stock_codes': [], # 涨停股票代码
            'heat_index': 0,      # 板块热度指数
            'next_day_premium': {  # 涨停股次日溢价情况
                'avg_open_premium': 0, # 平均开盘溢价
                'max_open_premium': 0, # 最大开盘溢价
                'premium_stocks': []   # 溢价详情
            }
        }
        
        # 确保stocks_data是字典格式的
        if isinstance(stocks_data, pd.DataFrame):
            # 如果是DataFrame，转换为字典格式 {股票代码: 数据行}
            stocks_dict = {}
            for _, row in stocks_data.iterrows():
                stock_code = row.get('code', row.get('stock_code', ''))
                if stock_code:
                    stocks_dict[stock_code] = row.to_dict()
            stocks_data = stocks_dict
        
        # 更新股票总数
        sector_features['stock_count'] = len(stocks_data)
        
        # 按换手率分组
        turnover_high = []  # 高换手率(>10%)
        turnover_medium = [] # 中等换手率(5-10%)
        turnover_low = []   # 低换手率(<5%)
        
        # 按市值分组
        market_cap_large = []  # 大市值(>100亿)
        market_cap_medium = [] # 中等市值(20-100亿)
        market_cap_small = []  # 小市值(<20亿)
        
        # 加载交易日历，用于新股过滤
        trading_calendar = self._load_trading_calendar()
        
        # 处理每只股票
        processed_stocks = []
        
        for stock_code, stock_data in stocks_data.items():
            if stock_code == '_metadata':
                continue  # 跳过元数据
                
            # 新增: 检查是否是新股上市不足5天，如果是则排除
            if trading_calendar is not None:
                try:
                    # 尝试获取股票的首个交易日
                    first_date = None
                    
                    # 从不同来源尝试获取首个交易日
                    if isinstance(stock_data, dict):
                        # 直接从数据中获取
                        if 'first_trade_date' in stock_data:
                            first_date = stock_data['first_trade_date']
                        elif 'list_date' in stock_data:
                            first_date = stock_data['list_date']
                    
                    # 如果没找到，尝试从元数据或其他来源获取
                    if not first_date:
                        try:
                            from to.data_loaders import get_stock_basic_info
                            stock_info = get_stock_basic_info(stock_code)
                            if stock_info and 'list_date' in stock_info:
                                first_date = stock_info['list_date']
                        except Exception as e:
                            self.logger.debug(f"无法获取股票{stock_code}的首个交易日: {e}")
                    
                    # 如果找到首个交易日，判断是否上市不足5天
                    if first_date:
                        # 确保first_date格式正确
                        if isinstance(first_date, datetime.datetime):
                            first_date = first_date.strftime('%Y%m%d')
                            
                        # 找到首个交易日在交易日历中的位置
                        if first_date in trading_calendar:
                            idx = trading_calendar.index(first_date)
                            # 计算上市第6天的交易日索引
                            sixth_day_idx = idx + 5
                            
                            # 如果当前日期早于上市第6天，则跳过该股票
                            if sixth_day_idx < len(trading_calendar):
                                sixth_day = trading_calendar[sixth_day_idx]
                                if date_str < sixth_day:
                                    self.logger.debug(f"股票 {stock_code} 上市不足5天，排除在龙头股分析之外")
                                    continue
                except Exception as e:
                    self.logger.debug(f"检查股票 {stock_code} 是否为新股时出错: {e}")
                    
            # 计算换手率分布
            turnover = stock_data.get('turnover_rate', stock_data.get('turnover', 0))
            if turnover > 10:
                turnover_high.append(stock_code)
                sector_features['turnover_distribution']['high'] += 1
            elif turnover > 5:
                turnover_medium.append(stock_code)
                sector_features['turnover_distribution']['medium'] += 1
            else:
                turnover_low.append(stock_code)
                sector_features['turnover_distribution']['low'] += 1
            
            # 计算市值分布
            total_mv = stock_data.get('total_mv', 0)
            circ_mv = stock_data.get('circ_mv', 0)
            
            # 市值单位统一为亿元
            if total_mv is not None:
                if total_mv > 100:  # 大市值 >100亿
                    market_cap_large.append(stock_code)
                    sector_features['market_cap_distribution']['large'] += 1
                elif total_mv > 20:  # 中等市值 20-100亿
                    market_cap_medium.append(stock_code)
                    sector_features['market_cap_distribution']['medium'] += 1
                else:  # 小市值 <20亿
                    market_cap_small.append(stock_code)
                    sector_features['market_cap_distribution']['small'] += 1
            
            # 判断是否涨停
            pct_chg = stock_data.get('pct_chg', stock_data.get('change_percent', 0))
            close = stock_data.get('close', None)
            pre_close = stock_data.get('pre_close', None)
            
            # 计算振幅
            amplitude = self._calculate_amplitude(stock_data)
            
            # 判断是否破板
            is_broken_limit = self._is_broken_limit(stock_data, stock_code)
            
            # 计算连续涨停天数
            consecutive_limit_days = self._calculate_consecutive_limit_days(stock_code, date, {date_str: stock_data})
            
            # 创建处理后的股票信息
            processed_stock = {
                'stock_code': stock_code,
                'name': stock_data.get('name', ''),
                'pct_chg': pct_chg,
                'turnover': turnover,
                'amount': stock_data.get('amount', 0),  # 成交额
                'volume': stock_data.get('volume', 0),  # 成交量
                'is_limit_up': self._is_limit_up(pct_chg, stock_code, close, pre_close),
                'consecutive_limit_days': consecutive_limit_days,
                'is_broken_limit': is_broken_limit,
                'amplitude': amplitude,
                'open': stock_data.get('open', 0),
                'high': stock_data.get('high', 0),
                'low': stock_data.get('low', 0),
                'total_mv': total_mv,  # 总市值（亿元）
                'circ_mv': circ_mv,    # 流通市值（亿元）
                'is_bse_stock': self._is_bse_stock(stock_code)  # 是否为北交所股票
            }
            
            # 如果是涨停，更新涨停计数
            if processed_stock['is_limit_up']:
                sector_features['limit_up_counts']['total'] += 1
                sector_features['limit_up_stock_codes'].append(stock_code)
                
                # 根据连板天数分类
                consecutive_days = processed_stock['consecutive_limit_days']
                if consecutive_days == 1:
                    sector_features['limit_up_counts']['first_limit'] += 1
                elif consecutive_days == 2:
                    sector_features['limit_up_counts']['second_limit'] += 1
                elif consecutive_days == 3:
                    sector_features['limit_up_counts']['third_limit'] += 1
                elif consecutive_days == 4:
                    sector_features['limit_up_counts']['fourth_limit'] += 1
                elif consecutive_days >= 5:
                    sector_features['limit_up_counts']['fifth_plus_limit'] += 1
            
            # 涨停炸板
            if processed_stock['is_broken_limit']:
                sector_features['limit_up_counts']['broken_limit'] += 1
            
            processed_stocks.append(processed_stock)
        
        # 记录处理后的股票数量和类型
        self.logger.info(f"板块 {sector_code} 处理了 {len(processed_stocks)} 只股票，准备识别龙头股")
        
        # 识别龙头股
        leading_results = self._identify_leading_stocks(processed_stocks)
        sector_features['leading_stocks'] = leading_results['leading_stocks']
        sector_features['high_elasticity_stocks'] = leading_results['high_elasticity_stocks']
        
        # 计算板块热度指数
        heat_index = self._calculate_sector_heat(sector_features)
        sector_features['heat_index'] = heat_index
        
        # 计算统计指标
        if sector_features['stock_count'] > 0:
            # 涨停率
            sector_features['limit_up_ratio'] = sector_features['limit_up_counts']['total'] / sector_features['stock_count']
            # 高换手率比例
            sector_features['high_turnover_ratio'] = sector_features['turnover_distribution']['high'] / sector_features['stock_count']
            # 龙头股比例
            sector_features['leading_stock_ratio'] = len(sector_features['leading_stocks']) / sector_features['stock_count']
        
        # 计算前一日涨停股今日溢价情况
        if prev_day_stocks_data is not None:
            self._calculate_next_day_premium(sector_features, prev_day_stocks_data, stocks_data)
        
        return sector_features

    def batch_process_sectors(self, sector_list, date_range):
        """
        批量处理多个板块的龙头股识别
        
        Args:
            sector_list: 板块代码列表
            date_range: 日期范围列表
        """
        self.logger.info(f"开始批量处理 {len(sector_list)} 个板块的龙头股，日期范围: {date_range[0]}至{date_range[-1]}")
        
        try:
            from to.stock_sectors_mapping import get_sector_stocks
        except ImportError as e:
            logger.error(f"批量处理时出错: {e}")
            logger.error(traceback.format_exc())
            return None
        # 导入所需函数
      
        # 使用进度条跟踪处理
        processed = 0
        total_tasks = len(sector_list)
        # 使用多线程并行处理多个板块
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
            futures = []
            
            for sector_code in sector_list:
                future = executor.submit(
                    self._process_sector_with_optimized_data_loading,
                    sector_code=sector_code,
                    date_range=date_range,
                    get_stock_list_func=get_sector_stocks,
                    get_daily_data_func=get_daily_data
                )
                futures.append((future, sector_code))
            
            # 收集结果
            for future, sector_code in futures:
                try:
                    result = future.result()
                    processed += 1
                    self.logger.info(f"完成处理板块 {sector_code} ({processed}/{total_tasks})")
                except Exception as e:
                    self.logger.error(f"处理板块 {sector_code} 时出错: {e}")
                    self.logger.error(traceback.format_exc())
        
        self.logger.info(f"批量处理完成，成功处理 {processed}/{total_tasks} 个板块")

    def _process_sector_with_optimized_data_loading(self, sector_code, date_range, get_stock_list_func, get_daily_data_func):
        """
        使用优化的数据加载方式处理单个板块
        
        Args:
            sector_code: 板块代码
            date_range: 日期范围
            get_stock_list_func: 获取板块股票列表的函数
            get_daily_data_func: 获取股票日线数据的函数
            
        Returns:
            处理结果
        """
        try:
            # 获取板块内股票列表
            stocks = []
            try:
                stocks = get_stock_list_func(sector_code)
                if not stocks:
                    self.logger.warning(f"板块 {sector_code} 没有股票，跳过处理")
                    return None
            except Exception as e:
                self.logger.error(f"获取板块 {sector_code} 股票列表时出错: {e}")
                self.logger.error(traceback.format_exc())
                return None
            # print(f"板块 {sector_code} 股票列表: {stocks}")
            # 计算需要加载的日期范围
            # 包括date_range前一天作为对比基准
            start_date = pd.to_datetime(date_range[0]) - pd.Timedelta(days=1)
            end_date = pd.to_datetime(date_range[-1])
            
            extended_date_range = pd.date_range(start=start_date, end=end_date)
            extended_date_range = [d.strftime('%Y%m%d') for d in extended_date_range]
            
            # 批量加载数据
            try:
                self.logger.info(f"开始批量加载板块 {sector_code} 的股票数据，日期范围：{date_range[0]}至{date_range[-1]}，股票数量：{len(stocks)}")
                stock_data, valid_trading_days = self._batch_load_stocks_data(stocks, date_range, get_daily_data_func)
                if not stock_data:
                    self.logger.warning(f"板块 {sector_code} 没有加载到股票数据，跳过处理")
                    return []
                
                # 打印有效数据统计
                valid_dates = set()
                stocks_with_data = set()
                
                for stock_code, date_data in stock_data.items():
                    self.logger.info(f"板块 {sector_code} 股票代码: {stock_code}, 日期数据: {date_data}")
                    if stock_code != '_metadata':  # 跳过元数据
                        stocks_with_data.add(stock_code)
                        valid_dates.update(date_data.keys())
                
                self.logger.info(f"板块 {sector_code} 成功加载数据：有效股票 {len(stocks_with_data)}/{len(stocks)}，有效日期 {len(valid_dates)}/{len(extended_date_range)}")
                
                # 检查是否所有关键日期都有数据
                # 只检查交易日是否有缺失
                missing_dates = [d for d in valid_trading_days if d not in valid_dates and d in date_range]
                if missing_dates:
                    self.logger.warning(f"板块 {sector_code} 以下交易日缺少数据: {missing_dates}")
            except Exception as e:
                self.logger.error(f"批量加载板块 {sector_code} 股票数据时出错: {e}")
                self.logger.error(traceback.format_exc())
                return None
            
            # 处理每个日期
            for date_idx, date in enumerate(date_range):
                try:
                    # 跳过非交易日
                    if valid_trading_days and date not in valid_trading_days:
                        self.logger.info(f"日期 {date} 不是交易日，跳过处理")
                        continue
                        
                    # 收集当日所有股票的数据
                    stocks_for_date = {}
                    valid_stock_count = 0
                    
                    # 检查当前日期是否为交易日
                    trading_calendar = self._load_trading_calendar()
                    if trading_calendar is not None and date not in trading_calendar:
                        self.logger.info(f"日期 {date} 不是交易日，跳过处理")
                        continue
                    
                    # 从各个股票中提取当日数据
                    for stock_code, date_data in stock_data.items():
                        if stock_code == '_metadata':
                            continue  # 跳过元数据
                        
                        # 获取该股票当天的数据
                        stock_daily_data = date_data.get(date)
                        if stock_daily_data:
                            stocks_for_date[stock_code] = stock_daily_data
                            valid_stock_count += 1
                    
                    # 检查当天是否有足够的股票数据
                    if valid_stock_count < 0.3 * len(stocks):  # 如果少于30%的股票有数据，视为无效
                        self.logger.warning(f"板块 {sector_code} 在日期 {date} 只有 {valid_stock_count}/{len(stocks)} 只股票有数据，跳过")
                        continue
                    
                    # 收集前一天所有股票的数据
                    # 查找前一个交易日
                    prev_date_str = None
                    if valid_trading_days:
                        # 在交易日列表中找到当前日期的前一个交易日
                        trading_days = sorted([d for d in valid_trading_days if d <= date])
                        if len(trading_days) >= 2:
                            prev_date_str = trading_days[-2]  # 倒数第二个是前一个交易日
                    
                    # 如果没有找到前一个交易日，则回退到简单的日历日期减一天
                    if prev_date_str is None:
                        prev_date = pd.to_datetime(date) - pd.Timedelta(days=1)
                        prev_date_str = prev_date.strftime('%Y%m%d')
                    
                    prev_stocks_data = {}
                    
                    # 从各个股票中提取前一天数据
                    for stock_code, date_data in stock_data.items():
                        if stock_code == '_metadata':
                            continue  # 跳过元数据
                        
                        # 获取该股票前一天的数据
                        stock_prev_data = date_data.get(prev_date_str)
                        if stock_prev_data:
                            prev_stocks_data[stock_code] = stock_prev_data
                    
                    # 处理当日数据
                    result = self.process_sector_data(
                        sector_code=sector_code,
                        date=date,
                        stocks_data=stocks_for_date,
                        prev_day_stocks_data=prev_stocks_data
                    )
                    
                    if result:
                        # 保存结果到缓存
                        self._save_results_to_cache(sector_code, date, result)
                except Exception as e:
                    self.logger.error(f"处理板块 {sector_code} 日期 {date} 时出错: {e}")
                    self.logger.error(traceback.format_exc())
            
            # 合并整个日期范围的结果
            merged_results = self._merge_sector_results(sector_code, date_range)
            return merged_results
            
        except Exception as e:
            self.logger.error(f"处理板块 {sector_code} 时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def _batch_load_stocks_data(self, stocks, date_range, get_daily_data_func):
        """批量加载股票数据
        
        Args:
            stocks: 股票代码列表
            date_range: 日期范围
            get_daily_data_func: 获取数据的函数
            
        Returns:
            tuple: (股票数据字典 {股票代码: {日期: 数据行}}, 有效交易日列表)
        """
        stock_data = {}
        valid_stocks_count = 0
        first_data_dates = {}
        
        self.logger.info(f"批量加载 {len(stocks)} 只股票数据，日期范围: {date_range[0]} - {date_range[-1]}")
        
        # 确保stocks是股票代码列表而不是字符串
        if isinstance(stocks, str):
            stocks = [stocks]  # 如果是单个股票代码字符串，转换为列表
        
        # 过滤有效交易日期，确保只包含非节假日
        trading_calendar = self._load_trading_calendar()
        valid_trading_days = None
        
        # 确保date_range中只有交易日
        if trading_calendar is not None:
            valid_trading_days = [day for day in date_range if day in trading_calendar]
            if len(valid_trading_days) < len(date_range):
                self.logger.info(f"从 {len(date_range)} 个日期中筛选出 {len(valid_trading_days)} 个交易日")
        else:
            valid_trading_days = date_range
            self.logger.warning("无法加载交易日历，使用原始日期范围")
        
        # 一次获取一个批次的股票数据
        try:
            # 调用外部传入的get_daily_data_func函数，传入股票列表和日期范围
            batch_data_df = get_daily_data_func(stocks, date_range[0], date_range[-1])
            
            if batch_data_df is not None and not batch_data_df.empty:
                # 确保date_str列存在
                if 'date_str' not in batch_data_df.columns and 'date' in batch_data_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(batch_data_df['date']):
                        batch_data_df['date_str'] = batch_data_df['date'].dt.strftime('%Y%m%d')
                
                # 按股票代码分组，并转换为所需的字典格式
                for stock_code, group in batch_data_df.groupby('stock_code'):
                    if stock_code in stocks:  # 确认是我们请求的股票
                        # 创建{日期: 数据行}的字典
                        date_data = {}
                        for _, row in group.iterrows():
                            date_key = row['date_str'] if 'date_str' in row else str(row['date'])
                            date_data[date_key] = row.to_dict()
                        
                        # 添加到结果字典
                        stock_data[stock_code] = date_data
                        valid_stocks_count += 1
                        
                        # 记录首个有数据的日期
                        if date_data:
                            sorted_dates = sorted(date_data.keys())
                            if sorted_dates:
                                first_data_dates[stock_code] = sorted_dates[0]
        except Exception as e:
            self.logger.error(f"批量加载股票数据时出错: {e}")
            self.logger.error(traceback.format_exc())
        
        # 添加元数据
        stock_data['_metadata'] = {
            'total_stocks': len(stocks),
            'valid_stocks': valid_stocks_count,
            'date_range': date_range,
            'valid_trading_days': valid_trading_days,
            'first_data_dates': first_data_dates
        }
        
        return stock_data, valid_trading_days


    def _calculate_next_day_premium(self, sector_features, prev_day_stocks_data, current_stocks_data):
        """
        计算涨停股票次日溢价情况
        
        Args:
            sector_features: 板块特征字典，将被更新
            prev_day_stocks_data: 前一日股票数据 {股票代码: 数据字典} 格式
            current_stocks_data: 当日股票数据 {股票代码: 数据字典} 格式
        """
        # 确保数据是字典格式
        if isinstance(prev_day_stocks_data, pd.DataFrame):
            # 如果是DataFrame，转换为字典格式 {股票代码: 数据行}
            prev_stocks_dict = {}
            for _, row in prev_day_stocks_data.iterrows():
                stock_code = row.get('code', row.get('stock_code', ''))
                if stock_code:
                    prev_stocks_dict[stock_code] = row.to_dict()
            prev_day_stocks_data = prev_stocks_dict
        
        if isinstance(current_stocks_data, pd.DataFrame):
            # 如果是DataFrame，转换为字典格式 {股票代码: 数据行}
            current_stocks_dict = {}
            for _, row in current_stocks_data.iterrows():
                stock_code = row.get('code', row.get('stock_code', ''))
                if stock_code:
                    current_stocks_dict[stock_code] = row.to_dict()
            current_stocks_data = current_stocks_dict
        
        # 找出前一日涨停的股票
        prev_limit_up_stocks = {}
        for stock_code, prev_stock in prev_day_stocks_data.items():
            if stock_code == '_metadata':
                continue  # 跳过元数据
                
            pct_chg = prev_stock.get('pct_chg', prev_stock.get('change_percent', 0))
            close = prev_stock.get('close', None)
            pre_close = prev_stock.get('pre_close', None)
            
            if self._is_limit_up(pct_chg, stock_code, close, pre_close):
                prev_limit_up_stocks[stock_code] = prev_stock
        
        # 计算溢价情况
        premium_stocks = []
        total_premium = 0
        max_premium = 0
        count = 0
        
        for stock_code, prev_stock in prev_limit_up_stocks.items():
            if stock_code in current_stocks_data:
                current_stock = current_stocks_data[stock_code]
                
                # 获取前一日收盘价和当日开盘价
                prev_close = prev_stock.get('close', 0)
                current_open = current_stock.get('open', 0)
                
                if prev_close > 0:
                    # 计算开盘溢价率
                    open_premium = (current_open - prev_close) / prev_close * 100
                    
                    # 记录溢价情况
                    premium_info = {
                        'stock_code': stock_code,
                        'name': current_stock.get('name', ''),
                        'prev_close': prev_close,
                        'current_open': current_open,
                        'open_premium': open_premium,
                        'current_close': current_stock.get('close', 0),
                        'day_change': current_stock.get('pct_chg', current_stock.get('change_percent', 0))
                    }
                    premium_stocks.append(premium_info)
                    
                    # 更新统计数据
                    total_premium += open_premium
                    max_premium = max(max_premium, open_premium)
                    count += 1
        
        # 更新板块特征
        if count > 0:
            avg_premium = total_premium / count
            sector_features['next_day_premium']['avg_open_premium'] = round(avg_premium, 2)
            sector_features['next_day_premium']['max_open_premium'] = round(max_premium, 2)
            sector_features['next_day_premium']['premium_stocks'] = premium_stocks

    def _save_results_to_cache(self, sector_code, date, sector_features):
        """
        保存处理结果到缓存
        
        Args:
            sector_code: 板块代码
            date: 日期
            sector_features: 板块特征数据
        """
        try:
            # 确保缓存目录存在
            daily_cache_dir = os.path.join(self.cache_dir, "daily")
            os.makedirs(daily_cache_dir, exist_ok=True)
            
            # 生成缓存文件路径
            if isinstance(date, datetime.datetime):
                date_str = date.strftime('%Y%m%d')
            else:
                date_str = str(date)
                
            cache_file = os.path.join(daily_cache_dir, f"{sector_code}_{date_str}.parquet")
            
            # 将结果转换为DataFrame并保存
            if isinstance(sector_features, dict):
                # 扁平化嵌套结构，便于存储
                flat_features = self._flatten_dict(sector_features)
                df = pd.DataFrame([flat_features])
                df.to_parquet(cache_file)
                self.logger.debug(f"板块 {sector_code} 日期 {date_str} 的特征已缓存")
            else:
                self.logger.warning(f"板块 {sector_code} 日期 {date_str} 的特征不是字典格式，无法缓存")
        
        except Exception as e:
            self.logger.error(f"保存板块 {sector_code} 日期 {date} 的缓存时出错: {e}")
            self.logger.error(traceback.format_exc())
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """
        将嵌套字典扁平化为单层字典
        
        Args:
            d: 要扁平化的字典
            parent_key: 父键前缀
            sep: 键分隔符
            
        Returns:
            dict: 扁平化后的字典
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                # 处理列表类型，将其转换为JSON字符串
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
                
        return dict(items)
    
    def _merge_sector_results(self, sector_code, date_range):
        """
        合并板块所有日期的结果为一个DataFrame
        
        Args:
            sector_code: 板块代码
            date_range: 日期范围
            
        Returns:
            DataFrame: 合并后的结果
        """
        try:
            # 检查缓存目录
            daily_cache_dir = os.path.join(self.cache_dir, "daily")
            if not os.path.exists(daily_cache_dir):
                self.logger.warning(f"缓存目录 {daily_cache_dir} 不存在")
                return None
                
            # 收集该板块所有日期的缓存文件
            all_files = []
            for date in date_range:
                date_str = date if isinstance(date, str) else date.strftime('%Y%m%d')
                cache_file = os.path.join(daily_cache_dir, f"{sector_code}_{date_str}.parquet")
                if os.path.exists(cache_file):
                    all_files.append(cache_file)
            
            if not all_files:
                self.logger.warning(f"板块 {sector_code} 没有找到任何缓存文件")
                return None
                
            # 合并所有文件
            dfs = []
            for file in all_files:
                try:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                except Exception as e:
                    self.logger.error(f"读取文件 {file} 时出错: {e}")
                    
            if not dfs:
                self.logger.warning(f"板块 {sector_code} 没有有效的缓存文件")
                return None
                
            # 合并所有DataFrame
            merged_df = pd.concat(dfs, ignore_index=True)
            
            # 确保日期列是datetime类型
            if 'date' in merged_df.columns:
                merged_df['date'] = pd.to_datetime(merged_df['date'])
                
            # 按日期排序
            merged_df = merged_df.sort_values('date')
            
            # 保存合并结果
            output_dir = os.path.join(self.output_dir, sector_code)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为parquet文件
            output_path = os.path.join(output_dir, f"{sector_code}_leading_stocks.parquet")
            merged_df.to_parquet(output_path)
            
            # 同时保存最新数据为CSV，便于查看
            latest_path = os.path.join(output_dir, f"{sector_code}_leading_stocks_latest.csv")
            # 只保存最近30天数据
            merged_df.tail(30).to_csv(latest_path, index=False, encoding='utf-8')
            
            self.logger.info(f"板块 {sector_code} 的龙头股特征合并完成，共 {len(merged_df)} 条记录，已保存到 {output_path}")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"合并板块 {sector_code} 结果时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    
    def add_sector_leading_indicators(self, features_df, sector_code, date):
        """
        为板块特征添加龙头股指标
        
        Args:
            features_df: 原始特征DataFrame
            sector_code: 板块代码
            date: 日期
            
        Returns:
            DataFrame: 添加龙头股指标后的特征
        """
        # 将日期转为字符串格式
        if isinstance(date, datetime.datetime):
            date_str = date.strftime('%Y%m%d')
        else:
            date_str = str(date)
        
        # 查找缓存路径
        cache_path = os.path.join(self.cache_dir, "daily", f"{sector_code}_{date_str}.parquet")
        
        # 如果缓存不存在，返回原特征
        if not os.path.exists(cache_path):
            self.logger.warning(f"板块 {sector_code} 日期 {date_str} 的龙头股缓存不存在")
            return features_df
        
        # 加载缓存
        try:
            leading_features_df = pd.read_parquet(cache_path)
            if leading_features_df.empty:
                return features_df
                
            # 提取第一行作为特征
            leading_features = leading_features_df.iloc[0].to_dict()
        except Exception as e:
            self.logger.error(f"加载板块 {sector_code} 日期 {date_str} 的龙头股缓存时出错: {e}")
            return features_df
        
        # 创建一个新的DataFrame复制原始DataFrame
        enhanced_df = features_df.copy()
        
        # 添加龙头股指标
        enhanced_df['leading_stock_count'] = len(json.loads(leading_features.get('leading_stocks', '[]')))
        enhanced_df['high_elasticity_count'] = len(json.loads(leading_features.get('high_elasticity_stocks', '[]')))
        
        # 添加涨停计数
        enhanced_df['limit_up_count'] = leading_features.get('limit_up_counts_total', 0)
        enhanced_df['first_limit_count'] = leading_features.get('limit_up_counts_first_limit', 0)
        enhanced_df['multi_limit_count'] = (
            leading_features.get('limit_up_counts_second_limit', 0) +
            leading_features.get('limit_up_counts_third_limit', 0) +
            leading_features.get('limit_up_counts_fourth_limit', 0) +
            leading_features.get('limit_up_counts_fifth_plus_limit', 0)
        )
        
        # 添加换手率分布
        enhanced_df['high_turnover_count'] = leading_features.get('turnover_distribution_high', 0)
        
        # 添加热度指数
        enhanced_df['sector_heat_index'] = leading_features.get('heat_index', 0)
        
        # 添加溢价指标
        enhanced_df['avg_open_premium'] = leading_features.get('next_day_premium_avg_open_premium', 0)
        enhanced_df['max_open_premium'] = leading_features.get('next_day_premium_max_open_premium', 0)
        
        # 计算行业集中度指标 (龙头股占比)
        stock_count = leading_features.get('stock_count', 0)
        if stock_count > 0:
            enhanced_df['leading_stock_ratio'] = enhanced_df['leading_stock_count'] / stock_count
            enhanced_df['limit_up_ratio'] = enhanced_df['limit_up_count'] / stock_count
            enhanced_df['high_turnover_ratio'] = enhanced_df['high_turnover_count'] / stock_count
        else:
            enhanced_df['leading_stock_ratio'] = 0
            enhanced_df['limit_up_ratio'] = 0
            enhanced_df['high_turnover_ratio'] = 0
            
        # 添加龙头股列表和高弹性标的代码
        try:
            # 解析JSON字符串
            leading_stocks = json.loads(leading_features.get('leading_stocks', '[]'))
            top_leading_codes = [stock.get('stock_code', '') for stock in leading_stocks[:3]]
            enhanced_df['top_leading_codes'] = ','.join(top_leading_codes) if top_leading_codes else ''
            
            high_elasticity_stocks = json.loads(leading_features.get('high_elasticity_stocks', '[]'))
            elasticity_codes = [stock.get('stock_code', '') for stock in high_elasticity_stocks]
            enhanced_df['high_elasticity_codes'] = ','.join(elasticity_codes) if elasticity_codes else ''
        except Exception as e:
            self.logger.error(f"解析龙头股列表时出错: {e}")
            enhanced_df['top_leading_codes'] = ''
            enhanced_df['high_elasticity_codes'] = ''
        
        return enhanced_df
        
    def load_leading_stocks_features(self, sector_code):
        """
        加载板块龙头股特征数据
        
        Args:
            sector_code: 板块代码
            
        Returns:
            DataFrame: 龙头股特征数据
        """
        # 特征文件路径
        features_path = os.path.join(self.output_dir, sector_code, f"{sector_code}_leading_stocks.parquet")
        
        # 检查文件是否存在
        if not os.path.exists(features_path):
            self.logger.warning(f"板块 {sector_code} 的龙头股特征文件不存在: {features_path}")
            return pd.DataFrame()
        
        try:
            # 加载parquet文件
            features_df = pd.read_parquet(features_path)
            
            # 确保日期列为datetime类型
            if 'date' in features_df.columns and not pd.api.types.is_datetime64_any_dtype(features_df['date']):
                features_df['date'] = pd.to_datetime(features_df['date'])
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"加载板块 {sector_code} 的龙头股特征时出错: {e}")
            return pd.DataFrame()

    def identify_leading_stocks(self, sector_code, date, lookback_days=20, top_n=5, use_cache=True):
        """
        识别板块内的龙头股
        
        Args:
            sector_code: 板块代码
            date: 当前日期，格式为YYYYMMDD
            lookback_days: 回溯的天数，用于计算相关指标
            top_n: 返回的龙头股数量
            use_cache: 是否使用缓存
            
        Returns:
            龙头股列表，每个元素为(股票代码, 综合得分)
        """
        try:
            self.logger.info(f"开始识别板块 {sector_code} 的龙头股，日期: {date}, 回溯天数: {lookback_days}")
            
            # 获取板块内股票列表
            stocks = self._get_sector_stocks(sector_code)
            if not stocks:
                self.logger.warning(f"板块 {sector_code} 没有找到股票")
                return []
            
            self.logger.info(f"板块 {sector_code} 包含 {len(stocks)} 只股票")
            
            # 获取日期范围
            date_range = self._get_date_range(date, lookback_days)
            if not date_range:
                self.logger.warning(f"无法获取日期 {date} 的回溯日期范围")
                return []
            
            self.logger.info(f"使用日期范围: {date_range[0]} 至 {date_range[-1]}")
            
            # 如果使用缓存，检查是否存在
            if use_cache:
                cached_result = self._load_from_cache(sector_code, date, lookback_days)
                if cached_result is not None:
                    self.logger.info(f"从缓存加载板块 {sector_code} 的龙头股")
                    return cached_result
            
            # 批量加载数据
            try:
                self.logger.info(f"开始批量加载板块 {sector_code} 的股票数据，日期范围：{date_range[0]}至{date_range[-1]}，股票数量：{len(stocks)}")
                stock_data, valid_trading_days = self._batch_load_stocks_data(stocks, date_range, get_daily_data)
                if not stock_data:
                    self.logger.warning(f"板块 {sector_code} 没有加载到股票数据，跳过处理")
                    return []
                
                # 统计有效的股票数据
                valid_stocks = []
                for stock_code, accessor in stock_data.items():
                    # 检查数据覆盖率
                    if accessor.is_valid():
                        valid_stocks.append(stock_code)
                    else:
                        self.logger.debug(f"股票 {stock_code} 数据覆盖率过低，跳过")
                
                if len(valid_stocks) < 5:  # 至少需要5只有效股票才能计算龙头股
                    self.logger.warning(f"板块 {sector_code} 只有 {len(valid_stocks)}/{len(stocks)} 只有效股票，无法识别龙头股")
                    return []
                
                self.logger.info(f"板块 {sector_code} 成功加载 {len(valid_stocks)}/{len(stocks)} 只有效股票数据")
            except Exception as e:
                self.logger.error(f"批量加载板块 {sector_code} 股票数据时出错: {e}")
                self.logger.error(traceback.format_exc())
                return []
            
            # 计算每只股票的各项指标
            stock_metrics = {}
            for stock_code in valid_stocks:
                try:
                    metrics = self._calculate_stock_metrics(stock_code, stock_data[stock_code], date_range)
                    if metrics:
                        stock_metrics[stock_code] = metrics
                except Exception as e:
                    self.logger.warning(f"计算股票 {stock_code} 指标时出错: {e}")
                    continue
            
            if not stock_metrics:
                self.logger.warning(f"板块 {sector_code} 没有计算出有效的股票指标")
                return []
                
            self.logger.info(f"成功计算 {len(stock_metrics)}/{len(valid_stocks)} 只股票的指标")
            
            # 计算综合得分并排序
            scored_stocks = self._score_stocks(stock_metrics)
            
            # 获取top_n的龙头股
            leading_stocks = scored_stocks[:min(top_n, len(scored_stocks))]
            
            # 保存到缓存
            self._save_to_cache(sector_code, date, lookback_days, leading_stocks)
            
            self.logger.info(f"识别出板块 {sector_code} 的 {len(leading_stocks)} 只龙头股")
            return leading_stocks
            
        except Exception as e:
            self.logger.error(f"识别板块 {sector_code} 龙头股时出错: {e}")
            self.logger.error(traceback.format_exc())
            return []

    def _get_date_range(self, end_date, lookback_days):
        """
        获取日期范围
        
        Args:
            end_date: 结束日期
            lookback_days: 回溯天数
            
        Returns:
            日期列表
        """
        try:
            # 尝试加载交易日历
            if not hasattr(self, 'trading_days') or self.trading_days is None:
                self._load_trading_calendar()
            
            # 确保日期格式正确
            if isinstance(end_date, str):
                end_date = end_date.replace('-', '')
            elif isinstance(end_date, (datetime, pd.Timestamp)):
                end_date = end_date.strftime('%Y%m%d')
            
            # 如果没有交易日历，使用日历日
            if not hasattr(self, 'trading_days') or self.trading_days is None:
                self.logger.warning("未找到交易日历，使用日历日替代")
                end_date_obj = datetime.strptime(end_date, '%Y%m%d')
                start_date_obj = end_date_obj - timedelta(days=lookback_days * 1.5)  # 考虑周末和节假日
                
                # 生成日期范围
                date_list = []
                current_date = start_date_obj
                while current_date <= end_date_obj:
                    date_list.append(current_date.strftime('%Y%m%d'))
                    current_date += timedelta(days=1)
                
                return date_list
            
            # 从交易日历中获取日期范围
            if end_date in self.trading_days:
                end_idx = self.trading_days.index(end_date)
                if end_idx >= lookback_days - 1:
                    start_idx = end_idx - lookback_days + 1
                    return self.trading_days[start_idx:end_idx+1]
                else:
                    self.logger.warning(f"交易日历中日期 {end_date} 之前的数据不足 {lookback_days} 天")
                    return self.trading_days[:end_idx+1]
            else:
                # 找到小于等于end_date的最大交易日
                valid_days = [d for d in self.trading_days if d <= end_date]
                if not valid_days:
                    self.logger.warning(f"交易日历中没有小于等于 {end_date} 的交易日")
                    return []
                
                end_date = max(valid_days)
                end_idx = self.trading_days.index(end_date)
                
                if end_idx >= lookback_days - 1:
                    start_idx = end_idx - lookback_days + 1
                    return self.trading_days[start_idx:end_idx+1]
                else:
                    self.logger.warning(f"交易日历中日期 {end_date} 之前的数据不足 {lookback_days} 天")
                    return self.trading_days[:end_idx+1]
        
        except Exception as e:
            self.logger.error(f"获取日期范围时出错: {e}")
            self.logger.error(traceback.format_exc())
            return []
            
    def _load_trading_calendar(self):
        """加载交易日历"""
        try:
            # 检查是否已经加载过交易日历
            if hasattr(self, 'trading_calendar') and self.trading_calendar is not None:
                self.logger.debug("使用已加载的交易日历")
                return self.trading_calendar
                
            # 尝试从ParquetDataStorage中获取
            sys.path.append(str(get_project_root() / "to"))
            from advanced_quant_data import ParquetDataStorage
            
            storage = ParquetDataStorage()
            
            # 获取较长时间范围的交易日历，但不需要太早的数据
            # 只需要包含训练日期和回溯期即可，'20220101'足够早
            cal_df = storage.get_trading_calendar('20221001', '20250331')
            
            if cal_df is not None and not cal_df.empty:
                if 'date' in cal_df.columns:
                    self.trading_calendar = cal_df['date'].dt.strftime('%Y%m%d').tolist()
                else:
                    self.trading_calendar = cal_df.iloc[:, 0].dt.strftime('%Y%m%d').tolist()
                    
                self.logger.info(f"成功加载交易日历，共 {len(self.trading_calendar)} 个交易日，范围: {self.trading_calendar[0]} - {self.trading_calendar[-1]}")
                return self.trading_calendar
            else:
                self.logger.warning("获取交易日历失败")
                self.trading_calendar = None
                return None
        except Exception as e:
            self.logger.error(f"加载交易日历出错: {e}")
            self.logger.error(traceback.format_exc())
            self.trading_calendar = None
            return None

    def _save_to_cache(self, sector_code, date, lookback_days, leading_stocks):
        """保存结果到缓存"""
        try:
            if not self.cache_dir:
                return
                
            # 构建缓存目录和文件
            cache_dir = os.path.join(self.cache_dir, sector_code)
            ensure_dir(cache_dir)
            
            cache_file = os.path.join(cache_dir, f"leading_stocks_{date}_{lookback_days}.pkl")
            
            # 保存到文件
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'sector_code': sector_code,
                    'date': date,
                    'lookback_days': lookback_days,
                    'leading_stocks': leading_stocks,
                    'timestamp': datetime.now().strftime('%Y%m%d%H%M%S')
                }, f)
                
            self.logger.debug(f"已缓存板块 {sector_code} 的龙头股结果: {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"保存板块 {sector_code} 龙头股缓存时出错: {e}")
    
    def _load_from_cache(self, sector_code, date, lookback_days):
        """从缓存加载结果"""
        try:
            if not self.cache_dir:
                return None
                
            # 构建缓存文件路径
            cache_dir = os.path.join(self.cache_dir, sector_code)
            cache_file = os.path.join(cache_dir, f"leading_stocks_{date}_{lookback_days}.pkl")
            
            # 检查缓存是否存在
            if not os.path.exists(cache_file):
                return None
                
            # 检查缓存是否过期（7天过期）
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            now = datetime.now()
            days_diff = (now - file_time).days
            
            if days_diff > 7:
                self.logger.debug(f"板块 {sector_code} 的龙头股缓存已过期: {cache_file}")
                return None
            
            # 加载缓存
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # 验证缓存数据
            if (cache_data['sector_code'] == sector_code and 
                cache_data['date'] == date and 
                cache_data['lookback_days'] == lookback_days):
                self.logger.debug(f"从缓存加载板块 {sector_code} 的龙头股结果: {cache_file}")
                return cache_data['leading_stocks']
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"加载板块 {sector_code} 龙头股缓存时出错: {e}")
            return None
    
    def _get_sector_stocks(self, sector_code):
        """获取板块内的股票列表"""
        try:
            # 从股票-板块映射文件加载
            mapping_file = os.path.join(get_project_root(), "to", "local_data", "sector_mapping", f"{sector_code}.json")
            
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    sector_data = json.load(f)
                    if 'stocks' in sector_data:
                        return sector_data['stocks']
                    elif isinstance(sector_data, list):
                        return sector_data
            
            # 尝试从其他来源获取
            try:
                # 导入可能的其他来源
                sys.path.append(str(get_project_root()))
                from to.stock_sectors_mapping import get_sector_stocks
                
                stocks = get_sector_stocks(sector_code)
                if stocks:
                    return stocks
            except Exception as e:
                self.logger.warning(f"从外部模块获取板块 {sector_code} 股票列表时出错: {e}")
            
            self.logger.warning(f"未能获取板块 {sector_code} 的股票列表")
            return []
            
        except Exception as e:
            self.logger.error(f"获取板块 {sector_code} 股票列表时出错: {e}")
            self.logger.error(traceback.format_exc())
            return []

    def _calculate_stock_metrics(self, stock_code, stock_data, date_range):
        """计算单只股票的各项指标
        
        Args:
            stock_code: 股票代码
            stock_data: 股票数据，{日期: 数据}格式
            date_range: 日期范围
            
        Returns:
            dict: 股票指标字典
        """
        try:
            # 股票数据可能为空或不是字典格式
            if not stock_data or not isinstance(stock_data, dict):
                return None
                
            # 需要至少2天数据才能计算指标
            dates = sorted([d for d in stock_data.keys() if d in date_range])
            if len(dates) < 2:
                return None
                
            # 初始化结果
            metrics = {
                'stock_code': stock_code,
                'data_days': len(dates),
                'total_days': len(date_range),
                'returns': [],
                'volumes': [],
                'first_date': dates[0],
                'last_date': dates[-1]
            }
            
            # 计算每日价格变动
            for i in range(1, len(dates)):
                prev_date = dates[i-1]
                curr_date = dates[i]
                
                prev_data = stock_data.get(prev_date, {})
                curr_data = stock_data.get(curr_date, {})
                
                # 获取收盘价
                prev_close = self._get_value(prev_data, 'close')
                curr_close = self._get_value(curr_data, 'close')
                
                # 计算涨跌幅
                if prev_close is not None and curr_close is not None and prev_close > 0:
                    daily_return = (curr_close - prev_close) / prev_close
                    metrics['returns'].append(daily_return)
                
                # 获取成交量
                curr_volume = self._get_value(curr_data, 'volume')
                if curr_volume is not None and curr_volume > 0:
                    metrics['volumes'].append(curr_volume)
            
            # 计算累计收益率 - 乘积(1+r_i) - 1
            if metrics['returns']:
                metrics['cumulative_return'] = np.prod([1 + r for r in metrics['returns']]) - 1
                metrics['avg_daily_return'] = np.mean(metrics['returns'])
                metrics['volatility'] = np.std(metrics['returns']) if len(metrics['returns']) > 1 else 0
                
                # 计算上涨天数比例
                up_days = sum(1 for r in metrics['returns'] if r > 0)
                metrics['up_ratio'] = up_days / len(metrics['returns']) if metrics['returns'] else 0
                
                # 计算最大连续上涨天数
                max_up_streak = 0
                current_streak = 0
                for r in metrics['returns']:
                    if r > 0:
                        current_streak += 1
                        max_up_streak = max(max_up_streak, current_streak)
                    else:
                        current_streak = 0
                metrics['max_up_streak'] = max_up_streak
                
                # 计算形态特征 - 最近三天的趋势
                if len(metrics['returns']) >= 3:
                    recent_returns = metrics['returns'][-3:]
                    metrics['recent_trend'] = sum(recent_returns)
                    metrics['trend_acceleration'] = recent_returns[-1] - recent_returns[0]
                else:
                    metrics['recent_trend'] = 0
                    metrics['trend_acceleration'] = 0
            
            # 计算成交量特征
            if metrics['volumes']:
                metrics['avg_volume'] = np.mean(metrics['volumes'])
                if len(metrics['volumes']) >= 3:
                    metrics['volume_change'] = metrics['volumes'][-1] / np.mean(metrics['volumes'][:-1]) if np.mean(metrics['volumes'][:-1]) > 0 else 1
                else:
                    metrics['volume_change'] = 1
            
            return metrics
        except Exception as e:
            self.logger.error(f"计算股票 {stock_code} 指标时发生错误: {e}")
            return None

    def _score_stocks(self, metrics):
        """对股票进行综合评分
        
        Args:
            metrics: 股票指标字典 {股票代码: 指标}
            
        Returns:
            dict: 股票评分字典 {股票代码: 评分}
        """
        scores = {}
        
        # 如果没有指标或只有一个股票，简单返回
        if not metrics or len(metrics) < 2:
            if metrics and len(metrics) == 1:
                stock = list(metrics.keys())[0]
                scores[stock] = 1.0
            return scores
        
        # 提取各种指标值用于归一化
        returns = {}
        volatilities = {}
        up_ratios = {}
        max_streaks = {}
        recent_trends = {}
        volume_changes = {}
        
        for stock, metric in metrics.items():
            returns[stock] = metric.get('cumulative_return', 0)
            volatilities[stock] = metric.get('volatility', 0)
            up_ratios[stock] = metric.get('up_ratio', 0)
            max_streaks[stock] = metric.get('max_up_streak', 0)
            recent_trends[stock] = metric.get('recent_trend', 0)
            volume_changes[stock] = metric.get('volume_change', 1)
        
        # 归一化函数
        def normalize(values, reverse=False, default=0.5):
            if not values:
                return {}
            min_val = min(values.values())
            max_val = max(values.values())
            if max_val == min_val:
                return {k: default for k in values}
            
            result = {}
            for k, v in values.items():
                if reverse:
                    # 反向归一化，值越小得分越高（如波动率）
                    result[k] = 1 - (v - min_val) / (max_val - min_val)
                else:
                    # 正向归一化，值越大得分越高（如收益率）
                    result[k] = (v - min_val) / (max_val - min_val)
            return result
        
        # 归一化各项指标
        norm_returns = normalize(returns)
        norm_volatilities = normalize(volatilities, reverse=True)
        norm_up_ratios = normalize(up_ratios)
        norm_max_streaks = normalize(max_streaks)
        norm_recent_trends = normalize(recent_trends)
        norm_volume_changes = normalize(volume_changes)
        
        # 计算综合得分，不同指标权重不同
        weights = {
            'return': 0.35,        # 累计收益权重
            'volatility': 0.15,    # 波动率权重（越低越好）
            'up_ratio': 0.10,      # 上涨天数比例权重
            'max_streak': 0.15,    # 最大连续上涨天数权重
            'recent_trend': 0.15,  # 最近趋势权重
            'volume_change': 0.10  # 成交量变化权重
        }
        
        # 计算最终得分
        for stock in metrics:
            score = (
                weights['return'] * norm_returns.get(stock, 0) +
                weights['volatility'] * norm_volatilities.get(stock, 0) +
                weights['up_ratio'] * norm_up_ratios.get(stock, 0) +
                weights['max_streak'] * norm_max_streaks.get(stock, 0) +
                weights['recent_trend'] * norm_recent_trends.get(stock, 0) +
                weights['volume_change'] * norm_volume_changes.get(stock, 0)
            )
            scores[stock] = score
        
        return scores

    def _get_value(self, data, key):
        """从数据中安全获取值
        
        Args:
            data: 数据字典或对象
            key: 键名
            
        Returns:
            对应的值，如果获取失败则返回None
        """
        if not data:
            return None
                
        # 如果是字典
        if isinstance(data, dict):
            return data.get(key)
                
        # 如果是对象
        try:
            return getattr(data, key, None)
        except:
            pass
                
        return None



    def _identify_leading_stocks_by_date_range(self, sector_code, start_date, end_date, stock_data=None):
        """识别板块龙头股 - 使用日期范围
        
        Args:
            sector_code (str): 板块代码
            start_date (str): 开始日期
            end_date (str): 结束日期
            stock_data (dict, optional): 股票数据，如果提供则不重新加载
            
        Returns:
            dict: 龙头股信息，包含leading_stocks列表
        """
        try:
            # 如果没有提供股票数据，则加载数据
            if stock_data is None:
                # 获取板块内所有股票
                sector_stocks = self._get_sector_stocks(sector_code)
                if not sector_stocks:
                    self.logger.warning(f"板块 {sector_code} 没有股票列表，无法识别龙头股")
                    return None
                
                # 获取日期范围
                date_range = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date))
                date_range = [d.strftime('%Y%m%d') for d in date_range]
                
                # 加载股票数据 - 直接使用get_daily_data_func函数
                stock_data, _ = self._batch_load_stocks_data(sector_stocks, date_range, get_daily_data)
                
                if not stock_data or len(stock_data) <= 1:  # 只有_metadata
                    self.logger.warning(f"板块 {sector_code} 没有加载到有效股票数据，无法识别龙头股")
                    return None
            
            # 移除元数据获取首个交易日期信息
            metadata = stock_data.pop('_metadata', {}) if '_metadata' in stock_data else {}
            first_data_dates = metadata.get('first_data_dates', {})
            
            # 初始化结果
            metrics = {}
            valid_stocks = {}
            
            # 获取日期范围
            if isinstance(start_date, str) and isinstance(end_date, str):
                date_range = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date))
                date_range = [d.strftime('%Y%m%d') for d in date_range]
            else:
                # 尝试从股票数据中提取日期范围
                all_dates = set()
                for stock_code, data in stock_data.items():
                    if stock_code != '_metadata':
                        all_dates.update(data.keys())
                date_range = sorted(list(all_dates))
            
            total_days = len(date_range)
            
            # 计算每只股票的指标
            for stock_code, data in stock_data.items():
                try:
                    # 获取首个交易日期
                    first_date = first_data_dates.get(stock_code)
                    
                    # 过滤上市前5天的数据（不设涨跌幅）
                    filtered_data = {}
                    if isinstance(data, dict) and first_date and first_date in date_range:
                        # 获取首个交易日在date_range中的索引
                        idx = date_range.index(first_date)
                        # 计算上市第6天的日期索引
                        sixth_day_idx = idx + 5
                        
                        # 复制数据，但排除上市前5天
                        for date, values in data.items():
                            # 检查日期是否在date_range中且不在上市前5天
                            if date in date_range:
                                date_idx = date_range.index(date)
                                if date_idx >= sixth_day_idx:
                                    filtered_data[date] = values
                                else:
                                    self.logger.debug(f"忽略股票 {stock_code} 上市前5天的数据: {date}")
                    else:
                        # 如果无法确定首个交易日或数据不是字典，使用原始数据
                        filtered_data = data
                    
                    # 计算股票指标 - 所有股票都参与计算，不使用覆盖率筛选
                    stock_metrics = self._calculate_stock_metrics(stock_code, filtered_data, date_range)
                    if stock_metrics:
                        # 记录有效的股票指标
                        metrics[stock_code] = stock_metrics
                        valid_stocks[stock_code] = filtered_data
                except Exception as e:
                    self.logger.error(f"计算股票 {stock_code} 指标时出错: {e}")
            
            if not metrics:
                self.logger.error(f"板块 {sector_code} 没有有效的股票指标")
                return None
            
            # 对股票进行评分 - 所有股票权重相同
            scores = self._score_stocks(metrics)
            
            # 按得分排序
            sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # 返回龙头股信息
            leading_stocks = {
                'sector_code': sector_code,
                'date_range': [date_range[0], date_range[-1]],
                'leading_stocks': sorted_stocks[:10],  # 返回前10名
                'high_elasticity_stocks': [],  # 为了与单日版本兼容
                'metrics': metrics
            }
            
            self.logger.info(f"板块 {sector_code} 龙头股识别完成，有效股票数: {len(valid_stocks)}")
            return leading_stocks
        
        except Exception as e:
            self.logger.error(f"识别板块 {sector_code} 龙头股时发生错误: {e}")
            self.logger.exception(e)
            return None

    def _calculate_consecutive_limit_days(self, stock_code, date, stock_data):
        """
        计算股票连续涨停天数
        
        Args:
            stock_code: 股票代码
            date: 当前日期
            stock_data: 股票数据，{日期: 数据}格式
            
        Returns:
            int: 连续涨停天数
        """
        try:
            # 确保日期是字符串格式
            if isinstance(date, datetime.datetime):
                date_str = date.strftime('%Y%m%d')
            else:
                date_str = str(date)
                
            # 获取交易日历
            trading_calendar = self._load_trading_calendar()
            if not trading_calendar:
                return 0
                
            # 如果不是交易日，直接返回0
            if date_str not in trading_calendar:
                return 0
                
            # 获取当前日期在交易日历中的索引
            current_idx = trading_calendar.index(date_str)
            
            # 计算连续涨停天数
            consecutive_days = 0
            
            # 从当前日期向前回溯，检查每个交易日是否涨停
            for i in range(current_idx, -1, -1):
                check_date = trading_calendar[i]
                
                # 当前日期的数据
                current_data = stock_data.get(check_date, None)
                if not current_data:
                    break
                
                # 获取涨跌幅和价格数据
                pct_chg = current_data.get('pct_chg', current_data.get('change_percent', 0))
                close = current_data.get('close', None)
                pre_close = current_data.get('pre_close', None)
                
                # 检查是否涨停
                if self._is_limit_up(pct_chg, stock_code, close, pre_close):
                    consecutive_days += 1
                else:
                    break
            
            return consecutive_days
            
        except Exception as e:
            self.logger.error(f"计算股票 {stock_code} 连续涨停天数时发生错误: {e}")
            return 0
            
    def _calculate_amplitude(self, stock_data):
        """
        计算股票振幅
        
        Args:
            stock_data: 单日股票数据字典
            
        Returns:
            float: 振幅百分比
        """
        try:
            high = stock_data.get('high', 0)
            low = stock_data.get('low', 0)
            pre_close = stock_data.get('pre_close', 0)
            
            if pre_close <= 0:
                return 0
                
            # 振幅 = (最高价 - 最低价) / 昨收价 * 100%
            amplitude = (high - low) / pre_close * 100
            
            return round(amplitude, 2)
            
        except Exception as e:
            self.logger.error(f"计算股票振幅时发生错误: {e}")
            return 0
            
    def _is_broken_limit(self, stock_data, stock_code):
        """
        判断股票是否涨停破板
        
        Args:
            stock_data: 单日股票数据字典
            stock_code: 股票代码
            
        Returns:
            bool: 是否涨停破板
        """
        try:
            # 获取涨跌幅和价格数据
            pct_chg = stock_data.get('pct_chg', stock_data.get('change_percent', 0))
            close = stock_data.get('close', None)
            pre_close = stock_data.get('pre_close', None)
            open_price = stock_data.get('open', None)
            high = stock_data.get('high', None)
            
            # 确保关键数据存在
            if pre_close is None or high is None or close is None:
                return False
                
            # 确定涨停幅度比率
            limit_ratio = self.limit_up_pct  # 默认10%
            
            # 根据股票类型调整涨停幅度
            if 'ST' in stock_code.upper():
                limit_ratio = self.special_limit_configs['ST']['up']
            elif stock_code.startswith('688'):  # 科创板股票
                limit_ratio = self.special_limit_configs['STAR']['up']
            elif stock_code.startswith('300'):  # 创业板股票
                limit_ratio = self.special_limit_configs['GEM']['up']
            elif '.BJ' in stock_code:  # 北交所股票
                limit_ratio = self.special_limit_configs['BSE']['up']
                
            # 计算理论涨停价
            theoretical_limit = pre_close * (1 + limit_ratio)
            # 四舍五入到0.01元
            rounded_limit = round(theoretical_limit, 2)
            
            # 如果最高价达到涨停价，但收盘价未达到涨停价，则视为涨停破板
            is_high_reached_limit = abs(high - rounded_limit) < 0.005
            is_close_not_limit = abs(close - rounded_limit) >= 0.005
            
            return is_high_reached_limit and is_close_not_limit
            
        except Exception as e:
            self.logger.error(f"判断股票 {stock_code} 是否涨停破板时发生错误: {e}")
            return False
            
    def _is_bse_stock(self, stock_code):
        """
        判断是否为北交所股票
        
        Args:
            stock_code: 股票代码
            
        Returns:
            bool: 是否为北交所股票
        """
        # 北交所股票代码以83开头或包含.BJ后缀
        return stock_code.startswith('83') or '.BJ' in stock_code

def get_project_root():
    """获取项目根目录"""
    return Path(os.path.abspath(__file__)).parent.parent


# 确保目录存在
def ensure_dir(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main():
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='板块龙头股特征生成工具')
    parser.add_argument('--sector', type=str, help='单个板块代码')
    parser.add_argument('--sector_list', type=str, help='包含板块列表的文件路径')
    parser.add_argument('--start_date', type=str, required=True, help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end_date', type=str, required=True, help='结束日期 (YYYYMMDD)')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--cache_dir', type=str, help='缓存目录')
    
    args = parser.parse_args()
    
    # 初始化分析器
    analyzer = SectorLeadingStocksAnalyzer(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    # 准备日期范围
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    date_range = pd.date_range(start=start_date, end=end_date)
    date_range = [d.strftime('%Y%m%d') for d in date_range]
    
    # 获取板块列表
    sector_list = get_sector_list()
    
    logger.info(f"开始处理 {len(sector_list)} 个板块，日期范围: {args.start_date} 至 {args.end_date}")
    
    # 批量处理板块
    analyzer.batch_process_sectors(sector_list, date_range)
    
    logger.info("处理完成")

if __name__ == "__main__":
    main()
