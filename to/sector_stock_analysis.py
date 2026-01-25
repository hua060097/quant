#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sector_stock_analysis.py

板块涨跌停板分析模块
用于统计板块内涨跌停板情况、连板股票、涨停类型和市值分布等特征
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import traceback
from datetime import timedelta
sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录


# log_file = 'to/log/sector_board_analysis.log'
log_file = 'log/sector_board_analysis.log'
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
        
class SectorBoardAnalyzer:
    """板块涨跌停板分析器"""
    
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
    
    def __init__(self, output_dir=None, cache_dir=None, enable_cache=True, update_mode=False):
        """
        初始化板块涨跌停板分析器

        Args:
            output_dir: 输出目录，默认为'to/local_data/sector_features'
            cache_dir: 缓存目录，默认为'to/local_data/features_cache/board_analysis'
            enable_cache: 是否启用缓存
            update_mode: 是否为更新模式，True时按日期合并到现有文件
        """
        self.logger = logger
        self.enable_cache = enable_cache
        self.update_mode = update_mode
        self.update_mode = update_mode
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # 获取项目根目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            self.output_dir = project_root / "to" / "local_data" / "sector_features_board"
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置缓存目录
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # 获取项目根目录
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            self.cache_dir = project_root / "to" / "local_data" / "features_cache" / "board_analysis"
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 涨跌停相关设置
        self.limit_up_pct = 0.1  # 默认涨停幅度10%
        self.limit_down_pct = -0.1  # 默认跌停幅度-10%
        
        # 不同市场的特殊涨跌停设置
        self.special_limit_configs = {
            'ST': {'up': 0.05, 'down': -0.05},  # ST股票
            'STAR': {'up': 0.2, 'down': -0.2},  # 科创板
            'GEM': {'up': 0.2, 'down': -0.2},   # 创业板
            'BSE': {'up': 0.3, 'down': -0.3}    # 北交所
        }

        
        # 板块涨停分权重
        self.board_analysis_weights = {
            'limit_up_count': 0.30,          # 涨停数量权重
            'limit_up_ratio': 0.20,          # 涨停占比权重
            'consecutive_boards': 0.25,      # 连板梯队权重
            'highest_board': 0.15,           # 最高连板权重
            'market_cap_dist': 0.10,         # 市值分布权重
            'broken_limit_penalty': 0.0     # 炸板惩罚权重（负分）
        }
        
        # 创建缓存目录结构
        (self.cache_dir / "daily").mkdir(exist_ok=True)
        (self.cache_dir / "stocks").mkdir(exist_ok=True)

        # 性能优化相关
        self.data_cache = {}  # 数据缓存
        self.trading_calendar = None  # 交易日历缓存

        self.logger.info(f"板块涨跌停板分析器初始化完成，输出目录: {self.output_dir}，缓存目录: {self.cache_dir}")
        self.logger.info(f"缓存状态: {'启用' if enable_cache else '禁用'}")

    def _is_limit_up(self, pct_chg, stock_code, close=None, preClose=None):
        """
        判断股票是否涨停，优化后的判断逻辑

        Args:
            pct_chg: 涨跌幅
            stock_code: 股票代码
            close: 收盘价(可选)
            preClose: 前收盘价(可选)

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
        if close is not None and preClose is not None and preClose > 0:
            # 计算理论涨停价
            theoretical_limit = preClose * (1 + limit_ratio)
            # 四舍五入到0.01元
            rounded_limit = round(theoretical_limit, 2)
            # 判断当前价格是否等于四舍五入后的涨停价
            return abs(close - rounded_limit) < 0.005
        else:
            # 如果未提供价格数据，则使用涨跌幅判断，允许一定误差
            limit_pct = limit_ratio * 100
            return abs(pct_chg - limit_pct) <= 0.15


    def _calculate_sector_heat(self, sector_features):
        """
        计算板块热度指数

        Args:
            sector_features: 板块特征数据

        Returns:
            float: 热度指数(0-100)
        """
        try:
            # 初始化热度分项
            heat_scores = {
                'limit_up_count': 0,         # 涨停数量得分
                'limit_up_ratio': 0,         # 涨停占比得分
                'consecutive_boards': 0,      # 连板梯队得分
                'highest_board': 0,          # 最高连板得分
                'market_cap_dist': 0,        # 市值分布得分
                'broken_limit_penalty': 0    # 炸板惩罚得分
            }
            
            # 计算涨停数量得分(0-100)
            limit_up_count = sector_features['limit_up_count']
            stock_count = max(1, sector_features['stock_count'])
            
            # 涨停数量得分 - 5只涨停或10%涨停比例满分
            limit_up_score = min(100, limit_up_count * 20)  # 每个涨停20分，最高100
            heat_scores['limit_up_count'] = limit_up_score
            
            # 涨停占比得分 - 20%涨停比例满分
            limit_up_ratio = limit_up_count / stock_count
            heat_scores['limit_up_ratio'] = min(100, limit_up_ratio * 500)  # 20%涨停比例满分
            
            # 连板梯队得分 - 根据不同连板数量加权
            board_counts = sector_features.get('board_counts', {})
            if board_counts:
                weights = {
                    "1板": 1,
                    "2板": 2,
                    "3板": 3,
                    "4板": 5,
                    "5板": 8,
                    "6板": 12,
                    "7板": 16,
                    "8板": 20,
                    "9板": 25,
                    "10板及以上": 30
                }
                
                # 计算连板加权得分
                board_score = 0
                for board_key, count in board_counts.items():
                    if board_key in weights:
                        board_score += count * weights[board_key]
                
                # 标准化得分，25分相当于5个首板或1个5板
                heat_scores['consecutive_boards'] = min(100, board_score * 4)
            
            # 最高连板得分 - 根据最高连板数
            highest_board = 1  # 默认为1板
            for board_key in ["10板及以上", "9板", "8板", "7板", "6板", "5板", "4板", "3板", "2板"]:
                if board_key in board_counts and board_counts[board_key] > 0:
                    highest_board_num = int(board_key.replace("板及以上", "").replace("板", ""))
                    highest_board = highest_board_num
                    break
            
            # 最高连板得分
            # 1-10板对应10-100分
            heat_scores['highest_board'] = min(100, highest_board * 10)
            
            # 炸板惩罚得分
            broken_limit_count = sector_features.get('broken_limit_up_count', 0)
            limit_up_count = sector_features.get('limit_up_count', 1)  # 避免除零
            if broken_limit_count > 0 and limit_up_count > 0:
                # 炸板率越高，惩罚越大
                broken_ratio = broken_limit_count / limit_up_count
                heat_scores['broken_limit_penalty'] = -min(50, broken_ratio * 100)  # 最大扣50分
            
            # 市值分布得分 - 使用实际存在的字段
            market_cap_dist = sector_features.get('limit_board_distribution', {})
            if market_cap_dist:
                # 权重设置 - 小市值权重高一些
                cap_weights = {
                    'small': 1.2,    # 小市值(<50亿)权重1.2
                    'medium': 1.0,   # 中市值(50-200亿)权重1.0
                    'large': 0.8     # 大市值(>200亿)权重0.8
                }

                # 计算加权得分
                cap_score = 0
                total_caps = 0
                for cap_key, count in market_cap_dist.items():
                    weight = cap_weights.get(cap_key, 1.0)
                    cap_score += count * weight
                    total_caps += count

                # 标准化得分
                if total_caps > 0:
                    heat_scores['market_cap_dist'] = min(100, cap_score * 15 / total_caps)
            
            # 加权计算最终热度指数
            final_heat = 0
            for metric, score in heat_scores.items():
                if metric in self.board_analysis_weights:
                    final_heat += score * self.board_analysis_weights[metric]
            
            # 返回最终热度指数(四舍五入到2位小数)
            return round(final_heat, 2)
        except Exception as e:
            self.logger.error(f"计算板块热度指数时出错: {e}")
            return 0

    def _calculate_premium(self, stock_data, prev_day_data):
        """
        计算股票的溢价率
        
        Args:
            stock_data: 当日股票数据
            prev_day_data: 前一日股票数据
            
        Returns:
            float: 溢价率 (%)
        """
        try:
            if not stock_data or not prev_day_data:
                return 0
                
            # 获取当日开盘价和前一日收盘价
            open_price = stock_data.get('open', 0)
            prev_close = prev_day_data.get('close', 0)
            
            if prev_close <= 0:
                return 0
                
            # 计算溢价率 = (今日开盘价 / 昨日收盘价 - 1) * 100%
            premium = (open_price / prev_close - 1) * 100
            
            return round(premium, 2)
        except Exception as e:
            self.logger.error(f"计算溢价率时出错: {e}")
            return 0
            
    def process_sector_data(self, sector_code, date, stocks_data, prev_day_stocks_data=None):
        """
        处理板块日度数据，计算板块指标
        
        Args:
            sector_code: 板块代码
            date: 日期，格式为YYYYMMDD
            stocks_data: 板块内所有股票的当日行情数据
            prev_day_stocks_data: 前一交易日行情数据(可选)，用于计算变化
            
        Returns:
            dict: 板块指标字典
        """
        try:
            # 确保输入有效
            if not stocks_data or not isinstance(stocks_data, dict):
                self.logger.warning(f"板块 {sector_code} 日期 {date} 的股票数据无效，跳过处理")
                return {}
            
            date_str = date
            if isinstance(date, (datetime.datetime, pd.Timestamp)):
                date_str = date.strftime('%Y%m%d')
                
            # 获取交易日历
            trading_calendar = self._load_trading_calendar()
            if trading_calendar and date_str in trading_calendar:
                # 获取前一交易日
                current_idx = trading_calendar.index(date_str)
                if current_idx > 0:
                    prev_date_str = trading_calendar[current_idx - 1]
                    
                    # 标记昨日涨停股票
                    if prev_day_stocks_data:
                        for stock_code, today_data in stocks_data.items():
                            if stock_code in prev_day_stocks_data:
                                prev_data = prev_day_stocks_data[stock_code]
                                
                                # 判断昨日是否涨停
                                prev_pct_chg = prev_data.get('pct_chg', prev_data.get('change_percent', 0))
                                prev_close = prev_data.get('close')
                                prev_preClose = prev_data.get('preClose')
                                turnover_rate = prev_data.get('turnover_rate', None)

                                if self._is_limit_up(prev_pct_chg, stock_code, prev_close, prev_preClose):
                                    # 标记为昨日涨停
                                    today_data['yesterday_limit_up'] = True

                                    # 计算连续涨停天数（优化版本，避免重复数据加载）
                                    consecutive_days = self._calculate_consecutive_limit_days_optimized(
                                        stock_code, prev_date_str, prev_day_stocks_data
                                    )
                                    today_data['yesterday_consecutive_days'] = consecutive_days
                                    self.logger.info(f"计算连续涨停天数: {stock_code}, 日期: {prev_date_str}, 连续涨停天数: {consecutive_days}, 换手率: {turnover_rate}")

                                    # 计算溢价
                                    premium = self._calculate_premium(today_data, prev_data)
                                    today_data['premium'] = premium
                                elif self._is_broken_limit(prev_data, stock_code):
                                    # 标记为昨日炸板
                                    today_data['yesterday_broken_limit'] = True
                                    today_data['yesterday_limit_up'] = False
                                    today_data['yesterday_consecutive_days'] = 0

                                    # 计算溢价
                                    premium = self._calculate_premium(today_data, prev_data)
                                    today_data['premium'] = premium
                                else:
                                    today_data['yesterday_limit_up'] = False
                                    today_data['yesterday_broken_limit'] = False
                                    today_data['yesterday_consecutive_days'] = 0
                                    today_data['premium'] = 0
            
            # 计算板块统计数据
            sector_stats = self._calculate_board_statistics(stocks_data, date_str)
            
            # 计算板块热度
            heat_index = self._calculate_sector_heat(sector_stats)
            sector_stats['heat_index'] = heat_index
            
            # 添加基本信息
            sector_stats['sector_code'] = sector_code
            sector_stats['date'] = date_str
            sector_stats['date_obj'] = pd.to_datetime(date_str)
            
            # 保存到缓存
            self._save_results_to_cache(sector_code, date_str, sector_stats)
            
            return sector_stats
            
        except Exception as e:
            self.logger.error(f"处理板块 {sector_code} 日期 {date} 数据时出错: {e}")
            self.logger.error(traceback.format_exc())
            return {}

    def optimized_batch_process(self, sector_list, date_range, max_workers=4, batch_size=10):
        """
        高性能批量处理方法

        Args:
            sector_list: 板块代码列表
            date_range: 日期范围列表
            max_workers: 最大并发数
            batch_size: 批处理大小

        Returns:
            list: 处理成功的板块列表
        """
        try:
            self.logger.info(f"开始高性能批量处理 {len(sector_list)} 个板块")
            self.logger.info(f"配置: 最大并发={max_workers}, 批处理大小={batch_size}")

            # 预加载交易日历
            self._load_trading_calendar()

            # 过滤有效交易日
            valid_trading_days = [day for day in date_range if day in self.trading_calendar] if self.trading_calendar else date_range
            self.logger.info(f"有效交易日: {len(valid_trading_days)} 天")

            # 分批处理板块
            processed_sectors = []
            total_batches = (len(sector_list) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(sector_list))
                batch_sectors = sector_list[start_idx:end_idx]

                self.logger.info(f"处理批次 {batch_idx + 1}/{total_batches}: {len(batch_sectors)} 个板块")

                # 批量处理当前批次
                batch_results = self._process_sector_batch(
                    batch_sectors,
                    valid_trading_days,
                    max_workers
                )

                processed_sectors.extend(batch_results)

                # 清理缓存，释放内存
                if hasattr(self, 'data_cache'):
                    self.data_cache.clear()

                self.logger.info(f"批次 {batch_idx + 1} 完成，处理了 {len(batch_results)} 个板块")

            self.logger.info(f"高性能批量处理完成，共处理 {len(processed_sectors)} 个板块")
            return processed_sectors

        except Exception as e:
            self.logger.error(f"高性能批量处理时出错: {e}")
            self.logger.error(traceback.format_exc())
            return []

    def _process_sector_batch(self, sector_list, date_range, max_workers):
        """
        处理一批板块

        Args:
            sector_list: 板块列表
            date_range: 日期范围
            max_workers: 最大并发数

        Returns:
            list: 处理成功的板块列表
        """
        try:
            # 导入数据加载函数
            from data_loaders import get_stock_list_in_sector, get_daily_data

            processed_sectors = []

            # 使用较少的并发数避免资源竞争
            actual_workers = min(max_workers, len(sector_list))

            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                # 提交任务
                futures = {}
                for sector_code in sector_list:
                    future = executor.submit(
                        self._process_single_sector_optimized,
                        sector_code,
                        date_range,
                        get_stock_list_in_sector,
                        get_daily_data
                    )
                    futures[future] = sector_code

                # 收集结果
                for future in as_completed(futures):
                    sector_code = futures[future]
                    try:
                        success = future.result()
                        if success:
                            processed_sectors.append(sector_code)
                            self.logger.debug(f"板块 {sector_code} 处理成功")
                        else:
                            self.logger.warning(f"板块 {sector_code} 处理失败")
                    except Exception as e:
                        self.logger.error(f"处理板块 {sector_code} 时出错: {e}")

            return processed_sectors

        except Exception as e:
            self.logger.error(f"批处理板块时出错: {e}")
            return []

    def _process_single_sector_optimized(self, sector_code, date_range, get_stock_list_func, get_daily_data_func):
        """
        优化的单板块处理方法

        Args:
            sector_code: 板块代码
            date_range: 日期范围
            get_stock_list_func: 获取股票列表函数
            get_daily_data_func: 获取数据函数

        Returns:
            bool: 是否处理成功
        """
        try:
            # 获取板块内股票列表
            stocks = get_stock_list_func(sector_code)
            if not stocks:
                
                self.logger.warning(f"板块 {sector_code} 没有股票")
                return False

            # 限制股票数量，避免内存过大
            # if len(stocks) > 200:
            #     self.logger.warning(f"板块 {sector_code} 股票数量过多({len(stocks)})，截取前200只")
            #     stocks = stocks[:200]

            # 一次性加载所有数据
            try:
                stock_data_df = get_daily_data_func(stocks, date_range[0], date_range[-1])
                if stock_data_df is None or stock_data_df.empty:
                    self.logger.warning(f"板块 {sector_code} 没有数据")
                    return False
            except Exception as e:
                self.logger.error(f"加载板块 {sector_code} 数据失败: {e}")
                return False

            # 转换数据格式
            stock_data_dict = self._convert_dataframe_to_dict(stock_data_df)

            # 按日期处理
            results_count = 0
            for i, date in enumerate(date_range):
                if date not in self.trading_calendar:
                    continue

                try:
                    # 获取当日数据
                    current_stocks = {}
                    for stock_code, date_dict in stock_data_dict.items():
                        if date in date_dict:
                            current_stocks[stock_code] = date_dict[date]

                    if not current_stocks:
                        continue

                    # 获取前一日数据
                    prev_day_data = None
                    if i > 0:
                        prev_date = date_range[i-1]
                        if prev_date in self.trading_calendar:
                            prev_day_data = {}
                            for stock_code, date_dict in stock_data_dict.items():
                                if prev_date in date_dict:
                                    prev_day_data[stock_code] = date_dict[prev_date]

                    # 处理板块数据
                    sector_features = self.process_sector_data(
                        sector_code,
                        date,
                        current_stocks,
                        prev_day_data
                    )

                    if sector_features:
                        results_count += 1

                except Exception as e:
                    self.logger.error(f"处理板块 {sector_code} 日期 {date} 时出错: {e}")
                    continue

            # 合并结果
            if results_count > 0:
                try:
                    merged_df = self._merge_sector_results(sector_code, date_range)
                    if merged_df is not None and not merged_df.empty:
                        # # 保存结果
                        # output_path = self.output_dir / sector_code
                        # output_path.mkdir(exist_ok=True)
                        # merged_df.to_parquet(output_path / f"{sector_code}_board_analysis.parquet", index=False)
                        self.logger.info(f"板块 {sector_code} 处理完成，{results_count} 个有效日期")
                        return True
                except Exception as e:
                    self.logger.error(f"保存板块 {sector_code} 结果时出错: {e}")

            return False

        except Exception as e:
            self.logger.error(f"处理板块 {sector_code} 时出错: {e}")
            return False

    def _convert_dataframe_to_dict(self, df):
        """
        将DataFrame转换为嵌套字典格式

        Args:
            df: 股票数据DataFrame

        Returns:
            dict: {股票代码: {日期: 数据}} 格式
        """
        try:
            # 确保有date_str列
            if 'date_str' not in df.columns and 'date' in df.columns:
                df['date_str'] = df['date'].dt.strftime('%Y%m%d')

            result = {}
            for stock_code, group in df.groupby('stock_code'):
                date_dict = {}
                for _, row in group.iterrows():
                    date_key = row.get('date_str', str(row.get('date', '')))
                    date_dict[date_key] = row.to_dict()
                result[stock_code] = date_dict

            return result

        except Exception as e:
            self.logger.error(f"转换数据格式时出错: {e}")
            return {}

    def _calculate_consecutive_limit_days_optimized(self, stock_code, date, prev_day_stocks_data):
        """
        优化的连续涨停天数计算（避免重复数据加载）

        Args:
            stock_code: 股票代码
            date: 当前日期
            prev_day_stocks_data: 前一日所有股票数据

        Returns:
            int: 连续涨停天数
        """
        try:
            # 如果没有前一日数据，返回1（当天首次涨停）
            if not prev_day_stocks_data or stock_code not in prev_day_stocks_data:
                return 1

            prev_data = prev_day_stocks_data[stock_code]

            # 检查前一日是否涨停
            prev_pct_chg = prev_data.get('pct_chg', prev_data.get('change_percent', 0))
            prev_close = prev_data.get('close')
            prev_preClose = prev_data.get('preClose')

            if not self._is_limit_up(prev_pct_chg, stock_code, prev_close, prev_preClose):
                return 1  # 前一日未涨停，当天是首次涨停

            # 前一日涨停，获取前一日的连续涨停天数
            prev_consecutive = prev_data.get('yesterday_consecutive_days', 0)

            # 当前连续涨停天数 = 前一日连续涨停天数 + 1
            return prev_consecutive + 1

        except Exception as e:
            self.logger.error(f"计算股票 {stock_code} 连续涨停天数时出错: {e}")
            return 1  # 出错时返回1
            
    def _process_sector_with_optimized_data_loading(self, sector_code, date_range, get_stock_list_func, get_daily_data_func):
        """
        使用优化的数据加载方式处理板块数据
        
        Args:
            sector_code: 板块代码
            date_range: 日期范围
            get_stock_list_func: 获取板块内股票列表的函数
            get_daily_data_func: 获取日线数据的函数
            
        Returns:
            dict: {日期: 特征} 格式的结果
        """
        try:
            self.logger.info(f"开始处理板块 {sector_code} 的数据，日期范围: {date_range[0]} - {date_range[-1]}")
            
            # 获取板块内股票列表
            stocks = get_stock_list_func(sector_code)
            
            if not stocks:
                self.logger.warning(f"板块 {sector_code} 没有找到股票列表，跳过处理")
                return {}
                
            self.logger.info(f"板块 {sector_code} 包含 {len(stocks)} 只股票")
            
            # 批量加载股票日线数据
            stock_data, valid_trading_days = self._batch_load_stocks_data(stocks, date_range, get_daily_data_func)
                
            if not stock_data:
                self.logger.warning(f"板块 {sector_code} 没有加载到股票数据，跳过处理")
                return {}
            
            self.logger.info(f"成功加载板块 {sector_code} 的数据，共 {len(valid_trading_days)} 个交易日")
            
            # 处理每个日期
            results = {}
            dates_list = valid_trading_days
            
            for i, date in enumerate(dates_list):
                try:
                    # 获取前一日数据，用于计算连板
                    prev_date = None
                    prev_day_data = None
                    
                    if i > 0:
                        prev_date = dates_list[i-1]
                        current_stocks = {}
                        for stock_code, date_dict in stock_data.items():
                            if stock_code != '_metadata' and prev_date in date_dict:
                                current_stocks[stock_code] = date_dict[prev_date]
                        prev_day_data = current_stocks
                    
                    # 处理当日数据
                    current_stocks = {}
                    for stock_code, date_dict in stock_data.items():
                        if stock_code != '_metadata' and date in date_dict:
                            current_stocks[stock_code] = date_dict[date]
                    
                    if not current_stocks:
                        continue
                        
                    sector_features = self.process_sector_data(
                        sector_code, 
                        date, 
                        current_stocks, 
                        prev_day_data
                    )
                    
                    if sector_features:
                        results[date] = sector_features
                        
                except Exception as e:
                    self.logger.error(f"处理板块 {sector_code} 日期 {date} 时出错: {e}")
                    self.logger.error(traceback.format_exc())
            
            self.logger.info(f"板块 {sector_code} 处理完成，有效日期数: {len(results)}/{len(dates_list)}")
            return results
            
        except Exception as e:
            self.logger.error(f"处理板块 {sector_code} 时出错: {e}")
            self.logger.error(traceback.format_exc())
            return {}

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
        支持更新模式：按日期合并到现有文件，而不是覆盖

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
            new_data_files = []
            for date in date_range:
                date_str = date if isinstance(date, str) else date.strftime('%Y%m%d')
                cache_file = os.path.join(daily_cache_dir, f"{sector_code}_{date_str}.parquet")
                if os.path.exists(cache_file):
                    new_data_files.append(cache_file)

            if not new_data_files:
                self.logger.warning(f"板块 {sector_code} 没有找到任何缓存文件")
                return None

            # 读取新数据
            new_dfs = []
            for file in new_data_files:
                try:
                    df = pd.read_parquet(file)
                    new_dfs.append(df)
                except Exception as e:
                    self.logger.error(f"读取文件 {file} 时出错: {e}")

            if not new_dfs:
                self.logger.warning(f"板块 {sector_code} 没有有效的缓存文件")
                return None

            # 合并新数据
            new_merged_df = pd.concat(new_dfs, ignore_index=True)

            # 确保日期列是datetime类型
            if 'date' in new_merged_df.columns:
                new_merged_df['date'] = pd.to_datetime(new_merged_df['date'])

            # 设置输出路径
            output_dir = os.path.join(self.output_dir, sector_code)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{sector_code}_board_analysis.parquet")

            # 根据模式处理数据
            if self.update_mode and os.path.exists(output_path):
                # 更新模式：读取现有文件并按日期合并
                try:
                    existing_df = pd.read_parquet(output_path)

                    # 确保现有数据的日期列也是datetime类型
                    if 'date' in existing_df.columns:
                        existing_df['date'] = pd.to_datetime(existing_df['date'])

                    # 获取新数据的日期范围
                    new_dates = set(new_merged_df['date'].dt.strftime('%Y-%m-%d'))

                    # 从现有数据中移除与新数据重复的日期（用新数据覆盖）
                    existing_df_filtered = existing_df[
                        ~existing_df['date'].dt.strftime('%Y-%m-%d').isin(new_dates)
                    ]

                    # 合并现有数据和新数据
                    final_df = pd.concat([existing_df_filtered, new_merged_df], ignore_index=True)

                    # 按日期排序，保持与文件内一致的排序
                    final_df = final_df.sort_values('date')

                    self.logger.info(f"板块 {sector_code} 更新模式：现有数据 {len(existing_df)} 条，"
                                   f"新数据 {len(new_merged_df)} 条，"
                                   f"合并后 {len(final_df)} 条")

                except Exception as e:
                    self.logger.error(f"读取现有文件 {output_path} 失败，将使用覆盖模式: {e}")
                    final_df = new_merged_df.sort_values('date')
            else:
                # 覆盖模式或文件不存在：直接使用新数据
                final_df = new_merged_df.sort_values('date')
                mode_desc = "覆盖模式" if not self.update_mode else "文件不存在，创建新文件"
                self.logger.info(f"板块 {sector_code} {mode_desc}：新数据 {len(final_df)} 条")

            # 保存合并结果
            final_df.to_parquet(output_path)

            # 同时保存最新数据为CSV，便于查看
            latest_path = os.path.join(output_dir, f"{sector_code}_leading_stocks_latest.csv")
            # 只保存最近30天数据
            final_df.tail(30).to_csv(latest_path, index=False, encoding='utf-8')

            self.logger.info(f"板块 {sector_code} 的板块分析特征合并完成，共 {len(final_df)} 条记录，已保存到 {output_path}")

            return final_df

        except Exception as e:
            self.logger.error(f"合并板块 {sector_code} 结果时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    
    def _calculate_consecutive_limit_days(self, stock_code, date, get_daily_data_func=None):
        """
        计算股票连续涨停天数

        Args:
            stock_code: 股票代码
            date: 当前日期
            get_daily_data_func: 获取历史数据的函数，如果为None则尝试导入默认函数

        Returns:
            int: 连续涨停天数
        """
        try:
            # 确保日期是字符串格式
            if isinstance(date, datetime.datetime):
                date_str = date.strftime('%Y%m%d')
            else:
                date_str = str(date)

            self.logger.debug(f"开始计算股票 {stock_code} 在 {date_str} 的连续涨停天数")

            # 获取交易日历
            trading_calendar = self._load_trading_calendar()
            if not trading_calendar:
                self.logger.error(f"无法获取交易日历")
                return 0

            # 如果不是交易日，直接返回0
            if date_str not in trading_calendar:
                self.logger.debug(f"当前日期 {date_str} 不是交易日")
                return 0

            # 获取当前日期在交易日历中的索引
            current_idx = trading_calendar.index(date_str)

            # 获取数据加载函数
            if get_daily_data_func is None:
                try:
                    sys.path.append(str(get_project_root() / "to"))
                    from data_loaders import get_daily_data
                    get_daily_data_func = get_daily_data
                except ImportError:
                    self.logger.error("无法导入数据加载函数")
                    return 0

            # 计算需要回溯的日期范围（最多回溯30个交易日）
            max_lookback = min(30, current_idx + 1)
            start_idx = max(0, current_idx - max_lookback + 1)

            # 获取回溯期间的日期列表
            lookback_dates = trading_calendar[start_idx:current_idx + 1]

            if not lookback_dates:
                return 0

            # 获取该股票的历史数据
            try:
                stock_data_df = get_daily_data_func([stock_code], lookback_dates[0], lookback_dates[-1])

                if stock_data_df is None or stock_data_df.empty:
                    self.logger.debug(f"股票 {stock_code} 在期间 {lookback_dates[0]}-{lookback_dates[-1]} 没有数据")
                    return 0

                # 过滤该股票的数据
                stock_data_df = stock_data_df[stock_data_df['stock_code'] == stock_code].copy()

                if stock_data_df.empty:
                    self.logger.debug(f"股票 {stock_code} 没有有效数据")
                    return 0

                # 确保有date_str列
                if 'date_str' not in stock_data_df.columns and 'date' in stock_data_df.columns:
                    stock_data_df['date_str'] = stock_data_df['date'].dt.strftime('%Y%m%d')

                # 转换为字典格式 {日期: 数据}
                stock_history = {}
                for _, row in stock_data_df.iterrows():
                    date_key = row.get('date_str', str(row.get('date', '')))
                    stock_history[date_key] = row.to_dict()

            except Exception as e:
                self.logger.error(f"获取股票 {stock_code} 历史数据时出错: {e}")
                return 0

            # 计算连续涨停天数
            consecutive_days = 0

            # 从当前日期向前回溯，检查每个交易日是否涨停
            for i in range(current_idx, -1, -1):
                check_date = trading_calendar[i]

                # 获取该日期的数据
                current_data = stock_history.get(check_date, None)

                if not current_data:
                    self.logger.debug(f"股票 {stock_code} 在日期 {check_date} 没有数据，停止回溯")
                    break

                # 获取涨跌幅和价格数据
                pct_chg = current_data.get('pct_chg', current_data.get('change_percent', 0))
                close = current_data.get('close', None)
                preClose = current_data.get('preClose', None)
                turnover_rate = current_data.get('turnover_rate', None)

                # 检查是否涨停
                if self._is_limit_up(pct_chg, stock_code, close, preClose):
                    consecutive_days += 1
                    self.logger.info(f"股票 {stock_code} 在 {check_date} 涨停，当前连板数: {consecutive_days}, 换手率: {turnover_rate}")
                else:
                    self.logger.debug(f"股票 {stock_code} 在 {check_date} 未涨停，停止回溯")
                    break

            self.logger.debug(f"股票 {stock_code} 在 {date_str} 的连续涨停天数: {consecutive_days}")
            return consecutive_days

        except Exception as e:
            self.logger.error(f"计算股票 {stock_code} 连续涨停天数时发生错误: {e}")
            self.logger.error(traceback.format_exc())
            return 0

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
            cal_df = storage.get_trading_calendar('20221001', '20250720')
            
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
            preClose = stock_data.get('preClose', 0)

            if preClose <= 0:
                return 0

            # 振幅 = (最高价 - 最低价) / 昨收价 * 100%
            amplitude = (high - low) / preClose * 100
            
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
            preClose = stock_data.get('preClose', None)
            open_price = stock_data.get('open', None)
            high = stock_data.get('high', None)

            # 确保关键数据存在
            if preClose is None or high is None or close is None:
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
            theoretical_limit = preClose * (1 + limit_ratio)
            # 四舍五入到0.01元
            rounded_limit = round(theoretical_limit, 2)
            
            # 如果最高价达到涨停价，但收盘价未达到涨停价，则视为涨停破板
            is_high_reached_limit = abs(high - rounded_limit) < 0.005
            is_close_not_limit = abs(close - rounded_limit) >= 0.005
            
            return is_high_reached_limit and is_close_not_limit
            
        except Exception as e:
            self.logger.error(f"判断股票 {stock_code} 是否涨停破板时发生错误: {e}")
            return False

    def _get_stock_limit_ratios(self, stock_code):
        """
        获取股票的涨跌停限制比例

        Args:
            stock_code: 股票代码

        Returns:
            tuple: (涨停比例, 跌停比例)
        """
        # 根据股票类型确定涨跌停幅度
        if 'ST' in stock_code.upper():
            return self.special_limit_configs['ST']['up'], self.special_limit_configs['ST']['down']
        elif stock_code.startswith('688'):  # 科创板股票
            return self.special_limit_configs['STAR']['up'], self.special_limit_configs['STAR']['down']
        elif stock_code.startswith('300'):  # 创业板股票
            return self.special_limit_configs['GEM']['up'], self.special_limit_configs['GEM']['down']
        elif '.BJ' in stock_code:  # 北交所股票
            return self.special_limit_configs['BSE']['up'], self.special_limit_configs['BSE']['down']
        else:
            return self.limit_up_pct, self.limit_down_pct

    def _is_limit_down(self, pct_chg, stock_code, close=None, preClose=None):
        """
        判断股票是否跌停，与涨停判断逻辑类似

        Args:
            pct_chg: 涨跌幅
            stock_code: 股票代码
            close: 收盘价(可选)
            preClose: 前收盘价(可选)

        Returns:
            bool: 是否跌停
        """
        # 确定跌停幅度比率
        limit_ratio = self.limit_down_pct  # 默认-10%
        
        # 根据股票类型调整跌停幅度
        if 'ST' in stock_code.upper():
            limit_ratio = self.special_limit_configs['ST']['down']
        elif stock_code.startswith('688'):  # 科创板股票
            limit_ratio = self.special_limit_configs['STAR']['down']
        elif stock_code.startswith('300'):  # 创业板股票
            limit_ratio = self.special_limit_configs['GEM']['down']
        elif '.BJ' in stock_code:  # 北交所股票
            limit_ratio = self.special_limit_configs['BSE']['down']
            
        # 如果提供了收盘价和前收盘价，使用精确计算
        if close is not None and preClose is not None and preClose > 0:
            # 计算理论跌停价
            theoretical_limit = preClose * (1 + limit_ratio)
            # 四舍五入到0.01元
            rounded_limit = round(theoretical_limit, 2)
            # 判断当前价格是否等于四舍五入后的跌停价
            return abs(close - rounded_limit) < 0.005
        else:
            # 如果未提供价格数据，则使用涨跌幅判断，允许一定误差
            limit_pct = limit_ratio * 100
            return abs(pct_chg - limit_pct) <= 0.15

    def _calculate_board_statistics(self, stock_data, date_str):
        """计算板块统计数据

        Args:
            stock_data: 股票数据
            date_str: 日期字符串

        Returns:
            dict: 板块统计数据
        """
        stats = {
            'stock_count': 0,          # 股票数量
            'up_count': 0,             # 上涨数量
            'down_count': 0,           # 下跌数量
            'flat_count': 0,           # 平盘数量
            'avg_pct_chg': 0,          # 平均涨跌幅
            'avg_turnover': 0,         # 平均换手率
            'avg_amplitude': 0,        # 平均振幅
            # 涨跌停统计
            'limit_up_count': 0,      # 涨停数量
            'limit_down_count': 0,    # 跌停数量
            'near_limit_up_count': 0, # 接近涨停数量
            'near_limit_down_count': 0, # 接近跌停数量
            'one_word_limit_up_count': 0,  # 一字涨停数量
            'max_consecutive_limit_up': 0,  # 最高连板数
            'high_elasticity_count': 0,  # 高弹性股数量
            'yesterday_limit_up_premium': 0,  # 昨日涨停股今日平均溢价
            'yesterday_max_limit_up_premium': 0,  # 昨日最高连板今日溢价
            'yesterday_broken_limit_premium': 0,  # 昨日炸板股今日平均溢价
            # 分布统计
            'market_cap_distribution': {
                'large': 0,   # 大市值 (>200亿)
                'medium': 0,  # 中市值 (50-200亿)
                'small': 0    # 小市值 (<50亿)
            },
            'limit_board_distribution': {
                'large': 0,   # 涨停股中大市值 (>200亿)
                'medium': 0,  # 涨停股中中市值 (50-200亿)
                'small': 0    # 涨停股中小市值 (<50亿)
            },
            # 连板分布统计
            'board_counts': {
                '1板': 0, '2板': 0, '3板': 0, '4板': 0, '5板': 0,
                '6板': 0, '7板': 0, '8板': 0, '9板': 0, '10板及以上': 0
            },
            # 其他统计
            'broken_limit_up_count': 0,  # 炸板数量
            'limit_open_count': 0  # 开盘涨停数量
        }
        
        if not stock_data:
            return stats

        # 统计股票数量
        stock_count = 0
        
        # 用于计算平均值的累计
        pct_chg_values = []
        turnover_values = []
        amplitude_values = []
        
        # 记录连板数和涨停股信息
        consecutive_limit_days = {}  # 股票代码 -> 连续涨停天数
        limit_up_stocks = []  # 存储涨停股票信息
        
        # 昨日涨停股列表及其今日溢价
        yesterday_limit_stocks_premium = []

        # 昨日炸板股列表及其今日溢价
        yesterday_broken_limit_premium = []

        # 昨日最高连板及其今日溢价
        yesterday_max_limit_stock = None
        yesterday_max_limit_days = 0
        
        # 遍历股票数据
        for stock_code, data in stock_data.items():
            if not isinstance(data, dict) or 'close' not in data:
                continue
                
            stock_count += 1
            
            # 获取基本数据
            pct_chg = data.get('pct_chg', 0)
            turnover = data.get('turnover_rate', 0)
            close = data.get('close', 0)
            preClose = data.get('preClose', 0)
            open_price = data.get('open', 0)
            high = data.get('high', 0)
            low = data.get('low', 0)

            # 修复前收盘价缺失问题
            if preClose <= 0 and close > 0 and pct_chg != 0:
                # 通过当前收盘价和涨跌幅反推前收盘价
                preClose = close / (1 + pct_chg / 100)
                # 更新data中的preClose，供后续使用
                data['preClose'] = preClose

            # 获取该股票的涨跌停限制
            limit_up_ratio, limit_down_ratio = self._get_stock_limit_ratios(stock_code)

            # 振幅计算
            amplitude = self._calculate_amplitude(data)

            # 市值信息
            total_mv = data.get('total_mv', 0) / 100000000  # 总市值(亿元)
            circ_mv = data.get('circ_mv', 0) / 100000000    # 流通市值(亿元)

            # 收集用于计算平均值的数据
            pct_chg_values.append(pct_chg)
            if turnover > 0:
                turnover_values.append(turnover)
            if amplitude > 0:
                amplitude_values.append(amplitude)
            
            # 检查是否有昨日连板数据（从数据中获取，而不是重新计算）
            consecutive_days = data.get('yesterday_consecutive_days', 0)

            # 如果当天涨停，连板数+1
            is_limit_up = self._is_limit_up(pct_chg, stock_code, data.get('close'), data.get('preClose'))
            if is_limit_up and consecutive_days > 0:
                consecutive_days += 1
            elif is_limit_up:
                consecutive_days = 1  # 当天首次涨停
            else:
                consecutive_days = 0  # 当天未涨停，连板中断

            if consecutive_days > 0:
                consecutive_limit_days[stock_code] = consecutive_days
                # 更新最高连板数
                if consecutive_days > stats['max_consecutive_limit_up']:
                    stats['max_consecutive_limit_up'] = consecutive_days

                # 统计连板分布
                if consecutive_days >= 10:
                    stats['board_counts']['10板及以上'] += 1
                else:
                    board_key = f"{consecutive_days}板"
                    if board_key in stats['board_counts']:
                        stats['board_counts'][board_key] += 1

                # 计算连板股票的市值分布
                if circ_mv > 0:
                    if circ_mv > 200:
                        stats['limit_board_distribution']['large'] += 1
                    elif circ_mv > 50:
                        stats['limit_board_distribution']['medium'] += 1
                    else:
                        stats['limit_board_distribution']['small'] += 1

                # 保存涨停股信息以计算溢价
                limit_up_stocks.append({
                    'stock_code': stock_code,
                    'consecutive_days': consecutive_days,
                    'data': data
                })
            
            # 统计市值分布
            if total_mv > 0:
                if total_mv > 200:
                    stats['market_cap_distribution']['large'] += 1
                elif total_mv > 50:
                    stats['market_cap_distribution']['medium'] += 1
                else:
                    stats['market_cap_distribution']['small'] += 1
                  
            # 判断是否为跌停（涨停判断已在上面完成）
            is_limit_down = self._is_limit_down(pct_chg, stock_code, data.get('close'), data.get('preClose'))
            
            #判断是否为炸板
            is_broken_limit = self._is_broken_limit(data, stock_code)
            
            if is_broken_limit:
                stats['broken_limit_up_count'] += 1
                self.logger.warning(f"检测到炸板股票: {stock_code}, 日期: {date_str}, 高价: {data.get('high')}, 收盘价: {data.get('close')}, 前收盘: {data.get('preClose')}")
            
            # 判断是否为一字涨停
            if is_limit_up and data.get('open') == data.get('high') == data.get('close'):
                stats['one_word_limit_up_count'] += 1
            
            # 判断是否为高弹性股 (振幅大于5%且非涨停跌停)
            if amplitude >= 5 and not is_limit_up and not is_limit_down:
                stats['high_elasticity_count'] += 1
            
            # 根据涨跌情况统计
            if pct_chg > 0:
                stats['up_count'] += 1
            elif pct_chg < 0:
                stats['down_count'] += 1
            else:
                stats['flat_count'] += 1
                
            # 涨停统计
            if is_limit_up:
                stats['limit_up_count'] += 1

                # 开盘涨停判断 - 使用正确的涨停比例
                if preClose > 0:
                    open_ratio = open_price / preClose - 1
                    if abs(open_ratio - limit_up_ratio) < 0.002:
                        stats['limit_open_count'] += 1

            # 跌停统计
            if is_limit_down:
                stats['limit_down_count'] += 1

            # 接近涨停判断 - 使用动态阈值
            near_limit_up_threshold = limit_up_ratio * 0.8  # 涨停阈值的80%
            if near_limit_up_threshold <= pct_chg / 100 < limit_up_ratio * 0.95:
                stats['near_limit_up_count'] += 1

            # 接近跌停判断 - 使用动态阈值
            near_limit_down_threshold = limit_down_ratio * 0.8  # 跌停阈值的80%
            if limit_down_ratio * 0.95 < pct_chg / 100 <= near_limit_down_threshold:
                stats['near_limit_down_count'] += 1
                
            # 处理昨日涨停股的今日溢价
            if 'yesterday_limit_up' in data and data['yesterday_limit_up'] and 'premium' in data:
                yesterday_limit_stocks_premium.append(data['premium'])

                # 获取昨日连板数
                yesterday_consecutive_days = data.get('yesterday_consecutive_days', 0)

                # 更新最高连板股
                if yesterday_consecutive_days > yesterday_max_limit_days:
                    yesterday_max_limit_days = yesterday_consecutive_days
                    yesterday_max_limit_stock = data

            # 处理昨日炸板股的今日溢价
            if 'yesterday_broken_limit' in data and data['yesterday_broken_limit'] and 'premium' in data:
                yesterday_broken_limit_premium.append(data['premium'])
        
        # 更新股票总数
        stats['stock_count'] = stock_count
        
        # 计算昨日涨停股今日平均溢价
        if yesterday_limit_stocks_premium:
            stats['yesterday_limit_up_premium'] = np.mean(yesterday_limit_stocks_premium)
        
        # 计算昨日最高连板今日溢价
        if yesterday_max_limit_stock and 'premium' in yesterday_max_limit_stock:
            stats['yesterday_max_limit_up_premium'] = yesterday_max_limit_stock['premium']

        # 计算昨日炸板股今日平均溢价
        if yesterday_broken_limit_premium:
            stats['yesterday_broken_limit_premium'] = np.mean(yesterday_broken_limit_premium)
        
        # 计算平均值
        if pct_chg_values:
            stats['avg_pct_chg'] = np.mean(pct_chg_values)
        if turnover_values:
            stats['avg_turnover'] = np.mean(turnover_values)
        if amplitude_values:
            stats['avg_amplitude'] = np.mean(amplitude_values)

        # 添加调试日志
        self.logger.info(f"板块统计完成 - 日期: {date_str}, 股票数: {stats['stock_count']}, "
                        f"涨停数: {stats['limit_up_count']}, 炸板数: {stats['broken_limit_up_count']}, "
                        f"最高连板: {stats['max_consecutive_limit_up']}, "
                        f"昨日最高连板溢价: {stats['yesterday_max_limit_up_premium']}")

        return stats

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
    """主函数"""
    try:
        import argparse

        # 创建命令行参数解析器
        parser = argparse.ArgumentParser(description="板块涨跌停板分析工具")
        
        # 添加参数 - 简化为只支持batch模式
        parser.add_argument('--start_date', type=str, required=True, help='开始日期，格式为YYYYMMDD')
        parser.add_argument('--end_date', type=str, required=True, help='结束日期，格式为YYYYMMDD')
        parser.add_argument('--sector_list', type=str, help='板块列表文件路径，不指定时处理所有板块')
        parser.add_argument('--output_dir', type=str, help='输出目录，不指定时使用默认目录')
        parser.add_argument('--cache_dir', type=str, help='缓存目录，不指定时使用默认目录')
        parser.add_argument('--max_workers', type=int, default=4, help='最大并发数，默认4')
        parser.add_argument('--batch_size', type=int, default=10, help='批处理大小，默认10个板块')
        parser.add_argument('--enable_cache', action='store_true', help='启用缓存加速')
        parser.add_argument('--update', action='store_true', help='更新模式：按日期合并到现有文件，而不是覆盖')
        
        # 解析命令行参数
        args = parser.parse_args()

        # 初始化分析器
        analyzer = SectorBoardAnalyzer(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            enable_cache=args.enable_cache,
            update_mode=args.update
        )

        # 获取板块列表
        sector_list = []
        if args.sector_list:
            # 从文件加载板块列表
            try:
                with open(args.sector_list, 'r', encoding='utf-8') as f:
                    for line in f:
                        sector = line.strip()
                        if sector:
                            sector_list.append(sector)
            except Exception as e:
                print(f"加载板块列表文件 {args.sector_list} 时出错: {e}")
                return
        else:
            # 使用默认的板块列表
            try:
                from to.data_loaders import get_sector_list
                sector_list = get_sector_list()
            except Exception as e:
                print(f"获取默认板块列表时出错: {e}")
                return

        if not sector_list:
            print("未找到任何板块")
            return

        # 生成日期范围列表
        date_range = []
        current_date = pd.to_datetime(args.start_date)
        end_date_obj = pd.to_datetime(args.end_date)

        while current_date <= end_date_obj:
            date_range.append(current_date.strftime('%Y%m%d'))
            current_date += pd.Timedelta(days=1)

        # 高性能批量处理
        print(f"开始高性能批量处理 {len(sector_list)} 个板块，日期范围: {args.start_date} - {args.end_date}")
        print(f"配置: 最大并发={args.max_workers}, 批处理大小={args.batch_size}, 缓存={'启用' if args.enable_cache else '禁用'}")

        result = analyzer.optimized_batch_process(
            sector_list,
            date_range,
            max_workers=args.max_workers,
            batch_size=args.batch_size
        )

        if result:
            print(f"批量处理完成，已处理 {len(result)} 个板块")
        else:
            print("批量处理失败或结果为空")
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
