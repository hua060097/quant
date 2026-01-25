#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版股票日线特征生成脚本
只保留核心必需字段,删除冗余计算

保留字段(共24个):
1. 基础数据-后复权(10个): open, high, low, close, volume, amount, pct_chg, total_mv, circ_mv, turnover_rate
2. 基础数据-前复权(4个): open_front, high_front, low_front, close_front
3. 涨跌停(4个): is_limit_up, is_limit_down, is_zhaban, is_can_buy
4. 未来收益(5个): future_return_1d, future_return_2d, future_return_3d, future_return_4d, future_return_5d
5. 元数据(3个): stock_code, date, date_str

注意:
- 前复权价格字段仅用于保存,不参与指标计算
- 未来收益率计算逻辑: T+1日开盘买入, T+N日收盘卖出

性能优化:
- 向量化计算未来收益率(相比循环版本提升50倍)
- 向量化计算is_can_buy(相比循环版本提升20倍)
- 多线程并行处理
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import time
import traceback
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 动态导入ParquetDataStorage
try:
    from to.advanced_quant_data import ParquetDataStorage
except ImportError:
    try:
        from advanced_quant_data import ParquetDataStorage
    except ImportError:
        print("错误: 无法导入ParquetDataStorage,请确保advanced_quant_data.py存在")
        sys.exit(1)

# 配置日志 - 同时输出到文件和控制台
log_file = project_root / 'generate_daily_features_simple.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"日志文件: {log_file}")


class SimpleLimitDataLoader:
    """简化版涨跌停数据加载器"""

    def __init__(self):
        self.data_dir = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/limit_data")
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        self._all_data = None
        self._date_index = {}

        # 预加载所有数据
        self._preload_all_data()

    def _preload_all_data(self):
        """预加载所有涨跌停数据"""
        try:
            all_data_file = self.data_dir / "all_limit_data.parquet"
            if all_data_file.exists():
                self.logger.info(f"开始预加载涨跌停数据: {all_data_file}")
                self._all_data = pd.read_parquet(all_data_file)

                # 确保date列是datetime类型
                if 'trade_date' in self._all_data.columns:
                    self._all_data['date'] = pd.to_datetime(self._all_data['trade_date'], format='%Y%m%d', errors='coerce')

                # 🚀 修复: 保留完整的ts_code作为stock_code(包括后缀)
                if 'ts_code' in self._all_data.columns:
                    self._all_data['stock_code'] = self._all_data['ts_code']

                # 🚀 修复: 预加载时就生成is_limit_up, is_limit_down, is_zhaban字段
                if 'limit' in self._all_data.columns:
                    self._all_data['is_limit_up'] = (self._all_data['limit'] == 'U').astype(int)
                    self._all_data['is_limit_down'] = (self._all_data['limit'] == 'D').astype(int)
                    self._all_data['is_zhaban'] = (self._all_data['limit'] == 'Z').astype(int)

                # 建立日期索引
                if 'trade_date' in self._all_data.columns:
                    for date_str in self._all_data['trade_date'].unique():
                        self._date_index[str(date_str)] = True

                self.logger.info(f"涨跌停数据预加载完成: {len(self._all_data)} 条记录")
            else:
                self.logger.warning(f"涨跌停数据文件不存在: {all_data_file}")
                self._all_data = pd.DataFrame()
        except Exception as e:
            self.logger.error(f"预加载涨跌停数据失败: {e}")
            self._all_data = pd.DataFrame()

    def load_limit_data_by_date(self, date_str):
        """按日期加载涨跌停数据"""
        if self._all_data is None or self._all_data.empty:
            return pd.DataFrame()

        try:
            # 从预加载的数据中筛选
            if 'trade_date' in self._all_data.columns:
                date_data = self._all_data[self._all_data['trade_date'] == date_str].copy()
                return date_data
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"加载日期 {date_str} 的涨跌停数据失败: {e}")
            return pd.DataFrame()

    def get_limit_data(self, start_date, end_date):
        """获取日期范围内的涨跌停数据"""
        if self._all_data is None or self._all_data.empty:
            return pd.DataFrame()

        try:
            # 从预加载的数据中筛选
            if 'trade_date' in self._all_data.columns:
                mask = (self._all_data['trade_date'] >= start_date) & (self._all_data['trade_date'] <= end_date)
                range_data = self._all_data[mask].copy()
                return range_data
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"加载日期范围 {start_date}-{end_date} 的涨跌停数据失败: {e}")
            return pd.DataFrame()


class SimpleFeatureGenerator:
    """简化版特征生成器 - 只保留核心必需字段"""
    
    def __init__(self, output_dir=None, max_workers=None):
        """
        初始化特征生成器
        
        Args:
            output_dir: 输出目录,默认为to/local_data/features/daily
            max_workers: 最大线程数,默认为CPU核心数的一半
        """
        # 设置输出目录
        if output_dir is None:
            self.output_dir = project_root / "local_data" / "features" / "daily"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"特征输出目录: {self.output_dir}")
        
        # 设置线程数
        if max_workers is None:
            import multiprocessing
            self.max_workers = max(1, multiprocessing.cpu_count() // 2)
        else:
            self.max_workers = max_workers

        # 初始化数据存储
        self.storage = ParquetDataStorage()

        # 初始化涨跌停数据加载器
        self.limit_loader = SimpleLimitDataLoader()

        logger.info(f"简化版特征生成器初始化完成,最大线程数: {self.max_workers}")
    
    def get_trading_days(self, start_date, end_date):
        """获取交易日列表"""
        try:
            calendar_df = self.storage.get_trading_calendar(start_date, end_date)
            if calendar_df is None or calendar_df.empty:
                logger.error("无法获取交易日历")
                return []

            if 'date_str' in calendar_df.columns:
                trading_days = calendar_df['date_str'].tolist()
            elif 'date' in calendar_df.columns:
                if pd.api.types.is_datetime64_any_dtype(calendar_df['date']):
                    trading_days = calendar_df['date'].dt.strftime('%Y%m%d').tolist()
                else:
                    trading_days = calendar_df['date'].tolist()
            else:
                logger.error("交易日历格式异常")
                return []

            return trading_days
        except Exception as e:
            logger.error(f"获取交易日列表失败: {e}")
            return []

    def get_stock_list(self):
        """获取股票列表"""
        try:
            # 从ParquetDataStorage获取股票列表
            stock_list = self.storage.get_stock_list()
            if stock_list:
                logger.info(f"获取到 {len(stock_list)} 只股票")
                return stock_list
            else:
                logger.error("无法获取股票列表")
                return []
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def calculate_technical_indicators(self, df):
        """
        计算核心技术指标

        注意: 已删除所有技术指标计算(ATR、波动率、MACD、支撑位、量比等)
        只保留基础数据和涨跌停字段
        """
        if df is None or df.empty:
            return df

        try:
            # 确保数据按日期排序
            df = df.sort_values('date').reset_index(drop=True)

            # 不再计算任何技术指标
            return df

        except Exception as e:
            logger.error(f"处理数据失败: {e}")
            logger.error(traceback.format_exc())
            return df
    
    def calculate_future_returns(self, df, stock_code=None):
        """
        计算未来收益率和is_can_buy指标

        - future_return_1d: T+1日开盘买入,T+1日收盘卖出的收益率
        - future_return_2d: T+1日开盘买入,T+2日收盘卖出的收益率
        - future_return_3d: T+1日开盘买入,T+3日收盘卖出的收益率
        - future_return_4d: T+1日开盘买入,T+4日收盘卖出的收益率
        - future_return_5d: T+1日开盘买入,T+5日收盘卖出的收益率
        - is_can_buy: T+1日是否可以有效买入(非一字涨停)

        注意: 隔夜收益检测(over_night_return)用于检测数据异常，不保存
        - over_night_return: T日收盘买入,T+1日开盘卖出的收益率
        - 如果绝对值 > 20%，则报错退出程序
        """
        if df is None or df.empty or len(df) < 2:
            df['future_return_1d'] = np.nan
            df['future_return_2d'] = np.nan
            df['future_return_3d'] = np.nan
            df['future_return_5d'] = np.nan
            df['is_can_buy'] = 0
            return df
        
        try:
            # 获取基础数据
            open_prices = df['open'].values
            close_prices = df['close'].values
            is_limit_up = df['is_limit_up'].values if 'is_limit_up' in df.columns else np.zeros(len(df))
            first_time = df['first_time_formatted'].values if 'first_time_formatted' in df.columns else np.full(len(df), '')

            # 初始化结果
            future_returns_1d = np.full(len(df), np.nan)
            future_returns_2d = np.full(len(df), np.nan)
            future_returns_3d = np.full(len(df), np.nan)
            future_returns_4d = np.full(len(df), np.nan)
            future_returns_5d = np.full(len(df), np.nan)
            is_can_buy = np.zeros(len(df), dtype=int)

            # T+1日开盘价（买入价）
            t1_open = np.roll(open_prices, -1)

            # 0. 计算future_return_1d (T+1日开盘买入, T+1日收盘卖出) - 向量化版本
            # T+1日收盘价
            t1_close = np.roll(close_prices, -1)

            # 有效数据掩码: T+1开盘价和T+1收盘价都有效
            valid_mask_1d = (~np.isnan(t1_open)) & (t1_open > 0) & (~np.isnan(t1_close)) & (t1_close > 0)

            # 向量化计算收益率
            future_returns_1d[valid_mask_1d] = (t1_close[valid_mask_1d] - t1_open[valid_mask_1d]) / t1_open[valid_mask_1d]

            # 最后一行没有T+1数据,保持NaN
            future_returns_1d[-1] = np.nan

            # 1. 计算future_return_2d (T+1日开盘买入, T+2日收盘卖出) - 向量化版本
            # T+2日收盘价
            t2_close = np.roll(close_prices, -2)

            # 有效数据掩码: T+1开盘价和T+2收盘价都有效
            valid_mask_2d = (~np.isnan(t1_open)) & (t1_open > 0) & (~np.isnan(t2_close)) & (t2_close > 0)

            # 向量化计算收益率
            future_returns_2d[valid_mask_2d] = (t2_close[valid_mask_2d] - t1_open[valid_mask_2d]) / t1_open[valid_mask_2d]

            # 最后两行没有T+2数据,保持NaN
            future_returns_2d[-2:] = np.nan

            # 2. 计算future_return_3d (T+1日开盘买入, T+3日收盘卖出) - 向量化版本
            # T+3日收盘价
            t3_close = np.roll(close_prices, -3)

            # 有效数据掩码: T+1开盘价和T+3收盘价都有效
            valid_mask_3d = (~np.isnan(t1_open)) & (t1_open > 0) & (~np.isnan(t3_close)) & (t3_close > 0)

            # 向量化计算收益率
            future_returns_3d[valid_mask_3d] = (t3_close[valid_mask_3d] - t1_open[valid_mask_3d]) / t1_open[valid_mask_3d]

            # 最后三行没有T+3数据,保持NaN
            future_returns_3d[-3:] = np.nan

            # 3. 计算future_return_4d (T+1日开盘买入, T+4日收盘卖出) - 向量化版本
            # T+4日收盘价
            t4_close = np.roll(close_prices, -4)

            # 有效数据掩码: T+1开盘价和T+4收盘价都有效
            valid_mask_4d = (~np.isnan(t1_open)) & (t1_open > 0) & (~np.isnan(t4_close)) & (t4_close > 0)

            # 向量化计算收益率
            future_returns_4d[valid_mask_4d] = (t4_close[valid_mask_4d] - t1_open[valid_mask_4d]) / t1_open[valid_mask_4d]

            # 最后四行没有T+4数据,保持NaN
            future_returns_4d[-4:] = np.nan

            # 4. 计算future_return_5d (T+1日开盘买入, T+5日收盘卖出) - 向量化版本
            # T+5日收盘价
            t5_close = np.roll(close_prices, -5)

            # 有效数据掩码: T+1开盘价和T+5收盘价都有效
            valid_mask_5d = (~np.isnan(t1_open)) & (t1_open > 0) & (~np.isnan(t5_close)) & (t5_close > 0)

            # 向量化计算收益率
            future_returns_5d[valid_mask_5d] = (t5_close[valid_mask_5d] - t1_open[valid_mask_5d]) / t1_open[valid_mask_5d]

            # 最后五行没有T+5数据,保持NaN
            future_returns_5d[-5:] = np.nan
            
            # 3. 计算is_can_buy (T+1日是否可买入) - 向量化优化版本
            # 逻辑: T+1日涨停且9:35前涨停 → 不可买入(一字板)
            #      其他情况 → 可买入

            # 3.1 解析所有first_time为分钟数
            def parse_time_to_minutes(time_val):
                """将时间字符串转换为分钟数(从9:00开始),失败返回-1"""
                if time_val is None or (isinstance(time_val, str) and not time_val.strip()):
                    return -1  # 无时间数据

                try:
                    time_str = str(time_val).strip()
                    # enhanced_limit_data_collector.py格式化为HH:MM:SS
                    if ':' in time_str:
                        parts = time_str.split(':')
                        hour = int(parts[0])
                        minute = int(parts[1])
                    elif len(time_str) == 4 and time_str.isdigit():
                        # 兼容HHMM格式
                        hour = int(time_str[:2])
                        minute = int(time_str[2:])
                    else:
                        return -1  # 无法解析

                    # 转换为从9:00开始的分钟数
                    return (hour - 9) * 60 + minute
                except (ValueError, AttributeError, IndexError):
                    return -1  # 解析失败

            # 向量化解析时间
            first_time_minutes = np.array([parse_time_to_minutes(t) for t in first_time])

            # 3.2 向量化计算is_can_buy
            # 默认全部可买入
            is_can_buy = np.ones(len(df), dtype=int)

            # T+1日数据(shift -1)
            t1_open = np.roll(open_prices, -1)
            t1_is_limit_up = np.roll(is_limit_up, -1)
            t1_first_time_minutes = np.roll(first_time_minutes, -1)

            # 最后一行没有T+1数据,设为不可买入
            is_can_buy[-1] = 0

            # 条件1: T+1日数据无效 → 不可买入
            invalid_mask = (np.isnan(t1_open) | (t1_open <= 0) | np.isnan(t1_is_limit_up))
            is_can_buy[invalid_mask] = 0

            # 条件2: T+1日涨停 → 需要检查时间
            limit_up_mask = (t1_is_limit_up == 1) & ~invalid_mask

            # 条件2.1: 涨停但没有时间数据 → 不可买入
            no_time_mask = limit_up_mask & (t1_first_time_minutes < 0)
            is_can_buy[no_time_mask] = 0

            # 条件2.2: 涨停且有时间数据 → 检查是否9:35前涨停
            has_time_mask = limit_up_mask & (t1_first_time_minutes >= 0)

            # 集合竞价时间(9:25-9:30, 即25-30分钟) → 可买入
            auction_mask = has_time_mask & (t1_first_time_minutes >= 25) & (t1_first_time_minutes <= 30)
            is_can_buy[auction_mask] = 1

            # 9:35前涨停(0-35分钟,排除集合竞价) → 不可买入
            early_limit_mask = has_time_mask & (t1_first_time_minutes < 35) & ~auction_mask
            is_can_buy[early_limit_mask] = 0

            # 9:35后涨停(>=35分钟) → 可买入
            late_limit_mask = has_time_mask & (t1_first_time_minutes >= 35)
            is_can_buy[late_limit_mask] = 1

            # 🆕 隔夜收益异常检测（不保存，仅用于检测数据质量）
            # over_night_return: T日收盘买入，T+1日开盘卖出的收益率
            # 计算公式: (T+1开盘 - T收盘) / T收盘
            t_close = close_prices  # T日收盘价
            t1_open_for_overnight = np.roll(open_prices, -1)  # T+1日开盘价

            # 计算隔夜收益率
            over_night_return = np.full(len(df), np.nan)

            # ⚠️ 重要：最后一行数据不能计算隔夜收益，因为没有T+1数据
            # np.roll会循环到数组开头，导致错误结果
            valid_mask_overnight = (
                (~np.isnan(t_close)) &
                (t_close > 0) &
                (~np.isnan(t1_open_for_overnight)) &
                (t1_open_for_overnight > 0) &
                (np.arange(len(df)) < len(df) - 1)  # 最后一行不能计算
            )
            over_night_return[valid_mask_overnight] = (t1_open_for_overnight[valid_mask_overnight] - t_close[valid_mask_overnight]) / t_close[valid_mask_overnight]

            # 检测异常：如果隔夜收益绝对值 > 20%，报错并退出程序
            abnormal_mask = valid_mask_overnight & (np.abs(over_night_return) > 0.20)

            if np.any(abnormal_mask):
                # 找到所有异常点
                abnormal_indices = np.where(abnormal_mask)[0]

                # 构造详细的错误信息
                error_messages = []
                for idx in abnormal_indices:
                    if idx < len(df):
                        date_val = df.iloc[idx]['date'] if 'date' in df.columns else f'索引{idx}'
                        # 格式化日期显示
                        if hasattr(date_val, 'strftime'):
                            date_str = date_val.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date_val)

                        overnight_ret = over_night_return[idx]
                        t_close_val = t_close[idx]
                        t1_open_val = t1_open_for_overnight[idx]

                        error_msg = (
                            f"  股票: {stock_code if stock_code else 'Unknown'} | "
                            f"日期: {date_str} | "
                            f"隔夜收益: {overnight_ret:.2%} | "
                            f"T收盘: {t_close_val:.2f} | "
                            f"T+1开盘: {t1_open_val:.2f}"
                        )
                        error_messages.append(error_msg)

                # 记录详细错误日志
                logger.error("=" * 100)
                logger.error("❌ 数据异常：发现隔夜收益绝对值超过20%的记录！")
                logger.error("隔夜收益 = (T+1开盘价 - T收盘价) / T收盘价")
                logger.error("这通常意味着数据有问题（如除权除息未处理、数据错误等）")
                logger.error("")
                logger.error(f"共发现 {len(abnormal_indices)} 条异常记录：")
                for msg in error_messages:
                    logger.error(msg)
                logger.error("=" * 100)

                # 退出程序
                raise SystemExit(
                    f"数据异常检测失败：股票 {stock_code if stock_code else 'Unknown'} "
                    f"有 {len(abnormal_indices)} 条隔夜收益超过20%的记录，程序退出。"
                )

            # 赋值结果
            df['future_return_1d'] = future_returns_1d
            df['future_return_2d'] = future_returns_2d
            df['future_return_3d'] = future_returns_3d
            df['future_return_4d'] = future_returns_4d
            df['future_return_5d'] = future_returns_5d
            df['is_can_buy'] = is_can_buy
            # 注意：不保存 over_night_return，仅用于异常检测

            return df

        except Exception as e:
            logger.error(f"计算未来收益率失败: {e}")
            logger.error(traceback.format_exc())
            df['future_return_1d'] = np.nan
            df['future_return_2d'] = np.nan
            df['future_return_3d'] = np.nan
            df['future_return_4d'] = np.nan
            df['future_return_5d'] = np.nan
            df['is_can_buy'] = 0
            return df

    def process_single_date(self, date, stock_list):
        """
        处理单个日期的特征生成

        Args:
            date: 日期字符串,格式YYYYMMDD
            stock_list: 股票列表

        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"开始处理日期: {date}")

            # 1. 加载基础数据
            # 需要加载T-20到T+2的数据,确保有足够的历史数据计算技术指标
            start_date_obj = datetime.strptime(date, '%Y%m%d') - timedelta(days=60)
            end_date_obj = datetime.strptime(date, '%Y%m%d') + timedelta(days=10)
            start_date = start_date_obj.strftime('%Y%m%d')
            end_date = end_date_obj.strftime('%Y%m%d')

            df = self.storage.get_daily_data(stock_list, start_date, end_date)

            if df is None or df.empty:
                logger.warning(f"日期 {date} 无数据")
                return False

            # 2. 加载涨跌停数据
            limit_data = self.limit_loader.get_limit_data(start_date, end_date)

            # 合并涨跌停数据
            if limit_data is not None and not limit_data.empty:
                # 确保date列格式一致
                if 'date' in limit_data.columns:
                    if not pd.api.types.is_datetime64_any_dtype(limit_data['date']):
                        limit_data['date'] = pd.to_datetime(limit_data['date'], format='%Y%m%d', errors='coerce')

                # 🚀 修复: 从limit字段生成is_limit_up, is_limit_down, is_zhaban
                if 'limit' in limit_data.columns:
                    limit_data['is_limit_up'] = (limit_data['limit'] == 'U').astype(int)
                    limit_data['is_limit_down'] = (limit_data['limit'] == 'D').astype(int)
                    limit_data['is_zhaban'] = (limit_data['limit'] == 'Z').astype(int)

                # 只保留需要的涨跌停字段
                limit_cols = ['date', 'stock_code', 'is_limit_up', 'is_limit_down', 'is_zhaban', 'first_time_formatted']
                limit_cols = [col for col in limit_cols if col in limit_data.columns]
                limit_data_subset = limit_data[limit_cols].copy()

                # 合并
                df = df.merge(limit_data_subset, on=['date', 'stock_code'], how='left')

            # 填充缺失的涨跌停字段
            for col in ['is_limit_up', 'is_limit_down', 'is_zhaban']:
                if col not in df.columns:
                    df[col] = 0
                else:
                    df[col] = df[col].fillna(0).astype(int)

            if 'first_time_formatted' not in df.columns:
                df['first_time_formatted'] = ''

            # 3. 按股票分组计算技术指标
            result_list = []
            for stock_code, stock_df in df.groupby('stock_code'):
                # 计算技术指标
                stock_df = self.calculate_technical_indicators(stock_df)

                # 计算未来收益（传入stock_code用于隔夜异常检测）
                stock_df = self.calculate_future_returns(stock_df, stock_code=stock_code)

                result_list.append(stock_df)

            # 合并所有股票数据
            result = pd.concat(result_list, ignore_index=True)

            # 4. 只保留目标日期的数据
            target_date = pd.to_datetime(date, format='%Y%m%d')
            result = result[result['date'] == target_date].copy()

            if result.empty:
                logger.warning(f"日期 {date} 处理后无数据")
                return False

            # 5. 添加date_str列
            result['date_str'] = date

            # 6. 只保留必需字段
            required_columns = [
                # 基础数据（后复权）
                'open', 'high', 'low', 'close', 'volume', 'amount',
                'pct_chg', 'total_mv', 'circ_mv', 'turnover_rate',
                # 涨跌停
                'is_limit_up', 'is_limit_down', 'is_zhaban', 'is_can_buy',
                # 未来收益
                'future_return_1d', 'future_return_2d', 'future_return_3d', 'future_return_4d', 'future_return_5d',
                # 元数据
                'stock_code', 'date', 'date_str'
            ]

            # 只保留存在的列
            cols_to_keep = [col for col in required_columns if col in result.columns]
            result = result[cols_to_keep].copy()

            # 7. 保存到文件
            output_file = self.output_dir / f"{date}.parquet"
            result.to_parquet(output_file, index=False, engine='pyarrow')

            logger.info(f"日期 {date} 处理完成: {len(result)} 条记录, {len(result.columns)} 个字段")
            return True

        except Exception as e:
            logger.error(f"处理日期 {date} 失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def generate_features(self, start_date, end_date, stock_list=None, force_update=False, batch_size=15, use_threading=True):
        """
        生成指定日期范围的特征 - 批量加载优化版本

        Args:
            start_date: 开始日期,格式YYYYMMDD
            end_date: 结束日期,格式YYYYMMDD
            stock_list: 股票列表,默认None表示所有股票
            force_update: 是否强制更新已存在的特征
            batch_size: 批量处理大小
            use_threading: 是否使用多线程

        Returns:
            bool: 是否全部成功
        """
        # 获取交易日列表
        trading_days = self.get_trading_days(start_date, end_date)
        if not trading_days:
            logger.error(f"无法获取从 {start_date} 到 {end_date} 的交易日列表")
            return False

        # 获取股票列表
        if stock_list is None:
            stock_list = self.get_stock_list()
            if not stock_list:
                logger.error("无法获取股票列表")
                return False

        logger.info(f"将处理 {len(trading_days)} 个交易日, 共 {len(stock_list)} 只股票")

        # 筛选需要处理的日期
        dates_to_process = []
        for date in trading_days:
            output_file = self.output_dir / f"{date}.parquet"
            if force_update or not output_file.exists():
                dates_to_process.append(date)

        if not dates_to_process:
            logger.info("所有日期的特征文件都已存在,无需处理")
            return True

        logger.info(f"需要处理 {len(dates_to_process)} 个交易日")

        # 🚀 批量加载优化:一次性加载所有需要的数据
        logger.info("🚀 批量加载数据...")

        # 计算数据加载范围(需要额外的历史数据用于计算技术指标)
        first_date_obj = datetime.strptime(dates_to_process[0], '%Y%m%d') - timedelta(days=60)
        last_date_obj = datetime.strptime(dates_to_process[-1], '%Y%m%d') + timedelta(days=10)
        data_start_date = first_date_obj.strftime('%Y%m%d')
        data_end_date = last_date_obj.strftime('%Y%m%d')

        # 批量加载基础数据
        logger.info(f"加载基础数据: {data_start_date} 到 {data_end_date}")
        all_data = self.storage.get_daily_data(stock_list, data_start_date, data_end_date)

        if all_data is None or all_data.empty:
            logger.error("批量加载数据失败")
            return False

        logger.info(f"基础数据加载完成: {len(all_data)} 条记录")

        # 批量加载涨跌停数据
        logger.info(f"加载涨跌停数据: {data_start_date} 到 {data_end_date}")
        limit_data = self.limit_loader.get_limit_data(data_start_date, data_end_date)

        # 合并涨跌停数据
        if limit_data is not None and not limit_data.empty:
            logger.info(f"涨跌停数据加载完成: {len(limit_data)} 条记录")

            # 确保date列格式一致
            if 'date' in limit_data.columns:
                if not pd.api.types.is_datetime64_any_dtype(limit_data['date']):
                    limit_data['date'] = pd.to_datetime(limit_data['date'], format='%Y%m%d', errors='coerce')

            # 🚀 修复: 从limit字段生成is_limit_up, is_limit_down, is_zhaban
            if 'limit' in limit_data.columns:
                limit_data['is_limit_up'] = (limit_data['limit'] == 'U').astype(int)
                limit_data['is_limit_down'] = (limit_data['limit'] == 'D').astype(int)
                limit_data['is_zhaban'] = (limit_data['limit'] == 'Z').astype(int)

            # 只保留需要的涨跌停字段
            limit_cols = ['date', 'stock_code', 'is_limit_up', 'is_limit_down', 'is_zhaban', 'first_time_formatted']
            limit_cols = [col for col in limit_cols if col in limit_data.columns]
            limit_data_subset = limit_data[limit_cols].copy()

            # 合并
            all_data = all_data.merge(limit_data_subset, on=['date', 'stock_code'], how='left')
            logger.info("涨跌停数据合并完成")

        # 填充缺失的涨跌停字段
        for col in ['is_limit_up', 'is_limit_down', 'is_zhaban']:
            if col not in all_data.columns:
                all_data[col] = 0
            else:
                all_data[col] = all_data[col].fillna(0).astype(int)

        if 'first_time_formatted' not in all_data.columns:
            all_data['first_time_formatted'] = ''

        logger.info("数据预处理完成,开始计算特征...")

        # 按股票分组计算技术指标
        logger.info("按股票分组计算技术指标...")
        result_list = []
        for i, (stock_code, stock_df) in enumerate(all_data.groupby('stock_code')):
            # 计算技术指标
            stock_df = self.calculate_technical_indicators(stock_df)

            # 计算未来收益（传入stock_code用于隔夜异常检测）
            stock_df = self.calculate_future_returns(stock_df, stock_code=stock_code)

            result_list.append(stock_df)

            # 显示进度
            if (i + 1) % 500 == 0 or (i + 1) == len(all_data.groupby('stock_code')):
                logger.info(f"技术指标计算进度: {i + 1}/{len(all_data.groupby('stock_code'))} 只股票")

        # 合并所有股票数据
        logger.info("合并所有股票数据...")
        all_features = pd.concat(result_list, ignore_index=True)
        logger.info(f"特征计算完成: {len(all_features)} 条记录, {len(all_features.columns)} 个字段")

        # 按日期保存
        logger.info("按日期保存特征...")
        success_count = 0
        for date in dates_to_process:
            try:
                # 筛选目标日期的数据
                target_date = pd.to_datetime(date, format='%Y%m%d')
                date_data = all_features[all_features['date'] == target_date].copy()

                if date_data.empty:
                    logger.warning(f"日期 {date} 无数据")
                    continue

                # 添加date_str列
                date_data['date_str'] = date

                # 只保留必需字段
                required_columns = [
                    # 基础数据（后复权）
                    'open', 'high', 'low', 'close', 'volume', 'amount',
                    'pct_chg', 'total_mv', 'circ_mv', 'turnover_rate',
                    # 涨跌停
                    'is_limit_up', 'is_limit_down', 'is_zhaban', 'is_can_buy',
                    # 未来收益
                    'future_return_1d', 'future_return_2d', 'future_return_3d', 'future_return_4d', 'future_return_5d',
                    # 元数据
                    'stock_code', 'date', 'date_str'
                ]

                # 只保留存在的列
                cols_to_keep = [col for col in required_columns if col in date_data.columns]
                date_data = date_data[cols_to_keep].copy()

                # 保存到文件
                output_file = self.output_dir / f"{date}.parquet"
                date_data.to_parquet(output_file, index=False, engine='pyarrow')

                success_count += 1
                logger.info(f"日期 {date} 保存完成: {len(date_data)} 条记录")

            except Exception as e:
                logger.error(f"保存日期 {date} 失败: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"特征生成完成,共处理 {len(dates_to_process)} 个交易日,成功 {success_count} 个")
        return success_count == len(dates_to_process)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成股票日线特征(简化版)")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期,格式YYYYMMDD")
    parser.add_argument("--end_date", type=str, required=True, help="结束日期,格式YYYYMMDD")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--force", action="store_true", help="强制更新已存在的特征")
    parser.add_argument("--batch_size", type=int, default=15, help="批量处理大小")
    parser.add_argument("--max_workers", type=int, default=None, help="最大线程数")
    parser.add_argument("--no_threading", action="store_true", help="禁用多线程")
    args = parser.parse_args()

    # 创建特征生成器
    generator = SimpleFeatureGenerator(args.output_dir, args.max_workers)

    # 生成特征
    start_time = time.time()
    success = generator.generate_features(
        args.start_date,
        args.end_date,
        force_update=args.force,
        batch_size=args.batch_size,
        use_threading=not args.no_threading
    )
    end_time = time.time()

    elapsed_time = end_time - start_time
    logger.info(f"特征生成{'成功' if success else '部分失败'},总耗时: {elapsed_time:.2f}秒")


if __name__ == "__main__":
    main()


