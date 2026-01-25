#!/usr/bin/env python3
"""
优化版分钟级回测系统 - 基于策略的架构
解决之前犯过的错误：
1. 时间循环vs TopK循环的架构问题
2. 状态管理混乱问题
3. 数据重复加载问题
4. 变量作用域错误问题
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
import logging
import os
import pyarrow.parquet as pq
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """策略类型枚举"""
    TOP_1 = 1
    TOP_2 = 2
    TOP_3 = 3
    TOP_5 = 5

@dataclass
class StrategyConfig:
    """策略配置类"""
    strategy_type: StrategyType
    stock_count: int
    holding_days: int = 3
    initial_capital: float = 1000000.0
    max_position_ratio: float = 0.25  # 单只股票最大仓位比例（用于多股票策略的资金分配）

@dataclass
class Position:
    """持仓记录类 - 解决之前字典结构不一致的问题"""
    stock_code: str
    buy_date: datetime.date
    buy_datetime: datetime
    buy_price: float
    quantity: int
    predicted_return: float
    actual_return: float = 0.0
    sell_price: float = 0.0
    sell_datetime: Optional[datetime] = None
    holding_end_date: Optional[datetime.date] = None
    entry_reason: str = "normal"
    exit_reason: str = ""

class BacktestStrategy:
    """单个策略的回测类 - 完全独立，避免状态干扰"""

    def __init__(self, config: StrategyConfig, prediction_data: pd.DataFrame, minute_data_path: str):
        self.config = config
        self.minute_data_path = minute_data_path

        # 标准化预测数据的日期格式 - 统一为datetime.date
        prediction_data = prediction_data.copy()
        if 'date' in prediction_data.columns:
            # 将pandas Timestamp转换为datetime.date
            prediction_data['date'] = pd.to_datetime(prediction_data['date']).dt.date
            logger.info(f"📅 标准化日期格式: pandas Timestamp -> datetime.date")

        self.prediction_data = prediction_data

        # 策略状态 - 独立管理
        self.positions: List[Position] = []
        self.cash: float = config.initial_capital
        self.current_holdings: Dict[str, Position] = {}

        # 添加买入周期状态跟踪
        self.last_buy_date: Optional[datetime.date] = None
        self.current_buy_cycle_date: Optional[datetime.date] = None
        self.last_position_end_date: Optional[datetime.date] = None  # 上一轮持仓到期日（下一轮的T日）
        self.is_holding: bool = False
        self.current_holding_end_date: Optional[datetime.date] = None

        # 正确的T+1逻辑：当前轮次的T日状态
        self.current_t_date: Optional[datetime.date] = None  # 当前轮次的T日

        # 回测结果
        self.trades: List[Dict] = []
        self.daily_portfolio_values: List[float] = [config.initial_capital]
        self.daily_returns: List[float] = [0.0]
        self.trading_dates_list: List[datetime.date] = []  # 保存交易日期列表，用于绘图
        self.skipped_buy_attempts: List[Dict] = []  # 保存被跳过的买入尝试（开盘风控等），用于绘图

        # 数据缓存
        self.minute_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.required_stocks_by_date: Dict[str, List[str]] = {}  # 按日期保存每日Top-K股票，按预测值排序

        logger.info(f"🚀 初始化策略: Top-{config.stock_count}, 初始资金: {config.initial_capital:,.0f}")

    def pre_analyze_required_stocks(self, max_k: int = 5) -> None:
        """
        预分析需要加载的股票 - 解决96.5%数据被错误过滤的问题
        """
        logger.info("🔍 开始智能分析候选股票...")

        self.required_stocks_by_date = {}  # 按日期保存每日Top-K股票
        trading_dates = sorted(self.prediction_data['date'].unique())

        for date in trading_dates:
            daily_predictions = self.prediction_data[self.prediction_data['date'] == date]
            logger.info(f"📅 {date}: 日线数据 {len(daily_predictions)} 条记录")
            filtered_stocks = []
            filtering_stats = {
                'total': len(daily_predictions),
                'is_can_buy_filtered': 0,
                'exchange_filtered': 0,
                'st_filtered': 0,
                'price_filtered': 0,
                'liquidity_filtered': 0,
                'limit_filtered': 0,
                'zhaban_filtered': 0,
                'passed': 0
            }

            for _, stock in daily_predictions.iterrows():
                stock_code = stock['stock_code']

                # 修复流动性过滤逻辑
                # if stock['is_can_buy'] != 1:
                #     filtering_stats['is_can_buy_filtered'] += 1
                #     continue
                if stock_code.startswith(('8', '4')):  # 北交所
                    filtering_stats['exchange_filtered'] += 1
                    continue
                if any(x in stock_code for x in ['ST', 'st', '*ST', '*st']):
                    filtering_stats['st_filtered'] += 1
                    continue
                if stock['close'] <= 1.0 or stock['close'] >= 10000:
                    filtering_stats['price_filtered'] += 1
                    continue
                # 使用成交额而不是成交量
                if 'amount' in stock and stock['amount'] < 10000000:  # 1000万
                    filtering_stats['liquidity_filtered'] += 1
                    continue
                elif 'amount' not in stock and stock['volume'] < 1000000:  # 回退到成交量
                    filtering_stats['liquidity_filtered'] += 1
                    continue
                if stock['is_limit_up'] == 1 or stock['is_limit_down'] == 1:
                    filtering_stats['limit_filtered'] += 1
                    # 🔍 添加调试日志：记录被过滤的涨停跌停股票
                    logger.info(f"🚫 涨跌停过滤: {stock['stock_code']} is_limit_up={stock.get('is_limit_up', 'N/A')}, is_limit_down={stock.get('is_limit_down', 'N/A')}, pct_chg={stock.get('pct_chg', 'N/A')}")
                    continue
                if stock['is_zhaban'] == 1:
                    filtering_stats['zhaban_filtered'] += 1
                    continue

                filtered_stocks.append(stock)
                filtering_stats['passed'] += 1
                # 🔍 添加调试日志：记录通过过滤的股票关键信息
                if stock['stock_code'] == '603778.SH':  # 专门跟踪问题股票
                    logger.info(f"✅ 603778.SH通过过滤: is_limit_up={stock.get('is_limit_up', 'N/A')}, is_limit_down={stock.get('is_limit_down', 'N/A')}, is_zhaban={stock.get('is_zhaban', 'N/A')}, pct_chg={stock.get('pct_chg', 'N/A')}, predicted_return={stock.get('predicted_return', 'N/A')}")

            # 记录过滤统计信息
            logger.info(f"🔍 {date} 过滤统计:")
            logger.info(f"   总候选: {filtering_stats['total']}只")
            logger.info(f"   可买入过滤: {filtering_stats['is_can_buy_filtered']}只")
            logger.info(f"   交易所过滤: {filtering_stats['exchange_filtered']}只")
            logger.info(f"   ST股过滤: {filtering_stats['st_filtered']}只")
            logger.info(f"   价格过滤: {filtering_stats['price_filtered']}只")
            logger.info(f"   流动性过滤: {filtering_stats['liquidity_filtered']}只")
            logger.info(f"   涨跌停过滤: {filtering_stats['limit_filtered']}只")
            logger.info(f"   炸板过滤: {filtering_stats['zhaban_filtered']}只")
            logger.info(f"   通过过滤: {filtering_stats['passed']}只")

            if filtered_stocks:
                # 按预测收益率排序，取前max_k名
                filtered_df = pd.DataFrame(filtered_stocks)
                filtered_sorted = filtered_df.sort_values('predicted_return', ascending=False)
                top_candidates = filtered_sorted.head(max_k)

                logger.info(f"🏆 {date} Top-{max_k}候选股票:")
                for i, (_, candidate) in enumerate(top_candidates.iterrows(), 1):
                    logger.info(f"   {i}. {candidate['stock_code']}: 预测收益={candidate['predicted_return']:.4f}, 收盘价={candidate['close']:.2f}")

                # 按日期保存Top-K股票列表，保持预测值从高到低的排序
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                self.required_stocks_by_date[date_str] = top_candidates['stock_code'].tolist()

                logger.info(f"💾 {date_str} 保存Top-{max_k}股票: {len(self.required_stocks_by_date[date_str])}只")
            else:
                logger.warning(f"⚠️ {date}: 无通过过滤的候选股票")
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                self.required_stocks_by_date[date_str] = []

        total_stocks = len(self.prediction_data['stock_code'].unique())
        total_top_stocks = sum(len(stocks) for stocks in self.required_stocks_by_date.values())
        total_dates = len(self.required_stocks_by_date)

        logger.info(f"🎯 智能分析完成:")
        logger.info(f"   总股票数: {total_stocks}")
        logger.info(f"   分析日期数: {total_dates}天")
        logger.info(f"   Top-K候选总数: {total_top_stocks}只")

        # 检查关键日期和股票
        for check_date, check_code in [('2025-07-11', '002809.SZ')]:
            if check_date in self.required_stocks_by_date:
                if check_code in self.required_stocks_by_date[check_date]:
                    logger.info(f"✅ {check_code} 已包含在{check_date}的Top-K候选股票中")
                else:
                    logger.warning(f"❌ {check_code} 未包含在{check_date}的Top-K候选股票中")
                    logger.info(f"   {check_date}候选股票: {self.required_stocks_by_date[check_date]}")
            else:
                logger.warning(f"❌ 未找到{check_date}的候选股票数据")

    def load_minute_data_on_demand(self, date_str: str, target_stocks: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        优化的按需加载5分钟数据 - 支持指定股票列表的精准加载
        """
        # 统一缓存Key策略：只按日期缓存，不按股票列表缓存
        cache_key = date_str
        if cache_key in self.minute_data_cache:
            cached_data = self.minute_data_cache[cache_key]
            # 如果指定了目标股票，检查缓存中是否包含这些股票
            if target_stocks:
                # 检查所有目标股票是否都在缓存中
                found_stocks = [stock for stock in target_stocks if stock in cached_data and not cached_data[stock].empty]
                missing_stocks = [stock for stock in target_stocks if stock not in found_stocks]

                if missing_stocks:
                    logger.info(f"🔍 {date_str}: 缓存中缺少{len(missing_stocks)}只目标股票，重新加载: {missing_stocks[:3]}...")
                    logger.debug(f"   缓存中已有: {list(cached_data.keys())[:5]}")
                    # 重新加载所有目标股票（避免复杂的数据合并逻辑）
                    return self._reload_target_stocks(date_str, target_stocks)
                else:
                    # 所有目标股票都在缓存中
                    filtered_data = {code: cached_data[code] for code in target_stocks if code in cached_data}
                    logger.info(f"📋 {date_str}: 使用缓存数据，共{len(cached_data)}只股票，返回{len(filtered_data)}只目标股票")
                    return filtered_data
            else:
                logger.info(f"📋 {date_str}: 使用缓存数据，{len(cached_data)}只股票")
                return cached_data

        try:
            logger.info(f"📂 开始加载{date_str}的5分钟数据...")
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            year_month = target_date.strftime('%Y%m')
            month_dir = os.path.join(self.minute_data_path, year_month)
            minute_file = os.path.join(month_dir, "minute_data.parquet")

            normalized_path = minute_file.replace(os.sep, '/')
            if not os.path.exists(minute_file):
                logger.warning(f"❌ 未找到{date_str}的数据文件: {normalized_path}")
                return {}

            logger.info(f"📁 数据文件路径: {normalized_path}")
            logger.info(f"✅ 文件存在，大小: {os.path.getsize(minute_file)/(1024*1024):.1f}MB")

            from datetime import datetime as dt
            start_time = dt.strptime(f'{date_str} 00:00:00', '%Y-%m-%d %H:%M:%S')
            end_time = dt.strptime(f'{date_str} 23:59:59', '%Y-%m-%d %H:%M:%S')

            # 策略选择：如果指定了目标股票且数量合理，使用精准加载
            if target_stocks and len(target_stocks) <= 20:
                return self._load_specific_stocks(date_str, minute_file, start_time, end_time, target_stocks)
            else:
                return self._load_all_stocks_optimized(date_str, minute_file, start_time, end_time)

        except Exception as e:
            logger.error(f"加载{date_str}分钟数据失败: {e}")
            return {}

    def _load_specific_stocks(self, date_str: str, minute_file: str, start_time: datetime,
                             end_time: datetime, target_stocks: List[str]) -> Dict[str, pd.DataFrame]:
        """
        精准加载指定的股票数据 - 避免加载不必要的数据
        """
        logger.info(f"🎯 精准加载{len(target_stocks)}只目标股票的5分钟数据: {target_stocks[:5]}...")

        try:
            # 组合过滤器：日期 + 指定股票列表
            filters = [
                ('datetime', '>=', start_time),
                ('datetime', '<=', end_time),
                ('stock_code', 'in', target_stocks)
            ]

            # 只读取必要的列，减少内存占用
            columns = ['datetime', 'stock_code', 'open', 'close', 'high', 'low', 'volume']

            logger.info(f"🔍 使用精准过滤器加载指定股票数据...")
            table = pq.read_table(minute_file, filters=filters, columns=columns, memory_map=True)
            df = table.to_pandas()

            logger.info(f"✅ 精准数据加载成功，实际数据{len(df)}条")

            # 按股票分组
            stock_mapping = {}
            if not df.empty and 'stock_code' in df.columns:
                for stock_code in target_stocks:
                    stock_data = df[df['stock_code'] == stock_code].sort_values('datetime')
                    if not stock_data.empty:
                        stock_mapping[stock_code] = stock_data
                        logger.debug(f"   ✓ {stock_code}: {len(stock_data)}条5分钟数据")
                    else:
                        logger.warning(f"   ⚠️ {stock_code}: 无数据")

            # 缓存结果 - 统一使用日期作为Key，覆盖式缓存避免复杂合并
            cache_key = date_str
            self.minute_data_cache[cache_key] = stock_mapping.copy()
            logger.info(f"📊 精准加载完成并缓存: {len(stock_mapping)}只股票 (缓存Key: {cache_key})")
            return stock_mapping

        except Exception as e:
            logger.error(f"精准加载{date_str}数据失败: {e}")
            # 降级到逐只股票加载
            return self._load_stocks_one_by_one(date_str, minute_file, start_time, end_time, target_stocks)

    def _load_stocks_one_by_one(self, date_str: str, minute_file: str, start_time: datetime,
                               end_time: datetime, target_stocks: List[str]) -> Dict[str, pd.DataFrame]:
        """
        降级方案：逐只股票加载，避免内存问题
        """
        logger.info(f"🔄 使用降级方案：逐只加载{len(target_stocks)}只股票...")

        stock_mapping = {}
        columns = ['datetime', 'stock_code', 'open', 'close', 'high', 'low', 'volume']

        for stock_code in target_stocks:
            try:
                # 单只股票过滤器
                filters = [
                    ('datetime', '>=', start_time),
                    ('datetime', '<=', end_time),
                    ('stock_code', '=', stock_code)  # 精确匹配单只股票
                ]

                table = pq.read_table(minute_file, filters=filters, columns=columns, memory_map=True)
                df = table.to_pandas()

                if not df.empty:
                    stock_mapping[stock_code] = df.sort_values('datetime')
                    logger.debug(f"   ✓ {stock_code}: {len(df)}条数据")
                else:
                    logger.warning(f"   ⚠️ {stock_code}: 无数据")

            except Exception as e:
                logger.error(f"   ❌ {stock_code}: 加载失败 - {e}")
                continue

        logger.info(f"📊 降级加载完成: {len(stock_mapping)}/{len(target_stocks)}只股票")

        # 缓存结果 - 统一使用日期作为Key，覆盖式缓存避免复杂合并
        if stock_mapping:  # 只有当有数据时才缓存
            cache_key = date_str
            self.minute_data_cache[cache_key] = stock_mapping.copy()
            logger.info(f"📊 降级加载完成并缓存: {len(stock_mapping)}只股票 (缓存Key: {cache_key})")

        return stock_mapping

    def _load_all_stocks_optimized(self, date_str: str, minute_file: str,
                                  start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """
        优化的全量股票加载 - 使用内存映射和列过滤
        """
        logger.info(f"📊 加载{date_str}所有股票的5分钟数据（优化模式）...")

        try:
            # 只使用日期过滤器，但限制列读取
            filters = [
                ('datetime', '>=', start_time),
                ('datetime', '<=', end_time)
            ]

            # 只读取必要的列
            columns = ['datetime', 'stock_code', 'open', 'close', 'high', 'low', 'volume']

            logger.info(f"🔍 使用内存映射和列过滤优化...")
            table = pq.read_table(minute_file, filters=filters, columns=columns, memory_map=True)
            df = table.to_pandas()

            logger.info(f"✅ 数据加载成功，原始数据{len(df)}条，内存使用优化")

            # 按股票分组
            stock_mapping = {}
            if not df.empty and 'stock_code' in df.columns:
                logger.info(f"🔄 按股票分组处理...")
                for stock_code, stock_data in df.groupby('stock_code'):
                    stock_data_sorted = stock_data.sort_values('datetime')
                    stock_mapping[stock_code] = stock_data_sorted
                logger.info(f"✅ 成功分组{len(stock_mapping)}只股票")

            # 缓存结果
            self.minute_data_cache[date_str] = stock_mapping
            logger.info(f"📊 {date_str}数据加载完成: {len(stock_mapping)}只股票")
            return stock_mapping

        except Exception as e:
            logger.error(f"优化加载{date_str}数据失败: {e}")
            return {}

    def _preload_stocks_data(self, dates: List[datetime.date], target_stocks: List[str]):
        """
        批量预加载指定日期和股票的5分钟数据 - 提高效率，避免重复IO
        """
        logger.info(f"🚀 开始批量预加载: {len(dates)}个日期 x {len(target_stocks)}只股票")

        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            try:
                # 使用精准批量加载
                stock_mapping = self.load_minute_data_on_demand(date_str, target_stocks)
                logger.info(f"📦 {date_str}: 预加载完成 {len(stock_mapping)}只股票数据")

            except Exception as e:
                logger.error(f"❌ {date_str}: 预加载失败 - {e}")
                continue

        logger.info(f"✅ 批量预加载完成")

    def _reload_target_stocks(self, date_str: str, target_stocks: List[str]) -> Dict[str, pd.DataFrame]:
        """
        重新加载指定日期的目标股票数据，覆盖现有缓存
        """
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            year_month = target_date.strftime('%Y%m')
            month_dir = os.path.join(self.minute_data_path, year_month)
            minute_file = os.path.join(month_dir, "minute_data.parquet")

            if not os.path.exists(minute_file):
                logger.warning(f"❌ 未找到{date_str}的数据文件")
                return {stock: pd.DataFrame() for stock in target_stocks}

            start_time = datetime.strptime(f'{date_str} 00:00:00', '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(f'{date_str} 23:59:59', '%Y-%m-%d %H:%M:%S')

            # 策略选择：如果目标股票较少，使用精准加载
            if len(target_stocks) <= 20:
                return self._load_specific_stocks(date_str, minute_file, start_time, end_time, target_stocks)
            else:
                # 加载所有股票然后过滤
                all_stocks_data = self._load_all_stocks_optimized(date_str, minute_file, start_time, end_time)
                # 更新缓存
                self.minute_data_cache[date_str] = all_stocks_data
                # 返回目标股票的数据
                return {stock: all_stocks_data.get(stock, pd.DataFrame()) for stock in target_stocks}

        except Exception as e:
            logger.error(f"重新加载{date_str}目标股票失败: {e}")
            return {stock: pd.DataFrame() for stock in target_stocks}

    def get_stock_data(self, stock_code: str, date: datetime.date) -> pd.DataFrame:
        """获取指定股票在指定日期的5分钟数据 - 使用精准查询优化"""
        date_str = date.strftime('%Y-%m-%d')

        try:
            # 使用精准加载：只加载指定的股票数据
            stock_mapping = self.load_minute_data_on_demand(date_str, [stock_code])
            return stock_mapping.get(stock_code, pd.DataFrame())

        except Exception as e:
            logger.error(f"获取{stock_code}在{date_str}的5分钟数据失败: {e}")
            return pd.DataFrame()

    def check_entry_signal(self, stock_code: str, buy_date: datetime.date,
                          prev_close: Optional[float], buy_open: float) -> Dict[str, Any]:
        """开盘风控检查"""
        logger.info(f"🔎 {stock_code} 开盘风控检查: buy_date={buy_date}, prev_close={prev_close}, buy_open={buy_open}")

        if prev_close is None:
            logger.info(f"✅ {stock_code}: 无前日数据，允许买入")
            return {'can_buy': True, 'reason': '无前日数据'}

        gap_ratio = (buy_open - prev_close) / prev_close
        logger.info(f"📊 {stock_code}: 跳空计算 (buy_open={buy_open} - prev_close={prev_close}) / prev_close = {gap_ratio:.4f} ({gap_ratio:.2%})")

        # 风控条件
        if gap_ratio < -0.03:  # 跌空超过3%
            logger.info(f"🚫 {stock_code}: 跌空超过3% ({gap_ratio:.2%})，禁止买入")
            return {'can_buy': False, 'reason': f'开盘跳空{-gap_ratio:.2%}', 'gap_ratio': gap_ratio}
        elif gap_ratio > 0.05:  # 高开超过5%
            logger.info(f"🚫 {stock_code}: 高开超过5% ({gap_ratio:.2%})，禁止买入")
            return {'can_buy': False, 'reason': f'开盘跳空{gap_ratio:.2%}', 'gap_ratio': gap_ratio}

        logger.info(f"✅ {stock_code}: 通过风控检查，跳空{gap_ratio:.2%}在范围内")
        return {'can_buy': True, 'reason': '通过风控', 'gap_ratio': gap_ratio}

    def check_position_risk(self, position: Position, current_date: datetime.date, trading_dates: List[datetime.date]) -> Dict[str, Any]:
        """持仓风控检查 - 使用交易日期列表，避免非交易日"""
        logger.info(f"🛡️ 开始风控检查: {position.stock_code}, 买入日: {position.buy_date}, 当前日: {current_date}")

        # 确保trading_dates按时间顺序排列
        trading_dates = sorted(trading_dates)

        try:
            # 找到买入日在交易日期列表中的索引
            buy_date_index = trading_dates.index(position.buy_date)
            current_date_index = trading_dates.index(current_date)
            logger.debug(f"   买入日索引: {buy_date_index}, 当前日索引: {current_date_index}")
        except ValueError as e:
            logger.error(f"❌ 日期索引查找失败: {e}")
            return {'should_exit': False, 'reason': '日期索引错误'}

        # 检查是否到了T+2（买入日的下1个交易日）
        if current_date_index <= buy_date_index:
            days_to_wait = (buy_date_index + 1) - current_date_index
            logger.info(f"⏸️ {position.stock_code}: 未到T+2，还需等待{days_to_wait}个交易日")
            return {'should_exit': False, 'reason': f'未到T+2(还需{days_to_wait}日)'}

        # 确定风控检查范围：T+2交易日（买入日的下1个交易日）~ min(T+holding_days交易日, 当前日)
        start_index = buy_date_index + 1  # T+2交易日（买入日的下一个交易日）
        end_index = min(
            buy_date_index + self.config.holding_days,  # T+holding_days交易日
            current_date_index + 1  # 当前日（包含）
        )

        # 获取风控检查的实际日期范围
        risk_check_dates = trading_dates[start_index:end_index]
        logger.info(f"📊 {position.stock_code} 风控检查范围: {risk_check_dates[0]} ~ {risk_check_dates[-1]} ({len(risk_check_dates)}个交易日)")

        # 获取所有风控检查日内的5分钟数据
        minute_data_list = []
        logger.info(f"📊 收集{position.stock_code}在{len(risk_check_dates)}个交易日内的5分钟数据进行风控检查...")

        for check_date in risk_check_dates:
            day_data = self.get_stock_data(position.stock_code, check_date)
            if not day_data.empty:
                logger.debug(f"   {check_date}: 获取到{len(day_data)}条5分钟数据")
                for _, row in day_data.iterrows():
                    minute_data_list.append(row.to_dict())
            else:
                logger.warning(f"   ⚠️ {check_date}: 无5分钟数据")

        logger.info(f"📈 {position.stock_code}: 风控检查范围内总共收集到{len(minute_data_list)}条5分钟数据")

        if len(minute_data_list) < 3:
            logger.info(f"⚠️ {position.stock_code}: 数据不足({len(minute_data_list)}条)，跳过阴线检查")
            return {'should_exit': False, 'reason': '数据不足'}

        # 统一时间序列风控检查 - 实现"先满足先触发"原则
        logger.debug(f"🛡️ 开始统一风控检查（累计下跌 vs 连续3根阴线）...")

        # 按时间顺序逐个K线检查，实现先满足先触发
        for i, minute_data in enumerate(minute_data_list):
            current_price = minute_data['close']
            current_datetime = minute_data['datetime']

            # 1. 检查累计下跌超5%（基于买入价）
            decline_ratio = (current_price - position.buy_price) / position.buy_price
            if decline_ratio <= -0.05:  # 累计下跌超过5%
                # 处理不同的datetime格式
                if isinstance(current_datetime, str):
                    exit_time = datetime.fromisoformat(current_datetime)
                elif hasattr(current_datetime, 'to_pydatetime'):  # pandas Timestamp
                    exit_time = current_datetime.to_pydatetime()
                elif isinstance(current_datetime, datetime):
                    exit_time = current_datetime
                else:
                    exit_time = datetime.now()

                logger.info(f"🚨 {position.stock_code}: 在第{i+1}根K线发现累计下跌超5%!")
                logger.debug(f"   跌幅K线: 时间={current_datetime}, 开盘={minute_data['open']:.4f}, 收盘={current_price:.4f}")
                logger.debug(f"   买入价: {position.buy_price:.4f}, 当前价: {current_price:.4f}, 跌幅: {decline_ratio:.2%}")

                return {
                    'should_exit': True,
                    'reason': '累计下跌超5%',
                    'exit_price': current_price,
                    'exit_time': exit_time
                }

            # 2. 检查连续3根阴线（需要至少3根K线）
            if i >= 2:  # 至少有3根K线可以检查连续性
                # 获取最近的3根K线：第i-2, i-1, i根
                candle_1 = minute_data_list[i - 2]
                candle_2 = minute_data_list[i - 1]
                candle_3 = minute_data

                # 检查这3根K线是否都是阴线
                is_red_1 = candle_1['close'] < candle_1['open']
                is_red_2 = candle_2['close'] < candle_2['open']
                is_red_3 = candle_3['close'] < candle_3['open']

                if is_red_1 and is_red_2 and is_red_3:
                    # 发现连续3根阴线，触发风控
                    logger.info(f"🚨 {position.stock_code}: 在第{i-1}-{i+1}根K线发现连续3根阴线!")
                    logger.debug(f"   阴线1: 时间={candle_1['datetime']}, 开盘={candle_1['open']:.4f}, 收盘={candle_1['close']:.4f}")
                    logger.debug(f"   阴线2: 时间={candle_2['datetime']}, 开盘={candle_2['open']:.4f}, 收盘={candle_2['close']:.4f}")
                    logger.debug(f"   阴线3: 时间={candle_3['datetime']}, 开盘={candle_3['open']:.4f}, 收盘={candle_3['close']:.4f}")

                    # 处理不同的datetime格式
                    if isinstance(current_datetime, str):
                        exit_time = datetime.fromisoformat(current_datetime)
                    elif hasattr(current_datetime, 'to_pydatetime'):  # pandas Timestamp
                        exit_time = current_datetime.to_pydatetime()
                    elif isinstance(current_datetime, datetime):
                        exit_time = current_datetime
                    else:
                        exit_time = datetime.now()

                    return {
                        'should_exit': True,
                        'reason': '连续3根阴线',
                        'exit_price': candle_3['close'],
                        'exit_time': exit_time
                    }

        logger.info(f"✅ {position.stock_code}: 未发现风控信号（累计下跌超5% 或 连续3根阴线）")
        return {'should_exit': False, 'reason': '无风控信号'}

    def select_top_k_stocks(self, filtered_df: pd.DataFrame, k: int) -> pd.DataFrame:
        """选择Top-K股票"""
        if len(filtered_df) < k:
            return filtered_df
        return filtered_df.nlargest(k, 'predicted_return')

    def execute_buy(self, stock_code: str, buy_date: datetime.date,
                    predicted_return: float) -> Optional[Position]:
        """执行买入操作"""
        logger.info(f"🛒 准备买入 {stock_code}, 预测收益: {predicted_return:.4f}")

        stock_data = self.get_stock_data(stock_code, buy_date)
        if stock_data.empty:
            logger.warning(f"⚠️ {stock_code} 买入日无分钟数据，跳过买入")
            return None

        logger.info(f"📊 {stock_code} 获取到{len(stock_data)}条5分钟数据")

        # 使用开盘价买入
        buy_price = stock_data.iloc[0]['open']
        raw_datetime = stock_data.iloc[0]['datetime']

        # 处理不同的datetime格式
        if isinstance(raw_datetime, str):
            buy_datetime = datetime.fromisoformat(raw_datetime)
        elif hasattr(raw_datetime, 'to_pydatetime'):  # pandas Timestamp
            buy_datetime = raw_datetime.to_pydatetime()
        elif isinstance(raw_datetime, datetime):
            buy_datetime = raw_datetime
        else:
            logger.warning(f"⚠️ 未知的datetime格式: {type(raw_datetime)}, 使用当前时间")
            buy_datetime = datetime.now()

        logger.info(f"💰 {stock_code} 开盘价: {buy_price:.4f}, 开盘时间: {buy_datetime}, 当前现金: {self.cash:.2f}")

        # ✅ 修复: 基于当前可用现金计算买入数量
        # Top-K策略中，每个策略应该基于当前可用现金进行分配

        # 检查现金是否足够购买至少1手
        min_cost = buy_price * 100  # 至少买1手的成本
        if self.cash < min_cost:
            logger.warning(f"⚠️ 现金不足，无法买入{stock_code}: 可用现金={self.cash:.2f}, 最少需要={min_cost:.2f}")
            return None

        # ✅ 修复: Top-K策略的资金分配逻辑
        # 根据当前持仓数量和策略配置，智能分配当前可用现金

        if self.config.stock_count == 1:
            # Top-1策略：用全部可用现金买入最优的1只股票
            position_amount = self.cash
            logger.info(f"📊 Top-1策略: 使用全部现金 {self.cash:.2f}")
        else:
            # Top-K策略(K>1): 考虑当前持仓情况，分配资金
            current_positions = len(self.current_holdings)
            remaining_slots = self.config.stock_count - current_positions

            if remaining_slots <= 1:
                # 这是最后一只股票，用剩余全部现金买入
                position_amount = self.cash
                logger.info(f"📊 Top-{self.config.stock_count}策略: 最后买入，使用剩余现金 {self.cash:.2f}")
            else:
                # 还有多个买入名额，为后续买入预留资金
                # 采用保守策略：当前这只股票只用部分现金，为后续留出空间
                reserved_ratio = 0.7  # 为后续买入预留30%资金
                position_amount = self.cash * reserved_ratio / remaining_slots
                logger.info(f"📊 Top-{self.config.stock_count}策略: 分配资金 {position_amount:.2f} (当前持仓: {current_positions}, 剩余额度: {remaining_slots})")
        quantity = int(position_amount / buy_price / 100) * 100  # 整手
        actual_cost = buy_price * quantity

        # 再次确保不会超支
        if actual_cost > self.cash:
            # 如果计算出的成本超过可用现金，减少买入数量
            quantity = int(self.cash / buy_price / 100) * 100
            actual_cost = buy_price * quantity
            if actual_cost > self.cash:  # 如果还是超支，调整到能买起的数量
                quantity = max(0, int(self.cash / buy_price / 100) * 100 - 100)
                actual_cost = buy_price * quantity

        # 最终验证
        if actual_cost > self.cash:
            logger.error(f"❌ 买入成本超支: 成本={actual_cost:.2f}, 可用现金={self.cash:.2f}")
            return None

        logger.info(f"💵 资金使用情况: 分配金额={position_amount:.2f}, 买入数量={quantity}股, 实际花费={actual_cost:.2f}")
        logger.info(f"💰 买入后现金: {self.cash - actual_cost:.2f} (买入前: {self.cash:.2f})")

        # 创建持仓记录
        position = Position(
            stock_code=stock_code,
            buy_date=buy_date,
            buy_datetime=buy_datetime,
            buy_price=buy_price,
            quantity=quantity,
            predicted_return=predicted_return,
            holding_end_date=buy_date + timedelta(days=self.config.holding_days - 1)  # 修正T+1逻辑：T+1买入，T+1+(holding_days-1)卖出
        )

        self.current_holdings[stock_code] = position
        self.cash -= buy_price * quantity

        # 更新买入周期状态 (关键修复)
        self.last_buy_date = buy_date
        self.current_buy_cycle_date = buy_date
        self.is_holding = True
        self.current_holding_end_date = position.holding_end_date

        # 修复T+1买入bug：买入成功后清除动态T日，使用正式的last_position_end_date
        self.dynamic_t_date = None  # 清除动态T日，买入后使用正式逻辑

        logger.info(f"✅ 买入成功: {stock_code}, 价格: {buy_price:.4f}, 数量: {quantity}, 时间: {buy_datetime}, 剩余现金: {self.cash:.2f}")
        logger.info(f"📅 买入周期状态: 最后买入日={self.last_buy_date}, 预计卖出日={self.current_holding_end_date}")

        return position

    def _should_attempt_buy(self, trade_date: datetime.date, trading_dates: List[datetime.date]) -> Dict[str, Any]:
        """检查当前交易日是否应该尝试买入 - 实现正确的T+1逻辑"""
        try:
            logger.debug(f"🤔 判断是否买入: trade_date={trade_date}, is_holding={self.is_holding}, "
                        f"last_buy_date={self.last_buy_date}, current_t_date={getattr(self, 'current_t_date', None)}")

            # 如果当前有持仓，不能买入
            if self.is_holding:
                logger.debug(f"🚫 当前有持仓，不允许买入")
                return {'should_buy': False, 'reason': f'持仓期间 (预计{self.current_holding_end_date}到期)', 'prediction_date': None}

            # 首次运行：设置current_t_date为交易日的前一日，允许买入
            if not self.last_buy_date:
                # 为了找到T日，我们需要找当前交易日之前的最近一个预测数据日期
                available_dates = sorted(self.prediction_data['date'].unique())
                earlier_dates = [d for d in available_dates if d < trade_date]

                if not earlier_dates:
                    return {'should_buy': False, 'reason': f'{trade_date} 是第一个交易日，无前一日预测数据', 'prediction_date': None}

                # 设置T日为最近的有预测数据的日期
                self.current_t_date = max(earlier_dates)
                logger.info(f"🆕 首次运行: 设置T日={self.current_t_date}, 允许在{trade_date}买入")
                return {'should_buy': True, 'reason': '首次买入', 'prediction_date': self.current_t_date}

            # 正常运行：按照用户指定的正确逻辑
            # "当前轮次的T+holding_days日，在T+holding_days日结束时，把该日当作下一轮的T日"

            try:
                # 获取当前交易日在交易日列表中的索引
                current_index = trading_dates.index(trade_date)

                if not hasattr(self, 'current_t_date') or self.current_t_date is None:
                    # 如果没有current_t_date，说明需要重新设置
                    # T日应该是当前交易日的上一交易日
                    if current_index > 0:
                        self.current_t_date = trading_dates[current_index - 1]
                        logger.info(f"🔄 重新设置T日: {self.current_t_date}")
                    else:
                        return {'should_buy': False, 'reason': '无法确定T日', 'prediction_date': None}

                # 检查当前交易日是否是T+1日
                t_index = trading_dates.index(self.current_t_date)
                days_since_t = current_index - t_index

                logger.debug(f"📅 T日逻辑: T日={self.current_t_date}, 当前={trade_date}, 距离T日={days_since_t}日")

                if days_since_t == 1:
                    # 正好是T+1日，可以买入
                    # 使用T日的预测数据
                    available_dates = sorted(self.prediction_data['date'].unique())
                    prediction_dates = [d for d in available_dates if d <= self.current_t_date]

                    if prediction_dates:
                        prediction_date = max(prediction_dates)
                    else:
                        prediction_date = self.current_t_date

                    logger.info(f"✅ T+1买入: T日={self.current_t_date}, T+1日={trade_date}, 使用预测日={prediction_date}")
                    return {'should_buy': True, 'reason': 'T+1买入', 'prediction_date': prediction_date}

                elif days_since_t < 1:
                    # 还没到T+1日
                    logger.debug(f"⏰ 还未到T+1买入日: {trade_date}（T日：{self.current_t_date}，还需{1-days_since_t}日）")
                    return {'should_buy': False, 'reason': f'未到T+1日 (T日={self.current_t_date}, 当前={trade_date}, 还需{1-days_since_t}日)', 'prediction_date': None}

                else:
                    # days_since_t > 1，已经过了T+1买入窗口
                    # 根据用户逻辑，此时应该等待T+holding_days日结束，然后将该日设为下一轮的T日
                    days_past_t1 = days_since_t - 1  # 已经过了T+1日的天数
                    days_in_holding_period = days_past_t1 + 1  # T+1日到当前日的总天数

                    logger.info(f"📅 已过T+1买入窗口: T日={self.current_t_date}, 当前={trade_date}, 已过T+1({days_past_t1}日)")

                    if days_in_holding_period >= self.config.holding_days:
                        # 已过持仓周期，按照用户逻辑：将当前日设为下一轮的T日
                        logger.info(f"🔄 持仓周期已过，更新T日: {self.current_t_date} → {trade_date}")
                        self.current_t_date = trade_date
                        return {'should_buy': False, 'reason': f'持仓周期已过，更新T日为{trade_date}', 'prediction_date': None}
                    else:
                        # 还在持仓周期内，不能买入
                        logger.debug(f"⏳ 持仓期间: 已过{days_in_holding_period}/{self.config.holding_days}日")
                        return {'should_buy': False, 'reason': f'已过T+1买入窗口，持仓中({days_in_holding_period}/{self.config.holding_days}日)', 'prediction_date': None}

            except ValueError as e:
                logger.warning(f"⚠️ 交易日索引错误: {e}")
                return {'should_buy': False, 'reason': '日期索引错误', 'prediction_date': None}

        except Exception as e:
            logger.error(f"❌ 买入判断逻辑异常: {e}")
            # 异常情况下默认不允许买入，避免错误状态
            return {'should_buy': False, 'reason': f'逻辑异常: {e}', 'prediction_date': None}

    def execute_sell(self, position: Position, sell_date: datetime.date,
                     exit_reason: str, exit_price: Optional[float] = None) -> Dict[str, Any]:
        """执行卖出操作"""
        if exit_price is None:
            stock_data = self.get_stock_data(position.stock_code, sell_date)
            if stock_data.empty:
                # 使用买入价止损卖出
                exit_price = position.buy_price * 0.98  # 假设2%止损
                sell_datetime = datetime.combine(sell_date, datetime.min.time())
                logger.warning(f"⚠️ {position.stock_code} 无卖出数据，使用止损价: {exit_price:.4f}")
            else:
                exit_price = stock_data.iloc[-1]['close']
                raw_sell_datetime = stock_data.iloc[-1]['datetime']

                # 处理不同的datetime格式
                if isinstance(raw_sell_datetime, str):
                    sell_datetime = datetime.fromisoformat(raw_sell_datetime)
                elif hasattr(raw_sell_datetime, 'to_pydatetime'):  # pandas Timestamp
                    sell_datetime = raw_sell_datetime.to_pydatetime()
                elif isinstance(raw_sell_datetime, datetime):
                    sell_datetime = raw_sell_datetime
                else:
                    logger.warning(f"⚠️ 卖出时未知的datetime格式: {type(raw_sell_datetime)}, 使用当前时间")
                    sell_datetime = datetime.now()
        else:
            sell_datetime = datetime.combine(sell_date, datetime.min.time())

        # 计算实际收益
        actual_return = (exit_price - position.buy_price) / position.buy_price
        position.actual_return = actual_return
        position.sell_price = exit_price
        position.sell_datetime = sell_datetime
        position.exit_reason = exit_reason

        # 更新现金
        sell_value = exit_price * position.quantity
        self.cash += sell_value

        # 创建交易记录
        trade_record = {
            'stock_code': position.stock_code,
            'buy_datetime': position.buy_datetime,
            'sell_datetime': sell_datetime,
            'buy_price': position.buy_price,
            'sell_price': exit_price,
            'quantity': position.quantity,
            'predicted_return': position.predicted_return,
            'actual_return': actual_return,
            'profit_loss': (exit_price - position.buy_price) * position.quantity,
            'profit_loss_pct': actual_return * 100,
            'holding_days': (sell_date - position.buy_date).days,
            'entry_reason': position.entry_reason,
            'exit_reason': exit_reason,
            'strategy_type': f"Top-{self.config.stock_count}"
        }

        self.trades.append(trade_record)
        del self.current_holdings[position.stock_code]

        # 检查是否所有持仓都已卖出，如果是则重置买入周期状态
        if not self.current_holdings:
            logger.info("🔄 所有持仓已清空，重置买入周期状态")
            self.is_holding = False
            self.current_holding_end_date = None
            # 关键修复：设置上一轮持仓到期日为下一轮的T日
            self.last_position_end_date = sell_date
            logger.info(f"📅 设置下一轮T日: {sell_date} (上一轮持仓到期日)")
            # 注意：保留last_buy_date用于下次买入周期判断

        logger.info(f"💰 卖出: {position.stock_code}, 价格: {exit_price:.4f}, "
                   f"收益: {actual_return:.4f} ({actual_return*100:.2f}%), "
                   f"原因: {exit_reason}")
        if not self.current_holdings:
            logger.info(f"📊 当前状态: 无持仓, 现金: {self.cash:.2f}, 最后买入日: {self.last_buy_date}")

        return trade_record

    def run_backtest(self, trading_dates: List[datetime.date]) -> Dict[str, Any]:
        """运行完整的回测"""
        logger.info(f"🚀 开始回测 Top-{self.config.stock_count} 策略")
        logger.info(f"   时间范围: {trading_dates[0]} ~ {trading_dates[-1]}")
        logger.info(f"   总交易日: {len(trading_dates)}")

        # 确保trading_dates按时间顺序排列
        trading_dates = sorted(trading_dates)
        logger.info(f"✅ 交易日期已按时间排序")

        # 预分析候选股票
        self.pre_analyze_required_stocks()

        for i, trade_date in enumerate(trading_dates):
            # 保存交易日期，用于后续绘图
            self.trading_dates_list.append(trade_date)

            logger.info(f"📅 交易日 {i+1}/{len(trading_dates)}: {trade_date}")
            logger.info(f"💼 当前持仓: {len(self.current_holdings)}只, 现金: {self.cash:.2f}")

            # 风控检查（优先执行 - 在到期卖出之前检查）
            risk_exited_positions = []
            positions_to_remove = []

            for stock_code, position in list(self.current_holdings.items()):
                # 使用交易日期列表进行精确的风控检查
                logger.debug(f"🔍 检查{position.stock_code}在{trade_date}的风控状态 (买入日: {position.buy_date})")
                risk_check = self.check_position_risk(position, trade_date, trading_dates)
                if risk_check['should_exit']:
                    logger.info(f"🛡️ 风控触发: {position.stock_code}, 原因: {risk_check['reason']}")
                    trade = self.execute_sell(
                        position, trade_date,
                        f"风控: {risk_check['reason']}",
                        risk_check.get('exit_price')
                    )
                    if trade:
                        risk_exited_positions.append(trade)
                        positions_to_remove.append(stock_code)

            # 检查持仓到期（在风控检查之后执行）
            expired_positions = []
            for stock_code, position in self.current_holdings.items():
                if stock_code not in positions_to_remove and position.holding_end_date and trade_date >= position.holding_end_date:
                    expired_positions.append(position)

            # 到期卖出
            if expired_positions:
                logger.info(f"⏰ {len(expired_positions)}只持仓到期")
                for position in expired_positions:
                    self.execute_sell(position, trade_date, "到期")

            # 空仓选股 - 检查买入时机
            if not self.current_holdings and i < len(trading_dates) - 1:
                # 检查是否是正确的买入时机
                should_buy = self._should_attempt_buy(trade_date, trading_dates)

                if not should_buy['should_buy']:
                    logger.info(f"⏸️ {should_buy['reason']}, 跳过买入")
                    continue

                # T+1交易逻辑：使用T日预测数据在T+1日买入
                prediction_date = should_buy['prediction_date']
                logger.info(f"🔍 T+1交易逻辑: T+1日={trade_date}, T日={prediction_date}，开始选股")

                # 获取预测数据 - 使用pre_analyze_required_stocks保存的Top-K股票列表
                prediction_date_str = prediction_date.strftime('%Y-%m-%d')
                top_k_stocks = self.required_stocks_by_date.get(prediction_date_str, [])

                if not top_k_stocks:
                    logger.warning(f"⚠️ {prediction_date} 无预分析的Top-K股票，跳过买入")
                    continue

                logger.info(f"📋 {prediction_date} 预分析Top-{len(top_k_stocks)}股票: {top_k_stocks[:3]}{'...' if len(top_k_stocks) > 3 else ''}")

                # 只获取预分析的Top-K股票的预测数据，保持数据一致性
                daily_predictions = self.prediction_data[
                    (self.prediction_data['date'] == prediction_date) &
                    (self.prediction_data['stock_code'].isin(top_k_stocks))
                ]

                if daily_predictions.empty:
                    logger.warning(f"⚠️ {prediction_date} Top-K股票无预测数据，跳过")
                    continue

                logger.info(f"📊 {prediction_date} Top-K预测数据: {len(daily_predictions)}只股票")

                # 直接使用预分析的排序结果，无需重新过滤和排序
                selected_stocks = daily_predictions.sort_values('predicted_return', ascending=False)

                logger.info(f"🎆 使用预分析结果Top-{len(selected_stocks)}股票进行买入检查:")
                for _, stock in selected_stocks.iterrows():
                    logger.info(f"   📈 {stock['stock_code']}: 预测收益={stock['predicted_return']:.4f}")

                logger.info(f"🔄 开始对Top-{self.config.stock_count}策略执行买入逻辑...")

                # 预加载策略优化：提前加载当日需要的所有股票数据
                target_stocks_for_today = [stock['stock_code'] for _, stock in selected_stocks.head(self.config.stock_count).iterrows()]
                logger.info(f"📦 预加载{trade_date}和{prediction_date}的{len(target_stocks_for_today)}只目标股票5分钟数据...")

                # 批量预加载T日和T+1日的股票数据
                self._preload_stocks_data([prediction_date, trade_date], target_stocks_for_today)

                # 执行买入 - 只处理前stock_count只股票
                for i, (_, stock) in enumerate(selected_stocks.iterrows()):
                    if i >= self.config.stock_count:  # 只处理前stock_count只股票
                        logger.info(f"✅ 已处理前{self.config.stock_count}只股票，跳过剩余股票")
                        break
                    logger.info(f"🎯 处理Top-{self.config.stock_count}策略第{i+1}只股票: {stock['stock_code']}")
                    stock_code = stock['stock_code']

                    # 正确的T+1风控检查：使用T日收盘价 vs T+1日开盘价
                    logger.info(f"📅 {stock_code} T+1买入逻辑: T日={prediction_date}(预测日), T+1日={trade_date}(执行日)")

                    # 获取T日（预测日）的收盘价用于风控（现在使用缓存数据）
                    logger.info(f"🔍 获取{stock_code} T日({prediction_date})数据用于风控检查...")
                    prediction_day_data = self.get_stock_data(stock_code, prediction_date)

                    if not prediction_day_data.empty:
                        prev_close = prediction_day_data.iloc[-1]['close']
                        logger.info(f"📊 {stock_code} T日({prediction_date})收盘价: {prev_close:.2f}")
                        logger.info(f"📊 {stock_code} T日数据条数: {len(prediction_day_data)}")
                        logger.info(f"📊 {stock_code} T日最后一条数据时间: {prediction_day_data.iloc[-1]['datetime']}")
                    else:
                        prev_close = None
                        logger.warning(f"⚠️ {stock_code} T日({prediction_date})无数据")

                    # 获取T+1日（执行日）的开盘价用于买入
                    logger.info(f"🔍 获取{stock_code} T+1日({trade_date})数据用于买入...")
                    execution_day_data = self.get_stock_data(stock_code, trade_date)

                    # 初始化buy_data变量
                    buy_data = execution_day_data

                    if not execution_day_data.empty:
                        buy_open = execution_day_data.iloc[0]['open']
                        logger.info(f"📊 {stock_code} T+1日({trade_date})开盘价: {buy_open:.2f}")
                        logger.info(f"📊 {stock_code} T+1日数据条数: {len(execution_day_data)}")
                        logger.info(f"📊 {stock_code} T+1日第一条数据时间: {execution_day_data.iloc[0]['datetime']}")
                    else:
                        buy_open = None

                    if not buy_data.empty:
                            # 重新获取buy_open（确保使用正确的值）
                            buy_open = buy_data.iloc[0]['open']
                            prev_close_str = f"{prev_close:.4f}" if prev_close is not None else "N/A"
                            logger.info(f"🔍 {stock_code} 买入检查: 开盘价={buy_open:.4f}, 前收盘={prev_close_str}")

                            entry_check = self.check_entry_signal(
                                stock_code, trade_date, prev_close, buy_open
                            )

                            logger.info(f"🚦 {stock_code} 买入信号: {entry_check['can_buy']}, 原因: {entry_check['reason']}")

                            if entry_check['can_buy']:
                                position = self.execute_buy(
                                    stock_code, trade_date, stock['predicted_return']  # 修复：应该是trade_date，不是next_date
                                )
                                if position:
                                    self.positions.append(position)
                                    logger.info(f"✅ {stock_code} 买入成功，添加到持仓列表")
                                else:
                                    logger.warning(f"❌ {stock_code} 买入失败，execute_buy返回None")
                            else:
                                logger.info(f"🚫 {stock_code} 买入被拒绝: {entry_check['reason']}")
                                # 保存被跳过的买入尝试，用于后续分析
                                skipped_record = {
                                    'stock_code': stock_code,
                                    'prediction_date': prediction_date,  # T日（预测日）
                                    'attempt_buy_date': trade_date,  # T+1日（尝试买入日）
                                    'buy_open': buy_open,
                                    'prev_close': prev_close,
                                    'gap_ratio': entry_check.get('gap_ratio', 0),
                                    'skip_reason': entry_check['reason'],
                                    'predicted_return': stock['predicted_return'],
                                    'skip_category': '开盘风控'  # 分类：开盘风控
                                }
                                self.skipped_buy_attempts.append(skipped_record)
                    else:
                        logger.warning(f"⚠️ {stock_code} {trade_date}无5分钟数据，跳过买入")
                else:
                    logger.warning(f"⚠️ {prediction_date} 无通过基本过滤的股票，跳过买入")

            # 计算当日收益
            daily_pnl = sum([trade['profit_loss'] for trade in self.trades
                             if trade['sell_datetime'].date() == trade_date])
            current_value = self.cash + sum([
                pos.quantity * self.get_stock_data(
                    pos.stock_code, trade_date
                ).iloc[-1]['close']
                if not self.get_stock_data(pos.stock_code, trade_date).empty
                else pos.buy_price
                for pos in self.current_holdings.values()
            ])

            portfolio_value = current_value
            daily_return = (portfolio_value / self.daily_portfolio_values[-1] - 1) if self.daily_portfolio_values else 0

            self.daily_portfolio_values.append(portfolio_value)
            self.daily_returns.append(daily_return)

            # 定期输出进度
            if (i + 1) % 10 == 0 or i == len(trading_dates) - 1:
                logger.info(f"📊 [{i+1}/{len(trading_dates)}] {trade_date}: "
                           f"净值 {portfolio_value:.0f}, "
                           f"当日收益 {daily_return:.4f} "
                           f"({daily_return*100:.2f}%)")

        # 最终统计
        final_value = self.daily_portfolio_values[-1]
        total_return = (final_value / self.config.initial_capital - 1) * 100

        # 交易统计
        total_trades = len(self.trades)
        profitable_trades = len([t for t in self.trades if t['actual_return'] > 0])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

        avg_return = np.mean([t['actual_return'] for t in self.trades]) if self.trades else 0

        result = {
            'strategy_config': self.config,
            'initial_capital': self.config.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': self._calculate_annual_return(
                trading_dates[0], trading_dates[-1], total_return
            ),
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(trading_dates),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'trades': self.trades,
            'daily_portfolio_values': self.daily_portfolio_values,
            'daily_returns': self.daily_returns,
            'trading_dates': self.trading_dates_list,  # 添加交易日期列表
            'skipped_buy_attempts': self.skipped_buy_attempts,  # 添加被跳过的买入尝试
            'final_positions': list(self.current_holdings.values())
        }

        logger.info(f"🎯 Top-{self.config.stock_count} 策略回测完成:")
        logger.info(f"   总收益: {total_return:.2f}%")
        logger.info(f"   年化收益: {result['annual_return'] * 100:.2f}%")
        logger.info(f"   最大回撤: {result['max_drawdown']:.2f}%")
        logger.info(f"   夏普比率: {result['sharpe_ratio']:.3f}")
        logger.info(f"   交易次数: {total_trades}")
        logger.info(f"   胜率: {win_rate:.1f}%")

        return result

    def _calculate_annual_return(self, start_date: datetime.date,
                                 end_date: datetime.date, total_return: float) -> float:
        """计算年化收益率"""
        days = (end_date - start_date).days
        if days == 0:
            return 0
        return (1 + total_return / 100) ** (365.25 / days) - 1

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.daily_portfolio_values:
            return 0

        peak = self.daily_portfolio_values[0]
        max_drawdown = 0

        for value in self.daily_portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _calculate_sharpe_ratio(self, trading_dates: List[datetime.date]) -> float:
        """计算夏普比率"""
        if len(self.daily_returns) < 2:
            return 0

        returns_array = np.array(self.daily_returns[1:])  # 跳过第一天的0收益
        days = (trading_dates[-1] - trading_dates[0]).days

        if days == 0:
            return 0

        annual_return = np.mean(returns_array) * 252  # 年化
        annual_volatility = np.std(returns_array) * np.sqrt(252)

        if annual_volatility == 0:
            return 0

        return annual_return / annual_volatility


class OptimizedBacktester:
    """优化版回测器 - 管理多个策略"""

    def __init__(self, prediction_data: pd.DataFrame, minute_data_path: str):
        self.prediction_data = prediction_data
        self.minute_data_path = minute_data_path
        self.strategies: List[BacktestStrategy] = []

    def add_strategy(self, config: StrategyConfig) -> None:
        """添加策略"""
        strategy = BacktestStrategy(config, self.prediction_data, self.minute_data_path)
        self.strategies.append(strategy)

    def run_strategies(self, stock_counts: List[int], trading_dates: List[datetime.date]) -> Dict[str, Any]:
        """运行所有策略"""
        logger.info(f"🚀 开始运行 {len(stock_counts)} 个策略")

        results = {}

        for stock_count in stock_counts:
            config = StrategyConfig(
                strategy_type=StrategyType(stock_count),
                stock_count=stock_count
            )

            self.add_strategy(config)

        # 每个策略独立运行完整的回测周期
        for strategy in self.strategies:
            result = strategy.run_backtest(trading_dates)
            results[f"Top-{strategy.config.stock_count}"] = result

        return results


# 向后兼容的接口函数
def run_optimized_backtest(prediction_data: pd.DataFrame,
                          stock_counts: List[int],
                          minute_data_path: str,
                          holding_days: int = 2) -> Dict[str, Any]:
    """
    向后兼容的回测接口
    """
    logger.info("🔄 启动优化版回测系统")

    backtester = OptimizedBacktester(prediction_data, minute_data_path)
    trading_dates = sorted(prediction_data['date'].unique())

    # 过滤有效的交易日期
    valid_trading_dates = []
    for date in trading_dates:
        if isinstance(date, pd.Timestamp):
            date = date.date()
        valid_trading_dates.append(date)

    logger.info(f"📅 交易日期范围: {valid_trading_dates[0]} ~ {valid_trading_dates[-1]}")

    results = backtester.run_strategies(stock_counts, valid_trading_dates)

    return results