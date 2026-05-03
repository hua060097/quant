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
from pathlib import Path

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """策略类型枚举"""
    TOP_1 = 1
    TOP_2 = 2
    TOP_3 = 3
    TOP_5 = 5


class RiskControlStrategy:
    """风控策略类型"""
    CONSECUTIVE_BEARISH = "consecutive_bearish"  # 连续阴线策略
    DAILY_MA5_BREAK = "daily_ma5_break"  # 日线MA5跌破策略
    T1_LOW_BREAK_WITH_HIGH_DRAWDOWN = "t1_low_break_with_high_drawdown"  # 🆕 T+1最低价跌破+高位回撤策略


class RiskControlConfig:
    """风控配置"""
    # 风控策略选择（默认使用日线MA5跌破策略）
    STRATEGY = RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN

    # 日线MA5跌破策略参数
    MA5_CONSECUTIVE_BREAKS = 2  # 连续跌破MA5的次数（连续N个5分钟收盘价低于MA5）

    # 连续阴线策略参数（保持原有逻辑）
    CONSECUTIVE_BEARISH_COUNT = 3  # 连续阴线数量

    # 🆕 T+1最低价跌破策略参数
    T1_LOW_CONSECUTIVE_BREAKS = 2  # 连续跌破T+1日最低价的次数（亏损状态下使用）


class EntryRiskControlStrategy:
    """开盘风控策略类型"""
    GAP_BASED = "gap_based"  # 基于涨跌幅的风控（原有逻辑）
    MA5_BASED = "ma5_based"  # 基于MA5的风控（新增逻辑）


class EntryRiskControlConfig:
    """开盘风控配置"""
    # 开盘风控策略选择（默认使用MA5风控）
    STRATEGY = EntryRiskControlStrategy.GAP_BASED

    # MA5风控参数
    MA5_MAX_UP_THRESHOLD = 0.05  # 高开超MA5 3%禁止买入
    MA5_MAX_DOWN_THRESHOLD = -0.03  # 低开超MA5 3%禁止买入

    # 涨跌幅风控参数（保持原有逻辑）
    GAP_MAX_DOWN_THRESHOLD = -0.03  # 跌空超过3%禁止买入
    GAP_MAX_UP_THRESHOLD = 0.05  # 高开超过5%禁止买入
    GAP_MAX_DOWN_THRESHOLD_NON_TOP1 = -0.05  # 非Top1跌空超过5%禁止买入

@dataclass
class StrategyConfig:
    """策略配置类"""
    strategy_type: StrategyType
    stock_count: int
    enable_dynamic_rank_switch: bool = True
    holding_days: int = 3
    initial_capital: float = 1000000.0
    max_concurrent_positions: int = 1
    fixed_position_pct: float = 1
    position_schedule: Optional[List[float]] = None

    # 持仓卖出止损阈值（用于T+2、T+3及以后的持仓风控）
    # 说明：
    #   - T+1日为买入日，不存在卖出风控
    #   - 从T+2日开始，如果累计收益低于此阈值，触发止损卖出
    #   - 示例：
    #       - stop_loss_threshold = -0.05  : 跌幅超过5%才卖出（当前默认值）
    #       - stop_loss_threshold = 0.00   : 收益为负（0%）就卖出（激进风控）
    #       - stop_loss_threshold = 0.01   : 收益低于1%就卖出（要求正收益）
    #       - stop_loss_threshold = -0.10  : 跌幅超过10%才卖出（宽松风控）
    stop_loss_threshold: float = -0.05  # 默认-5%，与当前双策略回测口径一致

    # 🆕 风控策略配置
    risk_control_strategy: str = RiskControlConfig.STRATEGY  # 风控策略选择
    risk_control_ma5_consecutive_breaks: int = RiskControlConfig.MA5_CONSECUTIVE_BREAKS  # MA5连续跌破次数
    risk_control_t1_low_consecutive_breaks: int = RiskControlConfig.T1_LOW_CONSECUTIVE_BREAKS  # 🆕 T+1最低价连续跌破次数

    # 🆕 开盘风控策略配置
    entry_risk_control_strategy: str = EntryRiskControlConfig.STRATEGY  # 开盘风控策略选择
    entry_risk_ma5_up_threshold: float = EntryRiskControlConfig.MA5_MAX_UP_THRESHOLD  # MA5高开阈值
    entry_risk_ma5_down_threshold: float = EntryRiskControlConfig.MA5_MAX_DOWN_THRESHOLD  # MA5低开阈值
    entry_risk_up_threshold: float = EntryRiskControlConfig.GAP_MAX_UP_THRESHOLD  # MA5高开阈值
    entry_risk_down_threshold: float = EntryRiskControlConfig.GAP_MAX_DOWN_THRESHOLD  # MA5低开阈值
    entry_risk_down_threshold_non_top1: float = EntryRiskControlConfig.GAP_MAX_DOWN_THRESHOLD_NON_TOP1  # 非Top1低开阈值

    # 🆕 高位回撤风控参数（用于MA5跌破策略）
    high_return_threshold: float = 0.04  # 最高浮盈阈值（4%），触发后开始监控回撤
    high_drawdown_threshold: float = -0.03  # 从最高点回撤阈值（-3%），超过则卖出
    profit_bearish_min_peak_return: float = 0.08  # 盈利状态止盈生效前，至少先达到8%最高浮盈
    profit_bearish_trailing_drawdown: float = -0.04  # 盈利状态下相对最高点至少回撤4%
    profit_bearish_min_current_return: float = 0.04  # 盈利状态止盈时，当前至少仍有4%浮盈
    profit_bearish_required_count: int = 3  # 缓慢转弱场景的连续阴线触发数量
    profit_bearish_min_single_drop: float = -0.012  # 快速转弱：单根中阴线阈值
    profit_bearish_two_bar_combined_drop: float = -0.018  # 快速转弱：两根合计跌幅阈值
    profit_bearish_slow_single_drop: float = -0.003  # 缓慢转弱：单根最小跌幅阈值

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
    holding_days: int = 3
    slot_index: int = 0

    # 🆕 跟踪买入后的最高价和最高浮盈（用于高位回撤风控）
    highest_price_since_buy: float = 0.0  # 买入后的最高价格
    highest_return_since_buy: float = -999.0  # 买入后的最高浮盈（相对买入价）
    t1_day_lowest_price: float = 0.0  # 🆕 T+1日（买入日）的最低价（从日线数据获取）

class BacktestStrategy:
    """单个策略的回测类 - 完全独立，避免状态干扰"""

    def __init__(self, config: StrategyConfig, prediction_data: pd.DataFrame, minute_data_path: str):
        self.config = config
        self.minute_data_path = minute_data_path

        # 标准化预测数据的日期格式 - 统一为datetime.date
        prediction_data = prediction_data.copy()
        if prediction_data.columns.duplicated().any():
            duplicated_columns = prediction_data.columns[prediction_data.columns.duplicated()].tolist()
            logger.warning(f"⚠️ 预测数据存在重复列，已保留首列并去重: {duplicated_columns}")
            prediction_data = prediction_data.loc[:, ~prediction_data.columns.duplicated()].copy()
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

        # 回测结果
        self.trades: List[Dict] = []
        self.daily_portfolio_values: List[float] = [config.initial_capital]
        self.daily_returns: List[float] = [0.0]
        self.trading_dates_list: List[datetime.date] = []  # 保存交易日期列表，用于绘图
        self.skipped_buy_attempts: List[Dict] = []  # 保存被跳过的买入尝试（开盘风控等），用于绘图

        # 数据缓存
        self.minute_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.required_stocks_by_date: Dict[str, List[str]] = {}  # 按日期保存每日Top-K股票，按预测值排序

        # 🆕 加载申万行业映射（用于行业跌停风控）
        self.stock_to_sw_industry = {}
        try:
            import sys
            import os
            # 添加父目录到路径以便导入sw_encoding_loader
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            from sw_encoding_loader import load_sw_sector_mapping
            self.stock_to_sw_industry = load_sw_sector_mapping()
            logger.info(f"✅ 成功加载申万行业映射: {len(self.stock_to_sw_industry)}只股票")
        except Exception as e:
            logger.warning(f"⚠️ 无法加载申万行业映射: {e}，将跳过行业跌停风控")
            self.stock_to_sw_industry = {}

        logger.info(f"🚀 初始化策略: Top-{config.stock_count}, 初始资金: {config.initial_capital:,.0f}")


    @staticmethod
    def _safe_logit(p: float, eps: float = 1e-6) -> float:
        """稳定logit，避免p=0/1导致无穷大。"""
        try:
            x = float(p)
        except Exception:
            return 0.0
        if np.isnan(x):
            return 0.0
        x = min(max(x, eps), 1.0 - eps)
        return float(np.log(x / (1.0 - x)))



    def _select_target_stock_for_top1_fusion(self, selected_stocks: pd.DataFrame, alpha: float = 0.1) -> pd.Series:
        """
        Top-1策略：在rank Top1/Top2之间，用融合分数二选一。
        - 需要列：predicted_return, p_cls, stock_code
        - 若缺少p_cls或候选不足2只，则退回rank Top1。
        """
        if selected_stocks is None or len(selected_stocks) == 0:
            return None
        if len(selected_stocks) < 2:
            return selected_stocks.iloc[0]
        if 'p_cls' not in selected_stocks.columns:
            return selected_stocks.iloc[0]

        r1 = selected_stocks.iloc[0]
        r2 = selected_stocks.iloc[1]
        s1 = float(r1.get('predicted_return', 0.0))
        s2 = float(r2.get('predicted_return', 0.0))
        p1 = r1.get('p_cls', np.nan)
        p2 = r2.get('p_cls', np.nan)
        z1 = self._safe_logit(p1)
        z2 = self._safe_logit(p2)
        score1 = s1 + float(alpha) * z1
        score2 = s2 + float(alpha) * z2

        logger.info(f"🧠 [FUSION] Top1/Top2融合打分: "
                    f"Top1({r1.get('stock_code')}): s={s1:.6f}, p={float(p1) if p1==p1 else np.nan:.4f}, logit={z1:.4f}, score={score1:.6f} | "
                    f"Top2({r2.get('stock_code')}): s={s2:.6f}, p={float(p2) if p2==p2 else np.nan:.4f}, logit={z2:.4f}, score={score2:.6f}")

        if score2 > score1:
            logger.info(f"✅ [FUSION] 选择Top2替代Top1: {r2.get('stock_code')} (score2>score1)")
            return r2
        else:
            logger.info(f"✅ [FUSION] 保持Top1: {r1.get('stock_code')} (score1>=score2)")
            return r1


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
                if stock['is_can_buy'] != 1:
                    filtering_stats['is_can_buy_filtered'] += 1
                    continue
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
                    # logger.info(f"🚫 涨跌停过滤: {stock['stock_code']} is_limit_up={stock.get('is_limit_up', 'N/A')}, is_limit_down={stock.get('is_limit_down', 'N/A')}, pct_chg={stock.get('pct_chg', 'N/A')}")
                    continue
                if stock['is_zhaban'] == 1:
                    filtering_stats['zhaban_filtered'] += 1
                    continue

                # 转为dict，避免重复列索引的Series在DataFrame重建时触发 InvalidIndexError
                filtered_stocks.append(stock.to_dict())
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

    def _calculate_daily_ma5_at_open(self, stock_code: str, buy_date: datetime.date,
                                     buy_open: float) -> Optional[float]:
        """
        计算买入日开盘时的日线MA5（前4天日线close + 买入日开盘价）

        Args:
            stock_code: 股票代码
            buy_date: 买入日期
            buy_open: 买入日开盘价

        Returns:
            MA5值，如果前4天数据不足返回None

        Note:
            MA5 = (前4天日线close之和 + 买入日开盘价) / 5

        Example:
            买入日: 2026-01-10, 开盘价: 10.5
            前4天日线close: [10.2, 10.3, 10.1, 10.4]
            MA5 = (10.2 + 10.3 + 10.1 + 10.4 + 10.5) / 5 = 10.3
        """
        # 获取前4个交易日的日线close
        daily_closes = self._get_daily_closes_before(stock_code, buy_date, 4)

        # 检查数据是否充足
        if len(daily_closes) < 4:
            logger.info(f"⚠️ {stock_code} 在{buy_date}前不足4个交易日数据")
            # return None

        # 计算MA5：前4天日线 + 买入日开盘
        ma5 = (sum(daily_closes) + buy_open) / 5.0

        logger.info(f"📊 {stock_code} 买入日开盘MA5计算:")
        logger.info(f"   前4天日线close: {daily_closes}")
        logger.info(f"   买入日开盘价: {buy_open}")
        logger.info(f"   MA5: {ma5:.4f}")

        return ma5

    def check_entry_signal(self, stock_code: str, buy_date: datetime.date,
                          prev_close: Optional[float], buy_open: float,
                          target_rank: int = 1) -> Dict[str, Any]:
        """
        开盘风控检查 - 支持配置化策略选择

        支持两种策略：
        1. GAP_BASED: 基于涨跌幅的风控（原有逻辑）
        2. MA5_BASED: 基于MA5的风控（新增逻辑）
        """
        logger.info(f"🔎 {stock_code} 开盘风控检查: buy_date={buy_date}, prev_close={prev_close}, buy_open={buy_open}")
        logger.info(f"📋 开盘风控策略: {self.config.entry_risk_control_strategy}")

        # ===== 根据配置选择策略 =====
        if self.config.entry_risk_control_strategy == EntryRiskControlStrategy.GAP_BASED:
            # ===== 原有策略: 基于涨跌幅的风控 =====
            if prev_close is None:
                logger.info(f"✅ {stock_code}: 无前日数据，允许买入")
                return {'can_buy': True, 'reason': '无前日数据'}

            gap_ratio = (buy_open - prev_close) / prev_close
            logger.info(f"📊 {stock_code}: 跳空计算 (buy_open={buy_open} - prev_close={prev_close}) / prev_close = {gap_ratio:.4f} ({gap_ratio:.2%})")

            down_threshold = (
                self.config.entry_risk_down_threshold
                if int(target_rank) == 1
                else self.config.entry_risk_down_threshold_non_top1
            )

            # 风控条件1: Top1 跌空超过3%，非Top1跌空超过5%
            if gap_ratio < down_threshold:
                logger.info(
                    f"🚫 {stock_code}: 跌空超过{abs(down_threshold):.0%} ({gap_ratio:.2%})，禁止买入"
                )
                return {
                    'can_buy': False,
                    'reason': f'开盘跳空{-gap_ratio:.2%}',
                    'gap_ratio': gap_ratio,
                    'skip_category': '开盘风控'
                }

            # 风控条件2: 高开超过5%
            elif gap_ratio > self.config.entry_risk_up_threshold:  # 使用0.05
                logger.info(f"🚫 {stock_code}: 高开超过5% ({gap_ratio:.2%})，禁止买入")
                return {
                    'can_buy': False,
                    'reason': f'开盘跳空{gap_ratio:.2%}',
                    'gap_ratio': gap_ratio,
                    'skip_category': '开盘风控'
                }

            logger.info(f"✅ {stock_code}: 通过所有风控检查，跳空{gap_ratio:.2%}在范围内")
            return {'can_buy': True, 'reason': '通过风控', 'gap_ratio': gap_ratio}

        elif self.config.entry_risk_control_strategy == EntryRiskControlStrategy.MA5_BASED:
            # ===== 🆕 新策略: 基于MA5的风控 =====
            # 计算MA5（前4天日线close + 买入日开盘价）
            ma5 = self._calculate_daily_ma5_at_open(stock_code, buy_date, buy_open)

            if ma5 is None:
                # 无法计算MA5，使用涨跌幅风控作为后备
                logger.warning(f"⚠️ {stock_code}: 无法计算MA5，使用涨跌幅风控作为后备")
                if prev_close is None:
                    logger.info(f"✅ {stock_code}: 无前日数据，允许买入")
                    return {'can_buy': True, 'reason': '无前日数据'}

                gap_ratio = (buy_open - prev_close) / prev_close
                # 简单涨跌幅风控
                if abs(gap_ratio) > 0.10:  # 跌幅或涨幅超10%才跳过
                    logger.info(f"🚫 {stock_code}: 跳空超10% ({gap_ratio:.2%})，禁止买入")
                    return {
                        'can_buy': False,
                        'reason': f'跳空超10%',
                        'gap_ratio': gap_ratio,
                        'skip_category': '开盘风控'
                    }

                return {'can_buy': True, 'reason': '后备涨跌幅风控通过', 'gap_ratio': gap_ratio}

            # 计算开盘价相对于MA5的涨跌幅
            ma5_gap_ratio = (buy_open - ma5) / ma5
            gap_ratio = (buy_open - prev_close) / prev_close
            logger.info(f"📊 {stock_code}: MA5跳空计算 (buy_open={buy_open} - MA5={ma5}) / MA5 = {ma5_gap_ratio:.4f} ({ma5_gap_ratio:.2%})")

            # 获取阈值
            down_threshold = self.config.entry_risk_ma5_down_threshold  # 默认-0.03
            up_threshold = self.config.entry_risk_ma5_up_threshold  # 默认0.03

            # 风控条件1: 低开超MA5 3%
            if ma5_gap_ratio < down_threshold:
                logger.info(f"🚫 {stock_code}: 低开超MA5 {abs(down_threshold):.0%} ({ma5_gap_ratio:.2%})，禁止买入")
                logger.info(f"   MA5: {ma5:.4f}, 开盘价: {buy_open:.4f}, 低开: {ma5_gap_ratio:.2%}")
                return {
                    'can_buy': False,
                    'reason': f'低开超MA5{abs(down_threshold):.0%}',
                    'gap_ratio': ma5_gap_ratio,
                    'skip_category': '开盘MA5风控',
                    'ma5_value': ma5  # 记录MA5值
                }

            # 风控条件2: 高开超MA5 3%
            elif gap_ratio > up_threshold:#ma5_gap_ratio > up_threshold:
                logger.info(f"🚫 {stock_code}: 高开超MA5 {up_threshold:.0%} ({ma5_gap_ratio:.2%})，禁止买入")
                logger.info(f"   MA5: {ma5:.4f}, 开盘价: {buy_open:.4f}, 高开: {ma5_gap_ratio:.2%}")
                return {
                    'can_buy': False,
                    'reason': f'高开超MA5{up_threshold:.0%}',
                    'gap_ratio': ma5_gap_ratio,
                    'skip_category': '开盘MA5风控',
                    'ma5_value': ma5  # 记录MA5值
                }

            logger.info(f"✅ {stock_code}: 通过所有风控检查，MA5跳空{ma5_gap_ratio:.2%}在范围内")
            return {
                'can_buy': True,
                'reason': '通过MA5风控',
                'gap_ratio': ma5_gap_ratio,
                'ma5_value': ma5  # 记录MA5值
            }

    def _get_daily_closes_before(self, stock_code: str, date: datetime.date, n: int) -> List[float]:
        """
        获取指定日期前N个交易日的日线close价格

        Args:
            stock_code: 股票代码
            date: 基准日期
            n: 需要获取的天数

        Returns:
            前N个交易日的close价格列表（按时间正序排列）
            如果数据不足，返回尽可能多的数据

        Example:
            如果 date=2026-01-10, n=4
            返回: [10.2, 10.3, 10.1, 10.4]  # 2026-01-06, 01-07, 01-08, 01-09的close
        """
        # 获取所有交易日期（已排序）
        all_dates = sorted(self.prediction_data['date'].unique())

        # 找到基准日期在交易日列表中的位置
        try:
            date_index = all_dates.index(date)
        except ValueError:
            # 如果基准日期不在交易日列表中，找最近的前一个交易日
            earlier_dates = [d for d in all_dates if d < date]
            if not earlier_dates:
                return []
            date = earlier_dates[-1]
            date_index = all_dates.index(date)

        # 获取前N个交易日
        start_index = max(0, date_index - n)
        target_dates = all_dates[start_index:date_index]

        # 获取这些日期的close价格
        closes = []
        for target_date in target_dates:
            daily_data = self.prediction_data[
                (self.prediction_data['date'] == target_date) &
                (self.prediction_data['stock_code'] == stock_code)
            ]
            if not daily_data.empty:
                closes.append(daily_data.iloc[0]['close'])
            else:
                logger.warning(f"⚠️ {stock_code} 在 {target_date} 无日线数据")

        logger.info(f"📊 {stock_code} 在{date}前{n}天的日线close: {closes}")
        return closes

    def _get_daily_low(self, stock_code: str, date: datetime.date) -> Optional[float]:
        """
        获取指定日期的日线最低价（用于T+1最低价跌破策略）

        Args:
            stock_code: 股票代码
            date: 指定日期（买入日/T+1日）

        Returns:
            指定日期的日线最低价，如果数据不存在返回None

        Example:
            如果 date=2026-01-10
            返回: 10.1  # 2026-01-10当天的最低价
        """
        try:
            date_str = pd.to_datetime(date).strftime('%Y%m%d')
            daily_file = Path(__file__).resolve().parents[2] / "local_data" / "features" / "daily" / f"{date_str}.parquet"

            if daily_file.exists():
                daily_df = pd.read_parquet(daily_file, columns=None)
                if 'ts_code' in daily_df.columns and 'stock_code' not in daily_df.columns:
                    daily_df = daily_df.rename(columns={'ts_code': 'stock_code'})
                row = daily_df[daily_df['stock_code'] == stock_code]
                if not row.empty:
                    low_value = row.iloc[0]['low'] if 'low' in row.columns and pd.notna(row.iloc[0]['low']) else row.iloc[0].get('close')
                    if pd.notna(low_value):
                        daily_low = float(low_value)
                        logger.info(f"📊 {stock_code} 在{date}的日线最低价(low): {daily_low:.4f}")
                        return daily_low

            # 回退到prediction_data
            daily_data = self.prediction_data[
                (self.prediction_data['date'] == date) &
                (self.prediction_data['stock_code'] == stock_code)
            ]

            if daily_data.empty:
                logger.warning(f"⚠️ {stock_code} 在 {date} 无日线数据")
                return None

            low_field = 'low' if 'low' in daily_data.columns else 'close'
            daily_low = float(daily_data.iloc[0][low_field])
            logger.info(f"📊 {stock_code} 在{date}的日线最低价({low_field}): {daily_low:.4f}")
            return daily_low

        except Exception as e:
            logger.error(f"❌ 获取{stock_code}在{date}的日线最低价失败: {e}")
            return None

    def _calculate_daily_ma5_realtime(self, stock_code: str, current_date: datetime.date,
                                      current_5min_close: float) -> Optional[float]:
        """
        实时计算日线MA5（前4天日线close + 当前5分钟close）

        Args:
            stock_code: 股票代码
            current_date: 当前日期
            current_5min_close: 当前5分钟K线的close价格

        Returns:
            MA5值，如果前4天数据不足返回None

        Note:
            MA5 = (前4天日线close之和 + 当前5分钟close) / 5

        Example:
            当前日期: 2026-01-10, 5分钟close: 10.5
            前4天日线close: [10.2, 10.3, 10.1, 10.4]
            MA5 = (10.2 + 10.3 + 10.1 + 10.4 + 10.5) / 5 = 10.3
        """
        # 获取前4个交易日的日线close
        daily_closes = self._get_daily_closes_before(stock_code, current_date, 4)

        # 检查数据是否充足
        if len(daily_closes) < 4:
            logger.info(f"⚠️ {stock_code} 在{current_date}前不足4个交易日数据")
            # return None

        # 计算MA5：前4天日线 + 当前5分钟
        ma5 = (sum(daily_closes) + current_5min_close) / 5.0

        logger.info(f"📊 {stock_code} 日线MA5计算:")
        logger.info(f"   前4天日线close: {daily_closes}")
        logger.info(f"   当前5分钟close: {current_5min_close}")
        logger.info(f"   MA5: {ma5:.4f}")

        return ma5

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
        effective_holding_days = int(getattr(position, 'holding_days', self.config.holding_days))
        end_index = min(
            buy_date_index + effective_holding_days,  # T+holding_days交易日
            current_date_index + 1  # 当前日（包含）
        )

        # 获取风控检查的实际日期范围
        # ⚠️ 关键修复：本函数每天都会被调用一次，若每次都把 T+2~当前日 的分钟数据重新拼接并从头扫描，
        # 会导致“历史分钟K线被重复检查”，而 position.highest_price_since_buy 等状态又是跨天累计的，
        # 从而在新的一天把上一交易日的第一根K线当成“第1根”重新计算，出现误触发。
        # 因此：每次只检查【当前交易日】的 5 分钟序列，状态（最高价/最高浮盈等）由 position 跨天携带。
        risk_check_dates = [current_date]
        logger.info(f"📊 {position.stock_code} 风控检查范围: {current_date} ~ {current_date} (当日{len(risk_check_dates)}个交易日)")

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
        logger.debug(f"🛡️ 开始统一风控检查（使用止损阈值: {self.config.stop_loss_threshold:.2%}）...")
        logger.info(f"📋 风控策略: {self.config.risk_control_strategy}")

        # 🆕 日线MA5策略状态变量
        consecutive_below_ma5 = 0  # 连续跌破MA5的次数
        last_check_date = None  # 上次检查的日期（用于跨天重置）


        # 初始化状态变量  对应T1_LOW_BREAK_WITH_HIGH_DRAWDOWN策略
        consecutive_below_t1_low = 0  # 连续跌破T+1日最低价的次数
        consecutive_bearish_in_profit = 0  # 🆕 盈利状态下连续阴线次数

        # 按时间顺序逐个K线检查，实现先满足先触发
        for i, minute_data in enumerate(minute_data_list):
            close_price = minute_data['close']
            high_price = minute_data['high']
            current_datetime = minute_data['datetime']

            # 提取当前K线的日期
            if isinstance(current_datetime, str):
                check_date = datetime.fromisoformat(current_datetime).date()
            elif hasattr(current_datetime, 'to_pydatetime'):
                check_date = current_datetime.to_pydatetime().date()
            elif isinstance(current_datetime, datetime):
                check_date = current_datetime.date()
            else:
                check_date = current_date

            # 🆕 跨天处理：如果日期变化，重置连续跌破计数
            if last_check_date is not None and check_date != last_check_date:
                if consecutive_below_ma5 > 0:
                    logger.info(f"📅 {position.stock_code}: 跨天检测，重置MA5跌破计数（从{consecutive_below_ma5}重置为0）")
                consecutive_below_ma5 = 0
            last_check_date = check_date

            # 1. 检查累计下跌/收益低于阈值（基于买入价，使用收盘价）
            decline_ratio = (close_price - position.buy_price) / position.buy_price
            if decline_ratio <= self.config.stop_loss_threshold:  # 使用配置的止损阈值
                # 处理不同的datetime格式
                if isinstance(current_datetime, str):
                    exit_time = datetime.fromisoformat(current_datetime)
                elif hasattr(current_datetime, 'to_pydatetime'):  # pandas Timestamp
                    exit_time = current_datetime.to_pydatetime()
                elif isinstance(current_datetime, datetime):
                    exit_time = current_datetime
                else:
                    exit_time = datetime.now()

                # 动态生成阈值描述
                threshold_desc = f"{self.config.stop_loss_threshold:.2%}"
                if self.config.stop_loss_threshold == 0.0:
                    threshold_desc = "0% (盈亏平衡)"
                elif self.config.stop_loss_threshold > 0:
                    threshold_desc = f"{self.config.stop_loss_threshold:.2%} (要求正收益)"

                logger.info(f"🚨 {position.stock_code}: 在第{i+1}根K线触发止损!")
                logger.debug(f"   止损K线: 时间={current_datetime}, 开盘={minute_data['open']:.4f}, 收盘={close_price:.4f}")
                logger.debug(f"   买入价: {position.buy_price:.4f}, 收盘价: {close_price:.4f}, 收益: {decline_ratio:.2%}")
                logger.debug(f"   止损阈值: {threshold_desc}")

                return {
                    'should_exit': True,
                    'reason': f'累计收益低于{threshold_desc}',
                    'exit_price': close_price,  # 使用收盘价成交
                    'exit_time': exit_time,
                    'decline_ratio': decline_ratio,
                    'stop_loss_threshold': self.config.stop_loss_threshold,  # 记录使用的阈值
                }

            # ===== 风控条件2: 根据配置选择策略 =====
            if self.config.risk_control_strategy == RiskControlStrategy.CONSECUTIVE_BEARISH:
                # ===== 原有策略: 连续阴线风控 =====
                if i >= 2:  # 至少有3根K线可以检查连续性
                    # 获取最近的3根K线：第i-2, i-1, i根
                    # candle_0 = minute_data_list[i -3]
                    candle_1 = minute_data_list[i - 2]
                    candle_2 = minute_data_list[i - 1]
                    candle_3 = minute_data

                    # 检查这3根K线是否都是阴线
                    # is_red_0 = candle_0['close'] < candle_0['open']
                    is_red_1 = candle_1['close'] < candle_1['open']
                    is_red_2 = candle_2['close'] < candle_2['open']
                    is_red_3 = candle_3['close'] < candle_3['open']

                    if is_red_1 and is_red_2 and is_red_3: #is_red_0 and 
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

            elif self.config.risk_control_strategy == RiskControlStrategy.DAILY_MA5_BREAK:
                # ===== 🆕 新风控：MA5跌破 + 高位回撤双条件策略 =====

                # 初始化退出标记和结果字典
                should_exit_flag = False
                exit_result = {}

                # 1️⃣ 计算当前收益和更新最高价
                current_return = (close_price - position.buy_price) / position.buy_price

                # 更新买入后的最高价和最高浮盈
                if high_price > position.highest_price_since_buy:
                    position.highest_price_since_buy = high_price
                    position.highest_return_since_buy = (high_price - position.buy_price) / position.buy_price

                # 2️⃣ 计算动态MA5（前4天日线 + 当前5分钟）
                ma5 = self._calculate_daily_ma5_realtime(
                    position.stock_code,
                    check_date,
                    close_price
                )

                if ma5 is None:
                    logger.info(f"⏸️ {position.stock_code}: 无法计算MA5，跳过检查")
                    # ✅ 修改：不直接return，而是跳过本次检查继续循环
                    continue

                # 3️⃣ 条件1：MA5跌破 + 当前收益≤0
                is_below_ma5 = close_price < ma5
                current_return_negative = current_return <= 0

                if is_below_ma5:
                    consecutive_below_ma5 += 1
                    logger.info(f"📉 {position.stock_code} 第{i+1}根K线:")
                    logger.info(f"   close({close_price:.4f}) < MA5({ma5:.4f}), 收益:{current_return*100:.2f}%")
                    logger.info(f"   连续跌破MA5: {consecutive_below_ma5}次")
                else:
                    if consecutive_below_ma5 > 0:
                        logger.info(f"📈 {position.stock_code} 第{i+1}根K线:")
                        logger.info(f"   close({close_price:.4f}) >= MA5({ma5:.4f}), 收益:{current_return*100:.2f}%")
                        logger.info(f"   重置跌破计数（从{consecutive_below_ma5}重置为0）")
                    consecutive_below_ma5 = 0

                # 检查条件1是否满足（亏损或持平状态 + 连续跌破）
                condition1_trigger = (current_return_negative and
                                  consecutive_below_ma5 >= self.config.risk_control_ma5_consecutive_breaks)

                # 4️⃣ 条件2：高位回撤
                # 判断是否曾经达到最高浮盈阈值（4%）
                highest_return_above_threshold = position.highest_return_since_buy >= self.config.high_return_threshold

                condition2_trigger = False
                if highest_return_above_threshold:
                    # 从最高点回撤超过3%（-0.03）
                    drawdown_from_highest = (close_price - position.highest_price_since_buy) / position.highest_price_since_buy
                    condition2_trigger = (drawdown_from_highest <= self.config.high_drawdown_threshold)

                    if condition2_trigger:
                        logger.info(f"📉 {position.stock_code} 第{i+1}根K线:")
                        logger.info(f"   close({close_price:.4f}), 最高:{position.highest_price_since_buy:.4f}")
                        logger.info(f"   最高浮盈:{position.highest_return_since_buy*100:.2f}%")
                        logger.info(f"   回撤:{abs(drawdown_from_highest)*100:.2f}%")

                # 5️⃣ 判断是否触发卖出（两个条件OR关系）
                if condition1_trigger or condition2_trigger:
                    # ✅ 修改：不直接return，而是记录退出状态
                    should_exit_flag = True

                    exit_reason_parts = []

                    if condition1_trigger:
                        exit_reason_parts.append(
                            f"亏损状态下连续{consecutive_below_ma5}次跌破MA5"
                        )

                    if condition2_trigger:
                        drawdown_from_highest = (close_price - position.highest_price_since_buy) / position.highest_price_since_buy
                        exit_reason_parts.append(
                            f"最高浮盈{position.highest_return_since_buy*100:.2f}%后回撤{abs(drawdown_from_highest)*100:.2f}%"
                        )

                    exit_reason = " | ".join(exit_reason_parts)

                    # 记录退出信息（稍后return）
                    # 标准化退出时间
                    # 处理不同的datetime格式
                    if isinstance(current_datetime, str):
                        exit_time_standardized = datetime.fromisoformat(current_datetime)
                    elif hasattr(current_datetime, 'to_pydatetime'):  # pandas Timestamp
                        exit_time_standardized = current_datetime.to_pydatetime()
                    elif isinstance(current_datetime, datetime):
                        exit_time_standardized = current_datetime
                    else:
                        exit_time_standardized = datetime.now()

                    exit_result = {
                        'should_exit': True,
                        'reason': exit_reason,
                        'exit_price': close_price,
                        'exit_time': exit_time_standardized,  # 标准化时间
                        'ma5_value': ma5,
                        'current_return_pct': current_return * 100,
                        'highest_return_pct': position.highest_return_since_buy * 100,
                        'drawdown_from_highest_pct': drawdown_from_highest * 100 if condition2_trigger else None
                    }
                    return exit_result

                    # ✅ 修改：触发后立即退出循环，不再检查后续K线
                continue

            elif self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
                # ===== 🆕 新风控：T+1最低价跌破 + 高位回撤双状态策略 =====

                close_price = minute_data['close']
                high_price = minute_data['high']
                current_datetime = minute_data['datetime']

                # 提取当前K线的日期
                if isinstance(current_datetime, str):
                    check_date = datetime.fromisoformat(current_datetime).date()
                elif hasattr(current_datetime, 'to_pydatetime'):
                    check_date = current_datetime.to_pydatetime().date()
                elif isinstance(current_datetime, datetime):
                    check_date = current_datetime.date()
                else:
                    check_date = current_date

                # 1. 计算当前收益和更新最高价
                current_return = (close_price - position.buy_price) / position.buy_price

                # 更新买入后的最高价和最高浮盈
                if high_price > position.highest_price_since_buy:
                    position.highest_price_since_buy = high_price
                    position.highest_return_since_buy = (high_price - position.buy_price) / position.buy_price

                # ===== 状态判断：亏损/持平 vs 盈利 =====
                if current_return <= 0:
                    # ===== 状态A：亏损/持平状态 - T+1最低价跌破策略 =====
                    continue
                    # 检查是否跌破T+1日最低价
                    is_below_t1_low = close_price < position.t1_day_lowest_price

                    if is_below_t1_low:
                        consecutive_below_t1_low += 1
                        logger.info(f"📉 {position.stock_code} 第{i+1}根K线:")
                        logger.info(f"   close({close_price:.4f}) < T+1最低价({position.t1_day_lowest_price:.4f})")
                        logger.info(f"   收益:{current_return*100:.2f}%")
                        logger.info(f"   连续跌破T+1最低价: {consecutive_below_t1_low}次")
                    else:
                        if consecutive_below_t1_low > 0:
                            logger.info(f"📈 {position.stock_code} 第{i+1}根K线:")
                            logger.info(f"   close({close_price:.4f}) >= T+1最低价({position.t1_day_lowest_price:.4f})")
                            logger.info(f"   重置跌破计数（从{consecutive_below_t1_low}重置为0）")
                        consecutive_below_t1_low = 0

                    # 检查是否触发卖出：连续跌破T+1日最低价
                    if consecutive_below_t1_low >= self.config.risk_control_t1_low_consecutive_breaks:
                        logger.info(f"🚨 {position.stock_code}: 在第{i+1}根K线触发T+1最低价跌破风控!")
                        logger.debug(f"   触发K线: 时间={current_datetime}, close={close_price:.4f}")
                        logger.debug(f"   T+1最低价: {position.t1_day_lowest_price:.4f}")
                        logger.debug(f"   连续跌破次数: {consecutive_below_t1_low}")
                        logger.debug(f"   当前收益: {current_return*100:.2f}%")

                        # 处理不同的datetime格式
                        if isinstance(current_datetime, str):
                            exit_time = datetime.fromisoformat(current_datetime)
                        elif hasattr(current_datetime, 'to_pydatetime'):
                            exit_time = current_datetime.to_pydatetime()
                        elif isinstance(current_datetime, datetime):
                            exit_time = current_datetime
                        else:
                            exit_time = datetime.now()

                        return {
                            'should_exit': True,
                            'reason': f'亏损状态下连续{consecutive_below_t1_low}次跌破T+1最低价',
                            'exit_price': close_price,
                            'exit_time': exit_time,
                            't1_lowest_price': position.t1_day_lowest_price,
                            'current_return_pct': current_return * 100,
                            'consecutive_breaks': consecutive_below_t1_low
                        }
                else:
                    # ===== 状态B：盈利状态 - 峰值回撤 + 阴线数量/跌幅确认 =====

                    # 检查当前K线是否为阴线
                    is_bearish = close_price < minute_data['open']
                    bar_return = (close_price - minute_data['open']) / minute_data['open'] if minute_data['open'] else 0.0

                    if is_bearish:
                        consecutive_bearish_in_profit += 1
                        logger.info(f"📉 {position.stock_code} 第{i+1}根K线(盈利状态):")
                        logger.info(f"   open={minute_data['open']:.4f}, close={close_price:.4f} (阴线)")
                        logger.info(f"   当前收益:{current_return*100:.2f}%")
                        logger.info(f"   单根跌幅:{bar_return*100:.2f}%")
                        logger.info(f"   连续阴线: {consecutive_bearish_in_profit}次")
                    else:
                        # 阳线，重置计数
                        if consecutive_bearish_in_profit > 0:
                            logger.info(f"📈 {position.stock_code} 第{i+1}根K线(盈利状态):")
                            logger.info(f"   open={minute_data['open']:.4f}, close={close_price:.4f} (阳线)")
                            logger.info(f"   重置阴线计数（从{consecutive_bearish_in_profit}重置为0）")
                        consecutive_bearish_in_profit = 0

                    # 只有先形成了足够浮盈，且从最高点出现明显回撤，才允许用连续阴线止盈。
                    drawdown_from_peak = 0.0
                    if position.highest_price_since_buy and position.highest_price_since_buy > 0:
                        drawdown_from_peak = (
                            close_price - position.highest_price_since_buy
                        ) / position.highest_price_since_buy

                    bearish_profit_exit_ready = (
                        position.highest_return_since_buy >= self.config.profit_bearish_min_peak_return
                        and drawdown_from_peak <= self.config.profit_bearish_trailing_drawdown
                        and current_return >= self.config.profit_bearish_min_current_return
                    )

                    fast_bearish_exit = False
                    slow_bearish_exit = False

                    if bearish_profit_exit_ready:
                        # 快速转弱：最近两根都是阴线，且至少一根属于中阴线，两根合计跌幅明显
                        if i >= 1:
                            prev_bar = minute_data_list[i - 1]
                            prev_is_bearish = prev_bar['close'] < prev_bar['open']
                            prev_bar_return = (
                                (prev_bar['close'] - prev_bar['open']) / prev_bar['open']
                                if prev_bar['open'] else 0.0
                            )
                            if (
                                is_bearish
                                and prev_is_bearish
                                and min(bar_return, prev_bar_return) <= self.config.profit_bearish_min_single_drop
                                and (bar_return + prev_bar_return) <= self.config.profit_bearish_two_bar_combined_drop
                            ):
                                fast_bearish_exit = True

                        # 缓慢转弱：连续三根实质阴线
                        if consecutive_bearish_in_profit >= self.config.profit_bearish_required_count and i >= 2:
                            recent_three = minute_data_list[i - 2:i + 1]
                            three_returns = []
                            all_bearish = True
                            for candle in recent_three:
                                if candle['close'] >= candle['open']:
                                    all_bearish = False
                                    break
                                one_ret = (
                                    (candle['close'] - candle['open']) / candle['open']
                                    if candle['open'] else 0.0
                                )
                                three_returns.append(one_ret)
                            if all_bearish and all(
                                one_ret <= self.config.profit_bearish_slow_single_drop
                                for one_ret in three_returns
                            ):
                                slow_bearish_exit = True

                    # 检查是否触发卖出：快速转弱 or 缓慢转弱
                    if fast_bearish_exit or slow_bearish_exit:
                        trigger_type = "快速转弱" if fast_bearish_exit else "连续阴线"
                        logger.info(f"🚨 {position.stock_code}: 在第{i+1}根K线触发{trigger_type}风控(盈利状态)!")
                        logger.info(f"   触发K线: 时间={current_datetime}")
                        logger.info(f"   连续阴线次数: {consecutive_bearish_in_profit}")
                        logger.info(f"   最高浮盈: {position.highest_return_since_buy*100:.2f}%")
                        logger.info(f"   当前收益: {current_return*100:.2f}%")
                        logger.info(f"   相对峰值回撤: {drawdown_from_peak*100:.2f}%")
                        logger.info(f"   当前K线跌幅: {bar_return*100:.2f}%")

                        # 处理不同的datetime格式
                        if isinstance(current_datetime, str):
                            exit_time = datetime.fromisoformat(current_datetime)
                        elif hasattr(current_datetime, 'to_pydatetime'):
                            exit_time = current_datetime.to_pydatetime()
                        elif isinstance(current_datetime, datetime):
                            exit_time = current_datetime
                        else:
                            exit_time = datetime.now()

                        return {
                            'should_exit': True,
                            'reason': (
                                f'盈利状态下快速转弱'
                                if fast_bearish_exit else
                                f'盈利状态下连续{consecutive_bearish_in_profit}根阴线'
                            ),
                            'exit_price': close_price,
                            'exit_time': exit_time,
                            'current_return_pct': current_return * 100,
                            'consecutive_bearish': consecutive_bearish_in_profit,
                            'highest_return_since_buy_pct': position.highest_return_since_buy * 100,
                            'drawdown_from_peak_pct': drawdown_from_peak * 100,
                        }

                # end for - T+1最低价跌破策略循环结束



        # 根据策略显示不同的日志信息
        if self.config.risk_control_strategy == RiskControlStrategy.CONSECUTIVE_BEARISH:
            logger.info(f"✅ {position.stock_code}: 未发现风控信号（止损阈值: {self.config.stop_loss_threshold:.2%} 或 连续3根阴线）")
        elif self.config.risk_control_strategy == RiskControlStrategy.DAILY_MA5_BREAK:
            logger.info(f"✅ {position.stock_code}: 未发现风控信号（止损阈值: {self.config.stop_loss_threshold:.2%} 或 日线MA5跌破）")
        elif self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
            logger.info(f"✅ {position.stock_code}: 未发现风控信号（止损阈值: {self.config.stop_loss_threshold:.2%} 或 T+1最低价跌破/盈利状态连续阴线）")
        else:
            logger.info(f"✅ {position.stock_code}: 未发现风控信号（止损阈值: {self.config.stop_loss_threshold:.2%}）")
        return {'should_exit': False, 'reason': '无风控信号'}



    def select_top_k_stocks(self, filtered_df: pd.DataFrame, k: int) -> pd.DataFrame:
        """选择Top-K股票"""
        if len(filtered_df) < k:
            return filtered_df
        return filtered_df.nlargest(k, 'predicted_return')

    def add_trading_days(self, start_date: datetime.date, n: int,
                        trading_dates: List[datetime.date]) -> Optional[datetime.date]:
        """
        从start_date开始，计算n个交易日后的日期

        Args:
            start_date: 起始日期
            n: 要增加的交易日数量
            trading_dates: 完整的交易日期列表

        Returns:
            n个交易日后的日期，如果start_date不在trading_dates中则返回None
        """
        try:
            start_index = trading_dates.index(start_date)
            end_index = start_index + n

            # 边界检查：确保索引不超过列表长度
            if end_index < len(trading_dates):
                return trading_dates[end_index]
            else:
                # 如果超出范围，返回最后一个交易日
                logger.warning(f"⚠️ 到期日超出交易日期范围，使用最后一个交易日: {trading_dates[-1]}")
                return trading_dates[-1]
        except ValueError:
            # 如果买入日期不在trading_dates中，返回None
            logger.error(f"❌ 买入日期 {start_date} 不在交易日期列表中")
            return None

    def _is_last_trade_stop_loss(self) -> bool:
        """
        检查上一次交易是否触发5%止损

        Returns:
            True: 上次交易止损（≤ -5%）
            False: 上次交易未止损，或无历史交易
        """
        if not self.trades:
            return False

        last_trade = self.trades[-1]
        actual_return = last_trade.get('actual_return', 0.0)

        # 方式1: 检查收益率是否 ≤ -5%（使用配置的止损阈值）
        if actual_return <= self.config.stop_loss_threshold:
            logger.info(f"🛡️ [止损检测] 上次交易触发{self.config.stop_loss_threshold*100:.1f}%止损: "
                       f"{last_trade['stock_code']}, return={actual_return:.4f}")
            return True

        # 方式2: 检查退出原因是否包含"止损"字样
        exit_reason = last_trade.get('exit_reason', '')
        if '止损' in exit_reason or 'stop_loss' in exit_reason.lower():
            logger.info(f"🛡️ [止损检测] 上次交易触发止损(原因): {last_trade['stock_code']}, reason={exit_reason}")
            return True

        return False


    def _has_consecutive_losses(self, n: int) -> bool:
        """
        检查是否连续N次交易都是负收益

        Args:
            n: 连续次数

        Returns:
            True: 最近连续N次都是亏损
            False: 不足N次，或有盈利
        """
        if len(self.trades) < n:
            return False

        # 获取最近N次交易
        recent_trades = self.trades[-n:]

        # 检查是否都是负收益
        all_losses = all((trade.get('actual_return', 0.0) < 0 and trade.get('actual_return') > self.config.stop_loss_threshold) for trade in recent_trades)

        if all_losses:
            returns = [trade.get('actual_return', 0.0) for trade in recent_trades]
            logger.info(f"📉 [连续亏损] 检测到连续{n}次亏损: {[f'{r*100:.2f}%' for r in returns]}")

        return all_losses


    def _has_consecutive_profits(self, n: int) -> bool:
        """
        检查是否连续N次交易都盈利

        Args:
            n: 连续次数

        Returns:
            True: 最近连续N次都是盈利
            False: 不足N次，或有亏损
        """
        if len(self.trades) < n:
            return False

        # 获取最近N次交易
        recent_trades = self.trades[-n:]

        # 检查是否都盈利（actual_return > 0）
        all_profits = all(trade.get('actual_return', 0.0) > 0 for trade in recent_trades)

        if all_profits:
            returns = [trade.get('actual_return', 0.0) for trade in recent_trades]
            logger.info(f"📈 [连续盈利] 检测到连续{n}次盈利: {[f'{r*100:.2f}%' for r in returns]}")

        return all_profits


    def _get_position_limit_by_trade_history(self) -> float:
        """根据最近交易结果动态调整 lead_z 的过滤阈值。

        状态机（用来做简易的“风控/降频”）：
          - NORMAL: 默认阈值较宽（更容易开仓）
          - DEFENSIVE1: 小幅收紧
          - DEFENSIVE: 大幅收紧（连续亏损后）

        触发逻辑：
          - 若上一笔交易触发 stop_loss -> 进入 DEFENSIVE1
          - 若连续亏损 >= 3 -> 进入 DEFENSIVE
          - 若在 DEFENSIVE/DEFENSIVE1 且连续盈利 >= 2 -> 回到 NORMAL

        返回值：
          - NORMAL     -> -0.4
          - DEFENSIVE1 ->  0.0
          - DEFENSIVE  ->  0.3
        """
        # 默认值
        base_normal = -0.4
        base_defensive1 = 0.0
        base_defensive = 0.3

        try:
            trade_history = getattr(self, "trade_history", []) or []
            if len(trade_history) == 0:
                return base_normal

            last_trade = trade_history[-1]
            last_reason = str(last_trade.get("exit_reason", "")).lower()

            # 连亏/连盈统计（从末尾往前数）
            consecutive_losses = 0
            consecutive_profits = 0
            for t in reversed(trade_history):
                ret = float(t.get("return_pct", 0.0))
                if ret <= 0:
                    consecutive_losses += 1
                    consecutive_profits = 0
                else:
                    consecutive_profits += 1
                    consecutive_losses = 0
                if consecutive_losses >= 3 or consecutive_profits >= 2:
                    break

            current_state = getattr(self, "position_control_state", "NORMAL")

            # 状态迁移
            if "stop_loss" in last_reason:
                current_state = "DEFENSIVE1"
            if consecutive_losses >= 3:
                current_state = "DEFENSIVE"
            if current_state in ("DEFENSIVE", "DEFENSIVE1") and consecutive_profits >= 2:
                current_state = "NORMAL"

            self.position_control_state = current_state

            # 阈值输出
            if current_state == "DEFENSIVE":
                limit = base_defensive
            elif current_state == "DEFENSIVE1":
                limit = base_defensive1
            else:
                limit = base_normal

            logger.info(
                f"📊 [仓位控制/lead_z] 状态={current_state}, 连亏={consecutive_losses}, 连赢={consecutive_profits}, z_limit={limit:.2f}"
            )
            return float(limit)
        except Exception as e:
            logger.warning(f"⚠️ 仓位控制状态机异常，使用默认阈值 {base_normal:.2f}: {e}")
            return float(base_normal)


    def _position_pct_by_lead(self, lead_z: float, trade_date: Optional[datetime.date] = None) -> float:
        """根据 lead_z（topK 内相对优势的鲁棒 z-score）给出仓位比例。

        lead_z 计算（使用过去 rolling_window 的历史 lead_raw，不含当天，避免 look-ahead）：
            lead_raw = top1_score - mean(topK_scores)
            med = median(hist)
            mad = median(|hist - med|)
            lead_z = (lead_raw - med) / (1.4826 * mad + eps)

        仓位/过滤规则（与历史交易结果联动）：
            - 通过历史交易结果得到动态阈值 z_limit（默认 -0.4；防御模式更严格为 0.0）
            - 若 lead_z < z_limit：跳过（0%）
            - 否则：全仓（100%）

        说明：
            你之前用 lead_raw 的固定阈值容易受模型尺度漂移影响；lead_z 通过滚动 robust 标准化减敏。
        """
        try:
            z = float(lead_z)
        except Exception:
            return 0.0

        z_limit = float(self._get_position_limit_by_trade_history())

        # if z < z_limit:
        #     return 0.0
        return 1.0

    def _resolve_position_pct(self, slot_index: int) -> float:
        """按固定槽位身份返回本次开仓目标仓位。"""
        # schedule = list(self.config.position_schedule or [])
        # if schedule:
        #     resolved_slot_index = max(0, int(slot_index))
        #     if resolved_slot_index < len(schedule):
        #         pct = float(schedule[resolved_slot_index])
        #     else:
        #         pct = float(schedule[-1])
        # else:
        #     pct = float(self.config.fixed_position_pct)
        # return max(0.0, min(1.0, pct))
        return 1.0

    def _get_first_free_slot_index(self, positions: Optional[Dict[str, "Position"]] = None) -> Optional[int]:
        """返回当前固定槽位体系下的第一个空槽位。"""
        positions = self.current_holdings if positions is None else positions
        occupied = {
            int(pos.slot_index) for pos in positions.values()
            if hasattr(pos, 'slot_index') and pos.slot_index is not None
        }
        max_slots = int(max(1, self.config.max_concurrent_positions))
        for slot_index in range(max_slots):
            if slot_index not in occupied:
                return slot_index
        return None

    def _robust_mad(self, values: list[float]) -> float:
        """MAD = median(|x - median(x)|). values 需为非空。"""
        if not values:
            return 0.0
        import numpy as np
        arr = np.asarray(values, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return mad

    def _calc_lead_z(self, lead_raw: float, rolling_window: int = 60, eps: float = 1e-9) -> float:
        """用历史 lead_raw（不含当前 lead_raw）计算鲁棒 z-score。"""
        try:
            x = float(lead_raw)
        except Exception:
            return 0.0

        hist = getattr(self, "_lead_history", [])
        if not hist:
            return 0.0

        # 只取最近 rolling_window
        h = [float(v) for v in hist[-int(rolling_window):] if v is not None]
        if len(h) < 10:
            return 0.0

        import numpy as np
        arr = np.asarray(h, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        denom = 1.4826 * mad + float(eps)
        return (x - med) / denom

    def _update_lead_history(self, lead_raw: float, rolling_window: int = 60) -> None:
        """更新 lead_raw 历史（滚动窗口）。"""
        try:
            x = float(lead_raw)
        except Exception:
            return
        if not hasattr(self, "_lead_history"):
            self._lead_history = []
        self._lead_history.append(x)
        # trim
        if len(self._lead_history) > int(rolling_window) * 2:
            # 保留更长一点，避免频繁 slice
            self._lead_history = self._lead_history[-int(rolling_window) * 2:]

    def execute_buy(self, stock_code: str, buy_date: datetime.date,
                    predicted_return: float, trading_dates: List[datetime.date],
                    position_pct: float, entry_reason: str = "normal",
                    holding_days_override: Optional[int] = None,
                    day_start_cash: Optional[float] = None,
                    day_start_equity: Optional[float] = None,
                    slot_index: int = 0) -> Optional[Position]:
        """执行买入操作"""
        logger.info(f"🛒 准备买入 {stock_code}, 预测收益: {predicted_return:.4f}")

        if stock_code in self.current_holdings:
            logger.warning(f"⚠️ {stock_code} 已在持仓中，跳过重复开仓，避免覆盖原持仓")
            return None

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
        # 🆕 新逻辑：所有策略都每天只买入1只股票（指定排名的那只），应该全仓买入

        # 检查现金是否足够购买至少1手
        min_cost = buy_price * 100  # 至少买1手的成本
        if self.cash < min_cost:
            logger.warning(f"⚠️ 现金不足，无法买入{stock_code}: 可用现金={self.cash:.2f}, 最少需要={min_cost:.2f}")
            return None

        position_pct = float(position_pct)
        position_pct = max(0.0, min(1.0, position_pct))
        day_start_cash = float(self.cash if day_start_cash is None else day_start_cash)
        holdings_cost = sum(p.buy_price * p.quantity for p in self.current_holdings.values())
        reference_equity = float(
            (self.cash + holdings_cost) if day_start_equity is None else day_start_equity
        )
        position_amount = min(day_start_cash, reference_equity * position_pct)
        logger.info(
            f"📊 固定仓位买入: 日初现金={day_start_cash:.2f}, 当前现金={self.cash:.2f}, "
            f"持仓成本={holdings_cost:.2f}, 日初参考权益={reference_equity:.2f}, "
            f"目标仓位={position_pct:.0%}, 分配金额={position_amount:.2f}"
        )
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

        effective_holding_days = int(holding_days_override or self.config.holding_days)
        effective_holding_days = max(1, effective_holding_days)

        # 计算持仓到期日 - 使用交易日而非自然日
        holding_end_date = self.add_trading_days(buy_date, effective_holding_days - 1, trading_dates)

        # 如果计算失败（买入日期不在交易列表中），回退到自然日计算
        if holding_end_date is None:
            logger.warning(f"⚠️ 无法通过交易日计算到期日，使用自然日计算")
            holding_end_date = buy_date + timedelta(days=effective_holding_days - 1)

        # 计算实际持有的交易日数量（用于日志）
        try:
            buy_index = trading_dates.index(buy_date)
            end_index = trading_dates.index(holding_end_date)
            actual_trading_days = end_index - buy_index + 1
            logger.info(f"📅 持仓计划: 买入日={buy_date} (交易日索引#{buy_index}), "
                       f"到期日={holding_end_date} (交易日索引#{end_index}), "
                       f"持有{actual_trading_days}个交易日 (配置: {effective_holding_days}天)")
        except (ValueError, Exception):
            logger.info(f"📅 持仓计划: 买入日={buy_date}, 到期日={holding_end_date} (配置: {effective_holding_days}天)")

        # 🆕 获取T+1日（买入日）的最低价（用于T+1最低价跌破策略）
        t1_day_lowest_price = 0.0
        if self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
            t1_day_lowest_price = self._get_daily_low(stock_code, buy_date)
            if t1_day_lowest_price is None:
                logger.warning(f"⚠️ 无法获取{stock_code}在买入日{buy_date}的最低价，使用0.0")
                t1_day_lowest_price = 0.0

        # 创建持仓记录
        position = Position(
            stock_code=stock_code,
            buy_date=buy_date,
            buy_datetime=buy_datetime,
            buy_price=buy_price,
            highest_price_since_buy=buy_price,
            quantity=quantity,
            predicted_return=predicted_return,
            holding_end_date=holding_end_date,
            entry_reason=entry_reason,
            holding_days=effective_holding_days,
            slot_index=int(slot_index),
            t1_day_lowest_price=t1_day_lowest_price  # 🆕 记录T+1日最低价
        )

        self.current_holdings[stock_code] = position
        self.cash -= buy_price * quantity

        self.last_buy_date = buy_date
        self.current_buy_cycle_date = buy_date
        self.is_holding = bool(self.current_holdings)
        self.current_holding_end_date = max(
            p.holding_end_date for p in self.current_holdings.values() if p.holding_end_date is not None
        )

        logger.info(f"✅ 买入成功: {stock_code}, 价格: {buy_price:.4f}, 数量: {quantity}, 时间: {buy_datetime}, 剩余现金: {self.cash:.2f}")
        logger.info(
            f"📅 买入后持仓状态: 槽位={position.slot_index + 1}/{self.config.max_concurrent_positions}, "
            f"持仓数={len(self.current_holdings)}/{self.config.max_concurrent_positions}, "
            f"最晚到期日={self.current_holding_end_date}"
        )

        return position

    def _should_reset_t_date(self, trade_date: datetime.date, trading_dates: List[datetime.date]) -> bool:
        """
        判断是否应该重置T日（空仓优化）

        重置条件：
        1. 当前没有持仓（is_holding = False）
        2. 距离上次T日更新已经过了一段时间（避免频繁重置）
        3. 不是在卖出当天（避免重复重置）

        Args:
            trade_date: 当前交易日
            trading_dates: 完整的交易日列表

        Returns:
            bool: 是否应该重置T日
        """
        # 如果没有current_t_date，需要初始化
        if not hasattr(self, 'current_t_date') or self.current_t_date is None:
            return True

        # 🆕 关键修复：如果T日就是当前交易日，说明刚在卖出时重置过，不需要再次重置
        if self.current_t_date == trade_date:
            logger.debug(f"📅 [重置判断] T日={trade_date}，刚在卖出时重置过 → 不重置")
            return False

        # 获取当前交易日的索引
        try:
            current_index = trading_dates.index(trade_date)
        except ValueError:
            return False

        # 获取T日的索引
        try:
            t_index = trading_dates.index(self.current_t_date)
        except ValueError:
            # T日不在交易日列表中，需要重置
            return True

        days_since_t = current_index - t_index

        # 🆕 核心判断逻辑：
        # 如果距离T日已经超过1天（已过T+1买入窗口），且没有持仓
        # 说明上一轮的买入机会已经完全错过，应该立即重置
        if days_since_t > 1 and not self.is_holding:
            logger.info(f"📅 [重置判断] 距T日{days_since_t}天，已过T+1窗口，且无持仓 → 应该重置")
            return True

        # 如果正好是T+1日，但在空仓状态（可能之前全部跳过）
        # 不需要重置，应该尝试买入
        if days_since_t == 1 and not self.is_holding:
            logger.info(f"📅 [重置判断] T+1日空仓，应该尝试买入 → 不重置")
            # 🆕 关键修复：不在判断时设置标志，而是在真正尝试买入时设置
            # 这样可以避免"判断为T+1日但实际未买入"的误判
            return False

        return False

    def _get_previous_trading_day(self, trade_date: datetime.date, trading_dates: List[datetime.date]) -> Optional[datetime.date]:
        """获取前一个交易日"""
        try:
            current_index = trading_dates.index(trade_date)
            if current_index > 0:
                return trading_dates[current_index - 1]
        except ValueError:
            pass
        return None

    def _should_attempt_buy(self, trade_date: datetime.date, trading_dates: List[datetime.date],
                            day_start_open_slots: Optional[int] = None) -> Dict[str, Any]:
        """检查当前交易日是否应该尝试买入。

        并行持仓版本使用“当日开盘前的可用槽位”决定是否允许买入。
        如果槽位是当日先卖出才腾出来的，则当天不能复用，只能从下一交易日开始使用。
        """
        try:
            max_slots = int(max(1, self.config.max_concurrent_positions))
            if day_start_open_slots is None:
                day_start_open_slots = max_slots - len(self.current_holdings)
            open_slots = int(day_start_open_slots)
            logger.debug(
                f"🤔 判断是否买入: trade_date={trade_date}, holdings={len(self.current_holdings)}, "
                f"day_start_open_slots={open_slots}/{max_slots}"
            )

            if open_slots <= 0:
                return {
                    'should_buy': False,
                    'reason': f'当日开盘前无可用槽位({max_slots - open_slots}/{max_slots})',
                    'prediction_date': None
                }

            prediction_dates = pd.to_datetime(self.prediction_data['date']).dt.date
            prediction_dates = sorted(set(prediction_dates.tolist()))
            earlier_dates = [d for d in prediction_dates if d < trade_date]
            if not earlier_dates:
                return {'should_buy': False, 'reason': '无前一交易日预测数据', 'prediction_date': None}

            prediction_date = max(earlier_dates)
            logger.info(
                f"✅ 并行持仓买入: 当日开盘前可用槽位{open_slots}/{max_slots}, 当前持仓{len(self.current_holdings)}/{max_slots}, "
                f"T日={prediction_date}, T+1日={trade_date}"
            )
            return {'should_buy': True, 'reason': '并行持仓T+1买入', 'prediction_date': prediction_date}

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

        if not self.current_holdings:
            logger.info("🔄 所有持仓已清空，重置买入周期状态")
            self.is_holding = False
            self.current_holding_end_date = None
            self.last_position_end_date = sell_date
        else:
            self.is_holding = True
            self.current_holding_end_date = max(
                p.holding_end_date for p in self.current_holdings.values() if p.holding_end_date is not None
            )

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
            day_start_positions = dict(self.current_holdings)
            day_start_holdings_count = len(self.current_holdings)
            day_start_open_slots = max(0, int(self.config.max_concurrent_positions) - day_start_holdings_count)
            day_start_free_slot_index = self._get_first_free_slot_index(day_start_positions)
            day_start_cash = float(self.cash)
            day_start_holdings_cost = sum(
                p.buy_price * p.quantity for p in self.current_holdings.values()
            )
            day_start_equity = day_start_cash + day_start_holdings_cost

            logger.info(f"📅 交易日 {i+1}/{len(trading_dates)}: {trade_date}")
            logger.info(f"💼 当前持仓: {len(self.current_holdings)}只, 现金: {self.cash:.2f}")

            # 先按早盘状态决定是否买入。A股是T+1，当天9:35可用于买入的资金只能来自早盘可用现金，
            # 不能使用同日盘中风控/到期卖出后才释放的资金。
            # 因此买入必须发生在当日卖出逻辑之前，且只允许使用当日开盘前本来就空着的槽位。
            if day_start_open_slots > 0 and i < len(trading_dates) - 1 and day_start_free_slot_index is not None:
                # 检查是否是正确的买入时机
                should_buy = self._should_attempt_buy(
                    trade_date,
                    trading_dates,
                    day_start_open_slots=day_start_open_slots
                )

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
                    overlay_desc = ""
                    if 'overlay_action' in stock.index and pd.notna(stock.get('overlay_action')):
                        overlay_desc = (
                            f", overlay={stock.get('overlay_action', 'none')}"
                            f", overlay_scale={float(stock.get('overlay_scale', 1.0)):.2f}"
                            f", gap12_rel={float(stock.get('overlay_margin12_rel', np.nan)):.4f}"
                            f", certainty={float(stock.get('overlay_std_certainty_pct', np.nan)):.4f}"
                        )
                    logger.info(f"   📈 {stock['stock_code']}: 预测收益={stock['predicted_return']:.4f}{overlay_desc}")

                logger.info(f"🔄 开始对Top-{self.config.stock_count}策略执行买入逻辑...")

                # 双策略分仓专用逻辑：
                # - stock_count=1: 保持原语义，买入排名第1的股票
                # - stock_count>1: 买入前N只股票，用于同日分仓
                # 若预测表中包含regime_target_rank，则优先使用日级状态层决定Top1/Top2切换
                target_rank = self.config.stock_count  # 目标排名或目标数量
                regime_override_active = False
                if (
                    self.config.enable_dynamic_rank_switch and
                    'regime_target_rank' in selected_stocks.columns
                ):
                    regime_rank_series = pd.to_numeric(
                        selected_stocks['regime_target_rank'], errors='coerce'
                    ).dropna()
                    if not regime_rank_series.empty:
                        target_rank = int(np.clip(regime_rank_series.iloc[0], 1, len(selected_stocks)))
                        regime_override_active = True
                        logger.info(
                            f"🧭 日级状态切换生效: prediction_date={prediction_date}, "
                            f"使用regime_target_rank={target_rank} 覆盖默认Top-{self.config.stock_count}"
                        )

                # 检查是否有足够的候选股票
                if len(selected_stocks) < target_rank:
                    logger.warning(f"⚠️ 候选股票数{len(selected_stocks)}不足目标数量/排名{target_rank}，跳过买入")
                    continue

                target_stock_rows = []
                if target_rank == 1 and not regime_override_active:
                    # 🧠 融合选择：仅对Top-1策略，在Top1/Top2之间用 rank+binary 融合二选一
                    target_stock = self._select_target_stock_for_top1_fusion(selected_stocks, alpha=0.1)
                    if target_stock is None:
                        logger.warning("⚠️ 候选股票为空，跳过买入")
                        continue
                    target_stock_rows = [target_stock]
                    target_stocks_for_today = [target_stock['stock_code']]
                    logger.info(f"📦 预加载{trade_date}和{prediction_date}的目标股票5分钟数据: 排名第1的{target_stock['stock_code']}")
                elif target_rank > 1 and not regime_override_active:
                    # 双策略分仓专用：买入前N只
                    target_stock_rows = [row for _, row in selected_stocks.head(target_rank).iterrows()]
                    target_stocks_for_today = [row['stock_code'] for row in target_stock_rows]
                    logger.info(
                        f"📦 预加载{trade_date}和{prediction_date}的目标股票5分钟数据: "
                        f"前{target_rank}只 {target_stocks_for_today}"
                    )
                else:
                    # regime 覆盖场景保持原始语义：买入指定排名那一只
                    target_stock = selected_stocks.iloc[target_rank - 1]
                    target_stock_rows = [target_stock]
                    target_stocks_for_today = [target_stock['stock_code']]
                    logger.info(f"📦 预加载{trade_date}和{prediction_date}的目标股票5分钟数据: 排名第{target_rank}的{target_stock['stock_code']}")

                # 批量预加载T日和T+1日的股票数据
                self._preload_stocks_data([prediction_date, trade_date], target_stocks_for_today)

                # 逐只处理目标股票
                if len(target_stock_rows) > 1:
                    logger.info(f"🎯 处理Top-{self.config.stock_count}策略: 买入前{len(target_stock_rows)}只股票")
                else:
                    logger.info(f"🎯 处理Top-{self.config.stock_count}策略: 买入排名第{target_rank}的股票")

                for rank_idx, stock in enumerate(target_stock_rows, start=1):
                    # 双策略分仓专用：达到最大槽位后，不再继续处理剩余候选
                    if len(self.current_holdings) >= self.config.max_concurrent_positions:
                        logger.info(
                            f"🛑 当前持仓已满 {len(self.current_holdings)}/{self.config.max_concurrent_positions}，"
                            f"停止继续买入剩余候选"
                        )
                        break

                    stock_code = stock['stock_code']
                    overlay_action = str(stock.get('overlay_action', 'none')).lower()
                    overlay_scale = float(stock.get('overlay_scale', 1.0) if pd.notna(stock.get('overlay_scale', 1.0)) else 1.0)
                    overlay_reason = str(stock.get('overlay_reason', '') or '')

                    if rank_idx == 1 and target_rank == 1 and overlay_action in {'skip', 'half', 'full'}:
                        logger.info(
                            f"🛡️ Top1 Overlay决策: {stock_code} -> {overlay_action.upper()} "
                            f"(scale={overlay_scale:.2f}, reason={overlay_reason}, "
                            f"gap12_rel={float(stock.get('overlay_margin12_rel', np.nan)):.4f}, "
                            f"certainty={float(stock.get('overlay_std_certainty_pct', np.nan)):.4f})"
                        )

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
                    buy_data = execution_day_data

                    if not execution_day_data.empty:
                        buy_open = execution_day_data.iloc[0]['open']
                        logger.info(f"📊 {stock_code} T+1日({trade_date})开盘价: {buy_open:.2f}")
                        logger.info(f"📊 {stock_code} T+1日数据条数: {len(execution_day_data)}")
                        logger.info(f"📊 {stock_code} T+1日第一条数据时间: {execution_day_data.iloc[0]['datetime']}")
                    else:
                        buy_open = None

                    if not buy_data.empty:
                        if rank_idx == 1 and target_rank == 1 and overlay_action == 'skip':
                            logger.info(f"🚫 {stock_code} 被Top1 Overlay跳过: {overlay_reason}")
                            skipped_record = {
                                'stock_code': stock_code,
                                'prediction_date': prediction_date,
                                'attempt_buy_date': trade_date,
                                'buy_open': buy_open,
                                'prev_close': prev_close,
                                'gap_ratio': np.nan,
                                'skip_reason': f"Top1 Overlay跳过: {overlay_reason}",
                                'predicted_return': stock['predicted_return'],
                                'skip_category': 'Top1 Overlay',
                                'overlay_action': overlay_action,
                                'overlay_scale': overlay_scale,
                                'overlay_margin12': stock.get('overlay_margin12', np.nan),
                                'overlay_margin12_rel': stock.get('overlay_margin12_rel', np.nan),
                                'overlay_std_certainty_pct': stock.get('overlay_std_certainty_pct', np.nan)
                            }
                            self.skipped_buy_attempts.append(skipped_record)
                            continue

                        buy_open = buy_data.iloc[0]['open']
                        prev_close_str = f"{prev_close:.4f}" if prev_close is not None else "N/A"
                        logger.info(f"🔍 {stock_code} 买入检查: 开盘价={buy_open:.4f}, 前收盘={prev_close_str}")

                        entry_check = self.check_entry_signal(
                            stock_code, trade_date, prev_close, buy_open, target_rank=rank_idx
                        )

                        logger.info(f"🚦 {stock_code} 买入信号: {entry_check['can_buy']}, 原因: {entry_check['reason']}")

                        if entry_check['can_buy']:
                            current_slot_index = min(len(self.current_holdings), self.config.max_concurrent_positions - 1)
                            current_holdings_count = len(self.current_holdings)
                            final_position_pct = self._resolve_position_pct(current_slot_index)
                            entry_reason = "normal"
                            logger.info(
                                f"🧭 分层并行仓位生效: {stock_code} 使用 {final_position_pct:.0%} 仓位, "
                                f"分配到槽位={current_slot_index + 1}/{self.config.max_concurrent_positions}, "
                                f"开盘前持仓数={current_holdings_count}/{self.config.max_concurrent_positions}"
                            )

                            holding_days_override = self.config.holding_days
                            logger.info(f"🧭 固定持仓天数生效: {stock_code} 使用 {holding_days_override} 天持仓")

                            position = self.execute_buy(
                                stock_code, trade_date, stock['predicted_return'], trading_dates,
                                position_pct=final_position_pct,
                                entry_reason=entry_reason,
                                holding_days_override=holding_days_override,
                                day_start_cash=day_start_cash,
                                day_start_equity=day_start_equity,
                                slot_index=current_slot_index
                            )
                            if position:
                                self.positions.append(position)
                                logger.info(f"✅ {stock_code} 买入成功，添加到持仓列表")
                            else:
                                logger.warning(f"❌ {stock_code} 买入失败，execute_buy返回None")
                        else:
                            logger.info(f"🚫 {stock_code} 买入被拒绝: {entry_check['reason']}")
                            skipped_record = {
                                'stock_code': stock_code,
                                'prediction_date': prediction_date,
                                'attempt_buy_date': trade_date,
                                'buy_open': buy_open,
                                'prev_close': prev_close,
                                'gap_ratio': entry_check.get('gap_ratio', 0),
                                'skip_reason': entry_check['reason'],
                                'predicted_return': stock['predicted_return'],
                                'skip_category': entry_check.get('skip_category', '其他风控')
                            }
                            self.skipped_buy_attempts.append(skipped_record)
                    else:
                        logger.warning(f"⚠️ {stock_code} {trade_date}无5分钟数据，跳过买入")

            # 风控检查（在买入之后、到期卖出之前执行）。这样当天卖出释放的资金不会错误参与早盘买入。
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

        # 🆕 生成风控统计信息
        summary_stats = self._generate_risk_control_stats()

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
            'final_positions': list(self.current_holdings.values()),
            'summary_stats': summary_stats  # 🆕 添加风控统计
        }

        logger.info(f"🎯 Top-{self.config.stock_count} 策略回测完成:")
        logger.info(f"   总收益: {total_return:.2f}%")
        logger.info(f"   年化收益: {result['annual_return'] * 100:.2f}%")
        logger.info(f"   最大回撤: {result['max_drawdown']:.2f}%")
        logger.info(f"   夏普比率: {result['sharpe_ratio']:.3f}")
        logger.info(f"   交易次数: {total_trades}")
        logger.info(f"   胜率: {win_rate:.1f}%")

        # 添加风控统计输出
        if summary_stats:
            logger.info(f"   📊 风控统计:")

            # 买入风控统计
            gap_down = summary_stats.get('gap_down_3_skipped', 0)
            gap_up = summary_stats.get('gap_up_5_skipped', 0)
            overlay_skipped = summary_stats.get('overlay_skipped', 0)

            if gap_down > 0 or gap_up > 0:
                logger.info(f"      买入风控 (开盘): {gap_down + gap_up}次")
                if gap_down > 0:
                    logger.info(f"        - 跌超3%: {gap_down}次")
                if gap_up > 0:
                    logger.info(f"        - 涨超5%: {gap_up}次")
            else:
                logger.info(f"      买入风控 (开盘): 无触发")

            if overlay_skipped > 0:
                logger.info(f"      Top1 Overlay跳过: {overlay_skipped}次")

            # 🆕 MA5开盘风控统计
            ma5_gap_down = summary_stats.get('ma5_gap_down_skipped', 0)
            ma5_gap_up = summary_stats.get('ma5_gap_up_skipped', 0)

            if ma5_gap_down > 0 or ma5_gap_up > 0:
                logger.info(f"      买入风控 (开盘-MA5): {ma5_gap_down + ma5_gap_up}次")
                if ma5_gap_down > 0:
                    logger.info(f"        - 低开超MA5 3%: {ma5_gap_down}次")
                if ma5_gap_up > 0:
                    logger.info(f"        - 高开超MA5 3%: {ma5_gap_up}次")

            # 卖出风控统计
            risk_details = summary_stats.get('risk_control_details', {})
            risk_exits = risk_details.get('total_risk_exits', 0)

            if risk_exits > 0:
                threshold_desc = risk_details.get('stop_loss_threshold_desc', '-5.00%')
                logger.info(f"      卖出风控 (持仓，止损阈值: {threshold_desc}): {risk_exits}次 / {total_trades}笔交易")

                cumulative = risk_details.get('cumulative_decline_exits', 0)
                consecutive = risk_details.get('consecutive_decline_exits', 0)
                ma5_break = risk_details.get('ma5_break_exits', 0)  # 🆕 获取MA5跌破次数
                t1_low_break = risk_details.get('t1_low_break_exits', 0)  # 🆕 获取T+1最低价跌破次数
                profit_bearish = risk_details.get('profit_bearish_exits', 0)  # 🆕 获取盈利状态连续阴线次数

                if cumulative > 0:
                    logger.info(f"        - 累计收益低于阈值: {cumulative}次")
                if consecutive > 0:
                    logger.info(f"        - 连续3根阴线: {consecutive}次")
                if ma5_break > 0:  # 🆕 显示MA5跌破统计
                    logger.info(f"        - 日线MA5跌破: {ma5_break}次")
                if t1_low_break > 0:  # 🆕 显示T+1最低价跌破统计
                    logger.info(f"        - T+1最低价跌破: {t1_low_break}次")
                if profit_bearish > 0:  # 🆕 显示盈利状态连续阴线统计
                    logger.info(f"        - 盈利状态连续阴线: {profit_bearish}次")

                risk_rate = risk_details.get('risk_exit_rate', 0)
                logger.info(f"        风控退出率: {risk_rate:.1%}")
            else:
                threshold_desc = risk_details.get('stop_loss_threshold_desc', '-5.00%')
                logger.info(f"      卖出风控 (持仓，止损阈值: {threshold_desc}): 无触发")

            # 总计风控保护
            total_skipped = summary_stats.get('skipped_positions', 0)
            if total_skipped > 0:
                logger.info(f"   🛡️ 总计风控保护: {total_skipped}次买入被跳过")

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

    def _generate_risk_control_stats(self) -> Dict[str, Any]:
        """
        🆕 生成风控统计信息

        统计各类风控跳过买入的次数

        Returns:
            Dict: 风控统计字典
        """
        # 统计各类风控跳过次数
        stats = {
            'skipped_positions': len(self.skipped_buy_attempts),  # 总跳过次数
            'gap_down_3_skipped': 0,  # 跌空超过3%
            'gap_up_5_skipped': 0,  # 高开超过5%
            'overlay_skipped': 0,  # Top1 overlay主动跳过
            # 🆕 添加MA5风控统计
            'ma5_gap_down_skipped': 0,  # 低开超MA5 3%
            'ma5_gap_up_skipped': 0,  # 高开超MA5 3%
            'entry_risk_strategy': self.config.entry_risk_control_strategy,  # 🆕 记录使用的策略
            'optimization_version': 'v3.0_ma5_entry_control',  # 🆕 版本号更新
            'risk_control': {
                'gap_control': True,
                'ma5_control': True  # 🆕 MA5风控启用
            }
        }

        # 按skip_category分类统计
        for skip_record in self.skipped_buy_attempts:
            skip_category = skip_record.get('skip_category', '其他风控')

            if skip_category == '开盘风控':
                gap_ratio = skip_record.get('gap_ratio', 0)
                if gap_ratio < self.config.entry_risk_down_threshold:
                    stats['gap_down_3_skipped'] += 1
                elif gap_ratio > self.config.entry_risk_up_threshold:
                    stats['gap_up_5_skipped'] += 1

            # 🆕 MA5风控统计
            elif skip_category == '开盘MA5风控':
                ma5_gap_ratio = skip_record.get('gap_ratio', 0)
                if ma5_gap_ratio < 0:
                    stats['ma5_gap_down_skipped'] += 1
                elif ma5_gap_ratio > 0:
                    stats['ma5_gap_up_skipped'] += 1
            elif skip_category == 'Top1 Overlay':
                stats['overlay_skipped'] += 1

        # 持仓风控详情统计
        risk_exits = [t for t in self.trades if '风控' in t.get('exit_reason', '')]
        normal_exits = [t for t in self.trades if '到期' in t.get('exit_reason', '')]

        # 🔍 调试：打印所有风控退出的exit_reason
        if risk_exits:
            logger.info(f"🔍 [风控统计调试] 所有风控退出的exit_reason:")
            for t in risk_exits:
                logger.info(f"   - {t.get('exit_reason', 'N/A')}")

        # 动态生成阈值描述，用于匹配
        threshold_desc = f"{self.config.stop_loss_threshold:.2%}"
        if self.config.stop_loss_threshold == 0.0:
            threshold_desc = "0% (盈亏平衡)"
        elif self.config.stop_loss_threshold > 0:
            threshold_desc = f"{self.config.stop_loss_threshold:.2%} (要求正收益)"

        # 统计各类风控退出次数（使用动态阈值描述）
        cumulative_decline_exits = len([t for t in risk_exits if f'累计收益低于{threshold_desc}' in t.get('exit_reason', '')])
        consecutive_decline_exits = len([t for t in risk_exits if '连续3根阴线' in t.get('exit_reason', '')])
        # 🆕 统计MA5跌破退出次数
        ma5_break_exits = len([t for t in risk_exits if '日线MA5' in t.get('exit_reason', '')])
        # 🆕 统计T+1最低价跌破退出次数
        t1_low_break_exits = len([t for t in risk_exits if 'T+1最低价' in t.get('exit_reason', '')])
        # 🆕 统计盈利状态连续阴线退出次数
        profit_bearish_exits = len([
            t for t in risk_exits
            if (
                ('盈利状态下连续' in t.get('exit_reason', '') and '阴线' in t.get('exit_reason', ''))
                or '盈利状态下快速转弱' in t.get('exit_reason', '')
            )
        ])

        # 🔍 调试：打印统计结果
        logger.info(f"🔍 [风控统计调试] 使用阈值: {threshold_desc}")
        logger.info(f"🔍 [风控统计调试] 风控策略: {self.config.risk_control_strategy}")
        logger.info(f"🔍 [风控统计调试] 累计收益低于阈值: {cumulative_decline_exits}次")
        if self.config.risk_control_strategy == RiskControlStrategy.CONSECUTIVE_BEARISH:
            logger.info(f"🔍 [风控统计调试] 连续3根阴线: {consecutive_decline_exits}次")
        elif self.config.risk_control_strategy == RiskControlStrategy.DAILY_MA5_BREAK:
            logger.info(f"🔍 [风控统计调试] 日线MA5跌破: {ma5_break_exits}次")
        elif self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
            logger.info(f"🔍 [风控统计调试] T+1最低价跌破: {t1_low_break_exits}次")
            logger.info(f"🔍 [风控统计调试] 盈利状态连续阴线: {profit_bearish_exits}次")

        stats['risk_control_details'] = {
            'total_risk_exits': len(risk_exits),
            'stop_loss_threshold': self.config.stop_loss_threshold,  # 🆕 记录使用的止损阈值
            'stop_loss_threshold_desc': threshold_desc,  # 🆕 阈值描述
            'cumulative_decline_exits': cumulative_decline_exits,
            'consecutive_decline_exits': consecutive_decline_exits,
            'ma5_break_exits': ma5_break_exits,  # 🆕 MA5跌破退出次数
            't1_low_break_exits': t1_low_break_exits,  # 🆕 T+1最低价跌破退出次数
            'profit_bearish_exits': profit_bearish_exits,  # 🆕 盈利状态连续阴线退出次数
            'normal_exits': len(normal_exits),
            'risk_exit_rate': len(risk_exits) / len(self.trades) if self.trades else 0,
            'avg_risk_exit_return': np.mean([t['actual_return'] for t in risk_exits]) if risk_exits else 0,
            'avg_normal_return': np.mean([t['actual_return'] for t in normal_exits]) if normal_exits else 0
        }

        return stats


class OptimizedBacktester:
    """优化版回测器 - 管理多个策略"""

    def __init__(self, prediction_data: pd.DataFrame, minute_data_path: str):
        self.prediction_data = prediction_data
        self.minute_data_path = minute_data_path
        self.strategies: List[BacktestStrategy] = []

    # =====================================================
    # 🆕 rank + binary 融合：Top1/Top2 二选一（只在Top-1策略生效）
    #   score = s_rank + alpha * logit(p_cls)
    # =====================================================
    @staticmethod
    def _safe_logit(p: float, eps: float = 1e-6) -> float:
        """稳定logit，避免p=0/1导致无穷大。"""
        try:
            x = float(p)
        except Exception:
            return 0.0
        if np.isnan(x):
            return 0.0
        x = min(max(x, eps), 1.0 - eps)
        return float(np.log(x / (1.0 - x)))

    def _select_target_stock_for_top1_fusion(self, selected_stocks: pd.DataFrame, alpha: float = 0.1) -> pd.Series:
        """
        Top-1策略：在rank Top1/Top2之间，用融合分数二选一。
        - 需要列：predicted_return, p_cls, stock_code
        - 若缺少p_cls或候选不足2只，则退回rank Top1。
        """
        if selected_stocks is None or len(selected_stocks) == 0:
            return None
        if len(selected_stocks) < 2:
            return selected_stocks.iloc[0]
        if 'p_cls' not in selected_stocks.columns:
            return selected_stocks.iloc[0]

        r1 = selected_stocks.iloc[0]
        r2 = selected_stocks.iloc[1]
        s1 = float(r1.get('predicted_return', 0.0))
        s2 = float(r2.get('predicted_return', 0.0))
        p1 = r1.get('p_cls', np.nan)
        p2 = r2.get('p_cls', np.nan)
        z1 = self._safe_logit(p1)
        z2 = self._safe_logit(p2)
        score1 = s1 + float(alpha) * z1
        score2 = s2 + float(alpha) * z2

        logger.info(f"🧠 [FUSION] Top1/Top2融合打分: "
                    f"Top1({r1.get('stock_code')}): s={s1:.6f}, p={float(p1) if p1==p1 else np.nan:.4f}, logit={z1:.4f}, score={score1:.6f} | "
                    f"Top2({r2.get('stock_code')}): s={s2:.6f}, p={float(p2) if p2==p2 else np.nan:.4f}, logit={z2:.4f}, score={score2:.6f}")

        if score2 > score1:
            logger.info(f"✅ [FUSION] 选择Top2替代Top1: {r2.get('stock_code')} (score2>score1)")
            return r2
        else:
            logger.info(f"✅ [FUSION] 保持Top1: {r1.get('stock_code')} (score1>=score2)")
            return r1

    def add_strategy(self, config: StrategyConfig) -> None:
        """添加策略"""
        strategy = BacktestStrategy(config, self.prediction_data, self.minute_data_path)
        self.strategies.append(strategy)

    def run_strategies(self,
                       stock_counts: List[int],
                       trading_dates: List[datetime.date],
                       enable_dynamic_rank_switch: bool = True,
                       max_concurrent_positions: int = 3,
                       fixed_position_pct: float = 0.5,
                       position_schedule: Optional[List[float]] = None) -> Dict[str, Any]:
        """运行所有策略"""
        logger.info(f"🚀 开始运行 {len(stock_counts)} 个策略")

        results = {}

        for stock_count in stock_counts:
            config = StrategyConfig(
                strategy_type=StrategyType(stock_count),
                stock_count=stock_count,
                enable_dynamic_rank_switch=enable_dynamic_rank_switch,
                max_concurrent_positions=max_concurrent_positions,
                fixed_position_pct=fixed_position_pct,
                position_schedule=position_schedule,
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
                          holding_days: int = 2,
                          enable_dynamic_rank_switch: bool = True,
                          max_concurrent_positions: int = 3,
                          fixed_position_pct: float = 0.5,
                          position_schedule: Optional[List[float]] = None) -> Dict[str, Any]:
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

    results = backtester.run_strategies(
        stock_counts,
        valid_trading_dates,
        enable_dynamic_rank_switch=enable_dynamic_rank_switch,
        max_concurrent_positions=max_concurrent_positions,
        fixed_position_pct=fixed_position_pct,
        position_schedule=position_schedule,
    )

    return results
