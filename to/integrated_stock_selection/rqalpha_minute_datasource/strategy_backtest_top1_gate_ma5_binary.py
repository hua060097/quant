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
import gc
import bisect
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
    # Top1 开关专用回测：默认使用“低开超过6%跳过” + “gap_up > 5% 跳过”
    STRATEGY = EntryRiskControlStrategy.GAP_BASED

    # MA5风控参数
    MA5_MAX_UP_THRESHOLD = 0.05  # 保留高开超过5%跳过
    MA5_MAX_DOWN_THRESHOLD = 0.00  # 仅作为文档占位；实际逻辑是 open < MA5 就跳过

    # 涨跌幅风控参数（保持原有逻辑）
    GAP_MAX_DOWN_THRESHOLD = -0.06  # 跌空超过6%禁止买入
    GAP_MAX_UP_THRESHOLD = 0.05  # 高开超过5%禁止买入
    GAP_MAX_DOWN_THRESHOLD_NON_TOP1 = -0.03  # 非Top1跌空超过5%禁止买入

@dataclass
class StrategyConfig:
    """策略配置类"""
    strategy_type: StrategyType
    stock_count: int
    enable_dynamic_rank_switch: bool = True
    holding_days: int = 3
    initial_capital: float = 1000000.0
    max_concurrent_positions: int = 2
    fixed_position_pct: float = 0.5
    position_schedule: Optional[List[float]] = None

    # 持仓卖出止损阈值（用于T+2、T+3及以后的持仓风控）
    # 说明：
    #   - T+1日为买入日，不存在卖出风控
    #   - 从T+2日开始，如果累计收益低于此阈值，触发止损卖出
    #   - 示例：
    #       - stop_loss_threshold = -0.05  : 跌幅超过5%才卖出（原有逻辑，默认值）
    #       - stop_loss_threshold = 0.00   : 收益为负（0%）就卖出（激进风控）
    #       - stop_loss_threshold = 0.01   : 收益低于1%就卖出（要求正收益）
    #       - stop_loss_threshold = -0.10  : 跌幅超过10%才卖出（宽松风控）
    stop_loss_threshold: float = -0.05  # 默认-5%硬止损阈值

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
    delayed_entry_mode: str = "disabled"

    # 🆕 高位回撤风控参数（用于MA5跌破策略）
    high_return_threshold: float = 0.04  # 最高浮盈阈值（4%），触发后开始监控回撤
    high_drawdown_threshold: float = -0.03  # 从最高点回撤阈值（-3%），超过则卖出
    profit_bearish_min_peak_return: float = 0.09  # 盈利状态连续阴线风控生效前，至少先达到9%最高浮盈
    profit_bearish_trailing_drawdown: float = -0.04  # 盈利状态下相对最高点至少回撤4%
    profit_bearish_min_current_return: float = 0.06  # 盈利状态连续阴线风控时，当前至少仍有6%浮盈
    profit_bearish_required_count: int = 3  # 盈利状态下连续阴线触发数量
    # Top1 专用三层分钟风控
    enable_t2_0935_hard_stop: bool = True
    t2_0935_hard_stop_threshold: float = -0.04
    enable_gate_hit_strict_mode: bool = False
    gate_hit_entry_gap_up_threshold: float = 0.02
    enable_t2_1000_soft_stop: bool = False
    t2_1000_soft_stop_threshold: float = -0.02
    enable_second_layer_time_stop: bool = True
    t2_close_breakeven_stop_threshold: float = 0.00
    t3_0935_breakeven_stop_threshold: float = 0.00
    enable_small_profit_failure_exit: bool = True
    small_profit_activation_return: float = 0.04
    small_profit_retrace_return: float = -0.005
    small_profit_trailing_drawdown: float = -0.04
    small_profit_bearish_required_count: int = 2
    small_profit_watch_window_bars: int = 3
    enable_profit_threshold_for_bearish_exit: bool = True
    profit_bearish_activation_return: float = 0.05
    enable_top1_state_machine_risk: bool = False

    # Exhaustion factor buy filter (高位衰竭因子买入过滤)
    # strategy: 'three_rule_v3' (默认) | 'runup_or_blowoff' | 'runup_ma5' | 'close_below_ma5'
    enable_exhaustion_filter: bool = False
    exhaustion_filter_strategy: str = 'three_rule_v3'
    # v3: three_rule_v3 基于50+162双日志鲁棒性分析
    exhaustion_v3_cpv_threshold: float = -0.4                # cpv_factor <= 此值（配合runup>0.8）
    exhaustion_v3_blowoff_threshold: float = 2.0              # prev_day_blowoff_risk >= 此值
    exhaustion_v3_sector_accel_threshold: float = 0.30        # sector_momentum_accel_rank <= 此值
    exhaustion_v3_sector_leadership_threshold: float = -0.18  # sector_leadership_persistence_v3 <= 此值
    exhaustion_v3_return_20d_intra_z_threshold: float = 0.40  # R3仅在个股20日相对强度也偏弱时触发
    # v3 Rule4: 大盘流动性弱 + 个股三项至少其一
    exhaustion_v3_market_liq_threshold: float = 0.94  # market_liquidity_change < 此值
    exhaustion_v3_ulcer_threshold: float = -2.25       # ulcer_index < 此值 (路径A)
    exhaustion_v3_mom_peak_threshold: float = 3.20     # momentum_peak_decay_3d > 此值 (路径B)
    exhaustion_v3_intra_sector_rank_threshold: float = 0.67  # intra_sector_momentum_rank < 此值 (路径C)
    # v3 Rule5: 市场熔断 — 安静熊市
    exhaustion_v3_mkt_disp_threshold: float = 0.98    # mkt_dispersion_shock_v2 < 此值
    exhaustion_v3_mkt_adr_threshold: float = 0.38     # market_advance_decline_ratio < 此值
    # v3 Rule6: 近3日跌停过滤
    enable_exhaustion_limit_down_3d: bool = True       # 是否启用近3日跌停过滤
    # v3 Rule7: 短强但散户资金流出
    exhaustion_v3_top10_strength_threshold: float = 0.70    # top10_10d_strength_level_v3 >= 此值
    exhaustion_v3_buy_sm_amount_threshold: float = -1.25    # buy_sm_amount_rate <= 此值
    # v3 Rule8: 连板情绪高分位 + 个股KDJ-J截面动能偏弱
    exhaustion_v3_non_first_limit_up_perf_pct20_threshold: float = 0.80
    exhaustion_v3_kdj_j_z_threshold: float = 0.75
    # v3 Rule9: 当日Top1模型分数相对候选池预测分布偏弱
    exhaustion_v3_score_z_in_day_threshold: float = 1.50
    # v3 Rule10: 个股近10日严重上涨占比处于当日最高2% + 市场涨跌比极热
    exhaustion_v3_reg_severe_up_10d_pct_threshold: float = 0.98
    exhaustion_v3_reg_hot_adr_threshold: float = 4.50
    # v3 Rule11: MA5乖离率 >= 9% + 均线收敛度 >= 0.20 → 高位衰竭
    exhaustion_v3_ma5_raw_threshold: float = 9.0   # price_ma5_ratio_raw >= 此值
    exhaustion_v3_ma_convergence_threshold: float = 0.20  # ma_convergence_score >= 此值
    # v3 Rule12: MA5回撤深度极深 + 短期加速上涨 → 高位回撤加速衰竭
    exhaustion_v3_ma5_pullback_depth_threshold: float = -0.05  # ma5_pullback_depth_10d <= 此值
    exhaustion_v3_ret_3d_vs_10d_accel_threshold: float = 6.0   # ret_3d_vs_10d_accel >= 此值
    # 日内拆分入场（无未来函数版本）
    enable_intraday_split_entry: bool = False          # 是否启用日内拆分入场
    intraday_split_first15_threshold: float = 0.0      # 保留兼容；当前无未来版本用首根5分钟K close > open 做确认
    intraday_split_confirm_cap: float = 0.015          # 兼容旧参数；当前不再用作补仓涨幅上限
    intraday_split_limit_up_buffer: float = 0.005      # 估算涨停价缓冲，避免接近/达到涨停时假设可买
    intraday_split_max_buy_premium: float = 0.15       # 第二笔相对首笔买入价的异常保护上限
    intraday_split_tail_stop: float = 0.05             # 尾盘止损线（从open算起的最大跌幅）
    # v2: runup_ma5 / runup_or_blowoff
    exhaustion_runup_ma5_threshold: float = 0.8
    exhaustion_top10_10d_strength_threshold: float = 0.5
    exhaustion_prev_day_blowoff_threshold: float = 4.15
    exhaustion_sector_momentum_accel_weak_threshold: float = 0.41
    exhaustion_market_advance_decline_ratio_threshold: float = 5.39559
    # v1: close_below_ma5
    exhaustion_filter_turnover_z_threshold: float = 1.0

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
    last_valid_mark_price: float = 0.0  # 最近一次有效估值价格，避免异常分钟close污染组合净值
    risk_state: str = "INIT"
    failed_cont_watch_active: bool = False
    failed_cont_watch_bars_seen: int = 0
    failed_cont_watch_bearish_bars: int = 0
    failed_cont_watch_break_level: float = 0.0
    failed_cont_watch_day_offset: int = -1
    strict_risk_mode: bool = False

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
        self.sorted_prediction_dates = sorted(self.prediction_data['date'].unique())
        self.daily_bar_lookup: Dict[tuple, Dict[str, float]] = {}
        self.stock_daily_history: Dict[str, Dict[str, list]] = {}
        if {'stock_code', 'date'}.issubset(self.prediction_data.columns):
            history_cols = [col for col in ['stock_code', 'date', 'close', 'low'] if col in self.prediction_data.columns]
            daily_history_df = (
                self.prediction_data[history_cols]
                .drop_duplicates(subset=['stock_code', 'date'], keep='first')
                .sort_values(['stock_code', 'date'])
                .reset_index(drop=True)
            )
            for row in daily_history_df.itertuples(index=False):
                key = (row.date, row.stock_code)
                self.daily_bar_lookup[key] = {
                    'close': float(getattr(row, 'close', np.nan)) if pd.notna(getattr(row, 'close', np.nan)) else np.nan,
                    'low': float(getattr(row, 'low', np.nan)) if pd.notna(getattr(row, 'low', np.nan)) else np.nan,
                }
            for stock_code, grp in daily_history_df.groupby('stock_code', sort=False):
                self.stock_daily_history[str(stock_code)] = {
                    'dates': grp['date'].tolist(),
                    'closes': grp['close'].tolist() if 'close' in grp.columns else [],
                    'lows': grp['low'].tolist() if 'low' in grp.columns else [],
                }

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
        self.shadow_skipped_opportunity_trades: List[Dict] = []  # 满仓等原因跳过时的影子机会交易，不影响组合净值

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

    def clear_minute_data_cache(self, dates: Optional[List[str]] = None):
        """释放分钟数据缓存，避免全量缓存构建时内存持续膨胀。"""
        if dates is None:
            self.minute_data_cache.clear()
            gc.collect()
            return
        for date_key in dates:
            self.minute_data_cache.pop(date_key, None)
        gc.collect()


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

    @staticmethod
    def _to_python_datetime(value) -> datetime:
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        if hasattr(value, 'to_pydatetime'):
            return value.to_pydatetime()
        if isinstance(value, datetime):
            return value
        return datetime.now()

    @classmethod
    def _extract_time_hm(cls, value) -> Optional[str]:
        return cls._to_python_datetime(value).strftime('%H:%M')

    @staticmethod
    def _reset_failed_cont_watch(position: Position) -> None:
        position.failed_cont_watch_active = False
        position.failed_cont_watch_bars_seen = 0
        position.failed_cont_watch_bearish_bars = 0
        position.failed_cont_watch_break_level = 0.0
        position.failed_cont_watch_day_offset = -1

    def _evaluate_top1_state_machine_risk(
        self,
        position: Position,
        close_price: float,
        open_price: float,
        current_datetime,
        holding_day_offset: Optional[int],
        decline_ratio: float,
        consecutive_bearish_in_profit: int,
    ) -> Optional[Dict[str, Any]]:
        """Top1分钟状态机：FailFast / WeakDrift / FailedContinuation / ProfitProtect。"""
        if holding_day_offset not in (1, 2):
            self._reset_failed_cont_watch(position)
            position.risk_state = "IDLE"
            return None

        current_time_hm = self._extract_time_hm(current_datetime)
        exit_time = self._to_python_datetime(current_datetime)
        current_return = decline_ratio
        body_ratio = 0.0
        if open_price not in (None, 0):
            body_ratio = (close_price - open_price) / open_price
        is_bearish = close_price < open_price
        is_neutral = abs(body_ratio) <= 0.0005
        is_weak_bar = is_bearish or is_neutral
        drawdown_from_highest = None
        if position.highest_price_since_buy and position.highest_price_since_buy > 0:
            drawdown_from_highest = (close_price - position.highest_price_since_buy) / position.highest_price_since_buy

        if position.failed_cont_watch_active and position.failed_cont_watch_day_offset != holding_day_offset:
            self._reset_failed_cont_watch(position)

        if holding_day_offset == 1 and current_time_hm == '09:35':
            position.risk_state = "FAIL_FAST_OPEN"
            if self.config.enable_t2_0935_hard_stop and current_return <= self.config.t2_0935_hard_stop_threshold:
                return {
                    'should_exit': True,
                    'reason': f'T+2 09:35失败止损({self.config.t2_0935_hard_stop_threshold:.2%})',
                    'exit_price': close_price,
                    'exit_time': exit_time,
                    'decline_ratio': decline_ratio,
                    'risk_state': position.risk_state,
                }

        if holding_day_offset == 1 and current_time_hm == '10:00':
            position.risk_state = "FAIL_FAST_DRIFT"
            if self.config.enable_t2_1000_soft_stop and current_return <= self.config.t2_1000_soft_stop_threshold:
                return {
                    'should_exit': True,
                    'reason': f'T+2 10:00继续走坏止损({self.config.t2_1000_soft_stop_threshold:.2%})',
                    'exit_price': close_price,
                    'exit_time': exit_time,
                    'decline_ratio': decline_ratio,
                    'risk_state': position.risk_state,
                }

        if self.config.enable_second_layer_time_stop:
            if holding_day_offset == 1 and current_time_hm == '14:55':
                position.risk_state = "WEAK_DRIFT_T2_CLOSE"
                if current_return <= self.config.t2_close_breakeven_stop_threshold:
                    return {
                        'should_exit': True,
                        'reason': f'T+2收盘仍未转正({self.config.t2_close_breakeven_stop_threshold:.2%})',
                        'exit_price': close_price,
                        'exit_time': exit_time,
                        'decline_ratio': decline_ratio,
                        'risk_state': position.risk_state,
                    }
            if holding_day_offset == 2 and current_time_hm == '09:35':
                position.risk_state = "WEAK_DRIFT_T3_OPEN"
                if current_return <= self.config.t3_0935_breakeven_stop_threshold:
                    return {
                        'should_exit': True,
                        'reason': f'T+3 09:35仍未转正({self.config.t3_0935_breakeven_stop_threshold:.2%})',
                        'exit_price': close_price,
                        'exit_time': exit_time,
                        'decline_ratio': decline_ratio,
                        'risk_state': position.risk_state,
                    }

        if self.config.enable_small_profit_failure_exit and holding_day_offset == 1:
            break_level = position.t1_day_lowest_price
            peak_armed = position.highest_return_since_buy >= self.config.small_profit_activation_return
            if peak_armed and break_level and break_level > 0:
                if not position.failed_cont_watch_active:
                    if close_price < break_level:
                        position.failed_cont_watch_active = True
                        position.failed_cont_watch_bars_seen = 0
                        position.failed_cont_watch_bearish_bars = 0
                        position.failed_cont_watch_break_level = break_level
                        position.failed_cont_watch_day_offset = holding_day_offset
                        position.risk_state = "FAILED_CONTINUATION_WATCH"
                else:
                    if close_price >= position.failed_cont_watch_break_level:
                        self._reset_failed_cont_watch(position)
                    else:
                        position.failed_cont_watch_bars_seen += 1
                        if is_bearish:
                            position.failed_cont_watch_bearish_bars += 1
                        else:
                            position.failed_cont_watch_bearish_bars = 0
                        if (
                            position.failed_cont_watch_bearish_bars >= self.config.small_profit_bearish_required_count
                        ):
                            position.risk_state = "FAILED_CONTINUATION"
                            return {
                                'should_exit': True,
                                'reason': (
                                    f'小盈利后延续失败: 跌破T+1低点后连续'
                                    f'{position.failed_cont_watch_bearish_bars}根阴线'
                                    f' | 峰值{position.highest_return_since_buy:.2%}'
                                    f' | 当前{current_return:.2%}'
                                    f' | T+1低点{position.failed_cont_watch_break_level:.2f}'
                                ),
                                'exit_price': close_price,
                                'exit_time': exit_time,
                                'current_return_pct': current_return * 100,
                                'consecutive_bearish': consecutive_bearish_in_profit,
                                'risk_state': position.risk_state,
                            }
            else:
                self._reset_failed_cont_watch(position)
        else:
            self._reset_failed_cont_watch(position)

        drawdown_armed = (
            drawdown_from_highest is not None
            and drawdown_from_highest <= self.config.profit_bearish_trailing_drawdown
        )
        current_profit_ok = current_return >= self.config.profit_bearish_min_current_return
        if drawdown_armed and current_profit_ok:
            position.risk_state = "PROFIT_PROTECT"
            if consecutive_bearish_in_profit >= self.config.profit_bearish_required_count:
                return {
                    'should_exit': True,
                    'reason': (
                        f'利润保护: 连续{consecutive_bearish_in_profit}根弱K线'
                        f' | 峰值{position.highest_return_since_buy:.2%}'
                        f' | 回撤{drawdown_from_highest:.2%}'
                    ),
                    'exit_price': close_price,
                    'exit_time': exit_time,
                    'current_return_pct': current_return * 100,
                    'consecutive_bearish': consecutive_bearish_in_profit,
                    'risk_state': position.risk_state,
                }

        position.risk_state = "TREND_HOLD" if current_return > 0 else "WEAK_HOLD"
        return None



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
        unique_stocks = list(dict.fromkeys([str(s) for s in target_stocks if pd.notna(s)]))
        logger.info(f"🚀 开始批量预加载: {len(dates)}个日期 x {len(unique_stocks)}只股票")

        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            try:
                if not unique_stocks:
                    stock_mapping = {}
                elif len(unique_stocks) <= 200:
                    stock_mapping = self.load_minute_data_on_demand(date_str, unique_stocks)
                else:
                    target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    year_month = target_date.strftime('%Y%m')
                    month_dir = os.path.join(self.minute_data_path, year_month)
                    minute_file = os.path.join(month_dir, "minute_data.parquet")
                    if not os.path.exists(minute_file):
                        logger.warning(f"❌ 未找到{date_str}的数据文件: {minute_file.replace(os.sep, '/')}")
                        stock_mapping = {stock: pd.DataFrame() for stock in unique_stocks}
                    else:
                        start_time = datetime.strptime(f'{date_str} 00:00:00', '%Y-%m-%d %H:%M:%S')
                        end_time = datetime.strptime(f'{date_str} 23:59:59', '%Y-%m-%d %H:%M:%S')
                        stock_mapping = {}
                        batch_size = 200
                        for start_idx in range(0, len(unique_stocks), batch_size):
                            batch_stocks = unique_stocks[start_idx:start_idx + batch_size]
                            batch_mapping = self._load_specific_stocks(
                                date_str, minute_file, start_time, end_time, batch_stocks
                            )
                            stock_mapping.update(batch_mapping)
                        self.minute_data_cache[date_str] = stock_mapping
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

    def _get_safe_mark_price(self, position: Position, trade_date: datetime.date) -> float:
        """
        获取用于组合净值估值的安全价格。

        目的：
        - 避免分钟数据异常 close（接近0、NaN、单日跳变异常）把组合净值瞬间打穿
        - 当日数据缺失时，优先沿用最近一次有效估值价，而不是让净值异常波动
        """
        fallback_price = (
            position.last_valid_mark_price
            if position.last_valid_mark_price and position.last_valid_mark_price > 0
            else position.buy_price
        )
        day_data = self.get_stock_data(position.stock_code, trade_date)
        if day_data.empty:
            logger.warning(
                f"⚠️ {position.stock_code} 在{trade_date}无分钟数据，估值回退到最近有效价格: {fallback_price:.4f}"
            )
            return fallback_price

        raw_close = day_data.iloc[-1].get('close', np.nan)
        if pd.isna(raw_close) or not np.isfinite(raw_close) or raw_close <= 0:
            logger.warning(
                f"⚠️ {position.stock_code} 在{trade_date}收盘价异常({raw_close})，估值回退到最近有效价格: {fallback_price:.4f}"
            )
            return fallback_price

        mark_price = float(raw_close)
        reference_price = fallback_price if fallback_price > 0 else position.buy_price
        if reference_price > 0:
            one_day_move = abs(mark_price / reference_price - 1.0)
            if one_day_move > 0.40:
                logger.warning(
                    f"⚠️ {position.stock_code} 在{trade_date}估值价异常跳变: last_valid={reference_price:.4f}, "
                    f"close={mark_price:.4f}, 单日变化={one_day_move:.2%}，回退到最近有效价格"
                )
                return fallback_price

        position.last_valid_mark_price = mark_price
        return mark_price

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
            # ===== Top1 开关专用新策略 =====
            # 条件1：如果 T+1 开盘价低于 (前4日close + 当日open) / 5 的开盘MA5，则跳过
            # 条件2：同时保留原有 gap_up > 5% 跳过
            ma5 = self._calculate_daily_ma5_at_open(stock_code, buy_date, buy_open)

            if ma5 is None:
                logger.warning(f"⚠️ {stock_code}: 无法计算开盘MA5，退化为仅使用 gap_up 风控")
                if prev_close is None:
                    logger.info(f"✅ {stock_code}: 无前日数据，允许买入")
                    return {'can_buy': True, 'reason': '无前日数据'}

                gap_ratio = (buy_open - prev_close) / prev_close
                if gap_ratio > self.config.entry_risk_up_threshold:
                    logger.info(f"🚫 {stock_code}: 高开超过5% ({gap_ratio:.2%})，禁止买入")
                    return {
                        'can_buy': False,
                        'reason': f'开盘跳空{gap_ratio:.2%}',
                        'gap_ratio': gap_ratio,
                        'skip_category': '开盘风控'
                    }

                return {'can_buy': True, 'reason': '无MA5时仅gap_up风控通过', 'gap_ratio': gap_ratio}

            gap_ratio = None
            if prev_close is not None and prev_close != 0:
                gap_ratio = (buy_open - prev_close) / prev_close

            logger.info(f"📊 {stock_code}: 开盘MA5风控检查: buy_open={buy_open:.4f}, open_ma5={ma5:.4f}")
            if gap_ratio is not None:
                logger.info(f"📊 {stock_code}: 相对前收跳空={gap_ratio:.4f} ({gap_ratio:.2%})")

            # 条件1：开盘价低于开盘MA5就跳过
            if buy_open < ma5:
                logger.info(f"🚫 {stock_code}: 开盘价低于开盘MA5，禁止买入")
                logger.info(f"   MA5: {ma5:.4f}, 开盘价: {buy_open:.4f}")
                return {
                    'can_buy': False,
                    'reason': '开盘价低于开盘MA5',
                    'gap_ratio': gap_ratio if gap_ratio is not None else (buy_open - ma5) / ma5,
                    'skip_category': '开盘MA5风控',
                    'ma5_value': ma5
                }

            # 条件2：保留原有高开超过5%跳过
            if gap_ratio is not None and gap_ratio > self.config.entry_risk_up_threshold:
                logger.info(f"🚫 {stock_code}: 高开超过5% ({gap_ratio:.2%})，禁止买入")
                logger.info(f"   MA5: {ma5:.4f}, 开盘价: {buy_open:.4f}")
                return {
                    'can_buy': False,
                    'reason': f'开盘跳空{gap_ratio:.2%}',
                    'gap_ratio': gap_ratio,
                    'skip_category': '开盘风控',
                    'ma5_value': ma5
                }

            logger.info(f"✅ {stock_code}: 通过 Top1 开关专用买入过滤")
            return {
                'can_buy': True,
                'reason': '通过开盘MA5+gap_up风控',
                'gap_ratio': gap_ratio if gap_ratio is not None else 0.0,
                'ma5_value': ma5
            }

        logger.warning(f"⚠️ {stock_code}: 未知开盘风控策略 {self.config.entry_risk_control_strategy}，默认允许买入")
        return {'can_buy': True, 'reason': '未知风控策略默认放行'}

    def _resolve_delayed_entry(
        self,
        stock_code: str,
        trade_date: datetime.date,
        execution_day_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """解析实际买入时点。

        默认仍使用第一根5分钟K线的open（对应09:30开盘价）。
        delayed_entry_mode='1000_not_drop_1pct' 时：
        - 使用 10:00 的 close 作为买入价
        - 仅当 10:00 close 相对 09:35 close 未跌超1% 才允许买入
        """
        mode = str(getattr(self.config, 'delayed_entry_mode', 'disabled') or 'disabled').lower()
        first_bar = execution_day_data.iloc[0]
        default_dt = pd.to_datetime(first_bar['datetime'], errors='coerce')
        default_dt = default_dt.to_pydatetime() if hasattr(default_dt, 'to_pydatetime') else default_dt
        result = {
            'can_buy': True,
            'buy_price': float(first_bar['open']),
            'buy_datetime': default_dt if isinstance(default_dt, datetime) else datetime.combine(trade_date, datetime.min.time()),
            'reason': 'default_open_buy',
        }
        if mode != '1000_not_drop_1pct':
            return result

        working_data = execution_day_data.copy()
        if 'time_hm' not in working_data.columns:
            dt_series = pd.to_datetime(working_data['datetime'], errors='coerce')
            working_data['time_hm'] = dt_series.dt.strftime('%H:%M')

        bar_0935 = working_data[working_data['time_hm'] == '09:35']
        bar_1000 = working_data[working_data['time_hm'] == '10:00']
        if bar_0935.empty or bar_1000.empty:
            return {
                'can_buy': False,
                'buy_price': np.nan,
                'buy_datetime': result['buy_datetime'],
                'reason': '延后买入缺少09:35或10:00分钟数据',
            }

        close_0935 = float(bar_0935.iloc[0]['close'])
        close_1000 = float(bar_1000.iloc[0]['close'])
        if close_0935 <= 0:
            return {
                'can_buy': False,
                'buy_price': np.nan,
                'buy_datetime': result['buy_datetime'],
                'reason': '09:35价格异常',
            }
        weaken_ratio = close_1000 / close_0935 - 1.0
        if weaken_ratio < -0.01:
            return {
                'can_buy': False,
                'buy_price': np.nan,
                'buy_datetime': result['buy_datetime'],
                'reason': f'10:00相对09:35走弱超1%({weaken_ratio:.2%})',
            }
        buy_dt = pd.to_datetime(bar_1000.iloc[0]['datetime'], errors='coerce')
        buy_dt = buy_dt.to_pydatetime() if hasattr(buy_dt, 'to_pydatetime') else buy_dt
        return {
            'can_buy': True,
            'buy_price': close_1000,
            'buy_datetime': buy_dt if isinstance(buy_dt, datetime) else result['buy_datetime'],
            'reason': f'1000_not_drop_1pct({weaken_ratio:.2%})',
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
        stock_history = self.stock_daily_history.get(str(stock_code))
        if not stock_history:
            return []

        dates = stock_history['dates']
        closes = stock_history['closes']
        if not dates or not closes:
            return []

        pos = bisect.bisect_left(dates, date)
        if pos == 0:
            return []

        start_index = max(0, pos - n)
        result = [float(x) for x in closes[start_index:pos] if pd.notna(x)]
        logger.info(f"📊 {stock_code} 在{date}前{n}天的日线close: {result}")
        return result

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
            bar = self.daily_bar_lookup.get((date, str(stock_code)))
            if bar is None:
                logger.warning(f"⚠️ {stock_code} 在 {date} 无日线数据")
                return None
            low_value = bar.get('low')
            if pd.notna(low_value):
                daily_low = float(low_value)
                logger.info(f"📊 {stock_code} 在{date}的日线最低价(low): {daily_low:.4f}")
                return daily_low
            close_value = bar.get('close')
            if pd.notna(close_value):
                daily_low = float(close_value)
                logger.info(f"📊 {stock_code} 在{date}的日线最低价(close回退): {daily_low:.4f}")
                return daily_low
            return None
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
            open_price = minute_data['open']
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

            # 0. Top1 专用三层分钟风控
            if self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
                try:
                    buy_date_index = trading_dates.index(position.buy_date)
                    check_date_index = trading_dates.index(check_date)
                    holding_day_offset = check_date_index - buy_date_index
                except ValueError:
                    holding_day_offset = None

                current_time_hm = None
                if isinstance(current_datetime, str):
                    current_time_hm = datetime.fromisoformat(current_datetime).strftime('%H:%M')
                elif hasattr(current_datetime, 'strftime'):
                    current_time_hm = current_datetime.strftime('%H:%M')

                if self.config.enable_top1_state_machine_risk:
                    state_machine_result = self._evaluate_top1_state_machine_risk(
                        position=position,
                        close_price=close_price,
                        open_price=open_price,
                        current_datetime=current_datetime,
                        holding_day_offset=holding_day_offset,
                        decline_ratio=decline_ratio,
                        consecutive_bearish_in_profit=consecutive_bearish_in_profit,
                    )
                    if state_machine_result:
                        return state_machine_result
                else:
                    # 兼容旧版平铺阈值逻辑
                    if (
                        self.config.enable_t2_0935_hard_stop
                        and holding_day_offset == 1
                        and current_time_hm == '09:35'
                        and decline_ratio <= self.config.t2_0935_hard_stop_threshold
                    ):
                        return {
                            'should_exit': True,
                            'reason': f'T+2 09:35失败止损({self.config.t2_0935_hard_stop_threshold:.2%})',
                            'exit_price': close_price,
                            'exit_time': self._to_python_datetime(current_datetime),
                            'decline_ratio': decline_ratio,
                        }

                    if (
                        self.config.enable_t2_1000_soft_stop
                        and holding_day_offset == 1
                        and current_time_hm == '10:00'
                        and decline_ratio <= self.config.t2_1000_soft_stop_threshold
                    ):
                        return {
                            'should_exit': True,
                            'reason': f'T+2 10:00继续走坏止损({self.config.t2_1000_soft_stop_threshold:.2%})',
                            'exit_price': close_price,
                            'exit_time': self._to_python_datetime(current_datetime),
                            'decline_ratio': decline_ratio,
                        }

                    if self.config.enable_second_layer_time_stop:
                        second_layer_trigger = (
                            (holding_day_offset == 1 and current_time_hm == '14:55'
                             and decline_ratio <= self.config.t2_close_breakeven_stop_threshold)
                            or
                            (holding_day_offset == 2 and current_time_hm == '09:35'
                             and decline_ratio <= self.config.t3_0935_breakeven_stop_threshold)
                        )
                        if second_layer_trigger:
                            reason = (
                                f'T+2收盘仍未转正({self.config.t2_close_breakeven_stop_threshold:.2%})'
                                if holding_day_offset == 1 else
                                f'T+3 09:35仍未转正({self.config.t3_0935_breakeven_stop_threshold:.2%})'
                            )
                            return {
                                'should_exit': True,
                                'reason': reason,
                                'exit_price': close_price,
                                'exit_time': self._to_python_datetime(current_datetime),
                                'decline_ratio': decline_ratio,
                            }

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

                # 连续3根5分钟阴线仅在 T+2 / T+3 生效，但不再要求“盈利状态”
                try:
                    buy_date_index = trading_dates.index(position.buy_date)
                    check_date_index = trading_dates.index(check_date)
                    holding_day_offset = check_date_index - buy_date_index
                except ValueError:
                    holding_day_offset = None

                if holding_day_offset not in (1, 2):
                    if consecutive_bearish_in_profit > 0:
                        logger.info(
                            f"📅 {position.stock_code}: 当前为持仓第{holding_day_offset + 1 if holding_day_offset is not None else '未知'}日，"
                            f"不在T+2/T+3范围，重置连续阴线计数（从{consecutive_bearish_in_profit}重置为0）"
                        )
                    consecutive_bearish_in_profit = 0
                    continue

                open_price = minute_data['open']
                body_ratio = 0.0
                if open_price not in (None, 0):
                    body_ratio = (close_price - open_price) / open_price

                is_bearish = close_price < open_price
                is_neutral = abs(body_ratio) <= 0.0005

                if is_bearish:
                    consecutive_bearish_in_profit += 1
                    logger.info(f"📉 {position.stock_code} 第{i+1}根K线(连续阴线监控):")
                    logger.info(f"   open={open_price:.4f}, close={close_price:.4f} (阴线)")
                    logger.info(f"   当前收益:{current_return*100:.2f}%")
                    logger.info(f"   连续走坏计数: {consecutive_bearish_in_profit}次")
                elif is_neutral:
                    if consecutive_bearish_in_profit > 0:
                        logger.info(f"➖ {position.stock_code} 第{i+1}根K线(连续阴线监控):")
                        logger.info(f"   open={open_price:.4f}, close={close_price:.4f} (十字/平盘)")
                        logger.info(
                            f"   保持连续走坏计数: {consecutive_bearish_in_profit}次"
                        )
                else:
                    if consecutive_bearish_in_profit > 0:
                        logger.info(f"📈 {position.stock_code} 第{i+1}根K线(连续阴线监控):")
                        logger.info(f"   open={open_price:.4f}, close={close_price:.4f} (阳线)")
                        logger.info(f"   重置连续走坏计数（从{consecutive_bearish_in_profit}重置为0）")
                    consecutive_bearish_in_profit = 0

                if self.config.enable_top1_state_machine_risk:
                    state_machine_result = self._evaluate_top1_state_machine_risk(
                        position=position,
                        close_price=close_price,
                        open_price=open_price,
                        current_datetime=current_datetime,
                        holding_day_offset=holding_day_offset,
                        decline_ratio=current_return,
                        consecutive_bearish_in_profit=consecutive_bearish_in_profit,
                    )
                    if state_machine_result:
                        return state_machine_result
                else:
                    bearish_exit_armed = True
                    if self.config.enable_profit_threshold_for_bearish_exit:
                        bearish_exit_armed = current_return >= self.config.profit_bearish_activation_return
                        if not bearish_exit_armed and consecutive_bearish_in_profit > 0:
                            logger.info(
                                f"⏸️ {position.stock_code}: 当前浮盈{current_return*100:.2f}%尚未达到"
                                f"{self.config.profit_bearish_activation_return*100:.2f}%，"
                                f"连续阴线止盈暂不生效"
                            )
                            consecutive_bearish_in_profit = 0

                    if bearish_exit_armed and consecutive_bearish_in_profit >= self.config.profit_bearish_required_count:
                        logger.info(f"🚨 {position.stock_code}: 在第{i+1}根K线触发连续阴线风控!")
                        logger.info(f"   触发K线: 时间={current_datetime}")
                        logger.info(f"   连续阴线次数: {consecutive_bearish_in_profit}")
                        logger.info(f"   当前收益: {current_return*100:.2f}%")

                        return {
                            'should_exit': True,
                            'reason': f'连续{consecutive_bearish_in_profit}根5分钟阴线',
                            'exit_price': close_price,
                            'exit_time': self._to_python_datetime(current_datetime),
                            'current_return_pct': current_return * 100,
                            'consecutive_bearish': consecutive_bearish_in_profit,
                        }

                # end for - T+1最低价跌破策略循环结束



        # 根据策略显示不同的日志信息
        if self.config.risk_control_strategy == RiskControlStrategy.CONSECUTIVE_BEARISH:
            logger.info(f"✅ {position.stock_code}: 未发现风控信号（止损阈值: {self.config.stop_loss_threshold:.2%} 或 连续3根阴线）")
        elif self.config.risk_control_strategy == RiskControlStrategy.DAILY_MA5_BREAK:
            logger.info(f"✅ {position.stock_code}: 未发现风控信号（止损阈值: {self.config.stop_loss_threshold:.2%} 或 日线MA5跌破）")
        elif self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
            logger.info(f"✅ {position.stock_code}: 未发现风控信号（止损阈值: {self.config.stop_loss_threshold:.2%} 或 T+1最低价跌破/连续阴线）")
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

    def _compute_minute_path_stats(
        self,
        stock_code: str,
        start_date: datetime.date,
        end_date: datetime.date,
        trading_dates: List[datetime.date],
        buy_price: float
    ) -> Dict[str, float]:
        """计算单笔交易分钟路径统计，用于缓存和分钟级复筛。"""
        min_return = 0.0
        worst_drawdown = 0.0
        peak_price = float(buy_price)

        try:
            start_index = trading_dates.index(start_date)
            end_index = trading_dates.index(end_date)
        except ValueError:
            return {
                'minute_min_return': 0.0,
                'minute_max_drawdown': 0.0
            }

        for check_date in trading_dates[start_index:end_index + 1]:
            day_data = self.get_stock_data(stock_code, check_date)
            if day_data.empty:
                continue
            for _, row in day_data.iterrows():
                low_price = row['low'] if 'low' in row and pd.notna(row['low']) else row['close']
                high_price = row['high'] if 'high' in row and pd.notna(row['high']) else row['close']

                if pd.notna(low_price) and buy_price > 0:
                    low_ret = (float(low_price) - float(buy_price)) / float(buy_price)
                    min_return = min(min_return, low_ret)

                if pd.notna(high_price):
                    peak_price = max(peak_price, float(high_price))

                if pd.notna(low_price) and peak_price > 0:
                    current_drawdown = (float(low_price) - float(peak_price)) / float(peak_price)
                    worst_drawdown = min(worst_drawdown, current_drawdown)

        return {
            'minute_min_return': float(min_return),
            'minute_max_drawdown': float(worst_drawdown)
        }

    def simulate_single_prediction_outcome(
        self,
        stock_code: str,
        prediction_date: datetime.date,
        trading_dates: List[datetime.date],
        predicted_return: float = 0.0,
        target_rank: int = 1,
        preloaded: bool = False
    ) -> Dict[str, Any]:
        """
        单股票单预测日模拟，用于离线缓存分钟交易结果。
        该函数不依赖策略历史交易状态，适合反复复用。
        """
        prediction_date = pd.to_datetime(prediction_date).date()
        result: Dict[str, Any] = {
            'date': prediction_date,
            'stock_code': stock_code,
            'predicted_return': float(predicted_return) if pd.notna(predicted_return) else 0.0,
            'buy_executable': False,
            'skip_reason': '',
            'buy_price': np.nan,
            'exit_price': np.nan,
            'exit_date': pd.NaT,
            'exit_reason': '',
            'minute_realized_return': 0.0,
            'minute_min_return': 0.0,
            'minute_max_drawdown': 0.0,
            'stoploss_hit': False,
            'profit_bearish_hit': False,
            'holding_days_realized': 0,
            'missing_minute_data': False,
        }

        trade_date = self.add_trading_days(prediction_date, 1, trading_dates)
        if trade_date is None:
            result['skip_reason'] = '无T+1交易日'
            result['missing_minute_data'] = True
            return result

        holding_end_date = self.add_trading_days(trade_date, self.config.holding_days - 1, trading_dates)
        if holding_end_date is None:
            holding_end_date = trade_date

        if not preloaded:
            preload_dates = [prediction_date]
            try:
                trade_index = trading_dates.index(trade_date)
                end_index = trading_dates.index(holding_end_date)
                preload_dates.extend(trading_dates[trade_index:end_index + 1])
            except ValueError:
                preload_dates.append(trade_date)
                if holding_end_date != trade_date:
                    preload_dates.append(holding_end_date)
            self._preload_stocks_data(preload_dates, [stock_code])

        prediction_day_data = self.get_stock_data(stock_code, prediction_date)
        prev_close = prediction_day_data.iloc[-1]['close'] if not prediction_day_data.empty else None

        execution_day_data = self.get_stock_data(stock_code, trade_date)
        if execution_day_data.empty:
            result['skip_reason'] = 'T+1无5分钟数据'
            result['missing_minute_data'] = True
            return result

        buy_open = float(execution_day_data.iloc[0]['open'])

        entry_check = self.check_entry_signal(
            stock_code, trade_date, prev_close, buy_open, target_rank=target_rank
        )
        if not entry_check.get('can_buy', False):
            result['skip_reason'] = str(entry_check.get('reason', '开盘风控拒绝'))
            return result

        delayed_entry = self._resolve_delayed_entry(stock_code, trade_date, execution_day_data)
        if not delayed_entry.get('can_buy', False):
            result['skip_reason'] = str(delayed_entry.get('reason', '延后买入条件未满足'))
            return result
        actual_buy_price = float(delayed_entry['buy_price'])
        buy_datetime = delayed_entry['buy_datetime']
        result['buy_price'] = actual_buy_price

        t1_day_lowest_price = 0.0
        if self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
            t1_low = self._get_daily_low(stock_code, trade_date)
            if t1_low is not None and pd.notna(t1_low):
                t1_day_lowest_price = float(t1_low)

        position = Position(
            stock_code=stock_code,
            buy_date=trade_date,
            buy_datetime=buy_datetime,
            buy_price=actual_buy_price,
            highest_price_since_buy=actual_buy_price,
            highest_return_since_buy=0.0,
            quantity=100,
            predicted_return=float(predicted_return) if pd.notna(predicted_return) else 0.0,
            holding_end_date=holding_end_date,
            entry_reason='cache_sim',
            holding_days=int(self.config.holding_days),
            slot_index=0,
            t1_day_lowest_price=t1_day_lowest_price,
            last_valid_mark_price=actual_buy_price,
            strict_risk_mode=False
        )

        result['buy_executable'] = True
        exit_price = actual_buy_price
        exit_date = holding_end_date
        exit_reason = '到期'

        try:
            trade_index = trading_dates.index(trade_date)
            end_index = trading_dates.index(holding_end_date)
            scan_dates = trading_dates[trade_index:end_index + 1]
        except ValueError:
            scan_dates = [trade_date]
            if holding_end_date != trade_date:
                scan_dates.append(holding_end_date)

        for check_date in scan_dates:
            risk_check = self.check_position_risk(position, check_date, trading_dates)
            if risk_check.get('should_exit', False):
                exit_date = check_date
                exit_price = float(risk_check.get('exit_price', actual_buy_price))
                exit_reason = f"风控: {risk_check.get('reason', '')}"
                break

        if exit_reason == '到期':
            exit_day_data = self.get_stock_data(stock_code, holding_end_date)
            if not exit_day_data.empty and pd.notna(exit_day_data.iloc[-1]['close']):
                exit_price = float(exit_day_data.iloc[-1]['close'])
            else:
                exit_price = actual_buy_price

        path_stats = self._compute_minute_path_stats(
            stock_code=stock_code,
            start_date=trade_date,
            end_date=exit_date,
            trading_dates=trading_dates,
            buy_price=actual_buy_price
        )
        realized_return = (float(exit_price) - float(actual_buy_price)) / float(actual_buy_price) if actual_buy_price > 0 else 0.0

        result.update({
            'exit_price': float(exit_price),
            'exit_date': exit_date,
            'exit_reason': exit_reason,
            'minute_realized_return': float(realized_return),
            'minute_min_return': float(path_stats['minute_min_return']),
            'minute_max_drawdown': float(path_stats['minute_max_drawdown']),
            'stoploss_hit': ('累计收益低于' in exit_reason),
            'profit_bearish_hit': ('连续' in exit_reason and '阴线' in exit_reason),
            'holding_days_realized': int(
                max(0, trading_dates.index(exit_date) - trading_dates.index(trade_date) + 1)
            ) if exit_date in trading_dates and trade_date in trading_dates else 0
        })
        return result

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
        schedule = list(self.config.position_schedule or [])
        if schedule:
            resolved_slot_index = max(0, int(slot_index))
            if resolved_slot_index < len(schedule):
                pct = float(schedule[resolved_slot_index])
            else:
                pct = float(schedule[-1])
        else:
            pct = float(self.config.fixed_position_pct)
        return max(0.0, min(1.0, pct))

    def _resolve_position_scale(self, selected_stocks: pd.DataFrame) -> float:
        if 'regime_position_scale' not in selected_stocks.columns:
            return 1.0
        try:
            first_scale = pd.to_numeric(selected_stocks['regime_position_scale'], errors='coerce').iloc[0]
            if pd.notna(first_scale):
                return max(0.0, min(1.0, float(first_scale)))
        except Exception:
            pass
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
                    slot_index: int = 0,
                    strict_risk_mode: bool = False,
                    prev_close: Optional[float] = None) -> Optional[Position]:
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

        # 当前实现的开盘风控仍基于首根5分钟K线open（即09:30开盘价），
        # 但实际买入时点可由 delayed_entry_mode 决定。
        delayed_entry = self._resolve_delayed_entry(stock_code, buy_date, stock_data)
        if not delayed_entry.get('can_buy', False):
            logger.info(f"⏭️ {stock_code} 延后买入条件未满足，跳过买入: {delayed_entry.get('reason', '')}")
            return None

        buy_price = float(delayed_entry['buy_price'])
        buy_datetime = delayed_entry['buy_datetime']
        logger.info(f"💰 {stock_code} 实际买入价: {buy_price:.4f}, 买入时间: {buy_datetime}, 当前现金: {self.cash:.2f}")
        if self.config.enable_gate_hit_strict_mode and strict_risk_mode and prev_close not in (None, 0):
            strict_gap_ratio = (buy_price - float(prev_close)) / float(prev_close)
            if strict_gap_ratio > self.config.gate_hit_entry_gap_up_threshold:
                logger.info(
                    f"⏭️ {stock_code} 命中严格模式高开过滤，跳过买入: "
                    f"gap_up={strict_gap_ratio:.2%} > {self.config.gate_hit_entry_gap_up_threshold:.2%}"
                )
                return None

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

        logger.info(
            f"💵 资金使用情况: 分配金额={position_amount:.2f}, 买入数量={quantity}股, "
            f"实际花费={actual_cost:.2f}"
        )
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
            t1_day_lowest_price=t1_day_lowest_price,  # 🆕 记录T+1日最低价
            last_valid_mark_price=buy_price,
            strict_risk_mode=bool(strict_risk_mode)
        )

        self.cash -= buy_price * quantity

        self.current_holdings[stock_code] = position

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

    def _get_latest_prediction_date_before(self, trade_date: datetime.date) -> Optional[datetime.date]:
        prediction_dates = pd.to_datetime(self.prediction_data['date']).dt.date
        prediction_dates = sorted(set(prediction_dates.tolist()))
        earlier_dates = [d for d in prediction_dates if d < trade_date]
        return max(earlier_dates) if earlier_dates else None

    def _select_target_stock_for_date(
        self,
        prediction_date: datetime.date,
        log_prefix: str = "",
    ) -> tuple[Optional[pd.Series], Optional[pd.DataFrame], str]:
        """按真实买入逻辑选择指定T日的目标股票。"""
        prediction_date_str = prediction_date.strftime('%Y-%m-%d')
        top_k_stocks = self.required_stocks_by_date.get(prediction_date_str, [])
        if not top_k_stocks:
            return None, None, f"{prediction_date} 无预分析Top-K股票"

        daily_predictions = self.prediction_data[
            (self.prediction_data['date'] == prediction_date) &
            (self.prediction_data['stock_code'].isin(top_k_stocks))
        ]
        if daily_predictions.empty:
            return None, None, f"{prediction_date} Top-K股票无预测数据"

        daily_predictions = self._attach_score_z_in_day(daily_predictions)
        selected_stocks = daily_predictions.sort_values('predicted_return', ascending=False)
        target_rank = self.config.stock_count
        regime_override_active = False

        if 'regime_skip_buy' in selected_stocks.columns:
            skip_series = selected_stocks['regime_skip_buy']
            if len(skip_series) > 0:
                first_skip = skip_series.iloc[0]
                regime_skip_buy = bool(first_skip) if pd.notna(first_skip) else False
                if regime_skip_buy:
                    return None, selected_stocks, f"{prediction_date} regime_skip_buy=True"

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

        if len(selected_stocks) < target_rank:
            return None, selected_stocks, f"候选股票数{len(selected_stocks)}不足目标排名{target_rank}"

        if self.config.enable_dynamic_rank_switch and target_rank == 1 and not regime_override_active:
            target_stock = self._select_target_stock_for_top1_fusion(selected_stocks, alpha=0.1)
            if target_stock is None:
                return None, selected_stocks, "候选股票为空"
        else:
            target_stock = selected_stocks.iloc[target_rank - 1]

        if log_prefix:
            logger.info(
                f"{log_prefix}目标股票: {target_stock['stock_code']}, "
                f"prediction_date={prediction_date}, target_rank={target_rank}"
            )
        return target_stock, selected_stocks, ""

    def _attach_score_z_in_day(self, daily_predictions: pd.DataFrame) -> pd.DataFrame:
        """补齐当日候选池内预测分数z-score，供Rule9使用。"""
        daily_predictions = daily_predictions.copy()
        if 'score_z_in_day' in daily_predictions.columns and daily_predictions['score_z_in_day'].notna().any():
            return daily_predictions
        scores = pd.to_numeric(daily_predictions.get('predicted_return', np.nan), errors='coerce')
        score_std = float(np.nanstd(scores, ddof=0))
        if score_std > 0:
            daily_predictions['score_z_in_day'] = (scores - float(np.nanmean(scores))) / score_std
        else:
            daily_predictions['score_z_in_day'] = np.nan
        return daily_predictions

    def _evaluate_exhaustion_for_stock(self, stock: pd.Series) -> Dict[str, Any]:
        strategy = self.config.exhaustion_filter_strategy
        if not self.config.enable_exhaustion_filter:
            return {
                'blocked': False,
                'strategy': strategy,
                'reason': '',
                'rule1_hit': False,
                'rule2_hit': False,
                'rule3_hit': False,
                'rule4_hit': False,
                'rule5_hit': False,
                'rule6_hit': False,
                'rule7_hit': False,
                'rule8_hit': False,
                'rule9_hit': False,
                'rule10_hit': False,
                'rule11_hit': False,
                'rule12_hit': False,
            }

        if strategy == 'three_rule_v3':
            th = self.config.exhaustion_runup_ma5_threshold
            cpv_th = self.config.exhaustion_v3_cpv_threshold
            rma_val = stock.get('runup_ma5_exhaustion', np.nan)
            cpv_val = stock.get('cpv_factor', np.nan)
            rule1_hit = (
                pd.notna(rma_val) and pd.notna(cpv_val)
                and rma_val > th
                and cpv_val <= cpv_th
            )

            blowoff_th = self.config.exhaustion_v3_blowoff_threshold
            sector_th = self.config.exhaustion_v3_sector_accel_threshold
            blowoff_val = stock.get('prev_day_blowoff_risk', np.nan)
            sector_val = stock.get('sector_momentum_accel_rank', np.nan)
            rule2_hit = (
                pd.notna(blowoff_val) and pd.notna(sector_val)
                and (
                    (blowoff_val >= blowoff_th and sector_val <= sector_th)
                    or (blowoff_val > 5.0 and sector_val <= 0.50)   # XR2a
                    or (blowoff_val > 1.0 and sector_val <= 0.15)   # XR2b
                )
            )

            sl_th = self.config.exhaustion_v3_sector_leadership_threshold
            ret20_th = self.config.exhaustion_v3_return_20d_intra_z_threshold
            sl_val = stock.get('sector_leadership_persistence_v3', np.nan)
            ret20_val = stock.get('return_20d_intra_z', np.nan)
            rule3_hit = (
                pd.notna(sl_val) and pd.notna(ret20_val)
                and sl_val <= sl_th
                and ret20_val <= ret20_th
            )

            # Rule 4: 大盘流动性弱 + 个股三项至少其一
            liq_th = self.config.exhaustion_v3_market_liq_threshold
            ulcer_th = self.config.exhaustion_v3_ulcer_threshold
            mom_peak_th = self.config.exhaustion_v3_mom_peak_threshold
            sector_rank_th = self.config.exhaustion_v3_intra_sector_rank_threshold
            liq_val = stock.get('market_liquidity_change', np.nan)
            ulcer_val = stock.get('ulcer_index', np.nan)
            mom_peak_val = stock.get('momentum_peak_decay_3d', np.nan)
            sector_rank_val = stock.get('intra_sector_momentum_rank', np.nan)
            rule4_hit = (
                pd.notna(liq_val)
                and liq_val < liq_th
                and (
                    (pd.notna(ulcer_val) and ulcer_val < ulcer_th)
                    or (pd.notna(mom_peak_val) and mom_peak_val > mom_peak_th)
                    or (pd.notna(sector_rank_val) and sector_rank_val < sector_rank_th)
                )
            )

            # Rule 6: 近3日有过跌停
            rule6_hit = (
                self.config.enable_exhaustion_limit_down_3d
                and stock.get('limit_down_3d', 0) == 1
            )

            # Rule 7: 短强但散户资金流出 — 主力出货嫌疑
            top10_th = self.config.exhaustion_v3_top10_strength_threshold
            buy_sm_th = self.config.exhaustion_v3_buy_sm_amount_threshold
            top10_val = stock.get('top10_10d_strength_level_v3', np.nan)
            buy_sm_val = stock.get('buy_sm_amount_rate', np.nan)
            rule7_hit = (
                pd.notna(top10_val) and pd.notna(buy_sm_val)
                and top10_val >= top10_th
                and buy_sm_val <= buy_sm_th
            )

            # Rule 8: 连板情绪高分位 + 个股KDJ-J截面动能偏弱
            non_first_pct20_th = self.config.exhaustion_v3_non_first_limit_up_perf_pct20_threshold
            kdj_j_th = self.config.exhaustion_v3_kdj_j_z_threshold
            non_first_pct20_val = stock.get('prev_non_first_limit_up_performance_percentile_20d', np.nan)
            kdj_j_val = stock.get('kdj_j_9', np.nan)
            rule8_hit = (
                pd.notna(non_first_pct20_val) and pd.notna(kdj_j_val)
                and non_first_pct20_val >= non_first_pct20_th
                and kdj_j_val <= kdj_j_th
            )

            # Rule 9: 当日Top1模型分数相对候选池预测分布偏弱
            score_z_th = self.config.exhaustion_v3_score_z_in_day_threshold
            score_z_val = stock.get('score_z_in_day', np.nan)
            rule9_hit = (
                pd.notna(score_z_val)
                and score_z_val <= score_z_th
            )

            # Rule11: MA5乖离率 >= 9% + 均线收敛度 >= 0.20
            ma5_raw_th = self.config.exhaustion_v3_ma5_raw_threshold
            ma_conv_th = self.config.exhaustion_v3_ma_convergence_threshold
            ma5_raw_val = stock.get('price_ma5_ratio_raw', np.nan)
            ma_conv_val = stock.get('ma_convergence_score', np.nan)
            rule11_hit = (
                pd.notna(ma5_raw_val) and pd.notna(ma_conv_val)
                and ma5_raw_val >= ma5_raw_th
                and ma_conv_val >= ma_conv_th
            )
            # Rule12: MA5回撤深度极深 + 短期加速上涨 → 高位回撤加速衰竭
            ma5_pullback_th = self.config.exhaustion_v3_ma5_pullback_depth_threshold
            ret_accel_th = self.config.exhaustion_v3_ret_3d_vs_10d_accel_threshold
            ma5_pullback_val = stock.get('ma5_pullback_depth_10d', np.nan)
            ret_accel_val = stock.get('ret_3d_vs_10d_accel', np.nan)
            rule12_hit = (
                pd.notna(ma5_pullback_val) and pd.notna(ret_accel_val)
                and ma5_pullback_val <= ma5_pullback_th
                and ret_accel_val >= ret_accel_th
            )

            # Rule 5: 市场熔断 — 安静熊市(dispersion低于均值+涨跌比极低)
            disp_th = self.config.exhaustion_v3_mkt_disp_threshold
            adr_th = self.config.exhaustion_v3_mkt_adr_threshold
            disp_val = stock.get('mkt_dispersion_shock_v2', np.nan)
            adr_val = stock.get('market_advance_decline_ratio', np.nan)
            rule5_hit = (
                pd.notna(disp_val) and pd.notna(adr_val)
                and disp_val < disp_th
                and adr_val < adr_th
            )
            reg_severe_pct_th = self.config.exhaustion_v3_reg_severe_up_10d_pct_threshold
            reg_hot_adr_th = self.config.exhaustion_v3_reg_hot_adr_threshold
            reg_severe_pct_val = stock.get('reg_severe_up_10d_ratio_pct', np.nan)
            rule10_hit = (
                pd.notna(reg_severe_pct_val) and pd.notna(adr_val)
                and reg_severe_pct_val >= reg_severe_pct_th
                and adr_val >= reg_hot_adr_th
            )

            parts = []
            if rule1_hit:
                parts.append(f'runup({rma_val:.2f})>{th} & cpv({cpv_val:.2f})<={cpv_th}')
            if rule2_hit:
                if blowoff_val >= blowoff_th and sector_val <= sector_th:
                    parts.append(f'blowoff({blowoff_val:.2f})>={blowoff_th} & sector_accel({sector_val:.2f})<={sector_th}')
                elif blowoff_val > 5.0:
                    parts.append(f'XR2a:blowoff({blowoff_val:.2f})>5.0 & sector_accel({sector_val:.2f})<=0.50')
                else:
                    parts.append(f'XR2b:blowoff({blowoff_val:.2f})>1.0 & sector_accel({sector_val:.2f})<=0.15')
            if rule3_hit:
                parts.append(
                    f'sector_leadership({sl_val:.2f})<={sl_th} '
                    f'& return_20d_intra_z({ret20_val:.2f})<={ret20_th}'
                )
            if rule4_hit:
                r4_reasons = []
                if pd.notna(ulcer_val) and ulcer_val < ulcer_th:
                    r4_reasons.append('ulcer=%.2f' % ulcer_val)
                if pd.notna(mom_peak_val) and mom_peak_val > mom_peak_th:
                    r4_reasons.append('mom_peak=%.1f' % mom_peak_val)
                if pd.notna(sector_rank_val) and sector_rank_val < sector_rank_th:
                    r4_reasons.append('sector_rank=%.2f' % sector_rank_val)
                parts.append(f'liq({liq_val:.2f})<{liq_th} & ({"/".join(r4_reasons)})')
            if rule5_hit:
                parts.append(f'disp({disp_val:.2f})<{disp_th} & adr({adr_val:.2f})<{adr_th}')
            if rule6_hit:
                parts.append('limit_down_3d=1')
            if rule7_hit:
                parts.append(f'top10({top10_val:.2f})>={top10_th} & buy_sm({buy_sm_val:.2f})<={buy_sm_th}')
            if rule8_hit:
                parts.append(f'non_first_limit_pct20({non_first_pct20_val:.2f})>={non_first_pct20_th} & kdj_j_9_z({kdj_j_val:.2f})<={kdj_j_th}')
            if rule9_hit:
                parts.append(f'score_z_in_day({score_z_val:.2f})<={score_z_th}')
            if rule10_hit:
                parts.append(f'reg_severe_pct({reg_severe_pct_val:.4f})>={reg_severe_pct_th} & adr({adr_val:.2f})>={reg_hot_adr_th}')
            if rule11_hit:
                parts.append(
                    f'ma5_raw({ma5_raw_val:.2f})>={ma5_raw_th}'
                    f' & ma_conv({ma_conv_val:.2f})>={ma_conv_th}'
                )
            if rule12_hit:
                parts.append(
                    f'ma5_pullback({ma5_pullback_val:.4f})<={ma5_pullback_th}'
                    f' & ret_accel({ret_accel_val:.2f})>={ret_accel_th}'
                )

            composite = np.nan
            if pd.notna(sl_val) and pd.notna(ret20_val):
                composite = max(0.0, -float(sl_val)) * max(0.0, float(ret20_th) - float(ret20_val))

            return {
                'blocked': bool(rule1_hit or rule2_hit or rule3_hit or rule4_hit or rule5_hit or rule6_hit or rule7_hit or rule8_hit or rule9_hit or rule10_hit or rule11_hit or rule12_hit),
                'strategy': strategy,
                'reason': ' | '.join(parts),
                'rule1_hit': bool(rule1_hit),
                'rule2_hit': bool(rule2_hit),
                'rule3_hit': bool(rule3_hit),
                'rule4_hit': bool(rule4_hit),
                'rule5_hit': bool(rule5_hit),
                'rule6_hit': bool(rule6_hit),
                'rule7_hit': bool(rule7_hit),
                'rule8_hit': bool(rule8_hit),
                'rule9_hit': bool(rule9_hit),
                'rule10_hit': bool(rule10_hit),
                'rule11_hit': bool(rule11_hit),
                'rule12_hit': bool(rule12_hit),
                'runup_ma5_exhaustion': rma_val,
                'cpv_factor': cpv_val,
                'prev_day_blowoff_risk': blowoff_val,
                'sector_momentum_accel_rank': sector_val,
                'sector_leadership_persistence_v3': sl_val,
                'return_20d_intra_z': ret20_val,
                'sector_leadership_break_no_alpha': composite,
                'score_z_in_day': score_z_val,
            }

        return {
            'blocked': False,
            'strategy': strategy,
            'reason': '',
            'rule1_hit': False,
            'rule2_hit': False,
            'rule3_hit': False,
            'rule4_hit': False,
            'rule5_hit': False,
            'rule6_hit': False,
            'rule7_hit': False,
            'rule8_hit': False,
            'rule9_hit': False,
            'rule10_hit': False,
        }

    def _build_shadow_trade_record(
        self,
        position: Position,
        sell_date: datetime.date,
        exit_reason: str,
        exit_price: Optional[float] = None,
        exit_time: Optional[datetime] = None,
        trading_dates: Optional[List[datetime.date]] = None,
    ) -> Dict[str, Any]:
        """构造影子交易记录，不更新现金、持仓或真实交易列表。"""
        if exit_time is not None:
            sell_datetime = exit_time
        elif exit_price is None:
            stock_data = self.get_stock_data(position.stock_code, sell_date)
            if stock_data.empty:
                exit_price = position.buy_price * 0.98
                sell_datetime = datetime.combine(sell_date, datetime.min.time())
            else:
                exit_price = float(stock_data.iloc[-1]['close'])
                raw_sell_datetime = stock_data.iloc[-1]['datetime']
                sell_datetime = self._to_python_datetime(raw_sell_datetime)
        else:
            sell_datetime = datetime.combine(sell_date, datetime.min.time())

        actual_return = (float(exit_price) - position.buy_price) / position.buy_price
        realized_holding_days = (sell_date - position.buy_date).days
        if trading_dates is not None:
            try:
                buy_idx = trading_dates.index(position.buy_date)
                sell_idx = trading_dates.index(sell_date)
                realized_holding_days = sell_idx - buy_idx + 1
            except ValueError:
                realized_holding_days = (sell_date - position.buy_date).days

        return {
            'stock_code': position.stock_code,
            'buy_datetime': position.buy_datetime,
            'sell_datetime': sell_datetime,
            'buy_price': position.buy_price,
            'sell_price': float(exit_price),
            'quantity': position.quantity,
            'predicted_return': position.predicted_return,
            'actual_return': actual_return,
            'profit_loss': (float(exit_price) - position.buy_price) * position.quantity,
            'profit_loss_pct': actual_return * 100,
            'holding_days': int(realized_holding_days),
            'entry_reason': position.entry_reason,
            'exit_reason': exit_reason,
            'strategy_type': f"Top-{self.config.stock_count}",
        }

    def _simulate_shadow_position_trade(
        self,
        stock: pd.Series,
        stock_code: str,
        prediction_date: datetime.date,
        trade_date: datetime.date,
        trading_dates: List[datetime.date],
        buy_price: float,
        buy_datetime: datetime,
        prev_close: Optional[float],
        shadow_skip_reason: str,
        entry_check: Dict[str, Any],
        exhaustion_info: Dict[str, Any],
        strict_risk_mode: bool = False,
    ) -> Dict[str, Any]:
        effective_holding_days = max(1, int(self.config.holding_days))
        holding_end_date = self.add_trading_days(trade_date, effective_holding_days - 1, trading_dates)
        if holding_end_date is None:
            holding_end_date = trade_date + timedelta(days=effective_holding_days - 1)

        virtual_notional = max(float(self.config.initial_capital) * float(self.config.fixed_position_pct), buy_price * 100.0)
        quantity = max(100, int(virtual_notional / buy_price / 100) * 100)
        t1_day_lowest_price = 0.0
        if self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
            t1_day_lowest = self._get_daily_low(stock_code, trade_date)
            t1_day_lowest_price = float(t1_day_lowest) if t1_day_lowest is not None else 0.0

        shadow_position = Position(
            stock_code=stock_code,
            buy_date=trade_date,
            buy_datetime=buy_datetime,
            buy_price=float(buy_price),
            highest_price_since_buy=float(buy_price),
            quantity=int(quantity),
            predicted_return=float(stock.get('predicted_return', np.nan)),
            holding_end_date=holding_end_date,
            entry_reason=f"shadow:{shadow_skip_reason}",
            holding_days=effective_holding_days,
            slot_index=-1,
            t1_day_lowest_price=t1_day_lowest_price,
            last_valid_mark_price=float(buy_price),
            strict_risk_mode=bool(strict_risk_mode),
        )

        buy_idx = trading_dates.index(trade_date)
        for check_date in trading_dates[buy_idx:]:
            risk_check = self.check_position_risk(shadow_position, check_date, trading_dates)
            if risk_check.get('should_exit'):
                trade_record = self._build_shadow_trade_record(
                    shadow_position,
                    check_date,
                    f"风控: {risk_check.get('reason', '')}",
                    risk_check.get('exit_price'),
                    exit_time=risk_check.get('exit_time'),
                    trading_dates=trading_dates,
                )
                break
            if check_date >= holding_end_date:
                trade_record = self._build_shadow_trade_record(
                    shadow_position,
                    check_date,
                    "到期",
                    trading_dates=trading_dates,
                )
                break
        else:
            trade_record = self._build_shadow_trade_record(
                shadow_position,
                trading_dates[-1],
                "到期",
                trading_dates=trading_dates,
            )

        trade_record.update({
            'shadow_trade': True,
            'shadow_skip_reason': shadow_skip_reason,
            'shadow_prediction_date': prediction_date,
            'shadow_attempt_buy_date': trade_date,
            'entry_can_buy': bool(entry_check.get('can_buy', False)),
            'entry_reason_detail': entry_check.get('reason', ''),
            'entry_gap_ratio': entry_check.get('gap_ratio', np.nan),
            'prev_close': prev_close,
            'exhaustion_blocked': bool(exhaustion_info.get('blocked', False)),
            'exhaustion_strategy': exhaustion_info.get('strategy', ''),
            'exhaustion_reason': exhaustion_info.get('reason', ''),
            'exhaustion_rule1_hit': bool(exhaustion_info.get('rule1_hit', False)),
            'exhaustion_rule2_hit': bool(exhaustion_info.get('rule2_hit', False)),
            'exhaustion_rule3_hit': bool(exhaustion_info.get('rule3_hit', False)),
            'exhaustion_rule4_hit': bool(exhaustion_info.get('rule4_hit', False)),
            'exhaustion_rule5_hit': bool(exhaustion_info.get('rule5_hit', False)),
            'exhaustion_rule6_hit': bool(exhaustion_info.get('rule6_hit', False)),
            'exhaustion_rule7_hit': bool(exhaustion_info.get('rule7_hit', False)),
            'exhaustion_rule8_hit': bool(exhaustion_info.get('rule8_hit', False)),
            'exhaustion_rule9_hit': bool(exhaustion_info.get('rule9_hit', False)),
            'exhaustion_rule10_hit': bool(exhaustion_info.get('rule10_hit', False)),
            'runup_ma5_exhaustion': exhaustion_info.get('runup_ma5_exhaustion', np.nan),
            'cpv_factor': exhaustion_info.get('cpv_factor', np.nan),
            'prev_day_blowoff_risk': exhaustion_info.get('prev_day_blowoff_risk', np.nan),
            'sector_momentum_accel_rank': exhaustion_info.get('sector_momentum_accel_rank', np.nan),
            'sector_leadership_persistence_v3': exhaustion_info.get('sector_leadership_persistence_v3', np.nan),
            'return_20d_intra_z': exhaustion_info.get('return_20d_intra_z', np.nan),
            'sector_leadership_break_no_alpha': exhaustion_info.get('sector_leadership_break_no_alpha', np.nan),
            'score_z_in_day': exhaustion_info.get('score_z_in_day', np.nan),
        })
        return trade_record

    def _record_full_position_shadow_opportunity(
        self,
        trade_date: datetime.date,
        trading_dates: List[datetime.date],
        shadow_skip_reason: str,
    ) -> None:
        """满仓跳过时记录Top1影子机会，不占仓位、不消耗现金、不改变组合结果。"""
        if trade_date >= trading_dates[-1]:
            return
        prediction_date = self._get_latest_prediction_date_before(trade_date)
        if prediction_date is None:
            return

        target_stock, selected_stocks, select_reason = self._select_target_stock_for_date(
            prediction_date,
            log_prefix="🧪 [SHADOW] ",
        )
        if target_stock is None:
            self.shadow_skipped_opportunity_trades.append({
                'shadow_trade': True,
                'shadow_status': 'selection_skipped',
                'shadow_skip_reason': shadow_skip_reason,
                'shadow_prediction_date': prediction_date,
                'shadow_attempt_buy_date': trade_date,
                'skip_reason': select_reason,
            })
            return

        stock_code = target_stock['stock_code']
        logger.info(
            f"🧪 [SHADOW] 满仓跳过机会评估: {stock_code}, "
            f"T日={prediction_date}, T+1日={trade_date}"
        )
        self._preload_stocks_data([prediction_date, trade_date], [stock_code])

        prediction_day_data = self.get_stock_data(stock_code, prediction_date)
        prev_close = None
        if not prediction_day_data.empty:
            prev_close = float(prediction_day_data.iloc[-1]['close'])

        execution_day_data = self.get_stock_data(stock_code, trade_date)
        if execution_day_data.empty:
            self.shadow_skipped_opportunity_trades.append({
                'shadow_trade': True,
                'shadow_status': 'no_minute_data',
                'stock_code': stock_code,
                'shadow_skip_reason': shadow_skip_reason,
                'shadow_prediction_date': prediction_date,
                'shadow_attempt_buy_date': trade_date,
                'predicted_return': target_stock.get('predicted_return', np.nan),
            })
            return

        buy_open = float(execution_day_data.iloc[0]['open'])
        entry_check = self.check_entry_signal(
            stock_code, trade_date, prev_close, buy_open, target_rank=self.config.stock_count
        )
        exhaustion_info = self._evaluate_exhaustion_for_stock(target_stock)

        delayed_entry = self._resolve_delayed_entry(stock_code, trade_date, execution_day_data)
        if not delayed_entry.get('can_buy', False):
            self.shadow_skipped_opportunity_trades.append({
                'shadow_trade': True,
                'shadow_status': 'delayed_entry_rejected',
                'stock_code': stock_code,
                'shadow_skip_reason': shadow_skip_reason,
                'shadow_prediction_date': prediction_date,
                'shadow_attempt_buy_date': trade_date,
                'buy_open': buy_open,
                'prev_close': prev_close,
                'predicted_return': target_stock.get('predicted_return', np.nan),
                'entry_can_buy': bool(entry_check.get('can_buy', False)),
                'entry_reason_detail': entry_check.get('reason', ''),
                'skip_reason': delayed_entry.get('reason', ''),
                'exhaustion_blocked': bool(exhaustion_info.get('blocked', False)),
                'exhaustion_reason': exhaustion_info.get('reason', ''),
            })
            return

        buy_price = float(delayed_entry['buy_price'])
        buy_datetime = delayed_entry['buy_datetime']
        strict_risk_mode = False
        if self.config.enable_gate_hit_strict_mode and prev_close not in (None, 0):
            strict_gap_ratio = (buy_price - float(prev_close)) / float(prev_close)
            if strict_gap_ratio > self.config.gate_hit_entry_gap_up_threshold:
                self.shadow_skipped_opportunity_trades.append({
                    'shadow_trade': True,
                    'shadow_status': 'strict_gap_rejected',
                    'stock_code': stock_code,
                    'shadow_skip_reason': shadow_skip_reason,
                    'shadow_prediction_date': prediction_date,
                    'shadow_attempt_buy_date': trade_date,
                    'buy_price': buy_price,
                    'prev_close': prev_close,
                    'predicted_return': target_stock.get('predicted_return', np.nan),
                    'entry_can_buy': bool(entry_check.get('can_buy', False)),
                    'entry_reason_detail': entry_check.get('reason', ''),
                    'exhaustion_blocked': bool(exhaustion_info.get('blocked', False)),
                    'exhaustion_reason': exhaustion_info.get('reason', ''),
                })
                return

        trade_record = self._simulate_shadow_position_trade(
            stock=target_stock,
            stock_code=stock_code,
            prediction_date=prediction_date,
            trade_date=trade_date,
            trading_dates=trading_dates,
            buy_price=buy_price,
            buy_datetime=buy_datetime,
            prev_close=prev_close,
            shadow_skip_reason=shadow_skip_reason,
            entry_check=entry_check,
            exhaustion_info=exhaustion_info,
            strict_risk_mode=strict_risk_mode,
        )
        trade_record['shadow_status'] = (
            'entry_rejected_but_simulated'
            if not entry_check.get('can_buy', False)
            else 'exhaustion_blocked_but_simulated'
            if exhaustion_info.get('blocked', False)
            else 'simulated'
        )
        self.shadow_skipped_opportunity_trades.append(trade_record)
        logger.info(
            f"🧪 [SHADOW] {stock_code} 满仓机会收益: "
            f"{trade_record.get('actual_return', np.nan) * 100:.2f}%, "
            f"status={trade_record['shadow_status']}, "
            f"exhaustion={trade_record.get('exhaustion_reason', '') or 'PASS'}"
        )

    def execute_sell(self, position: Position, sell_date: datetime.date,
                     exit_reason: str, exit_price: Optional[float] = None,
                     exit_time: Optional[datetime] = None,
                     trading_dates: Optional[List[datetime.date]] = None) -> Dict[str, Any]:
        """执行卖出操作"""
        if exit_time is not None:
            sell_datetime = exit_time
        elif exit_price is None:
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

        realized_holding_days = (sell_date - position.buy_date).days
        if trading_dates is not None:
            try:
                buy_idx = trading_dates.index(position.buy_date)
                sell_idx = trading_dates.index(sell_date)
                realized_holding_days = sell_idx - buy_idx + 1
            except ValueError:
                realized_holding_days = (sell_date - position.buy_date).days

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
            'holding_days': int(realized_holding_days),
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

        class _SkipBuyDay(Exception):
            """只跳过当日买入流程，不跳过后续风控与到期卖出。"""
            pass

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
            if day_start_open_slots <= 0:
                logger.info(f"⏸️ 开盘前无可用槽位(持仓{day_start_holdings_count}/{self.config.max_concurrent_positions}), 跳过买入")
                self._record_full_position_shadow_opportunity(
                    trade_date=trade_date,
                    trading_dates=trading_dates,
                    shadow_skip_reason=f"开盘前无可用槽位(持仓{day_start_holdings_count}/{self.config.max_concurrent_positions})",
                )
            if day_start_open_slots > 0 and i < len(trading_dates) - 1 and day_start_free_slot_index is not None:
                try:
                    # 检查是否是正确的买入时机
                    should_buy = self._should_attempt_buy(
                        trade_date,
                        trading_dates,
                        day_start_open_slots=day_start_open_slots
                    )

                    if not should_buy['should_buy']:
                        logger.info(f"⏸️ {should_buy['reason']}, 跳过买入")
                        raise _SkipBuyDay()

                    # T+1交易逻辑：使用T日预测数据在T+1日买入
                    prediction_date = should_buy['prediction_date']
                    logger.info(f"🔍 T+1交易逻辑: T+1日={trade_date}, T日={prediction_date}，开始选股")

                    # 获取预测数据 - 使用pre_analyze_required_stocks保存的Top-K股票列表
                    prediction_date_str = prediction_date.strftime('%Y-%m-%d')
                    top_k_stocks = self.required_stocks_by_date.get(prediction_date_str, [])

                    if not top_k_stocks:
                        logger.warning(f"⚠️ {prediction_date} 无预分析的Top-K股票，跳过买入")
                        raise _SkipBuyDay()

                    logger.info(f"📋 {prediction_date} 预分析Top-{len(top_k_stocks)}股票: {top_k_stocks[:3]}{'...' if len(top_k_stocks) > 3 else ''}")

                    # 只获取预分析的Top-K股票的预测数据，保持数据一致性
                    daily_predictions = self.prediction_data[
                        (self.prediction_data['date'] == prediction_date) &
                        (self.prediction_data['stock_code'].isin(top_k_stocks))
                    ]

                    if daily_predictions.empty:
                        logger.warning(f"⚠️ {prediction_date} Top-K股票无预测数据，跳过")
                        raise _SkipBuyDay()

                    daily_predictions = self._attach_score_z_in_day(daily_predictions)
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

                    target_rank = self.config.stock_count
                    regime_override_active = False
                    regime_skip_buy = False
                    if 'regime_skip_buy' in selected_stocks.columns:
                        skip_series = selected_stocks['regime_skip_buy']
                        if len(skip_series) > 0:
                            first_skip = skip_series.iloc[0]
                            regime_skip_buy = bool(first_skip) if pd.notna(first_skip) else False
                            if regime_skip_buy:
                                logger.info(
                                    f"🧭 Top1开关空仓生效: prediction_date={prediction_date}, "
                                    f"跳过当日所有买入"
                                )
                                raise _SkipBuyDay()
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
                    if (
                        self.config.enable_dynamic_rank_switch and
                        'regime_skip_buy' in selected_stocks.columns and
                        regime_skip_buy
                    ):
                        logger.info(
                            f"🧭 日级状态空仓生效: prediction_date={prediction_date}, "
                            f"跳过当日所有买入"
                        )
                        raise _SkipBuyDay()

                    if len(selected_stocks) < target_rank:
                        logger.warning(f"⚠️ 候选股票数{len(selected_stocks)}不足目标排名{target_rank}，跳过买入")
                        raise _SkipBuyDay()

                    if self.config.enable_dynamic_rank_switch and target_rank == 1 and not regime_override_active:
                        target_stock = self._select_target_stock_for_top1_fusion(selected_stocks, alpha=0.1)
                        if target_stock is None:
                            logger.warning(f"⚠️ 候选股票为空，跳过买入")
                            raise _SkipBuyDay()
                        target_stocks_for_today = [target_stock['stock_code']]
                    else:
                        target_stock = selected_stocks.iloc[target_rank - 1]
                        target_stocks_for_today = [target_stock['stock_code']]
                    logger.info(f"📦 预加载{trade_date}和{prediction_date}的目标股票5分钟数据: 排名第{target_rank}的{target_stock['stock_code']}")

                    self._preload_stocks_data([prediction_date, trade_date], target_stocks_for_today)

                    logger.info(f"🎯 处理Top-{self.config.stock_count}策略: 买入排名第{target_rank}的股票")
                    stock = target_stock
                    stock_code = stock['stock_code']
                    overlay_action = str(stock.get('overlay_action', 'none')).lower()
                    overlay_scale = float(stock.get('overlay_scale', 1.0) if pd.notna(stock.get('overlay_scale', 1.0)) else 1.0)
                    overlay_reason = str(stock.get('overlay_reason', '') or '')

                    if target_rank == 1 and overlay_action in {'skip', 'half', 'full'}:
                        logger.info(
                            f"🛡️ Top1 Overlay决策: {stock_code} -> {overlay_action.upper()} "
                            f"(scale={overlay_scale:.2f}, reason={overlay_reason}, "
                            f"gap12_rel={float(stock.get('overlay_margin12_rel', np.nan)):.4f}, "
                            f"certainty={float(stock.get('overlay_std_certainty_pct', np.nan)):.4f})"
                        )

                    logger.info(f"📅 {stock_code} T+1买入逻辑: T日={prediction_date}(预测日), T+1日={trade_date}(执行日)")
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

                    if buy_data.empty:
                        logger.warning(f"⚠️ {stock_code} {trade_date}无5分钟数据，跳过买入")
                        raise _SkipBuyDay()

                    if target_rank == 1 and overlay_action == 'skip':
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
                        raise _SkipBuyDay()

                    buy_open = buy_data.iloc[0]['open']
                    prev_close_str = f"{prev_close:.4f}" if prev_close is not None else "N/A"
                    logger.info(f"🔍 {stock_code} 买入检查: 开盘价={buy_open:.4f}, 前收盘={prev_close_str}")

                    entry_check = self.check_entry_signal(
                        stock_code, trade_date, prev_close, buy_open, target_rank=target_rank
                    )

                    logger.info(f"🚦 {stock_code} 买入信号: {entry_check['can_buy']}, 原因: {entry_check['reason']}")

                    if entry_check['can_buy']:
                        # ================================================================
                        # Exhaustion filter: 检查模型Top1选中的股票是否触发高位衰竭信号
                        # 注意：仅检查具体选中的股票，不预过滤候选池，避免模型选第二名
                        # ================================================================
                        if self.config.enable_exhaustion_filter:
                            strategy = self.config.exhaustion_filter_strategy
                            blocked = False
                            block_reason = ''

                            if strategy == 'three_rule_v3':
                                # ========================================================
                                # v3: 三规则过滤 (基于Top-1鲁棒性分析)
                                # ========================================================
                                th = self.config.exhaustion_runup_ma5_threshold
                                cpv_th = self.config.exhaustion_v3_cpv_threshold
                                rma_val = stock.get('runup_ma5_exhaustion', 0)
                                cpv_val = stock.get('cpv_factor', 0)
                                rule1_hit = (
                                    pd.notna(rma_val) and pd.notna(cpv_val)
                                    and rma_val > th
                                    and cpv_val <= cpv_th
                                )

                                blowoff_th = self.config.exhaustion_v3_blowoff_threshold
                                sector_th = self.config.exhaustion_v3_sector_accel_threshold
                                blowoff_val = stock.get('prev_day_blowoff_risk', 0)
                                sector_val = stock.get('sector_momentum_accel_rank', 0)
                                rule2_hit = (
                                    pd.notna(blowoff_val) and pd.notna(sector_val)
                                    and (
                                        (blowoff_val >= blowoff_th and sector_val <= sector_th)
                                        or (blowoff_val > 5.0 and sector_val <= 0.50)   # XR2a
                                        or (blowoff_val > 1.0 and sector_val <= 0.15)   # XR2b
                                    )
                                )

                                sl_th = self.config.exhaustion_v3_sector_leadership_threshold
                                ret20_th = self.config.exhaustion_v3_return_20d_intra_z_threshold
                                sl_val = stock.get('sector_leadership_persistence_v3', 0)
                                ret20_val = stock.get('return_20d_intra_z', np.nan)
                                rule3_hit = (
                                    pd.notna(sl_val) and pd.notna(ret20_val)
                                    and sl_val <= sl_th
                                    and ret20_val <= ret20_th
                                )

                                # Rule 4: 大盘流动性弱 + 个股三项至少其一
                                liq_th = self.config.exhaustion_v3_market_liq_threshold
                                ulcer_th = self.config.exhaustion_v3_ulcer_threshold
                                mom_peak_th = self.config.exhaustion_v3_mom_peak_threshold
                                sector_rank_th = self.config.exhaustion_v3_intra_sector_rank_threshold
                                liq_val = stock.get('market_liquidity_change', np.nan)
                                ulcer_val = stock.get('ulcer_index', np.nan)
                                mom_peak_val = stock.get('momentum_peak_decay_3d', np.nan)
                                sector_rank_val = stock.get('intra_sector_momentum_rank', np.nan)
                                rule4_hit = (
                                    pd.notna(liq_val)
                                    and liq_val < liq_th
                                    and (
                                        (pd.notna(ulcer_val) and ulcer_val < ulcer_th)
                                        or (pd.notna(mom_peak_val) and mom_peak_val > mom_peak_th)
                                        or (pd.notna(sector_rank_val) and sector_rank_val < sector_rank_th)
                                    )
                                )

                                # Rule 5: 市场熔断 — 安静熊市
                                disp_th = self.config.exhaustion_v3_mkt_disp_threshold
                                adr_th = self.config.exhaustion_v3_mkt_adr_threshold
                                disp_val = stock.get('mkt_dispersion_shock_v2', np.nan)
                                adr_val = stock.get('market_advance_decline_ratio', np.nan)
                                rule5_hit = (
                                    pd.notna(disp_val) and pd.notna(adr_val)
                                    and disp_val < disp_th
                                    and adr_val < adr_th
                                )

                                # Rule 6: 近3日有过跌停
                                rule6_hit = (
                                    self.config.enable_exhaustion_limit_down_3d
                                    and stock.get('limit_down_3d', 0) == 1
                                )

                                # Rule 7: 短强但散户资金流出
                                top10_th = self.config.exhaustion_v3_top10_strength_threshold
                                buy_sm_th = self.config.exhaustion_v3_buy_sm_amount_threshold
                                top10_val = stock.get('top10_10d_strength_level_v3', np.nan)
                                buy_sm_val = stock.get('buy_sm_amount_rate', np.nan)
                                rule7_hit = (
                                    pd.notna(top10_val) and pd.notna(buy_sm_val)
                                    and top10_val >= top10_th
                                    and buy_sm_val <= buy_sm_th
                                )

                                # Rule 8: 连板情绪高分位 + 个股KDJ-J截面动能偏弱
                                non_first_pct20_th = self.config.exhaustion_v3_non_first_limit_up_perf_pct20_threshold
                                kdj_j_th = self.config.exhaustion_v3_kdj_j_z_threshold
                                non_first_pct20_val = stock.get('prev_non_first_limit_up_performance_percentile_20d', np.nan)
                                kdj_j_val = stock.get('kdj_j_9', np.nan)
                                rule8_hit = (
                                    pd.notna(non_first_pct20_val) and pd.notna(kdj_j_val)
                                    and non_first_pct20_val >= non_first_pct20_th
                                    and kdj_j_val <= kdj_j_th
                                )

                                # Rule 9: 当日Top1模型分数相对候选池预测分布偏弱
                                score_z_th = self.config.exhaustion_v3_score_z_in_day_threshold
                                score_z_val = stock.get('score_z_in_day', np.nan)
                                rule9_hit = (
                                    pd.notna(score_z_val)
                                    and score_z_val <= score_z_th
                                )

                                # Rule 10: 个股严重上涨占比极端 + 市场广度过热
                                reg_severe_pct_th = self.config.exhaustion_v3_reg_severe_up_10d_pct_threshold
                                reg_hot_adr_th = self.config.exhaustion_v3_reg_hot_adr_threshold
                                reg_severe_pct_val = stock.get('reg_severe_up_10d_ratio_pct', np.nan)
                                rule10_hit = (
                                    pd.notna(reg_severe_pct_val) and pd.notna(adr_val)
                                    and reg_severe_pct_val >= reg_severe_pct_th
                                    and adr_val >= reg_hot_adr_th
                                )

                                if rule1_hit or rule2_hit or rule3_hit or rule4_hit or rule5_hit or rule6_hit or rule7_hit or rule8_hit or rule9_hit or rule10_hit:
                                    blocked = True
                                    parts = []
                                    if rule1_hit:
                                        parts.append(f'runup({rma_val:.2f})>{th} & cpv({cpv_val:.2f})<={cpv_th}')
                                    if rule2_hit:
                                        if blowoff_val >= blowoff_th and sector_val <= sector_th:
                                            parts.append(f'blowoff({blowoff_val:.2f})>={blowoff_th} & sector_accel({sector_val:.2f})<={sector_th}')
                                        elif blowoff_val > 5.0:
                                            parts.append(f'XR2a:blowoff({blowoff_val:.2f})>5.0 & sector_accel({sector_val:.2f})<=0.50')
                                        else:
                                            parts.append(f'XR2b:blowoff({blowoff_val:.2f})>1.0 & sector_accel({sector_val:.2f})<=0.15')
                                    if rule3_hit:
                                        parts.append(
                                            f'sector_leadership({sl_val:.2f})<={sl_th} '
                                            f'& return_20d_intra_z({ret20_val:.2f})<={ret20_th}'
                                        )
                                    if rule4_hit:
                                        r4_reasons = []
                                        if pd.notna(ulcer_val) and ulcer_val < ulcer_th:
                                            r4_reasons.append('ulcer=%.2f' % ulcer_val)
                                        if pd.notna(mom_peak_val) and mom_peak_val > mom_peak_th:
                                            r4_reasons.append('mom_peak=%.1f' % mom_peak_val)
                                        if pd.notna(sector_rank_val) and sector_rank_val < sector_rank_th:
                                            r4_reasons.append('sector_rank=%.2f' % sector_rank_val)
                                        parts.append(f'liq({liq_val:.2f})<{liq_th} & ({"/".join(r4_reasons)})')
                                    if rule5_hit:
                                        parts.append(f'disp({disp_val:.2f})<{disp_th} & adr({adr_val:.2f})<{adr_th}')
                                    if rule6_hit:
                                        parts.append('limit_down_3d=1')
                                    if rule7_hit:
                                        parts.append(f'top10({top10_val:.2f})>={top10_th} & buy_sm({buy_sm_val:.2f})<={buy_sm_th}')
                                    if rule8_hit:
                                        parts.append(f'non_first_limit_pct20({non_first_pct20_val:.2f})>={non_first_pct20_th} & kdj_j_9_z({kdj_j_val:.2f})<={kdj_j_th}')
                                    if rule9_hit:
                                        parts.append(f'score_z_in_day({score_z_val:.2f})<={score_z_th}')
                                    if rule10_hit:
                                        parts.append(f'reg_severe_pct({reg_severe_pct_val:.4f})>={reg_severe_pct_th} & adr({adr_val:.2f})>={reg_hot_adr_th}')
                                    block_reason = ' | '.join(parts)

                            elif strategy in ('runup_ma5', 'runup_or_blowoff'):
                                # v2: 连续因子 + 头部中期强度环境确认
                                th = self.config.exhaustion_runup_ma5_threshold
                                top10_th = self.config.exhaustion_top10_10d_strength_threshold
                                val = stock.get('runup_ma5_exhaustion', 0)
                                top10_val = stock.get('top10_10d_strength_level_v3', 0)
                                runup_hit = (
                                    pd.notna(val)
                                    and pd.notna(top10_val)
                                    and val > th
                                    and top10_val >= top10_th
                                )
                                if strategy == 'runup_ma5' and runup_hit:
                                    blocked = True
                                    block_reason = (
                                        f'runup_ma5_exhaustion({val:.2f})>{th} '
                                        f'AND top10_10d_strength_level_v3({top10_val:.2f})>={top10_th}'
                                    )
                                elif strategy == 'runup_or_blowoff':
                                    blowoff_th = self.config.exhaustion_prev_day_blowoff_threshold
                                    sector_weak_th = self.config.exhaustion_sector_momentum_accel_weak_threshold
                                    blowoff_val = stock.get('prev_day_blowoff_risk', 0)
                                    sector_val = stock.get('sector_momentum_accel_rank', 0)
                                    blowoff_hit = (
                                        pd.notna(blowoff_val)
                                        and pd.notna(sector_val)
                                        and blowoff_val >= blowoff_th
                                        and sector_val <= sector_weak_th
                                    )
                                    market_ad_th = self.config.exhaustion_market_advance_decline_ratio_threshold
                                    market_ad_val = stock.get('market_advance_decline_ratio', np.nan)
                                    market_ad_hit = (
                                        pd.notna(market_ad_val)
                                        and market_ad_val >= market_ad_th
                                    )
                                    if runup_hit or blowoff_hit or market_ad_hit:
                                        blocked = True
                                        parts = []
                                        if runup_hit:
                                            parts.append(
                                                f'runup_ma5_exhaustion({val:.2f})>{th} '
                                                f'AND top10_10d_strength_level_v3({top10_val:.2f})>={top10_th}'
                                            )
                                        if blowoff_hit:
                                            parts.append(
                                                f'prev_day_blowoff_risk({blowoff_val:.2f})>={blowoff_th} '
                                                f'AND sector_momentum_accel_rank({sector_val:.2f})<={sector_weak_th}'
                                            )
                                        if market_ad_hit:
                                            parts.append(
                                                f'market_advance_decline_ratio({market_ad_val:.2f})>={market_ad_th}'
                                            )
                                        block_reason = ' OR '.join(parts)
                            else:
                                # v1: close_below_ma5 AND turnover_z 双重确认
                                has_flag = False
                                flag_val = stock.get('close_below_ma5_after_runup', 0)
                                if pd.notna(flag_val) and flag_val == 1:
                                    has_flag = True
                                tz_th = self.config.exhaustion_filter_turnover_z_threshold
                                tz_val = stock.get('turnover_z_20', 0)
                                has_tz = pd.notna(tz_val) and tz_val > tz_th
                                if has_flag and has_tz:
                                    blocked = True
                                    block_reason = f'close_below_ma5=1 AND tz({tz_val:.2f})>{tz_th}'

                            if blocked:
                                entry_check = {
                                    'can_buy': False,
                                    'reason': f'高位衰竭({strategy}): {block_reason}',
                                    'skip_category': '高位衰竭过滤'
                                }
                                logger.info(
                                    f"🛡️ {stock_code} exhaustion BLOCKED [{strategy}]: {block_reason}"
                                )
                            else:
                                # 通过过滤也打印因子值，便于观察
                                if strategy == 'three_rule_v3':
                                    rma_val = stock.get('runup_ma5_exhaustion', None)
                                    cpv_val = stock.get('cpv_factor', None)
                                    blowoff_val = stock.get('prev_day_blowoff_risk', None)
                                    sector_val = stock.get('sector_momentum_accel_rank', None)
                                    sl_val = stock.get('sector_leadership_persistence_v3', None)
                                    ret20_val = stock.get('return_20d_intra_z', None)
                                    liq_val = stock.get('market_liquidity_change', None)
                                    mfp_pct_val = stock.get('money_flow_pressure_pct', None)
                                    parts = []
                                    if rma_val is not None and pd.notna(rma_val):
                                        parts.append(f"runup={rma_val:.2f}")
                                    if cpv_val is not None and pd.notna(cpv_val):
                                        parts.append(f"cpv={cpv_val:.2f}")
                                    if blowoff_val is not None and pd.notna(blowoff_val):
                                        parts.append(f"blowoff={blowoff_val:.2f}")
                                    if sector_val is not None and pd.notna(sector_val):
                                        parts.append(f"sector_accel={sector_val:.2f}")
                                    if sl_val is not None and pd.notna(sl_val):
                                        parts.append(f"sector_leadership={sl_val:.2f}")
                                    if ret20_val is not None and pd.notna(ret20_val):
                                        parts.append(f"return_20d_intra_z={ret20_val:.2f}")
                                    disp_val = stock.get('mkt_dispersion_shock_v2', None)
                                    adr_val = stock.get('market_advance_decline_ratio', None)
                                    if liq_val is not None and pd.notna(liq_val):
                                        parts.append(f"liq={liq_val:.2f}")
                                    sector_rank_pass_val = stock.get('intra_sector_momentum_rank', None)
                                    if sector_rank_pass_val is not None and pd.notna(sector_rank_pass_val):
                                        parts.append(f"sector_rank={sector_rank_pass_val:.2f}")
                                    ulcer_pass_val = stock.get('ulcer_index', None)
                                    if ulcer_pass_val is not None and pd.notna(ulcer_pass_val):
                                        parts.append(f"ulcer={ulcer_pass_val:.2f}")
                                    mom_pass_val = stock.get('momentum_peak_decay_3d', None)
                                    if mom_pass_val is not None and pd.notna(mom_pass_val):
                                        parts.append(f"mom_peak={mom_pass_val:.2f}")
                                    if disp_val is not None and pd.notna(disp_val):
                                        parts.append(f"disp={disp_val:.2f}")
                                    if adr_val is not None and pd.notna(adr_val):
                                        parts.append(f"adr={adr_val:.2f}")
                                    lim3d_pass = stock.get('limit_down_3d', 0)
                                    if lim3d_pass:
                                        parts.append("limit_3d=1")
                                    top10_pass = stock.get('top10_10d_strength_level_v3', None)
                                    buy_sm_pass = stock.get('buy_sm_amount_rate', None)
                                    if top10_pass is not None and pd.notna(top10_pass):
                                        parts.append(f"top10={top10_pass:.2f}")
                                    if buy_sm_pass is not None and pd.notna(buy_sm_pass):
                                        parts.append(f"buy_sm={buy_sm_pass:.2f}")
                                    logger.info(
                                        f"✅ {stock_code} exhaustion PASS "
                                        f"[{', '.join(parts)}]"
                                    )
                                else:
                                    rma_val = stock.get('runup_ma5_exhaustion', None)
                                    top10_val = stock.get('top10_10d_strength_level_v3', None)
                                    market_ad_val = stock.get('market_advance_decline_ratio', None)
                                    if rma_val is not None and pd.notna(rma_val):
                                        top10_text = f"{top10_val:.2f}" if top10_val is not None and pd.notna(top10_val) else "NaN"
                                        market_ad_text = (
                                            f"{market_ad_val:.2f}"
                                            if market_ad_val is not None and pd.notna(market_ad_val)
                                            else "NaN"
                                        )
                                        logger.info(
                                            f"✅ {stock_code} exhaustion PASS "
                                            f"[runup_ma5_exhaustion={rma_val:.2f}, "
                                            f"top10_10d_strength_level_v3={top10_text}, "
                                            f"market_advance_decline_ratio={market_ad_text}]"
                                        )
                        # ================================================================
                        if entry_check['can_buy']:
                            base_position_pct = self._resolve_position_pct(day_start_free_slot_index)
                            regime_position_scale = self._resolve_position_scale(selected_stocks)
                            strict_risk_mode = False
                            if 'regime_strict_risk_mode' in selected_stocks.columns:
                                strict_series = selected_stocks['regime_strict_risk_mode']
                                if len(strict_series) > 0:
                                    first_strict = strict_series.iloc[0]
                                    strict_risk_mode = bool(first_strict) if pd.notna(first_strict) else False
                            final_position_pct = base_position_pct * regime_position_scale
                            entry_reason = "normal"
                            logger.info(
                                f"🧭 分层并行仓位生效: {stock_code} 使用 {final_position_pct:.0%} 仓位, "
                                f"(基础槽位={base_position_pct:.0%}, 日级缩放={regime_position_scale:.2f}), "
                                f"严格风控={strict_risk_mode}, "
                                f"分配到槽位={day_start_free_slot_index + 1}/{self.config.max_concurrent_positions}, "
                                f"开盘前持仓数={day_start_holdings_count}/{self.config.max_concurrent_positions}"
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
                                slot_index=day_start_free_slot_index,
                                strict_risk_mode=strict_risk_mode,
                                prev_close=prev_close
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
                except _SkipBuyDay:
                    pass

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
                        risk_check.get('exit_price'),
                        exit_time=risk_check.get('exit_time'),
                        trading_dates=trading_dates
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
                    self.execute_sell(position, trade_date, "到期", trading_dates=trading_dates)

            # 计算当日收益
            daily_pnl = sum([trade['profit_loss'] for trade in self.trades
                             if trade['sell_datetime'].date() == trade_date])
            current_value = self.cash + sum([
                pos.quantity * self._get_safe_mark_price(pos, trade_date)
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
            'shadow_skipped_opportunity_trades': self.shadow_skipped_opportunity_trades,
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
        logger.info(f"   🧪 影子满仓机会: {len(self.shadow_skipped_opportunity_trades)}笔")

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
                profit_bearish = risk_details.get('profit_bearish_exits', 0)  # 获取连续阴线退出次数

                if cumulative > 0:
                    logger.info(f"        - 累计收益低于阈值: {cumulative}次")
                if consecutive > 0:
                    logger.info(f"        - 连续3根阴线: {consecutive}次")
                if ma5_break > 0:  # 🆕 显示MA5跌破统计
                    logger.info(f"        - 日线MA5跌破: {ma5_break}次")
                if t1_low_break > 0:  # 🆕 显示T+1最低价跌破统计
                    logger.info(f"        - T+1最低价跌破: {t1_low_break}次")
                if profit_bearish > 0:
                    logger.info(f"        - 连续3根阴线(T+2/T+3): {profit_bearish}次")

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
        worst_idx = 0

        for idx, value in enumerate(self.daily_portfolio_values):
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                worst_idx = idx

        if max_drawdown > 80:
            worst_value = self.daily_portfolio_values[worst_idx]
            worst_date = None
            if worst_idx > 0 and len(self.trading_dates_list) >= worst_idx:
                worst_date = self.trading_dates_list[worst_idx - 1]
            logger.warning(
                f"⚠️ 检测到异常大回撤: {max_drawdown:.2f}%, "
                f"worst_value={worst_value:.2f}, worst_date={worst_date}"
            )

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
            'shadow_skipped_opportunities': len(self.shadow_skipped_opportunity_trades),
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
        # 连续阴线退出次数（当前规则下不再区分是否盈利状态）
        profit_bearish_exits = len([t for t in risk_exits if '连续3根' in t.get('exit_reason', '') and '阴线' in t.get('exit_reason', '')])

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
            logger.info(f"🔍 [风控统计调试] 连续3根阴线: {profit_bearish_exits}次")

        stats['risk_control_details'] = {
            'total_risk_exits': len(risk_exits),
            'stop_loss_threshold': self.config.stop_loss_threshold,  # 🆕 记录使用的止损阈值
            'stop_loss_threshold_desc': threshold_desc,  # 🆕 阈值描述
            'cumulative_decline_exits': cumulative_decline_exits,
            'consecutive_decline_exits': consecutive_decline_exits,
            'ma5_break_exits': ma5_break_exits,  # 🆕 MA5跌破退出次数
            't1_low_break_exits': t1_low_break_exits,  # 🆕 T+1最低价跌破退出次数
            'profit_bearish_exits': profit_bearish_exits,  # 连续阴线退出次数
            'normal_exits': len(normal_exits),
            'risk_exit_rate': len(risk_exits) / len(self.trades) if self.trades else 0,
            'avg_risk_exit_return': np.mean([t['actual_return'] for t in risk_exits]) if risk_exits else 0,
            'avg_normal_return': np.mean([t['actual_return'] for t in normal_exits]) if normal_exits else 0
        }

        return stats


def get_minute_trade_cache_rule_signature(config: Optional[StrategyConfig] = None) -> str:
    """为分钟交易缓存生成规则签名，便于规则变动后自动失效。"""
    cfg = config or StrategyConfig(strategy_type=StrategyType.TOP_1, stock_count=1)
    return (
        f"hd{int(cfg.holding_days)}"
        f"_entry{cfg.entry_risk_control_strategy}"
        f"_down{cfg.entry_risk_down_threshold:.3f}"
        f"_downn{cfg.entry_risk_down_threshold_non_top1:.3f}"
        f"_up{cfg.entry_risk_up_threshold:.3f}"
        f"_dem{cfg.delayed_entry_mode}"
        f"_risk{cfg.risk_control_strategy}"
        f"_sl{cfg.stop_loss_threshold:.3f}"
        f"_pb{int(cfg.profit_bearish_required_count)}"
        f"_t2s{int(cfg.enable_t2_0935_hard_stop)}"
        f"_t2m{cfg.t2_0935_hard_stop_threshold:.3f}"
        f"_gstr{int(cfg.enable_gate_hit_strict_mode)}"
        f"_gsup{cfg.gate_hit_entry_gap_up_threshold:.3f}"
        f"_sm{int(cfg.enable_top1_state_machine_risk)}"
        f"_s2{int(cfg.enable_second_layer_time_stop)}"
        f"_t2c{cfg.t2_close_breakeven_stop_threshold:.3f}"
        f"_t3o{cfg.t3_0935_breakeven_stop_threshold:.3f}"
        f"_spf{int(cfg.enable_small_profit_failure_exit)}"
        f"_spa{cfg.small_profit_activation_return:.3f}"
        f"_spw{int(cfg.small_profit_watch_window_bars)}"
        f"_spn{int(cfg.small_profit_bearish_required_count)}"
        f"_pba{int(cfg.enable_profit_threshold_for_bearish_exit)}"
        f"_pbr{cfg.profit_bearish_activation_return:.3f}"
        f"_pbp{cfg.profit_bearish_min_peak_return:.3f}"
        f"_pbd{cfg.profit_bearish_trailing_drawdown:.3f}"
        f"_pbc{cfg.profit_bearish_min_current_return:.3f}"
        f"_exf{int(cfg.enable_exhaustion_filter)}"
        f"_exs{cfg.exhaustion_filter_strategy}"
        f"_exv3c{cfg.exhaustion_v3_cpv_threshold:.3f}"
        f"_exv3b{cfg.exhaustion_v3_blowoff_threshold:.3f}"
        f"_exv3a{cfg.exhaustion_v3_sector_accel_threshold:.3f}"
        f"_exv3l{cfg.exhaustion_v3_sector_leadership_threshold:.3f}"
        f"_exv3r20{cfg.exhaustion_v3_return_20d_intra_z_threshold:.3f}"
        f"_exv3sz{cfg.exhaustion_v3_score_z_in_day_threshold:.3f}"
        f"_exr{cfg.exhaustion_runup_ma5_threshold:.3f}"
        f"_ext{cfg.exhaustion_top10_10d_strength_threshold:.3f}"
        f"_exb{cfg.exhaustion_prev_day_blowoff_threshold:.3f}"
        f"_exw{cfg.exhaustion_sector_momentum_accel_weak_threshold:.3f}"
        f"_exm{cfg.exhaustion_market_advance_decline_ratio_threshold:.3f}"
        f"_exz{cfg.exhaustion_filter_turnover_z_threshold:.3f}"
        f"_isplit{int(cfg.enable_intraday_split_entry)}"
        f"_isbuf{cfg.intraday_split_limit_up_buffer:.3f}"
        f"_isbuy{cfg.intraday_split_max_buy_premium:.3f}"
        f"_istail{cfg.intraday_split_tail_stop:.3f}"
    )


def build_minute_trade_cache(
    target_prediction_rows: pd.DataFrame,
    context_prediction_data: pd.DataFrame,
    minute_data_path: str,
    trading_dates: Optional[List[datetime.date]] = None,
    cache_path: Optional[Path] = None,
    holding_days: int = 3,
    overwrite: bool = False,
    log_prefix: str = "[MINUTE_CACHE]"
) -> pd.DataFrame:
    """
    为验证集/selection集股票生成分钟交易结果缓存。
    仅缓存股票-预测日级最终结果，而非完整分钟明细。
    """
    if cache_path is not None and Path(cache_path).exists() and not overwrite:
        logger.info(f"{log_prefix} 命中缓存: {cache_path}")
        return pd.read_parquet(cache_path)

    if target_prediction_rows is None or target_prediction_rows.empty:
        logger.warning(f"{log_prefix} 目标数据为空，返回空缓存")
        empty_df = pd.DataFrame(columns=[
            'date', 'stock_code', 'predicted_return', 'buy_executable', 'skip_reason',
            'buy_price', 'exit_price', 'exit_date', 'exit_reason', 'minute_realized_return',
            'minute_min_return', 'minute_max_drawdown', 'stoploss_hit',
            'profit_bearish_hit', 'holding_days_realized', 'missing_minute_data'
        ])
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_parquet(cache_path, index=False)
        return empty_df

    target_rows = target_prediction_rows.copy()
    context_rows = context_prediction_data.copy()
    target_rows['date'] = pd.to_datetime(target_rows['date']).dt.date
    context_rows['date'] = pd.to_datetime(context_rows['date']).dt.date

    if trading_dates is None:
        trading_dates = sorted(pd.to_datetime(context_rows['date']).dt.date.unique().tolist())
    else:
        trading_dates = sorted(pd.to_datetime(pd.Series(trading_dates)).dt.date.unique().tolist())

    target_rows = (
        target_rows
        .sort_values(['date', 'stock_code'])
        .drop_duplicates(['date', 'stock_code'])
        .reset_index(drop=True)
    )

    cfg = StrategyConfig(
        strategy_type=StrategyType.TOP_1,
        stock_count=1,
        enable_dynamic_rank_switch=False,
        holding_days=holding_days,
        max_concurrent_positions=2,
        fixed_position_pct=0.8,
        position_schedule=[0.8, 0.2]
    )
    strategy = BacktestStrategy(cfg, context_rows, minute_data_path)

    results: List[Dict[str, Any]] = []
    progress_logger = logging.getLogger(f"{__name__}.minute_cache_progress")
    grouped = target_rows.groupby('date', sort=True)
    total_dates = len(grouped)

    for group_idx, (prediction_date, day_df) in enumerate(grouped, 1):
        stock_list = day_df['stock_code'].dropna().astype(str).unique().tolist()
        preload_dates = [prediction_date]
        trade_date = strategy.add_trading_days(prediction_date, 1, trading_dates)
        if trade_date is not None:
            holding_end_date = strategy.add_trading_days(trade_date, holding_days - 1, trading_dates) or trade_date
            try:
                trade_index = trading_dates.index(trade_date)
                end_index = trading_dates.index(holding_end_date)
                preload_dates.extend(trading_dates[trade_index:end_index + 1])
            except ValueError:
                preload_dates.append(trade_date)
                if holding_end_date != trade_date:
                    preload_dates.append(holding_end_date)

        previous_logger_disabled = logger.disabled
        try:
            # 缓存构建是离线批处理，逐票 INFO 日志会极大拖慢速度并放大 IO/内存压力。
            logger.disabled = True
            strategy._preload_stocks_data(preload_dates, stock_list)

            for row in day_df.itertuples(index=False):
                pred_value = getattr(row, 'predicted_return', 0.0)
                results.append(
                    strategy.simulate_single_prediction_outcome(
                        stock_code=row.stock_code,
                        prediction_date=prediction_date,
                        trading_dates=trading_dates,
                        predicted_return=float(pred_value) if pd.notna(pred_value) else 0.0,
                        target_rank=1,
                        preloaded=True
                    )
                )
        finally:
            logger.disabled = previous_logger_disabled

        # 关键优化：每天处理完成后立即释放该批日期的分钟缓存，避免峰值内存持续累积。
        preload_cache_keys = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in preload_dates]
        strategy.clear_minute_data_cache(preload_cache_keys)

        if group_idx % 10 == 0 or group_idx == total_dates:
            progress_logger.info(
                f"{log_prefix} 已生成 {group_idx}/{total_dates} 个交易日缓存, "
                f"累计记录 {len(results):,} 条"
            )

    cache_df = pd.DataFrame(results)
    if not cache_df.empty and 'date' in cache_df.columns:
        cache_df['date'] = pd.to_datetime(cache_df['date'])
    if not cache_df.empty and 'exit_date' in cache_df.columns:
        cache_df['exit_date'] = pd.to_datetime(cache_df['exit_date'], errors='coerce')

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_parquet(cache_path, index=False)
        logger.info(f"{log_prefix} 缓存已保存: {cache_path}")

    return cache_df


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
                       holding_days: int = 2,
                       enable_dynamic_rank_switch: bool = True,
                       max_concurrent_positions: int = 2,
                       fixed_position_pct: float = 0.5,
                       position_schedule: Optional[List[float]] = None,
                       strategy_config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """运行所有策略"""
        logger.info(f"🚀 开始运行 {len(stock_counts)} 个策略")
        logger.info(
            f"🧭 生效参数: holding_days={holding_days}, "
            f"max_concurrent_positions={max_concurrent_positions}, "
            f"fixed_position_pct={fixed_position_pct:.2f}, "
            f"position_schedule={position_schedule}, "
            f"enable_dynamic_rank_switch={enable_dynamic_rank_switch}"
        )

        results = {}

        for stock_count in stock_counts:
            config_kwargs = dict(strategy_config_overrides or {})
            # 显式构造参数已单独传入，避免 overrides 中重复传参导致 TypeError
            for reserved_key in [
                'strategy_type',
                'stock_count',
                'holding_days',
                'enable_dynamic_rank_switch',
                'max_concurrent_positions',
                'fixed_position_pct',
                'position_schedule',
            ]:
                config_kwargs.pop(reserved_key, None)
            config = StrategyConfig(
                strategy_type=StrategyType(stock_count),
                stock_count=stock_count,
                holding_days=holding_days,
                enable_dynamic_rank_switch=enable_dynamic_rank_switch,
                max_concurrent_positions=max_concurrent_positions,
                fixed_position_pct=fixed_position_pct,
                position_schedule=position_schedule,
                **config_kwargs,
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
                          max_concurrent_positions: int = 2,
                          fixed_position_pct: float = 0.5,
                          position_schedule: Optional[List[float]] = None,
                          strategy_config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        holding_days=holding_days,
        enable_dynamic_rank_switch=enable_dynamic_rank_switch,
        max_concurrent_positions=max_concurrent_positions,
        fixed_position_pct=fixed_position_pct,
        position_schedule=position_schedule,
        strategy_config_overrides=strategy_config_overrides,
    )

    return results
