"""
5分钟K线风控监控模块

功能:
1. 订阅5分钟K线数据
2. 实时监控持仓
3. 累计下跌风控（5%）
4. 连续阴线风控（3根阴线）
5. 自动卖出
"""

import json
import time
import logging
import threading
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .config import PathConfig


class PositionMonitor5Min:
    """
    5分钟K线风控监控器

    风控规则：
    1. 累计下跌风控：累计下跌 > 5%（相对于买入价）→ 立即卖出
    2. 连续阴线风控：T+2日开始，出现3根连续的5分钟K线阴线 → 立即卖出
    """

    def __init__(self, trader, scheduler):
        self.logger = logging.getLogger(__name__)
        self.trader = trader
        self.scheduler = scheduler

        # 持仓信息: {stock_code: position_info}
        self.positions = {}

        # 5分钟K线数据缓存: {stock_code: DataFrame}
        self.kline_cache = {}

        # 监控线程
        self.monitor_thread = None
        self.is_monitoring = False

        # 交易日历（用于判断T+2）
        self.trading_dates = []
        self._load_trading_calendar()

        # 监控间隔（5分钟）
        self.monitor_interval = 5 * 60  # 300秒

        # 是否启用尾盘卖出（第3天为True，第2天为False）
        self.enable_closing_sell = False

        # 尾盘卖出时间（14:55）
        self.closing_sell_time = "14:55:00"

    def _load_trading_calendar(self):
        """加载交易日历"""
        try:
            from xtquant import xtdata

            # 获取最近的交易日
            trading_dates = xtdata.get_trading_dates(
                market='SH',
                count=100  # 最近100个交易日
            )

            self.logger.info(f"从xtdata获取到 {len(trading_dates) if trading_dates is not None else 0} 个原始交易日")

            # 转换为date对象（处理多种数据格式）
            self.trading_dates = []
            if trading_dates is None or len(trading_dates) == 0:
                self.logger.warning("未获取到交易日历数据")
                return

            for d in trading_dates:
                try:
                    # 如果是整数，判断长度
                    if isinstance(d, int):
                        date_str = str(d)
                        # 13位时间戳（毫秒级Unix时间戳）
                        if len(date_str) == 13:
                            timestamp = d / 1000  # 转换为秒
                            date_obj = datetime.fromtimestamp(timestamp).date()
                            self.trading_dates.append(date_obj)
                        # 8位数字（YYYYMMDD）
                        elif len(date_str) == 8:
                            date_obj = datetime.strptime(date_str, '%Y%m%d').date()
                            self.trading_dates.append(date_obj)
                        else:
                            self.logger.warning(f"日期格式不正确（长度={len(date_str)}）: {d}")
                    # 如果是字符串，直接解析
                    elif isinstance(d, str):
                        # 尝试不同的格式
                        try:
                            date_obj = datetime.strptime(d, '%Y%m%d').date()
                        except ValueError:
                            # 如果包含时间戳，取前8位
                            date_obj = datetime.strptime(d[:8], '%Y%m%d').date()
                        self.trading_dates.append(date_obj)
                    # 如果是numpy.int64等类型
                    elif hasattr(d, 'item'):
                        date_int = d.item()
                        date_str = str(date_int)
                        # 13位时间戳（毫秒级Unix时间戳）
                        if len(date_str) == 13:
                            timestamp = date_int / 1000  # 转换为秒
                            date_obj = datetime.fromtimestamp(timestamp).date()
                            self.trading_dates.append(date_obj)
                        # 8位数字（YYYYMMDD）
                        elif len(date_str) == 8:
                            date_obj = datetime.strptime(date_str, '%Y%m%d').date()
                            self.trading_dates.append(date_obj)
                        else:
                            self.logger.warning(f"numpy日期格式不正确（长度={len(date_str)}）: {d}")
                    else:
                        self.logger.warning(f"未知的日期类型 {type(d)}: {d}")
                except Exception as e:
                    self.logger.warning(f"无法解析日期 {d} (类型={type(d).__name__}): {e}")
                    continue

            self.logger.info(f"加载交易日历: {len(self.trading_dates)}个交易日")

            if len(self.trading_dates) > 0:
                self.logger.info(f"  最早日期: {self.trading_dates[0]}")
                self.logger.info(f"  最晚日期: {self.trading_dates[-1]}")

        except Exception as e:
            self.logger.error(f"加载交易日历失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            self.trading_dates = []

    def add_position_5m(self, stock_code: str, entry_price: float, entry_date: str):
        """
        添加持仓监控（5分钟K线版本）

        Args:
            stock_code: 股票代码
            entry_price: 买入价格
            entry_date: 买入日期 (YYYY-MM-DD)
        """
        self.positions[stock_code] = {
            'stock_code': stock_code,
            'entry_price': entry_price,
            'entry_date': datetime.strptime(entry_date, '%Y-%m-%d').date(),
            'entry_datetime': datetime.now(),
            'klines_after_entry': [],  # 买入后的5分钟K线
            'consecutive_bearish': 0,  # 连续阴线数
            'max_decline_from_entry': 0.0,  # 最大回撤（相对于买入价）
            'should_exit': False,
            'exit_reason': None,
        }

        self.logger.info(f"添加5分钟监控: {stock_code}")
        self.logger.info(f"  买入价: {entry_price:.2f}")
        self.logger.info(f"  买入日期: {entry_date}")

    def remove_position(self, stock_code: str):
        """移除持仓监控"""
        if stock_code in self.positions:
            del self.positions[stock_code]
            self.logger.info(f"移除监控: {stock_code}")

    def start_5m_monitoring(self):
        """启动5分钟K线监控"""
        if self.is_monitoring:
            self.logger.warning("监控已在运行")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._5m_monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()

        self.logger.info("✅ 启动5分钟K线监控")

    def stop_5m_monitoring(self):
        """停止5分钟K线监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("停止5分钟K线监控")

    def _5m_monitor_loop(self):
        """5分钟监控循环"""
        self.logger.info("5分钟监控循环已启动")

        while self.is_monitoring:
            try:
                # 检查是否为交易时间
                if not self._is_trading_time():
                    # 优化：智能午间休市处理
                    self._sleep_through_lunch_break()
                    continue

                # ===== 收集5分钟K线数据（在5分钟整数倍时间点）=====
                self._collect_5min_kline_data()

                # ===== 打印持仓摘要（每5分钟打印一次）=====
                self._print_positions_summary()

                # 检查所有持仓
                for stock_code in list(self.positions.keys()):
                    position = self.positions[stock_code]

                    # 检查是否需要退出
                    if position['should_exit']:
                        continue

                    # 检查是否到了T+2（买入日的下一个交易日）
                    if not self._is_after_t2(position['entry_date']):
                        self.logger.debug(f"{stock_code}: 未到T+2，跳过风控检查")
                        continue

                    # 执行5分钟K线风控检查（与回测系统一致，等到K线收盘后检查）
                    self._check_position_risk_5m(stock_code, position)

                # 休眠直到下个5分钟整数倍
                self._sleep_to_next_5min()

            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                time.sleep(10)

    def _check_opening_decline_risk(self, stock_code: str, position: Dict, current_price: float):
        """
        开盘风控检查（已废弃，保留用于兼容性）

        ⚠️ 注意：此函数已不再使用，与回测系统保持一致
        实盘监控现在使用K线数据进行风控检查，与回测系统逻辑一致

        Args:
            stock_code: 股票代码
            position: 持仓信息
            current_price: 当前最新价格

        Returns:
            bool: 是否已卖出（始终返回False）
        """
        try:
            # 此函数已废弃，与回测系统保持一致
            # 实盘监控现在使用K线数据进行风控检查
            return False

        except Exception as e:
            self.logger.error(f"开盘风控检查失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _check_position_risk_5m(self, stock_code: str, position: Dict):
        """
        5分钟K线风控检查（与回测系统完全一致）

        风控规则：
        1. 累计收益风控：累计收益低于阈值（相对于买入价）→ 立即卖出
           - 阈值从配置读取（StopLossConfig.MAX_STOP_LOSS_PCT）
           - 使用每根K线的收盘价检查
           - 从第一根5分钟K线（9:35）开始检查
        2. 连续阴线风控：连续3根5分钟K线阴线（同一天内）→ 立即卖出

        执行时机：
        - 从T+2日开始检查
        - 从第一根5分钟K线开始检查（不要求3根K线）
        - 按时间顺序逐个K线检查

        Args:
            stock_code: 股票代码
            position: 持仓信息
        """
        try:
            # 导入配置
            from live_trading.config import StopLossConfig

            # 从缓存获取5分钟K线数据
            if stock_code not in self.kline_cache or len(self.kline_cache[stock_code]) == 0:
                self.logger.debug(f"{stock_code}: 暂无5分钟K线数据，跳过风控检查")
                return

            latest_klines = self.kline_cache[stock_code]

            # 筛选买入后的K线（T+2日开始的）
            entry_date_str = position['entry_date'].strftime('%Y%m%d')

            # 筛选今天和之后的K线
            from datetime import datetime
            today_str = datetime.now().strftime('%Y%m%d')

            # 筛选今天的K线
            today_klines = [
                kline for kline in latest_klines
                if kline['time'][:8] == today_str
            ]

            if len(today_klines) == 0:
                self.logger.debug(f"{stock_code}: 今天的5分钟K线数据不足（{len(today_klines)}根），跳过风控检查")
                return

            self.logger.info(f"🛡️ {stock_code}: 开始5分钟风控检查（止损阈值: {StopLossConfig.MAX_STOP_LOSS_PCT:.2%}），今天K线数: {len(today_klines)}")

            # ===== 按时间顺序逐个K线检查（与回测系统一致）=====
            # 不限制K线数量，从第一根K线开始检查

            # 记录上一根K线的日期（用于检测跨天）
            prev_kline_date = None

            for kline in today_klines:
                open_price = kline['open']
                close_price = kline['close']
                kline_time = kline['time']
                kline_date = kline_time[:8]  # 提取YYYYMMDD
                kline_time_str = kline_time[9:]  # 提取HH:MM:00

                # 检查是否跨天（新的一天开始）
                if prev_kline_date is not None and kline_date != prev_kline_date:
                    # 跨天了，重置连续阴线计数
                    if position['consecutive_bearish'] > 0:
                        self.logger.info(f"{stock_code}: 跨天检测，重置连续阴线计数（从{position['consecutive_bearish']}重置为0）")
                        position['consecutive_bearish'] = 0

                # 更新上一根K线的日期
                prev_kline_date = kline_date

                # ===== 风控条件1: 累计收益低于阈值（与回测系统完全一致）=====
                # 计算累计收益（基于买入价，使用收盘价）
                decline_ratio = (close_price - position['entry_price']) / position['entry_price']

                # 更新最大回撤记录
                position['max_decline_from_entry'] = min(
                    position['max_decline_from_entry'],
                    decline_ratio
                )

                # 检查是否触发止损（使用配置的阈值）
                if decline_ratio <= StopLossConfig.MAX_STOP_LOSS_PCT:
                    self.logger.warning(f"🚨 {stock_code}: 在{kline_time_str}触发累计收益止损!")
                    self.logger.warning(f"  买入价: {position['entry_price']:.2f}")
                    self.logger.warning(f"  收盘价: {close_price:.2f}")
                    self.logger.warning(f"  累计收益: {decline_ratio:.2%}")
                    self.logger.warning(f"  止损阈值: {StopLossConfig.MAX_STOP_LOSS_PCT:.2%}")

                    # 标记退出
                    position['should_exit'] = True
                    position['exit_reason'] = f'累计收益低于{StopLossConfig.MAX_STOP_LOSS_PCT:.2%}'

                    # 立即卖出
                    self._execute_sell_5m(stock_code, position, position['exit_reason'])
                    return

                # ===== 风控条件2: 连续3根阴线（需要至少3根K线）=====
                # 判断是否为阴线：收盘价 < 开盘价
                is_bearish = close_price < open_price

                if is_bearish:
                    position['consecutive_bearish'] += 1
                    self.logger.info(f"📉 {stock_code}: 检测到阴线（{kline_date} {kline_time_str}），连续阴线: {position['consecutive_bearish']}根")
                else:
                    if position['consecutive_bearish'] > 0:
                        self.logger.info(f"📈 {stock_code}: 检测到阳线，重置连续阴线计数（从{position['consecutive_bearish']}重置为0）")
                        position['consecutive_bearish'] = 0

                # 风控条件3: 连续阴线风控（同一天内3根）
                if position['consecutive_bearish'] >= 3:
                    self.logger.warning(f"🚨 {stock_code}: 触发连续阴线风控!")
                    self.logger.warning(f"  连续阴线: {position['consecutive_bearish']}根")
                    self.logger.warning(f"  日期: {kline_date}")

                    # 标记退出
                    position['should_exit'] = True
                    position['exit_reason'] = '连续阴线风控'

                    # 立即卖出
                    self._execute_sell_5m(stock_code, position, '连续阴线风控')
                    return

            # 检查是否到尾盘卖出时间（14:55）
            if self.enable_closing_sell and self._is_closing_sell_time():
                self.logger.info(f"⏰ {stock_code}: 14:55尾盘卖出时间到，执行卖出")
                position['should_exit'] = True
                position['exit_reason'] = '尾盘卖出'
                self._execute_sell_5m(stock_code, position, '尾盘卖出')
                return

            # 检查是否超过最大持仓期（3个交易日）- 异常情况
            if self._is_max_holding_reached(position):
                self.logger.warning(f"⚠️ {stock_code}: 持仓异常（超过3个交易日），执行卖出")
                position['should_exit'] = True
                position['exit_reason'] = '持仓异常'
                self._execute_sell_5m(stock_code, position, '持仓异常')

        except Exception as e:
            self.logger.error(f"5分钟风控检查失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _execute_sell_5m(self, stock_code: str, position: Dict, reason: str):
        """
        执行5分钟级别卖出（全仓卖出）

        Args:
            stock_code: 股票代码
            position: 持仓信息
            reason: 卖出原因
        """
        try:
            # 查询持仓
            positions = self.trader.query_stock_positions(self.trader.account)

            if not positions:
                self.logger.warning(f"{stock_code}: 无持仓")
                return

            # 获取持仓数量
            volume = 0
            for pos in positions:
                if pos.stock_code == stock_code:
                    volume = pos.can_use_volume
                    break

            if volume < 100:
                self.logger.warning(f"{stock_code}: 可用数量不足1手")
                return

            # 获取最新价格
            latest_price = self.trader.get_current_price(stock_code)

            if not latest_price:
                self.logger.warning(f"{stock_code}: 无法获取最新价")
                return

            # 计算收益
            entry_price = position['entry_price']
            profit_pct = (latest_price - entry_price) / entry_price * 100

            self.logger.info(f"执行5分钟卖出: {stock_code}")
            self.logger.info(f"  原因: {reason}")
            self.logger.info(f"  买入价: {entry_price:.2f}")
            self.logger.info(f"  卖出价: {latest_price:.2f}")
            self.logger.info(f"  收益率: {profit_pct:+.2f}%")
            self.logger.info(f"  卖出数量: {volume}股（全仓卖出）")

            # 执行卖出（全仓）
            order_id = self.trader.order_manager.smart_sell_full_position(
                stock_code,
                volume
            )

            if order_id:
                # 保存记录
                self.scheduler.save_trade_record(
                    datetime.now().strftime('%Y%m%d'),
                    'sell',
                    [stock_code],
                    {stock_code: order_id}
                )

                # 移除监控
                self.remove_position(stock_code)
            else:
                self.logger.error(f"卖出失败: {stock_code}")

        except Exception as e:
            self.logger.error(f"执行5分钟卖出失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _is_after_t2(self, entry_date: datetime.date) -> bool:
        """
        判断当前是否已到T+2（买入日的下一个交易日）

        Args:
            entry_date: 买入日期

        Returns:
            bool: 是否已到T+2
        """
        try:
            today = datetime.now().date()

            # 找到买入日在交易日历中的索引
            if entry_date not in self.trading_dates:
                # 如果买入日不在交易日历中，使用自然日
                return (today - entry_date).days >= 1

            entry_index = self.trading_dates.index(entry_date)

            # T+2的索引
            t2_index = entry_index + 1

            # 检查今天是否已超过T+2
            today_index = self.trading_dates.index(today) if today in self.trading_dates else -1

            return today_index >= t2_index

        except Exception as e:
            self.logger.error(f"判断T+2失败: {e}")
            return False

    def _is_max_holding_reached(self, position: Dict) -> bool:
        """
        判断是否达到最大持仓期（3个交易日）

        Args:
            position: 持仓信息

        Returns:
            bool: 是否达到最大持仓期
        """
        try:
            entry_date = position['entry_date']
            today = datetime.now().date()

            # 计算持仓的交易日数
            if entry_date in self.trading_dates:
                entry_index = self.trading_dates.index(entry_date)

                if today in self.trading_dates:
                    today_index = self.trading_dates.index(today)
                    trading_days_held = today_index - entry_index

                    # 持仓满3个交易日
                    return trading_days_held >= 3
                else:
                    return False
            else:
                # 使用自然日
                return (today - entry_date).days >= 3

        except Exception as e:
            self.logger.error(f"判断持仓期失败: {e}")
            return False

    def _is_trading_time(self) -> bool:
        """
        判断当前是否为交易时间

        交易时间：
        - 早盘: 09:30 - 11:30
        - 午盘: 13:00 - 14:55（预留5分钟用于尾盘卖出）

        注意：14:55之后返回False，停止监控和风控检查
        """
        now = datetime.now()
        current_time = now.strftime('%H:%M:%S')

        # 早盘: 09:30 - 11:30
        morning_start = "09:30:00"
        morning_end = "11:30:00"

        # 午盘: 13:00 - 14:55（注意：14:55停止监控）
        afternoon_start = "13:00:00"
        afternoon_end = "14:55:00"

        is_trading = (morning_start <= current_time <= morning_end or
                      afternoon_start <= current_time <= afternoon_end)

        # 收盘后自动停止监控（15:05后停止，给最后5分钟留点余量）
        if not is_trading and current_time >= "15:05:00" and self.is_monitoring:
            self.logger.info("=" * 60)
            self.logger.info("⏰ 收盘时间已过，自动停止5分钟监控")
            self.logger.info("=" * 60)
            self.is_monitoring = False

        return is_trading

    def _is_closing_sell_time(self) -> bool:
        """
        判断当前是否为尾盘卖出时间（14:55）

        Returns:
            bool: 是否到了尾盘卖出时间
        """
        now = datetime.now()
        current_time = now.strftime('%H:%M:%S')

        # 判断当前时间 >= 14:55
        return current_time >= self.closing_sell_time

    def _collect_5min_kline_data(self):
        """
        收集5分钟实时K线数据

        在每个5分钟整数倍时间点（如9:35, 9:40）获取当前最新价格，
        构建实时5分钟K线：
        - 第一根K线（9:35）：open=9:30开盘价, close=9:35价格
        - 后续K线：open=上一根的close, close=当前价格
        """
        try:
            from datetime import datetime

            # 获取当前时间
            now = datetime.now()
            time_str = now.strftime('%Y%m%d %H:%M:00')

            # 检查是否已经收集过这个时间点的数据
            for stock_code in list(self.positions.keys()):
                if stock_code not in self.kline_cache:
                    self.kline_cache[stock_code] = []

                # 检查最新一根K线的时间
                if len(self.kline_cache[stock_code]) > 0:
                    last_kline_time = self.kline_cache[stock_code][-1]['time']
                    if last_kline_time == time_str:
                        # 已经收集过这个时间点的数据，跳过
                        self.logger.debug(f"{stock_code}: 5分钟K线数据已存在（{time_str}），跳过收集")
                        continue

                # 获取当前最新价格
                current_price = self.trader.get_current_price(stock_code)

                if current_price is None or current_price <= 0:
                    self.logger.warning(f"{stock_code}: 获取当前价格失败，跳过5分钟K线收集")
                    continue

                # 确定open价格
                if len(self.kline_cache[stock_code]) > 0:
                    # 后续K线：使用上一根K线的close作为当前K线的open
                    open_price = self.kline_cache[stock_code][-1]['close']
                else:
                    # 第一根K线：获取今日开盘价（9:30的价格）
                    try:
                        # 通过 trader.order_manager 获取今日开盘价
                        open_price = self.trader.order_manager._get_today_open_price_from_tick(stock_code)
                        if open_price is None or open_price <= 0:
                            self.logger.warning(f"{stock_code}: 获取今日开盘价失败，使用当前价格")
                            open_price = current_price
                        else:
                            self.logger.info(f"✅ {stock_code}: 第一根K线使用今日开盘价 {open_price:.2f}")
                    except Exception as e:
                        self.logger.error(f"{stock_code}: 获取今日开盘价异常 - {e}，使用当前价格")
                        open_price = current_price

                # 构建K线数据
                kline = {
                    'time': time_str,
                    'open': open_price,
                    'close': current_price
                }

                # 添加到缓存
                self.kline_cache[stock_code].append(kline)

                # 保持缓存大小（最多保存最近100根K线）
                if len(self.kline_cache[stock_code]) > 100:
                    self.kline_cache[stock_code] = self.kline_cache[stock_code][-100:]

                # 判断涨跌
                is_bearish = current_price < open_price
                change_pct = (current_price - open_price) / open_price * 100

                self.logger.info(
                    f"📊 {stock_code} 5分钟K线: {time_str} | "
                    f"开:{open_price:.2f} 收:{current_price:.2f} | "
                    f"跌幅:{change_pct:+.2f}% | "
                    f"{'🔴阴线' if is_bearish else '🟢阳线'}"
                )

        except Exception as e:
            self.logger.error(f"收集5分钟K线数据失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _sleep_through_lunch_break(self):
        """
        智能午间休市处理：一次性sleep整个午间休市时间

        午间休市时间：11:30 - 13:00（90分钟）

        优化效果：
        - 避免每60秒检查一次（原方案需90次循环）
        - 一次性sleep整个休市时间（只需1次sleep）
        - 减少CPU唤醒次数和日志输出
        """
        now = datetime.now()
        current_time = now.strftime('%H:%M:%S')

        # 午间休市：11:30 - 13:00
        lunch_start = "11:30:00"
        lunch_end = "13:00:00"

        # 判断是否在午间休市时间
        if lunch_start <= current_time < lunch_end:
            # 计算距离13:00还有多少秒
            lunch_end_datetime = now.replace(hour=13, minute=0, second=0, microsecond=0)
            wait_seconds = (lunch_end_datetime - now).total_seconds()

            # 额外加30秒缓冲，确保13:00已经开市
            wait_seconds += 30

            if wait_seconds > 0:
                self.logger.info("=" * 60)
                self.logger.info(f"🍽️  午间休市（11:30-13:00）")
                self.logger.info(f"💤 系统将休眠 {int(wait_seconds // 60)}分钟 {int(wait_seconds % 60)}秒")
                self.logger.info(f"⏰ 预计唤醒时间: 13:00:30")
                self.logger.info("=" * 60)

                # 一次性sleep整个午间休市时间
                time.sleep(wait_seconds)

                self.logger.info("=" * 60)
                self.logger.info("☕️ 午间休市结束，恢复监控")
                self.logger.info("=" * 60)
        elif current_time < lunch_start:
            # 在早盘，但接近午间休市（11:25-11:30）
            if current_time >= "11:25:00":
                # 计算距离11:30还有多少秒
                lunch_start_datetime = now.replace(hour=11, minute=30, second=0, microsecond=0)
                wait_seconds = (lunch_start_datetime - now).total_seconds()

                if wait_seconds > 0 and wait_seconds < 300:  # 5分钟内
                    self.logger.info(f"⏰ 距离午间休市还有 {int(wait_seconds)}秒，提前休眠")
                    time.sleep(wait_seconds)

    def _sleep_to_next_5min(self):
        """休眠到下一个5分钟整数倍"""
        now = datetime.now()

        # 计算下一个5分钟整数倍
        minute = now.minute
        next_minute = ((minute // 5) + 1) * 5

        if next_minute >= 60:
            # 下个小时
            next_time = now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
        else:
            next_time = now.replace(minute=next_minute, second=0, microsecond=0)

        wait_seconds = (next_time - now).total_seconds()

        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _print_positions_summary(self):
        """
        打印持仓摘要（在监控循环中每5分钟调用一次）

        不返回数据，直接打印到日志
        """
        if not self.positions:
            return

        try:
            self.logger.info("=" * 60)
            self.logger.info("📊 持仓摘要")
            self.logger.info("=" * 60)

            for stock_code, position in self.positions.items():
                # 获取当前价格
                current_price = self.trader.get_current_price(stock_code)
                if current_price is None:
                    current_price = position.get('current_price', position['entry_price'])

                # 计算收益率
                profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100

                # 计算持仓天数
                from datetime import datetime
                entry_date = position['entry_date']
                today = datetime.now().date()
                days_held = (today - entry_date).days

                # 打印摘要
                self.logger.info(f"股票: {stock_code}")
                self.logger.info(f"  买入价: {position['entry_price']:.2f}")
                self.logger.info(f"  当前价: {current_price:.2f}")
                self.logger.info(f"  收益率: {profit_pct:+.2f}%")
                self.logger.info(f"  持仓天数: {days_held}天")
                self.logger.info(f"  最大回撤: {position['max_decline_from_entry']:.2%}")
                self.logger.info(f"  连续阴线: {position['consecutive_bearish']}根")
                self.logger.info("-" * 60)

        except Exception as e:
            self.logger.error(f"打印持仓摘要失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def get_positions_summary(self) -> List[Dict]:
        """
        获取持仓摘要

        Returns:
            List[Dict]: 持仓摘要列表
        """
        summary = []

        for stock_code, position in self.positions.items():
            # 获取最新价
            current_price = self.trader.get_current_price(stock_code)

            if current_price:
                profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100

                # 计算持仓天数
                days_held = (datetime.now().date() - position['entry_date']).days

                summary.append({
                    'stock_code': stock_code,
                    'entry_price': position['entry_price'],
                    'current_price': current_price,
                    'profit_pct': profit_pct,
                    'days_held': days_held,
                    'max_decline': position['max_decline_from_entry'] * 100,  # 转为百分比
                    'consecutive_bearish': position['consecutive_bearish'],
                })

        return summary
