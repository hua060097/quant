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

from .config import PathConfig, StopLossConfig


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

        # 状态机诊断轨迹 JSONL 输出（按日切分）
        self._state_trace_fh = None
        self._state_trace_day = None
        self._state_trace_path = None

        # 5分钟K线采样边界容忍秒数
        self.kline_boundary_tolerance_seconds = 10
        # 收盘边界读取的行情必须足够新；否则宁可漏一根，也不使用陈旧快照伪造K线
        self.kline_max_quote_age_seconds = 30.0

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

    def add_position_5m(self, stock_code: str, entry_price: float, entry_date: str,
                        slot_index: int = None, target_pct: float = None,
                        enable_closing_sell: bool = False,
                        open_price: float = None):
        """
        添加持仓监控（5分钟K线版本）

        Args:
            stock_code: 股票代码
            entry_price: 买入价格（实际成交均价，用于计算实际盈亏）
            entry_date: 买入日期 (YYYY-MM-DD)
            open_price: T+1日开盘价（用于止盈止损的基准价，与回测一致）
        """
        # 止盈止损基准价：优先使用开盘价（与回测一致），fallback到成交均价
        if hasattr(self.trader, "ensure_market_quote_subscription"):
            self.trader.ensure_market_quote_subscription([stock_code])

        risk_ref_price = open_price if (open_price and open_price > 0) else entry_price
        entry_date_obj = datetime.strptime(entry_date, '%Y-%m-%d').date()
        historical_high = self._load_holding_period_high(stock_code, entry_date_obj, datetime.now().date())
        seed_high = max(risk_ref_price, historical_high or risk_ref_price)
        seed_high_return = (seed_high - risk_ref_price) / risk_ref_price if risk_ref_price > 0 else 0.0

        self.positions[stock_code] = {
            'stock_code': stock_code,
            'entry_price': entry_price,
            'open_price': open_price if open_price else 0.0,
            'risk_ref_price': risk_ref_price,
            'entry_date': entry_date_obj,
            'entry_datetime': datetime.now(),
            'klines_after_entry': [],  # 买入后的5分钟K线
            'consecutive_bearish': 0,  # 兼容展示字段
            'consecutive_bearish_in_profit': 0,  # 盈利状态下连续阴线数
            'consecutive_weak_in_profit': 0,  # 盈利状态下连续弱K线数（阴线/十字）
            'max_decline_from_entry': 0.0,  # 最大回撤（相对于买入价）
            'should_exit': False,
            'exit_reason': None,
            'exit_in_progress': False,
            'exit_order_id': None,
            'exit_submitted_at': None,
            'slot_index': slot_index,
            'target_pct': target_pct,
            'enable_closing_sell': enable_closing_sell,
            'highest_price_since_buy': seed_high,
            'highest_return_since_buy': seed_high_return,
            't1_day_lowest_price': self._load_entry_day_low(stock_code, entry_date) or 0.0,
            'risk_state': 'IDLE',
            'failed_cont_watch_active': False,
            'failed_cont_watch_bars_seen': 0,
            'failed_cont_watch_bearish_bars': 0,
            'failed_cont_watch_break_level': 0.0,
            'failed_cont_watch_day_offset': -1,
            'last_checked_kline_time': None,
        }

        self.logger.info(f"添加5分钟监控: {stock_code}")
        self.logger.info(f"  买入价(实际均价): {entry_price:.2f}")
        if open_price and open_price > 0:
            self.logger.info(f"  开盘价(风控基准): {open_price:.2f}")
        self.logger.info(f"  历史持仓高点初始化: high={seed_high:.2f}, peak_return={seed_high_return:.2%}")
        self.logger.info(f"  买入日期: {entry_date}")

    def remove_position(self, stock_code: str):
        """移除持仓监控"""
        if stock_code in self.positions:
            del self.positions[stock_code]
            self.logger.info(f"移除监控: {stock_code}")

    def _load_entry_day_low(self, stock_code: str, entry_date: str) -> Optional[float]:
        """加载买入日(T+1)日线最低价，供 FailedContinuation 使用。"""
        try:
            from xtquant import xtdata

            entry_compact = datetime.strptime(entry_date, '%Y-%m-%d').strftime('%Y%m%d')
            daily_data = xtdata.get_market_data_ex(
                stock_list=[stock_code],
                period='1d',
                start_time=entry_compact,
                end_time=entry_compact,
                count=1,
                dividend_type="front",
            )
            if stock_code in daily_data and len(daily_data[stock_code]) > 0:
                day_low = float(daily_data[stock_code].iloc[0]['low'])
                self.logger.info(f"{stock_code}: T+1日低点={day_low:.2f}")
                return day_low
        except Exception as e:
            self.logger.warning(f"{stock_code}: 读取T+1日低点失败: {e}")
        return None

    def _load_holding_period_high(
        self,
        stock_code: str,
        entry_date: datetime.date,
        current_date: datetime.date,
    ) -> Optional[float]:
        """加载从T+1买入日至当前监控日前一交易日的最高价。

        T+2启动时补T+1 high；T+3启动时补T+1/T+2 high。
        当天盘中的高点由实时5分钟K线继续更新。
        """
        try:
            from xtquant import xtdata

            if current_date <= entry_date:
                return None

            if entry_date in self.trading_dates and current_date in self.trading_dates:
                entry_idx = self.trading_dates.index(entry_date)
                current_idx = self.trading_dates.index(current_date)
                end_idx = current_idx - 1
                if end_idx < entry_idx:
                    return None
                start_date = self.trading_dates[entry_idx]
                end_date = self.trading_dates[end_idx]
            else:
                # 兜底：至少排除当前日，只补已经收盘的历史日线。
                start_date = entry_date
                end_date = current_date - timedelta(days=1)
                if end_date < start_date:
                    return None

            start_compact = start_date.strftime('%Y%m%d')
            end_compact = end_date.strftime('%Y%m%d')
            expected_count = 1
            if start_date in self.trading_dates and end_date in self.trading_dates:
                expected_count = self.trading_dates.index(end_date) - self.trading_dates.index(start_date) + 1
            daily_data = xtdata.get_market_data_ex(
                stock_list=[stock_code],
                period='1d',
                start_time=start_compact,
                end_time=end_compact,
                count=max(expected_count, 1),
                dividend_type="front",
            )
            if stock_code not in daily_data or len(daily_data[stock_code]) == 0:
                self.logger.warning(f"{stock_code}: 未获取到持仓历史日线高点 {start_compact}~{end_compact}")
                return None

            high_series = pd.to_numeric(daily_data[stock_code].get('high'), errors='coerce').dropna()
            if high_series.empty:
                self.logger.warning(f"{stock_code}: 持仓历史日线high为空 {start_compact}~{end_compact}")
                return None

            historical_high = float(high_series.max())
            self.logger.info(
                f"{stock_code}: 持仓历史高点 {start_date}~{end_date} high={historical_high:.2f}"
            )
            return historical_high

        except Exception as e:
            self.logger.warning(f"{stock_code}: 读取持仓历史高点失败: {e}")
            return None

    @staticmethod
    def _extract_time_hm(kline_time: str) -> str:
        try:
            dt = datetime.strptime(kline_time, '%Y%m%d %H:%M:00')
            return dt.strftime('%H:%M')
        except Exception:
            return ''

    def _get_holding_day_offset(self, entry_date: datetime.date, current_date: datetime.date) -> Optional[int]:
        """返回与回测一致的 holding_day_offset：买入日=0, T+2=1, T+3=2。"""
        try:
            if entry_date in self.trading_dates and current_date in self.trading_dates:
                return self.trading_dates.index(current_date) - self.trading_dates.index(entry_date)
            return (current_date - entry_date).days
        except Exception:
            return None

    @staticmethod
    def _reset_failed_cont_watch(position: Dict):
        position['failed_cont_watch_active'] = False
        position['failed_cont_watch_bars_seen'] = 0
        position['failed_cont_watch_bearish_bars'] = 0
        position['failed_cont_watch_break_level'] = 0.0
        position['failed_cont_watch_day_offset'] = -1

    @staticmethod
    def _is_expiry_bar(holding_day_offset: Optional[int], time_hm: str) -> bool:
        if holding_day_offset is None:
            return False
        return (
            holding_day_offset >= StopLossConfig.MAX_HOLDING_DAYS - 1
            and time_hm == '14:55'
        )

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
        # 关闭状态机轨迹文件句柄
        if self._state_trace_fh is not None:
            try:
                self._state_trace_fh.flush()
                self._state_trace_fh.close()
            except Exception:
                pass
            self._state_trace_fh = None
            self._state_trace_day = None
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

                # 只在严格的5分钟边界采样，避免启动时落在09:44/09:49这类错位分钟
                if not self._wait_until_5min_boundary():
                    continue

                if not self._is_trading_time():
                    continue

                self._sync_pending_exit_positions()

                # ===== 收集5分钟K线数据（在5分钟整数倍时间点）=====
                self._collect_5min_kline_data()

                # ===== 打印持仓摘要（每5分钟打印一次）=====
                self._print_positions_summary()

                # 检查所有持仓
                for stock_code in list(self.positions.keys()):
                    position = self.positions[stock_code]

                    # 检查是否需要退出
                    if position.get('should_exit') or position.get('exit_in_progress'):
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

    def _write_state_trace(self, record: Dict):
        """
        将单根5分钟K线的状态机诊断变量以一行 JSON 形式写入当日轨迹文件。

        输出路径: live_trading/logs/state_trace_YYYYMMDD.jsonl
        每行一个独立的 JSON 对象，便于事后用 pandas.read_json(lines=True) 解析。
        """
        try:
            today_str = datetime.now().strftime('%Y%m%d')
            if self._state_trace_day != today_str:
                # 跨天时关闭旧句柄并切换到新文件
                if self._state_trace_fh is not None:
                    try:
                        self._state_trace_fh.close()
                    except Exception:
                        pass
                trace_path = PathConfig.LOG_DIR / f"state_trace_{today_str}.jsonl"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                self._state_trace_fh = open(trace_path, 'a', encoding='utf-8')
                self._state_trace_day = today_str
                self._state_trace_path = trace_path
                self.logger.info(f"📝 状态机轨迹文件: {trace_path}")
            self._state_trace_fh.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            self._state_trace_fh.flush()
        except Exception as e:
            # 状态轨迹写入失败不能影响交易主流程
            self.logger.warning(f"写入状态机轨迹失败: {e}")

    def _check_position_risk_5m(self, stock_code: str, position: Dict):
        """与当前分钟回测一致的状态机风控。"""
        try:
            if stock_code not in self.kline_cache or len(self.kline_cache[stock_code]) == 0:
                self.logger.debug(f"{stock_code}: 暂无5分钟K线数据，跳过风控检查")
                return

            today = datetime.now().date()
            today_str = today.strftime('%Y%m%d')
            today_klines = [k for k in self.kline_cache[stock_code] if k['time'][:8] == today_str]
            last_checked_time = position.get('last_checked_kline_time')
            if last_checked_time and last_checked_time[:8] != today_str:
                position['consecutive_bearish_in_profit'] = 0
                position['consecutive_weak_in_profit'] = 0
                position['consecutive_bearish'] = 0
            if last_checked_time:
                today_klines = [k for k in today_klines if k['time'] > last_checked_time]
            if not today_klines:
                return

            self.logger.info(
                f"🛡️ {stock_code}: 开始5分钟状态机风控检查"
                f"（硬止损: {StopLossConfig.MAX_STOP_LOSS_PCT:.2%}），今天K线数: {len(today_klines)}"
            )

            holding_day_offset = self._get_holding_day_offset(position['entry_date'], today)
            prev_kline_date = None

            for kline in today_klines:
                open_price = float(kline['open'])
                close_price = float(kline['close'])
                high_price = float(kline.get('high', max(open_price, close_price)))
                kline_time = kline['time']
                kline_date = kline_time[:8]
                time_hm = self._extract_time_hm(kline_time)

                if prev_kline_date is not None and kline_date != prev_kline_date:
                    position['consecutive_bearish_in_profit'] = 0
                    position['consecutive_weak_in_profit'] = 0
                prev_kline_date = kline_date

                risk_ref = position.get('risk_ref_price', position['entry_price'])
                current_return = (close_price - risk_ref) / risk_ref
                position['max_decline_from_entry'] = min(position['max_decline_from_entry'], current_return)
                position['highest_price_since_buy'] = max(position.get('highest_price_since_buy', risk_ref), high_price)
                position['highest_return_since_buy'] = max(
                    position.get('highest_return_since_buy', 0.0),
                    (position['highest_price_since_buy'] - risk_ref) / risk_ref
                )

                body_ratio = (close_price - open_price) / open_price if open_price else 0.0
                is_bearish = close_price < open_price
                is_neutral = abs(body_ratio) <= 0.0005

                # 回测里这一路变量名仍叫 consecutive_bearish_in_profit，
                # 实际语义是连续走坏：阴线递增，十字/平盘保持，阳线重置。
                if is_bearish:
                    position['consecutive_bearish_in_profit'] += 1
                    position['consecutive_weak_in_profit'] = position['consecutive_bearish_in_profit']
                    position['consecutive_bearish'] = position['consecutive_bearish_in_profit']
                elif is_neutral:
                    position['consecutive_weak_in_profit'] = position['consecutive_bearish_in_profit']
                else:
                    position['consecutive_bearish_in_profit'] = 0
                    position['consecutive_weak_in_profit'] = 0
                    position['consecutive_bearish'] = 0

                # 计算回撤（同时供下方 ProfitProtect 与诊断日志共用）
                highest_price = float(position.get('highest_price_since_buy') or 0.0)
                drawdown_from_highest = None
                if highest_price > 0:
                    drawdown_from_highest = (close_price - highest_price) / highest_price

                # 诊断日志：状态机每根K线的中间变量，便于回放判断路径
                peak_armed = position.get('highest_return_since_buy', 0.0) >= StopLossConfig.SMALL_PROFIT_ACTIVATION_RETURN
                drawdown_armed = (
                    drawdown_from_highest is not None
                    and drawdown_from_highest <= StopLossConfig.PROFIT_BEARISH_TRAILING_DRAWDOWN
                )
                current_profit_ok = current_return >= StopLossConfig.PROFIT_BEARISH_MIN_CURRENT_RETURN
                t1_low_dbg = position.get('t1_day_lowest_price')
                t1_low_str = f"{float(t1_low_dbg):.2f}" if t1_low_dbg else "N/A"
                drawdown_str = f"{drawdown_from_highest:.2%}" if drawdown_from_highest is not None else "N/A"
                self.logger.info(
                    f"[STATE] {stock_code} {kline_time} HD{holding_day_offset} {time_hm} "
                    f"O={open_price:.2f} C={close_price:.2f} H={high_price:.2f} "
                    f"ref={risk_ref:.2f} ret={current_return:.2%} "
                    f"hi={highest_price:.2f} hi_ret={position.get('highest_return_since_buy', 0.0):.2%} "
                    f"dd={drawdown_str} "
                    f"bearish={position['consecutive_bearish_in_profit']} "
                    f"peak_armed={peak_armed} dd_armed={drawdown_armed} cur_ok={current_profit_ok} "
                    f"t1_low={t1_low_str} "
                    f"fc_active={position.get('failed_cont_watch_active', False)} "
                    f"fc_break={position.get('failed_cont_watch_break_level', 0.0):.2f} "
                    f"fc_bear={position.get('failed_cont_watch_bearish_bars', 0)}"
                )

                # 同步写入结构化 JSONL 轨迹（便于事后用 pandas 解析复盘）
                self._write_state_trace({
                    'wall_ts': datetime.now().isoformat(timespec='seconds'),
                    'kline_time': kline_time,
                    'stock_code': stock_code,
                    'entry_date': str(position.get('entry_date')),
                    'entry_price': position.get('entry_price'),
                    'risk_ref_price': risk_ref,
                    'holding_day_offset': holding_day_offset,
                    'time_hm': time_hm,
                    'open': open_price,
                    'high': high_price,
                    'close': close_price,
                    'body_ratio': body_ratio,
                    'is_bearish': is_bearish,
                    'is_neutral': is_neutral,
                    'current_return': current_return,
                    'highest_price_since_buy': highest_price,
                    'highest_return_since_buy': position.get('highest_return_since_buy', 0.0),
                    'drawdown_from_highest': drawdown_from_highest,
                    'consecutive_bearish_in_profit': position['consecutive_bearish_in_profit'],
                    'consecutive_weak_in_profit': position.get('consecutive_weak_in_profit', 0),
                    'peak_armed': peak_armed,
                    'drawdown_armed': drawdown_armed,
                    'current_profit_ok': current_profit_ok,
                    't1_day_lowest_price': float(t1_low_dbg) if t1_low_dbg else None,
                    'failed_cont_watch_active': position.get('failed_cont_watch_active', False),
                    'failed_cont_watch_break_level': position.get('failed_cont_watch_break_level', 0.0),
                    'failed_cont_watch_bearish_bars': position.get('failed_cont_watch_bearish_bars', 0),
                    'failed_cont_watch_bars_seen': position.get('failed_cont_watch_bars_seen', 0),
                    'risk_state': position.get('risk_state'),
                    'max_decline_from_entry': position.get('max_decline_from_entry'),
                    'thresholds': {
                        'small_profit_activation': StopLossConfig.SMALL_PROFIT_ACTIVATION_RETURN,
                        'small_profit_bearish_required': StopLossConfig.SMALL_PROFIT_BEARISH_REQUIRED_COUNT,
                        'profit_bearish_min_current': StopLossConfig.PROFIT_BEARISH_MIN_CURRENT_RETURN,
                        'profit_bearish_trailing_dd': StopLossConfig.PROFIT_BEARISH_TRAILING_DRAWDOWN,
                        'profit_bearish_required_count': StopLossConfig.PROFIT_BEARISH_REQUIRED_COUNT,
                        'profit_bearish_activation': StopLossConfig.PROFIT_BEARISH_ACTIVATION_RETURN,
                        'profit_bearish_min_peak': StopLossConfig.PROFIT_BEARISH_MIN_PEAK_RETURN,
                        't2_0935_hard_stop': StopLossConfig.T2_0935_HARD_STOP_THRESHOLD,
                        't2_1000_soft_stop': StopLossConfig.T2_1000_SOFT_STOP_THRESHOLD,
                        't2_close_breakeven': StopLossConfig.T2_CLOSE_BREAKEVEN_STOP_THRESHOLD,
                        't3_0935_breakeven': StopLossConfig.T3_0935_BREAKEVEN_STOP_THRESHOLD,
                        'max_stop_loss': StopLossConfig.MAX_STOP_LOSS_PCT,
                    },
                })

                # FailFast
                if (
                    StopLossConfig.ENABLE_TOP1_STATE_MACHINE_RISK
                    and holding_day_offset == 1
                    and time_hm == '09:35'
                    and current_return <= StopLossConfig.T2_0935_HARD_STOP_THRESHOLD
                ):
                    reason = f"T+2 09:35失败止损({StopLossConfig.T2_0935_HARD_STOP_THRESHOLD:.2%})"
                    position['should_exit'] = True
                    position['exit_reason'] = reason
                    self._execute_sell_5m(stock_code, position, reason)
                    return

                if (
                    StopLossConfig.ENABLE_TOP1_STATE_MACHINE_RISK
                    and StopLossConfig.ENABLE_T2_1000_SOFT_STOP
                    and holding_day_offset == 1
                    and time_hm == '10:00'
                    and current_return <= StopLossConfig.T2_1000_SOFT_STOP_THRESHOLD
                ):
                    reason = f"T+2 10:00继续走坏止损({StopLossConfig.T2_1000_SOFT_STOP_THRESHOLD:.2%})"
                    position['should_exit'] = True
                    position['exit_reason'] = reason
                    self._execute_sell_5m(stock_code, position, reason)
                    return

                # WeakDrift
                if StopLossConfig.ENABLE_SECOND_LAYER_TIME_STOP:
                    if (
                        holding_day_offset == 1
                        and time_hm == '14:55'
                        and current_return <= StopLossConfig.T2_CLOSE_BREAKEVEN_STOP_THRESHOLD
                    ):
                        reason = f"T+2收盘仍未转正({StopLossConfig.T2_CLOSE_BREAKEVEN_STOP_THRESHOLD:.2%})"
                        position['should_exit'] = True
                        position['exit_reason'] = reason
                        self._execute_sell_5m(stock_code, position, reason)
                        return
                    if (
                        holding_day_offset == 2
                        and time_hm == '09:35'
                        and current_return <= StopLossConfig.T3_0935_BREAKEVEN_STOP_THRESHOLD
                    ):
                        reason = f"T+3 09:35仍未转正({StopLossConfig.T3_0935_BREAKEVEN_STOP_THRESHOLD:.2%})"
                        position['should_exit'] = True
                        position['exit_reason'] = reason
                        self._execute_sell_5m(stock_code, position, reason)
                        return

                # FailedContinuation: 跌破T+1低点后，从下一根开始要求连续2根阴线；收回即失效
                if StopLossConfig.ENABLE_SMALL_PROFIT_FAILURE_EXIT and holding_day_offset == 1:
                    break_level = float(position.get('t1_day_lowest_price') or 0.0)
                    peak_armed = position.get('highest_return_since_buy', 0.0) >= StopLossConfig.SMALL_PROFIT_ACTIVATION_RETURN
                    if peak_armed and break_level > 0:
                        if not position['failed_cont_watch_active']:
                            if close_price < break_level:
                                position['failed_cont_watch_active'] = True
                                position['failed_cont_watch_bars_seen'] = 0
                                position['failed_cont_watch_bearish_bars'] = 0
                                position['failed_cont_watch_break_level'] = break_level
                                position['failed_cont_watch_day_offset'] = holding_day_offset
                                position['risk_state'] = 'FAILED_CONTINUATION_WATCH'
                        else:
                            if close_price >= position['failed_cont_watch_break_level']:
                                self._reset_failed_cont_watch(position)
                            else:
                                position['failed_cont_watch_bars_seen'] += 1
                                if is_bearish:
                                    position['failed_cont_watch_bearish_bars'] += 1
                                else:
                                    position['failed_cont_watch_bearish_bars'] = 0
                                if position['failed_cont_watch_bearish_bars'] >= StopLossConfig.SMALL_PROFIT_BEARISH_REQUIRED_COUNT:
                                    reason = (
                                        f"小盈利后延续失败: 跌破T+1低点后连续"
                                        f"{position['failed_cont_watch_bearish_bars']}根阴线"
                                        f" | 峰值{position['highest_return_since_buy']:.2%}"
                                        f" | 当前{current_return:.2%}"
                                        f" | T+1低点{position['failed_cont_watch_break_level']:.2f}"
                                    )
                                    position['should_exit'] = True
                                    position['exit_reason'] = reason
                                    self._execute_sell_5m(stock_code, position, reason)
                                    return
                    else:
                        self._reset_failed_cont_watch(position)
                else:
                    self._reset_failed_cont_watch(position)

                # ProfitProtect（drawdown_from_highest 已在前面计算）
                if (
                    StopLossConfig.ENABLE_PROFIT_THRESHOLD_FOR_BEARISH_EXIT
                    and drawdown_from_highest is not None
                    and drawdown_from_highest <= StopLossConfig.PROFIT_BEARISH_TRAILING_DRAWDOWN
                    and current_return >= StopLossConfig.PROFIT_BEARISH_MIN_CURRENT_RETURN
                    and position['consecutive_bearish_in_profit'] >= StopLossConfig.PROFIT_BEARISH_REQUIRED_COUNT
                ):
                    reason = (
                        f"利润保护: 连续{position['consecutive_bearish_in_profit']}根弱K线"
                        f" | 峰值{position['highest_return_since_buy']:.2%}"
                        f" | 回撤{drawdown_from_highest:.2%}"
                    )
                    position['should_exit'] = True
                    position['exit_reason'] = reason
                    self._execute_sell_5m(stock_code, position, reason)
                    return

                # 全局硬止损兜底
                if current_return <= StopLossConfig.MAX_STOP_LOSS_PCT:
                    reason = f"累计收益低于{StopLossConfig.MAX_STOP_LOSS_PCT:.2%}"
                    position['should_exit'] = True
                    position['exit_reason'] = reason
                    self._execute_sell_5m(stock_code, position, reason)
                    return

                # 到期卖出：回测 holding_days=3 时，到期日为买入日后第2个交易日；
                # 回测用当日最后一根分钟K收盘价卖出，实盘在14:55主动卖出以避开尾盘集合竞价。
                if self._is_expiry_bar(holding_day_offset, time_hm):
                    position['should_exit'] = True
                    position['exit_reason'] = '到期'
                    self._execute_sell_5m(stock_code, position, '到期')
                    return

                position['last_checked_kline_time'] = kline_time

            if self._is_max_holding_reached(position):
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
            if position.get('exit_in_progress'):
                self.logger.info(
                    f"{stock_code}: 卖出已在执行中，跳过重复卖出 "
                    f"(order_id={position.get('exit_order_id')}, reason={position.get('exit_reason')})"
                )
                return

            position['should_exit'] = True
            position['exit_reason'] = reason

            # 查询持仓
            positions = self.trader.query_stock_positions(self.trader.account)

            if not positions:
                self.logger.warning(f"{stock_code}: 无持仓")
                self.remove_position(stock_code)
                return

            # 获取持仓数量
            volume = 0
            for pos in positions:
                if pos.stock_code == stock_code:
                    volume = pos.can_use_volume
                    break

            if volume < 100:
                self.logger.warning(f"{stock_code}: 可用数量不足1手")
                self._mark_exit_in_progress(stock_code, position, None, reason)
                return

            # 获取最新价格
            latest_price = self.trader.get_current_price(stock_code)

            if not latest_price:
                self.logger.warning(f"{stock_code}: 无法获取最新价")
                position['should_exit'] = False
                position['exit_reason'] = None
                return

            # 计算收益
            entry_price = position['entry_price']
            risk_ref = position.get('risk_ref_price', entry_price)
            profit_pct = (latest_price - entry_price) / entry_price * 100
            risk_return_pct = (latest_price - risk_ref) / risk_ref * 100

            self.logger.info(f"执行5分钟卖出: {stock_code}")
            self.logger.info(f"  原因: {reason}")
            self.logger.info(f"  买入价(均价): {entry_price:.2f}")
            if risk_ref != entry_price:
                self.logger.info(f"  开盘价(风控基准): {risk_ref:.2f}")
            self.logger.info(f"  卖出价: {latest_price:.2f}")
            self.logger.info(f"  实际收益率: {profit_pct:+.2f}%")
            if risk_ref != entry_price:
                self.logger.info(f"  风控收益率(基准=开盘): {risk_return_pct:+.2f}%")
            self.logger.info(f"  卖出数量: {volume}股（全仓卖出）")

            # 执行卖出（全仓）
            order_id = self.trader.order_manager.smart_sell_full_position(
                stock_code,
                volume
            )

            if order_id:
                self._mark_exit_in_progress(stock_code, position, order_id, reason)
                # 保存记录
                self.scheduler.save_trade_record(
                    datetime.now().strftime('%Y%m%d'),
                    'sell',
                    [stock_code],
                    {stock_code: order_id},
                    exit_reason=reason,
                    profit_pct=profit_pct
                )

                self._sync_pending_exit_positions()
            else:
                submission = getattr(self.trader.order_manager, 'last_sell_submission', None)
                if submission and submission.get('stock_code') == stock_code:
                    submitted_order_id = submission.get('order_id')
                    self._mark_exit_in_progress(stock_code, position, submitted_order_id, reason)
                    self.logger.warning(
                        f"{stock_code}: 卖出函数超时/失败返回，但检测到卖单已提交 "
                        f"(order_id={submitted_order_id})，冻结监控等待成交/持仓同步"
                    )
                    self._sync_pending_exit_positions()
                else:
                    position['should_exit'] = False
                    position['exit_reason'] = None
                    self.logger.error(f"卖出失败且未检测到已提交卖单: {stock_code}")

        except Exception as e:
            self.logger.error(f"执行5分钟卖出失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _mark_exit_in_progress(self, stock_code: str, position: Dict, order_id: Optional[str], reason: str):
        """标记退出中，避免成交回报迟到时继续风控或重复卖出。"""
        position['should_exit'] = True
        position['exit_reason'] = reason
        position['exit_in_progress'] = True
        position['exit_order_id'] = str(order_id) if order_id else position.get('exit_order_id')
        position['exit_submitted_at'] = datetime.now()
        self.logger.info(
            f"{stock_code}: 已进入卖出等待确认状态 "
            f"(order_id={position.get('exit_order_id')}, reason={reason})"
        )

    def _sync_pending_exit_positions(self):
        """对退出中持仓做账户同步；账户已无持仓时立即移除监控。"""
        pending_codes = [
            code for code, position in self.positions.items()
            if position.get('exit_in_progress') or position.get('should_exit')
        ]
        if not pending_codes:
            return

        try:
            account_positions = self.trader.query_stock_positions(self.trader.account) or []
            held_volume = {}
            for pos in account_positions:
                code = getattr(pos, 'stock_code', None)
                if not code:
                    continue
                total_volume = int(getattr(pos, 'volume', 0) or 0)
                can_use_volume = int(getattr(pos, 'can_use_volume', 0) or 0)
                held_volume[code] = max(total_volume, can_use_volume)

            for stock_code in pending_codes:
                if stock_code not in self.positions:
                    continue
                if held_volume.get(stock_code, 0) < 100:
                    self.logger.info(f"{stock_code}: 账户已无可监控持仓，确认移除5分钟监控")
                    self.remove_position(stock_code)

        except Exception as e:
            self.logger.warning(f"同步退出中持仓失败: {e}")

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
        收集由全推逐笔行情聚合出的真实5分钟OHLC。

        不再使用“上一根收盘 + 当前快照”拼接伪K线。完整bar尚未形成或
        行情已经陈旧时直接跳过，等待下一轮，避免错误阴阳线触发风控。
        """
        try:
            from datetime import datetime

            # 获取当前时间
            now = datetime.now()

            # 只允许在5分钟边界附近采样，避免生成09:44、09:49这类伪5分钟K线
            if now.minute % 5 != 0 or now.second > self.kline_boundary_tolerance_seconds:
                self.logger.debug(
                    f"当前时间 {now.strftime('%H:%M:%S')} 不在5分钟边界容忍范围内，跳过K线采集"
                )
                return

            boundary_time = now.replace(second=0, microsecond=0)
            boundary_hm = boundary_time.strftime('%H:%M')
            if boundary_hm in {'09:30', '13:00'}:
                session = "开盘" if boundary_hm == '09:30' else "午后开盘"
                self.logger.info(
                    f"{boundary_hm}{session}点不生成风控K线，"
                    f"等待{'09:35' if boundary_hm == '09:30' else '13:05'}首根5分钟K收盘"
                )
                return

            time_str = boundary_time.strftime('%Y%m%d %H:%M:00')

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

                quote_cache = getattr(self.trader, "quote_cache", None)
                if quote_cache is None:
                    self.logger.warning(f"{stock_code}: 全推行情缓存不可用，跳过真实5分钟K线收集")
                    continue

                fresh_tick = quote_cache.get_tick(
                    stock_code,
                    max_age_seconds=self.kline_max_quote_age_seconds,
                )
                if not fresh_tick:
                    self.logger.warning(
                        f"{stock_code}: 行情超过{self.kline_max_quote_age_seconds:.0f}秒未更新，"
                        f"拒绝采集 {time_str} K线"
                    )
                    continue

                cached_bar = quote_cache.get_realtime_5m_bar(stock_code, time_str)
                if not cached_bar:
                    self.logger.warning(
                        f"{stock_code}: {time_str} 真实5分钟OHLC尚未形成，跳过本轮"
                    )
                    continue

                try:
                    open_price = float(cached_bar['open'])
                    high_price = float(cached_bar['high'])
                    low_price = float(cached_bar['low'])
                    current_price = float(cached_bar['close'])
                except (KeyError, TypeError, ValueError) as e:
                    self.logger.warning(f"{stock_code}: {time_str} OHLC字段无效({e})，跳过")
                    continue

                prices = (open_price, high_price, low_price, current_price)
                if any(price <= 0 for price in prices):
                    self.logger.warning(f"{stock_code}: {time_str} OHLC含非正价格，跳过")
                    continue
                if high_price < max(open_price, current_price) or low_price > min(open_price, current_price):
                    self.logger.warning(
                        f"{stock_code}: {time_str} OHLC关系异常 "
                        f"O={open_price}, H={high_price}, L={low_price}, C={current_price}，跳过"
                    )
                    continue

                kline = {
                    'time': time_str,
                    'open': open_price,
                    'close': current_price,
                    'high': high_price,
                    'low': low_price,
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
                    f"开:{open_price:.2f} 高:{high_price:.2f} "
                    f"低:{low_price:.2f} 收:{current_price:.2f} | "
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

    def _wait_until_5min_boundary(self) -> bool:
        """
        等待到下一个5分钟边界。

        返回:
            bool: 是否仍应继续执行本轮监控
        """
        now = datetime.now()

        # 已经在5分钟边界附近，直接采样
        if now.minute % 5 == 0 and now.second <= self.kline_boundary_tolerance_seconds:
            return True

        minute = now.minute
        next_minute = ((minute // 5) + 1) * 5

        if next_minute >= 60:
            next_time = now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
        else:
            next_time = now.replace(minute=next_minute, second=0, microsecond=0)

        wait_seconds = (next_time - now).total_seconds()
        if wait_seconds > 0:
            self.logger.info(
                f"⏱ 当前时间 {now.strftime('%H:%M:%S')} 未对齐5分钟边界，等待 {wait_seconds:.1f}s 至 {next_time.strftime('%H:%M:%S')}"
            )
            time.sleep(wait_seconds)

        return self.is_monitoring

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
                if position.get('exit_in_progress'):
                    continue
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
            if position.get('exit_in_progress'):
                continue
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
