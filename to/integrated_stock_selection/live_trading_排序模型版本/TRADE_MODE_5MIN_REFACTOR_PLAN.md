# live_trading Trade模式重构方案

## 📋 重构目标

将 `live_trading` 的 `trade` 模式从**日线级别交易**改为**5分钟K线级别交易**，并采用与 `main.py` 中5分钟回测类似的**买入、卖出和风控逻辑**。

---

## 🔍 现状分析

### 当前 live_trading trade 模式

**交易周期**: 日线级别
**买入逻辑**:
- T+1日开盘一次性买入
- 使用 `get_full_tick()` 获取最新价

**持仓监控**:
- 监控周期：每3秒检查一次
- 止盈：+1.8x ATR
- 止损：-0.9x ATR 或 -5%
- 最大持仓期：3天（自然日）

**数据来源**:
- `get_full_tick([stock_code])` - 获取分笔最新价

### 5分钟回测逻辑（main.py）

**交易周期**: 5分钟K线级别
**买入风控**:
```python
def check_entry_signal():
    # 跳空检查
    gap_ratio = (buy_open - prev_close) / prev_close

    # 跌空超过3% → 禁止买入
    if gap_ratio < -0.03:
        return False

    # 高开超过5% → 禁止买入
    if gap_ratio > 0.05:
        return False

    return True
```

**持仓风控** (从T+2交易日开始):
```python
def check_position_risk():
    """
    两种风控条件，先满足先触发：

    1. 累计下跌风控：
       - 累计下跌 > 3% 且 日内跌幅 > 2% → 卖出

    2. 连续阴线风控：
       - 连续3根5分钟阴线 且 累计跌幅 > 2% → 卖出
    """
    # 获取T+2 ~ 当前的所有5分钟K线数据
    # 按时间顺序检查每个5分钟K线
    # 先满足先触发
```

**卖出时机**:
- T+2交易日（买入日的下一个交易日）开始风控检查
- 触发任一风控条件立即卖出
- 到期日强制卖出

**数据来源**:
- 从缓存加载5分钟K线数据

---

## 🎯 重构方案设计

### 1. 整体架构

```
┌─────────────────────────────────────────────────────┐
│                   run_trade_mode()                   │
│  ┌──────────────────────────────────────────────┐   │
│  │  1. 加载选股结果（select模式选出的股票）      │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  2. T+1日开盘前：                            │   │
│  │     - 订阅5分钟K线行情                       │   │
│  │     - 执行开盘风控检查（跳空检查）            │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  3. T+1日开盘后：                            │   │
│  │     - 通过风控的股票执行买入                 │   │
│  │     - 添加到5分钟监控循环                    │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  4. 5分钟监控循环（实时运行）：              │   │
│  │     - 每5分钟检查一次（新K线生成时）        │   │
│  │     - T+2日开始风控检查                     │   │
│  │     - 触发风控立即卖出                       │   │
│  │     - 到期日强制卖出                         │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

### 2. 核心模块重构

#### 2.1 数据获取模块 - `XtQuantTrader`

**新增方法**: 订阅并获取5分钟K线数据

```python
class XtQuantTrader:
    def subscribe_5m_data(self, stock_codes: List[str]) -> bool:
        """
        订阅5分钟K线行情数据

        Args:
            stock_codes: 股票代码列表

        Returns:
            bool: 订阅是否成功
        """
        try:
            from xtquant import xtdata

            # 对每只股票订阅5分钟K线
            for stock_code in stock_codes:
                # 订阅5分钟K线，设置回调
                seq = xtdata.subscribe_quote(
                    stock_code,
                    period='5m',
                    count=0,  # 只订阅实时数据，不取历史
                    callback=self._on_5m_data_callback
                )

                if seq > 0:
                    self.logger.info(f"✅ 订阅5分钟K线: {stock_code}, 订阅号={seq}")
                else:
                    self.logger.error(f"❌ 订阅失败: {stock_code}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"订阅5分钟K线失败: {e}")
            return False

    def get_5m_kline(self, stock_code: str, count: int = 100) -> pd.DataFrame:
        """
        获取5分钟K线数据（包含历史和实时）

        Args:
            stock_code: 股票代码
            count: 获取数量（默认100根K线）

        Returns:
            pd.DataFrame: 5分钟K线数据
            """
        try:
            from xtquant import xtdata

            # 先确保已订阅
            self.subscribe_5m_data([stock_code])

            # 获取数据
            data = xtdata.get_market_data_ex(
                stock_list=[stock_code],
                period='5m',
                count=count,
                dividend_type='none'  # 不复权
            )

            if stock_code in data:
                df = data[stock_code]
                self.logger.info(f"获取{stock_code} 5分钟K线: {len(df)}条")
                return df
            else:
                self.logger.warning(f"未获取到{stock_code}的数据")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取5分钟K线失败: {e}")
            return pd.DataFrame()

    def _on_5m_data_callback(self, data):
        """
        5分钟K线数据回调（实时推送）

        Args:
            data: 格式为 {stock_code: [kline1, kline2, ...]}
        """
        try:
            for stock_code, klines in data.items():
                if stock_code in self.minute_data_cache:
                    # 将新K线添加到缓存
                    self.minute_data_cache[stock_code].extend(klines)

                    # 保持缓存大小（例如只保留最近500根K线）
                    if len(self.minute_data_cache[stock_code]) > 500:
                        self.minute_data_cache[stock_code] = \
                            self.minute_data_cache[stock_code][-500:]
                else:
                    self.minute_data_cache[stock_code] = klines

            # 触发风控检查
            self._trigger_risk_check()

        except Exception as e:
            self.logger.error(f"处理5分钟数据回调失败: {e}")

    def _trigger_risk_check(self):
        """触发风控检查"""
        # 通知监控器进行风控检查
        if self.monitor and self.monitor.is_monitoring:
            self.monitor.check_risk_control_5m()
```

**关键点**:
- 使用 `subscribe_quote(period='5m')` 订阅5分钟K线
- 使用 `get_market_data_ex(period='5m')` 获取数据
- 设置回调函数接收实时推送
- 缓存5分钟K线数据用于风控检查

---

#### 2.2 买入模块 - `TradingScheduler`

**修改**: `execute_buy_orders()` 方法

```python
class TradingScheduler:
    def execute_buy_orders_with_5m_check(self, stocks: List[Dict],
                                         total_amount: float,
                                         num_stocks: int,
                                         trading_date: datetime.date) -> Dict[str, str]:
        """
        执行买入订单（带5分钟开盘风控检查）

        买入流程：
        1. T+1日开盘前订阅5分钟K线
        2. 等待开盘获取第一根5分钟K线
        3. 执行开盘风控检查（跳空检查）
        4. 通过风控的股票执行买入

        Args:
            stocks: 股票列表
            total_amount: 总金额
            num_stocks: 买入数量
            trading_date: 交易日期

        Returns:
            Dict: {stock_code: order_id}
        """
        self.logger.info("=" * 60)
        self.logger.info(f"开始T+1日买入检查: {trading_date}")

        # 1. 订阅5分钟K线
        stock_codes = [s['code'] for s in stocks[:num_stocks]]
        if not self.trader.subscribe_5m_data(stock_codes):
            self.logger.error("订阅5分钟K线失败")
            return {}

        # 2. 等待开盘
        self.wait_for_market_open()

        # 3. 等待第一根5分钟K线生成（09:35）
        self.logger.info("等待第一根5分钟K线生成...")
        time.sleep(60 * 6)  # 等待到09:35

        # 4. 获取前一日收盘价和今日开盘价
        buy_results = {}
        amount_per_stock = total_amount / num_stocks

        for stock in stocks[:num_stocks]:
            stock_code = stock['code']

            # 获取5分钟K线数据
            kline_5m = self.trader.get_5m_kline(stock_code, count=2)  # 获取最近2根

            if kline_5m.empty:
                self.logger.warning(f"{stock_code}: 未获取到5分钟K线，跳过")
                continue

            # 第一根K线（今日开盘）
            today_kline = kline_5m.iloc[-1]
            buy_open = today_kline['open']

            # 获取前一日收盘价（从历史数据）
            prev_close = self._get_prev_close(stock_code, trading_date)

            # 执行开盘风控检查
            risk_check = self._check_opening_gap_risk(
                stock_code, trading_date, prev_close, buy_open
            )

            if not risk_check['can_buy']:
                self.logger.warning(
                    f"{stock_code}: 未通过开盘风控 ({risk_check['reason']})，跳过买入"
                )
                continue

            # 通过风控，执行买入
            self.logger.info(f"{stock_code}: 通过风控检查，开始买入...")

            order_id = self.trader.order_manager.smart_buy_with_retry(
                stock_code,
                amount_per_stock,
                buy_open  # 参考价
            )

            if order_id:
                buy_results[stock_code] = order_id
            else:
                self.logger.warning(f"{stock_code}: 买入失败")

        self.logger.info("=" * 60)
        self.logger.info(f"买入完成: 成功{len(buy_results)}/{num_stocks}")

        return buy_results

    def _check_opening_gap_risk(self, stock_code: str, buy_date: datetime.date,
                                 prev_close: Optional[float], buy_open: float) -> Dict:
        """
        开盘跳空风控检查（与5分钟回测一致）

        Args:
            stock_code: 股票代码
            buy_date: 买入日期
            prev_close: 前一日收盘价
            buy_open: 今日开盘价

        Returns:
            Dict: {'can_buy': bool, 'reason': str}
        """
        self.logger.info(f"🔎 {stock_code} 开盘风控检查")
        self.logger.info(f"  前收盘: {prev_close}, 今开盘: {buy_open}")

        # 无前日数据
        if prev_close is None:
            self.logger.info(f"  ✅ 无前日数据，允许买入")
            return {'can_buy': True, 'reason': '无前日数据'}

        # 计算跳空幅度
        gap_ratio = (buy_open - prev_close) / prev_close
        self.logger.info(f"  跳空幅度: {gap_ratio:.4f} ({gap_ratio:.2%})")

        # 风控1: 跌空超过3%
        if gap_ratio < -0.03:
            self.logger.warning(f"  🚫 跌空超过3% ({gap_ratio:.2%})，禁止买入")
            return {
                'can_buy': False,
                'reason': f'跌空{-gap_ratio:.2%}',
                'skip_category': '开盘风控'
            }

        # 风控2: 高开超过5%
        elif gap_ratio > 0.05:
            self.logger.warning(f"  🚫 高开超过5% ({gap_ratio:.2%})，禁止买入")
            return {
                'can_buy': False,
                'reason': f'高开{gap_ratio:.2%}',
                'skip_category': '开盘风控'
            }

        self.logger.info(f"  ✅ 通过风控检查，跳空{gap_ratio:.2%}在范围内")
        return {'can_buy': True, 'reason': '通过风控', 'gap_ratio': gap_ratio}

    def _get_prev_close(self, stock_code: str, current_date: datetime.date) -> Optional[float]:
        """
        获取前一日收盘价

        Args:
            stock_code: 股票代码
            current_date: 当前日期

        Returns:
            float: 前一日收盘价
        """
        try:
            from xtquant import xtdata

            # 获取最近2天的日线数据
            daily_data = xtdata.get_market_data_ex(
                stock_list=[stock_code],
                period='1d',
                count=2
            )

            if stock_code in daily_data and len(daily_data[stock_code]) >= 2:
                # 倒数第二根就是前一日
                prev_close = daily_data[stock_code].iloc[-2]['close']
                return float(prev_close)
            else:
                self.logger.warning(f"{stock_code}: 无法获取前一日收盘价")
                return None

        except Exception as e:
            self.logger.error(f"获取前一日收盘价失败: {e}")
            return None
```

**关键变更**:
1. ✅ 订阅5分钟K线行情
2. ✅ 等待第一根5分钟K线生成
3. ✅ 执行跳空风控检查（与回测一致）
4. ✅ 只对通过风控的股票买入

---

#### 2.3 监控模块 - `PositionMonitor`

**完全重构**: 从日线监控改为5分钟K线监控

```python
class PositionMonitor:
    def __init__(self, trader, scheduler):
        self.trader = trader
        self.scheduler = scheduler
        self.logger = logging.getLogger(__name__)

        # 持仓信息: {stock_code: position_info}
        self.positions = {}

        # 5分钟K线数据缓存: {stock_code: [kline1, kline2, ...]}
        self.minute_data_cache = {}

        # 监控线程
        self.monitor_thread = None
        self.is_monitoring = False

        # 交易日历（用于判断T+2）
        self.trading_dates = []

        # 加载交易日历
        self._load_trading_calendar()

    def add_position_5m(self, stock_code: str, entry_price: float,
                        entry_date: datetime.date):
        """
        添加持仓监控（5分钟K线版本）

        Args:
            stock_code: 股票代码
            entry_price: 买入价格（开盘价）
            entry_date: 买入日期
        """
        self.positions[stock_code] = {
            'stock_code': stock_code,
            'entry_price': entry_price,
            'entry_date': entry_date,
            'entry_datetime': datetime.now(),
            'klines_after_entry': [],  # 买入后的5分钟K线
            'cumulative_decline': 0.0,   # 累计下跌幅度
            'consecutive_bearish': 0,     # 连续阴线数
            'risk_triggered': False,      # 是否触发风控
            'risk_reason': None,          # 风控原因
            'should_exit': False,         # 是否应该退出
        }

        self.logger.info(f"添加5分钟监控: {stock_code}, 买入价: {entry_price:.2f}, 买入日: {entry_date}")

    def start_5m_monitoring(self):
        """
        启动5分钟K线监控循环

        监控逻辑：
        - 等待每根新5分钟K线生成
        - T+2日开始风控检查
        - 检查累计下跌和连续阴线
        - 触发风控立即卖出
        """
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

    def _5m_monitor_loop(self):
        """5分钟监控循环"""
        while self.is_monitoring:
            try:
                # 检查是否为交易时间
                if not self.scheduler.is_trading_time():
                    time.sleep(60)
                    continue

                # 检查所有持仓
                for stock_code in list(self.positions.keys()):
                    position = self.positions[stock_code]

                    # 检查是否需要退出
                    if position['should_exit']:
                        continue

                    # 检查是否到了T+2（买入日的下一个交易日）
                    if not self._is_after_t2(position['entry_date']):
                        continue

                    # 执行5分钟风控检查
                    self._check_position_risk_5m(stock_code, position)

                # 休眠直到下个5分钟（对齐到5分钟整数倍）
                self._sleep_to_next_5min()

            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                time.sleep(10)

    def _check_position_risk_5m(self, stock_code: str, position: Dict):
        """
        5分钟K线风控检查（与回测完全一致）

        两种风控条件，先满足先触发：

        1. 累计下跌风控：
           - 累计下跌 > 3% 且 当日5分钟跌幅 > 2% → 卖出

        2. 连续阴线风控：
           - 连续3根5分钟阴线 且 累计跌幅 > 2% → 卖出
        """
        try:
            # 获取最新的5分钟K线数据
            latest_klines = self.trader.get_5m_kline(stock_code, count=100)

            if latest_klines.empty:
                return

            # 筛选买入后的K线（T+2日开始的）
            entry_date_str = position['entry_date'].strftime('%Y%m%d')
            after_entry_klines = latest_klines[
                latest_klines.index >= entry_date_str
            ]

            if len(after_entry_klines) < 3:
                return

            self.logger.info(f"🛡️ {stock_code}: 开始5分钟风控检查，K线数: {len(after_entry_klines)}")

            # 按时间顺序检查每根K线（先满足先触发）
            for i, kline in after_entry_klines.iterrows():
                current_price = kline['close']
                open_price = kline['open']

                # 更新累计下跌（从买入价开始计算）
                decline_from_entry = (current_price - position['entry_price']) / position['entry_price']

                # 判断是否为阴线
                is_bearish = current_price < open_price

                # 更新连续阴线计数
                if is_bearish:
                    position['consecutive_bearish'] += 1
                else:
                    position['consecutive_bearish'] = 0

                # ===== 风控条件1: 累计下跌风控 =====
                if decline_from_entry < -0.03:  # 累计下跌超过3%
                    # 检查当日5分钟跌幅是否超过2%
                    intraday_decline = (current_price - open_price) / open_price

                    if intraday_decline < -0.02:  # 当日跌幅超过2%
                        self.logger.warning(f"🚨 {stock_code}: 触发累计下跌风控!")
                        self.logger.warning(f"  累计下跌: {decline_from_entry:.2%}")
                        self.logger.warning(f"  日内跌幅: {intraday_decline:.2%}")

                        # 标记退出
                        position['should_exit'] = True
                        position['risk_reason'] = '累计下跌风控'

                        # 立即卖出
                        self._execute_sell_5m(stock_code, position, '累计下跌风控')
                        return

                # ===== 风控条件2: 连续阴线风控 =====
                if position['consecutive_bearish'] >= 3:
                    # 计算这3根阴线的累计跌幅
                    if len(after_entry_klines) >= i + 1:
                        recent_3_klines = after_entry_klines.iloc[max(0, i-2):i+1]
                        decline_3_klines = (
                            recent_3_klines.iloc[-1]['close'] - recent_3_klines.iloc[0]['open']
                        ) / recent_3_klines.iloc[0]['open']

                        if decline_3_klines < -0.02:  # 3根阴线累计跌幅超过2%
                            self.logger.warning(f"🚨 {stock_code}: 触发连续阴线风控!")
                            self.logger.warning(f"  连续阴线: {position['consecutive_bearish']}根")
                            self.logger.warning(f"  累计跌幅: {decline_3_klines:.2%}")

                            # 标记退出
                            position['should_exit'] = True
                            position['risk_reason'] = '连续阴线风控'

                            # 立即卖出
                            self._execute_sell_5m(stock_code, position, '连续阴线风控')
                            return

            # 检查是否超过最大持仓期（3个交易日）
            if self._is_max_holding_reached(position):
                self.logger.info(f"⏰ {stock_code}: 到期（持仓3个交易日），执行卖出")
                position['should_exit'] = True
                position['risk_reason'] = '到期'
                self._execute_sell_5m(stock_code, position, '到期')

        except Exception as e:
            self.logger.error(f"5分钟风控检查失败: {e}")

    def _execute_sell_5m(self, stock_code: str, position: Dict, reason: str):
        """
        执行5分钟级别卖出

        Args:
            stock_code: 股票代码
            position: 持仓信息
            reason: 卖出原因
        """
        try:
            # 获取当前持仓数量
            positions = self.trader.get_positions()

            if stock_code not in positions:
                self.logger.warning(f"{stock_code}: 不在持仓中")
                return

            volume = positions[stock_code]['can_use_volume']

            if volume < 100:
                self.logger.warning(f"{stock_code}: 可用数量不足1手")
                return

            # 获取最新5分钟K线的收盘价作为卖出价
            latest_klines = self.trader.get_5m_kline(stock_code, count=1)

            if not latest_klines.empty:
                current_price = latest_klines.iloc[-1]['close']

                # 计算收益
                entry_price = position['entry_price']
                profit_pct = (current_price - entry_price) / entry_price * 100

                self.logger.info(f"执行5分钟卖出: {stock_code}")
                self.logger.info(f"  原因: {reason}")
                self.logger.info(f"  买入价: {entry_price:.2f}")
                self.logger.info(f"  卖出价: {current_price:.2f}")
                self.logger.info(f"  收益率: {profit_pct:+.2f}%")

                # 执行卖出
                order_id = self.trader.sell_stock(stock_code, volume)

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

    def _sleep_to_next_5min(self):
        """
        休眠到下一个5分钟整数倍

        例如: 如果现在是09:32:15，则休眠到09:35:00
        """
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

    def _load_trading_calendar(self):
        """加载交易日历"""
        try:
            from xtquant import xtdata

            # 获取最近的交易日
            trading_dates = xtdata.get_trading_dates(
                market='SH',
                count=100  # 最近100个交易日
            )

            # 转换为date对象
            self.trading_dates = [
                datetime.strptime(str(d), '%Y%m%d').date()
                for d in trading_dates
            ]

            self.logger.info(f"加载交易日历: {len(self.trading_dates)}个交易日")

        except Exception as e:
            self.logger.error(f"加载交易日历失败: {e}")
            self.trading_dates = []
```

**关键变更**:
1. ✅ 监控周期从3秒改为5分钟
2. ✅ 使用5分钟K线数据进行风控检查
3. ✅ 实现**累计下跌风控**和**连续阴线风控**
4. ✅ 与5分钟回测逻辑完全一致

---

### 3. 配置参数调整

#### 3.1 `config.py` - 新增5分钟监控配置

```python
class TradingConfig5Min:
    """5分钟交易配置"""

    # 5分钟K线周期
    KLINE_PERIOD = '5m'

    # 5分钟数据缓存大小
    KLINES_CACHE_SIZE = 500

    # 风控检查参数（与回测一致）
    CUMULATIVE_DECLINE_THRESHOLD = -0.03  # 累计下跌3%
    INTRADAY_DECLINE_THRESHOLD = -0.02   # 日内下跌2%
    CONSECUTIVE_BEARISH_COUNT = 3         # 连续3根阴线
    CONSECUTIVE_BEARISH_DECLINE = -0.02   # 3根阴线累计下跌2%

    # 持仓期（3个交易日）
    MAX_HOLDING_TRADING_DAYS = 3

    # 监控间隔（5分钟）
    MONITOR_INTERVAL_5M = 5 * 60  # 300秒
```

---

### 4. 交易流程图

```
T日（选股日）
  └─> select模式选出股票
      └─> 保存到 stock_selections.json

T+1日（买入日）
  ├─> 09:25前
  │     └─> 订阅5分钟K线行情
  │
  ├─> 09:30
  │     └─> 等待第一根5分钟K线生成
  │
  ├─> 09:35
  │     ├─> 获取第一根5分钟K线
  │     ├─> 执行开盘风控检查（跳空）
  │     │  ├─> 跌空 > 3% → 跳过买入
  │     │  └─> 高开 > 5% → 跳过买入
  │     └─> 对通过风控的股票执行买入
  │
  └─> 09:40 ~ 15:00
        └─> 持仓监控（不检查风控，T+1日不触发）

T+2日（风控检查开始）
  ├─> 09:35 ~ 15:00
  │     ├─> 每5分钟检查一次
  │     ├─> 获取最新的5分钟K线
  │     ├─> 风控检查：
  │     │  ├─> 条件1: 累计下跌 > 3% 且 日内跌幅 > 2%
  │     │  │        └─> 立即卖出
  │     │  └─> 条件2: 连续3根阴线 且 累计跌幅 > 2%
  │     │  │        └─> 立即卖出
  │     └─> 检查持仓期（满3个交易日）
  │              └─> 到期立即卖出
  │
  └─> 如果卖出
        └─> 移除监控

T+3日、T+4日
  └─> 继续监控（逻辑同T+2）
```

---

### 5. 需要修改的文件清单

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `trader.py` | 新增订阅5分钟K线方法 | 高 |
| `trader.py` | 新增获取5分钟K线方法 | 高 |
| `trader.py` | 新增5分钟数据回调处理 | 高 |
| `trader.py` | 修改minute_data_cache结构 | 高 |
| `scheduler.py` | 修改execute_buy_orders()为5m版本 | 高 |
| `scheduler.py` | 新增_check_opening_gap_risk() | 高 |
| `scheduler.py` | 新增_get_prev_close() | 中 |
| `monitor.py` | 完全重构为5m版本 | 高 |
| `monitor.py` | 新增_check_position_risk_5m() | 高 |
| `monitor.py` | 新增5分钟监控循环 | 高 |
| `monitor.py` | 新增_is_after_t2() | 中 |
| `monitor.py` | 新增_is_max_holding_reached() | 中 |
| `monitor.py` | 新增_sleep_to_next_5min() | 中 |
| `monitor.py` | 新增_load_trading_calendar() | 中 |
| `config.py` | 新增TradingConfig5Min配置 | 低 |
| `main.py` | 修改run_trade_mode()调用新方法 | 高 |

---

## ⚠️ 关键注意事项

### 1. 数据订阅限制

- **单股订阅数量**: 不宜超过50只
- **解决方案**: 只订阅当前持仓的股票（通常3-5只）

### 2. A股T+1规则

- **T+1日**: 买入当天不检查风控
- **T+2日**: 开始风控检查并可以卖出
- **实现**: 使用交易日历判断T+2

### 3. 5分钟K线时间点

- **第一根K线**: 09:35（09:30-09:35）
- **最后一根K线**: 15:00（14:55-15:00）
- **共48根K线/天** (上午24根，下午24根)

### 4. 与回测的差异

| 维度 | 5分钟回测 | 实盘trade |
|------|------------|------------|
| 数据来源 | 缓存文件 | 迅投实时 |
| 买入时机 | 开盘价买入 | 第一根5分钟K线生成后买入 |
| 卖出时机 | 当根K线收盘卖出 | 触发后立即市价卖出 |
| 风控触发 | 模拟触发 | 实时触发 |

---

## 📝 实现建议

### 分阶段实施

**阶段1**: 数据层重构
- 修改 `trader.py`，新增5分钟K线订阅和获取
- 测试数据订阅和获取是否正常

**阶段2**: 买入逻辑重构
- 修改 `scheduler.py`，实现5分钟开盘风控检查
- 测试买入流程

**阶段3**: 监控逻辑重构
- 完全重写 `monitor.py`，实现5分钟风控检查
- 测试监控循环

**阶段4**: 集成测试
- 端到端测试完整流程
- 模拟盘验证

---

## 🎯 预期效果

重构后，live_trading的trade模式将：

1. ✅ **与5分钟回测逻辑一致**
   - 相同的买入风控（跳空检查）
   - 相同的持仓风控（累计下跌 + 连续阴线）
   - 相同的持仓期（3个交易日）

2. ✅ **响应更快速**
   - 5分钟级别监控（而非日线）
   - 实时触发风控卖出

3. ✅ **更精细的风控**
   - 检查每根5分钟K线
   - 累计下跌和连续阴线双重保护

4. ✅ **数据实时获取**
   - 不依赖缓存文件
   - 使用迅投实时行情

---

## 📚 参考资料

- [迅投xtdata行情模块文档](https://dict.thinktrader.net/nativeApi/xtdata.html)
- [迅投交易函数文档](https://dict.thinktrader.net/nativeApi/xttrader.html)
- 5分钟回测逻辑: `rqalpha_minute_datasource/strategy_backtest.py`
