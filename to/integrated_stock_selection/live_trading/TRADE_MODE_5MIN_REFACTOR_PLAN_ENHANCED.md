# live_trading Trade模式5分钟重构方案（增强版）

## 📋 重构目标

将 `live_trading` 的 `trade` 模式从**日线级别交易**改为**5分钟K线级别交易**，采用与 `main.py` 中5分钟回测类似的**买入、卖出和风控逻辑**，并实现**完整的订单生命周期管理**。

---

## 🔍 关键修正点

### ⚠️ 修正1: 买入时机（CRITICAL）

**错误方案**: 等待09:35第一根5分钟K线生成后买入

**正确方案**: **09:30开盘立即买入**，不等待5分钟K线

**理由**:
- 5分钟回测使用开盘价买入，开盘价就是09:30集合竞价产生的价格
- 实盘应在09:30开盘后立即下单，而不是等到09:35
- 09:30集合竞价确定开盘价后，即可下单买入

**实现方法**:
```python
# T+1日09:25: 开始订阅行情
subscribe_quote(stock_codes, period='tick')  # 订阅tick数据

# 等待到09:30
wait_until_time('09:30:00')

# 获取最新价（此时开盘价已确定）
current_price = get_full_tick([stock_code])[stock_code]['lastPrice']

# 立即下单买入
order_stock_async(account, stock_code, ...)
```

---

### ⚠️ 修正2: 完整的订单生命周期管理（CRITICAL）

根据 [迅投交易API文档](https://dict.thinktrader.net/nativeApi/xttrader.html?id=CcRiHZ)，实现完整的订单管理：

#### 2.1 订单状态码

| 状态码 | 状态名称 | 说明 |
|--------|----------|------|
| 48 | 未报 | 订单已创建但未上报 |
| 49 | 待报 | 订单已上报但交易所未确认 |
| 50 | 已报 | 订单已提交到交易所 |
| 51 | 已报待撤 | 订单部分成交后等待撤单 |
| 52 | 部成 | 订单部分成交 |
| 53 | 已成 | 订单全部成交 ✅ |
| 54 | 已撤 | 订单已撤销 |
| 55 | 部撤 | 订单部分撤销 |
| 56 | 废单 | 订单被拒绝 |
| 57 | 未知 | 状态未知 |

#### 2.2 订单回调机制

**必须实现的回调函数**:

```python
from xtquant.xttrader import XtQuantTraderCallback
from xtquant.xttype import XtOrder, XtTrade, XtOrderError

class LiveTradingCallback(XtQuantTraderCallback):
    """实盘交易回调类"""

    def on_stock_order(self, order: XtOrder):
        """
        订单状态更新回调

        Args:
            order: 订单对象
                - order_id: 订单号
                - order_status: 订单状态（48-57）
                - order_time: 订单时间
                - traded_volume: 已成交数量
                - traded_price: 成交均价
                - total_volume: 委托数量
                - order_price: 委托价格
        """
        order_id = order.order_id
        status = order.order_status
        stock_code = order.stock_code

        logger.info(f"[订单回调] {stock_code} 订单{order_id} 状态: {status}")

        # 状态处理
        if status == 53:  # 已成
            logger.info(f"✅ 订单全部成交: {order_id}")
            logger.info(f"  成交量: {order.traded_volume}")
            logger.info(f"  成均价: {order.traded_price:.2f}")

            # 更新持仓记录
            self._on_order_filled(order)

        elif status == 52:  # 部成
            logger.warning(f"⚠️ 订单部分成交: {order_id}")
            logger.warning(f"  已成交: {order.traded_volume}/{order.total_volume}")

            # 判断是否需要撤单重试
            if order.traded_volume == 0:
                # 部成但成交量为0，可能是价格问题
                logger.warning("  部成但成交为0，准备撤单重试")
                self._schedule_cancel_retry(order_id)

        elif status == 56:  # 废单
            logger.error(f"❌ 订单被拒绝（废单）: {order_id}")
            logger.error(f"  可能原因: 价格不合法、数量不合法、账户问题等")

            # 立即重试
            self._on_order_rejected(order)

        elif status == 54:  # 已撤
            logger.info(f"🔄 订单已撤销: {order_id}")

    def on_stock_trade(self, trade: XtTrade):
        """
        成交回报回调

        Args:
            trade: 成交对象
                - order_id: 订单号
                - traded_id: 成交号
                - traded_price: 成交价格
                - traded_volume: 成交数量
                - traded_time: 成交时间
        """
        logger.info(f"[成交回报] 订单{trade.order_id}")
        logger.info(f"  成交号: {trade.traded_id}")
        logger.info(f"  价格: {trade.traded_price:.2f}")
        logger.info(f"  数量: {trade.traded_volume}")
        logger.info(f"  时间: {trade.traded_time}")

        # 保存成交记录
        self._save_trade_record(trade)

    def on_order_error(self, order_error: XtOrderError):
        """
        订单错误回调

        Args:
            order_error: 错误信息
                - order_id: 订单号
                - error_id: 错误代码
                - error_msg: 错误信息
        """
        logger.error(f"[订单错误] 订单{order_error.order_id}")
        logger.error(f"  错误代码: {order_error.error_id}")
        logger.error(f"  错误信息: {order_error.error_msg}")

        # 根据错误类型处理
        if '资金不足' in order_error.error_msg:
            logger.error("  资金不足，无法买入")
        elif '价格异常' in order_error.error_msg:
            logger.error("  价格异常，需要调整价格")
```

#### 2.3 智能订单管理器

**完整实现买入、撤单、重试逻辑**:

```python
class SmartOrderManager:
    """智能订单管理器"""

    def __init__(self, trader, callback):
        self.trader = trader
        self.callback = callback
        self.logger = logging.getLogger(__name__)

        # 订单追踪: {order_id: order_info}
        self.active_orders = {}

        # 重试配置
        self.MAX_RETRY_TIMES = 3  # 最大重试次数
        self.CANCEL_WAIT_SECONDS = 3  # 撤单等待时间（秒）
        self.ORDER_TIMEOUT_SECONDS = 10  # 订单超时时间（秒）

    def smart_buy_with_retry(self, stock_code: str, amount: float,
                            reference_price: float) -> Optional[str]:
        """
        智能买入（带重试机制）

        流程:
        1. 查询账户资金，计算可买数量
        2. 使用参考价下单
        3. 监控订单状态
        4. 部成或超时 → 撤单重试
        5. 全部成交 → 返回订单号
        6. 废单 → 调整参数重试

        Args:
            stock_code: 股票代码
            amount: 买入金额
            reference_price: 参考价（开盘价）

        Returns:
            str: 最终成交的订单号（失败返回None）
        """
        for retry in range(1, self.MAX_RETRY_TIMES + 1):
            self.logger.info(f"[买入重试 {retry}/{self.MAX_RETRY_TIMES}] {stock_code}")

            # Step 1: 查询账户资金
            asset = self.trader.query_stock_asset(self.trader.account)
            if not asset:
                self.logger.error("查询资金失败")
                continue

            available_cash = asset.cash

            # 检查资金是否足够
            if available_cash < amount:
                self.logger.error(
                    f"资金不足: 需要{amount:.2f}元, 可用{available_cash:.2f}元"
                )
                return None

            # Step 2: 计算买入数量（向下取整到100的倍数）
            buy_volume = self._calculate_buy_volume(
                available_cash,
                reference_price,
                buffer_ratio=0.95  # 留5%缓冲，防止价格变动导致资金不足
            )

            if buy_volume < 100:
                self.logger.error(
                    f"可买数量不足1手: {buy_volume}股"
                )
                return None

            self.logger.info(
                f"计算买入数量: 资金{available_cash:.2f}元, "
                f"价格{reference_price:.2f}元, 数量{buy_volume}股"
            )

            # Step 3: 下单
            order_id = self._place_buy_order(
                stock_code,
                buy_volume,
                reference_price
            )

            if not order_id:
                self.logger.error("下单失败")
                continue

            # Step 4: 等待成交
            filled = self._wait_for_order_fill(order_id, timeout=self.ORDER_TIMEOUT_SECONDS)

            if filled:
                self.logger.info(f"✅ 买入成功: 订单{order_id}")
                return order_id

            # 未成交，判断是否需要撤单重试
            if retry < self.MAX_RETRY_TIMES:
                self.logger.warning(f"订单未成交，撤单重试...")
                self._cancel_order_with_retry(order_id)
            else:
                self.logger.error(f"❌ 买入失败: 已重试{self.MAX_RETRY_TIMES}次")
                return None

        return None

    def _calculate_buy_volume(self, cash: float, price: float,
                            buffer_ratio: float = 0.95) -> int:
        """
        计算买入数量（向下取整到100的倍数）

        Args:
            cash: 可用资金
            price: 参考价格
            buffer_ratio: 缓冲比例（防止价格滑点）

        Returns:
            int: 买入数量（100的整数倍）
        """
        # 考虑缓冲比例
        effective_cash = cash * buffer_ratio

        # 计算数量
        volume = int(effective_cash / price / 100) * 100

        return volume

    def _place_buy_order(self, stock_code: str, volume: int,
                        price: float) -> Optional[str]:
        """
        下单买入

        Args:
            stock_code: 股票代码
            volume: 买入数量
            price: 买入价格

        Returns:
            str: 订单号（失败返回None）
        """
        try:
            from xtquant import xtconstant

            # 使用最新价下单（更灵活）
            order_id = self.trader.order_stock_async(
                account=self.trader.account,
                stock_code=stock_code,
                order_type=xtconstant.STOCK_BUY,
                order_volume=volume,
                price_type=xtconstant.LATEST_PRICE,  # 最新价
                price=price,  # 参考价（用于LATEST_PRICE模式）
                strategy_name='live_trading_5m',
                order_remark=stock_code
            )

            if order_id > 0:
                self.logger.info(f"下单成功: {stock_code} 买入{volume}股, 订单号={order_id}")
                return str(order_id)
            else:
                self.logger.error(f"下单失败: {stock_code}, order_id={order_id}")
                return None

        except Exception as e:
            self.logger.error(f"下单异常: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _wait_for_order_fill(self, order_id: str,
                            timeout: int = 10) -> bool:
        """
        等待订单成交

        Args:
            order_id: 订单号
            timeout: 超时时间（秒）

        Returns:
            bool: 是否成交
        """
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            # 查询订单状态
            order_status = self._query_order_status(order_id)

            if order_status == 53:  # 已成
                return True
            elif order_status in [54, 56]:  # 已撤、废单
                return False
            elif order_status == 52:  # 部成
                # 检查是否部分成交了至少50%
                order_info = self._get_order_info(order_id)
                if order_info:
                    filled_ratio = order_info['traded_volume'] / order_info['total_volume']
                    if filled_ratio >= 0.5:
                        self.logger.info(f"部分成交{filled_ratio:.1%}，视为成功")
                        return True

            # 休眠0.5秒后重试
            time.sleep(0.5)

        # 超时
        self.logger.warning(f"等待订单成交超时: {order_id}")
        return False

    def _query_order_status(self, order_id: str) -> Optional[int]:
        """
        查询订单状态

        Args:
            order_id: 订单号

        Returns:
            int: 订单状态（48-57，失败返回None）
        """
        try:
            orders = self.trader.query_stock_orders(self.trader.account)

            for order in orders:
                if str(order.order_id) == order_id:
                    return order.order_status

            return None

        except Exception as e:
            self.logger.error(f"查询订单状态失败: {e}")
            return None

    def _get_order_info(self, order_id: str) -> Optional[Dict]:
        """
        获取订单详细信息

        Args:
            order_id: 订单号

        Returns:
            Dict: 订单信息
        """
        try:
            orders = self.trader.query_stock_orders(self.trader.account)

            for order in orders:
                if str(order.order_id) == order_id:
                    return {
                        'order_id': str(order.order_id),
                        'stock_code': order.stock_code,
                        'order_status': order.order_status,
                        'order_volume': order.total_volume,
                        'traded_volume': order.traded_volume,
                        'traded_price': order.traded_price,
                        'order_price': order.order_price,
                    }

            return None

        except Exception as e:
            self.logger.error(f"获取订单信息失败: {e}")
            return None

    def _cancel_order_with_retry(self, order_id: str,
                                max_retry: int = 3) -> bool:
        """
        撤单（带重试）

        Args:
            order_id: 订单号
            max_retry: 最大重试次数

        Returns:
            bool: 是否成功
        """
        for retry in range(1, max_retry + 1):
            self.logger.info(f"[撤单重试 {retry}/{max_retry}] {order_id}")

            # 先检查订单状态
            order_status = self._query_order_status(order_id)

            if order_status == 53:  # 已成
                self.logger.info("订单已成交，无需撤单")
                return True
            elif order_status in [54, 56]:  # 已撤、废单
                self.logger.info("订单已撤销或废单，无需撤单")
                return True

            # 执行撤单
            result = self.trader.cancel_order_stock(self.trader.account, order_id)

            if result:
                self.logger.info(f"撤单请求成功: {order_id}")

                # 等待撤单生效
                import time
                time.sleep(self.CANCEL_WAIT_SECONDS)

                # 检查撤单结果
                order_status = self._query_order_status(order_id)
                if order_status in [54, 55, 56]:  # 已撤、部撤、废单
                    return True

            # 撤单失败，重试
            if retry < max_retry:
                time.sleep(1)

        self.logger.error(f"撤单失败: {order_id}")
        return False
```

#### 2.4 持仓和资金查询

**实时监控持仓和账户资金**:

```python
class PositionAndAssetManager:
    """持仓和资金管理器"""

    def __init__(self, trader):
        self.trader = trader
        self.logger = logging.getLogger(__name__)

    def query_positions(self) -> Dict[str, Dict]:
        """
        查询当前持仓

        Returns:
            Dict: {stock_code: position_info}
        """
        try:
            positions = self.trader.query_stock_positions(self.trader.account)

            position_dict = {}
            for pos in positions:
                position_dict[pos.stock_code] = {
                    'stock_code': pos.stock_code,
                    'volume': pos.volume,  # 持仓数量
                    'can_use_volume': pos.can_use_volume,  # 可用数量
                    'open_price': pos.open_price,  # 开仓价
                    'pnl': pos.pnl,  # 浮动盈亏
                    'pnl_ratio': pos.pnl_ratio,  # 盈亏比例
                }

            return position_dict

        except Exception as e:
            self.logger.error(f"查询持仓失败: {e}")
            return {}

    def query_asset(self) -> Optional[Dict]:
        """
        查询账户资金

        Returns:
            Dict: 资金信息
        """
        try:
            asset = self.trader.query_stock_asset(self.trader.account)

            if asset:
                return {
                    'cash': asset.cash,  # 可用资金
                    'total_asset': asset.total_asset,  # 总资产
                    'market_value': asset.market_value,  # 市值
                    'pnl': asset.pnl,  # 盈亏
                    'pnl_ratio': asset.pnl_ratio,  # 盈亏比例
                }
            else:
                return None

        except Exception as e:
            self.logger.error(f"查询资金失败: {e}")
            return None

    def get_buyable_volume(self, stock_code: str, price: float,
                          cash_ratio: float = 0.95) -> int:
        """
        计算可买入数量

        Args:
            stock_code: 股票代码
            price: 参考价格
            cash_ratio: 资金使用比例（默认95%，留5%缓冲）

        Returns:
            int: 可买入数量（100的整数倍）
        """
        asset_info = self.query_asset()

        if not asset_info:
            return 0

        available_cash = asset_info['cash'] * cash_ratio

        # 计算可买数量（向下取整到100）
        buyable_volume = int(available_cash / price / 100) * 100

        return buyable_volume
```

---

## 🎯 完整的交易流程

### T日（选股日）

```
1. 运行 select 模式
   └─> 选出N只股票
   └─> 保存到 stock_selections.json
```

### T+1日（买入日）

```
09:25 ──┐
         ├─> 连接迅投API
         ├─> 注册交易回调（LiveTradingCallback）
         └─> 订阅tick行情（用于获取最新价）

09:30 ──┐
         ├─> 等待开盘
         ├─> 获取最新价（开盘价已确定）
         │
         ├─> 【核心修改】立即下单买入（不等到09:35）
         │   ├─> 查询账户资金
         │   ├─> 计算可买数量
         │   ├─> 调用 smart_buy_with_retry()
         │   │   ├─> 下单
         │   │   ├─> 监控订单状态（通过回调）
         │   │   ├─> 部成/超时 → 撤单重试
         │   │   └─> 全成 → 记录持仓
         │   └─> 保存买入记录
         │
         └─> 订阅5分钟K线（用于后续监控）

09:35 ~ 15:00 ──┐
                  ├─> 持仓监控启动
                  ├─> 但T+1日不触发风控
                  └─> 每5分钟获取K线数据（预热）
```

### T+2日及以后（风控日）

```
09:30 ~ 15:00 ──┐
                ├─> 每5分钟检查一次
                │
                ├─> 获取最新5分钟K线
                ├─> 风控检查：
                │   ├─> 条件1: 累计下跌 > 3% 且 日内跌幅 > 2%
                │   │   └─> 立即卖出
                │   └─> 条件2: 连续3根阴线 且 累计跌幅 > 2%
                │       └─> 立即卖出
                │
                ├─> 卖出流程:
                │   ├─> 调用 smart_sell_with_retry()
                │   │   ├─> 下单卖出
                │   │   ├─> 监控订单状态
                │   │   └─> 未成交 → 撤单重试
                │   └─> 保存卖出记录
                │
                └─> 检查持仓期
                    ├─> 满3个交易日
                    └─> 强制卖出
```

---

## 📝 核心代码实现

### 3.1 XtQuantTrader 增强

```python
class XtQuantTrader:
    """迅投交易客户端（增强版）"""

    def __init__(self, session_id=None):
        self.account = None
        self.callback = None
        self.logger = logging.getLogger(__name__)

        # 订单管理器
        self.order_manager = None

        # 持仓和资金管理器
        self.asset_manager = PositionAndAssetManager(self)

    def connect(self) -> bool:
        """连接迅投"""
        try:
            from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback

            # 连接
            trader = XtQuantTrader(session_id)

            # 注册回调
            self.callback = LiveTradingCallback(self)
            trader.register_callback(self.callback)

            # 连接交易终端
            if trader.connect() == 0:
                self.account = trader.get_stock_account()
                self.logger.info(f"连接成功: {self.account}")

                # 初始化订单管理器
                self.order_manager = SmartOrderManager(self, self.callback)

                return True
            else:
                self.logger.error("连接失败")
                return False

        except Exception as e:
            self.logger.error(f"连接异常: {e}")
            return False

    def subscribe_5m_data(self, stock_codes: List[str]) -> bool:
        """订阅5分钟K线"""
        try:
            from xtquant import xtdata

            for stock_code in stock_codes:
                seq = xtdata.subscribe_quote(
                    stock_code,
                    period='5m',
                    count=0
                )

                if seq > 0:
                    self.logger.info(f"✅ 订阅5分钟K线: {stock_code}")
                else:
                    self.logger.error(f"❌ 订阅失败: {stock_code}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"订阅5分钟K线失败: {e}")
            return False

    def get_current_price(self, stock_code: str) -> Optional[float]:
        """获取最新价"""
        try:
            from xtquant import xtdata

            tick_data = xtdata.get_full_tick([stock_code])

            if stock_code in tick_data:
                return tick_data[stock_code]['lastPrice']
            else:
                return None

        except Exception as e:
            self.logger.error(f"获取最新价失败: {e}")
            return None

    def get_5m_kline(self, stock_code: str, count: int = 100) -> pd.DataFrame:
        """获取5分钟K线数据"""
        try:
            from xtquant import xtdata

            data = xtdata.get_market_data_ex(
                stock_list=[stock_code],
                period='5m',
                count=count,
                dividend_type='none'
            )

            if stock_code in data:
                return data[stock_code]
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取5分钟K线失败: {e}")
            return pd.DataFrame()
```

### 3.2 开盘风控检查

**关键修改**: 09:30立即买入，不等待5分钟K线

```python
def execute_buy_orders_at_open(self, stocks: List[Dict],
                               total_amount: float,
                               num_stocks: int,
                               buy_date: datetime.date) -> Dict[str, str]:
    """
    T+1日09:30开盘立即买入

    流程:
    1. 等待到09:30
    2. 获取最新价（开盘价）
    3. 执行开盘风控检查
    4. 对通过风控的股票立即买入

    Args:
        stocks: 股票列表
        total_amount: 总金额
        num_stocks: 买入数量
        buy_date: 买入日期

    Returns:
        Dict: {stock_code: order_id}
    """
    self.logger.info("=" * 60)
    self.logger.info(f"开始T+1日买入: {buy_date} 09:30")

    # 1. 等待到09:30
    self._wait_until_time('09:30:00')

    # 2. 订阅tick行情
    stock_codes = [s['code'] for s in stocks[:num_stocks]]
    if not self._subscribe_tick_data(stock_codes):
        self.logger.error("订阅tick行情失败")
        return {}

    # 3. 获取前一日收盘价
    prev_closes = {}
    for stock in stocks[:num_stocks]:
        stock_code = stock['code']
        prev_closes[stock_code] = self._get_prev_close(stock_code, buy_date)

    # 4. 执行买入
    buy_results = {}
    amount_per_stock = total_amount / num_stocks

    for stock in stocks[:num_stocks]:
        stock_code = stock['code']

        # 获取最新价（开盘价）
        current_price = self.trader.get_current_price(stock_code)

        if not current_price:
            self.logger.warning(f"{stock_code}: 无法获取最新价，跳过")
            continue

        self.logger.info(f"{stock_code}: 最新价={current_price:.2f}")

        # 执行开盘风控检查
        prev_close = prev_closes.get(stock_code)
        risk_check = self._check_opening_gap_risk(
            stock_code, buy_date, prev_close, current_price
        )

        if not risk_check['can_buy']:
            self.logger.warning(
                f"{stock_code}: 未通过开盘风控 ({risk_check['reason']})，跳过买入"
            )
            continue

        # 通过风控，立即买入
        self.logger.info(f"{stock_code}: ✅ 通过风控，开始买入...")

        # 使用智能订单管理器
        order_id = self.trader.order_manager.smart_buy_with_retry(
            stock_code,
            amount_per_stock,
            current_price
        )

        if order_id:
            buy_results[stock_code] = order_id
            self.logger.info(f"✅ 买入成功: {stock_code}, 订单号={order_id}")
        else:
            self.logger.error(f"❌ 买入失败: {stock_code}")

    self.logger.info("=" * 60)
    self.logger.info(f"买入完成: 成功{len(buy_results)}/{num_stocks}")

    # 5. 订阅5分钟K线（用于后续监控）
    if buy_results:
        self.trader.subscribe_5m_data(list(buy_results.keys()))

    return buy_results

def _wait_until_time(self, time_str: str):
    """
    等待到指定时间

    Args:
        time_str: 时间字符串，如 '09:30:00'
    """
    import time
    from datetime import datetime

    target_time = datetime.strptime(time_str, '%H:%M:%S').time()
    now = datetime.now()

    # 计算目标时间
    target = datetime.combine(now.date(), target_time)

    # 如果已过目标时间，等待下一秒
    if now >= target:
        target = target + pd.Timedelta(days=1)

    wait_seconds = (target - now).total_seconds()

    if wait_seconds > 0:
        self.logger.info(f"等待到 {time_str}，还需等待 {wait_seconds:.0f} 秒")
        time.sleep(wait_seconds)
```

---

## 📊 关键数据结构

### 订单追踪

```python
# self.order_manager.active_orders
{
    'order_id_123': {
        'stock_code': '600000.SH',
        'order_type': 'buy',  # buy/sell
        'order_volume': 100,
        'traded_volume': 0,
        'order_status': 50,  # 已报
        'create_time': '09:30:01',
        'update_time': '09:30:05',
        'retry_count': 0,
    }
}
```

### 持仓追踪

```python
# position_monitor.positions
{
    '600000.SH': {
        'stock_code': '600000.SH',
        'entry_price': 10.50,
        'entry_date': date(2025, 1, 9),
        'entry_datetime': datetime(2025, 1, 9, 9, 30, 5),
        'klines_after_entry': [
            {'time': '09:35', 'open': 10.48, 'close': 10.45, ...},
            {'time': '09:40', 'open': 10.45, 'close': 10.43, ...},
            # ...
        ],
        'cumulative_decline': -0.0286,  # 累计下跌-2.86%
        'consecutive_bearish': 2,  # 连续2根阴线
        'risk_triggered': False,
        'risk_reason': None,
        'should_exit': False,
    }
}
```

---

## ⚙️ 配置参数

```python
# config.py - 新增配置

class OrderConfig:
    """订单管理配置"""

    # 重试配置
    MAX_ORDER_RETRY_TIMES = 3
    ORDER_WAIT_TIMEOUT_SECONDS = 10
    CANCEL_WAIT_SECONDS = 3

    # 买入配置
    BUY_CASH_RATIO = 0.95  # 使用95%资金，留5%缓冲

    # 卖出配置
    SELL_ORDER_TIMEOUT_SECONDS = 5


class TradingConfig5Min:
    """5分钟交易配置"""

    # 时间配置
    MARKET_OPEN_TIME = '09:30:00'
    FIRST_5M_KLINE_TIME = '09:35:00'
    MARKET_CLOSE_TIME = '15:00:00'

    # 风控参数
    GAP_DOWN_THRESHOLD = -0.03  # 跌空3%
    GAP_UP_THRESHOLD = 0.05  # 高开5%
    CUMULATIVE_DECLINE_THRESHOLD = -0.03
    INTRADAY_DECLINE_THRESHOLD = -0.02
    CONSECUTIVE_BEARISH_COUNT = 3
    CONSECUTIVE_BEARISH_DECLINE = -0.02

    # 持仓期
    MAX_HOLDING_TRADING_DAYS = 3
```

---

## 📝 实施步骤

### 阶段1: 订单管理基础（1天）

1. ✅ 实现 `LiveTradingCallback` 回调类
2. ✅ 实现 `SmartOrderManager` 智能订单管理器
3. ✅ 实现 `PositionAndAssetManager` 持仓资金管理器
4. ✅ 测试下单、撤单、重试流程

### 阶段2: 09:30开盘买入（1天）

1. ✅ 修改买入时机为09:30（不等到09:35）
2. ✅ 实现 `_wait_until_time()` 等待函数
3. ✅ 实现 `_subscribe_tick_data()` tick订阅
4. ✅ 测试开盘买入流程

### 阶段3: 5分钟风控（2天）

1. ✅ 实现 `subscribe_5m_data()` 5分钟K线订阅
2. ✅ 重写 `_check_position_risk_5m()` 风控检查
3. ✅ 实现 `_5m_monitor_loop()` 5分钟监控循环
4. ✅ 测试风控触发

### 阶段4: 集成测试（1天）

1. ✅ 端到端测试完整流程
2. ✅ 模拟盘验证
3. ✅ 性能优化

---

## 🎯 预期效果

重构完成后：

1. ✅ **买入时机正确**: 09:30开盘立即买入
2. ✅ **订单管理完善**: 完整的下单、监控、撤单、重试机制
3. ✅ **风控逻辑一致**: 与5分钟回测完全一致
4. ✅ **持仓监控精准**: 5分钟级别实时监控
5. ✅ **资金利用合理**: 自动计算可买数量，留出缓冲

---

## 📚 参考资料

- [迅投交易API文档](https://dict.thinktrader.net/nativeApi/xttrader.html?id=CcRiHZ)
- [迅投行情API文档](https://dict.thinktrader.net/nativeApi/xtdata.html?id=CcRiHZ)
- [代码示例](https://dict.thinktrader.net/nativeApi/code_examples.html?id=CcRiHZ)
- 5分钟回测逻辑: `rqalpha_minute_datasource/strategy_backtest.py`

---

## ⚠️ 重要提醒

### 买入时机修正

- **错误**: 等待09:35第一根5分钟K线
- **正确**: 09:30开盘立即买入

### 订单管理要点

- 使用 `order_stock_async()` 异步下单
- 注册 `LiveTradingCallback` 接收订单回调
- 实现完整的撤单重试机制
- 实时查询持仓和资金
- 计算可买数量时留5%缓冲

### 风控逻辑要点

- T+1日不触发风控
- T+2日开始风控检查
- 先满足先触发（累计下跌 vs 连续阴线）
- 5分钟级别实时监控
