# live_trading Trade模式5分钟重构实施方案

## 📋 核心变更

### 变更1: 只买入top1股票

**原方案**: 买入前N只股票（通过--num_stocks参数指定，默认3只）

**新方案**: **只买入rank=1的top1股票**

**理由**: 集中资金买入最优质的股票

**实现**:
```python
# 读取选股结果
selections = json.load(stock_selections_file)
selected_stocks = selections[previous_trading_date]['stocks']

# 只买入rank=1的股票
top1_stock = [s for s in selected_stocks if s['rank'] == 1][0]
stock_code = top1_stock['code']
```

---

### 变更2: 全仓买入（不留缓冲）

**原方案**: 使用95%资金，留5%缓冲

**新方案**: **全仓买入，买入后剩余多少算多少**

**实现**:
```python
# 计算可买数量（全仓买入，不留缓冲）
def calculate_buy_volume_full(cash: float, price: float) -> int:
    """
    全仓买入计算（不留缓冲）

    Args:
        cash: 可用资金
        price: 参考价格

    Returns:
        int: 买入数量（100的整数倍）
    """
    volume = int(cash / price / 100) * 100
    return volume
```

---

### 变更3: 移除num_stocks参数

**原方案**: `python main.py --mode trade --date 20250109 --total_amount 30000 --num_stocks 3`

**新方案**: `python main.py --mode trade --date 20250109`

**理由**: 只买入top1，不需要num_stocks参数

**实现**:
- 移除命令行参数 `--num_stocks`
- 移除 `--total_amount` 参数（全仓买入，不需要指定金额）
- 简化为: `python main.py --mode trade --date 20250109`

---

## 🎯 完整的交易流程

### T日（选股日）

```bash
# 运行选股模式
python live_trading/main.py --mode select --date 20250109

# 选股结果保存到 stock_selections.json
{
  "20250109": {
    "stocks": [
      {"code": "600000.SH", "score": 2.345, "rank": 1},  # top1，第二天买入
      {"code": "000001.SZ", "score": 2.123, "rank": 2},
      {"code": "601318.SH", "score": 1.987, "rank": 3}
    ],
    "num_stocks": 3,
    "timestamp": "2025-01-09 20:00:00"
  }
}
```

### T+1日（买入日）

```bash
# 运行交易模式（注意：不需要--num_stocks和--total_amount参数）
python live_trading/main.py --mode trade --date 20250109
```

**执行流程**:

```
09:25 ──┐
         ├─> 连接迅投API
         ├─> 注册交易回调（LiveTradingCallback）
         └─> 订阅tick行情

09:30 ──┐
         ├─> 等待开盘
         ├─> 加载T日的选股结果（20250109）
         │  └─> 从stock_selections.json读取
         │
         ├─> 提取top1股票（rank=1）
         │  └─> stock_code = "600000.SH"
         │
         ├─> 获取最新价（开盘价）
         │  └─> current_price = 10.50
         │
         ├─> 查询账户资金
         │  └─> available_cash = 50000.00
         │
         ├─> 计算全仓买入数量
         │  └─> volume = int(50000 / 10.50 / 100) * 100 = 4700股
         │
         ├─> 执行开盘风控检查
         │  ├─> 跌空 > 3% → 跳过买入
         │  └─> 高开 > 5% → 跳过买入
         │
         └─> 立即下单买入（全仓）
            └─> 买入4700股，订单号=12345

09:35 ~ 15:00 ──┐
                  ├─> 订阅5分钟K线
                  ├─> 持仓监控启动
                  └─> T+1日不触发风控
```

### T+2日及以后（风控日）

```
09:30 ~ 15:00 ──┐
                ├─> 每5分钟检查一次
                │
                ├─> 获取最新5分钟K线
                ├─> 风控检查：
                │  ├─> 条件1: 累计下跌 > 3% 且 日内跌幅 > 2%
                │  │   └─> 立即卖出（全仓卖出）
                │  └─> 条件2: 连续3根阴线 且 累计跌幅 > 2%
                │      └─> 立即卖出（全仓卖出）
                │
                └─> 检查持仓期
                    ├─> 满3个交易日
                    └─> 强制卖出（全仓卖出）
```

---

## 📝 实现细节

### 1. 修改命令行参数

**文件**: `live_trading/main.py`

```python
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实盘交易系统')

    parser.add_argument('--mode', type=str, required=True, choices=['select', 'trade'],
                       help='运行模式: select-选股模式, trade-交易模式')

    parser.add_argument('--date', type=str, required=True,
                       help='日期(格式: YYYYMMDD)')

    # 移除 --num_stocks 和 --total_amount 参数

    args = parser.parse_args()
```

---

### 2. 读取前一个交易日的选股结果

**文件**: `live_trading/main.py`

```python
def run_trade_mode(args):
    """运行交易模式"""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("交易模式（买入top1股票）")
    logger.info("=" * 60)
    logger.info(f"选股日期(T日): {args.date}")

    # 1. 加载选股结果
    logger.info("\n步骤1: 加载选股结果...")

    if not PathConfig.STOCK_SELECTIONS_FILE.exists():
        logger.error(f"选股文件不存在: {PathConfig.STOCK_SELECTIONS_FILE}")
        return False

    try:
        with open(PathConfig.STOCK_SELECTIONS_FILE, 'r', encoding='utf-8') as f:
            selections = json.load(f)
    except Exception as e:
        logger.error(f"加载选股文件失败: {e}")
        return False

    if args.date not in selections:
        logger.error(f"未找到日期{args.date}的选股结果")
        logger.info(f"可用日期: {list(selections.keys())}")
        return False

    selected_stocks = selections[args.date]['stocks']

    if not selected_stocks:
        logger.warning("选股结果为空,无法交易")
        return False

    # 提取top1股票（rank=1）
    top1_stocks = [s for s in selected_stocks if s['rank'] == 1]

    if not top1_stocks:
        logger.error("未找到rank=1的股票")
        return False

    top1_stock = top1_stocks[0]
    stock_code = top1_stock['code']
    score = top1_stock['score']

    logger.info(f"✅ 找到top1股票: {stock_code}, 分数: {score:.3f}")

    # 2. 连接迅投API
    logger.info("\n步骤2: 连接迅投API...")

    trader = XtQuantTrader()
    if not trader.connect():
        logger.error("连接失败")
        return False

    # 3. 查询账户信息
    logger.info("\n步骤3: 查询账户信息...")

    account_info = trader.get_account_info()
    if not account_info:
        logger.error("查询账户信息失败")
        trader.disconnect()
        return False

    available_cash = account_info['cash']
    logger.info(f"可用资金: {available_cash:.2f}元")

    # 4. 计算交易日期
    logger.info("\n步骤4: 计算交易日期...")

    t_date = pd.to_datetime(args.date, format='%Y%m%d')
    logger.info(f"T日(选股日): {t_date.strftime('%Y-%m-%d')}")

    # T+1日: 买入日
    buy_date = t_date + pd.Timedelta(days=1)
    while buy_date.weekday() >= 5:
        buy_date += pd.Timedelta(days=1)
    logger.info(f"T+1日(买入日): {buy_date.strftime('%Y-%m-%d')}")

    # 5. T+1日买入
    logger.info("\n步骤5: T+1日买入...")

    today = datetime.now().date()
    buy_date_date = buy_date.date()

    if today < buy_date_date:
        logger.info(f"当前日期{today}, 买入日{buy_date_date}, 请在买入日运行")
        trader.disconnect()
        return False
    elif today > buy_date_date:
        logger.warning(f"当前日期{today}已过买入日{buy_date_date}, 跳过买入")
    else:
        # 执行买入
        logger.info("开始买入top1股票...")

        # 使用智能订单管理器（全仓买入）
        order_id = trader.order_manager.smart_buy_full_position(
            stock_code,
            buy_date.strftime('%Y-%m-%d')
        )

        if order_id:
            logger.info(f"✅ 买入成功: {stock_code}, 订单号={order_id}")

            # 保存记录
            trader.scheduler.save_trade_record(
                args.date,
                'buy',
                [stock_code],
                {stock_code: order_id}
            )

            # 添加5分钟监控
            # （后续实现）
        else:
            logger.error(f"❌ 买入失败: {stock_code}")
            trader.disconnect()
            return False

    # 6. 断开连接
    trader.disconnect()

    logger.info("\n" + "=" * 60)
    logger.info("交易完成")
    logger.info("=" * 60)

    return True
```

---

### 3. 全仓买入实现

**文件**: `live_trading/order_manager.py`（新增）

```python
class SmartOrderManager:
    """智能订单管理器（全仓买入版本）"""

    def smart_buy_full_position(self, stock_code: str,
                                buy_date: str) -> Optional[str]:
        """
        全仓买入top1股票

        流程:
        1. 等待到09:30
        2. 获取最新价（开盘价）
        3. 查询账户资金
        4. 计算全仓买入数量（不留缓冲）
        5. 执行开盘风控检查
        6. 立即下单买入
        7. 监控订单状态
        8. 部成/超时 → 撤单重试
        9. 全部成交 → 返回订单号

        Args:
            stock_code: 股票代码
            buy_date: 买入日期

        Returns:
            str: 订单号（失败返回None）
        """
        self.logger.info("=" * 60)
        self.logger.info(f"开始全仓买入: {stock_code}")
        self.logger.info("=" * 60)

        # 1. 等待到09:30
        self._wait_until_market_open()

        # 2. 获取最新价（开盘价）
        current_price = self.trader.get_current_price(stock_code)

        if not current_price:
            self.logger.error(f"无法获取{stock_code}的最新价")
            return None

        self.logger.info(f"当前价格: {current_price:.2f}元")

        # 3. 查询账户资金
        asset = self.trader.query_stock_asset(self.trader.account)

        if not asset:
            self.logger.error("查询账户资金失败")
            return None

        available_cash = asset.cash
        self.logger.info(f"可用资金: {available_cash:.2f}元")

        # 4. 计算全仓买入数量（不留缓冲）
        buy_volume = self._calculate_buy_volume_full(available_cash, current_price)

        if buy_volume < 100:
            self.logger.error(
                f"资金不足: 可买{buy_volume}股，不足1手（100股）"
            )
            return None

        self.logger.info(f"买入数量: {buy_volume}股")
        self.logger.info(f"预计金额: {buy_volume * current_price:.2f}元")
        self.logger.info(f"剩余资金: {available_cash - buy_volume * current_price:.2f}元")

        # 5. 执行开盘风控检查
        prev_close = self._get_prev_close(stock_code, buy_date)
        risk_check = self._check_opening_gap_risk(
            stock_code, buy_date, prev_close, current_price
        )

        if not risk_check['can_buy']:
            self.logger.warning(
                f"未通过开盘风控 ({risk_check['reason']})，取消买入"
            )
            return None

        # 6. 下单买入（全仓）
        self.logger.info("✅ 通过风控检查，开始下单...")

        for retry in range(1, self.MAX_RETRY_TIMES + 1):
            self.logger.info(f"[买入重试 {retry}/{self.MAX_RETRY_TIMES}]")

            # 下单
            order_id = self._place_buy_order(stock_code, buy_volume, current_price)

            if not order_id:
                self.logger.error("下单失败")
                continue

            # 等待成交
            filled = self._wait_for_order_fill(order_id, timeout=self.ORDER_TIMEOUT_SECONDS)

            if filled:
                self.logger.info(f"✅ 全仓买入成功: 订单{order_id}")
                return order_id

            # 未成交，撤单重试
            if retry < self.MAX_RETRY_TIMES:
                self.logger.warning("订单未成交，撤单重试...")
                self._cancel_order_with_retry(order_id)
            else:
                self.logger.error(f"❌ 买入失败: 已重试{self.MAX_RETRY_TIMES}次")
                return None

        return None

    def _calculate_buy_volume_full(self, cash: float, price: float) -> int:
        """
        全仓买入计算（不留缓冲）

        Args:
            cash: 可用资金
            price: 参考价格

        Returns:
            int: 买入数量（100的整数倍）
        """
        volume = int(cash / price / 100) * 100
        return volume
```

---

### 4. 5分钟监控（针对单只股票）

**文件**: `live_trading/monitor.py`

```python
class PositionMonitor:
    """持仓监控器（单只股票版本）"""

    def add_position_5m(self, stock_code: str, entry_price: float,
                        entry_date: str):
        """
        添加持仓监控

        Args:
            stock_code: 股票代码（只有1只）
            entry_price: 买入价格
            entry_date: 买入日期
        """
        self.positions[stock_code] = {
            'stock_code': stock_code,
            'entry_price': entry_price,
            'entry_date': entry_date,
            'klines_after_entry': [],
            'cumulative_decline': 0.0,
            'consecutive_bearish': 0,
            'risk_triggered': False,
            'risk_reason': None,
            'should_exit': False,
        }

        self.logger.info(f"添加5分钟监控: {stock_code}, 买入价: {entry_price:.2f}")

    def _execute_sell_5m(self, stock_code: str, position: Dict, reason: str):
        """
        执行卖出（全仓卖出）

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
```

---

## 📂 需要修改的文件清单

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `live_trading/order_manager.py` | 新增：智能订单管理器（全仓买入） | 高 |
| `live_trading/main.py` | 移除--num_stocks和--total_amount参数 | 高 |
| `live_trading/main.py` | 修改run_trade_mode()：只买入top1 | 高 |
| `live_trading/trader.py` | 增强XtQuantTrader：添加订单管理器 | 高 |
| `live_trading/monitor.py` | 修改为单只股票监控 | 高 |
| `live_trading/config.py` | 移除num_stocks相关配置 | 低 |
| `live_trading/README.md` | 更新文档 | 高 |

---

## 🔄 实施步骤

### 阶段1: 创建订单管理器（第1天）

1. ✅ 创建 `live_trading/order_manager.py`
2. ✅ 实现 `SmartOrderManager` 类
3. ✅ 实现 `smart_buy_full_position()` 全仓买入
4. ✅ 实现订单回调 `LiveTradingCallback`
5. ✅ 实现撤单重试机制

### 阶段2: 修改主程序（第1天）

1. ✅ 修改 `main.py` 的命令行参数
2. ✅ 修改 `run_trade_mode()` 读取top1股票
3. ✅ 移除 `--num_stocks` 和 `--total_amount` 参数
4. ✅ 实现全仓买入逻辑

### 阶段3: 修改trader.py（第1天）

1. ✅ 集成 `SmartOrderManager`
2. ✅ 实现 `get_current_price()` 获取最新价
3. ✅ 实现 `query_stock_asset()` 查询资金
4. ✅ 实现 `get_5m_kline()` 获取5分钟K线

### 阶段4: 修改monitor.py（第2天）

1. ✅ 修改为单只股票监控
2. ✅ 实现 `smart_sell_full_position()` 全仓卖出
3. ✅ 实现5分钟风控检查
4. ✅ 实现监控循环

### 阶段5: 更新文档（第2天）

1. ✅ 更新 `README.md`
2. ✅ 更新使用说明
3. ✅ 更新示例代码

---

## 📖 使用示例

### 选股（T日）

```bash
# 选股（选出top3，但只买top1）
python live_trading/main.py --mode select --date 20250109

# 输出:
# ✅ 选股完成
# 1. 600000.SH - 分数: 2.345  <- top1，明天买入
# 2. 000001.SZ - 分数: 2.123
# 3. 601318.SH - 分数: 1.987
```

### 交易（T+1日）

```bash
# 交易（自动买入top1，全仓）
python live_trading/main.py --mode trade --date 20250109

# 输出:
# ✅ 找到top1股票: 600000.SH, 分数: 2.345
# 可用资金: 50000.00元
# 当前价格: 10.50元
# 买入数量: 4700股
# 预计金额: 49350.00元
# 剩余资金: 650.00元
# ✅ 全仓买入成功: 订单12345
```

---

## ⚙️ 配置文件

**live_trading/config.py**:

```python
class TradingConfig:
    """交易配置"""

    # 全仓买入配置
    FULL_POSITION_BUY = True  # 全仓买入
    CASH_BUFFER_RATIO = 0.0  # 不留缓冲（原0.05）

    # 订单管理配置
    MAX_ORDER_RETRY_TIMES = 3
    ORDER_WAIT_TIMEOUT_SECONDS = 10
    CANCEL_WAIT_SECONDS = 3

class StopLossConfig:
    """风控配置"""

    # 持仓期
    MAX_HOLDING_DAYS = 3  # 3个交易日

    # 5分钟风控参数
    CUMULATIVE_DECLINE_THRESHOLD = -0.03  # 累计下跌3%
    INTRADAY_DECLINE_THRESHOLD = -0.02   # 日内下跌2%
    CONSECUTIVE_BEARISH_COUNT = 3         # 连续3根阴线
    CONSECUTIVE_BEARISH_DECLINE = -0.02   # 3根阴线累计下跌2%

    # 开盘风控
    GAP_DOWN_THRESHOLD = -0.03  # 跌空3%
    GAP_UP_THRESHOLD = 0.05  # 高开5%
```

---

## 📝 总结

### 核心变更

1. ✅ **只买入top1股票**: 从rank=1的股票买入
2. ✅ **全仓买入**: 不留5%缓冲，全部资金买入
3. ✅ **移除num_stocks参数**: 简化命令行参数
4. ✅ **移除total_amount参数**: 自动查询账户资金

### 优势

- **资金利用率最大化**: 全仓买入，不留闲置资金
- **策略更聚焦**: 集中买入最优质的股票
- **操作更简单**: 命令行参数更简洁
- **逻辑更清晰**: 只买1只，更容易监控和管理
