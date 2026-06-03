# 持仓天数判断逻辑详解

## 📊 核心概念

### 持仓天数定义

**持仓天数**：基于**交易日历**计算的持仓时间，而非自然日。

| holding_days | 含义 | 说明 | 操作 |
|-------------|------|------|------|
| **0** | 买入当天 | T日买入，T日当天 | ✅ 无法卖出（T+1制度），退出程序 |
| **1** | 第2天 | T日买入，T+1日 | 🛡️ 启动5分钟监控（无尾盘卖出） |
| **2** | 第3天 | T日买入，T+2日 | 🛡️ 启动5分钟监控（14:55尾盘卖出） |
| **≥3** | 异常 | 持仓超过3天 | ⚠️ 立即卖出 |

---

## 🧮 计算公式

### 核心公式
```python
holding_days = today_index - entry_index
```

**示例**：
```python
# 交易日历
trading_dates = [
    '2025-01-08',  # 索引 0
    '2025-01-09',  # 索引 1
    '2025-01-10',  # 索引 2 ← 买入日
    '2025-01-13',  # 索引 3
    '2025-01-14',  # 索引 4
    '2025-01-15',  # 索引 5
    ...
]

# 场景1：买入当天（2025-01-10）
entry_date = '2025-01-10'  # 索引 2
today = '2025-01-10'        # 索引 2
holding_days = 2 - 2 = 0  # ✅ 买入当天

# 场景2：第2天（2025-01-13）
entry_date = '2025-01-10'  # 索引 2
today = '2025-01-13'        # 索引 3
holding_days = 3 - 2 = 1  # ✅ 第2天

# 场景3：第3天（2025-01-14）
entry_date = '2025-01-10'  # 索引 2
today = '2025-01-14'        # 索引 4
holding_days = 4 - 2 = 2  # ✅ 第3天
```

---

## 🔄 完整判断流程

### 步骤1：加载交易日历

**位置**：`trader.py:TradingScheduler._load_trading_calendar()`

```python
# 从迅投API获取交易日历
trading_dates = xtdata.get_trading_dates(market='SH', start_time='', end_time='')

# 格式：[date, date, date, ...]
# 示例：[datetime.date(2025, 1, 8), datetime.date(2025, 1, 9), ...]
```

**交易日历特点**：
- ✅ 只包含交易日（周一至周五，排除节假日）
- ❌ 不包含周末（周六、周日）
- ❌ 不包含节假日（春节、国庆等）
- ✅ 按时间顺序排列

---

### 步骤2：加载买入记录

**位置**：`trader.py:TradingScheduler.load_latest_buy_record()`

```python
# 从 trade_records.json 加载买入记录
{
  "20250110": [
    {
      "action": "buy",
      "stocks": ["600000.SH"],
      "entry_date": "2025-01-10",  # ← 买入日期
      "entry_price": 10.50,           # ← 买入价格
      "volume": 4700                   # ← 买入数量
    }
  ]
}
```

**关键信息**：
- `entry_date`：买入日期（YYYY-MM-DD格式）
- 用于后续计算持仓天数

---

### 步骤3：计算持仓天数

**位置**：`trader.py:TradingScheduler.calculate_holding_days()`

```python
def calculate_holding_days(self, entry_date_str: str) -> int:
    """
    计算持仓天数（基于交易日历）

    Args:
        entry_date_str: 买入日期字符串 (YYYY-MM-DD)

    Returns:
        int: 持仓天数
            0 = 买入当天
            1 = 第2天
            2 = 第3天
            >=3 = 异常
    """
    try:
        # 1. 确保交易日历已加载
        if not self.trading_dates:
            self.logger.warning("交易日历未加载，无法计算持仓天数")
            return 0

        # 2. 转换日期格式
        entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d').date()
        today = datetime.now().date()

        # 3. 检查日期是否在交易日历中
        if entry_date not in self.trading_dates:
            # 容错：使用自然日计算
            return (today - entry_date).days

        if today not in self.trading_dates:
            # 容错：使用自然日计算
            return (today - entry_date).days

        # 4. 在交易日历中查找索引
        entry_index = self.trading_dates.index(entry_date)
        today_index = self.trading_dates.index(today)

        # 5. 计算持仓天数
        holding_days = today_index - entry_index

        return holding_days

    except Exception as e:
        self.logger.error(f"计算持仓天数失败: {e}")
        return 0
```

---

### 步骤4：根据持仓天数执行不同操作

**位置**：`main.py:run_trade_mode()`

```python
# 遍历所有持仓
for stock_code in positions:
    # 1. 加载买入记录
    buy_record = trader.scheduler.load_latest_buy_record(stock_code)

    # 2. 提取关键信息
    entry_date = buy_record['entry_date']
    entry_price = buy_record['entry_price']

    # 3. 计算持仓天数
    holding_days = trader.calculate_holding_days(entry_date)

    # 4. 根据持仓天数执行不同操作
    if holding_days == 0:
        # ========================================
        # 买入当天（T日）
        # ========================================
        logger.info(f"✅ {stock_code}: 当天买入（T+0），无法卖出，无需监控")
        continue  # 退出程序

    elif holding_days == 1:
        # ========================================
        # 第2天（T+1日）
        # ========================================
        logger.info(f"🛡️ {stock_code}: 持仓第2天，启动5分钟风控监控（无尾盘卖出）")

        # 创建监控器
        monitor_5m = PositionMonitor5Min(trader, scheduler)
        monitor_5m.enable_closing_sell = False  # 第2天不尾盘卖出

        # 添加持仓并启动监控
        monitor_5m.add_position_5m(stock_code, entry_price, entry_date)
        monitor_5m.start_5m_monitoring()

        # 持续监控直到用户按Ctrl+C或收盘
        # ...

    elif holding_days == 2:
        # ========================================
        # 第3天（T+2日）
        # ========================================
        logger.info(f"🛡️ {stock_code}: 持仓第3天，启动5分钟风控监控（14:55尾盘卖出）")

        # 创建监控器
        monitor_5m = PositionMonitor5Min(trader, scheduler)
        monitor_5m.enable_closing_sell = True  # 第3天启用尾盘卖出

        # 添加持仓并启动监控
        monitor_5m.add_position_5m(stock_code, entry_price, entry_date)
        monitor_5m.start_5m_monitoring()

        # 持续监控，14:55自动卖出
        # ...

    elif holding_days >= 3:
        # ========================================
        # 异常情况（持仓超过3天）
        # ========================================
        logger.warning(f"⚠️ {stock_code}: 持仓异常（{holding_days}天），立即卖出")

        # 立即卖出
        order_id = trader.order_manager.smart_sell_full_position(stock_code, volume)
```

---

## 📅 实际案例演示

### 案例1：买入当天再次运行

**时间线**：
```
2025-01-10 (周五) 09:35 - 买入600000.SH，程序退出
2025-01-10 (周五) 14:00 - 再次运行程序
```

**执行流程**：
```python
# 1. 查询持仓
positions = ['600000.SH']

# 2. 加载买入记录
entry_date = '2025-01-10'

# 3. 加载交易日历
trading_dates = [
    datetime.date(2025, 1, 8),   # 索引 0
    datetime.date(2025, 1, 9),   # 索引 1
    datetime.date(2025, 1, 10),  # 索引 2 ← 今天
    ...
]

# 4. 计算持仓天数
today = datetime.date(2025, 1, 10)  # 索引 2
entry_index = 2
today_index = 2
holding_days = 2 - 2 = 0

# 5. 判断操作
if holding_days == 0:
    logger.info("当天买入（T+0），无法卖出，无需监控")
    # 程序退出
```

**输出**：
```
发现持仓: 1只股票
处理持仓: 600000.SH
  持仓天数: 0天

✅ 600000.SH: 当天买入（T+0），无法卖出，无需监控
============================================================
```

---

### 案例2：跨周末持仓

**时间线**：
```
2025-01-10 (周五) - 买入600000.SH
2025-01-11 (周六) - 周末，不交易
2025-01-12 (周日) - 周末，不交易
2025-01-13 (周一) - 运行程序
```

**执行流程**：
```python
# 交易日历
trading_dates = [
    datetime.date(2025, 1, 8),   # 周三，索引 0
    datetime.date(2025, 1, 9),   # 周四，索引 1
    datetime.date(2025, 1, 10),  # 周五，索引 2 ← 买入日
    datetime.date(2025, 1, 13),  # 周一，索引 3 ← 今天
    datetime.date(2025, 1, 14),  # 周二，索引 4
    ...
]

# 计算持仓天数
entry_date = datetime.date(2025, 1, 10)  # 索引 2
today = datetime.date(2025, 1, 13)        # 索引 3
holding_days = 3 - 2 = 1

# 判断操作
if holding_days == 1:
    logger.info("持仓第2天，启动5分钟监控（无尾盘卖出）")
    # 启动监控
```

**输出**：
```
发现持仓: 1只股票
处理持仓: 600000.SH
  买入日期: 2025-01-10
  买入价格: 10.50元
  持仓天数: 1天

🛡️ 600000.SH: 持仓第2天，启动5分钟风控监控（无尾盘卖出）
✅ 5分钟K线监控已启动
```

**关键点**：
- ✅ 使用交易日历，周末不计入持仓天数
- ✅ 周五买入，周一运行，holding_days = 1（正确）

---

### 案例3：跨节假日持仓

**时间线**：
```
2025-01-10 (周五) - 买入前，发现不是交易日（假设春节）
2025-01-10 (周五) - 实际买入日
2025-01-13 (周一) - 春节假期
2025-01-14 (周二) - 春节假期
...
2025-02-03 (周一) - 节后第一天，运行程序
```

**执行流程**：
```python
# 交易日历
trading_dates = [
    ...
    datetime.date(2025, 1, 10),  # 索引 100 ← 买入日
    datetime.date(2025, 1, 17),  # 索引 101
    datetime.date(2025, 1, 24),  # 索引 102
    datetime.date(2025, 2, 3),   # 索引 103 ← 今天
    ...
]

# 计算持仓天数
entry_date = datetime.date(2025, 1, 10)  # 索引 100
today = datetime.date(2025, 2, 3)        # 索引 103
holding_days = 103 - 100 = 3

# 判断操作
if holding_days >= 3:
    logger.warning(f"持仓异常（3天），立即卖出")
    # 立即卖出
```

**输出**：
```
发现持仓: 1只股票
处理持仓: 600000.SH
  持仓天数: 3天

⚠️ 600000.SH: 持仓异常（3天），立即卖出
  买入价: 10.50, 卖出价: 11.03, 收益率: +5.05%
✅ 卖出成功: 600000.SH, 订单号=12346
```

---

## 🎯 关键要点

### 1. 使用交易日历，而非自然日

**原因**：
- A股只在交易日交易
- 周末、节假日不交易
- 持仓天数应该只计算交易日

**示例**：
```
自然日计算（错误）：
周五买入 → 周六(1天) → 周日(2天) → 周一(3天) ❌

交易日历计算（正确）：
周五买入(索引2) → 周一(索引3) → holding_days = 3-2 = 1天 ✅
```

---

### 2. 买入当天（holding_days=0）无法卖出

**原因**：A股T+1制度
- T日买入的股票，T日无法卖出
- 即使有持仓，也不启动监控
- 程序直接退出

---

### 3. 第2天（holding_days=1）vs 第3天（holding_days=2）

| 特性 | 第2天 (holding_days=1) | 第3天 (holding_days=2) |
|------|---------------------|---------------------|
| **是否监控** | ✅ 是 | ✅ 是 |
| **尾盘卖出** | ❌ 否 | ✅ 是（14:55） |
| **风控规则** | 累计下跌5% OR 连续3阴线 | 累计下跌5% OR 连续3阴线 OR 14:55 |

---

### 4. 持仓≥3天（异常情况）

**原因**：
- 正常情况下第3天14:55已经卖出
- 如果第4天还有持仓，说明异常
- 立即卖出控制风险

---

## 🔧 容错机制

### 如果日期不在交易日历中？

**容错逻辑**：
```python
if entry_date not in self.trading_dates:
    # 使用自然日计算
    return (today - entry_date).days

if today not in self.trading_dates:
    # 使用自然日计算
    return (today - entry_date).days
```

**场景**：
- 今天是周末（不在交易日历）
- 或买入日期是节假日（不在交易日历）
- 使用自然日作为后备方案

---

## 📊 数据流图

```
┌─────────────────────────────────────────┐
│ 1. 加载交易日历                         │
│    trading_dates = [date, date, ...]    │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 2. 加载买入记录                         │
│    entry_date = "2025-01-10"            │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 3. 计算持仓天数                         │
│    entry_index = trading_dates.index(entry_date) │
│    today_index = trading_dates.index(today)       │
│    holding_days = today_index - entry_index      │
└──────────────┬──────────────────────────┘
               ↓
       ┌───────┼───────┬──────────┐
       ↓       ↓       ↓          ↓
   holding_days = 0   = 1   = 2    >= 3
       ↓           ↓     ↓         ↓
   退出程序    启动监控 启动监控  立即卖出
              (无尾盘) (14:55)
```

---

## 🎓 总结

### 核心逻辑
1. **交易日历**：只包含交易日，排除周末和节假日
2. **索引差值**：`holding_days = today_index - entry_index`
3. **场景判断**：根据holding_days的值执行不同操作
4. **容错机制**：如果日期不在交易日历，使用自然日计算

### 优势
- ✅ 精确计算持仓天数
- ✅ 自动处理周末和节假日
- ✅ 符合A股T+1制度
- ✅ 避免误判和错误操作
