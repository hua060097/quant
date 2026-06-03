# live_trading 5分钟监控修复总结

## 修复日期
2026-01-13

## 问题描述

### 原始错误
```
2026-01-13 09:30:56,261 - live_trading.monitor - ERROR - 5分钟风控检查失败: 'XtQuantTrader' object has no attribute 'get_5m_kline'
2026-01-13 09:30:56,291 - live_trading.monitor - ERROR - Traceback (most recent call last):
  File "C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\live_trading\monitor.py", line 297, in _check_position_risk_5m
    latest_klines = self.trader.get_5m_kline(stock_code, count=100)
                    ^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'XtQuantTrader' object has no attribute 'get_5m_kline'
```

### 问题根源

用户指出了核心问题：

> "xtdata.get_market_data_ex是收盘后获取历史k线的函数，我需要的是在每个5分钟整数倍的时间获取当时的最新价格来作为5分钟的close，上一个5分钟的价格作为这个5分钟的open，这样来判断5分钟k线的涨跌幅和计算连续阴线"

**错误的设计**：
- 原代码调用 `xtdata.get_market_data_ex` 获取历史5分钟K线数据
- 这个函数只能在**收盘后**获取历史K线，无法在交易时段实时获取
- 不适用于实时风控监控

**正确的设计**：
- 在每个5分钟整数倍时间点（如9:35, 9:40, 9:45...）
- 获取**当前最新价格**作为这个5分钟的 **close**
- 使用**上一个5分钟的价格**作为这个5分钟的 **open**
- 构建实时5分钟K线数据结构
- 用来判断涨跌和连续阴线

## 解决方案

### 核心思路

**实时构建5分钟K线，而不是查询历史K线**

```
时间线：
9:30:00  - 获取最新价格 = 10.50  → 第一根K线: open=10.50, close=10.50
9:35:00  - 获取最新价格 = 10.30  → 第二根K线: open=10.50, close=10.30 (阴线)
9:40:00  - 获取最新价格 = 10.20  → 第三根K线: open=10.30, close=10.20 (阴线)
9:45:00  - 获取最新价格 = 10.10  → 第四根K线: open=10.20, close=10.10 (阴线) → 触发连续3根阴线风控
```

### 实现细节

#### 1. 删除错误的API方法

**文件**: `live_trading/trader.py`

**删除的方法**：
- `XtQuantTrader.get_5m_kline()` (第419-449行)
- `TradingScheduler.get_5m_kline()` (第1176-1187行)

**原因**：这些方法使用 `xtdata.get_market_data_ex` 只能获取历史K线，不适合实时监控

#### 2. 添加实时5分钟K线数据收集方法

**文件**: `live_trading/monitor.py`

**新增方法**: `_collect_5min_kline_data()` (第620-690行)

```python
def _collect_5min_kline_data(self):
    """
    收集5分钟实时K线数据

    在每个5分钟整数倍时间点（如9:35, 9:40）获取当前最新价格，
    构建实时5分钟K线：open=上一根的close, close=当前价格
    """
    # 1. 获取当前时间
    now = datetime.now()
    time_str = now.strftime('%Y%m%d %H:%M:00')

    # 2. 遍历所有监控的股票
    for stock_code in list(self.positions.keys()):
        # 3. 检查是否已收集过这个时间点的数据
        if len(self.kline_cache[stock_code]) > 0:
            last_kline_time = self.kline_cache[stock_code][-1]['time']
            if last_kline_time == time_str:
                continue  # 已收集，跳过

        # 4. 获取当前最新价格
        current_price = self.trader.get_current_price(stock_code)

        # 5. 确定open价格（上一根K线的close）
        if len(self.kline_cache[stock_code]) > 0:
            open_price = self.kline_cache[stock_code][-1]['close']
        else:
            open_price = current_price  # 第一根K线

        # 6. 构建K线数据
        kline = {
            'time': time_str,
            'open': open_price,
            'close': current_price
        }

        # 7. 添加到缓存
        self.kline_cache[stock_code].append(kline)
```

**关键特性**：
- ✅ 去重检查：避免同一时间点重复收集
- ✅ 自动缓存管理：最多保留最近100根K线
- ✅ 实时价格获取：使用 `trader.get_current_price()` 获取最新价
- ✅ 详细的日志输出：显示每根K线的开、收、涨跌幅

#### 3. 修改监控循环

**文件**: `live_trading/monitor.py`

**修改位置**: `_5m_monitor_loop()` 方法 (第203行)

**修改前**：
```python
while self.is_monitoring:
    try:
        # 检查是否为交易时间
        if not self._is_trading_time():
            time.sleep(60)
            continue

        # 检查所有持仓
        for stock_code in list(self.positions.keys()):
            # ... 风控检查
            self._check_position_risk_5m(stock_code, position)

        # 休眠直到下个5分钟整数倍
        self._sleep_to_next_5min()
```

**修改后**：
```python
while self.is_monitoring:
    try:
        # 检查是否为交易时间
        if not self._is_trading_time():
            time.sleep(60)
            continue

        # ===== 收集5分钟K线数据（在5分钟整数倍时间点）=====
        self._collect_5min_kline_data()

        # 检查所有持仓
        for stock_code in list(self.positions.keys()):
            # ... 风控检查
            self._check_position_risk_5m(stock_code, position)

        # 休眠直到下个5分钟整数倍
        self._sleep_to_next_5min()
```

**改进**：
- ✅ 在风控检查之前先收集5分钟K线数据
- ✅ 确保每次循环都获取最新的价格数据

#### 4. 修改风控检查方法

**文件**: `live_trading/monitor.py`

**修改方法**: `_check_position_risk_5m()` (第286-428行)

**修改前**：
```python
def _check_position_risk_5m(self, stock_code: str, position: Dict):
    try:
        # ❌ 调用不存在的API方法
        latest_klines = self.trader.get_5m_kline(stock_code, count=100)

        if latest_klines is None or latest_klines.empty:
            return

        # 使用DataFrame的index筛选
        today_klines = after_entry_klines[
            after_entry_klines.index.str[:8] == today_str
        ]

        for i, kline in recent_klines.iterrows():
            open_price = kline['open']
            close_price = kline['close']
            # ...
```

**修改后**：
```python
def _check_position_risk_5m(self, stock_code: str, position: Dict):
    try:
        # ✅ 从缓存获取5分钟K线数据
        if stock_code not in self.kline_cache or len(self.kline_cache[stock_code]) == 0:
            self.logger.debug(f"{stock_code}: 暂无5分钟K线数据，跳过风控检查")
            return

        latest_klines = self.kline_cache[stock_code]

        # 筛选今天的K线
        today_klines = [
            kline for kline in latest_klines
            if kline['time'][:8] == today_str
        ]

        # 只检查最近3根K线
        recent_klines = today_klines[-3:]

        for kline in recent_klines:
            open_price = kline['open']
            close_price = kline['close']
            kline_time = kline['time']
            # ...
```

**改进**：
- ✅ 从缓存读取而不是API调用
- ✅ 使用字典列表而不是DataFrame（更轻量）
- ✅ 适配新的K线数据结构 `{'time': ..., 'open': ..., 'close': ...}`
- ✅ 保持所有风控逻辑不变（累计下跌、连续阴线、尾盘卖出）

## 技术细节

### K线数据结构

**旧格式** (DataFrame):
```python
# xtdata.get_market_data_ex 返回的DataFrame
                  open   close    volume
time
20250113 09:30:00  10.50  10.50   1000000
20250113 09:35:00  10.50  10.30   500000
20250113 09:40:00  10.30  10.20   300000
```

**新格式** (字典列表):
```python
# kline_cache[stock_code] = [
#     {'time': '20250113 09:30:00', 'open': 10.50, 'close': 10.50},
#     {'time': '20250113 09:35:00', 'open': 10.50, 'close': 10.30},
#     {'time': '20250113 09:40:00', 'open': 10.30, 'close': 10.20},
# ]
```

### 时间同步

**关键逻辑**：
```python
# 检查是否已经收集过这个时间点的数据
if len(self.kline_cache[stock_code]) > 0:
    last_kline_time = self.kline_cache[stock_code][-1]['time']
    if last_kline_time == time_str:
        # 已经收集过这个时间点的数据，跳过
        continue
```

**原因**：
- 避免在同一分钟内重复收集数据
- 确保每根K线的时间戳唯一

### 缓存管理

**自动清理**：
```python
# 保持缓存大小（最多保存最近100根K线）
if len(self.kline_cache[stock_code]) > 100:
    self.kline_cache[stock_code] = self.kline_cache[stock_code][-100:]
```

**好处**：
- 限制内存使用
- 保留足够的历史数据用于风控分析（100根 × 5分钟 = 8.3小时）

## 运行流程

### 完整的时间线

```
09:30:00  - 监控循环启动
           ↓
         _sleep_to_next_5min() 计算下一个5分钟时间点 = 09:35:00
           ↓
         等待5分钟...
           ↓
09:35:00  - 循环唤醒
           ↓
         _collect_5min_kline_data()
           ├─ 获取当前价格 = 10.30
           ├─ 构建K线: time='20250113 09:35:00', open=10.50, close=10.30
           └─ 日志: 📊 600506.SH 5分钟K线: 20250113 09:35:00 | 开:10.50 收:10.30 | 跌幅:-1.90% | 🔴阴线
           ↓
         _check_position_risk_5m()
           ├─ 从缓存读取K线数据
           ├─ 检查累计下跌 > 5%
           ├─ 检查连续阴线 ≥ 3根
           └─ 执行风控卖出（如果触发）
           ↓
         _sleep_to_next_5min() → 09:40:00
           ↓
09:40:00  - 重复上述流程
```

## 修改文件清单

### 1. `live_trading/trader.py`
- **删除**: `XtQuantTrader.get_5m_kline()` 方法
- **删除**: `TradingScheduler.get_5m_kline()` 方法

### 2. `live_trading/monitor.py`
- **新增**: `_collect_5min_kline_data()` 方法 (第620-690行)
  - 收集实时5分钟K线数据
  - 构建K线结构并缓存
  - 自动去重和缓存管理
- **修改**: `_5m_monitor_loop()` 方法 (第203行)
  - 在风控检查前调用 `_collect_5min_kline_data()`
- **修改**: `_check_position_risk_5m()` 方法 (第286-428行)
  - 从缓存读取K线数据而不是调用API
  - 适配新的字典列表格式
  - 保持所有风控逻辑不变

## 验证方法

### 预期日志输出

```
2026-01-13 09:35:00 - INFO - 📊 600506.SH 5分钟K线: 20250113 09:35:00 | 开:10.50 收:10.30 | 跌幅:-1.90% | 🔴阴线
2026-01-13 09:35:05 - INFO - 🛡️ 600506.SH: 开始5分钟风控检查，今天K线数: 5
2026-01-13 09:35:06 - INFO - 📉 600506.SH: 检测到阴线（20250113 09:35:00），连续阴线: 1根

2026-01-13 09:40:00 - INFO - 📊 600506.SH 5分钟K线: 20250113 09:40:00 | 开:10.30 收:10.20 | 跌幅:-0.97% | 🔴阴线
2026-01-13 09:40:05 - INFO - 📉 600506.SH: 检测到阴线（20250113 09:40:00），连续阴线: 2根

2026-01-13 09:45:00 - INFO - 📊 600506.SH 5分钟K线: 20250113 09:45:00 | 开:10.20 收:10.10 | 跌幅:-0.98% | 🔴阴线
2026-01-13 09:45:05 - WARNING - 🚨 600506.SH: 触发连续阴线风控!
2026-01-13 09:45:05 - WARNING -   连续阴线: 3根
2026-01-13 09:45:05 - WARNING -   日期: 20250113
2026-01-13 09:45:06 - INFO - 💰 600506.SH: 执行5分钟风控卖出
2026-01-13 09:45:10 - INFO - ✅ 600506.SH: 卖出成功，订单号: 123456
```

### 测试步骤

1. **启动实时监控**：
   ```bash
   cd C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection
   python -m live_trading.main trade --mode trade
   ```

2. **观察日志输出**：
   - 确认每5分钟有K线数据收集日志
   - 确认K线数据格式正确（开、收价格）
   - 确认连续阴线检测正常工作

3. **验证风控触发**：
   - 如果出现连续3根阴线，应该触发风控卖出
   - 检查日志中的风控原因和执行结果

## 常见问题

### Q1: 为什么不使用 `xtdata.get_market_data_ex`？
**A**: 这个函数只能获取**历史K线数据**，在交易时段无法获取当天正在形成的K线。对于实时风控监控，必须使用**当前最新价格**来构建K线。

### Q2: 第一根K线的open和close为什么相同？
**A**: 第一根K线没有"上一根K线"，所以open=close=当前价格。这不影响风控判断，因为后续的K线会正确计算涨跌。

### Q3: 如何确保K线时间同步？
**A**: 使用时间字符串去重检查：
```python
if last_kline_time == time_str:
    continue  # 已收集，跳过
```
确保每个5分钟时间点只收集一次数据。

### Q4: 缓存大小100根K线够吗？
**A**: 足够了。100根 × 5分钟 = 8.3小时，覆盖整个交易时段（9:30-15:00 = 5.5小时）。

## 总结

### 修复内容
✅ **删除错误的API方法**：移除只能获取历史K线的 `get_5m_kline()` 方法
✅ **实现实时K线收集**：在每个5分钟时间点获取最新价格并构建K线
✅ **优化监控循环**：在风控检查前先收集数据
✅ **修改风控检查**：从缓存读取数据而不是API调用
✅ **保持风控逻辑不变**：累计下跌、连续阴线、尾盘卖出等逻辑完全保持

### 关键改进
| 方面 | 修改前 | 修改后 |
|------|--------|--------|
| K线数据来源 | `xtdata.get_market_data_ex` (历史) | `trader.get_current_price()` (实时) |
| 数据结构 | DataFrame | 字典列表 |
| 适用场景 | 收盘后查询 | 实时监控 |
| API依赖 | 依赖外部API | 只依赖实时价格 |
| 风控及时性 | 延迟（需等收盘） | 实时（5分钟级别） |

### 用户体验
- ✅ **实时性提升**：从"收盘后才能判断"提升到"5分钟实时判断"
- ✅ **风控响应更快**：连续3根阴线立即触发，无需等待
- ✅ **日志更清晰**：每根K线的开、收、涨跌幅一目了然
- ✅ **错误修复**：解决 `AttributeError` 问题

---

**修改人**: Claude Code Assistant
**修改时间**: 2026-01-13
**核心改进**: 从历史K线查询改为实时K线构建
