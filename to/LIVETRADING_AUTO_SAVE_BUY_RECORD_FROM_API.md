# Live Trading 自动保存买入记录功能实现

## 功能描述

当系统检测到持仓但找不到本地买入记录时，**自动从迅投API查询当日成交记录，并保存到本地**，避免手动干预。

## 实现逻辑

### **文件修改**
**文件**：`live_trading/trader.py` 第909-994行
**函数**：`TradingScheduler.load_latest_buy_record()`

### **执行流程**

```
1. 系统检测到持仓 002718.SZ
   ↓
2. 调用 load_latest_buy_record('002718.SZ')
   ↓
3. [优先] 查询本地 trade_records.json
   ├─ 找到 → 返回记录 ✅
   └─ 找不到 → 继续下一步
   ↓
4. [备用] 调用迅投API查询当日成交记录
   ├─ query_stock_trades(account)
   └─ 筛选 order_type == STOCK_BUY (买入)
   ↓
5a. 找到当日买入记录
    ├─ 构建买入记录对象
    ├─ 调用 save_trade_record() 保存到本地
    └─ 返回记录 ✅
    ↓
5b. 当日成交中也找不到
    └─ 返回 None，提示用户手动管理 ❌
```

## 关键代码

### **1. 查询当日成交记录**
```python
from xtquant import xtconstant
trades = self.trader.query_stock_trades(self.trader.account)
```

### **2. 筛选买入记录**
```python
for trade in trades:
    if (trade.stock_code == stock_code and
        trade.order_type == xtconstant.STOCK_BUY):  # 23 = 买入
```

### **3. 构建并保存记录**
```python
today_str = datetime.now().strftime('%Y-%m-%d')
buy_record = {
    'action': 'buy',
    'stocks': [stock_code],
    'order_ids': {stock_code: trade.order_id},
    'entry_date': today_str,
    'entry_price': trade.traded_price,
    'volume': trade.traded_volume
}

# 保存到本地
self.save_trade_record(
    date=datetime.now().strftime('%Y%m%d'),
    action='buy',
    stocks=[stock_code],
    order_ids={stock_code: trade.order_id},
    entry_date=today_str,
    entry_price=trade.traded_price,
    volume=trade.traded_volume
)
```

## 日志输出示例

### **场景1：本地有记录**
```log
✅ 从本地找到002718.SZ的买入记录: 日期=2026-01-12, 价格=15.50, 数量=700
```

### **场景2：本地无记录，从API获取**
```log
⚠️ 本地未找到002718.SZ的买入记录，尝试从API查询当日成交...
✅ 从API获取并保存002718.SZ的买入记录: 日期=2026-01-12, 价格=15.50, 数量=700
交易记录已保存: C:\Users\...\trade_records.json
```

### **场景3：API也找不到**
```log
⚠️ 本地未找到002718.SZ的买入记录，尝试从API查询当日成交...
❌ 当日成交记录中也没有找到002718.SZ的买入记录
  未找到002718.SZ的买入记录，无法监控
  请手动管理此持仓
```

## 使用场景

### **适用场景** ✅
1. **T+1日首次监控**：昨天（T日）选股，今天（T+1日）买入，当天启动监控
2. **程序崩溃重启**：买入后程序崩溃，本地记录未保存，重启后自动恢复
3. **手动买入补录**：通过QMT手动买入股票，启动程序后自动补录

### **限制场景** ⚠️
1. **只支持当日成交**：只能查询今天的成交记录，无法查询历史成交
2. **需要持仓第1天**：最适合T+1日（持仓第1天）使用
3. **持仓第2天及以后**：如果本地记录丢失，API无法找回（因为不是当日买入）

## 数据结构

### **保存的买入记录格式**
```json
{
  "20260112": [
    {
      "action": "buy",
      "stocks": ["002718.SZ"],
      "order_ids": {
        "002718.SZ": 12345
      },
      "entry_date": "2026-01-12",
      "entry_price": 15.50,
      "volume": 700,
      "timestamp": "2026-01-12 10:20:30"
    }
  ]
}
```

## 优势

### **1. 自动化** ✅
- 无需手动输入买入信息
- 系统自动识别并补录

### **2. 健壮性** ✅
- 本地记录丢失时自动恢复
- 减少人工干预

### **3. 数据准确性** ✅
- 直接从券商系统获取真实成交数据
- 避免手动输入错误

## 测试验证

### **测试步骤**
```bash
# 1. 删除本地买入记录（模拟丢失）
rm trade_records.json

# 2. 确保今天有买入股票（通过QMT手动买入或程序买入）

# 3. 运行交易模式
cd C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\live_trading
python main.py --mode trade --date 20260112
```

### **预期结果**
```log
步骤2: 查询账户持仓...
📊 查询到持仓数量: 1
  持仓1: 002718.SZ, 持仓数量: 700股, 可用数量: 0股
发现有效持仓: 1只股票
  002718.SZ: 700股

⚠️ 本地未找到002718.SZ的买入记录，尝试从API查询当日成交...
✅ 从API获取并保存002718.SZ的买入记录: 日期=2026-01-12, 价格=15.50, 数量=700
交易记录已保存: trade_records.json

✅ 找到002718.SZ的买入记录: 日期=2026-01-12, 价格=15.50, 数量=700
  买入日期: 2026-01-12
  买入价格: 15.50元
  买入数量: 700股
  持仓天数: 0天
```

## 相关文件

1. **live_trading/trader.py** - 修改 `load_latest_buy_record()` 函数
2. **live_trading/main.py** - 调用 `load_latest_buy_record()` 的业务逻辑
3. **trade_records.json** - 交易记录存储文件（自动生成）

## 技术细节

### **迅投API字段**
- `trade.stock_code`: 股票代码（如 "002718.SZ"）
- `trade.order_type`: 委托类型（23=买入，24=卖出）
- `trade.traded_price`: 成交均价
- `trade.traded_volume`: 成交数量
- `trade.traded_amount`: 成交金额
- `trade.order_id`: 订单编号
- `trade.traded_time`: 成交时间

### **常量定义**
```python
from xtquant import xtconstant
xtconstant.STOCK_BUY = 23   # 买入
xtconstant.STOCK_SELL = 24  # 卖出
```

## 总结

✅ **功能已实现**

现在系统可以：
1. **优先使用本地记录**（快速查询）
2. **本地无记录时自动从API获取**（数据恢复）
3. **自动保存到本地**（下次直接使用）
4. **清晰的日志提示**（便于调试）

这样即使本地记录丢失或首次启动，系统也能自动识别当日买入的股票并正常监控。

---

**修改时间**: 2026-01-12
**修改人**: Claude Code Assistant
**依赖**: 迅投XTQuant API - query_stock_trades()
