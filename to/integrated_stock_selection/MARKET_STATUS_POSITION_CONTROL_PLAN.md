# strategy_backtest.py 市场状态仓位控制修改方案
# Market Status Based Position Control Modification Plan

**日期 / Date**: 2026-02-14
**状态 / Status**: 📋 设计方案（待实施）

---

## 📋 需求概述 / Requirement Overview

### 核心需求
在 `run_backtest` 函数中加载 `market_status.json`，根据**市场状态**动态调整仓位控制阈值：

**判断逻辑**:
- 如果当日**指数 close < 指数 ma20**（市场弱势），则 `limit = 0.01`（提高门槛）
- 如果当日**指数 close >= 指数 ma20**（市场强势），则 `limit = 0.0`（降低门槛）

### 目标文件
- **市场状态数据**: `C:\Users\86151\.trae-cn\pythons\path\to\local_data\market_status\daily\market_status.json`
- **待修改文件**: `rqalpha_minute_datasource/strategy_backtest.py`

---

## 🔍 代码现状分析 / Current Code Analysis

### 1. `_position_pct_by_lead` 函数（第 1276-1291 行）

**当前实现**:
```python
def _position_pct_by_lead(self, lead_vs_top5mean: float) -> float:
    """根据 lead（topK 内相对优势）给出仓位比例。
    lead_vs_top5mean = score(stock) - mean(score(topK))
    规则：
      - lead >= 0.02 -> 100%
      - lead >= 0.01 -> 70%
      - else -> 0%（跳过）
    """
    limit = 0.01  # ⚠️ 固定阈值
    try:
        lead = float(lead_vs_top5mean)
    except Exception:
        return 0.0
    if lead >= limit:
        return 1.0
    return 0.0
```

**问题**:
- ❌ `limit = 0.01` 是**硬编码**的固定值
- ❌ 无法根据市场状态动态调整
- ❌ 注释中的规则（0.02/0.01）与实际代码不符

### 2. 调用位置（第 1885 行）

**调用上下文**:
```python
# 第 1877-1885 行
if entry_check['can_buy']:
    scores = pd.to_numeric(selected_stocks["predicted_return"], errors="coerce").to_numpy(dtype=float)
    scores = scores[np.isfinite(scores)]  # 去掉 NaN/inf

    if scores.size == 0:
        topk_mean = float(stock["predicted_return"])
    else:
        topk_mean = float(scores.mean())
    lead_vs_top5mean = float(stock['predicted_return']) - float(topk_mean)
    position_pct = self._position_pct_by_lead(lead_vs_top5mean)  # ⚠️ 调用点

    if position_pct <= 0.0:
        logger.info(f"🚫 {stock_code} 触发预测强度过滤: lead_vs_top5mean={lead_vs_top5mean:.4f} (<0.0100)，跳过买入")
        continue
```

**问题**:
- ❌ 调用时没有传递当日日期
- ❌ 无法获取当日市场状态

### 3. `run_backtest` 函数签名（第 1707 行）

**当前签名**:
```python
def run_backtest(self, trading_dates: List[datetime.date]) -> Dict[str, Any]:
    """运行完整的回测"""
    logger.info(f"🚀 开始回测 Top-{self.config.stock_count} 策略")
    logger.info(f"   时间范围: {trading_dates[0]} ~ {trading_dates[-1]}")
    logger.info(f"   总交易日: {len(trading_dates)}")
```

**问题**:
- ❌ 没有加载 `market_status.json`
- ❌ 没有将市场状态数据传递给 `_position_pct_by_lead`

---

## 📊 market_status.json 数据结构分析

### 数据结构

```json
{
  "20220705": {
    "individual_status": {
      "399006.SZ": {  // 创业板指
        "technical_data": {
          "close": 2825.13,
          "ma5": 2812.53,
          "ma10": 2789.22,
          "ma20": 2756.98,  // ✅ 需要的字段
          "ma60": 2638.56,
          ...
        }
      }
    }
  },
  "20220706": { ... },
  ...
}
```

### 关键信息

- **日期格式**: `"YYYYMMDD"` (字符串格式，如 `"20260212"`)
- **指数代码**: `"399006.SZ"` (创业板指)
- **关键字段**:
  - `individual_status.399006.SZ.technical_data.close` - 当日收盘价
  - `individual_status.399006.SZ.technical_data.ma20` - 20日均线
- **数据范围**: 2022-07-05 到 2026-02-12（879个交易日）

---

## 🎯 修改方案设计 / Modification Plan Design

### 方案概述

**核心思路**:
1. 在 `run_backtest` 开头加载 `market_status.json`
2. 将市场状态数据存储为实例变量
3. 修改 `_position_pct_by_lead` 函数签名，添加 `trade_date` 参数
4. 在函数内部根据当日市场状态动态计算 `limit`
5. 更新所有调用点，传递 `trade_date` 参数

---

## 🔧 详细修改方案 / Detailed Modification Plan

### 修改点1: 加载市场状态数据

**位置**: `run_backtest` 函数开头（第 1707 行之后）

**新增代码**:
```python
def run_backtest(self, trading_dates: List[datetime.date]) -> Dict[str, Any]:
    """运行完整的回测"""
    logger.info(f"🚀 开始回测 Top-{self.config.stock_count} 策略")
    logger.info(f"   时间范围: {trading_dates[0]} ~ {trading_dates[-1]}")
    logger.info(f"   总交易日: {len(trading_dates)}")

    # 🆕 加载市场状态数据
    self._load_market_status_data()

    # 确保trading_dates按时间顺序排列
    trading_dates = sorted(trading_dates)
    logger.info(f"✅ 交易日期已按时间排序")
```

### 修改点2: 实现数据加载方法

**位置**: 在 `_position_pct_by_lead` 方法之前添加（第 1274 行附近）

**新增方法**:
```python
def _load_market_status_data(self):
    """
    加载市场状态数据，用于动态仓位控制

    数据结构:
    {
        "20220705": {
            "individual_status": {
                "399006.SZ": {
                    "technical_data": {
                        "close": 2825.13,
                        "ma20": 2756.98,
                        ...
                    }
                }
            }
        }
    }
    """
    import json
    market_status_path = r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\market_status\daily\market_status.json"

    try:
        with open(market_status_path, 'r', encoding='utf-8') as f:
            self.market_status_data = json.load(f)

        logger.info(f"✅ 成功加载市场状态数据")
        logger.info(f"   数据路径: {market_status_path}")
        logger.info(f"   日期范围: {min(self.market_status_data.keys())} ~ {max(self.market_status_data.keys())}")
        logger.info(f"   总交易日: {len(self.market_status_data)}")

        # 验证数据格式
        sample_date = list(self.market_status_data.keys())[0]
        if "individual_status" in self.market_status_data[sample_date]:
            logger.info(f"   ✓ 数据格式正确: 包含 individual_status")
        else:
            logger.warning(f"   ⚠️ 数据格式可能不正确: 缺少 individual_status")

    except FileNotFoundError:
        logger.error(f"❌ 市场状态文件不存在: {market_status_path}")
        self.market_status_data = {}
    except Exception as e:
        logger.error(f"❌ 加载市场状态数据失败: {e}")
        self.market_status_data = {}
```

### 修改点3: 修改 `_position_pct_by_lead` 函数签名和实现

**位置**: 第 1276-1291 行

**当前代码**:
```python
def _position_pct_by_lead(self, lead_vs_top5mean: float) -> float:
    """根据 lead（topK 内相对优势）给出仓位比例。
    lead_vs_top5mean = score(stock) - mean(score(topK))
    规则：
      - lead >= 0.02 -> 100%
      - lead >= 0.01 -> 70%
      - else -> 0%（跳过）
    """
    limit = 0.01  # ⚠️ 固定阈值
    try:
        lead = float(lead_vs_top5mean)
    except Exception:
        return 0.0
    if lead >= limit:
        return 1.0
    return 0.0
```

**修改后代码**:
```python
def _position_pct_by_lead(self, lead_vs_top5mean: float, trade_date: datetime.date) -> float:
    """
    根据 lead（topK 内相对优势）和市场状态给出仓位比例。

    Args:
        lead_vs_top5mean: 预测得分 - topK平均得分
        trade_date: 交易日期（用于获取当日市场状态）

    仓位控制规则：
    1. 市场弱势判断（基于创业板指 399006.SZ）:
       - 如果 指数close < 指数ma20，则 limit = 0.01（提高门槛）
       - 如果 指数close >= 指数ma20，则 limit = 0.0（降低门槛）

    2. 仓位分配:
       - lead >= limit -> 100% 仓位
       - lead < limit -> 0% 仓位（跳过）
    """
    # 🆕 根据市场状态动态计算 limit
    limit = self._get_position_limit_by_market_status(trade_date)

    logger.info(f"📊 市场状态仓位控制: trade_date={trade_date}, limit={limit}")

    try:
        lead = float(lead_vs_top5mean)
    except Exception:
        return 0.0

    if lead >= limit:
        return 1.0
    return 0.0
```

### 修改点4: 实现市场状态判断方法

**位置**: 在 `_position_pct_by_lead` 方法之后添加（第 1292 行之后）

**新增方法**:
```python
def _get_position_limit_by_market_status(self, trade_date: datetime.date) -> float:
    """
    根据市场状态动态计算仓位控制阈值

    规则:
    - 如果指数 close < 指数 ma20（市场弱势），则 limit = 0.01
    - 如果指数 close >= 指数 ma20（市场强势），则 limit = 0.0

    Args:
        trade_date: 交易日期

    Returns:
        limit: 仓位控制阈值（0.0 或 0.01）
    """
    # 默认值（市场弱势，提高门槛）
    limit = 0.01

    # 检查市场状态数据是否已加载
    if not hasattr(self, 'market_status_data') or not self.market_status_data:
        logger.warning(f"⚠️ 市场状态数据未加载，使用默认 limit={limit}")
        return limit

    # 转换日期格式: datetime.date -> "YYYYMMDD"
    date_str = trade_date.strftime('%Y%m%d')

    # 获取当日市场状态
    if date_str not in self.market_status_data:
        logger.warning(f"⚠️ 日期 {date_str} 不在市场状态数据中，使用默认 limit={limit}")
        return limit

    daily_status = self.market_status_data[date_str]

    # 提取创业板指（399006.SZ）的技术数据
    try:
        index_code = "399006.SZ"
        tech_data = daily_status["individual_status"][index_code]["technical_data"]

        close_price = tech_data["close"]
        ma20_price = tech_data["ma20"]

        logger.info(f"📊 {date_str} 创业板指(399006.SZ): close={close_price:.2f}, ma20={ma20_price:.2f}")

        # 判断市场状态
        if close_price < ma20_price:
            # 市场弱势：指数低于MA20，提高门槛
            limit = 0.01
            logger.info(f"🐻 市场弱势: close({close_price:.2f}) < ma20({ma20_price:.2f}), limit={limit}")
        else:
            # 市场强势：指数高于MA20，降低门槛
            limit = 0.0
            logger.info(f"🐂 市场强势: close({close_price:.2f}) >= ma20({ma20_price:.2f}), limit={limit}")

    except KeyError as e:
        logger.warning(f"⚠️ 市场状态数据缺少关键字段: {e}，使用默认 limit={limit}")
    except Exception as e:
        logger.error(f"❌ 解析市场状态数据失败: {e}，使用默认 limit={limit}")

    return limit
```

### 修改点5: 更新调用点（第 1885 行）

**当前代码**:
```python
lead_vs_top5mean = float(stock['predicted_return']) - float(topk_mean)
position_pct = self._position_pct_by_lead(lead_vs_top5mean)  # ⚠️ 缺少 trade_date
```

**修改后代码**:
```python
lead_vs_top5mean = float(stock['predicted_return']) - float(topk_mean)
position_pct = self._position_pct_by_lead(lead_vs_top5mean, trade_date)  # ✅ 传递 trade_date
```

**注意**: `trade_date` 变量在上下文中已存在（第 1720 行的循环变量）

---

## 📋 修改清单 / Modification Checklist

| # | 位置 | 修改类型 | 描述 | 优先级 |
|---|------|---------|------|--------|
| 1 | 第 1707 行后 | 新增代码 | 在 `run_backtest` 开头调用 `_load_market_status_data()` | 🔴 高 |
| 2 | 第 1274 行后 | 新增方法 | 添加 `_load_market_status_data()` 方法 | 🔴 高 |
| 3 | 第 1276-1291 行 | 修改方法 | 更新 `_position_pct_by_lead()` 签名和实现，添加 `trade_date` 参数 | 🔴 高 |
| 4 | 第 1292 行后 | 新增方法 | 添加 `_get_position_limit_by_market_status()` 方法 | 🔴 高 |
| 5 | 第 1885 行 | 更新调用 | 传递 `trade_date` 参数给 `_position_pct_by_lead()` | 🔴 高 |

---

## 🎨 代码流程图 / Code Flow Diagram

### 修改后的调用流程

```
run_backtest(trading_dates)
    │
    ├─→ _load_market_status_data()  🆕 加载市场状态数据
    │   └─→ self.market_status_data = {...}
    │
    └─→ for trade_date in trading_dates:
            │
            ├─→ lead_vs_top5mean = ...
            │
            ├─→ _position_pct_by_lead(lead_vs_top5mean, trade_date)  ✅ 传递日期
            │   │
            │   ├─→ _get_position_limit_by_market_status(trade_date)  🆕 动态获取 limit
            │   │   │
            │   │   ├─→ date_str = trade_date.strftime('%Y%m%d')
            │   │   ├─→ close = market_status_data[date_str]["individual_status"]["399006.SZ"]["technical_data"]["close"]
            │   │   ├─→ ma20 = market_status_data[date_str]["individual_status"]["399006.SZ"]["technical_data"]["ma20"]
            │   │   └─→ limit = 0.01 if close < ma20 else 0.0
            │   │
            │   └─→ return 1.0 if lead >= limit else 0.0
            │
            └─→ position_pct = ...
```

---

## 🧪 测试场景 / Test Scenarios

### 场景1: 市场强势（close >= ma20）

**输入**:
- `trade_date = datetime.date(2026, 2, 10)`
- 假设当日创业板指: `close = 3200.0`, `ma20 = 3100.0`
- `lead_vs_top5mean = 0.005`

**预期输出**:
- `limit = 0.0` (市场强势)
- `lead (0.005) >= limit (0.0)` → `position_pct = 1.0` (买入)
- 日志: `🐂 市场强势: close(3200.00) >= ma20(3100.00), limit=0.0`

### 场景2: 市场弱势（close < ma20）

**输入**:
- `trade_date = datetime.date(2026, 2, 10)`
- 假设当日创业板指: `close = 3000.0`, `ma20 = 3100.0`
- `lead_vs_top5mean = 0.005`

**预期输出**:
- `limit = 0.01` (市场弱势)
- `lead (0.005) < limit (0.01)` → `position_pct = 0.0` (不买入)
- 日志: `🐻 市场弱势: close(3000.00) < ma20(3100.00), limit=0.01`

### 场景3: 数据缺失（降级处理）

**输入**:
- `trade_date = datetime.date(2022, 1, 1)` (不在 market_status.json 中)
- `lead_vs_top5mean = 0.005`

**预期输出**:
- `limit = 0.01` (默认值)
- `lead (0.005) < limit (0.01)` → `position_pct = 0.0` (不买入)
- 日志: `⚠️ 日期 20220101 不在市场状态数据中，使用默认 limit=0.01`

---

## ⚠️ 注意事项 / Important Notes

### 1. 日期格式转换

**问题**: `trade_date` 是 `datetime.date` 对象，但 `market_status.json` 的 key 是字符串 `"YYYYMMDD"`

**解决**: 使用 `trade_date.strftime('%Y%m%d')` 进行转换

### 2. 数据完整性

**风险**: `market_status.json` 可能不包含所有回测日期

**处理**: 添加降级逻辑，如果日期不存在，使用默认 `limit = 0.01`

### 3. 指数选择

**当前选择**: 创业板指 (`399006.SZ`)

**可选替代**:
- 上证指数 (`000001.SH`)
- 深证成指 (`399001.SZ`)

**建议**: 保持当前选择，除非有特殊需求

### 4. 性能影响

**影响**: 每次调用 `_position_pct_by_lead` 都需要查询字典

**评估**: 字典查询是 O(1) 操作，性能影响可忽略

### 5. 向后兼容性

**兼容性**: 新增参数 `trade_date` 在调用时必须传递

**建议**: 在方法签名中使用默认值 `None`，提供更好的兼容性

**修改后的签名**:
```python
def _position_pct_by_lead(self, lead_vs_top5mean: float, trade_date: Optional[datetime.date] = None) -> float:
```

---

## 🚀 实施步骤 / Implementation Steps

1. **阶段1: 准备工作**
   - [ ] 备份 `strategy_backtest.py`
   - [ ] 确认 `market_status.json` 路径正确
   - [ ] 验证数据格式

2. **阶段2: 新增方法**
   - [ ] 添加 `_load_market_status_data()` 方法
   - [ ] 添加 `_get_position_limit_by_market_status()` 方法

3. **阶段3: 修改方法**
   - [ ] 更新 `_position_pct_by_lead()` 方法签名
   - [ ] 修改方法实现

4. **阶段4: 更新调用**
   - [ ] 在 `run_backtest` 开头调用加载方法
   - [ ] 更新所有调用点传递 `trade_date`

5. **阶段5: 测试验证**
   - [ ] 单元测试市场状态判断
   - [ ] 运行完整回测
   - [ ] 检查日志输出

---

## 📊 预期效果 / Expected Results

### 策略行为变化

**修改前**:
- 固定 `limit = 0.01`
- 所有交易日使用相同的买入门槛

**修改后**:
- 市场强势时: `limit = 0.0` → 降低门槛，更多买入机会
- 市场弱势时: `limit = 0.01` → 提高门槛，减少买入风险

### 回测指标预期变化

| 指标 | 市场强势 | 市场弱势 | 整体 |
|------|---------|---------|------|
| 买入次数 | ↑ 增加 | ↓ 减少 | 优化 |
| 胜率 | ↑ 提高 | ↑ 提高 | ↑ 提高 |
| 最大回撤 | ↓ 降低 | ↓ 降低 | ↓ 降低 |
| 夏普比率 | ↑ 提高 | ↑ 提高 | ↑ 提高 |

---

**文档版本**: v1.0
**最后更新**: 2026-02-14
**状态**: ✅ 设计完成，待实施
