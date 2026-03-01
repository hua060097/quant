# 基于历史交易结果的动态仓位控制方案
# Trade History Based Position Control Modification Plan

**日期 / Date**: 2026-02-14
**状态 / Status**: 📋 设计方案（待实施）

---

## 📋 需求概述 / Requirement Overview

### 核心需求

**替换原有市场状态（均线）判断逻辑，改为基于历史交易结果的动态调整：**

**规则 1** - 提高门槛（防御模式）:
- 如果**上一次交易触发5%止损**，或者**连续3次交易都是负收益**
- 则设置 `limit = 0.01`

**规则 2** - 降低门槛（进攻模式）:
- 如果**最近连续两次交易都盈利**
- 则设置 `limit = 0.0`

**规则 3** - 默认模式:
- 其他情况，`limit = 0.0`（默认）

### 实现位置
- **文件**: `rqalpha_minute_datasource/strategy_backtest.py`
- **方法**: `_position_pct_by_lead` 和新增的辅助方法

---

## 🔍 代码现状分析 / Current Code Analysis

### 1. 交易记录结构 (第 1790-1807 行)

**`execute_sell` 方法创建的交易记录**:
```python
trade_record = {
    'stock_code': position.stock_code,
    'buy_datetime': position.buy_datetime,
    'sell_datetime': sell_datetime,
    'buy_price': position.buy_price,
    'sell_price': exit_price,
    'quantity': position.quantity,
    'predicted_return': position.predicted_return,
    'actual_return': actual_return,  # ✅ 实际收益率
    'profit_loss': (exit_price - position.buy_price) * position.quantity,
    'profit_loss_pct': actual_return * 100,  # ✅ 盈亏百分比
    'holding_days': (sell_date - position.buy_date).days,
    'entry_reason': position.entry_reason,
    'exit_reason': exit_reason,  # ✅ 退出原因（如 "风控: T+1最低价跌破"）
    'strategy_type': f"Top-{self.config.stock_count}"
}

self.trades.append(trade_record)  # ✅ 存储到交易列表
```

### 2. 止损识别逻辑

**止损阈值配置** (第 90 行):
```python
stop_loss_threshold: float = -0.05  # 默认-5%
```

**止损触发特征**:
- `exit_reason` 包含 "止损" 字样
- 或者 `actual_return <= -0.05` (≤ -5%)

### 3. 现有问题

**当前市场状态方案的问题**:
- ❌ 使用创业板指 MA20 判断，与个股交易结果无关
- ❌ 无法反映策略实际表现
- ❌ 缺乏自适应能力

---

## 🎯 新方案设计 / New Solution Design

### 方案架构

```
_position_pct_by_lead(lead_vs_top5mean, trade_date)
    │
    ├─→ _get_position_limit_by_trade_history()  🆕 基于历史计算 limit
    │   │
    │   ├─→ 检查最近N次交易结果
    │   ├─→ 判断规则1: 上次止损 OR 连续3次亏损 → limit = 0.01
    │   ├─→ 判断规则2: 连续2次盈利 → limit = 0.0
    │   └─→ 默认: limit = 0.0
    │
    └─→ return 1.0 if lead >= limit else 0.0
```

### 核心逻辑

```python
def _get_position_limit_by_trade_history(self) -> float:
    """基于历史交易结果动态计算仓位控制阈值"""

    # 规则1: 提高门槛（防御模式）
    if self._is_last_trade_stop_loss() or self._has_consecutive_losses(3):
        return 0.01

    # 规则2: 降低门槛（进攻模式）
    if self._has_consecutive_profits(2):
        return 0.0

    # 规则3: 默认
    return 0.0
```

---

## 🔧 详细修改方案 / Detailed Modification Plan

### 修改点1: 移除市场状态相关代码

**位置**: 多处

**需要删除的内容**:
1. 第 1275-1318 行: `_load_market_status_data()` 方法
2. 第 1321-1378 行: `_get_position_limit_by_market_status()` 方法
3. 第 1833 行: `run_backtest` 中的调用

### 修改点2: 添加历史交易分析方法

**位置**: 在 `_position_pct_by_lead` 之前添加（第 1320 行附近）

**新增方法1**: 检查上次交易是否止损

```python
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

    # 方式1: 检查收益率是否 ≤ -5%
    if actual_return <= -0.05:
        logger.info(f"🛡️ [止损检测] 上次交易触发5%止损: {last_trade['stock_code']}, return={actual_return:.4f}")
        return True

    # 方式2: 检查退出原因是否包含"止损"字样
    exit_reason = last_trade.get('exit_reason', '')
    if '止损' in exit_reason or 'stop_loss' in exit_reason.lower():
        logger.info(f"🛡️ [止损检测] 上次交易触发止损(原因): {last_trade['stock_code']}, reason={exit_reason}")
        return True

    return False
```

**新增方法2**: 检查连续N次亏损

```python
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
    all_losses = all(trade.get('actual_return', 0.0) < 0 for trade in recent_trades)

    if all_losses:
        returns = [trade.get('actual_return', 0.0) for trade in recent_trades]
        logger.info(f"📉 [连续亏损] 检测到连续{n}次亏损: {returns}")

    return all_losses
```

**新增方法3**: 检查连续N次盈利

```python
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
        logger.info(f"📈 [连续盈利] 检测到连续{n}次盈利: {returns}")

    return all_profits
```

**新增方法4**: 主判断逻辑

```python
def _get_position_limit_by_trade_history(self) -> float:
    """
    基于历史交易结果动态计算仓位控制阈值

    规则:
    1. 如果上一次交易触发5%止损，或连续3次交易都是负收益 → limit = 0.01
    2. 如果最近连续两次交易都盈利 → limit = 0.0
    3. 其他情况 → limit = 0.0 (默认)

    Returns:
        limit: 仓位控制阈值（0.0 或 0.01）
    """
    limit = 0.0  # 默认值
    reason = "默认"

    # 规则1: 提高门槛（防御模式）
    if self._is_last_trade_stop_loss():
        limit = 0.01
        reason = "上次交易触发5%止损"
    elif self._has_consecutive_losses(3):
        limit = 0.01
        reason = "连续3次交易亏损"

    # 规则2: 降低门槛（进攻模式）
    elif self._has_consecutive_profits(2):
        limit = 0.0
        reason = "连续2次交易盈利"

    # 规则3: 默认模式已在上面初始化

    logger.info(f"📊 [历史仓位控制] {reason}, limit={limit}")

    # 如果有交易历史，显示最近几次结果
    if self.trades:
        recent_count = min(5, len(self.trades))
        recent_trades = self.trades[-recent_count:]
        logger.info(f"   最近{recent_count}次交易结果:")
        for i, trade in enumerate(recent_trades, 1):
            ret = trade.get('actual_return', 0.0)
            reason_short = trade.get('exit_reason', '')[:15]
            logger.info(f"     {i}. {trade['stock_code']}: return={ret*100:.2f}%, reason={reason_short}")

    return limit
```

### 修改点3: 更新 `_position_pct_by_lead` 方法

**位置**: 第 1381-1410 行

**当前代码**:
```python
def _position_pct_by_lead(self, lead_vs_top5mean: float, trade_date: datetime.date) -> float:
    """
    根据 lead（topK 内相对优势）和市场状态给出仓位比例。
    ...
    """
    # 根据市场状态动态计算 limit
    limit = self._get_position_limit_by_market_status(trade_date)  # ❌ 旧逻辑

    logger.info(f"📊 市场状态仓位控制: trade_date={trade_date}, limit={limit}")

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
def _position_pct_by_lead(self, lead_vs_top5mean: float, trade_date: Optional[datetime.date] = None) -> float:
    """
    根据 lead（topK 内相对优势）和历史交易结果给出仓位比例。

    Args:
        lead_vs_top5mean: 预测得分 - topK平均得分
        trade_date: 交易日期（保留参数以兼容，但不再使用）

    仓位控制规则（基于历史交易结果）:
    1. 防御模式: 上次止损 OR 连续3次亏损 → limit = 0.01
    2. 进攻模式: 连续2次盈利 → limit = 0.0
    3. 默认模式: 其他情况 → limit = 0.0

    仓位分配:
    - lead >= limit -> 100% 仓位
    - lead < limit -> 0% 仓位（跳过）
    """
    # 根据历史交易结果动态计算 limit
    limit = self._get_position_limit_by_trade_history()  # ✅ 新逻辑

    try:
        lead = float(lead_vs_top5mean)
    except Exception:
        return 0.0

    if lead >= limit:
        return 1.0
    return 0.0
```

### 修改点4: 移除 `run_backtest` 中的市场状态加载

**位置**: 第 1832-1833 行

**当前代码**:
```python
def run_backtest(self, trading_dates: List[datetime.date]) -> Dict[str, Any]:
    """运行完整的回测"""
    logger.info(f"🚀 开始回测 Top-{self.config.stock_count} 策略")
    logger.info(f"   时间范围: {trading_dates[0]} ~ {trading_dates[-1]}")
    logger.info(f"   总交易日: {len(trading_dates)}")

    # 加载市场状态数据（用于动态仓位控制）
    self._load_market_status_data()  # ❌ 删除此行

    # 确保trading_dates按时间顺序排列
    trading_dates = sorted(trading_dates)
```

**修改后代码**:
```python
def run_backtest(self, trading_dates: List[datetime.date]) -> Dict[str, Any]:
    """运行完整的回测"""
    logger.info(f"🚀 开始回测 Top-{self.config.stock_count} 策略")
    logger.info(f"   时间范围: {trading_dates[0]} ~ {trading_dates[-1]}")
    logger.info(f"   总交易日: {len(trading_dates)}")

    # 确保trading_dates按时间顺序排列
    trading_dates = sorted(trading_dates)
```

---

## 📊 测试场景 / Test Scenarios

### 场景1: 上次交易止损

**交易历史**:
```
1. 000001.SZ: actual_return = -0.06 (触发止损)
```

**预期输出**:
```
🛡️ [止损检测] 上次交易触发5%止损: 000001.SZ, return=-0.0600
📊 [历史仓位控制] 上次交易触发5%止损, limit=0.01
lead=0.005 < limit(0.01) → 不买入
```

### 场景2: 连续3次亏损

**交易历史**:
```
1. 000001.SZ: actual_return = -0.02
2. 000002.SZ: actual_return = -0.01
3. 000003.SZ: actual_return = -0.03
```

**预期输出**:
```
📉 [连续亏损] 检测到连续3次亏损: [-0.02, -0.01, -0.03]
📊 [历史仓位控制] 连续3次交易亏损, limit=0.01
lead=0.005 < limit(0.01) → 不买入
```

### 场景3: 连续2次盈利

**交易历史**:
```
1. 000001.SZ: actual_return = 0.03
2. 000002.SZ: actual_return = 0.02
```

**预期输出**:
```
📈 [连续盈利] 检测到连续2次盈利: [0.03, 0.02]
📊 [历史仓位控制] 连续2次交易盈利, limit=0.0
lead=0.005 >= limit(0.0) → 买入
```

### 场景4: 无历史交易（首次运行）

**交易历史**: 空

**预期输出**:
```
📊 [历史仓位控制] 默认, limit=0.0
lead=0.005 >= limit(0.0) → 买入
```

### 场景5: 混合结果（不触发任何规则）

**交易历史**:
```
1. 000001.SZ: actual_return = 0.02
2. 000002.SZ: actual_return = -0.01
3. 000003.SZ: actual_return = 0.03
```

**预期输出**:
```
📊 [历史仓位控制] 默认, limit=0.0
lead=0.005 >= limit(0.0) → 买入
```

---

## 📋 修改清单 / Modification Checklist

| # | 位置 | 修改类型 | 描述 | 优先级 |
|---|------|---------|------|--------|
| 1 | 第 1275-1318 行 | 删除方法 | 删除 `_load_market_status_data()` 方法 | 🔴 高 |
| 2 | 第 1321-1378 行 | 删除方法 | 删除 `_get_position_limit_by_market_status()` 方法 | 🔴 高 |
| 3 | 第 1833 行 | 删除调用 | 删除 `run_backtest` 中的市场状态加载调用 | 🔴 高 |
| 4 | 第 1320 行后 | 新增方法 | 添加 `_is_last_trade_stop_loss()` 方法 | 🔴 高 |
| 5 | 第 1320 行后 | 新增方法 | 添加 `_has_consecutive_losses(n)` 方法 | 🔴 高 |
| 6 | 第 1320 行后 | 新增方法 | 添加 `_has_consecutive_profits(n)` 方法 | 🔴 高 |
| 7 | 第 1320 行后 | 新增方法 | 添加 `_get_position_limit_by_trade_history()` 方法 | 🔴 高 |
| 8 | 第 1381-1410 行 | 修改方法 | 更新 `_position_pct_by_lead()` 方法实现 | 🔴 高 |
| 9 | 第 2007 行 | 更新调用 | 移除 `trade_date` 参数（可选，保留也可以） | 🟡 中 |

---

## 🎨 代码流程图 / Code Flow Diagram

### 修改后的调用流程

```
run_backtest(trading_dates)
    │
    └─→ for trade_date in trading_dates:
            │
            ├─→ ... 买入逻辑 ...
            │
            ├─→ lead_vs_top5mean = ...
            │
            ├─→ _position_pct_by_lead(lead_vs_top5mean, trade_date)
            │   │
            │   ├─→ _get_position_limit_by_trade_history()  🆕 基于历史判断
            │   │   │
            │   │   ├─→ _is_last_trade_stop_loss()  🆕 检查上次止损
            │   │   │   └─→ return True/False
            │   │   │
            │   │   ├─→ _has_consecutive_losses(3)  🆕 检查连续亏损
            │   │   │   └─→ return True/False
            │   │   │
            │   │   ├─→ _has_consecutive_profits(2)  🆕 检查连续盈利
            │   │   │   └─→ return True/False
            │   │   │
            │   │   └─→ return limit (0.0 或 0.01)
            │   │
            │   └─→ return 1.0 if lead >= limit else 0.0
            │
            └─→ execute_buy() ... execute_sell() → self.trades.append()
```

---

## ⚠️ 注意事项 / Important Notes

### 1. 交易历史可用性

**首次运行**（`self.trades` 为空）:
- 所有方法返回 False
- 使用默认 `limit = 0.0`
- 行为正常，不影响首次交易

### 2. 止损阈值对齐

**问题**: 如何确保止损阈值与实际使用的一致？

**解决**:
- 使用 `self.config.stop_loss_threshold` 而不是硬编码 `-0.05`
- 修改 `_is_last_trade_stop_loss` 方法:

```python
def _is_last_trade_stop_loss(self) -> bool:
    """检查上一次交易是否触发止损"""
    if not self.trades:
        return False

    last_trade = self.trades[-1]
    actual_return = last_trade.get('actual_return', 0.0)

    # 使用配置的止损阈值
    if actual_return <= self.config.stop_loss_threshold:
        logger.info(f"🛡️ [止损检测] 上次交易触发{self.config.stop_loss_threshold*100:.1f}%止损: "
                   f"{last_trade['stock_code']}, return={actual_return:.4f}")
        return True

    # 同时检查退出原因
    exit_reason = last_trade.get('exit_reason', '')
    if '止损' in exit_reason or 'stop_loss' in exit_reason.lower():
        logger.info(f"🛡️ [止损检测] 上次交易触发止损(原因): {last_trade['stock_code']}, reason={exit_reason}")
        return True

    return False
```

### 3. 性能影响

**影响**: 每次买入都需要检查历史交易

**评估**:
- 历史交易数量通常 ≤ 1000
- 列表切片和遍历开销很小
- 性能影响可忽略

### 4. 向后兼容性

**兼容性**: 移除 `trade_date` 参数的影响

**建议**: 保留 `trade_date` 参数但标记为可选（`Optional[datetime.date] = None`）

**好处**:
- 不破坏现有调用代码
- 便于将来扩展

---

## 🚀 实施步骤 / Implementation Steps

1. **阶段1: 清理旧代码**
   - [ ] 删除 `_load_market_status_data()` 方法
   - [ ] 删除 `_get_position_limit_by_market_status()` 方法
   - [ ] 删除 `run_backtest` 中的调用

2. **阶段2: 添加新方法**
   - [ ] 添加 `_is_last_trade_stop_loss()` 方法
   - [ ] 添加 `_has_consecutive_losses(n)` 方法
   - [ ] 添加 `_has_consecutive_profits(n)` 方法
   - [ ] 添加 `_get_position_limit_by_trade_history()` 方法

3. **阶段3: 修改主方法**
   - [ ] 更新 `_position_pct_by_lead()` 方法实现
   - [ ] 更新方法文档字符串

4. **阶段4: 测试验证**
   - [ ] 创建测试脚本
   - [ ] 测试各种交易历史场景
   - [ ] 运行完整回测验证效果

---

## 📊 预期效果 / Expected Results

### 策略行为变化

**修改前（市场状态方案）**:
- 基于创业板指 MA20 判断
- 与策略实际表现脱节

**修改后（历史交易方案）**:
- 基于策略自身历史表现
- 自适应调整仓位控制

### 回测指标预期变化

| 场景 | 旧方案 | 新方案 | 改进 |
|------|-------|-------|------|
| 连续亏损后 | 仍按原limit | 提高limit→减少买入 | 降低风险 |
| 连续盈利后 | 仍按原limit | 降低limit→增加买入 | 提高收益 |
| 触发止损后 | 仍按原limit | 提高limit→减少买入 | 防止亏损扩大 |
| 正常交易 | 根据大盘MA20 | 根据自身表现 | 更贴合策略 |

---

**文档版本**: v1.0
**最后更新**: 2026-02-14
**状态**: ✅ 设计完成，待实施
