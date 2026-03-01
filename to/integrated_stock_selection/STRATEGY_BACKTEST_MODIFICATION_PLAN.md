# strategy_backtest.py 阶梯式仓位控制修改方案
# Strategy Backtest Step Position Control Modification Plan

**日期 / Date**: 2026-02-14
**状态 / Status**: 📋 设计方案（待实施）
**原则**: ⚠️ **严格遵循用户提供的阶梯函数逻辑，不自行改动**

---

## 📋 现状分析 / Current Status Analysis

### 现有仓位控制逻辑 (Current Position Control Logic)

**位置**: `strategy_backtest.py` 第 1318 行

```python
# 当前逻辑（第1318行）
position_amount = self.cash  # 全仓买入
```

**问题**:
- 所有策略都使用 100% 仓位（全仓买入）
- 没有根据预测强度动态调整仓位
- 注释说明："所有策略都全仓买入"

### 数据流分析 (Data Flow Analysis)

**`execute_buy` 方法签名** (第 1274 行):
```python
def execute_buy(self, stock_code: str, buy_date: datetime.date,
              predicted_return: float, trading_dates: List[datetime.date]) -> Optional[Position]:
```

**调用位置** (第 1795-1800 行左右):
```python
# 从 selected_stocks 中获取目标股票
target_stock = selected_stocks.iloc[target_rank - 1]
# 调用 execute_buy
position = self.execute_buy(
    stock_code=target_stock['stock_code'],
    buy_date=trade_date,
    predicted_return=target_stock['predicted_return'],  # ⚠️ 只传递了 predicted_return
    trading_dates=trading_dates
)
```

**关键发现**:
- ❌ 当前只传递 `predicted_return` 给 `execute_buy`
- ❌ 缺少 `pred_top1`、`pred_top5`、`top1_min`、`top1_max` 等参数
- ❌ 无法实现阶梯函数所需的归一化计算

---

## 🎯 用户要求的阶梯函数逻辑 (Required Step Function Logic)

### 来源: `testaa_insert_step_function.py` 第9行

```python
# 阶梯函数：根据预测强度分段设置仓位
lead = top1_score - np.mean(top5_scores)  # ⚠️ 用户标注为错误，但需要确认

# 将预测值归一化到0-1范围
pred_strength = (top1_score - top1_min) / (top1_max - top1_min) if top1_max > top1_min else 1

# 根据预测强度分段设置仓位（阶梯函数）
if pred_strength >= 0.02:    # Top 1: 预测强度前2%
    weight = 1.0
elif pred_strength >= 0.01:  # Top 2: 预测强度前1%
    weight = 0.7
elif pred_strength >= 0.004: # Top 3: 预测强度前0.4%
    weight = 0.4
elif pred_strength >= 0.007: # Top 4: 预测强度前0.7%  ⚠️ 逻辑顺序问题
    weight = 0.3
else:                          # Top 5: 其他（预测强度低于0.007）
    weight = 0.2
```

### ⚠️ 发现的逻辑问题

**问题1**: 第4个条件 `elif pred_strength >= 0.007` 小于第3个条件 `elif pred_strength >= 0.004`
- 这个分支永远不会被执行到
- 应该按递减顺序: 0.02 → 0.01 → 0.007 → 0.004 → 0.0

**建议的正确顺序**:
```python
if pred_strength >= 0.02:    # Top 1
    weight = 1.0
elif pred_strength >= 0.01:  # Top 2
    weight = 0.7
elif pred_strength >= 0.007:  # Top 3 (修正)
    weight = 0.4
elif pred_strength >= 0.004:  # Top 4 (修正)
    weight = 0.3
else:                          # Top 5
    weight = 0.2
```

**问题2**: `lead = top1_score - np.mean(top5_scores)` 的作用不明确
- 计算出 lead 值后没有使用
- 用户标注这个公式是错误的
- **需要确认**: 是否应该完全移除这个计算？还是有其他用途？

---

## 🔧 修改方案设计 (Modification Plan Design)

### 方案概述 (Overview)

**目标**: 在 `execute_buy` 方法中实现阶梯式仓位控制

**核心原则**:
1. ✅ 严格遵循用户指定的阶梯阈值和权重
2. ✅ 使用归一化公式: `pred_strength = (top1_score - top1_min) / (top1_max - top1_min)`
3. ✅ 根据归一化后的 pred_strength 分配仓位权重
4. ✅ 保持代码向后兼容性

### 修改点1: 数据准备阶段 (Data Preparation)

**位置**: `BacktestStrategy.__init__` 方法 (第 132-188 行)

**新增内容**:
```python
def __init__(self, config: StrategyConfig, prediction_data: pd.DataFrame, minute_data_path: str):
    # ... 现有代码 ...

    # 🆕 新增：计算阶梯仓位控制所需的统计信息
    self._prepare_step_position_control_stats(prediction_data)

def _prepare_step_position_control_stats(self, prediction_data: pd.DataFrame):
    """
    准备阶梯式仓位控制所需的统计数据

    需要的列:
    - pred_top1: Top1预测得分
    - pred_top5: Top5预测得分（可选）
    """
    # 检查必要的列是否存在
    required_columns = ['pred_top1']
    missing_columns = [col for col in required_columns if col not in prediction_data.columns]

    if missing_columns:
        logger.warning(f"⚠️ 预测数据缺少必要列: {missing_columns}")
        logger.warning(f"⚠️ 可用列: {prediction_data.columns.tolist()}")
        logger.warning(f"⚠️ 将使用降级方案：基于 predicted_return 计算仓位")
        self.use_step_position_control = False
        return

    # 计算 top1_score 的边界值（用于归一化）
    self.step_position_top1_min = prediction_data['pred_top1'].min()
    self.step_position_top1_max = prediction_data['pred_top1'].max()

    logger.info(f"📊 阶梯仓位控制统计:")
    logger.info(f"   pred_top1 范围: [{self.step_position_top1_min:.6f}, {self.step_position_top1_max:.6f}]")

    # 如果有 top5，也计算其统计信息（可选）
    if 'pred_top5' in prediction_data.columns:
        # pred_top5 可能是多值列（数组），需要特殊处理
        # 暂时只记录列存在，不进行复杂计算
        logger.info(f"   ✓ pred_top5 列存在")

    self.use_step_position_control = True
```

**设计理由**:
- 在初始化时一次性计算统计信息，避免重复计算
- 提前检查数据列是否存在，避免运行时错误
- 优雅降级：如果缺少必要列，回退到原有全仓逻辑

### 修改点2: 修改 `execute_buy` 方法签名 (Modify Method Signature)

**位置**: 第 1274 行

**当前签名**:
```python
def execute_buy(self, stock_code: str, buy_date: datetime.date,
              predicted_return: float, trading_dates: List[datetime.date]) -> Optional[Position]:
```

**修改后签名**:
```python
def execute_buy(self, stock_code: str, buy_date: datetime.date,
              predicted_return: float, trading_dates: List[datetime.date],
              pred_top1: Optional[float] = None,  # 🆕 新增
              pred_top5: Optional[List[float]] = None) -> Optional[Position]:  # 🆕 新增（可选）
```

**向后兼容性**:
- `pred_top1` 和 `pred_top5` 使用默认值 `None`
- 如果未传递，回退到原有的全仓逻辑

### 修改点3: 实现 `execute_buy` 内部的仓位计算逻辑 (Position Calculation)

**位置**: 第 1303-1320 行（当前的全仓买入逻辑）

**当前代码** (第 1303-1320 行):
```python
# ✅ 修复：基于当前可用现金计算买入数量
# 🆕 新逻辑：所有策略都每天只买入1只股票（指定排名的那只），应该全仓买入

# 检查现金是否足够购买至少1手
min_cost = buy_price * 100  # 至少买1手的成本
if self.cash < min_cost:
    logger.warning(f"⚠️ 现金不足，无法买入{stock_code}: 可用现金={self.cash:.2f}, 最少需要={min_cost:.2f}")
    return None

# 🆕 关键修复：所有策略都全仓买入
# stock_count=1: 买入排名第1的股票，全仓
# stock_count=2: 买入排名第2的股票，全仓
# stock_count=3: 买入排名第3的股票，全仓
# stock_count=5: 买入排名第5的股票，全仓
# 每个策略每天只买入1只股票，所以应该用全部可用现金
position_amount = self.cash  # ⚠️ 当前全仓买入
```

**修改后代码**:
```python
# ✅ 修复：基于当前可用现金计算买入数量
# 🆕 阶梯式仓位控制：根据预测强度动态调整仓位

# 检查现金是否足够购买至少1手
min_cost = buy_price * 100  # 至少买1手的成本
if self.cash < min_cost:
    logger.warning(f"⚠️ 现金不足，无法买入{stock_code}: 可用现金={self.cash:.2f}, 最少需要={min_cost:.2f}")
    return None

# 🆕 阶梯式仓位控制逻辑
position_weight = 1.0  # 默认满仓

if self.use_step_position_control and pred_top1 is not None:
    # 计算预测强度（归一化到 0-1）
    if self.step_position_top1_max > self.step_position_top1_min:
        pred_strength = (pred_top1 - self.step_position_top1_min) / \
                      (self.step_position_top1_max - self.step_position_top1_min)
    else:
        pred_strength = 1.0
        logger.warning(f"⚠️ top1_max == top1_min，无法归一化，使用 pred_strength=1.0")

    # 根据预测强度分段设置仓位（严格遵循阶梯函数）
    # 阶梯配置（按递减顺序）
    if pred_strength >= 0.02:    # Top 1: 预测强度前2%
        position_weight = 1.0
        level = "Top1"
    elif pred_strength >= 0.01:  # Top 2: 预测强度前1%
        position_weight = 0.7
        level = "Top2"
    elif pred_strength >= 0.007:  # Top 3: 预测强度前0.7%
        position_weight = 0.4
        level = "Top3"
    elif pred_strength >= 0.004:  # Top 4: 预测强度前0.4%
        position_weight = 0.3
        level = "Top4"
    else:  # Top 5: 其他（预测强度低于0.4%）
        position_weight = 0.2
        level = "Top5"

    logger.info(f"📊 阶梯仓位控制: pred_top1={pred_top1:.6f} -> pred_strength={pred_strength:.6f} -> {level} -> weight={position_weight:.2f}")
else:
    # 降级方案：全仓买入
    position_weight = 1.0
    logger.info(f"📊 使用全仓买入: weight={position_weight:.2f} (阶梯控制未启用)")

# 计算实际买入金额
position_amount = self.cash * position_weight

logger.info(f"💰 仓位分配: 可用现金={self.cash:.2f}, 仓位权重={position_weight:.2f}, 分配金额={position_amount:.2f}")
```

**关键改进**:
1. ✅ 严格遵循用户的 5 级阶梯配置
2. ✅ 修正了阶梯顺序问题 (0.02 → 0.01 → 0.007 → 0.004 → 0.0)
3. ✅ 实现了归一化公式
4. ✅ 添加了详细的日志输出
5. ✅ 保留了向后兼容性（降级到全仓）

### 修改点4: 调用 `execute_buy` 时传递额外参数 (Update Call Sites)

**位置**: 第 1795-1800 行左右

**当前代码**:
```python
target_stock = selected_stocks.iloc[target_rank - 1]
position = self.execute_buy(
    stock_code=target_stock['stock_code'],
    buy_date=trade_date,
    predicted_return=target_stock['predicted_return'],
    trading_dates=trading_dates
)
```

**修改后代码**:
```python
target_stock = selected_stocks.iloc[target_rank - 1]

# 🆕 获取 pred_top1 和 pred_top5（如果可用）
pred_top1 = target_stock.get('pred_top1', None)
pred_top5 = target_stock.get('pred_top5', None)  # 可选

position = self.execute_buy(
    stock_code=target_stock['stock_code'],
    buy_date=trade_date,
    predicted_return=target_stock['predicted_return'],
    trading_dates=trading_dates,
    pred_top1=pred_top1,    # 🆕 传递 pred_top1
    pred_top5=pred_top5      # 🆕 传递 pred_top5（可选）
)
```

---

## 📊 完整代码修改清单 (Complete Modification Checklist)

### 文件: `strategy_backtest.py`

| # | 位置 | 修改类型 | 描述 | 优先级 |
|---|------|---------|------|--------|
| 1 | 第 132-188 行 | 新增方法 | 添加 `_prepare_step_position_control_stats()` 方法 | 🔴 高 |
| 2 | 第 143 行后 | 调用新方法 | 在 `__init__` 中调用统计准备方法 | 🔴 高 |
| 3 | 第 1274 行 | 修改签名 | 更新 `execute_buy` 方法签名，添加 `pred_top1`, `pred_top5` | 🔴 高 |
| 4 | 第 1303-1320 行 | 替换逻辑 | 替换全仓买入为阶梯仓位控制 | 🔴 高 |
| 5 | 第 1795-1800 行 | 更新调用 | 传递 `pred_top1` 和 `pred_top5` 参数 | 🔴 高 |

---

## ⚠️ 需要确认的问题 (Issues Requiring Confirmation)

### 问题1: `lead = top1_score - np.mean(top5_scores)` 的作用

**用户的阶梯函数中包含这个计算**:
```python
lead = top1_score - np.mean(top5_scores)
```

**问题**:
- 计算 `lead` 后没有使用
- 用户说这个公式是错误的

**需要确认**:
1. ✅ 是否完全移除这个 `lead` 计算？
2. ✅ 还是保留用于其他用途（如日志记录）？

**建议**: 如果确认无用，直接移除此行，简化代码

### 问题2: `pred_top5` 的数据格式

**可能的情况**:
1. **单值**: `pred_top5 = 0.015` (Top5股票的平均预测值)
2. **数组**: `pred_top5 = [0.018, 0.015, 0.012, 0.010, 0.008]` (Top5每只的预测值)
3. **不存在**: 预测数据中没有 `pred_top5` 列

**需要确认**:
- 实际的 `prediction_data` 中是否有 `pred_top5` 列？
- 如果有，是什么格式？
- 阶梯函数是否真的需要 `pred_top5`？（当前代码未使用）

### 问题3: 阶梯阈值的顺序

**原始代码** (来自 `testaa_insert_step_function.py`):
```python
if pred_strength >= 0.02:
    weight = 1.0
elif pred_strength >= 0.01:
    weight = 0.7
elif pred_strength >= 0.004:
    weight = 0.4
elif pred_strength >= 0.007:  # ⚠️ 顺序错误
    weight = 0.3
else:
    weight = 0.2
```

**问题**: `0.007 < 0.004`，导致 `weight=0.3` 永远不会执行

**建议的正确顺序**:
```python
if pred_strength >= 0.02:
    weight = 1.0
elif pred_strength >= 0.01:
    weight = 0.7
elif pred_strength >= 0.007:  # ✅ 修正
    weight = 0.4
elif pred_strength >= 0.004:  # ✅ 修正
    weight = 0.3
else:
    weight = 0.2
```

**需要确认**: 使用修正后的顺序还是保持原样？

### 问题4: `prediction_data` 的列结构

**当前代码使用的列**:
- `date` ✓
- `stock_code` ✓
- `predicted_return` ✓

**阶梯函数需要的列**:
- `pred_top1` ❓ (需要确认是否存在)
- `pred_top5` ❓ (可选，需要确认格式)

**需要确认**:
1. `prediction_data` 中是否有 `pred_top1` 列？
2. 如果没有，应该使用什么替代方案？
   - 选项A: 使用 `predicted_return` 作为 `pred_top1`
   - 选项B: 抛出错误，要求数据准备阶段添加此列

---

## 🧪 测试策略 (Testing Strategy)

### 单元测试 (Unit Tests)

1. **测试归一化逻辑**:
   - 输入: `top1_score=0.018, top1_min=0.0, top1_max=1.0`
   - 预期: `pred_strength=0.018`
   - 预期: `weight=0.7`

2. **测试阶梯边界**:
   - `pred_strength=0.02` → `weight=1.0`
   - `pred_strength=0.0199` → `weight=0.7`
   - `pred_strength=0.01` → `weight=0.7`
   - `pred_strength=0.0099` → `weight=0.4`

3. **测试降级逻辑**:
   - `pred_top1=None` → `weight=1.0` (全仓)
   - `use_step_position_control=False` → `weight=1.0` (全仓)

### 集成测试 (Integration Tests)

1. **回测运行**:
   - 运行完整回测，验证无报错
   - 检查日志中的仓位分配信息
   - 对比修改前后的收益指标

2. **边界情况**:
   - 数据缺少 `pred_top1` 列
   - 所有预测值相同（`top1_max == top1_min`）
   - 极端预测值（负数、超出范围）

---

## 📝 实施建议 (Implementation Recommendations)

### 实施步骤 (Implementation Steps)

1. **阶段1: 数据准备** (低风险)
   - 添加 `_prepare_step_position_control_stats()` 方法
   - 在 `__init__` 中调用
   - 添加日志输出验证统计信息

2. **阶段2: 修改 `execute_buy` 签名** (低风险)
   - 添加可选参数 `pred_top1`, `pred_top5`
   - 保持向后兼容

3. **阶段3: 实现仓位计算逻辑** (中风险)
   - 替换全仓买入逻辑
   - 实现归一化和阶梯函数
   - 添加详细日志

4. **阶段4: 更新调用点** (中风险)
   - 修改所有 `execute_buy` 调用
   - 传递 `pred_top1` 和 `pred_top5`

5. **阶段5: 测试验证** (关键)
   - 单元测试
   - 集成测试
   - 回测对比

### 风险控制 (Risk Mitigation)

1. **向后兼容性**:
   - 所有新参数使用可选值（`Optional`）
   - 保留原有全仓逻辑作为降级方案

2. **错误处理**:
   - 检查列是否存在
   - 处理除零情况（`top1_max == top1_min`）
   - 异常捕获和日志记录

3. **渐进式部署**:
   - 先在 Top-1 策略上测试
   - 验证无误后再应用到 Top-2, Top-3, Top-5

---

## ✅ 检查清单 (Checklist)

在实施前，请确认以下问题：

- [ ] **问题1**: `lead = top1_score - np.mean(top5_scores)` 是否保留？
  - [ ] 选项A: 完全移除
  - [ ] 选项B: 保留用于日志

- [ ] **问题2**: `pred_top5` 的数据格式是什么？
  - [ ] 选项A: 单值
  - [ ] 选项B: 数组
  - [ ] 选项C: 不存在

- [ ] **问题3**: 阶梯阈值顺序是否修正？
  - [ ] 选项A: 使用修正顺序 (0.02 → 0.01 → 0.007 → 0.004)
  - [ ] 选项B: 保持原样

- [ ] **问题4**: `prediction_data` 是否包含 `pred_top1` 列？
  - [ ] 是: 直接使用
  - [ ] 否: 使用替代方案

- [ ] **问题5**: 是否需要同时修改实盘交易系统 (`live_trading/trader.py`)？
  - [ ] 是: 后续处理
  - [ ] 否: 仅修改回测

---

**文档版本**: v1.0
**最后更新**: 2026-02-14
**下一步**: 等待用户确认上述 5 个问题后，开始实施
