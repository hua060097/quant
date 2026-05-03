# is_monster 权重增强修改方案

## 一、需求概述

在 `_calculate_professional_sample_weights` 函数中，根据 `is_monster` 字段设置不同的样本权重：
- **is_monster == 1**：权重为 **5**（重点关注的优质样本）
- **is_monster == 0**：权重为 **1**（普通样本）

## 二、现状分析

### 2.1 `is_monster` 字段生成位置

**文件：** `main_alpha原始_v2_parallel_validate_fixed.py`
**函数：** `_generate_training_labels_for_all_data` (line 7686)

```python
# is_monster 定义：卖出价>=10% 且最低点回撤>=-5%
is_monster = np.where((sell_price >= 0.10) & (min_low_return >= -0.05), 1.0, 0.0)
stock_data['is_monster'] = is_monster
```

### 2.2 当前权重计算流程

**调用位置：** `_train_final_model_for_window` (lines 13908-13924)

当前传递的参数：
- `train_relevance_array` - relevance标签
- `train_dates` - 训练日期
- `train_turnover_rate` - 换手率
- `train_circ_mv` - 流通市值
- `label_rank` - relevance值（用于relevance boost）
- 其他参数...

**问题：** 当前没有传递 `is_monster` 字段到权重计算函数

### 2.3 当前权重计算函数结构

**函数：** `_calculate_professional_sample_weights` (lines 1409-1592)

**当前逻辑：**
1. 基于 turnover_rate 和 circ_mv 计算基础权重
2. 应用时间衰减
3. （已注释）relevance 增强
4. （已注释）全局 winsorize 和 clip

**问题：** 没有使用 `is_monster` 字段

## 三、修改方案设计

### 3.1 方案概述

**策略：** 在权重计算的最开始阶段，根据 `is_monster` 设置基础权重，然后在此基础上应用其他权重调整。

**权重设置方式：**
- 将 `is_monster` 作为**基础权重乘数**
- is_monster == 1：基础乘数 = 5.0
- is_monster == 0：基础乘数 = 1.0
- 其他逻辑（时间衰减等）在此基础之上应用

### 3.2 详细修改步骤

#### **步骤1：修改函数签名**

**位置：** `_calculate_professional_sample_weights` 函数定义 (line 1409)

**修改前：**
```python
def _calculate_professional_sample_weights(
    y_train: np.ndarray,
    imbalance_ratio: float,
    logger,
    boundary_distances: np.ndarray = None,
    is_regression: bool = False,
    y_train_original: np.ndarray = None,
    train_dates: np.ndarray = None,
    time_decay_half_life: float = 365.0,
    reference_date: str = None,
    turnover_rate: np.ndarray = None,
    circ_mv: np.ndarray = None,
    label_rank: np.ndarray = None,
    label_boost_beta: float = 0.1,
    w_min: float = 0.8,
    w_max: float = 1.2
) -> np.ndarray:
```

**修改后：**
```python
def _calculate_professional_sample_weights(
    y_train: np.ndarray,
    imbalance_ratio: float,
    logger,
    boundary_distances: np.ndarray = None,
    is_regression: bool = False,
    y_train_original: np.ndarray = None,
    train_dates: np.ndarray = None,
    time_decay_half_life: float = 365.0,
    reference_date: str = None,
    turnover_rate: np.ndarray = None,
    circ_mv: np.ndarray = None,
    label_rank: np.ndarray = None,
    label_boost_beta: float = 0.1,
    w_min: float = 0.8,
    w_max: float = 1.2,
    is_monster: np.ndarray = None  # 🆕 新增参数
) -> np.ndarray:
```

#### **步骤2：添加 is_monster 权重逻辑**

**位置：** 函数内部，在基本统计之后，检查输入之前 (line 1446 之后)

**插入代码：**
```python
# ---------- 基本统计 ----------
logger.info("标签统计:")
logger.info(f"  min={np.nanmin(y_train):.6f}, max={np.nanmax(y_train):.6f}, mean={np.nanmean(y_train):.6f}, std={np.nanstd(y_train):.6f}")
logger.info(f"  NaN count in labels: {int(np.isnan(y_train).sum())}, Inf count: {int(np.isinf(y_train).sum())}")

# ---------- 🆕 is_monster 权重设置 ----------
# is_monster == 1: 权重乘数 5.0（优质样本）
# is_monster == 0: 权重乘数 1.0（普通样本）
monster_weight_multiplier = np.ones(n)  # 默认全部为1.0

if is_monster is not None:
    is_monster_arr = np.asarray(is_monster, dtype=float).copy()

    # 统计 is_monster 分布
    monster_count = int(np.sum(is_monster_arr == 1.0))
    normal_count = int(np.sum(is_monster_arr == 0.0))
    other_count = len(is_monster_arr) - monster_count - normal_count

    logger.info(f"[is_monster] 统计:")
    logger.info(f"  is_monster==1 (优质样本): {monster_count} ({monster_count/n*100:.2f}%)")
    logger.info(f"  is_monster==0 (普通样本): {normal_count} ({normal_count/n*100:.2f}%)")
    if other_count > 0:
        logger.warning(f"  is_monster 其他值: {other_count} (将视为普通样本)")

    # 设置权重乘数
    monster_mask = (is_monster_arr == 1.0)
    monster_weight_multiplier[monster_mask] = 5.0  # 优质样本权重为5
    # 其他样本（包括 is_monster==0）保持权重为1.0

    logger.info(f"[is_monster] 权重乘数设置:")
    logger.info(f"  优质样本权重乘数: 5.0")
    logger.info(f"  普通样本权重乘数: 1.0")
    logger.info(f"  权重乘数统计: min={monster_weight_multiplier.min():.1f}, max={monster_weight_multiplier.max():.1f}")
else:
    logger.info("[is_monster] 未提供 is_monster 字段，所有样本使用统一权重乘数 1.0")

# ---------- 检查输入 ----------
```

#### **步骤3：将 is_monster 权重整合到权重计算中**

**位置：** 在计算时间衰减后，合成权重时 (line 1539 附近)

**修改前：**
```python
w_time = w_base * decay
w_final = w_time
logger.info(f"time adjusted weight 统计: min={np.nanmin(w_time):.4f}, max={np.nanmax(w_time):.4f}, mean={np.nanmean(w_time):.4f}, NaN={int(np.isnan(w_time).sum())}")
```

**修改后：**
```python
w_time = w_base * decay
logger.info(f"time adjusted weight 统计: min={np.nanmin(w_time):.4f}, max={np.nanmax(w_time):.4f}, mean={np.nanmean(w_time):.4f}, NaN={int(np.isnan(w_time).sum())}")

# ---------- 🆕 应用 is_monster 权重乘数 ----------
w_monster = w_time * monster_weight_multiplier
logger.info(f"[is_monster] 应用权重乘数后的权重统计:")
logger.info(f"  min={np.nanmin(w_monster):.4f}, max={np.nanmax(w_monster):.4f}, mean={np.nanmean(w_monster):.4f}, NaN={int(np.isnan(w_monster).sum())}")

w_final = w_monster
```

#### **步骤4：在调用处提取和传递 is_monster**

**位置：** `_train_final_model_for_window` 函数 (line 13863 之后)

**插入代码：**
```python
# 🆕 提取换手率和流通市值（用于基础权重计算）
train_turnover_rate = None
train_circ_mv = None
val_turnover_rate = None
val_circ_mv = None

# ... (原有的 turnover_rate 和 circ_mv 提取代码保持不变) ...

# 🆕 提取 is_monster 字段（用于权重增强）
train_is_monster = None
val_is_monster = None
if 'is_monster' in train_data_valid.columns:
    train_is_monster = train_data_valid['is_monster'].values[:len(X_train_processed)]
    monster_count = int(np.sum(train_is_monster == 1.0))
    logger.info(f"[OK] 提取训练集 is_monster 字段（优质样本数: {monster_count}）")
else:
    logger.warning("[WARNING] 训练数据中未包含is_monster列，所有样本将使用统一权重")

if 'is_monster' in val_data_valid.columns:
    val_is_monster = val_data_valid['is_monster'].values[:len(X_val_processed)]
    monster_count = int(np.sum(val_is_monster == 1.0))
    logger.info(f"[OK] 提取验证集 is_monster 字段（优质样本数: {monster_count}）")
else:
    logger.warning("[WARNING] 验证数据中未包含is_monster列，所有样本将使用统一权重")
```

#### **步骤5：在函数调用时传递 is_monster 参数**

**位置：** `_train_final_model_for_window` 函数 (lines 13908-13924, 13930-13946)

**训练集调用修改：**
```python
# 计算训练集权重
unified_sample_weights = _calculate_professional_sample_weights(
    train_relevance_array,  # 使用relevance标签
    imbalance_ratio=1.0,
    logger=logger,
    boundary_distances=None,
    is_regression=True,
    y_train_original=None,
    train_dates=train_dates,
    time_decay_half_life=180.0,  # 统一使用180天
    reference_date=reference_date,  # [WRENCH] 修复: 使用训练集最新日期
    turnover_rate=train_turnover_rate,
    circ_mv=train_circ_mv,
    label_rank=train_relevance_array,  # relevance值(0-3)
    label_boost_beta=0.15,  # 统一使用0.15
    w_min=0.8,
    w_max=1.2,
    is_monster=train_is_monster  # 🆕 传递 is_monster
)
```

**验证集调用修改：**
```python
# 计算验证集权重（使用相同的参数）
unified_val_weights = _calculate_professional_sample_weights(
    val_relevance_array,  # 使用relevance标签
    imbalance_ratio=1.0,
    logger=logger,
    boundary_distances=None,
    is_regression=True,
    y_train_original=None,
    train_dates=val_dates,  # 验证集使用自己的日期
    time_decay_half_life=180.0,  # 统一使用180天
    reference_date=reference_date,  # 使用训练集最新日期作为基准
    turnover_rate=val_turnover_rate,
    circ_mv=val_circ_mv,
    label_rank=val_relevance_array,  # relevance值(0-3)
    label_boost_beta=0.15,  # 统一使用0.15
    w_min=0.8,
    w_max=1.2,
    is_monster=val_is_monster  # 🆕 传递 is_monster
)
```

## 四、修改影响分析

### 4.1 功能影响

**正面影响：**
- ✅ 重点关注优质样本（is_monster==1），提高模型对这些样本的学习
- ✅ 保持普通样本（is_monster==0）的训练
- ✅ 权重差异明确（5倍），易于理解和调试

**潜在风险：**
- ⚠️ 过度关注优质样本可能导致对普通样本的预测能力下降
- ⚠️ 需要监控验证集性能，确保没有过拟合

### 4.2 数据流影响

**数据流向：**
```
_generate_training_labels_for_all_data
  ↓ 生成 is_monster 字段
train_data / val_data (DataFrame包含is_monster列)
  ↓ 提取 is_monster.values
train_is_monster / val_is_monster (ndarray)
  ↓ 传递给
_calculate_professional_sample_weights
  ↓ 应用权重乘数
monster_weight_multiplier (1.0 或 5.0)
  ↓ 应用到最终权重
unified_sample_weights / unified_val_weights
```

### 4.3 性能影响

- **计算开销：** 极小（仅增加一个数组乘法操作）
- **内存开销：** 极小（仅增加一个长度为 n 的临时数组）
- **训练速度：** 无明显影响

## 五、日志输出示例

修改后，运行代码将看到以下日志：

```log
[OK] 提取训练集 is_monster 字段（优质样本数: 1,234）
[OK] 提取验证集 is_monster 字段（优质样本数: 456）

样本权重计算 (修复版) 开始
  总样本数: 100,000
标签统计:
  min=-0.150000, max=0.200000, mean=0.010000, std=0.080000
  NaN count in labels: 0, Inf count in labels: 0

[is_monster] 统计:
  is_monster==1 (优质样本): 1,234 (1.23%)
  is_monster==0 (普通样本): 98,766 (98.77%)
[is_monster] 权重乘数设置:
  优质样本权重乘数: 5.0
  普通样本权重乘数: 1.0
  权重乘数统计: min=1.0, max=5.0

... (其他权重计算逻辑) ...

[is_monster] 应用权重乘数后的权重统计:
  min=0.8000, max=6.0000, mean=1.0492
```

## 六、测试验证要点

### 6.1 功能验证

1. **字段提取验证**
   - [ ] 检查 train_is_monster 和 val_is_monster 是否正确提取
   - [ ] 验证数组长度与 X_train_processed 一致
   - [ ] 验证 is_monster==1 的样本数量合理

2. **权重计算验证**
   - [ ] 检查 monster_weight_multiplier 是否正确设置
   - [ ] 验证 is_monster==1 的样本权重是否为 5.0
   - [ ] 验证 is_monster==0 的样本权重是否为 1.0

3. **最终权重验证**
   - [ ] 检查 w_final 中 is_monster==1 的样本权重是否约为普通样本的 5 倍
   - [ ] 验证最终权重范围合理（0.8-6.0 之间）

### 6.2 性能验证

1. **训练集性能**
   - [ ] 检查模型对 is_monster==1 样本的预测准确性
   - [ ] 检查模型整体性能（IC, RankIC）

2. **验证集性能**
   - [ ] 检查验证集上的性能指标
   - [ ] 确保没有过拟合

### 6.3 边界情况测试

1. **is_monster 为 None**
   - [ ] 当数据中不包含 is_monster 列时，应使用统一权重
   - [ ] 日志应输出警告信息

2. **is_monster 值异常**
   - [ ] 当 is_monster 不是 0 或 1 时，应视为普通样本
   - [ ] 日志应输出警告信息

3. **长度不匹配**
   - [ ] 当 is_monster 长度与 y_train 不一致时，应自动截断
   - [ ] 日志应输出警告信息

## 七、回滚方案

如果修改后效果不佳，可以通过以下方式回滚：

1. **简单回滚：** 将 is_monster 参数设置为 None，函数将使用统一权重
2. **参数调整：** 修改权重乘数（如将 5.0 改为 3.0 或 2.0）
3. **完全回滚：** 删除所有 is_monster 相关代码

## 八、后续优化建议

1. **权重乘数可调：** 将 5.0 作为参数（monster_weight），便于调优
2. **分级权重：** 根据未来收益率设置更细粒度的权重
3. **动态权重：** 根据模型训练进度动态调整 is_monster 权重
4. **监控指标：** 增加对 is_monster 样本的专门监控指标

---

**方案设计日期：** 2026-02-19
**待实现状态：** ⏳ 等待用户确认后实施
