# LightGBM 训练错误分析报告

## 📋 错误摘要

**错误信息**:
```
[LightGBM] [Fatal] Reducing `min_data_in_leaf` with `feature_pre_filter=true` may cause unexpected behaviour for features that were pre-filtered by the larger `min_data_in_leaf`.
You need to set `feature_pre_filter=false` to dynamically change the `min_data_in_leaf`.
```

**发生位置**: `main.py` 第 2902 行 `lgb.train()` 调用
**发生时机**: 参数搜索第 1 次迭代后，第 2 次或后续迭代时

---

## 🔍 根本原因分析

### 1. 代码流程分析

#### Dataset 创建（第 2867-2869 行）
```python
# 创建 Dataset 对象
train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=w_train)
val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, weight=w_val, reference=train_ds)
```

**关键点**:
- Dataset 在参数搜索循环**之前**创建
- **没有指定** `feature_pre_filter` 参数
- 因此使用 LightGBM 的**默认值** `feature_pre_filter=True`

#### 参数搜索空间（第 2774 行）
```python
'min_data_in_leaf': [100, 200, 300, 600], # 覆盖稀疏到密集
```

**关键点**:
- `min_data_in_leaf` 有 4 个候选值：`100, 200, 300, 600`
- 在 20 次迭代中会随机采样不同的值

#### 参数搜索循环（第 2893-2920 行）
```python
for i in range(SEARCH_ITER):
    # 随机采样参数
    current_params = base_params.copy()
    for key, choices in search_space.items():
        current_params[key] = random.choice(choices)
    current_params['label_gain'] = random.choice(label_gain_options)

    # 快速训练
    tmp_model = lgb.train(
        current_params,
        train_ds,  # ← 使用同一个 Dataset 对象
        num_boost_round=600,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
```

### 2. LightGBM 的 Feature Pre-filter 机制

**什么是 Feature Pre-filter?**
- LightGBM 为了加速训练，会在训练开始前过滤掉"不重要的特征"
- 过滤标准基于 `min_data_in_leaf` 参数
- 如果某个特征的大多数分裂点的样本数 < `min_data_in_leaf`，该特征会被过滤掉

**问题所在**:
1. **第 1 次迭代**:
   - 随机选择到 `min_data_in_leaf=300`
   - LightGBM 基于 300 个样本对特征进行预过滤
   - 某些特征因为样本数不足 300 被过滤掉
   - Dataset 对象内部记录了"已过滤的特征列表"

2. **第 2 次迭代**（或后续某次迭代）:
   - 随机选择到 `min_data_in_leaf=100` 或 `200`（小于 300）
   - LightGBM 检测到：`min_data_in_leaf` 减小了
   - 但是之前被过滤的特征**无法恢复**（因为 `feature_pre_filter=True`）
   - 抛出错误：`Reducing min_data_in_leaf with feature_pre_filter=true may cause unexpected behaviour`

### 3. 为什么会出现这个问题？

**设计缺陷**:
- Dataset 对象在循环外创建，**只创建一次**
- 同一个 Dataset 对象被多次训练，每次使用不同的 `min_data_in_leaf`
- `feature_pre_filter=True` 时，Dataset 会缓存第一次训练的特征过滤结果
- 后续训练如果使用更小的 `min_data_in_leaf`，会导致不一致

**为什么会随机到不同的值？**
- 参数搜索空间: `[100, 200, 300, 600]`
- 第 1 次迭代随机到 `300`
- 第 2 次迭代可能随机到 `100` 或 `200`
- 一旦出现**递减**的情况，就会报错

---

## ✅ 解决方案设计

### 方案 1: 禁用 Feature Pre-filter（推荐）

**原理**:
- 设置 `feature_pre_filter=False`
- 禁用特征预过滤，让 LightGBM 在每次训练时动态考虑所有特征
- 允许 `min_data_in_leaf` 在不同迭代之间变化

**修改位置**: 第 2868-2869 行

**修改代码**:
```python
# 创建 Dataset 对象（禁用 feature_pre_filter）
train_ds = lgb.Dataset(
    X_train,
    label=y_train,
    group=train_groups,
    weight=w_train,
    params={'feature_pre_filter': False}  # ← 新增
)
val_ds = lgb.Dataset(
    X_val,
    label=y_val,
    group=val_groups,
    weight=w_val,
    reference=train_ds,
    params={'feature_pre_filter': False}  # ← 新增
)
```

**优点**:
- ✅ 简单直接，一行代码修复
- ✅ 允许参数搜索自由探索 `min_data_in_leaf` 的所有候选值
- ✅ 不影响参数搜索的完整性和准确性

**缺点**:
- ⚠️ 每次训练会考虑所有特征，训练速度略微变慢
- ⚠️ 但对参数搜索阶段影响很小（只有 600 轮迭代）

---

### 方案 2: 每次迭代重新创建 Dataset

**原理**:
- 在参数搜索循环内部，每次迭代都创建新的 Dataset 对象
- 每个 Dataset 对象只使用一次，避免特征过滤状态冲突

**修改位置**: 第 2893-2908 行

**修改代码**:
```python
for i in range(SEARCH_ITER):
    # 随机采样参数
    current_params = base_params.copy()
    for key, choices in search_space.items():
        current_params[key] = random.choice(choices)
    current_params['label_gain'] = random.choice(label_gain_options)

    # ← 新增：每次迭代重新创建 Dataset
    iter_train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=w_train)
    iter_val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, weight=w_val)

    # 快速训练
    tmp_model = lgb.train(
        current_params,
        iter_train_ds,  # ← 使用新创建的 Dataset
        num_boost_round=600,
        valid_sets=[iter_val_ds],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
```

**优点**:
- ✅ 每次迭代都是独立的，不会有状态冲突
- ✅ 可以保留 `feature_pre_filter=True` 的性能优势

**缺点**:
- ❌ 每次迭代都要重新构建 Dataset，增加开销
- ❌ 代码复杂度增加
- ❌ 内存占用增加（虽然旧 Dataset 会被 GC 回收）

---

### 方案 3: 固定 min_data_in_leaf（不推荐）

**原理**:
- 从参数搜索空间中移除 `min_data_in_leaf`
- 使用固定值，不参与参数搜索

**修改位置**: 第 2774 行

**修改代码**:
```python
# 移除 'min_data_in_leaf': [100, 200, 300, 600],
# 或改为固定值
'min_data_in_leaf': [300],  # 只使用固定值
```

**优点**:
- ✅ 避免了特征过滤冲突

**缺点**:
- ❌ 失去了参数搜索的意义
- ❌ 可能错过更优的参数组合
- ❌ 不符合参数搜索的设计初衷

---

### 方案 4: 确保单调递增（复杂，不推荐）

**原理**:
- 确保 `min_data_in_leaf` 在搜索过程中单调递增或不变
- 从大到小或从小到大搜索

**缺点**:
- ❌ 与随机搜索矛盾
- ❌ 实现复杂
- ❌ 不推荐

---

## 📊 方案对比

| 方案 | 难度 | 性能影响 | 推荐度 | 说明 |
|------|------|---------|--------|------|
| **方案1: 禁用 pre-filter** | ⭐ 极简单 | ⚠️ 轻微 | ⭐⭐⭐⭐⭐ | 最佳方案 |
| 方案2: 重建 Dataset | ⭐⭐ 中等 | ⚠️⚠️ 中等 | ⭐⭐ | 次优方案 |
| 方案3: 固定参数 | ⭐ 极简单 | ✅ 无影响 | ⭐ | 不推荐 |
| 方案4: 单调递增 | ⭐⭐⭐⭐ 复杂 | ✅ 无影响 | ❌ | 不推荐 |

---

## 🎯 推荐实施方案

**选择**: **方案 1 - 禁用 Feature Pre-filter**

**实施步骤**:
1. 修改第 2868-2869 行的 Dataset 创建代码
2. 添加 `params={'feature_pre_filter': False}` 参数
3. 保持其他代码不变

**预期效果**:
- ✅ 错误完全消除
- ✅ 参数搜索正常完成 20 次迭代
- ✅ 可以自由探索所有 `min_data_in_leaf` 候选值
- ⚠️ 每次迭代训练时间略微增加（约 5-10%）

**测试验证**:
- 运行训练，确认没有 `Reducing min_data_in_leaf` 错误
- 检查参数搜索日志，确认 20 次迭代都成功
- 对比修复前后的训练时间（应该差异不大）

---

## 📝 附加说明

### 为什么第一次迭代成功了？

从日志看：
```
[Iter 1] New Best Top-1 Return: 5.4879 | LR: 0.02 | Leaves: 31 | MinData: 300
```

- 第 1 次迭代使用 `min_data_in_leaf=300`
- Dataset 首次训练，基于 300 进行特征预过滤
- 没有冲突，成功完成

**第二次迭代失败的原因**:
- 第 2 次迭代随机到了 `min_data_in_leaf=100` 或 `200`
- 小于 300，触发了 LightGBM 的保护机制
- 抛出错误防止训练结果不一致

### 为什么全量训练阶段可能没问题？

全量训练使用的是：
- 第 2943 行创建的 `full_ds`
- 第 2950-2956 行的训练代码
- 使用固定的 `best_params`（不会变化）
- 因此不会触发这个问题

---

## 🔧 代码修改位置总结

**文件**: `main.py`
**行号**: 2868-2869
**函数**: `_train_lightgbm_model`

**修改前**:
```python
train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=w_train)
val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, weight=w_val, reference=train_ds)
```

**修改后**:
```python
train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=w_train,
                       params={'feature_pre_filter': False})
val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, weight=w_val, reference=train_ds,
                     params={'feature_pre_filter': False})
```

---

**分析完成时间**: 2026-01-27
**推荐方案**: 方案 1 - 禁用 feature_pre_filter
**预计修复时间**: 1 分钟（只需修改 2 行代码）
