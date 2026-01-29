# 删除 sector_data 相关代码修改方案

## 📋 背景分析

### 当前逻辑
1. **旧板块数据来源**：通过 `data_loader.load_sector_features()` 加载板块数据
2. **新板块数据来源**：通过 `extra_context['stock_to_sector']` 传入 SW2 行业映射
3. **冲突**：sector_data 加载逻辑已经不需要，但代码中仍在使用

### 新的板块数据流程
```
load_sw_sector_mapping() → extra_context['stock_to_sw_sector']
→ normalization_config['sw_sector_mapping']
→ 因子计算使用 SW 行业信息
```

---

## 🎯 修改目标

删除所有 sector_data 相关代码，使用新的 SW 行业映射系统。

---

## 📂 修改范围

### 文件清单
1. **main.py**: `run_generate_factors_mode` 函数
2. **factor_system/factor_lab/commands.py**: `generate_and_store` 函数
3. **main.py**: `_load_sector_features_basic` 函数（整函数删除）

---

## 🔧 详细修改方案

### 修改1: main.py - 删除 sector_data 加载逻辑

**位置**: 第 6400-6409 行

**删除代码**:
```python
try:
    # 修复：板块数据也需要历史窗口（最大需求20天，使用extended_start_date确保足够）
    sector_data = data_loader.load_sector_features(extended_start_date, end_date)
except Exception as exc:
    logger.warning('sector data loader raised an exception: %s', exc)
    sector_data = None

if sector_data is None or sector_data.empty:
    logger.warning('sector data missing, falling back to local cache')
    sector_data = _load_sector_features_basic(extended_start_date, end_date)
```

**替换为**:
```python
# 板块数据已改用 extra_context['stock_to_sector'] 传入，不再需要单独加载 sector_data
sector_data = None
logger.info("[INFO] 板块数据通过 extra_context['stock_to_sw_sector'] 传入，跳过单独加载")
```

---

### 修改2: main.py - 删除 sector_data 检查和打印逻辑

**位置**: 第 6432-6457 行

**删除代码**:
```python
if sector_data is None or sector_data.empty:

    logger.warning('no valid sector data, sector-related factors may be missing')
```

**原因**: sector_data 现在固定为 None，不再需要检查

---

### 修改3: main.py - 删除 sector_data 日期格式转换

**位置**: 第 6441-6445 行

**原代码**:
```python
for dataset in (stock_data, sector_data, market_data):

    if dataset is not None and not dataset.empty and 'date' in dataset.columns:

        dataset['date'] = pd.to_datetime(dataset['date'])
```

**修改为**:
```python
for dataset in (stock_data, market_data):  # 移除 sector_data

    if dataset is not None and not dataset.empty and 'date' in dataset.columns:

        dataset['date'] = pd.to_datetime(dataset['date'])
```

**原因**: sector_data 不再需要格式转换

---

### 修改4: main.py - 删除 sector_data 记录数打印

**位置**: 第 6449-6453 行

**删除代码**:
```python
if sector_data is not None and not sector_data.empty:
    print(f"? 板块记录数: {len(sector_data)}")

else:
    print('[warning] sector data missing, using stock-only factors')
```

**原因**: 不再需要打印 sector_data 记录数

---

### 修改5: main.py - 修改 generate_and_store 调用

**位置**: 第 6731-6746 行

**原代码**:
```python
pipeline_result = generate_and_store(
    factor_names=factors_to_generate,
    data=stock_data,
    start_date=start_date,
    end_date=end_date,
    store=store,
    pipeline=pipeline,
    market_data=market_data,
    sector_data=sector_data,  # ← 删除此参数
    context_extra=extra_context,
    enable_batch_save=enable_batch_save,
    batch_size=batch_size,
    normalization_config=normalization_config,
    incremental_mode=incremental_save_mode,
    add_factor_mode=add_factor_mode,
)
```

**修改为**:
```python
pipeline_result = generate_and_store(
    factor_names=factors_to_generate,
    data=stock_data,
    start_date=start_date,
    end_date=end_date,
    store=store,
    pipeline=pipeline,
    market_data=market_data,
    sector_data=None,  # ← 明确传入 None（或删除此参数，使用默认值）
    context_extra=extra_context,
    enable_batch_save=enable_batch_save,
    batch_size=batch_size,
    normalization_config=normalization_config,
    incremental_mode=incremental_save_mode,
    add_factor_mode=add_factor_mode,
)
```

**说明**:
- 选项A: 传入 `sector_data=None`（明确）
- 选项B: 删除此参数，使用函数默认值（更简洁）

---

### 修改6: commands.py - 删除从 sector_data 提取 SW 映射的逻辑

**位置1**: 第 442-453 行

**删除代码**:
```python
# 🔧 修复：确保sector_data被正确转换为SW行业映射
if sector_data is not None and 'sw_sector_mapping' not in normalization_config:
    if isinstance(sector_data, pd.DataFrame):
        if sector_data.index.name == 'stock_code' and 'SW' in sector_data.columns:
            sw_mapping = sector_data['SW'].to_dict()
            normalization_config['sw_sector_mapping'] = sw_mapping
            logger.info(f"从sector_data构建SW行业映射: {len(sw_mapping)}个股票")
        elif 'stock_code' in sector_data.columns and 'SW' in sector_data.columns:
            sw_mapping = dict(zip(sector_data['stock_code'], sector_data['SW']))
            normalization_config['sw_sector_mapping'] = sw_mapping
            logger.info(f"从sector_data构建SW行业映射: {len(sw_mapping)}个股票")
```

**位置2**: 第 827-836 行

**删除代码**:
```python
# 准备sector mapping
if sector_data is not None and normalization_config and 'sw_sector_mapping' not in normalization_config:
    if isinstance(sector_data, pd.DataFrame):
        if sector_data.index.name == 'stock_code' and 'SW' in sector_data.columns:
            sw_mapping = sector_data['SW'].to_dict()
            normalization_config['sw_sector_mapping'] = sw_mapping
            logger.info(f"从sector_data构建SW行业映射: {len(sw_mapping)}个股票")
        elif 'stock_code' in sector_data.columns and 'SW' in sector_data.columns:
            sw_mapping = dict(zip(sector_data['stock_code'], sector_data['SW']))
            normalization_config['sw_sector_mapping'] = sw_mapping
            logger.info(f"从sector_data构建SW行业映射: {len(sw_mapping)}个股票")
```

**原因**:
- SW 行业映射现在通过 `normalization_config['sw_sector_mapping']` 直接传入
- 不再需要从 sector_data 中提取

---

### 修改7: main.py - 删除 _load_sector_features_basic 函数

**位置**: 第 7089-7135 行

**删除整个函数**:
```python
def _load_sector_features_basic(start_date: str, end_date: str):
    """从本地缓存的板块文件中加载数据，供因子生成兜底使用"""
    # ... 整个函数删除
```

**原因**: 此函数已不再被调用

---

## ✅ 验证检查清单

### 功能验证

- [ ] SW 行业映射正确加载到 `extra_context['stock_to_sw_sector']`
- [ ] `normalization_config['sw_sector_mapping']` 包含正确的映射
- [ ] 行业内归一化因子（intra_sector_rank, intra_sector_zscore）正常计算
- [ ] 板块相关因子正常生成
- [ ] 无 sector_data 相关的错误日志

### 日志验证

**应该看到的日志**:
```
[INFO] 板块数据通过 extra_context['stock_to_sw_sector'] 传入，跳过单独加载
? SW申万行业映射数量: XXXX
```

**不应该看到的日志**:
```
[WARNING] sector data missing
[WARNING] no valid sector data
? 板块记录数: XXX
```

### 性能验证

- [ ] 减少了不必要的数据加载操作
- [ ] 减少了 DataFrame 格式转换开销
- [ ] 减少了 SW 映射提取的开销（直接使用传入的映射）

---

## 📝 修改优先级

### 高优先级（必须修改）
1. ✅ main.py 第 6400-6409 行：删除 sector_data 加载
2. ✅ main.py 第 6739 行：修改 generate_and_store 调用
3. ✅ commands.py 第 442-453, 827-836 行：删除 SW 映射提取逻辑

### 中优先级（建议修改）
4. ✅ main.py 第 6432-6457 行：删除 sector_data 检查
5. ✅ main.py 第 6441-6445 行：删除 sector_data 格式转换
6. ✅ main.py 第 6449-6453 行：删除 sector_data 打印

### 低优先级（清理代码）
7. ✅ main.py 第 7089-7135 行：删除 _load_sector_features_basic 函数

---

## 🎯 修改后预期效果

### 代码简洁性
- 删除约 150 行代码
- 移除冗余的 sector_data 处理逻辑
- 统一使用 SW 行业映射系统

### 可维护性
- 单一数据源：`extra_context['stock_to_sw_sector']`
- 清晰的数据流向
- 减少代码复杂度

### 性能提升
- 避免重复加载板块数据
- 避免 DataFrame 格式转换
- 避免 SW 映射的重复提取

---

## ⚠️ 风险评估

### 低风险
- sector_data 参数在 commands.py 中是 `Optional[pd.DataFrame] = None`
- 传入 None 符合函数设计预期
- SW 行业映射已有其他来源

### 需要验证
- 确认所有板块相关因子不依赖 sector_data
- 确认行业内归一化使用正确的 SW 映射
- 确认因子生成流程完整

---

## 🔄 回滚方案

如果修改后出现问题：

1. **快速回滚**:
   ```python
   # main.py 第 6400-6409 行恢复原代码
   sector_data = data_loader.load_sector_features(extended_start_date, end_date)
   ```

2. **验证点**:
   - 检查日志中是否有 "no valid sector data" 警告
   - 检查因子计算是否正常
   - 检查行业内归一化因子是否正常

3. **诊断步骤**:
   - 打印 `extra_context['stock_to_sw_sector']` 是否存在
   - 打印 `normalization_config['sw_sector_mapping']` 是否正确
   - 检查因子定义是否依赖 sector_data

---

## 📊 修改影响范围

### 受影响的函数
1. `run_generate_factors_mode` - main.py
2. `generate_and_store` - commands.py
3. `generate_and_store_optimized` - commands.py
4. `generate_and_store_batch_processing` - commands.py
5. `_load_sector_features_basic` - main.py（删除）

### 不受影响的代码
- ✅ SW 行业映射加载（第 6507-6513 行）
- ✅ extra_context 构建逻辑
- ✅ normalization_config 构建逻辑
- ✅ 因子计算流程
- ✅ 因子存储逻辑

---

## 🚀 实施步骤

1. **备份代码**: 创建 git commit 或备份文件
2. **按优先级修改**: 先修改高优先级项目
3. **测试运行**: 运行因子生成，验证功能
4. **检查日志**: 确认无 sector_data 相关错误
5. **验证因子**: 检查板块相关因子是否正常
6. **提交代码**: 功能正常后提交

---

**方案完成时间**: 2026-01-29
**预计影响**: 清理约 150 行代码，提升代码可维护性
**风险等级**: 低（sector_data 参数已是 Optional，传入 None 符合设计）
