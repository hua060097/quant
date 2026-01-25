# download_daily.py 数据重复问题修复方案

## 问题根源分析

### 当前问题
- 20260109的数据中，每只股票有约500条重复记录
- 5,001只股票 → 2,495,600行数据（应该只有5,001行）
- 数据被混乱分配：每只股票的500条记录实际上是约500个不同股票的数据

### 根本原因推测（未确认）

**最可能的原因**：批量下载时 `ts.pro_bar` 返回的数据结构异常，导致：
1. 某些批次的 `ts_code` 字段值错误（都变成了同一个值）
2. 或者过滤逻辑 `batch_df_raw[batch_df_raw['ts_code'] == stock_code]` 失效
3. 或者累积数据 `accumulated_data` 在某处被重复追加

**次要原因**：去重逻辑无法处理 `new_data` **内部**的重复
- `PartitionCache.append_data` 只检查新数据和旧数据之间的重复
- 如果 `new_data` 本身包含重复，会被全部保存

## 修复方案

### 方案1：增强去重逻辑（核心方案）

**目标**：在数据处理的关键环节增加去重，确保即使数据混乱也能被纠正

#### 修复点1：批次数据去重（最关键）
**位置**：主循环中，合并批次数据后
**修复内容**：
```python
# 合并当前批次的数据
if batch_data:
    batch_df = pd.concat(batch_data, ignore_index=True)

    # 【修复】去重：确保每只股票每天只有一条记录
    before_dedup = len(batch_df)
    batch_df = batch_df.drop_duplicates(subset=['stock_code', 'date_str'], keep='first')
    after_dedup = len(batch_df)
    if before_dedup != after_dedup:
        logger.warning(f"  [去重] 批次数据: {before_dedup} -> {after_dedup} 行 (移除 {before_dedup - after_dedup} 条重复)")

    accumulated_data.append(batch_df)
```

#### 修复点2：累积数据去重
**位置**：保存前，合并所有累积数据后
**修复内容**：
```python
if accumulated_data:
    all_data_df = pd.concat(accumulated_data, ignore_index=True)

    # 【修复】全局去重：确保跨批次的数据也不重复
    before_dedup = len(all_data_df)
    all_data_df = all_data_df.drop_duplicates(subset=['stock_code', 'date_str'], keep='first')
    after_dedup = len(all_data_df)
    if before_dedup != after_dedup:
        logger.warning(f"  [去重] 累积数据: {before_dedup} -> {after_dedup} 行 (移除 {before_dedup - after_dedup} 条重复)")
```

#### 修复点3：缓存去重逻辑增强
**位置**：`PartitionCache.append_data` 方法
**修复内容**：
```python
def append_data(self, partition_key: str, new_data: pd.DataFrame):
    # ... 现有代码 ...

    if len(existing_df) == 0:
        # 【修复】首次追加也要去重（防止new_data内部重复）
        self.cache[partition_key] = new_data.drop_duplicates(subset=['stock_code', 'date_str'], keep='first')
    else:
        # 先对new_data去重
        new_data_dedup = new_data.drop_duplicates(subset=['stock_code', 'date_str'], keep='first')

        # 再执行现有逻辑
        existing_keys = set(zip(existing_df['stock_code'], existing_df['date_str']))
        current_keys = set(zip(new_data_dedup['stock_code'], new_data_dedup['date_str']))
        # ...
```

### 方案2：增加数据验证（辅助方案）

**目标**：在数据处理的关键点增加检查，提前发现问题

#### 修复点4：批量下载后验证
**位置**：批量下载完成后
**修复内容**：
```python
batch_df_raw = ts.pro_bar(...)

if batch_df_raw is not None and len(batch_df_raw) > 0:
    # 【验证】检查ts_code字段的唯一性
    unique_codes = batch_df_raw['ts_code'].nunique()
    if unique_codes != len(batch_stocks):
        logger.warning(f"  [异常] ts_code唯一值({unique_codes}) != 批次股票数({len(batch_stocks)})")

    # 【验证】检查数据行数是否合理
    expected_rows = len(batch_stocks)  # 假设只下载1天
    if len(batch_df_raw) > expected_rows * 2:  # 允许一定冗余
        logger.warning(f"  [异常] 数据行数({len(batch_df_raw)})远超预期({expected_rows})")
```

#### 修复点5：日志增强
**位置**：各个关键处理点
**修复内容**：
- 记录每批次的去重情况
- 记录每只股票的处理结果
- 记录数据保存前后的行数变化

## 修复策略

### 优先级
1. **P0（必须）**：修复点1、2 - 批次和累积数据去重
2. **P1（强烈建议）**：修复点3 - 缓存去重增强
3. **P2（建议）**：修复点4、5 - 数据验证和日志增强

### 实施步骤
1. 实施修复点1、2（最关键，能直接解决问题）
2. 实施修复点3（增强防御）
3. 实施修复点4、5（监控和诊断）
4. 清理20260109的脏数据
5. 重新下载20260109验证修复效果

## 预期效果

### 修复前
- 20260109: 2,495,600行（500倍重复）
- 文件大小: 约125 MB

### 修复后
- 20260109: 约5,001行（正常）
- 文件大小: 约0.5 MB
- 特征文件: 约0.56 MB（与20260108一致）

## 风险评估

### 低风险
- 去重操作使用 `drop_duplicates(subset=['stock_code', 'date_str'], keep='first')`
- 只保留第一条记录，不会丢失唯一数据
- 即使数据混乱，也能保证每只股票每天只有一条记录

### 注意事项
- 如果 `ts.pro_bar` 确实返回了错误数据，去重后可能保留了错误的记录
- 建议修复后重新下载数据，而不是依赖去重

## 后续行动

1. 立即实施P0修复（修复点1、2）
2. 清理20260109的脏数据
3. 重新下载20260109
4. 验证数据质量
5. 如果确认 `ts.pro_bar` 批量下载有问题，考虑回退到逐股票下载
