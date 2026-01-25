# download_daily.py 修复总结

## 修复日期
2025-01-11

## 问题描述
20260109的数据中每只股票有约500条重复记录，导致：
- 日线数据：2,495,600行（应该只有5,001行）
- 特征文件：2,469,100行（应该只有4,949行）
- 文件大小膨胀222倍

## 修复内容

### 修复点1：批次数据去重（P0 - 必须）
**位置**：`download_daily.py` lines 240-245
**修复内容**：在合并批次数据后，立即进行去重
```python
# [修复点1] 去重：确保每只股票每天只有一条记录
before_dedup = len(batch_df)
batch_df = batch_df.drop_duplicates(subset=['stock_code', 'date_str'], keep='first')
after_dedup = len(batch_df)
if before_dedup != after_dedup:
    logger.warning(f"  [去重] 批次数据: {before_dedup} -> {after_dedup} 行 (移除 {before_dedup - after_dedup} 条重复)")
```
**作用**：防止批次数据内部重复累积

### 修复点2：累积数据去重（P0 - 必须）
**位置**：`download_daily.py` lines 259-264
**修复内容**：在保存前，对所有累积数据进行全局去重
```python
# [修复点2] 全局去重：确保跨批次的数据也不重复
before_dedup = len(all_data_df)
all_data_df = all_data_df.drop_duplicates(subset=['stock_code', 'date_str'], keep='first')
after_dedup = len(all_data_df)
if before_dedup != after_dedup:
    logger.warning(f"  [去重] 累积数据: {before_dedup} -> {after_dedup} 行 (移除 {before_dedup - after_dedup} 条重复)")
```
**作用**：确保跨批次的数据也不重复

### 修复点3：缓存去重逻辑增强（P1 - 强烈建议）
**位置**：`download_daily.py` lines 482-505
**修复内容**：
1. 首次追加时也进行去重
2. 在增量去重前，先对new_data内部去重
```python
if len(existing_df) == 0:
    # [修复点3-1] 首次追加也要去重（防止new_data内部重复）
    before_dedup = len(new_data)
    self.cache[partition_key] = new_data.drop_duplicates(subset=['stock_code', 'date_str'], keep='first')
    after_dedup = len(self.cache[partition_key])
    if before_dedup != after_dedup:
        logger.warning(f"  [缓存去重] {partition_key}: {before_dedup} -> {after_dedup} 行 (移除 {before_dedup - after_dedup} 条重复)")
else:
    # [修复点3-2] 先对new_data去重，再进行增量去重
    new_data_dedup = new_data.drop_duplicates(subset=['stock_code', 'date_str'], keep='first')
    if len(new_data_dedup) < len(new_data):
        logger.warning(f"  [缓存去重] new_data内部去重: {len(new_data)} -> {len(new_data_dedup)} 行")
```
**作用**：增强缓存层面的去重防御

### 修复点4：数据验证（P2 - 建议）
**位置**：`download_daily.py` lines 194-209
**修复内容**：
1. 验证ts_code字段的唯一性
2. 验证数据行数是否合理
```python
# [修复点4-1] 验证ts_code字段的唯一性
unique_codes = batch_df_raw['ts_code'].nunique()
if unique_codes != len(batch_stocks):
    logger.warning(f"  [异常] ts_code唯一值({unique_codes}) != 批次股票数({len(batch_stocks)})，可能存在数据混乱")

# [修复点4-2] 验证数据行数是否合理
# 估算：每只股票的数据行数 = (结束日期 - 开始日期 + 1)
start_dt = datetime.strptime(start_date, '%Y%m%d')
end_dt = datetime.strptime(end_date, '%Y%m%d')
expected_days = (end_dt - start_dt).days + 1
expected_rows = len(batch_stocks) * expected_days
# 允许一定的冗余（比如某些股票有额外数据）
tolerance = expected_rows * 3
if len(batch_df_raw) > tolerance:
    logger.warning(f"  [异常] 数据行数({len(batch_df_raw)})远超预期({expected_rows})，可能存在重复数据")
```
**作用**：提前发现数据异常，便于诊断

## 修复策略

### 防御层次
1. **第一道防线**（修复点4）：批量下载后验证，提前发现异常
2. **第二道防线**（修复点1）：批次数据去重，防止批次内重复累积
3. **第三道防线**（修复点2）：累积数据去重，确保全局无重复
4. **第四道防线**（修复点3）：缓存去重增强，最后一道保障

### 去重原则
- 使用 `drop_duplicates(subset=['stock_code', 'date_str'], keep='first')`
- 保留第一条记录，移除后续重复
- 适用于所有去重场景，确保一致性

## 使用说明

### 立即生效
所有修复立即生效，无需重启。从现在起下载的数据会自动去重。

### 清理历史数据
使用提供的脚本清理20260109的脏数据：

```bash
# 清理20260109的重复数据
python clean_and_redownload_20260109.py

# 重新下载20260109（可选，因为清理脚本已经保留了去重后的数据）
python download_daily.py --start 20260109 --end 20260109

# 重新生成特征
python generate_daily_features_simple.py --start_date 20260109 --end_date 20260109
```

### 监控日志
下载时注意观察日志中的去重信息：
- `[去重] 批次数据: X -> Y 行 (移除 Z 条重复)`
- `[去重] 累积数据: X -> Y 行 (移除 Z 条重复)`
- `[缓存去重] new_data内部去重: X -> Y 行`

如果出现这些警告，说明数据源存在异常，去重逻辑正在工作。

## 预期效果

### 修复后下载20260109
- 日线数据：约5,001行（每只股票1条）
- 特征文件：约4,949行
- 文件大小：约0.5 MB

### 日志示例（如果有重复）
```
[批次 1/50] 下载股票 1-100
  批量下载 100 只股票...
  批量下载成功，共 500 行数据
  [异常] 数据行数(500)远超预期(100)，可能存在重复数据
  [去重] 批次数据: 500 -> 100 行 (移除 400 条重复)
  批次完成，获取 100 行数据
```

## 风险评估

### 低风险
- 去重使用 `keep='first'`，保留第一条记录
- 即使数据混乱，也能保证每只股票每天只有一条记录
- 不影响正常数据下载

### 注意事项
- 如果 `ts.pro_bar` 批量下载确实返回错误数据，去重后可能保留了错误的记录
- 建议监控日志，如果出现大量去重警告，考虑回退到逐股票下载
- 如果确认批量下载有问题，可以修改 `batch_size=1` 回退到逐股票模式

## 验证步骤

1. **清理历史数据**：
   ```bash
   python clean_and_redownload_20260109.py
   ```

2. **重新下载**（可选，因为清理脚本已去重）：
   ```bash
   python download_daily.py --start 20260109 --end 20260109
   ```

3. **重新生成特征**：
   ```bash
   python generate_daily_features_simple.py --start_date 20260109 --end_date 20260109
   ```

4. **验证修复效果**：
   ```bash
   python analyze_feature_file_size.py
   ```

5. **预期结果**：
   - 20260108.parquet: 0.56 MB, 4,949行
   - 20260109.parquet: 0.56 MB, 4,949行（修复前是125.24 MB, 2,469,100行）

## 后续优化建议

1. **短期**：监控下载日志，观察是否还有重复数据
2. **中期**：如果确认批量下载有问题，考虑回退到逐股票下载
3. **长期**：向Tushare官方反馈批量下载的bug

## 相关文件

- `download_daily.py` - 已修复的主脚本
- `clean_and_redownload_20260109.py` - 清理脏数据脚本
- `analyze_feature_file_size.py` - 分析文件大小脚本
- `DOWNLOAD_DAILY_FIX_PLAN.md` - 修复方案文档
- `FEATURE_FILE_SIZE_ISSUE_ANALYSIS.md` - 问题分析报告

---

**修复完成时间**: 2025-01-11
**修复版本**: v1.1 (增强去重版)
**状态**: 已完成修复，待验证
