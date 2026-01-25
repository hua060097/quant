# 特征文件大小异常分析报告

## 问题概述

`20260109.parquet` 文件大小异常（125.24 MB），是 `20260108.parquet`（0.56 MB）的 **222倍**。

---

## 数据对比

| 文件 | 大小 | 行数 | 股票数 | 重复率 | 平均记录/股票 |
|------|------|------|--------|--------|---------------|
| 20260108.parquet | 0.56 MB | 4,949 | 4,949 | 0% | 1.00 ✅ |
| 20260109.parquet | 125.24 MB | 2,469,100 | 4,948 | **99.80%** ❌ | **499.02** ❌ |

---

## 根本原因

### 问题定位

经过深入分析，发现问题**不在特征生成脚本** `generate_daily_features_simple.py`，而在**数据下载脚本** `download_daily.py`。

### 数据混乱模式

对日线数据 `local_data/daily/202601/daily_data.parquet` 的分析显示：

```
20260109数据统计:
- 总行数: 2,495,600
- 股票数: 5,001
- 每只股票平均记录数: 499.02（应该只有1条！）

每只股票的唯一close价格数量:
- 平均值: 297.68
- 中位数: 317.00
- 最小值: 9
- 最大值: 483

样本数据（000001.SZ）:
stock_code       date         date_str   open  close
000001.SZ  2026-01-09   20260109   11.53  11.46
000001.SZ  2026-01-09   20260109    0.77   0.77
000001.SZ  2026-01-09   20260109    1.92   1.91
000001.SZ  2026-01-09   20260109    0.24   0.24
...（共500条，价格各不相同）
```

### 结论

**5001只股票的数据被完全混淆！每只股票的约500条记录，实际上是约500个不同股票的数据，但都被错误地标记成了同一个stock_code。**

---

## 可能的原因分析

### 疑点1: 批量下载时的数据混乱

在 `download_daily.py` 的批量处理逻辑中（lines 194-205）：

```python
for stock_code in batch_stocks:
    stock_df = batch_df_raw[batch_df_raw['ts_code'] == stock_code].copy()
    if len(stock_df) == 0:
        logger.warning(f"    {stock_code} 无数据，跳过")
        failed_stocks.append(stock_code)
        continue

    # 字段映射与转换
    stock_df = _transform_tushare_data(stock_df, stock_code)
    batch_data.append(stock_df)
```

**关键问题**：如果 `ts.pro_bar` 批量下载时返回的数据结构异常（比如所有数据的 `ts_code` 字段都是同一个值），或者过滤逻辑有问题，就会导致数据混乱。

### 疑点2: 保存时的去重逻辑失效

`PartitionCache.append_data` 方法的去重逻辑（lines 470-491）：

```python
existing_keys = set(zip(existing_df['stock_code'], existing_df['date_str']))
current_keys = set(zip(new_data['stock_code'], new_data['date_str']))

new_keys = current_keys - existing_keys
```

**问题**：如果 `new_data` 中有500条 `(stock_code='000001.SZ', date_str='20260109')` 的重复记录，`current_keys` 仍然只会有1个元素（因为set会去重），导致去重逻辑失效。

---

## 特征文件放大的原因

特征文件 `20260109.parquet` 的数据来自 `generate_daily_features_simple.py`：

1. **数据加载** (line 247)：
   ```python
   df_all = self.storage.get_daily_data(stock_list, load_start_date, load_end_date)
   ```
   这里加载了包含500倍重复数据的日线数据

2. **合并涨跌停数据** (line 266)：
   ```python
   df_all = df_all.merge(limit_data[cols_to_use], on=['date', 'stock_code'], how='left')
   ```
   日线数据的重复被保留（涨跌停数据本身没有重复）

3. **保存特征** (lines 298-308)：
   ```python
   for date_str, group in df_all.groupby('date_str'):
       save_df = group[valid_cols].copy()
       save_df.to_parquet(output_path, index=False)
   ```
   重复数据被原样保存到特征文件

最终结果：
- 日线数据中每只股票有500条重复记录
- 特征文件继承了这个重复
- 导致文件大小膨胀222倍

---

## 为什么20260108正常？

**可能原因**：
1. 20260108的数据是在优化**之前**下载的（使用的是逐股票下载逻辑）
2. 20260109是使用批量下载优化后的逻辑下载的，触发了bug
3. 或者20260109被重复下载了多次（每次都写入了重复数据）

---

## 修复建议（仅分析，不实施修改）

用户要求"只分析，不修改"，因此不提供具体修复代码，但可以提出修复方向：

1. **验证批量下载的返回数据结构**
   - 检查 `ts.pro_bar` 批量下载时 `ts_code` 字段的唯一性
   - 确保过滤逻辑 `batch_df_raw[batch_df_raw['ts_code'] == stock_code]` 正确工作

2. **增强去重逻辑**
   - 在保存前对 `new_data` 进行去重
   - 使用 `drop_duplicates(subset=['stock_code', 'date_str'])` 确保每条记录唯一

3. **数据验证**
   - 在保存前检查 `(stock_code, date_str)` 组合的唯一性
   - 如果发现重复，记录警告日志并跳过

4. **清理脏数据**
   - 删除 `local_data/daily/202601/daily_data.parquet` 中20260109的重复数据
   - 重新下载20260109的数据

---

## 分析结论

1. **问题源头**：`download_daily.py` 的批量下载逻辑存在bug，导致5001只股票的数据被混淆
2. **影响范围**：仅影响20260109（20260108正常）
3. **文件膨胀**：从正常的0.56 MB膨胀到125.24 MB（222倍）
4. **数据质量**：99.80%的数据是重复的，每只股票有约500条重复记录

---

**生成时间**: 2025-01-11
**分析者**: Claude Code
**状态**: 分析完成，等待用户修复指示
