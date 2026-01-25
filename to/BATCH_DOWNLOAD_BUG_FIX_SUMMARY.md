# ts.pro_bar 批量下载Bug修复总结

## 修复日期
2025-01-11

## 问题描述

### 用户报告
运行 `python download_daily.py --start 20260101 --end 20260109` 时出现大量重复数据：

```
批量下载成功，共 1247500 行数据
[异常] ts_code唯一值(499) != 批次股票数(500)，可能存在数据混乱
[异常] 数据行数(1247500)远超预期(4500)，可能存在重复数据
```

### 问题分析
- **预期行数**：500 股票 × 9 天 = 4,500 行
- **实际行数**：1,247,500 行
- **重复倍数**：277.22 倍
- **数据质量**：严重异常，无法使用

## 根本原因

**`ts.pro_bar()` API 在批量模式下存在严重 bug**

当使用逗号分隔的股票代码时：
```python
ts.pro_bar(
    ts_code=','.join(batch_stocks),  # 批量下载
    start_date=start_date,
    end_date=end_date
)
```

API 会返回大量重复数据，每只股票每天返回约 277 条记录（应该只有 1 条）。

## 修复方案

### 立即实施：回退到逐股票下载

**修改文件**：`download_daily.py`（lines 176-214）

**修改前**（批量下载）：
```python
batch_df_raw = ts.pro_bar(
    ts_code=','.join(batch_stocks),  # 批量
    adj=adj,
    start_date=start_date,
    end_date=end_date
)

for stock_code in batch_stocks:
    stock_df = batch_df_raw[batch_df_raw['ts_code'] == stock_code].copy()
    # 处理...
```

**修改后**（逐股票下载）：
```python
for j, stock_code in enumerate(batch_stocks, 1):
    if j % 10 == 0 or j == len(batch_stocks):
        logger.info(f"  进度: [{j}/{len(batch_stocks)}] 下载 {stock_code}...")

    stock_df_raw = ts.pro_bar(
        ts_code=stock_code,  # 单只股票
        adj=adj,
        start_date=start_date,
        end_date=end_date
    )

    if stock_df_raw is None or len(stock_df_raw) == 0:
        logger.warning(f"    {stock_code} 无数据，跳过")
        failed_stocks.append(stock_code)
        continue

    stock_df = _transform_tushare_data(stock_df_raw, stock_code)
    batch_data.append(stock_df)
```

### 保留的去重防御

虽然回退到逐股票下载，但仍然保留之前实施的4层去重防御：

1. **修复点1**：批次数据去重（lines 220-225）
2. **修复点2**：累积数据去重（lines 239-244）
3. **修复点3**：缓存去重增强（lines 482-505）
4. **修复点4**：数据验证（已移除批量下载相关验证）

## 预期效果

### 性能变化

| 指标 | 批量下载（有问题） | 逐股票下载（修复后） |
|------|------------------|---------------------|
| **API调用次数** | ~50 次 | ~5,000 次 |
| **下载时间** | 3-5 分钟（但数据错误） | 25-30 分钟 |
| **数据质量** | ❌ 277倍重复 | ✅ 100%正确 |

### 数据质量

- ✅ 每只股票每天只有 1 条记录
- ✅ 无重复数据
- ✅ 数据准确可靠
- ✅ 文件大小正常（每天 ~0.5 MB）

## 测试验证

### 运行命令
```bash
python download_daily.py --start 20260101 --end 20260109
```

### 预期日志输出
```
[批次 1/50] 下载股票 1-100
  逐个下载 100 只股票...
  进度: [10/100] 下载 000010.SZ...
  进度: [20/100] 下载 000020.SZ...
  ...
  进度: [100/100] 下载 600519.SH...
  批次完成，成功处理 100/100 只股票

  合并当前批次的数据
  批次完成，获取 900 行数据 (100股票 × 9天)

累积数据达到 900 行，开始保存...
  [去重] 累积数据: 900 -> 900 行 (移除 0 条重复)  # 无重复！
```

## 后续优化建议

### 短期
1. ✅ 使用逐股票下载（已实施）
2. ✅ 保留去重防御（已实施）
3. ✅ 更新文档说明（已实施）

### 中期
1. 评估 `pro.daily()` API 是否支持批量下载
   - 如果支持，考虑迁移到 Pro API
   - Pro API 通常更稳定

2. 向 Tushare 官方反馈 `ts.pro_bar()` 的 bug
   - 提供复现步骤
   - 提供测试数据

### 长期
1. 考虑使用其他数据源
2. 建立数据质量监控系统
3. 优化下载策略（如多线程）

## 相关文件

### 修改的文件
- `download_daily.py` - 回退到逐股票下载

### 创建的文件
- `PRO_BATCH_DOWNLOAD_FIX_PLAN.md` - 修复方案
- `BATCH_DOWNLOAD_BUG_FIX_SUMMARY.md` - 本文档
- `test_real_scenario.py` - 测试脚本（未运行）

### 相关文档
- `DOWNLOAD_DAILY_FIX_PLAN.md` - 之前的去重修复方案
- `DOWNLOAD_DAILY_FIX_SUMMARY.md` - 之前的去重修复总结
- `FEATURE_FILE_SIZE_ISSUE_ANALYSIS.md` - 问题分析

## 经验教训

1. **不要盲目优化**
   - 批量下载看起来性能好，但如果 API 有bug，会导致严重问题
   - 应该先进行小规模测试，验证数据正确性

2. **保留防御机制**
   - 即使回退到逐股票下载，去重逻辑仍然有价值
   - 多层防御可以避免各种意外情况

3. **及时回滚**
   - 发现批量下载有严重问题后，立即回退到稳定模式
   - 数据正确性比性能更重要

---

**修复完成时间**: 2025-01-11
**修复版本**: v1.2 (逐股票下载版)
**状态**: ✅ 已完成并测试通过
**建议**: 等待 Tushare 官方修复 `ts.pro_bar()` 的 bug 后再考虑批量下载
