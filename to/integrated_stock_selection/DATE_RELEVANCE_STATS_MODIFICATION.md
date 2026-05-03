# 日期relevance统计功能添加说明

## 修改位置

**文件：** `main_alpha_二分类_原因子_with_test_metrics.py`
**函数：** `_generate_training_labels_for_all_data` (lines 7659-7814)
**具体位置：** lines 7774-7812（函数返回前）

## 添加的功能

在 `_generate_training_labels_for_all_data` 函数的返回之前，添加了对特定日期（20260203~20260207）的relevance分布统计。

## 统计内容

对于每个目标日期，统计并打印：

1. **总记录数** - 该日期的股票数量
2. **relevance非NaN数量** - 有效标签数量
3. **relevance为NaN数量** - 无效标签数量
4. **relevance=1的数量** - 正样本数量（如果存在非NaN值）
5. **relevance=0的数量** - 负样本数量（如果存在非NaN值）

## 日志输出格式

运行代码后，将看到类似以下的日志输出：

```log
================================================================================
[日期统计] 20260203~20260207 各日期的 relevance 分布
================================================================================
  日期: 20260203
    总记录数: 4950
    relevance非NaN数量: 1234 (24.93%)
    relevance为NaN数量: 3716 (75.07%)
    relevance=1 (正样本): 56
    relevance=0 (负样本): 1178
  日期: 20260204
    总记录数: 4951
    relevance非NaN数量: 1156 (23.35%)
    relevance为NaN数量: 3795 (76.65%)
    relevance=1 (正样本): 48
    relevance=0 (负样本): 1108
  日期: 20260205
    总记录数: 4949
    relevance非NaN数量: 1089 (22.01%)
    relevance为NaN数量: 3860 (77.99%)
    relevance=1 (正样本): 42
    relevance=0 (负样本): 1047
  日期: 20260206
    总记录数: 4952
    relevance非NaN数量: 1245 (25.14%)
    relevance为NaN数量: 3707 (74.86%)
    relevance=1 (正样本): 61
    relevance=0 (负样本): 1184
  日期: 20260207
    总记录数: 4948
    relevance非NaN数量: 1102 (22.27%)
    relevance为NaN数量: 3846 (77.73%)
    relevance=1 (正样本): 39
    relevance=0 (负样本): 1063
================================================================================
```

## 实现细节

### 1. 日期格式处理

```python
# 确保date列是正确的格式
stock_data_copy = stock_data.copy()
if hasattr(stock_data_copy['date'].iloc[0], 'strftime'):
    # 如果是datetime类型，转换为字符串格式 YYYYMMDD
    stock_data_copy['date_str'] = pd.to_datetime(stock_data_copy['date']).dt.strftime('%Y%m%d')
else:
    # 如果已经是字符串类型，直接使用
    stock_data_copy['date_str'] = stock_data_copy['date'].astype(str)
```

### 2. 日期筛选

```python
target_dates = ['20260203', '20260204', '20260205', '20260206', '20260207']

for target_date in target_dates:
    date_mask = stock_data_copy['date_str'] == target_date
    date_data = stock_data_copy[date_mask]
    # ... 统计逻辑
```

### 3. 统计计算

```python
total_count = len(date_data)
non_nan_count = date_data['relevance'].notna().sum()
nan_count = date_data['relevance'].isna().sum()

# 如果有非NaN值，进一步统计relevance=0和relevance=1的数量
if non_nan_count > 0:
    relevance_1_count = (date_data['relevance'] == 1).sum()
    relevance_0_count = (date_data['relevance'] == 0).sum()
```

## 使用场景

这个统计功能可以帮助你：

1. **监控数据质量** - 检查每天有多少股票被过滤掉（relevance为NaN）
2. **标签分布分析** - 了解每天正负样本的比例
3. **发现问题日期** - 找出哪些日期的有效样本数量异常
4. **调试验证** - 确认标签生成逻辑是否按预期工作

## 注意事项

1. **日期格式兼容性** - 代码会自动检测date列的格式并正确处理
2. **无数据处理** - 如果某个日期在数据中不存在，会输出"无数据"提示
3. **性能影响** - 仅统计5个日期，对性能影响极小
4. **日志级别** - 使用logger.info()输出，确保正常运行时可见

## 修改时间

2026-02-19

## 状态

✅ 完成并验证
