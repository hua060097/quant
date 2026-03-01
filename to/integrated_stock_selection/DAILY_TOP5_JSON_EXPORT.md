# 每日 Top1~Top5 JSON 导出功能

## 功能说明

在 `main_alpha原始.py` 的 `_backtest_on_window` 函数中，新增了每日 top1~top5 详细信息的 JSON 导出功能。

## 修改位置

**文件：** `main_alpha原始.py`
**函数：** `_backtest_on_window` (line 15570)
**具体位置：** lines 15878-15954

## 修改内容

### 1. 初始化字典 (line 15878-15879)

```python
# 🆕 新增：保存每日 top1~top5 详细信息
daily_top_details = {}  # 格式: {20250601: {top1: {pred: ..., real_return: ...}, top2: {...}, ...}}
```

### 2. 收集每日详细信息 (lines 15917-15937)

在每个日期循环内，收集 top1~top5 的预测值和真实收益：

```python
# 🆕 收集每日 top1~top5 详细信息
date_str = date.strftime('%Y%m%d')
daily_top_details[date_str] = {}

for k in [1, 2, 3, 4, 5]:
    if len(date_sorted_filtered) >= k:
        top_k_stock = date_sorted_filtered.iloc[k-1:k]
        if len(top_k_stock) > 0:
            pred_value = float(top_k_stock['predicted_return'].values[0])
            real_return_value = float(top_k_stock['atr_return_label'].values[0])

            # 处理 NaN 值
            if np.isnan(pred_value):
                pred_value = None
            if np.isnan(real_return_value):
                real_return_value = None

            daily_top_details[date_str][f'top{k}'] = {
                'pred': pred_value,
                'real_return': real_return_value
            }
```

### 3. 保存到 JSON 文件 (lines 15939-15954)

在日期循环结束后，保存数据到 JSON：

```python
# 🆕 保存每日 top1~top5 详细信息到 JSON
if daily_top_details:
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    window_name = window['name'].replace(' ', '_').replace('/', '_')
    json_filename = f"results/daily_top5_details_{window_name}_{timestamp}.json"

    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(daily_top_details, f, indent=2, ensure_ascii=False)
        logger.info(f"[JSON] 已保存每日 top1~top5 详细信息到: {json_filename}")
        logger.info(f"[JSON] 共保存 {len(daily_top_details)} 个交易日的数据")
    except Exception as e:
        logger.error(f"[JSON] 保存失败: {e}")
```

## JSON 文件格式

### 文件命名规则

```
daily_top5_details_{窗口名称}_{时间戳}.json
```

**示例：** `daily_top5_details_Window_3_20260219_223045.json`

### JSON 内容结构

```json
{
  "20250203": {
    "top1": {
      "pred": 0.0325,
      "real_return": 0.0456
    },
    "top2": {
      "pred": 0.0298,
      "real_return": 0.0321
    },
    "top3": {
      "pred": 0.0276,
      "real_return": 0.0189
    },
    "top4": {
      "pred": 0.0254,
      "real_return": 0.0156
    },
    "top5": {
      "pred": 0.0238,
      "real_return": 0.0098
    }
  },
  "20250204": {
    "top1": {
      "pred": 0.0289,
      "real_return": 0.0234
    },
    ...
  }
}
```

### 字段说明

- **日期键值** (如 `"20250203"`): 交易日期，格式为 `YYYYMMDD`
- **top1~top5**: 当天按预测值排序的第1~5名股票
  - **pred**: 模型预测的收益率 (`predicted_return`)
  - **real_return**: 真实的未来收益率 (`atr_return_label`)

### 特殊值处理

- **NaN 值**: 转换为 `null` (JSON 中的 null)
- **缺失的排名**: 如果某天过滤后不足5个股票，该日期不会出现在 JSON 中

## 数据来源

### 数据过滤

与原有统计逻辑保持一致：

1. 按预测值 `predicted_return` 降序排序
2. 过滤掉涨跌停股票 (`is_limit_up`, `is_limit_down`)
3. 过滤掉炸板股票 (`is_zhaban`)
4. 过滤后股票不足5个的日期跳过

### 数据字段

- **预测值**: 来自 `test_data_with_pred['predicted_return']`
  - 这是模型预测的未来收益率

- **真实收益**: 来自 `test_data_with_pred['atr_return_label']`
  - 这是原始未来收益（未做风险惩罚）
  - 与训练集和验证集的评估保持一致

## 日志输出

### 成功日志

```log
[JSON] 已保存每日 top1~top5 详细信息到: results/daily_top5_details_Window_3_20260219_223045.json
[JSON] 共保存 45 个交易日的数据
```

### 失败日志

```log
[JSON] 保存失败: [Errno 2] No such file or directory: 'results/daily_top5_details_...'
```

## 使用示例

### 读取 JSON 文件

```python
import json

# 读取 JSON 文件
with open('results/daily_top5_details_Window_3_20260219_223045.json', 'r') as f:
    daily_data = json.load(f)

# 遍历每个交易日
for date_str, tops in daily_data.items():
    print(f"日期: {date_str}")
    for top_key, values in tops.items():
        print(f"  {top_key}: pred={values['pred']:.4f}, real_return={values['real_return']:.4f}")
```

### 分析预测准确性

```python
import json
import pandas as pd

# 读取数据
with open('results/daily_top5_details_...json', 'r') as f:
    daily_data = json.load(f)

# 转换为 DataFrame 进行分析
records = []
for date_str, tops in daily_data.items():
    for top_key in ['top1', 'top2', 'top3', 'top4', 'top5']:
        if top_key in tops:
            records.append({
                'date': date_str,
                'rank': top_key,
                'pred': tops[top_key]['pred'],
                'real_return': tops[top_key]['real_return']
            })

df = pd.DataFrame(records)

# 计算预测误差
df['pred_error'] = df['pred'] - df['real_return']
print(df.groupby('rank')['pred_error'].describe())
```

## 技术细节

### 性能考虑

1. **内存占用**: 每个交易日约 200 字节，100 个交易日约 20 KB
2. **序列化时间**: 使用 `json.dump` 自动处理，速度快
3. **存储位置**: 在日期循环外保存，不影响主流程性能

### 数据一致性

- 与原有的 `top_k_stats` 统计使用完全相同的数据
- 保证 JSON 中的数据与日志输出的统计完全一致

### 错误处理

- 使用 try-except 捕获保存异常
- 保存失败不影响主流程
- 错误信息记录到日志

## 验证要点

运行代码后，检查：

1. ✅ `results/` 目录下是否生成了 `daily_top5_details_*.json` 文件
2. ✅ 日志中是否输出 `[JSON] 已保存每日 top1~top5 详细信息到: ...`
3. ✅ JSON 文件格式是否正确（可使用 JSON 验证工具）
4. ✅ 日期格式是否为 `YYYYMMDD`（如 `20250203`）
5. ✅ 每个日期是否包含完整的 top1~top5
6. ✅ pred 和 real_return 值是否合理（范围约 -0.2 到 0.2）
7. ✅ NaN 值是否正确转换为 `null`

## 修改日期

2026-02-19

## 修改原因

用户需要每日 top1~top5 的详细预测值和真实收益数据，用于后续分析和可视化。
