# Walk-Forward 模式修复报告

## 问题描述

运行 `python main.py --mode test_model_walkforward --window 3 --models lightgbm` 后发现两个问题：

1. **年化收益显示为 0.00%**：汇总表格中年化收益列全部显示为 0
2. **选股结果文件为空**：`walkforward_selections_*.json` 文件中没有详细的选股数据

## 问题分析

### 1. 年化收益计算问题

**根本原因**：
- `_print_walkforward_summary` 函数中计算年化收益时，`count_metrics` 中没有 `trading_days` 字段
- 使用了简单的线性年化方法 `total_return_net * (252 / total_trading_days)`，而不是复利年化

**具体位置**：
- 文件：`main.py`
- 函数：`_print_walkforward_summary`
- 行号：12507-12512

### 2. 选股结果保存问题

**根本原因**：
- `all_selections` 的数据结构不匹配
- `_display_daily_stock_selections` 函数期望的格式与实际存储的格式不一致

**数据流**：
1. `_run_stock_selection_backtest` 返回 `detailed_trades`，格式为 `{1: [...], 2: [...], 3: [...], 5: [...]}`
2. `all_selections.append(window_result['selections'])` 直接追加这个字典
3. 但显示函数期望的是包含窗口信息的结构

## 修复方案

### 1. 修复年化收益计算

```python
# 修复前
total_trading_days = sum(m.get('trading_days', 0) for m in count_metrics)
if total_trading_days > 0:
    annual_return = total_return_net * (252 / total_trading_days)
else:
    annual_return = 0

# 修复后
# 计算年化收益
# 获取总交易天数
if count_metrics and 'summary_stats' in count_metrics[0]:
    # 从summary_stats获取总交易日
    total_trading_days = count_metrics[0]['summary_stats'].get('total_trading_days',
                                    len(trading_dates) if 'trading_dates' in count_metrics[0]['summary_stats'] else 0)
else:
    # 计算实际交易天数
    total_trading_days = sum(m.get('total_trades', 0) * m.get('holding_days', 3) for m in count_metrics) / len(count_metrics)
    if total_trading_days > 0:
        total_trading_days = int(total_trading_days)

# 使用复利计算年化收益
if total_return_net != 0 and total_trading_days > 0:
    # 总净收益率转换为净值
    final_value = 1 + total_return_net
    # 年化收益率
    annual_return = (final_value ** (252 / total_trading_days)) - 1.0
else:
    annual_return = 0.0
```

### 2. 修复选股结果保存

```python
# 修复前
all_selections.extend(window_result['selections'])

# 修复后
all_selections.append({
    'window_name': window['name'],
    'selections': window_result['selections']  # {1: [...], 2: [...], 3: [...], 5: [...]}
})
```

### 3. 修复选股结果显示

```python
# 修复 _display_daily_stock_selections 函数以兼容新旧数据结构
for window_data in all_selections:
    # 兼容新旧数据结构
    # 新结构：window_data = {'window_name': ..., 'selections': {1: [...], 2: [...], ...}}
    # 旧结构：window_data = {1: [...], 2: [...], ...} (直接的selections字典)
    if isinstance(window_data, dict) and 'selections' in window_data:
        window_selections = window_data['selections']
        window_name = window_data.get('window_name', 'Unknown')

        logger.info(f"\n📊 窗口 {window_name} 选股记录:")
        logger.info("-" * 50)
    elif isinstance(window_data, dict) and any(k in window_data for k in [1, 2, 3, 5]):
        # 旧的数据结构，直接是 selections 字典
        window_selections = window_data
        window_name = 'Unknown Window'

        logger.info(f"\n📊 窗口 {window_name} 选股记录:")
        logger.info("-" * 50)
```

## 修复效果

1. **年化收益正确显示**：
   - 使用复利公式计算年化收益
   - 正确获取交易天数
   - 年化收益不再显示为 0.00%

2. **选股结果正确保存**：
   - 文件中包含每个窗口的选股记录
   - 数据结构清晰，包含窗口名称
   - 详细显示每日选股的股票代码、买卖日期、价格等信息

## 相关文件

- `main.py` - 主程序文件，包含所有修复代码
- `walkforward_selections_*.json` - 修复后包含完整选股数据
- `daily_selections_detail_*.json` - 新增的详细选股记录文件

## 使用说明

修复后的代码保持了向后兼容性，同时支持新的数据结构。运行命令保持不变：

```bash
python main.py --mode test_model_walkforward --window 3 --models lightgbm
```

## 验证方法

1. 检查日志输出中年化收益是否正确计算
2. 查看 `walkforward_selections_*.json` 文件是否包含选股数据
3. 查看 `daily_selections_detail_*.json` 文件是否有详细的交易记录