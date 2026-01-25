# Run Stock Selection Backtest 修复报告

## 问题描述

运行 Walk-Forward 模式后发现以下问题：

1. **显示格式问题**：
   - 日期显示为 `<12` 而不是正常日期格式
   - 收益率格式中 `%` 符号与数值间距过大
   - 显示了不需要的股票名称列

2. **硬编码问题**：
   - 只显示选股 1、2、3 的情况，缺少选股 5 的情况
   - 代码中硬编码了 `[1, 2, 3]` 而不是使用传入的 `stock_counts` 参数

3. **数据保存问题**：
   - `portfolio_values` 列表前 4 个值都是 1.0
   - 保存的 JSON 文件结构不够清晰，不易阅读

## 修复方案

### 1. 修复显示格式

**修改位置**：`_display_daily_stock_selections` 函数（第12647-12660行）

```python
# 修复前
logger.info(f"{'股票代码':<10} {'股票名称':<12} {'买入日期':<12} {'卖出日期':<12} {'买入价格':<8} {'卖出价格':<8} {'收益率':<8} {'持仓天数':<6}")
logger.info(f"{trade['stock_code']:<10} {trade['stock_name']:<12} {trade['buy_date']:<12} {trade['sell_date']:<12} "
           f"{trade['buy_price']:<8.2f} {trade['sell_price']:<8.2f} {net_return_pct:<8.2f}% {trade['holding_days']:<6}")

# 修复后
logger.info(f"{'股票代码':<10} {'买入日期':<12} {'卖出日期':<12} {'买入价格':<8} {'卖出价格':<8} {'收益率':<10} {'持仓天数':<6}")
buy_date_str = pd.to_datetime(trade['buy_date']).strftime('%Y-%m-%d') if trade['buy_date'] else 'N/A'
sell_date_str = pd.to_datetime(trade['sell_date']).strftime('%Y-%m-%d') if trade['sell_date'] else 'N/A'
logger.info(f"{trade['stock_code']:<10} {buy_date_str:<12} {sell_date_str:<12} "
           f"{trade['buy_price']:<8.2f} {trade['sell_price']:<8.2f} {net_return_pct:>7.2f}% {trade['holding_days']:<6}")
```

**改进**：
- 删除了股票名称列
- 使用 `pd.to_datetime().strftime()` 格式化日期显示
- 调整了收益率列宽和对齐方式，使 `%` 紧贴数值

### 2. 修复硬编码问题

**修改位置**：`_display_daily_stock_selections` 函数（第12641-12642行和12693行）

```python
# 修复前
for stock_count in [1, 2, 3]:
# ...
for stock_count in [1, 2, 3]:

# 修复后
# 初始化时使用空字典
selections_by_count = {}

# 动态获取所有存在的档位
available_stock_counts = sorted([k for k in selections_by_count.keys() if selections_by_count[k]])
for stock_count in available_stock_counts:
```

**改进**：
- 使用动态获取的 `available_stock_counts` 列表
- 遍历 `window_selections` 中的所有键，支持 `[1, 2, 3, 5]` 的任意组合

### 3. 分析 portfolio_values 前4个值为1.0的原因

**分析结果**：
这是**正确的**行为，原因如下：

1. **第0天**：初始值 `portfolio_values = [1.0]`
2. **第1天**：`holding_days=3`，持仓期未满，无到期股票，净值保持1.0
3. **第2天**：同理，净值保持1.0
4. **第3天**：同理，净值保持1.0

从第4天（`date_idx >= holding_days`）开始，才会有第一批持仓到期并计算收益。

### 4. 优化保存数据结构

**修改位置**：`_display_daily_stock_selections` 函数（第12691-12753行）

```python
# 新的数据结构
detailed_data = {
    "summary": {
        "total_trades_by_count": {
            "top_1": 62,
            "top_2": 124,
            "top_3": 186,
            "top_5": 310
        },
        "trading_windows": 3,
        "date_range": {
            "start": "2024-01-01",
            "end": "2024-12-31"
        }
    },
    "trades_by_stock_count": {
        "top_1": {
            "statistics": {
                "total_trades": 62,
                "profitable_trades": 28,
                "win_rate": 45.16,
                "avg_return": -1.23,
                "max_profit": 15.67,
                "max_loss": -9.63
            },
            "daily_transactions": {
                "Window_1": [
                    {
                        "date": "2024-01-02",
                        "stock_code": "002189.SZ",
                        "buy_price": 29.88,
                        "sell_price": 27.05,
                        "return_pct": -9.63,
                        "holding_days": 3
                    }
                ]
            }
        }
    }
}
```

**改进**：
- 添加了汇总统计信息
- 按窗口组织交易数据
- 收益率以百分比形式保存
- 价格保留2位小数
- 更清晰的层次结构

## 修复效果

1. **显示格式改进**：
   - ✅ 日期正常显示为 YYYY-MM-DD 格式
   - ✅ 收益率格式紧凑，`%` 紧贴数值
   - ✅ 删除了不需要的股票名称列

2. **动态支持所有档位**：
   - ✅ 自动识别并显示所有存在的选股数量（1, 2, 3, 5）
   - ✅ 不再硬编码，支持任意 `stock_counts` 配置

3. **数据保存优化**：
   - ✅ JSON 文件结构清晰，易于阅读
   - ✅ 包含完整的统计信息
   - ✅ 按窗口组织，便于分析

## 使用说明

修复后的代码：
- 自动适配 `stock_counts` 参数
- 生成更易读的日志和JSON文件
- 保持所有原有功能不变

运行命令保持不变：
```bash
python main.py --mode test_model_walkforward --window 3 --models lightgbm
```