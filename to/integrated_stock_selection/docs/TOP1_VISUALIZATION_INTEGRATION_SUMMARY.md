# Top1策略资金收益曲线可视化集成总结

## 修改日期
2025-12-28

## 修改目的
根据用户需求，将5分钟回测的可视化从Top5策略改为Top1策略，每天一只股票，逻辑更简单。在一张图上展示总资金/净值变化曲线，标注所有买卖点和开盘风控跳过的买入。

## 修改文件
- `main.py`

## 主要修改内容

### 1. 修改回测结果获取逻辑（第9193行）

**修改前：**
```python
top5_result = result.get(5, None) if isinstance(result, dict) else None
```

**修改后：**
```python
# 修改为使用Top1策略进行可视化（每天一只股票，逻辑更简单）
top1_result = result.get(1, None) if isinstance(result, dict) else None
```

**说明：** 从获取Top5策略的回测结果改为获取Top1策略的结果。

### 2. 更新所有相关日志信息（第9195-9219行）

将所有日志中的"Top5"替换为"Top1"，包括：
- 变量名：`top5_result` → `top1_result`
- 日志信息："Top5策略数据检查" → "Top1策略数据检查"
- 成功信息："Top5策略个股交易曲线生成完成" → "Top1策略资金收益曲线生成完成"
- 警告信息："Top5策略无交易记录..." → "Top1策略无交易记录..."
- 错误信息："Top5个股曲线生成失败" → "Top1资金曲线生成失败"

### 3. 修改绘图函数调用（第9212-9216行）

**修改前：**
```python
_plot_top5_individual_stock_curves(
    top5_result=top5_result,
    window_name=window_name,
    save_path="./results/top5_individual_stocks"
)
```

**修改后：**
```python
_plot_top1_equity_curve(
    top1_result=top1_result,
    window_name=window_name,
    save_path="./results/top1_equity_curves"
)
```

**说明：** 调用新的资金收益曲线绘制函数，保存路径也相应更新。

## 新函数特性：_plot_top1_equity_curve

该函数已存在于main.py（第8884-9103行），无需新增。

### 主要功能

1. **资金/净值曲线图**（上图）
   - X轴：交易日序列（0到N-1）
   - Y轴：总资金/净值（人民币）
   - 显示初始资金线和最终收益
   - 跟随`daily_portfolio_values`的总资金变化

2. **买卖点标注**
   - **买入点**：绿色向上三角形（^），标注：
     - 交易序号和股票代码
     - 买入价格
     - 买入日期
   - **卖出点**：红色向下三角形（v），标注：
     - 交易序号和股票代码
     - 卖出价格
     - 收益率百分比
     - 退出原因（到期/风控：累计下跌超5%/连续3根阴线）
     - 卖出日期
   - **颜色编码**：盈利用浅绿色，亏损用浅红色

3. **开盘风控跳过标注**
   - 灰色叉号（X）标记
   - 标注内容：
     - 股票代码
     - 跳过原因（开盘跳空-3.2% 或 开盘跳空+5.8%）
     - 预测收益
     - 日期

4. **单笔交易收益柱状图**（下图）
   - X轴：交易序号
   - Y轴：收益率（%）
   - 绿色柱：盈利交易
   - 红色柱：亏损交易
   - 每柱标注：收益率 + 股票代码
   - 统计信息：总交易数、胜率、平均收益

5. **图表保存**
   - 文件名格式：`top1_equity_curve_[窗口名称].png`
   - 保存路径：`./results/top1_equity_curves/`
   - 分辨率：150 DPI
   - 尺寸：18x12英寸

### 数据结构要求

`top1_result`需要包含以下字段：

```python
{
    'trades': [...],                    # 交易记录列表
    'trading_dates': [...],             # 交易日列表
    'skipped_buy_attempts': [...],      # 被跳过的买入尝试
    'daily_portfolio_values': [...],    # 每日资金净值
    'initial_capital': 1000000          # 初始资金
}
```

每个trade记录需要包含：
```python
{
    'buy_datetime': datetime,           # 买入时间
    'sell_datetime': datetime,          # 卖出时间
    'stock_code': str,                  # 股票代码
    'buy_price': float,                 # 买入价格
    'sell_price': float,                # 卖出价格
    'actual_return': float,             # 实际收益率（小数）
    'exit_reason': str                  # 退出原因
}
```

每个skipped_buy_attempt记录需要包含：
```python
{
    'stock_code': str,                  # 股票代码
    'attempt_buy_date': date,           # 尝试买入日期
    'skip_reason': str,                 # 跳过原因
    'predicted_return': float,          # 预测收益
    'skip_category': str                # 跳过分类（如"开盘风控"）
}
```

## 与原Top5实现的区别

| 特性 | Top5个股曲线（原） | Top1资金曲线（新） |
|-----|------------------|------------------|
| 图表数量 | 每只股票一张图（185张） | 一张总图 |
| X轴 | 交易序号 | 交易日序列 |
| Y轴 | 累计收益率（%） | 总资金净值（元） |
| 买卖点显示 | 在个股曲线上 | 在总资金曲线上 |
| 开盘风控显示 | 无 | 灰色X标记 |
| 交易收益图 | 无 | 底部柱状图 |
| 资金跟踪 | 不跟踪总资金 | 跟踪daily_portfolio_values |

## 测试建议

运行5分钟回测后，检查：
1. 日志中显示"Top1策略数据检查: 交易记录=X, 被跳过买入=Y"
2. 日志中显示"✅ Top1策略资金收益曲线生成完成"
3. 文件`./results/top1_equity_curves/top1_equity_curve.png`已生成
4. 图表内容完整：
   - 上图：资金曲线 + 所有买卖点标注 + 跳过买入标注
   - 下图：单笔交易收益柱状图
   - 标题显示初始资金、最终资金、总收益

## 关键改进

1. **逻辑简化**：Top1每天只有一只股票，避免了Top5的多股票复杂逻辑
2. **可视化效果**：一张图展示完整的资金变化轨迹，更直观
3. **信息完整**：包含了买卖点、收益、开盘风控等所有关键信息
4. **性能优化**：只生成一张图，而非185张个股图，大幅减少IO操作

## 相关文件

- `main.py` - 主文件，包含_run_minute_level_backtest和_plot_top1_equity_curve函数
- `rqalpha_minute_datasource/strategy_backtest.py` - 回测引擎，生成trades和skipped_buy_attempts数据
- `rqalpha_minute_datasource/minute_backtest.py` - 包装器，转换结果格式

## 备注

- 修改不影响回测逻辑，仅改变可视化方式
- 所有TopK策略（1, 2, 3, 5）的回测结果依然完整返回
- 只有Top1的结果用于可视化展示
