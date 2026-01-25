# RQAlpha 5分钟数据源和选股回测系统

基于RQAlpha框架扩展的5分钟K线数据源和选股回测系统，支持高频交易策略回测。

## 🎯 功能特点

### ✅ 已实现功能
- **5分钟K线数据源**: 基于本地parquet文件的高性能数据访问
- **多时间频率支持**: 1m, 5m, 15m, 30m, 60m等多种时间周期
- **Top-K选股策略**: K=1,2,3,5多档位选股回测
- **完整的选股过滤**: 复用原有的技术指标和风控条件
- **可配置持仓时间**: 支持15分钟、30分钟、60分钟等多种持仓周期
- **向后兼容**: 保持原有日线回测功能不变
- **性能优化**: 数据缓存和索引优化，支持大规模回测

### 🔄 与原有系统的关系
- **完全兼容**: 现有的日线回测逻辑保持不变
- **无缝切换**: 通过`frequency`参数在日线和5分钟间切换
- **数据复用**: 使用相同的预测数据和技术指标
- **策略一致**: 选股逻辑和过滤条件完全复用

## 📁 项目结构

```
rqalpha_minute_datasource/
├── __init__.py                 # 包初始化文件
├── minute_data_source.py       # 5分钟数据源实现
├── mod.py                      # RQAlpha Mod集成
├── minute_backtest.py          # 5分钟选股回测逻辑
├── config.yaml                 # 配置文件
├── example_minute_backtest.py  # 使用示例
└── README.md                   # 说明文档
```

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的依赖：
```bash
pip install pandas numpy pyarrow rqalpha
```

### 2. 数据准备

将5分钟K线数据放置在`../local_data/minute/`目录下：
```
../local_data/minute/
├── 202409/
│   └── minute_data.parquet
├── 202410/
│   └── minute_data.parquet
├── 202411/
│   └── minute_data.parquet
└── ...
```

数据格式要求：
- 字段：`stock_code`, `datetime`, `open`, `high`, `low`, `close`, `volume`, `amount`
- 时间：精确到分钟，如 `2024-10-08 09:35:00`

### 3. 基本使用

#### 日线回测（原有功能）
```python
from main import _run_stock_selection_backtest

# 日线回测
result = _run_stock_selection_backtest(
    backtest_data=prediction_data,
    stock_counts=[1, 2, 3, 5],
    window_name="daily_test",
    frequency='daily'  # 可省略，默认为日线
)
```

#### 5分钟回测（新功能）
```python
# 5分钟回测
result = _run_stock_selection_backtest(
    backtest_data=prediction_data,
    stock_counts=[1, 2, 3, 5],
    window_name="minute_test",
    frequency='minute',        # 指定5分钟回测
    holding_minutes=30         # 持仓30分钟
)
```

### 4. 运行示例

```bash
# 运行完整的示例演示
python rqalpha_minute_datasource/example_minute_backtest.py
```

## 📊 回测结果格式

回测结果包含以下信息：

```python
{
    'backtest_results': {
        1: {  # Top-1选股结果
            'total_trades': 150,           # 总交易次数
            'total_return': 0.0523,        # 总收益
            'annual_return': 0.1245,       # 年化收益
            'sharpe_ratio': 1.234,         # 夏普比率
            'max_drawdown': 0.0234,        # 最大回撤
            'win_rate': 0.6543,            # 胜率
            'avg_profit': 0.0023,          # 平均盈利
            'avg_loss': -0.0015            # 平均亏损
        },
        2: { ... },  # Top-2选股结果
        3: { ... },  # Top-3选股结果
        5: { ... },  # Top-5选股结果
    },
    'detailed_trades': {
        1: [  # Top-1详细交易记录
            {
                'stock_code': '000001.SZ',
                'buy_datetime': '2024-10-08 09:35:00',
                'sell_datetime': '2024-10-08 10:05:00',
                'buy_price': 12.34,
                'sell_price': 12.56,
                'predicted_return': 0.0123,
                'actual_return': 0.0178,
                'holding_minutes': 30,
                'profit_loss': 0.22,
                'profit_loss_pct': 1.78
            },
            # ... 更多交易记录
        ],
        2: [ ... ],  # Top-2详细交易记录
        # ...
    },
    'summary_stats': {
        'trading_days': 20,         # 交易日数量
        'holding_minutes': 30,       # 持仓分钟数
        'trading_cost': 0.0016      # 交易成本
    }
}
```

## ⚙️ 配置选项

### 基本配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `frequency` | str | `'daily'` | 回测频率：`'daily'`或`'minute'` |
| `holding_minutes` | int | `30` | 5分钟回测时的持仓分钟数 |
| `stock_counts` | list | `[1,2,3,5]` | 选股数量列表 |
| `data_path` | str | `'../local_data/minute/'` | 5分钟数据路径 |
| `cache_size` | int | `100` | 数据缓存的股票数量 |

### 交易成本配置

```python
# 在minute_backtest.py中可调整
SLIPPAGE_RATE = 0.001      # 滑点0.1%
COMMISSION_RATE = 0.0003   # 手续费0.03%
TRADING_COST_TOTAL = 0.0016  # 总交易成本
```

### 选股过滤条件

系统复用了原有的选股过滤逻辑，包括：

- **基础过滤**: 北交所、ST股、新股排除
- **价格过滤**: 价格区间限制
- **成交额过滤**: 最低成交额要求
- **技术指标过滤**: ATR、RSI、布林带等
- **涨跌停过滤**: 排除涨停跌停股票
- **量比过滤**: 成交量比例要求

## 🔧 高级用法

### 1. 自定义持仓时间

```python
# 不同持仓时间测试
holding_periods = [15, 30, 60, 120]  # 15分钟到2小时

for minutes in holding_periods:
    result = _run_stock_selection_backtest(
        backtest_data=prediction_data,
        stock_counts=[1, 3, 5],
        frequency='minute',
        holding_minutes=minutes
    )
    print(f"持仓{minutes}分钟 - Top-3收益: {result['backtest_results'][3]['total_return']:.4f}")
```

### 2. 多频率对比

```python
# 日线 vs 5分钟对比
daily_result = _run_stock_selection_backtest(
    backtest_data=prediction_data,
    stock_counts=[3],
    frequency='daily'
)

minute_result = _run_stock_selection_backtest(
    backtest_data=prediction_data,
    stock_counts=[3],
    frequency='minute',
    holding_minutes=30
)

print(f"日线回测收益: {daily_result['backtest_results'][3]['total_return']:.4f}")
print(f"5分钟回测收益: {minute_result['backtest_results'][3]['total_return']:.4f}")
```

### 3. RQAlpha直接使用

```python
from rqalpha_minute_datasource import MinuteKDataSource

# 创建数据源
data_source = MinuteKDataSource(
    data_path="../local_data/minute/",
    cache_size=200
)

# 获取K线数据
bar_data = data_source.get_bar(instrument, datetime, '5m')

# 获取历史数据
history_data = data_source.history_bars(
    instrument, 100, '5m',
    ['open', 'high', 'low', 'close', 'volume'], datetime
)
```

## 🐛 常见问题

### Q1: 如何切换到5分钟回测？
A: 在调用`_run_stock_selection_backtest`时设置`frequency='minute'`参数。

### Q2: 5分钟数据从哪里获取？
A: 需要自行下载5分钟K线数据并转换为parquet格式，放置在指定目录下。

### Q3: 内存使用过高怎么办？
A: 可以调整`cache_size`参数来控制数据缓存大小。

### Q4: 如何修改持仓时间？
A: 通过`holding_minutes`参数设置，支持任意分钟数。

### Q5: 与原日线回测结果对比如何？
A: 可以通过运行示例中的对比功能来分析不同频率的表现。

## 📈 性能特点

- **数据访问**: 优化的parquet文件读取和缓存机制
- **内存使用**: 智能缓存策略，支持大规模数据
- **计算效率**: 向量化计算，快速回测
- **可扩展性**: 支持自定义时间频率和过滤条件

## 🔮 未来扩展

- [ ] 支持tick级数据
- [ ] 更多技术指标计算
- [ ] 实盘交易接口
- [ ] Web可视化界面
- [ ] 分布式回测支持

## 📝 更新日志

### v1.0.0 (2025-11-22)
- ✅ 初始版本发布
- ✅ 支持5分钟K线数据源
- ✅ 实现Top-K选股回测
- ✅ 完整的选股过滤逻辑
- ✅ RQAlpha Mod集成
- ✅ 配置文件和示例代码

## 📞 联系方式

如有问题或建议，请联系：Claude Code Assistant

---

**注意**: 本系统仅用于研究和回测目的，实盘交易请谨慎评估风险。