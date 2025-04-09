# Advanced Quant Data 使用指南

## 1. 模块概述

`advanced_quant_data.py` 是一个高级量化数据处理工具，用于快速获取、缓存和操作股票市场数据。该模块提供了一套完整的接口，用于管理股票列表、交易日历、日线数据和分钟级数据，支持数据的下载、存储、查询和更新等操作。

主要特点：
- 支持股票基础信息管理
- 支持交易日历管理
- 支持日线数据下载与处理
- 支持5分钟级别数据下载与处理
- 提供与 `xtdata` 类似的市场数据接口
- 数据本地缓存，优化性能
- 多线程并行下载支持

## 2. 主要功能与接口

### 2.1 数据初始化与管理

#### 初始化数据存储

```python
from to.advanced_quant_data import AdvancedQuantData

# 创建数据管理实例
storage = AdvancedQuantData()

# 下载所有基础数据
from to.advanced_quant_data import download_all_data
download_all_data(storage, start_date='20230101', end_date='20240331')
```

#### 下载股票列表

```python
# 单独下载/更新股票列表
storage.download_stock_list()

# 获取股票列表
stocks = storage.metadata['stock_list']
```

#### 下载交易日历

```python
# 单独下载/更新交易日历
storage.download_trading_calendar(start_date='20230101', end_date='20240331')

# 获取交易日历
calendar = storage.get_trading_calendar(start_date='20230101', end_date='20240331')
```

### 2.2 日线数据操作

#### 下载日线数据

```python
# 下载指定日期范围内的日线数据
storage.download_daily_data(
    stock_list=['000001.SZ', '600519.SH'],  # 可选，None表示所有股票
    start_date='20230101',
    end_date='20240331',
    batch_size=100,  # 每批处理的股票数
    max_workers=5    # 并行下载的线程数
)
```

#### 查询日线数据

```python
# 获取日线数据
daily_data = storage.get_daily_data(
    stock_list=['000001.SZ', '600519.SH'],  # 可选，None表示所有股票
    start_date='20230101',
    end_date='20240331',
    fields=['open', 'high', 'low', 'close', 'volume']  # 可选，None表示所有字段
)
```

### 2.3 分钟数据操作

#### 下载5分钟数据

```python
# 下载指定日期范围内的5分钟数据
storage.download_minute_data(
    stock_list=['000001.SZ', '600519.SH'],  # 可选，None表示所有股票
    start_date='20230101',
    end_date='20240331',
    batch_size=50,  # 每批处理的股票数
    max_workers=5   # 并行下载的线程数
)
```

#### 查询5分钟数据

```python
# 获取5分钟数据（支持精确时间范围）
minute_data = storage.get_minute_data(
    stock_list=['000001.SZ', '600519.SH'],  # 可选，None表示所有股票
    start_date='20240318093000',  # 支持YYYYMMDD或YYYYMMDDHHMMSS格式
    end_date='20240318150000',    # 支持YYYYMMDD或YYYYMMDDHHMMSS格式
    fields=['open', 'high', 'low', 'close', 'volume']  # 可选，None表示所有字段
)
```

### 2.4 统一市场数据接口

与`xtdata.get_market_data`接口类似的市场数据获取接口：

```python
# 获取市场数据（兼容xtdata.get_market_data接口格式）
market_data = storage.get_market_data(
    field_list=['open', 'high', 'low', 'close', 'volume'],
    stock_list=['000001.SZ', '600519.SH'],
    period='1d',  # '1d'表示日线数据，'5m'表示5分钟数据
    start_time='20240101',  # 日线使用YYYYMMDD，分钟线使用YYYYMMDDHHMMSS
    end_time='20240331'
)

# 获取开盘价数据
open_prices = market_data['open']  # DataFrame，行索引为股票代码，列索引为日期/时间

# 获取时间序列
times = market_data['time']
```

## 3. 数据更新

### 3.1 定期更新所有数据

```python
# 更新所有数据
from to.advanced_quant_data import update_all_data
update_all_data(storage)
```

### 3.2 单独更新特定数据

```python
# 更新股票列表
storage.update_stock_list()

# 更新交易日历
storage.update_trading_calendar()

# 更新日线数据
storage.update_daily_data(stock_list=None)  # None表示所有股票

# 更新5分钟数据
storage.update_minute_data(stock_list=None)  # None表示所有股票
```

## 4. 数据测试与验证

### 4.1 测试数据完整性

```python
# 运行所有测试
from to.advanced_quant_data import run_all_tests
run_all_tests(storage, test_date='20240101')

# 测试交易日历
storage.test_trading_calendar()

# 测试日线数据
storage.test_daily_data(start_date='20240101', end_date='20240131')

# 测试5分钟数据
storage.test_minute_data(start_date='20240101', end_date='20240103')
```

### 4.2 测试数据查询性能

```python
# 测试日线数据查询性能
from to.advanced_quant_data import test_bulk_daily_data_performance
test_bulk_daily_data_performance(storage, stock_count=100, days_range=30)

# 测试5分钟数据查询性能
from to.advanced_quant_data import test_bulk_minute_data_performance
test_bulk_minute_data_performance(storage, stock_count=50, days_range=10)
```

## 5. 回测使用示例

以下是使用该模块进行量化回测的基本示例：

```python
from to.advanced_quant_data import AdvancedQuantData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 初始化数据存储
storage = AdvancedQuantData()

# 设置回测参数
start_date = '20230101'
end_date = '20231231'
stock_list = ['000001.SZ', '600519.SH', '300059.SZ']

# 获取回测数据
market_data = storage.get_market_data(
    field_list=['open', 'high', 'low', 'close', 'volume'],
    stock_list=stock_list,
    period='1d',  # 可以是'1d'或'5m'，取决于回测精度需求
    start_time=start_date,
    end_time=end_date
)

# 计算技术指标（示例：20日均线）
close_prices = market_data['close']
ma20 = close_prices.rolling(window=20, axis=1).mean()

# 生成交易信号（示例：金叉死叉策略）
ma5 = close_prices.rolling(window=5, axis=1).mean()
signals = pd.DataFrame(0, index=ma5.index, columns=ma5.columns)
signals[ma5 > ma20] = 1  # 买入信号
signals[ma5 < ma20] = -1  # 卖出信号

# 模拟交易（简化示例）
positions = signals.shift(1, axis=1).fillna(0)  # 下一个交易日执行
returns = (positions * close_prices.pct_change(axis=1)).fillna(0)

# 计算累计收益
cumulative_returns = (1 + returns).cumprod(axis=1) - 1

# 打印回测结果
print(f"总收益率: {cumulative_returns.iloc[:, -1].mean():.2%}")
```

## 6. 其他技巧与注意事项

### 6.1 数据存储路径

数据默认存储在以下目录：
- 股票列表：`/to/local_data/metadata/`
- 交易日历：`/to/local_data/calendar/`
- 日线数据：`/to/local_data/daily/`
- 5分钟数据：`/to/local_data/minute/`

### 6.2 数据分区

日线和分钟数据按年份-月份进行分区存储，例如：
- 2023年1月数据：`/to/local_data/daily/202301/`
- 2023年2月数据：`/to/local_data/minute/202302/`

### 6.3 性能优化建议

- 使用`batch_size`和`max_workers`参数调整下载性能
- 按需下载数据，避免下载不必要的数据
- 对于回测，优先选择必要的股票和时间范围
- 使用`fields`参数仅获取需要的字段，减少内存占用

### 6.4 常见问题解决

1. **数据下载失败**：检查网络连接和API权限，适当增加超时时间。
2. **数据缺失**：使用`test_`系列函数检查数据完整性。
3. **查询速度慢**：考虑减少查询范围，或优化分区结构。
4. **内存占用过高**：分批处理数据，释放不再使用的数据。

## 7. 与xtdata的兼容性

本模块的`get_market_data`接口设计与`xtdata.get_market_data`保持高度兼容，可以轻松替换现有代码中的xtdata调用：

```python
# 原有代码使用xtdata
import xtquant.xtdata as xtdata
data = xtdata.get_market_data(
    field_list=['close'], 
    stock_list=['000001.SZ'], 
    period='1d',
    start_time='20230101', 
    end_time='20231231'
)

# 替换为advanced_quant_data
from to.advanced_quant_data import AdvancedQuantData
storage = AdvancedQuantData()
data = storage.get_market_data(
    field_list=['close'], 
    stock_list=['000001.SZ'], 
    period='1d',
    start_time='20230101', 
    end_time='20231231'
)
```

主要区别：
1. 本模块支持更灵活的时间范围处理
2. 数据首先从本地缓存获取，提高查询速度
3. 可以通过调用更新函数保持数据最新 