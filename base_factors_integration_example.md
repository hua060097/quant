# 基础因子集成功能使用说明

## 功能概述

在 `main.py` 的 `run_test_factors_mode` 函数中新增了基础因子集成功能，允许从 `stock_data` 中提取基础因子字段，并与指定的因子组数据进行合并。

## 主要修改

### 1. `run_test_factors_mode` 函数修改

在调用 `_load_specified_factor_groups` 之前添加了基础因子字段配置：

```python
# 🚀 新增：定义基础因子字段列表
base_factor_fields = [
    # 价格相关基础因子
    'close', 'open', 'high', 'low', 'volume',
    # 可以添加更多基础因子字段，例如：
    # 'pct_chg', 'turnover_rate', 'pe_ratio', 'pb_ratio', 'market_cap'
]

# 检查stock_data中是否包含所需的基础因子字段
available_base_factors = [field for field in base_factor_fields if field in stock_data.columns]

# 加载指定因子组的数据，并传入基础因子相关参数
factor_data = _load_specified_factor_groups(
    factor_groups=specified_factor_groups, 
    start_date=start_date, 
    end_date=end_date,
    base_factor_fields=available_base_factors,
    stock_data=stock_data,
    merge_base_factors=True  # 控制是否合并基础因子
)
```

### 2. `_load_specified_factor_groups` 函数修改

#### 函数签名更新
```python
def _load_specified_factor_groups(factor_groups: List[str], start_date: str, end_date: str,
                                base_factor_fields: List[str] = None, 
                                stock_data: pd.DataFrame = None,
                                merge_base_factors: bool = False) -> pd.DataFrame:
```

#### 新增参数说明
- `base_factor_fields`: 基础因子字段列表，从stock_data中提取
- `stock_data`: 包含基础因子数据的DataFrame
- `merge_base_factors`: 是否合并基础因子数据的控制开关

#### 基础因子处理逻辑
```python
# 🚀 新增：处理基础因子数据合并
if merge_base_factors and base_factor_fields and stock_data is not None:
    # 1. 从stock_data中提取基础因子数据
    # 2. 过滤日期范围
    # 3. 检查重复列并处理
    # 4. 合并基础因子数据和因子组数据
```

## 使用方法

### 1. 配置基础因子字段

在 `run_test_factors_mode` 函数中的 `base_factor_fields` 列表中添加需要的基础因子字段：

```python
base_factor_fields = [
    'close', 'open', 'high', 'low', 'volume',  # 基础价格和成交量
    'pct_chg',                                  # 涨跌幅
    'turnover_rate',                           # 换手率
    'pe_ratio',                                # 市盈率
    'pb_ratio',                                # 市净率
    'market_cap'                               # 市值
]
```

### 2. 控制合并行为

通过 `merge_base_factors` 参数控制是否合并基础因子：

```python
# 启用基础因子合并
factor_data = _load_specified_factor_groups(
    factor_groups=specified_factor_groups,
    start_date=start_date,
    end_date=end_date,
    base_factor_fields=available_base_factors,
    stock_data=stock_data,
    merge_base_factors=True  # 启用合并
)

# 禁用基础因子合并（保持原有行为）
factor_data = _load_specified_factor_groups(
    factor_groups=specified_factor_groups,
    start_date=start_date,
    end_date=end_date,
    merge_base_factors=False  # 禁用合并
)
```

## 数据合并逻辑

### 1. 数据提取
- 从 `stock_data` 中提取指定的基础因子字段
- 自动包含 `date` 和 `stock_code` 作为合并键

### 2. 日期过滤
- 基础因子数据会根据 `start_date` 和 `end_date` 进行过滤
- 确保数据范围与因子组数据一致

### 3. 重复列处理
- 自动检测基础因子数据和因子组数据中的重复列
- 优先保留因子组数据中的值，从基础因子数据中移除重复列

### 4. 数据合并
- 使用 `left join` 方式合并，以因子组数据为主
- 合并键为 `['date', 'stock_code']`

## 日志输出示例

```
🎯 指定因子组: ['kline_pattern_factors']
📊 基础因子字段: ['close', 'open', 'high', 'low', 'volume']
✅ 可用基础因子字段: ['close', 'open', 'high', 'low', 'volume']
开始加载指定因子组: ['kline_pattern_factors']
🔧 开始合并基础因子数据...
  基础因子数据: 15000 条记录, 7 个字段
  基础因子合并完成: 15000 -> 12000 条记录 (原12000条)
  合并后总字段数: 25
  新增基础因子字段: ['close', 'open', 'high', 'low', 'volume']
```

## 注意事项

1. **字段检查**: 系统会自动检查 `stock_data` 中是否包含指定的基础因子字段
2. **缺失字段**: 如果某些基础因子字段在 `stock_data` 中不存在，会发出警告但不会中断执行
3. **重复列处理**: 如果基础因子字段与因子组数据中的字段重复，优先保留因子组数据
4. **错误处理**: 如果基础因子合并过程中发生错误，会返回原始的因子组数据
5. **性能考虑**: 基础因子数据会根据日期范围进行过滤，避免处理不必要的数据

## 扩展建议

可以根据需要在 `base_factor_fields` 中添加更多基础因子字段，例如：
- 技术指标：RSI、MACD、布林带等
- 基本面指标：ROE、ROA、负债率等
- 市场指标：相对强弱、行业排名等
- 资金流指标：主力资金净流入、大单净流入等
