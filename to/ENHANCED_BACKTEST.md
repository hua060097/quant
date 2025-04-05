# 增强回测系统使用说明

本文档介绍了增强回测系统的主要特性和使用方法，该系统通过集成市场模拟器，显著提高了回测的真实性和准确性。

## 主要特性

增强回测系统在原有回测引擎的基础上，增加了以下关键特性：

1. **真实滑点模拟**：
   - 固定滑点模型：设置固定的滑点百分比
   - 基于成交量的滑点模型：根据交易量动态计算滑点
   - 随机滑点模型：在设定范围内随机生成滑点

2. **交易成本计算**：
   - 佣金计算：支持按比例计算佣金，并设置最低佣金
   - 印花税：对卖出交易收取印花税（A股特性）
   - 过户费：上交所股票交易的过户费

3. **A股市场限制**：
   - 涨跌停限制：实现涨跌停板限制，拒绝超过限制的交易
   - T+1交易规则：当日买入的股票不能当日卖出

4. **订单管理**：
   - 订单状态跟踪：记录订单的创建、执行和拒绝状态
   - 交易拒绝原因：详细记录订单被拒绝的原因

5. **持仓管理**：
   - 精确的持仓跟踪：记录每个股票的买入时间、价格和数量
   - 持仓市值更新：每日收盘时更新持仓市值

6. **业绩评估**：
   - 详细的交易记录：记录每笔交易的执行价格、数量和手续费
   - 多维度性能指标：计算总收益率、年化收益率、夏普比率、最大回撤等

## 系统架构

增强回测系统由以下几个主要组件构成：

1. **市场模拟器（`market_simulator.py`）**：
   - 核心模拟组件，实现各种交易规则和限制
   - 包含滑点模型、交易成本模型、价格限制模型等

2. **回测引擎补丁（`backtest_engine_patch.py`）**：
   - 通过Monkey Patching技术增强原有回测引擎
   - 保留原有功能的同时，集成市场模拟器的新特性

3. **回测示例（`enhanced_backtest_example.py`）**：
   - 展示如何使用增强回测系统
   - 提供不同配置的比较测试

## 使用方法

### 1. 基本使用

要使用增强回测系统，首先需要应用补丁并创建回测引擎：

```python
# 导入补丁模块
from backtest_engine_patch import patch_backtest_engine, create_enhanced_backtest_engine

# 应用补丁
patch_backtest_engine()

# 创建增强回测引擎
engine = create_enhanced_backtest_engine(
    start_date='20240101',
    end_date='20240331',
    initial_capital=1000000,
    position_size=0.1,
    max_positions=10,
    slippage_model_type='volume',  # 可选：'fixed', 'volume', 'random'
    slippage_param=0.0001,
    commission_rate=0.00025,
    min_commission=5.0,
    price_limit_pct=0.1,
    t_plus_1=True
)

# 加载交易信号
# ... 加载或生成交易信号 ...

# 运行回测
engine.run_backtest()

# 分析结果
print(f"总收益率: {engine.total_return:.2%}")
print(f"年化收益率: {engine.annual_return:.2%}")
print(f"夏普比率: {engine.sharpe_ratio:.2f}")
```

### 2. 参数说明

创建增强回测引擎时可以设置以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `start_date` | 回测开始日期（格式：'YYYYMMDD'） | 必填 |
| `end_date` | 回测结束日期（格式：'YYYYMMDD'） | 必填 |
| `initial_capital` | 初始资金 | 1000000 |
| `position_size` | 单个股票仓位比例 | 0.1 |
| `max_positions` | 最大持仓数量 | 10 |
| `slippage_model_type` | 滑点模型类型（'fixed', 'volume', 'random'） | 'volume' |
| `slippage_param` | 滑点参数 | 0.0001 |
| `commission_rate` | 佣金率 | 0.00025 |
| `min_commission` | 最低佣金 | 5.0 |
| `price_limit_pct` | 涨跌停百分比 | 0.1 |
| `t_plus_1` | 是否启用T+1规则 | True |

### 3. 运行示例

系统提供了一个示例脚本，展示了不同配置下的回测比较：

```bash
python enhanced_backtest_example.py
```

示例包括：
- 不同滑点模型的比较测试
- T+1规则影响测试
- 涨跌停限制影响测试

## 自定义扩展

### 1. 自定义滑点模型

可以通过继承`SlippageModel`基类来创建自定义的滑点模型：

```python
from market_simulator import SlippageModel

class MyCustomSlippageModel(SlippageModel):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def calculate_slippage(self, price, volume, direction, market_data=None):
        # 实现自定义滑点计算逻辑
        return my_slippage_amount
```

### 2. 自定义交易成本模型

类似地，可以通过继承`CommissionModel`基类来创建自定义的交易成本模型：

```python
from market_simulator import CommissionModel

class MyCustomCommissionModel(CommissionModel):
    def __init__(self, rate, min_commission):
        self.rate = rate
        self.min_commission = min_commission
    
    def calculate_commission(self, price, volume, direction):
        # 实现自定义交易成本计算逻辑
        return my_commission_amount
```

## 性能考量

增强回测系统由于添加了更多的真实市场模拟功能，运行速度可能会比原始回测引擎稍慢。为了提高性能，可以考虑：

1. 在不需要特定功能时禁用它们（如设置`t_plus_1=False`）
2. 对于初步测试，可以使用较短的回测时间段
3. 在最终回测前，确保交易信号已经完全准备好

## 常见问题

1. **回测结果与原始引擎差异较大**
   - 这是正常的，因为增强系统考虑了更多现实因素，如滑点、交易成本和市场限制
   - 差异大说明原始回测可能过于理想化

2. **交易数量比预期少**
   - 检查是否有交易被T+1规则或涨跌停限制拒绝
   - 查看日志中的拒绝原因

3. **收益率低于原始回测**
   - 这通常是由于加入了更真实的交易成本和滑点所致
   - 这种情况下，增强系统的结果更接近实际交易情况

## 联系与支持

如有任何问题或建议，请联系开发团队。

---

*本文档最后更新于：2024-05-10* 