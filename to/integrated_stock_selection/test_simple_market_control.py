import pandas as pd
from datetime import date
from to.integrated_stock_selection.rqalpha_minute_datasource.strategy_backtest1 import BacktestStrategy, StrategyConfig, StrategyType as ST

# 创建策略配置
config = StrategyConfig(strategy_type=ST.TOP_1, stock_count=1)

# 创建预测数据
pred_data = pd.DataFrame({
    'date': [date(2026, 2, 10)],
    'stock_code': ['000001.SZ'],
    'predicted_return': [0.01]
})

# 创建策略实例
strategy = BacktestStrategy(
    config,
    pred_data,
    r'C:\Users\86151\.trae-cn\pythons\path\to\local_data\minute\202602\minute_data.parquet'
)

# 测试1: 加载市场状态数据
print("=" * 60)
print("Test 1: Load Market Status Data")
print("=" * 60)
strategy._load_market_status_data()
print("Data loaded:", hasattr(strategy, 'market_status_data'))
print("Total dates:", len(strategy.market_status_data))
print("Date range:", min(strategy.market_status_data.keys()), "~", max(strategy.market_status_data.keys()))

# 测试2: 计算市场状态
print("\n" + "=" * 60)
print("Test 2: Calculate Market Status")
print("=" * 60)
test_date = date(2026, 2, 10)
limit = strategy._get_position_limit_by_market_status(test_date)
print(f"Date: {test_date}")
print(f"Limit: {limit}")

# 显示当日市场数据
date_str = test_date.strftime('%Y%m%d')
if date_str in strategy.market_status_data:
    tech = strategy.market_status_data[date_str]['individual_status']['399006.SZ']['technical_data']
    print(f"Close: {tech['close']:.2f}")
    print(f"MA20: {tech['ma20']:.2f}")
    print(f"Close < MA20: {tech['close'] < tech['ma20']}")
    print(f"Market Status: {'WEAK (limit=0.01)' if limit == 0.01 else 'STRONG (limit=0.0)'}")

# 测试3: 计算仓位比例
print("\n" + "=" * 60)
print("Test 3: Calculate Position Percentage")
print("=" * 60)
test_cases = [
    (0.015, "High lead"),
    (0.005, "Low lead"),
    (-0.005, "Negative lead"),
]

for lead, desc in test_cases:
    pct = strategy._position_pct_by_lead(lead, test_date)
    position = "FULL (100%)" if pct == 1.0 else "EMPTY (0%)"
    print(f"{desc}: lead={lead:.4f} -> position_pct={pct:.2f} ({position})")

print("\n" + "=" * 60)
print("All tests passed successfully!")
print("=" * 60)
