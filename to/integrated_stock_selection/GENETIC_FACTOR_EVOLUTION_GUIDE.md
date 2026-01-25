# 遗传编程因子进化使用指南

## 🎯 功能概述

基于gplearn库实现的遗传编程因子进化功能，通过遗传算法自动进化出新的超级因子组合，用于量化交易中的股票选择。

## 🔧 核心特性

### 1. 智能因子组合
- **基础因子池**: 包含50+个技术、动量、成交量、价格位置、基本面等因子
- **自动组合**: 通过遗传编程自动发现因子间的最优数学组合
- **超级因子**: 生成比单一因子更强的预测能力的复合因子

### 2. 多空组合优化
- **优化目标**: 基于多空组合的夏普比率（使用T+2日收益率）
- **风险控制**: 内置最大回撤否决机制（>15%时给予极低分数）
- **收益评估**: 做多前20%，做空后20%的股票构建组合

### 3. 高级遗传算法
- **种群进化**: 支持大规模种群（默认1000个个体）
- **多代进化**: 支持长期进化（默认20代）
- **并行计算**: 自动利用多核CPU加速计算
- **自定义函数**: 包含保护性数学函数（log, sqrt, exp等）

### 4. 风险控制机制
- **否决门槛**: 最大回撤>15%或夏普比率<0.5时给予极低适应度
- **数据清洗**: 自动处理异常值、无穷值、缺失值
- **收益限制**: 限制收益率在合理范围内（-20%到20%）

## 📋 使用方法

### 基本用法
```bash
# 使用默认参数运行遗传编程
python main.py --mode genetic_factor

# 自定义参数运行
python main.py --mode genetic_factor \
    --population_size 1000 \
    --generations 20 \
    --n_best_factors 10 \
    --output_dir ./genetic_results
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--population_size` | 1000 | 遗传编程种群大小，越大越能找到更好的解 |
| `--generations` | 20 | 进化代数，越多越能充分进化 |
| `--n_best_factors` | 10 | 返回最佳因子的数量 |
| `--output_dir` | ./results | 结果输出目录 |

### 快速测试
```bash
# 小规模快速测试（适合验证功能）
python main.py --mode genetic_factor \
    --population_size 100 \
    --generations 5 \
    --n_best_factors 5
```

### 生产环境
```bash
# 大规模深度进化（适合生产使用）
python main.py --mode genetic_factor \
    --population_size 2000 \
    --generations 50 \
    --n_best_factors 20
```

## 📊 输出结果

### 1. 控制台输出
```
🏆 最佳因子:
   公式: add(mul(rsi6, momentum_5d), div(volume_ratio_5d, volatility20))
   夏普比率: 1.2345
   最大回撤: 0.0876
   年化收益: 0.1543
   胜率: 0.6234
   信息系数: 0.0876

📊 进化统计:
   总代数: 20
   种群大小: 1000
   收敛代数: 15
   种群多样性: 0.7234

🔝 前5个最佳因子:
   1. 夏普比率: 1.2345
      公式: add(mul(rsi6, momentum_5d), div(volume_ratio_5d, volatility20))
      最大回撤: 0.0876
```

### 2. 文件输出

#### 详细结果文件
- **文件名**: `genetic_factors_YYYYMMDD_HHMMSS.json`
- **内容**: 完整的进化结果，包括所有最佳因子的详细信息
- **格式**: JSON格式，便于程序读取

#### 因子公式文件
- **文件名**: `best_factor_formulas_YYYYMMDD_HHMMSS.txt`
- **内容**: 人类可读的因子公式和性能指标
- **格式**: 文本格式，便于查看和分析

## 🔬 技术实现

### 1. 数据预处理
```python
# 自动数据清洗
- 处理无穷大值和NaN值
- 异常值检测（5个标准差外）
- Z-score标准化
- 数据范围限制（-5到5）
```

### 2. 适应度函数
```python
def custom_fitness(y_true, y_pred, w):
    # 构建多空组合
    long_stocks = top_20_percent_by_factor
    short_stocks = bottom_20_percent_by_factor
    
    # 计算组合收益
    portfolio_returns = long_returns - short_returns
    
    # 计算夏普比率
    sharpe_ratio = mean(portfolio_returns) / std(portfolio_returns) * sqrt(252)
    
    # 风险控制
    if max_drawdown > 0.15:
        return -100.0  # 否决
    
    return sharpe_ratio * (1 - max_drawdown * 0.5)
```

### 3. 遗传编程配置
```python
SymbolicRegressor(
    population_size=1000,
    generations=20,
    p_crossover=0.7,        # 交叉概率
    p_subtree_mutation=0.1, # 子树变异概率
    p_hoist_mutation=0.05,  # 提升变异概率
    p_point_mutation=0.1,   # 点变异概率
    max_samples=0.8,        # 采样比例
    tournament_size=20,     # 锦标赛选择大小
    parsimony_coefficient=0.01  # 简洁性系数
)
```

## 📈 性能优化

### 1. 并行计算
- **自动检测**: 自动检测CPU核心数
- **并行限制**: 最大使用8个核心，避免系统过载
- **内存管理**: 智能内存管理，避免内存溢出

### 2. 数据采样
- **训练采样**: 使用80%的数据进行训练，提高速度
- **批量处理**: 批量处理大规模数据集
- **缓存机制**: 智能缓存中间结果

### 3. 早停机制
- **收敛检测**: 自动检测进化收敛
- **适应度阈值**: 达到目标适应度时提前停止
- **时间限制**: 可设置最大运行时间

## ⚠️ 注意事项

### 1. 计算资源
- **内存需求**: 大种群需要较多内存（建议16GB+）
- **CPU需求**: 多核CPU能显著提升速度
- **时间需求**: 完整进化可能需要数小时

### 2. 数据质量
- **因子完整性**: 确保基础因子数据质量良好
- **时间序列**: 需要足够的历史数据（建议1年+）
- **股票数量**: 需要足够的股票样本（建议500+只）

### 3. 结果解释
- **过拟合风险**: 复杂公式可能存在过拟合
- **实盘验证**: 需要在实盘环境中验证效果
- **定期更新**: 市场环境变化时需要重新进化

## 🚀 高级用法

### 1. 自定义因子池
```python
# 在genetic_factor_evolution.py中修改required_factors列表
required_factors = [
    'your_custom_factor_1',
    'your_custom_factor_2',
    # ... 其他因子
]
```

### 2. 自定义适应度函数
```python
# 修改_custom_fitness_function方法
def _custom_fitness_function(self, y_true, y_pred, w):
    # 实现你的自定义逻辑
    return your_fitness_score
```

### 3. 自定义数学函数
```python
# 在_create_custom_functions中添加新函数
def _your_custom_function(x1, x2):
    return your_calculation

custom_func = make_function(
    function=_your_custom_function,
    name='your_func',
    arity=2
)
```

## 📚 相关资源

- **gplearn文档**: https://gplearn.readthedocs.io/
- **遗传编程理论**: 《Genetic Programming: An Introduction》
- **量化交易**: 《Quantitative Trading》by Ernest Chan
- **因子投资**: 《Factor Investing》by Andrew Ang

## 🎉 总结

遗传编程因子进化功能为量化交易提供了强大的因子挖掘工具，能够：

1. **自动发现**: 自动发现复杂的因子组合关系
2. **风险控制**: 内置多重风险控制机制
3. **高效计算**: 利用并行计算提升效率
4. **易于使用**: 简单的命令行接口
5. **结果可解释**: 输出人类可读的数学公式

通过合理使用这个工具，可以显著提升量化交易策略的因子挖掘能力和投资收益！
