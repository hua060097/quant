# 板块轮动量化交易系统

这是一个基于真实股票数据进行板块轮动交易的量化系统，集成了数据处理、特征工程、模型训练、预测和回测功能。

## 项目结构

```
to/
  ├── Liquid/                # 液体神经网络模型
  │   └── nn/liquid/         # 液体神经网络实现
  ├── local_data/            # 本地数据
  │   ├── calendar/          # 交易日历
  │   ├── sector_features/   # 板块特征数据
  │   └── market_status/     # 市场状态数据
  ├── log/                   # 日志文件
  ├── models/                # 保存的模型
  │   └── sector_models/     # 板块模型
  ├── results/               # 结果文件
  │   └── backtest/          # 回测结果
  ├── data_processor.py      # 数据处理模块
  ├── feature_processor.py   # 特征处理模块
  ├── model_manager.py       # 模型管理模块
  ├── logger_config.py       # 日志配置模块
  ├── sector_model.py        # 板块模型主API
  └── README.md              # 项目说明文档
```

## 模块说明

- **data_processor.py**: 负责数据加载、缓存和基本处理，提供统一的数据访问接口
- **feature_processor.py**: 负责特征转换、归一化和特征工程，保证训练和预测过程中特征处理的一致性
- **model_manager.py**: 负责模型的训练、评估、保存、加载和预测，提供模型生命周期管理
- **logger_config.py**: 提供统一的日志配置
- **sector_model.py**: 主API模块，整合以上组件，提供训练、预测和回测功能

## 主要改进

1. **统一的特征处理流水线**：
   - 集中在FeatureProcessor类中实现特征处理逻辑
   - 确保训练和预测流程中特征处理的一致性
   - 增强数据清洗、缺失值处理和异常处理能力

2. **自适应模型权重机制**：
   - 实现了动态计算模型权重的方法
   - 基于历史RMSE、市场状态和预测确信度调整权重
   - 支持在不同市场环境下自动调整模型权重

3. **LNN特征生成优化**：
   - 改进LNN特征覆盖率阈值，确保高质量特征
   - 实现基于PCA的替代特征生成方法
   - 增加特征生成过程的可追溯性

4. **代码重构与优化**：
   - 将重复逻辑提取为独立的工具类和方法
   - 模块化设计，提高代码可维护性
   - 完善错误处理和日志记录，提高系统健壮性

## 使用方法

### 安装依赖

```bash
pip install numpy pandas scikit-learn xgboost lightgbm torch
```

### 训练板块轮动模型

```bash
python sector_model.py --mode train --start_date 20230101 --end_date 20231231 --use_lnn
```

### 预测板块未来收益

```bash
python sector_model.py --mode predict --end_date 20240601 --output predictions.csv
```

### 回测板块轮动策略

```bash
python sector_model.py --mode backtest --start_date 20230101 --end_date 20231231 --top_n 5 --output backtest_results.json
```

## API说明

### SectorModel类

这是主要的API类，提供以下方法：

#### 初始化

```python
from to.sector_model import SectorModel

# 初始化模型
model = SectorModel(model_dir=None, data_dir=None, use_lnn=True)
```

#### 训练模型

```python
# 训练板块轮动模型
results = model.train(sector_list=None, start_date='20230101', end_date='20231231')
```

#### 预测未来收益

```python
# 预测板块未来收益
predictions = model.predict(sector_list=None, date='20240601')
```

#### 回测策略

```python
# 回测板块轮动策略
backtest_results = model.backtest(sector_list=None, start_date='20230101', end_date='20231231', top_n=5)
```

## 注意事项

1. 确保数据目录结构正确，特别是板块特征数据和市场状态数据
2. 液体神经网络(LNN)在训练时需要更多计算资源，可以根据需要开启或关闭
3. 回测结果仅作为参考，实际交易时需要考虑更多因素如交易成本、滑点等
4. 建议定期重新训练模型以适应最新的市场环境 