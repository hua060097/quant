# 机器学习K线形态选股系统 - 新版main.py使用说明

## 📋 概述

新版main.py已经完全重构，专注于机器学习K线形态选股功能。原来的`ml_pattern_selection`模式已被拆分为三个独立的运行模式，提供更清晰的工作流程。

## 🔄 主要变化

### 删除的模式
- ❌ `sector_backtest` - 板块策略回测模式
- ❌ `train_selection` - 训练集选股模式  
- ❌ `test_selection` - 测试集选股模式
- ❌ `full_pipeline` - 完整流程模式
- ❌ `ml_pattern_selection` - 原机器学习选股模式

### 新增的模式
- ✅ `calculate_features` - 特征计算模式（特征工程）
- ✅ `train_model` - 模型训练模式
- ✅ `predict_stocks` - 选股预测模式

## 🚀 使用方法

### 1. 特征计算模式
计算和增强股票数据的特征，为模型训练做准备。

```bash
# 基础用法
python main.py --mode calculate_features

# 自定义参数
python main.py --mode calculate_features \
    --start_date 20230601 \
    --end_date 20250725 \
    --output_dir ./features_output
```

**功能说明：**
- 加载指定时间范围的股票数据
- 使用FeatureCalculator计算缺失特征
- 生成增强后的特征数据集
- 保存特征数据和特征列表

### 2. 模型训练模式
使用增强后的特征数据训练机器学习模型。

```bash
# 基础用法
python main.py --mode train_model

# 自定义训练期间
python main.py --mode train_model \
    --start_date 20230601 \
    --train_end_date 20241230 \
    --output_dir ./model_output
```

**功能说明：**
- 加载训练期间的数据
- 进行特征工程增强
- 生成训练标签（次日涨幅排名前10%）
- 训练XGBoost分类模型
- 保存训练好的模型到缓存

### 3. 选股预测模式
使用训练好的模型进行股票选择预测。

```bash
# 基础用法
python main.py --mode predict_stocks

# 自定义预测期间
python main.py --mode predict_stocks \
    --start_date 20230601 \
    --train_end_date 20241230 \
    --end_date 20250725 \
    --output_dir ./predictions_output
```

**功能说明：**
- 加载训练好的模型
- 加载测试期间的数据
- 进行特征工程增强
- 使用模型进行选股预测
- 保存预测结果

## 📊 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | `calculate_features` | 运行模式 |
| `--output_dir` | str | `./results` | 输出目录 |
| `--log_level` | str | `INFO` | 日志级别 |
| `--save_results` | bool | `True` | 是否保存结果 |
| `--start_date` | str | `20230601` | 开始日期 |
| `--end_date` | str | `20250725` | 结束日期 |
| `--train_end_date` | str | `20241230` | 训练结束日期 |

## 🔧 核心功能

### 特征工程
- **DTW形态相似度特征**: W底、头肩顶、杯柄等形态识别
- **形态解构特征**: 波动率结构、成交量剖面、相对强度变化
- **微观结构特征**: 影线比例、实体比例、反转强度
- **技术指标特征**: RSI、MACD、布林带等
- **基本面特征**: 市值、换手率、流动性等

### 机器学习模型
- **算法**: XGBoost分类器
- **标签**: 次日涨幅排名前10%（二分类）
- **特征选择**: 自动筛选可用特征
- **数据预处理**: 标准化、缺失值处理
- **模型缓存**: 自动保存和加载训练好的模型

### 选股策略
- **风险控制**: 避免追高、过滤大跌股
- **概率预测**: 输出选股概率和置信度
- **每日选股**: 选择概率最高的前2只股票
- **结果保存**: JSON格式保存详细预测结果

## 📁 输出文件

### 特征计算模式输出
- `enhanced_features_YYYYMMDD_HHMMSS.pkl` - 增强特征数据
- `feature_list_YYYYMMDD_HHMMSS.json` - 特征列表信息

### 模型训练模式输出
- `./model_cache/ml_model_STARTDATE_ENDDATE.pkl` - 训练好的模型缓存

### 选股预测模式输出
- `ml_predictions_YYYYMMDD_HHMMSS.json` - 选股预测结果

## 🔄 工作流程

推荐的完整工作流程：

```bash
# 1. 计算特征
python main.py --mode calculate_features --start_date 20230601 --end_date 20250725

# 2. 训练模型
python main.py --mode train_model --start_date 20230601 --train_end_date 20241230

# 3. 进行预测
python main.py --mode predict_stocks --start_date 20230601 --train_end_date 20241230 --end_date 20250725
```

## ⚠️ 注意事项

1. **数据依赖**: 确保有足够的历史数据用于特征计算
2. **模型缓存**: 相同训练期间的模型会被缓存，避免重复训练
3. **内存使用**: 大量数据处理时注意内存使用情况
4. **特征一致性**: 训练和预测时使用相同的特征集合

## 🎯 优势

- **模块化设计**: 三个独立模式，职责清晰
- **缓存机制**: 避免重复计算，提高效率
- **错误处理**: 完善的异常处理和日志记录
- **参数灵活**: 支持自定义时间范围和输出目录
- **专业特征**: 基于顶级量化研究的特征工程

## 🔍 故障排除

如果遇到导入错误，系统会自动使用简化的替代类，确保基本功能正常运行。详细的错误信息会记录在日志中。

---

**注意**: 这是从原`ml_pattern_selection`模式重构而来的专业版本，保留了所有核心功能，但提供了更清晰的工作流程和更好的可维护性。
