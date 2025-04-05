# 板块轮动模型系统代码索引

本文档提供了板块轮动模型系统中关键类和函数的索引，方便快速定位和理解代码。按照功能模块进行分类整理。

## 1. 数据加载与处理模块

### 数据存储与获取

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `ParquetDataStorage` | `to/advanced_quant_data.py` | 高效的数据存储和访问类，提供行情数据的保存和读取功能 |
| `download_daily_data()` | `to/advanced_quant_data.py` | 下载日线数据并保存为Parquet格式 |
| `download_calendar()` | `to/advanced_quant_data.py` | 下载交易日历并保存 |
| `get_market_data()` | `to/advanced_quant_data.py` | 获取格式化的市场数据 |
| `get_daily_data()` | `to/advanced_quant_data.py` | 获取股票日线数据 |
| `DataLoaderWrapper` | `to/generate_sector_features.py` | 数据加载器包装类，提供统一的接口 |

### 板块-股票映射

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `load_data()` | `to/stock_sectors_mapping.py` | 加载板块-股票映射数据 |
| `update_mapping()` | `to/stock_sectors_mapping.py` | 更新板块-股票映射关系 |
| `get_stocks_in_sector()` | `to/stock_sectors_mapping.py` | 获取指定板块的所有股票 |
| `get_sectors_for_stock()` | `to/stock_sectors_mapping.py` | 获取指定股票所属的所有板块 |

## 2. 特征生成模块

### 板块特征生成

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `SectorFeatureManager` | `to/generate_sector_features.py` | 板块特征管理类，负责生成和管理板块特征 |
| `generate_features()` | `to/generate_sector_features.py` | 生成板块特征的主函数，支持增量更新和全量更新 |
| `process_date()` | `to/generate_sector_features.py` | 处理单个日期的板块特征生成 |
| `optimize_data_structure()` | `to/generate_sector_features.py` | 优化数据结构以提高处理效率 |
| `split_sectors_by_worker()` | `to/generate_sector_features.py` | 将板块按照工作进程数量分割，用于并行处理 |

### 向量化特征提取

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `calculate_ma_fast()` | `to/generate_sector_features.py` | 使用Numba加速的移动平均计算 |
| `calculate_ema_fast()` | `to/generate_sector_features.py` | 使用Numba加速的指数移动平均计算 |
| `calculate_macd_fast()` | `to/generate_sector_features.py` | 使用Numba加速的MACD指标计算 |
| `VectorizedFeatureExtractor` | `to/vectorized_features.py` | 向量化特征提取器类，提供高效的特征计算方法 |

### 股票排名相关

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `calculate_ten_day_returns()` | `to/stock_daily_rankings.py` | 计算十日涨幅排名 |
| `calculate_daily_returns_and_amount()` | `to/stock_daily_rankings.py` | 计算当日涨跌幅和成交额 |
| `get_rankings_for_date()` | `to/stock_daily_rankings.py` | 获取特定日期的排名数据 |
| `update_all_rankings()` | `to/stock_daily_rankings.py` | 更新所有排名数据 |

### 市场状态分析

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `MarketAnalyzer` | `to/market_status/market_analyzer.py` | 大盘分析器，专注于创业板指数分析 |
| `analyze_market_status()` | `to/market_status/market_analyzer.py` | 分析市场状态，返回市场类型 |
| `_calculate_trend_score()` | `to/market_status/market_analyzer.py` | 计算市场趋势得分 |
| `_identify_market_cycle()` | `to/market_status/market_analyzer.py` | 识别市场周期 |

## 3. 模型训练模块

### 板块轮动模型

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `train_sector_rotation_model()` | `to/train_models.py` | 训练板块轮动模型，使用板块特征和大盘特征 |
| `run_offline_training()` | `to/train_models.py` | 运行离线训练流程，支持批量处理 |
| `train_and_save_sector()` | `to/train_models.py` | 训练并保存单个板块的模型 |
| `load_features_for_training()` | `to/train_models.py` | 加载特征数据用于模型训练 |

### 数据处理与特征组合

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `_load_sector_features()` | `to/train_models.py` | 加载板块特征数据 |
| `_load_market_features()` | `to/train_models.py` | 加载市场特征数据 |
| `_combine_all_features()` | `to/train_models.py` | 整合所有特征数据 |
| `_build_training_dataset()` | `to/train_models.py` | 构建训练数据集 |
| `_add_time_decay_weights()` | `to/train_models.py` | 添加时间衰减权重 |

### 增量训练

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `IncrementalTrainer` | `to/incremental_training.py` | 增量训练器类，支持模型的增量更新 |
| `update_model_incrementally()` | `to/incremental_training.py` | 增量更新模型 |
| `merge_datasets()` | `to/incremental_training.py` | 合并新旧数据集 |

### 模型类

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `EnsembleModel` | `to/model_classes.py` | 集成模型，结合XGBoost、LightGBM、CatBoost和LNN |
| `SectorRotationEnsemble` | `to/models/ensemble_model.py` | 板块轮动集成模型 |
| `LiquidNeuralNetwork` | `to/advanced_quant_model.py` | 液体神经网络模型 |

## 4. 液体神经网络模块

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `LiquidCell` | `to/Liquid/nn/liquid/liquid_network.py` | 液体神经元细胞，可自适应调整时间常数和非线性激活函数 |
| `LiquidNetwork` | `to/Liquid/nn/liquid/liquid_network.py` | 液体神经网络，由多层液体神经元细胞组成 |
| `LiquidNetModel` | `to/Liquid/nn/liquid/liquid_network.py` | 液体神经网络模型包装类，提供训练和预测接口 |
| `LiquidTimeConstantNetwork` | `to/market_status_features.py` | 基于液体时间常数的神经网络，用于时序预测 |

## 5. 运行工具模块

### 主运行脚本

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `run_sector_rotation_model()` | `to/simple_runner.py` | 运行板块轮动模型的主函数 |
| `main()` | `to/simple_runner.py` | 命令行入口函数 |
| `parse_args()` | `to/simple_runner.py` | 解析命令行参数 |
| `generate_date_list()` | `to/simple_runner.py` | 生成日期列表 |

### 配置管理

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `get_config()` | `to/config.py` | 获取系统配置 |
| `PROJECT_ROOT` | `to/config.py` | 项目根目录路径常量 |
| `DATA_ROOT` | `to/config.py` | 数据根目录路径常量 |
| `MODEL_DIR` | `to/config.py` | 模型目录路径常量 |

### 性能优化

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `NumbaAccelerator` | `to/numba_accelerator.py` | Numba加速器类，提供各种技术指标的加速计算 |
| `accelerate_backtest()` | `to/accelerate_backtest.py` | 加速回测的入口函数 |
| `batch_load_data()` | `to/generate_sector_features.py` | 批量加载数据函数，提高数据加载效率 |

### 实用工具

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `create_data_directories()` | `to/create_data_directories.py` | 创建系统所需的数据目录 |
| `log_memory_usage()` | `to/generate_sector_features.py` | 记录内存使用情况 |
| `split_date_range()` | `to/generate_sector_features.py` | 将日期范围分割为多个小区间 |

## 6. 信号生成模块

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `generate_stock_signals_from_sector_prediction()` | `to/train_models.py` | 基于板块预测生成个股交易信号 |
| `_select_top_stocks_in_sector()` | `to/train_models.py` | 从板块中选择最优的股票 |
| `_calculate_relative_strength()` | `to/train_models.py` | 计算相对强度 |
| `_calculate_fund_inflow()` | `to/train_models.py` | 计算资金流入得分 |
| `_calculate_technical_scores()` | `to/train_models.py` | 计算技术指标得分 |
| `_calculate_volume_scores()` | `to/train_models.py` | 计算成交量相关得分 |

## 7. 测试与验证模块

| 类/函数 | 文件路径 | 功能简述 |
|---------|---------|---------|
| `test_sector_rotation()` | `to/test_sector_rotation.py` | 测试板块轮动模型 |
| `test_feature_loading()` | `to/test_feature_loading.py` | 测试特征加载功能 |
| `test_generate_features()` | `to/test_sector_features.py` | 测试板块特征生成功能 |
| `test_model_training()` | `to/test_sector_model.py` | 测试模型训练功能 | 