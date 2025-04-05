# 板块轮动模型系统

本系统实现了基于板块特征的板块轮动模型，使用板块和大盘特征进行训练，结合长短周期捕捉轮动特征，支持使用LNN(液体神经网络)模型更好地反映时间序列特征。

## 系统结构

```
project_root/
├── to/                         # 主要代码目录
│   ├── Liquid/                 # 液体神经网络相关代码
│   ├── local_data/             # 本地数据存储目录
│   │   ├── calendar/           # 交易日历数据
│   │   ├── daily/              # 日线数据
│   │   ├── features/           # 特征数据
│   │   ├── metadata/           # 元数据
│   │   ├── minute/             # 分钟线数据
│   │   ├── models/             # 训练出的模型
│   │   │   └── sector/         # 板块相关模型
│   │   ├── sector_features/    # 板块特征数据
│   │   │   └── TGN*/           # 各个TGN板块目录
│   │   └── sector_stocks_cache.pkl  # 板块股票映射缓存
│   ├── log/                    # 日志文件目录
│   ├── models/                 # 模型定义和模型保存目录
│   ├── results/                # 回测和分析结果
│   ├── sector_analysis/        # 板块分析相关代码和数据
│   ├── sector_features/        # 板块特征生成相关代码
│   ├── signals/                # 生成的交易信号
│   └── *.py                    # 其他Python模块文件
├── data/                       # 外部数据目录
├── data_cache/                 # 数据缓存目录
├── log/                        # 根目录日志文件
├── models/                     # 备份模型目录
├── results/                    # 备份结果目录
└── *.py                        # 根目录的Python模块文件
```

## 功能特点

1. **板块轮动策略**：基于板块表现预测，选择表现最好的板块，再从中选择最优股票
2. **多周期特征**：结合短期和长期特征，捕捉板块轮动趋势
3. **LNN模型支持**：支持使用液体神经网络模型更好地反映时间序列特征
4. **灵活配置**：可配置参数包括回溯天数、选股数量、模型类型等

## 修复历史

### 2024-04-04 板块特征生成修复
- 修复了板块特征生成程序中数据读取和处理的问题
- 增强了预加载数据处理逻辑，支持不同的数据格式和结构
  - 增加了对批处理格式数据(batch_*)的支持
  - 添加了对不同日期格式的兼容处理
  - 修复了获取股票数据时可能的异常情况
- 增加了详细的日志记录，便于调试和问题排查
  - 记录预加载数据结构的详细信息
  - 记录股票数据处理过程中的状态和异常
- 修复后可以正确生成板块特征，包括开盘价、收盘价、成交量等均值数据
- 添加了对一些常用技术指标的计算，如MA5、MA10、MA20等
  - 支持更多的技术分析特征，如价格相对均线位置比率
  - 支持交易量分析特征，如交易量变化率、交易量均线等

### 2024-04-05 优化路径处理
- 修复了路径处理中可能导致创建重复目录结构(如to/to)的问题
- 添加了路径处理工具模块`to/path_utils.py`，提供统一的路径处理函数
  - `get_project_path()`: 获取相对于项目根目录的路径
  - `get_data_path()`: 获取相对于数据目录的路径
  - `get_model_path()`: 获取相对于模型目录的路径
  - `get_log_path()`: 获取相对于日志目录的路径
  - `get_results_path()`: 获取相对于结果目录的路径
  - `ensure_dir()`: 确保目录存在
  - `normalize_path()`: 标准化路径字符串
- 添加了路径使用规范文档`to/README_PATH.md`
- 更新了`create_data_directories.py`以使用新的路径处理工具
- 更新了多个文件中的路径处理方式，确保路径使用一致性

## 使用方法

### 路径处理

系统现在提供了统一的路径处理方式，以避免路径混乱和重复目录的问题。请参考以下示例：

```python
# 导入路径工具
from to.path_utils import get_data_path, get_model_path, ensure_dir

# 获取数据目录下的特征文件路径
feature_path = get_data_path('sector_features/TGN人工智能/features_20230103.parquet')

# 确保输出目录存在
output_dir = ensure_dir(get_data_path('new_features'))

# 获取模型保存路径
model_path = get_model_path('sector_rotation/model_20230103.pkl')
```

更多详情请参考`to/README_PATH.md`文件。

### 板块特征生成

```bash
# 生成所有TGN板块的特征
python -m to.generate_sector_features --start_date 20230101 --end_date 20230110

# 生成指定板块的特征
python -m to.generate_sector_features --start_date 20230101 --end_date 20230110 --sectors "TGN人工智能,TGN芯片概念"

# 生成特征并同时保存CSV格式
python -m to.generate_sector_features --start_date 20230101 --end_date 20230110 --save_csv
```

更多参数请运行`python -m to.generate_sector_features --help`查看。

### 基本使用

运行单个日期的板块轮动模型：

```bash
python -m to.simple_runner --date 20230105
```

### 参数说明

- `--date`：指定日期，格式为YYYYMMDD，默认为当前日期
- `--start_date` 和 `--end_date`：指定日期范围
- `--lookback_short`：短期回溯天数，默认30
- `--lookback_long`：长期回溯天数，默认90
- `--forward_days`：预测未来几天，默认5
- `--use_lnn`：使用LNN模型
- `--top_sectors`：选择排名靠前的板块数量，默认3
- `--stocks_per_sector`：每个板块选择的股票数量，默认3
- `--no_signals`：不生成交易信号

### 批量处理

处理一段时间范围内的数据：

```bash
python -m to.simple_runner --start_date 20230101 --end_date 20230110
```

### 使用LNN模型

```bash
python -m to.simple_runner --date 20230105 --use_lnn
```

## 模型原理

板块轮动模型基于以下原理：

1. **板块特征提取**：
   - 短期特征：最近5-30天内的价格、成交量等指标
   - 长期特征：最近60-90天内的趋势和波动性指标
   - 大盘关联：板块与大盘的相关性特征

2. **板块排名预测**：
   - 使用机器学习模型预测未来表现最好的板块
   - 支持随机森林、GBDT、LNN等多种模型

3. **选股策略**：
   - 相对强度：选择板块内动量最强的股票
   - 资金流向：考虑资金流入因素
   - 技术指标：结合均线、波动率等技术指标
   - 成交量特征：考虑量价配合情况

## 依赖库

- numpy
- pandas
- scikit-learn
- torch (LNN模型需要)
- joblib

## 注意事项

1. 确保板块特征数据已经生成
2. 使用LNN模型需要安装PyTorch
3. 板块-股票映射需要定期更新 