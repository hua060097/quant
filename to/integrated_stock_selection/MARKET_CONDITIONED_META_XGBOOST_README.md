# Market-Conditioned Meta-XGBoost 系统

## 🎯 系统概述

Market-Conditioned Meta-XGBoost是一个基于市场状态的专家集成量化选股系统，通过LSTM Autoencoder学习市场环境表示，训练多个时间窗口专家，并根据当前市场状态动态加权预测。

## 🏗️ 系统架构

### 1. Market Embedding模块 (`market_embedding.py`)
- **LSTM Autoencoder**: 将20日市场因子序列编码为8维embedding
- **自监督学习**: 通过重构损失学习市场状态表示
- **时间窗口**: 20个交易日的市场因子序列
- **输出维度**: 8维市场状态embedding

### 2. Expert Manager模块 (`expert_manager.py`)
- **专家池**: 6个时间窗口的XGBoost专家模型
- **Prototype计算**: 每个专家对应时间窗口的平均embedding
- **相似度加权**: 余弦相似度 + softmax温度参数
- **动态预测**: 基于市场状态的专家权重分配

### 3. 主系统集成 (`main.py`)
- **训练流程**: Market Embedding训练 → 专家池训练
- **预测流程**: 当日embedding生成 → 专家权重计算 → 加权预测
- **回退机制**: 专家池失败时使用传统单一模型

## 🚀 使用流程

### 步骤1: 训练Market Embedding
```bash
python main.py market_state_test
```
**功能**:
- 分析市场状态因子的择时能力和开关效应
- 训练LSTM Autoencoder学习市场状态表示
- 生成历史交易日的market embeddings
- 保存模型到 `models/market_embedding/`

### 步骤2: 训练专家池
```bash
python main.py train_model
```
**功能**:
- 加载Market Embedding模型
- 按时间窗口划分训练数据（6个专家）
- 训练每个专家的XGBoost模型
- 计算专家prototype embeddings
- 保存专家池到 `models/expert_pool/`

### 步骤3: Meta-XGBoost预测
```bash
python main.py test_model
```
**功能**:
- 加载专家池和Market Embedding模型
- 按日期生成market embeddings
- 计算专家相似度权重
- 执行加权预测和选股

## 📊 核心参数

### Market Embedding参数
```python
window_length = 20      # 市场因子序列长度
embedding_dim = 8       # Embedding维度
hidden_dim = 32         # LSTM隐藏层维度
epochs = 50            # 训练轮数
batch_size = 128       # 批次大小
learning_rate = 0.001  # 学习率
```

### 专家池参数
```python
n_experts = 6          # 专家数量
temperature = 0.5      # Softmax温度参数
max_depth = 6          # XGBoost最大深度
learning_rate = 0.05   # XGBoost学习率
n_estimators = 500     # XGBoost树数量
```

## 🔧 技术特性

### 1. 市场状态编码
- **输入**: 16个市场因子的20日序列 (20×16)
- **编码器**: 双层LSTM + 全连接层
- **解码器**: LSTM + 重构层
- **损失函数**: 均方误差(MSE)

### 2. 专家池管理
- **时间分割**: 6个连续时间窗口（18-24个月/窗口）
- **专家训练**: 每个窗口独立训练XGBoost
- **Prototype**: 窗口内所有embedding的均值
- **特征**: 仅使用股票因子（不包含市场因子）

### 3. 动态加权预测
- **相似度**: 余弦相似度计算
- **权重分配**: Softmax(similarity/temperature)
- **最终预测**: Σ(weight_k × prediction_k)
- **温度控制**: tau=0.5平衡权重分布

## 📁 文件结构

```
models/
├── market_embedding/
│   ├── lstm_autoencoder.pth    # LSTM模型权重
│   ├── scaler.pkl              # 数据标准化器
│   └── config.pkl              # 模型配置
└── expert_pool/
    ├── experts.pkl             # 专家模型集合
    ├── prototypes.pkl          # 专家prototype embeddings
    └── config.pkl              # 专家池配置
```

## 🎯 性能优势

### 1. 市场适应性
- **动态权重**: 根据当前市场状态调整专家权重
- **时间多样性**: 6个专家覆盖不同市场环境
- **状态感知**: LSTM捕捉市场状态的时序特征

### 2. 预测稳定性
- **集成学习**: 多专家预测降低单一模型风险
- **相似度加权**: 相似市场环境的专家获得更高权重
- **回退机制**: 专家池失败时自动使用传统模型

### 3. 系统鲁棒性
- **模块化设计**: 各组件独立，便于维护和升级
- **错误处理**: 完善的异常处理和日志记录
- **持久化**: 模型状态完整保存和加载

## 🔍 监控指标

### 训练阶段
- Market Embedding重构损失
- 专家池训练样本数量
- Prototype embedding质量

### 预测阶段
- 专家权重分布
- 相似度计算结果
- 预测值统计分析

### 回测评估
- Baseline vs Meta-XGBoost性能对比
- 按market embedding分位的表现分析
- IC、RankIC、Top10%收益等指标

## ⚠️ 注意事项

1. **数据要求**: 需要至少20个交易日的历史市场因子数据
2. **计算资源**: LSTM训练需要一定的计算时间
3. **模型依赖**: 专家池预测依赖Market Embedding模型
4. **参数调优**: 温度参数和专家数量可根据实际效果调整

## 🚀 扩展建议

1. **增加专家数量**: 可调整为8-10个专家获得更细粒度的市场状态划分
2. **优化embedding维度**: 根据市场因子复杂度调整embedding维度
3. **集成其他模型**: 可将LSTM替换为Transformer或其他序列模型
4. **动态专家选择**: 实现基于置信度的专家动态选择机制

---

**系统状态**: ✅ 已完成集成并通过全面测试
**下一步**: 在实际数据上运行完整流程并评估效果
