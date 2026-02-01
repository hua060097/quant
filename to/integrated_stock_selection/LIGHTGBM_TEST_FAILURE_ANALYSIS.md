# LightGBM 测试集效果差 - 大师级分析报告

**分析时间**: 2026-01-30
**函数**: `_train_lightgbm_model` (main.py:2809-3101)
**问题**: 模型使用了参数搜索 + TimeSeriesSplit 交叉验证 + 过拟合惩罚，但测试集效果依旧很差

---

## 🔍 当前实现分析

### 1. 训练流程概览

```
数据准备 (合并 Train + Val)
    ↓
TimeSeriesSplit (3折交叉验证)
    ↓
Random Search (25组参数)
    ↓
Max-Min Sharpe 评估
    ↓
过拟合惩罚 (Gap Penalty)
    ↓
全量训练 (使用最优参数)
    ↓
返回模型
```

### 2. 核心优化策略

#### 策略1: TimeSeriesSplit 交叉验证
```python
# 第2868-2889行
tscv = TimeSeriesSplit(n_splits=3)
folds = []
for train_date_idx, val_date_idx in tscv.split(unique_dates):
    train_mask = np.isin(dates_all, train_dates_split)
    val_mask = np.isin(dates_all, val_dates_split)
    folds.append((train_mask, val_mask))
```

**目的**: 避免未来数据泄露

#### 策略2: Max-Min Sharpe 评估
```python
# 第2995-3017行
min_val_score = np.min(fold_val_scores)  # 最差一折
avg_val_score = np.mean(fold_val_scores)  # 平均表现
robust_score = (min_val_score * 0.7 + avg_val_score * 0.3) - penalty
```

**目的**: 选择在最差时期表现最好的参数（稳健性）

#### 策略3: 过拟合惩罚
```python
# 第3004-3013行
gap = avg_train_score - avg_val_score  # 泛化差距
if gap > 0.5:
    penalty = (gap - 0.5) * 0.5
```

**目的**: 惩罚训练集和验证集差距过大的参数

---

## ❌ 问题诊断：9大核心缺陷

### 缺陷1: 无真正的测试集（最致命）

**问题代码**:
```python
# 第2857-2862行
X_all = np.vstack((X_train_raw, X_val_raw))  # ← 训练集 + 验证集合并
y_all = np.concatenate((y_train_raw, y_val_raw))
```

**问题分析**:
- **训练集 + 验证集** = 全部数据
- **没有独立的测试集**
- 模型在全量数据上训练（第3037-3054行）
- 最后20%数据的评估（第3070-3072行）**不是测试集**，而是训练数据的一部分

**根本问题**:
```
真正的时间序列:
Train Period | Val Period | Test Period (未来)
    ↓            ↓            ↓
   使用        使用          未使用

当前实现:
Train Period | Val Period | (无测试集)
    ↓            ↓
   合并训练  ← ← ← ← ← ← ← ← ←
```

**后果**:
- 模型在验证集上可能表现良好
- 但在全新的测试集上表现差
- **因为没有验证模型的泛化能力**

---

### 缺陷2: Look-ahead Bias（前视偏差）

**问题代码**:
```python
# 第2857-2862行
X_all = np.vstack((X_train_raw, X_val_raw))  # ← 包含未来数据
```

**问题分析**:
- `X_val_raw` 是**未来**的数据（相对于 `X_train_raw`）
- 在交叉验证时，每个 fold 的验证集都是**未来**数据
- 但在全量训练时，模型**看到了未来的数据**

**Look-ahead Bias 示例**:
```
Fold 1: Train[2022-01~2022-08] | Val[2022-09~2022-12]
Fold 2: Train[2022-01~2022-12] | Val[2023-01~2023-04]
Fold 3: Train[2022-01~2023-04] | Val[2023-05~2023-08]

全量训练: Train + Val = [2022-01~2023-08]
         ↑ 包含了所有验证集数据
```

**后果**:
- 模型在全量数据上训练时，看到了"未来的验证集"
- 这会导致模型在训练时记忆未来数据
- **但测试集是真正的未来，模型无法预测**

---

### 缺陷3: 评估指标不匹配（目标错位）

**训练目标** (第2900-2904行):
```python
'objective': 'lambdarank',
'metric': 'ndcg',
'ndcg_eval_at': [1, 3],
```

**评估指标** (第2721-2775行):
```python
def _calculate_val_metric(y_pred, y_real, dates, metric='composite'):
    # Top-1 Sharpe + Median 组合
    sharpe = (mean_ret / std_ret) * 15.87
    median_score = median_ret * 500.0
    final_score = sharpe_score + median_score
```

**问题分析**:
1. **训练目标**: NDCG@1 和 NDCG@3（排序质量）
2. **评估目标**: Sharpe + Median（收益质量）
3. **测试目标**: Top-1 收益率

**错位问题**:
```
优化目标: NDCG (排序)
    ↓
评估目标: Sharpe (收益)
    ↓
实际目标: Top-1 收益 (最终)
```

**根本矛盾**:
- **NDCG 优化** → 排序准确，但不保证收益
- **Sharpe 优化** → 风险调整收益，但测试时不考虑
- **Top-1 优化** → 只关注第一名，不考虑风险

**示例**:
```
预测排序: 股票A (0.95), 股票B (0.90), 股票C (0.10)
实际排序: 股票A (+2%), 股票B (-1%), 股票C (-10%)
NDCG: 高 (排序准确)
Sharpe: 中 (A涨B跌抵消)
Top-1: 选中A → +2% ✓

预测排序: 股票A (0.80), 股票B (0.75), 股票C (0.05)
实际排序: 股票A (-0.5%), 股票B (+3%), 股票C (+2%)
NDCG: 高 (排序部分正确)
Sharpe: 高 (B和C都涨)
Top-1: 选中A → -0.5% ✗ (预测最高的反而跌)
```

---

### 缺陷4: LambdaRank 目标函数的局限性

**当前配置** (第2900-2904行):
```python
'objective': 'lambdarank',
'lambdarank_norm': True,
'lambdarank_truncation_level': [5, 10, 30],
```

**LambdaRank 的工作原理**:
- 优化成对排序（pairwise ranking）
- 只关注**截断位置以上**的排序（如 top-5, top-10, top-30）
- **不关心**截断位置以下的样本

**问题**:
```
测试时: 只使用 Top-1
训练时: 优化 Top-5/10/30 的排序

错位:
训练优化: [第1名 vs 第2-30名]
测试使用: [只看第1名]

结果:
- 模型可能擅长区分 top-30
- 但不擅长选准 top-1
```

**示例**:
```
预测: A(0.9), B(0.85), C(0.8), D(0.75), E(0.7), ...
实际: B(+5%), C(+4%), A(+1%), D(-1%), E(-2%), ...

LambdaRank 评估 (truncation_level=5):
  优化 A vs B/C/D/E 的排序
  虽然 A 排第1，但实际收益不是最高

Top-1 选择:
  选中A → 只赚1%，不如选B(+5%)
```

---

### 缺陷5: 交叉验证的边界泄露

**问题代码**:
```python
# 第2880-2887行
for train_date_idx, val_date_idx in tscv.split(unique_dates):
    train_dates_split = unique_dates[train_date_idx]
    val_dates_split = unique_dates[val_date_idx]
    train_mask = np.isin(dates_all, train_dates_split)
    val_mask = np.isin(dates_all, val_dates_split)
```

**问题分析**:
TimeSeriesSplit 在时间序列上的问题：
```
Fold 1:
  Train: [Date1~Date100]
  Val:   [Date101~Date200]

Fold 2:
  Train: [Date1~Date200]  ← ← 包含了 Fold 1 的验证集！
  Val:   [Date201~Date300]

Fold 3:
  Train: [Date1~Date300]  ← ← 包含了 Fold 1 和 2 的验证集！
  Val:   [Date301~Date400]
```

**边界泄露问题**:
- Fold 2 的训练集包含了 Fold 1 的验证集
- Fold 3 的训练集包含了 Fold 1 和 2 的验证集
- **这不是真正的"时间序列交叉验证"**

**正确做法应该是**:
```
Fold 1: Train[1~100], Val[101~200]
Fold 2: Train[1~200], Val[201~300]  ← 不应该看到 101~200 的数据
Fold 3: Train[1~300], Val[301~400]  ← 不应该看到 101~300 的数据

但特征计算可能使用了未来信息！
例如: 30日均线在 Date200 需要 Date171~Date200
如果 Date171~Date200 包含 Fold 1 的验证集数据
就产生了前视偏差
```

---

### 缺陷6: 权重计算过于复杂（引入噪声）

**权重计算** (第2847-2849行):
```python
w_train_raw = np.ones(len(y_train_raw))  # ← 简化为全1
w_val_raw = np.ones(len(y_val_raw))
```

**但实际调用** (第8372-8373行):
```python
unified_sample_weights = _calculate_professional_sample_weights(
    train_relevance_for_ensemble,
    time_decay_half_life=180.0,  # ← 时间衰减
    turnover_rate=train_turnover_rate,  # ← 换手率
    circ_mv=train_circ_mv,  # ← 流通市值
    label_boost_beta=0.15,
    ...
)
```

**问题分析**:
权重计算引入了多个维度：
1. 时间衰减（近期数据权重高）
2. 换手率（低换手股票权重高）
3. 流通市值（大盘股权重高）
4. label_boost（边缘样本权重调整）

**引入的噪声**:
```
理想情况:
  特征 → 模型 → 预测 → Top-1 → 收益

实际情况:
  特征 + 复杂权重 → 模型 → 预测 → Top-1 → 收益
       ↑
    权重噪声:
    - 时间衰减可能惩罚了重要历史样本
    - 换手率偏好可能导致过度交易
    - 流通市值偏好可能导致忽视小盘股
```

**问题**:
- 复杂权重**不代表更好**
- 可能引入过拟合（权重本身也是超参数）
- 可能削弱模型学习特征的能力

---

### 缺陷7: 过拟合惩罚机制失效

**惩罚逻辑** (第3004-3017行):
```python
gap = avg_train_score - avg_val_score
if gap > 0.5:
    penalty = (gap - 0.5) * 0.5
robust_score = (min_val_score * 0.7 + avg_val_score * 0.3) - penalty
```

**问题分析**:
1. **容差太大**: `gap > 0.5` 意味着允许训练集比验证集高 0.5 分
   - 如果 avg_val_score = 2.0, avg_train_score = 2.5
   - gap = 0.5，不惩罚
   - 但训练集比验证集高 25%

2. **惩罚系数太小**: `(gap - 0.5) * 0.5`
   - 如果 gap = 1.0，penalty = 0.25
   - 如果 min_val_score = 2.0，robust_score = 2.0 * 0.7 + avg * 0.3 - 0.25
   - 惩罚相对于得分来说太小

3. **只看平均分**:
   - `avg_train_score` 是所有 fold 的平均
   - 不考虑某些 fold 严重过拟合的情况

**示例**:
```
Fold 1: Train=3.0, Val=1.0, gap=2.0
Fold 2: Train=2.5, Val=2.0, gap=0.5
Fold 3: Train=2.2, Val=2.1, gap=0.1

avg_train = 2.57, avg_val = 1.7
gap = 0.87
penalty = (0.87 - 0.5) * 0.5 = 0.185 ← 很小

robust_score = (1.0 * 0.7 + 1.7 * 0.3) - 0.185
            = 0.7 + 0.51 - 0.185
            = 1.025

虽然 Fold 1 严重过拟合，但因为其他 fold 表现好，整体得分还行
```

---

### 缺陷8: 特征重要性不代表预测能力

**特征重要性计算** (第3057-3066行):
```python
importance = final_model.feature_importance(importance_type='gain')
feature_imp_dict = dict(zip(feature_columns, importance))
```

**问题分析**:
- `importance_type='gain'` 计算的是**特征带来的总分裂增益**
- **高增益 ≠ 高预测准确率**
- 可能特征被大量使用，但预测能力一般

**示例**:
```
特征A: 增益=1000，使用次数=1000，每次增益=1
特征B: 增益=800，使用次数=100，每次增益=8

特征A的重要性更高，但特征B的"效率"更高
```

**问题**:
- 特征A可能是"噪声特征"，使用多但贡献小
- 特征B可能是"强特征"，使用少但贡献大
- 当前逻辑无法区分

---

### 缺陷9: 缺少真正的 Walk-Forward 验证

**当前实现** (第3031-3054行):
```python
# 全量训练
full_ds = lgb.Dataset(X_all, label=y_all, group=full_groups, ...)
final_model = lgb.train(best_params, full_ds, num_boost_round=final_boost_round, ...)
```

**问题分析**:
- **全量训练**使用了所有数据（Train + Val）
- **没有 Walk-Forward 验证**
- **没有测试集验证**

**Walk-Forward 应该是**:
```
Period 1: Train[1~200], Test[201~250]
Period 2: Train[1~250], Test[251~300]
Period 3: Train[1~300], Test[301~350]
...

每个 Period:
  1. 用 Train 训练模型
  2. 在 Test 上评估
  3. 记录 Test 的表现
```

**当前做法**:
```
只用 Train+Val 训练一次模型
在最后 20% 数据上"伪验证"
没有真正的向前滚动验证
```

---

## 💡 大师级解决方案

### 方案1: 引入真正的测试集（最优先）⭐⭐⭐⭐⭐

#### 1.1 重新划分数据集

```
原始数据: [2022-01 ~ 2023-12]
         ↓
按时间划分:
  Train: [2022-01 ~ 2023-06]  (70%)
  Val:   [2023-07 ~ 2023-09]  (15%)
  Test:  [2023-10 ~ 2023-12]  (15%)  ← 新增

重要: Test 必须在 Val 之后（真正的未来数据）
```

#### 1.2 修改训练流程

```python
# 步骤1: 在 Train + Val 上做交叉验证和参数搜索
tscv = TimeSeriesSplit(n_splits=3)
folds = create_folds(train_dates + val_dates)

# 步骤2: 参数搜索
for params in search_space:
    for fold in folds:
        train_model(params, fold.train_data, fold.val_data)

    # 评估：使用 Val 的 Max-Min Sharpe
    robust_score = evaluate(params, val_data)

# 步骤3: 选择最优参数
best_params = select_best(robust_scores)

# 步骤4: 全量训练（只用 Train + Val）
final_model = train(best_params, train_data + val_data)

# 步骤5: 测试集评估（关键！）
test_performance = evaluate(final_model, test_data)
logger.info(f"Test Sharpe: {test_performance['sharpe']}")
```

#### 1.3 测试集评估指标

```python
# 必须在测试集上评估：
1. Sharpe Ratio
2. Top-1 Return
3. Top-3 Return
4. Win Rate
5. Max Drawdown

如果测试集表现远低于验证集，说明：
- 过拟合验证集
- 特征失效
- 市场风格切换
```

---

### 方案2: 使用 Purged TimeSeriesSplit（防止信息泄露）⭐⭐⭐⭐⭐

#### 2.1 Purged CV 原理

```
传统 TimeSeriesSplit:
  Fold 1: Train[1~100], Val[101~200]
  Fold 2: Train[1~200], Val[201~300]  ← 泄露！

Purged TimeSeriesSplit:
  Fold 1: Train[1~100], Val[101~200], Purged[201~210]
  Fold 2: Train[1~100], Purged[101~210], Val[211~310]
                                   ↑
  训练和验证之间留出 gap，防止特征泄露
```

#### 2.2 实现方案

```python
class PurgedTimeSeriesSplit:
    def __init__(self, n_splits=3, gap_size=20):
        self.n_splits = n_splits
        self.gap_size = gap_size  # 训练和验证之间的 gap

    def split(self, dates):
        fold_size = len(dates) // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = i * fold_size
            val_start = train_end + 1 + self.gap_size  # ← 加 gap
            val_end = val_start + fold_size

            train_dates = dates[:train_end]
            val_dates = dates[val_start:val_end]

            yield train_dates, val_dates
```

#### 2.3 使用 Purged CV

```python
# 替换第2870行
purged_cv = PurgedTimeSeriesSplit(n_splits=3, gap_size=20)
folds = list(purged_cv.split(unique_dates))

# 后续逻辑不变
for fold_idx, (train_dates, val_dates) in enumerate(folds):
    train_mask = np.isin(dates_all, train_dates)
    val_mask = np.isin(dates_all, val_dates)
    ...
```

---

### 方案3: 对齐训练目标和评估目标（关键）⭐⭐⭐⭐⭐

#### 3.1 问题根源

```
当前:
  训练目标: NDCG (排序)
  评估目标: Sharpe + Median (收益)
  测试目标: Top-1 收益

三个目标不一致！
```

#### 3.2 解决方案：统一为 Sharpe

**选项A**: 使用 Regression 目标
```python
# 改用回归目标（直接预测收益）
'objective': 'regression',
'metric': 'rmse',

# 标签使用未来收益
label = future_return  # 而不是 relevance rank

# 评估
def evaluate_sharpe(y_pred, y_true, dates):
    """直接优化 Sharpe"""
    daily_returns = select_top1(y_pred, y_true, dates)
    sharpe = calculate_sharpe(daily_returns)
    return sharpe
```

**选项B**: 自定义 LambdaRank 目标
```python
# 使用 Python API 自定义目标函数
def custom_lambda_rank(preds, train_data):
    """自定义 LambdaRank，直接优化 Sharpe"""
    labels = train_data.get_label()

    # 计算成对梯度时，考虑 Sharpe
    grad, hess = lambda_rank_sharpe_grad(
        preds, labels,
        top_k=1,  # 只关注 Top-1
        sharpe_weight=1.0  # Sharpe 权重
    )

    return grad, hess

# 训练时使用自定义目标
lgb.train(
    params,
    train_set,
    fobj=custom_lambda_rank,  # ← 自定义目标
    ...
)
```

**选项C**: 使用 Learning to Rank + 后处理
```python
# 保持 LambdaRank 目标
# 但在预测时，不只是选 Top-1

def predict_with_risk_adjustment(model, X, market_data):
    """预测时考虑风险"""
    raw_scores = model.predict(X)

    # 风险调整
    for date, stocks in group_by_date(X):
        scores = raw_scores[stocks]
        volatility = market_data[stocks]['volatility']

        # 风险调整后的分数
        adjusted_scores = scores / (1 + volatility)

        # 选调整后的 Top-1
        top1 = stocks[adjusted_scores.argmax()]

    return top1_predictions
```

---

### 方案4: 增加 Top-1 专门的评估和优化⭐⭐⭐⭐

#### 4.1 Top-1 专项评估

```python
def top1_accuracy_score(y_pred, y_true, dates):
    """
    专门评估 Top-1 的准确率

    Returns:
        top1_hit_rate: Top-1 命中率
        top1_avg_return: Top-1 平均收益
        top1_sharpe: Top-1 Sharpe
    """
    df = pd.DataFrame({
        'date': dates,
        'pred': y_pred,
        'return': y_true
    })

    # 按 date 和 pred 排序，取 Top-1
    top1_returns = df.sort_values(['date', 'pred'], ascending=[True, False]) \
                    .groupby('date').head(1)['return'].values

    # Top-1 指标
    top1_hit_rate = (top1_returns > 0).mean()
    top1_avg_return = top1_returns.mean()
    top1_sharpe = calculate_sharpe(top1_returns)

    return {
        'hit_rate': top1_hit_rate,
        'avg_return': top1_avg_return,
        'sharpe': top1_sharpe
    }
```

#### 4.2 Top-1 专项优化

```python
# 在交叉验证中，加入 Top-1 评估
for fold_idx, (train_mask, val_mask) in enumerate(folds):
    tmp_model = lgb.train(...)

    # 评估 1: NDCG（原始指标）
    ndcg_score = evaluate_ndcg(tmp_model, X_val, y_val)

    # 评估 2: Top-1 Sharpe（关键！）
    top1_metrics = top1_accuracy_score(
        tmp_model.predict(X_val), y_val, dates_val
    )

    # 综合评分
    final_score = (
        ndcg_score * 0.3 +  # 排序质量
        top1_metrics['sharpe'] * 0.7  # Top-1 收益
    )
```

---

### 方案5: 特征工程改进⭐⭐⭐⭐

#### 5.1 添加时间衰减特征

```python
def add_time_decay_features(X, dates, decay_halflife=[30, 60, 90]):
    """
    添加时间衰减特征

    原理:
    - 近期特征更重要
    - 远期特征逐渐衰减
    """
    X_new = X.copy()

    for halflife in decay_halflife:
        # 计算每个样本的时间权重
        reference_date = dates.max()
        days_ago = (reference_date - dates).days

        # 指数衰减
        time_weight = np.exp(-np.log(2) * days_ago / halflife)

        # 原始特征 × 时间权重
        X_new = np.hstack([X_new, X * time_weight.reshape(-1, 1)])

    return X_new
```

#### 5.2 添加交叉特征

```python
def add_cross_features(X, feature_names, top_k=10):
    """
    添加重要特征的交叉项

    原理:
    - 单特征可能不够
    - 特征组合可能更有预测力
    """
    # 选择 Top-K 重要特征
    importance = model.feature_importance()
    top_features = np.argsort(importance)[-top_k:]

    # 生成交叉特征
    cross_features = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            feat_i = top_features[i]
            feat_j = top_features[j]

            # 相乘交叉
            cross_feat = X[:, feat_i] * X[:, feat_j]
            cross_features.append(cross_feat)

    return np.hstack([X, np.array(cross_features).T])
```

#### 5.3 滚动窗口统计特征

```python
def add_rolling_features(X, dates, windows=[5, 10, 20]):
    """
    添加滚动窗口统计特征

    原理:
    - 不仅看当前值
    - 还要看历史趋势
    """
    X_rolling = []

    for window in windows:
        # 对每个特征计算滚动统计
        for col in range(X.shape[1]):
            # 滚动均值
            rolling_mean = pd.Series(X[:, col]) \
                           .rolling(window, min_periods=1) \
                           .mean().values

            # 滚动标准差
            rolling_std = pd.Series(X[:, col]) \
                          .rolling(window, min_periods=1) \
                          .std().fillna(0).values

            X_rolling.append(rolling_mean)
            X_rolling.append(rolling_std)

    return np.hstack([X, np.array(X_rolling).T])
```

---

### 方案6: 集成学习（Ensemble）⭐⭐⭐⭐

#### 6.1 Bagging

```python
def train_ensemble_with_bagging(X_train, y_train, n_models=5):
    """
    训练多个模型，每个使用不同的数据采样
    """
    models = []

    for i in range(n_models):
        # Bootstrap 采样
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_sample = X_train[indices]
        y_sample = y_train[indices]

        # 训练模型
        model = lgb.train(params, lgb.Dataset(X_sample, y_sample))
        models.append(model)

    return models

def predict_ensemble(models, X):
    """集成预测"""
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0)  # 平均
```

#### 6.2 时间窗口集成

```python
def train_time_ensemble(dates, X, y, window_size=180):
    """
    在不同时间窗口上训练模型
    """
    models = {}

    unique_dates = np.unique(dates)
    for i in range(0, len(unique_dates), window_size):
        window_dates = unique_dates[i:i+window_size]
        mask = np.isin(dates, window_dates)

        # 训练该窗口的模型
        X_window = X[mask]
        y_window = y[mask]

        model = lgb.train(params, lgb.Dataset(X_window, y_window))
        models[f'model_{i}'] = model

    return models

def predict_time_ensemble(models, X, dates, current_date):
    """
    使用最近的窗口模型预测
    """
    # 找到最近的模型
    for model_name in sorted(models.keys(), reverse=True):
        model_date = extract_date_from_name(model_name)
        if model_date <= current_date:
            return models[model_name].predict(X)

    # 如果没有合适的，使用最早的模型
    return models[list(models.keys())[0]].predict(X)
```

---

### 方案7: 超参数搜索策略改进⭐⭐⭐⭐

#### 7.1 Bayesian Optimization

```python
import optuna

def objective(trial):
    """
    Optuna 优化目标
    """
    params = {
        'learning_rate': trial.suggest_loguniform('lr', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 7, 127),
        'min_data_in_leaf': trial.suggest_int('min_data', 50, 500),
        'feature_fraction': trial.suggest_uniform('feat_frac', 0.4, 0.9),
        'bagging_fraction': trial.suggest_uniform('bag_frac', 0.4, 0.9),
        'reg_alpha': trial.suggest_loguniform('reg_a', 0.01, 20),
        'reg_lambda': trial.suggest_loguniform('reg_l', 0.01, 20),
    }

    # 交叉验证
    cv_scores = []
    for fold in folds:
        model = lgb.train(params, fold.train_data)
        score = evaluate_sharpe(model, fold.val_data)
        cv_scores.append(score)

    # 返回平均 Sharpe（Optuna 会自动最大化）
    return np.mean(cv_scores)

# 运行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
```

#### 7.2 Hyperband

```python
def hyperband_search(X, y, dates, max_iter=100):
    """
    Hyperband 搜索

    原理:
    - 快速淘汰表现差的参数组合
    - 在有潜力的参数上多花时间
    """
    for bracket in range(max_iter):
        # 随机采样参数
        params_list = [sample_random_params() for _ in range(n)]

        # 部分训练
        for params in params_list:
            model = lgb.train(params, data, num_boost_round=bracket)
            score = evaluate(model)

            # 淘汰表现差的
            if score < threshold:
                continue

            # 有潜力的继续训练
            model = lgb.train(params, data, num_boost_round=max_iter-bracket, init_model=model)
```

---

### 方案8: 后处理策略（Post-processing）⭐⭐⭐⭐

#### 8.1 风险过滤

```python
def predict_with_risk_filter(model, X, market_data, threshold=0.05):
    """
    预测时过滤高风险股票
    """
    raw_scores = model.predict(X)

    predictions = []
    for i, score in enumerate(raw_scores):
        stock_volatility = market_data[i]['volatility']

        # 高风险股票降低分数
        if stock_volatility > threshold:
            adjusted_score = score * 0.5
        else:
            adjusted_score = score

        predictions.append(adjusted_score)

    return predictions
```

#### 8.2 行业中性化

```python
def predict_sector_neutral(model, X, sectors):
    """
    行业中性化预测

    原理:
    - 在每个行业内做排名
    - 而不是全局排名
    """
    raw_scores = model.predict(X)

    neutral_scores = np.zeros_like(raw_scores)

    for sector in np.unique(sectors):
        mask = sectors == sector
        sector_scores = raw_scores[mask]

        # 行业内排名
        sector_ranks = rankdata(sector_scores)
        neutral_scores[mask] = sector_ranks

    return neutral_scores
```

#### 8.3 动态阈值

```python
def predict_with_dynamic_threshold(model, X, dates, window=20):
    """
    动态阈值：根据历史表现调整选择标准

    原理:
    - 如果最近表现差，提高阈值（更严格）
    - 如果最近表现好，降低阈值（更激进）
    """
    predictions = model.predict(X)

    # 计算历史 Sharpe
    historical_sharpe = calculate_rolling_sharpe(dates, window)

    # 动态调整
    if historical_sharpe[-1] < 0:  # 最近表现差
        threshold = np.percentile(predictions, 90)  # 只选前10%
    else:  # 最近表现好
        threshold = np.percentile(predictions, 70)  # 选前30%

    # 超过阈值的股票
    selected = predictions > threshold

    return selected
```

---

### 方案9: 样本权重简化⭐⭐⭐

#### 9.1 使用简单权重

```python
# 当前复杂权重 → 简化权重

# 方案A: 时间衰减（只用一个维度）
def simple_time_decay_weights(dates, halflife=180):
    """
    简单时间衰减权重
    """
    reference_date = dates.max()
    days_ago = (reference_date - dates).days

    weights = np.exp(-np.log(2) * days_ago / halflife)

    # 归一化
    weights = weights / weights.sum()

    return weights

# 方案B: 均匀权重（最简单）
def uniform_weights(len):
    """
    均匀权重
    """
    return np.ones(len) / len

# 方案C: 基于 relevance 的简单权重
def simple_relevance_weights(relevance):
    """
    基于 relevance 的简单权重

    原理:
    - relevance 高（好股票）权重高
    - relevance 低（差股票）权重低
    """
    # relevance 是 0-3 的整数
    # 映射到权重
    weight_map = {0: 0.5, 1: 0.8, 2: 1.0, 3: 1.2}
    weights = np.array([weight_map[r] for r in relevance])

    # 归一化
    weights = weights / weights.mean()

    return weights
```

---

### 方案10: 真正的 Walk-Forward 验证⭐⭐⭐⭐⭐

#### 10.1 Walk-Forward 实现框架

```python
def walk_forward_validation(dates, X, y,
                           train_size=252,  # 约1年
                           val_size=63,     # 约1季度
                           test_size=63):    # 约1季度
    """
    Walk-Forward 验证

    结构:
      Train: 252天
      Val:   63天 (参数选择)
      Test:  63天 (真实评估)

    然后滚动:
      Train: 252+63=315天
      Val:   63天
      Test:  63天
    """
    results = []

    # 总样本数
    n_samples = len(dates)

    # 计算滚动次数
    step = val_size + test_size  # 每次滚动的步长
    n_walks = (n_samples - train_size) // step

    for i in range(n_walks):
        # 计算索引
        train_start = 0
        train_end = train_start + train_size
        val_start = train_end
        val_end = val_start + val_size
        test_start = val_end
        test_end = test_start + test_size

        # 提取数据
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]

        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]

        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        # 参数搜索（在 Val 上）
        best_params = parameter_search(X_train, y_train, X_val, y_val)

        # 训练（使用 Train + Val）
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])

        model = lgb.train(best_params, lgb.Dataset(X_train_val, y_train_val))

        # 测试（在 Test 上）
        test_metrics = evaluate(model, X_test, y_test)

        results.append({
            'walk': i,
            'test_period': (dates[test_start], dates[test_end]),
            'test_sharpe': test_metrics['sharpe'],
            'test_return': test_metrics['return'],
            'params': best_params
        })

        logger.info(f"Walk {i+1}/{n_walks}: Test Sharpe={test_metrics['sharpe']:.2f}")

    return results
```

#### 10.2 分析 Walk-Forward 结果

```python
def analyze_walk_forward_results(walk_results):
    """
    分析 Walk-Forward 结果
    """
    test_sharpes = [r['test_sharpe'] for r in walk_results]

    # 统计
    mean_sharpe = np.mean(test_sharpes)
    std_sharpe = np.std(test_sharpes)

    win_rate = (np.array(test_sharpes) > 0).mean()

    max_drawdown = calculate_max_drawdown(test_sharpes)

    logger.info("="*60)
    logger.info("Walk-Forward 验证结果:")
    logger.info(f"  平均 Sharpe: {mean_sharpe:.2f}")
    logger.info(f"  标准差: {std_sharpe:.2f}")
    logger.info(f"  胜率: {win_rate*100:.1f}%")
    logger.info(f"  最大回撤: {max_drawdown:.2f}")
    logger.info("="*60)

    return {
        'mean_sharpe': mean_sharpe,
        'std_sharpe': std_sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown
    }
```

---

## 📊 推荐的实施优先级

### 🔴 紧急（立即实施）

1. **引入真正的测试集** (方案1)
   - 重新划分 Train/Val/Test
   - 在 Test 上评估
   - 诊断过拟合程度

2. **使用 Purged TimeSeriesSplit** (方案2)
   - 防止信息泄露
   - 更干净的交叉验证

3. **对齐训练目标和评估目标** (方案3)
   - 统一为 Sharpe 优化
   - 或者直接优化 Top-1 收益

### 🟡 重要（1-2周内）

4. **增加 Top-1 专项评估** (方案4)
   - 添加 Top-1 命中率
   - 添加 Top-1 Sharpe

5. **样本权重简化** (方案9)
   - 去除复杂的多维度权重
   - 使用简单时间衰减

6. **特征工程改进** (方案5)
   - 时间衰减特征
   - 交叉特征

### 🟢 优化（长期）

7. **集成学习** (方案6)
   - Bagging
   - 时间窗口集成

8. **超参数搜索改进** (方案7)
   - Bayesian Optimization
   - Hyperband

9. **后处理策略** (方案8)
   - 风险过滤
   - 行业中性化

10. **Walk-Forward 验证** (方案10)
    - 完整的滚动验证
    - 更真实的评估

---

## 🎯 核心建议

### 最关键的3个改动

1. **引入测试集**: 诊断真正的泛化能力
2. **Purged CV**: 防止信息泄露
3. **统一目标**: 训练和评估都用 Sharpe

### 预期效果

- **测试集 Sharpe > 0**: 基本可用
- **测试集 Sharpe > 1.0**: 良好
- **测试集 Sharpe > 2.0**: 优秀

---

**分析完成时间**: 2026-01-30
**建议**: 优先实施紧急方案（方案1-3），然后逐步实施其他方案
