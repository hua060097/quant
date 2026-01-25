# 量化交易特征工程分析报告

## 1. 问题识别：不适合直接输入模型的基础特征

### 1.1 价格类特征（存在严重问题）
**问题特征：**
- `open`, `high`, `low`, `close` - 绝对价格值
- `volume`, `amount` - 绝对成交量/金额

**问题分析：**
1. **尺度差异**：不同股票价格差异巨大（几元到几百元），导致模型偏向高价股
2. **非平稳性**：绝对价格包含趋势和季节性，不符合机器学习假设
3. **无法跨股票比较**：10元股票涨到11元 vs 100元股票涨到110元，涨幅相同但绝对值差异巨大

### 1.2 技术指标类特征（部分问题）
**问题特征：**
- `ma5`, `ma10`, `ma20`, `ma30`, `ma60`, `ma120` - 绝对均线值
- `ema12`, `ema26` - 绝对指数移动平均值
- `bb_upper`, `bb_middle`, `bb_lower` - 绝对布林带值

**问题分析：**
1. **绝对值依赖**：均线的绝对值受股价影响，无法标准化比较
2. **缺乏相对性**：没有体现价格相对于均线的位置关系

### 1.3 成交量类特征（部分问题）
**问题特征：**
- `volume` - 绝对成交量
- `amount` - 绝对成交金额

**问题分析：**
1. **股本差异**：不同股票流通股本差异巨大
2. **行业差异**：不同行业成交活跃度差异显著

## 2. 推荐的高级衍生特征

### 2.1 价格相对特征（替代绝对价格）
```python
# 价格位置特征
'price_position_in_range_5d'   # 当前价格在5日高低点中的位置 [0,1]
'price_position_in_range_20d'  # 当前价格在20日高低点中的位置 [0,1]
'price_position_in_range_60d'  # 当前价格在60日高低点中的位置 [0,1]

# 价格相对强度
'price_vs_ma5_ratio'           # 当前价格/MA5 - 1
'price_vs_ma20_ratio'          # 当前价格/MA20 - 1
'price_vs_ma60_ratio'          # 当前价格/MA60 - 1

# 价格动量特征
'price_momentum_5d'            # (close - close_5d_ago) / close_5d_ago
'price_momentum_20d'           # (close - close_20d_ago) / close_20d_ago
'price_acceleration_5d'        # 5日动量的变化率
```

### 2.2 均线相对特征（替代绝对均线）
```python
# 均线排列特征
'ma_alignment_score'           # 均线多头/空头排列强度 [-1,1]
'ma_convergence_score'         # 均线收敛程度 [0,1]
'ma_slope_5d'                  # MA5的5日斜率（标准化）
'ma_slope_20d'                 # MA20的5日斜率（标准化）

# 均线距离特征
'ma5_ma20_distance_pct'        # (MA5 - MA20) / MA20
'ma20_ma60_distance_pct'       # (MA20 - MA60) / MA60
'price_ma_distance_zscore'     # 价格偏离均线的Z-Score
```

### 2.3 成交量相对特征（替代绝对成交量）
```python
# 成交量相对强度
'volume_ratio_5d'              # 当日成交量 / 5日平均成交量
'volume_ratio_20d'             # 当日成交量 / 20日平均成交量
'volume_percentile_60d'        # 当日成交量在60日内的分位数 [0,1]

# 量价关系特征
'volume_price_correlation_20d' # 20日量价相关性
'volume_weighted_return'       # 成交量加权收益率
'volume_surge_indicator'       # 成交量异常放大指标 [0,1]
```

### 2.4 波动率标准化特征
```python
# 标准化波动率
'volatility_zscore_20d'        # 20日波动率Z-Score
'volatility_percentile_60d'    # 波动率在60日内的分位数
'volatility_regime_indicator'  # 波动率状态指标（低/中/高）

# 波动率结构特征
'realized_vs_implied_vol'      # 已实现波动率 vs 隐含波动率
'vol_term_structure_slope'     # 波动率期限结构斜率
```

### 2.5 布林带相对特征（替代绝对布林带）
```python
# 布林带位置特征（已存在但需验证）
'bb_position'                  # 价格在布林带中的位置 [0,1]
'bb_width_percentile'          # 布林带宽度分位数
'bb_squeeze_indicator'         # 布林带收缩指标

# 布林带突破特征
'bb_breakout_strength'         # 突破强度（距离带边的标准差倍数）
'bb_mean_reversion_score'      # 均值回归概率评分
```

## 3. 特征工程最佳实践

### 3.1 标准化原则
1. **相对化**：将绝对值转换为相对值或比率
2. **标准化**：使用Z-Score或分位数标准化
3. **时间窗口**：使用滚动窗口计算，避免未来信息泄露

### 3.2 特征组合策略
1. **多时间框架**：结合短期（5日）、中期（20日）、长期（60日）特征
2. **交叉验证**：不同指标的交叉确认（如量价配合）
3. **状态识别**：识别市场状态（趋势/震荡/突破）

### 3.3 避免的常见错误
1. **直接使用绝对价格**：应转换为收益率或相对位置
2. **忽略股本差异**：成交量应标准化处理
3. **未来信息泄露**：确保所有特征都基于历史数据

## 4. 现有特征分析结果

### 4.1 已计算的特征（97个）
**分类统计：**
- 价格衍生特征: 15个
- 技术指标衍生: 14个
- 成交量衍生: 8个
- 形态衍生: 17个（包括新的DTW相似度特征）
- 其他衍生: 43个

### 4.2 需要新增的特征（15个）

**价格位置特征（3个）：**
- `price_position_in_range_5d` - 当前价格在5日高低点中的位置
- `price_position_in_range_20d` - 当前价格在20日高低点中的位置
- `price_position_in_range_60d` - 当前价格在60日高低点中的位置

**价格相对特征（3个）：**
- `price_vs_ma5_ratio` - 当前价格相对MA5的比率
- `price_vs_ma20_ratio` - 当前价格相对MA20的比率
- `price_vs_ma60_ratio` - 当前价格相对MA60的比率

**均线相对特征（4个）：**
- `ma_alignment_score` - 均线排列评分
- `ma_convergence_score` - 均线收敛评分
- `ma5_ma20_distance_pct` - MA5与MA20的距离百分比
- `ma20_ma60_distance_pct` - MA20与MA60的距离百分比

**成交量相对特征（3个）：**
- `volume_ratio_5d` - 相对5日平均成交量比率
- `volume_ratio_20d` - 相对20日平均成交量比率
- `volume_percentile_60d` - 60日成交量分位数

**波动率特征（2个）：**
- `volatility_zscore_20d` - 20日波动率Z-Score
- `volatility_percentile_60d` - 60日波动率分位数

## 5. 实施计划

1. **第一阶段**：删除不适合的基础特征，添加高优先级衍生特征
2. **第二阶段**：验证新特征的有效性，调整特征权重
3. **第三阶段**：添加中优先级特征，优化特征组合

---
*本分析基于量化交易最佳实践和机器学习特征工程原理*
