# 分析方案：因子极端值是否支持收紧开盘跌幅阈值

## 核心问题

当前Top1的买入规则是：`gap_ratio = (buy_open - prev_close) / prev_close < -0.06` 时跳过买入。
问题是：在某些因子处于极端值时，是否需要将这个阈值从-6%收紧（如-4%）？

## 分析步骤

### Step 1: 构建主数据集
- 从日志中解析117笔真实交易的gap_ratio
- 加载28笔影子交易（已有gap_ratio和因子值）
- 从factors.parquet合并6个关键因子：sector_momentum_interaction, return_20d_intra_z, price_ma60_ratio, ma60_slope, trend_acceleration, ma_convergence_score
- 计算因子截面分位数（在同日全市场中的百分位）

### Step 2: 描述性分析
- gap分档(-6%~-4%, -4%~-2%等) × 因子极端值 × 盈亏结果的交叉表
- (-6%, -4%]档内10笔交易的详细清单（因子值+分位数+收益）
- 每个因子在该档内的大亏vs大赚对比

### Step 3: 因子条件阈值收紧评估
- 测试每条规则："当因子F处于极端区间时，将阈值从-6%收紧到-4%"
- 评估指标：新拦截的交易数、拦截的大亏数、误杀的大赚数、净收益影响
- 最少样本要求：新拦截交易>=3笔
- 稳定性检查：前后半段时间一致性

### Step 4: 影子交易反事实分析
- 4笔被-6%门控拦截的影子交易的结局
- 如果门控是-4%，它们是否能通过

### Step 5: 决策表
- 汇总所有因子条件规则的效果
- 只有当选择性比率>2且净效应为正时才建议修改

## 数据文件
- 日志: `stock_selection_test_mode_walkforward_分类模型_原因子169_optuna.log`
- 因子: `factor_lab_data/factors.parquet`
- 影子: `results/walkforward_shadow_skipped_opportunities_20260513_230111.csv`

## 输出
- `results/factor_conditional_gate_master.csv`
- 终端输出决策表
