# RISK RULE ANALYSIS FRAMEWORK

量化风控规则的设计、分析、验证框架。用于在回测日志上测试风控规则组合，评估拦截大亏 / 误杀盈利的效果及阈值鲁棒性。

## 前置要求

- **两份回测日志**：无风控版（基准） + 风控版（含现有规则），用于对比分析
- **因子 parquet 文件**：包含所有候选因子，T日截面值
- **影子满仓机会 CSV**：模型选股但仓位已满的模拟交易日志（可选，用于补充验证）
- 回测日志中每笔卖出必须有 `buy_date`（买入日/T+1日）

## 分析流程

### Step 1: 数据提取

从回测日志中提取所有交易的 `stock_code` + `buy_date` + `return_pct`，与因子 parquet 按 `stock_code` 和 `prediction_date`（=`buy_date` 前一个交易日）匹配，得到每笔交易的因子值表。

```python
# 匹配逻辑：prediction_date = buy_date 前最后一个交易日
buy_dt = pd.Timestamp(buy_date)
before = stock_data[stock_data['date'] < buy_dt]
pred_date = before['date'].max()
```

### Step 2: 因子分离度分析

计算每个因子在 **大亏(< -5%)** 和 **大赚(> 10%)** 两组间的分离度：

```
separation = abs(loss_mean - win_mean) / pooled_std
```

排序后取 Top 20-30 因子作为候选。同时观察每个因子的方向性：大亏偏高还是偏低？

### Step 3: 单因子阈值扫描

对每个候选因子，扫描不同分位数阈值（p5~p50），检查拦截大亏数和误杀大赚数。保留 `catch >= 3` 且 `catch_rate > fp_rate` 的组合。

### Step 4: 组合规则扫描

用单因子扫描中表现最好的因子做 **2因子 AND 组合** 和 **大盘因子 + 个股因子组合**。大盘因子阈值用全交易样本的分位数，个股因子阈值用绝对值或截面分位。

评分函数：`score = catch - 2 * FP_bigwin`（惩罚误杀大赚）。

### Step 5: 影子满仓机会验证

对影子 CSV 中的 28 笔交易匹配因子值，检查候选规则是否误伤盈利交易。

### Step 6: 阈值鲁棒性分析

在候选阈值附近做精细 Grid Search（步长 0.01-0.02），检查：
- **零误杀边界**：FP 从 0 跳到非零的阈值点
- **Sharp boundary 警告**：阈值微小变化（如 1.0→1.5）导致 FP 剧烈跳变（0→3），说明鲁棒性不足

绝对阈值 vs 截面分位对比：
- 绝对值阈值在 mfp 等分布偏斜的因子上可能极敏感
- 截面分位（`rank(pct=True)`）自适应每日分布，通常在阈值附近更鲁棒

### Step 7: 新旧规则独立性检查

计算新规则拦截的大亏与现有规则拦截的大亏的交集，确认增量贡献。

## 分析脚本

以下脚本在项目根目录，可按需修改后运行：

| 脚本 | 用途 |
|---|---|
| `analyze_risk_rules.py` | Step 2-4: 因子分离度 + 单因子扫描 + 2-3因子组合 + 误杀详情 |
| `analyze_risk_rules_v2.py` | Step 4: 大盘因子+个股因子组合扫描，分位鲁棒性，V3漏网分析 |
| `analyze_r4_robustness.py` | Step 5-6: 影子交易验证 + Grid Search 鲁棒性 |
| `analyze_r4_robustness_v2.py` | Step 6: 四方案对比（绝对值 vs 截面分位，不同个股因子） |

## 关键注意事项

### T日 vs T+1日
- 所有因子值必须用 **预测日（T日）**的值，不是买入日（T+1日）
- 代码中 `date < buy_date` 取 max，即 T日
- 确认回测代码中的 V3 过滤用的也是 T日值：`daily_predictions[date == prediction_date]`

### 因子值格式
- 部分因子在 parquet 中有 `_pct` 后缀版本（如 `money_flow_pressure_pct`），是每日截面分位（0~1），与 `rank(pct=True)` 等效
- V3 规则使用的 6 个因子在 parquet 中是 **原始值**（非 zscore），阈值直接可比
- 新增因子（如 `post_accumulation_supply_tightness`）也是原始值 [0,1] 范围

### 样本量限制
- 150 笔交易的统计意义有限，结论仅供参考
- 零误杀大赚（0/42）时需注意置信区间：样本 42 笔大赚，0 误杀 ≠ 真实误杀率为 0
- 建议：规则正式上线前，在更多回测窗口上交叉验证

### 过度拟合风险
- 总规则数 <= 4 条（V3 已有 3 条，新增不超过 1 条）
- 每条规则用 2 个因子，逻辑清晰可解释
- 避免在交易样本上过度调参：阈值用全市场分位而非交易样本分位

### 规则可解释性模板
```
Rule X: factor_A (direction) threshold_A AND factor_B (direction) threshold_B
含义：A 意味着什么 + B 意味着什么 → 为什么同时成立时就是风险
```

## 示例：R4 分析过程

### 输入
- 162 日志（无风控，127 笔交易，28 大亏，42 大赚）
- 169 日志（V3 风控，117 笔交易，20 大亏）
- 影子 CSV（28 笔模拟交易）
- factors.parquet（201 个因子列，2023-06 ~ 2026-05）

### 过程
1. 162 日志因子匹配 → 28 大亏 vs 42 大赚因子分离度 → Top 10 候选
2. 单因子扫描 → `money_flow_pressure`、`market_liquidity_change`、`close_position_60d` 等表现最佳
3. 组合扫描 → `market_liquidity_change < 0.95 & money_flow_pressure < 1.0` 拦截 7/28，零误杀
4. 改用截面分位 → `market_liquidity_change < 0.95 & money_flow_pressure_pct < 0.30` 拦截 6/28，零误杀，鲁棒性更好
5. 影子交易验证 → Rule 4 拦 2 笔小亏，零误杀盈利
6. 与 V3 重叠检查 → 3 笔重叠，2 笔独有增量

### 结论
新增 Rule 4：同 V3 的 Rule 1-3 并列 OR，合共 4 条规则。
