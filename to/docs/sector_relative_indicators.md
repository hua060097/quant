# 板块相对于创业板的相对指标分析

## 概述

本文档详细介绍了量化交易系统中板块相对于创业板指数的相对指标计算方法。这些指标旨在帮助投资者识别相对强弱、评估超额收益，以及进行板块轮动策略的决策支持。

## 相对指标计算方法

### 1. 基础相对变化指标

- **相对涨幅差值 (relative_change_diff)**
  
  计算公式：`sector_change_percent - cyb_change_percent`
  
  含义：板块涨跌幅与创业板涨跌幅的差值，正值表示板块表现优于创业板，负值表示板块表现弱于创业板。

- **相对涨幅比值 (relative_change_ratio)**
  
  计算公式：`sector_change_percent / cyb_change_percent`（当创业板涨跌幅接近0时使用差值代替）
  
  含义：板块涨跌幅与创业板涨跌幅的比值，大于1表示板块涨幅大于创业板，小于1表示板块涨幅小于创业板。

### 2. 进阶相对强度指标

- **相对强度指数 (relative_strength_index)**
  
  计算公式：采用指数平滑方式 `alpha * relative_change_diff + (1 - alpha) * previous_rsi`
  
  含义：通过指数平滑处理的板块相对于创业板的累积相对强度，减少单日异常值的影响，反映中期相对强弱趋势。
  
  参数：alpha = 0.3（平滑系数）

### 3. 风险调整指标

- **Beta估计值 (beta_estimate)**
  
  计算公式：`sector_volatility / market_volatility`
  
  含义：衡量板块相对于市场的波动性，Beta大于1表示板块波动大于市场，小于1表示波动小于市场。

- **Alpha估计值 (alpha_estimate)**
  
  计算公式：`sector_change_percent - beta_estimate * cyb_change_percent`
  
  含义：风险调整后的超额收益，正值表示板块相对于其风险水平取得了超额收益。

### 4. 相对强度分类

- **相对强度类别 (relative_strength_category)**
  
  分类标准：
  - `stronger`：当relative_change_diff > 0.5%
  - `weaker`：当relative_change_diff < -0.5%
  - `neutral`：当relative_change_diff在-0.5%到0.5%之间
  
  含义：将板块相对强弱程度分类，便于直观判断板块相对表现。

## 应用场景

1. **板块轮动策略**：利用相对强度指数识别强势板块，进行择时轮动。
   
2. **超额收益分析**：通过Alpha估计值评估板块的风险调整后收益，识别具有持续超额收益的板块。
   
3. **风险控制**：结合Beta估计值对投资组合进行风险评估和调整。
   
4. **趋势跟踪**：利用相对强度类别变化捕捉板块由弱转强或由强转弱的转折点。

## 指标解读示例

- 当relative_strength_index持续为正且增加，同时relative_strength_category为"stronger"时，表明该板块处于相对强势，可能具有持续上涨的动力。
   
- 当relative_change_diff由负转正，且alpha_estimate转为正值时，可能意味着板块正在形成相对强势，是潜在的买入时机。
   
- 当beta_estimate较高而alpha_estimate为负时，表明板块承担了较高风险但未能获得相应回报，投资者应当谨慎。

## 注意事项

1. 相对指标应结合板块基本面、市场环境等综合分析，不宜单独作为投资决策依据。
   
2. Beta和Alpha估计值是近似计算，不同于传统CAPM模型的精确值，仅供参考。
   
3. 相对强度阈值(0.5%)可根据市场波动特性和投资策略进行调整。

## 后续优化方向

1. 引入多时间周期的相对强度分析，如5日、10日、20日相对强度对比。
   
2. 考虑板块特性差异，对不同性质板块设置差异化的相对强度阈值。
   
3. 开发基于相对指标的自适应交易信号系统。 