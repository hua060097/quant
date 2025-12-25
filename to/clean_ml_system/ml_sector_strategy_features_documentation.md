# 机器学习选股系统特征文档

## 1. 概述

本文档详细描述了基于train_sector_models.py系统中使用的所有特征，包括个股特征、板块特征和市场特征。这些特征是机器学习选股系统的基础数据。

## 2. 个股特征 (Stock Features) - 实际特征105个

**当前状态**: 基于实际数据检查结果
**特征总数**: 105个特征
**数据来源**:
- 基础数据：xtdata API (OHLCV + 基本面数据)
- 涨跌停数据：enhanced_limit_data_collector.py
- 技术指标：generate_daily_features.py
- 高级量化特征：基于顶级量化研究论文设计
**完整率**: 100%验证

### 2.1 基础信息特征 (3个)
- **stock_code**: 股票代码
- **date**: 日期
- **date_str**: 日期字符串

### 2.2 价格特征 (10个)
- **open**: 开盘价
- **high**: 最高价
- **low**: 最低价
- **close**: 收盘价
- **bb_lower**: 布林带下轨
- **volume_weighted_price**: 成交量加权价格
- **vol_price_corr_20d**: 20日价量相关性
- **price_vol_change_ratio**: 价量变化比率
- **price_volume_divergence**: 价量背离
- **open_times**: 开盘次数

### 2.3 收益率特征 (6个)
- **pct_chg**: 涨跌幅
- **return_5d**: 5日收益率
- **return_20d**: 20日收益率
- **risk_adjusted_return**: 风险调整收益率
- **returns_zscore**: 收益率Z分数
- **future_return_2d**: 2日后收益率(目标变量)

### 2.4 成交量特征 (14个)
- **volume**: 成交量
- **amount**: 成交额
- **turnover_rate**: 换手率
- **volatility20**: 20日波动率
- **volume_zscore**: 成交量Z分数
- **parkinson_vol**: Parkinson波动率
- **vol_term_structure**: 波动率期限结构
- **volatility_skew**: 波动率偏度
- **volatility_adjusted_momentum**: 波动率调整动量
- **fd_amount**: 封单金额
- **limit_amount**: 板上成交金额
- **turnover_efficiency**: 换手率效率
- **limit_up_volume_ratio**: 涨停日成交量比率
- **limit_up_turnover_ratio**: 涨停日换手率比率

### 2.5 技术指标特征 (17个)
#### 移动平均线系列 (6个)
- **ma5**: 5日移动平均线
- **ma10**: 10日移动平均线
- **ma20**: 20日移动平均线
- **ma30**: 30日移动平均线
- **ma60**: 60日移动平均线
- **ma120**: 120日移动平均线

#### RSI指标 (3个)
- **rsi6**: 6日RSI
- **rsi12**: 12日相对强弱指数
- **rsi14**: 14日RSI

#### MACD指标 (5个)
- **ema12**: 12日指数移动平均线
- **ema26**: 26日指数移动平均线
- **macd**: MACD线
- **macd_signal**: MACD信号线
- **macd_hist**: MACD直方图

#### 其他技术指标 (3个)
- **rsi_macd_agreement**: RSI-MACD一致性
- **first_time_formatted**: 格式化的首次封板时间
- **last_time_formatted**: 格式化的最后封板时间

### 2.6 涨跌停特征 (11个) - A股特色
- **is_limit_up**: 是否涨停
- **is_limit_down**: 是否跌停
- **limit_times**: 连板数
- **limit_strength**: 涨跌停强度
- **time_to_limit_minutes**: 到达涨跌停时间(分钟)
- **limit_quality_score**: 涨跌停质量评分
- **limit_up_days_5d**: 近5日涨停次数
- **limit_down_days_5d**: 近5日跌停次数
- **limit_up_days_20d**: 近20日涨停次数
- **consecutive_limit_up**: 连续涨停天数
- **consecutive_limit_down**: 连续跌停天数

### 2.7 基本面特征 (2个)
- **total_mv**: 总市值
- **circ_mv**: 流通市值

### 2.8 市值特征 (3个)
- **float_ratio**: 流通比例
- **mv_factor**: 市值因子
- **capital_intensity**: 资金强度

### 2.9 其他特征 (39个)
#### 高级技术特征 (8个)
- **bb_upper**: 布林带上轨
- **bb_middle**: 布林带中轨
- **bb_width**: 布林带宽度
- **bb_position**: 价格在布林带中的位置
- **atr14**: 14日平均真实波动范围
- **trend_strength**: 趋势强度
- **trend_consistency**: 趋势一致性
- **bb_breakout_signal**: 布林带突破信号

#### 形态特征 (3个)
- **candlestick_strength**: K线强度
- **pattern_engulfing_bullish**: 看涨吞没形态
- **pattern_engulfing_bearish**: 看跌吞没形态

#### 支撑阻力特征 (3个)
- **risk_reward_ratio**: 风险收益比
- **near_support**: 是否接近支撑位
- **near_resistance**: 是否接近阻力位

#### 相对强度特征 (2个)
- **relative_strength_20d**: 20日相对强度
- **momentum_acceleration**: 动量加速度

#### 综合评分特征 (8个)
- **tech_momentum_signal**: 技术动量信号
- **technical_consensus**: 技术面一致性
- **composite_trading_signal**: 综合交易信号
- **momentum_score**: 动量评分
- **technical_score**: 技术评分
- **composite_score**: 综合评分
- **board_quality**: 板块质量评分
- **pattern_score**: 形态质量评分

#### 其他量化特征 (15个)
- **short_reversal**: 短期反转信号
- **amihud_illiquidity**: Amihud非流动性指标
- **overbought_oversold**: 超买超卖指标
- **multi_period_trend_agreement**: 多周期趋势一致性
- **time_efficiency_score**: 时间效率评分
- **stability_score**: 稳定性评分
- **continuity_factor**: 连板因子
- **trend_score**: 趋势一致性评分
- **first_time**: 首次封板时间
- **last_time**: 最后封板时间
- **up_stat**: 涨停统计
- **is_zhaban**: 是否炸板
- **change_pct**: 价格变化百分比
- **pre_close**: 前收盘价
- **adj_factor**: 复权因子

### 2.10 特征总结
- **总特征数**: 105个
- **基础信息**: 3个
- **价格特征**: 10个
- **收益率特征**: 6个
- **成交量特征**: 14个
- **技术指标**: 17个
- **涨跌停特征**: 11个 (A股特色)
- **基本面特征**: 2个
- **市值特征**: 3个
- **其他特征**: 39个
- **特征覆盖率**: 100%
- **数据完整性**: 已验证

## 3. 板块特征 (Sector Features) - 实际特征141个

基于实际数据检查结果，系统包含141个板块特征，分为以下类别：

### 3.1 基础信息特征 (3个)
- **date**: 日期
- **date_str**: 日期字符串
- **sector_code**: 板块代码

### 3.2 价格特征 (22个)
- **close**: 板块收盘价（加权平均）
- **open**: 板块开盘价
- **high**: 板块最高价
- **low**: 板块最低价
- **change_percent**: 板块涨跌幅
- **amplitude**: 板块振幅
- **price_ma5_ratio**: 价格与MA5比值
- **price_ma20_ratio**: 价格与MA20比值
- **price_ma60_ratio**: 价格与MA60比值
- **price_above_ma5**: 价格是否在MA5之上
- **price_above_ma20**: 价格是否在MA20之上
- **price_above_ma60**: 价格是否在MA60之上
- **close_5d_ago**: 5日前收盘价
- **close_10d_ago**: 10日前收盘价
- **close_20d_ago**: 20日前收盘价
- **change_pct**: 涨跌幅百分比
- **volume_change_ratio**: 成交量变化比率
- **limit_open_ratio**: 开盘涨停比例
- **fund_flow_momentum**: 资金流动动量
- **avg_amplitude**: 平均振幅
- **limit_open_count**: 开盘涨停数量
- **high_elasticity_count**: 高弹性股票数量

### 3.3 成交量特征 (14个)
- **volume**: 板块总成交量
- **amount**: 板块总成交额
- **turnover_rate**: 板块换手率
- **volume_ratio_5**: 5日成交量比率
- **volume_ratio_10**: 10日成交量比率
- **volume_ratio_20**: 20日成交量比率
- **volume_ratio_60**: 60日成交量比率
- **volume_trend**: 成交量趋势
- **avg_turnover_rate**: 平均换手率
- **daily_amount_count**: 日成交额计数
- **daily_amount_ratio**: 日成交额比率
- **amount_ma5**: 成交额5日均线
- **volume_intensity**: 成交量密集度
- **avg_turnover**: 平均换手

### 3.2 板块结构特征层
#### 资金流动特征（需计算生成）
- **fund_flow_momentum**: 资金流动动量（需计算：当日成交额/5日平均成交额）
- **volume_intensity**: 成交量密集度（需计算：成交量/20日波动率）
- **up_volume_ratio**: 上涨成交量比率（需计算：上涨股票成交量/总成交量）

#### 技术指标特征
##### 均线系统特征
- **ma5_10_ratio**: MA5与MA10比值
- **ma5_20_ratio**: MA5与MA20比值
- **ma10_20_ratio**: MA10与MA20比值
- **ma20_60_ratio**: MA20与MA60比值
- **ma60_120_ratio**: MA60与MA120比值
- **price_ma5_ratio**: 价格与MA5比值
- **price_ma20_ratio**: 价格与MA20比值
- **price_ma60_ratio**: 价格与MA60比值

##### 价格位置特征
- **price_above_ma5**: 价格是否在MA5之上
- **price_above_ma20**: 价格是否在MA20之上
- **price_above_ma60**: 价格是否在MA60之上

##### 趋势特征
- **trend_direction_score**: 趋势方向评分
- **ma_bull_trend**: 多头趋势信号
- **ma_bear_trend**: 空头趋势信号
- **ma_neutral_trend**: 中性趋势信号
- **ma5_slope**: MA5斜率
- **ma20_slope**: MA20斜率

##### 动量特征
- **momentum_5d**: 5日动量
- **momentum_10d**: 10日动量
- **momentum_20d**: 20日动量
- **sector_change_pct**: 板块相对变化率

#### 波动率特征
- **volatility**: 当前波动率
- **volatility_5d**: 5日波动率
- **volatility_10d**: 10日波动率
- **volatility_20d**: 20日波动率
- **volatility_60d**: 60日波动率
- **volatility_ratio**: 波动率比率

#### 成交量特征
- **volume_ratio_5**: 5日成交量比率
- **volume_ratio_10**: 10日成交量比率
- **volume_ratio_20**: 20日成交量比率
- **volume_ratio_60**: 60日成交量比率
- **volume_change_ratio**: 成交量变化比率
- **volume_trend**: 成交量趋势

#### 高级技术特征
- **rsi_macd_bull_signal**: RSI-MACD多头信号
- **rsi_macd_bear_signal**: RSI-MACD空头信号
- **momentum_volatility_ratio**: 动量波动率比率
- **volume_momentum_confirm**: 量价确认指标
- **volume_momentum_divergence**: 量价背离指标

#### 板块生命周期特征
- **trend_strength**: 趋势强度
- **consecutive_up_days**: 连续上涨天数
- **consecutive_down_days**: 连续下跌天数
- **swing_pattern**: 摆动模式

#### 支撑阻力特征
- **support_distance_pct**: 支撑位距离百分比
- **resistance_distance_pct**: 阻力位距离百分比

#### 技术评分特征
- **tech_score**: 技术评分
- **market_sentiment**: 市场情绪
- **signal_strength**: 信号强度

### 3.3 涨跌停板结构特征（基于sector_stock_analysis.py板块分析）
#### 基础统计特征
- **stock_count**: 板块内股票总数
- **up_count**: 上涨股票数量
- **down_count**: 下跌股票数量
- **flat_count**: 平盘股票数量
- **avg_pct_chg**: 平均涨跌幅
- **avg_turnover**: 平均换手率
- **avg_amplitude**: 平均振幅

#### 涨跌停统计特征
- **limit_up_count**: 涨停股票数量
- **limit_down_count**: 跌停股票数量
- **near_limit_up_count**: 接近涨停股票数量
- **near_limit_down_count**: 接近跌停股票数量
- **one_word_limit_up_count**: 一字涨停股票数量
- **broken_limit_up_count**: 炸板股票数量
- **limit_open_count**: 开盘涨停股票数量

#### 连板特征
- **max_consecutive_limit_up**: 最高连板数
- **high_elasticity_count**: 高弹性股票数量
- **board_counts_1板**: 1板股票数量
- **board_counts_2板**: 2板股票数量
- **board_counts_3板**: 3板股票数量
- **board_counts_4板**: 4板股票数量
- **board_counts_5板**: 5板股票数量
- **board_counts_6板**: 6板股票数量
- **board_counts_7板**: 7板股票数量
- **board_counts_8板**: 8板股票数量
- **board_counts_9板**: 9板股票数量
- **board_counts_10板及以上**: 10板及以上股票数量

#### 溢价特征
- **yesterday_limit_up_premium**: 昨日涨停溢价
- **yesterday_max_limit_up_premium**: 昨日最高涨停溢价
- **yesterday_broken_limit_premium**: 昨日炸板溢价

#### 市值分布特征
- **market_cap_distribution_large**: 大市值股票数量
- **market_cap_distribution_medium**: 中市值股票数量
- **market_cap_distribution_small**: 小市值股票数量

#### 连板分布特征
- **limit_board_distribution_large**: 大市值连板股票数量
- **limit_board_distribution_medium**: 中市值连板股票数量
- **limit_board_distribution_small**: 小市值连板股票数量

#### 综合评价特征
- **heat_index**: 板块热度指数
- **yesterday_limit_up_premium**: 昨日涨停溢价
- **yesterday_max_limit_up_premium**: 昨日最高涨停溢价

#### 市值分布特征
- **market_cap_small/medium/large**: 小/中/大市值股票数量
- **limit_board_small/medium/large**: 涨停板小/中/大市值分布

#### 涨停质量特征（需计算生成）
- **limit_up_ratio**: 涨停占比（需计算：limit_up_count / stock_count）
- **limit_down_ratio**: 跌停占比（需计算：limit_down_count / stock_count）
- **one_word_limit_ratio**: 一字板占比（需计算：one_word_limit_up_count / limit_up_count）
- **broken_ratio**: 炸板率（需计算：broken_limit_up_count / (limit_up_count + broken_limit_up_count)）

#### 综合指标（需计算生成）
- **sector_strength**: 板块强度指标（需计算：基于avg_pct_chg, avg_turnover, avg_amplitude的综合评分）
- **board_heat_index**: 板块热度指数（已有heat_index，可直接使用）
- **sector_momentum_score**: 板块动量评分（需计算：基于avg_pct_chg和limit_up_ratio的综合评分）

### 3.4 排名特征
- **ten_day_rise_count**: 十日涨幅排名数量
- **ten_day_rise_ratio**: 十日涨幅排名比例
- **daily_rise_count**: 日涨幅排名数量
- **daily_rise_ratio**: 日涨幅排名比例
- **daily_fall_count**: 日跌幅排名数量
- **daily_fall_ratio**: 日跌幅排名比例
- **daily_amount_count**: 日成交额排名数量
- **daily_amount_ratio**: 日成交额排名比例
- **ten_day_rise_weighted**: 十日涨幅加权

### 3.5 技术指标一致性特征
- **indicator_consensus**: 技术指标一致程度
- **strong_bull_consensus**: 强多头一致信号
- **strong_bear_consensus**: 强空头一致信号

## 4. 市场特征 (Market Features)

基于_extract_market_environment_features函数的实现：

### 4.1 基础市场状态特征
- **market_type**: 市场类型
- **market_cycle**: 市场周期
- **market_trend_strength**: 市场趋势强度
- **market_group**: 市场分组（1:牛市, 2:熊市, 3:震荡, 4:极端）

### 4.2 市场信号特征
- **market_risk_level**: 市场风险等级
- **market_trading_signal**: 市场交易信号
- **market_signal_strength**: 市场信号强度

### 4.3 市场基准特征（创业板指数）
- **market_benchmark_change**: 市场基准变化
- **market_rsi6**: 市场RSI6
- **market_rsi12**: 市场RSI12
- **market_macd**: 市场MACD
- **market_bb_width**: 市场布林带宽度
- **market_ma5**: 市场MA5
- **market_ma10**: 市场MA10
- **market_profit_loss_ratio**: 市场盈亏比

### 4.4 板块与市场相对特征
- **relative_change_percent**: 相对涨跌幅
- **rsi_vs_market**: RSI相对市场比值
- **rsi_diff_market**: RSI与市场差值
- **macd_vs_market**: MACD相对市场比值
- **macd_diff_market**: MACD与市场差值
- **volatility_vs_market**: 波动率相对市场比值
- **momentum_vs_market_rsi**: 动量相对市场RSI比值
- **ma_ratio_vs_market**: 均线比率相对市场比值
- **profit_loss_ratio_vs_market**: 盈亏比相对市场比值

## 5. 特征工程说明

### 5.1 特征计算原理
1. **个股特征**: 基于OHLCV数据计算各种技术指标和量化因子
2. **板块特征**: 通过板块内个股数据聚合计算，包括加权平均、求和、比例等
3. **市场特征**: 基于市场整体数据计算的宏观指标
4. **相对特征**: 板块相对于市场的相对强度指标

### 5.2 特征质量控制
1. **异常值处理**: 使用分位数方法识别和处理异常值
2. **缺失值填充**: 使用前向填充、中位数填充等方法
3. **标准化**: 部分特征进行Z-score标准化
4. **特征筛选**: 基于相关性和重要性筛选有效特征

### 5.3 特征更新频率
- **日频特征**: 每个交易日更新
- **周频特征**: 每周更新一次
- **月频特征**: 每月更新一次

## 6. 数据来源和存储

### 6.1 数据来源
- **基础数据**: 通过xtdata API获取股票和板块的OHLCV数据
- **技术指标**: 基于基础数据计算生成
- **市场数据**: 从指数数据中提取
- **涨跌停数据**: 基于价格限制计算

### 6.2 存储格式
- **个股特征**: 存储在`to/local_data/features/daily/`目录下，按日期分文件
- **板块特征**: 存储在`to/local_data/sector_features/`目录下，按板块分目录
- **缓存机制**: 使用Parquet格式提高读取效率

## 7. 特征使用建议

### 7.1 特征选择原则
1. **相关性**: 选择与目标变量相关性高的特征
2. **稳定性**: 选择在不同市场环境下表现稳定的特征
3. **可解释性**: 优先选择具有明确经济含义的特征
4. **多样性**: 组合不同类型的特征以提高模型鲁棒性

### 7.2 特征组合策略
1. **技术指标组合**: 结合趋势、动量、波动率等不同类型指标
2. **多时间框架**: 结合短期、中期、长期特征
3. **量价结合**: 同时考虑价格和成交量特征
4. **个股板块结合**: 结合个股和板块层面的特征
5. **涨跌停特征**: 充分利用A股市场特有的涨跌停制度特征

## 8. 注意事项

1. **前瞻偏差**: 确保不使用未来信息
2. **数据质量**: 定期检查数据完整性和准确性
3. **特征漂移**: 监控特征分布的变化
4. **计算效率**: 优化特征计算的性能
5. **存储管理**: 合理管理特征数据的存储空间
6. **涨跌停处理**: 正确处理涨跌停数据的特殊性
