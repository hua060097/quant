# 特征缺失分析报告

## 1. 数据源特征分析

### 1.1 load_all_data加载的基础特征（来自data_loader）
基于`main_sector_strategy/data/data_loader.py`的`_get_stock_feature_columns`方法，共38个基础特征：

**基础价格数据 (6个)**
- open, high, low, close, volume, amount

**原始特征 (4个)**
- pct_chg, total_mv, circ_mv, turnover_rate

**技术指标 (9个)**
- ma20, ma60, rsi12, macd, macd_hist, bb_width, bb_position, atr14, volatility20

**高级模式和趋势特征 (5个)**
- trend_strength, trend_consistency, candlestick_strength, pattern_engulfing_bullish, pattern_engulfing_bearish

**支撑阻力特征 (3个)**
- risk_reward_ratio, near_support, near_resistance

**量价特征 (4个)**
- volume_weighted_price, vol_price_corr_20d, price_vol_change_ratio, volume_zscore

**动量特征 (4个)**
- return_5d, return_20d, relative_strength_20d, momentum_acceleration

**波动率特征 (3个)**
- parkinson_vol, vol_term_structure, volatility_skew

**流动性和反转特征 (3个)**
- amihud_illiquidity, short_reversal, overbought_oversold

**综合指标 (6个)**
- tech_momentum_signal, risk_adjusted_return, returns_zscore, rsi_macd_agreement, bb_breakout_signal, price_volume_divergence

**高级ML特征 (4个)**
- volatility_adjusted_momentum, technical_consensus, composite_trading_signal, multi_period_trend_agreement

### 1.2 calculate_missing_features当前生成的特征
基于当前`feature_calculator.py`实现，主要包括：

**市值相对特征 (7个)**
- log_market_cap, market_cap_percentile, market_cap_zscore, size_factor_score, liquidity_adjusted_mv, market_cap_momentum, sector_relative_mv

**风险特征 (3个)**
- high_volatility, recent_limit_down, low_quality_limit

**涨停质量特征 (6个)**
- limit_up_count_60d, limit_up_next_day_positive_rate, first_board_seal_rate, consecutive_limit_rate, limit_up_premium_5pct_count, limit_up_quality_score

**累计收益率特征 (3个)**
- cumulative_return_3d, cumulative_return_5d, cumulative_return_10d

**均线相关特征 (6个)**
- ma_convergence, price_ma5_distance, price_ma10_distance, price_ma20_distance, price_ma60_distance, ma_support_strength

**股价低点距离特征 (6个)**
- price_low_distance_3d, price_low_distance_5d, price_low_distance_10d, price_low_distance_20d, price_low_distance_60d, price_low_distance_weighted

**MA交叉特征 (2个)**
- ma5_above_ma10_strength, ma5_below_ma10_strength

## 2. _get_ml_feature_columns需要的特征分析

### 2.1 严重缺失的特征类别

**K线市场微观结构特征 (67个) - 完全缺失**
- 单日微观结构特征 (11个): lower_shadow_ratio, upper_shadow_ratio, body_ratio, reversal_score, overnight_gap_pct, intraday_momentum, price_position_ratio, volatility_efficiency, buying_pressure, selling_pressure, market_structure_score
- 基于DTW的多日形态相似度特征 (6个): w_bottom_similarity, head_shoulders_similarity, cup_handle_similarity, ascending_triangle_similarity, flag_pattern_similarity, double_top_similarity
- 多日形态解构特征 (50个): W底解构、头肩顶解构、杯柄解构等专业量化特征

**横截面排名特征 (9个) - 完全缺失**
- rank_return_20d, rank_volatility20, rank_turnover_rate, rank_total_mv, rank_cumulative_return_5d, rank_cumulative_return_10d, rank_rsi14, rank_return_1d, rank_return_3d, rank_momentum_composite, rank_quality_composite, rank_risk_composite

**多周期股票市场相关性特征 (16个) - 完全缺失**
- 基础滚动指标: corr_5d, corr_10d, corr_20d, corr_60d, corr_120d, beta_5d, beta_10d, beta_20d, beta_60d, beta_120d
- 期限结构特征: corr_slope_5_20, corr_slope_10_60, corr_slope_20_120, beta_slope_5_20, beta_slope_10_60, beta_slope_20_120
- 动态变化特征: corr_10d_mom5, corr_20d_mom5, corr_60d_mom5, beta_10d_mom5, beta_20d_mom5, beta_60d_mom5
- 相对位置特征: corr_20d_zscore250, corr_60d_zscore250, beta_20d_zscore250, beta_60d_zscore250

**高级价格位置特征 (6个) - 部分缺失**
- price_position_in_range_5d, price_position_in_range_20d, price_position_in_range_60d
- price_vs_ma5_ratio, price_vs_ma20_ratio, price_vs_ma60_ratio

**成交量相对特征 (5个) - 部分缺失**
- volume_ratio_5d, volume_ratio_20d, volume_percentile_60d
- volatility_zscore_20d, volatility_percentile_60d

**均线相对特征 (4个) - 部分缺失**
- ma_alignment_score, ma_convergence_score
- ma5_ma20_distance_pct, ma20_ma60_distance_pct

### 2.2 需要删除的过时特征

**单日K线形态特征 (11个) - 需删除**
- pattern_bare_head_yang, pattern_bare_head_yin, pattern_hammer, pattern_hanging_man, pattern_doji, pattern_morning_star, pattern_evening_star, pattern_engulfing_bull, pattern_engulfing_bear, pattern_piercing, pattern_dark_cloud

**多日K线形态特征 - 需删除**
- 传统主观形态识别特征，已被专业微观结构特征替代

## 3. 修复优先级

### 高优先级（立即修复）
1. **K线市场微观结构特征** - 专业量化机构核心特征
2. **横截面排名特征** - 机器学习模型关键输入
3. **多周期相关性特征** - 风险管理和alpha发现

### 中优先级（后续修复）
1. **高级价格位置特征** - 技术分析增强
2. **成交量相对特征** - 流动性分析
3. **均线相对特征** - 趋势分析

### 低优先级（清理工作）
1. **删除过时的单日/多日形态特征**
2. **优化现有特征计算效率**

## 4. 实施计划

1. **第一阶段**: 实现K线市场微观结构特征计算
2. **第二阶段**: 实现横截面排名特征和相关性特征
3. **第三阶段**: 补充其他缺失特征
4. **第四阶段**: 清理过时特征，优化性能
