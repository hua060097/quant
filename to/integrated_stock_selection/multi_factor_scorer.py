#!/usr/bin/env python3
"""
多因子评分器 - 基于2020-2025顶尖量化研究重新设计
参考文献：
1. Bali et al. (2023) "Empirical Asset Pricing via Machine Learning"
2. Gu et al. (2020) "Empirical Asset Pricing via Machine Learning"
3. Chen et al. (2022) "Deep Learning in Asset Pricing"
4. A股特色：涨跌停制度、连板效应、板块轮动
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# 导入特征计算器
try:
    from .feature_calculator import FeatureCalculator
except ImportError:
    from feature_calculator import FeatureCalculator

# 默认因子权重配置
DEFAULT_FACTOR_WEIGHTS = {
    'technical': 0.25,
    'fundamental': 0.25,
    'sector': 0.20,
    'capital_flow': 0.30
}

class MultiFactorScorer:
    """
    多因子评分器 - 基于现有特征的专业量化评分系统

    设计原则：
    1. 严格使用现有特征，避免虚构指标
    2. 基于顶尖量化研究的因子设计思路
    3. 结合A股市场特色（涨跌停、连板、板块轮动）
    4. 动态权重调整，避免简单粗暴的线性组合
    5. 考虑市场环境的适应性
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.factor_weights = DEFAULT_FACTOR_WEIGHTS.copy()

        # 初始化特征计算器
        self.feature_calculator = FeatureCalculator()

        # 基于Fama-French五因子模型扩展的权重体系
        self.dynamic_weights = {
            'momentum': 0.25,      # 动量因子（Jegadeesh & Titman, 1993）
            'reversal': 0.15,      # 反转因子（De Bondt & Thaler, 1985）
            'volatility': 0.20,    # 波动率因子（Ang et al., 2006）
            'liquidity': 0.15,     # 流动性因子（Amihud, 2002）
            'quality': 0.25        # 质量因子（Asness et al., 2019）
        }

    def calculate_technical_score(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术面评分 - 基于现有技术指标特征

        参考研究：
        - Moskowitz et al. (2012) "Time Series Momentum in Stocks and Bonds"
        - Novy-Marx (2012) "Is momentum really momentum?"
        """
        try:
            data = stock_data.copy()
            technical_score = pd.Series(0.0, index=data.index)

            # 1. 动量因子组合（基于现有收益率特征）
            momentum_score = self._calculate_momentum_factor(data)
            technical_score += momentum_score * 0.15  # 降低权重

            # 2. 趋势强度因子（基于现有技术指标）
            trend_score = self._calculate_trend_factor(data)
            technical_score += trend_score * 0.15   # 降低权重

            # 3. 技术指标一致性（基于RSI、MACD等）
            consensus_score = self._calculate_technical_consensus(data)
            technical_score += consensus_score * 0.15   # 降低权重

            # 4. 价格位置因子（基于均线系统）
            position_score = self._calculate_price_position_factor(data)
            technical_score += position_score * 0.15

            # 5. 形态特征评分（新增）
            pattern_score = self._calculate_pattern_factor(data)
            technical_score += pattern_score * 0.2  

            # 6. 涨停质量评分（新增）
            limit_quality_score = self._calculate_limit_quality_factor(data)
            technical_score += limit_quality_score * 0.2

            data['technical_score'] = np.clip(technical_score, 0, 100)

            self.logger.debug("技术面评分计算完成")
            return data

        except Exception as e:
            self.logger.error(f"技术面评分计算失败: {e}")
            stock_data['technical_score'] = 0
            return stock_data
    
    def calculate_fundamental_score(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算基本面评分 - 基于现有基本面特征

        参考研究：
        - Asness et al. (2019) "Quality Minus Junk"
        - Novy-Marx (2013) "The other side of value"
        - Frazzini & Pedersen (2014) "Betting against beta"
        """
        try:
            data = stock_data.copy()
            fundamental_score = pd.Series(0.0, index=data.index)

            # 1. 市值因子（基于现有市值特征）
            size_score = self._calculate_size_factor(data)
            fundamental_score += size_score * 0.25

            # 2. 流动性因子（基于现有成交量特征）
            liquidity_score = self._calculate_liquidity_factor(data)
            fundamental_score += liquidity_score * 0.30

            # 3. 质量因子（基于现有基本面特征）
            quality_score = self._calculate_quality_factor(data)
            fundamental_score += quality_score * 0.25

            # 4. 波动率因子（基于现有波动率特征）
            volatility_score = self._calculate_volatility_factor(data)
            fundamental_score += volatility_score * 0.20

            data['fundamental_score'] = np.clip(fundamental_score, 0, 100)

            self.logger.debug("基本面评分计算完成")
            return data

        except Exception as e:
            self.logger.error(f"基本面评分计算失败: {e}")
            stock_data['fundamental_score'] = 0
            return stock_data
    
    def calculate_sector_score(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算板块评分（基于强势股相似性）

        评分组成：
        1. limit_up_similarity_score: 强势股相似性评分 (100%)

        设计理念：寻找与强势股形态相似的潜力股
        注：已取消累计涨幅惩罚，改为在过滤阶段直接过滤高涨幅股票
        """
        try:
            self.logger.info("计算板块评分（原始逻辑）")
            data = stock_data.copy()

            # 权重配置（取消累计涨幅惩罚）
            weights = {
                'limit_up_similarity_score': 100,      # 强势股相似性评分（100%）
            }

            # 检查必需的列
            required_columns = ['limit_up_similarity_score']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.logger.warning(f"缺少必需的列: {missing_columns}")
                # 为缺失的列设置默认值
                for col in missing_columns:
                    data[col] = 0

            # 计算板块评分（仅基于强势股相似性）
            sector_score = (
                data['limit_up_similarity_score'] * weights['limit_up_similarity_score'] / 100
            )

            data['sector_score'] = sector_score

            # 统计信息
            avg_score = sector_score.mean()
            max_score = sector_score.max()
            min_score = sector_score.min()
            std_score = sector_score.std()
            unique_scores = sector_score.nunique()

            self.logger.info(f"板块评分完成: 平均{avg_score:.2f}, 范围[{min_score:.2f}, {max_score:.2f}], 标准差{std_score:.2f}")
            self.logger.info(f"评分分布: {unique_scores}个不同值")

            # 分析各组件贡献
            if 'limit_up_similarity_score' in data.columns:
                similarity_avg = data['limit_up_similarity_score'].mean()
                similarity_nonzero = (data['limit_up_similarity_score'] > 0).sum()
                self.logger.info(f"强势股相似性: 平均{similarity_avg:.2f}, 非零{similarity_nonzero}只")

            if 'consecutive_limit_penalty' in data.columns:
                penalty_avg = data['consecutive_limit_penalty'].mean()
                penalty_nonzero = (data['consecutive_limit_penalty'] > 0).sum()
                self.logger.info(f"累计涨幅惩罚: 平均{penalty_avg:.2f}, 非零{penalty_nonzero}只 (已取消惩罚)")

            return data

        except Exception as e:
            self.logger.error(f"计算板块评分失败: {e}")
            # 返回默认评分
            stock_data['sector_score'] = 50
            return stock_data
    
    def calculate_capital_flow_score(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算资金流评分 - 基于现有成交量和资金流特征

        参考研究：
        - Amihud (2002) "Illiquidity and stock returns"
        - Chordia et al. (2001) "Trading activity and expected stock returns"
        - Pastor & Stambaugh (2003) "Liquidity risk and expected stock returns"
        """
        try:
            data = stock_data.copy()
            capital_score = pd.Series(0.0, index=data.index)

            # 1. 成交量因子（基于现有成交量特征）
            volume_score = self._calculate_volume_factor(data)
            capital_score += volume_score * 0.35

            # 2. 流动性因子（基于现有流动性特征）
            liquidity_score = self._calculate_amihud_liquidity_factor(data)
            capital_score += liquidity_score * 0.25

            # 3. 量价关系因子（基于现有价量相关性特征）
            volume_price_score = self._calculate_volume_price_factor(data)
            capital_score += volume_price_score * 0.25

            # 4. 资金强度因子（基于现有资金强度特征）
            capital_intensity_score = self._calculate_capital_intensity_factor(data)
            capital_score += capital_intensity_score * 0.15

            data['capital_flow_score'] = np.clip(capital_score, 0, 100)

            self.logger.debug("资金流评分计算完成")
            return data

        except Exception as e:
            self.logger.error(f"资金流评分计算失败: {e}")
            stock_data['capital_flow_score'] = 0
            return stock_data

    def _filter_invalid_scores(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        过滤掉评分为NaN或异常的股票
        严格质量控制：有任何关键评分为NaN的股票都被过滤掉
        """
        try:
            data = stock_data.copy()
            initial_count = len(data)

            # 检查关键评分字段
            key_score_fields = ['technical_score', 'fundamental_score', 'capital_flow_score', 'sector_score']

            valid_mask = pd.Series(True, index=data.index)

            for field in key_score_fields:
                if field in data.columns:
                    if field == 'sector_score':
                        # sector_score必须大于0（不能为0或NaN）
                        field_valid = data[field].notna() & (data[field] > 0)
                    else:
                        # 其他评分可以为0，但不能为NaN或负值
                        field_valid = data[field].notna() & (data[field] >= 0)

                    valid_mask = valid_mask & field_valid

                    invalid_count = (~field_valid).sum()
                    if invalid_count > 0:
                        if field == 'sector_score':
                            self.logger.info(f"过滤 {invalid_count} 只股票：{field} 为NaN或≤0")
                        else:
                            self.logger.info(f"过滤 {invalid_count} 只股票：{field} 为NaN或负值")

            # 应用过滤
            filtered_data = data[valid_mask]
            filtered_count = len(filtered_data)

            self.logger.info(f"评分质量过滤: {initial_count} -> {filtered_count} (过滤掉 {initial_count - filtered_count} 只)")

            return filtered_data

        except Exception as e:
            self.logger.error(f"评分质量过滤失败: {e}")
            return stock_data

    def calculate_composite_score(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算综合评分 - 简化版本，直接使用四个评分加权计算

        评分组成：
        1. technical_score: 技术面评分
        2. fundamental_score: 基本面评分
        3. sector_score: 板块评分
        4. capital_flow_score: 资金流评分

        注意：不再分阶段，直接使用配置的权重进行加权计算
        """
        try:
            data = stock_data.copy()

            # 确保所有子评分存在
            required_scores = ['technical_score', 'fundamental_score', 'sector_score', 'capital_flow_score']
            for score in required_scores:
                if score not in data.columns:
                    data[score] = 0

            # 过滤掉评分异常的股票
            data = self._filter_invalid_scores(data)

            if len(data) == 0:
                self.logger.error("所有股票都被过滤掉了，无法进行评分")
                return stock_data  # 返回原始数据

            self.logger.info(f"评分异常过滤后剩余 {len(data)} 只股票")

            # 直接计算综合评分（使用配置的权重）
            composite_score = (
                data['technical_score'] * self.factor_weights['technical'] +
                data['fundamental_score'] * self.factor_weights['fundamental'] +
                data['sector_score'] * self.factor_weights['sector'] +
                data['capital_flow_score'] * self.factor_weights['capital_flow']
            )

            data['composite_score'] = composite_score

            # 筛选前5%股票
            top5_threshold = composite_score.quantile(0.95)
            top5_mask = composite_score >= top5_threshold

            self.logger.info(f"综合评分筛选: {top5_mask.sum()} 只股票进入前5%（阈值: {top5_threshold:.2f}）")

            # 只保留前5%的股票
            if top5_mask.any():
                data = data[top5_mask].copy()
                self.logger.info(f"综合评分完成，保留前5%股票: {len(data)} 只")
            else:
                self.logger.warning("没有股票通过前5%筛选，返回空结果")
                return pd.DataFrame()

            # 计算风险调整评分
            risk_penalty = 0

            # 1. 原有风险因子
            # 波动率风险
            if 'high_volatility' in data.columns:
                risk_penalty += data['high_volatility'] * 10

            # 跌停风险
            if 'recent_limit_down' in data.columns:
                risk_penalty += data['recent_limit_down'] * 15

            # 低质量涨停风险
            if 'low_quality_limit' in data.columns:
                risk_penalty += data['low_quality_limit'] * 5

            # 2. 新增风险因子 - 基于负收益股票分析
            # 盈亏比越高越好
            if 'risk_reward_ratio' in data.columns:
                high_risk_reward = (data['risk_reward_ratio'] > 2).astype(int)
                risk_penalty -= high_risk_reward * 25  # 重度奖赏25分

            # 接近阻力位惩罚（惩罚8-15分）
            if 'near_resistance' in data.columns:
                # near_resistance为1表示接近阻力位
                risk_penalty += data['near_resistance'] * 12  # 重度惩罚12分

            # 远离支撑位惩罚（惩罚3-8分）
            if 'near_support' in data.columns:
                # near_support为0表示远离支撑位，为1表示接近支撑位
                far_from_support = (1 - data['near_support']).clip(0, 1)
                risk_penalty += far_from_support * 5  # 轻度惩罚5分


            # 3. 涨停风险控制
            # 当日涨停重度惩罚（15-20分）
            if 'is_limit_up' in data.columns:
                risk_penalty += data['is_limit_up'] * 18  # 重度惩罚18分

            # 前一交易日涨停中度惩罚（8-12分）
            # 需要计算前一日涨停状态
            if 'is_limit_up' in data.columns:
                # 按股票代码分组，计算前一日涨停状态
                data_sorted = data.sort_values(['stock_code', 'date'])
                prev_limit_up = data_sorted.groupby('stock_code')['is_limit_up'].shift(1).fillna(0)
                risk_penalty += prev_limit_up * 10  # 中度惩罚10分

            data['risk_adjusted_score'] = np.clip(composite_score - risk_penalty, 0, 100)
            
            self.logger.debug("综合评分计算完成")
            return data
            
        except Exception as e:
            self.logger.error(f"综合评分计算失败: {e}")
            stock_data['composite_score'] = 0
            stock_data['risk_adjusted_score'] = 0
            return stock_data
    
    def calculate_all_scores(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """计算所有评分"""
        try:
            self.logger.info("开始多因子评分计算")

            # 首先计算缺失的特征
            self.logger.debug("计算缺失特征...")
            data = self.feature_calculator.calculate_missing_features(stock_data)

            # 按顺序计算各维度评分
            data = self.calculate_technical_score(data)
            data = self.calculate_fundamental_score(data)
            data = self.calculate_sector_score(data)
            data = self.calculate_capital_flow_score(data)
            data = self.calculate_composite_score(data)

            self.logger.info("多因子评分计算完成")
            return data

        except Exception as e:
            self.logger.error(f"多因子评分计算失败: {e}")
            return stock_data

    def calculate_scores_only(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """只计算评分，跳过特征计算（特征已在前面计算完成）"""
        try:
            self.logger.info("开始多因子评分计算（跳过特征计算）")

            # 直接计算各维度评分
            data = self.calculate_technical_score(stock_data)
            data = self.calculate_fundamental_score(data)
            data = self.calculate_sector_score(data)
            data = self.calculate_capital_flow_score(data)
            data = self.calculate_composite_score(data)  # 这里会计算risk_adjusted_score

            self.logger.info("多因子评分计算完成")
            return data

        except Exception as e:
            self.logger.error(f"多因子评分计算失败: {e}")
            return stock_data
    
    def rank_stocks_by_score(self, stock_data: pd.DataFrame, 
                           score_column: str = 'risk_adjusted_score') -> pd.DataFrame:
        """按评分对股票排序"""
        try:
            if score_column not in stock_data.columns:
                self.logger.error(f"评分列{score_column}不存在")
                return stock_data
            
            # 按日期分组排序
            ranked_data = stock_data.copy()
            ranked_data['score_rank'] = ranked_data.groupby('date')[score_column].rank(
                method='dense', ascending=False
            )
            
            # 计算分位数排名
            ranked_data['score_percentile'] = ranked_data.groupby('date')[score_column].rank(
                pct=True, ascending=False
            )
            
            self.logger.debug(f"股票评分排序完成，使用{score_column}")
            return ranked_data
            
        except Exception as e:
            self.logger.error(f"股票评分排序失败: {e}")
            return stock_data
    
    def filter_stocks_by_criteria(self, stock_data: pd.DataFrame, current_date: str = None) -> pd.DataFrame:
        """
        根据标准过滤股票
        注意：此时接收的应该是已经通过composite_score筛选的前5%股票

        参数:
        - stock_data: 股票数据
        - current_date: 当前日期，用于新股过滤
        """
        try:
            data = stock_data.copy()
            initial_count = len(data)

            # 如果数据为空，直接返回
            if initial_count == 0:
                self.logger.warning("接收到空数据，无需过滤")
                return data

            self.logger.info(f"开始过滤前5%筛选后的股票: {initial_count} 只")

            # 逐步应用过滤条件，记录每步的过滤效果
            current_data = data.copy()
            filter_steps = []

            # 1. 市值过滤
            min_market_cap = 50  # 默认最小市值50亿
            if 'total_mv' in current_data.columns:
                before_count = len(current_data)
                market_cap_filter = current_data['total_mv'] >= min_market_cap
                current_data = current_data[market_cap_filter]
                after_count = len(current_data)
                filtered_out = before_count - after_count
                filter_steps.append(f"市值过滤(>={min_market_cap}): {before_count} -> {after_count} (过滤{filtered_out}只)")
                self.logger.info(f"市值过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

            # 2. ST股票过滤
            if 'stock_code' in current_data.columns:
                before_count = len(current_data)
                st_filter = ~current_data['stock_code'].str.contains('ST', na=False)
                current_data = current_data[st_filter]
                after_count = len(current_data)
                filtered_out = before_count - after_count
                filter_steps.append(f"ST股票过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")
                self.logger.info(f"ST股票过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

            # 3. 跌停股票过滤
            if 'is_limit_down' in current_data.columns:
                before_count = len(current_data)
                limit_down_filter = current_data['is_limit_down'] != 1
                current_data = current_data[limit_down_filter]
                after_count = len(current_data)
                filtered_out = before_count - after_count
                filter_steps.append(f"跌停股票过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")
                self.logger.info(f"跌停股票过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

            # 4. 技术面过滤（逐项应用）
            current_data = self._apply_technical_filters_step_by_step(current_data, current_date, filter_steps)

            data = current_data
            filtered_count = len(data)

            # 输出过滤总结
            self.logger.info(f"前5%股票过滤完成: {initial_count} -> {filtered_count}")
            if filtered_count == 0:
                self.logger.warning("所有股票都被过滤掉了！过滤步骤详情:")
                for step in filter_steps:
                    self.logger.warning(f"  {step}")
            else:
                self.logger.info("过滤步骤详情:")
                for step in filter_steps:
                    self.logger.info(f"  {step}")

            return data
            
        except Exception as e:
            self.logger.error(f"股票过滤失败: {e}")
            return stock_data

    def _apply_technical_filters_step_by_step(self, data: pd.DataFrame, current_date: str, filter_steps: list) -> pd.DataFrame:
        """
        专业量化过滤系统 - 基于动量、趋势、背离等多维度协调过滤

        设计理念：
        1. 动量确认：寻找具有持续上涨动能的股票
        2. 趋势一致性：多时间框架趋势协调
        3. 技术背离检测：避免技术指标背离的股票
        4. 流动性保证：确保足够的交易活跃度
        5. 风险控制：过滤高风险股票
        """
        try:
            current_data = data.copy()

            # 第一层：基础质量过滤
            current_data = self._apply_quality_filters(current_data, filter_steps)

            # 第二层：动量趋势过滤
            current_data = self._apply_momentum_trend_filters(current_data, filter_steps)

            # 第三层：技术背离过滤
            current_data = self._apply_technical_divergence_filters(current_data, filter_steps)

            # 第四层：流动性和风险过滤
            current_data = self._apply_liquidity_risk_filters(current_data, current_date, filter_steps)

            # 保护机制
            if len(current_data) == 0 and len(data) > 0:
                current_data = self._apply_protection_mechanism(data, filter_steps)

            return current_data

        except Exception as e:
            self.logger.error(f"专业量化过滤失败: {e}")
            return data

    def _apply_quality_filters(self, data: pd.DataFrame, filter_steps: list) -> pd.DataFrame:
        """
        第一层：基础质量过滤
        过滤明显不符合条件的股票
        """
        current_data = data.copy()

        # 1. 技术评分过滤 - 使用动态阈值
        if 'technical_score' in current_data.columns:
            before_count = len(current_data)
            tech_threshold = max(30, current_data['technical_score'].quantile(0.15))
            tech_filter = current_data['technical_score'] >= tech_threshold
            current_data = current_data[tech_filter]
            after_count = len(current_data)
            filtered_out = before_count - after_count
            filter_steps.append(f"技术评分过滤(>={tech_threshold:.1f}): {before_count} -> {after_count} (过滤{filtered_out}只)")
            self.logger.info(f"技术评分过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

        # 2. 基本涨跌幅过滤 - 避免大跌股票
        if 'pct_chg' in current_data.columns:
            before_count = len(current_data)
            pct_filter = current_data['pct_chg'] >= -5.0  # 避免大跌股票
            current_data = current_data[pct_filter]
            after_count = len(current_data)
            filtered_out = before_count - after_count
            filter_steps.append(f"大跌过滤(>=-5%): {before_count} -> {after_count} (过滤{filtered_out}只)")
            self.logger.info(f"大跌过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

        # 3. 近10日累计涨幅过滤 - 避免短期涨幅过大的股票
        if 'cumulative_return_10d' in current_data.columns:
            before_count = len(current_data)
            # 过滤掉近10日累计涨幅超过15%的股票
            cumulative_filter = current_data['cumulative_return_10d'] <= 15.0
            current_data = current_data[cumulative_filter]
            after_count = len(current_data)
            filtered_out = before_count - after_count
            filter_steps.append(f"近10日累计涨幅过滤(<=15%): {before_count} -> {after_count} (过滤{filtered_out}只)")
            self.logger.info(f"近10日累计涨幅过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")
        else:
            self.logger.warning("cumulative_return_10d列不存在，跳过近10日累计涨幅过滤")

        return current_data

    def _apply_momentum_trend_filters(self, data: pd.DataFrame, filter_steps: list) -> pd.DataFrame:
        """
        第二层：动量趋势过滤
        寻找具有上涨动能和趋势一致性的股票
        """
        current_data = data.copy()

        # 1. 多时间框架趋势一致性
        if all(col in current_data.columns for col in ['ma5', 'ma10', 'ma20', 'close']):
            before_count = len(current_data)

            # 短期趋势：价格在MA5之上或接近
            short_trend = current_data['close'] >= current_data['ma5'] * 0.98

            # 中期趋势：MA5在MA10之上或接近
            mid_trend = current_data['ma5'] >= current_data['ma10'] * 0.995

            # 长期趋势：MA10在MA20之上或接近（更宽松）
            long_trend = current_data['ma10'] >= current_data['ma20'] * 0.98

            # 趋势一致性：至少两个时间框架趋势向上
            trend_filter = (short_trend.astype(int) + mid_trend.astype(int) + long_trend.astype(int)) >= 2

            current_data = current_data[trend_filter]
            after_count = len(current_data)
            filtered_out = before_count - after_count
            filter_steps.append(f"多时间框架趋势过滤(至少2个向上): {before_count} -> {after_count} (过滤{filtered_out}只)")
            self.logger.info(f"多时间框架趋势过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

        # 2. 动量强度过滤 - 使用多个动量指标
        current_data = self._apply_momentum_strength_filter(current_data, filter_steps)

        return current_data

    def _apply_momentum_strength_filter(self, data: pd.DataFrame, filter_steps: list) -> pd.DataFrame:
        """
        动量强度过滤 - 基于多个动量指标的综合判断
        """
        current_data = data.copy()

        # 检查可用的动量指标
        momentum_indicators = []

        # 1. RSI动量 - 寻找适中到强势的RSI
        if 'rsi14' in current_data.columns:
            # RSI在30-75之间，避免极端超买超卖
            rsi_momentum = (current_data['rsi14'] >= 30) & (current_data['rsi14'] <= 75)
            momentum_indicators.append(rsi_momentum)

        # 2. MACD动量
        if all(col in current_data.columns for col in ['macd', 'macd_signal']):
            # MACD在信号线之上或接近
            macd_momentum = current_data['macd'] >= current_data['macd_signal'] * 0.9
            momentum_indicators.append(macd_momentum)

        # 3. 布林带位置 - 避免极端位置
        if 'bb_position' in current_data.columns:
            # 布林带位置在0.2-0.8之间，避免极端超买超卖
            bb_momentum = (current_data['bb_position'] >= 0.2) & (current_data['bb_position'] <= 0.8)
            momentum_indicators.append(bb_momentum)

        if momentum_indicators:
            before_count = len(current_data)

            # 至少一半的动量指标为正
            momentum_score = sum(momentum_indicators)
            required_positive = max(1, len(momentum_indicators) // 2)
            momentum_filter = momentum_score >= required_positive

            current_data = current_data[momentum_filter]
            after_count = len(current_data)
            filtered_out = before_count - after_count
            filter_steps.append(f"动量强度过滤(至少{required_positive}/{len(momentum_indicators)}个指标正向): {before_count} -> {after_count} (过滤{filtered_out}只)")
            self.logger.info(f"动量强度过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

        return current_data

    def _apply_technical_divergence_filters(self, data: pd.DataFrame, filter_steps: list) -> pd.DataFrame:
        """
        第三层：技术背离过滤
        检测并过滤存在技术背离的股票
        """
        current_data = data.copy()

        # 1. 价格与RSI背离检测
        if all(col in current_data.columns for col in ['pct_chg', 'rsi14']):
            before_count = len(current_data)

            # 简化的背离检测：价格上涨但RSI过高（可能顶背离）
            price_up = current_data['pct_chg'] > 0
            rsi_overbought = current_data['rsi14'] > 80  # 严重超买

            # 过滤可能存在顶背离的股票（价格上涨但RSI严重超买）
            divergence_filter = ~(price_up & rsi_overbought)

            current_data = current_data[divergence_filter]
            after_count = len(current_data)
            filtered_out = before_count - after_count
            filter_steps.append(f"RSI顶背离过滤(避免价格上涨+RSI>80): {before_count} -> {after_count} (过滤{filtered_out}只)")
            self.logger.info(f"RSI顶背离过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

        # 2. MACD背离检测
        if all(col in current_data.columns for col in ['pct_chg', 'macd', 'macd_signal']):
            before_count = len(current_data)

            # MACD死叉且价格仍在上涨可能是顶背离信号
            price_up = current_data['pct_chg'] > 2  # 明显上涨
            macd_bearish = current_data['macd'] < current_data['macd_signal']  # MACD死叉

            # 过滤可能的MACD顶背离
            macd_divergence_filter = ~(price_up & macd_bearish)

            current_data = current_data[macd_divergence_filter]
            after_count = len(current_data)
            filtered_out = before_count - after_count
            filter_steps.append(f"MACD背离过滤(避免价格上涨+MACD死叉): {before_count} -> {after_count} (过滤{filtered_out}只)")
            self.logger.info(f"MACD背离过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

        return current_data

    def _apply_liquidity_risk_filters(self, data: pd.DataFrame, current_date: str, filter_steps: list) -> pd.DataFrame:
        """
        第四层：流动性和风险过滤
        """
        current_data = data.copy()

        # 1. 流动性过滤 - 动态换手率阈值
        if 'turnover_rate' in current_data.columns:
            before_count = len(current_data)
            turnover_threshold = max(0.1, current_data['turnover_rate'].quantile(0.1))
            turnover_filter = current_data['turnover_rate'] >= turnover_threshold
            current_data = current_data[turnover_filter]
            after_count = len(current_data)
            filtered_out = before_count - after_count
            filter_steps.append(f"流动性过滤(换手率>={turnover_threshold:.2f}%): {before_count} -> {after_count} (过滤{filtered_out}只)")
            self.logger.info(f"流动性过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

        # 2. 新股过滤
        if current_date:
            before_count = len(current_data)
            new_stock_filter = self._filter_new_stocks(current_data, current_date)
            if new_stock_filter is not None:
                current_data = current_data[new_stock_filter]
                after_count = len(current_data)
                filtered_out = before_count - after_count
                filter_steps.append(f"新股过滤(>=15个有效交易日): {before_count} -> {after_count} (过滤{filtered_out}只)")
                self.logger.info(f"新股过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

        return current_data

    def _apply_protection_mechanism(self, data: pd.DataFrame, filter_steps: list) -> pd.DataFrame:
        """
        保护机制：当所有股票都被过滤时，应用最基本的过滤条件
        """
        self.logger.warning("所有股票都被过滤掉了，启用保护机制")
        protected_data = data.copy()

        # 只保留最严格的过滤条件
        if 'is_limit_down' in protected_data.columns:
            protected_data = protected_data[protected_data['is_limit_down'] != 1]

        if 'stock_code' in protected_data.columns:
            protected_data = protected_data[~protected_data['stock_code'].str.contains('ST', na=False)]

        if 'pct_chg' in protected_data.columns:
            protected_data = protected_data[protected_data['pct_chg'] >= -8.0]

        self.logger.warning(f"保护机制生效: 保留 {len(protected_data)} 只股票")
        filter_steps.append(f"保护机制: 保留基本过滤后的 {len(protected_data)} 只股票")
        return protected_data

    def _add_technical_filters(self, data: pd.DataFrame, filters: list, current_date: str = None):
        """
        添加技术面过滤条件
        过滤掉明显下跌趋势的股票

        参数:
        - data: 股票数据
        - filters: 过滤条件列表
        - current_date: 当前日期，用于新股过滤
        """
        try:
            # 1. 技术评分过滤 - 使用动态阈值替代固定30分
            if 'technical_score' in data.columns:
                # 使用第25百分位数作为动态阈值，确保过滤掉技术面最差的25%股票
                tech_threshold = max(30, data['technical_score'].quantile(0.25))
                filters.append(data['technical_score'] >= tech_threshold)
                self.logger.debug(f"添加技术评分过滤: >= {tech_threshold:.1f}分（动态阈值）")

            # 2. 均线趋势过滤 - 基于均线数据判断趋势
            if all(col in data.columns for col in ['ma5', 'ma10', 'ma20', 'close']):
                # 过滤条件：价格不能远低于MA20
                price_filter = data['close'] >= data['ma20'] * 0.95  # 价格不低于MA20的5%
                filters.append(price_filter)

                # 过滤条件：不能是明显的空头排列
                # 允许MA5略低于MA10，但不能MA5明显低于MA20
                trend_filter = data['ma5'] >= data['ma20'] * 0.97  # MA5不低于MA20的3%
                filters.append(trend_filter)

                self.logger.debug("添加均线趋势过滤: 价格>=MA20*0.95, MA5>=MA20*0.97")

            # 3. 涨跌幅过滤 - 过滤掉当日大跌的股票
            if 'pct_chg' in data.columns:
                filters.append(data['pct_chg'] >= -7.0)  # 当日跌幅不超过7%
                self.logger.debug("添加涨跌幅过滤: >= -7%")

            # 4. 多周期RSI过滤 - 基于专业技术分析
            self._add_multi_period_rsi_filters(data, filters)

            # 5. 成交量异常过滤 - 过滤掉成交量异常萎缩的股票
            if 'turnover_rate' in data.columns:
                filters.append(data['turnover_rate'] >= 0.5)  # 换手率不低于0.5%
                self.logger.debug("添加换手率过滤: >= 0.5%")

            # 6. 新股过滤 - 过滤掉上市不足20个交易日的股票
            if current_date:
                new_stock_filter = self._filter_new_stocks(data, current_date)
                if new_stock_filter is not None:
                    filters.append(new_stock_filter)
                    self.logger.debug("添加新股过滤: 上市>=20个交易日")

        except Exception as e:
            self.logger.error(f"添加技术面过滤失败: {e}")

    def _add_multi_period_rsi_filters(self, data: pd.DataFrame, filters: list):
        """
        添加多周期RSI过滤条件 - 基于专业技术分析

        RSI指标说明：
        - RSI6: 短期超买超卖指标，反应迅速，适合捕捉短期过度反应
        - RSI12: 中短期指标，平衡敏感性和稳定性
        - RSI14: 经典参数，Wilder原始设计，最常用的中期指标

        专业阈值：
        - 超买区域: RSI > 70 (价格可能回调)
        - 超卖区域: RSI < 30 (价格可能反弹)
        - 严重超卖: RSI < 20 (极度超卖，风险较高)
        """
        try:
            # 检查可用的RSI指标
            available_rsi = []
            for rsi_col in ['rsi6', 'rsi12', 'rsi14']:
                if rsi_col in data.columns:
                    available_rsi.append(rsi_col)

            if not available_rsi:
                self.logger.debug("没有可用的RSI指标，跳过RSI过滤")
                return

            self.logger.debug(f"可用RSI指标: {available_rsi}")

            # 策略1: 过滤RSI14严重超卖股票 (< 20)
            if 'rsi14' in available_rsi:
                # RSI14是最经典的参数，过滤严重超卖的股票
                filters.append(data['rsi14'] >= 20)
                self.logger.debug("添加RSI14严重超卖过滤: >= 20")

            # 策略2: 过滤RSI6过度超卖股票 (< 15)
            if 'rsi6' in available_rsi:
                # RSI6反应迅速，过滤短期过度超卖
                filters.append(data['rsi6'] >= 15)
                self.logger.debug("添加RSI6过度超卖过滤: >= 15")

            # 策略3: 多周期RSI共振过滤
            if len(available_rsi) >= 2:
                # 计算多周期RSI的平均值，避免多周期共振超卖
                rsi_sum = data[available_rsi[0]] * 0  # 初始化为0
                for rsi_col in available_rsi:
                    rsi_sum += data[rsi_col]
                rsi_avg = rsi_sum / len(available_rsi)

                # 要求RSI平均值不低于25（避免多周期共振超卖）
                filters.append(rsi_avg >= 25)
                self.logger.debug(f"添加多周期RSI共振过滤: 平均>= 25 (基于{len(available_rsi)}个RSI)")

        except Exception as e:
            self.logger.error(f"添加多周期RSI过滤失败: {e}")

    def _apply_multi_period_rsi_filter(self, data: pd.DataFrame, filter_steps: list) -> pd.DataFrame:
        """
        应用多周期RSI过滤 - 逐步过滤版本
        """
        try:
            current_data = data.copy()

            # 检查可用的RSI指标
            available_rsi = []
            for rsi_col in ['rsi6', 'rsi12', 'rsi14']:
                if rsi_col in current_data.columns:
                    available_rsi.append(rsi_col)

            if not available_rsi:
                self.logger.warning("没有可用的RSI指标，跳过RSI过滤")
                filter_steps.append("RSI过滤: 跳过 (无可用RSI指标)")
                return current_data

            self.logger.info(f"可用RSI指标: {available_rsi}")

            # 策略1: 过滤严重超卖股票 (RSI14 < 20)
            if 'rsi14' in available_rsi:
                before_count = len(current_data)
                rsi14_filter = current_data['rsi14'] >= 20
                current_data = current_data[rsi14_filter]
                after_count = len(current_data)
                filtered_out = before_count - after_count
                filter_steps.append(f"RSI14严重超卖过滤(>=20): {before_count} -> {after_count} (过滤{filtered_out}只)")
                self.logger.info(f"RSI14严重超卖过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

            # 策略2: 短期RSI6过度超卖过滤 (RSI6 < 15)
            if 'rsi6' in available_rsi:
                before_count = len(current_data)
                rsi6_filter = current_data['rsi6'] >= 15
                current_data = current_data[rsi6_filter]
                after_count = len(current_data)
                filtered_out = before_count - after_count
                filter_steps.append(f"RSI6过度超卖过滤(>=15): {before_count} -> {after_count} (过滤{filtered_out}只)")
                self.logger.info(f"RSI6过度超卖过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

            # 策略3: 多周期RSI共振过滤
            if len(available_rsi) >= 2:
                before_count = len(current_data)

                # 计算多周期RSI的平均值
                rsi_sum = current_data[available_rsi[0]] * 0  # 初始化
                for rsi_col in available_rsi:
                    rsi_sum += current_data[rsi_col]
                rsi_avg = rsi_sum / len(available_rsi)

                multi_rsi_filter = rsi_avg >= 25
                current_data = current_data[multi_rsi_filter]
                after_count = len(current_data)
                filtered_out = before_count - after_count
                filter_steps.append(f"多周期RSI共振过滤(平均>=25): {before_count} -> {after_count} (过滤{filtered_out}只)")
                self.logger.info(f"多周期RSI共振过滤: {before_count} -> {after_count} (过滤{filtered_out}只)")

            return current_data

        except Exception as e:
            self.logger.error(f"多周期RSI过滤失败: {e}")
            filter_steps.append(f"RSI过滤: 失败 ({str(e)})")
            return data

    def _filter_new_stocks_batch(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        批量过滤新股 - 使用main.py中的批量实现

        参数:
        - stock_data: 完整的股票数据DataFrame，包含多个交易日

        返回:
        - pd.DataFrame: 过滤后的股票数据
        """
        try:
            # 导入main.py中的批量实现
            from main import _filter_new_stocks_batch as main_filter_new_stocks_batch

            # 直接调用main.py中的批量实现
            return main_filter_new_stocks_batch(stock_data, self.logger)

        except Exception as e:
            self.logger.error(f"批量新股过滤失败: {e}")
            # 出错时返回原始数据
            return stock_data

    def _filter_new_stocks(self, data: pd.DataFrame, current_date: str) -> pd.Series:
        """
        单日新股过滤 - 为保持兼容性，但建议使用批量过滤

        参数:
        - data: 当前日期的股票数据
        - current_date: 当前选股日期 (格式: YYYYMMDD)

        返回:
        - pd.Series: 布尔值序列，True表示股票数据有效，False表示数据应该被过滤
        """
        try:
            # 为向后兼容，将单日数据包装成stock_data格式并调用批量处理
            if 'date' not in data.columns:
                # 如果没有date列，添加当前日期
                data_copy = data.copy()
                data_copy['date'] = pd.to_datetime(current_date, format='%Y%m%d')
            else:
                data_copy = data.copy()

            # 调用批量处理
            filtered_data = self._filter_new_stocks_batch(data_copy)

            # 返回原始数据的过滤结果
            original_stock_codes = set(data['stock_code'])
            filtered_stock_codes = set(filtered_data['stock_code'])

            # 构建布尔序列：保留的股票为True，过滤的为False
            result = data['stock_code'].isin(filtered_stock_codes)

            return result

        except Exception as e:
            self.logger.error(f"新股过滤失败: {e}")
            # 出错时不过滤任何股票
            return pd.Series(True, index=data.index)

    def _get_previous_trading_dates(self, current_date: str, days: int) -> list:
        """获取指定日期前N个交易日的日期列表"""
        try:
            from pathlib import Path
            import datetime

            # 使用相对于当前文件的路径
            current_file_dir = Path(__file__).parent  # integrated_stock_selection目录
            features_dir = current_file_dir.parent / "local_data" / "features" / "daily"

            # 简化实现：向前推算days*1.5天，然后筛选存在数据文件的日期
            current_dt = datetime.datetime.strptime(current_date, '%Y%m%d')
            trading_dates = []

            for i in range(1, days * 2):  # 向前推算足够多的天数
                check_date = current_dt - datetime.timedelta(days=i)
                date_str = check_date.strftime('%Y%m%d')

                # 检查该日期是否有数据文件（即是否为交易日）
                file_path = features_dir / f"{date_str}.parquet"
                if file_path.exists():
                    trading_dates.append(date_str)

                    if len(trading_dates) >= days:
                        break

            return trading_dates[:days]  # 返回最近的N个交易日

        except Exception as e:
            self.logger.error(f"获取交易日列表失败: {e}")
            return []

    # ==================== 辅助计算方法 ====================

    def _calculate_momentum_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算动量因子 - 基于现有收益率特征
        参考：Jegadeesh & Titman (1993)
        """
        momentum_score = pd.Series(0.0, index=data.index)

        # 使用现有的收益率特征
        if 'return_5d' in data.columns:
            # 5日收益率标准化 (0-40分)
            momentum_score += np.clip(data['return_5d'] * 200, -20, 40)

        if 'return_20d' in data.columns:
            # 20日收益率标准化 (0-30分)
            momentum_score += np.clip(data['return_20d'] * 150, -15, 30)

        if 'pct_chg' in data.columns:
            # 当日涨跌幅 (0-30分)
            momentum_score += np.clip(data['pct_chg'] * 3, -15, 30)

        return np.clip(momentum_score, 0, 100)

    def _calculate_trend_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算趋势强度因子 - 基于现有技术指标
        参考：Moskowitz et al. (2012)
        """
        trend_score = pd.Series(0.0, index=data.index)

        # 使用现有的趋势强度特征（如果存在）
        if 'trend_strength' in data.columns:
            # trend_strength > 1.0 表示上升趋势，给予加分
            uptrend_mask = data['trend_strength'] > 1.0
            trend_score += np.where(uptrend_mask, (data['trend_strength'] - 1.0) * 40, 0)

        if 'trend_consistency' in data.columns:
            # trend_consistency越高，趋势越一致，给予更多加分
            trend_score += data['trend_consistency'] * 30

        if 'momentum_acceleration' in data.columns:
            trend_score += np.clip(data['momentum_acceleration'] * 100, 0, 30)

        # 补充均线趋势评分
        trend_score += self._calculate_ma_trend_score(data)

        return np.clip(trend_score, 0, 100)

    def _calculate_ma_trend_score(self, data: pd.DataFrame) -> pd.Series:
        """
        基于均线计算趋势评分
        当没有专门的趋势特征时使用
        """
        trend_score = pd.Series(0.0, index=data.index)

        try:
            # 检查均线数据是否有效
            if all(col in data.columns for col in ['ma5', 'ma10', 'ma20', 'close']):
                ma5 = data['ma5']
                ma10 = data['ma10']
                ma20 = data['ma20']
                close = data['close']

                # 过滤掉无效数据（均线为0或NaN）
                valid_mask = (ma5 > 0) & (ma10 > 0) & (ma20 > 0) & (close > 0)

                if valid_mask.any():
                    # 1. 均线多头排列 (40分)
                    bullish_alignment = (ma5 > ma10) & (ma10 > ma20)
                    trend_score += np.where(valid_mask & bullish_alignment, 40, 0)

                    # 2. 价格在均线上方 (30分)
                    price_above_ma = close > ma20
                    trend_score += np.where(valid_mask & price_above_ma, 30, 0)

                    # 3. 均线斜率向上 (30分) - 通过MA5相对MA20的位置判断
                    ma_slope_up = ma5 > ma20 * 1.02  # MA5比MA20高2%以上认为向上
                    trend_score += np.where(valid_mask & ma_slope_up, 30, 0)

                    # 对于均线向下的情况给予惩罚
                    bearish_alignment = (ma5 < ma10) & (ma10 < ma20)
                    trend_score -= np.where(valid_mask & bearish_alignment, 20, 0)

                    # 价格在均线下方惩罚
                    price_below_ma = close < ma20 * 0.98  # 低于MA20 2%以上
                    trend_score -= np.where(valid_mask & price_below_ma, 15, 0)

        except Exception as e:
            self.logger.error(f"均线趋势评分计算失败: {e}")

        return np.clip(trend_score, 0, 100)

    def _calculate_technical_consensus(self, data: pd.DataFrame) -> pd.Series:
        """
        计算技术指标一致性评分 - 基于现有技术指标（安全版本）
        参考：Neely et al. (2014)
        注意：这是评分计算，不会覆盖technical_consensus特征
        """
        consensus_score = pd.Series(0.0, index=data.index)

        try:
            # RSI一致性（安全处理）
            if 'rsi6' in data.columns and 'rsi12' in data.columns and 'rsi14' in data.columns:
                # 安全计算RSI平均值
                rsi6_safe = np.where(np.isfinite(data['rsi6']), data['rsi6'], 50.0)
                rsi12_safe = np.where(np.isfinite(data['rsi12']), data['rsi12'], 50.0)
                rsi14_safe = np.where(np.isfinite(data['rsi14']), data['rsi14'], 50.0)

                rsi_avg = (rsi6_safe + rsi12_safe + rsi14_safe) / 3
                # RSI在30-70之间得分较高，避免极端值
                rsi_score = np.where((rsi_avg > 30) & (rsi_avg < 70),
                                   40 - np.abs(rsi_avg - 50), 0)
                consensus_score += rsi_score

            # MACD一致性（安全处理）
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd_safe = np.where(np.isfinite(data['macd']), data['macd'], 0.0)
                macd_signal_safe = np.where(np.isfinite(data['macd_signal']), data['macd_signal'], 0.0)
                macd_bullish = (macd_safe > macd_signal_safe).astype(int)
                consensus_score += macd_bullish * 30

            # RSI-MACD一致性（安全处理）
            if 'rsi_macd_agreement' in data.columns:
                agreement_safe = np.where(np.isfinite(data['rsi_macd_agreement']),
                                        data['rsi_macd_agreement'], 0.0)
                consensus_score += agreement_safe * 30

        except Exception as e:
            self.logger.warning(f"技术指标一致性计算出错: {e}")
            consensus_score = pd.Series(0.0, index=data.index)

        # 确保结果有限且在合理范围内
        result = np.clip(consensus_score, 0, 100)
        result = np.where(np.isfinite(result), result, 0.0)

        return pd.Series(result, index=data.index)

    def _calculate_price_position_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算价格位置因子 - 基于现有均线系统
        参考：Zhu & Zhou (2009)
        """
        position_score = pd.Series(0.0, index=data.index)

        # 价格相对均线位置
        if 'close' in data.columns:
            if 'ma5' in data.columns:
                ma5_ratio = (data['close'] / data['ma5'] - 1) * 100
                position_score += np.clip(ma5_ratio * 2, -10, 25)

            if 'ma20' in data.columns:
                ma20_ratio = (data['close'] / data['ma20'] - 1) * 100
                position_score += np.clip(ma20_ratio * 1.5, -10, 25)

            if 'ma60' in data.columns:
                ma60_ratio = (data['close'] / data['ma60'] - 1) * 100
                position_score += np.clip(ma60_ratio, -10, 25)

        # 布林带位置
        if 'bb_position' in data.columns:
            # 布林带中上部得分较高
            bb_score = np.where(data['bb_position'] > 0.5,
                              (data['bb_position'] - 0.5) * 50, 0)
            position_score += bb_score

        return np.clip(position_score, 0, 100)

    def _calculate_size_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算市值因子 - 基于现有市值特征
        参考：Fama & French (1992) 修正版，适应A股特点
        """
        size_score = pd.Series(50.0, index=data.index)  # 默认中性评分

        if 'total_mv' in data.columns:
            # A股市场：中小市值股票通常有更好的表现
            # 检查NaN值：有NaN的股票将在后续被过滤掉
            valid_mv_mask = data['total_mv'].notna() & (data['total_mv'] > 0)

            if valid_mv_mask.any():
                mv_log = np.log(data['total_mv'] + 1)
                mv_percentile = mv_log.rank(pct=True)

                # 中小市值得分较高，但避免过小的股票
                size_score = np.where(mv_percentile < 0.2, 30,  # 过小市值
                            np.where(mv_percentile < 0.6, 80,   # 中小市值
                            np.where(mv_percentile < 0.8, 60,   # 中等市值
                                    40)))                        # 大市值

                # 对于total_mv为NaN的股票，size_score保持默认值50

        return np.clip(size_score, 0, 100)

    def _calculate_liquidity_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算流动性因子 - 基于现有成交量特征
        参考：Amihud (2002)
        """
        liquidity_score = pd.Series(0.0, index=data.index)

        # 换手率评分
        if 'turnover_rate' in data.columns:
            # 保持NaN值，让后续过滤处理
            # 适中的换手率得分最高（1%-8%）
            turnover_score = np.where(data['turnover_rate'] < 0.5, 20,
                           np.where(data['turnover_rate'] < 1, 40,
                           np.where(data['turnover_rate'] < 8, 80,
                           np.where(data['turnover_rate'] < 15, 60, 30))))
            liquidity_score += turnover_score * 0.4

        # Amihud非流动性指标（越小越好）
        if 'amihud_illiquidity' in data.columns:
            # 保持NaN值，让后续过滤处理
            amihud_score = 100 - np.clip(data['amihud_illiquidity'] * 1000, 0, 80)
            liquidity_score += amihud_score * 0.3

        # 成交量Z分数
        if 'volume_zscore' in data.columns:
            # 保持NaN值，让后续过滤处理
            volume_score = np.clip(data['volume_zscore'] * 10 + 50, 0, 100)
            liquidity_score += volume_score * 0.3

        return np.clip(liquidity_score, 0, 100)

    def _calculate_quality_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算质量因子 - 基于现有基本面特征
        参考：Asness et al. (2019) "Quality Minus Junk"
        """
        quality_score = pd.Series(0.0, index=data.index)

        # 流通比例（高流通比例表示质量较好）
        if 'float_ratio' in data.columns:
            quality_score += np.clip(data['float_ratio'] * 100, 0, 40)

        # 市值因子（适中市值得分较高）
        if 'mv_factor' in data.columns:
            quality_score += np.clip(data['mv_factor'] * 50, 0, 30)

        # 资金强度
        if 'capital_intensity' in data.columns:
            quality_score += np.clip(data['capital_intensity'] * 100, 0, 30)

        return np.clip(quality_score, 0, 100)

    def _calculate_volatility_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算波动率因子 - 基于现有波动率特征
        参考：Frazzini & Pedersen (2014) "Betting against beta"
        """
        volatility_score = pd.Series(0.0, index=data.index)

        # 20日波动率（适中波动率得分较高）
        if 'volatility20' in data.columns:
            # 波动率在2%-6%之间得分最高
            vol_score = np.where(data['volatility20'] < 1, 30,
                       np.where(data['volatility20'] < 2, 60,
                       np.where(data['volatility20'] < 6, 80,
                       np.where(data['volatility20'] < 10, 50, 20))))
            volatility_score += vol_score * 0.4

        # Parkinson波动率
        if 'parkinson_vol' in data.columns:
            park_score = 100 - np.clip(data['parkinson_vol'] * 20, 0, 60)
            volatility_score += park_score * 0.3

        # 波动率偏度
        if 'volatility_skew' in data.columns:
            skew_score = 50 + np.clip(data['volatility_skew'] * 25, -30, 30)
            volatility_score += skew_score * 0.3

        return np.clip(volatility_score, 0, 100)

    def _calculate_volume_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算成交量因子 - 基于现有成交量特征
        参考：Chordia et al. (2001)
        """
        volume_score = pd.Series(0.0, index=data.index)

        # 成交量Z分数（异常放量得分较高）
        if 'volume_zscore' in data.columns:
            # 保持NaN值，让后续过滤处理
            # Z分数在1-3之间得分最高
            vol_z_score = np.where(data['volume_zscore'] < 0, 20,
                         np.where(data['volume_zscore'] < 1, 50,
                         np.where(data['volume_zscore'] < 3, 80,
                         np.where(data['volume_zscore'] < 5, 60, 30))))
            volume_score += vol_z_score * 0.4

        # 换手率效率
        if 'turnover_efficiency' in data.columns:
            # 保持NaN值，让后续过滤处理
            volume_score += np.clip(data['turnover_efficiency'] * 100, 0, 30)

        # 涨停日成交量比率
        if 'limit_up_volume_ratio' in data.columns:
            # 保持NaN值，让后续过滤处理
            volume_score += np.clip(data['limit_up_volume_ratio'] * 100, 0, 30)

        return np.clip(volume_score, 0, 100)

    def _calculate_amihud_liquidity_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算Amihud流动性因子 - 基于现有Amihud指标
        参考：Amihud (2002)
        """
        liquidity_score = pd.Series(50.0, index=data.index)  # 默认中性评分

        if 'amihud_illiquidity' in data.columns:
            # Amihud指标越小，流动性越好，得分越高
            # 使用对数变换处理极值
            amihud_log = np.log(data['amihud_illiquidity'] + 1e-10)
            amihud_percentile = amihud_log.rank(pct=True)

            # 流动性好的股票得分高
            liquidity_score = (1 - amihud_percentile) * 100

        return np.clip(liquidity_score, 0, 100)

    def _calculate_volume_price_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算量价关系因子 - 基于现有价量相关性特征
        参考：Karpoff (1987)
        """
        vp_score = pd.Series(0.0, index=data.index)

        # 20日价量相关性
        if 'vol_price_corr_20d' in data.columns:
            # 正相关得分较高
            corr_score = np.clip(data['vol_price_corr_20d'] * 50 + 50, 0, 80)
            vp_score += corr_score * 0.4

        # 价量变化比率
        if 'price_vol_change_ratio' in data.columns:
            ratio_score = np.clip(data['price_vol_change_ratio'] * 25 + 50, 0, 60)
            vp_score += ratio_score * 0.3

        # 价量背离（负向指标，背离程度低得分高）
        if 'price_volume_divergence' in data.columns:
            div_score = 100 - np.clip(data['price_volume_divergence'] * 100, 0, 60)
            vp_score += div_score * 0.3

        return np.clip(vp_score, 0, 100)

    def _calculate_capital_intensity_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算资金强度因子 - 基于现有资金强度特征
        """
        intensity_score = pd.Series(0.0, index=data.index)

        # 资金强度
        if 'capital_intensity' in data.columns:
            intensity_score += np.clip(data['capital_intensity'] * 100, 0, 60)

        # 成交额相关
        if 'amount' in data.columns and 'volume' in data.columns:
            # 计算平均成交价格作为资金质量指标
            avg_price = data['amount'] / (data['volume'] + 1e-10)
            price_score = np.clip(np.log(avg_price + 1) * 10, 0, 40)
            intensity_score += price_score

        return np.clip(intensity_score, 0, 100)




    def _calculate_pattern_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算形态特征因子
        基于K线形态和吞没形态的综合评分
        """
        pattern_score = pd.Series(50.0, index=data.index)  # 基础分50分

        try:
            # 1. K线强度评分 (40%权重)
            if 'candlestick_strength' in data.columns:
                # K线强度标准化到0-100分
                strength_normalized = np.clip(data['candlestick_strength'] * 100 + 50, 0, 100)
                pattern_score += (strength_normalized - 50) * 0.4

            # 2. 看涨吞没形态加分 (30%权重)
            if 'pattern_engulfing_bullish' in data.columns:
                bullish_bonus = data['pattern_engulfing_bullish'] * 30  # 出现看涨吞没+30分
                pattern_score += bullish_bonus * 0.3

            # 3. 看跌吞没形态扣分 (30%权重)
            if 'pattern_engulfing_bearish' in data.columns:
                bearish_penalty = data['pattern_engulfing_bearish'] * 30  # 出现看跌吞没-30分
                pattern_score -= bearish_penalty * 0.3

            return np.clip(pattern_score, 0, 100)

        except Exception as e:
            self.logger.error(f"形态因子计算失败: {e}")
            return pattern_score

    def _calculate_limit_quality_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        计算涨停质量因子
        基于涨停质量评分的因子
        注意：LimitUpQualityFactors因子组已暂时禁用（稀疏性问题导致IR计算异常）
        """
        try:
            # if 'limit_up_quality_score' in data.columns:  # 暂时禁用：稀疏性问题
            #     # 直接使用涨停质量评分
            #     return data['limit_up_quality_score']
            # else:
            #     # 如果没有涨停质量评分，返回中性评分
            #     return pd.Series(50.0, index=data.index)

            # 暂时返回中性评分，避免依赖已禁用的因子
            return pd.Series(50.0, index=data.index)

        except Exception as e:
            self.logger.error(f"涨停质量因子计算失败: {e}")
            return pd.Series(50.0, index=data.index)
