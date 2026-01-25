#!/usr/bin/env python3
"""
增强版板块潜力分析器
充分利用141个板块特征进行深度分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

@dataclass
class EnhancedSectorResult:
    """增强版板块分析结果"""
    sector_code: str
    market_env: str
    
    # 四大维度评分
    technical_score: float
    capital_score: float
    sentiment_score: float
    relative_score: float
    
    # 新增维度评分
    momentum_score: float
    volatility_score: float
    structure_score: float
    quality_score: float
    
    # 综合评分
    total_score: float
    
    # 详细信息
    limit_up_count: int
    board_2_plus: int
    change_percent: float
    volume_ratio: float
    
    # 风险评估
    risk_level: str
    confidence: float

class EnhancedSectorAnalyzer:
    """增强版板块潜力分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 优化权重配置 - 基于回测结果的最优配置
        self.enhanced_weights = {
            'technical': 0.30,      # 技术面（提升，因为技术主导配置表现最佳）
            'capital': 0.15,        # 资金面
            'sentiment': 0.20,      # 情绪面（降低，平衡收益率）
            'relative': 0.15,       # 相对强度
            'momentum': 0.10,       # 动量
            'volatility': 0.05,     # 波动率
            'structure': 0.03,      # 结构（降低）
            'quality': 0.02         # 质量（降低）
        }
        
        # 市场环境调整因子
        self.env_adjustments = {
            'strong_market': {
                'technical': 1.2, 'capital': 1.0, 'sentiment': 1.4, 'relative': 0.9,
                'momentum': 1.3, 'volatility': 0.8, 'structure': 1.1, 'quality': 1.0
            },
            'weak_market': {
                'technical': 0.9, 'capital': 1.2, 'sentiment': 0.8, 'relative': 1.3,
                'momentum': 0.8, 'volatility': 1.2, 'structure': 1.2, 'quality': 1.3
            },
            'volatile_market': {
                'technical': 1.0, 'capital': 1.1, 'sentiment': 1.2, 'relative': 1.2,
                'momentum': 1.1, 'volatility': 1.0, 'structure': 1.0, 'quality': 1.1
            },
            'normal_market': {
                'technical': 1.0, 'capital': 1.0, 'sentiment': 1.0, 'relative': 1.0,
                'momentum': 1.0, 'volatility': 1.0, 'structure': 1.0, 'quality': 1.0
            }
        }
    
    def analyze_enhanced_daily_potential(self, daily_sectors: pd.DataFrame, 
                                       market_features: pd.DataFrame) -> List[EnhancedSectorResult]:
        """增强版每日板块潜力分析"""
        try:
            if daily_sectors.empty:
                return []
            
            # 评估市场环境
            market_env = self._assess_enhanced_market_environment(daily_sectors, market_features)
            
            # 计算8个维度评分
            technical_scores = self._calculate_enhanced_technical_scores(daily_sectors)
            capital_scores = self._calculate_enhanced_capital_scores(daily_sectors)
            sentiment_scores = self._calculate_enhanced_sentiment_scores(daily_sectors)
            relative_scores = self._calculate_enhanced_relative_scores(daily_sectors)
            momentum_scores = self._calculate_enhanced_momentum_scores(daily_sectors)
            volatility_scores = self._calculate_enhanced_volatility_scores(daily_sectors)
            structure_scores = self._calculate_enhanced_structure_scores(daily_sectors)
            quality_scores = self._calculate_enhanced_quality_scores(daily_sectors)
            
            # 生成结果
            results = []
            env_adj = self.env_adjustments.get(market_env, self.env_adjustments['normal_market'])
            
            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')
                
                # 获取各维度评分
                tech_score = technical_scores.get(sector_code, 0) * env_adj['technical']
                cap_score = capital_scores.get(sector_code, 0) * env_adj['capital']
                sent_score = sentiment_scores.get(sector_code, 0) * env_adj['sentiment']
                rel_score = relative_scores.get(sector_code, 0) * env_adj['relative']
                mom_score = momentum_scores.get(sector_code, 0) * env_adj['momentum']
                vol_score = volatility_scores.get(sector_code, 0) * env_adj['volatility']
                struct_score = structure_scores.get(sector_code, 0) * env_adj['structure']
                qual_score = quality_scores.get(sector_code, 0) * env_adj['quality']
                
                # 计算综合评分
                total_score = (
                    tech_score * self.enhanced_weights['technical'] +
                    cap_score * self.enhanced_weights['capital'] +
                    sent_score * self.enhanced_weights['sentiment'] +
                    rel_score * self.enhanced_weights['relative'] +
                    mom_score * self.enhanced_weights['momentum'] +
                    vol_score * self.enhanced_weights['volatility'] +
                    struct_score * self.enhanced_weights['structure'] +
                    qual_score * self.enhanced_weights['quality']
                )
                
                # 风险评估和置信度
                risk_level, confidence = self._assess_risk_and_confidence(row, total_score)
                
                # 创建结果对象
                result = EnhancedSectorResult(
                    sector_code=sector_code,
                    market_env=market_env,
                    technical_score=tech_score,
                    capital_score=cap_score,
                    sentiment_score=sent_score,
                    relative_score=rel_score,
                    momentum_score=mom_score,
                    volatility_score=vol_score,
                    structure_score=struct_score,
                    quality_score=qual_score,
                    total_score=total_score,
                    limit_up_count=int(row.get('limit_up_count', 0)) if not pd.isna(row.get('limit_up_count', 0)) else 0,
                    board_2_plus=int(row.get('board_counts_2板', 0) + row.get('board_counts_3板', 0)) if not pd.isna(row.get('board_counts_2板', 0)) else 0,
                    change_percent=row.get('change_percent', 0),
                    volume_ratio=row.get('volume_ratio_5', 1),
                    risk_level=risk_level,
                    confidence=confidence
                )
                
                results.append(result)
            
            # 按综合评分排序
            results.sort(key=lambda x: x.total_score, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"增强版板块潜力分析失败: {e}")
            return []
    
    def _assess_enhanced_market_environment(self, daily_sectors: pd.DataFrame, 
                                          market_features: pd.DataFrame) -> str:
        """增强版市场环境评估"""
        try:
            # 基础统计
            avg_change = daily_sectors['change_percent'].mean()
            up_ratio = (daily_sectors['change_percent'] > 0).mean()
            
            # 涨跌停统计
            total_limit_up = daily_sectors['limit_up_count'].fillna(0).sum()
            total_limit_down = daily_sectors['limit_down_count'].fillna(0).sum()
            total_stocks = daily_sectors['stock_count'].sum()
            
            # 连板统计
            total_board_2_plus = (daily_sectors['board_counts_2板'].fillna(0) + 
                                daily_sectors['board_counts_3板'].fillna(0)).sum()
            
            # 成交量活跃度
            volume_activity = daily_sectors['volume_ratio_5'].fillna(1).mean()
            
            # 市场活跃度综合评分
            limit_up_ratio = total_limit_up / max(total_stocks, 1)
            board_ratio = total_board_2_plus / max(total_stocks, 1)
            
            # 强势市场判断
            if (avg_change > 0.02 and up_ratio > 0.7 and 
                limit_up_ratio > 0.05 and volume_activity > 1.2):
                return 'strong_market'
            
            # 弱势市场判断
            elif (avg_change < -0.02 and up_ratio < 0.3 and 
                  total_limit_down > total_limit_up):
                return 'weak_market'
            
            # 震荡市场判断
            elif abs(avg_change) < 0.01 and 0.4 < up_ratio < 0.6:
                return 'volatile_market'
            
            else:
                return 'normal_market'
                
        except Exception as e:
            self.logger.warning(f"市场环境评估失败: {e}")
            return 'normal_market'
    
    def _calculate_enhanced_technical_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """增强版技术面评分 - 利用更多技术指标"""
        scores = {}
        
        try:
            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')
                score = 0
                
                # 价格趋势 (25分) - 优化收益率导向
                change_pct = row.get('change_percent', 0)
                if change_pct > 0.03:  # 降低阈值，更关注稳定上涨
                    score += 25
                elif change_pct > 0.02:
                    score += 22
                elif change_pct > 0.01:
                    score += 18
                elif change_pct > 0.005:  # 增加小幅上涨的评分
                    score += 15
                elif change_pct > 0:
                    score += 10
                elif change_pct > -0.01:  # 小幅下跌也给一定分数
                    score += 5
                
                # 均线系统 (20分)
                ma_bull_trend = row.get('ma_bull_trend', 0)
                price_above_ma5 = row.get('price_above_ma5', 0)
                price_above_ma20 = row.get('price_above_ma20', 0)
                
                if ma_bull_trend and price_above_ma5 and price_above_ma20:
                    score += 20
                elif price_above_ma5 and price_above_ma20:
                    score += 15
                elif price_above_ma5:
                    score += 10
                
                # 技术指标 (20分)
                rsi6 = row.get('rsi6', 50)
                macd_hist = row.get('macd_histogram', 0)
                
                if 30 < rsi6 < 70 and macd_hist > 0:
                    score += 20
                elif 20 < rsi6 < 80:
                    score += 15
                elif rsi6 > 20:
                    score += 10
                
                # 趋势强度 (15分)
                trend_strength = row.get('trend_strength', 0)
                consecutive_up_days = row.get('consecutive_up_days', 0)
                
                if trend_strength > 0.7:
                    score += 15
                elif trend_strength > 0.5:
                    score += 12
                elif trend_strength > 0.3:
                    score += 8
                
                if consecutive_up_days >= 3:
                    score += 5
                elif consecutive_up_days >= 2:
                    score += 3
                
                # 技术评分 (10分)
                tech_score = row.get('tech_score', 50)
                if tech_score > 80:
                    score += 10
                elif tech_score > 60:
                    score += 8
                elif tech_score > 40:
                    score += 5
                
                # 信号强度 (10分)
                signal_strength = row.get('signal_strength', 0)
                if signal_strength > 0.8:
                    score += 10
                elif signal_strength > 0.6:
                    score += 8
                elif signal_strength > 0.4:
                    score += 5
                
                scores[sector_code] = min(100, score)
                
        except Exception as e:
            self.logger.warning(f"技术面评分计算失败: {e}")
        
        return scores

    def _calculate_enhanced_capital_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """增强版资金面评分 - 利用更多成交量特征"""
        scores = {}

        try:
            # 计算成交量分位数
            volume_data = daily_sectors['volume'].replace(0, np.nan).dropna()
            amount_data = daily_sectors['amount'].replace(0, np.nan).dropna()

            if len(volume_data) > 0:
                volume_75 = volume_data.quantile(0.75)
                volume_90 = volume_data.quantile(0.90)
                amount_75 = amount_data.quantile(0.75) if len(amount_data) > 0 else 0

                for idx, row in daily_sectors.iterrows():
                    sector_code = row.get('sector_code', f'sector_{idx}')
                    score = 0

                    # 成交量放大 (30分)
                    volume = row.get('volume', 0)
                    volume_ratio_5 = row.get('volume_ratio_5', 1)
                    volume_ratio_20 = row.get('volume_ratio_20', 1)

                    if volume_ratio_5 > 2.0 and volume_ratio_20 > 1.5:
                        score += 30
                    elif volume_ratio_5 > 1.5:
                        score += 25
                    elif volume_ratio_5 > 1.2:
                        score += 20
                    elif volume_ratio_5 > 1.0:
                        score += 15

                    # 成交额活跃度 (25分)
                    amount = row.get('amount', 0)
                    daily_amount_ratio = row.get('daily_amount_ratio', 0)

                    if amount > amount_75 and daily_amount_ratio > 0.8:
                        score += 25
                    elif amount > amount_75:
                        score += 20
                    elif daily_amount_ratio > 0.6:
                        score += 15

                    # 换手率 (20分)
                    turnover_rate = row.get('turnover_rate', 0)
                    avg_turnover_rate = row.get('avg_turnover_rate', 0)

                    if turnover_rate > avg_turnover_rate * 2:
                        score += 20
                    elif turnover_rate > avg_turnover_rate * 1.5:
                        score += 15
                    elif turnover_rate > avg_turnover_rate:
                        score += 10

                    # 资金流动动量 (15分)
                    fund_flow_momentum = row.get('fund_flow_momentum', 1)
                    volume_intensity = row.get('volume_intensity', 0)

                    if fund_flow_momentum > 1.5:
                        score += 15
                    elif fund_flow_momentum > 1.2:
                        score += 12
                    elif fund_flow_momentum > 1.0:
                        score += 8

                    # 成交量趋势 (10分)
                    volume_trend = row.get('volume_trend', 0)
                    if volume_trend > 0.5:
                        score += 10
                    elif volume_trend > 0.2:
                        score += 8
                    elif volume_trend > 0:
                        score += 5

                    scores[sector_code] = min(100, score)

        except Exception as e:
            self.logger.warning(f"资金面评分计算失败: {e}")

        return scores

    def _calculate_enhanced_sentiment_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """增强版情绪面评分 - 充分利用涨跌停特征"""
        scores = {}

        try:
            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')
                score = 0

                # 基础数据
                stock_count = max(row.get('stock_count', 1), 1)
                limit_up_count = row.get('limit_up_count', 0)
                if pd.isna(limit_up_count):
                    limit_up_count = 0

                # 涨停数量和比例 (35分)
                limit_up_ratio = limit_up_count / stock_count
                one_word_limit_up_count = row.get('one_word_limit_up_count', 0)

                if limit_up_ratio > 0.15:  # 超过15%涨停
                    score += 35
                elif limit_up_ratio > 0.10:  # 超过10%涨停
                    score += 30
                elif limit_up_ratio > 0.05:  # 超过5%涨停
                    score += 25
                elif limit_up_ratio > 0.02:  # 超过2%涨停
                    score += 20
                elif limit_up_count > 0:
                    score += 15

                # 一字板质量 (10分)
                if one_word_limit_up_count > 0 and limit_up_count > 0:
                    one_word_ratio = one_word_limit_up_count / limit_up_count
                    if one_word_ratio > 0.5:
                        score += 10
                    elif one_word_ratio > 0.3:
                        score += 8
                    elif one_word_ratio > 0.1:
                        score += 5

                # 连板梯队 (25分)
                board_1 = row.get('board_counts_1板', 0) if not pd.isna(row.get('board_counts_1板', 0)) else 0
                board_2 = row.get('board_counts_2板', 0) if not pd.isna(row.get('board_counts_2板', 0)) else 0
                board_3 = row.get('board_counts_3板', 0) if not pd.isna(row.get('board_counts_3板', 0)) else 0

                # 计算高阶连板
                board_4_plus = 0
                for board_level in ['board_counts_4板', 'board_counts_5板', 'board_counts_6板',
                                  'board_counts_7板', 'board_counts_8板', 'board_counts_9板', 'board_counts_10板及以上']:
                    value = row.get(board_level, 0)
                    if not pd.isna(value):
                        board_4_plus += value

                if board_4_plus > 0:
                    score += 25  # 有高位连板
                elif board_3 > 0:
                    score += 20  # 有3板
                elif board_2 > 0:
                    score += 15  # 有2板
                elif board_1 >= 3:
                    score += 10  # 首板较多
                elif board_1 > 0:
                    score += 5   # 有首板

                # 炸板率 (负面因子，-15分)
                broken_limit_up_count = row.get('broken_limit_up_count', 0)
                if broken_limit_up_count > 0 and limit_up_count > 0:
                    broken_ratio = broken_limit_up_count / (limit_up_count + broken_limit_up_count)
                    if broken_ratio > 0.5:
                        score -= 15
                    elif broken_ratio > 0.3:
                        score -= 10
                    elif broken_ratio > 0.1:
                        score -= 5

                # 涨跌比例 (15分)
                up_count = row.get('up_count', 0)
                down_count = row.get('down_count', 0)
                if stock_count > 0:
                    up_ratio = up_count / stock_count
                    if up_ratio > 0.8:
                        score += 15
                    elif up_ratio > 0.6:
                        score += 12
                    elif up_ratio > 0.5:
                        score += 8
                    elif up_ratio > 0.4:
                        score += 5

                # 热度指数 (10分)
                heat_index = row.get('heat_index', 0)
                if heat_index > 80:
                    score += 10
                elif heat_index > 60:
                    score += 8
                elif heat_index > 40:
                    score += 5

                # 跌停惩罚
                limit_down_count = row.get('limit_down_count', 0)
                if pd.isna(limit_down_count):
                    limit_down_count = 0
                limit_down_penalty = min(30, limit_down_count * 5)

                scores[sector_code] = min(100, max(0, score - limit_down_penalty))

        except Exception as e:
            self.logger.warning(f"情绪面评分计算失败: {e}")

        return scores

    def _calculate_enhanced_relative_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """增强版相对强度评分"""
        scores = {}

        try:
            # 计算排名百分位
            change_ranks = daily_sectors['change_percent'].rank(pct=True)
            volume_ranks = daily_sectors['volume'].rank(pct=True)

            # 涨停排名
            limit_up_data = daily_sectors['limit_up_count'].fillna(0)
            limit_up_ranks = limit_up_data.rank(pct=True)

            # 连板强度排名
            board_strength = (daily_sectors['board_counts_1板'].fillna(0) * 1 +
                            daily_sectors['board_counts_2板'].fillna(0) * 2 +
                            daily_sectors['board_counts_3板'].fillna(0) * 3)
            board_ranks = board_strength.rank(pct=True)

            # 排名特征
            daily_rise_ranks = daily_sectors.get('daily_rise_ratio', pd.Series(0.5, index=daily_sectors.index)).rank(pct=True)
            ten_day_rise_ranks = daily_sectors.get('ten_day_rise_ratio', pd.Series(0.5, index=daily_sectors.index)).rank(pct=True)

            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')

                # 相对涨幅排名 (25分)
                change_rank = change_ranks.loc[idx]
                change_score = change_rank * 25

                # 相对成交量排名 (20分)
                volume_rank = volume_ranks.loc[idx]
                volume_score = volume_rank * 20

                # 相对涨停排名 (25分)
                limit_up_rank = limit_up_ranks.loc[idx]
                limit_up_score = limit_up_rank * 25

                # 相对连板强度排名 (20分)
                board_rank = board_ranks.loc[idx]
                board_score = board_rank * 20

                # 历史排名表现 (10分)
                daily_rise_rank = daily_rise_ranks.loc[idx]
                ten_day_rise_rank = ten_day_rise_ranks.loc[idx]
                history_score = (daily_rise_rank + ten_day_rise_rank) * 5

                total_score = change_score + volume_score + limit_up_score + board_score + history_score
                scores[sector_code] = min(100, total_score)

        except Exception as e:
            self.logger.warning(f"相对强度评分计算失败: {e}")

        return scores

    def _calculate_enhanced_momentum_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """动量评分 - 基于动量特征"""
        scores = {}

        try:
            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')
                score = 0

                # 多周期动量 (40分)
                momentum_5d = row.get('momentum_5d', 0)
                momentum_10d = row.get('momentum_10d', 0)
                momentum_20d = row.get('momentum_20d', 0)

                if momentum_5d > 0.05 and momentum_10d > 0.03:
                    score += 40
                elif momentum_5d > 0.03:
                    score += 30
                elif momentum_5d > 0.01:
                    score += 20
                elif momentum_5d > 0:
                    score += 10

                # 连续上涨天数 (30分)
                consecutive_up_days = row.get('consecutive_up_days', 0)
                if consecutive_up_days >= 5:
                    score += 30
                elif consecutive_up_days >= 3:
                    score += 25
                elif consecutive_up_days >= 2:
                    score += 15
                elif consecutive_up_days >= 1:
                    score += 10

                # 动量加速度 (20分)
                if momentum_5d > momentum_10d > momentum_20d:
                    score += 20  # 动量加速
                elif momentum_5d > momentum_10d:
                    score += 15  # 动量改善
                elif momentum_5d > 0:
                    score += 10  # 正动量

                # 摆动模式 (10分)
                swing_pattern = row.get('swing_pattern', 0)
                if swing_pattern > 0.7:
                    score += 10
                elif swing_pattern > 0.5:
                    score += 8
                elif swing_pattern > 0.3:
                    score += 5

                scores[sector_code] = min(100, score)

        except Exception as e:
            self.logger.warning(f"动量评分计算失败: {e}")

        return scores

    def _calculate_enhanced_volatility_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """波动率评分 - 适度波动率更优"""
        scores = {}

        try:
            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')
                score = 50  # 基础分

                # 当前波动率 (40分)
                volatility = row.get('volatility', 0)
                volatility_20d = row.get('volatility_20d', 0)

                # 适度波动率最优 (0.02-0.06)
                if 0.02 <= volatility <= 0.06:
                    score += 40
                elif 0.01 <= volatility <= 0.08:
                    score += 30
                elif volatility <= 0.10:
                    score += 20
                else:
                    score += 10  # 过高波动率

                # 波动率比率 (30分)
                volatility_ratio = row.get('volatility_ratio', 1)
                if 0.8 <= volatility_ratio <= 1.5:
                    score += 30  # 适度波动
                elif 0.5 <= volatility_ratio <= 2.0:
                    score += 20
                else:
                    score += 10  # 极端波动

                # 振幅 (30分)
                amplitude = row.get('amplitude', 0)
                avg_amplitude = row.get('avg_amplitude', 0)

                if 0.03 <= amplitude <= 0.08:
                    score += 30  # 适度振幅
                elif 0.02 <= amplitude <= 0.10:
                    score += 20
                elif amplitude > 0:
                    score += 10

                scores[sector_code] = min(100, max(0, score))

        except Exception as e:
            self.logger.warning(f"波动率评分计算失败: {e}")

        return scores

    def _calculate_enhanced_structure_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """结构评分 - 基于市值分布和板块结构"""
        scores = {}

        try:
            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')
                score = 0

                # 市值分布均衡性 (40分)
                large_cap = row.get('market_cap_distribution_large', 0)
                medium_cap = row.get('market_cap_distribution_medium', 0)
                small_cap = row.get('market_cap_distribution_small', 0)
                total_cap = large_cap + medium_cap + small_cap

                if total_cap > 0:
                    # 均衡分布更优
                    large_ratio = large_cap / total_cap
                    medium_ratio = medium_cap / total_cap
                    small_ratio = small_cap / total_cap

                    # 理想分布：大盘30%，中盘40%，小盘30%
                    balance_score = 100 - abs(large_ratio - 0.3) * 100 - abs(medium_ratio - 0.4) * 100 - abs(small_ratio - 0.3) * 100
                    score += max(0, balance_score * 0.4)

                # 连板分布 (35分)
                limit_large = row.get('limit_board_distribution_large', 0)
                limit_medium = row.get('limit_board_distribution_medium', 0)
                limit_small = row.get('limit_board_distribution_small', 0)

                # 小盘股连板更活跃为优
                if limit_small > limit_medium > limit_large:
                    score += 35
                elif limit_small > 0:
                    score += 25
                elif limit_medium > 0:
                    score += 15
                elif limit_large > 0:
                    score += 10

                # 支撑阻力位置 (25分)
                support_distance = row.get('support_distance_pct', 0)
                resistance_distance = row.get('resistance_distance_pct', 0)

                # 接近支撑位且远离阻力位为优
                if support_distance < 0.05 and resistance_distance > 0.1:
                    score += 25
                elif support_distance < 0.1:
                    score += 15
                elif resistance_distance > 0.05:
                    score += 10

                scores[sector_code] = min(100, score)

        except Exception as e:
            self.logger.warning(f"结构评分计算失败: {e}")

        return scores

    def _calculate_enhanced_quality_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """质量评分 - 基于综合质量指标"""
        scores = {}

        try:
            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')
                score = 0

                # 技术指标一致性 (30分)
                indicator_consensus = row.get('indicator_consensus', 0)
                strong_bull_consensus = row.get('strong_bull_consensus', 0)

                if strong_bull_consensus > 0.8:
                    score += 30
                elif indicator_consensus > 0.7:
                    score += 25
                elif indicator_consensus > 0.5:
                    score += 20
                elif indicator_consensus > 0.3:
                    score += 15

                # 市场情绪 (25分)
                market_sentiment = row.get('market_sentiment', 50)
                if market_sentiment > 80:
                    score += 25
                elif market_sentiment > 60:
                    score += 20
                elif market_sentiment > 40:
                    score += 15
                elif market_sentiment > 20:
                    score += 10

                # 信号强度 (25分)
                signal_strength = row.get('signal_strength', 0)
                if signal_strength > 0.8:
                    score += 25
                elif signal_strength > 0.6:
                    score += 20
                elif signal_strength > 0.4:
                    score += 15
                elif signal_strength > 0.2:
                    score += 10

                # 综合评分验证 (20分)
                tech_score = row.get('tech_score', 50)
                if tech_score > 80:
                    score += 20
                elif tech_score > 60:
                    score += 15
                elif tech_score > 40:
                    score += 10
                elif tech_score > 20:
                    score += 5

                scores[sector_code] = min(100, score)

        except Exception as e:
            self.logger.warning(f"质量评分计算失败: {e}")

        return scores

    def _assess_risk_and_confidence(self, row: pd.Series, total_score: float) -> Tuple[str, float]:
        """评估风险等级和置信度"""
        try:
            # 风险因子
            risk_factors = 0

            # 波动率风险
            volatility = row.get('volatility', 0)
            if volatility > 0.1:
                risk_factors += 2
            elif volatility > 0.08:
                risk_factors += 1

            # 炸板风险
            broken_limit_up_count = row.get('broken_limit_up_count', 0)
            limit_up_count = row.get('limit_up_count', 0)
            if broken_limit_up_count > 0 and limit_up_count > 0:
                broken_ratio = broken_limit_up_count / (limit_up_count + broken_limit_up_count)
                if broken_ratio > 0.5:
                    risk_factors += 2
                elif broken_ratio > 0.3:
                    risk_factors += 1

            # 跌停风险
            limit_down_count = row.get('limit_down_count', 0)
            if limit_down_count > 0:
                risk_factors += 1

            # 成交量异常风险
            volume_ratio_5 = row.get('volume_ratio_5', 1)
            if volume_ratio_5 > 5:  # 成交量过度放大
                risk_factors += 1

            # 风险等级
            if risk_factors >= 4:
                risk_level = 'high'
            elif risk_factors >= 2:
                risk_level = 'medium'
            else:
                risk_level = 'low'

            # 置信度计算
            confidence_factors = 0

            # 评分置信度
            if total_score > 80:
                confidence_factors += 3
            elif total_score > 60:
                confidence_factors += 2
            elif total_score > 40:
                confidence_factors += 1

            # 数据完整性置信度
            key_fields = ['limit_up_count', 'board_counts_1板', 'volume_ratio_5', 'change_percent']
            complete_fields = sum(1 for field in key_fields if not pd.isna(row.get(field, np.nan)))
            confidence_factors += complete_fields

            # 一致性置信度
            indicator_consensus = row.get('indicator_consensus', 0)
            if indicator_consensus > 0.7:
                confidence_factors += 2
            elif indicator_consensus > 0.5:
                confidence_factors += 1

            # 置信度标准化到0-1
            confidence = min(1.0, confidence_factors / 10)

            return risk_level, confidence

        except Exception as e:
            self.logger.warning(f"风险评估失败: {e}")
            return 'medium', 0.5
