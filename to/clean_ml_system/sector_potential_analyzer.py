#!/usr/bin/env python3
"""
板块上涨潜力分析器
基于多维度评估识别最具继续上涨潜力的板块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass

@dataclass
class PotentialScore:
    """潜力评分结果"""
    sector_code: str
    total_score: float
    technical_score: float
    capital_score: float
    sentiment_score: float
    relative_score: float
    market_env: str
    confidence: float

class SectorPotentialAnalyzer:
    """板块上涨潜力分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 评估维度权重（根据市场环境动态调整）
        self.base_weights = {
            'technical': 0.30,    # 技术面：趋势、突破、支撑阻力
            'capital': 0.25,      # 资金面：成交量、资金流入
            'sentiment': 0.25,    # 情绪面：涨停、连板、热度
            'relative': 0.20      # 相对强度：与其他板块对比
        }
        
        # 市场环境权重调整
        self.env_weight_adjustments = {
            'strong_market': {'technical': 1.2, 'capital': 1.0, 'sentiment': 1.3, 'relative': 0.8},
            'weak_market': {'technical': 1.1, 'capital': 1.4, 'sentiment': 0.7, 'relative': 1.2},
            'volatile_market': {'technical': 0.9, 'capital': 1.1, 'sentiment': 1.1, 'relative': 1.3},
            'normal_market': {'technical': 1.0, 'capital': 1.0, 'sentiment': 1.0, 'relative': 1.0}
        }
    
    def analyze_daily_potential(self, sector_data: pd.DataFrame, market_features: pd.DataFrame = None) -> List[PotentialScore]:
        """
        分析当日所有板块的上涨潜力
        
        Args:
            sector_data: 板块数据
            market_features: 市场特征数据
            
        Returns:
            按潜力评分排序的板块列表
        """
        try:
            results = []
            
            # 按日期分组处理
            for date, daily_sectors in sector_data.groupby('date'):
                self.logger.info(f"分析日期 {date.strftime('%Y-%m-%d')} 的板块潜力")
                
                # 获取市场环境
                market_env = self._assess_market_environment(daily_sectors, market_features, date)
                self.logger.info(f"市场环境：{market_env}")
                
                # 计算各维度评分
                technical_scores = self._calculate_technical_scores(daily_sectors)
                capital_scores = self._calculate_capital_scores(daily_sectors)
                sentiment_scores = self._calculate_sentiment_scores(daily_sectors)
                relative_scores = self._calculate_relative_scores(daily_sectors)
                
                # 获取当前市场环境的权重
                weights = self._get_adjusted_weights(market_env)
                
                # 计算综合潜力评分
                for idx, row in daily_sectors.iterrows():
                    sector_code = row.get('sector_code', f'sector_{idx}')
                    
                    # 获取各维度评分
                    tech_score = technical_scores.get(sector_code, 0)
                    cap_score = capital_scores.get(sector_code, 0)
                    sent_score = sentiment_scores.get(sector_code, 0)
                    rel_score = relative_scores.get(sector_code, 0)
                    
                    # 计算加权总分
                    total_score = (
                        tech_score * weights['technical'] +
                        cap_score * weights['capital'] +
                        sent_score * weights['sentiment'] +
                        rel_score * weights['relative']
                    )
                    
                    # 计算置信度
                    confidence = self._calculate_confidence(tech_score, cap_score, sent_score, rel_score, market_env)
                    
                    result = PotentialScore(
                        sector_code=sector_code,
                        total_score=total_score,
                        technical_score=tech_score,
                        capital_score=cap_score,
                        sentiment_score=sent_score,
                        relative_score=rel_score,
                        market_env=market_env,
                        confidence=confidence
                    )
                    results.append(result)
            
            # 按总分排序
            results.sort(key=lambda x: x.total_score, reverse=True)

            # 输出综合评分前10名
            if results:
                self.logger.info(f"🏆 综合评分前10名:")
                for i, result in enumerate(results[:10], 1):
                    self.logger.info(f"   {i:2d}. {result.sector_code}: {result.total_score:.1f}分 "
                                   f"(技术:{result.technical_score:.1f} 资金:{result.capital_score:.1f} "
                                   f"情绪:{result.sentiment_score:.1f} 相对:{result.relative_score:.1f} "
                                   f"置信度:{result.confidence:.2f})")

                # 输出置信度前10名
                confidence_sorted = sorted(results, key=lambda x: x.confidence, reverse=True)[:10]
                self.logger.info(f"🎯 置信度前10名:")
                for i, result in enumerate(confidence_sorted, 1):
                    self.logger.info(f"   {i:2d}. {result.sector_code}: 置信度{result.confidence:.2f} "
                                   f"(总分:{result.total_score:.1f})")

            self.logger.info(f"完成 {len(results)} 个板块的潜力分析")
            return results
            
        except Exception as e:
            self.logger.error(f"板块潜力分析失败: {e}")
            return []
    
    def _assess_market_environment(self, daily_sectors: pd.DataFrame, market_features: pd.DataFrame, date: pd.Timestamp) -> str:
        """评估市场环境（基于涨跌停数据）"""
        try:
            # 获取市场数据
            if market_features is not None and len(market_features) > 0:
                market_row = market_features[market_features['date'] == date]
                if len(market_row) > 0:
                    market_change = market_row.iloc[0].get('daily_change_pct', 0)
                    market_volume_level = market_row.iloc[0].get('market_volume_level', 'medium')
                else:
                    market_change = 0
                    market_volume_level = 'medium'
            else:
                market_change = 0
                market_volume_level = 'medium'

            # 计算板块整体表现
            avg_change = daily_sectors['change_percent'].mean()
            up_ratio = (daily_sectors['change_percent'] > 0).mean()

            # 使用可用数据评估市场强弱
            total_stocks = daily_sectors['stock_count'].sum()

            # 尝试获取涨跌停数据
            if 'limit_up_count' in daily_sectors.columns:
                total_limit_up = daily_sectors['limit_up_count'].fillna(0).sum()
                total_limit_down = daily_sectors['limit_down_count'].fillna(0).sum()
            else:
                # 使用比率数据估算
                limit_up_ratio_sum = daily_sectors.get('limit_up_ratio', pd.Series(0, index=daily_sectors.index)).sum()
                limit_down_ratio_sum = daily_sectors.get('limit_down_ratio', pd.Series(0, index=daily_sectors.index)).sum()
                total_limit_up = limit_up_ratio_sum * total_stocks / len(daily_sectors)
                total_limit_down = limit_down_ratio_sum * total_stocks / len(daily_sectors)

            # 连板情况
            total_board_2_plus = 0
            if 'board_counts_2板' in daily_sectors.columns:
                total_board_2_plus = (daily_sectors['board_counts_2板'].fillna(0).sum() +
                                    daily_sectors['board_counts_3板'].fillna(0).sum())

                # 安全地添加4板以上
                if 'board_counts_4板' in daily_sectors.columns:
                    total_board_2_plus += daily_sectors['board_counts_4板'].fillna(0).sum()
            else:
                # 没有连板数据，使用涨停数据估算
                total_board_2_plus = max(0, total_limit_up * 0.3)  # 估算连板数量

            # 计算涨跌停比率
            limit_up_ratio = total_limit_up / max(total_stocks, 1)
            limit_down_ratio = total_limit_down / max(total_stocks, 1)

            # 强势板块数量（涨幅>2%或有涨停）
            if 'limit_up_count' in daily_sectors.columns:
                strong_sectors = len(daily_sectors[
                    (daily_sectors['change_percent'] > 0.02) |
                    (daily_sectors['limit_up_count'].fillna(0) > 0)
                ])
            else:
                strong_sectors = len(daily_sectors[daily_sectors['change_percent'] > 0.02])

            # 弱势板块数量（跌幅>2%或有跌停）
            if 'limit_down_count' in daily_sectors.columns:
                weak_sectors = len(daily_sectors[
                    (daily_sectors['change_percent'] < -0.02) |
                    (daily_sectors['limit_down_count'].fillna(0) > 0)
                ])
            else:
                weak_sectors = len(daily_sectors[daily_sectors['change_percent'] < -0.02])

            # 判断市场环境
            if (market_change > 0.01 and up_ratio > 0.6 and
                limit_up_ratio > 0.02 and total_board_2_plus > 5):
                return 'strong_market'
            elif (market_change < -0.01 and up_ratio < 0.4 and
                  limit_down_ratio > 0.01 and weak_sectors > strong_sectors):
                return 'weak_market'
            elif (abs(market_change) > 0.015 or market_volume_level in ['very_high', 'very_low'] or
                  total_limit_up + total_limit_down > total_stocks * 0.05):
                return 'volatile_market'
            else:
                return 'normal_market'

        except Exception as e:
            self.logger.warning(f"市场环境评估失败: {e}")
            return 'normal_market'
    
    def _calculate_technical_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """计算技术面评分"""
        scores = {}
        
        try:
            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')
                
                # 技术指标
                change_pct = row.get('change_percent', 0)
                ma5_slope = row.get('ma5_slope', 0)
                trend_strength = row.get('trend_strength', 0)
                
                # 突破信号
                breakthrough_score = 0
                if change_pct > 0.02:  # 大涨突破
                    breakthrough_score += 30
                elif change_pct > 0.01:  # 中等突破
                    breakthrough_score += 15
                elif change_pct > 0.005:  # 小幅突破
                    breakthrough_score += 5
                
                # 趋势评分
                trend_score = 0
                if ma5_slope > 0.001:
                    trend_score += 20
                elif ma5_slope > 0:
                    trend_score += 10
                
                if trend_strength > 0.5:
                    trend_score += 15
                elif trend_strength > 0.3:
                    trend_score += 8
                
                # 技术面总分
                tech_score = min(100, breakthrough_score + trend_score + change_pct * 1000)
                scores[sector_code] = max(0, tech_score)
                
        except Exception as e:
            self.logger.warning(f"技术面评分计算失败: {e}")

        # 输出技术面评分前10名
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
            self.logger.info(f"📊 技术面评分前10名:")
            for i, (sector, score) in enumerate(sorted_scores, 1):
                self.logger.info(f"   {i:2d}. {sector}: {score:.1f}分")

        return scores

    def _calculate_sentiment_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """计算情绪面评分（基于可用字段）"""
        scores = {}

        try:
            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')

                # 基础数据
                change_percent = row.get('change_percent', 0)
                up_count = row.get('up_count', 0)
                down_count = row.get('down_count', 0)
                stock_count = max(row.get('stock_count', 1), 1)

                # 尝试获取涨跌停数据，如果不存在则使用比率数据
                if 'limit_up_count' in row.index and not pd.isna(row['limit_up_count']):
                    limit_up_count = row['limit_up_count']
                    limit_down_count = row.get('limit_down_count', 0)
                    if pd.isna(limit_down_count):
                        limit_down_count = 0
                else:
                    # 使用比率数据估算
                    limit_up_ratio = row.get('limit_up_ratio', 0)
                    limit_down_ratio = row.get('limit_down_ratio', 0)
                    limit_up_count = limit_up_ratio * stock_count
                    limit_down_count = limit_down_ratio * stock_count

                # 尝试获取连板数据
                board_1 = 0
                board_2 = 0
                board_3 = 0
                board_4_plus = 0

                if 'board_counts_1板' in row.index and not pd.isna(row['board_counts_1板']):
                    board_1 = row['board_counts_1板']
                    board_2 = row.get('board_counts_2板', 0)
                    board_3 = row.get('board_counts_3板', 0)

                    # 计算4板以上
                    for board_level in ['board_counts_4板', 'board_counts_5板', 'board_counts_6板',
                                      'board_counts_7板', 'board_counts_8板', 'board_counts_9板', 'board_counts_10板及以上']:
                        if board_level in row.index and not pd.isna(row[board_level]):
                            board_4_plus += row[board_level]
                else:
                    # 如果没有连板数据，使用涨停数据估算
                    if limit_up_count > 0:
                        board_1 = max(1, limit_up_count * 0.7)  # 估算首板数量

                # 涨停评分（40分）
                limit_up_ratio = limit_up_count / stock_count
                limit_up_score = 0
                if limit_up_ratio > 0.1:  # 超过10%涨停
                    limit_up_score = 40
                elif limit_up_ratio > 0.05:  # 超过5%涨停
                    limit_up_score = 30
                elif limit_up_ratio > 0.02:  # 超过2%涨停
                    limit_up_score = 20
                elif limit_up_count > 0:
                    limit_up_score = 10

                # 连板评分（35分）
                cascade_score = 0
                if board_4_plus > 0:
                    cascade_score = 35  # 有4板以上高位连板
                elif board_3 > 0:
                    cascade_score = 28  # 有3板
                elif board_2 > 0:
                    cascade_score = 20  # 有2板
                elif board_1 >= 3:
                    cascade_score = 12  # 首板较多
                elif board_1 > 0:
                    cascade_score = 6   # 有首板

                # 涨跌比例评分（15分）
                if stock_count > 0:
                    up_ratio = up_count / stock_count
                    ratio_score = min(15, up_ratio * 20)
                else:
                    ratio_score = 0

                # 涨幅强度评分（10分）
                strength_score = 0
                if change_percent > 0.05:
                    strength_score = 10
                elif change_percent > 0.03:
                    strength_score = 8
                elif change_percent > 0.02:
                    strength_score = 6
                elif change_percent > 0.01:
                    strength_score = 4

                # 跌停惩罚
                limit_down_penalty = min(30, limit_down_count * 5)

                # 情绪面总分
                sentiment_score = min(100, max(0, limit_up_score + cascade_score + ratio_score + strength_score - limit_down_penalty))
                scores[sector_code] = sentiment_score
                

        except Exception as e:
            self.logger.warning(f"情绪面评分计算失败: {e}")

        # 输出情绪面评分前10名
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
            self.logger.info(f"🔥 情绪面评分前10名:")
            for i, (sector, score) in enumerate(sorted_scores, 1):
                self.logger.info(f"   {i:2d}. {sector}: {score:.1f}分")

        return scores

    def _calculate_relative_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """计算相对强度评分（基于可用字段）"""
        scores = {}

        try:
            # 计算各项指标的分位数
            change_data = daily_sectors['change_percent']
            volume_data = daily_sectors['volume'].replace(0, np.nan).dropna()
            up_count_data = daily_sectors['up_count']

            # 尝试获取涨停数据
            if 'limit_up_count' in daily_sectors.columns:
                self.logger.info(f"limit_up_count存在")
                limit_up_data = daily_sectors['limit_up_count'].fillna(0)
            else:
                # 使用比率数据估算
                limit_up_ratio_data = daily_sectors.get('limit_up_ratio', pd.Series(0, index=daily_sectors.index))
                stock_count_data = daily_sectors['stock_count']
                limit_up_data = limit_up_ratio_data * stock_count_data

            # 计算连板强度
            board_strength_data = pd.Series(0, index=daily_sectors.index)

            if 'board_counts_1板' in daily_sectors.columns:
                # 有连板数据
                board_strength_data = (
                    daily_sectors['board_counts_1板'].fillna(0) * 1 +
                    daily_sectors['board_counts_2板'].fillna(0) * 2 +
                    daily_sectors['board_counts_3板'].fillna(0) * 3
                )

                # 安全地添加高阶连板
                for board_level in ['board_counts_4板', 'board_counts_5板', 'board_counts_6板',
                                  'board_counts_7板', 'board_counts_8板', 'board_counts_9板', 'board_counts_10板及以上']:
                    if board_level in daily_sectors.columns:
                        multiplier = int(board_level.split('_')[2][0]) if board_level != 'board_counts_10板及以上' else 10
                        board_strength_data += daily_sectors[board_level].fillna(0) * multiplier
            else:
                # 没有连板数据，使用涨停数据估算
                self.logger.info(f"没有连板数据，使用涨停数据估算")
                board_strength_data = limit_up_data * 1.5  # 简单估算

            # 计算排名百分位
            change_ranks = change_data.rank(pct=True)
            volume_ranks = daily_sectors['volume'].rank(pct=True) if len(volume_data) > 0 else pd.Series(0.5, index=daily_sectors.index)
            limit_up_ranks = limit_up_data.rank(pct=True)
            board_strength_ranks = board_strength_data.rank(pct=True)

            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')

                # 相对涨幅排名（25分）
                change_rank = change_ranks.loc[idx]
                change_score = change_rank * 25

                # 相对成交量排名（20分）
                volume_rank = volume_ranks.loc[idx]
                volume_score = volume_rank * 20

                # 相对涨停数量排名（30分）
                limit_up_rank = limit_up_ranks.loc[idx]
                limit_up_score = limit_up_rank * 30

                # 相对连板强度排名（25分）
                board_strength_rank = board_strength_ranks.loc[idx]
                board_score = board_strength_rank * 25

                # 相对强度总分
                relative_score = min(100, change_score + volume_score + limit_up_score + board_score)
                scores[sector_code] = relative_score

        except Exception as e:
            self.logger.warning(f"相对强度评分计算失败: {e}")

        # 输出相对强度评分前10名
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
            self.logger.info(f"📈 相对强度评分前10名:")
            for i, (sector, score) in enumerate(sorted_scores, 1):
                self.logger.info(f"   {i:2d}. {sector}: {score:.1f}分")

        return scores

    def _get_adjusted_weights(self, market_env: str) -> Dict[str, float]:
        """获取根据市场环境调整后的权重"""
        try:
            adjustments = self.env_weight_adjustments.get(market_env, self.env_weight_adjustments['normal_market'])

            adjusted_weights = {}
            for dimension, base_weight in self.base_weights.items():
                adjusted_weights[dimension] = base_weight * adjustments[dimension]

            # 归一化权重
            total_weight = sum(adjusted_weights.values())
            for dimension in adjusted_weights:
                adjusted_weights[dimension] /= total_weight

            return adjusted_weights

        except Exception as e:
            self.logger.warning(f"权重调整失败: {e}")
            return self.base_weights

    def _calculate_confidence(self, tech_score: float, cap_score: float, sent_score: float, rel_score: float, market_env: str) -> float:
        """计算置信度"""
        try:
            # 基础置信度：各维度评分的一致性
            scores = [tech_score, cap_score, sent_score, rel_score]
            avg_score = np.mean(scores)
            std_score = np.std(scores)

            # 一致性越高，置信度越高
            consistency = max(0, 1 - std_score / max(avg_score, 1))

            # 市场环境调整
            env_adjustment = {
                'strong_market': 1.1,
                'normal_market': 1.0,
                'volatile_market': 0.9,
                'weak_market': 0.8
            }.get(market_env, 1.0)

            # 综合置信度
            confidence = min(1.0, consistency * env_adjustment)

            # 详细的置信度计算日志（仅在调试时启用）
            if hasattr(self, '_debug_confidence') and self._debug_confidence:
                self.logger.debug(f"置信度计算详情: 技术{tech_score:.1f} 资金{cap_score:.1f} "
                                f"情绪{sent_score:.1f} 相对{rel_score:.1f} -> "
                                f"平均{avg_score:.1f} 标准差{std_score:.1f} "
                                f"一致性{consistency:.2f} 环境调整{env_adjustment:.1f} "
                                f"最终置信度{confidence:.2f}")

            return confidence

        except Exception as e:
            self.logger.warning(f"置信度计算失败: {e}")
            return 0.5

    def get_top_potential_sectors(self, results: List[PotentialScore], top_n: int = 5, min_confidence: float = 0.3) -> List[PotentialScore]:
        """获取最具潜力的前N个板块"""
        try:
            # 过滤低置信度结果
            filtered_results = [r for r in results if r.confidence >= min_confidence]

            # 按总分排序并取前N个
            top_sectors = filtered_results[:top_n]

            return top_sectors

        except Exception as e:
            self.logger.warning(f"获取顶级潜力板块失败: {e}")
            return []
    
    def _calculate_capital_scores(self, daily_sectors: pd.DataFrame) -> Dict[str, float]:
        """计算资金面评分"""
        scores = {}
        
        try:
            # 计算成交量分位数
            volume_data = daily_sectors['volume'].replace(0, np.nan).dropna()
            if len(volume_data) > 0:
                volume_median = volume_data.median()
                volume_75 = volume_data.quantile(0.75)
                volume_90 = volume_data.quantile(0.90)
            else:
                volume_median = volume_75 = volume_90 = 1
            
            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')
                
                volume = row.get('volume', 0)
                change_pct = row.get('change_percent', 0)
                
                # 成交量评分
                volume_score = 0
                if volume > volume_90:
                    volume_score = 40
                elif volume > volume_75:
                    volume_score = 25
                elif volume > volume_median:
                    volume_score = 15
                else:
                    volume_score = 5
                
                # 量价配合评分
                volume_price_score = 0
                if change_pct > 0 and volume > volume_median:
                    volume_price_score = 30  # 放量上涨
                elif change_pct > 0 and volume <= volume_median:
                    volume_price_score = 10  # 缩量上涨
                elif change_pct < 0 and volume > volume_median:
                    volume_price_score = -10  # 放量下跌
                
                # 资金面总分
                capital_score = min(100, max(0, volume_score + volume_price_score))
                scores[sector_code] = capital_score
                
        except Exception as e:
            self.logger.warning(f"资金面评分计算失败: {e}")

        # 输出资金面评分前10名
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
            self.logger.info(f"💰 资金面评分前10名:")
            for i, (sector, score) in enumerate(sorted_scores, 1):
                self.logger.info(f"   {i:2d}. {sector}: {score:.1f}分")

        return scores
