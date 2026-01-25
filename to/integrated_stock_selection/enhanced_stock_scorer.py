#!/usr/bin/env python3
"""
增强个股评分器
包含板块内涨停股票相似性评分和连板评分
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from config import system_config
from sector_stock_mapper import SectorStockMapper

class EnhancedStockScorer:
    """增强个股评分器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sector_mapper = SectorStockMapper()
        self.lookback_days = system_config.max_lookback_days
        self.penalty_factor = system_config.limit_up_penalty_factor
    
    def calculate_limit_up_similarity_score(self, stock_data: pd.DataFrame,
                                          sector_results: List,
                                          current_date: str) -> pd.DataFrame:
        """
        计算板块内强势股票相似性评分（逻辑修复版）

        正确逻辑：
        1. 使用当日数据识别强势股
        2. 获取这些强势股的T-1日数据
        3. 用强势股T-1日数据与目标股票T日数据计算相似度
        4. 预测目标股票的上涨潜力
        """
        try:
            self.logger.info("计算板块内强势股票相似性评分（逻辑修复版）")

            data = stock_data.copy()
            current_date_obj = pd.to_datetime(current_date)

            # 获取当日数据
            daily_data = data[data['date'] == current_date_obj].copy()

            if len(daily_data) == 0:
                self.logger.warning(f"没有{current_date}的股票数据")
                data['limit_up_similarity_score'] = 0
                return data

            # 获取前一个交易日（从stock_data中的日期获取）
            available_dates = sorted(data['date'].unique())
            current_date_index = None
            for i, date in enumerate(available_dates):
                if date == current_date_obj:
                    current_date_index = i
                    break

            if current_date_index is None or current_date_index == 0:
                self.logger.warning(f"没有{current_date}的前一交易日数据，使用降级处理")
                data['limit_up_similarity_score'] = 30  # 基础分30分
                return data

            # 获取前一个交易日
            previous_date_obj = available_dates[current_date_index - 1]
            previous_data = data[data['date'] == previous_date_obj].copy()

            if len(previous_data) == 0:
                self.logger.warning(f"前一交易日数据为空，使用降级处理")
                data['limit_up_similarity_score'] = 30
                return data

            # 获取前5个优质板块
            top_sectors = [result.sector_code for result in sector_results[:5]]

            # 正确逻辑：用当日数据识别强势股，用强势股T-1日数据计算相似度
            similarity_scores = self._calculate_similarity_scores_with_correct_logic(
                daily_data, previous_data, top_sectors
            )

            # 将评分添加到数据中
            data['limit_up_similarity_score'] = data['stock_code'].map(similarity_scores).fillna(0)

            avg_score = data['limit_up_similarity_score'].mean()
            non_zero_count = (data['limit_up_similarity_score'] > 0).sum()

            self.logger.info(f"强势股相似性评分完成，平均分: {avg_score:.2f}, 非零评分: {non_zero_count}只")

            return data

        except Exception as e:
            self.logger.error(f"计算强势股相似性评分失败: {e}")
            stock_data['limit_up_similarity_score'] = 0
            return stock_data

    def _calculate_similarity_scores_with_correct_logic(self, daily_data: pd.DataFrame,
                                                       previous_data: pd.DataFrame,
                                                       top_sectors: List[str]) -> dict:
        """
        计算相似性评分（正确逻辑版）

        正确逻辑：
        1. 用当日数据识别强势股
        2. 获取这些强势股的T-1日数据
        3. 用强势股T-1日数据与目标股票T日数据计算相似度

        Args:
            daily_data: 当日数据（用于识别强势股和目标股票）
            previous_data: 前一日数据（用于获取强势股的历史特征）
            top_sectors: 优质板块列表
        """
        try:
            # 1. 预构建股票-板块映射
            stock_sector_map = self._build_stock_sector_map_batch(daily_data['stock_code'].tolist())

            # 2. 使用当日数据识别强势股票（简化逻辑，不限板块）
            strong_stocks = self._get_all_sector_strong_stocks_vectorized(
                daily_data, previous_data
            )

            # 3. 提取当日目标股票特征
            target_features = self._extract_features_vectorized(daily_data)

            # 4. 提取前一日所有股票特征（用于获取强势股的历史特征）
            previous_features = self._extract_features_vectorized(previous_data)

            # 5. 批量计算相似性评分
            similarity_scores = {}

            # 如果没有强势股，所有股票评分为0
            if not strong_stocks:
                for _, stock in daily_data.iterrows():
                    similarity_scores[stock['stock_code']] = 0
                return similarity_scores

            for _, stock in daily_data.iterrows():
                stock_code = stock['stock_code']
                stock_sectors = stock_sector_map.get(stock_code, [])

                # 检查目标股票是否属于优质板块
                overlap_sectors = [s for s in stock_sectors if s in top_sectors]
                if not overlap_sectors:
                    similarity_scores[stock_code] = 0
                    continue

                # 检查目标股票当日涨幅是否<4%
                current_pct_chg = stock.get('pct_chg', 0)
                if current_pct_chg >= 4.0:
                    similarity_scores[stock_code] = 0
                    continue

                # 计算与所有强势股的相似性（不再按板块过滤强势股）
                target_stock_features = target_features.get(stock_code, {})
                similarity_score = self._calculate_similarity_with_correct_logic(
                    target_stock_features, strong_stocks, previous_features
                )

                similarity_scores[stock_code] = similarity_score

            return similarity_scores

        except Exception as e:
            self.logger.error(f"向量化相似性计算失败: {e}")
            return {}

    def _build_stock_sector_map_batch(self, stock_codes: List[str]) -> dict:
        """批量构建股票-板块映射，避免重复查询"""
        try:
            stock_sector_map = {}

            # 批量查询，减少单次查询开销
            for stock_code in stock_codes:
                sectors = self.sector_mapper.get_stock_sectors(stock_code)
                stock_sector_map[stock_code] = sectors

            return stock_sector_map

        except Exception as e:
            self.logger.error(f"构建股票板块映射失败: {e}")
            return {}

    def _get_all_sector_strong_stocks_vectorized(self, daily_data: pd.DataFrame,
                                               previous_data: pd.DataFrame) -> List[dict]:
        """
        获取强势股票（简化逻辑）

        筛选条件：
        1. 当日涨幅 > 8%
        2. T-1日涨幅 < 4%
        3. 不限制板块

        Args:
            daily_data: 当日数据
            previous_data: 前一日数据

        Returns:
            强势股票列表
        """
        try:
            # 1. 从当日数据中找出涨幅>8%的股票
            current_strong = daily_data[daily_data['pct_chg'] > 8.0].copy()

            if current_strong.empty:
                self.logger.info("当日没有涨幅>8%的强势股")
                return []

            # 2. 检查这些股票在T-1日的涨幅是否<4%
            strong_stocks = []
            for _, stock in current_strong.iterrows():
                stock_code = stock['stock_code']

                # 查找该股票在前一日的数据
                prev_stock = previous_data[previous_data['stock_code'] == stock_code]
                if not prev_stock.empty:
                    prev_pct_chg = prev_stock.iloc[0]['pct_chg']
                    # 只有T-1日涨幅<4%的股票才算强势股
                    if prev_pct_chg < 4.0:
                        strong_stocks.append({
                            'stock_code': stock_code,
                            'pct_chg': stock['pct_chg'],
                            'prev_pct_chg': prev_pct_chg
                        })

            self.logger.info(f"识别到{len(strong_stocks)}只强势股（当日>8%，T-1<4%）")
            return strong_stocks

        except Exception as e:
            self.logger.error(f"获取强势股票失败: {e}")
            return []

    def _extract_features_vectorized(self, daily_data: pd.DataFrame) -> dict:
        """向量化提取所有股票的特征"""
        try:
            all_features = {}

            # 定义需要的特征列
            feature_columns = [
                'total_mv', 'turnover_rate', 'amount', 'pct_chg',
                'ma5', 'ma10', 'ma20', 'rsi', 'macd', 'macd_signal', 'bb_position',
                'pattern_golden_spoon', 'pattern_w_bottom', 'pattern_n_shape',
                'pattern_ascending_triangle', 'pattern_cup_handle', 'pattern_flag',
                'pattern_breakout', 'candlestick_strength', 'pattern_engulfing_bullish',
                'pattern_engulfing_bearish'
            ]

            # 向量化提取特征
            for _, stock in daily_data.iterrows():
                stock_code = stock['stock_code']
                features = {}

                for col in feature_columns:
                    if col in stock:
                        features[col] = stock[col]
                    else:
                        # 设置默认值
                        if col == 'rsi':
                            features[col] = 50
                        elif col == 'bb_position':
                            features[col] = 0.5
                        else:
                            features[col] = 0

                all_features[stock_code] = features

            return all_features

        except Exception as e:
            self.logger.error(f"向量化特征提取失败: {e}")
            return {}

    def _calculate_similarity_with_correct_logic(self, target_features: dict,
                                               strong_stocks: List[dict],
                                               previous_features: dict) -> float:
        """
        计算相似性评分（正确逻辑版）

        Args:
            target_features: 目标股票当日特征
            strong_stocks: 当日识别的强势股列表（全市场，不限板块）
            previous_features: 前一日所有股票特征
        """
        try:
            if not strong_stocks or not target_features:
                return 30  # 基础相似性

            similarity_scores = []

            # 对每个强势股，获取其前一日特征并计算相似度
            for strong_stock in strong_stocks:
                strong_code = strong_stock['stock_code']
                # 获取该强势股的前一日特征
                strong_previous_features = previous_features.get(strong_code, {})

                if not strong_previous_features:
                    continue

                # 计算目标股票当日特征与强势股前一日特征的相似性
                similarity = self._calculate_fast_similarity(target_features, strong_previous_features)
                similarity_scores.append(similarity)

            # 返回最高相似性评分
            max_similarity = max(similarity_scores) if similarity_scores else 0.3
            return max_similarity * 100

        except Exception as e:
            self.logger.error(f"相似性计算失败: {e}")
            return 30

    def _calculate_similarity_fast(self, target_features: dict,
                                 strong_stocks: List[dict],
                                 all_features: dict) -> float:
        """
        快速计算相似性评分（增加二次过滤逻辑）
        """
        try:
            if not strong_stocks or not target_features:
                return 0

            similarity_scores = []
            filtered_strong_stocks = []

            # 二次过滤逻辑修复：不应该过滤强势股本身，而是在选择目标股票时避免高风险
            # 这里我们保留所有识别到的强势股用于相似性计算
            for strong_stock in strong_stocks:
                strong_code = strong_stock['stock_code']
                strong_features = all_features.get(strong_code, {})

                if strong_features:
                    filtered_strong_stocks.append((strong_stock, strong_features))

            # 如果没有强势股特征，降低要求
            if not filtered_strong_stocks:
                for strong_stock in strong_stocks:
                    # 即使没有完整特征，也给予基础相似性
                    filtered_strong_stocks.append((strong_stock, {}))

            # 计算与过滤后强势股的相似性
            for strong_stock, strong_features in filtered_strong_stocks:
                similarity = self._calculate_fast_similarity(target_features, strong_features)
                similarity_scores.append(similarity)

            # 返回最高相似性评分
            max_similarity = max(similarity_scores) if similarity_scores else 0
            return max_similarity * 100

        except Exception as e:
            self.logger.error(f"快速相似性计算失败: {e}")
            return 0

    def _calculate_fast_similarity(self, target_features: dict, strong_previous_features: dict) -> float:
        """
        快速相似性计算（正确逻辑版）
        比较目标股票当日特征与强势股前日特征的相似性
        目标：寻找具有强势股昨日特征的今日股票
        """
        try:
            if not target_features and not strong_previous_features:
                return 0.3  # 基础相似性

            similarities = []

            # 1. 技术形态相似性（权重40%）- 关键：比较形态而非涨幅
            target_rsi = target_features.get('rsi', 50)
            strong_rsi = strong_previous_features.get('rsi', 50)

            # RSI相似性 - 更宽松的计算
            if target_rsi > 0 and strong_rsi > 0:
                rsi_diff = abs(target_rsi - strong_rsi)
                rsi_similarity = max(0.3, 1 - rsi_diff / 100)  # 更宽松的阈值
            else:
                rsi_similarity = 0.5  # 默认中等相似性

            # 均线位置相似性
            target_ma_position = self._get_ma_position(target_features)
            strong_ma_position = self._get_ma_position(strong_previous_features)
            ma_similarity = 1.0 if target_ma_position == strong_ma_position else 0.6

            technical_similarity = (rsi_similarity + ma_similarity) / 2
            similarities.append(technical_similarity * 0.4)

            # 2. 市值匹配度（权重30%）- 相似市值的股票更可能有相似表现
            target_mv = target_features.get('total_mv', 0)
            strong_mv = strong_previous_features.get('total_mv', 0)
            if target_mv > 0 and strong_mv > 0:
                mv_ratio = min(target_mv, strong_mv) / max(target_mv, strong_mv)
                mv_similarity = max(0.4, mv_ratio)  # 提高基础分
            else:
                mv_similarity = 0.5  # 提高默认相似性
            similarities.append(mv_similarity * 0.3)

            # 3. K线形态相似性（权重20%）- 比较K线形态特征
            pattern_similarity = self._calculate_pattern_similarity(target_features, strong_previous_features)
            similarities.append(pattern_similarity * 0.2)

            # 4. 流动性匹配度（权重10%）- 换手率相似性
            target_turnover = target_features.get('turnover_rate', 0)
            strong_turnover = strong_previous_features.get('turnover_rate', 0)
            if target_turnover > 0 and strong_turnover > 0:
                turnover_ratio = min(target_turnover, strong_turnover) / max(target_turnover, strong_turnover)
                turnover_similarity = max(0.5, turnover_ratio)  # 提高基础分
            else:
                turnover_similarity = 0.6  # 提高默认相似性
            similarities.append(turnover_similarity * 0.1)

            return sum(similarities) if similarities else 0

        except Exception as e:
            self.logger.error(f"快速相似性计算失败: {e}")
            return 0

    def _get_ma_position(self, features: dict) -> str:
        """获取均线位置"""
        try:
            close = features.get('close', 0)
            ma5 = features.get('ma5', 0)
            ma10 = features.get('ma10', 0)
            ma20 = features.get('ma20', 0)

            # 如果数据不完整，返回默认值
            if close <= 0 or ma5 <= 0:
                return "neutral"

            if close > ma5 > ma10 > ma20 and ma10 > 0 and ma20 > 0:
                return "strong_up"
            elif close > ma5 > ma10 and ma10 > 0:
                return "up"
            elif close > ma5:
                return "weak_up"
            elif close < ma5 < ma10 < ma20 and ma10 > 0 and ma20 > 0:
                return "strong_down"
            else:
                return "neutral"
        except:
            return "neutral"

    def _calculate_pattern_similarity(self, target_features: dict, strong_previous_features: dict) -> float:
        """计算K线形态相似性"""
        try:
            pattern_features = [
                'pattern_golden_spoon', 'pattern_w_bottom', 'pattern_n_shape',
                'pattern_ascending_triangle', 'pattern_cup_handle', 'pattern_flag',
                'pattern_breakout', 'pattern_engulfing_bullish'
            ]

            matches = 0
            total = 0

            for pattern in pattern_features:
                target_val = target_features.get(pattern, 0)
                strong_val = strong_previous_features.get(pattern, 0)

                if target_val > 0 or strong_val > 0:
                    total += 1
                    if target_val > 0 and strong_val > 0:
                        matches += 1

            if total == 0:
                return 0.5  # 默认中等相似性

            return matches / total

        except:
            return 0.5

    def _get_sector_strong_stocks(self, daily_data: pd.DataFrame,
                                 sectors: List[str], exclude_stock: str) -> List[Dict]:
        """获取板块内的强势股票（基于实际数据）"""
        try:
            strong_stocks = []

            for sector in sectors:
                # 获取板块内的股票
                sector_stocks = self.sector_mapper.get_sector_stocks(sector)

                # 过滤出当日强势的股票（涨幅前30%）
                sector_data_list = []
                for stock_code in sector_stocks:
                    if stock_code == exclude_stock:
                        continue

                    stock_data = daily_data[daily_data['stock_code'] == stock_code]
                    if len(stock_data) == 0:
                        continue

                    stock_row = stock_data.iloc[0]
                    pct_chg = stock_row.get('pct_chg', 0)

                    # 只考虑上涨的股票
                    if pct_chg > 0:
                        sector_data_list.append({
                            'stock_code': stock_code,
                            'sector': sector,
                            'pct_chg': pct_chg,
                            'market_cap': stock_row.get('total_mv', 0),
                            'volume_ratio': stock_row.get('turnover_rate', 0),
                            'amount': stock_row.get('amount', 0)
                        })

                # 按涨幅排序，取前30%作为强势股票
                if sector_data_list:
                    sector_data_list.sort(key=lambda x: x['pct_chg'], reverse=True)
                    top_count = max(1, len(sector_data_list) // 3)  # 至少1只，最多前1/3
                    strong_stocks.extend(sector_data_list[:top_count])

            self.logger.debug(f"找到{len(strong_stocks)}只强势股票")
            return strong_stocks

        except Exception as e:
            self.logger.error(f"获取板块强势股票失败: {e}")
            return []
    
    def _calculate_stock_similarity(self, target_stock: pd.Series,
                                  strong_stocks: List[Dict],
                                  daily_data: pd.DataFrame) -> float:
        """
        计算股票与强势股票的相似性（基于强势股昨日特征数据）

        改进逻辑：
        1. 与强势股昨日的技术指标对比
        2. 与强势股昨日的K线形态对比
        3. 与强势股昨日的涨幅相似性对比
        4. 融入新增的K线形态特征
        """
        try:
            if not strong_stocks:
                return 0

            # 获取目标股票当日特征
            target_features = self._extract_target_features(target_stock)

            similarity_scores = []

            for strong_stock in strong_stocks:
                # 获取强势股昨日特征数据
                strong_yesterday_features = self._get_strong_stock_yesterday_features(
                    strong_stock, daily_data
                )

                if not strong_yesterday_features:
                    continue

                # 计算各维度相似性
                similarity_components = {}

                # 1. 技术指标相似性 (40%权重)
                similarity_components['technical'] = self._calculate_technical_similarity(
                    target_features, strong_yesterday_features
                )

                # 2. K线形态相似性 (30%权重) - 使用新增的形态特征
                similarity_components['pattern'] = self._calculate_pattern_similarity(
                    target_features, strong_yesterday_features
                )

                # 3. 涨幅相似性 (20%权重) - 与强势股昨日涨幅对比
                similarity_components['return'] = self._calculate_return_similarity(
                    target_features, strong_yesterday_features
                )

                # 4. 市值流动性相似性 (10%权重)
                similarity_components['liquidity'] = self._calculate_liquidity_similarity(
                    target_features, strong_yesterday_features
                )

                # 综合相似性评分
                total_similarity = (
                    similarity_components['technical'] * 0.25 +
                    similarity_components['pattern'] * 0.5 +
                    similarity_components['return'] * 0.15 +
                    similarity_components['liquidity'] * 0.1
                )

                similarity_scores.append(total_similarity)

            # 返回最高相似性评分
            max_similarity = max(similarity_scores) if similarity_scores else 0

            # 转换为0-100分制
            return max_similarity * 100

        except Exception as e:
            self.logger.error(f"计算股票相似性失败: {e}")
            return 0

    def _extract_target_features(self, target_stock: pd.Series) -> dict:
        """提取目标股票的特征"""
        return {
            # 基础特征
            'total_mv': target_stock.get('total_mv', 0),
            'turnover_rate': target_stock.get('turnover_rate', 0),
            'amount': target_stock.get('amount', 0),
            'pct_chg': target_stock.get('pct_chg', 0),

            # 技术指标
            'ma5': target_stock.get('ma5', 0),
            'ma10': target_stock.get('ma10', 0),
            'ma20': target_stock.get('ma20', 0),
            'rsi': target_stock.get('rsi', 50),
            'macd': target_stock.get('macd', 0),
            'macd_signal': target_stock.get('macd_signal', 0),
            'bb_position': target_stock.get('bb_position', 0.5),

            # K线形态特征（新增）
            'pattern_golden_spoon': target_stock.get('pattern_golden_spoon', 0),
            'pattern_w_bottom': target_stock.get('pattern_w_bottom', 0),
            'pattern_n_shape': target_stock.get('pattern_n_shape', 0),
            'pattern_ascending_triangle': target_stock.get('pattern_ascending_triangle', 0),
            'pattern_cup_handle': target_stock.get('pattern_cup_handle', 0),
            'pattern_flag': target_stock.get('pattern_flag', 0),
            'pattern_breakout': target_stock.get('pattern_breakout', 0),
            'candlestick_strength': target_stock.get('candlestick_strength', 0),
            'pattern_engulfing_bullish': target_stock.get('pattern_engulfing_bullish', 0),
            'pattern_engulfing_bearish': target_stock.get('pattern_engulfing_bearish', 0),
        }

    def _get_strong_stock_yesterday_features(self, strong_stock: dict, daily_data: pd.DataFrame) -> dict:
        """获取强势股昨日的特征数据"""
        try:
            stock_code = strong_stock.get('stock_code', '')
            if not stock_code:
                return {}

            # 查找该股票昨日的数据
            stock_yesterday = daily_data[daily_data['stock_code'] == stock_code]

            if stock_yesterday.empty:
                return {}

            # 取最新的一条记录（应该是昨日数据）
            yesterday_row = stock_yesterday.iloc[-1]

            return self._extract_target_features(yesterday_row)

        except Exception as e:
            self.logger.error(f"获取强势股昨日特征失败: {e}")
            return {}

    def _calculate_technical_similarity(self, target: dict, strong_yesterday: dict) -> float:
        """计算技术指标相似性"""
        try:
            similarities = []

            # 均线位置相似性
            for ma_period in ['ma5', 'ma10', 'ma20']:
                target_val = target.get(ma_period, 0)
                strong_val = strong_yesterday.get(ma_period, 0)

                if target_val > 0 and strong_val > 0:
                    # 计算价格相对均线的位置相似性
                    target_ratio = target.get('pct_chg', 0) / 100 + 1  # 当日涨幅
                    strong_ratio = strong_yesterday.get('pct_chg', 0) / 100 + 1  # 昨日涨幅

                    ratio_similarity = min(target_ratio, strong_ratio) / max(target_ratio, strong_ratio)
                    similarities.append(ratio_similarity)

            # RSI相似性
            target_rsi = target.get('rsi', 50)
            strong_rsi = strong_yesterday.get('rsi', 50)
            rsi_diff = abs(target_rsi - strong_rsi)
            rsi_similarity = max(0, 1 - rsi_diff / 50)  # RSI差异在50以内
            similarities.append(rsi_similarity)

            # MACD相似性
            target_macd = target.get('macd', 0)
            strong_macd = strong_yesterday.get('macd', 0)
            if abs(target_macd) > 0.001 and abs(strong_macd) > 0.001:
                macd_similarity = min(abs(target_macd), abs(strong_macd)) / max(abs(target_macd), abs(strong_macd))
                similarities.append(macd_similarity)

            # 布林带位置相似性
            target_bb = target.get('bb_position', 0.5)
            strong_bb = strong_yesterday.get('bb_position', 0.5)
            bb_diff = abs(target_bb - strong_bb)
            bb_similarity = max(0, 1 - bb_diff)
            similarities.append(bb_similarity)

            return sum(similarities) / len(similarities) if similarities else 0.5

        except Exception as e:
            self.logger.error(f"计算技术指标相似性失败: {e}")
            return 0.5

    def _calculate_pattern_similarity(self, target: dict, strong_yesterday: dict) -> float:
        """计算K线形态相似性"""
        try:
            pattern_features = [
                'pattern_golden_spoon', 'pattern_w_bottom', 'pattern_n_shape',
                'pattern_ascending_triangle', 'pattern_cup_handle', 'pattern_flag',
                'pattern_breakout', 'pattern_engulfing_bullish', 'pattern_engulfing_bearish'
            ]

            similarities = []

            # 形态匹配度
            for pattern in pattern_features:
                target_val = target.get(pattern, 0)
                strong_val = strong_yesterday.get(pattern, 0)

                # 如果都有该形态，相似性为1；都没有为0；一个有一个没有为0
                if target_val == 1 and strong_val == 1:
                    similarities.append(1.0)  # 完全匹配
                    self.logger.info(f"k线形态匹配：{pattern}")
                elif target_val == 0 and strong_val == 0:
                    similarities.append(0.0)  # 都没有，中性
                else:
                    similarities.append(0.0)  # 不匹配

            # K线强度相似性
            target_strength = target.get('candlestick_strength', 0)
            strong_strength = strong_yesterday.get('candlestick_strength', 0)

            if abs(target_strength) > 0.01 and abs(strong_strength) > 0.01:
                strength_similarity = min(abs(target_strength), abs(strong_strength)) / max(abs(target_strength), abs(strong_strength))
                # 如果方向相同（都为正或都为负），加分
                if (target_strength > 0 and strong_strength > 0) or (target_strength < 0 and strong_strength < 0):
                    strength_similarity *= 1.2  # 方向一致加分
                similarities.append(min(strength_similarity, 1.0))

            return sum(similarities) / len(similarities) if similarities else 0.5

        except Exception as e:
            self.logger.error(f"计算形态相似性失败: {e}")
            return 0.5

    def _calculate_return_similarity(self, target: dict, strong_yesterday: dict) -> float:
        """计算涨幅相似性（与强势股昨日涨幅对比）"""
        try:
            target_return = target.get('pct_chg', 0)
            strong_return = strong_yesterday.get('pct_chg', 0)

            # 涨幅差异
            return_diff = abs(target_return - strong_return)

            # 10%差异内为相似，差异越小相似性越高
            similarity = max(0, 1 - return_diff / 10)

            # 如果都是上涨，额外加分
            if target_return > 0 and strong_return > 0:
                similarity *= 1.1

            return min(similarity, 1.0)

        except Exception as e:
            self.logger.error(f"计算涨幅相似性失败: {e}")
            return 0.5

    def _calculate_liquidity_similarity(self, target: dict, strong_yesterday: dict) -> float:
        """计算市值流动性相似性"""
        try:
            similarities = []

            # 市值相似性
            target_mv = target.get('total_mv', 0)
            strong_mv = strong_yesterday.get('total_mv', 0)

            if target_mv > 0 and strong_mv > 0:
                mv_ratio = min(target_mv, strong_mv) / max(target_mv, strong_mv)
                mv_similarity = mv_ratio ** 0.5  # 开方减少差异影响
                similarities.append(mv_similarity)

            # 换手率相似性
            target_turnover = target.get('turnover_rate', 0)
            strong_turnover = strong_yesterday.get('turnover_rate', 0)

            if target_turnover > 0 and strong_turnover > 0:
                turnover_ratio = min(target_turnover, strong_turnover) / max(target_turnover, strong_turnover)
                turnover_similarity = turnover_ratio ** 0.3  # 换手率差异容忍度更高
                similarities.append(turnover_similarity)

            # 成交额相似性
            target_amount = target.get('amount', 0)
            strong_amount = strong_yesterday.get('amount', 0)

            if target_amount > 0 and strong_amount > 0:
                amount_ratio = min(target_amount, strong_amount) / max(target_amount, strong_amount)
                amount_similarity = amount_ratio ** 0.3
                similarities.append(amount_similarity)

            return sum(similarities) / len(similarities) if similarities else 0.5

        except Exception as e:
            self.logger.error(f"计算流动性相似性失败: {e}")
            return 0.5
    
    def calculate_consecutive_limit_penalty(self, stock_data: pd.DataFrame,
                                          current_date: str) -> pd.DataFrame:
        """
        计算近10天累计涨幅惩罚评分

        逻辑：
        1. 计算所有个股近10天累计涨幅
        2. 排除新股（单日涨幅>30%）
        3. 找出最大累计涨幅
        4. 计算每个股票距离最大涨幅的惩罚
        """
        try:
            self.logger.info("计算近10天累计涨幅惩罚评分")

            data = stock_data.copy()

            # 高效计算：一次性加载近10天的历史数据
            cumulative_returns, max_cumulative_return = self._calculate_all_cumulative_returns(
                current_date, lookback_days=10
            )

            if max_cumulative_return is None:
                self.logger.warning("无法计算累计涨幅，使用默认惩罚0")
                data['consecutive_limit_penalty'] = 0
                return data

            # 计算每个股票的惩罚评分
            penalty_scores = {}
            for stock_code, cumulative_return in cumulative_returns.items():
                penalty = self._calculate_cumulative_return_penalty(
                    cumulative_return, max_cumulative_return
                )
                penalty_scores[stock_code] = penalty

            # 将惩罚评分和累计涨幅数据添加到数据中
            data['consecutive_limit_penalty'] = data['stock_code'].map(penalty_scores).fillna(0)
            data['cumulative_return_10d'] = data['stock_code'].map(cumulative_returns).fillna(0)

            # 统计信息
            avg_penalty = data['consecutive_limit_penalty'].mean()
            non_zero_penalty = (data['consecutive_limit_penalty'] > 0).sum()

            self.logger.info(f"累计涨幅惩罚评分完成")
            self.logger.info(f"  最大10天累计涨幅: {max_cumulative_return:.2f}%")
            self.logger.info(f"  平均惩罚: {avg_penalty:.2f}, 非零惩罚: {non_zero_penalty}只")

            return data

        except Exception as e:
            self.logger.error(f"计算累计涨幅惩罚评分失败: {e}")
            stock_data['consecutive_limit_penalty'] = 0
            return stock_data
    
    def _calculate_all_cumulative_returns(self, current_date: str, lookback_days: int = 10) -> Tuple[Dict[str, float], float]:
        """
        高效计算所有股票的近N天累计涨幅

        Returns:
            Tuple[Dict[股票代码: 累计涨幅], 最大累计涨幅]
        """
        try:
            from pathlib import Path
            import pandas as pd

            # 获取近N个交易日的日期列表
            trading_dates = self._get_recent_trading_dates(current_date, lookback_days)

            if len(trading_dates) < 2:
                self.logger.warning(f"交易日数据不足，无法计算累计涨幅")
                return {}, None

            # 批量加载历史数据（高效方式）
            # 使用相对于当前文件的路径
            current_file_dir = Path(__file__).parent  # integrated_stock_selection目录
            features_dir = current_file_dir.parent / "local_data" / "features" / "daily"

            all_historical_data = []
            for date_str in trading_dates:
                file_path = features_dir / f"{date_str}.parquet"
                if file_path.exists():
                    try:
                        daily_data = pd.read_parquet(file_path)
                        if 'stock_code' in daily_data.columns and 'pct_chg' in daily_data.columns:
                            daily_data['date'] = date_str
                            all_historical_data.append(daily_data[['stock_code', 'date', 'pct_chg']])
                    except Exception as e:
                        self.logger.warning(f"加载{date_str}数据失败: {e}")

            if not all_historical_data:
                return {}, None

            # 合并所有历史数据
            historical_df = pd.concat(all_historical_data, ignore_index=True)

            # 向量化计算累计涨幅
            cumulative_returns = {}
            valid_stocks = set()

            for stock_code in historical_df['stock_code'].unique():
                stock_data_subset = historical_df[historical_df['stock_code'] == stock_code].copy()

                # 按日期排序
                stock_data_subset = stock_data_subset.sort_values('date')

                # 检查是否为新股（单日涨幅>30%）
                if self._is_new_stock(stock_data_subset):
                    continue

                # 计算累计涨幅：(1+r1)*(1+r2)*...*(1+rn) - 1
                pct_changes = stock_data_subset['pct_chg'].fillna(0) / 100.0  # 转换为小数

                # 过滤异常值
                pct_changes = pct_changes.clip(-0.3, 0.3)  # 限制在-20%到20%之间

                if len(pct_changes) >= 2:  # 至少需要2天数据
                    cumulative_return = (1 + pct_changes).prod() - 1
                    cumulative_return_pct = cumulative_return * 100  # 转换为百分比

                    cumulative_returns[stock_code] = cumulative_return_pct
                    valid_stocks.add(stock_code)

            # 找出最大累计涨幅
            if cumulative_returns:
                max_cumulative_return = max(cumulative_returns.values())
                self.logger.info(f"计算了{len(cumulative_returns)}只股票的累计涨幅，最大涨幅: {max_cumulative_return:.2f}%")
            else:
                max_cumulative_return = None
                self.logger.warning("没有有效的累计涨幅数据")

            return cumulative_returns, max_cumulative_return

        except Exception as e:
            self.logger.error(f"计算累计涨幅失败: {e}")
            return {}, None

    def _get_recent_trading_dates(self, current_date: str, days: int) -> list:
        """获取指定日期前N个交易日的日期列表"""
        try:
            from pathlib import Path
            import datetime

            # 使用相对于当前文件的路径
            current_file_dir = Path(__file__).parent  # integrated_stock_selection目录
            features_dir = current_file_dir.parent / "local_data" / "features" / "daily"

            # 向前推算days*1.5天，然后筛选存在数据文件的日期
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

    def _is_new_stock(self, stock_data: pd.DataFrame) -> bool:
        """
        判断是否为新股
        如果某一天涨幅超过30%，说明这个股票是新股
        """
        try:
            if 'pct_chg' not in stock_data.columns or len(stock_data) == 0:
                return True

            # 检查是否有单日涨幅超过30%
            max_daily_change = stock_data['pct_chg'].max()
            if max_daily_change > 30:
                return True

            # 检查是否有异常的连续大涨（可能是新股特征）
            large_gains = (stock_data['pct_chg'] > 15).sum()
            if large_gains >= 3:  # 3天以上大涨15%+
                return True

            return False

        except Exception as e:
            self.logger.error(f"判断新股失败: {e}")
            return True  # 出错时保守处理，认为是新股

    def _calculate_cumulative_return_penalty(self, cumulative_return: float,
                                           max_cumulative_return: float) -> float:
        """
        计算基于累计涨幅的惩罚评分

        Args:
            cumulative_return: 个股近10天累计涨幅(%)
            max_cumulative_return: 市场最大累计涨幅(%)

        Returns:
            惩罚评分 (0-50分)
        """
        try:
            if max_cumulative_return <= 0 or cumulative_return <= 0:
                return 0

            # 计算相对涨幅比例
            relative_ratio = cumulative_return / max_cumulative_return

            # # 分段惩罚函数：距离最大涨幅越近，惩罚越重（降低阈值，更严格）
            # if relative_ratio >= 0.1:  # 达到10%以上最大涨幅
            #     penalty = 10 + 10 * (relative_ratio - 0.1) / 0.1  # 10-20分惩罚
            # else:
            #     penalty = 0

            if relative_ratio >= 0.8:  # 达到80%以上最大涨幅
                penalty = 35 + 15 * (relative_ratio - 0.8) / 0.2  # 35-50分惩罚
            elif relative_ratio >= 0.6:  # 达到60%-80%
                penalty = 20 + 15 * (relative_ratio - 0.6) / 0.2  # 20-35分惩罚
            elif relative_ratio >= 0.4:  # 达到40%-60%
                penalty = 10 + 10 * (relative_ratio - 0.4) / 0.2  # 10-20分惩罚
            elif relative_ratio >= 0.3:  # 达到30%-40%
                penalty = 5 + 5 * (relative_ratio - 0.3) / 0.1   # 5-10分惩罚
            elif relative_ratio >= 0.2:  # 达到20%-30%
                penalty = 2 + 3 * (relative_ratio - 0.2) / 0.1   # 2-5分惩罚
            else:  # 低于20%
                penalty = 0

            # 额外考虑绝对涨幅
            if cumulative_return >= 50:  # 累计涨幅超过50%额外惩罚
                extra_penalty = min((cumulative_return - 50) / 10, 10)  # 最多额外10分
                penalty += extra_penalty

            return min(penalty, 50)  # 最大惩罚50分

        except Exception as e:
            self.logger.error(f"计算累计涨幅惩罚失败: {e}")
            return 0

    def calculate_cumulative_return_data_only(self, data: pd.DataFrame,
                                            current_date: str) -> pd.DataFrame:
        """
        只计算近10日累计涨幅数据，不进行惩罚评分
        用于后续过滤阶段使用
        """
        try:
            self.logger.info("计算近10天累计涨幅数据（仅用于过滤）")

            # 获取累计涨幅数据
            cumulative_returns = self._calculate_all_cumulative_returns(data, current_date)

            if not cumulative_returns:
                self.logger.warning("没有计算到任何累计涨幅数据")
                data['cumulative_return_10d'] = 0
                return data

            max_cumulative_return = max(cumulative_returns.values())
            self.logger.info(f"计算了{len(cumulative_returns)}只股票的累计涨幅，最大涨幅: {max_cumulative_return:.2f}%")

            # 只添加累计涨幅数据，不计算惩罚评分
            data['cumulative_return_10d'] = data['stock_code'].map(cumulative_returns).fillna(0)
            # 设置惩罚评分为0（取消惩罚）
            data['consecutive_limit_penalty'] = 0

            self.logger.info("累计涨幅数据计算完成（已取消惩罚评分）")
            self.logger.info(f"  最大10天累计涨幅: {max_cumulative_return:.2f}%")

            return data

        except Exception as e:
            self.logger.error(f"计算累计涨幅数据失败: {e}")
            # 设置默认值
            data['cumulative_return_10d'] = 0
            data['consecutive_limit_penalty'] = 0
            return data


    def calculate_enhanced_scores(self, stock_data: pd.DataFrame,
                                sector_results: List, 
                                current_date: str) -> pd.DataFrame:
        """计算所有增强评分"""
        try:
            self.logger.info("计算增强个股评分")
            
            # 计算涨停相似性评分
            data = self.calculate_limit_up_similarity_score(
                stock_data, sector_results, current_date
            )

            # 计算近10日累计涨幅数据（用于后续过滤，不进行惩罚评分）
            data = self.calculate_cumulative_return_data_only(data, current_date)
            
            self.logger.info("增强个股评分计算完成")
            
            return data
            
        except Exception as e:
            self.logger.error(f"计算增强评分失败: {e}")
            return stock_data
