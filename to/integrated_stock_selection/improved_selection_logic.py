#!/usr/bin/env python3
"""
改进的选股逻辑 - 基于分析结果的多重改进
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class ImprovedStockSelector:
    """改进的股票选择器"""
    
    def __init__(self):
        self.calibration_params = None
        self.dynamic_thresholds = None
        self.pattern_weights = None
        
    def calibrate_confidence(self, probabilities: np.ndarray, true_labels: np.ndarray) -> Dict:
        """
        使用温度缩放校准置信度
        基于Platt Scaling的改进版本
        """
        from scipy.optimize import minimize_scalar
        
        def temperature_scaling_loss(temperature):
            """温度缩放损失函数"""
            scaled_probs = self._apply_temperature_scaling(probabilities, temperature)
            # 使用交叉熵损失
            epsilon = 1e-15
            scaled_probs = np.clip(scaled_probs, epsilon, 1 - epsilon)
            loss = -np.mean(np.log(scaled_probs[np.arange(len(true_labels)), true_labels]))
            return loss
        
        # 优化温度参数
        result = minimize_scalar(temperature_scaling_loss, bounds=(0.1, 10.0), method='bounded')
        optimal_temperature = result.x
        
        print(f"📊 置信度校准完成，最优温度参数: {optimal_temperature:.3f}")
        
        return {
            'temperature': optimal_temperature,
            'calibration_loss': result.fun
        }
    
    def _apply_temperature_scaling(self, probabilities: np.ndarray, temperature: float) -> np.ndarray:
        """应用温度缩放"""
        # 将概率转换为logits
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        logits = np.log(probabilities / (1 - probabilities + epsilon))
        
        # 应用温度缩放
        scaled_logits = logits / temperature
        
        # 转换回概率
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        
        # 归一化确保概率和为1
        scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)
        
        return scaled_probs
    
    def calculate_dynamic_thresholds(self, training_data: pd.DataFrame) -> Dict:
        """
        基于训练数据计算动态阈值
        考虑市场环境、波动率等因素
        """
        thresholds = {}
        
        # 基础阈值（保守设置）
        base_confidence = 0.65
        base_advantage = 0.20
        
        # 根据市场波动率调整
        volatility = training_data['pct_chg'].std()
        if volatility > 3.0:  # 高波动市场
            thresholds['confidence'] = base_confidence + 0.10
            thresholds['advantage'] = base_advantage + 0.05
            thresholds['market_regime'] = 'high_volatility'
        elif volatility < 1.5:  # 低波动市场
            thresholds['confidence'] = base_confidence - 0.05
            thresholds['advantage'] = base_advantage - 0.05
            thresholds['market_regime'] = 'low_volatility'
        else:  # 正常市场
            thresholds['confidence'] = base_confidence
            thresholds['advantage'] = base_advantage
            thresholds['market_regime'] = 'normal'
        
        # 根据历史胜率调整
        if 'future_return_1d' in training_data.columns:
            win_rate = (training_data['future_return_1d'] > 0).mean()
            if win_rate < 0.5:  # 历史胜率低，提高阈值
                thresholds['confidence'] += 0.05
                thresholds['advantage'] += 0.05
        
        print(f"📊 动态阈值计算完成:")
        print(f"   市场环境: {thresholds['market_regime']}")
        print(f"   置信度阈值: {thresholds['confidence']:.3f}")
        print(f"   优势阈值: {thresholds['advantage']:.3f}")
        
        return thresholds
    
    def calculate_pattern_weights(self, training_data: pd.DataFrame) -> Dict:
        """
        基于K线形态分析计算权重
        """
        weights = {
            'bald_bearish_penalty': 0.0,
            'consecutive_decline_penalty': 0.0,
            'high_rsi_penalty': 0.0,
            'long_upper_shadow_penalty': 0.0,
            'below_all_ma_penalty': 0.0
        }

        # 分析光头阴线的负面影响
        # 基于之前的分析结果：光头阴线次日下跌概率48.9%，略高于上涨概率48.6%
        weights['bald_bearish_penalty'] = 0.15  # 15%的惩罚

        # 连续下跌惩罚
        weights['consecutive_decline_penalty'] = 0.10  # 10%的惩罚

        # 高RSI惩罚（过热信号）
        weights['high_rsi_penalty'] = 0.12  # 12%的惩罚

        # 长上影线惩罚（高位压力信号）
        weights['long_upper_shadow_penalty'] = 0.20  # 20%的惩罚

        # 股价在所有均线下方惩罚（趋势向下）
        weights['below_all_ma_penalty'] = 0.18  # 18%的惩罚
        
        print(f"📊 形态权重计算完成:")
        for key, value in weights.items():
            print(f"   {key}: {value:.3f}")
        
        return weights
    
    def enhanced_stock_selection(self, 
                                daily_data: pd.DataFrame,
                                predictions: np.ndarray,
                                probabilities: np.ndarray,
                                feature_names: List[str]) -> List[Dict]:
        """
        增强的股票选择逻辑
        """
        if self.dynamic_thresholds is None:
            raise ValueError("请先调用calculate_dynamic_thresholds设置动态阈值")
        
        if self.pattern_weights is None:
            raise ValueError("请先调用calculate_pattern_weights设置形态权重")
        
        # 应用置信度校准
        if self.calibration_params is not None:
            calibrated_probs = self._apply_temperature_scaling(
                probabilities, self.calibration_params['temperature']
            )
        else:
            calibrated_probs = probabilities
        
        candidate_stocks = []

        # 统计各种过滤情况
        total_stocks = len(predictions)
        confidence_filtered = 0
        advantage_filtered = 0
        prediction_filtered = 0
        pattern_filtered = 0

        # 添加日志记录器
        import logging
        logger = logging.getLogger(__name__)

        for i in range(len(predictions)):
            # 二分类模型：0=其他股票, 1=Top10%股票
            prob_other = calibrated_probs[i, 0]  # 其他股票概率
            prob_top10 = calibrated_probs[i, 1]  # Top10%股票概率

            # 对于二分类，Top10%概率就是上涨优势
            total_rise_prob = prob_top10
            total_fall_prob = prob_other
            rise_advantage = prob_top10 - prob_other
            
            # 统计过滤原因
            if total_rise_prob < self.dynamic_thresholds['confidence']:
                confidence_filtered += 1
                continue

            if rise_advantage < self.dynamic_thresholds['advantage']:
                advantage_filtered += 1
                continue

            # 二分类：只选择预测为Top10%的股票
            if predictions[i] != 1:
                prediction_filtered += 1
                continue

            # 基础筛选条件通过，计算基础评分
            if predictions[i] == 1:  # Top10%
                base_score = prob_top10 + rise_advantage
                category = "Top10%"
            else:  # 其他（理论上不会到这里，因为上面已经过滤了）
                base_score = prob_other + rise_advantage
                category = "其他"
                
            # 应用形态惩罚
            penalty_score = 0.0
            penalty_reasons = []

            # 检查K线形态惩罚
            row = daily_data.iloc[i]

            # 1. 光头阴线惩罚
            if self._is_bald_bearish_candle(row):
                penalty_score += self.pattern_weights['bald_bearish_penalty']
                penalty_reasons.append("光头阴线")

            # 2. 长上影线惩罚
            if self._is_long_upper_shadow(row):
                penalty_score += self.pattern_weights['long_upper_shadow_penalty']
                penalty_reasons.append("长上影线")

            # 3. 股价在所有均线下方惩罚
            if self._is_below_all_ma(row):
                penalty_score += self.pattern_weights['below_all_ma_penalty']
                penalty_reasons.append("均线下方")

            # 4. 高RSI惩罚
            if self._is_high_rsi(row):
                penalty_score += self.pattern_weights['high_rsi_penalty']
                penalty_reasons.append("RSI过热")

            # 5. 连续下跌惩罚
            if self._is_consecutive_decline(row):
                penalty_score += self.pattern_weights['consecutive_decline_penalty']
                penalty_reasons.append("连续下跌")
                
            # 计算最终评分
            final_score = base_score - penalty_score

            # 只有最终评分仍然为正才考虑
            if final_score > 0:
                candidate_stocks.append({
                    'index': i,
                    'stock_code': row['stock_code'],
                    'base_score': base_score,
                    'penalty_score': penalty_score,
                    'final_score': final_score,
                    'category': category,
                    'rise_prob': total_rise_prob,
                    'fall_prob': total_fall_prob,
                    'rise_advantage': rise_advantage,
                    'penalty_reasons': penalty_reasons,
                    'prediction': predictions[i]
                })
            else:
                # 记录被形态惩罚过滤的股票
                pattern_filtered += 1

        # 只在候选股票为0时输出诊断信息
        if len(candidate_stocks) == 0 and total_stocks > 0:
            logger.warning(f"❌ 没有选出任何股票，过滤统计:")
            logger.warning(f"   总股票数: {total_stocks}")
            logger.warning(f"   置信度不足(<{self.dynamic_thresholds['confidence']:.1%}): {confidence_filtered}")
            logger.warning(f"   优势不足(<{self.dynamic_thresholds['advantage']:.1%}): {advantage_filtered}")
            logger.warning(f"   预测类别非上涨: {prediction_filtered}")
            logger.warning(f"   形态惩罚过滤: {pattern_filtered}")

            # 分析概率分布
            if len(calibrated_probs) > 0:
                # 二分类：使用Top10%概率
                rise_probs = calibrated_probs[:, 1]  # Top10%概率
                logger.warning(f"   Top10%概率分布: 平均{rise_probs.mean():.1%}, 最大{rise_probs.max():.1%}")
                logger.warning(f"   达到置信度要求的股票数: {(rise_probs >= self.dynamic_thresholds['confidence']).sum()}")

        return candidate_stocks
    
    def _is_bald_bearish_candle(self, row: pd.Series) -> bool:
        """判断是否为光头阴线（使用已计算的特征）"""
        try:
            # 优先使用已计算的光头阴线特征
            if 'pattern_bare_head_yin' in row and pd.notna(row['pattern_bare_head_yin']):
                # 如果特征值大于0.5，认为是明显的光头阴线（提高阈值）
                return row['pattern_bare_head_yin'] > 0.5
            else:
                # 降级到手动计算
                is_negative = row['close'] < row['open']
                if not is_negative:
                    return False

                upper_shadow = row['high'] - max(row['open'], row['close'])
                body_size = abs(row['open'] - row['close'])

                return upper_shadow / body_size < 0.1 if body_size > 0 else False
        except:
            return False
    
    def _is_high_rsi(self, row: pd.Series) -> bool:
        """判断RSI是否过热"""
        try:
            # 检查多个RSI指标
            rsi_fields = ['rsi6', 'rsi12', 'rsi14']
            high_rsi_count = 0
            
            for field in rsi_fields:
                if field in row and pd.notna(row[field]):
                    if row[field] > 80:  # RSI超过80认为过热
                        high_rsi_count += 1
            
            return high_rsi_count >= 2  # 至少2个RSI指标过热
        except:
            return False
    
    def _is_consecutive_decline(self, row: pd.Series) -> bool:
        """判断是否连续下跌（基于累计涨跌幅特征）"""
        try:
            # 使用已计算的累计涨跌幅特征来判断连续下跌
            # 优先使用5日累计涨跌幅，如果不存在则使用3日

            if 'cumulative_return_5d' in row and pd.notna(row['cumulative_return_5d']):
                # 5日累计跌幅超过8%认为是连续下跌
                return row['cumulative_return_5d'] < -0.08
            elif 'cumulative_return_3d' in row and pd.notna(row['cumulative_return_3d']):
                # 3日累计跌幅超过6%认为是连续下跌
                return row['cumulative_return_3d'] < -0.06
            else:
                # 降级到当日跌幅判断
                return row['pct_chg'] < -3.0
        except:
            return False

    def _is_long_upper_shadow(self, row: pd.Series) -> bool:
        """判断是否为长上影线（使用已计算的特征）"""
        try:
            # 优先使用已计算的长上影线特征
            if 'pattern_long_upper_shadow' in row and pd.notna(row['pattern_long_upper_shadow']):
                # 长上影线特征是负值评分，小于-0.3认为是明显的长上影线
                return row['pattern_long_upper_shadow'] < -0.3
            else:
                # 降级到手动计算
                open_price = row['open']
                close_price = row['close']
                high_price = row['high']

                upper_shadow = high_price - max(open_price, close_price)
                body_size = abs(open_price - close_price)

                # 长上影线：上影线长度是实体的2倍以上
                return upper_shadow / body_size > 2.0 if body_size > 0 else False
        except:
            return False

    def _is_below_all_ma(self, row: pd.Series) -> bool:
        """判断股价是否在所有均线下方（使用已计算的特征）"""
        try:
            # 优先使用已计算的趋势一致性特征
            if 'trend_consistency' in row and pd.notna(row['trend_consistency']):
                # 趋势一致性低于0.2认为是明显的下跌趋势（股价在均线下方）
                return row['trend_consistency'] < 0.2
            else:
                # 降级到手动计算
                close_price = row['close']
                ma_fields = ['ma5', 'ma10', 'ma20']

                # 检查是否在所有均线下方
                for field in ma_fields:
                    if field not in row or pd.isna(row[field]):
                        return False
                    if close_price >= row[field]:
                        return False

                return True
        except:
            return False
