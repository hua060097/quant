#!/usr/bin/env python3
"""
特征工程模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from config import system_config

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def engineer_technical_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """工程化技术面特征（简化版）"""
        try:
            data = stock_data.copy()

            # 核心价格动量特征
            if 'close' in data.columns:
                data['price_momentum_5d'] = data.groupby('stock_code')['close'].pct_change(5)

            # 核心均线特征
            if all(col in data.columns for col in ['close', 'ma5', 'ma20']):
                data['price_vs_ma5'] = (data['close'] - data['ma5']) / data['ma5']
                data['ma_bullish'] = (data['ma5'] > data['ma20']).astype(int)

            # 核心RSI特征
            if 'rsi14' in data.columns:
                data['rsi_neutral'] = ((data['rsi14'] >= 30) & (data['rsi14'] <= 70)).astype(int)

            # 核心MACD特征
            if all(col in data.columns for col in ['macd', 'macd_signal']):
                data['macd_bullish'] = (data['macd'] > data['macd_signal']).astype(int)

            # 核心趋势特征
            if 'trend_strength' in data.columns:
                data['strong_trend'] = (data['trend_strength'] > 0.7).astype(int)

            self.logger.info("技术面特征工程完成（简化版）")
            return data

        except Exception as e:
            self.logger.error(f"技术面特征工程失败: {e}")
            return stock_data
    
    def engineer_fundamental_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """工程化基本面特征（简化版）"""
        try:
            data = stock_data.copy()

            # 核心市值特征
            if 'total_mv' in data.columns:
                data['market_cap_rank'] = data.groupby('date')['total_mv'].rank(pct=True)
                data['mid_cap'] = ((data['market_cap_rank'] >= 0.2) &
                                 (data['market_cap_rank'] <= 0.8)).astype(int)

            # 核心换手率特征
            if 'turnover_rate' in data.columns:
                data['turnover_rank'] = data.groupby('date')['turnover_rate'].rank(pct=True)
                data['high_turnover'] = (data['turnover_rank'] > 0.7).astype(int)

            self.logger.info("基本面特征工程完成（简化版）")
            return data

        except Exception as e:
            self.logger.error(f"基本面特征工程失败: {e}")
            return stock_data
    
    def engineer_capital_flow_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """工程化资金流特征（简化版）"""
        try:
            data = stock_data.copy()

            # 核心成交量特征
            if 'volume' in data.columns:
                data['volume_rank'] = data.groupby('date')['volume'].rank(pct=True)
                data['high_volume'] = (data['volume_rank'] > 0.7).astype(int)

            # 核心量价配合特征
            if all(col in data.columns for col in ['pct_chg', 'volume_rank']):
                data['volume_price_match'] = ((data['pct_chg'] > 0) &
                                            (data['volume_rank'] > 0.6)).astype(int)

            self.logger.info("资金流特征工程完成（简化版）")
            return data

        except Exception as e:
            self.logger.error(f"资金流特征工程失败: {e}")
            return stock_data
    
    def engineer_limit_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """工程化涨跌停特征（简化版）"""
        try:
            data = stock_data.copy()

            # 核心涨停质量特征
            if 'limit_quality_score' in data.columns:
                data['high_quality_limit'] = (data['limit_quality_score'] > 0.7).astype(int)

            # 核心涨停强度特征
            if 'limit_strength' in data.columns:
                data['strong_limit'] = (data['limit_strength'] > 0.7).astype(int)

            self.logger.info("涨跌停特征工程完成（简化版）")
            return data

        except Exception as e:
            self.logger.error(f"涨跌停特征工程失败: {e}")
            return stock_data
    
    def engineer_composite_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """工程化复合特征（简化版）"""
        try:
            data = stock_data.copy()

            # 核心技术评分
            technical_features = ['ma_bullish', 'macd_bullish', 'strong_trend']
            available_technical = [f for f in technical_features if f in data.columns]

            if available_technical:
                data['composite_technical'] = data[available_technical].mean(axis=1)

            self.logger.info("复合特征工程完成（简化版）")
            return data

        except Exception as e:
            self.logger.error(f"复合特征工程失败: {e}")
            return stock_data
    
    def engineer_all_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """执行所有特征工程"""
        try:
            self.logger.info("开始特征工程")
            
            # 按顺序执行各类特征工程
            data = self.engineer_technical_features(stock_data)
            data = self.engineer_fundamental_features(data)
            data = self.engineer_capital_flow_features(data)
            data = self.engineer_limit_features(data)
            data = self.engineer_composite_features(data)
            
            # 填充缺失值
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(0)
            
            self.logger.info(f"特征工程完成，新增特征数: {len(data.columns) - len(stock_data.columns)}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"特征工程失败: {e}")
            return stock_data
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """获取特征重要性权重"""
        return {
            # 技术面特征权重
            'price_momentum_5d': 0.8,
            'price_momentum_10d': 0.6,
            'price_vs_ma5': 0.7,
            'ma_bullish': 0.9,
            'rsi_neutral': 0.5,
            'macd_bullish': 0.8,
            'strong_trend': 0.9,
            'composite_momentum': 1.0,
            'composite_technical': 1.0,
            
            # 基本面特征权重
            'mid_cap': 0.6,
            'high_float': 0.4,
            'high_turnover': 0.7,
            
            # 资金流特征权重
            'high_volume': 0.8,
            'volume_surge': 0.9,
            'volume_price_match': 1.0,
            
            # 涨跌停特征权重
            'recent_limit_up': 0.7,
            'high_quality_limit': 0.9,
            'strong_limit': 0.8,
            
            # 板块特征权重
            'overlap_score': 1.0,
            'weighted_sector_score': 1.0,
            
            # 复合特征权重
            'composite_quality': 1.0
        }
