#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强特征处理流水线，统一特征工程过程
"""

import numpy as np
import pandas as pd
import traceback

def _enhanced_feature_pipeline(self, features_df, market_status=None, mode='train'):
    """
    统一的特征工程流水线，确保训练和预测时使用完全一致的特征处理
    
    Args:
        features_df: 输入特征DataFrame
        market_status: 市场状态数据
        mode: 'train'、'validation'或'predict'，指示当前操作模式
        
    Returns:
        处理后的特征DataFrame
    """
    try:
        # 创建副本以避免修改原数据
        result_df = features_df.copy()
        
        self.logger.info(f"应用增强特征流水线 (模式:{mode})，输入特征数: {len(result_df.columns)}")
        
        # 1. 基础特征处理和清理
        # 1.1 处理缺失值和异常值
        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
        # 排除目标变量和ID类列
        exclude_patterns = ['sector_code', 'date', 'label_date']
        exclude_patterns += [col for col in result_df.columns if col.startswith('future_return') and col != 'future_return_2d']
        
        # 有效特征列
        valid_feature_cols = [col for col in numeric_cols if not any(pattern in col for pattern in exclude_patterns)]
        
        # 处理缺失值和异常值
        for col in valid_feature_cols:
            if col in result_df.columns:
                # 替换无穷值
                result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                # 填充缺失值
                if result_df[col].isna().any():
                    if mode == 'train':
                        # 训练模式：用中位数填充
                        median_val = result_df[col].median()
                        result_df[col] = result_df[col].fillna(median_val)
                    else:
                        # 预测模式：用0填充
                        result_df[col] = result_df[col].fillna(0)
                        
                # 处理极端值 - 使用winsorize裁剪至1%和99%分位数
                try:
                    if result_df[col].nunique() > 10:  # 只处理有足够唯一值的列
                        q_low, q_high = result_df[col].quantile([0.01, 0.99])
                        result_df[col] = result_df[col].clip(q_low, q_high)
                except Exception as e:
                    self.logger.warning(f"处理列 {col} 的极端值时出错: {e}")
        
        # 2. 添加核心交互特征 - 统一训练和预测中的特征工程
        # 2.1 添加热度与限价指标交互特征
        if 'heat_index' in result_df.columns and 'limit_up_count' in result_df.columns: 
            result_df['heat_limit_interaction'] = result_df['heat_index'] * result_df['limit_up_count']
            
        if 'leading_stock_count' in result_df.columns and 'high_turnover_ratio' in result_df.columns:
            result_df['leading_turnover_interaction'] = result_df['leading_stock_count'] * result_df['high_turnover_ratio']
            
        if 'multi_limit_ratio' in result_df.columns and 'broken_limit_ratio' in result_df.columns:
            result_df['limit_quality_score'] = result_df['multi_limit_ratio'] - result_df['broken_limit_ratio']
        
        # 2.2 添加市场状态与动量特征交互
        momentum_cols = [col for col in valid_feature_cols if 'momentum' in col or 'relative_' in col]
        market_cols = [col for col in valid_feature_cols if 'market_' in col or 'trend_' in col]
        
        if momentum_cols and market_cols:
            # 为避免特征爆炸，仅选择重要特征
            main_momentum = momentum_cols[:2] if len(momentum_cols) > 2 else momentum_cols
            main_market = market_cols[:2] if len(market_cols) > 2 else market_cols
            
            for mom_col in main_momentum:
                for market_col in main_market:
                    if mom_col in result_df.columns and market_col in result_df.columns:
                        feat_name = f'interact_{mom_col}_{market_col}'
                        result_df[feat_name] = result_df[mom_col] * result_df[market_col]
        
        # 2.3 添加强度与波动率交互特征
        strength_cols = [col for col in valid_feature_cols if 'strength' in col or 'power' in col]
        volatility_cols = [col for col in valid_feature_cols if 'volatility' in col or 'std' in col]
        
        if strength_cols and volatility_cols:
            main_strength = strength_cols[:2] if len(strength_cols) > 2 else strength_cols
            main_volatility = volatility_cols[:2] if len(volatility_cols) > 2 else volatility_cols
            
            for strength_col in main_strength:
                for vol_col in main_volatility:
                    if strength_col in result_df.columns and vol_col in result_df.columns:
                        feat_name = f'interact_{strength_col}_{vol_col}'
                        result_df[feat_name] = result_df[strength_col] * result_df[vol_col]
        
        # 2.4 添加多项式特征
        polynomial_candidates = ['momentum_5d', 'heat_index', 'leading_strength', 'turnover_ratio']
        for col in polynomial_candidates:
            if col in result_df.columns:
                # 创建平方项
                result_df[f'{col}_squared'] = result_df[col] ** 2
                
                # 如果有市场趋势强度特征，创建交互特征
                if 'market_trend_strength' in result_df.columns:
                    result_df[f'{col}_market_trend'] = result_df[col] * result_df['market_trend_strength']
        
        # 3. 确保LNN特征存在
        if mode == 'predict' or mode == 'validation' or 'lnn_output_feature' in result_df.columns:
            result_df = self._ensure_lnn_features(result_df)
        
        # 4. 添加时间特征与季节性特征
        if 'date' in result_df.columns:
            # 确保date是datetime类型
            if not pd.api.types.is_datetime64_dtype(result_df['date']):
                result_df['date'] = pd.to_datetime(result_df['date'])
            
            # 提取月份和星期
            result_df['month'] = result_df['date'].dt.month
            result_df['weekday'] = result_df['date'].dt.weekday
            
            # 添加周期性编码
            result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
            result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
            result_df['weekday_sin'] = np.sin(2 * np.pi * result_df['weekday'] / 7)
            result_df['weekday_cos'] = np.cos(2 * np.pi * result_df['weekday'] / 7)
            
            # 季度特征
            result_df['quarter'] = result_df['date'].dt.quarter
            
            # 是否月初/月末
            result_df['is_month_start'] = result_df['date'].dt.is_month_start.astype(int)
            result_df['is_month_end'] = result_df['date'].dt.is_month_end.astype(int)
        
        # 5. 保存特征工程元数据
        if mode == 'train':
            # 在训练模式下记录特征工程元数据，以便在预测时复现
            feature_metadata = {
                'feature_cols': result_df.columns.tolist(),
                'numeric_cols': numeric_cols,
                'categorical_cols': [col for col in result_df.columns if col not in numeric_cols],
                'interaction_features': [col for col in result_df.columns if 'interact_' in col],
                'polynomial_features': [col for col in result_df.columns if '_squared' in col or '_market_trend' in col]
            }
            self.logger.info(f"训练阶段生成特征元数据: {len(feature_metadata['feature_cols'])} 个特征，"
                           f"{len(feature_metadata['interaction_features'])} 个交互特征")
        
        # 记录新创建的特征
        original_cols = set(features_df.columns)
        new_cols = set(result_df.columns) - original_cols
        self.logger.info(f"特征流水线完成 (模式: {mode})，新增 {len(new_cols)} 个特征，总特征数: {len(result_df.columns)}")
        
        return result_df
        
    except Exception as e:
        self.logger.error(f"特征流水线处理出错: {e}")
        self.logger.error(traceback.format_exc())
        return features_df  # 出错时返回原始特征 