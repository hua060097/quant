#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型管理器模块
负责模型的训练、评估、保存、加载和预测
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
import traceback
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

from to.logger_config import configure_logger
from to.Liquid.nn.liquid.liquid_network import LiquidNetModel
from to.Liquid.nn.liquid.improved_liquid_network import ImprovedLiquidNetModel

# 配置日志
logger = configure_logger(__name__, 'model_manager.log')

class ModelManager:
    """
    模型管理器类
    负责模型的训练、评估、保存、加载和预测
    """
    
    def __init__(self, model_dir=None):
        """
        初始化模型管理器
        
        Args:
            model_dir: 模型保存目录
        """
        # 获取项目根目录
        self.project_root = self._get_project_root()
        
        # 设置模型目录
        self.model_dir = model_dir or os.path.join(self.project_root, "to", "models", "sector_models")
        
        # 确保目录存在
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 日志记录器
        self.logger = logger
        
        # 模型缓存
        self._model_cache = {}
        
        # 模型性能历史记录
        self.performance_history = defaultdict(dict)
    
    def _get_project_root(self):
        """获取项目根目录"""
        current_file = Path(__file__).resolve()
        return current_file.parent.parent
    
    def train_model(self, X, y, model_type='xgboost', model_params=None, feature_names=None):
        """
        训练模型
        
        Args:
            X: 特征数据
            y: 目标数据
            model_type: 模型类型，支持'xgboost'、'lightgbm'、'rf'、'gbdt'、'ridge'、'lasso'、'lnn'
            model_params: 模型参数
            feature_names: 特征名称列表
            
        Returns:
            训练好的模型
        """
        try:
            start_time = time.time()
            self.logger.info(f"开始训练 {model_type} 模型...")
            
            # 设置默认参数
            if model_params is None:
                model_params = self._get_default_params(model_type)
            
            # 创建指定类型的模型
            model = None
            
            if model_type == 'xgboost':
                model = xgb.XGBRegressor(**model_params)
                model.fit(X, y)
                
            elif model_type == 'lightgbm':
                model = lgb.LGBMRegressor(**model_params)
                model.fit(X, y)
                
            elif model_type == 'rf':
                model = RandomForestRegressor(**model_params)
                model.fit(X, y)
                
            elif model_type == 'gbdt':
                model = GradientBoostingRegressor(**model_params)
                model.fit(X, y)
                
            elif model_type == 'ridge':
                model = Ridge(**model_params)
                model.fit(X, y)
                
            elif model_type == 'lasso':
                model = Lasso(**model_params)
                model.fit(X, y)
                
            elif model_type == 'lnn':
                # 液体神经网络模型
                input_size = X.shape[1]
                model = LiquidNetModel(input_size=input_size, output_size=1, **model_params)
                model.fit(X, y, epochs=100, batch_size=32, early_stopping=True)
                
            elif model_type == 'improved_lnn':
                # 改进的液体神经网络模型
                input_size = X.shape[1]
                model = ImprovedLiquidNetModel(input_size=input_size, output_size=1, **model_params)
                model.fit(X, y, epochs=100, batch_size=32, early_stopping=True)
                
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"模型训练完成，耗时 {elapsed_time:.2f} 秒")
            
            return model
            
        except Exception as e:
            self.logger.error(f"训练模型时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _get_default_params(self, model_type):
        """
        获取模型的默认参数
        
        Args:
            model_type: 模型类型
            
        Returns:
            默认参数字典
        """
        if model_type == 'xgboost':
            return {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        elif model_type == 'lightgbm':
            return {
                'n_estimators': 200,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'regression',
                'random_state': 42
            }
        elif model_type == 'rf':
            return {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        elif model_type == 'gbdt':
            return {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'random_state': 42
            }
        elif model_type == 'ridge':
            return {
                'alpha': 1.0
            }
        elif model_type == 'lasso':
            return {
                'alpha': 0.1
            }
        elif model_type == 'lnn':
            return {
                'hidden_size': 64,
                'num_layers': 2,
                'learning_rate': 0.001
            }
        elif model_type == 'improved_lnn':
            return {
                'hidden_size': 64,
                'num_layers': 2,
                'learning_rate': 0.001,
                'dropout_rate': 0.2
            }
        else:
            return {}
    
    def evaluate_model(self, model, X, y, model_type='xgboost'):
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            X: 测试特征数据
            y: 测试目标数据
            model_type: 模型类型
            
        Returns:
            性能指标字典
        """
        try:
            # 预测
            if model_type in ['lnn', 'improved_lnn']:
                y_pred = model.predict(X)
            else:
                y_pred = model.predict(X)
            
            # 计算评估指标
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # 计算预测方向准确率（对于收益率预测）
            direction_accuracy = None
            if 'return' in str(y.name).lower():
                actual_direction = np.sign(y)
                pred_direction = np.sign(y_pred)
                direction_accuracy = np.mean(actual_direction == pred_direction)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            if direction_accuracy is not None:
                metrics['direction_accuracy'] = direction_accuracy
            
            self.logger.info(f"模型评估结果: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            if direction_accuracy is not None:
                self.logger.info(f"方向准确率: {direction_accuracy:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"评估模型时出错: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def save_model(self, model, model_type, sector_code, model_name, feature_columns=None, metadata=None):
        """
        保存模型
        
        Args:
            model: 训练好的模型
            model_type: 模型类型
            sector_code: 板块代码
            model_name: 模型名称
            feature_columns: 特征列名列表
            metadata: 其他元数据
            
        Returns:
            是否成功保存
        """
        try:
            # 构建模型保存路径
            sector_model_dir = os.path.join(self.model_dir, sector_code)
            os.makedirs(sector_model_dir, exist_ok=True)
            
            model_filename = f"{model_name}.pkl"
            model_path = os.path.join(sector_model_dir, model_filename)
            
            metadata_filename = f"{model_name}_metadata.json"
            metadata_path = os.path.join(sector_model_dir, metadata_filename)
            
            # 根据模型类型进行保存
            if model_type in ['lnn', 'improved_lnn']:
                # 液体神经网络模型需要使用专门的保存方法
                model.save(model_path)
            else:
                # 其他模型使用pickle保存
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # 保存元数据
            model_metadata = {
                'model_type': model_type,
                'sector_code': sector_code,
                'model_name': model_name,
                'feature_columns': feature_columns,
                'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            # 合并自定义元数据
            if metadata:
                model_metadata.update(metadata)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(model_metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"模型已保存: {model_path}")
            self.logger.info(f"元数据已保存: {metadata_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存模型时出错: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def load_model(self, sector_code, model_name, use_cache=True):
        """
        加载模型
        
        Args:
            sector_code: 板块代码
            model_name: 模型名称
            use_cache: 是否使用缓存
            
        Returns:
            (model, metadata) 元组
        """
        try:
            # 检查缓存
            cache_key = f"{sector_code}_{model_name}"
            if use_cache and cache_key in self._model_cache:
                return self._model_cache[cache_key]
            
            # 构建模型路径
            sector_model_dir = os.path.join(self.model_dir, sector_code)
            model_path = os.path.join(sector_model_dir, f"{model_name}.pkl")
            metadata_path = os.path.join(sector_model_dir, f"{model_name}_metadata.json")
            
            # 检查文件是否存在
            if not os.path.exists(model_path):
                self.logger.warning(f"模型文件不存在: {model_path}")
                return None, None
                
            if not os.path.exists(metadata_path):
                self.logger.warning(f"元数据文件不存在: {metadata_path}")
                
            # 加载元数据
            metadata = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # 根据模型类型加载模型
            model = None
            model_type = metadata.get('model_type') if metadata else None
            
            if model_type in ['lnn', 'improved_lnn']:
                # 液体神经网络模型
                try:
                    if model_type == 'lnn':
                        model = LiquidNetModel(input_size=1, output_size=1)
                        model.load(model_path)
                    else:
                        model = ImprovedLiquidNetModel(input_size=1, output_size=1)
                        model.load(model_path)
                except Exception as e:
                    self.logger.error(f"加载液体神经网络模型时出错: {e}")
                    return None, metadata
            else:
                # 其他模型使用pickle加载
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                except Exception as e:
                    self.logger.error(f"加载模型时出错: {e}")
                    return None, metadata
            
            # 保存到缓存
            if use_cache:
                self._model_cache[cache_key] = (model, metadata)
            
            self.logger.info(f"成功加载模型: {model_path}")
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"加载模型时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None, None
    
    def predict(self, model, X, model_type='xgboost'):
        """
        使用模型进行预测
        
        Args:
            model: 训练好的模型
            X: 预测特征数据
            model_type: 模型类型
            
        Returns:
            预测结果
        """
        try:
            # 根据模型类型进行预测
            if model_type in ['lnn', 'improved_lnn']:
                return model.predict(X)
            else:
                return model.predict(X)
                
        except Exception as e:
            self.logger.error(f"预测时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def calculate_adaptive_weights(self, model_package, market_status=None):
        """
        计算自适应模型权重
        
        Args:
            model_package: 模型包，包含性能历史记录
            market_status: 市场状态数据
            
        Returns:
            tuple: (strength_weight, lnn_weight) 强度模型和LNN模型的权重
        """
        # 默认权重
        strength_weight, lnn_weight = 0.6, 0.4
        
        # 如果没有性能历史记录，使用默认权重
        performance_history = model_package.get('performance_history', {})
        if not performance_history:
            return strength_weight, lnn_weight
            
        try:
            # 基于历史性能调整权重
            strength_rmse = performance_history.get('strength_model_rmse', 0.05)
            lnn_rmse = performance_history.get('lnn_model_rmse', 0.05)
            
            # 避免除以零
            if strength_rmse <= 0 or lnn_rmse <= 0:
                return strength_weight, lnn_weight
            
            # 根据RMSE的比例调整权重
            total_rmse = strength_rmse + lnn_rmse
            
            # 使用倒数关系，误差越小权重越高
            strength_weight = (1/strength_rmse) / ((1/strength_rmse) + (1/lnn_rmse))
            lnn_weight = 1 - strength_weight
            
            # 如果有市场状态信息，进一步调整权重
            if market_status:
                market_type = market_status.get('market_type', 'neutral')
                volatility = market_status.get('volatility', 0.5)
                
                # 在高波动性市场中增加LNN模型权重
                if volatility > 0.7:
                    adjustment = min(0.1, volatility - 0.7)
                    strength_weight -= adjustment
                    lnn_weight += adjustment
                
                # 在不同市场类型下调整权重
                if market_type == 'bull':
                    # 牛市中增加强度模型权重
                    strength_weight += 0.05
                    lnn_weight -= 0.05
                elif market_type == 'bear':
                    # 熊市中增加LNN模型权重
                    strength_weight -= 0.05
                    lnn_weight += 0.05
            
            # 确保权重在0-1之间
            strength_weight = max(0.2, min(0.8, strength_weight))
            lnn_weight = 1 - strength_weight
            
            # 记录调整后的权重
            self.logger.info(f"自适应权重: 强度模型={strength_weight:.2f}, LNN模型={lnn_weight:.2f}")
            
            return strength_weight, lnn_weight
            
        except Exception as e:
            self.logger.error(f"计算自适应权重时出错: {e}")
            self.logger.error(traceback.format_exc())
            return 0.6, 0.4  # 出错时返回默认权重
    
    def update_performance_history(self, sector_code, model_name, metrics, model_type='strength'):
        """
        更新模型性能历史记录
        
        Args:
            sector_code: 板块代码
            model_name: 模型名称
            metrics: 性能指标字典
            model_type: 模型类型（strength、lnn等）
            
        Returns:
            更新后的性能历史记录
        """
        try:
            # 获取当前记录
            key = f"{sector_code}_{model_name}"
            history = self.performance_history.get(key, {})
            
            # 更新性能指标
            history[f'{model_type}_model_rmse'] = metrics.get('rmse', 0)
            history[f'{model_type}_model_mae'] = metrics.get('mae', 0)
            history[f'{model_type}_model_r2'] = metrics.get('r2', 0)
            
            if 'direction_accuracy' in metrics:
                history[f'{model_type}_model_dir_acc'] = metrics['direction_accuracy']
            
            # 记录时间戳
            history['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 保存更新后的记录
            self.performance_history[key] = history
            
            self.logger.info(f"已更新 {sector_code} 的 {model_name} ({model_type}) 性能历史记录")
            
            return history
            
        except Exception as e:
            self.logger.error(f"更新性能历史记录时出错: {e}")
            self.logger.error(traceback.format_exc())
            return {}
    
    def get_performance_history(self, sector_code, model_name):
        """
        获取模型性能历史记录
        
        Args:
            sector_code: 板块代码
            model_name: 模型名称
            
        Returns:
            性能历史记录字典
        """
        key = f"{sector_code}_{model_name}"
        return self.performance_history.get(key, {})
    
    def clear_model_cache(self):
        """清除模型缓存"""
        self._model_cache.clear()
        self.logger.info("已清除模型缓存") 