#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增量训练模块
实现模型的增量训练功能，减少重复训练时间，提升回测效率
"""

import os
import sys
import time
import logging
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

# 配置日志
os.makedirs('to/log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/incremental_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("incremental_training")

class IncrementalTrainingManager:
    """增量训练管理器，负责协调各类模型的增量训练过程"""
    
    def __init__(self, model_dir: str = 'to/models/incremental', 
                 version_control: bool = True,
                 max_versions: int = 5,
                 performance_monitoring: bool = True):
        """
        初始化增量训练管理器
        
        参数:
            model_dir: 模型存储目录
            version_control: 是否启用版本控制
            max_versions: 最大保存版本数
            performance_monitoring: 是否启用性能监控
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.version_control = version_control
        self.max_versions = max_versions
        self.performance_monitoring = performance_monitoring
        
        # 性能统计
        self.performance_stats = {
            'full_training_time': 0,
            'incremental_training_time': 0,
            'training_count': 0,
            'incremental_count': 0,
            'model_versions': {}
        }
        
        # 当前活跃模型
        self.active_models = {}
        
        # 各模型类型的训练器
        self.trainers = {
            'xgboost': self._train_xgboost_incremental,
            'lightgbm': self._train_lightgbm_incremental,
            'catboost': self._train_catboost_incremental,
            'lnn': self._train_lnn_incremental,
            'ensemble': self._train_ensemble_incremental
        }
        
        logger.info(f"增量训练管理器初始化完成，模型存储目录: {self.model_dir}")
    
    def train(self, 
              model_type: str, 
              model_id: str, 
              X_new: np.ndarray, 
              y_new: np.ndarray, 
              **kwargs) -> Any:
        """
        训练或增量更新模型
        
        参数:
            model_type: 模型类型 ('xgboost', 'lightgbm', 'catboost', 'lnn', 'ensemble')
            model_id: 模型唯一标识符，如 'sector_finance_short'
            X_new: 新训练数据特征
            y_new: 新训练数据标签
            **kwargs: 额外参数，传递给具体的训练函数
            
        返回:
            训练后的模型
        """
        if model_type not in self.trainers:
            logger.error(f"不支持的模型类型: {model_type}")
            return None
        
        # 记录开始时间
        start_time = time.time()
        
        # 检查是否存在之前的模型
        existing_model = self._load_model(model_type, model_id)
        is_incremental = existing_model is not None
        
        # 确保训练数据格式正确
        X_new, y_new = self._validate_data(X_new, y_new)
        if X_new is None or y_new is None:
            return None
        
        try:
            # 执行训练
            if is_incremental:
                logger.info(f"对模型 {model_id} 进行增量训练，新样本数量: {len(X_new)}")
                model = self.trainers[model_type](existing_model, X_new, y_new, **kwargs)
                self.performance_stats['incremental_count'] += 1
            else:
                logger.info(f"模型 {model_id} 不存在，执行全量训练，样本数量: {len(X_new)}")
                model = self._train_full(model_type, X_new, y_new, **kwargs)
            
            # 记录训练时间
            training_time = time.time() - start_time
            if is_incremental:
                self.performance_stats['incremental_training_time'] += training_time
                logger.info(f"增量训练完成，耗时: {training_time:.2f}秒")
            else:
                self.performance_stats['full_training_time'] += training_time
                logger.info(f"全量训练完成，耗时: {training_time:.2f}秒")
            
            self.performance_stats['training_count'] += 1
            
            # 保存模型
            if model is not None:
                self._save_model(model, model_type, model_id)
                # 更新活跃模型字典
                self.active_models[f"{model_type}_{model_id}"] = model
            
            return model
            
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _train_full(self, model_type: str, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """执行全量训练"""
        logger.info(f"执行{model_type}模型全量训练")
        
        if model_type == 'xgboost':
            return self._train_xgboost_full(X, y, **kwargs)
        elif model_type == 'lightgbm':
            return self._train_lightgbm_full(X, y, **kwargs)
        elif model_type == 'catboost':
            return self._train_catboost_full(X, y, **kwargs)
        elif model_type == 'lnn':
            return self._train_lnn_full(X, y, **kwargs)
        elif model_type == 'ensemble':
            return self._train_ensemble_full(X, y, **kwargs)
        else:
            logger.error(f"不支持的模型类型: {model_type}")
            return None
    
    def _train_xgboost_full(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """XGBoost全量训练"""
        try:
            import xgboost as xgb
            
            # 设置默认参数
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'use_label_encoder': False,
                'verbosity': 0,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6
            }
            
            # 更新自定义参数
            params.update(kwargs.get('params', {}))
            
            # 创建DMatrix对象（可选）
            if kwargs.get('use_dmatrix', False):
                dtrain = xgb.DMatrix(X, label=y)
                model = xgb.train(params, dtrain, num_boost_round=params.get('n_estimators', 100))
            else:
                # 创建分类器
                model = xgb.XGBClassifier(**params)
                
                # 训练
                sample_weight = kwargs.get('sample_weight', None)
                eval_set = kwargs.get('eval_set', None)
                
                model.fit(
                    X, y,
                    sample_weight=sample_weight,
                    eval_set=eval_set,
                    verbose=kwargs.get('verbose', False)
                )
            
            return model
            
        except ImportError:
            logger.error("未安装XGBoost库")
            return None
        except Exception as e:
            logger.error(f"XGBoost训练出错: {e}")
            return None
    
    def _train_xgboost_incremental(self, model, X_new: np.ndarray, y_new: np.ndarray, **kwargs) -> Any:
        """XGBoost增量训练"""
        try:
            import xgboost as xgb
            
            if isinstance(model, xgb.XGBClassifier):
                # XGBClassifier增量训练
                sample_weight = kwargs.get('sample_weight', None)
                eval_set = kwargs.get('eval_set', None)
                
                model.fit(
                    X_new, y_new,
                    sample_weight=sample_weight,
                    eval_set=eval_set,
                    verbose=kwargs.get('verbose', False),
                    xgb_model=model  # 关键参数：使用现有模型作为起点
                )
            
            elif isinstance(model, xgb.Booster):
                # Booster对象增量训练
                dtrain = xgb.DMatrix(X_new, label=y_new)
                model = xgb.train(
                    kwargs.get('params', {}),
                    dtrain,
                    num_boost_round=kwargs.get('num_boost_round', 10),
                    xgb_model=model  # 使用现有模型作为起点
                )
            
            else:
                logger.error(f"不支持的XGBoost模型类型: {type(model)}")
                return None
            
            return model
            
        except Exception as e:
            logger.error(f"XGBoost增量训练出错: {e}")
            return None
    
    def _train_lightgbm_full(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """LightGBM全量训练"""
        try:
            import lightgbm as lgb
            
            # 设置默认参数
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31
            }
            
            # 更新自定义参数
            params.update(kwargs.get('params', {}))
            
            # 创建数据集（可选）
            if kwargs.get('use_lgb_dataset', False):
                train_data = lgb.Dataset(X, label=y)
                model = lgb.train(
                    params, 
                    train_data, 
                    num_boost_round=params.get('n_estimators', 100),
                    valid_sets=kwargs.get('valid_sets', None)
                )
            else:
                # 创建分类器
                model = lgb.LGBMClassifier(**params)
                
                # 训练
                sample_weight = kwargs.get('sample_weight', None)
                eval_set = kwargs.get('eval_set', None)
                
                model.fit(
                    X, y,
                    sample_weight=sample_weight,
                    eval_set=eval_set,
                    verbose=kwargs.get('verbose', False)
                )
            
            return model
            
        except ImportError:
            logger.error("未安装LightGBM库")
            return None
        except Exception as e:
            logger.error(f"LightGBM训练出错: {e}")
            return None
    
    def _train_lightgbm_incremental(self, model, X_new: np.ndarray, y_new: np.ndarray, **kwargs) -> Any:
        """LightGBM增量训练"""
        try:
            import lightgbm as lgb
            
            if isinstance(model, lgb.LGBMClassifier):
                # LGBMClassifier增量训练
                model.fit(
                    X_new, y_new,
                    sample_weight=kwargs.get('sample_weight', None),
                    eval_set=kwargs.get('eval_set', None),
                    verbose=kwargs.get('verbose', False),
                    init_model=model  # 关键参数：使用现有模型作为起点
                )
            
            elif isinstance(model, lgb.Booster):
                # Booster对象增量训练
                train_data = lgb.Dataset(X_new, label=y_new)
                model = lgb.train(
                    kwargs.get('params', {}),
                    train_data,
                    num_boost_round=kwargs.get('num_boost_round', 10),
                    init_model=model  # 使用现有模型作为起点
                )
            
            else:
                logger.error(f"不支持的LightGBM模型类型: {type(model)}")
                return None
            
            return model
            
        except Exception as e:
            logger.error(f"LightGBM增量训练出错: {e}")
            return None
    
    def _train_catboost_full(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """CatBoost全量训练"""
        try:
            from catboost import CatBoostClassifier, Pool
            
            # 设置默认参数
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'verbose': False,
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6
            }
            
            # 更新自定义参数
            params.update(kwargs.get('params', {}))
            
            # 创建分类器
            model = CatBoostClassifier(**params)
            
            # 获取分类特征索引
            cat_features = kwargs.get('cat_features', None)
            
            # 训练
            if kwargs.get('use_pool', False) or cat_features is not None:
                train_pool = Pool(X, y, cat_features=cat_features)
                eval_set = kwargs.get('eval_set', None)
                
                if eval_set is not None:
                    eval_pool = Pool(*eval_set[0], cat_features=cat_features)
                    model.fit(train_pool, eval_set=eval_pool, verbose=params.get('verbose', False))
                else:
                    model.fit(train_pool, verbose=params.get('verbose', False))
            else:
                model.fit(
                    X, y,
                    eval_set=kwargs.get('eval_set', None),
                    verbose=params.get('verbose', False)
                )
            
            return model
            
        except ImportError:
            logger.error("未安装CatBoost库")
            return None
        except Exception as e:
            logger.error(f"CatBoost训练出错: {e}")
            return None
    
    def _train_catboost_incremental(self, model, X_new: np.ndarray, y_new: np.ndarray, **kwargs) -> Any:
        """CatBoost增量训练"""
        try:
            from catboost import CatBoostClassifier, Pool
            
            if not isinstance(model, CatBoostClassifier):
                logger.error(f"不支持的CatBoost模型类型: {type(model)}")
                return None
            
            # 获取分类特征索引
            cat_features = kwargs.get('cat_features', None)
            
            # 增量训练
            if kwargs.get('use_pool', False) or cat_features is not None:
                train_pool = Pool(X_new, y_new, cat_features=cat_features)
                eval_set = kwargs.get('eval_set', None)
                
                if eval_set is not None:
                    eval_pool = Pool(*eval_set[0], cat_features=cat_features)
                    model.fit(train_pool, eval_set=eval_pool, init_model=model, verbose=kwargs.get('verbose', False))
                else:
                    model.fit(train_pool, init_model=model, verbose=kwargs.get('verbose', False))
            else:
                model.fit(
                    X_new, y_new,
                    eval_set=kwargs.get('eval_set', None),
                    init_model=model,  # 关键参数：使用现有模型作为起点
                    verbose=kwargs.get('verbose', False)
                )
            
            return model
            
        except Exception as e:
            logger.error(f"CatBoost增量训练出错: {e}")
            return None
    
    def _train_lnn_full(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """LNN全量训练"""
        try:
            # 检查是否提供了自定义的LNN创建函数
            create_lnn_func = kwargs.get('create_lnn_func', None)
            if create_lnn_func is None:
                logger.error("未提供LNN模型创建函数")
                return None
            
            # 创建LNN模型
            input_size = X.shape[1]
            hidden_size = kwargs.get('hidden_size', 64)
            output_size = kwargs.get('output_size', 2)  # 二分类默认为2
            num_layers = kwargs.get('num_layers', 2)
            
            # 调用创建函数
            model = create_lnn_func(input_size, hidden_size, output_size, num_layers)
            
            # 训练参数
            epochs = kwargs.get('epochs', 50)
            batch_size = kwargs.get('batch_size', 32)
            lr = kwargs.get('learning_rate', 0.001)
            device = kwargs.get('device', 'cpu')
            
            # 训练函数
            train_func = kwargs.get('train_func', None)
            if train_func is None:
                logger.error("未提供LNN模型训练函数")
                return None
            
            # 执行训练
            model = train_func(model, X, y, epochs, batch_size, lr, device)
            
            return model
            
        except ImportError:
            logger.error("未安装必要的深度学习库")
            return None
        except Exception as e:
            logger.error(f"LNN训练出错: {e}")
            return None
    
    def _train_lnn_incremental(self, model, X_new: np.ndarray, y_new: np.ndarray, **kwargs) -> Any:
        """LNN增量训练"""
        try:
            # 训练参数
            epochs = kwargs.get('epochs', 10)  # 增量训练的轮数通常少于全量训练
            batch_size = kwargs.get('batch_size', 32)
            lr = kwargs.get('learning_rate', 0.0005)  # 增量训练通常使用较小的学习率
            device = kwargs.get('device', 'cpu')
            
            # 训练函数
            train_func = kwargs.get('train_func', None)
            if train_func is None:
                logger.error("未提供LNN模型训练函数")
                return None
            
            # 使用更低的学习率执行增量训练
            model = train_func(model, X_new, y_new, epochs, batch_size, lr, device)
            
            return model
            
        except Exception as e:
            logger.error(f"LNN增量训练出错: {e}")
            return None
    
    def _train_ensemble_full(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """集成模型全量训练"""
        try:
            # 获取子模型类型
            sub_models = kwargs.get('sub_models', ['xgboost', 'lightgbm', 'catboost'])
            
            # 创建子模型字典
            models = {}
            
            # 训练每个子模型
            for model_type in sub_models:
                logger.info(f"训练子模型: {model_type}")
                
                if model_type == 'xgboost':
                    models[model_type] = self._train_xgboost_full(X, y, **kwargs.get('xgboost_kwargs', {}))
                elif model_type == 'lightgbm':
                    models[model_type] = self._train_lightgbm_full(X, y, **kwargs.get('lightgbm_kwargs', {}))
                elif model_type == 'catboost':
                    models[model_type] = self._train_catboost_full(X, y, **kwargs.get('catboost_kwargs', {}))
                elif model_type == 'lnn':
                    models[model_type] = self._train_lnn_full(X, y, **kwargs.get('lnn_kwargs', {}))
            
            # 获取模型权重
            weights = kwargs.get('weights', {})
            
            # 创建集成模型字典
            ensemble = {
                'models': models,
                'weights': weights
            }
            
            return ensemble
            
        except Exception as e:
            logger.error(f"集成模型训练出错: {e}")
            return None
    
    def _train_ensemble_incremental(self, ensemble, X_new: np.ndarray, y_new: np.ndarray, **kwargs) -> Any:
        """集成模型增量训练"""
        try:
            if not isinstance(ensemble, dict) or 'models' not in ensemble:
                logger.error("无效的集成模型格式")
                return None
            
            # 获取现有子模型
            models = ensemble['models']
            
            # 更新每个子模型
            for model_type, model in models.items():
                logger.info(f"增量更新子模型: {model_type}")
                
                if model_type == 'xgboost':
                    models[model_type] = self._train_xgboost_incremental(
                        model, X_new, y_new, **kwargs.get('xgboost_kwargs', {})
                    )
                elif model_type == 'lightgbm':
                    models[model_type] = self._train_lightgbm_incremental(
                        model, X_new, y_new, **kwargs.get('lightgbm_kwargs', {})
                    )
                elif model_type == 'catboost':
                    models[model_type] = self._train_catboost_incremental(
                        model, X_new, y_new, **kwargs.get('catboost_kwargs', {})
                    )
                elif model_type == 'lnn':
                    models[model_type] = self._train_lnn_incremental(
                        model, X_new, y_new, **kwargs.get('lnn_kwargs', {})
                    )
            
            # 更新集成模型
            ensemble['models'] = models
            
            # 如果提供了新权重，更新权重
            if 'weights' in kwargs:
                ensemble['weights'] = kwargs['weights']
            
            return ensemble
            
        except Exception as e:
            logger.error(f"集成模型增量训练出错: {e}")
            return None
    
    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """验证并预处理数据"""
        try:
            # 确保X是二维数组
            if X is None or y is None:
                logger.error("输入数据为空")
                return None, None
                
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            
            # 确保y是一维数组
            if len(y.shape) > 1 and y.shape[1] > 1:
                logger.warning("标签y不是一维数组，尝试转换")
                y = np.argmax(y, axis=1)
            
            # 检查样本数量是否匹配
            if X.shape[0] != y.shape[0]:
                logger.error(f"特征和标签样本数量不匹配: X={X.shape[0]}, y={y.shape[0]}")
                return None, None
            
            # 检查是否有无效值
            if np.isnan(X).any() or np.isinf(X).any():
                logger.warning("特征中存在NaN或Inf值，尝试替换")
                X = np.nan_to_num(X)
            
            return X, y
            
        except Exception as e:
            logger.error(f"数据验证过程中出错: {e}")
            return None, None
    
    def _load_model(self, model_type: str, model_id: str) -> Any:
        """加载现有模型"""
        # 检查活跃模型缓存
        cache_key = f"{model_type}_{model_id}"
        if cache_key in self.active_models:
            return self.active_models[cache_key]
        
        # 构建模型文件路径
        model_path = self.model_dir / f"{model_type}_{model_id}.pkl"
        
        if not model_path.exists():
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"从 {model_path} 加载现有模型成功")
            return model
            
        except Exception as e:
            logger.error(f"加载模型 {model_path} 失败: {e}")
            return None
    
    def _save_model(self, model: Any, model_type: str, model_id: str) -> bool:
        """保存模型"""
        try:
            base_path = self.model_dir / f"{model_type}_{model_id}"
            
            # 版本控制
            if self.version_control:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = base_path.with_suffix(f".{timestamp}.pkl")
                
                # 记录版本信息
                if model_id not in self.performance_stats['model_versions']:
                    self.performance_stats['model_versions'][model_id] = []
                
                self.performance_stats['model_versions'][model_id].append(str(model_path))
                
                # 限制版本数量
                if len(self.performance_stats['model_versions'][model_id]) > self.max_versions:
                    # 删除最旧的版本文件
                    old_version = self.performance_stats['model_versions'][model_id].pop(0)
                    old_path = Path(old_version)
                    if old_path.exists():
                        old_path.unlink()
                        logger.info(f"删除旧版本模型: {old_path}")
            else:
                model_path = base_path.with_suffix(".pkl")
            
            # 保存模型
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 如果不使用版本控制，同时创建一个不带版本号的标准路径
            if self.version_control:
                standard_path = base_path.with_suffix(".pkl")
                with open(standard_path, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info(f"模型已保存至 {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False
    
    def get_active_models(self) -> Dict[str, Any]:
        """获取所有活跃模型"""
        return self.active_models
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计数据"""
        # 计算时间节省比例
        total_incremental_time = self.performance_stats['incremental_training_time']
        incremental_count = self.performance_stats['incremental_count']
        
        # 估算全量训练需要的时间
        avg_full_time = (self.performance_stats['full_training_time'] / 
                         (self.performance_stats['training_count'] - incremental_count)
                         if self.performance_stats['training_count'] > incremental_count else 0)
        
        estimated_full_time = avg_full_time * incremental_count
        
        # 计算节省的时间
        time_saved = estimated_full_time - total_incremental_time if estimated_full_time > 0 else 0
        time_saved_pct = (time_saved / estimated_full_time * 100 
                          if estimated_full_time > 0 else 0)
        
        # 添加计算结果到统计数据
        stats = dict(self.performance_stats)
        stats.update({
            'avg_full_training_time': avg_full_time,
            'avg_incremental_training_time': (total_incremental_time / incremental_count 
                                             if incremental_count > 0 else 0),
            'estimated_full_time': estimated_full_time,
            'time_saved': time_saved,
            'time_saved_pct': time_saved_pct
        })
        
        return stats
    
    def print_performance_report(self) -> None:
        """打印性能报告"""
        stats = self.get_performance_stats()
        
        logger.info("\n===== 增量训练性能报告 =====")
        logger.info(f"总训练次数: {stats['training_count']}")
        logger.info(f"全量训练次数: {stats['training_count'] - stats['incremental_count']}")
        logger.info(f"增量训练次数: {stats['incremental_count']}")
        
        if stats['incremental_count'] > 0:
            logger.info(f"\n全量训练总时间: {stats['full_training_time']:.2f}秒")
            logger.info(f"增量训练总时间: {stats['incremental_training_time']:.2f}秒")
            logger.info(f"平均全量训练时间: {stats['avg_full_training_time']:.2f}秒/次")
            logger.info(f"平均增量训练时间: {stats['avg_incremental_training_time']:.2f}秒/次")
            
            if stats['avg_full_training_time'] > 0:
                speedup = stats['avg_full_training_time'] / stats['avg_incremental_training_time']
                logger.info(f"平均加速比: {speedup:.2f}倍")
            
            logger.info(f"估算节省时间: {stats['time_saved']:.2f}秒 ({stats['time_saved_pct']:.2f}%)")
        
        # 模型版本信息
        if self.version_control and stats['model_versions']:
            logger.info("\n模型版本信息:")
            for model_id, versions in stats['model_versions'].items():
                logger.info(f"  {model_id}: {len(versions)}个版本")
        
        logger.info("=============================")

# 全局增量训练管理器
_GLOBAL_INCREMENTAL_MANAGER = None

def get_incremental_manager(model_dir=None, version_control=True, max_versions=5):
    """获取全局增量训练管理器实例"""
    global _GLOBAL_INCREMENTAL_MANAGER
    if _GLOBAL_INCREMENTAL_MANAGER is None:
        if model_dir is None:
            model_dir = 'to/models/incremental'
        _GLOBAL_INCREMENTAL_MANAGER = IncrementalTrainingManager(
            model_dir=model_dir,
            version_control=version_control,
            max_versions=max_versions
        )
    return _GLOBAL_INCREMENTAL_MANAGER

# 测试代码
if __name__ == "__main__":
    # 测试增量训练功能
    logger.info("测试增量训练功能")
    
    # 创建模拟数据
    np.random.seed(42)
    X1 = np.random.rand(100, 10)
    y1 = np.random.randint(0, 2, 100)
    
    X2 = np.random.rand(50, 10)
    y2 = np.random.randint(0, 2, 50)
    
    # 获取增量训练管理器
    manager = get_incremental_manager()
    
    # 训练XGBoost模型
    try:
        import xgboost as xgb
        
        # 第一次训练（全量）
        model = manager.train('xgboost', 'test_model', X1, y1)
        
        if model:
            logger.info("XGBoost模型首次训练成功")
            
            # 第二次训练（增量）
            model = manager.train('xgboost', 'test_model', X2, y2)
            
            if model:
                logger.info("XGBoost模型增量训练成功")
                manager.print_performance_report()
    except ImportError:
        logger.warning("XGBoost未安装，跳过测试")
    
    logger.info("增量训练模块加载成功") 