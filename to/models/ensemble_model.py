#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块轮动模型融合框架

结合多种机器学习模型的优势，特别优化用于捕捉股票市场板块轮动特征
实现灵活的模型融合策略，提高预测稳定性和准确性
"""

import numpy as np
import pandas as pd
import logging
import os
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

# 导入自定义LNN模型
try:
    from to.models.liquid_neural_network import LNNClassifier
except ImportError:
    # 如果在不同路径下导入
    try:
        from models.liquid_neural_network import LNNClassifier
    except ImportError:
        logging.warning("无法导入LNNClassifier，模型融合将不包含LNN")
        LNNClassifier = None

logger = logging.getLogger(__name__)

class SectorRotationEnsemble(BaseEstimator, ClassifierMixin):
    """
    板块轮动模型融合器
    
    融合多种模型的预测结果，专门用于板块轮动预测
    支持XGBoost、LightGBM、CatBoost和LNN模型
    
    特点:
    1. 自动处理不同时间尺度的特征
    2. 支持样本权重和时间衰减权重
    3. 针对金融市场类别不平衡问题优化
    4. 提供灵活的模型配置和权重调整选项
    """
    
    def __init__(self, use_xgb=True, use_lgbm=True, use_catboost=True, use_lnn=True,
                 model_weights=None, val_size=0.2, random_state=42,
                 xgb_params=None, lgbm_params=None, catboost_params=None, lnn_params=None,
                 n_jobs=-1, verbose=1):
        """
        初始化模型融合器
        
        Args:
            use_xgb: 是否使用XGBoost模型
            use_lgbm: 是否使用LightGBM模型
            use_catboost: 是否使用CatBoost模型
            use_lnn: 是否使用LNN (液体神经网络) 模型
            model_weights: 各模型预测权重, 如None则使用默认权重
            val_size: 验证集比例
            random_state: 随机种子
            xgb_params: XGBoost模型参数
            lgbm_params: LightGBM模型参数 
            catboost_params: CatBoost模型参数
            lnn_params: LNN模型参数
            n_jobs: 并行任务数
            verbose: 详细程度
        """
        self.use_xgb = use_xgb
        self.use_lgbm = use_lgbm
        self.use_catboost = use_catboost
        self.use_lnn = use_lnn and LNNClassifier is not None
        
        # 设置默认模型权重
        if model_weights is None:
            self.model_weights = {
                'xgb': 0.3,
                'lgbm': 0.3,
                'catboost': 0.2,
                'lnn': 0.2
            }
        else:
            self.model_weights = model_weights
            
        self.val_size = val_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # 设置各模型默认参数
        self.xgb_params = xgb_params or {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'scale_pos_weight': 1,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        
        self.lgbm_params = lgbm_params or {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary',
            'class_weight': 'balanced',
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        
        self.catboost_params = catboost_params or {
            'iterations': 200,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'Logloss',
            'random_seed': random_state,
            'thread_count': n_jobs if n_jobs > 0 else None,
            'verbose': verbose > 0
        }
        
        self.lnn_params = lnn_params or {
            'hidden_sizes': [64, 32],
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'dropout': 0.3,
            'tau_init': 1.0,
            'early_stopping': True,
            'patience': 10,
            'class_weight': 'balanced'
        }
        
        # 初始化模型和缩放器
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance_ = None
        self.is_fitted = False
    
    def fit(self, X, y, sample_weight=None):
        """
        训练模型融合器
        
        Args:
            X: 特征数据
            y: 目标变量
            sample_weight: 样本权重
            
        Returns:
            self
        """
        logger.info("开始训练板块轮动模型融合器")
        
        # 数据预处理
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # 划分训练集和验证集
        if self.val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, 
                test_size=self.val_size, 
                random_state=self.random_state,
                stratify=y
            )
            
            if sample_weight is not None:
                _, sample_weight_val = train_test_split(
                    sample_weight, 
                    test_size=self.val_size, 
                    random_state=self.random_state,
                    stratify=y
                )
            else:
                sample_weight_val = None
        else:
            X_train, y_train = X_scaled, y
            X_val, y_val = None, None
            sample_weight_val = None
        
        # 初始化特征重要性
        self.feature_importance_ = np.zeros(X.shape[1])
        
        # 训练XGBoost模型
        if self.use_xgb:
            logger.info("训练XGBoost模型")
            self.models['xgb'] = XGBClassifier(**self.xgb_params)
            self.models['xgb'].fit(
                X_train, y_train, 
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                early_stopping_rounds=20 if X_val is not None else None,
                verbose=self.verbose
            )
            
            # 更新特征重要性
            if hasattr(self.models['xgb'], 'feature_importances_'):
                self.feature_importance_ += (
                    self.models['xgb'].feature_importances_ * self.model_weights['xgb']
                )
        
        # 训练LightGBM模型
        if self.use_lgbm:
            logger.info("训练LightGBM模型")
            self.models['lgbm'] = LGBMClassifier(**self.lgbm_params)
            self.models['lgbm'].fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                early_stopping_rounds=20 if X_val is not None else None,
                verbose=self.verbose
            )
            
            # 更新特征重要性
            if hasattr(self.models['lgbm'], 'feature_importances_'):
                self.feature_importance_ += (
                    self.models['lgbm'].feature_importances_ * self.model_weights['lgbm']
                )
        
        # 训练CatBoost模型
        if self.use_catboost:
            logger.info("训练CatBoost模型")
            self.models['catboost'] = CatBoostClassifier(**self.catboost_params)
            
            if X_val is not None:
                eval_set = Pool(X_val, y_val, weight=sample_weight_val)
            else:
                eval_set = None
                
            self.models['catboost'].fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=eval_set,
                early_stopping_rounds=20 if eval_set is not None else None,
                verbose=self.verbose > 0
            )
            
            # 更新特征重要性
            if hasattr(self.models['catboost'], 'feature_importances_'):
                self.feature_importance_ += (
                    self.models['catboost'].feature_importances_ * self.model_weights['catboost']
                )
        
        # 训练LNN模型
        if self.use_lnn and LNNClassifier is not None:
            logger.info("训练LNN模型")
            self.models['lnn'] = LNNClassifier(**self.lnn_params)
            self.models['lnn'].fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                sample_weight=sample_weight
            )
            
            # LNN目前不提供特征重要性
        
        # 归一化特征重要性
        if np.sum(self.feature_importance_) > 0:
            self.feature_importance_ = self.feature_importance_ / np.sum(self.feature_importance_)
        
        self.is_fitted = True
        logger.info("板块轮动模型融合器训练完成")
        return self
    
    def predict_proba(self, X):
        """
        预测类别概率
        
        Args:
            X: 特征数据
            
        Returns:
            预测概率数组 [n_samples, n_classes]
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 数据预处理
        X_scaled = self.scaler.transform(X)
        
        # 收集各模型的预测结果
        probas = []
        weights = []
        
        # XGBoost预测
        if 'xgb' in self.models:
            xgb_proba = self.models['xgb'].predict_proba(X_scaled)
            probas.append(xgb_proba)
            weights.append(self.model_weights['xgb'])
        
        # LightGBM预测
        if 'lgbm' in self.models:
            lgbm_proba = self.models['lgbm'].predict_proba(X_scaled)
            probas.append(lgbm_proba)
            weights.append(self.model_weights['lgbm'])
        
        # CatBoost预测
        if 'catboost' in self.models:
            catboost_proba = self.models['catboost'].predict_proba(X_scaled)
            probas.append(catboost_proba)
            weights.append(self.model_weights['catboost'])
        
        # LNN预测
        if 'lnn' in self.models:
            lnn_proba = self.models['lnn'].predict_proba(X_scaled)
            probas.append(lnn_proba)
            weights.append(self.model_weights['lnn'])
        
        # 加权平均
        weights = np.array(weights) / np.sum(weights)  # 归一化权重
        weighted_proba = np.zeros_like(probas[0])
        
        for i, proba in enumerate(probas):
            weighted_proba += proba * weights[i]
        
        return weighted_proba
    
    def predict(self, X):
        """
        预测类别
        
        Args:
            X: 特征数据
            
        Returns:
            预测类别数组 [n_samples]
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def get_feature_importance(self, feature_names=None):
        """
        获取特征重要性
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            特征重要性DataFrame
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance_
        })
        
        return importance_df.sort_values('importance', ascending=False)
    
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存各个子模型
        submodels = {}
        for name, model in self.models.items():
            if name != 'lnn':  # 常规模型直接保存
                submodels[name] = model
            else:  # LNN模型需要特殊处理
                model_path = f"{path}_{name}"
                model.save(model_path)
                submodels[name] = f"{model_path}"  # 保存路径引用
        
        # 保存模型配置和缩放器
        config = {
            'submodels': submodels,
            'scaler': self.scaler,
            'feature_importance_': self.feature_importance_,
            'model_weights': self.model_weights,
            'use_xgb': self.use_xgb,
            'use_lgbm': self.use_lgbm,
            'use_catboost': self.use_catboost,
            'use_lnn': self.use_lnn
        }
        
        joblib.dump(config, f"{path}.joblib")
        logger.info(f"模型已保存到 {path}")
    
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            self
        """
        # 加载配置
        config = joblib.load(f"{path}.joblib")
        
        # 恢复参数
        self.model_weights = config['model_weights']
        self.use_xgb = config['use_xgb']
        self.use_lgbm = config['use_lgbm']
        self.use_catboost = config['use_catboost']
        self.use_lnn = config['use_lnn']
        self.scaler = config['scaler']
        self.feature_importance_ = config['feature_importance_']
        
        # 恢复模型
        self.models = {}
        submodels = config['submodels']
        
        for name, model in submodels.items():
            if name != 'lnn' or isinstance(model, str):  # 常规模型直接加载
                self.models[name] = model
            else:  # LNN模型特殊处理
                model_path = model
                self.models[name] = LNNClassifier().load(model_path)
        
        self.is_fitted = True
        logger.info(f"模型已从 {path} 加载")
        return self 