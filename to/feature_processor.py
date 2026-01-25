#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征处理器模块
负责特征数据的规范化、变换和预处理，确保训练和预测时特征处理的一致性
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from collections import defaultdict

from to.logger_config import configure_logger

# 配置日志
logger = configure_logger(__name__, 'feature_processor.log')

class FeatureProcessor:
    """
    特征处理器类
    提供统一的特征处理流水线，确保训练和预测时特征处理的一致性
    """
    
    def __init__(self):
        """
        初始化特征处理器
        """
        # 预处理器
        self.scalers = {}  # 特征标准化器
        self.imputers = {}  # 特征缺失值填充器
        self.categorical_encoders = {}  # 分类特征编码器
        self.dynamic_feature_generators = {}  # 动态特征生成器
        self.pca_models = {}  # PCA降维模型
        self.feature_stats = defaultdict(dict)  # 特征统计信息
        
        # 特征清单
        self.feature_columns = None  # 所有有效特征列
        self.numeric_columns = None  # 数值型特征
        self.categorical_columns = None  # 分类特征
        self.interaction_features = None  # 交互特征
        self.sequence_features = None  # 序列特征
        
        # 元数据
        self.metadata = {}
        
        # 日志
        self.logger = logger
    
    def initialize_from_data(self, df, categorical_cols=None):
        """
        从数据中初始化特征处理器
        
        Args:
            df: 特征DataFrame
            categorical_cols: 分类特征列表
        """
        try:
            # 识别特征类型
            if categorical_cols is None:
                # 自动识别分类特征
                categorical_cols = []
                for col in df.columns:
                    if df[col].dtype == 'object' or df[col].nunique() < 10:
                        categorical_cols.append(col)
            
            self.categorical_columns = categorical_cols
            self.numeric_columns = [col for col in df.columns if col not in categorical_cols]
            self.feature_columns = list(df.columns)
            
            # 记录特征统计信息
            for col in self.numeric_columns:
                if col in df.columns:
                    self.feature_stats[col]['mean'] = df[col].mean()
                    self.feature_stats[col]['std'] = df[col].std()
                    self.feature_stats[col]['median'] = df[col].median()
                    self.feature_stats[col]['min'] = df[col].min()
                    self.feature_stats[col]['max'] = df[col].max()
                    self.feature_stats[col]['missing_rate'] = df[col].isnull().mean()
            
            # 初始化预处理器
            for col in self.numeric_columns:
                # 缺失值填充器
                self.imputers[col] = SimpleImputer(strategy='median')
                if col in df.columns:
                    self.imputers[col].fit(df[[col]])
                
                # 标准化器
                self.scalers[col] = StandardScaler()
                if col in df.columns and not df[col].isnull().all():
                    # 填充缺失值后再拟合标准化器
                    col_data = df[[col]].copy()
                    col_data = pd.DataFrame(self.imputers[col].transform(col_data), columns=[col])
                    self.scalers[col].fit(col_data)
            
            # 初始化分类特征编码器
            for col in self.categorical_columns:
                if col in df.columns:
                    unique_values = df[col].dropna().unique()
                    self.categorical_encoders[col] = {val: idx for idx, val in enumerate(unique_values)}
            
            # 准备元数据
            self.metadata = {
                'feature_columns': self.feature_columns,
                'numeric_columns': self.numeric_columns,
                'categorical_columns': self.categorical_columns,
                'feature_stats': dict(self.feature_stats),
                'categorical_encoders': self.categorical_encoders,
            }
            
            self.logger.info(f"成功从数据初始化特征处理器，共 {len(self.feature_columns)} 个特征")
            
        except Exception as e:
            self.logger.error(f"初始化特征处理器时出错: {e}")
            self.logger.error(traceback.format_exc())
            
    def initialize_from_metadata(self, metadata):
        """
        从元数据初始化特征处理器
        
        Args:
            metadata: 特征处理元数据字典
        """
        try:
            self.metadata = metadata
            self.feature_columns = metadata.get('feature_columns', [])
            self.numeric_columns = metadata.get('numeric_columns', [])
            self.categorical_columns = metadata.get('categorical_columns', [])
            self.feature_stats = defaultdict(dict, metadata.get('feature_stats', {}))
            self.categorical_encoders = metadata.get('categorical_encoders', {})
            
            # 重建预处理器
            for col in self.numeric_columns:
                stats = self.feature_stats.get(col, {})
                
                # 重建缺失值填充器
                self.imputers[col] = SimpleImputer(strategy='median')
                if 'median' in stats:
                    self.imputers[col].statistics_ = np.array([stats['median']])
                
                # 重建标准化器
                self.scalers[col] = StandardScaler()
                if 'mean' in stats and 'std' in stats:
                    self.scalers[col].mean_ = np.array([stats['mean']])
                    self.scalers[col].scale_ = np.array([stats['std']])
                    self.scalers[col].var_ = np.array([stats['std'] ** 2])
            
            self.logger.info(f"成功从元数据初始化特征处理器，共 {len(self.feature_columns)} 个特征")
            
        except Exception as e:
            self.logger.error(f"从元数据初始化特征处理器时出错: {e}")
            self.logger.error(traceback.format_exc())
    
    def process_features(self, df, is_training=False, create_interactions=True):
        """
        处理特征
        
        Args:
            df: 特征DataFrame
            is_training: 是否为训练模式
            create_interactions: 是否创建交互特征
            
        Returns:
            处理后的特征DataFrame
        """
        try:
            if df.empty:
                self.logger.warning("输入DataFrame为空")
                return df
            
            # 创建工作副本
            processed_df = df.copy()
            
            # 1. 处理缺失值
            processed_df = self._handle_missing_values(processed_df)
            
            # 2. 处理异常值
            processed_df = self._handle_outliers(processed_df)
            
            # 3. 标准化数值特征
            processed_df = self._standardize_features(processed_df)
            
            # 4. 编码分类特征
            processed_df = self._encode_categorical_features(processed_df)
            
            # 5. 创建交互特征
            if create_interactions:
                processed_df = self._create_interaction_features(processed_df)
            
            # 6. 创建时间特征
            processed_df = self._create_time_features(processed_df)
            
            # 训练模式下，更新特征统计信息
            if is_training:
                self._update_feature_stats(processed_df)
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"处理特征时出错: {e}")
            self.logger.error(traceback.format_exc())
            
            # 发生错误时，尽量返回原始DataFrame
            return df
    
    def _handle_missing_values(self, df):
        """
        处理缺失值
        
        Args:
            df: 特征DataFrame
            
        Returns:
            处理后的DataFrame
        """
        try:
            # 创建工作副本
            result_df = df.copy()
            
            # 数值型特征使用中位数填充
            for col in self.numeric_columns:
                if col in result_df.columns and result_df[col].isnull().any():
                    if col in self.imputers:
                        col_data = result_df[[col]]
                        result_df[col] = self.imputers[col].transform(col_data)
                    else:
                        # 如果没有预训练的imputer，使用当前数据的中位数
                        result_df[col] = result_df[col].fillna(result_df[col].median())
            
            # 分类特征使用众数填充
            for col in self.categorical_columns:
                if col in result_df.columns and result_df[col].isnull().any():
                    mode_value = result_df[col].mode().iloc[0] if not result_df[col].dropna().empty else "未知"
                    result_df[col] = result_df[col].fillna(mode_value)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"处理缺失值时出错: {e}")
            return df
    
    def _handle_outliers(self, df):
        """
        处理异常值

        Args:
            df: 特征DataFrame

        Returns:
            处理后的DataFrame
        """
        try:
            # 创建工作副本
            result_df = df.copy()

            # 对数值型特征进行异常值处理
            for col in self.numeric_columns:
                if col in result_df.columns and not result_df[col].isnull().all():
                    col_values = result_df[col].values

                    # 根据特征类型选择处理方法
                    if col in ['total_mv', 'circ_mv'] or 'mv' in col:
                        # 市值特征：使用分位数方法
                        p1 = np.nanpercentile(col_values, 1)
                        p99_5 = np.nanpercentile(col_values, 99.5)
                        lower_bound = max(0, p1)  # 确保市值不为负
                        upper_bound = p99_5
                    elif col in ['amount', 'volume'] or 'amount' in col or 'volume' in col:
                        # 成交量/成交额特征：使用分位数方法
                        p1 = np.nanpercentile(col_values, 1)
                        p99 = np.nanpercentile(col_values, 99)
                        lower_bound = max(0, p1)  # 确保不为负
                        upper_bound = p99
                    elif 'log_' in col or 'ratio' in col:
                        # 对数特征和比率特征：使用分位数方法
                        p0_5 = np.nanpercentile(col_values, 0.5)
                        p99_5 = np.nanpercentile(col_values, 99.5)
                        lower_bound, upper_bound = p0_5, p99_5
                    else:
                        # 其他特征使用5σ方法（比3σ更宽松）
                        col_mean = result_df[col].mean()
                        col_std = result_df[col].std()
                        lower_bound = col_mean - 5 * col_std
                        upper_bound = col_mean + 5 * col_std

                    # 限制异常值
                    result_df[col] = result_df[col].clip(lower_bound, upper_bound)

            return result_df

        except Exception as e:
            self.logger.error(f"处理异常值时出错: {e}")
            return df
    
    def _standardize_features(self, df):
        """
        标准化数值特征
        
        Args:
            df: 特征DataFrame
            
        Returns:
            处理后的DataFrame
        """
        try:
            # 创建工作副本
            result_df = df.copy()
            
            # 使用预训练的标准化器对数值特征进行标准化
            for col in self.numeric_columns:
                if col in result_df.columns and col in self.scalers:
                    col_data = result_df[[col]]
                    result_df[col] = self.scalers[col].transform(col_data)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"标准化特征时出错: {e}")
            return df
    
    def _encode_categorical_features(self, df):
        """
        编码分类特征
        
        Args:
            df: 特征DataFrame
            
        Returns:
            处理后的DataFrame
        """
        try:
            # 创建工作副本
            result_df = df.copy()
            
            # 对分类特征进行编码
            for col in self.categorical_columns:
                if col in result_df.columns:
                    if col in self.categorical_encoders:
                        # 使用预定义的编码器
                        encoder = self.categorical_encoders[col]
                        
                        # 处理未见过的类别
                        result_df[col] = result_df[col].apply(lambda x: encoder.get(x, -1) if pd.notna(x) else -1)
                    else:
                        # 如果没有预定义的编码器，使用one-hot编码
                        dummies = pd.get_dummies(result_df[col], prefix=col, dummy_na=True)
                        result_df = pd.concat([result_df, dummies], axis=1)
                        result_df = result_df.drop(col, axis=1)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"编码分类特征时出错: {e}")
            return df
    
    def _create_interaction_features(self, df):
        """
        创建交互特征
        
        Args:
            df: 特征DataFrame
            
        Returns:
            处理后的DataFrame
        """
        try:
            # 创建工作副本
            result_df = df.copy()
            
            # 定义要创建交互特征的基本特征
            base_features = [
                col for col in self.numeric_columns 
                if col in result_df.columns 
                and 'return' in col.lower() or 'volume' in col.lower() or 'strength' in col.lower()
            ]
            
            # 限制基本特征数量，避免创建过多交互特征
            if len(base_features) > 5:
                base_features = base_features[:5]
            
            # 创建交互特征
            interactions_created = []
            
            # 创建乘积特征
            for i in range(len(base_features)):
                for j in range(i+1, len(base_features)):
                    col1 = base_features[i]
                    col2 = base_features[j]
                    
                    # 检查两列是否都存在
                    if col1 in result_df.columns and col2 in result_df.columns:
                        interaction_name = f"interaction_{col1}_{col2}"
                        result_df[interaction_name] = result_df[col1] * result_df[col2]
                        interactions_created.append(interaction_name)
            
            # 创建比率特征
            for i in range(len(base_features)):
                for j in range(len(base_features)):
                    if i != j:
                        col1 = base_features[i]
                        col2 = base_features[j]
                        
                        # 检查两列是否都存在
                        if col1 in result_df.columns and col2 in result_df.columns:
                            ratio_name = f"ratio_{col1}_{col2}"
                            # 避免除以零
                            denominator = result_df[col2].replace(0, np.nan)
                            result_df[ratio_name] = result_df[col1] / denominator
                            # 处理无穷大值
                            result_df[ratio_name] = result_df[ratio_name].replace([np.inf, -np.inf], np.nan)
                            # 填充缺失值
                            result_df[ratio_name] = result_df[ratio_name].fillna(0)
                            interactions_created.append(ratio_name)
            
            # 更新交互特征列表
            self.interaction_features = interactions_created
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"创建交互特征时出错: {e}")
            return df
    
    def _create_time_features(self, df):
        """
        创建时间特征
        
        Args:
            df: 特征DataFrame
            
        Returns:
            处理后的DataFrame
        """
        try:
            # 创建工作副本
            result_df = df.copy()
            
            # 检查是否有日期列
            if 'date' not in result_df.columns:
                return result_df
            
            # 确保日期列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(result_df['date']):
                result_df['date'] = pd.to_datetime(result_df['date'])
            
            # 提取日期特征
            result_df['day_of_week'] = result_df['date'].dt.dayofweek
            result_df['month'] = result_df['date'].dt.month
            result_df['quarter'] = result_df['date'].dt.quarter
            result_df['year'] = result_df['date'].dt.year
            result_df['day_of_month'] = result_df['date'].dt.day
            result_df['week_of_year'] = result_df['date'].dt.isocalendar().week
            
            # 创建季节性特征
            result_df['is_month_start'] = result_df['date'].dt.is_month_start.astype(int)
            result_df['is_month_end'] = result_df['date'].dt.is_month_end.astype(int)
            result_df['is_quarter_start'] = result_df['date'].dt.is_quarter_start.astype(int)
            result_df['is_quarter_end'] = result_df['date'].dt.is_quarter_end.astype(int)
            result_df['is_year_start'] = result_df['date'].dt.is_year_start.astype(int)
            result_df['is_year_end'] = result_df['date'].dt.is_year_end.astype(int)
            
            # 创建月份的正弦和余弦特征，以捕捉周期性
            result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
            result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
            
            # 创建周几的正弦和余弦特征
            result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
            result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"创建时间特征时出错: {e}")
            return df
    
    def _update_feature_stats(self, df):
        """
        更新特征统计信息
        
        Args:
            df: 特征DataFrame
        """
        try:
            # 更新数值型特征的统计信息
            for col in self.numeric_columns:
                if col in df.columns:
                    self.feature_stats[col]['mean'] = df[col].mean()
                    self.feature_stats[col]['std'] = df[col].std()
                    self.feature_stats[col]['median'] = df[col].median()
                    self.feature_stats[col]['min'] = df[col].min()
                    self.feature_stats[col]['max'] = df[col].max()
                    self.feature_stats[col]['missing_rate'] = df[col].isnull().mean()
            
            # 更新元数据
            self.metadata['feature_stats'] = dict(self.feature_stats)
            
        except Exception as e:
            self.logger.error(f"更新特征统计信息时出错: {e}")
    
    def create_sequence_features(self, df, seq_length=10):
        """
        创建序列特征
        
        Args:
            df: 特征DataFrame
            seq_length: 序列长度
            
        Returns:
            序列特征字典，格式为 {(sector_code, date): sequence_array}
        """
        try:
            # 检查必要的列是否存在
            if 'sector_code' not in df.columns or 'date' not in df.columns:
                self.logger.error("DataFrame缺少必要的列: sector_code 或 date")
                return {}
            
            # 确保日期列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # 排序DataFrame
            sorted_df = df.sort_values(['sector_code', 'date'])
            
            # 选择用于序列特征的列
            seq_feature_cols = [
                col for col in self.numeric_columns 
                if col in sorted_df.columns 
                and ('return' in col.lower() or 'strength' in col.lower() or 'volume' in col.lower())
            ]
            
            # 如果没有找到合适的特征列，使用所有数值列
            if not seq_feature_cols:
                seq_feature_cols = [col for col in self.numeric_columns if col in sorted_df.columns]
            
            # 限制特征数量，避免序列维度过高
            if len(seq_feature_cols) > 10:
                seq_feature_cols = seq_feature_cols[:10]
            
            # 初始化序列特征字典
            sequence_features = {}
            
            # 按板块分组
            for sector_code, sector_group in sorted_df.groupby('sector_code'):
                sector_data = sector_group.sort_values('date')
                
                # 为每个日期创建序列特征
                for i in range(seq_length, len(sector_data)):
                    # 获取当前行信息
                    current_row = sector_data.iloc[i]
                    current_date = current_row['date']
                    
                    # 获取前seq_length行的数据
                    sequence_data = sector_data.iloc[i-seq_length:i][seq_feature_cols].values
                    
                    # 存储序列特征
                    sequence_features[(sector_code, current_date)] = sequence_data
            
            # 记录使用的序列特征列
            self.sequence_features = seq_feature_cols
            
            return sequence_features
            
        except Exception as e:
            self.logger.error(f"创建序列特征时出错: {e}")
            self.logger.error(traceback.format_exc())
            return {}
    
    def create_pca_features(self, df, n_components=5, feature_prefix='pca_'):
        """
        创建PCA特征
        
        Args:
            df: 特征DataFrame
            n_components: PCA组件数量
            feature_prefix: PCA特征前缀
            
        Returns:
            添加了PCA特征的DataFrame
        """
        try:
            # 创建工作副本
            result_df = df.copy()
            
            # 选择数值特征
            numeric_features = [col for col in self.numeric_columns if col in result_df.columns]
            
            # 检查数值特征是否足够
            if len(numeric_features) < n_components:
                self.logger.warning(f"数值特征数量 ({len(numeric_features)}) 小于请求的PCA组件数量 ({n_components})，将使用所有特征")
                n_components = len(numeric_features)
            
            # 提取特征数据
            feature_data = result_df[numeric_features].copy()
            
            # 处理缺失值
            feature_data = feature_data.fillna(feature_data.median())
            
            # 创建并拟合PCA模型
            pca_model = PCA(n_components=n_components)
            pca_components = pca_model.fit_transform(feature_data)
            
            # 添加PCA特征到结果DataFrame
            for i in range(n_components):
                result_df[f"{feature_prefix}{i+1}"] = pca_components[:, i]
            
            # 保存PCA模型
            model_key = '_'.join(sorted(numeric_features))
            self.pca_models[model_key] = pca_model
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"创建PCA特征时出错: {e}")
            self.logger.error(traceback.format_exc())
            return df
    
    def save_metadata(self, filepath):
        """
        保存特征处理器元数据
        
        Args:
            filepath: 文件路径
        """
        try:
            metadata = {
                'feature_columns': self.feature_columns,
                'numeric_columns': self.numeric_columns,
                'categorical_columns': self.categorical_columns,
                'interaction_features': self.interaction_features,
                'sequence_features': self.sequence_features,
                'feature_stats': dict(self.feature_stats),
                'categorical_encoders': self.categorical_encoders,
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存元数据
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"特征处理器元数据已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存特征处理器元数据时出错: {e}")
            self.logger.error(traceback.format_exc())
    
    def load_metadata(self, filepath):
        """
        加载特征处理器元数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否成功加载
        """
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"元数据文件不存在: {filepath}")
                return False
            
            # 加载元数据
            with open(filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 初始化处理器
            self.initialize_from_metadata(metadata)
            
            self.logger.info(f"从文件加载特征处理器元数据: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载特征处理器元数据时出错: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def get_feature_importance(self, model):
        """
        获取特征重要性
        
        Args:
            model: 训练好的模型
            
        Returns:
            特征重要性字典
        """
        try:
            feature_importance = {}
            
            # 检查模型类型并获取特征重要性
            if hasattr(model, 'feature_importances_'):
                # 树模型（如RandomForest, XGBoost等）
                importances = model.feature_importances_
                feature_names = model.feature_names_ if hasattr(model, 'feature_names_') else self.feature_columns
                
                for name, importance in zip(feature_names, importances):
                    feature_importance[name] = float(importance)
                    
            elif hasattr(model, 'coef_'):
                # 线性模型（如LinearRegression, Lasso等）
                coefficients = model.coef_
                feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else self.feature_columns
                
                if coefficients.ndim > 1:
                    # 多目标或多类别模型
                    coefficients = np.abs(coefficients).mean(axis=0)
                
                for name, coef in zip(feature_names, coefficients):
                    feature_importance[name] = float(abs(coef))
            
            # 对特征重要性进行排序
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            self.logger.error(f"获取特征重要性时出错: {e}")
            self.logger.error(traceback.format_exc())
            return {} 