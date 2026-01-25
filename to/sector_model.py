#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块轮动模型API
提供板块轮动模型的训练、预测和回测功能的统一接口
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import traceback
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from to.logger_config import configure_logger
from to.data_processor import DataProcessor
from to.feature_processor import FeatureProcessor
from to.model_manager import ModelManager

# 配置日志
logger = configure_logger(__name__, 'sector_model.log')

class SectorModel:
    """
    板块轮动模型类
    整合数据处理、特征工程和模型管理，提供统一的API接口
    """
    
    def __init__(self, model_dir=None, data_dir=None, use_lnn=True):
        """
        初始化板块轮动模型
        
        Args:
            model_dir: 模型保存目录
            data_dir: 数据目录
            use_lnn: 是否使用液体神经网络
        """
        # 获取项目根目录
        self.project_root = self._get_project_root()
        
        # 初始化子组件
        self.data_processor = DataProcessor(data_dir)
        self.feature_processor = FeatureProcessor()
        self.model_manager = ModelManager(model_dir)
        
        # 配置参数
        self.use_lnn = use_lnn
        
        # 日志记录器
        self.logger = logger
        
        self.logger.info("板块轮动模型已初始化")
    
    def _get_project_root(self):
        """获取项目根目录"""
        current_file = Path(__file__).resolve()
        return current_file.parent.parent
    
    def train(self, sector_list=None, start_date=None, end_date=None):
        """
        训练板块轮动模型
        
        Args:
            sector_list: 板块列表，如果为None则加载所有板块
            start_date: 训练开始日期，格式为'YYYYMMDD'
            end_date: 训练结束日期，格式为'YYYYMMDD'
            
        Returns:
            训练结果字典，包含每个板块的模型信息
        """
        try:
            start_time = time.time()
            self.logger.info(f"开始训练板块轮动模型，时间范围: {start_date} 至 {end_date}")
            
            # 加载板块列表
            if sector_list is None:
                sector_list = self.data_processor.load_sector_list()
                self.logger.info(f"已加载 {len(sector_list)} 个板块")
            
            # 批量加载特征数据
            features_df = self.data_processor.batch_load_features(sector_list, start_date, end_date)
            
            if features_df.empty:
                self.logger.error("加载特征数据失败，无法训练模型")
                return {}
            
            self.logger.info(f"已加载特征数据，共 {len(features_df)} 行，包含 {features_df['sector_code'].nunique()} 个板块")
            
            # 初始化特征处理器
            categorical_cols = ['sector_code', 'market_type']
            self.feature_processor.initialize_from_data(features_df, categorical_cols)
            
            # 处理特征
            processed_df = self.feature_processor.process_features(features_df, is_training=True)
            
            # 创建序列特征
            sequence_features = self.feature_processor.create_sequence_features(processed_df, seq_length=10)
            
            # 创建训练目标（未来2天收益率）
            self._create_future_returns(processed_df)
            
            # 训练每个板块的模型
            models_result = {}
            
            for sector_code in sector_list:
                self.logger.info(f"开始训练板块 {sector_code} 的模型")
                
                # 获取板块数据
                sector_df = processed_df[processed_df['sector_code'] == sector_code].copy()
                
                if len(sector_df) < 60:  # 至少需要60个样本
                    self.logger.warning(f"板块 {sector_code} 数据量不足，跳过训练")
                    continue
                
                # 训练强度模型
                strength_model = self._train_strength_model(sector_df, sector_code)
                
                # 训练LNN模型
                lnn_model = None
                if self.use_lnn:
                    lnn_model = self._train_lnn_model(sector_df, sector_code, sequence_features)
                
                # 保存模型
                models_result[sector_code] = {
                    'strength_model': strength_model is not None,
                    'lnn_model': lnn_model is not None
                }
            
            # 保存特征处理器元数据
            metadata_path = os.path.join(self.model_manager.model_dir, "feature_processor_metadata.json")
            self.feature_processor.save_metadata(metadata_path)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"板块轮动模型训练完成，耗时 {elapsed_time:.2f} 秒")
            
            return models_result
            
        except Exception as e:
            self.logger.error(f"训练板块轮动模型时出错: {e}")
            self.logger.error(traceback.format_exc())
            return {}
    
    def _train_strength_model(self, sector_df, sector_code):
        """
        训练板块强度模型
        
        Args:
            sector_df: 板块特征DataFrame
            sector_code: 板块代码
            
        Returns:
            训练好的模型
        """
        try:
            # 准备特征和目标变量
            feature_cols = [col for col in sector_df.columns if col not in ['date', 'sector_code', 'future_return_2d', 'future_return_5d', 'future_return_10d']]
            
            X = sector_df[feature_cols].copy()
            y = sector_df['future_return_2d']
            
            # 分割训练集和测试集（最后30天作为测试集）
            train_size = max(len(sector_df) - 30, int(len(sector_df) * 0.8))
            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            X_test = X.iloc[train_size:]
            y_test = y.iloc[train_size:]
            
            # 训练模型
            model = self.model_manager.train_model(
                X_train, y_train, 
                model_type='xgboost', 
                feature_names=feature_cols
            )
            
            if model is None:
                self.logger.error(f"训练板块 {sector_code} 的强度模型失败")
                return None
            
            # 评估模型
            if not X_test.empty and not y_test.empty:
                metrics = self.model_manager.evaluate_model(model, X_test, y_test, model_type='xgboost')
                self.model_manager.update_performance_history(sector_code, 'strength_model', metrics, model_type='strength')
            
            # 保存模型
            self.model_manager.save_model(
                model, 
                model_type='xgboost',
                sector_code=sector_code,
                model_name='strength_model',
                feature_columns=feature_cols,
                metadata={'feature_importance': self.feature_processor.get_feature_importance(model)}
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"训练板块 {sector_code} 强度模型时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _train_lnn_model(self, sector_df, sector_code, sequence_features):
        """
        训练板块LNN模型
        
        Args:
            sector_df: 板块特征DataFrame
            sector_code: 板块代码
            sequence_features: 序列特征字典
            
        Returns:
            训练好的模型
        """
        if not self.use_lnn:
            return None
            
        try:
            # 准备序列特征数据
            seq_data = []
            seq_targets = []
            
            for index, row in sector_df.iterrows():
                date = row['date']
                key = (sector_code, date)
                
                if key in sequence_features:
                    seq_data.append(sequence_features[key])
                    seq_targets.append(row['future_return_2d'])
            
            if not seq_data:
                self.logger.warning(f"板块 {sector_code} 没有可用的序列特征，跳过LNN模型训练")
                return None
            
            # 转换为numpy数组
            X = np.array(seq_data)
            y = np.array(seq_targets)
            
            # 分割训练集和测试集
            train_size = max(len(X) - 30, int(len(X) * 0.8))
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_test = X[train_size:]
            y_test = y[train_size:]
            
            # 训练模型
            model = self.model_manager.train_model(
                X_train, y_train, 
                model_type='improved_lnn'
            )
            
            if model is None:
                self.logger.error(f"训练板块 {sector_code} 的LNN模型失败")
                return None
            
            # 评估模型
            if len(X_test) > 0 and len(y_test) > 0:
                metrics = self.model_manager.evaluate_model(model, X_test, y_test, model_type='improved_lnn')
                self.model_manager.update_performance_history(sector_code, 'lnn_model', metrics, model_type='lnn')
            
            # 保存模型
            self.model_manager.save_model(
                model, 
                model_type='improved_lnn',
                sector_code=sector_code,
                model_name='lnn_model'
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"训练板块 {sector_code} LNN模型时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def predict(self, sector_list=None, date=None):
        """
        预测板块未来收益
        
        Args:
            sector_list: 板块列表，如果为None则加载所有板块
            date: 预测日期，格式为'YYYYMMDD'，如果为None则使用最新日期
            
        Returns:
            预测结果DataFrame，包含板块代码、预测收益率和预测确信度
        """
        try:
            self.logger.info(f"开始预测板块未来收益，日期: {date}")
            
            # 如果没有指定日期，使用最新日期
            if date is None:
                trading_days = self.data_processor.trading_calendar
                if trading_days:
                    date = trading_days[-1]
                else:
                    self.logger.error("无法确定最新交易日，请指定预测日期")
                    return pd.DataFrame()
            
            # 加载板块列表
            if sector_list is None:
                sector_list = self.data_processor.load_sector_list()
                self.logger.info(f"已加载 {len(sector_list)} 个板块")
            
            # 加载市场状态
            market_status = self.data_processor.load_market_status(date)
            
            # 加载特征处理器元数据
            metadata_path = os.path.join(self.model_manager.model_dir, "feature_processor_metadata.json")
            if os.path.exists(metadata_path):
                self.feature_processor.load_metadata(metadata_path)
            else:
                self.logger.warning("未找到特征处理器元数据，将使用默认设置")
            
            # 收集预测结果
            predictions = []
            
            for sector_code in sector_list:
                # 加载特征数据
                features = self.data_processor.load_sector_features(sector_code, date)
                if not features:
                    self.logger.warning(f"无法加载板块 {sector_code} 的特征数据，跳过预测")
                    continue
                
                # 构建特征DataFrame
                if 'features' in features:
                    feature_dict = features['features']
                else:
                    feature_dict = features
                
                # 添加板块代码和日期
                feature_dict['sector_code'] = sector_code
                feature_dict['date'] = pd.to_datetime(date)
                
                # 转换为DataFrame
                feature_df = pd.DataFrame([feature_dict])
                
                # 处理特征
                processed_df = self.feature_processor.process_features(feature_df, is_training=False)
                
                # 预测
                prediction = self._predict_sector(sector_code, processed_df, market_status)
                
                if prediction:
                    predictions.append({
                        'sector_code': sector_code,
                        'date': pd.to_datetime(date),
                        'predicted_return': prediction['predicted_return'],
                        'confidence': prediction['confidence'],
                        'strength_weight': prediction['strength_weight'],
                        'lnn_weight': prediction['lnn_weight']
                    })
            
            # 转换为DataFrame
            if predictions:
                result_df = pd.DataFrame(predictions)
                result_df = result_df.sort_values('predicted_return', ascending=False).reset_index(drop=True)
                self.logger.info(f"预测完成，共 {len(result_df)} 个板块")
                return result_df
            else:
                self.logger.warning("没有生成任何预测结果")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"预测板块未来收益时出错: {e}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _predict_sector(self, sector_code, feature_df, market_status):
        """
        预测单个板块的未来收益
        
        Args:
            sector_code: 板块代码
            feature_df: 特征DataFrame
            market_status: 市场状态数据
            
        Returns:
            预测结果字典
        """
        try:
            # 获取特征列
            metadata_path = os.path.join(self.model_manager.model_dir, sector_code, "strength_model_metadata.json")
            feature_columns = None
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    feature_columns = metadata.get('feature_columns')
            
            if feature_columns is None:
                feature_columns = [col for col in feature_df.columns if col not in ['date', 'sector_code']]
            
            # 加载强度模型
            strength_model, strength_metadata = self.model_manager.load_model(sector_code, 'strength_model')
            
            # 加载LNN模型
            lnn_model, lnn_metadata = self.model_manager.load_model(sector_code, 'lnn_model')
            
            # 如果没有模型，返回None
            if strength_model is None and lnn_model is None:
                self.logger.warning(f"板块 {sector_code} 没有可用的模型，跳过预测")
                return None
            
            # 确保特征列存在
            feature_columns = [col for col in feature_columns if col in feature_df.columns]
            
            # 准备模型包
            model_package = {
                'performance_history': self.model_manager.get_performance_history(sector_code, 'strength_model')
            }
            
            # 计算自适应权重
            strength_weight, lnn_weight = self.model_manager.calculate_adaptive_weights(model_package, market_status)
            
            # 预测结果
            predicted_return = 0.0
            confidence = 0.5
            
            # 如果有强度模型，使用强度模型预测
            if strength_model is not None:
                X = feature_df[feature_columns]
                strength_pred = self.model_manager.predict(strength_model, X, model_type='xgboost')
                
                if strength_pred is not None:
                    predicted_return = float(strength_pred[0]) * strength_weight
            
            # 如果有LNN模型，使用LNN模型预测
            if lnn_model is not None and self.use_lnn:
                # 创建LNN特征
                lnn_feature = self._ensure_lnn_features(feature_df)
                
                if lnn_feature is not None:
                    # 预测
                    X = np.array([lnn_feature])
                    lnn_pred = self.model_manager.predict(lnn_model, X, model_type='improved_lnn')
                    
                    if lnn_pred is not None:
                        predicted_return += float(lnn_pred[0]) * lnn_weight
                        
                        # 更新置信度
                        if lnn_pred[0] > 0.02:  # 较大的正向预测
                            confidence = 0.7
                        elif lnn_pred[0] < -0.02:  # 较大的负向预测
                            confidence = 0.7
                        else:  # 接近中性的预测
                            confidence = 0.5
            
            return {
                'predicted_return': predicted_return,
                'confidence': confidence,
                'strength_weight': strength_weight,
                'lnn_weight': lnn_weight
            }
            
        except Exception as e:
            self.logger.error(f"预测板块 {sector_code} 未来收益时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _ensure_lnn_features(self, feature_df):
        """
        确保LNN特征存在，不存在则生成
        
        Args:
            feature_df: 特征DataFrame
            
        Returns:
            LNN特征数组
        """
        try:
            # 如果已有LNN特征，使用现有特征
            if 'lnn_output_feature' in feature_df.columns:
                return feature_df['lnn_output_feature'].values
            
            # 创建PCA特征作为备用
            pca_df = self.feature_processor.create_pca_features(feature_df, n_components=10)
            
            # 提取数值型特征
            numeric_features = [col for col in pca_df.columns 
                               if col.startswith('pca_') 
                               or 'return' in col.lower() 
                               or 'strength' in col.lower()]
            
            if not numeric_features:
                self.logger.warning("没有找到可用于LNN的特征")
                return None
            
            # 限制特征数量
            if len(numeric_features) > 10:
                numeric_features = numeric_features[:10]
            
            # 构建特征序列
            feature_seq = []
            for col in numeric_features:
                if col in pca_df.columns:
                    feature_seq.append(pca_df[col].values[0])
            
            # 如果特征不足，填充0
            while len(feature_seq) < 10:
                feature_seq.append(0.0)
            
            return np.array(feature_seq)
            
        except Exception as e:
            self.logger.error(f"生成LNN特征时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _create_future_returns(self, df, days_list=[2, 5, 10]):
        """
        创建未来收益率特征
        
        Args:
            df: 特征DataFrame
            days_list: 未来天数列表
            
        Returns:
            添加了未来收益率的DataFrame
        """
        try:
            # 确保日期列是日期时间类型
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # 按板块分组
            for sector_code, sector_group in df.groupby('sector_code'):
                # 按日期排序
                sector_data = sector_group.sort_values('date')
                
                # 获取日期索引
                dates = sector_data['date'].tolist()
                
                for days in days_list:
                    future_return_col = f'future_return_{days}d'
                    
                    # 初始化为NaN
                    df.loc[sector_data.index, future_return_col] = np.nan
                    
                    # 计算未来收益率
                    for i in range(len(dates) - days):
                        current_idx = sector_data.iloc[i].name
                        future_idx = sector_data.iloc[i + days].name
                        
                        # 如果有收盘价列，使用收盘价计算
                        if 'close_price' in sector_data.columns:
                            current_price = sector_data.iloc[i]['close_price']
                            future_price = sector_data.iloc[i + days]['close_price']
                            future_return = (future_price / current_price) - 1
                        else:
                            # 否则使用强度值的变化作为代理
                            current_strength = sector_data.iloc[i]['strength']
                            future_strength = sector_data.iloc[i + days]['strength']
                            future_return = future_strength - current_strength
                        
                        df.loc[current_idx, future_return_col] = future_return
            
            return df
            
        except Exception as e:
            self.logger.error(f"创建未来收益率特征时出错: {e}")
            self.logger.error(traceback.format_exc())
            return df
    
    def backtest(self, sector_list=None, start_date=None, end_date=None, top_n=5):
        """
        回测板块轮动策略
        
        Args:
            sector_list: 板块列表，如果为None则加载所有板块
            start_date: 回测开始日期，格式为'YYYYMMDD'
            end_date: 回测结束日期，格式为'YYYYMMDD'
            top_n: 每天选择的前N个板块
            
        Returns:
            回测结果字典
        """
        try:
            self.logger.info(f"开始回测板块轮动策略，时间范围: {start_date} 至 {end_date}，选择前 {top_n} 个板块")
            
            # 加载板块列表
            if sector_list is None:
                sector_list = self.data_processor.load_sector_list()
                self.logger.info(f"已加载 {len(sector_list)} 个板块")
            
            # 获取日期范围内的交易日
            trading_days = [d for d in self.data_processor.trading_calendar 
                           if (start_date is None or d >= start_date) 
                           and (end_date is None or d <= end_date)]
            
            if not trading_days:
                self.logger.error(f"在指定的日期范围内没有交易日")
                return {}
            
            # 回测结果
            backtest_result = {
                'daily_returns': [],
                'cumulative_returns': [],
                'positions': [],
                'metrics': {}
            }
            
            # 初始资金
            initial_capital = 1000000.0
            current_capital = initial_capital
            
            # 每日收益
            daily_returns = []
            
            # 加载特征处理器元数据
            metadata_path = os.path.join(self.model_manager.model_dir, "feature_processor_metadata.json")
            if os.path.exists(metadata_path):
                self.feature_processor.load_metadata(metadata_path)
            
            # 遍历每个交易日
            for i, date in enumerate(trading_days):
                self.logger.info(f"回测日期: {date}，进度: {i+1}/{len(trading_days)}")
                
                # 获取当日市场状态
                market_status = self.data_processor.load_market_status(date)
                
                # 预测当日板块收益
                predictions = self.predict(sector_list, date)
                
                if predictions.empty:
                    self.logger.warning(f"日期 {date} 没有预测结果，跳过")
                    continue
                
                # 选择前N个板块
                selected_sectors = predictions.head(top_n)
                
                # 记录持仓
                positions = selected_sectors['sector_code'].tolist()
                backtest_result['positions'].append({
                    'date': date,
                    'positions': positions
                })
                
                # 如果是最后一个交易日，跳过计算收益
                if i == len(trading_days) - 1:
                    continue
                
                # 计算下一个交易日的收益
                next_date = trading_days[i + 1]
                
                # 每个板块分配相同资金
                position_size = current_capital / top_n
                
                # 计算持仓收益
                position_returns = []
                
                for sector_code in positions:
                    # 加载下一个交易日的特征
                    next_features = self.data_processor.load_sector_features(sector_code, next_date)
                    current_features = self.data_processor.load_sector_features(sector_code, date)
                    
                    if not next_features or not current_features:
                        position_returns.append(0.0)
                        continue
                    
                    # 计算实际收益率
                    if 'close_price' in next_features and 'close_price' in current_features:
                        current_price = current_features['close_price']
                        next_price = next_features['close_price']
                        actual_return = (next_price / current_price) - 1
                    else:
                        # 使用强度值变化作为代理
                        current_strength = current_features.get('strength', 0)
                        next_strength = next_features.get('strength', 0)
                        actual_return = next_strength - current_strength
                    
                    position_returns.append(actual_return)
                
                # 计算当日组合收益率
                if position_returns:
                    daily_return = sum(position_returns) / len(position_returns)
                else:
                    daily_return = 0.0
                
                # 更新资金
                current_capital *= (1 + daily_return)
                
                # 记录收益
                daily_returns.append({
                    'date': next_date,
                    'daily_return': daily_return,
                    'cumulative_return': (current_capital / initial_capital) - 1
                })
            
            # 转换为DataFrame
            if daily_returns:
                returns_df = pd.DataFrame(daily_returns)
                
                # 计算回测指标
                total_return = returns_df['cumulative_return'].iloc[-1]
                annual_return = (1 + total_return) ** (252 / len(returns_df)) - 1
                daily_std = returns_df['daily_return'].std()
                annual_std = daily_std * (252 ** 0.5)
                sharpe_ratio = annual_return / annual_std if annual_std > 0 else 0
                
                # 计算最大回撤
                cumulative_returns = (1 + returns_df['daily_return']).cumprod()
                running_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns / running_max) - 1
                max_drawdown = drawdown.min()
                
                # 记录指标
                backtest_result['metrics'] = {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'annual_volatility': annual_std,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': (returns_df['daily_return'] > 0).mean()
                }
                
                backtest_result['daily_returns'] = returns_df.to_dict('records')
                
                self.logger.info(f"回测完成，总收益率: {total_return:.2%}，年化收益率: {annual_return:.2%}，夏普比率: {sharpe_ratio:.2f}")
            
            return backtest_result
            
        except Exception as e:
            self.logger.error(f"回测板块轮动策略时出错: {e}")
            self.logger.error(traceback.format_exc())
            return {}

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="板块轮动模型")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'backtest'], 
                      help='运行模式: train, predict, backtest')
    parser.add_argument('--start_date', type=str, help='开始日期，格式为YYYYMMDD')
    parser.add_argument('--end_date', type=str, help='结束日期，格式为YYYYMMDD')
    parser.add_argument('--use_lnn', action='store_true', help='是否使用液体神经网络')
    parser.add_argument('--top_n', type=int, default=5, help='回测时选择的前N个板块')
    parser.add_argument('--output', type=str, help='输出文件路径')
    
    args = parser.parse_args()
    
    # 初始化模型
    model = SectorModel(use_lnn=args.use_lnn)
    
    # 根据模式执行相应操作
    if args.mode == 'train':
        # 训练模型
        result = model.train(start_date=args.start_date, end_date=args.end_date)
        print(f"训练结果: {len(result)} 个板块的模型已训练")
        
    elif args.mode == 'predict':
        # 预测未来收益
        predictions = model.predict(date=args.end_date)
        
        if not predictions.empty:
            print(f"预测结果: {len(predictions)} 个板块")
            print(predictions.head())
            
            # 如果指定了输出文件，保存预测结果
            if args.output:
                predictions.to_csv(args.output, index=False, encoding='utf-8')
                print(f"预测结果已保存至: {args.output}")
        else:
            print("没有生成预测结果")
            
    elif args.mode == 'backtest':
        # 回测策略
        result = model.backtest(
            start_date=args.start_date, 
            end_date=args.end_date,
            top_n=args.top_n
        )
        
        if result and 'metrics' in result:
            metrics = result['metrics']
            print(f"回测结果:")
            print(f"总收益率: {metrics['total_return']:.2%}")
            print(f"年化收益率: {metrics['annual_return']:.2%}")
            print(f"年化波动率: {metrics['annual_volatility']:.2%}")
            print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
            print(f"最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"胜率: {metrics['win_rate']:.2%}")
            
            # 如果指定了输出文件，保存回测结果
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"回测结果已保存至: {args.output}")
        else:
            print("回测失败，没有生成结果")

if __name__ == "__main__":
    main() 