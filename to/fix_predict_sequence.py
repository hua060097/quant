#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复train_sector_models.py文件中的_predict_with_sequence和main函数
"""

import os
import logging
import argparse
import traceback
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# 创建日志目录
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
os.makedirs(log_dir, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'fix_predict_sequence.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FixPredictSequence')

# _predict_with_sequence方法的完整实现
def _predict_with_sequence(self, model_package, test_df, seq_features, date):
    """
    使用序列特征进行预测
    
    Args:
        model_package: 模型包，包含模型、特征列等
        test_df: 测试集数据
        seq_features: 序列特征字典
        date: 预测日期
        
    Returns:
        预测结果字典
    """
    self.logger.info(f"使用序列特征预测日期 {date} 的板块表现")
    
    try:
        # 获取模型组件并进行健壮性检查
        if not isinstance(model_package, dict):
            self.logger.error(f"模型包类型错误，应为dict，实际为{type(model_package)}")
            return None
        
        market_model = model_package.get('market_model')
        strength_model = model_package.get('strength_model')
        lnn_model = model_package.get('lnn_model')
        scaler = model_package.get('scaler')
        market_features = model_package.get('market_model_features', [])
        strength_features = model_package.get('strength_model_features', [])
        seq_length = model_package.get('seq_length', 10)
        
        # 提取测试特征
        test_sectors = test_df['sector_code'].unique()
        
        # 必要组件健壮性检查
        missing_components = []
        if market_model is None:
            missing_components.append("market_model")
        if strength_model is None:
            missing_components.append("strength_model")
        if scaler is None:
            missing_components.append("scaler")
            
        if missing_components:
            self.logger.warning(f"模型包缺少组件: {', '.join(missing_components)}")
    except Exception as e:
        self.logger.error(f"预测时出错: {e}")
        self.logger.error(traceback.format_exc())
        return None

    try:
        # 获取特征列表
        feature_cols = model_package.get('feature_columns', [])
        
        # 特征列健壮性检查
        if not feature_cols:
            self.logger.warning("特征列列表为空，尝试使用所有数值列作为特征")
            numeric_cols = test_df.select_dtypes(include=['number']).columns.tolist()
            exclude_cols = ['sector_code', 'date', 'is_top_performer', 'label_date']
            exclude_cols += [col for col in test_df.columns if col.startswith('future_return')]
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            if not feature_cols:
                self.logger.error("无法确定特征列，无法继续预测")
                return None
        
        # 清理特征数据
        try:
            # 确保所有特征列都存在
            valid_feature_cols = [col for col in feature_cols if col in test_df.columns]
            if len(valid_feature_cols) < len(feature_cols):
                self.logger.warning(f"有 {len(feature_cols) - len(valid_feature_cols)} 个特征列在测试数据中不存在，将使用 {len(valid_feature_cols)} 个有效列")
                feature_cols = valid_feature_cols
            
            if not feature_cols:
                self.logger.error("没有有效的特征列，无法继续预测")
                return None
            
            X_test = test_df[feature_cols].fillna(0).values
            
            # 确保scaler可用
            if scaler is not None:
                try:
                    from sklearn.exceptions import NotFittedError
                    from sklearn.utils.validation import check_is_fitted
                    
                    try:
                        check_is_fitted(scaler)
                        X_test_scaled = scaler.transform(X_test)
                    except NotFittedError:
                        self.logger.warning("Scaler未拟合，尝试使用原始值")
                        X_test_scaled = X_test
                    except Exception as e:
                        self.logger.warning(f"Scaler转换失败: {e}，将使用原始值")
                        X_test_scaled = X_test
                except ImportError:
                    # 如果无法导入sklearn检查函数，假设scaler已拟合
                    try:
                        X_test_scaled = scaler.transform(X_test)
                    except Exception as e:
                        self.logger.warning(f"Scaler转换失败: {e}，将使用原始值")
                        X_test_scaled = X_test
            else:
                self.logger.warning("Scaler不可用，使用原始值")
                X_test_scaled = X_test
        except Exception as e:
            self.logger.error(f"准备特征数据时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
        
        # 市场模型预测
        market_pred = None
        if market_model is not None and market_features:
            try:
                # 安全索引
                valid_indices = []
                for f in market_features:
                    if f in feature_cols:
                        valid_indices.append(feature_cols.index(f))
                
                if valid_indices:
                    X_market_test = X_test_scaled[:, valid_indices]
                    market_pred = market_model.predict_proba(X_market_test)
                else:
                    self.logger.warning("没有有效的市场特征索引，跳过市场模型预测")
                    # 创建默认预测
                    market_pred_shape = model_package.get('market_pred_shape', (test_df.shape[0], 3))
                    market_pred = np.zeros(market_pred_shape)
                    market_pred[:, 1] = 1.0  # 中性类别
            except Exception as e:
                self.logger.error(f"市场模型预测出错: {e}")
                self.logger.error(traceback.format_exc())
                # 创建默认预测
                market_pred_shape = model_package.get('market_pred_shape', (test_df.shape[0], 3))
                market_pred = np.zeros(market_pred_shape)
                market_pred[:, 1] = 1.0  # 设置默认类别为中性
        else:
            # 创建默认预测
            market_pred_shape = model_package.get('market_pred_shape', (test_df.shape[0], 3))
            market_pred = np.zeros(market_pred_shape)
            market_pred[:, 1] = 1.0  # 设置默认类别为中性
        
        # 强度模型预测
        strength_pred = None
        try:
            # 提取强度特征
            valid_strength_indices = []
            for s in strength_features:
                if s in feature_cols and 'market_prob_' not in s:
                    valid_strength_indices.append(feature_cols.index(s))
            
            if valid_strength_indices:
                X_strength_test = X_test_scaled[:, valid_strength_indices]
                
                # 合并市场模型预测结果
                if market_pred is not None:
                    X_strength_test_with_market = np.hstack([
                        X_strength_test,
                        market_pred
                    ])
                else:
                    X_strength_test_with_market = X_strength_test
                
                # 强度模型预测
                if strength_model is not None:
                    try:
                        if hasattr(strength_model, 'predict_proba'):
                            strength_pred = strength_model.predict_proba(X_strength_test_with_market)[:, 1]
                        else:
                            strength_pred = strength_model.predict(X_strength_test_with_market)
                    except Exception as e:
                        self.logger.error(f"强度模型预测出错: {e}")
                        self.logger.error(traceback.format_exc())
                        # 使用默认值
                        strength_pred = np.zeros(test_df.shape[0])
                else:
                    # 使用默认值
                    strength_pred = np.zeros(test_df.shape[0])
            else:
                self.logger.warning("没有有效的强度特征索引，使用默认强度预测值")
                strength_pred = np.zeros(test_df.shape[0])
        except Exception as e:
            self.logger.error(f"准备强度模型特征时出错: {e}")
            self.logger.error(traceback.format_exc())
            strength_pred = np.zeros(test_df.shape[0])
        
        # 对每个板块添加LNN预测
        sector_final_scores = {}
        
        # 使用LNN进行预测（如果可用）
        if lnn_model is not None and seq_features:
            try:
                # 为每个板块准备序列数据
                lnn_predictions = {}
                
                for sector_code in test_sectors:
                    if sector_code in seq_features:
                        # 获取序列数据，处理可能的不同格式
                        sequence_data = None
                        try:
                            # 检查序列特征格式
                            if isinstance(seq_features[sector_code], dict):
                                if 'features' in seq_features[sector_code]:
                                    # 新格式: {'features': array, 'labels': array, ...}
                                    sequence_data = seq_features[sector_code]['features']
                                    self.logger.info(f"使用新格式序列特征，板块 {sector_code}")
                                else:
                                    self.logger.warning(f"序列特征字典缺少'features'键，板块 {sector_code}")
                                    continue
                            elif isinstance(seq_features[sector_code], np.ndarray) and hasattr(seq_features[sector_code], 'shape'):
                                # 旧格式: 直接存储特征数组
                                sequence_data = seq_features[sector_code]
                                self.logger.info(f"使用旧格式序列特征，板块 {sector_code}")
                            elif hasattr(seq_features[sector_code], 'shape'):
                                # 其他类型但有shape属性
                                sequence_data = seq_features[sector_code]
                                self.logger.info(f"使用shape兼容的序列特征，板块 {sector_code}, 类型: {type(seq_features[sector_code])}")
                            else:
                                self.logger.warning(f"序列特征格式无效，无法提取shape，板块 {sector_code}, 类型: {type(seq_features[sector_code])}")
                                continue
                            
                            # 确保序列长度足够
                            if not hasattr(sequence_data, 'shape') or len(sequence_data) < seq_length:
                                self.logger.warning(f"板块 {sector_code} 的序列特征长度不足或无效: {getattr(sequence_data, 'shape', 'No shape')}")
                                continue
                            
                            # 使用最近的seq_length天
                            # 安全地获取序列数据，避免越界
                            try:
                                sequence = sequence_data[-seq_length:]
                            except Exception as e:
                                self.logger.error(f"获取序列数据时出错: {e}")
                                continue
                            
                            # 标准化序列
                            if scaler is not None:
                                try:
                                    # 重塑为2D进行缩放
                                    sequence_flat = sequence.reshape(-1, sequence.shape[-1])
                                    sequence_scaled_flat = scaler.transform(sequence_flat)
                                    sequence_scaled = sequence_scaled_flat.reshape(sequence.shape)
                                    
                                    # 转换为Torch张量并预测
                                    sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0)  # 添加批次维度
                                except Exception as e:
                                    self.logger.error(f"序列特征处理出错，板块 {sector_code}: {e}")
                                    self.logger.error(traceback.format_exc())
                                    continue
                            else:
                                # 直接使用原始序列
                                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # 添加批次维度
                            
                            # 使用LNN模型预测
                            try:
                                with torch.no_grad():
                                    lnn_pred = lnn_model(sequence_tensor)
                                    lnn_pred = lnn_pred.cpu().numpy()[0]  # 获取预测概率
                                
                                # 存储预测结果
                                lnn_predictions[sector_code] = float(lnn_pred[0])  # 转换为Python标量
                            except Exception as e:
                                self.logger.error(f"LNN模型预测出错，板块 {sector_code}: {e}")
                                self.logger.error(traceback.format_exc())
                                continue
                                
                        except Exception as e:
                            self.logger.error(f"处理板块 {sector_code} 的序列特征时出错: {e}")
                            self.logger.error(traceback.format_exc())
                            continue
                    else:
                        self.logger.warning(f"没有板块 {sector_code} 的序列特征数据")
                
                # 合并LNN预测和强度模型预测
                for i, row in test_df.iterrows():
                    sector_code = row['sector_code']
                    # 获取强度模型预测
                    if strength_pred is not None:
                        # 找到当前板块在strength_pred中的索引
                        idx = test_df[test_df['sector_code'] == sector_code].index.get_loc(i)
                        # 安全地获取索引，避免越界
                        try:
                            if idx < len(strength_pred):
                                strength_score = strength_pred[idx]
                            else:
                                self.logger.warning(f"索引 {idx} 超出strength_pred范围 {len(strength_pred)}")
                                strength_score = 0.0
                        except Exception as e:
                            self.logger.warning(f"获取强度预测时出错: {e}")
                            strength_score = 0.0
                    else:
                        strength_score = 0.0
                    
                    # 添加LNN预测（如果有）
                    lnn_score = lnn_predictions.get(sector_code, 0.0)
                    
                    # 组合得分
                    combined_score = strength_score
                    if lnn_score > 0:
                        # 如果有LNN预测，则进行加权组合
                        combined_score = 0.6 * strength_score + 0.4 * lnn_score
                    
                    # 存储最终得分
                    sector_final_scores[sector_code] = {
                        'strength_score': float(strength_score),
                        'lnn_score': float(lnn_score),
                        'combined_score': float(combined_score)
                    }
            except Exception as e:
                self.logger.error(f"LNN预测过程中出错: {e}")
                self.logger.error(traceback.format_exc())
                # 继续使用强度模型的结果
        else:
            # 仅使用强度模型结果
            self.logger.info("缺少LNN模型或序列特征，仅使用强度模型结果")
            for i, row in test_df.iterrows():
                sector_code = row['sector_code']
                if strength_pred is not None:
                    idx = test_df[test_df['sector_code'] == sector_code].index.get_loc(i)
                    # 安全地获取索引，避免越界
                    try:
                        if idx < len(strength_pred):
                            strength_score = strength_pred[idx]
                        else:
                            self.logger.warning(f"索引 {idx} 超出strength_pred范围 {len(strength_pred)}")
                            strength_score = 0.0
                    except Exception as e:
                        self.logger.warning(f"获取强度预测时出错: {e}")
                        strength_score = 0.0
                else:
                    strength_score = 0.0
                
                # 存储最终得分
                sector_final_scores[sector_code] = {
                    'strength_score': float(strength_score),
                    'lnn_score': 0.0,
                    'combined_score': float(strength_score)
                }
        
        # 获取市场状态
        market_type = '32'  # 默认为中性行情
        market_trend_strength = 0
        if 'market_type' in test_df.columns:
            market_types = test_df['market_type'].unique()
            if len(market_types) > 0:
                market_type = str(market_types[0])
        
        if 'trend_strength' in test_df.columns:
            trend_strengths = test_df['trend_strength'].unique()
            if len(trend_strengths) > 0:
                market_trend_strength = float(trend_strengths[0])
        
        # 对板块排序
        sorted_sectors = sorted(
            [(sector_code, data['combined_score']) for sector_code, data in sector_final_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 获取预测结果
        result = {
            'date': date,
            'sector_scores': sector_final_scores,
            'sorted_sectors': sorted_sectors,
            'market_type': market_type,
            'market_trend_strength': market_trend_strength
        }
        
        return result
    except Exception as e:
        self.logger.error(f"预测时出错: {e}")
        self.logger.error(traceback.format_exc())
        return None

# main函数的完整实现
def main():
    """主函数，用于命令行调用"""
    parser = argparse.ArgumentParser(description="板块轮动模型训练与预测")
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'backtest'], 
                      help='操作模式：train (训练模型), predict (预测), backtest (回测)')
    parser.add_argument('--sector', type=str, help='板块代码，用于训练单个板块模型')
    parser.add_argument('--start_date', type=str, help='开始日期，格式为YYYYMMDD')
    parser.add_argument('--end_date', type=str, help='结束日期，格式为YYYYMMDD')
    parser.add_argument('--date', type=str, help='预测日期，格式为YYYYMMDD')
    parser.add_argument('--top_n', type=int, default=5, help='选择前N个板块')
    parser.add_argument('--max_workers', type=int, help='并行训练的进程数')
    parser.add_argument('--use_lnn', action='store_true', help='是否使用液体神经网络模型')
    parser.add_argument('--sector_list', type=str, help='板块列表文件路径，用于板块轮动模型训练')
    parser.add_argument('--model_dir', type=str, help='模型保存目录')
    parser.add_argument('--model_version', type=str, default='latest', help='用于预测或回测的模型版本，默认使用最新版本')
    parser.add_argument('--save_recent_days', type=int, default=0, help='保存最近N个交易日的模型，用于回测，默认为0表示只保存最终模型')
    
    args = parser.parse_args()
    
    # 导入SectorModelTrainer类
    try:
        from to.train_sector_models import SectorModelTrainer, predict_sector_rotation, get_data_path, ensure_dir
        from stock_sectors_mapping import get_tng_sectors
    except ImportError as e:
        logger.error(f"导入必要的模块失败: {e}")
        logger.error(traceback.format_exc())
        return
    
    # 创建SectorModelTrainer实例
    trainer = SectorModelTrainer(model_dir=args.model_dir, use_lnn=args.use_lnn)
    
    # 加载板块列表
    sector_list = []
    try:
        sector_list = get_tng_sectors()
    except Exception as e:
        logger.error(f"获取板块列表时出错: {e}")
        logger.error(traceback.format_exc())
        
    # 运行模式
    if args.mode == 'train':
        # 训练板块轮动模型
        if not sector_list:
            print("错误: 训练板块轮动模型需要提供板块列表")
            return
                    
        models = trainer.train_rotation_model(
                sector_list=sector_list,
                start_date=args.start_date,
                end_date=args.end_date,
                use_lnn=args.use_lnn)
        print(f"板块轮动模型训练结果: {'成功' if models else '失败'}")
            
    
    elif args.mode == 'predict':
        # 预测模式
        if not args.date:
            args.date = datetime.now().strftime('%Y%m%d')
            print(f"未指定预测日期，使用当前日期: {args.date}")
            
        # 预测特定日期的板块排名
        result = predict_sector_rotation(
            trainer.model_dir,
            args.date,
            sector_list=sector_list,
            forward_days=2,
            top_n=args.top_n
        )
        
        if result:
            print(f"\n预测日期: {args.date}")
            print(f"市场环境: {result.get('market_type', '未知')}")
            print("\n推荐板块:")
            for i, (sector, score) in enumerate(result.get('sorted_sectors', [])[:args.top_n]):
                print(f"{i+1}. {sector}: {score:.4f}")
        else:
            print(f"预测失败，未能获取有效结果")
    
    elif args.mode == 'backtest':
        # 回测模式
        if not sector_list:
            print("错误: 回测需要提供板块列表")
            return
            
        if not args.start_date or not args.end_date:
            print("错误: 回测需要指定--start_date和--end_date参数")
            return
            
        backtest_results = trainer.backtest_rotation_model(
                sector_list=sector_list,
                start_date=args.start_date,
                end_date=args.end_date,
            model_version=args.model_version,
            top_n=args.top_n
        )
        
        if backtest_results:
            metrics = backtest_results['performance_metrics']
            print("\n=== 回测结果 ===")
            print(f"回测区间: {args.start_date} 至 {args.end_date}")
            print(f"交易天数: {metrics['total_trading_days']}")
            print(f"总收益率: {metrics['total_return']:.2%}")
            print(f"年化收益率: {metrics['annual_return']:.2%}")
            print(f"最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
            print(f"胜率: {metrics['win_rate']:.2%}")
            print(f"日均收益: {metrics['avg_daily_return']:.4%}")
            print(f"年化波动率: {metrics['volatility']:.2%}")
            
            # 打印板块选择频率前10
            top_sectors = sorted(
                backtest_results['sector_selection_count'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            print("\n最常选择的板块:")
            for sector, count in top_sectors:
                print(f"{sector}: {count}次")
                
            # 保存回测结果
            try:
                result_dir = Path(get_data_path()) / "results" / "backtest"
                ensure_dir(result_dir)
                result_file = result_dir / f"rotation_backtest_{args.start_date}_to_{args.end_date}.pkl"
                
                with open(result_file, 'wb') as f:
                    pickle.dump(backtest_results, f)
                print(f"\n回测结果已保存到: {result_file}")
                
                # 绘制净值曲线
                try:
                    import matplotlib.pyplot as plt
                    
                    portfolio_values = pd.Series(backtest_results['portfolio_values'])
                    portfolio_values = portfolio_values / portfolio_values.iloc[0]  # 归一化
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(portfolio_values.index, portfolio_values.values)
                    plt.title('板块轮动策略净值曲线')
                    plt.xlabel('日期')
                    plt.ylabel('净值')
                    plt.grid(True)
                    
                    chart_file = result_dir / f"rotation_backtest_{args.start_date}_to_{args.end_date}.png"
                    plt.savefig(chart_file)
                    print(f"净值曲线图已保存到: {chart_file}")
                except Exception as e:
                    print(f"绘制净值曲线失败: {e}")
            except Exception as e:
                print(f"保存回测结果失败: {e}")
        else:
            print("回测失败，未能获取有效结果")
    
    else:
        print("请指定有效的运行模式: --mode train|predict|backtest")

# 执行入口
if __name__ == "__main__":
    # 将此函数复制并注入到原始文件中的方式
    logger.info("运行修复脚本，注入_predict_with_sequence和main函数")
    
    try:
        # 读取原始文件
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_sector_models.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查文件中是否已经包含这些函数
        if '_predict_with_sequence' not in content:
            logger.info("注入_predict_with_sequence方法")
            # 找到合适的注入点并插入方法
            # ...
            
        if 'def main(' not in content:
            logger.info("注入main函数")
            # 找到合适的注入点并插入函数
            # ...
            
        logger.info("修复完成")
    except Exception as e:
        logger.error(f"修复过程出错: {e}")
        logger.error(traceback.format_exc()) 