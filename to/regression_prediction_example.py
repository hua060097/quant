"""
回归预测示例

此脚本展示如何使用回归模型和校准器进行板块轮动预测。
主要包括以下步骤：
1. 加载和准备特征数据
2. 训练回归模型
3. 校准预测结果
4. 评估模型性能

使用方法:
python -m to.regression_prediction_example --sector_list TGNAI语料 TGN算力租赁 TGN新能源汽车 --start_date 20230101 --end_date 20230630
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import datetime
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import traceback

# 导入自定义模块
from to.train_sector_models import SectorModelTrainer, get_project_root
from to.regression_model_calibrator import EnsembleCalibrator, ReturnCalibrator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(get_project_root(), 'to', 'log', 'regression_prediction.log'))
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='回归预测示例')
    parser.add_argument('--sector_list', nargs='+', help='板块列表', required=False)
    parser.add_argument('--start_date', type=str, help='开始日期，格式YYYYMMDD', required=False)
    parser.add_argument('--end_date', type=str, help='结束日期，格式YYYYMMDD', required=False)
    parser.add_argument('--model_name', type=str, default='regression_sector_model', help='模型名称')
    parser.add_argument('--train_only', action='store_true', help='仅训练模型，不进行回测')
    parser.add_argument('--backtest_only', action='store_true', help='仅进行回测，不训练模型')
    parser.add_argument('--use_calibration', action='store_true', help='使用校准器校准预测结果')
    
    return parser.parse_args()

def format_date(date_str):
    """格式化日期字符串为datetime"""
    if date_str is None:
        return None
    
    if len(date_str) == 8:  # YYYYMMDD
        return datetime.datetime.strptime(date_str, '%Y%m%d')
    else:
        raise ValueError(f"日期格式错误: {date_str}, 应为YYYYMMDD格式")

def plot_prediction_results(results, title='回归预测结果', output_path=None):
    """绘制预测结果图表"""
    plt.figure(figsize=(12, 8))
    
    # 提取结果
    dates = [result['date'] for result in results if 'metrics' in result]
    
    if not dates:
        logger.warning("没有可用的预测结果数据进行绘图")
        return
    
    # 创建指标列表
    metrics = ['mae', 'rmse', 'r2', 'direction_accuracy', 'top5_avg_return']
    plot_data = {metric: [] for metric in metrics}
    
    # 收集数据
    for result in results:
        if 'metrics' in result:
            for metric in metrics:
                if metric in result['metrics']:
                    plot_data[metric].append(result['metrics'][metric])
                else:
                    plot_data[metric].append(0)
    
    # 绘制图表
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 5 * len(metrics)))
    
    for i, (metric, values) in enumerate(plot_data.items()):
        ax = axes[i]
        ax.plot(dates, values, 'o-', linewidth=2)
        ax.set_title(f'{metric.upper()} 随时间变化')
        ax.set_ylabel(metric)
        ax.set_xlabel('日期')
        ax.grid(True)
        
        # 添加均值线
        mean_value = np.mean(values)
        ax.axhline(y=mean_value, color='r', linestyle='--', label=f'均值: {mean_value:.4f}')
        ax.legend()
    
    plt.tight_layout()
    
    # 保存图表
    if output_path:
        plt.savefig(output_path)
        logger.info(f"预测结果图表已保存到: {output_path}")
    
    plt.close()

def evaluate_calibration_effect(results):
    """评估校准器效果"""
    if not results or 'calibration_comparison' not in results[0]:
        logger.warning("没有可用的校准效果数据")
        return
    
    # 提取校准前后的指标
    before_calibration = []
    after_calibration = []
    
    for result in results:
        if 'calibration_comparison' in result:
            before_calibration.append(result['calibration_comparison']['before'])
            after_calibration.append(result['calibration_comparison']['after'])
    
    if not before_calibration:
        logger.warning("没有可用的校准效果数据")
        return
        
    # 计算平均指标
    metrics = ['mae', 'rmse', 'r2']
    before_avg = {metric: np.mean([d[metric] for d in before_calibration]) for metric in metrics}
    after_avg = {metric: np.mean([d[metric] for d in after_calibration]) for metric in metrics}
    
    # 计算改进百分比
    improvements = {}
    for metric in metrics:
        if before_avg[metric] != 0:
            if metric == 'r2':  # R2为越高越好
                improvements[metric] = (after_avg[metric] - before_avg[metric]) / abs(before_avg[metric]) * 100
            else:  # MAE, RMSE越低越好
                improvements[metric] = (before_avg[metric] - after_avg[metric]) / before_avg[metric] * 100
    
    # 输出结果
    logger.info("校准效果评估:")
    logger.info(f"校准前平均指标: {before_avg}")
    logger.info(f"校准后平均指标: {after_avg}")
    logger.info(f"改进百分比: {improvements}")
    
    # 返回评估结果
    return {
        'before': before_avg,
        'after': after_avg,
        'improvements': improvements
    }

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 默认参数
        if args.sector_list is None:
            args.sector_list = ["TGN新能源汽车", "TGN半导体", "TGN人工智能", "TGN算力租赁", "TGN光伏概念"]
            
        if args.start_date is None:
            args.start_date = "20230101"
            
        if args.end_date is None:
            args.end_date = "20230630"
        
        # 格式化日期
        start_date = format_date(args.start_date)
        end_date = format_date(args.end_date)
        
        logger.info(f"启动回归预测示例")
        logger.info(f"参数: 板块列表={args.sector_list}, 开始日期={start_date}, 结束日期={end_date}")
        
        # 创建模型训练器
        trainer = SectorModelTrainer()
        
        # 训练模型
        if not args.backtest_only:
            logger.info("开始训练回归模型...")
            model_package = trainer.train_rotation_model(
                sector_list=args.sector_list,
                start_date=start_date,
                end_date=end_date,
                model_name=args.model_name,
                use_lnn=True
            )
            
            if model_package is None:
                logger.error("模型训练失败")
                return
                
            logger.info(f"模型训练完成: {args.model_name}")
        
        # 回测
        if not args.train_only:
            logger.info("开始回测...")
            
            # 加载最新训练的模型
            model_dir = Path(get_project_root()) / "to" / "models" / "sector_models"
            model_path = model_dir / f"{args.model_name}.pkl"
            
            if not model_path.exists() and not args.backtest_only:
                logger.error(f"模型文件不存在: {model_path}")
                return
                
            # 创建回测日期列表
            # 每隔7天进行一次回测，最多10个回测点
            from pandas.tseries.offsets import BDay
            
            calendar_df = pd.read_csv(os.path.join(get_project_root(), "to", "local_data", "calendar", "trading_days.csv"))
            calendar_df['date'] = pd.to_datetime(calendar_df['date'])
            
            # 筛选日期范围内的交易日
            trading_days = calendar_df[(calendar_df['date'] >= start_date) & (calendar_df['date'] <= end_date)]['date'].tolist()
            
            # 选择回测点
            backtest_dates = trading_days[::7][:10]  # 每7天一个点，最多10个
            
            if not backtest_dates:
                logger.error("回测日期为空")
                return
                
            logger.info(f"回测日期: {[d.strftime('%Y-%m-%d') for d in backtest_dates]}")
            
            # 加载特征数据
            logger.info("加载特征数据...")
            features_df = trainer._batch_load_features(args.sector_list, start_date, end_date)
            
            if features_df is None or features_df.empty:
                logger.error("特征数据为空")
                return
                
            logger.info(f"成功加载特征数据: {features_df.shape[0]} 行, {features_df.shape[1]} 列")
            
            # 为每个回测日期创建预测任务
            backtest_results = []
            
            for test_date in backtest_dates:
                logger.info(f"回测日期: {test_date}")
                
                # 分割训练集和测试集
                train_df = features_df[features_df['date'] < test_date].copy()
                test_df = features_df[features_df['date'] == test_date].copy()
                
                if train_df.empty or test_df.empty:
                    logger.warning(f"日期 {test_date} 的训练集或测试集为空，跳过")
                    continue
                
                # 创建标签
                train_df = trainer._create_return_prediction_labels(
                    train_df, 
                    future_return_col='future_return_2d'
                )
                
                test_df = trainer._create_return_prediction_labels(
                    test_df, 
                    future_return_col='future_return_2d'
                )
                
                # 删除没有future_return_2d的行
                train_df = train_df.dropna(subset=['future_return_2d'])
                test_df = test_df.dropna(subset=['future_return_2d'])
                
                if train_df.empty or test_df.empty:
                    logger.warning(f"处理标签后，日期 {test_date} 的训练集或测试集为空，跳过")
                    continue
                
                # 应用特征工程
                train_df = trainer._apply_feature_pipeline(train_df, mode='train')
                test_df = trainer._apply_feature_pipeline(test_df, mode='test')
                
                # 创建序列特征
                seq_features = trainer._create_sequence_features(
                    features_df, 
                    args.sector_list, 
                    end_date=test_date,
                    seq_length=10
                )
                
                # 训练当前日期前的模型
                current_model = trainer._train_rolling_day_model(
                    train_df,
                    test_date - BDay(1),  # 使用前一个交易日
                    training_window_days=252,  # 使用约一年的训练窗口
                    model_name=f"{args.model_name}_{test_date.strftime('%Y%m%d')}"
                )
                
                if current_model is None:
                    logger.warning(f"为日期 {test_date} 训练模型失败，跳过")
                    continue
                
                # 使用校准器
                if args.use_calibration:
                    logger.info("创建并应用校准器...")
                    
                    # 在训练集上拟合校准器
                    try:
                        # 准备数据
                        valid_feature_cols = [col for col in current_model['feature_columns'] if col in train_df.columns]
                        X_train = train_df[valid_feature_cols].fillna(0).values
                        y_train = train_df['future_return_2d'].values
                        
                        # 使用当前模型生成训练集预测
                        if current_model['scaler'] is not None:
                            X_train_scaled = current_model['scaler'].transform(X_train)
                        else:
                            X_train_scaled = X_train
                            
                        train_pred = current_model['model'].predict(X_train_scaled)
                        
                        # 创建并拟合校准器
                        calibrator = EnsembleCalibrator()
                        calibrator.fit(train_pred, y_train)
                        
                        # 将校准器添加到模型包中
                        current_model['calibrator'] = calibrator
                        
                        # 保存校准器
                        calibrator_dir = model_dir / "calibrators"
                        os.makedirs(calibrator_dir, exist_ok=True)
                        calibrator_path = calibrator_dir / f"{args.model_name}_{test_date.strftime('%Y%m%d')}_calibrator"
                        calibrator.save(str(calibrator_path))
                        
                        logger.info(f"校准器已保存到: {calibrator_path}")
                    except Exception as e:
                        logger.error(f"创建校准器时出错: {e}")
                        logger.error(traceback.format_exc())
                
                # 使用模型预测
                result = trainer._predict_with_sequence(current_model, test_df, seq_features, test_date)
                
                if result is None:
                    logger.warning(f"为日期 {test_date} 预测失败，跳过")
                    continue
                
                # 添加回测日期
                result['backtest_date'] = test_date
                
                # 添加到回测结果
                backtest_results.append(result)
                
                logger.info(f"日期 {test_date} 的回测完成")
                
                # 输出预测结果
                if 'metrics' in result:
                    logger.info(f"预测指标: MAE={result['metrics'].get('mae', 0):.6f}, "
                               f"RMSE={result['metrics'].get('rmse', 0):.6f}, "
                               f"R2={result['metrics'].get('r2', 0):.4f}, "
                               f"方向准确率={result['metrics'].get('direction_accuracy', 0):.4f}")
                
                # 输出预测的top5板块
                top_sectors = result.get('top_sectors', [])
                logger.info(f"预测top5板块: {top_sectors}")
            
            # 保存回测结果
            results_dir = Path(get_project_root()) / "to" / "results" / "backtest"
            os.makedirs(results_dir, exist_ok=True)
            
            results_path = results_dir / f"{args.model_name}_backtest_results.json"
            
            # 将datetime对象转换为字符串
            for result in backtest_results:
                if isinstance(result['date'], (datetime.datetime, pd.Timestamp)):
                    result['date'] = result['date'].strftime('%Y-%m-%d')
                    
                if 'backtest_date' in result and isinstance(result['backtest_date'], (datetime.datetime, pd.Timestamp)):
                    result['backtest_date'] = result['backtest_date'].strftime('%Y-%m-%d')
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(backtest_results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"回测结果已保存到: {results_path}")
            
            # 绘制预测结果图表
            plot_path = results_dir / f"{args.model_name}_prediction_results.png"
            plot_prediction_results(backtest_results, title='回归预测结果', output_path=plot_path)
            
            # 评估校准效果
            if args.use_calibration:
                calibration_result = evaluate_calibration_effect(backtest_results)
                if calibration_result:
                    # 保存校准效果评估结果
                    calibration_results_path = results_dir / f"{args.model_name}_calibration_results.json"
                    with open(calibration_results_path, 'w', encoding='utf-8') as f:
                        json.dump(calibration_result, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"校准效果评估结果已保存到: {calibration_results_path}")
            
            logger.info("回测完成")
    
    except Exception as e:
        logger.error(f"运行出错: {e}")
        logger.error(traceback.format_exc())
        
if __name__ == "__main__":
    main() 