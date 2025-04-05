#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sector_model(sector_name, date, model_dir=None):
    """加载板块模型
    
    Args:
        sector_name: 板块名称
        date: 日期字符串，格式YYYYMMDD
        model_dir: 模型目录，如果为None则使用默认路径
        
    Returns:
        (model, features_list)元组，如果加载失败则返回(None, None)
    """
    # 设置默认模型目录
    if model_dir is None:
        model_dir = Path(os.getcwd()) / "to" / "local_data" / "models" / "sector"
    else:
        model_dir = Path(model_dir)
    
    # 构建模型文件路径
    sector_model_dir = model_dir / sector_name
    model_file = sector_model_dir / f"{date}.pkl"
    
    # 检查文件是否存在
    if not model_file.exists():
        logger.warning(f"模型文件不存在: {model_file}")
        return None, None
    
    try:
        # 加载模型
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        # 提取模型和特征列表
        model = model_data.get('model')
        features = model_data.get('features', [])
        
        logger.info(f"成功加载板块 {sector_name} 的模型，日期: {date}")
        return model, features
    
    except Exception as e:
        logger.error(f"加载板块 {sector_name} 的模型时出错: {str(e)}")
        return None, None

def load_sector_features(sector_name, start_date, end_date, data_dir=None):
    """加载板块特征数据
    
    Args:
        sector_name: 板块名称
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        data_dir: 数据目录，如果为None则使用默认路径
        
    Returns:
        板块特征的DataFrame，如果加载失败则返回空DataFrame
    """
    # 添加当前目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    try:
        # 导入特征加载函数
        from sector_loader import load_sector_features as _load_sector_features
        return _load_sector_features(sector_name, data_dir, start_date, end_date)
    
    except Exception as e:
        logger.error(f"加载板块 {sector_name} 的特征数据时出错: {str(e)}")
        return pd.DataFrame()

def predict_sector_trend(model, features, feature_data):
    """预测板块趋势
    
    Args:
        model: 板块模型
        features: 特征列表
        feature_data: 特征数据DataFrame
        
    Returns:
        预测结果DataFrame，包含日期和预测概率
    """
    if model is None or feature_data.empty:
        logger.warning("模型或特征数据为空，无法进行预测")
        return pd.DataFrame()
    
    # 确保特征数据包含所有必要的特征
    missing_features = [f for f in features if f not in feature_data.columns]
    if missing_features:
        logger.warning(f"特征数据缺少必要的特征: {missing_features}")
        return pd.DataFrame()
    
    # 准备预测数据
    X = feature_data[features].values.astype(np.float32)
    
    # 进行预测
    try:
        # 获取预测概率
        probs = model.predict_proba(X)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'date': feature_data.index,
            'prob': probs[:, 1] if len(probs.shape) > 1 else probs  # 取第二列作为正类概率
        })
        
        results.set_index('date', inplace=True)
        
        logger.info(f"成功预测 {len(results)} 个日期的趋势")
        return results
    
    except Exception as e:
        logger.error(f"预测板块趋势时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def plot_prediction_results(sector_name, predictions):
    """绘制预测结果
    
    Args:
        sector_name: 板块名称
        predictions: 预测结果DataFrame
    """
    if predictions.empty:
        logger.warning("预测结果为空，无法绘制图表")
        return
    
    try:
        plt.figure(figsize=(12, 6))
        
        # 绘制预测概率线
        plt.plot(predictions.index, predictions['prob'], 'b-', label='预测上涨概率')
        
        # 绘制阈值线
        plt.axhline(y=0.5, color='r', linestyle='--', label='决策阈值')
        
        # 设置图表属性
        plt.title(f'板块 {sector_name} 上涨趋势预测')
        plt.xlabel('日期')
        plt.ylabel('上涨概率')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # 保存图表
        save_dir = Path(os.getcwd()) / "to" / "results"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{sector_name}_prediction.png"
        plt.savefig(save_path)
        
        logger.info(f"预测结果图表已保存到: {save_path}")
        
        # 显示图表
        plt.close()
    
    except Exception as e:
        logger.error(f"绘制预测结果图表时出错: {str(e)}")

def backtest_sector_model(sector_name, start_date, end_date, model_dir=None, data_dir=None):
    """回测板块模型
    
    Args:
        sector_name: 板块名称
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        model_dir: 模型目录
        data_dir: 数据目录
        
    Returns:
        回测结果字典
    """
    # 加载特征数据
    feature_data = load_sector_features(sector_name, start_date, end_date, data_dir)
    
    if feature_data.empty:
        logger.warning(f"没有找到板块 {sector_name} 在日期范围 {start_date}-{end_date} 的特征数据")
        return {}
    
    # 加载第一个日期的模型
    model, features = load_sector_model(sector_name, start_date, model_dir)
    
    if model is None:
        logger.warning(f"无法加载板块 {sector_name} 在日期 {start_date} 的模型")
        return {}
    
    # 进行预测
    predictions = predict_sector_trend(model, features, feature_data)
    
    if predictions.empty:
        logger.warning(f"预测结果为空")
        return {}
    
    # 绘制预测结果
    plot_prediction_results(sector_name, predictions)
    
    # 计算预测准确率（这里假设我们有真实标签）
    # 由于我们没有真实标签，这里只是示例性地计算一些统计指标
    stats = {
        'total_days': len(predictions),
        'up_days': (predictions['prob'] > 0.5).sum(),
        'down_days': (predictions['prob'] <= 0.5).sum(),
        'avg_prob': predictions['prob'].mean(),
        'max_prob': predictions['prob'].max(),
        'min_prob': predictions['prob'].min()
    }
    
    logger.info(f"回测结果统计:")
    logger.info(f"- 总交易日: {stats['total_days']}") 
    logger.info(f"- 预测上涨天数: {stats['up_days']} ({stats['up_days']/stats['total_days']*100:.1f}%)")
    logger.info(f"- 预测下跌天数: {stats['down_days']} ({stats['down_days']/stats['total_days']*100:.1f}%)")
    logger.info(f"- 平均上涨概率: {stats['avg_prob']:.4f}")
    
    return {'predictions': predictions, 'stats': stats}

def parse_args():
    """解析命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试板块模型')
    
    parser.add_argument('--sector', type=str, required=True, help='板块名称')
    parser.add_argument('--start_date', type=str, required=True, help='开始日期，格式YYYYMMDD')
    parser.add_argument('--end_date', type=str, required=True, help='结束日期，格式YYYYMMDD')
    parser.add_argument('--model_dir', type=str, help='模型目录')
    parser.add_argument('--data_dir', type=str, help='数据目录')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 回测板块模型
    results = backtest_sector_model(
        sector_name=args.sector,
        start_date=args.start_date,
        end_date=args.end_date,
        model_dir=args.model_dir,
        data_dir=args.data_dir
    )
    
    if not results:
        logger.error("回测失败")
        return
    
    logger.info("回测完成")

if __name__ == "__main__":
    main() 