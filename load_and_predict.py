#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演示如何加载模型并进行预测的简单脚本
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型目录
MODEL_DIR = "models"

def get_available_sectors():
    """获取所有可用的板块名称"""
    if not os.path.exists(MODEL_DIR):
        logger.error(f"模型目录不存在: {MODEL_DIR}")
        return []
    
    sectors = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
    return sectors

def get_available_dates(sector_name):
    """获取指定板块的所有可用日期"""
    sector_dir = os.path.join(MODEL_DIR, sector_name)
    if not os.path.exists(sector_dir):
        logger.error(f"板块目录不存在: {sector_dir}")
        return []
    
    dates = [f.split('.')[0] for f in os.listdir(sector_dir) if f.endswith('.pkl')]
    return sorted(dates)

def load_model(sector_name, date):
    """加载指定板块和日期的模型"""
    model_path = os.path.join(MODEL_DIR, sector_name, f"{date}.pkl")
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None
    
    logger.info(f"加载模型: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        return None

def create_test_features(feature_columns, stock_codes=None, num_stocks=10):
    """创建测试特征数据"""
    # 如果没有提供股票代码，则生成随机股票代码
    if stock_codes is None:
        stock_codes = [f"00000{i}.SZ" for i in range(1, num_stocks+1)]
    else:
        num_stocks = len(stock_codes)
    
    # 创建随机特征值
    data = np.random.randn(num_stocks, len(feature_columns))
    
    # 创建DataFrame
    df = pd.DataFrame(data, index=stock_codes, columns=feature_columns)
    
    return df

def predict_with_model(model, features):
    """使用模型进行预测"""
    logger.info(f"使用模型进行预测，特征形状: {features.shape}")
    
    # 获取短期和长期模型
    model_short = model['model_short']
    model_long = model['model_long']
    
    # 提取特征值
    X = features.values
    
    # 预测
    short_probs = model_short.predict_proba(X)[:, 1]
    long_probs = model_long.predict_proba(X)[:, 1]
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'stock_code': features.index,
        'short_prob': short_probs,
        'long_prob': long_probs
    })
    
    # 按短期概率排序
    results = results.sort_values('short_prob', ascending=False)
    
    return results

def main():
    """主函数"""
    # 获取所有可用板块
    sectors = get_available_sectors()
    logger.info(f"可用板块: {sectors}")
    
    if not sectors:
        return
    
    # 选择第一个板块
    sector_name = sectors[0]
    logger.info(f"选择板块: {sector_name}")
    
    # 获取该板块的所有日期
    dates = get_available_dates(sector_name)
    logger.info(f"可用日期: {dates}")
    
    if not dates:
        return
    
    # 选择最新的日期
    date = dates[-1]
    logger.info(f"选择日期: {date}")
    
    # 加载模型
    model = load_model(sector_name, date)
    if model is None:
        return
    
    # 打印模型信息
    logger.info(f"板块名称: {model['sector_name']}")
    logger.info(f"日期: {model['date']}")
    logger.info(f"特征列: {model['feature_columns']}")
    logger.info(f"短期模型分数: {model['short_score']}")
    logger.info(f"长期模型分数: {model['long_score']}")
    logger.info(f"样本数量: {model['sample_count']}")
    logger.info(f"是否为模拟数据: {model['is_simulated']}")
    
    # 创建测试特征
    features = create_test_features(model['feature_columns'], num_stocks=20)
    
    # 使用模型进行预测
    results = predict_with_model(model, features)
    
    # 打印预测结果
    logger.info("预测结果(排名前5):")
    print(results.head(5))
    
    logger.info("预测结果(排名后5):")
    print(results.tail(5))

if __name__ == "__main__":
    main() 