#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块轮动模型测试脚本
"""

import os
import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import time

# 将项目根目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 导入项目模块
from to.train_models import _load_sector_features, _load_market_features
from to.train_models import _prepare_sector_rotation_dataset, train_sector_rotation_model
from to.train_models import generate_stock_signals_from_sector_prediction

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sector_features_loading():
    """测试板块特征加载功能"""
    # 测试日期
    start_date = '20230101'
    end_date = '20230110'
    
    logger.info(f"测试加载板块特征数据，日期范围: {start_date} 到 {end_date}")
    sector_features = _load_sector_features(start_date, end_date)
    
    if sector_features is None or sector_features.empty:
        logger.error("无法加载板块特征数据，测试失败")
        return False
    
    # 显示基本信息
    logger.info(f"成功加载板块特征数据:")
    logger.info(f"- 数据形状: {sector_features.shape}")
    logger.info(f"- 日期范围: {sector_features['date'].min()} 到 {sector_features['date'].max()}")
    logger.info(f"- 包含的板块数量: {sector_features['sector_name'].nunique()}")
    logger.info(f"- 板块列表(前5个): {sector_features['sector_name'].unique()[:5]}")
    logger.info(f"- 特征列表: {list(sector_features.columns)}")
    
    # 展示样本数据
    logger.info(f"\n样本数据(前3行):\n{sector_features.head(3)}")
    
    return True

def test_prepare_dataset():
    """测试数据集准备功能"""
    # 测试日期
    start_date = '20230101'
    end_date = '20230110'
    target_date = '20230105'
    
    logger.info(f"测试准备数据集，目标日期: {target_date}")
    
    # 加载特征数据
    sector_features = _load_sector_features(start_date, end_date)
    market_features = _load_market_features(start_date, end_date)
    
    if sector_features is None or sector_features.empty:
        logger.error("无法加载板块特征数据，测试失败")
        return False
    
    # 准备数据集
    X, y, sectors, feature_names = _prepare_sector_rotation_dataset(
        sector_features, market_features, target_date, 
        lookback_short=5, lookback_long=10, forward_days=5
    )
    
    if X is None:
        logger.error("准备数据集失败")
        return False
    
    logger.info(f"成功准备数据集:")
    logger.info(f"- 特征矩阵形状: {X.shape}")
    logger.info(f"- 标签向量形状: {y.shape}")
    logger.info(f"- 板块数量: {len(sectors)}")
    logger.info(f"- 特征数量: {len(feature_names)}")
    logger.info(f"- 正样本比例: {np.mean(y):.2f}")
    
    return True

def test_sector_rotation_model():
    """测试板块轮动模型训练"""
    # 测试日期
    date = '20230105'
    
    logger.info(f"测试板块轮动模型训练，日期: {date}")
    
    # 创建模型保存目录
    model_dir = os.path.join('to', 'local_data', 'models', 'test_sector_rotation')
    os.makedirs(model_dir, exist_ok=True)
    
    # 训练模型
    model_components = train_sector_rotation_model(
        date=date,
        lookback_short=5,  # 使用较小的值加速测试
        lookback_long=10,  # 使用较小的值加速测试
        feature_dir=None,
        model_dir=model_dir,
        use_lnn=False,  # 默认不使用LNN
        forward_days=5
    )
    
    if model_components is None:
        logger.error("模型训练失败")
        return False
    
    # 检查模型组件
    logger.info(f"模型训练成功:")
    logger.info(f"- 模型类型: {model_components.get('model_type')}")
    logger.info(f"- 板块数量: {len(model_components.get('sectors', []))}")
    
    return True

def test_stock_signals_generation():
    """测试股票信号生成"""
    # 测试日期
    date = '20230105'
    
    logger.info(f"测试股票信号生成，日期: {date}")
    
    # 创建模型保存目录
    model_dir = os.path.join('to', 'local_data', 'models', 'test_sector_rotation')
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建信号保存目录
    signal_dir = os.path.join('to', 'signals', 'test')
    os.makedirs(signal_dir, exist_ok=True)
    
    # 训练模型
    model_components = train_sector_rotation_model(
        date=date,
        lookback_short=5,
        lookback_long=10,
        feature_dir=None,
        model_dir=model_dir,
        use_lnn=False,
        forward_days=5
    )
    
    if model_components is None:
        logger.error("模型训练失败，无法生成信号")
        return False
    
    # 生成股票信号
    selected_stocks, weights = generate_stock_signals_from_sector_prediction(
        date=date,
        model_components=model_components,
        top_n_sectors=3,
        stocks_per_sector=2,
        feature_dir=None,
        signal_dir=signal_dir
    )
    
    if selected_stocks is None:
        logger.error("信号生成失败")
        return False
    
    logger.info(f"信号生成成功:")
    logger.info(f"- 选出的股票数量: {len(selected_stocks)}")
    logger.info(f"- 选出的股票列表: {selected_stocks}")
    
    # 打印权重信息
    if weights:
        for stock, weight in weights.items():
            logger.info(f"- {stock}: 权重 {weight:.4f}")
    
    return True

def run_all_tests():
    """运行所有测试"""
    tests = [
        ("板块特征加载测试", test_sector_features_loading),
        ("数据集准备测试", test_prepare_dataset),
        ("板块轮动模型训练测试", test_sector_rotation_model),
        ("股票信号生成测试", test_stock_signals_generation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'-'*50}")
        logger.info(f"开始测试: {test_name}")
        
        start_time = time.time()
        try:
            success = test_func()
            results[test_name] = "通过" if success else "失败"
        except Exception as e:
            logger.error(f"测试出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[test_name] = "错误"
            
        elapsed = time.time() - start_time
        logger.info(f"测试完成: {test_name}, 耗时: {elapsed:.2f}秒, 结果: {results[test_name]}")
    
    # 打印汇总结果
    logger.info(f"\n{'-'*50}")
    logger.info("测试结果汇总:")
    for test_name, result in results.items():
        logger.info(f"- {test_name}: {result}")
    
    # 计算通过率
    passed = sum(1 for result in results.values() if result == "通过")
    logger.info(f"通过率: {passed}/{len(tests)} ({passed/len(tests)*100:.1f}%)")
    
    return all(result == "通过" for result in results.values())

if __name__ == "__main__":
    logger.info("开始板块轮动功能测试...")
    
    # 运行所有测试
    success = run_all_tests()
    
    exit_code = 0 if success else 1
    logger.info(f"测试{'全部通过' if success else '有失败项'}，退出码: {exit_code}")
    
    sys.exit(exit_code) 