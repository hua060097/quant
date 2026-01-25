#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试collect_stock_sector_change_point_features函数的修改
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "to"))

def create_test_data():
    """创建测试数据"""
    
    # 创建股票特征数据
    dates = ['20240101', '20240102', '20240103']
    stocks = ['000001.SZ', '000002.SZ']
    
    stock_data = []
    for stock in stocks:
        for date in dates:
            stock_data.append({
                'stock_code': stock,
                'date_str': date,
                'close': np.random.uniform(10, 50),
                'pct_chg': np.random.uniform(-5, 5),
                'vol': np.random.uniform(1000000, 10000000),
                'turnover_rate': np.random.uniform(1, 10),
                'amplitude': np.random.uniform(1, 8),
                'is_limit_up': np.random.choice([True, False]),
                'is_limit_down': np.random.choice([True, False]),
                'consecutive_limit_days': np.random.randint(0, 5),
                'is_one_word_limit': np.random.choice([True, False]),
                'is_high_elasticity': np.random.choice([True, False]),
                'market_cap_type': np.random.choice(['large', 'medium', 'small']),
                'rsi14': np.random.uniform(20, 80),
                'ma5_slope': np.random.uniform(-2, 2),
                'momentum_10d': np.random.uniform(-10, 10)
            })
    
    stock_features_df = pd.DataFrame(stock_data)
    
    # 创建板块特征数据
    sectors = ['TGN银行', 'TGN房地产']
    
    sector_data = []
    for sector in sectors:
        for date in dates:
            sector_data.append({
                'sector_code': sector,
                'date': date,
                'pct_chg': np.random.uniform(-3, 3),
                'vol': np.random.uniform(5000000, 50000000),
                'turnover_rate': np.random.uniform(2, 8),
                'amplitude': np.random.uniform(2, 6),
                'limit_up_count': np.random.randint(0, 10),
                'limit_down_count': np.random.randint(0, 5),
                'rsi14': np.random.uniform(30, 70),
                'ma5_slope': np.random.uniform(-1, 1),
                'momentum_10d': np.random.uniform(-5, 5),
                'avg_pct_chg': np.random.uniform(-2, 2),
                'avg_turnover': np.random.uniform(3, 7),
                'avg_amplitude': np.random.uniform(2, 5)
            })
    
    sector_features_df = pd.DataFrame(sector_data)
    
    # 创建变化点数据
    change_points = {
        '000001.SZ': {
            'change_points': [0, 1],
            'dates': ['20240101', '20240102']
        },
        '000002.SZ': {
            'change_points': [1],
            'dates': ['20240102']
        }
    }
    
    return stock_features_df, sector_features_df, change_points

def test_collect_features():
    """测试collect_stock_sector_change_point_features函数"""
    
    try:
        # 导入训练器类
        from to.train_sector_models import SectorModelTrainer
        
        # 创建测试数据
        stock_features_df, sector_features_df, change_points = create_test_data()
        
        print("创建的测试数据:")
        print(f"股票特征数据: {stock_features_df.shape}")
        print(f"板块特征数据: {sector_features_df.shape}")
        print(f"变化点数据: {change_points}")
        print()
        
        # 创建训练器实例
        trainer = SectorModelTrainer()
        
        # 调用修改后的函数
        print("调用collect_stock_sector_change_point_features函数...")
        result_df, relations_df = trainer.collect_stock_sector_change_point_features(
            stock_features_df=stock_features_df,
            sector_features_df=sector_features_df,
            change_points=change_points,
            is_training=True,
            use_cache=False  # 不使用缓存以便测试
        )
        
        print("函数执行完成!")
        print()
        
        # 检查结果
        print("=== 结果分析 ===")
        print(f"result_df形状: {result_df.shape}")
        print(f"relations_df形状: {relations_df.shape}")
        print()
        
        # 检查result_df中的新列
        print("result_df中的列:")
        for col in sorted(result_df.columns):
            print(f"  - {col}")
        print()
        
        # 检查relations_df中的列
        print("relations_df中的列:")
        for col in sorted(relations_df.columns):
            print(f"  - {col}")
        print()
        
        # 检查是否有最相关板块信息
        if 'most_relevant_sector' in result_df.columns:
            relevant_sectors = result_df['most_relevant_sector'].dropna().unique()
            print(f"找到的最相关板块: {relevant_sectors}")
        
        # 检查relations_df中的板块特征
        sector_feature_cols = [col for col in relations_df.columns if col.startswith('sector_')]
        print(f"relations_df中的板块特征列数量: {len(sector_feature_cols)}")
        if sector_feature_cols:
            print("板块特征列示例:")
            for col in sector_feature_cols[:5]:  # 只显示前5个
                print(f"  - {col}")
        
        # 检查股票特征是否正确集成
        stock_feature_cols = [col for col in result_df.columns if col.startswith('stock_')]
        print(f"result_df中的股票特征列数量: {len(stock_feature_cols)}")
        
        print("\n=== 测试成功! ===")
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("开始测试collect_stock_sector_change_point_features函数的修改...")
    print("=" * 60)
    
    success = test_collect_features()
    
    if success:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 测试失败!")
