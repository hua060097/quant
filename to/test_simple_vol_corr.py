#!/usr/bin/env python3
"""
简单测试量价相关系数修复
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_vol_corr_fix():
    """测试量价相关系数修复"""
    try:
        from generate_daily_features import add_advanced_ml_features
        
        # 创建足够长的测试数据（>=60行）
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 70,
            'date': pd.date_range('2024-01-01', periods=70),
            'open': np.random.uniform(10, 20, 70),
            'high': np.random.uniform(15, 25, 70),
            'low': np.random.uniform(8, 15, 70),
            'close': np.random.uniform(10, 20, 70),
            'volume': np.random.randint(1000, 100000, 70),
            'amount': np.random.randint(100000, 10000000, 70),
            'pct_chg': np.random.uniform(-5, 5, 70),
            'total_mv': np.random.randint(1000000, 100000000, 70),
            'circ_mv': np.random.randint(800000, 80000000, 70),
            'turnover_rate': np.random.uniform(0, 10, 70)
        })
        
        print('测试数据准备完成:', test_data.shape)
        
        # 直接测试 add_advanced_ml_features 函数
        print('开始计算高级ML特征...')
        result = add_advanced_ml_features(test_data)
        
        print('检查结果:')
        print(f'输入数据: {test_data.shape}')
        print(f'输出数据: {result.shape}')
        
        if 'vol_price_corr_20d' in result.columns:
            print('✅ vol_price_corr_20d 特征已生成')
            dtype = result['vol_price_corr_20d'].dtype
            print(f'  数据类型: {dtype}')
            non_zero_count = (result['vol_price_corr_20d'] != 0).sum()
            print(f'  非零值: {non_zero_count}')
            sample_values = result['vol_price_corr_20d'].head().tolist()
            print(f'  样本值: {sample_values}')
            return True
        else:
            print('❌ vol_price_corr_20d 特征未生成')
            print('可用特征:')
            for col in result.columns:
                print(f'  {col}')
            return False
            
    except Exception as e:
        print(f'❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vol_corr_fix()
    if success:
        print('\n🎉 测试成功！')
    else:
        print('\n💥 测试失败！')
