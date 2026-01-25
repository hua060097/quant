#!/usr/bin/env python3
"""
测试SW申万行业数据加载和行业中性化逻辑
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factor_system.factor_lab.definitions.momentum_reversal import (
    _pct_chg_0d, _pct_chg_1d, _pct_chg_2d,
    _return_3d, _return_5d, _return_10d, _return_20d,
    _momentum_reversal, _trend_acceleration, _reversal_strength,
    _low_volatility_reversal
)

def test_factor_functions():
    """测试因子函数是否能正确导入和调用"""
    print("[OK] 测试因子函数导入...")

    # 模拟数据
    import pandas as pd
    import numpy as np

    # 创建模拟数据
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    stock_codes = ['000001.SZ', '000002.SZ', '000003.SZ']

    data = []
    for date in dates:
        for stock in stock_codes:
            data.append({
                'stock_code': stock,
                'date': date,
                'pct_chg': np.random.uniform(-5, 5),
                'close': np.random.uniform(10, 100),
                'high': np.random.uniform(15, 110),
                'low': np.random.uniform(5, 90),
                'volume': np.random.uniform(10000, 1000000),
                'atr14': np.random.uniform(1, 5)
            })

    df = pd.DataFrame(data)

    # 模拟context
    class MockContext:
        def __init__(self):
            # 创建模拟的SW申万行业映射
            self.extra = {
                'stock_to_sw_sector': pd.Series({
                    '000001.SZ': 'SW1金融',
                    '000002.SZ': 'SW2地产',
                    '000003.SZ': 'SW3工业'
                }),
                'stock_to_sector': pd.Series({
                    '000001.SZ': '金融',
                    '000002.SZ': '地产',
                    '000003.SZ': '工业'
                })
            }

    context = MockContext()

    # 测试每个因子函数
    test_functions = [
        ('_pct_chg_0d', _pct_chg_0d),
        ('_pct_chg_1d', _pct_chg_1d),
        ('_pct_chg_2d', _pct_chg_2d),
        ('_return_3d', _return_3d),
        ('_return_5d', _return_5d),
        ('_return_10d', _return_10d),
        ('_return_20d', _return_20d),
        ('_momentum_reversal', _momentum_reversal),
        ('_trend_acceleration', _trend_acceleration),
        ('_reversal_strength', _reversal_strength),
        ('_low_volatility_reversal', _low_volatility_reversal)
    ]

    for func_name, func in test_functions:
        try:
            result = func(df, context)
            print(f"[OK] {func_name}: 成功，结果形状 {result.shape}")
            # 检查结果是否有NaN
            if result.isna().any():
                print(f"  [WARN] {func_name}: 结果包含NaN值")
        except Exception as e:
            print(f"[FAIL] {func_name}: 失败 - {e}")

    # 测试没有行业数据的情况
    print("\n[OK] 测试没有行业数据的情况...")
    context_no_sector = MockContext()
    context_no_sector.extra = {}  # 没有行业映射

    try:
        result = _pct_chg_0d(df, context_no_sector)
        print(f"[OK] _pct_chg_0d (无行业数据): 成功，结果形状 {result.shape}")
    except Exception as e:
        print(f"[FAIL] _pct_chg_0d (无行业数据): 失败 - {e}")

    print("\n[OK] 所有测试完成！")

if __name__ == "__main__":
    test_factor_functions()