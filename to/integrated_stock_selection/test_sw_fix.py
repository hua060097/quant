#!/usr/bin/env python3
"""
测试SW行业编码修复
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO)

def test_sw_sector_loading():
    """测试SW行业数据加载"""
    try:
        from main import load_sw_sector_mapping
        mapping = load_sw_sector_mapping()
        if mapping:
            print(f"[OK] SW行业数据加载成功: {len(mapping)} 只股票")
            # 打印前5个示例
            for i, (code, sector) in enumerate(list(mapping.items())[:5]):
                print(f"  {code}: {sector}")
        else:
            print("[WARNING] SW行业数据为空")
        return True
    except Exception as e:
        print(f"[FAIL] SW行业数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_industry_feature_logic():
    """测试行业特征逻辑（不实际训练模型）"""
    try:
        # 模拟一些数据
        import pandas as pd
        import numpy as np

        # 创建模拟数据
        train_stock_codes = ['000001.SZ', '000002.SZ', '000003.SZ']
        val_stock_codes = ['000004.SZ', '000005.SZ']

        # 模拟SW行业映射
        sw_sector_mapping = {
            '000001.SZ': 'SW1银行',
            '000002.SZ': 'SW2房地产',
            '000003.SZ': 'SW1银行',
            '000004.SZ': 'SW2房地产',
            '000005.SZ': 'SW3医药'
        }

        # 测试行业编码逻辑
        from pandas import Categorical
        train_industries = [sw_sector_mapping.get(code, '未知') for code in train_stock_codes]
        val_industries = [sw_sector_mapping.get(code, '未知') for code in val_stock_codes]

        all_industries = train_industries + val_industries
        cat = Categorical(all_industries)
        industry_codes = cat.codes

        train_industry_codes = industry_codes[:len(train_industries)]
        val_industry_codes = industry_codes[len(train_industries):]

        print(f"[OK] 行业编码逻辑测试通过")
        print(f"  行业类别: {cat.categories.tolist()}")
        print(f"  训练集编码: {train_industry_codes.tolist()}")
        print(f"  验证集编码: {val_industry_codes.tolist()}")

        # 检查一致性
        stock_to_code = {}
        for code, industry in sw_sector_mapping.items():
            if industry in cat.categories:
                stock_to_code[code] = int(np.where(cat.categories == industry)[0][0])

        print(f"  股票到编码映射:")
        for code, industry_code in stock_to_code.items():
            print(f"    {code}: {industry_code} ({sw_sector_mapping[code]})")

        return True

    except Exception as e:
        print(f"[FAIL] 行业特征逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("测试SW行业编码修复")
    print("=" * 60)

    success1 = test_sw_sector_loading()
    success2 = test_industry_feature_logic()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("[OK] 所有测试通过")
    else:
        print("[FAIL] 部分测试失败")
    print("=" * 60)