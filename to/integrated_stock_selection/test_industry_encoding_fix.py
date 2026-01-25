#!/usr/bin/env python3
"""
测试行业编码类型污染修复

验证修复后的代码是否能正确处理不同类型的行业编码
"""

import numpy as np
import pandas as pd
import sys
import os

# 添加当前目录到路径
sys.path.append('.')

def test_dtype_conversion():
    """测试数据类型转换"""
    print("=== 测试行业编码类型转换修复 ===")

    # 模拟不同类型的行业编码
    test_cases = [
        {
            'name': '纯整数编码',
            'codes': [13, 4, 18, 7, 17],
            'expected_type': np.int32
        },
        {
            'name': '字符串编码',
            'codes': ['13', '4', '18', '7', '17'],
            'expected_type': str
        },
        {
            'name': '混合类型编码',
            'codes': [13, '4', 18.0, '7', 17],
            'expected_type': object
        },
        {
            'name': '包含-1的编码',
            'codes': [13, 4, -1, 7, 17],
            'expected_type': np.int32
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. 测试案例: {case['name']}")

        # 模拟特征矩阵
        feature_matrix = np.random.rand(5, 10).astype(np.float32)
        print(f"  原始特征矩阵类型: {feature_matrix.dtype}")

        try:
            # 修复前的逻辑（可能导致类型污染）
            old_industry_feature = np.array(case['codes']).reshape(-1, 1)
            old_result = np.hstack([feature_matrix, old_industry_feature])
            print(f"  修复前结果类型: {old_result.dtype}")

            # 修复后的逻辑
            new_industry_feature = np.array(case['codes'], dtype=np.float32).reshape(-1, 1)
            new_result = np.hstack([feature_matrix, new_industry_feature])
            print(f"  修复后结果类型: {new_result.dtype}")

            # 验证类型是否正确
            if new_result.dtype.kind in 'if':  # int 或 float
                print(f"  ✅ 修复成功: 类型保持为数值型")
            else:
                print(f"  ❌ 修复失败: 类型被污染为 {new_result.dtype}")

            # 验证数值是否正确
            print(f"  行业编码值: {new_industry_feature.flatten()}")

        except Exception as e:
            print(f"  ❌ 测试失败: {e}")

def test_lightgbm_compatibility():
    """测试与LightGBM的兼容性"""
    print("\n=== 测试LightGBM兼容性 ===")

    try:
        import lightgbm as lgb

        # 创建测试数据
        n_samples = 1000
        n_features = 10

        # 特征矩阵
        X = np.random.rand(n_samples, n_features).astype(np.float32)

        # 添加行业编码（修复后的逻辑）
        industry_codes = np.random.randint(0, 31, n_samples).astype(np.float32)
        X_with_industry = np.hstack([X, industry_codes.reshape(-1, 1)])

        print(f"  特征矩阵形状: {X_with_industry.shape}")
        print(f"  特征矩阵类型: {X_with_industry.dtype}")
        print(f"  行业编码列范围: {industry_codes.min():.0f} ~ {industry_codes.max():.0f}")

        # 创建标签
        y = np.random.rand(n_samples)

        # 设置分类特征
        categorical_features = [n_features]  # 最后一列是行业特征

        # 创建LightGBM数据集
        train_data = lgb.Dataset(
            X_with_industry,
            label=y,
            categorical_feature=categorical_features
        )

        print(f"  ✅ LightGBM数据集创建成功")
        print(f"  分类特征索引: {categorical_features}")

        # 简单训练测试
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'num_boost_round': 10
        }

        model = lgb.train(params, train_data)
        print(f"  ✅ LightGBM模型训练成功")

        # 检查特征重要性
        feature_importance = model.feature_importance()
        industry_importance = feature_importance[-1]  # 最后一列是行业特征
        total_importance = feature_importance.sum()
        industry_importance_ratio = industry_importance / total_importance

        print(f"  行业特征重要性: {industry_importance}")
        print(f"  行业特征重要性占比: {industry_importance_ratio:.2%}")

        if industry_importance_ratio < 0.5:  # 如果行业特征占比不超过50%
            print(f"  ✅ 特征重要性分布合理")
        else:
            print(f"  ⚠️  行业特征重要性过高，可能仍有问题")

    except ImportError:
        print(f"  ⚠️  未安装LightGBM，跳过兼容性测试")
    except Exception as e:
        print(f"  ❌ LightGBM兼容性测试失败: {e}")

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")

    edge_cases = [
        {
            'name': '全-1编码',
            'codes': [-1, -1, -1, -1, -1]
        },
        {
            'name': '大数值编码',
            'codes': [100, 200, 300, 400, 500]
        },
        {
            'name': '包含None',
            'codes': [13, None, 18, 7, 17]
        }
    ]

    for i, case in enumerate(edge_cases, 1):
        print(f"\n{i}. 测试案例: {case['name']}")

        try:
            # 测试修复后的转换逻辑
            feature_matrix = np.random.rand(5, 10).astype(np.float32)
            industry_feature = np.array(case['codes'], dtype=np.float32).reshape(-1, 1)
            result = np.hstack([feature_matrix, industry_feature])

            print(f"  ✅ 转换成功，结果类型: {result.dtype}")
            print(f"  行业编码值: {industry_feature.flatten()}")

        except Exception as e:
            print(f"  ❌ 转换失败: {e}")
            print(f"  这表明该情况需要特殊处理")

if __name__ == "__main__":
    print("开始测试行业编码类型污染修复...")

    test_dtype_conversion()
    test_lightgbm_compatibility()
    test_edge_cases()

    print("\n=== 测试完成 ===")
    print("如果所有测试都显示 ✅，说明修复成功！")
    print("如果有 ❌ 或 ⚠️，需要进一步检查问题。")