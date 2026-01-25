#!/usr/bin/env python3
import pickle
import numpy as np

# 检查两个特定的模型文件
model1_path = 'integrated_stock_selection/results/xgboost_model_20250824_213638.pkl'
model2_path = 'integrated_stock_selection/results/xgboost_model_20250825_171429.pkl'

print("检查模型文件:")
print(f"模型1: {model1_path}")
print(f"模型2: {model2_path}")

try:
    with open(model1_path, 'rb') as f:
        model1 = pickle.load(f)
    print("✅ 模型1加载成功")
except Exception as e:
    print(f"❌ 模型1加载失败: {e}")
    exit(1)

try:
    with open(model2_path, 'rb') as f:
        model2 = pickle.load(f)
    print("✅ 模型2加载成功")
except Exception as e:
    print(f"❌ 模型2加载失败: {e}")
    exit(1)

# 检查特征名称
features1 = model1['feature_names']
features2 = model2['feature_names']

print(f"\n特征对比:")
print(f"模型1特征数: {len(features1)}")
print(f"模型2特征数: {len(features2)}")
print(f"特征名称相同: {features1 == features2}")

if features1 != features2:
    print("❌ 特征名称不同!")
    for i, (f1, f2) in enumerate(zip(features1, features2)):
        if f1 != f2:
            print(f"  位置{i}: '{f1}' vs '{f2}'")
    exit(1)

# 检查标准化参数
scaler1 = model1['scaler']
scaler2 = model2['scaler']

mean_same = np.allclose(scaler1.mean_, scaler2.mean_, atol=1e-8)
scale_same = np.allclose(scaler1.scale_, scaler2.scale_, atol=1e-8)

print(f"\n标准化参数对比:")
print(f"均值参数相同: {mean_same}")
print(f"标准差参数相同: {scale_same}")

if not mean_same:
    print("❌ 均值参数不同!")
    diff_indices = np.where(~np.isclose(scaler1.mean_, scaler2.mean_, atol=1e-8))[0]
    print(f"不同的特征数量: {len(diff_indices)}")
    for idx in diff_indices[:3]:
        print(f"  {features1[idx]}: {scaler1.mean_[idx]:.8f} vs {scaler2.mean_[idx]:.8f}")
    exit(1)

if not scale_same:
    print("❌ 标准差参数不同!")
    diff_indices = np.where(~np.isclose(scaler1.scale_, scaler2.scale_, atol=1e-8))[0]
    print(f"不同的特征数量: {len(diff_indices)}")
    for idx in diff_indices[:3]:
        print(f"  {features1[idx]}: {scaler1.scale_[idx]:.8f} vs {scaler2.scale_[idx]:.8f}")
    exit(1)

print("✅ 所有参数完全相同")
print("问题不在模型文件本身")