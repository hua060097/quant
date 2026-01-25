#!/usr/bin/env python3
"""
检查模型文件结构
"""

import pickle
from pathlib import Path

def check_model_structure():
    """检查模型文件的详细结构"""

    model_path = Path('./results/lightgbm_model_window3_20251216_200900.pkl')

    if not model_path.exists():
        print("模型文件不存在")
        return

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        print("模型文件中的键:")
        for key in model_data.keys():
            value = model_data[key]
            value_type = type(value)
            print(f"  {key}: {value_type}")

            # 如果是模型对象，显示更多详情
            if key == 'model':
                print(f"    模型类型: {type(value)}")
                if hasattr(value, 'feature_name'):
                    try:
                        features = value.feature_name()
                        print(f"    特征数量: {len(features)}")
                        print(f"    前5个特征: {features[:5]}")
                    except:
                        print("    无法获取特征信息")

                if hasattr(value, 'best_iteration'):
                    print(f"    最佳迭代: {value.best_iteration}")

                if hasattr(value, 'num_feature'):
                    print(f"    特征数量: {value.num_feature()}")

        # 检查可能的特征字段名称
        possible_feature_keys = ['feature_columns', 'features', 'columns', 'feature_names']
        print(f"\n检查可能的特征字段:")
        for key in possible_feature_keys:
            if key in model_data:
                features = model_data[key]
                print(f"  {key}: {type(features)}")
                if hasattr(features, '__len__'):
                    print(f"    长度: {len(features)}")
                    if len(features) > 0:
                        print(f"    前5个: {features[:5]}")

    except Exception as e:
        print(f"检查模型结构失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_structure()