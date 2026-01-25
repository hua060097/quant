#!/usr/bin/env python3
"""
检查模型特征名称与因子数据的映射关系
"""

import pandas as pd
import pickle
from pathlib import Path

def check_feature_mapping():
    """检查特征名称映射"""

    print("=== 检查特征名称映射 ===")

    # 1. 加载模型特征信息
    model_path = Path('./results/lightgbm_model_window3_20251216_200900.pkl')

    if not model_path.exists():
        print("模型文件不存在")
        return

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model_features = model_data.get('feature_names', [])
        print(f"模型特征数量: {len(model_features)}")
        print(f"模型特征前10个: {model_features[:10]}")

        # 2. 加载因子数据特征
        factor_file = Path("../local_data/features/daily/20251216.parquet")
        if not factor_file.exists():
            print("因子文件不存在")
            return

        factor_data = pd.read_parquet(factor_file)

        # 获取数值类型的列（排除股票代码和日期等）
        numeric_columns = []
        for col in factor_data.columns:
            if factor_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_columns.append(col)

        print(f"因子数据数值列数量: {len(numeric_columns)}")
        print(f"因子数据前10个数值列: {numeric_columns[:10]}")

        # 3. 检查映射关系
        print(f"\n=== 特征映射分析 ===")

        # 检查是否有完全匹配的特征
        exact_matches = set(model_features) & set(numeric_columns)
        print(f"完全匹配的特征数量: {len(exact_matches)}")
        if exact_matches:
            print(f"匹配的特征: {list(exact_matches)[:5]}")

        # 检查模式匹配（如Column_X vs alpha_XXX）
        print(f"\n模型特征模式: {model_features[0]}")
        print(f"因子特征模式: {numeric_columns[0] if numeric_columns else 'None'}")

        # 如果模型特征是Column_X格式，我们需要映射
        if model_features and model_features[0].startswith('Column_'):
            print(f"\n⚠️  模型使用通用特征名称，需要进行映射")
            print(f"建议的映射策略:")
            print(f"1. 直接使用model.feature_name()获取特征映射")
            print(f"2. 或者使用固定的特征名称映射")

    except Exception as e:
        print(f"检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_feature_mapping()