#!/usr/bin/env python3
"""
最终测试pred_stocks模式功能
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def test_pred_stocks_complete():
    """完整测试pred_stocks模式功能"""

    print("=== 最终测试pred_stocks模式 ===")

    prediction_date = "20251216"

    try:
        # 1. 加载因子数据
        print("1. 加载因子数据...")
        factor_file = Path(f"../local_data/features/daily/{prediction_date}.parquet")

        if not factor_file.exists():
            print(f"因子文件不存在: {factor_file}")
            return False

        factor_data = pd.read_parquet(factor_file)
        print(f"   成功加载: {len(factor_data)} 只股票, {len(factor_data.columns)} 个特征")

        # 2. 加载模型
        print("2. 加载模型...")
        model_path = Path("./results/lightgbm_model_window3_20251216_200900.pkl")

        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return False

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        feature_columns = model_data.get('feature_names', [])

        print(f"   成功加载模型: {len(feature_columns)} 个特征")

        # 3. 加载SW行业映射
        print("3. 加载SW行业映射...")
        try:
            from main import load_sw_sector_mapping
            sw_sector_mapping = load_sw_sector_mapping()
            print(f"   SW行业映射: {len(sw_sector_mapping)} 只股票")
        except:
            sw_sector_mapping = {}
            print("   SW行业映射: 无")

        # 4. 数据预处理
        print("4. 数据预处理...")
        # 添加SW行业信息
        if sw_sector_mapping:
            factor_data['sw_sector'] = factor_data['stock_code'].map(sw_sector_mapping)

        # 检查特征列
        available_features = [f for f in feature_columns if f in factor_data.columns]
        print(f"   可用特征: {len(available_features)}/{len(feature_columns)}")

        if len(available_features) < len(feature_columns) * 0.8:
            print(f"   错误: 可用特征太少: {len(available_features)}/{len(feature_columns)}")
            return False

        # 准备预测数据
        X_pred = factor_data[available_features].copy()
        X_pred = X_pred.fillna(X_pred.median())
        X_pred = X_pred.replace([np.inf, -np.inf], 0)

        print(f"   预测数据准备完成: {len(X_pred)} 只股票")

        # 5. 模型预测
        print("5. 执行模型预测...")
        predictions = model.predict(X_pred, num_iteration=model.best_iteration)

        # 添加预测值到数据中
        factor_data['prediction'] = predictions
        factor_data = factor_data.sort_values('prediction', ascending=False)

        print(f"   预测完成，预测值范围: [{predictions.min():.6f}, {predictions.max():.6f}]")

        # 6. 输出Top5结果
        print("6. 输出预测结果")
        top5_stocks = factor_data.head(5)

        print(f"\n Top 5 预测结果 ({prediction_date}):")
        print("=" * 80)
        print(f"{'排名':<4} {'股票代码':<12} {'预测值':<12} {'SW行业':<12}")
        print("-" * 80)

        for rank, (_, row) in enumerate(top5_stocks.iterrows(), 1):
            stock_code = row['stock_code']
            sw_sector = row.get('sw_sector', '未知')
            prediction = row['prediction']

            print(f"{rank:<4} {stock_code:<12} {prediction:<12.6f} {sw_sector:<12}")

        print("=" * 80)

        # 7. 保存结果
        results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)

        result_file = results_dir / f"pred_stocks_{prediction_date}.csv"
        factor_data.to_csv(result_file, index=False, encoding='utf-8-sig')

        print(f" 结果已保存到: {result_file}")
        print("\npred_stocks模式测试成功!")

        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pred_stocks_complete()