#!/usr/bin/env python3
"""简单验证修复后的因子数据"""

import pandas as pd

def simple_verify():
    """简单验证修复结果"""
    print("=== 智能裁剪修复验证 ===")

    # 读取统计信息
    try:
        with open('C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/factor_lab_data/factors_corrected_stats.txt', 'r', encoding='utf-8') as f:
            stats = f.read()
        print("修复统计信息:")
        print(stats)

        # 读取修复后的数据
        corrected_df = pd.read_parquet('C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/factor_lab_data/factors_corrected.parquet')
        print(f"修复后数据形状: {corrected_df.shape}")

        # 检查关键因子
        key_factors = ['corr_slope_20_120_rank', 'ma_convergence_score_rank', 'pct_chg_1d_z', 'return_3d_z']
        print("\n关键因子检查:")
        for factor in key_factors:
            if factor in corrected_df.columns:
                series = corrected_df[factor]
                extreme_ratio = 0
                if '_z' in factor:
                    extreme_ratio = ((series.abs() > 3).sum() / len(series))
                print(f"{factor}: 均值={series.mean():.4f}, 标准差={series.std():.4f}, 范围=[{series.min():.4f}, {series.max():.4f}]", end="")
                if extreme_ratio > 0:
                    print(f", 极端值比例={extreme_ratio:.2%}")
                else:
                    print()

        print("\n修复完成！建议使用 factors_corrected.parquet 作为新的因子数据文件。")

    except Exception as e:
        print(f"验证失败: {e}")

if __name__ == "__main__":
    simple_verify()