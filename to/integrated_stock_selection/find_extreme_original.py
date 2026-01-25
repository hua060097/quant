#!/usr/bin/env python3
"""在原始factors.parquet中查找pct_chg_1d异常值1913.573332"""

import pandas as pd
import numpy as np

def find_extreme_original():
    """在原始文件中查找异常值"""
    print("=== 在原始factors.parquet中查找pct_chg_1d异常值 ===\n")

    try:
        # 读取原始因子数据
        original_df = pd.read_parquet('C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/factor_lab_data/factors.parquet')
        print(f"原始数据形状: {original_df.shape}")

        # 检查pct_chg_1d
        if 'pct_chg_1d' not in original_df.columns:
            print("pct_chg_1d不存在，查找所有pct相关因子:")
            pct_cols = [col for col in original_df.columns if 'pct_chg' in col]
            print(f"找到的pct相关因子: {pct_cols}")
            return

        # 分析pct_chg_1d
        pct_chg_series = original_df['pct_chg_1d']
        print(f"pct_chg_1d统计:")
        print(f"  均值: {pct_chg_series.mean():.6f}")
        print(f"  标准差: {pct_chg_series.std():.6f}")
        print(f"  最小值: {pct_chg_series.min():.6f}")
        print(f"  最大值: {pct_chg_series.max():.6f}")

        # 查找1913.573332附近的值
        target_value = 1913.573332
        tolerance = 0.001

        # 查找匹配
        mask = abs(pct_chg_series - target_value) < tolerance
        matches = original_df[mask]

        if len(matches) > 0:
            print(f"\n找到 {len(matches)} 个匹配的异常值:")
            for idx, row in matches.iterrows():
                stock_code = row['stock_code']
                date = row['date']
                pct_value = row['pct_chg_1d']
                print(f"  股票代码: {stock_code}, 日期: {date}, pct_chg_1d: {pct_value:.6f}")
        else:
            print(f"\n未找到精确匹配 {target_value} 的值")

            # 查找最接近的值
            closest_diff = abs(pct_chg_series - target_value).min()
            closest_value = pct_chg_series.iloc[abs(pct_chg_series - target_value).argmin()]
            closest_idx = abs(pct_chg_series - target_value).argmin()

            print(f"最接近的值:")
            print(f"  数值: {closest_value:.6f}")
            print(f"  差异: {closest_diff:.6f}")

            # 查找该异常值的位置
            closest_row = original_df.iloc[[closest_idx]]
            stock_code = closest_row['stock_code'].iloc[0]
            date = closest_row['date'].iloc[0]
            print(f"  股票代码: {stock_code}, 日期: {date}")

        # 查找超过1000的极端值
        extreme_mask = pct_chg_series > 1000
        extreme_values = original_df[extreme_mask]

        if len(extreme_values) > 0:
            print(f"\n找到 {len(extreme_values)} 个超过1000的极端值:")
            for idx, row in extreme_values.iterrows():
                stock_code = row['stock_code']
                date = row['date']
                pct_value = row['pct_chg_1d']
                print(f"  {stock_code} {date}: {pct_value:.6f}")
        else:
            print("\n没有找到超过1000的值")

        # 查找前10个最大值
        print("\n前10个最大值:")
        top_10 = original_df.nlargest(10, 'pct_chg_1d')[['stock_code', 'date', 'pct_chg_1d']]
        for idx, row in top_10.iterrows():
            print(f"  {row['stock_code']} {row['date']}: {row['pct_chg_1d']:.6f}")

        # 分析分布
        print(f"\n分布分析:")
        print(f"  > 10 的数量: {(pct_chg_series > 10).sum()}")
        print(f"  > 50 的数量: {(pct_chg_series > 50).sum()}")
        print(f"  > 100 的数量: {(pct_chg_series > 100).sum()}")
        print(f"  > 500 的数量: {(pct_chg_series > 500).sum()}")
        print(f"  > 1000 的数量: {(pct_chg_series > 1000).sum()}")

    except Exception as e:
        print(f"查找失败: {e}")

if __name__ == "__main__":
    find_extreme_original()