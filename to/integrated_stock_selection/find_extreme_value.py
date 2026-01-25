#!/usr/bin/env python3
"""查找pct_chg_1d因子中的异常值1913.573332"""

import pandas as pd
import numpy as np

def find_extreme_value():
    """查找异常值的股票和日期"""
    print("=== 查找pct_chg_1d异常值1913.573332 ===\n")

    # 读取修复后的因子数据
    try:
        corrected_df = pd.read_parquet('C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/factor_lab_data/factors_corrected.parquet')
        print(f"数据形状: {corrected_df.shape}")

        # 检查pct_chg_1d原始因子是否存在
        if 'pct_chg_1d' not in corrected_df.columns:
            print("pct_chg_1d原始因子不存在，检查所有相关因子...")
            pct_cols = [col for col in corrected_df.columns if 'pct_chg_1d' in col]
            print(f"找到的相关因子: {pct_cols}")
            return

        # 分析pct_chg_1d原始因子
        pct_chg_series = corrected_df['pct_chg_1d']
        print(f"pct_chg_1d统计:")
        print(f"  均值: {pct_chg_series.mean():.6f}")
        print(f"  标准差: {pct_chg_series.std():.6f}")
        print(f"  最小值: {pct_chg_series.min():.6f}")
        print(f"  最大值: {pct_chg_series.max():.6f}")
        print(f"  中位数: {pct_chg_series.median():.6f}")

        # 查找1913.573332附近的值
        target_value = 1913.573332
        tolerance = 0.001  # 容忍度

        # 查找匹配的行
        mask = abs(pct_chg_series - target_value) < tolerance
        matches = corrected_df[mask]

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

            # 查找超过1000的极端值
            extreme_mask = pct_chg_series > 1000
            extreme_values = corrected_df[extreme_mask]

            if len(extreme_values) > 0:
                print(f"\n找到 {len(extreme_values)} 个超过1000的极端值:")
                for idx, row in extreme_values.iterrows():
                    stock_code = row['stock_code']
                    date = row['date']
                    pct_value = row['pct_chg_1d']
                    print(f"  股票代码: {stock_code}, 日期: {date}, pct_chg_1d: {pct_value:.6f}")
            else:
                print("没有找到超过1000的值")

        # 查找前10个最大值
        print("\n前10个最大值:")
        top_10 = corrected_df.nlargest(10, 'pct_chg_1d')[['stock_code', 'date', 'pct_chg_1d']]
        for idx, row in top_10.iterrows():
            print(f"  {row['stock_code']} {row['date']}: {row['pct_chg_1d']:.6f}")

        # 检查该股票的基本信息
        if len(matches) > 0:
            sample_stock = matches.iloc[0]['stock_code']
            sample_date = matches.iloc[0]['date']
            print(f"\n检查股票 {sample_stock} 在 {sample_date} 的其他因子:")
            stock_mask = (corrected_df['stock_code'] == sample_stock) & (corrected_df['date'] == sample_date)
            stock_data = corrected_df[stock_mask]

            # 显示一些关键因子
            key_factors = ['pct_chg_0d', 'pct_chg_2d', 'return_3d', 'amount', 'close', 'open']
            for factor in key_factors:
                if factor in stock_data.columns:
                    value = stock_data[factor].iloc[0]
                    print(f"  {factor}: {value}")

    except Exception as e:
        print(f"查找异常值失败: {e}")

if __name__ == "__main__":
    find_extreme_value()