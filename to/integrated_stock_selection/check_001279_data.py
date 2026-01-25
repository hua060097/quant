#!/usr/bin/env python3
"""检查001279.SZ股票在20241114的数据"""

import pandas as pd
import numpy as np

def check_001279_data():
    """检查001279.SZ股票在20241114的数据"""
    print("=== 检查001279.SZ股票在20241114的数据 ===\n")

    try:
        # 读取20241114的因子数据
        df = pd.read_parquet("C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/20241114.parquet")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 查找001279.SZ股票
        stock_mask = df['stock_code'] == '001279.SZ'
        stock_data = df[stock_mask]

        if len(stock_data) > 0:
            print(f"\n找到001279.SZ股票数据 ({len(stock_data)}条记录):")
            for idx, row in stock_data.iterrows():
                print(f"  记录 {idx}:")
                print(f"    股票代码: {row['stock_code']}")
                print(f"    日期: {row['date']}")

                # 查找涨跌幅相关列
                pct_chg_cols = [col for col in df.columns if 'pct_chg' in col.lower()]
                print(f"    涨跌幅相关列: {pct_chg_cols}")

                for col in pct_chg_cols:
                    if col in row and pd.notna(row[col]):
                        print(f"    {col}: {row[col]:.6f}")

                # 查找其他关键价格数据
                price_cols = ['open', 'close', 'high', 'low']
                for col in price_cols:
                    if col in row and pd.notna(row[col]):
                        print(f"    {col}: {row[col]:.6f}")

                print()
        else:
            print("未找到001279.SZ股票数据")

        # 检查所有pct_chg_1d的数据范围
        if 'pct_chg_1d' in df.columns:
            pct_chg_1d_data = df['pct_chg_1d'].dropna()
            print(f"pct_chg_1d统计:")
            print(f"  总数: {len(pct_chg_1d_data)}")
            print(f"  均值: {pct_chg_1d_data.mean():.6f}")
            print(f"  标准差: {pct_chg_1d_data.std():.6f}")
            print(f"  最小值: {pct_chg_1d_data.min():.6f}")
            print(f"  最大值: {pct_chg_1d_data.max():.6f}")
            print(f"  中位数: {pct_chg_1d_data.median():.6f}")

            # 查找极端值
            extreme_mask = pct_chg_1d_data > 1000
            extreme_count = extreme_mask.sum()
            if extreme_count > 0:
                print(f"  超过1000的极端值数量: {extreme_count}")
                extreme_data = df[extreme_mask][['stock_code', 'date', 'pct_chg_1d']].sort_values('pct_chg_1d', ascending=False)
                print("  最极端的10个值:")
                for idx, row in extreme_data.head(10).iterrows():
                    print(f"    {row['stock_code']} {row['date']}: {row['pct_chg_1d']:.6f}")

        # 检查原始pct_chg列（如果存在）
        if 'pct_chg' in df.columns:
            pct_chg_data = df['pct_chg'].dropna()
            print(f"\n原始pct_chg统计:")
            print(f"  总数: {len(pct_chg_data)}")
            print(f"  均值: {pct_chg_data.mean():.6f}")
            print(f"  标准差: {pct_chg_data.std():.6f}")
            print(f"  最小值: {pct_chg_data.min():.6f}")
            print(f"  最大值: {pct_chg_data.max():.6f}")

            # 检查001279.SZ的原始pct_chg
            if len(stock_data) > 0:
                for idx, row in stock_data.iterrows():
                    if 'pct_chg' in row and pd.notna(row['pct_chg']):
                        print(f"  001279.SZ原始pct_chg: {row['pct_chg']:.6f}")

    except Exception as e:
        print(f"检查失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    check_001279_data()