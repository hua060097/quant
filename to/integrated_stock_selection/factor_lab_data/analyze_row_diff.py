"""
分析行数差异对因子计算的影响
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_row_difference(file1_path, file2_path):
    """分析行数差异及其影响"""

    print("=" * 100)
    print("行数差异影响分析")
    print("=" * 100)

    # 读取文件
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    print(f"\n[基本信息]")
    print(f"  文件1 (副本): {df1.shape[0]:,} 行 × {df1.shape[1]} 列")
    print(f"  文件2 (当前): {df2.shape[0]:,} 行 × {df2.shape[1]} 列")
    print(f"  行数差异: {df2.shape[0] - df1.shape[0]:+,} 行")

    # 使用复合键对齐
    if 'stock_code' in df1.columns and 'date' in df1.columns:
        df1['_key'] = df1['stock_code'].astype(str) + '_' + df1['date'].astype(str)
        df2['_key'] = df2['stock_code'].astype(str) + '_' + df2['date'].astype(str)

        keys1 = set(df1['_key'])
        keys2 = set(df2['_key'])

        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        common_keys = keys1 & keys2

        print(f"\n[行级差异分析]")
        print(f"  共同行: {len(common_keys):,} ({len(common_keys)/max(len(keys1), len(keys2))*100:.2f}%)")
        print(f"  仅在文件1: {len(only_in_1):,} ({len(only_in_1)/len(keys1)*100:.2f}%)")
        print(f"  仅在文件2: {len(only_in_2):,} ({len(only_in_2)/len(keys2)*100:.2f}%)")

        # 找出差异行的详细信息
        if len(only_in_1) > 0:
            print(f"\n[仅在文件1（副本）中的行 - {len(only_in_1)}行]")
            diff_df1 = df1[df1['_key'].isin(only_in_1)]
            print(f"  日期分布:")
            for date, count in diff_df1['date'].value_counts().head(10).items():
                print(f"    {date}: {count}行")
            print(f"  股票分布:")
            for stock, count in diff_df1['stock_code'].value_counts().head(10).items():
                print(f"    {stock}: {count}行")

        if len(only_in_2) > 0:
            print(f"\n[仅在文件2（当前）中的行 - {len(only_in_2)}行]")
            diff_df2 = df2[df2['_key'].isin(only_in_2)]
            print(f"  日期分布:")
            for date, count in diff_df2['date'].value_counts().head(10).items():
                print(f"    {date}: {count}行")
            print(f"  股票分布:")
            for stock, count in diff_df2['stock_code'].value_counts().head(10).items():
                print(f"    {stock}: {count}行")

            # 分析这些新增行对market-level因子的影响
            print(f"\n[新增行对关键因子的影响分析]")

            key_factors = ['market_trend_adaptive_beta', 'sector_broken_resilience_alpha',
                          'market_trend_strength', 'market_volatility_context']

            for factor in key_factors:
                if factor in df2.columns:
                    # 新增行的因子值
                    new_rows_values = diff_df2[factor]

                    # 所有行的因子值（文件2）
                    all_values = df2[factor]

                    # 共同行的因子值
                    common_df2 = df2[df2['_key'].isin(common_keys)]
                    common_values = common_df2[factor]

                    print(f"\n  {factor}:")
                    print(f"    新增行统计:")
                    print(f"      均值: {new_rows_values.mean():.6f}")
                    print(f"      标准差: {new_rows_values.std():.6f}")
                    print(f"      最小值: {new_rows_values.min():.6f}")
                    print(f"      最大值: {new_rows_values.max():.6f}")
                    print(f"    全部数据 vs 共同数据:")
                    print(f"      全部均值: {all_values.mean():.6f}")
                    print(f"      共同均值: {common_values.mean():.6f}")
                    print(f"      均值差异: {abs(all_values.mean() - common_values.mean()):.6f}")

        # 分析market-level因子在不同子集上的差异
        print(f"\n[关键因子受数据集影响的分析]")

        # 只在共同行上比较两个文件
        common_df1 = df1[df1['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)
        common_df2 = df2[df2['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)

        # 比较market_trend_adaptive_beta在共同行上的差异
        if 'market_trend_adaptive_beta' in common_df1.columns:
            print(f"\n  market_trend_adaptive_beta (仅共同行):")
            col1 = common_df1['market_trend_adaptive_beta']
            col2 = common_df2['market_trend_adaptive_beta']

            diff = col2 - col1
            abs_diff = np.abs(diff)

            print(f"    完全相同: {(diff == 0).sum()} / {len(diff)} ({(diff == 0).sum()/len(diff)*100:.2f}%)")
            print(f"    最大差异: {abs_diff.max():.6f}")
            print(f"    平均差异: {abs_diff.mean():.6f}")

            # 检查是否仍有异常值
            abnormal1 = (abs(col1) > 50.0).sum()
            abnormal2 = (abs(col2) > 50.0).sum()
            print(f"    文件1异常值数量(>50): {abnormal1} ({abnormal1/len(col1)*100:.2f}%)")
            print(f"    文件2异常值数量(>50): {abnormal2} ({abnormal2/len(col2)*100:.2f}%)")

            # 检查异常值是否相同
            abnormal_mask1 = abs(col1) > 50.0
            abnormal_mask2 = abs(col2) > 50.0

            both_abnormal = (abnormal_mask1 & abnormal_mask2).sum()
            only_file1_abnormal = (abnormal_mask1 & ~abnormal_mask2).sum()
            only_file2_abnormal = (~abnormal_mask1 & abnormal_mask2).sum()

            print(f"    异常值对比:")
            print(f"      两文件都异常: {both_abnormal}")
            print(f"      仅文件1异常: {only_file1_abnormal}")
            print(f"      仅文件2异常: {only_file2_abnormal}")

        # 分析为什么market_trend_adaptive_beta会有差异
        print(f"\n[market_trend_adaptive_beta差异原因推测]")

        # 检查这个因子可能依赖的输入因子
        potential_inputs = [
            'pct_chg_1d', 'beta_60d', 'sector_mkt_corr_20d',
            'market_trend_strength', 'market_volatility_context'
        ]

        print(f"  检查依赖因子的差异（在共同行上）:")
        for input_factor in potential_inputs:
            if input_factor in common_df1.columns:
                in_col1 = pd.to_numeric(common_df1[input_factor], errors='coerce')
                in_col2 = pd.to_numeric(common_df2[input_factor], errors='coerce')

                valid = ~(in_col1.isna() | in_col2.isna())
                if valid.sum() > 0:
                    in_diff = np.abs(in_col2 - in_col1)
                    max_diff = in_diff[valid].max()
                    mean_diff = in_diff[valid].mean()
                    pct_diff = (in_diff[valid] > 0.01).sum() / valid.sum() * 100

                    if pct_diff > 1.0:  # 差异超过1%
                        print(f"    {input_factor}:")
                        print(f"      最大差异: {max_diff:.6f}")
                        print(f"      平均差异: {mean_diff:.6f}")
                        print(f"      差异>0.01占比: {pct_diff:.2f}%")

        # 检查是否是计算顺序或缓存问题
        print(f"\n[可能的根本原因]")
        print(f"  1. 数据集差异（28行）导致市场层面统计量重新计算")
        print(f"  2. 因子计算的先后顺序或依赖关系变化")
        print(f"  3. 缓存未生效导致重新计算结果不同")
        print(f"  4. 浮点数累加顺序差异")

        # 检查sector_broken_resilience_alpha
        if 'sector_broken_resilience_alpha' in common_df1.columns:
            print(f"\n  sector_broken_resilience_alpha (仅共同行):")
            col1 = common_df1['sector_broken_resilience_alpha']
            col2 = common_df2['sector_broken_resilience_alpha']

            diff = col2 - col1
            abs_diff = np.abs(diff)

            print(f"    完全相同: {(diff == 0).sum()} / {len(diff)} ({(diff == 0).sum()/len(diff)*100:.2f}%)")
            print(f"    最大差异: {abs_diff.max():.6f}")
            print(f"    平均差异: {abs_diff.mean():.6f}")

    print(f"\n[分析完成]")

if __name__ == "__main__":
    # 文件路径
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")
    file1 = base_path / "factors - 副本.parquet"
    file2 = base_path / "factors.parquet"

    analyze_row_difference(file1, file2)
