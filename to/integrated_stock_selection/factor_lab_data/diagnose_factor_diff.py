"""
深度诊断 sector_broken_resilience_alpha 和 market_trend_adaptive_beta 的差异原因
"""
import pandas as pd
import numpy as np
from pathlib import Path

def diagnose_factor_differences(file1_path, file2_path):
    """诊断两个因子的差异原因"""

    print("=" * 100)
    print("深度差异诊断分析")
    print("=" * 100)

    # 读取文件
    print("\n[读取文件]")
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    print(f"  文件1 (副本): {df1.shape}")
    print(f"  文件2 (当前): {df2.shape}")

    # 找共同列
    common_cols = list(set(df1.columns) & set(df2.columns))
    print(f"  共同列数: {len(common_cols)}")

    # 对齐数据
    print("\n[数据对齐]")
    # 使用索引对齐
    df1_common = df1[common_cols].copy()
    df2_common = df2[common_cols].copy()

    # 检查索引是否一致
    if not df1_common.index.equals(df2_common.index):
        print("  索引不同，通过merge对齐...")
        aligned = df1_common.merge(
            df2_common,
            left_index=True,
            right_index=True,
            suffixes=('_file1', '_file2'),
            how='inner',
            indicator=True
        )
        print(f"  对齐后行数: {len(aligned)}")
        print(f"  仅在文件1: {(aligned['_merge'] == 'left_only').sum()}")
        print(f"  仅在文件2: {(aligned['_merge'] == 'right_only').sum()}")
        print(f"  共同行: {(aligned['_merge'] == 'both').sum()}")

        # 分离对齐后的数据
        df1_aligned = aligned[[col for col in aligned.columns if col.endswith('_file1')]]
        df2_aligned = aligned[[col for col in aligned.columns if col.endswith('_file2')]]
        df1_aligned.columns = [col[:-7] for col in df1_aligned.columns]
        df2_aligned.columns = [col[:-7] for col in df2_aligned.columns]
    else:
        print("  索引相同，直接对齐")
        df1_aligned = df1_common
        df2_aligned = df2_common

    # 分析的两个目标因子
    target_factors = ['sector_broken_resilience_alpha', 'market_trend_adaptive_beta']

    for factor in target_factors:
        if factor not in common_cols:
            print(f"\n[WARNING] {factor} 不在共同列中，跳过")
            continue

        print("\n" + "=" * 100)
        print(f"分析因子: {factor}")
        print("=" * 100)

        col1 = df1_aligned[factor]
        col2 = df2_aligned[factor]

        # 1. 基本统计
        print(f"\n[基本统计]")
        print(f"  文件1:")
        print(f"    均值: {col1.mean():.6f}")
        print(f"    标准差: {col1.std():.6f}")
        print(f"    最小值: {col1.min():.6f}")
        print(f"    最大值: {col1.max():.6f}")
        print(f"    中位数: {col1.median():.6f}")

        print(f"  文件2:")
        print(f"    均值: {col2.mean():.6f}")
        print(f"    标准差: {col2.std():.6f}")
        print(f"    最小值: {col2.min():.6f}")
        print(f"    最大值: {col2.max():.6f}")
        print(f"    中位数: {col2.median():.6f}")

        # 2. 缺失值分析
        print(f"\n[缺失值分析]")
        nan1 = col1.isna()
        nan2 = col2.isna()

        print(f"  文件1缺失: {nan1.sum()} ({nan1.sum()/len(col1)*100:.2f}%)")
        print(f"  文件2缺失: {nan2.sum()} ({nan2.sum()/len(col2)*100:.2f}%)")
        print(f"  共同缺失: {(nan1 & nan2).sum()}")
        print(f"  仅文件1缺失: {(nan1 & ~nan2).sum()}")
        print(f"  仅文件2缺失: {(~nan1 & nan2).sum()}")

        # 3. 完全相同的比例
        exact_match = (col1 == col2).sum()
        print(f"\n[完全相同]")
        print(f"  完全相同的行数: {exact_match} / {len(col1)} ({exact_match/len(col1)*100:.2f}%)")

        # 4. 数值差异分析
        print(f"\n[数值差异分析]")
        col1_numeric = pd.to_numeric(col1, errors='coerce')
        col2_numeric = pd.to_numeric(col2, errors='coerce')

        diff = col2_numeric - col1_numeric
        abs_diff = np.abs(diff)

        # 过滤掉NaN
        valid_mask = ~(np.isnan(col1_numeric) | np.isnan(col2_numeric))
        diff_valid = diff[valid_mask]
        abs_diff_valid = abs_diff[valid_mask]

        print(f"  有效比较行数: {valid_mask.sum()}")
        print(f"  差异=0的行数: {(diff_valid == 0).sum()} ({(diff_valid == 0).sum()/len(diff_valid)*100:.2f}%)")
        print(f"  差异>0的行数: {(diff_valid > 0).sum()} ({(diff_valid > 0).sum()/len(diff_valid)*100:.2f}%)")
        print(f"  差异<0的行数: {(diff_valid < 0).sum()} ({(diff_valid < 0).sum()/len(diff_valid)*100:.2f}%)")
        print(f"  最大差异: {abs_diff_valid.max():.6f}")
        print(f"  平均差异: {abs_diff_valid.mean():.6f}")
        print(f"  差异中位数: {abs_diff_valid.median():.6f}")
        print(f"  差异标准差: {diff_valid.std():.6f}")

        # 5. 差异分布
        print(f"\n[差异分布]")
        diff_ranges = [
            (0, 1e-6, "0 ~ 1e-6 (浮点误差)"),
            (1e-6, 0.01, "1e-6 ~ 0.01"),
            (0.01, 0.1, "0.01 ~ 0.1"),
            (0.1, 1.0, "0.1 ~ 1.0"),
            (1.0, 10.0, "1.0 ~ 10.0"),
            (10.0, float('inf'), "> 10.0")
        ]

        for low, high, label in diff_ranges:
            count = ((abs_diff_valid >= low) & (abs_diff_valid < high)).sum()
            pct = count / len(diff_valid) * 100
            print(f"  {label:<30} {count:>10} ({pct:>6.2f}%)")

        # 6. 大差异样本分析
        print(f"\n[大差异样本分析 (差异>1.0)]")
        large_diff_mask = abs_diff_valid > 1.0
        large_diff_count = large_diff_mask.sum()

        if large_diff_count > 0:
            print(f"  大差异样本数: {large_diff_count} ({large_diff_count/len(diff_valid)*100:.2f}%)")

            # 找出前10个最大差异的索引
            top_large_diff_indices = abs_diff_valid[large_diff_mask].nlargest(10).index

            print(f"  前10个最大差异样本:")
            for idx in top_large_diff_indices:
                val1 = col1_numeric[idx]
                val2 = col2_numeric[idx]
                diff_val = diff[idx]
                print(f"    索引={idx}: 文件1={val1:.6f}, 文件2={val2:.6f}, 差异={diff_val:.6f}")
        else:
            print(f"  无大差异样本")

        # 7. 值分布分析
        print(f"\n[值分布分析]")
        # 创建分桶统计
        bins = [-float('inf'), -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, float('inf')]
        bin_labels = ["<-10", "-10~-5", "-5~-2", "-2~-1", "-1~-0.5", "-0.5~0",
                      "0~0.5", "0.5~1", "1~2", "2~5", "5~10", ">10"]

        dist1 = pd.cut(col1_numeric, bins=bins, labels=bin_labels).value_counts().sort_index()
        dist2 = pd.cut(col2_numeric, bins=bins, labels=bin_labels).value_counts().sort_index()

        print(f"  {'区间':<10} {'文件1数量':>12} {'文件1占比':>12} {'文件2数量':>12} {'文件2占比':>12}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for label in bin_labels:
            count1 = dist1.get(label, 0)
            count2 = dist2.get(label, 0)
            pct1 = count1 / len(col1_numeric) * 100
            pct2 = count2 / len(col2_numeric) * 100
            print(f"  {label:<10} {count1:>12} {pct1:>11.2f}% {count2:>12} {pct2:>11.2f}%")

        # 8. 特殊值检查
        print(f"\n[特殊值检查]")
        print(f"  文件1:")
        print(f"    零值数量: {(col1_numeric == 0).sum()} ({(col1_numeric == 0).sum()/len(col1_numeric)*100:.2f}%)")
        print(f"    正值数量: {(col1_numeric > 0).sum()} ({(col1_numeric > 0).sum()/len(col1_numeric)*100:.2f}%)")
        print(f"    负值数量: {(col1_numeric < 0).sum()} ({(col1_numeric < 0).sum()/len(col1_numeric)*100:.2f}%)")

        print(f"  文件2:")
        print(f"    零值数量: {(col2_numeric == 0).sum()} ({(col2_numeric == 0).sum()/len(col2_numeric)*100:.2f}%)")
        print(f"    正值数量: {(col2_numeric > 0).sum()} ({(col2_numeric > 0).sum()/len(col2_numeric)*100:.2f}%)")
        print(f"    负值数量: {(col2_numeric < 0).sum()} ({(col2_numeric < 0).sum()/len(col2_numeric)*100:.2f}%)")

        # 9. 相关性分析
        print(f"\n[相关性分析]")
        # 只在非缺失值上计算相关性
        both_valid = ~(np.isnan(col1_numeric) | np.isnan(col2_numeric))
        if both_valid.sum() > 0:
            correlation = col1_numeric[both_valid].corr(col2_numeric[both_valid])
            print(f"  文件1和文件2的相关系数: {correlation:.6f}")
        else:
            print(f"  无有效数据计算相关性")

        # 10. 检查是否与行索引有关
        print(f"\n[索引差异分析]")
        if not df1_aligned.index.equals(df2_aligned.index):
            print(f"  索引不完全相同")
            diff_indices1 = set(df1_aligned.index) - set(df2_aligned.index)
            diff_indices2 = set(df2_aligned.index) - set(df1_aligned.index)
            if diff_indices1:
                print(f"    仅在文件1中的索引数量: {len(diff_indices1)}")
                print(f"    示例: {list(diff_indices1)[:5]}")
            if diff_indices2:
                print(f"    仅在文件2中的索引数量: {len(diff_indices2)}")
                print(f"    示例: {list(diff_indices2)[:5]}")

    # 11. 检查其他可能相关的列
    print(f"\n[检查可能影响因子计算的列]")
    # 检查常见的输入列
    potential_input_cols = [
        'pct_chg_1d', 'return_5d_intra_z', 'return_10d_intra_z', 'return_20d_intra_z',
        'sector_mkt_corr_20d', 'beta_60d', 'market_trend_strength', 'market_volatility_context'
    ]

    for col in potential_input_cols:
        if col in common_cols:
            col1_data = pd.to_numeric(df1_aligned[col], errors='coerce')
            col2_data = pd.to_numeric(df2_aligned[col], errors='coerce')

            valid_mask = ~(np.isnan(col1_data) | np.isnan(col2_data))
            diff = np.abs(col1_data - col2_data)

            if valid_mask.sum() > 0:
                max_diff = diff[valid_mask].max()
                mean_diff = diff[valid_mask].mean()
                pct_diff = (diff[valid_mask] > 0.01).sum() / valid_mask.sum() * 100

                print(f"  {col}:")
                print(f"    最大差异: {max_diff:.6f}, 平均差异: {mean_diff:.6f}, 差异>0.01占比: {pct_diff:.2f}%")

    print(f"\n[分析完成]")

if __name__ == "__main__":
    # 文件路径
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")
    file1 = base_path / "factors - 副本.parquet"
    file2 = base_path / "factors.parquet"

    diagnose_factor_differences(file1, file2)
