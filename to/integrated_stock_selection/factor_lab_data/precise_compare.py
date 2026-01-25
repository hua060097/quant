"""
精确对比：相同日期范围 + 相同因子的数据差异
"""
import pandas as pd
import numpy as np
from pathlib import Path

def precise_compare_factors(file1_path, file2_path):
    """精确对比相同日期和相同因子的数据"""

    print("=" * 100)
    print("精确对比：相同日期范围 + 相同因子")
    print("=" * 100)

    # 读取文件
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    print(f"\n[原始文件信息]")
    print(f"  文件1 (副本): {df1.shape[0]:,} 行 × {df1.shape[1]} 列")
    print(f"  文件2 (当前): {df2.shape[0]:,} 行 × {df2.shape[1]} 列")

    # 找共同因子（排除stock_code和date）
    exclude_cols = {'stock_code', 'date'}
    cols1 = set(df1.columns) - exclude_cols
    cols2 = set(df2.columns) - exclude_cols
    common_cols = sorted(list(cols1 & cols2))

    print(f"\n[共同因子统计]")
    print(f"  文件1因子数: {len(cols1)}")
    print(f"  文件2因子数: {len(cols2)}")
    print(f"  共同因子数: {len(common_cols)}")

    # 仅在文件1的因子（新增的6个）
    only_in_1 = sorted(list(cols1 - cols2))
    print(f"  仅在文件1: {len(only_in_1)}个")
    if only_in_1:
        print(f"    {', '.join(only_in_1)}")

    # 日期过滤：2023-06-01 至 2025-12-26
    start_date = pd.Timestamp('2023-06-01')
    end_date = pd.Timestamp('2025-12-26')

    df1['_date'] = pd.to_datetime(df1['date'])
    df2['_date'] = pd.to_datetime(df2['date'])

    df1_filtered = df1[(df1['_date'] >= start_date) & (df1['_date'] <= end_date)].copy()
    df2_filtered = df2[(df2['_date'] >= start_date) & (df2['_date'] <= end_date)].copy()

    print(f"\n[日期范围过滤]")
    print(f"  目标日期范围: 2023-06-01 至 2025-12-26")
    print(f"  文件1过滤后: {df1_filtered.shape[0]:,} 行")
    print(f"  文件2过滤后: {df2_filtered.shape[0]:,} 行")

    # 创建复合键对齐
    df1_filtered['_key'] = df1_filtered['stock_code'].astype(str) + '_' + df1_filtered['_date'].astype(str)
    df2_filtered['_key'] = df2_filtered['stock_code'].astype(str) + '_' + df2_filtered['_date'].astype(str)

    keys1 = set(df1_filtered['_key'])
    keys2 = set(df2_filtered['_key'])

    print(f"\n[行级对齐]")
    print(f"  文件1唯一键: {len(keys1):,}")
    print(f"  文件2唯一键: {len(keys2):,}")
    print(f"  共同键: {len(keys1 & keys2):,}")
    print(f"  仅在文件1: {len(keys1 - keys2):,}")
    print(f"  仅在文件2: {len(keys2 - keys1):,}")

    # 使用共同键进行对齐
    common_keys = keys1 & keys2

    df1_aligned = df1_filtered[df1_filtered['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)
    df2_aligned = df2_filtered[df2_filtered['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)

    print(f"\n[对齐后数据]")
    print(f"  对齐行数: {len(df1_aligned):,}")
    print(f"  对比因子数: {len(common_cols)}")

    # 对比目标因子
    target_factors = ['sector_broken_resilience_alpha', 'market_trend_adaptive_beta']

    for factor in target_factors:
        if factor not in common_cols:
            continue

        print("\n" + "=" * 100)
        print(f"因子: {factor}")
        print("=" * 100)

        col1 = df1_aligned[factor]
        col2 = df2_aligned[factor]

        # 基本统计
        print(f"\n[基本统计]")
        print(f"  文件1:")
        print(f"    均值: {col1.mean():.6f}")
        print(f"    标准差: {col1.std():.6f}")
        print(f"    最小值: {col1.min():.6f}")
        print(f"    最大值: {col1.max():.6f}")

        print(f"  文件2:")
        print(f"    均值: {col2.mean():.6f}")
        print(f"    标准差: {col2.std():.6f}")
        print(f"    最小值: {col2.min():.6f}")
        print(f"    最大值: {col2.max():.6f}")

        # 完全相同比例
        exact_match = (col1 == col2).sum()
        print(f"\n[完全相同]")
        print(f"  完全相同的行数: {exact_match:,} / {len(col1):,} ({exact_match/len(col1)*100:.4f}%)")

        # 数值差异分析
        col1_numeric = pd.to_numeric(col1, errors='coerce')
        col2_numeric = pd.to_numeric(col2, errors='coerce')

        valid_mask = ~(col1_numeric.isna() | col2_numeric.isna())
        diff = col2_numeric - col1_numeric
        abs_diff = np.abs(diff)

        if valid_mask.sum() > 0:
            diff_valid = diff[valid_mask]
            abs_diff_valid = abs_diff[valid_mask]

            print(f"\n[数值差异分析]")
            print(f"  有效比较行数: {valid_mask.sum():,}")
            print(f"  差异=0: {(diff_valid == 0).sum():,} ({(diff_valid == 0).sum()/len(diff_valid)*100:.4f}%)")
            print(f"  差异>0: {(diff_valid > 0).sum():,} ({(diff_valid > 0).sum()/len(diff_valid)*100:.4f}%)")
            print(f"  差异<0: {(diff_valid < 0).sum():,} ({(diff_valid < 0).sum()/len(diff_valid)*100:.4f}%)")
            print(f"  最大差异: {abs_diff_valid.max():.6f}")
            print(f"  平均差异: {abs_diff_valid.mean():.6f}")
            print(f"  差异中位数: {abs_diff_valid.median():.6f}")

            # 差异分布
            print(f"\n[差异分布]")
            bins = [0, 1e-9, 1e-6, 0.01, 0.1, 1.0, 10.0, float('inf')]
            labels = ["=0", "0~1e-9", "1e-6~0.01", "0.01~0.1", "0.1~1.0", "1.0~10.0", ">10.0"]

            for i in range(len(bins)-1):
                low, high = bins[i], bins[i+1]
                if i == len(bins)-2:
                    count = (abs_diff_valid >= low).sum()
                else:
                    count = ((abs_diff_valid >= low) & (abs_diff_valid < high)).sum()
                pct = count / len(diff_valid) * 100
                print(f"  {labels[i]:<15} {count:>10,} ({pct:>8.4f}%)")

            # 大差异分析
            large_diff_threshold = 1.0
            large_diff_mask = abs_diff_valid > large_diff_threshold
            large_diff_count = large_diff_mask.sum()

            if large_diff_count > 0:
                print(f"\n[大差异样本分析 (差异>{large_diff_threshold})]")
                print(f"  大差异行数: {large_diff_count:,} ({large_diff_count/len(diff_valid)*100:.4f}%)")

                # 找出前10个最大差异
                top_indices = abs_diff_valid.nlargest(10).index
                print(f"  最大的10个差异:")
                for idx in top_indices:
                    val1 = col1_numeric[idx]
                    val2 = col2_numeric[idx]
                    diff_val = diff[idx]
                    print(f"    索引={idx}: 文件1={val1:.6f}, 文件2={val2:.6f}, 差异={diff_val:.6f}")

            # 相关性分析
            print(f"\n[相关性分析]")
            correlation = col1_numeric[valid_mask].corr(col2_numeric[valid_mask])
            print(f"  相关系数: {correlation:.8f}")

            # 尝试线性拟合
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                col1_numeric[valid_mask].dropna(),
                col2_numeric[valid_mask].dropna()
            )
            print(f"  线性拟合: col2 = {slope:.8f} * col1 + {intercept:.8f}")
            print(f"  R^2: {r_value**2:.8f}")

    # 检查所有共同因子的差异
    print(f"\n" + "=" * 100)
    print("[所有共同因子差异汇总]")
    print("=" * 100)

    factor_diff_summary = []

    for factor in common_cols:
        col1 = df1_aligned[factor]
        col2 = df2_aligned[factor]

        col1_numeric = pd.to_numeric(col1, errors='coerce')
        col2_numeric = pd.to_numeric(col2, errors='coerce')

        valid_mask = ~(col1_numeric.isna() | col2_numeric.isna())

        if valid_mask.sum() > 0:
            abs_diff = np.abs(col2_numeric - col1_numeric)
            max_diff = abs_diff[valid_mask].max()
            mean_diff = abs_diff[valid_mask].mean()
            exact_match = ((col2_numeric - col1_numeric) == 0).sum()
            pct_diff = (abs_diff[valid_mask] > 0.01).sum() / valid_mask.sum() * 100

            factor_diff_summary.append({
                'factor': factor,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'pct_diff': pct_diff,
                'exact_match_pct': exact_match / valid_mask.sum() * 100
            })

    # 排序显示
    factor_diff_summary.sort(key=lambda x: x['pct_diff'], reverse=True)

    print(f"\n差异超过1%的因子 (按差异行数比例排序):")
    print(f"{'因子名':<50} {'最大差异':>12} {'平均差异':>12} {'差异>0.01占比':>15} {'完全相同占比':>15}")
    print("-" * 105)

    count_diff_over_1pct = 0
    for item in factor_diff_summary:
        if item['pct_diff'] > 1.0:
            count_diff_over_1pct += 1
            print(f"{item['factor']:<50} {item['max_diff']:>12.6f} {item['mean_diff']:>12.6f} {item['pct_diff']:>14.2f}% {item['exact_match_pct']:>14.2f}%")

    print(f"\n总结:")
    print(f"  总共同因子数: {len(common_cols)}")
    print(f"  差异>1%的因子数: {count_diff_over_1pct}")
    print(f"  完全相同的因子数: {len(common_cols) - count_diff_over_1pct}")

    print(f"\n[分析完成]")

if __name__ == "__main__":
    # 文件路径
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")
    file1 = base_path / "factors.parquet"
    file2 = base_path / "factors-correct.parquet"

    precise_compare_factors(file1, file2)
