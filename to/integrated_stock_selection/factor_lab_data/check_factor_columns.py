"""
检查目标因子在两个文件中的存在情况
"""
import pandas as pd
from pathlib import Path

def check_factor_existence(file1_path, file2_path):
    """检查因子列的存在情况"""

    print("=" * 100)
    print("检查因子列存在情况")
    print("=" * 100)

    # 读取文件
    print("\n[读取文件]")
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    print(f"  文件1 (副本): {df1.shape[1]} 列")
    print(f"  文件2 (当前): {df2.shape[1]} 列")

    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    common_cols = cols1 & cols2
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1

    print(f"\n[列统计]")
    print(f"  共同列: {len(common_cols)}")
    print(f"  仅在文件1: {len(only_in_1)}")
    print(f"  仅在文件2: {len(only_in_2)}")

    # 检查目标因子
    target_factors = [
        'sector_broken_resilience_alpha',
        'market_trend_adaptive_beta'
    ]

    print(f"\n[目标因子检查]")
    for factor in target_factors:
        in_file1 = factor in cols1
        in_file2 = factor in cols2
        in_common = factor in common_cols

        print(f"\n  {factor}:")
        print(f"    在文件1: {'是' if in_file1 else '否'}")
        print(f"    在文件2: {'是' if in_file2 else '否'}")
        print(f"    在共同列: {'是' if in_common else '否'}")

        if in_file1 and not in_file2:
            print(f"    状态: 仅在文件1（副本）中存在")
            # 显示该列的基本信息
            col_data = df1[factor]
            print(f"    数据类型: {col_data.dtype}")
            print(f"    非空值: {col_data.notna().sum()} / {len(col_data)}")
            print(f"    均值: {col_data.mean():.6f}")
            print(f"    范围: [{col_data.min():.6f}, {col_data.max():.6f}]")
        elif in_file2 and not in_file1:
            print(f"    状态: 仅在文件2（当前）中存在")
            # 显示该列的基本信息
            col_data = df2[factor]
            print(f"    数据类型: {col_data.dtype}")
            print(f"    非空值: {col_data.notna().sum()} / {len(col_data)}")
            print(f"    均值: {col_data.mean():.6f}")
            print(f"    范围: [{col_data.min():.6f}, {col_data.max():.6f}]")
        elif in_common:
            print(f"    状态: 在两个文件中都存在，可以对比")

    # 显示仅在文件1的列
    if only_in_1:
        print(f"\n[仅在文件1（副本）中的列]")
        for col in sorted(only_in_1):
            print(f"  - {col}")

    # 显示仅在文件2的列
    if only_in_2:
        print(f"\n[仅在文件2（当前）中的列]")
        for col in sorted(only_in_2):
            print(f"  - {col}")

    # 对于 market_trend_adaptive_beta (如果存在)
    if 'market_trend_adaptive_beta' in common_cols:
        print(f"\n[详细分析 market_trend_adaptive_beta]")
        col1 = df1['market_trend_adaptive_beta']
        col2 = df2['market_trend_adaptive_beta']

        # 确保索引对齐
        if df1.index.equals(df2.index):
            print(f"  索引相同")
            aligned1 = col1
            aligned2 = col2
        else:
            print(f"  索引不同，通过merge对齐...")
            # 只取这个列进行merge
            temp_df1 = pd.DataFrame({'value1': col1}, index=col1.index)
            temp_df2 = pd.DataFrame({'value2': col2}, index=col2.index)

            aligned = temp_df1.merge(
                temp_df2,
                left_index=True,
                right_index=True,
                how='inner'
            )
            aligned1 = aligned['value1']
            aligned2 = aligned['value2']
            print(f"  对齐后行数: {len(aligned)}")

        # 计算差异
        diff = aligned2 - aligned1
        abs_diff = abs(diff)

        print(f"  文件1统计:")
        print(f"    均值: {aligned1.mean():.6f}")
        print(f"    标准差: {aligned1.std():.6f}")
        print(f"    最小值: {aligned1.min():.6f}")
        print(f"    最大值: {aligned1.max():.6f}")

        print(f"  文件2统计:")
        print(f"    均值: {aligned2.mean():.6f}")
        print(f"    标准差: {aligned2.std():.6f}")
        print(f"    最小值: {aligned2.min():.6f}")
        print(f"    最大值: {aligned2.max():.6f}")

        print(f"  差异分析:")
        print(f"    完全相同: {(diff == 0).sum()} / {len(diff)} ({(diff == 0).sum()/len(diff)*100:.2f}%)")
        print(f"    最大差异: {abs_diff.max():.6f}")
        print(f"    平均差异: {abs_diff.mean():.6f}")
        print(f"    差异>1.0: {(abs_diff > 1.0).sum()} ({(abs_diff > 1.0).sum()/len(diff)*100:.2f}%)")
        print(f"    差异>10.0: {(abs_diff > 10.0).sum()} ({(abs_diff > 10.0).sum()/len(diff)*100:.2f}%)")
        print(f"    差异>50.0: {(abs_diff > 50.0).sum()} ({(abs_diff > 50.0).sum()/len(diff)*100:.2f}%)")

        # 显示大差异样本
        print(f"  最大差异的前10个样本:")
        top_diff_indices = abs_diff.nlargest(10).index
        for idx in top_diff_indices:
            print(f"    索引={idx}: 文件1={aligned1[idx]:.6f}, 文件2={aligned2[idx]:.6f}, 差异={diff[idx]:.6f}")

if __name__ == "__main__":
    # 文件路径
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")
    file1 = base_path / "factors - 副本.parquet"
    file2 = base_path / "factors.parquet"

    check_factor_existence(file1, file2)
