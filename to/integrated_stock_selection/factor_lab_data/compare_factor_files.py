"""
比较两个因子文件的差异
"""
import pandas as pd
import numpy as np
from pathlib import Path

def compare_parquet_files(file1_path, file2_path):
    """比较两个parquet文件的差异"""

    print("=" * 80)
    print("因子文件差异对比分析")
    print("=" * 80)

    # 1. 读取文件
    print(f"\n[读取文件]")
    print(f"  文件1: {file1_path}")
    print(f"  文件2: {file2_path}")

    try:
        df1 = pd.read_parquet(file1_path)
        df2 = pd.read_parquet(file2_path)
        print(f"  [OK] 文件1读取成功: {df1.shape}")
        print(f"  [OK] 文件2读取成功: {df2.shape}")
    except Exception as e:
        print(f"  [ERROR] 读取失败: {e}")
        return

    # 2. 比较基本信息
    print(f"\n[基本信息]")
    print(f"  文件1形状: {df1.shape} (行, 列)")
    print(f"  文件2形状: {df2.shape} (行, 列)")

    if df1.shape != df2.shape:
        print(f"  [WARNING] 形状不同!")
    else:
        print(f"  [OK] 形状相同")

    # 3. 比较列名
    print(f"\n[列名比较]")
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    if cols1 == cols2:
        print(f"  [OK] 列名相同 (共 {len(cols1)} 列)")
    else:
        only_in_1 = cols1 - cols2
        only_in_2 = cols2 - cols1
        if only_in_1:
            print(f"  [WARNING] 仅在文件1中: {only_in_1}")
        if only_in_2:
            print(f"  [WARNING] 仅在文件2中: {only_in_2}")

    # 4. 比较索引（如果有）
    print(f"\n[索引比较]")
    print(f"  文件1索引名: {df1.index.names}")
    print(f"  文件2索引名: {df2.index.names}")

    # 5. 对齐数据进行比较
    print(f"\n[数据内容比较]")

    # 获取共同列
    common_cols = list(cols1 & cols2)
    print(f"  共同列数: {len(common_cols)}")

    # 确保索引对齐（如果需要）
    if not df1.index.equals(df2.index):
        print(f"  [WARNING] 索引不同，尝试通过索引对齐...")
        # 使用inner join对齐
        merged = df1[common_cols].merge(
            df2[common_cols],
            left_index=True,
            right_index=True,
            suffixes=('_1', '_2'),
            how='inner',
            indicator=True
        )
        print(f"  对齐后行数: {len(merged)} (原始: 文件1={len(df1)}, 文件2={len(df2)})")
        df1_aligned = merged[[col for col in merged.columns if col.endswith('_1')]]
        df2_aligned = merged[[col for col in merged.columns if col.endswith('_2')]]
        df1_aligned.columns = [col[:-2] for col in df1_aligned.columns]
        df2_aligned.columns = [col[:-2] for col in df2_aligned.columns]
    else:
        print(f"  [OK] 索引相同")
        df1_aligned = df1[common_cols]
        df2_aligned = df2[common_cols]

    # 6. 逐列比较差异
    print(f"\n[逐列差异统计]")
    print(f"{'列名':<30} {'数据类型':<15} {'完全相同':<12} {'浮点数近似':<15} {'最大差异':<15} {'差异行数':<10}")
    print("-" * 100)

    significant_diff_columns = []

    for col in common_cols:
        col1 = df1_aligned[col]
        col2 = df2_aligned[col]

        # 统计相同行数
        exact_match = (col1 == col2).sum()
        total_rows = len(col1)

        # 浮点数近似比较（容差1e-6）
        try:
            col1_float = pd.to_numeric(col1, errors='coerce')
            col2_float = pd.to_numeric(col2, errors='coerce')

            # 计算绝对差异
            diff = np.abs(col1_float - col2_float)
            max_diff = diff.max()

            # 近似相同（容差1e-6）
            approx_match = (diff < 1e-6).sum()
            diff_rows = (diff >= 1e-6).sum()

            dtype_str = str(col1.dtype)

            # 检查是否有明显差异（大于0.01或大于相对值的1%）
            if max_diff > 0.01 or max_diff > (np.abs(col1_float).max() * 0.01):
                significant_diff_columns.append({
                    'column': col,
                    'max_diff': max_diff,
                    'diff_rows': diff_rows,
                    'mean_1': col1_float.mean(),
                    'mean_2': col2_float.mean(),
                    'dtype': dtype_str
                })

            print(f"{col:<30} {dtype_str:<15} {exact_match:<12} {approx_match:<15} {max_diff:<15.6f} {diff_rows:<10}")

        except Exception as e:
            # 非数值类型
            dtype_str = str(col1.dtype)
            print(f"{col:<30} {dtype_str:<15} {exact_match:<12} {'N/A':<15} {'N/A':<15} {'N/A':<10}")

    # 7. 显示有明显差异的列
    if significant_diff_columns:
        print(f"\n[WARNING] 发现明显差异的列（差异>0.01或相对差异>1%）:")
        print("=" * 100)
        for col_info in significant_diff_columns:
            print(f"\n列名: {col_info['column']}")
            print(f"  数据类型: {col_info['dtype']}")
            print(f"  最大差异: {col_info['max_diff']:.6f}")
            print(f"  差异行数: {col_info['diff_rows']} / {total_rows} ({col_info['diff_rows']/total_rows*100:.2f}%)")
            print(f"  文件1均值: {col_info['mean_1']:.6f}")
            print(f"  文件2均值: {col_info['mean_2']:.6f}")
            print(f"  均值差异: {abs(col_info['mean_1'] - col_info['mean_2']):.6f}")

            # 显示前几个差异样本
            col1 = df1_aligned[col_info['column']]
            col2 = df2_aligned[col_info['column']]
            col1_float = pd.to_numeric(col1, errors='coerce')
            col2_float = pd.to_numeric(col2, errors='coerce')
            diff = np.abs(col1_float - col2_float)

            # 找出最大的5个差异
            top_diff_indices = diff.nlargest(5).index
            print(f"  最大的5个差异样本:")
            for idx in top_diff_indices:
                print(f"    索引={idx}: 文件1={col1_float[idx]:.6f}, 文件2={col2_float[idx]:.6f}, 差异={diff[idx]:.6f}")
    else:
        print(f"\n[OK] 未发现明显差异（所有列的差异均≤0.01且相对差异≤1%）")

    # 8. 检查是否有NaN值的位置不同
    print(f"\n[NaN值位置比较]")
    for col in common_cols:
        col1 = df1_aligned[col]
        col2 = df2_aligned[col]

        nan1 = col1.isna()
        nan2 = col2.isna()

        # 检查NaN位置是否完全相同
        if not nan1.equals(nan2):
            only_nan_in_1 = nan1 & ~nan2
            only_nan_in_2 = nan2 & ~nan1

            print(f"  [WARNING] 列 '{col}' 的NaN位置不同:")
            print(f"    仅在文件1中为NaN: {only_nan_in_1.sum()} 行")
            print(f"    仅在文件2中为NaN: {only_nan_in_2.sum()} 行")

    print(f"\n[OK] 对比分析完成!")

if __name__ == "__main__":
    # 文件路径
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")
    file1 = base_path / "factors_correct.parquet"
    file2 = base_path / "factors.parquet"

    compare_parquet_files(file1, file2)
