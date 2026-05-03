"""
分析 parquet 文件中的数据质量
检查股票指标数据是否正常
"""

import pandas as pd
import numpy as np

def analyze_parquet_data():
    file_path = r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\features\daily\20260205.parquet"

    print("=" * 80)
    print("Parquet 文件数据分析")
    print("=" * 80)
    print(f"\n文件路径: {file_path}")

    # 1. 读取文件
    print("\n正在读取文件...")
    try:
        df = pd.read_parquet(file_path)
        print("[OK] 文件读取成功")
    except Exception as e:
        print(f"[ERROR] 文件读取失败: {e}")
        return

    # 2. 基本信息
    print(f"\n{'='*80}")
    print("基本信息")
    print(f"{'='*80}")
    print(f"总行数: {len(df):,}")
    print(f"总列数: {len(df.columns):,}")
    print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 3. 列名和数据类型
    print(f"\n{'='*80}")
    print("列名和数据类型")
    print(f"{'='*80}")

    col_info = []
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = null_count / len(df) * 100
        col_info.append({
            '列名': col,
            '数据类型': str(dtype),
            '缺失值': null_count,
            '缺失率(%)': f"{null_pct:.2f}"
        })

    col_df = pd.DataFrame(col_info)
    print(col_df.to_string(index=False))

    # 4. 检查关键列
    print(f"\n{'='*80}")
    print("关键列检查")
    print(f"{'='*80}")

    key_columns = ['stock_code', 'date']
    for col in key_columns:
        if col in df.columns:
            print(f"\n[OK] {col}: 存在")
            if col == 'stock_code':
                unique_stocks = df[col].nunique()
                print(f"  唯一股票数: {unique_stocks:,}")
            elif col == 'date':
                unique_dates = df[col].nunique()
                print(f"  唯一日期数: {unique_dates}")
                if unique_dates > 0:
                    min_date = df[col].min()
                    max_date = df[col].max()
                    print(f"  日期范围: {min_date} 到 {max_date}")
        else:
            print(f"\n[WARNING] {col}: 不存在")

    # 5. 缺失值分析
    print(f"\n{'='*80}")
    print("缺失值分析")
    print(f"{'='*80}")

    total_cells = len(df) * len(df.columns)
    total_missing = df.isnull().sum().sum()
    missing_pct = total_missing / total_cells * 100

    print(f"总单元格数: {total_cells:,}")
    print(f"缺失单元格数: {total_missing:,} ({missing_pct:.2f}%)")

    # 每列缺失值统计
    missing_by_col = df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)

    if len(missing_by_col) > 0:
        print(f"\n有缺失值的列 ({len(missing_by_col)} 列):")
        for col, count in missing_by_col.head(20).items():
            pct = count / len(df) * 100
            print(f"  {col:<40}: {count:>8,} ({pct:>6.2f}%)")
    else:
        print("\n[OK] 无缺失值")

    # 6. 异常值分析（Inf, -Inf, NaN）
    print(f"\n{'='*80}")
    print("异常值分析")
    print(f"{'='*80}")

    # 只检查数值类型的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols[:20]:  # 只检查前20个数值列
        inf_count = np.isinf(df[col]).sum()
        nan_count = df[col].isnull().sum()
        total_values = df[col].notna().sum()

        if inf_count > 0 or nan_count > 0:
            print(f"  {col:<40}: Inf={inf_count:>6,}, NaN={nan_count:>6,}, 有效值={total_values:>8,}")

    # 7. 数值列统计
    print(f"\n{'='*80}")
    print("数值列统计（前20个）")
    print(f"{'='*80}")

    stats_summary = []
    for col in numeric_cols[:20]:
        if df[col].notna().sum() > 0:
            stats = df[col].describe()
            stats_summary.append({
                '列名': col,
                '最小值': f"{stats['min']:.4f}",
                '最大值': f"{stats['max']:.4f}",
                '均值': f"{stats['mean']:.4f}",
                '标准差': f"{stats['std']:.4f}",
                '有效值': f"{int(stats['count']):,}"
            })

    if stats_summary:
        stats_df = pd.DataFrame(stats_summary)
        print(stats_df.to_string(index=False))

    # 8. 检查极端值（超出合理范围）
    print(f"\n{'='*80}")
    print("极端值检查（异常大的值）")
    print(f"{'='*80}")

    for col in numeric_cols[:20]:
        if df[col].notna().sum() > 0:
            # 检查是否有超过10的值（对于大部分因子来说这可能是异常）
            large_values = (df[col].abs() > 10).sum()
            if large_values > 0:
                max_val = df[col].max()
                min_val = df[col].min()
                print(f"  {col:<40}: 最大值={max_val:.2f}, 最小值={min_val:.2f}, >10的值有{large_values:>6,}个")

    # 9. 股票级别检查
    if 'stock_code' in df.columns:
        print(f"\n{'='*80}")
        print("股票级别检查")
        print(f"{'='*80}")

        # 每只股票的数据量
        stock_counts = df['stock_code'].value_counts()
        print(f"每只股票的数据量统计:")
        print(f"  最小值: {stock_counts.min()}")
        print(f"  最大值: {stock_counts.max()}")
        print(f"  平均值: {stock_counts.mean():.2f}")
        print(f"  中位数: {stock_counts.median()}")

        # 检查是否有股票数据量异常
        low_count_stocks = stock_counts[stock_counts < stock_counts.median()]
        if len(low_count_stocks) > 0:
            print(f"\n数据量低于中位数的股票数: {len(low_count_stocks)}")
            print(f"示例（前5个）:")
            for stock, count in low_count_stocks.head().items():
                print(f"  {stock}: {count} 条记录")

    # 10. 数据一致性检查
    print(f"\n{'='*80}")
    print("数据一致性检查")
    print(f"{'='*80}")

    # 检查是否有重复行
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        print(f"[WARNING] 发现重复行: {duplicate_rows} 条")
    else:
        print(f"[OK] 无重复行")

    # 检查是否所有股票都有完整的数据
    if 'stock_code' in df.columns and 'date' in df.columns:
        stock_date_counts = df.groupby('stock_code')['date'].count()
        expected_count = df['date'].nunique()
        incomplete_stocks = stock_date_counts[stock_date_counts < expected_count]

        if len(incomplete_stocks) > 0:
            print(f"[WARNING] 数据不完整的股票数: {len(incomplete_stocks)}")
            print(f"  期望每只股票有 {expected_count} 条记录")
            print(f"  示例（前5个）:")
            for stock, count in incomplete_stocks.head().items():
                print(f"    {stock}: {count} 条记录")
        else:
            print(f"[OK] 所有股票数据完整")

    # 11. 数据预览
    print(f"\n{'='*80}")
    print("数据预览（前5行）")
    print(f"{'='*80}")

    display_cols = df.columns[:10].tolist()  # 只显示前10列
    print(df[display_cols].head())

    # 12. 总结
    print(f"\n{'='*80}")
    print("数据质量总结")
    print(f"{'='*80}")

    issues = []

    # 检查1: 缺失值
    if total_missing > 0:
        issues.append(f"存在 {total_missing:,} 个缺失值 ({missing_pct:.2f}%)")

    # 检查2: 异常值
    total_inf = 0
    for col in numeric_cols:
        total_inf += np.isinf(df[col]).sum()
    if total_inf > 0:
        issues.append(f"存在 {total_inf:,} 个无穷值")

    # 检查3: 极端值
    extreme_count = 0
    for col in numeric_cols[:20]:
        extreme_count += (df[col].abs() > 100).sum()
    if extreme_count > 0:
        issues.append(f"存在 {extreme_count:,} 个极端值（绝对值>100）")

    # 检查4: 重复行
    if duplicate_rows > 0:
        issues.append(f"存在 {duplicate_rows} 条重复行")

    if len(issues) == 0:
        print("[OK] 数据质量良好，未发现明显问题")
    else:
        print("发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    print(f"\n{'='*80}")
    print("分析完成")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    analyze_parquet_data()
