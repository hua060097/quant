"""
分析 market_trend_adaptive_beta 差异最大的数据对应的股票和日期
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

def analyze_max_diff_details(file1_path, file2_path):
    """分析最大差异的详细信息"""

    print("=" * 100)
    print("market_trend_adaptive_beta 最大差异详细分析")
    print("=" * 100)

    # 读取文件
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    print(f"\n[原始文件信息]")
    print(f"  文件1: {df1.shape[0]:,} 行 × {df1.shape[1]} 列")
    print(f"  文件2: {df2.shape[0]:,} 行 × {df2.shape[1]} 列")

    # 日期过滤
    start_date = pd.Timestamp('2023-06-01')
    end_date = pd.Timestamp('2025-12-26')

    df1['_date'] = pd.to_datetime(df1['date'])
    df2['_date'] = pd.to_datetime(df2['date'])

    df1_filtered = df1[(df1['_date'] >= start_date) & (df1['_date'] <= end_date)].copy()
    df2_filtered = df2[(df2['_date'] >= start_date) & (df2['_date'] <= end_date)].copy()

    print(f"\n[日期范围过滤后]")
    print(f"  文件1: {df1_filtered.shape[0]:,} 行")
    print(f"  文件2: {df2_filtered.shape[0]:,} 行")

    # 创建复合键对齐
    df1_filtered['_key'] = df1_filtered['stock_code'].astype(str) + '_' + df1_filtered['_date'].astype(str)
    df2_filtered['_key'] = df2_filtered['stock_code'].astype(str) + '_' + df2_filtered['_date'].astype(str)

    common_keys = set(df1_filtered['_key']) & set(df2_filtered['_key'])

    df1_aligned = df1_filtered[df1_filtered['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)
    df2_aligned = df2_filtered[df2_filtered['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)

    print(f"\n[对齐后数据]")
    print(f"  对齐行数: {len(df1_aligned):,}")

    # 分析 market_trend_adaptive_beta
    factor = 'market_trend_adaptive_beta'

    if factor not in df1_aligned.columns or factor not in df2_aligned.columns:
        print(f"\n[ERROR] 因子 {factor} 不存在")
        return

    print("\n" + "=" * 100)
    print(f"因子分析: {factor}")
    print("=" * 100)

    col1 = pd.to_numeric(df1_aligned[factor], errors='coerce')
    col2 = pd.to_numeric(df2_aligned[factor], errors='coerce')

    valid_mask = ~(col1.isna() | col2.isna())

    if valid_mask.sum() == 0:
        print(f"\n[WARNING] 没有有效数据")
        return

    diff = col2 - col1
    abs_diff = np.abs(diff)

    diff_valid = diff[valid_mask]
    abs_diff_valid = abs_diff[valid_mask]

    print(f"\n[差异统计]")
    print(f"  有效比较行数: {valid_mask.sum():,}")
    print(f"  完全相同: {(diff_valid == 0).sum():,} ({(diff_valid == 0).sum()/len(diff_valid)*100:.2f}%)")
    print(f"  最大差异: {abs_diff_valid.max():.6f}")
    print(f"  平均差异: {abs_diff_valid.mean():.6f}")
    print(f"  中位数差异: {abs_diff_valid.median():.6f}")

    # 找出差异最大的前100个样本
    top_n = 100
    top_diff_indices = abs_diff_valid.nlargest(top_n).index

    print(f"\n[前{top_n}个最大差异样本的详细信息]")
    print(f"{'排名':<6} {'索引':<10} {'股票代码':<15} {'日期':<12} {'文件1值':<12} {'文件2值':<12} {'差异值':<12}")
    print("-" * 100)

    # 创建详细分析DataFrame
    max_diff_details = []

    for rank, idx in enumerate(top_diff_indices, 1):
        stock_code = df1_aligned.loc[idx, 'stock_code']
        date = df1_aligned.loc[idx, 'date']
        val1 = col1[idx]
        val2 = col2[idx]
        diff_val = diff[idx]

        max_diff_details.append({
            'rank': rank,
            'index': idx,
            'stock_code': stock_code,
            'date': date,
            'file1_value': val1,
            'file2_value': val2,
            'difference': diff_val,
            'abs_difference': abs(diff_val)
        })

        print(f"{rank:<6} {idx:<10} {stock_code:<15} {str(date):<12} {val1:<12.6f} {val2:<12.6f} {diff_val:<12.6f}")

    # 分析股票分布
    print(f"\n[前{top_n}个最大差异样本的股票分布]")
    stock_counter = Counter([d['stock_code'] for d in max_diff_details])
    print(f"{'股票代码':<15} {'出现次数':<10} {'占比':<10}")
    print("-" * 40)
    for stock, count in stock_counter.most_common(20):
        pct = count / top_n * 100
        print(f"{stock:<15} {count:<10} {pct:<10.2f}%")

    # 分析日期分布
    print(f"\n[前{top_n}个最大差异样本的日期分布]")
    date_counter = Counter([d['date'] for d in max_diff_details])
    print(f"{'日期':<12} {'出现次数':<10} {'占比':<10}")
    print("-" * 40)
    for date, count in date_counter.most_common(20):
        pct = count / top_n * 100
        print(f"{str(date):<12} {count:<10} {pct:<10.2f}%")

    # 分析差异值的分布
    print(f"\n[前{top_n}个最大差异的值分布]")
    print(f"{'差异范围':<20} {'样本数':<10} {'占比':<10}")
    print("-" * 45)

    abs_diffs = [d['abs_difference'] for d in max_diff_details]

    ranges = [
        (30, 40, "30~40"),
        (20, 30, "20~30"),
        (10, 20, "10~20"),
        (5, 10, "5~10"),
        (1, 5, "1~5"),
    ]

    for low, high, label in ranges:
        count = sum(1 for d in abs_diffs if low <= d < high)
        pct = count / top_n * 100
        print(f"{label:<20} {count:<10} {pct:<10.2f}%")

    # 分析差异>1.0的所有样本
    large_diff_threshold = 1.0
    large_diff_mask = abs_diff_valid > large_diff_threshold
    large_diff_count = large_diff_mask.sum()

    print(f"\n[所有差异>{large_diff_threshold}的样本分析]")
    print(f"  总样本数: {large_diff_count:,} ({large_diff_count/len(diff_valid)*100:.2f}%)")

    if large_diff_count > 0:
        # 获取所有大差异样本的索引
        all_large_diff_indices = abs_diff_valid[large_diff_mask].index

        # 分析股票分布
        print(f"\n  股票分布 (Top 20):")
        large_diff_stocks = [df1_aligned.loc[idx, 'stock_code'] for idx in all_large_diff_indices]
        large_stock_counter = Counter(large_diff_stocks)
        for stock, count in large_stock_counter.most_common(20):
            pct = count / large_diff_count * 100
            print(f"    {stock:<15} {count:>8,} ({pct:>6.2f}%)")

        # 分析日期分布
        print(f"\n  日期分布 (Top 20):")
        large_diff_dates = [df1_aligned.loc[idx, 'date'] for idx in all_large_diff_indices]
        large_date_counter = Counter(large_diff_dates)
        for date, count in large_date_counter.most_common(20):
            pct = count / large_diff_count * 100
            print(f"    {str(date):<12} {count:>8,} ({pct:>6.2f}%)")

        # 分析差异模式
        print(f"\n  差异方向分析:")
        positive_diff = (diff_valid[large_diff_mask] > 0).sum()
        negative_diff = (diff_valid[large_diff_mask] < 0).sum()
        print(f"    文件2 > 文件1: {positive_diff:,} ({positive_diff/large_diff_count*100:.2f}%)")
        print(f"    文件2 < 文件1: {negative_diff:,} ({negative_diff/large_diff_count*100:.2f}%)")

        # 显示典型的差异模式
        print(f"\n  典型差异模式:")
        if positive_diff > 0:
            positive_vals = diff_valid[large_diff_mask & (diff_valid > 0)]
            print(f"    文件2 > 文件1 的范围: {positive_vals.min():.2f} 至 {positive_vals.max():.2f}")
        if negative_diff > 0:
            negative_vals = diff_valid[large_diff_mask & (diff_valid < 0)]
            print(f"    文件2 < 文件1 的范围: {negative_vals.min():.2f} 至 {negative_vals.max():.2f}")

    # 检查是否有特定的值反复出现
    print(f"\n[文件1和文件2中的特殊值检查]")

    # 文件1中的特殊值
    val1_counts = col1[valid_mask].value_counts()
    print(f"\n  文件1中出现频率最高的值 (Top 10):")
    for val, count in val1_counts.head(10).items():
        pct = count / valid_mask.sum() * 100
        print(f"    {val:>12.6f}  {count:>10,} ({pct:>6.2f}%)")

    # 文件2中的特殊值
    val2_counts = col2[valid_mask].value_counts()
    print(f"\n  文件2中出现频率最高的值 (Top 10):")
    for val, count in val2_counts.head(10).items():
        pct = count / valid_mask.sum() * 100
        print(f"    {val:>12.6f}  {count:>10,} ({pct:>6.2f}%)")

    print(f"\n[分析完成]")

if __name__ == "__main__":
    # 文件路径
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")
    file1 = base_path / "factors.parquet"
    file2 = base_path / "factors-correct.parquet"

    analyze_max_diff_details(file1, file2)
