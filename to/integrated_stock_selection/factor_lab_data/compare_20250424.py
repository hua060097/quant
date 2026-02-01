"""
对比 factors-correct.parquet 和 factors-1.parquet 在 2025-04-24 至 2025-04-29 期间的因子差异
并打印出具体的数值对比
"""
import pandas as pd
import numpy as np
from pathlib import Path

def compare_factors_in_date_range(file1_path, file2_path, start_date, end_date, top_n=20):
    """
    对比两个因子文件在指定日期范围内的差异

    参数:
        file1_path: 文件1路径
        file2_path: 文件2路径
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        top_n: 显示差异最大的前N个因子
    """

    print("=" * 120)
    print(f"对比因子文件在 {start_date} 至 {end_date} 期间的差异")
    print("=" * 120)

    # 读取文件
    print(f"\n[读取文件]")
    print(f"  文件1: {file1_path.name}")
    print(f"  文件2: {file2_path.name}")

    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    print(f"  文件1形状: {df1.shape}")
    print(f"  文件2形状: {df2.shape}")

    # 日期过滤
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    df1['_date'] = pd.to_datetime(df1['date'])
    df2['_date'] = pd.to_datetime(df2['date'])

    df1_filtered = df1[(df1['_date'] >= start_ts) & (df1['_date'] <= end_ts)].copy()
    df2_filtered = df2[(df2['_date'] >= start_ts) & (df2['_date'] <= end_ts)].copy()

    print(f"\n[日期范围过滤]")
    print(f"  目标范围: {start_ts.date()} 至 {end_ts.date()}")
    print(f"  文件1过滤后: {len(df1_filtered):,} 行")
    print(f"  文件2过滤后: {len(df2_filtered):,} 行")

    # 创建复合键对齐
    df1_filtered['_key'] = df1_filtered['stock_code'].astype(str) + '_' + df1_filtered['_date'].astype(str)
    df2_filtered['_key'] = df2_filtered['stock_code'].astype(str) + '_' + df2_filtered['_date'].astype(str)

    common_keys = set(df1_filtered['_key']) & set(df2_filtered['_key'])

    df1_aligned = df1_filtered[df1_filtered['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)
    df2_aligned = df2_filtered[df2_filtered['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)

    print(f"\n[数据对齐]")
    print(f"  共同的(stock_code, date)组合: {len(common_keys):,}")

    # 找共同因子
    exclude_cols = {'stock_code', 'date', '_date', '_key'}
    cols1 = set(df1_aligned.columns) - exclude_cols
    cols2 = set(df2_aligned.columns) - exclude_cols
    common_cols = sorted(list(cols1 & cols2))

    print(f"  共同因子数量: {len(common_cols)}")

    # 分析每个因子的差异
    factor_stats = []

    for factor in common_cols:
        col1 = pd.to_numeric(df1_aligned[factor], errors='coerce')
        col2 = pd.to_numeric(df2_aligned[factor], errors='coerce')

        valid_mask = ~(col1.isna() | col2.isna())

        if valid_mask.sum() > 0:
            diff = col2 - col1
            abs_diff = np.abs(diff)

            pct_diff_gt_001 = (abs_diff[valid_mask] > 0.01).sum() / valid_mask.sum() * 100
            pct_diff_gt_01 = (abs_diff[valid_mask] > 0.1).sum() / valid_mask.sum() * 100
            pct_diff_gt_1 = (abs_diff[valid_mask] > 1.0).sum() / valid_mask.sum() * 100

            max_diff = abs_diff[valid_mask].max()
            mean_diff = abs_diff[valid_mask].mean()
            median_diff = abs_diff[valid_mask].median()

            exact_match_pct = ((diff[valid_mask]) == 0).sum() / valid_mask.sum() * 100

            # 计算相关系数
            correlation = col1[valid_mask].corr(col2[valid_mask])

            factor_stats.append({
                'factor': factor,
                'pct_diff_gt_001': pct_diff_gt_001,
                'pct_diff_gt_01': pct_diff_gt_01,
                'pct_diff_gt_1': pct_diff_gt_1,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'median_diff': median_diff,
                'exact_match_pct': exact_match_pct,
                'correlation': correlation,
                'valid_count': valid_mask.sum()
            })

    # 转为DataFrame并排序
    stats_df = pd.DataFrame(factor_stats)
    stats_df = stats_df.sort_values('pct_diff_gt_001', ascending=False)

    # 打印统计摘要
    print(f"\n{'=' * 120}")
    print(f"因子差异统计摘要（按差异>0.01的比例排序）")
    print(f"{'=' * 120}")
    print(f"{'因子名':<45} {'差异>0.01':>10} {'差异>0.1':>10} {'差异>1.0':>10} {'最大差异':>10} {'平均差异':>10} {'相关系数':>10}")
    print("-" * 120)

    for _, row in stats_df.head(20).iterrows():
        print(f"{row['factor']:<45} {row['pct_diff_gt_001']:>9.2f}% {row['pct_diff_gt_01']:>9.2f}% {row['pct_diff_gt_1']:>9.2f}% {row['max_diff']:>10.4f} {row['mean_diff']:>10.6f} {row['correlation']:>10.4f}")

    # 找出差异最大的因子，并详细对比
    print(f"\n{'=' * 120}")
    print(f"差异最大的前{top_n}个因子的详细数值对比")
    print(f"{'=' * 120}")

    for rank, (_, row) in enumerate(stats_df.head(top_n).iterrows(), 1):
        factor_name = row['factor']

        print(f"\n{'#' * 120}")
        print(f"#{rank}. 因子: {factor_name}")
        print(f"{'#' * 120}")
        print(f"  统计信息:")
        print(f"    差异>0.01占比: {row['pct_diff_gt_001']:.2f}%")
        print(f"    差异>0.1占比:  {row['pct_diff_gt_01']:.2f}%")
        print(f"    差异>1.0占比:  {row['pct_diff_gt_1']:.2f}%")
        print(f"    最大差异:     {row['max_diff']:.6f}")
        print(f"    平均差异:     {row['mean_diff']:.6f}")
        print(f"    相关系数:     {row['correlation']:.6f}")
        print(f"    完全相同占比: {row['exact_match_pct']:.2f}%")

        # 获取该因子的数据
        col1 = pd.to_numeric(df1_aligned[factor_name], errors='coerce')
        col2 = pd.to_numeric(df2_aligned[factor_name], errors='coerce')

        valid_mask = ~(col1.isna() | col2.isna())
        abs_diff = np.abs(col2 - col1)

        # 找出差异最大的样本
        top_diff_indices = abs_diff[valid_mask].nlargest(13).index

        print(f"\n  差异最大的10个样本:")
        print(f"  {'排名':<6} {'股票代码':<15} {'日期':<12} {'文件1值':<15} {'文件2值':<15} {'差异':<12} {'绝对差异':<12}")
        print("  " + "-" * 100)

        for i, idx in enumerate(top_diff_indices, 1):
            stock_code = df1_aligned.loc[idx, 'stock_code']
            date = df1_aligned.loc[idx, '_date']
            val1 = col1[idx]
            val2 = col2[idx]
            diff_val = col2[idx] - col1[idx]
            abs_diff_val = abs_diff[idx]

            date_str = str(date.date())
            print(f"  {i:<6} {stock_code:<15} {date_str:<12} {val1:<15.6f} {val2:<15.6f} {diff_val:<12.6f} {abs_diff_val:<12.6f}")

        # 按日期统计差异
        df1_aligned['_diff'] = abs_diff
        df1_aligned['_valid'] = valid_mask

        date_stats = df1_aligned[df1_aligned['_valid']].groupby('_date').agg({
            '_diff': ['mean', 'max', 'count'],
            factor_name: ['mean', 'std']
        }).round(6)

        date_stats.columns = ['mean_diff', 'max_diff', 'count', 'mean_val', 'std_val']
        date_stats = date_stats.sort_values('mean_diff', ascending=False)

        print(f"\n  按日期的差异统计:")
        print(f"  {'日期':<12} {'平均差异':<12} {'最大差异':<12} {'样本数':<10} {'因子均值':<12} {'因子标准差':<12}")
        print("  " + "-" * 80)

        # for date, stat_row in date_stats.iterrows():
        #     date_str = str(date.date())
        #     print(f"  {date_str:<12} {stat_row['mean_diff']:<12.6f} {stat_row['max_diff']:<12.6f} {int(stat_row['count']):<10} {stat_row['mean_val']:<12.6f} {stat_row['std_val']:<12.6f}")

    print(f"\n{'=' * 120}")
    print(f"对比完成")
    print(f"{'=' * 120}")

if __name__ == "__main__":
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")

    file1 = base_path / "factors_correct.parquet"
    file2 = base_path / "factors.parquet"

    # 对比 2025-04-24 至 2025-04-29 的数据
    compare_factors_in_date_range(
        file1,
        file2,
        start_date='20250727',
        end_date='20251031',
        top_n=10  # 显示差异最大的前10个因子
    )
