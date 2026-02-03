"""
统计 factors.parquet 文件中所有因子的统计信息
日期范围: 20230601 至 20251231
统计指标: 最大值、最小值、均值、中位数、标准差
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_factors_statistics(file_path, start_date, end_date):
    """
    分析因子文件在指定日期范围内的统计信息

    参数:
        file_path: 因子文件路径
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
    """

    print("=" * 120)
    print(f"统计因子文件在 {start_date} 至 {end_date} 期间的统计信息")
    print("=" * 120)

    # 读取文件
    print(f"\n[读取文件]")
    print(f"  文件路径: {file_path.name}")

    df = pd.read_parquet(file_path)

    print(f"  数据形状: {df.shape}")
    print(f"  列名: {list(df.columns)}")

    # 日期过滤
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    df['_date'] = pd.to_datetime(df['date'])

    df_filtered = df[(df['_date'] >= start_ts) & (df['_date'] <= end_ts)].copy()

    print(f"\n[日期范围过滤]")
    print(f"  目标范围: {start_ts.date()} 至 {end_ts.date()}")
    print(f"  过滤后行数: {len(df_filtered):,}")
    print(f"  过滤后数据日期范围: {df_filtered['_date'].min().date()} 至 {df_filtered['_date'].max().date()}")

    # 获取因子列（排除 stock_code, date, _date）
    exclude_cols = {'stock_code', 'date', '_date'}
    factor_columns = [col for col in df_filtered.columns if col not in exclude_cols]

    print(f"\n[因子信息]")
    print(f"  因子数量: {len(factor_columns)}")

    # 计算每个因子的统计信息
    factor_stats = []

    print(f"\n[计算统计信息]")
    for i, factor in enumerate(factor_columns, 1):
        if i % 50 == 0:
            print(f"  已处理 {i}/{len(factor_columns)} 个因子...")

        # 转换为数值类型
        factor_series = pd.to_numeric(df_filtered[factor], errors='coerce')

        # 计算统计指标
        valid_data = factor_series.dropna()

        if len(valid_data) > 0:
            stats = {
                'factor': factor,
                'count': len(valid_data),
                'max': valid_data.max(),
                'min': valid_data.min(),
                'mean': valid_data.mean(),
                'median': valid_data.median(),
                'std': valid_data.std(),
                'nan_count': factor_series.isna().sum(),
                'nan_pct': factor_series.isna().sum() / len(factor_series) * 100
            }
            factor_stats.append(stats)
        else:
            # 如果没有有效数据，记录 NaN
            factor_stats.append({
                'factor': factor,
                'count': 0,
                'max': np.nan,
                'min': np.nan,
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'nan_count': len(factor_series),
                'nan_pct': 100.0
            })

    print(f"  已完成处理 {len(factor_columns)} 个因子")

    # 转换为 DataFrame
    stats_df = pd.DataFrame(factor_stats)

    # 打印统计摘要
    print(f"\n{'=' * 120}")
    print(f"因子统计信息汇总（按因子名称排序）")
    print(f"{'=' * 120}")
    print(f"{'因子名':<45} {'样本数':>12} {'最大值':>15} {'最小值':>15} {'均值':>15} {'中位数':>15} {'标准差':>15} {'缺失%':>10}")
    print("-" * 140)

    for _, row in stats_df.iterrows():
        factor_name = row['factor']
        count = row['count']

        if count > 0:
            max_val = row['max']
            min_val = row['min']
            mean_val = row['mean']
            median_val = row['median']
            std_val = row['std']
            nan_pct = row['nan_pct']

            print(f"{factor_name:<45} {count:>12,} {max_val:>15.6f} {min_val:>15.6f} {mean_val:>15.6f} {median_val:>15.6f} {std_val:>15.6f} {nan_pct:>9.2f}%")
        else:
            print(f"{factor_name:<45} {count:>12,} {'NaN':>15} {'NaN':>15} {'NaN':>15} {'NaN':>15} {'NaN':>15} {row['nan_pct']:>9.2f}%")

    # 打印额外的统计摘要
    print(f"\n{'=' * 120}")
    print(f"额外统计信息")
    print(f"{'=' * 120}")

    # 按均值排序的前10和后10因子
    valid_stats = stats_df[stats_df['count'] > 0].copy()

    if len(valid_stats) > 0:
        print(f"\n[均值最大的前10个因子]")
        top10_mean = valid_stats.nlargest(10, 'mean')[['factor', 'mean', 'std', 'count']]
        print(f"{'因子名':<45} {'均值':>15} {'标准差':>15} {'样本数':>12}")
        print("-" * 90)
        for _, row in top10_mean.iterrows():
            print(f"{row['factor']:<45} {row['mean']:>15.6f} {row['std']:>15.6f} {row['count']:>12,}")

        print(f"\n[均值最小的后10个因子]")
        bottom10_mean = valid_stats.nsmallest(10, 'mean')[['factor', 'mean', 'std', 'count']]
        print(f"{'因子名':<45} {'均值':>15} {'标准差':>15} {'样本数':>12}")
        print("-" * 90)
        for _, row in bottom10_mean.iterrows():
            print(f"{row['factor']:<45} {row['mean']:>15.6f} {row['std']:>15.6f} {row['count']:>12,}")

        print(f"\n[标准差最大的前10个因子（波动最大）]")
        top10_std = valid_stats.nlargest(10, 'std')[['factor', 'std', 'mean', 'count']]
        print(f"{'因子名':<45} {'标准差':>15} {'均值':>15} {'样本数':>12}")
        print("-" * 90)
        for _, row in top10_std.iterrows():
            print(f"{row['factor']:<45} {row['std']:>15.6f} {row['mean']:>15.6f} {row['count']:>12,}")

        print(f"\n[标准差最小的后10个因子（波动最小）]")
        bottom10_std = valid_stats.nsmallest(10, 'std')[['factor', 'std', 'mean', 'count']]
        print(f"{'因子名':<45} {'标准差':>15} {'均值':>15} {'样本数':>12}")
        print("-" * 90)
        for _, row in bottom10_std.iterrows():
            print(f"{row['factor']:<45} {row['std']:>15.6f} {row['mean']:>15.6f} {row['count']:>12,}")

        print(f"\n[缺失值最多的前10个因子]")
        top10_nan = stats_df.nlargest(10, 'nan_pct')[['factor', 'nan_pct', 'nan_count', 'count']]
        print(f"{'因子名':<45} {'缺失比例':>12} {'缺失数量':>12} {'总样本数':>12}")
        print("-" * 85)
        for _, row in top10_nan.iterrows():
            total_count = row['count'] + row['nan_count']
            print(f"{row['factor']:<45} {row['nan_pct']:>11.2f}% {row['nan_count']:>12,} {total_count:>12,}")

    print(f"\n{'=' * 120}")
    print(f"统计完成")
    print(f"{'=' * 120}")

    return stats_df

if __name__ == "__main__":
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")

    factor_file = base_path / "factors.parquet"

    # 统计 2023-06-01 至 2025-12-31 的因子统计信息
    analyze_factors_statistics(
        factor_file,
        start_date='20230601',
        end_date='20251231'
    )
