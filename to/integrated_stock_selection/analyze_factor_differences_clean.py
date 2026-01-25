#!/usr/bin/env python3
"""
分析factor_lab_data/factors1.parquet和factors.parquet中2023年数据的区别
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def analyze_factor_differences():
    """分析两个因子文件的差异"""

    # 文件路径
    file1 = "factor_lab_data/factors1.parquet"
    file2 = "factor_lab_data/factors.parquet"

    print("=" * 80)
    print("因子文件差异分析报告")
    print("=" * 80)

    # 检查文件是否存在
    if not os.path.exists(file1):
        print(f"[ERROR] 文件不存在: {file1}")
        return

    if not os.path.exists(file2):
        print(f"[ERROR] 文件不存在: {file2}")
        return

    print(f"[INFO] 分析文件:")
    print(f"  文件1: {file1}")
    print(f"  文件2: {file2}")
    print()

    # 读取数据
    print("[INFO] 正在读取数据...")
    try:
        df1 = pd.read_parquet(file1)
        df2 = pd.read_parquet(file2)
        print(f"[SUCCESS] 文件1: {len(df1):,} 条记录, {len(df1.columns)} 个字段")
        print(f"[SUCCESS] 文件2: {len(df2):,} 条记录, {len(df2.columns)} 个字段")
    except Exception as e:
        print(f"[ERROR] 读取数据失败: {e}")
        return

    # 检查必需列
    required_cols = ['date', 'stock_code']
    for col in required_cols:
        if col not in df1.columns:
            print(f"[ERROR] 文件1缺少列: {col}")
        if col not in df2.columns:
            print(f"[ERROR] 文件2缺少列: {col}")

    if not all(col in df1.columns for col in required_cols):
        return

    # 转换日期列
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])

    # 过滤2023年数据
    df1_2023 = df1[(df1['date'] >= '2023-01-01') & (df1['date'] <= '2023-12-31')].copy()
    df2_2023 = df2[(df2['date'] >= '2023-01-01') & (df2['date'] <= '2023-12-31')].copy()

    print(f"[STATS] 2023年数据统计:")
    print(f"  文件1: {len(df1_2023):,} 条记录")
    print(f"  文件2: {len(df2_2023):,} 条记录")

    if len(df1_2023) == 0 or len(df2_2023) == 0:
        print("[ERROR] 没有2023年数据可供对比")
        return

    # 检查日期范围
    date_range1 = (df1_2023['date'].min(), df1_2023['date'].max())
    date_range2 = (df2_2023['date'].min(), df2_2023['date'].max())

    print(f"  文件1日期范围: {date_range1[0].strftime('%Y-%m-%d')} ~ {date_range1[1].strftime('%Y-%m-%d')}")
    print(f"  文件2日期范围: {date_range2[0].strftime('%Y-%m-%d')} ~ {date_range2[1].strftime('%Y-%m-%d')}")
    print()

    # 找出重叠的日期
    dates1 = set(df1_2023['date'].unique())
    dates2 = set(df2_2023['date'].unique())
    overlap_dates = dates1.intersection(dates2)

    print(f"[DATE] 重叠日期分析:")
    print(f"  文件1独有日期: {len(dates1 - dates2)} 天")
    print(f"  文件2独有日期: {len(dates2 - dates1)} 天")
    print(f"  重叠日期: {len(overlap_dates)} 天")
    print(f"  重叠日期范围: {min(overlap_dates)} ~ {max(overlap_dates)}")
    print()

    if len(overlap_dates) == 0:
        print("[ERROR] 没有重叠日期，无法比较因子值")
        return

    # 过滤重叠日期的数据
    df1_overlap = df1_2023[df1_2023['date'].isin(overlap_dates)].copy()
    df2_overlap = df2_2023[df2_2023['date'].isin(overlap_dates)].copy()

    print(f"[DATA] 重叠数据统计:")
    print(f"  文件1重叠数据: {len(df1_overlap):,} 条记录")
    print(f"  文件2重叠数据: {len(df2_overlap):,} 条记录")
    print()

    # 对齐数据
    print("[PROCESS] 正在对齐数据...")
    # 按日期和股票代码对齐
    df1_aligned = df1_overlap.sort_values(['date', 'stock_code']).reset_index(drop=True)
    df2_aligned = df2_overlap.sort_values(['date', 'stock_code']).reset_index(drop=True)

    # 检查对齐后的数据量是否一致
    if len(df1_aligned) != len(df2_aligned):
        print("[WARNING] 警告：对齐后数据量不一致")
        min_len = min(len(df1_aligned), len(df2_aligned))
        df1_aligned = df1_aligned.head(min_len)
        df2_aligned = df2_aligned.head(min_len)
        print(f"  截断到前 {min_len} 条记录进行比较")

    # 比较数据一致性
    if 'stock_code' in df1_aligned.columns and 'date' in df1_aligned.columns:
        mask = (df1_aligned['stock_code'] == df2_aligned['stock_code']) & (df1_aligned['date'] == df2_aligned['date'])
        if not mask.all():
            print("[WARNING] 警告：存在日期或股票代码不匹配的对齐数据")
            df1_aligned = df1_aligned[mask]
            df2_aligned = df2_aligned[mask]

    # 获取共同因子列
    cols1 = set(df1_aligned.columns) - {'date', 'stock_code'}
    cols2 = set(df2_aligned.columns) - {'date', 'stock_code'}
    common_factors = cols1.intersection(cols2)

    print(f"[FACTORS] 共同因子数量: {len(common_factors)}")
    print(f"  文件1独有因子: {len(cols1 - common_factors)} 个")
    print(f"  文件2独有因子: {len(cols2 - common_factors)} 个")
    print()

    if len(common_factors) == 0:
        print("[ERROR] 没有共同的因子可以比较")
        return

    # 详细比较每个因子
    print("[ANALYSIS] 因子差异详细分析:")
    print("=" * 80)

    factor_differences = []
    significant_factors = []

    for i, factor_name in enumerate(sorted(common_factors), 1):
        print(f"[{i:2d}/{len(common_factors)}] 分析因子: {factor_name}")

        try:
            # 获取因子值
            values1 = df1_aligned[factor_name].values
            values2 = df2_aligned[factor_name].values

            # 计算差异
            abs_diff = np.abs(values1 - values2)

            # 统计信息
            diff_stats = {
                'factor': factor_name,
                'max_diff': np.max(abs_diff),
                'mean_diff': np.mean(abs_diff),
                'median_diff': np.median(abs_diff),
                'std_diff': np.std(abs_diff),
                'non_zero_count': np.sum(abs_diff > 1e-10),
                'identical_count': np.sum(abs_diff <= 1e-10),
                'total_count': len(abs_diff),
            }

            # 相对误差
            non_zero_mask = values2 != 0
            if np.any(non_zero_mask):
                rel_error = np.abs((values1[non_zero_mask] - values2[non_zero_mask]) / values2[non_zero_mask])
                diff_stats['max_relative_error'] = np.max(rel_error)
                diff_stats['mean_relative_error'] = np.mean(rel_error)
            else:
                diff_stats['max_relative_error'] = 0
                diff_stats['mean_relative_error'] = 0

            factor_differences.append(diff_stats)

            # 判断是否有显著差异
            if diff_stats['max_diff'] > 1e-6:  # 设置一个很小的阈值来判断是否真的有差异
                print(f"  [DIFF] 存在差异:")
                print(f"     最大差异: {diff_stats['max_diff']:.10f}")
                print(f"     平均差异: {diff_stats['mean_diff']:.10f}")
                print(f"     中位数差异: {diff_stats['median_diff']:.10f}")
                print(f"     不同值数量: {diff_stats['non_zero_count']:,}/{diff_stats['total_count']:,}")

                if diff_stats['max_relative_error'] > 1e-6:
                    print(f"     最大相对误差: {diff_stats['max_relative_error']:.6f}")
                    print(f"     平均相对误差: {diff_stats['mean_relative_error']:.6f}")

                # 记录显著差异的因子
                if diff_stats['max_diff'] > 1e-3:  # 更严格的阈值
                    significant_factors.append(factor_name)
            else:
                print(f"  [OK] 完全一致")

        except Exception as e:
            print(f"  [ERROR] 分析失败: {e}")

        print()

    # 生成差异统计报告
    print("[SUMMARY] 差异统计汇总:")
    print("=" * 80)

    # 创建差异DataFrame
    diff_df = pd.DataFrame(factor_differences)

    if len(diff_df) > 0:
        # 按最大差异排序
        diff_df_sorted = diff_df.sort_values('max_diff', ascending=False)

        print("Top 10 差异最大的因子:")
        for i, (idx, row) in enumerate(diff_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['factor']:30s} "
                  f"最大差异: {row['max_diff']:.2e} "
                  f"平均差异: {row['mean_diff']:.2e} "
                  f"不同值数: {row['non_zero_count']:5d}/{row['total_count']:5d}")

        print("\n[STATS] 差异分布:")
        print(f"  有差异的因子: {(diff_df['max_diff'] > 1e-6).sum()}/{len(diff_df)} 个")
        print(f"  完全一致的因子: {(diff_df['max_diff'] <= 1e-6).sum()}/{len(diff_df)} 个")

        # 统计差异程度
        max_diffs = diff_df['max_diff']
        print(f"  最大差异范围: {np.min(max_diffs):.2e} ~ {np.max(max_diffs):.2e}")
        print(f"  平均最大差异: {np.mean(max_diffs):.2e}")

        # 查找最可能的问题因子
        highly_variable_factors = diff_df[diff_df['max_diff'] > 1e-3]['factor'].tolist()
        if highly_variable_factors:
            print(f"\n[WARNING] 高差异因子 (最大差异 > 1e-3):")
            for factor in highly_variable_factors[:10]:
                row = diff_df[diff_df['factor'] == factor].iloc[0]
                print(f"  - {factor}: 最大差异 {row['max_diff']:.2e}")

        # 保存差异分析结果
        output_file = "factor_differences_2023_analysis.csv"
        diff_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n[SAVE] 详细分析结果已保存到: {output_file}")

        # 显示显著差异因子
        if significant_factors:
            print(f"\n[CRITICAL] 需要重点关注的因子 (最大差异 > 1e-3):")
            for factor in significant_factors:
                row = diff_df[diff_df['factor'] == factor].iloc[0]
                print(f"  - {factor}:")
                print(f"    最大差异: {row['max_diff']:.2e}")
                print(f"    平均差异: {row['mean_diff']:.2e}")
                print(f"    不同值比例: {row['non_zero_count']/row['total_count']*100:.2f}%")

    else:
        print("[OK] 所有因子完全一致，无任何差异")

if __name__ == "__main__":
    analyze_factor_differences()