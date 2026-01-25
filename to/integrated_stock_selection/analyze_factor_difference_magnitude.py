#!/usr/bin/env python3
"""
分析2023年因子差异的详细幅度分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_factor_difference_magnitude():
    """详细分析因子差异的幅度等级"""

    print("=" * 80)
    print("2023年因子差异幅度详细分析")
    print("=" * 80)

    # 读取数据
    print("[INFO] 正在读取数据...")
    df1 = pd.read_parquet('factor_lab_data/factors1.parquet')
    df2 = pd.read_parquet('factor_lab_data/factors.parquet')

    # 转换日期列
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])

    # 过滤2023年数据
    df1_2023 = df1[(df1['date'] >= '2023-01-01') & (df1['date'] <= '2023-12-31')].copy()
    df2_2023 = df2[(df2['date'] >= '2023-01-01') & (df2['date'] <= '2023-12-31')].copy()

    print(f"[DATA] 2023年数据: factors1={len(df1_2023):,}, factors={len(df2_2023):,}")

    # 找到重叠的日期并对齐数据
    dates1 = set(df1_2023['date'].unique())
    dates2 = set(df2_2023['date'].unique())
    overlap_dates = dates1.intersection(dates2)

    df1_overlap = df1_2023[df1_2023['date'].isin(overlap_dates)].copy()
    df2_overlap = df2_2023[df2_2023['date'].isin(overlap_dates)].copy()

    # 对齐数据
    df1_aligned = df1_overlap.sort_values(['date', 'stock_code']).reset_index(drop=True)
    df2_aligned = df2_overlap.sort_values(['date', 'stock_code']).reset_index(drop=True)

    # 确保对齐
    min_len = min(len(df1_aligned), len(df2_aligned))
    df1_aligned = df1_aligned.head(min_len)
    df2_aligned = df2_aligned.head(min_len)

    print(f"[ALIGN] 对齐数据量: {len(df1_aligned):,}")

    # 获取共同因子列
    common_cols = set(df1_aligned.columns) & set(df2_aligned.columns)
    factor_cols = sorted(list(common_cols - {'date', 'stock_code'}))

    print(f"[FACTORS] 共同因子数量: {len(factor_cols)}")

    # 分析每个因子的差异幅度
    print("\n[ANALYSIS] 因子差异幅度详细分析:")
    print("-" * 80)

    # 定义差异等级
    magnitude_levels = {
        'tiny': (1e-10, 1e-6, '微小差异 (浮点精度级)'),
        'small': (1e-6, 1e-3, '小差异 (千分级)'),
        'medium': (1e-3, 1e-1, '中等差异 (百分级)'),
        'large': (1e-1, 1.0, '大差异 (十分级)'),
        'huge': (1.0, float('inf'), '巨大差异 (数量级差异)')
    }

    factor_analysis = []

    for factor_name in factor_cols:
        if factor_name not in df1_aligned.columns or factor_name not in df2_aligned.columns:
            continue

        values1 = df1_aligned[factor_name].values
        values2 = df2_aligned[factor_name].values

        # 计算差异
        abs_diff = np.abs(values1 - values2)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        median_diff = np.median(abs_diff)

        # 计算相对差异（对于非零值）
        non_zero_mask = values2 != 0
        if np.any(non_zero_mask):
            rel_diff = np.abs((values1[non_zero_mask] - values2[non_zero_mask]) / values2[non_zero_mask])
            max_rel_diff = np.max(rel_diff)
            mean_rel_diff = np.mean(rel_diff)
        else:
            max_rel_diff = 0
            mean_rel_diff = 0

        # 统计不同差异等级的数量
        diff_counts = {}
        for level, (lower, upper, desc) in magnitude_levels.items():
            count = np.sum((abs_diff >= lower) & (abs_diff < upper))
            diff_counts[level] = count
            diff_counts[f'{level}_pct'] = count / len(abs_diff) * 100

        # 判断差异等级
        if max_diff < 1e-6:
            magnitude = 'identical'
            magnitude_desc = '完全一致'
        elif max_diff < 1e-3:
            magnitude = 'tiny'
            magnitude_desc = magnitude_levels['tiny'][2]
        elif max_diff < 1e-1:
            magnitude = 'small'
            magnitude_desc = magnitude_levels['small'][2]
        elif max_diff < 1.0:
            magnitude = 'medium'
            magnitude_desc = magnitude_levels['medium'][2]
        else:
            magnitude = 'large_or_huge'
            magnitude_desc = '大差异或巨大差异'

        factor_analysis.append({
            'factor': factor_name,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'median_diff': median_diff,
            'max_rel_diff': max_rel_diff,
            'mean_rel_diff': mean_rel_diff,
            'magnitude': magnitude,
            'magnitude_desc': magnitude_desc,
            **diff_counts
        })

    # 创建DataFrame
    analysis_df = pd.DataFrame(factor_analysis)

    # 按最大差异排序
    analysis_df_sorted = analysis_df.sort_values('max_diff', ascending=False)

    print("差异幅度分布统计:")
    magnitude_stats = analysis_df['magnitude'].value_counts()
    for magnitude, count in magnitude_stats.items():
        pct = count / len(analysis_df) * 100
        # 查找对应的描述
        desc_row = analysis_df[analysis_df['magnitude'] == magnitude].iloc[0]
        desc = desc_row['magnitude_desc']
        print(f"  {desc}: {count}个因子 ({pct:.1f}%)")

    print(f"\n差异最大的前15个因子详细分析:")
    print("-" * 80)

    for i, (_, row) in enumerate(analysis_df_sorted.head(15).iterrows()):
        print(f"{i+1:2d}. {row['factor']:35s}")
        print(f"    差异等级: {row['magnitude_desc']}")
        print(f"    最大差异: {row['max_diff']:.10f}")
        print(f"    平均差异: {row['mean_diff']:.10f}")
        print(f"    中位数差异: {row['median_diff']:.10f}")

        if row['max_rel_diff'] > 0:
            print(f"    最大相对差异: {row['max_rel_diff']:.6f}")
            print(f"    平均相对差异: {row['mean_rel_diff']:.6f}")

        # 显示差异分布
        if row['max_diff'] > 1e-6:
            print(f"    差异分布: 千分级以上{row.get('small_pct', 0):.1f}%, "
                  f"百分级以上{row.get('medium_pct', 0):.1f}%, "
                  f"十分级以上{row.get('large_pct', 0):.1f}%")
        print()

    # 按差异等级分组显示
    print("按差异等级分组:")
    print("-" * 50)

    for magnitude in ['huge', 'large', 'medium', 'small', 'tiny']:
        if magnitude == 'tiny':
            mask = analysis_df['max_diff'] < 1e-3
            desc = '微小差异 (<1e-3)'
        elif magnitude == 'small':
            mask = (analysis_df['max_diff'] >= 1e-3) & (analysis_df['max_diff'] < 1e-1)
            desc = '小差异 (1e-3 ~ 1e-1)'
        elif magnitude == 'medium':
            mask = (analysis_df['max_diff'] >= 1e-1) & (analysis_df['max_diff'] < 1.0)
            desc = '中等差异 (1e-1 ~ 1.0)'
        elif magnitude == 'large':
            mask = (analysis_df['max_diff'] >= 1.0) & (analysis_df['max_diff'] < 10.0)
            desc = '大差异 (1.0 ~ 10.0)'
        else:  # huge
            mask = analysis_df['max_diff'] >= 10.0
            desc = '巨大差异 (>=10.0)'

        factors_in_category = analysis_df[mask]
        if len(factors_in_category) > 0:
            print(f"\n{desc} ({len(factors_in_category)}个因子):")
            for _, row in factors_in_category.iterrows():
                print(f"  - {row['factor']}: 最大差异 {row['max_diff']:.6f}")

    # 保存详细分析结果
    output_file = "factor_magnitude_analysis_2023.csv"
    analysis_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n[SAVE] 详细分析结果已保存到: {output_file}")

    # 总结
    print("\n" + "=" * 80)
    print("总结:")
    print("=" * 80)

    total_factors = len(analysis_df)
    identical = (analysis_df['max_diff'] < 1e-6).sum()
    tiny_diff = ((analysis_df['max_diff'] >= 1e-6) & (analysis_df['max_diff'] < 1e-3)).sum()
    small_diff = ((analysis_df['max_diff'] >= 1e-3) & (analysis_df['max_diff'] < 1e-1)).sum()
    medium_diff = ((analysis_df['max_diff'] >= 1e-1) & (analysis_df['max_diff'] < 1.0)).sum()
    large_diff = (analysis_df['max_diff'] >= 1.0).sum()

    print(f"总因子数: {total_factors}")
    print(f"完全一致: {identical}个 ({identical/total_factors*100:.1f}%)")
    print(f"微小差异(浮点精度): {tiny_diff}个 ({tiny_diff/total_factors*100:.1f}%)")
    print(f"小差异(千分级到百分级): {small_diff}个 ({small_diff/total_factors*100:.1f}%)")
    print(f"中等差异(百分级到十分级): {medium_diff}个 ({medium_diff/total_factors*100:.1f}%)")
    print(f"大差异(十分级以上): {large_diff}个 ({large_diff/total_factors*100:.1f}%)")

    print(f"\n关键发现:")
    print(f"1. 大部分因子({identical/total_factors*100:.1f}%)完全一致，说明计算过程稳定")
    print(f"2. 有差异的因子主要分为两类:")
    print(f"   - 数值计算类因子: 差异在千分级到百分级，可能是浮点运算精度问题")
    print(f"   - 排序归一化类因子: 差异在十分级以上，可能是排序不稳定导致")
    print(f"3. 建议重点检查support_distance、resistance_distance等因子的计算稳定性")

if __name__ == "__main__":
    analyze_factor_difference_magnitude()