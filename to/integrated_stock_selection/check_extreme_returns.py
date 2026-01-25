#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查测试集时间范围内future_return_3d的异常高值
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import glob

def check_extreme_returns(start_date='2025-05-05', end_date='2025-10-15', threshold=0.75):
    """
    检查指定时间范围内future_return_3d超过阈值的记录

    Args:
        start_date: 开始日期
        end_date: 结束日期
        threshold: 收益率阈值
    """
    print(f"检查时间范围: {start_date} ~ {end_date}")
    print(f"查找future_return_3d > {threshold}的记录...")

    # 获取指定时间范围内的所有parquet文件
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    current_dt = start_dt
    extreme_records = []
    all_files = []

    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y%m%d')
        file_pattern = f"../local_data/features/daily/{date_str}.parquet"

        if os.path.exists(file_pattern):
            all_files.append(file_pattern)

        current_dt += timedelta(days=1)

        # 跳过周末
        while current_dt.weekday() >= 5:  # 5=周六, 6=周日
            current_dt += timedelta(days=1)

    print(f"找到 {len(all_files)} 个交易日文件")

    # 检查每个文件
    for file_path in all_files:
        try:
            # 只读取需要的列以提高性能
            df = pd.read_parquet(file_path, columns=['stock_code', 'future_return_3d'])

            # 查找极端高收益记录
            extreme_mask = df['future_return_3d'] > threshold
            extreme_df = df[extreme_mask].copy()

            if len(extreme_df) > 0:
                # 从文件路径提取日期
                date_str = os.path.basename(file_path).replace('.parquet', '')
                date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

                extreme_df['date'] = date_formatted
                extreme_records.append(extreme_df)

                print(f"[EXTREME] 发现极端收益记录 {date_formatted}:")
                for _, row in extreme_df.iterrows():
                    print(f"  股票: {row['stock_code']}, future_return_3d: {row['future_return_3d']:.6f}")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    # 汇总结果
    if extreme_records:
        print(f"\n📊 汇总结果:")
        print(f"在 {start_date} ~ {end_date} 期间")
        print(f"共发现 {len(extreme_records)} 个交易日有 future_return_3d > {threshold} 的记录")

        # 合并所有极端记录
        all_extreme = pd.concat(extreme_records, ignore_index=True)

        print(f"\n📈 统计信息:")
        print(f"总记录数: {len(all_extreme)}")
        print(f"最大收益: {all_extreme['future_return_3d'].max():.6f}")
        print(f"平均收益: {all_extreme['future_return_3d'].mean():.6f}")
        print(f"中位数收益: {all_extreme['future_return_3d'].median():.6f}")

        # 按收益排序显示前10名
        top_10 = all_extreme.nlargest(10, 'future_return_3d')
        print(f"\n🏆 收益最高的前10条记录:")
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"{i:2d}. {row['date']} {row['stock_code']} {row['future_return_3d']:.6f}")

        # 保存详细结果
        output_file = "extreme_returns_analysis.csv"
        all_extreme.to_csv(output_file, index=False)
        print(f"\n💾 详细结果已保存到: {output_file}")

    else:
        print(f"✅ 在 {start_date} ~ {end_date} 期间未发现 future_return_3d > {threshold} 的记录")

        # 检查整体收益分布
        print(f"\n📊 检查整体收益分布...")
        sample_returns = []

        # 采样检查部分文件的收益分布
        sample_files = all_files[::10]  # 每10个文件取1个
        for file_path in sample_files:
            try:
                df = pd.read_parquet(file_path, columns=['future_return_3d'])
                returns = df['future_return_3d'].dropna()
                if len(returns) > 0:
                    sample_returns.extend(returns.tolist())
            except:
                continue

        if sample_returns:
            sample_returns = np.array(sample_returns)
            print(f"采样收益统计 (基于 {len(sample_returns)} 条记录):")
            print(f"  最大值: {np.max(sample_returns):.6f}")
            print(f"  99%分位数: {np.percentile(sample_returns, 99):.6f}")
            print(f"  95%分位数: {np.percentile(sample_returns, 95):.6f}")
            print(f"  平均值: {np.mean(sample_returns):.6f}")
            print(f"  最小值: {np.min(sample_returns):.6f}")

def check_specific_value(target_value=0.7730, tolerance=0.001):
    """
    检查是否有接近特定值的future_return_3d
    """
    print(f"\n🔍 查找future_return_3d ≈ {target_value} (±{tolerance})的记录...")

    # 获取测试集时间范围内的所有文件
    test_files = glob.glob("../local_data/features/daily/20250[56789]*.parquet")
    test_files.extend(glob.glob("../local_data/features/daily/202510[012345]*.parquet"))

    found_records = []

    for file_path in test_files:
        try:
            df = pd.read_parquet(file_path, columns=['stock_code', 'future_return_3d'])

            # 查找接近目标值的记录
            diff = np.abs(df['future_return_3d'] - target_value)
            close_mask = diff <= tolerance
            close_df = df[close_mask].copy()

            if len(close_df) > 0:
                date_str = os.path.basename(file_path).replace('.parquet', '')
                date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

                close_df['date'] = date_formatted
                close_df['diff_from_target'] = diff[close_mask]
                found_records.append(close_df)

                print(f"🎯 发现接近目标值的记录 {date_formatted}:")
                for _, row in close_df.iterrows():
                    print(f"  股票: {row['stock_code']}, future_return_3d: {row['future_return_3d']:.6f}, 差异: {row['diff_from_target']:.6f}")

        except Exception as e:
            continue

    if found_records:
        print(f"\n✅ 共找到 {sum(len(r) for r in found_records)} 条接近 {target_value} 的记录")
    else:
        print(f"❌ 未找到接近 {target_value} 的记录")

if __name__ == "__main__":
    # 检查极端高收益
    check_extreme_returns(start_date='2025-05-05', end_date='2025-10-15', threshold=0.75)

    # 检查特定值0.7730
    check_specific_value(target_value=0.7730, tolerance=0.001)