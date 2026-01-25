#!/usr/bin/env python3
"""
检查20251021-20251216期间的volume字段数据质量
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta

def check_volume_quality():
    """检查volume字段的数据质量"""

    # 数据目录
    data_dir = Path("../local_data/features/daily")

    # 检查的日期范围
    start_date = "20251021"
    end_date = "20251216"

    print(f"Checking date range: {start_date} to {end_date}")
    print("=" * 60)

    # 统计信息
    total_files = 0
    volume_stats = []
    problematic_dates = []

    # 直接检查指定日期的文件
    for year in [2025]:
        for month in range(10, 13):
            for day in range(1, 32):
                # 构建日期字符串
                date_str = f"{year}{month:02d}{day:02d}"

                # 检查是否在目标范围内
                if date_str < start_date or date_str > end_date:
                    continue

                file_path = data_dir / f"{date_str}.parquet"

                if not file_path.exists():
                    continue

                try:
                    # 读取数据
                    df = pd.read_parquet(file_path)
                    total_files += 1

                    # 检查volume字段
                    if 'volume' not in df.columns:
                        print(f"{date_str}: Missing volume field")
                        problematic_dates.append({
                            'date': date_str,
                            'issue': 'Missing volume field',
                            'total_records': len(df)
                        })
                        continue

                    # 计算volume统计
                    total_records = len(df)
                    volume_null_count = df['volume'].isna().sum()
                    volume_zero_count = (df['volume'] == 0).sum()
                    volume_valid_count = total_records - volume_null_count

                    missing_ratio = volume_null_count / total_records if total_records > 0 else 0
                    valid_ratio = volume_valid_count / total_records if total_records > 0 else 0

                    volume_stats.append({
                        'date': date_str,
                        'total_records': total_records,
                        'volume_missing': volume_null_count,
                        'volume_valid': volume_valid_count,
                        'missing_ratio': missing_ratio,
                        'valid_ratio': valid_ratio
                    })

                    # 如果问题严重，打印详细信息
                    if missing_ratio > 0.05 or total_records < 1000:
                        print(f"{date_str}: Total={total_records:,}, Volume Missing={volume_null_count:,} ({missing_ratio:.4f}), Volume Valid={volume_valid_count:,} ({valid_ratio:.4f})")
                        if missing_ratio > 0.05:
                            problematic_dates.append({
                                'date': date_str,
                                'issue': f'High volume missing ratio: {missing_ratio:.4f}',
                                'total_records': total_records,
                                'volume_missing': volume_null_count,
                                'missing_ratio': missing_ratio
                            })

                except Exception as e:
                    print(f"{date_str}: Error reading file - {e}")
                    continue

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Files checked: {total_files}")

    if volume_stats:
        # 转换为DataFrame进行分析
        stats_df = pd.DataFrame(volume_stats)

        print(f"\nVolume Statistics:")
        print(f"Total records: {stats_df['total_records'].sum():,}")
        print(f"Volume missing records: {stats_df['volume_missing'].sum():,}")
        print(f"Volume valid records: {stats_df['volume_valid'].sum():,}")

        print(f"\nRatios:")
        print(f"Average missing ratio: {stats_df['missing_ratio'].mean():.4f}")
        print(f"Maximum missing ratio: {stats_df['missing_ratio'].max():.4f}")
        print(f"Average valid ratio: {stats_df['valid_ratio'].mean():.4f}")

        # 找出严重问题
        high_missing = stats_df[stats_df['missing_ratio'] > 0.1]
        if len(high_missing) > 0:
            print(f"\nSERIOUS ISSUES - Volume missing ratio > 10% ({len(high_missing)} dates):")
            for _, row in high_missing.iterrows():
                print(f"  {row['date']}: {row['missing_ratio']:.4f}, missing {row['volume_missing']:,} out of {row['total_records']:,}")

    if problematic_dates:
        print(f"\nPROBLEMATIC DATES ({len(problematic_dates)} total):")
        for problem in problematic_dates:
            print(f"  {problem['date']}: {problem['issue']}")

if __name__ == "__main__":
    check_volume_quality()