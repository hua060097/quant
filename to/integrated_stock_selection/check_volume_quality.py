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

    # 生成日期列表
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")

    # 生成所有工作日（排除周末）
    date_list = []
    current_dt = start_dt
    while current_dt <= end_dt:
        # 排除周末
        if current_dt.weekday() < 5:  # 0-4 是周一到周五
            date_list.append(current_dt.strftime("%Y%m%d"))
        current_dt += timedelta(days=1)

    print(f"📅 检查日期范围: {start_date} 到 {end_date}")
    print(f"📊 检查的总日期数: {len(date_list)} 个工作日")
    print("=" * 60)

    # 统计信息
    total_files = 0
    volume_missing_stats = []
    detailed_stats = []

    for date_str in date_list:
        file_path = data_dir / f"{date_str}.parquet"

        if not file_path.exists():
            print(f"❌ 文件不存在: {date_str}.parquet")
            continue

        try:
            # 读取数据
            df = pd.read_parquet(file_path)
            total_files += 1

            # 检查volume字段
            if 'volume' not in df.columns:
                print(f"❌ {date_str}: 缺少volume字段")
                volume_missing_stats.append({
                    'date': date_str,
                    'total_records': len(df),
                    'volume_missing': len(df),
                    'missing_ratio': 1.0,
                    'has_volume_field': False
                })
                continue

            # 计算volume缺失统计
            total_records = len(df)
            volume_null_count = df['volume'].isna().sum()
            volume_zero_count = (df['volume'] == 0).sum()
            volume_valid_count = total_records - volume_null_count - volume_zero_count

            missing_ratio = volume_null_count / total_records if total_records > 0 else 0
            zero_ratio = volume_zero_count / total_records if total_records > 0 else 0
            valid_ratio = volume_valid_count / total_records if total_records > 0 else 0

            volume_missing_stats.append({
                'date': date_str,
                'total_records': total_records,
                'volume_missing': volume_null_count,
                'volume_zero': volume_zero_count,
                'volume_valid': volume_valid_count,
                'missing_ratio': missing_ratio,
                'zero_ratio': zero_ratio,
                'valid_ratio': valid_ratio,
                'has_volume_field': True
            })

            # 如果缺失率很高，记录详细信息
            if missing_ratio > 0.1:  # 缺失率超过10%
                detailed_stats.append({
                    'date': date_str,
                    'missing_ratio': missing_ratio,
                    'total_records': total_records,
                    'volume_missing': volume_null_count,
                    'volume_stats': {
                        'mean': df['volume'].mean() if volume_valid_count > 0 else 'NaN',
                        'std': df['volume'].std() if volume_valid_count > 0 else 'NaN',
                        'min': df['volume'].min() if volume_valid_count > 0 else 'NaN',
                        'max': df['volume'].max() if volume_valid_count > 0 else 'NaN'
                    }
                })

            print(f"📅 {date_str}: 总记录={total_records:,}, volume缺失={volume_null_count:,} ({missing_ratio:.4f}), volume为零={volume_zero_count:,} ({zero_ratio:.4f}), volume有效={volume_valid_count:,} ({valid_ratio:.4f})")

        except Exception as e:
            print(f"❌ {date_str}: 读取文件失败 - {e}")
            continue

    print("\n" + "=" * 60)
    print("📋 数据质量总结:")
    print(f"✅ 成功检查的文件数: {total_files}")

    if volume_missing_stats:
        # 转换为DataFrame进行分析
        stats_df = pd.DataFrame(volume_missing_stats)

        print(f"\n📊 Volume字段统计:")
        print(f"   总记录数: {stats_df['total_records'].sum():,}")
        print(f"   Volume缺失记录数: {stats_df['volume_missing'].sum():,}")
        print(f"   Volume为零记录数: {stats_df['volume_zero'].sum():,}")
        print(f"   Volume有效记录数: {stats_df['volume_valid'].sum():,}")

        print(f"\n📈 比例统计:")
        print(f"   平均缺失率: {stats_df['missing_ratio'].mean():.4f}")
        print(f"   最高缺失率: {stats_df['missing_ratio'].max():.4f}")
        print(f"   平均零值率: {stats_df['zero_ratio'].mean():.4f}")
        print(f"   平均有效率: {stats_df['valid_ratio'].mean():.4f}")

        # 找出问题日期
        high_missing_dates = stats_df[stats_df['missing_ratio'] > 0.1]
        if len(high_missing_dates) > 0:
            print(f"\n⚠️  Volume缺失率>10%的日期 ({len(high_missing_dates)}个):")
            for _, row in high_missing_dates.iterrows():
                print(f"   {row['date']}: 缺失率={row['missing_ratio']:.4f}, 缺失数={row['volume_missing']:,}")

        # 检查是否有volume字段缺失的文件
        no_volume_files = stats_df[~stats_df['has_volume_field']]
        if len(no_volume_files) > 0:
            print(f"\n❌ 缺少volume字段的文件 ({len(no_volume_files)}个):")
            for _, row in no_volume_files.iterrows():
                print(f"   {row['date']}")

    if detailed_stats:
        print(f"\n🔍 详细问题分析:")
        for stat in detailed_stats[:5]:  # 显示前5个最严重的
            print(f"   {stat['date']}: 缺失率={stat['missing_ratio']:.4f}, 统计信息={stat['volume_stats']}")

if __name__ == "__main__":
    check_volume_quality()