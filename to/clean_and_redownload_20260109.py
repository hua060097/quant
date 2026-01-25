#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理20260109的脏数据并重新下载

问题：20260109的数据中每只股票有约500条重复记录
解决方案：删除重复数据，只保留每只股票的第一条记录
"""

import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

def clean_daily_data():
    """清理日线数据中的重复记录"""
    print("=" * 80)
    print("步骤1: 清理日线数据中的重复记录")
    print("=" * 80)

    daily_file = Path('local_data/daily/202601/daily_data.parquet')

    if not daily_file.exists():
        print(f"文件不存在: {daily_file}")
        return False

    # 备份原文件
    backup_file = daily_file.parent / f"daily_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    print(f"\n[备份] 复制原文件到: {backup_file.name}")
    shutil.copy(daily_file, backup_file)

    # 读取数据
    print(f"\n[读取] 加载数据...")
    df = pd.read_parquet(daily_file)
    original_count = len(df)
    print(f"  原始行数: {original_count:,}")

    # 检查20260109的数据
    df_0609 = df[df['date_str'] == '20260109'].copy()
    print(f"\n[检查] 20260109数据:")
    print(f"  行数: {len(df_0609):,}")
    print(f"  股票数: {df_0609['stock_code'].nunique()}")

    # 去重
    print(f"\n[去重] 删除重复记录...")
    before_dedup = len(df)
    df_dedup = df.drop_duplicates(subset=['stock_code', 'date_str'], keep='first')
    after_dedup = len(df_dedup)
    removed = before_dedup - after_dedup

    print(f"  去重前: {before_dedup:,} 行")
    print(f"  去重后: {after_dedup:,} 行")
    print(f"  移除: {removed:,} 行 ({removed/before_dedup*100:.2f}%)")

    # 验证20260109去重后的数据
    df_0609_dedup = df_dedup[df_dedup['date_str'] == '20260109'].copy()
    print(f"\n[验证] 20260109去重后:")
    print(f"  行数: {len(df_0609_dedup):,}")
    print(f"  股票数: {df_0609_dedup['stock_code'].nunique()}")

    if len(df_0609_dedup) == df_0609_dedup['stock_code'].nunique():
        print(f"  [OK] 每只股票只有一条记录，去重成功！")
    else:
        print(f"  [FAIL] 仍有重复，需要进一步检查")
        return False

    # 保存清理后的数据
    print(f"\n[保存] 写入清理后的数据...")
    df_dedup = df_dedup.sort_values(['stock_code', 'date'])
    df_dedup.to_parquet(daily_file, index=False, compression='snappy', engine='pyarrow')
    print(f"  保存完成: {daily_file}")

    # 立即验证保存的数据
    print(f"\n[验证] 重新加载数据以确认...")
    df_verify = pd.read_parquet(daily_file)
    df_verify_0609 = df_verify[df_verify['date_str'] == '20260109'].copy()
    print(f"  验证行数: {len(df_verify_0609):,}")

    if len(df_verify_0609) == len(df_0609_dedup):
        print(f"  [OK] 数据保存成功，行数一致")

        # 现在可以安全删除备份
        print(f"\n[清理] 删除备份文件: {backup_file.name}")
        backup_file.unlink()
        print(f"  删除完成")

        return True
    else:
        print(f"  [FAIL] 数据保存失败，行数不一致")
        print(f"  预期: {len(df_0609_dedup):,}, 实际: {len(df_verify_0609):,}")
        return False


def clean_feature_file():
    """删除特征文件，需要重新生成"""
    print("\n" + "=" * 80)
    print("步骤2: 删除特征文件（需要重新生成）")
    print("=" * 80)

    feature_file = Path('local_data/features/daily/20260109.parquet')

    if feature_file.exists():
        print(f"\n[删除] 特征文件: {feature_file}")
        file_size_mb = feature_file.stat().st_size / (1024 * 1024)
        print(f"  文件大小: {file_size_mb:.2f} MB")
        feature_file.unlink()
        print(f"  删除完成")
        return True
    else:
        print(f"\n[跳过] 特征文件不存在")
        return False


def main():
    print("\n" + "=" * 80)
    print("清理20260109脏数据并准备重新下载")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 步骤1: 清理日线数据
    success = clean_daily_data()

    if not success:
        print("\n[失败] 日线数据清理失败")
        return 1

    # 步骤2: 删除特征文件
    clean_feature_file()

    print("\n" + "=" * 80)
    print("清理完成！")
    print("=" * 80)
    print("\n后续步骤:")
    print("1. 重新下载20260109的日线数据:")
    print("   python download_daily.py --start 20260109 --end 20260109")
    print("\n2. 重新生成20260109的特征:")
    print("   python generate_daily_features_simple.py --start_date 20260109 --end_date 20260109")
    print("\n3. 验证修复效果:")
    print("   python analyze_feature_file_size.py")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
