#!/usr/bin/env python3
"""
增量因子更新分析脚本 - 简化版

分析incremental_save模式下生成的factors.parquet文件，
检查增量更新是否成功执行。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import glob

def analyze_incremental_update():
    """分析增量更新结果"""
    print("=" * 80)
    print("Incremental Factor Update Analysis")
    print("=" * 80)

    # 1. 检查factors.parquet文件基本信息
    print("\n1. Factors file basic information:")
    factors_file = Path("factor_lab_data/factors.parquet")

    if not factors_file.exists():
        print("[FAIL] factors.parquet file does not exist")
        return False

    # 读取文件
    try:
        df = pd.read_parquet(factors_file)
        print("[OK] File read successfully")
        print(f"   File size: {factors_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   Record count: {len(df):,}")
        print(f"   Stock count: {df['stock_code'].nunique()}")
        print(f"   Date count: {df['date'].nunique()}")
    except Exception as e:
        print(f"[FAIL] File read failed: {e}")
        return False

    # 2. 检查时间范围
    print("\n2. Time range check:")
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min()
    max_date = df['date'].max()

    print(f"   Start date: {min_date.strftime('%Y-%m-%d')}")
    print(f"   End date: {max_date.strftime('%Y-%m-%d')}")
    print(f"   Time span: {(max_date - min_date).days + 1} days")

    # 检查是否包含预期的日期范围
    expected_start = pd.to_datetime('2023-06-01')
    expected_end = pd.to_datetime('2023-07-01')

    if min_date <= expected_start and max_date >= expected_end:
        print("[OK] Time range matches expectation")
    else:
        print("[WARNING] Time range may not match expectation")
        print(f"   Expected range: {expected_start.strftime('%Y-%m-%d')} ~ {expected_end.strftime('%Y-%m-%d')}")

    # 3. 按日期统计记录数量
    print("\n3. Daily record count statistics:")
    daily_counts = df.groupby('date').size().sort_index()

    print(f"   Number of dates: {len(daily_counts)}")
    print(f"   Average daily stocks: {daily_counts.mean():.1f}")
    print(f"   Max daily stocks: {daily_counts.max()}")
    print(f"   Min daily stocks: {daily_counts.min()}")

    # 显示关键日期的统计
    key_dates = ['2023-06-01', '2023-06-19', '2023-06-20', '2023-06-21', '2023-07-01']
    print("\n   Key date statistics:")
    for date_str in key_dates:
        date = pd.to_datetime(date_str)
        if date in daily_counts.index:
            count = daily_counts[date]
            print(f"     {date_str}: {count:4d} stocks")
        else:
            print(f"     {date_str}: NO DATA")

    # 4. 检查因子列
    print("\n4. Factor column check:")
    factor_cols = [col for col in df.columns if col not in ['date', 'stock_code']]

    print(f"   Number of factors: {len(factor_cols)}")
    print(f"   Factor list: {factor_cols}")

    # 检查pct_chg_1d因子是否存在
    if 'pct_chg_1d' in factor_cols:
        print("[OK] pct_chg_1d factor exists")

        # 检查pct_chg_1d的统计信息
        pct_chg_stats = df['pct_chg_1d'].describe()
        print(f"\n   pct_chg_1d statistics:")
        print(f"     Count: {pct_chg_stats['count']:,.0f}")
        print(f"     Mean: {pct_chg_stats['mean']:.6f}")
        print(f"     Std: {pct_chg_stats['std']:.6f}")
        print(f"     Min: {pct_chg_stats['min']:.6f}")
        print(f"     Max: {pct_chg_stats['max']:.6f}")
        print(f"     Missing values: {df['pct_chg_1d'].isna().sum()}")
    else:
        print("[FAIL] pct_chg_1d factor does not exist")

    # 5. 检查数据完整性
    print("\n5. Data integrity check:")

    # 检查是否有缺失值
    missing_stats = {}
    for col in factor_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_stats[col] = missing_count

    if missing_stats:
        print("[WARNING] Missing values found:")
        for col, count in missing_stats.items():
            print(f"     {col}: {count:,} missing values ({count/len(df)*100:.2f}%)")
    else:
        print("[OK] No missing values found")

    # 检查是否有异常值
    if 'pct_chg_1d' in factor_cols:
        # 检查pct_chg_1d是否有异常值（超过±20%）
        extreme_count = ((df['pct_chg_1d'].abs() > 0.2)).sum()
        if extreme_count > 0:
            print(f"[WARNING] Found {extreme_count:,} extreme values (|pct_chg_1d| > 20%)")
        else:
            print("[OK] No extreme values found")

    # 6. 检查是否为增量更新结果
    print("\n6. Incremental update feature check:")

    # 检查2023-06-20前后的数据是否连续
    cutoff_date = pd.to_datetime('2023-06-20')

    # 统计6月20日之前的天数
    before_days = df[df['date'] < cutoff_date]['date'].nunique()
    # 统计6月20日及之后的天数
    after_days = df[df['date'] >= cutoff_date]['date'].nunique()

    print(f"   Trading days before 2023-06-20: {before_days} days")
    print(f"   Trading days on/after 2023-06-20: {after_days} days")
    print(f"   Total trading days: {before_days + after_days} days")

    # 检查是否覆盖了完整的日期范围
    expected_before_days = 14  # 2023-06-01 到 2023-06-20 (不包含)
    expected_after_days = 9    # 2023-06-20 到 2023-07-01 (包含)

    if before_days >= expected_before_days and after_days >= expected_after_days:
        print("[OK] Date coverage matches incremental update expectation")
    else:
        print("[WARNING] Date coverage may not match expectation")

    # 7. 检查是否存在备份文件或临时文件
    print("\n7. Related files check:")

    # 检查是否有备份文件
    backup_pattern = "factor_lab_data/factors_*.parquet"
    backup_files = glob.glob(backup_pattern)

    if backup_files:
        print(f"   Found {len(backup_files)} backup files:")
        for file in backup_files[:5]:  # 只显示前5个
            stat = Path(file).stat()
            size_mb = stat.st_size / 1024 / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"     {file}: {size_mb:.2f}MB, modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("   No backup files found")

    # 检查日志文件
    log_pattern = "log/*.log"
    log_files = glob.glob(log_pattern)

    if log_files:
        print(f"\n   Found {len(log_files)} log files:")
        for file in log_files:
            stat = Path(file).stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"     {file}: {size_kb:.1f}KB, modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    # 8. 结论
    print("\n8. Conclusion:")

    success_indicators = 0
    total_indicators = 0

    # 检查各项指标
    if len(df) > 0:
        print("[OK] Data file exists and readable")
        success_indicators += 1
    total_indicators += 1

    if min_date <= expected_start and max_date >= expected_end:
        print("[OK] Time range covers expectation")
        success_indicators += 1
    total_indicators += 1

    if 'pct_chg_1d' in factor_cols:
        print("[OK] Contains specified pct_chg_1d factor")
        success_indicators += 1
    total_indicators += 1

    if len(missing_stats) == 0:
        print("[OK] Data integrity is good")
        success_indicators += 1
    total_indicators += 1

    if before_days >= expected_before_days and after_days >= expected_after_days:
        print("[OK] Incremental update date coverage is complete")
        success_indicators += 1
    total_indicators += 1

    # 总体评估
    success_rate = success_indicators / total_indicators
    print(f"\n   Success indicators: {success_indicators}/{total_indicators}")
    print(f"   Success rate: {success_rate*100:.1f}%")

    if success_rate >= 0.8:
        print("\n[SUCCESS] Incremental update executed successfully!")
        print("   Data file has been successfully updated with expected date range and factor data.")
    elif success_rate >= 0.6:
        print("\n[PARTIAL] Incremental update partially successful")
        print("   Data file has been updated, but there may be some issues to check.")
    else:
        print("\n[FAIL] Incremental update may have failed")
        print("   Recommend checking log files and re-running incremental update.")

    return success_rate >= 0.8

def check_date_continuity():
    """检查日期连续性"""
    print("\n" + "=" * 80)
    print("Date Continuity Analysis")
    print("=" * 80)

    try:
        df = pd.read_parquet("factor_lab_data/factors.parquet")
        df['date'] = pd.to_datetime(df['date'])

        # 获取所有日期并排序
        all_dates = sorted(df['date'].unique())

        print(f"\nTotal trading days: {len(all_dates)}")
        print(f"Date range: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}")

        # 检查关键日期是否在数据中
        key_dates = [
            '2023-06-01',  # 第一次运行的开始日期
            '2023-06-19',  # 第一次运行的结束日期前
            '2023-06-20',  # 第二次运行的开始日期
            '2023-06-30',  # 第二次运行的结束日期前
            '2023-07-01'   # 第二次运行的结束日期
        ]

        print("\nKey dates presence check:")
        for date_str in key_dates:
            date = pd.to_datetime(date_str)
            if date in all_dates:
                stock_count = df[df['date'] == date]['stock_code'].nunique()
                print(f"  {date_str}: PRESENT ({stock_count} stocks)")
            else:
                print(f"  {date_str}: MISSING")

        # 检查日期间隔
        print("\nDate interval analysis:")
        for i in range(1, len(all_dates)):
            days_gap = (all_dates[i] - all_dates[i-1]).days
            if days_gap > 1:
                print(f"  Gap: {all_dates[i-1].strftime('%Y-%m-%d')} -> {all_dates[i].strftime('%Y-%m-%d')} ({days_gap} days)")

        if len(all_dates) > 1:
            total_days = (all_dates[-1] - all_dates[0]).days + 1
            coverage_rate = len(all_dates) / total_days * 100
            print(f"\nDate coverage rate: {coverage_rate:.1f}% ({len(all_dates)}/{total_days} days)")

            if coverage_rate >= 90:
                print("[OK] Date coverage is good")
            else:
                print("[WARNING] Date coverage may be incomplete")

    except Exception as e:
        print(f"[FAIL] Error analyzing date continuity: {e}")

if __name__ == "__main__":
    try:
        # 执行主要分析
        success = analyze_incremental_update()

        # 执行日期连续性检查
        check_date_continuity()

        print("\n" + "=" * 80)
        print("Analysis Complete")
        print("=" * 80)

        if success:
            print("[SUCCESS] Incremental update analysis completed with good results")
        else:
            print("[WARNING] Incremental update analysis completed, further checks recommended")

    except Exception as e:
        print(f"[FAIL] Error during analysis: {e}")
        import traceback
        traceback.print_exc()