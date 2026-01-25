#!/usr/bin/env python3
"""
增量因子更新分析脚本

分析incremental_save模式下生成的factors.parquet文件，
检查增量更新是否成功执行。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import glob
import os

def analyze_incremental_update():
    """分析增量更新结果"""
    print("=" * 80)
    print("增量因子更新分析")
    print("=" * 80)

    # 1. 检查factors.parquet文件基本信息
    print("\n1. 检查factors.parquet文件基本信息:")
    factors_file = Path("factor_lab_data/factors.parquet")

    if not factors_file.exists():
        print("❌ factors.parquet文件不存在")
        return

    # 读取文件
    try:
        df = pd.read_parquet(factors_file)
        print(f"✅ 文件读取成功")
        print(f"   文件大小: {factors_file.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   记录数量: {len(df):,}")
        print(f"   股票数量: {df['stock_code'].nunique()}")
        print(f"   日期数量: {df['date'].nunique()}")
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        return

    # 2. 检查时间范围
    print("\n2. 检查时间范围:")
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min()
    max_date = df['date'].max()

    print(f"   起始日期: {min_date.strftime('%Y-%m-%d')}")
    print(f"   结束日期: {max_date.strftime('%Y-%m-%d')}")
    print(f"   时间跨度: {(max_date - min_date).days + 1}天")

    # 检查是否包含预期的日期范围
    expected_start = pd.to_datetime('2023-06-01')
    expected_end = pd.to_datetime('2023-07-01')

    if min_date <= expected_start and max_date >= expected_end:
        print("✅ 时间范围符合预期")
    else:
        print("⚠️ 时间范围可能不符合预期")
        print(f"   预期范围: {expected_start.strftime('%Y-%m-%d')} ~ {expected_end.strftime('%Y-%m-%d')}")

    # 3. 按日期统计记录数量
    print("\n3. 按日期统计记录数量:")
    daily_counts = df.groupby('date').size().sort_index()

    print(f"   日期数量: {len(daily_counts)}")
    print(f"   平均每日股票数: {daily_counts.mean():.1f}")
    print(f"   最大每日股票数: {daily_counts.max()}")
    print(f"   最小每日股票数: {daily_counts.min()}")

    # 显示前后10天的统计
    print("\n   前10天统计:")
    for date, count in daily_counts.head(10).items():
        print(f"     {date.strftime('%Y-%m-%d')}: {count:4d}只股票")

    print("\n   后10天统计:")
    for date, count in daily_counts.tail(10).items():
        print(f"     {date.strftime('%Y-%m-%d')}: {count:4d}只股票")

    # 4. 检查因子列
    print("\n4. 检查因子列:")
    factor_cols = [col for col in df.columns if col not in ['date', 'stock_code']]

    print(f"   因子数量: {len(factor_cols)}")
    print(f"   因子列表: {factor_cols}")

    # 检查pct_chg_1d因子是否存在
    if 'pct_chg_1d' in factor_cols:
        print("✅ pct_chg_1d因子存在")

        # 检查pct_chg_1d的统计信息
        pct_chg_stats = df['pct_chg_1d'].describe()
        print(f"\n   pct_chg_1d统计:")
        print(f"     总数: {pct_chg_stats['count']:,.0f}")
        print(f"     均值: {pct_chg_stats['mean']:.6f}")
        print(f"     标准差: {pct_chg_stats['std']:.6f}")
        print(f"     最小值: {pct_chg_stats['min']:.6f}")
        print(f"     最大值: {pct_chg_stats['max']:.6f}")
        print(f"     缺失值: {df['pct_chg_1d'].isna().sum()}")
    else:
        print("❌ pct_chg_1d因子不存在")

    # 5. 检查数据完整性
    print("\n5. 检查数据完整性:")

    # 检查是否有缺失值
    missing_stats = {}
    for col in factor_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_stats[col] = missing_count

    if missing_stats:
        print("⚠️ 发现缺失值:")
        for col, count in missing_stats.items():
            print(f"     {col}: {count:,}个缺失值 ({count/len(df)*100:.2f}%)")
    else:
        print("✅ 没有发现缺失值")

    # 检查是否有异常值
    if 'pct_chg_1d' in factor_cols:
        # 检查pct_chg_1d是否有异常值（超过±20%）
        extreme_count = ((df['pct_chg_1d'].abs() > 0.2)).sum()
        if extreme_count > 0:
            print(f"⚠️ 发现{extreme_count:,}个极端值（|pct_chg_1d| > 20%）")
        else:
            print("✅ 没有发现极端值")

    # 6. 检查是否为增量更新结果
    print("\n6. 检查增量更新特征:")

    # 检查2023-06-20前后的数据是否连续
    cutoff_date = pd.to_datetime('2023-06-20')

    # 统计6月20日之前的天数
    before_days = df[df['date'] < cutoff_date]['date'].nunique()
    # 统计6月20日及之后的天数
    after_days = df[df['date'] >= cutoff_date]['date'].nunique()

    print(f"   2023-06-20之前交易日: {before_days}天")
    print(f"   2023-06-20及之后交易日: {after_days}天")
    print(f"   总交易日: {before_days + after_days}天")

    # 检查是否覆盖了完整的日期范围
    expected_before_days = 14  # 2023-06-01 到 2023-06-20 (不包含)
    expected_after_days = 9    # 2023-06-20 到 2023-07-01 (包含)

    if before_days >= expected_before_days and after_days >= expected_after_days:
        print("✅ 日期覆盖范围符合增量更新预期")
    else:
        print("⚠️ 日期覆盖范围可能不符合预期")

    # 7. 检查是否存在备份文件或临时文件
    print("\n7. 检查相关文件:")

    # 检查是否有备份文件
    backup_pattern = "factor_lab_data/factors_*.parquet"
    backup_files = glob.glob(backup_pattern)

    if backup_files:
        print(f"   发现{len(backup_files)}个备份文件:")
        for file in backup_files[:5]:  # 只显示前5个
            stat = Path(file).stat()
            size_mb = stat.st_size / 1024 / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"     {file}: {size_mb:.2f}MB, 修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("   没有发现备份文件")

    # 检查日志文件
    log_pattern = "log/*.log"
    log_files = glob.glob(log_pattern)

    if log_files:
        print(f"\n   发现{len(log_files)}个日志文件:")
        for file in log_files:
            stat = Path(file).stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"     {file}: {size_kb:.1f}KB, 修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    # 8. 结论
    print("\n8. 结论:")

    success_indicators = 0
    total_indicators = 0

    # 检查各项指标
    if len(df) > 0:
        print("✅ 数据文件存在且可读取")
        success_indicators += 1
    total_indicators += 1

    if min_date <= expected_start and max_date >= expected_end:
        print("✅ 时间范围覆盖预期")
        success_indicators += 1
    total_indicators += 1

    if 'pct_chg_1d' in factor_cols:
        print("✅ 包含指定的pct_chg_1d因子")
        success_indicators += 1
    total_indicators += 1

    if len(missing_stats) == 0:
        print("✅ 数据完整性良好")
        success_indicators += 1
    total_indicators += 1

    if before_days >= expected_before_days and after_days >= expected_after_days:
        print("✅ 增量更新日期覆盖完整")
        success_indicators += 1
    total_indicators += 1

    # 总体评估
    success_rate = success_indicators / total_indicators
    print(f"\n   成功指标: {success_indicators}/{total_indicators}")
    print(f"   成功率: {success_rate*100:.1f}%")

    if success_rate >= 0.8:
        print("\n🎉 增量更新执行成功！")
        print("   数据文件已成功更新，包含了预期的日期范围和因子数据。")
    elif success_rate >= 0.6:
        print("\n⚠️ 增量更新部分成功")
        print("   数据文件已更新，但可能存在一些问题需要检查。")
    else:
        print("\n❌ 增量更新可能失败")
        print("   建议检查日志文件和重新运行增量更新。")

    return success_rate >= 0.8

def check_file_comparison():
    """检查是否可以与之前的文件进行对比"""
    print("\n" + "=" * 80)
    print("文件对比分析")
    print("=" * 80)

    # 检查是否有其他版本的factors文件
    factor_files = glob.glob("factor_lab_data/factors*.parquet")
    factor_files = [f for f in factor_files if "factors.parquet" not in f or f == "factor_lab_data/factors.parquet"]

    if len(factor_files) <= 1:  # 只有主文件
        print("没有找到其他版本的factors文件用于对比")
        return

    print(f"发现{len(factor_files)}个factors相关文件:")
    for file in sorted(factor_files):
        if Path(file).exists():
            stat = Path(file).stat()
            size_mb = stat.st_size / 1024 / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"  {file}: {size_mb:.2f}MB, 修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        # 执行主要分析
        success = analyze_incremental_update()

        # 执行文件对比检查
        check_file_comparison()

        print("\n" + "=" * 80)
        print("分析完成")
        print("=" * 80)

        if success:
            print("✅ 增量更新分析完成，结果良好")
        else:
            print("⚠️ 增量更新分析完成，建议进一步检查")

    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()