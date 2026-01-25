#!/usr/bin/env python3
"""
分析涨跌幅超过20%的异常数据

区分正常的高波动性和真正的异常数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_extreme_changes():
    """分析涨跌幅超过20%的异常数据"""
    print("=" * 80)
    print("涨跌幅超过20%的异常数据分析")
    print("=" * 80)

    # 读取factors.parquet文件
    factors_file = Path("factor_lab_data/factors.parquet")
    if not factors_file.exists():
        print("❌ factors.parquet文件不存在")
        return

    try:
        df = pd.read_parquet(factors_file)
        print(f"✅ 成功读取数据: {len(df):,}条记录")

        # 确保有pct_chg_1d列
        if 'pct_chg_1d' not in df.columns:
            print("❌ 未找到pct_chg_1d列")
            return

        # 找出涨跌幅超过20%的记录
        extreme_mask = df['pct_chg_1d'].abs() > 0.2
        extreme_data = df[extreme_mask].copy()

        print(f"\n📊 异常数据统计:")
        print(f"   超过20%涨跌幅的记录数: {len(extreme_data):,}")
        print(f"   占总记录比例: {len(extreme_data)/len(df)*100:.3f}%")

        if len(extreme_data) == 0:
            print("✅ 没有发现超过20%的涨跌幅数据")
            return

        # 转换日期格式
        extreme_data['date'] = pd.to_datetime(extreme_data['date'])

        # 按涨跌幅大小分类
        extreme_data['abs_pct_chg'] = extreme_data['pct_chg_1d'].abs()
        extreme_data['change_type'] = np.where(extreme_data['pct_chg_1d'] > 0, '上涨', '下跌')

        # 按涨跌幅幅度分类
        def categorize_extremeness(abs_change):
            if 0.2 <= abs_change < 0.3:
                return '轻度异常(20-30%)'
            elif 0.3 <= abs_change < 0.5:
                return '中度异常(30-50%)'
            elif 0.5 <= abs_change < 1.0:
                return '重度异常(50-100%)'
            else:
                return '极度异常(>100%)'

        extreme_data['extreme_level'] = extreme_data['abs_pct_chg'].apply(categorize_extremeness)

        # 按异常程度统计
        print(f"\n📈 异常程度分布:")
        level_stats = extreme_data['extreme_level'].value_counts()
        for level, count in level_stats.items():
            print(f"   {level}: {count:,}条")

        # 按涨跌方向统计
        print(f"\n📊 涨跌方向分布:")
        direction_stats = extreme_data['change_type'].value_counts()
        for direction, count in direction_stats.items():
            print(f"   {direction}: {count:,}条")

        # 按日期统计异常数据
        print(f"\n📅 按日期统计异常数据:")
        date_stats = extreme_data.groupby(extreme_data['date'].dt.strftime('%Y-%m-%d')).agg({
            'stock_code': 'count',
            'pct_chg_1d': ['max', 'min', 'mean']
        }).round(6)

        date_stats.columns = ['异常数量', '最大涨幅', '最大跌幅', '平均涨跌幅']
        date_stats = date_stats.sort_values('异常数量', ascending=False)

        for date, row in date_stats.head(10).iterrows():
            print(f"   {date}: {row['异常数量']:3d}条, 最大涨幅:{row['最大涨幅']:+.2%}, 最大跌幅:{row['最大跌幅']:+.2%}")

        # 找出最严重的异常数据
        print(f"\n🚨 最严重的20条异常数据:")
        extreme_sorted = extreme_data.sort_values('abs_pct_chg', ascending=False)

        for i, (_, row) in enumerate(extreme_sorted.head(20).iterrows(), 1):
            date_str = row['date'].strftime('%Y-%m-%d')
            stock_code = row['stock_code']
            pct_chg = row['pct_chg_1d']
            level = row['extreme_level']
            change_type = row['change_type']

            print(f"   {i:2d}. {date_str} {stock_code} {change_type}{pct_chg:+.2%} ({level})")

        # 统计哪些股票最容易出现异常
        print(f"\n🏢 异常数据最多的股票 (前20只):")
        stock_stats = extreme_data.groupby('stock_code').agg({
            'pct_chg_1d': 'count',
            'abs_pct_chg': 'max'
        }).round(6)
        stock_stats.columns = ['异常次数', '最大涨跌幅']
        stock_stats = stock_stats.sort_values('异常次数', ascending=False)

        for i, (stock_code, row) in enumerate(stock_stats.head(20).iterrows(), 1):
            print(f"   {i:2d}. {stock_code}: {row['异常次数']:2d}次, 最大涨跌幅:{row['最大涨跌幅']:+.2%}")

        # 分析正常与异常的界限
        print(f"\n📋 正常vs异常判断标准:")

        # 计算不同区间的分布
        ranges = [
            (0.2, 0.25, "20-25%: 可能正常的高波动"),
            (0.25, 0.3, "25-30%: 边缘异常，需关注"),
            (0.3, 0.4, "30-40%: 明显异常"),
            (0.4, 0.5, "40-50%: 严重异常"),
            (0.5, 1.0, "50-100%: 极度异常"),
            (1.0, float('inf'), "100%+: 数据错误或重大事件")
        ]

        for min_val, max_val, description in ranges:
            count = len(extreme_data[(extreme_data['abs_pct_chg'] >= min_val) &
                                   (extreme_data['abs_pct_chg'] < max_val)])
            print(f"   {description}: {count:,}条")

        # 给出具体建议
        print(f"\n💡 分析建议:")
        print(f"   1. 20-25%涨跌幅: 相对正常，可能是ST股票或新股的正常波动")
        print(f"   2. 25-30%涨跌幅: 需要重点关注，检查是否有特殊原因")
        print(f"   3. 30%以上涨跌幅: 强烈建议检查数据质量和重大事件")
        print(f"   4. 50%以上涨跌幅: 很可能是数据错误，需要人工核实")
        print(f"   5. 100%以上涨跌幅: 几乎可以肯定是数据问题")

        # 保存详细报告
        report_file = "extreme_changes_report.csv"
        extreme_sorted.to_csv(report_file, index=False, encoding='utf-8')
        print(f"\n💾 详细报告已保存至: {report_file}")

        return extreme_sorted

    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_data_quality():
    """检查数据质量"""
    print("\n" + "=" * 80)
    print("数据质量检查")
    print("=" * 80)

    try:
        df = pd.read_parquet("factor_lab_data/factors.parquet")

        # 检查pct_chg_1d的基本统计
        print(f"\n📊 pct_chg_1d基本统计:")
        stats = df['pct_chg_1d'].describe()
        print(f"   总记录数: {stats['count']:,.0f}")
        print(f"   均值: {stats['mean']:+.6f}")
        print(f"   标准差: {stats['std']:.6f}")
        print(f"   最小值: {stats['min']:+.6f}")
        print(f"   最大值: {stats['max']:+.6f}")
        print(f"   中位数: {df['pct_chg_1d'].median():+.6f}")

        # 检查是否有明显错误的数据
        print(f"\n🔍 明显错误数据检查:")

        # 涨跌幅超过±50%的（除了一些特殊情况，这通常不合理）
        very_extreme = df[df['pct_chg_1d'].abs() > 0.5]
        print(f"   涨跌幅超过±50%: {len(very_extreme):,}条")

        if len(very_extreme) > 0:
            print(f"   样例:")
            for _, row in very_extreme.head(5).iterrows():
                print(f"     {row['date']} {row['stock_code']} {row['pct_chg_1d']:+.2%}")

        # 检查涨跌幅为±100%的数据（通常是数据错误）
        exactly_100 = df[df['pct_chg_1d'].abs() == 1.0]
        print(f"   涨跌幅恰好±100%: {len(exactly_100):,}条（可能是数据错误）")

        # 检查缺失值
        missing_count = df['pct_chg_1d'].isna().sum()
        print(f"   缺失值: {missing_count:,}条")

        # 检查无穷值
        inf_count = np.isinf(df['pct_chg_1d']).sum()
        print(f"   无穷值: {inf_count:,}条")

    except Exception as e:
        print(f"❌ 数据质量检查失败: {e}")

if __name__ == "__main__":
    try:
        # 执行异常数据分析
        extreme_data = analyze_extreme_changes()

        # 执行数据质量检查
        check_data_quality()

        print("\n" + "=" * 80)
        print("分析完成")
        print("=" * 80)

        if extreme_data is not None:
            print("✅ 异常数据分析完成")
            print("💡 建议重点关注涨跌幅超过30%的数据，这些可能包含数据质量问题")
        else:
            print("❌ 异常数据分析失败")

    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()