#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查缓存文件中20260106日的数据质量
"""

import pandas as pd
from pathlib import Path

def check_cache_data():
    """检查缓存数据"""
    cache_file = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\cache\stock_features_20250708_20260106.parquet")
    target_date = "20260106"

    print("=" * 100)
    print(f"检查缓存文件: {cache_file}")
    print(f"目标日期: {target_date}")
    print("=" * 100)
    print()

    # 检查文件是否存在
    if not cache_file.exists():
        print(f"错误: 缓存文件不存在: {cache_file}")
        return

    try:
        # 读取缓存文件
        print("正在读取缓存文件...")
        df = pd.read_parquet(cache_file)

        print(f"✅ 文件读取成功")
        print(f"总数据行数: {len(df):,}")
        print(f"总列数: {len(df.columns)}")
        print(f"总字段: {list(df.columns)}")
        print()

        # 检查date字段
        if 'date' not in df.columns:
            print("错误: 数据中没有 'date' 字段")
            return

        # 转换date字段
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # 检查日期范围
        print(f"数据日期范围: {df['date'].min()} ~ {df['date'].max()}")
        print(f"包含的不同日期数: {df['date'].nunique()}")
        print()

        # 筛选目标日期
        print(f"筛选日期: {target_date}")
        target_date_dt = pd.to_datetime(target_date)

        # 尝试不同的日期格式匹配
        if df['date'].dtype == 'object':
            # 如果是字符串类型
            df_filtered = df[df['date'].astype(str).str.contains(target_date)]
        else:
            # 如果是datetime类型
            df_filtered = df[df['date'].astype(str).str.contains(target_date)]

        if len(df_filtered) == 0:
            print(f"警告: 未找到 {target_date} 的数据")
            print()
            print("检查可用的日期:")
            available_dates = df['date'].unique()
            if len(available_dates) <= 20:
                for d in sorted(available_dates):
                    print(f"  - {d}")
            else:
                print(f"  最新10个日期:")
                for d in sorted(available_dates)[-10:]:
                    print(f"  - {d}")
            return

        print(f"✅ 找到 {len(df_filtered):,} 条数据")
        print()

        # 详细分析目标日期的数据
        print("=" * 100)
        print(f"{target_date} 数据详细分析")
        print("=" * 100)
        print()

        # 基本统计
        print(f"1. 数据量统计:")
        print(f"   总行数: {len(df_filtered):,}")
        if 'stock_code' in df_filtered.columns:
            unique_stocks = df_filtered['stock_code'].nunique()
            print(f"   股票数量: {unique_stocks:,}")
            print(f"   平均每只股票数据行数: {len(df_filtered) / unique_stocks:.1f}")
        print()

        # 字段完整性检查
        print(f"2. 字段完整性检查:")
        for col in df_filtered.columns:
            total = len(df_filtered)
            non_null = df_filtered[col].notna().sum()
            null_count = total - non_null
            null_pct = (null_count / total) * 100

            status = "✅" if null_count == 0 else "⚠️" if null_pct < 50 else "❌"
            print(f"   {status} {col}: {non_null}/{total} ({100-null_pct:.1f}%)")

            if null_count > 0:
                print(f"      缺失 {null_count:,} 行 ({null_pct:.2f}%)")
        print()

        # 关键数值字段统计
        numeric_cols = df_filtered.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            print(f"3. 数值字段统计 (前10个):")
            for col in numeric_cols[:10]:
                if col in ['stock_code', 'date']:
                    continue
                series = df_filtered[col]
                valid_data = series.dropna()

                if len(valid_data) > 0:
                    print(f"   {col}:")
                    print(f"     有效数据: {len(valid_data):,} 行")
                    print(f"     最小值: {valid_data.min():.6f}")
                    print(f"     最大值: {valid_data.max():.6f}")
                    print(f"     平均值: {valid_data.mean():.6f}")
                    print(f"     标准差: {valid_data.std():.6f}")
                    print(f"     唯一值数量: {valid_data.nunique()}")
                else:
                    print(f"   {col}: 全部缺失 ❌")
            print()

        # 检查是否有异常值
        print(f"4. 异常值检查:")
        for col in ['close', 'volume', 'amount']:  # 检查常见字段
            if col in df_filtered.columns:
                series = df_filtered[col]
                valid_data = series.dropna()

                if len(valid_data) > 0:
                    # 检查是否有负值
                    negative_count = (valid_data < 0).sum()
                    if negative_count > 0:
                        print(f"   ⚠️ {col}: 发现 {negative_count} 个负值")

                    # 检查是否有零值（对于价格和成交量可能异常）
                    zero_count = (valid_data == 0).sum()
                    if zero_count > 0:
                        print(f"   ⚠️ {col}: 发现 {zero_count} 个零值")

                    # 检查是否有极端值
                    q1 = valid_data.quantile(0.25)
                    q3 = valid_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    outliers = ((valid_data < lower_bound) | (valid_data > upper_bound)).sum()
                    if outliers > 0:
                        print(f"   ⚠️ {col}: 发现 {outliers} 个极端异常值 (IQR方法)")
        print()

        # 数据示例
        print(f"5. 数据示例 (前5行):")
        display_cols = df_filtered.columns.tolist()[:10]  # 只显示前10列
        print(df_filtered[display_cols].head().to_string())
        print()

        print("=" * 100)
        print(f"✅ {target_date} 数据检查完成")
        print("=" * 100)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_cache_data()
