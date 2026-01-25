#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看 factor_lab_data/factors.parquet 文件中的因子统计
"""

import pandas as pd
from pathlib import Path

def check_factors_count():
    """检查因子文件"""
    # 文件路径（可能是绝对路径或相对路径）
    possible_paths = [
        Path("C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/factor_lab_data/factors.parquet"),
        Path("C:/Users/86151/.trae-cn/pythons/path/to/factor_lab_data/factors.parquet"),
        Path("../factor_lab_data/factors.parquet"),
        Path("factor_lab_data/factors.parquet"),
    ]

    factors_file = None
    for path in possible_paths:
        if path.exists():
            factors_file = path
            break

    print("=" * 100)
    print("因子文件检查")
    print("=" * 100)

    if factors_file is None:
        print("\n[ERROR] 未找到 factors.parquet 文件")
        print("\n已尝试的路径:")
        for path in possible_paths:
            print(f"  - {path}")
        return

    print(f"\n[OK] 找到文件: {factors_file}")
    print(f"文件大小: {factors_file.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        # 读取数据
        print("\n[*] 正在读取数据...")
        df = pd.read_parquet(factors_file)

        print(f"[OK] 成功读取数据")
        print(f"  总行数: {len(df):,}")
        print(f"  总列数: {len(df.columns):,}")

        # 显示列名（因子名）
        print("\n" + "=" * 100)
        print("因子列表（所有列名）:")
        print("=" * 100)

        columns = list(df.columns)
        factor_count = len(columns)

        # 排除可能的非因子列
        non_factor_cols = ['date', 'stock_code', 'date_str', 'time']
        factor_cols = [col for col in columns if col not in non_factor_cols]

        print(f"\n总列数: {factor_count}")
        print(f"因子列数（排除日期/股票代码）: {len(factor_cols)}")
        print()

        # 分组显示
        print("所有列名:")
        for i, col in enumerate(columns, 1):
            col_type = "元数据" if col in non_factor_cols else "因子"
            print(f"  {i:3d}. {col:50s} [{col_type}]")

        # 检查数据样本
        print("\n" + "=" * 100)
        print("数据样本（前5行）:")
        print("=" * 100)

        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)

        print(df.head().to_string())

        # 检查缺失值
        print("\n" + "=" * 100)
        print("缺失值统计:")
        print("=" * 100)

        missing_stats = df.isnull().sum()
        missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)

        if len(missing_stats) > 0:
            print(f"\n有缺失值的列数: {len(missing_stats)}")
            print("\n前20个缺失值最多的列:")
            for col, count in missing_stats.head(20).items():
                missing_pct = count / len(df) * 100
                print(f"  {col:50s}: {count:6d} ({missing_pct:5.2f}%)")
        else:
            print("\n✓ 所有列都没有缺失值")

        # 检查数据范围
        print("\n" + "=" * 100)
        print("数据类型统计:")
        print("=" * 100)

        dtype_stats = df.dtypes.value_counts()
        for dtype, count in dtype_stats.items():
            print(f"  {dtype}: {count} 列")

    except Exception as e:
        print(f"\n[ERROR] 读取失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_factors_count()
