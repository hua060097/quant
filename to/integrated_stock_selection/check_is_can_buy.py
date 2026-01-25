#!/usr/bin/env python3
"""
检查指定日期的特征文件中 is_can_buy == 0 的股票统计
"""

import pandas as pd
from pathlib import Path

def check_is_can_buy_stats(date_str: str = "20251021"):
    """
    检查指定日期的 is_can_buy 统计信息

    Args:
        date_str: 日期字符串，格式 YYYYMMDD
    """
    # 构建文件路径
    file_path = Path(__file__).parent.parent / "local_data" / "features" / "daily" / f"{date_str}.parquet"

    print("=" * 80)
    print(f"检查文件: {file_path}")
    print("=" * 80)

    # 检查文件是否存在
    if not file_path.exists():
        print(f"[ERROR] 文件不存在: {file_path}")
        return

    # 读取数据
    print(f"\n[*] 正在读取数据...")
    try:
        df = pd.read_parquet(file_path)
        print(f"[OK] 成功读取数据: {len(df):,} 行")
    except Exception as e:
        print(f"[ERROR] 读取失败: {e}")
        return

    # 基本统计
    print(f"\n{'=' * 80}")
    print(f"基本统计信息")
    print(f"{'=' * 80}")
    print(f"总行数: {len(df):,}")
    print(f"总列数: {len(df.columns)}")

    # 检查 is_can_buy 列是否存在
    if 'is_can_buy' not in df.columns:
        print(f"\n[ERROR] 数据中不存在 'is_can_buy' 列")
        print(f"可用列: {list(df.columns)[:10]}...")
        return

    # 统计 is_can_buy 的分布
    print(f"\n{'=' * 80}")
    print(f"is_can_buy 字段统计")
    print(f"{'=' * 80}")

    value_counts = df['is_can_buy'].value_counts().sort_index()
    total = len(df)

    for value, count in value_counts.items():
        percentage = (count / total) * 100
        print(f"  is_can_buy = {value}: {count:,} ({percentage:.2f}%)")

    # 详细统计 is_can_buy == 0
    can_buy_0_df = df[df['is_can_buy'] == 0]
    print(f"\n{'=' * 80}")
    print(f"is_can_buy == 0 详细统计")
    print(f"{'=' * 80}")
    print(f"股票数量: {len(can_buy_0_df):,}")
    print(f"占比: {(len(can_buy_0_df) / total) * 100:.2f}%")

    # 显示前20个股票
    if 'stock_code' in df.columns:
        print(f"\n前20个 is_can_buy == 0 的股票:")
        stock_codes = can_buy_0_df['stock_code'].head(20).tolist()
        for i, code in enumerate(stock_codes, 1):
            print(f"  {i:2d}. {code}")

        if len(can_buy_0_df) > 20:
            print(f"  ... 还有 {len(can_buy_0_df) - 20:,} 只股票")

    # 检查是否有其他相关字段
    print(f"\n{'=' * 80}")
    print(f"相关字段检查")
    print(f"{'=' * 80}")

    if 'stock_code' in df.columns:
        unique_stocks = df['stock_code'].nunique()
        print(f"唯一股票数: {unique_stocks:,}")

    if 'date' in df.columns:
        unique_dates = df['date'].unique()
        print(f"日期范围: {unique_dates}")

    # 检查可能的布尔类型
    if can_buy_0_df is not None and len(can_buy_0_df) > 0:
        print(f"\n{'=' * 80}")
        print(f"is_can_buy == 0 的数据样本")
        print(f"{'=' * 80}")
        sample_cols = ['stock_code', 'is_can_buy'] if 'stock_code' in df.columns else ['is_can_buy']

        # 添加可能相关的列
        related_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['buy', 'limit', 'gap', 'industry', 'sector'])]
        sample_cols.extend(related_cols[:5])  # 最多显示5个相关列

        available_cols = [col for col in sample_cols if col in df.columns]
        print(can_buy_0_df[available_cols].head(10).to_string())

    print(f"\n{'=' * 80}")
    print(f"[OK] 检查完成")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    import sys

    # 支持命令行参数指定日期
    date_str = sys.argv[1] if len(sys.argv) > 1 else "20251021"

    check_is_can_buy_stats(date_str)
