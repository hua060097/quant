#!/usr/bin/env python3
"""
在整个数据集中查找17.12这个价格的来源
"""

import pandas as pd
from pathlib import Path

def search_price_17_12():
    """搜索17.12价格"""
    target_price = 17.12
    stock_code = '000002.SZ'

    print("=" * 100)
    print(f"搜索价格: {target_price}")
    print(f"目标股票: {stock_code}")
    print("=" * 100)

    base_dir = Path(__file__).parent.parent / "local_data"

    # 搜索所有月度数据文件
    daily_dir = base_dir / "daily"

    if not daily_dir.exists():
        print(f"[ERROR] 日线数据目录不存在: {daily_dir}")
        return

    # 获取所有parquet文件
    parquet_files = list(daily_dir.rglob("*.parquet"))

    print(f"\n[*] 找到 {len(parquet_files)} 个数据文件")

    found_records = []

    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)

            # 筛选目标股票
            stock_df = df[df['stock_code'] == stock_code].copy()

            if len(stock_df) == 0:
                continue

            # 检查open和close列
            for col in ['open', 'close', 'high', 'low']:
                if col in stock_df.columns:
                    # 使用浮点数比较，允许小的误差
                    matches = stock_df[abs(stock_df[col] - target_price) < 0.01]

                    if len(matches) > 0:
                        for idx, row in matches.iterrows():
                            record = {
                                'file': file_path.name,
                                'date': row.get('date', 'Unknown'),
                                'column': col,
                                'price': row[col],
                                'open': row.get('open', 'N/A'),
                                'close': row.get('close', 'N/A')
                            }
                            found_records.append(record)

        except Exception as e:
            print(f"[WARNING] 读取文件失败 {file_path.name}: {e}")

    # 显示结果
    print(f"\n{'=' * 100}")
    print(f"搜索结果:")
    print(f"{'=' * 100}")

    if len(found_records) > 0:
        print(f"\n找到 {len(found_records)} 条记录:\n")

        for i, record in enumerate(found_records, 1):
            print(f"{i}. 文件: {record['file']}")
            print(f"   日期: {record['date']}")
            print(f"   字段: {record['column']} = {record['price']:.2f}")
            print(f"   Open: {record['open']}, Close: {record['close']}")
            print()
    else:
        print("\n未找到17.12这个价格")
        print("\n可能的原因:")
        print("  1. 该价格不在日线数据中")
        print("  2. 该价格是前复权价格，需要检查前复权字段")
        print("  3. 该价格是在特征生成过程中计算错误产生的")

        # 检查前复权字段
        print(f"\n检查前复权字段...")
        for file_path in parquet_files[:3]:  # 只检查前3个文件
            try:
                df = pd.read_parquet(file_path)
                stock_df = df[df['stock_code'] == stock_code].copy()

                if len(stock_df) > 0:
                    front_cols = [col for col in stock_df.columns if 'front' in col.lower()]
                    if front_cols:
                        print(f"\n文件 {file_path.name} 包含前复权字段:")
                        print(f"  {front_cols}")

                        # 显示前复权数据样本
                        for col in front_cols[:2]:  # 只显示前2个
                            if col in stock_df.columns:
                                print(f"\n  {col} 样本数据（前5条）:")
                                sample = stock_df[['date', col]].head()
                                for idx, row in sample.iterrows():
                                    print(f"    {row['date']}: {row[col]:.2f}")

            except Exception as e:
                pass


if __name__ == "__main__":
    search_price_17_12()
