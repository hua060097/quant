"""
诊断涨跌停数据是否有重复记录
检查是否因为 merge 导致特征文件重复
"""

import pandas as pd
from pathlib import Path

def main():
    limit_file = Path('C:/Users/86151/.trae-cn/pythons/path/to/local_data/limit_data/all_limit_data.parquet')

    if not limit_file.exists():
        print(f"文件不存在: {limit_file}")
        return

    print("=" * 80)
    print("涨跌停数据分析")
    print("=" * 80)

    # 读取数据
    df = pd.read_parquet(limit_file)

    print(f"\n[基本信息]")
    print(f"总行数: {len(df):,}")
    print(f"列名: {list(df.columns)}")

    # 检查20260109的数据
    print(f"\n[20260109数据分析]")

    if 'trade_date' in df.columns:
        df_0609 = df[df['trade_date'] == '20260109'].copy()
    elif 'date' in df.columns:
        df_0609 = df[df['date'].astype(str).str[:10] == '2026-01-09'].copy()
    else:
        print("未找到日期列")
        return

    print(f"20260109总行数: {len(df_0609):,}")

    # 检查是否有重复的 (ts_code, trade_date) 组合
    if 'ts_code' in df_0609.columns and 'trade_date' in df_0609.columns:
        duplicates = df_0609.duplicated(subset=['ts_code', 'trade_date'], keep=False)
        dup_count = duplicates.sum()

        print(f"\n[重复检查]")
        print(f"重复的(ts_code, trade_date)组合: {dup_count:,} 行")

        if dup_count > 0:
            print(f"\n[重复详情]")
            dup_stocks = df_0609[duplicates].groupby('ts_code').size().sort_values(ascending=False)
            print(f"涉及股票数: {len(dup_stocks)}")
            print(f"\n重复次数最多的前10只股票:")
            print(dup_stocks.head(10))

            # 检查000001.SZ
            if '000001.SZ' in df_0609['ts_code'].values:
                stock_000001 = df_0609[df_0609['ts_code'] == '000001.SZ']
                print(f"\n[000001.SZ详情]")
                print(f"000001.SZ在20260109的记录数: {len(stock_000001)}")
                print(stock_000001)
        else:
            print("✓ 无重复数据")
    else:
        print(f"列名检查: 'ts_code'存在: {'ts_code' in df.columns}, 'trade_date'存在: {'trade_date' in df.columns}")

    # 显示样本数据
    print(f"\n[样本数据 - 前5行]")
    print(df_0609.head(5))

    # 统计每只股票在20260109的记录数
    if 'ts_code' in df_0609.columns:
        print(f"\n[记录数分布]")
        record_counts = df_0609.groupby('ts_code').size()
        print(f"每只股票的平均记录数: {record_counts.mean():.2f}")
        print(f"最大记录数: {record_counts.max()}")
        print(f"最小记录数: {record_counts.min()}")

        if record_counts.max() > 1:
            print(f"\n有超过1条记录的股票数: {(record_counts > 1).sum()}")
            print(f"\n记录数>1的股票统计:")
            multi_record_stocks = record_counts[record_counts > 1].value_counts().sort_index()
            print(multi_record_stocks)

if __name__ == "__main__":
    main()
