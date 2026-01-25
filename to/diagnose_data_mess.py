"""
深度诊断：分析20260109数据混乱的原因

检查5001只股票的数据是否都只包含一个唯一的价格（即所有股票数据混在一起）
"""

import pandas as pd
from pathlib import Path

def main():
    daily_file = Path('local_data/daily/202601/daily_data.parquet')

    if not daily_file.exists():
        print(f"文件不存在: {daily_file}")
        return

    print("=" * 80)
    print("20260109数据混乱诊断")
    print("=" * 80)

    # 读取数据
    df = pd.read_parquet(daily_file)
    df_0609 = df[df['date_str'] == '20260109'].copy()

    print(f"\n[基本信息]")
    print(f"20260109总行数: {len(df_0609):,}")
    print(f"唯一股票代码数: {df_0609['stock_code'].nunique()}")

    # 关键检查：如果数据混乱，每只股票应该有多个不同的价格
    print(f"\n[混乱模式分析]")
    print(f"检查每只股票的价格分布...")

    # 统计每只股票的唯一价格数量
    stock_price_counts = df_0609.groupby('stock_code')['close'].nunique()

    print(f"\n每只股票的唯一close价格数量统计:")
    print(f"  平均值: {stock_price_counts.mean():.2f}")
    print(f"  中位数: {stock_price_counts.median():.2f}")
    print(f"  最小值: {stock_price_counts.min()}")
    print(f"  最大值: {stock_price_counts.max()}")

    # 如果只有1个唯一价格，说明数据完全混乱
    if stock_price_counts.max() == 1:
        print(f"\n[结论] 数据完全混乱！每只股票只有一个唯一的close价格。")
        print(f"这说明5001只股票的数据被混在一起，然后所有记录都被标记成了同一个stock_code。")

        # 检查实际有多少个不同的价格
        unique_prices = df_0609['close'].nunique()
        print(f"\n实际数据中包含 {unique_prices} 个不同的close价格")
        print(f"理论上有5001只股票，所以平均每只股票应该对应 {unique_prices / 5001:.2f} 个价格")

        # 查看stock_code的分布
        stock_counts = df_0609['stock_code'].value_counts()
        print(f"\nstock_code出现次数最多的前10个:")
        print(stock_counts.head(10))

        # 检查是否有500个股票出现次数都是500
        exactly_500 = (stock_counts == 500).sum()
        print(f"\n恰好出现500次的股票数量: {exactly_500}")

    else:
        print(f"\n[结论] 数据未完全混乱。每只股票有多个价格记录。")
        print(f"这可能是重复写入问题，而不是数据混乱。")

        # 显示一些样本
        print(f"\n[样本数据 - 随机选择3只股票]")
        sample_stocks = df_0609['stock_code'].unique()[:3]
        for stock in sample_stocks:
            stock_data = df_0609[df_0609['stock_code'] == stock]
            print(f"\n{stock}: {len(stock_data)} 条记录")
            print(stock_data[['date', 'open', 'close']].head(5))

if __name__ == "__main__":
    main()
