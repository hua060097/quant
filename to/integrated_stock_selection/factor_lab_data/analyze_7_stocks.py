"""
详细分析7只股票的数据分布情况
"""
import pandas as pd
from pathlib import Path

def analyze_7_stocks_details(file1_path, file2_path):
    """详细分析7只新增股票的数据"""

    print("=" * 100)
    print("7只新增股票详细分析")
    print("=" * 100)

    # 读取文件
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    print(f"\n[基本信息]")
    print(f"  文件1 (副本): {df1.shape[0]:,} 行")
    print(f"  文件2 (当前): {df2.shape[0]:,} 行")

    # 创建复合键
    df1['_key'] = df1['stock_code'].astype(str) + '_' + df1['date'].astype(str)
    df2['_key'] = df2['stock_code'].astype(str) + '_' + df2['date'].astype(str)

    keys1 = set(df1['_key'])
    keys2 = set(df2['_key'])

    only_in_2 = keys2 - keys1
    only_in_1 = keys1 - keys2

    print(f"\n[行级差异]")
    print(f"  仅在文件1: {len(only_in_1):,} 行")
    print(f"  仅在文件2: {len(only_in_2):,} 行")

    # 分析仅在文件2中的行
    if len(only_in_2) > 0:
        diff_df2 = df2[df2['_key'].isin(only_in_2)].copy()

        # 获取这7只股票
        stocks_7 = diff_df2['stock_code'].unique().tolist()

        print(f"\n[7只股票详情]")
        print(f"  股票数量: {len(stocks_7)}")
        print(f"  股票代码: {', '.join(stocks_7)}")

        # 显示每只股票的详细信息
        print(f"\n[每只股票的新增数据]")
        for stock in stocks_7:
            stock_data = diff_df2[diff_df2['stock_code'] == stock]
            dates = sorted(stock_data['date'].unique())

            print(f"\n  {stock}:")
            print(f"    新增行数: {len(stock_data)}")
            print(f"    日期: {', '.join(str(d) for d in dates)}")

            # 显示一些关键因子的值
            if len(stock_data) > 0:
                sample_row = stock_data.iloc[0]
                print(f"    示例数据（第一行 {sample_row['date']}）:")

                key_factors = ['pct_chg_1d', 'market_trend_adaptive_beta',
                              'sector_broken_resilience_alpha', 'beta_60d']

                for factor in key_factors:
                    if factor in sample_row:
                        val = sample_row[factor]
                        if pd.isna(val):
                            print(f"      {factor}: NaN")
                        else:
                            print(f"      {factor}: {val:.6f}")

        # 检查这7只股票在两个文件中的完整数据
        print(f"\n[7只股票在两个文件中的完整数据分布]")
        for stock in stocks_7:
            stock_df1 = df1[df1['stock_code'] == stock]
            stock_df2 = df2[df2['stock_code'] == stock]

            print(f"\n  {stock}:")
            print(f"    文件1: {len(stock_df1):,} 行")
            print(f"    文件2: {len(stock_df2):,} 行")
            print(f"    差异: {len(stock_df2) - len(stock_df1):+,} 行")

            if len(stock_df1) > 0:
                dates1 = sorted(stock_df1['date'].unique())
                print(f"    文件1日期范围: {dates1[0]} 至 {dates1[-1]} (共{len(dates1)}个交易日)")

            if len(stock_df2) > 0:
                dates2 = sorted(stock_df2['date'].unique())
                print(f"    文件2日期范围: {dates2[0]} 至 {dates2[-1]} (共{len(dates2)}个交易日)")

            # 找出这只股票在文件2中独有日期的数据
            stock_only_in_2 = diff_df2[diff_df2['stock_code'] == stock]
            if len(stock_only_in_2) > 0:
                only_dates = sorted(stock_only_in_2['date'].unique())
                print(f"    仅在文件2中的日期: {', '.join(str(d) for d in only_dates)}")

        # 检查这7只股票是否在副本文件中完全不存在
        print(f"\n[7只股票在副本文件中的存在情况]")
        for stock in stocks_7:
            stock_in_file1 = df1[df1['stock_code'] == stock]

            if len(stock_in_file1) == 0:
                print(f"  {stock}: 副本文件中完全不存在（0行）")
            else:
                dates1 = sorted(stock_in_file1['date'].unique())
                print(f"  {stock}: 副本文件中存在（{len(stock_in_file1)}行，{len(dates1)}个交易日）")
                print(f"           日期范围: {dates1[0]} 至 {dates1[-1]}")

        # 检查整个数据集的日期分布
        print(f"\n[两个文件的日期范围对比]")

        dates1 = sorted(df1['date'].unique())
        dates2 = sorted(df2['date'].unique())

        print(f"  文件1:")
        print(f"    日期范围: {dates1[0]} 至 {dates1[-1]}")
        print(f"    交易日数量: {len(dates1)}")

        print(f"  文件2:")
        print(f"    日期范围: {dates2[0]} 至 {dates2[-1]}")
        print(f"    交易日数量: {len(dates2)}")

        # 找出新增的日期
        dates_only_in_2 = set(dates2) - set(dates1)
        if dates_only_in_2:
            print(f"\n  文件2中新增的日期 ({len(dates_only_in_2)}个):")
            for date in sorted(dates_only_in_2):
                date_df = df2[df2['date'] == date]
                stock_count = date_df['stock_code'].nunique()
                print(f"    {date}: {stock_count}只股票")

        # 分析这7只股票的板块信息（如果有）
        if 'sw_sector' in df2.columns:
            print(f"\n[7只股票的板块信息]")
            for stock in stocks_7:
                stock_data = df2[df2['stock_code'] == stock].iloc[0]
                sector = stock_data.get('sw_sector', '未知')
                print(f"  {stock}: {sector}")

    print(f"\n[分析完成]")

if __name__ == "__main__":
    # 文件路径
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")
    file1 = base_path / "factors - 副本.parquet"
    file2 = base_path / "factors.parquet"

    analyze_7_stocks_details(file1, file2)
