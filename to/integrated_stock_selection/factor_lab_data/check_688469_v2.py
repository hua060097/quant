"""
检查股票 688469.SH 在 2023-05-10 至 2023-06-04 之间的原始数据
"""
import pandas as pd
from pathlib import Path

def check_stock_data(stock_code, start_date, end_date):
    """
    检查指定股票在日期范围内的原始数据
    """

    print("=" * 100)
    print(f"检查股票 {stock_code} 在 {start_date} 至 {end_date} 之间的原始数据")
    print("=" * 100)

    # 数据目录
    data_base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\daily")

    # 目标日期范围
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # 需要检查的年月目录
    months_to_check = ['202305', '202306']

    all_data = []

    for month in months_to_check:
        month_path = data_base_path / month / "daily_data.parquet"

        if not month_path.exists():
            print(f"\n[WARNING] 文件不存在: {month_path}")
            continue

        print(f"\n[读取文件] {month}")
        try:
            df_month = pd.read_parquet(month_path)
            print(f"  文件形状: {df_month.shape}")

            # 过滤目标股票
            stock_data = df_month[df_month['stock_code'] == stock_code].copy()

            if len(stock_data) == 0:
                print(f"  未找到股票 {stock_code}")
                continue

            print(f"  找到 {len(stock_data)} 条记录（所有日期）")

            # 转换日期列
            stock_data['date'] = pd.to_datetime(stock_data['date'])

            # 过滤日期范围
            mask = (stock_data['date'] >= start_ts) & (stock_data['date'] <= end_ts)
            data_in_range = stock_data[mask]

            if len(data_in_range) > 0:
                all_data.append(data_in_range)
                print(f"  在日期范围内找到 {len(data_in_range)} 条记录")

                # 显示详细数据
                for _, row in data_in_range.iterrows():
                    print(f"    - {row['date'].date()}: close={row['close']:.2f}, pct_chg={row['pct_chg']:.2f}%")
            else:
                print(f"  在日期范围内无记录")

        except Exception as e:
            print(f"  [ERROR] 读取文件失败: {str(e)}")
            import traceback
            traceback.print_exc()

    # 汇总结果
    print(f"\n{'=' * 100}")
    print(f"汇总结果")
    print(f"{'=' * 100}")

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values('date').reset_index(drop=True)

        print(f"\n在 {start_date} 至 {end_date} 之间:")
        print(f"  总记录数: {len(combined_data)}")

        print(f"\n详细数据列表:")
        print(f"{'序号':<6} {'日期':<15} {'开盘':<10} {'最高':<10} {'最低':<10} {'收盘':<10} {'涨跌幅%':<10}")
        print("-" * 90)

        for idx, row in combined_data.iterrows():
            print(f"{idx+1:<6} {str(row['date'].date()):<15} {row['open']:<10.2f} {row['high']:<10.2f} {row['low']:<10.2f} {row['close']:<10.2f} {row['pct_chg']:<10.2f}")

        # 验证新股过滤逻辑
        print(f"\n[验证新股过滤逻辑]")
        print(f"  股票代码: {stock_code}")
        print(f"  IPO日期: {start_date}")
        print(f"  过滤至: {end_date}")
        print(f"  日志提示: '过滤 2 条记录'")
        print(f"  实际数据: {len(combined_data)} 条记录")

        if len(combined_data) == 2:
            print(f"  [OK] 数据一致！日志正确。")
        else:
            print(f"  [WARNING] 数据不一致！应该过滤 {len(combined_data)} 条记录。")

        # 检查数据完整性
        print(f"\n[数据完整性检查]")
        for col in ['open', 'high', 'low', 'close', 'pct_chg']:
            null_count = combined_data[col].isna().sum()
            if null_count > 0:
                print(f"  {col}: {null_count} 个空值")
            else:
                print(f"  {col}: 无空值")

    else:
        print(f"\n在 {start_date} 至 {end_date} 之间没有找到数据")
        print(f"  [WARNING] 这可能意味着:")
        print(f"    1. 股票 {stock_code} 不在数据集中")
        print(f"    2. 数据文件缺失")
        print(f"    3. 日期范围内无交易")
        print(f"    4. 股票代码格式不对（可能是其他交易所）")

        # 尝试搜索类似的股票代码
        print(f"\n[尝试搜索相似的股票代码]")
        for month in months_to_check:
            month_path = data_base_path / month / "daily_data.parquet"
            if month_path.exists():
                df_month = pd.read_parquet(month_path)
                similar = df_month[df_month['stock_code'].str.contains('688469', na=False)]
                if len(similar) > 0:
                    print(f"  在 {month} 找到: {similar['stock_code'].unique().tolist()}")

if __name__ == "__main__":
    # 检查股票 688469.SH 在 2023-05-10 至 2023-06-04 之间的数据
    check_stock_data(
        stock_code='688469.SH',
        start_date='2023-05-10',
        end_date='2023-06-04'
    )
