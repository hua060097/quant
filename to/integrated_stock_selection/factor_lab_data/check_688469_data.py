"""
检查股票 688469.SH 在 2023-05-10 至 2023-06-04 之间的原始数据
"""
import pandas as pd
import numpy as np
from pathlib import Path

def check_stock_data(stock_code, start_date, end_date):
    """
    检查指定股票在日期范围内的原始数据

    参数:
        stock_code: 股票代码（如 '688469.SH'）
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    """

    print("=" * 100)
    print(f"检查股票 {stock_code} 在 {start_date} 至 {end_date} 之间的原始数据")
    print("=" * 100)

    # 数据目录
    data_base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\daily")

    # 目标日期范围
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # 需要检查的年月目录：202305和202306
    months_to_check = ['202305', '202306']

    all_data = []

    for month in months_to_check:
        month_path = data_base_path / month / "daily_data.parquet"

        if not month_path.exists():
            print(f"\n[WARNING] 文件不存在: {month_path}")
            continue

        print(f"\n[读取文件] {month_path.name}")
        try:
            df_month = pd.read_parquet(month_path)
            print(f"  文件形状: {df_month.shape}")

            # 检查是否包含目标股票
            if stock_code in df_month.columns:
                print(f"  找到股票 {stock_code}")
            else:
                print(f"  未找到股票 {stock_code}")
                continue

            # 提取该股票的数据（需要转置，因为日期是索引）
            # 但首先需要确认数据结构
            print(f"  列名示例: {list(df_month.columns[:5])}")
            print(f"  索引类型: {type(df_month.index)}")

            # 假设数据格式是：index是股票代码，columns是日期
            # 或者：index是日期，columns是股票代码

            # 先检查索引
            if stock_code in df_month.index:
                # 股票代码在索引中
                stock_data = df_month.loc[stock_code, :]

                # 转换为Series以便处理
                if isinstance(stock_data, pd.Series):
                    # 将索引（日期）转换为datetime
                    stock_data.index = pd.to_datetime(stock_data.index, format='%Y%m%d', errors='coerce')

                    # 过滤日期范围
                    mask = (stock_data.index >= start_ts) & (stock_data.index <= end_ts)
                    data_in_range = stock_data[mask]

                    if len(data_in_range) > 0:
                        all_data.append(data_in_range)
                        print(f"  在该月找到 {len(data_in_range)} 条记录")
                        for date, val in data_in_range.items():
                            if pd.notna(val):
                                print(f"    - {date.date()}: {val}")
                else:
                    print(f"  [WARNING] 数据格式异常")

            elif stock_code in df_month.columns:
                # 股票代码在列中
                # 索引应该是日期
                stock_series = df_month[stock_code]

                # 转换索引为datetime
                try:
                    stock_series.index = pd.to_datetime(stock_series.index, format='%Y%m%d', errors='coerce')
                except:
                    stock_series.index = pd.to_datetime(stock_series.index, errors='coerce')

                # 过滤日期范围
                mask = (stock_series.index >= start_ts) & (stock_series.index <= end_ts)
                data_in_range = stock_series[mask]

                if len(data_in_range) > 0:
                    all_data.append(data_in_range)
                    print(f"  在该月找到 {len(data_in_range)} 条记录")
                    for date, val in data_in_range.items():
                        if pd.notna(val):
                            print(f"    - {date.date()}: {val}")

        except Exception as e:
            print(f"  [ERROR] 读取文件失败: {str(e)}")
            import traceback
            traceback.print_exc()

    # 汇总结果
    print(f"\n{'=' * 100}")
    print(f"汇总结果")
    print(f"{'=' * 100}")

    if all_data:
        combined_data = pd.concat(all_data)
        combined_data = combined_data.sort_index()

        print(f"\n在 {start_date} 至 {end_date} 之间:")
        print(f"  总记录数: {len(combined_data)}")

        # 统计非空值
        non_null_count = combined_data.notna().sum()
        print(f"  非空记录数: {non_null_count}")

        print(f"\n详细数据:")
        print(f"{'日期':<15} {'值':<20} {'是否非空':<10}")
        print("-" * 50)

        for date, val in combined_data.items():
            is_non_null = pd.notna(val)
            print(f"{str(date.date()):<15} {str(val):<20} {'是' if is_non_null else '否':<10}")

        # 验证新股过滤逻辑
        print(f"\n[验证新股过滤逻辑]")
        print(f"  IPO日期: {start_date}")
        print(f"  过滤至: {end_date}")
        print(f"  日志提示: '过滤 2 条记录'")
        print(f"  实际数据: {non_null_count} 条记录")

        if non_null_count == 2:
            print(f"  ✓ 数据一致！日志正确。")
        else:
            print(f"  ✗ 数据不一致！应该过滤 {non_null_count} 条记录。")

    else:
        print(f"\n在 {start_date} 至 {end_date} 之间没有找到数据")
        print(f"  [WARNING] 这可能意味着:")
        print(f"    1. 股票 {stock_code} 不在数据集中")
        print(f"    2. 数据文件缺失")
        print(f"    3. 日期范围内无交易")

if __name__ == "__main__":
    # 检查股票 688469.SH 在 2023-05-10 至 2023-06-04 之间的数据
    check_stock_data(
        stock_code='688469.SH',
        start_date='2023-05-10',
        end_date='2023-06-04'
    )
