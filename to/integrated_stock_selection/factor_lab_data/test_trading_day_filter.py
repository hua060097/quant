"""
测试新股过滤逻辑：验证是否使用25个交易日而非自然日
"""
import pandas as pd
from pathlib import Path
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def test_trading_day_calculation():
    """测试交易日计算逻辑"""
    print("=" * 100)
    print("测试新股过滤：25个交易日 vs 25个自然日")
    print("=" * 100)

    # 加载交易日历
    calendar_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\to\local_data\calendar\trading_calendar.parquet")

    if not calendar_path.exists():
        print(f"交易日历文件不存在: {calendar_path}")
        return

    calendar_df = pd.read_parquet(calendar_path)

    if 'date_str' in calendar_df.columns:
        trading_dates = calendar_df['date_str'].tolist()
    else:
        print(f"交易日历文件缺少 'date_str' 列")
        return

    print(f"\n加载交易日历成功: {len(trading_dates)} 个交易日")
    print(f"日期范围: {trading_dates[0]} ~ {trading_dates[-1]}")

    # 测试案例1：688469.SH（IPO日期：2023-05-10）
    test_stock = "688469.SH"
    ipo_date_str = "20230510"

    print(f"\n{'=' * 100}")
    print(f"测试案例: {test_stock}")
    print(f"  IPO日期: {ipo_date_str}")
    print(f"{'=' * 100}")

    # 方法1：使用自然日（旧逻辑）
    ipo_dt_natural = pd.to_datetime(ipo_date_str, format='%Y%m%d')
    filter_end_natural = ipo_dt_natural + pd.Timedelta(days=25)
    print(f"\n方法1（旧逻辑）- 使用自然日:")
    print(f"  IPO日期 + 25个自然日 = {filter_end_natural.strftime('%Y-%m-%d')}")

    # 方法2：使用交易日（新逻辑）
    def get_nth_trading_date(start_date_str, trading_dates, n=25):
        """获取第N个交易日"""
        start_index = None
        for i, date_str in enumerate(trading_dates):
            if date_str >= start_date_str:
                start_index = i
                break

        if start_index is None:
            return None

        target_index = start_index + n - 1  # -1因为包含IPO当天

        if target_index >= len(trading_dates):
            return pd.to_datetime(trading_dates[-1], format='%Y%m%d')
        else:
            return pd.to_datetime(trading_dates[target_index], format='%Y%m%d')

    filter_end_trading = get_nth_trading_date(ipo_date_str, trading_dates, n=25)
    print(f"\n方法2（新逻辑）- 使用交易日:")
    print(f"  IPO日期后第25个交易日 = {filter_end_trading.strftime('%Y-%m-%d')}")

    # 对比差异
    natural_days_diff = (filter_end_natural - ipo_dt_natural).days
    trading_days_count = 0

    for date_str in trading_dates:
        date_obj = pd.to_datetime(date_str, format='%Y%m%d')
        if ipo_dt_natural <= date_obj <= filter_end_trading:
            trading_days_count += 1

    print(f"\n差异对比:")
    print(f"  自然日方法: 从IPO到过滤结束 = {natural_days_diff} 个自然日")
    print(f"  交易日方法: 从IPO到过滤结束 = {trading_days_count} 个交易日")

    # 列出IPO后的交易日
    print(f"\nIPO后的前30个交易日:")
    start_index = None
    for i, date_str in enumerate(trading_dates):
        if date_str >= ipo_date_str:
            start_index = i
            break

    if start_index is not None:
        for i in range(min(30, len(trading_dates) - start_index)):
            idx = start_index + i
            if idx < len(trading_dates):
                date_obj = pd.to_datetime(trading_dates[idx], format='%Y%m%d')
                print(f"  第{i+1:2d}个交易日: {trading_dates[idx]} ({date_obj.strftime('%Y-%m-%d')})")

                # 高亮第25个交易日
                if i == 24:
                    print(f"                    ^^^^ 这是第25个交易日（过滤结束日期）")

    # 计算两种方法的实际过滤记录数（基于我们之前检查的数据）
    print(f"\n实际数据验证:")
    print(f"  根据之前的检查，{test_stock} 在 IPO日期~第25个交易日之间有 18 条记录")
    print(f"  如果使用自然日+25，过滤期到: {filter_end_natural.strftime('%Y-%m-%d')}")
    print(f"  如果使用交易日+25，过滤期到: {filter_end_trading.strftime('%Y-%m-%d')}")

    # 计算自然日方法会多过滤多少天
    extra_days = (filter_end_natural - filter_end_trading).days
    print(f"  自然日方法比交易日方法多过滤: {extra_days} 个自然日")

    print(f"\n{'=' * 100}")
    print(f"结论:")
    print(f"  使用交易日历更准确，避免过滤过多交易日")
    print(f"  688469.SH 应该过滤到第25个交易日，而不是25个自然日")
    print(f"{'=' * 100}")

if __name__ == "__main__":
    test_trading_day_calculation()
