"""
查找哪个交易日期向前扩展180个交易日会正好到2025-04-24
"""
import pandas as pd
from pathlib import Path

def find_start_date(target_date_str, lookback_days=180):
    """
    查找哪个交易日期向前扩展N个交易日会正好到目标日期

    参数:
        target_date_str: 目标日期字符串 (YYYYMMDD 或 YYYY-MM-DD)
        lookback_days: 向前扩展的交易日数量

    返回:
        找到的起始日期
    """
    # 读取交易日历
    calendar_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\to\local_data\calendar\trading_calendar.parquet")

    print("=" * 80)
    print(f"查找从哪个日期向前扩展{lookback_days}个交易日会到 {target_date_str}")
    print("=" * 80)

    # 读取交易日历
    try:
        calendar_df = pd.read_parquet(calendar_path)
        print(f"\n[交易日历信息]")
        print(f"  文件路径: {calendar_path}")
        print(f"  交易日数量: {len(calendar_df):,}")

        # 获取交易日期列（假设列名为 trade_date 或 date）
        date_col = None
        for col in calendar_df.columns:
            if 'date' in col.lower() or 'trade' in col.lower():
                date_col = col
                break

        if date_col is None:
            # 如果找不到，尝试使用第一列
            date_col = calendar_df.columns[0]

        print(f"  日期列名: {date_col}")
        print(f"  日期范围: {calendar_df[date_col].min()} 至 {calendar_df[date_col].max()}")

        # 转换目标日期
        if '-' in target_date_str:
            target_date = pd.to_datetime(target_date_str)
        else:
            target_date = pd.to_datetime(target_date_str, format='%Y%m%d')

        print(f"\n[目标日期]")
        print(f"  目标日期: {target_date.strftime('%Y-%m-%d')}")

        # 将交易日历转换为datetime类型
        calendar_df['_date'] = pd.to_datetime(calendar_df[date_col])

        # 查找目标日期在交易日历中的索引
        target_idx = calendar_df[calendar_df['_date'] == target_date].index

        if len(target_idx) == 0:
            print(f"\n[ERROR] 目标日期 {target_date.strftime('%Y-%m-%d')} 不在交易日历中")
            print(f"  可能原因: 该日期不是交易日")

            # 找到最接近的交易日
            calendar_df['diff'] = (calendar_df['_date'] - target_date).abs()
            closest = calendar_df.loc[calendar_df['diff'].idxmin()]

            print(f"\n  最接近的交易日:")
            print(f"    日期: {closest['_date'].strftime('%Y-%m-%d')}")
            print(f"    相差: {closest['diff'].days} 天")

            return None

        target_idx = target_idx[0]

        print(f"  在交易日历中的索引位置: {target_idx:,}")

        # 从未来的某一天向前扩展lookback_days个交易日到目标日期
        # 所以起始索引 = 目标索引 + lookback_days
        start_idx = target_idx + lookback_days

        if start_idx >= len(calendar_df):
            print(f"\n[WARNING] 交易日历数据不足")
            print(f"  需要的索引: {start_idx}")
            print(f"  交易日历最大索引: {len(calendar_df) - 1}")
            print(f"  缺少: {start_idx - len(calendar_df) + 1} 个交易日")

            # 使用交易日历的最后一个日期
            start_idx = len(calendar_df) - 1
            start_date = calendar_df.loc[start_idx, '_date']
            actual_trading_days = start_idx - target_idx

            print(f"\n[使用最远可用日期]")
            print(f"  未来日期索引: {start_idx:,}")
            print(f"  未来日期: {start_date.strftime('%Y-%m-%d')}")
            print(f"  实际向前交易日数: {actual_trading_days}")

            # 验证：列出从最远日期向前推到目标日期的所有交易日
            trading_dates = calendar_df.loc[target_idx:start_idx-1, '_date'].tolist()

            print(f"\n[验证]")
            print(f"  从未来日期向前到目标日期的交易日数量: {len(trading_dates)}")
            print(f"  最接近未来日期的前5个交易日:")
            for i, date in enumerate(trading_dates[-5:][::-1], 1):
                print(f"    {i}. {date.strftime('%Y-%m-%d')}")
            if len(trading_dates) > 10:
                print(f"    ...")
                print(f"  最接近目标日期的后5个交易日:")
                for i, date in enumerate(trading_dates[:5], 1):
                    print(f"    {i}. {date.strftime('%Y-%m-%d')}")

            # 计算自然日天数
            natural_days = (start_date - target_date).days
            print(f"\n[额外信息]")
            print(f"  自然日天数: {natural_days}")
            print(f"  平均每周交易日: {len(trading_dates) / (natural_days / 7):.2f}")

            return start_date

        # 正常情况：数据充足
        # 获取起始日期
        start_date = calendar_df.loc[start_idx, '_date']

        print(f"\n[计算结果]")
        print(f"  未来日期索引: {start_idx:,}")
        print(f"  未来日期: {start_date.strftime('%Y-%m-%d')}")
        print(f"  目标索引: {target_idx:,}")
        print(f"  目标日期: {target_date.strftime('%Y-%m-%d')}")
        print(f"  向前交易日数: {lookback_days}")

        # 验证：列出从起始日期向前推到目标日期的所有交易日（不包括起始日期，包括目标日期）
        trading_dates = calendar_df.loc[target_idx:start_idx-1, '_date'].tolist()

        print(f"\n[验证]")
        print(f"  从起始日期向前到目标日期的交易日数量: {len(trading_dates)}")
        print(f"  最接近起始日期的前5个交易日:")
        for i, date in enumerate(trading_dates[-5:][::-1], 1):
            print(f"    {i}. {date.strftime('%Y-%m-%d')}")
        if len(trading_dates) > 10:
            print(f"    ...")
            print(f"  最接近目标日期的后5个交易日:")
            for i, date in enumerate(trading_dates[:5], 1):
                print(f"    {i}. {date.strftime('%Y-%m-%d')}")

        # 计算自然日天数（未来日期 - 过去日期 = 正数）
        natural_days = (start_date - target_date).days
        print(f"\n[额外信息]")
        print(f"  自然日天数: {natural_days}")
        print(f"  平均每周交易日: {len(trading_dates) / (natural_days / 7):.2f}")

        return start_date

    except FileNotFoundError:
        print(f"\n[ERROR] 找不到交易日历文件: {calendar_path}")
        return None
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 查找从哪个日期向前扩展180个交易日会到2025-04-24
    result = find_start_date('20250424', lookback_days=180)

    if result is not None:
        print(f"\n[最终答案]")
        print(f"  从 {result.strftime('%Y-%m-%d')} 开始，")
        print(f"  向前扩展180个交易日，")
        print(f"  会正好到 2025-04-24")
