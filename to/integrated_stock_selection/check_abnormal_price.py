#!/usr/bin/env python3
"""
检查异常价格的来源

检查000002.SZ在2025-12-26前后的价格数据
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def check_abnormal_price():
    """检查异常价格来源"""
    stock_code = '000002.SZ'
    target_date = '2025-12-26'

    print("=" * 100)
    print(f"检查股票: {stock_code}")
    print(f"目标日期: {target_date} (T日)")
    print(f"隔夜异常: T收盘=4.77, T+1开盘=17.12, 隔夜收益=258.91%")
    print("=" * 100)

    # 构建文件路径
    # T日是2025-12-26，T+1日是2025-12-29（周末跳过）
    # T日数据在202512月度文件中，T+1日数据也应该在202512月度文件中
    base_dir = Path(__file__).parent.parent / "local_data" / "daily"
    december_file = base_dir / "202512" / "daily_data.parquet"

    print(f"\n[*] 读取数据文件: {december_file}")

    if not december_file.exists():
        print(f"[ERROR] 文件不存在: {december_file}")
        return

    try:
        # 读取数据
        df = pd.read_parquet(december_file)
        print(f"[OK] 成功读取数据: {len(df):,} 行")

        # 筛选目标股票
        stock_df = df[df['stock_code'] == stock_code].copy()
        print(f"[OK] 找到 {stock_code} 数据: {len(stock_df)} 条记录")

        # 转换日期格式
        stock_df['date'] = pd.to_datetime(stock_df['date'])

        # 筛选2025年12月的数据
        december_data = stock_df[
            (stock_df['date'] >= '2025-12-01') &
            (stock_df['date'] <= '2025-12-31')
        ].sort_values('date')

        print(f"\n{'=' * 100}")
        print(f"2025年12月完整数据:")
        print(f"{'=' * 100}")

        # 显示所有相关字段
        display_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'pct_chg']
        available_cols = [col for col in display_cols if col in december_data.columns]

        if available_cols:
            print(december_data[available_cols].to_string(index=False))
        else:
            print("[WARNING] 缺少必要的列")
            print(f"可用列: {list(december_data.columns)}")

        # 重点检查T日和T+1日
        print(f"\n{'=' * 100}")
        print(f"重点检查 T日({target_date})和T+1日:")
        print(f"{'=' * 100}")

        target_date_dt = pd.to_datetime(target_date)

        # 找T日（2025-12-26，星期五）
        t_day_data = december_data[december_data['date'] == target_date_dt]
        if len(t_day_data) > 0:
            print(f"\n[T日] {target_date} ({'星期五' if t_day_data.iloc[0]['date'].dayofweek == 4 else '未知'}):")
            for col in available_cols:
                if col in t_day_data.columns:
                    print(f"  {col}: {t_day_data.iloc[0][col]}")
        else:
            print(f"\n[WARNING] 未找到T日({target_date})的数据")

        # 找T+1日（应该是2025-12-29，星期一）
        t1_date_dt = target_date_dt + timedelta(days=3)  # 跳过周末
        t1_day_data = december_data[december_data['date'] == t1_date_dt]
        if len(t1_day_data) > 0:
            print(f"\n[T+1日] {t1_date_dt.strftime('%Y-%m-%d')} ({'星期一' if t1_day_data.iloc[0]['date'].dayofweek == 0 else '未知'}):")
            for col in available_cols:
                if col in t1_day_data.columns:
                    print(f"  {col}: {t1_day_data.iloc[0][col]}")
        else:
            print(f"\n[WARNING] 未找到T+1日({t1_date_dt.strftime('%Y-%m-%d')})的数据")

        # 检查是否有其他交易日
        print(f"\n{'=' * 100}")
        print(f"12月所有交易日:")
        print(f"{'=' * 100}")
        for idx, row in december_data.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            weekday = ['一', '二', '三', '四', '五', '六', '日'][row['date'].dayofweek]
            print(f"  {date_str} (星期{weekday}): open={row['open']:.2f}, close={row['close']:.2f}")

        # 检查是否有前复权数据
        if 'open_front' in december_data.columns or 'close_front' in december_data.columns:
            print(f"\n{'=' * 100}")
            print(f"前复权数据对比:")
            print(f"{'=' * 100}")

            front_cols = ['date', 'open_front', 'close_front', 'open', 'close']
            available_front_cols = [col for col in front_cols if col in december_data.columns]

            if available_front_cols:
                print(december_data[available_front_cols].to_string(index=False))

        # 分析异常原因
        print(f"\n{'=' * 100}")
        print(f"异常原因分析:")
        print(f"{'=' * 100}")

        if len(t_day_data) > 0 and len(t1_day_data) > 0:
            t_close = t_day_data.iloc[0]['close']
            t1_open = t1_day_data.iloc[0]['open']
            actual_overnight_return = (t1_open - t_close) / t_close

            print(f"  T日收盘: {t_close:.2f}")
            print(f"  T+1日开盘: {t1_open:.2f}")
            print(f"  实际隔夜收益: {actual_overnight_return:.2%}")
            print(f"  报告的隔夜收益: 258.91% (T收盘=4.77, T+1开盘=17.12)")

            if abs(t_close - 4.77) < 0.01:
                print(f"\n  ✓ T日收盘价匹配 (4.77)")
            else:
                print(f"\n  ✗ T日收盘价不匹配! 期望4.77, 实际{t_close:.2f}")

            if abs(t1_open - 17.12) < 0.01:
                print(f"  ✓ T+1日开盘价匹配 (17.12)")
            else:
                print(f"  ✗ T+1日开盘价不匹配! 期望17.12, 实际{t1_open:.2f}")

            # 检查是否有17.12这个价格出现在任何日期
            print(f"\n{'=' * 100}")
            print(f"查找17.12这个价格:")
            print(f"{'=' * 100}")

            price_17_12_found = False
            for idx, row in december_data.iterrows():
                if abs(row['open'] - 17.12) < 0.01:
                    print(f"  找到! {row['date'].strftime('%Y-%m-%d')} 开盘价: {row['open']:.2f}")
                    price_17_12_found = True
                if abs(row['close'] - 17.12) < 0.01:
                    print(f"  找到! {row['date'].strftime('%Y-%m-%d')} 收盘价: {row['close']:.2f}")
                    price_17_12_found = True

            if not price_17_12_found:
                print(f"  未在12月数据中找到17.12这个价格")
                print(f"  可能的来源:")
                print(f"    1. 数据来自其他月份（需要检查1月数据）")
                print(f"    2. 数据来自特征文件而非源数据")
                print(f"    3. 数据在某次处理中被修改")

    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_abnormal_price()
