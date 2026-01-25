#!/usr/bin/env python3
"""
检查程序如何处理未上市股票的情况

模拟：
1. 股票在2025-12-20上市
2. 处理2025-12-01的特征
3. 查看程序如何处理2025-12-01到2025-12-19期间的数据
"""

import pandas as pd
import numpy as np
from pathlib import Path


def simulate_unlisted_stock():
    """模拟未上市股票的处理"""
    print("=" * 100)
    print("模拟场景：股票在2025-12-20上市，处理2025-12-01的特征")
    print("=" * 100)

    # 模拟数据：股票只有12-20之后的数据
    data = {
        'stock_code': ['300619.SZ'] * 10,
        'date': pd.to_datetime([
            '2025-12-20', '2025-12-21', '2025-12-22', '2025-12-23',
            '2025-12-24', '2025-12-25', '2025-12-26', '2025-12-29',
            '2025-12-30', '2025-12-31'
        ]),
        'open': [20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9],
        'close': [19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8],
        'high': [20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 21.0, 21.1],
        'low': [19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7],
        'volume': [1000000] * 10,
        'amount': [20000000] * 10,
        'pct_chg': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'total_mv': [100000000] * 10,
        'circ_mv': [50000000] * 10,
        'turnover_rate': [5.0] * 10,
    }

    df = pd.DataFrame(data)

    print("\n[模拟数据] 该股票只有2025-12-20开始的数据:")
    print(df[['date', 'open', 'close']].to_string(index=False))

    # 模拟程序的处理流程
    print("\n" + "=" * 100)
    print("程序处理流程模拟")
    print("=" * 100)

    # 步骤1: 按stock_code分组（实际程序中）
    print("\n步骤1: 按股票代码分组")
    stock_df = df  # 这里已经只有一只股票的数据
    print(f"  股票数据行数: {len(stock_df)}")

    # 步骤2: 计算技术指标
    print("\n步骤2: 计算技术指标")
    print("  当前实现：不计算任何指标，直接返回")

    # 步骤3: 计算未来收益
    print("\n步骤3: 计算未来收益")

    # 模拟calculate_future_returns的逻辑
    if len(stock_df) < 2:
        print(f"  ⚠️ 数据行数({len(stock_df)}) < 2，无法计算未来收益")
        print(f"  结果: future_return字段全部为NaN, is_can_buy=0")
    else:
        print(f"  ✓ 数据行数({len(stock_df)}) >= 2，可以计算未来收益")

        # 模拟计算future_return_1d
        open_prices = stock_df['close'].values
        close_prices = stock_df['close'].values

        t1_open = np.roll(open_prices, -1)
        t1_close = np.roll(close_prices, -1)

        future_returns_1d = np.full(len(stock_df), np.nan)
        valid_mask = (~np.isnan(t1_open)) & (t1_open > 0) & (~np.isnan(t1_close)) & (t1_close > 0)
        future_returns_1d[valid_mask] = (t1_close[valid_mask] - t1_open[valid_mask]) / t1_open[valid_mask]
        future_returns_1d[-1] = np.nan  # 最后一行设为NaN

        print(f"\n  计算结果示例（前5行）:")
        for i in range(min(5, len(stock_df))):
            date_str = stock_df.iloc[i]['date'].strftime('%Y-%m-%d')
            ret_val = future_returns_1d[i] if not np.isnan(future_returns_1d[i]) else "NaN"
            print(f"    {date_str}: future_return_1d = {ret_val}")

    # 步骤4: 只保留目标日期的数据
    print("\n步骤4: 只保留目标日期的数据")
    target_date = pd.to_datetime('2025-12-01')
    result = stock_df[stock_df['date'] == target_date]

    if len(result) == 0:
        print(f"  ⚠️ 目标日期(2025-12-01)在数据中不存在")
        print(f"  结果: 该股票不会被保存到特征文件中")
    else:
        print(f"  ✓ 目标日期存在，保存该记录")

    # 结论
    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)

    print("""
情况1：如果storage.get_daily_data只返回有数据的日期
  - 结果：该股票不会出现在2025-12-01的特征文件中 ✓

情况2：如果storage.get_daily_data返回了2025-12-01的记录（但价格字段为NaN）
  - 结果：该股票会出现在特征文件中，但：
    - open, close等字段为NaN
    - future_return字段为NaN
    - is_can_buy = 0

建议：检查storage.get_daily_data的实际行为
    """)


def check_actual_storage_behavior():
    """检查storage的实际行为"""
    print("\n" + "=" * 100)
    print("检查storage.get_daily_data的实际行为")
    print("=" * 100)

    from advanced_quant_data_fixed import LocalDataStorage

    storage = LocalDataStorage()

    # 测试：加载一只新上市股票的数据
    test_stock = '301678.SZ'  # 假设这是一只较新的股票
    test_date = '20251201'  # 测试一个较早的日期

    print(f"\n测试股票: {test_stock}")
    print(f"测试日期: {test_date}")
    print(f"数据范围: {test_date} 前60天到后10天")

    try:
        data = storage.get_daily_data(
            stock_list=[test_stock],
            start_date='20251001',
            end_date='20251210'
        )

        if data is not None and not data.empty:
            stock_data = data[data['stock_code'] == test_stock]

            print(f"\n实际返回数据行数: {len(stock_data)}")
            print(f"数据日期范围:")

            if len(stock_data) > 0:
                stock_data = stock_data.sort_values('date')
                print(f"  最早日期: {stock_data.iloc[0]['date']}")
                print(f"  最晚日期: {stock_data.iloc[-1]['date']}")

                # 检查2025-12-01是否存在
                target_date = pd.to_datetime('2025-12-01')
                has_target_date = target_date in stock_data['date'].values

                print(f"\n目标日期(2025-12-01)是否存在: {has_target_date}")

                if has_target_date:
                    target_record = stock_data[stock_data['date'] == target_date].iloc[0]
                    print(f"  目标日期的记录:")
                    print(f"    open: {target_record['open']}")
                    print(f"    close: {target_record['close']}")
                    print(f"    是否为NaN: {pd.isna(target_record['open'])}")
                else:
                    print(f"  目标日期不存在，该股票不会保存到特征文件中")

            else:
                print("  返回的数据为空")
        else:
            print("  返回的数据为None")

    except Exception as e:
        print(f"  [ERROR] 查询失败: {e}")


if __name__ == "__main__":
    simulate_unlisted_stock()
    # check_actual_storage_behavior()  # 这个可能需要很长时间，暂时注释
