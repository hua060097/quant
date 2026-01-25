#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接使用xtdata获取603363.SH的原始数据
"""

from xtquant import xtdata
import pandas as pd
import datetime

# 禁用xtdata的hello消息
try:
    xtdata.disable_hello_message()
except:
    pass

def get_stock_data_from_xtdata():
    """从xtdata直接获取股票数据"""

    stock_code = '603363.SH'
    start_date = '20241224'
    end_date = '20241226'

    # 需要获取的字段
    fields = ['open', 'close', 'high', 'low', 'volume', 'amount', 'time']

    print('=' * 100)
    print('直接从xtdata获取原始数据')
    print('=' * 100)
    print(f'股票代码: {stock_code}')
    print(f'日期范围: {start_date} ~ {end_date}')
    print(f'字段列表: {fields}')
    print('=' * 100)
    print()

    try:
        # 方法1: 使用get_market_data（前复权）
        print('[方法1] 使用get_market_data (dividend_type="front" 前复权):')
        print('-' * 100)

        data_front = xtdata.get_market_data(
            field_list=fields,
            stock_list=[stock_code],
            period='1d',
            start_time=start_date,
            end_time=end_date,
            dividend_type="front"
        )

        if data_front and stock_code in data_front:
            df_front = data_front[stock_code]
            df_front['date'] = pd.to_datetime(df_front['time']).dt.strftime('%Y-%m-%d')

            print('\n前复权数据:')
            display_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']
            print(df_front[display_cols].to_string(index=False))
        else:
            print('未获取到数据')

        # 方法2: 使用get_market_data（后复权）
        print('\n' + '=' * 100)
        print('[方法2] 使用get_market_data (dividend_type="back" 后复权):')
        print('-' * 100)

        data_back = xtdata.get_market_data(
            field_list=fields,
            stock_list=[stock_code],
            period='1d',
            start_time=start_date,
            end_time=end_date,
            dividend_type="back"
        )

        if data_back and stock_code in data_back:
            df_back = data_back[stock_code]
            df_back['date'] = pd.to_datetime(df_back['time']).dt.strftime('%Y-%m-%d')

            print('\n后复权数据:')
            print(df_back[display_cols].to_string(index=False))
        else:
            print('未获取到数据')

        # 方法3: 使用get_market_data（不复权）
        print('\n' + '=' * 100)
        print('[方法3] 使用get_market_data (dividend_type="none" 不复权):')
        print('-' * 100)

        data_none = xtdata.get_market_data(
            field_list=fields,
            stock_list=[stock_code],
            period='1d',
            start_time=start_date,
            end_time=end_date,
            dividend_type="none"
        )

        if data_none and stock_code in data_none:
            df_none = data_none[stock_code]
            df_none['date'] = pd.to_datetime(df_none['time']).dt.strftime('%Y-%m-%d')

            print('\n不复权数据:')
            print(df_none[display_cols].to_string(index=False))
        else:
            print('未获取到数据')

        # 对比分析
        print('\n' + '=' * 100)
        print('数据对比分析:')
        print('=' * 100)

        if all([data_front.get(stock_code) is not None,
                data_back.get(stock_code) is not None,
                data_none.get(stock_code) is not None]):

            df_front = data_front[stock_code]
            df_back = data_back[stock_code]
            df_none = data_none[stock_code]

            df_front['date'] = pd.to_datetime(df_front['time']).dt.strftime('%Y-%m-%d')
            df_back['date'] = pd.to_datetime(df_back['time']).dt.strftime('%Y-%m-%d')
            df_none['date'] = pd.to_datetime(df_none['time']).dt.strftime('%Y-%m-%d')

            for idx, row in df_front.iterrows():
                date = row['date']

                print(f'\n日期: {date}')
                print(f'  前复权: 开={row["open"]:.2f}, 收={row["close"]:.2f}, 高={row["high"]:.2f}, 低={row["low"]:.2f}')

                back_row = df_back[df_back['date'] == date].iloc[0] if len(df_back[df_back['date'] == date]) > 0 else None
                none_row = df_none[df_none['date'] == date].iloc[0] if len(df_none[df_none['date'] == date]) > 0 else None

                if back_row is not None:
                    print(f'  后复权: 开={back_row["open"]:.2f}, 收={back_row["close"]:.2f}, 高={back_row["high"]:.2f}, 低={back_row["low"]:.2f}')

                if none_row is not None:
                    print(f'  不复权: 开={none_row["open"]:.2f}, 收={none_row["close"]:.2f}, 高={none_row["high"]:.2f}, 低={none_row["low"]:.2f}')

                # 检查跳空
                if idx > 0:
                    prev_close = df_front.iloc[idx-1]['close']
                    curr_open = row['open']
                    gap_ratio = (curr_open - prev_close) / prev_close * 100

                    if abs(gap_ratio) > 50:
                        print(f'  [异常] 前复权跳空: {gap_ratio:+.2f}%')

                    if back_row is not None and idx > 0:
                        prev_close_back = df_back.iloc[idx-1]['close']
                        curr_open_back = back_row['open']
                        gap_ratio_back = (curr_open_back - prev_close_back) / prev_close_back * 100
                        print(f'  后复权跳空: {gap_ratio_back:+.2f}%')

                    if none_row is not None and idx > 0:
                        prev_close_none = df_none.iloc[idx-1]['close']
                        curr_open_none = none_row['open']
                        gap_ratio_none = (curr_open_none - prev_close_none) / prev_close_none * 100
                        print(f'  不复权跳空: {gap_ratio_none:+.2f}%')

        print('\n' + '=' * 100)
        print('数据获取完成')
        print('=' * 100)

    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    get_stock_data_from_xtdata()
