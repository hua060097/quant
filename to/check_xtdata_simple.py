#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接使用xtdata获取603363.SH的原始数据
"""

try:
    from xtquant import xtdata
    print("成功导入xtdata")
except ImportError as e:
    print(f"无法导入xtdata: {e}")
    print("\n请确认：")
    print("1. 是否安装了xtquant库")
    print("2. 是否在正确的Python环境中")
    print("3. 尝试安装: pip install xtquant")
    exit(1)

import pandas as pd

# 禁用xtdata的hello消息
try:
    xtdata.disable_hello_message()
except:
    pass

stock_code = '603363.SH'
start_date = '20241224'
end_date = '20241226'
fields = ['open', 'close', 'high', 'low', 'volume', 'amount', 'time']

print('=' * 100)
print('直接从xtdata获取原始数据')
print('=' * 100)
print(f'股票代码: {stock_code}')
print(f'日期范围: {start_date} ~ {end_date}')
print('=' * 100)
print()

# 获取前复权数据
print('[前复权数据]')
data_front = xtdata.get_market_data(
    field_list=fields,
    stock_list=[stock_code],
    period='1d',
    start_time=start_date,
    end_time=end_date,
    dividend_type="front"
)

if data_front and stock_code in data_front:
    df = data_front[stock_code]
    df['date'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')

    print('\n数据:')
    print(df[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']].to_string(index=False))

    print('\n跳空分析:')
    for i in range(1, len(df)):
        prev_close = df.iloc[i-1]['close']
        curr_open = df.iloc[i]['open']
        gap_ratio = (curr_open - prev_close) / prev_close * 100

        date = df.iloc[i]['date']
        print(f'  {date}: 开盘({curr_open:.2f}) - 前一日收盘({prev_close:.2f}) = {gap_ratio:+.2f}%')

        if abs(gap_ratio) > 50:
            print(f'    [异常] 跳空幅度超过50%!')
else:
    print('未获取到数据')

print('\n' + '=' * 100)

# 获取不复权数据
print('[不复权数据]')
data_none = xtdata.get_market_data(
    field_list=fields,
    stock_list=[stock_code],
    period='1d',
    start_time=start_date,
    end_time=end_date,
    dividend_type="none"
)

if data_none and stock_code in data_none:
    df = data_none[stock_code]
    df['date'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')

    print('\n数据:')
    print(df[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']].to_string(index=False))

    print('\n跳空分析:')
    for i in range(1, len(df)):
        prev_close = df.iloc[i-1]['close']
        curr_open = df.iloc[i]['open']
        gap_ratio = (curr_open - prev_close) / prev_close * 100

        date = df.iloc[i]['date']
        print(f'  {date}: 开盘({curr_open:.2f}) - 前一日收盘({prev_close:.2f}) = {gap_ratio:+.2f}%')

        if abs(gap_ratio) > 50:
            print(f'    [注意] 跳空幅度超过50% (可能是除权除息)')
else:
    print('未获取到数据')

print('\n' + '=' * 100)
print('检查完成')
print('=' * 100)
