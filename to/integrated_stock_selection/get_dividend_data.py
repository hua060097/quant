#!/usr/bin/env python3
"""
获取指定股票在不同复权类型下的开收盘价数据

复权类型：
- "none": 不复权
- "front": 前复权
- "back": 后复权
"""

import pandas as pd
from datetime import datetime
from xtquant import xtdata

# 禁用xtdata的hello消息
try:
    xtdata.enable_hello = False
except:
    pass

def get_dividend_comparison_data(
    stock_list=None,
    start_date='20250605',
    end_date='20250606'
):
    """
    获取股票在不同复权类型下的数据

    Args:
        stock_list: 股票代码列表
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
    """
    if stock_list is None:
        stock_list = ['300619.SZ', '301150.SZ']

    # 需要获取的字段
    fields = ['open', 'close']

    # 复权类型列表
    dividend_types = {
        'none': '不复权',
        'front': '前复权',
        'back': '后复权'
    }

    print("=" * 100)
    print(f"获取股票数据对比：{stock_list}")
    print(f"日期范围：{start_date} - {end_date}")
    print("=" * 100)

    results = {}

    # 遍历三种复权类型
    for dividend_type, type_name in dividend_types.items():
        print(f"\n{'=' * 100}")
        print(f"复权类型：{type_name} (dividend_type='{dividend_type}')")
        print(f"{'=' * 100}")

        try:
            # 获取数据
            data_back = xtdata.get_market_data(
                field_list=fields,
                stock_list=stock_list,
                period='1d',
                start_time=start_date,
                end_time=end_date,
                dividend_type=dividend_type
            )

            if data_back is None or len(data_back) == 0:
                print(f"[WARNING] 未获取到数据")
                continue

            # 处理数据
            for field in fields:
                if field in data_back:
                    field_data = data_back[field]

                    print(f"\n{field.upper()} 数据：")
                    print("-" * 100)

                    # 转换为DataFrame便于查看
                    df_list = []
                    for stock_code in stock_list:
                        if stock_code in field_data.columns:
                            stock_series = field_data[stock_code].dropna()
                            for date, value in stock_series.items():
                                df_list.append({
                                    'stock_code': stock_code,
                                    'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                                    field: value
                                })

                    if df_list:
                        df = pd.DataFrame(df_list)
                        df_pivot = df.pivot(index='date', columns='stock_code', values=field)
                        print(df_pivot.to_string())

                    # 保存结果
                    results[f"{dividend_type}_{field}"] = field_data

        except Exception as e:
            print(f"[ERROR] 获取{type_name}数据失败: {e}")
            import traceback
            traceback.print_exc()

    # 生成对比表
    print(f"\n{'=' * 100}")
    print("数据对比汇总")
    print(f"{'=' * 100}")

    generate_comparison_table(stock_list, start_date, end_date, results, dividend_types)

    return results


def generate_comparison_table(stock_list, start_date, end_date, results, dividend_types):
    """生成对比表格"""
    comparison_data = []

    for stock_code in stock_list:
        for date_str in [start_date, end_date]:
            date = pd.to_datetime(date_str)

            row = {
                'stock_code': stock_code,
                'date': date.strftime('%Y-%m-%d')
            }

            # 获取三种复权类型的数据
            for dividend_type, type_name in dividend_types.items():
                for field in ['open', 'close']:
                    key = f"{dividend_type}_{field}"
                    if key in results:
                        field_data = results[key]
                        if stock_code in field_data.columns:
                            value = field_data[stock_code].get(date)
                            row[f"{field}_{type_name}"] = f"{value:.2f}" if pd.notna(value) else "N/A"
                        else:
                            row[f"{field}_{type_name}"] = "N/A"
                    else:
                        row[f"{field}_{type_name}"] = "N/A"

            comparison_data.append(row)

    # 创建DataFrame并显示
    df_comparison = pd.DataFrame(comparison_data)

    # 重新排列列顺序
    column_order = ['stock_code', 'date']
    for field in ['open', 'close']:
        for type_name in ['不复权', '前复权', '后复权']:
            column_order.append(f"{field}_{type_name}")

    df_comparison = df_comparison[column_order]

    print("\n详细对比表：")
    print(df_comparison.to_string(index=False))

    # 保存为CSV
    output_file = f"dividend_comparison_{start_date}_{end_date}.csv"
    df_comparison.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 对比结果已保存到: {output_file}")


def main():
    """主函数"""
    # 配置参数
    stock_list = ['300619.SZ', '301150.SZ']
    start_date = '20250605'
    end_date = '20250606'

    # 获取数据
    results = get_dividend_comparison_data(
        stock_list=stock_list,
        start_date=start_date,
        end_date=end_date
    )

    print(f"\n{'=' * 100}")
    print("[OK] 数据获取完成")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
