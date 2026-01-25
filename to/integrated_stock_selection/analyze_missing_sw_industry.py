#!/usr/bin/env python3
"""
分析没有SW行业分类的股票

检查factors.parquet中哪些股票缺少申万行业分类
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys

def analyze_missing_sw_industry():
    """分析没有SW行业的股票"""
    print("=== 分析没有SW行业分类的股票 ===")

    # 1. 读取因子数据
    print("\n1. 读取因子数据...")
    try:
        df = pd.read_parquet('factor_lab_data/factors.parquet')
        df['date'] = pd.to_datetime(df['date'])

        print(f"   因子数据基本信息:")
        print(f"   - 总记录数: {len(df):,}")
        print(f"   - 股票数量: {df['stock_code'].nunique()}")
        print(f"   - 日期范围: {df['date'].min()} ~ {df['date'].max()}")

        # 获取所有股票
        all_stocks = sorted(df['stock_code'].unique())
        print(f"   - 所有股票数: {len(all_stocks):,}")

    except Exception as e:
        print(f"   读取因子数据失败: {e}")
        return

    # 2. 检查因子数据中的行业相关列
    print("\n2. 检查因子数据中的行业相关列...")
    industry_cols = [col for col in df.columns if 'industry' in col.lower() or 'sector' in col.lower() or 'sw' in col.lower()]
    print(f"   行业相关列: {industry_cols}")

    missing_industry_stocks = set()

    if industry_cols:
        for col in industry_cols:
            null_count = df[col].isna().sum()
            if null_count > 0:
                col_missing_stocks = df[df[col].isna()]['stock_code'].unique()
                missing_industry_stocks.update(col_missing_stocks)
                print(f"   {col}: {len(col_missing_stocks):,}只股票缺失")
    else:
        print("   未发现行业相关列")
        missing_industry_stocks = set(all_stocks)

    # 3. 检查外部行业映射文件
    print("\n3. 检查外部行业映射文件...")

    # 检查stock_sectors_mapping.py
    mapped_stocks = set()

    try:
        # 尝试导入模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("stock_sectors_mapping", "stock_sectors_mapping.py")
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 查找可能的函数
            for attr in dir(module):
                if not attr.startswith('_'):
                    try:
                        func = getattr(module, attr)
                        if callable(func):
                            result = func()
                            if isinstance(result, dict):
                                # 提取股票代码
                                for key in result.keys():
                                    if isinstance(key, str):
                                        # 标准化股票代码
                                        stock_code = key
                                        if '.' not in stock_code and len(stock_code) == 6:
                                            # 补全交易所后缀
                                            if stock_code.startswith('6'):
                                                stock_code = f'{stock_code}.SH'
                                            else:
                                                stock_code = f'{stock_code}.SZ'
                                        mapped_stocks.add(stock_code)
                                        else:
                                            mapped_stocks.add(key)
                                break
                    except:
                        continue

            print(f"   从stock_sectors_mapping获取到 {len(mapped_stocks)} 只股票的行业映射")

    except Exception as e:
        print(f"   无法加载stock_sectors_mapping.py: {e}")

    # 4. 检查SW行业编码文件
    try:
        # 检查pkl文件
        if os.path.exists('config/sw_industry_encoding.pkl'):
            with open('config/sw_industry_encoding.pkl', 'rb') as f:
                sw_encoding = pickle.load(f)

            print(f"   SW编码文件结构: {type(sw_encoding)}")
            if isinstance(sw_encoding, dict):
                # 尝试提取股票映射
                for key, value in sw_encoding.items():
                    if key == 'mappings' and isinstance(value, dict):
                        for stock, industry in value.items():
                            if isinstance(stock, str):
                                mapped_stocks.add(stock)
                        break
                print(f"   从SW编码文件解析到 {len(mapped_stocks)} 只股票")

    except Exception as e:
        print(f"   读取SW编码文件失败: {e}")

    # 5. 汇总分析结果
    print(f"\n=== 汇总分析结果 ===")

    factor_stocks = set(all_stocks)
    total_stocks = len(factor_stocks)

    if mapped_stocks:
        missing_stocks = factor_stocks - mapped_stocks
        print(f"因子数据中的股票总数: {total_stocks:,}")
        print(f"有行业映射的股票数: {len(mapped_stocks):,}")
        print(f"没有SW行业的股票数: {len(missing_stocks):,}")
        print(f"缺失比例: {len(missing_stocks)/total_stocks*100:.2f}%")

        # 6. 输出没有SW行业的股票示例
        if missing_stocks:
            print(f"\n=== 没有SW行业的股票示例（前10个） ===")
            missing_list = sorted(list(missing_stocks))[:10]
            for i, stock in enumerate(missing_list, 1):
                print(f"{i:2d}. {stock}")
    else:
        print("没有找到任何行业映射数据")
        print(f"所有 {total_stocks:,} 只股票都可能缺少SW行业分类")

    # 7. 按交易所分类统计
    print(f"\n=== 按交易所分类统计 ===")

    sh_stocks = [stock for stock in all_stocks if stock.endswith('.SH')]
    sz_stocks = [stock for stock in all_stocks if stock.endswith('.SZ')]

    print(f"上海证券交易所: {len(sh_stocks):,}只股票")
    print(f"深圳证券交易所: {len(sz_stocks):,}只股票")

    if mapped_stocks:
        sh_mapped = [stock for stock in mapped_stocks if stock.endswith('.SH')]
        sz_mapped = [stock for stock in mapped_stocks if stock.endswith('.SZ')]

        print(f"\n有SW分类统计:")
        print(f"上海证券交易所: {len(sh_mapped):,}/{len(sh_stocks)} ({len(sh_mapped)/len(sh_stocks)*100:.1f}%)")
        print(f"深圳证券交易所: {len(sz_mapped):,}/{len(sz_stocks)} ({len(sz_mapped)/len(sz_stocks)*100:.1f}%)")

    # 8. 股票代码格式分析
    print(f"\n=== 股票代码格式分析 ===")

    # 检查不同格式的股票代码
    format_6digit = [stock for stock in all_stocks if len(stock) == 6 and not '.' in stock]
    format_with_suffix = [stock for stock in all_stocks if '.' in stock]

    print(f"6位数字格式: {len(format_6digit):,}只股票")
    print(f"带交易所后缀: {len(format_with_suffix):,}只股票")

    if format_6digit:
        print(f"6位格式示例: {format_6digit[:5]}")
    if format_with_suffix:
        print(f"带后缀示例: {format_with_suffix[:5]}")

    print(f"\n=== 建议解决方案 ===")
    print("1. 检查股票代码格式是否统一（建议使用带交易所后缀的格式）")
    print("2. 从外部数据源获取申万行业分类数据")
    print("3. 创建或更新股票代码到申万行业的映射文件")
    print("4. 考虑使用股票代码标准化函数统一格式")

    # 9. 创建输出文件
    print(f"\n=== 创建缺失股票列表 ===")

    if mapped_stocks:
        final_missing_stocks = factor_stocks - mapped_stocks
    else:
        final_missing_stocks = factor_stocks

    # 保存缺失股票列表
    missing_list = sorted(list(final_missing_stocks))
    output_file = "missing_sw_industry_stocks.txt"

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"没有SW行业分类的股票列表\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n")
            f.write(f"总缺失股票数: {len(final_missing_stocks):,}\n")
            f.write(f"缺失比例: {len(final_missing_stocks)/total_stocks*100:.2f}%\n")
            f.write(f"\n股票列表:\n")
            for stock in missing_list:
                f.write(f"{stock}\n")

        print(f"   已保存到文件: {output_file}")
        print(f"   包含 {len(missing_list):,} 只股票")

    except Exception as e:
        print(f"   保存文件失败: {e}")

    return len(final_missing_stocks)

if __name__ == "__main__":
    try:
        missing_count = analyze_missing_sw_industry()
        print(f"\n分析完成！")
        print(f"总计 {missing_count:,} 只股票没有SW行业分类")

    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()