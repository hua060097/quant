#!/usr/bin/env python3
"""
分析没有SW行业分类的股票

从daily特征数据中读取股票，然后与sw_sector.json对比，找出缺失SW行业的股票
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def analyze_missing_sw_industry():
    """分析没有SW行业的股票"""
    print("=== 分析没有SW行业分类的股票 ===")

    # 1. 读取2025年某一天的daily特征数据
    print("\n1. 读取daily特征数据...")

    # 寻找2025年的数据文件
    daily_path = Path("local_data/features/daily")
    if not daily_path.exists():
        print(f"   错误: 目录不存在 {daily_path}")
        return

    # 获取2025年的文件列表
    year_2025_files = list(daily_path.glob("2025*.parquet"))
    if not year_2025_files:
        print("   错误: 没有找到2025年的数据文件")
        return

    # 选择第一个2025年文件
    sample_file = sorted(year_2025_files)[0]
    print(f"   使用文件: {sample_file}")

    try:
        df_daily = pd.read_parquet(sample_file)
        print(f"   成功读取数据: {len(df_daily):,}条记录")
        print(f"   股票数量: {df_daily['stock_code'].nunique():,}")
        print(f"   日期: {df_daily['date'].unique()}")

        # 获取所有股票代码
        daily_stocks = set(df_daily['stock_code'].unique())
        print(f"   从daily数据中获取到 {len(daily_stocks):,}只股票")

    except Exception as e:
        print(f"   读取daily数据失败: {e}")
        return

    # 2. 加载SW行业映射文件
    print("\n2. 加载SW行业映射文件...")

    sw_mapping_path = Path("local_data/sw_sector/sw_sector.json")
    if not sw_mapping_path.exists():
        print(f"   错误: SW映射文件不存在 {sw_mapping_path}")
        return

    try:
        with open(sw_mapping_path, 'r', encoding='utf-8') as f:
            sw_mapping = json.load(f)

        print(f"   成功加载SW映射文件")
        print(f"   映射条目数: {len(sw_mapping):,}")

        # 获取有SW行业的股票
        sw_stocks = set(sw_mapping.keys())
        print(f"   有SW分类的股票: {len(sw_stocks):,}只")

        # 显示一些SW分类示例
        sw_examples = list(sw_mapping.items())[:5]
        print(f"   SW分类示例:")
        for stock, sector in sw_examples:
            print(f"     {stock}: {sector}")

    except Exception as e:
        print(f"   读取SW映射文件失败: {e}")
        return

    # 3. 对比分析
    print(f"\n=== 对比分析结果 ===")

    # 找到没有SW行业的股票
    missing_sw_stocks = daily_stocks - sw_stocks
    has_sw_stocks = daily_stocks.intersection(sw_stocks)

    total_stocks = len(daily_stocks)

    print(f"daily数据中股票总数: {total_stocks:,}")
    print(f"有SW行业分类的股票: {len(has_sw_stocks):,} ({len(has_sw_stocks)/total_stocks*100:.1f}%)")
    print(f"没有SW行业分类的股票: {len(missing_sw_stocks):,} ({len(missing_sw_stocks)/total_stocks*100:.1f}%)")

    # 4. 显示没有SW行业的股票示例
    if missing_sw_stocks:
        print(f"\n=== 没有SW行业分类的股票示例（前10个） ===")
        missing_list = sorted(list(missing_sw_stocks))[:10]
        for i, stock in enumerate(missing_list, 1):
            print(f"{i:2d}. {stock}")

    # 5. 按交易所分类统计
    print(f"\n=== 按交易所分类统计 ===")

    def classify_exchange(stock_list):
        sh_count = sum(1 for stock in stock_list if stock.endswith('.SH'))
        sz_count = sum(1 for stock in stock_list if stock.endswith('.SZ'))
        bj_count = sum(1 for stock in stock_list if stock.endswith('.BJ'))
        return {'SH': sh_count, 'SZ': sz_count, 'BJ': bj_count}

    # 所有股票分布
    all_exchange = classify_exchange(daily_stocks)
    print(f"所有股票分布:")
    print(f"  上海证券交易所(SH): {all_exchange['SH']:,}只")
    print(f"  深圳证券交易所(SZ): {all_exchange['SZ']:,}只")
    print(f"  北京证券交易所(BJ): {all_exchange['BJ']:,}只")

    # 有SW分类的股票分布
    has_sw_exchange = classify_exchange(has_sw_stocks)
    print(f"\n有SW分类的股票分布:")
    print(f"  上海证券交易所(SH): {has_sw_exchange['SH']:,}只 ({has_sw_exchange['SH']/all_exchange['SH']*100:.1f}%)")
    print(f"  深圳证券交易所(SZ): {has_sw_exchange['SZ']:,}只 ({has_sw_exchange['SZ']/all_exchange['SZ']*100:.1f}%)")
    print(f"  北京证券交易所(BJ): {has_sw_exchange['BJ']:,}只 ({has_sw_exchange['BJ']/all_exchange['BJ']*100:.1f}%)")

    # 缺失SW分类的股票分布
    missing_sw_exchange = classify_exchange(missing_sw_stocks)
    print(f"\n缺失SW分类的股票分布:")
    print(f"  上海证券交易所(SH): {missing_sw_exchange['SH']:,}只")
    print(f"  深圳证券交易所(SZ): {missing_sw_exchange['SZ']:,}只")
    print(f"  北京证券交易所(BJ): {missing_sw_exchange['BJ']:,}只")

    # 6. 分析SW行业分布
    print(f"\n=== SW行业分布统计 ===")

    sector_count = {}
    for stock, sector in sw_mapping.items():
        if stock in daily_stocks:  # 只统计当前股票池中的
            sector_count[sector] = sector_count.get(sector, 0) + 1

    # 按股票数量排序
    sorted_sectors = sorted(sector_count.items(), key=lambda x: x[1], reverse=True)

    print(f"SW行业数量: {len(sorted_sectors)}个")
    print(f"股票数量最多的前10个行业:")
    for i, (sector, count) in enumerate(sorted_sectors[:10], 1):
        print(f"{i:2d}. {sector:20s}: {count:4d}只股票")

    # 7. 保存结果
    print(f"\n=== 保存分析结果 ===")

    output_file = "missing_sw_industry_analysis.txt"

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"SW行业分类缺失分析报告\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n")
            f.write(f"分析文件: {sample_file}\n")
            f.write(f"SW映射文件: {sw_mapping_path}\n\n")

            f.write(f"=== 统计摘要 ===\n")
            f.write(f"daily数据中股票总数: {total_stocks:,}\n")
            f.write(f"有SW行业分类的股票: {len(has_sw_stocks):,} ({len(has_sw_stocks)/total_stocks*100:.1f}%)\n")
            f.write(f"没有SW行业分类的股票: {len(missing_sw_stocks):,} ({len(missing_sw_stocks)/total_stocks*100:.1f}%)\n\n")

            f.write(f"=== 按交易所分布 ===\n")
            f.write(f"所有股票 - SH:{all_exchange['SH']:,}, SZ:{all_exchange['SZ']:,}, BJ:{all_exchange['BJ']:,}\n")
            f.write(f"有SW分类 - SH:{has_sw_exchange['SH']:,}, SZ:{has_sw_exchange['SZ']:,}, BJ:{has_sw_exchange['BJ']:,}\n")
            f.write(f"缺失SW分类 - SH:{missing_sw_exchange['SH']:,}, SZ:{missing_sw_exchange['SZ']:,}, BJ:{missing_sw_exchange['BJ']:,}\n\n")

            f.write(f"=== 缺失SW分类的股票列表 ===\n")
            missing_list = sorted(list(missing_sw_stocks))
            for stock in missing_list:
                f.write(f"{stock}\n")

        print(f"   已保存到文件: {output_file}")

    except Exception as e:
        print(f"   保存文件失败: {e}")

    return len(missing_sw_stocks)

if __name__ == "__main__":
    try:
        missing_count = analyze_missing_sw_industry()
        print(f"\n分析完成！")
        print(f"总计 {missing_count:,} 只股票没有SW行业分类")

    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()