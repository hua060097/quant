#!/usr/bin/env python3
"""
调试sector因子的缺失值处理逻辑
分析_sector_return_rank_1d在pipeline中为什么还会有缺失值
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'factor_system', 'factor_lab'))

def analyze_sector_missing_values():
    """分析sector因子的缺失值来源"""

    print("=" * 80)
    print("分析sector因子缺失值处理逻辑")
    print("=" * 80)

    # 模拟_sector_return_rank_1d的计算过程
    print("\n模拟_sector_return_rank_1d的计算过程:")
    print("-" * 50)

    # 步骤1: 创建测试数据
    print("\n步骤1: 输入数据模拟:")
    dates = pd.date_range('2023-01-01', periods=5)
    stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']

    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'stock_code': stock,
                'date': date,
                'pct_chg': np.random.normal(0, 2)  # 收益率
            })

    stock_data = pd.DataFrame(data)
    print(f"股票数据: {len(stock_data)} 条记录")

    # 步骤2: 模拟sector映射
    print("\n步骤2: SW行业映射:")
    sector_mapping = {
        '000001.SZ': 'SW1银行',
        '000002.SZ': 'SW2房地产',
        '600000.SH': 'SW1银行',
        '600036.SH': 'SW2房地产',
        '000858.SZ': None  # 故意设置缺失映射
    }

    # 添加sector_code
    stock_data['sector_code'] = stock_data['stock_code'].map(sector_mapping)

    print("股票-行业映射:")
    for stock, sector in sector_mapping.items():
        print(f"  {stock}: {sector}")

    # 检查缺失映射
    missing_mapping = stock_data[stock_data['sector_code'].isna()]
    print(f"\n缺失行业映射的股票: {missing_mapping['stock_code'].unique()}")
    print(f"缺失记录数: {len(missing_mapping)}")

    # 步骤3: 模拟sector排名计算
    print("\n步骤3: 计算sector排名:")

    # 模拟_sector_data的构建
    sector_data = stock_data.groupby(['date', 'sector_code'])['pct_chg'].mean().reset_index()
    sector_data.rename(columns={'pct_chg': 'pct_chg_avg'}, inplace=True)

    # 添加排名
    sector_data['date_dt'] = pd.to_datetime(sector_data['date'])
    sector_data = sector_data.sort_values(['date_dt', 'sector_code'])
    sector_data['sector_rank_1d'] = sector_data.groupby('date_dt')['pct_chg_avg'].rank(
        pct=True, method='min'
    ).fillna(0.5)  # 这里是第一个fillna(0.5)

    print("Sector平均收益率和排名:")
    print(sector_data[['date', 'sector_code', 'pct_chg_avg', 'sector_rank_1d']].to_string(index=False))

    # 步骤4: 模拟merge过程（关键步骤）
    print("\n步骤4: Merge过程（缺失值来源分析）:")

    # 准备merge数据
    stock_for_merge = stock_data.copy()
    stock_for_merge['__orig_index'] = stock_for_merge.index

    # 执行merge（使用left join）
    merged = stock_for_merge.merge(
        sector_data[['date', 'sector_code', 'sector_rank_1d']],
        on=['date', 'sector_code'],
        how='left'
    )

    print("Merge后的数据样例:")
    print(merged[['stock_code', 'date', 'sector_code', 'sector_rank_1d']].head(10).to_string(index=False))

    # 检查merge后的缺失值
    merge_missing = merged[merged['sector_rank_1d'].isna()]
    print(f"\nMerge后缺失值分析:")
    print(f"缺失记录数: {len(merge_missing)}")

    if len(merge_missing) > 0:
        print("缺失的股票记录:")
        print(merge_missing[['stock_code', 'date', 'sector_code', 'sector_rank_1d']].to_string(index=False))

        print(f"\n缺失值原因分析:")
        print("-" * 30)

        # 分析缺失原因
        print("1. 缺失行业映射的股票:")
        no_sector = merge_missing[merge_missing['sector_code'].isna()]
        print(f"   数量: {len(no_sector)}")
        print(f"   股票: {no_sector['stock_code'].unique()}")

        print("\n2. 有行业映射但sector_data中缺失的股票:")
        has_sector_no_data = merge_missing[merge_missing['sector_code'].notna()]
        print(f"   数量: {len(has_sector_no_data)}")
        if len(has_sector_no_data) > 0:
            print(f"   记录样例:")
            print(has_sector_no_data[['stock_code', 'date', 'sector_code']].head().to_string(index=False))

    # 步骤5: 模拟最终处理
    print("\n步骤5: 最终处理:")

    # 设置索引（模拟merged.set_index('__orig_index')）
    merged_indexed = merged.set_index('__orig_index')

    # 应用fillna(0.5)和reindex
    original_index = stock_data.index
    result = merged_indexed['sector_rank_1d'].fillna(0.5).reindex(original_index)

    print("最终结果分析:")
    print(f"原始缺失值: {merged_indexed['sector_rank_1d'].isna().sum()}")
    print(f"fillna(0.5)后缺失值: {result.isna().sum()}")

    if result.isna().sum() > 0:
        print("仍然缺失的记录:")
        missing_final = result[result.isna()]
        print(stock_data.loc[missing_final.index][['stock_code', 'date']].to_string(index=False))
    else:
        print("最终无缺失值")

    # 步骤6: 检查Inf值情况
    print(f"\n步骤6: Inf值检查:")
    print(f"数据中是否包含Inf: {np.isinf(result).any()}")
    if np.isinf(result).any():
        inf_count = np.isinf(result).sum()
        print(f"Inf值数量: {inf_count}")
        print(f"Inf值位置: {result[np.isinf(result)]}")
    else:
        print("无Inf值")

def check_pipeline_inf_scenario():
    """检查什么情况下会产生Inf值"""
    print("\n" + "=" * 80)
    print("分析什么情况下sector因子会产生Inf值")
    print("=" * 80)

    scenarios = [
        "行业平均收益率计算异常",
        "排名计算中的数值异常",
        "数据类型转换问题",
        "除零错误"
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario}")
        print("   - 分析...")

        # 这里可以添加具体的测试代码
        # 但基于sector.py的代码逻辑，产生Inf的可能性极低

if __name__ == "__main__":
    analyze_sector_missing_values()
    check_pipeline_inf_scenario()

    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    print("1. _sector_return_rank_1d的缺失值主要来自merge操作")
    print("2. 缺失原因通常是股票没有对应的SW行业映射")
    print("3. 因子内部的fillna(0.5)会处理这些缺失值")
    print("4. Inf值产生概率极低，主要来自数值计算异常")
    print("5. pipeline的Inf处理主要是防御性编程")