#!/usr/bin/env python3
"""
分析为什么之前的因子生成没有报错
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import IntegratedDataLoader

def analyze_data_availability():
    """分析不同日期范围的数据可用性"""

    print("=== 分析数据可用性对因子计算的影响 ===")

    data_loader = IntegratedDataLoader()

    # 模拟两次运行的参数
    scenarios = [
        {
            "name": "之前成功运行",
            "start_date": "20230601",
            "end_date": "20251022",
            "extend_history_days": 120
        },
        {
            "name": "当前失败运行",
            "start_date": "20251021",
            "end_date": "20251216",
            "extend_history_days": 120
        }
    ]

    for scenario in scenarios:
        print(f"\n=== {scenario['name']} ===")
        print(f"目标范围: {scenario['start_date']} - {scenario['end_date']}")

        # 计算扩展开始日期
        extended_start_date = data_loader._calculate_extended_start_date(
            scenario['start_date'], scenario['end_date'], scenario['extend_history_days']
        )

        print(f"扩展开始日期: {extended_start_date}")
        print(f"总历史天数: {(pd.to_datetime(scenario['end_date']) - pd.to_datetime(extended_start_date)).days}天")

        # 加载数据
        stock_data = data_loader.load_stock_features(extended_start_date, scenario['end_date'])

        if stock_data is None:
            print("无法加载数据")
            continue

        print(f"总记录数: {len(stock_data):,}")
        print(f"股票数量: {stock_data['stock_code'].nunique():,}")

        # 分析数据分布
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        start_ts = pd.to_datetime(scenario['start_date'])
        end_ts = pd.to_datetime(scenario['end_date'])

        target_mask = (stock_data['date'] >= start_ts) & (stock_data['date'] <= end_ts)

        print(f"目标范围内记录数: {target_mask.sum():,}")
        print(f"目标范围外记录数: {(~target_mask).sum():,}")

        # 计算target_mask有效率
        valid_rate = target_mask.sum() / len(stock_data)
        print(f"target_mask有效率: {valid_rate:.4f}")

        # 分析每只股票的历史数据长度
        records_per_stock = stock_data['stock_code'].value_counts()
        print(f"平均每只股票记录数: {records_per_stock.mean():.1f}")
        print(f"最小记录数: {records_per_stock.min()}")

        # 分析institutional_activity_score的影响
        print(f"\n对institutional_activity_score的影响:")
        print(f"基于target_mask的有效率估算: {valid_rate:.4f} ({valid_rate*100:.2f}%)")

        if valid_rate > 0.5:
            print("✅ 数据充足，因子计算成功率较高")
        elif valid_rate > 0.1:
            print("⚠️  数据偏少，可能出现部分缺失")
        else:
            print("❌ 数据严重不足，因子计算很可能失败")

def analyze_lookback_days_vs_data():
    """分析lookback_days参数与实际数据需求的关系"""

    print("\n=== lookback_days参数分析 ===")

    # 查看不同因子的lookback_days设置
    from factor_system.factor_lab.definitions import support
    import inspect

    # 获取所有因子定义
    factor_definitions = {}
    for name in dir(support):
        obj = getattr(support, name)
        if hasattr(obj, 'name') and hasattr(obj, 'lookback_days'):
            factor_definitions[obj.name] = obj.lookback_days

    print("各因子的lookback_days设置:")
    for factor_name, lookback_days in sorted(factor_definitions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {factor_name}: {lookback_days}天")

    print(f"\n关键发现:")
    print(f"• institutional_activity_score: {factor_definitions.get('institutional_activity_score', 'N/A')}天")
    print(f"• near_support_count_60d: {factor_definitions.get('near_support_count_60d', 'N/A')}天")

    print(f"\nlookback_days的作用:")
    print("• lookback_days告诉因子管道需要加载多少天历史数据")
    print("• 但不保证每只股票都有完整的历史数据")
    print("• 数据质量问题（如新股上市）会导致实际数据不足")

if __name__ == "__main__":
    analyze_data_availability()
    analyze_lookback_days_vs_data()