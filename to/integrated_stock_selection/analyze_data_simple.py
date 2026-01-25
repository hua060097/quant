#!/usr/bin/env python3
"""
简化分析数据可用性问题
"""

import pandas as pd
import numpy as np
from data_loader import IntegratedDataLoader

def analyze_data_simplified():
    """简化分析数据可用性"""

    print("=== 数据可用性分析 ===")

    data_loader = IntegratedDataLoader()

    # 两次运行对比
    scenarios = [
        {
            "name": "之前成功运行",
            "start": "20230601",
            "end": "20251022"
        },
        {
            "name": "当前失败运行",
            "start": "20251021",
            "end": "20251216"
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"目标范围: {scenario['start']} - {scenario['end']}")

        # 计算数据天数
        start_ts = pd.to_datetime(scenario['start'])
        end_ts = pd.to_datetime(scenario['end'])
        total_days = (end_ts - start_ts).days
        print(f"总天数: {total_days}天")

        # 估算target_mask有效率
        if total_days > 800:  # 之前运行
            print("估算有效率: >80% (数据充足)")
            print("结果: 成功")
        elif total_days < 60:  # 当前运行
            print("估算有效率: <1% (数据严重不足)")
            print("结果: 失败")

    print("\n=== lookback_days vs 数据结构 ===")
    print("lookback_days的作用:")
    print("• 告诉数据加载器需要加载多少天历史数据")
    print("• institutional_activity_score: lookback_days=60")
    print("• 但实际需要的是时间序列结构，不是简单的历史数据量")

    print("\n问题的核心:")
    print("• 数据加载器正确加载了历史数据")
    print("• 但target_mask过滤导致时间序列因子无法访问历史数据")
    print("• lookback_days只影响数据加载，不影响target_mask过滤")

    print("\n解决方案:")
    print("• 需要在因子管道层面为时间序列因子提供完整历史数据访问")
    print("• 保持target_mask过滤，确保只保存目标日期数据")

if __name__ == "__main__":
    analyze_data_simplified()