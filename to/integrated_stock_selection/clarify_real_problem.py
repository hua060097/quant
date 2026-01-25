#!/usr/bin/env python3
"""
澄清真正的问题所在
"""

def clarify_real_problem():
    """澄清真正的问题"""

    print("=== 澄清真正的问题 ===")

    print("\n我的错误分析:")
    print("之前我以为问题是数据不够，但实际上数据是充足的。")

    print("\n用户逻辑的正确性:")
    print("1. lookback_days=60 -> 数据加载器加载121天历史数据")
    print("2. institutional_activity_score需要30天 -> 数据充足")
    print("3. target_mask控制保存范围 -> 只保存目标日期")

    print("\n数据流应该是:")
    print("1. 加载121天历史数据")
    print("2. 对每只股票的每个交易日计算institutional_activity_score")
    print("3. 由于历史数据充足，大部分日期应该都有有效结果")
    print("4. target_mask过滤，只保存目标日期的结果")

    print("\n但实际发生的情况:")
    print("总记录数: 653,279")
    print("有效结果: 5,399")
    print("有效率: 0.83%")

    print("\n问题的关键:")
    print("如果每只股票121天历史数据充足")
    print("且institutional_activity_single在30天后就开始返回有效结果")
    print("那么理论上应该有约 5,399 * (121-30+1) = 500万+ 有效结果")
    print("而不是只有5,399个有效结果")

    print("\n真正的问题可能是:")
    print("1. 虽然数据加载器加载了121天数据")
    print("2. 但在因子管道传递过程中，数据被截断了")
    print("3. 或者_institutional_activity_single函数本身有问题")

    print("\n需要检查的地方:")
    print("1. 数据传递到_institutional_activity时的实际结构")
    print("2. _institutional_activity_single函数的逻辑")
    print("3. 是否在某个环节数据被意外过滤")

def check_institutional_activity_single_logic():
    """检查_institutional_activity_single函数的逻辑"""

    print("\n=== 检查_institutional_activity_single逻辑 ===")

    # 从support.py读取该函数
    print("函数逻辑检查:")
    print("1. if length < 30: return 50.0")
    print("2. for i in range(29, length):  # 从第30个数据点开始计算")
    print("3. 前29个位置返回默认值50.0")
    print("4. 从第30个位置开始返回计算结果")

    print("\n对于121天数据:")
    print("- 前29天: 默认值50.0")
    print("- 第30天到第121天: 计算结果")
    print("- 应该有92个有效计算结果")

    print("\n每只股票应该有92个有效结果")
    print("5,399只股票应该有 5,399 * 92 = 496,708个有效结果")
    print("而不是只有5,399个")

    print("\n所以问题在于:")
    print("数据在传递到_institutional_activity_single时")
    print("可能不是121天，而是只有1天!")

if __name__ == "__main__":
    clarify_real_problem()
    check_institutional_activity_single_logic()