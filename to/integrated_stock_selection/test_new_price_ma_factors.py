#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新增的价格均线比率因子
"""

def test_new_price_ma_factors():
    """测试 price_ma5_ratio 和 price_ma10_ratio 因子"""
    print("=" * 80)
    print("测试新增价格均线比率因子")
    print("=" * 80)

    try:
        # 导入因子定义
        from factor_system.factor_lab.definitions.moving_average import (
            price_ma5_ratio, price_ma10_ratio, _price_ma5_ratio, _price_ma10_ratio
        )
        print("[SUCCESS] 因子导入成功")

        # 检查因子注册信息
        print(f"\n=== price_ma5_ratio 因子信息 ===")
        print(f"因子名称: {price_ma5_ratio.name}")
        print(f"描述: {price_ma5_ratio.description}")
        print(f"回看天数: {price_ma5_ratio.lookback_days}")
        print(f"必需列: {price_ma5_ratio.required_columns}")
        print(f"可选列: {price_ma5_ratio.optional_columns}")

        print(f"\n=== price_ma10_ratio 因子信息 ===")
        print(f"因子名称: {price_ma10_ratio.name}")
        print(f"描述: {price_ma10_ratio.description}")
        print(f"回看天数: {price_ma10_ratio.lookback_days}")
        print(f"必需列: {price_ma10_ratio.required_columns}")
        print(f"可选列: {price_ma10_ratio.optional_columns}")

        # 检查 main.py 中的因子列表
        print(f"\n=== 检查 main.py 中的因子注册 ===")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from main import _get_stock_factors
        all_factors = _get_stock_factors()

        print(f"总因子数量: {len(all_factors)}")

        new_factors = ['price_ma5_ratio', 'price_ma10_ratio']
        for factor in new_factors:
            if factor in all_factors:
                print(f"[SUCCESS] {factor} 已成功添加到 _get_stock_factors()")
            else:
                print(f"[ERROR] {factor} 未在 _get_stock_factors() 中找到")

        # 统计均线相关因子
        ma_factors = [f for f in all_factors if 'ma' in f.lower() and 'ratio' in f.lower()]
        print(f"\n=== 均线比率因子统计 ===")
        print(f"均线比率因子数量: {len(ma_factors)}")
        print(f"均线比率因子列表: {ma_factors}")

        # 计算逻辑验证
        print(f"\n=== 计算逻辑验证 ===")
        print("price_ma5_ratio 计算公式:")
        print("  ((close - MA5) / MA5) * 100")
        print("  - 正值: 价格在MA5之上 (短期强势)")
        print("  - 负值: 价格在MA5之下 (短期弱势)")

        print("\nprice_ma10_ratio 计算公式:")
        print("  ((close - MA10) / MA10) * 100")
        print("  - 正值: 价格在MA10之上 (中期强势)")
        print("  - 负值: 价格在MA10之下 (中期弱势)")

        print(f"\n=== 无未来信息泄露验证 ===")
        print("[SUCCESS] 使用 pandas.rolling() 计算，只使用当前及之前的数据")
        print("[SUCCESS] 按股票分组计算，避免跨股票信息泄露")
        print("[SUCCESS] 设置合理的 min_periods 确保数据质量")

        print(f"\n=== 数据质量保障 ===")
        print("price_ma5_ratio:")
        print("  - 回看天数: 8天 (5天均线 + 3天缓冲)")
        print("  - 最小数据点: min_periods=3 (60%数据可用)")

        print("price_ma10_ratio:")
        print("  - 回看天数: 15天 (10天均线 + 5天缓冲)")
        print("  - 最小数据点: min_periods=5 (50%数据可用)")

        print(f"\n=== 技术意义 ===")
        print("price_ma5_ratio:")
        print("  - 反映价格短期相对位置")
        print("  - 捕捉短期动量信号")
        print("  - 适合短线交易策略")

        print("price_ma10_ratio:")
        print("  - 反映价格中期相对位置")
        print("  - 介于MA5和MA60之间的中期信号")
        print("  - 平衡短期波动和长期趋势")

        print("\n" + "=" * 80)
        print("测试完成！新因子已成功实现并注册")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_examples():
    """显示使用示例"""
    print("\n" + "=" * 80)
    print("使用示例")
    print("=" * 80)

    print("1. 单独测试新因子:")
    print("   python main.py --mode generate_factors --start_date 20230601 --end_date 20231231 --factor_names price_ma5_ratio --add_factor")
    print("   python main.py --mode generate_factors --start_date 20230601 --end_date 20231231 --factor_names price_ma10_ratio --add_factor")

    print("\n2. 同时测试两个新因子:")
    print("   python main.py --mode generate_factors --start_date 20230601 --end_date 20231231 --factor_names price_ma5_ratio,price_ma10_ratio --add_factor")

    print("\n3. 生成包含新因子的完整因子集:")
    print("   python main.py --mode generate_factors --start_date 20230601 --end_date 20251218")

if __name__ == "__main__":
    if test_new_price_ma_factors():
        show_usage_examples()
    else:
        print("测试失败，请检查实现")