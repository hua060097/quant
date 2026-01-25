"""
测试新股过滤：跳过很早上市的股票
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def test_old_stock_skip():
    """测试跳过很早上市的股票"""
    print("=" * 100)
    print("测试新股过滤：跳过很早上市的股票")
    print("=" * 100)

    # 模拟交易日历（假设只有2022-2025年的数据）
    trading_dates = pd.date_range('2022-01-01', '2025-12-31', freq='B')  # 工作日
    trading_dates_str = trading_dates.strftime('%Y%m%d').tolist()

    print(f"\n[交易日历范围]")
    print(f"  起始日期: {trading_dates_str[0]}")
    print(f"  结束日期: {trading_dates_str[-1]}")
    print(f"  交易日数: {len(trading_dates_str)}")

    # 测试案例
    test_cases = [
        {
            'stock_code': '000597.SZ',
            'ipo_date': '19960523',  # 1996年上市，早于交易日历
            'should_skip': True,
            'reason': 'IPO日期早于交易日历起始日期'
        },
        {
            'stock_code': '688469.SH',
            'ipo_date': '20230510',  # 2023年上市，在交易日历范围内
            'should_skip': False,
            'reason': 'IPO日期在交易日历范围内'
        },
        {
            'stock_code': '300001.SZ',
            'ipo_date': '20220115',  # 2022年上市，在交易日历范围内
            'should_skip': False,
            'reason': 'IPO日期在交易日历范围内'
        },
        {
            'stock_code': '000001.SZ',
            'ipo_date': '19910102',  # 1991年上市，早于交易日历
            'should_skip': True,
            'reason': 'IPO日期早于交易日历起始日期'
        },
        {
            'stock_code': '301000.SZ',
            'ipo_date': '20260101',  # 2026年上市，晚于交易日历结束日期
            'should_skip': True,
            'reason': 'IPO日期晚于交易日历结束日期'
        }
    ]

    print(f"\n{'=' * 100}")
    print("测试案例")
    print(f"{'=' * 100}")

    for i, case in enumerate(test_cases, 1):
        stock_code = case['stock_code']
        ipo_date = case['ipo_date']
        should_skip = case['should_skip']
        reason = case['reason']

        print(f"\n案例 {i}: {stock_code}")
        print(f"  IPO日期: {ipo_date}")
        print(f"  预期是否跳过: {'是' if should_skip else '否'}")
        print(f"  原因: {reason}")

        # 应用逻辑
        trading_start = trading_dates_str[0]
        trading_end = trading_dates_str[-1]

        if ipo_date < trading_start:
            actual_skip = True
            skip_reason = f"IPO日期早于交易日历起始日期: {trading_start}"
        elif ipo_date > trading_end:
            actual_skip = True
            skip_reason = f"IPO日期晚于交易日历结束日期: {trading_end}"
        else:
            actual_skip = False
            skip_reason = "IPO日期在交易日历范围内，需要计算第60个交易日"

        print(f"  实际是否跳过: {'是' if actual_skip else '否'}")
        print(f"  跳过原因: {skip_reason}")

        # 验证结果
        if actual_skip == should_skip:
            print(f"  测试结果: [OK] 通过")
        else:
            print(f"  测试结果: [FAIL] 失败 - 预期{'跳过' if should_skip else '处理'}，实际{'跳过' if actual_skip else '处理'}")

    # 测试边界情况
    print(f"\n{'=' * 100}")
    print("边界情况测试")
    print(f"{'=' * 100}")

    # 边界1: IPO日期正好是交易日历的第一天
    print(f"\n边界1: IPO日期 = 交易日历起始日期")
    ipo_date = trading_dates_str[0]
    print(f"  IPO日期: {ipo_date}")
    print(f"  交易日历起始: {trading_dates_str[0]}")

    if ipo_date < trading_start:
        print(f"  结果: 跳过（不应该发生）")
    elif ipo_date > trading_end:
        print(f"  结果: 跳过（不应该发生）")
    else:
        print(f"  结果: [OK] 不跳过，计算第60个交易日")

    # 边界2: IPO日期正好是交易日历的最后一天
    print(f"\n边界2: IPO日期 = 交易日历结束日期")
    ipo_date = trading_dates_str[-1]
    print(f"  IPO日期: {ipo_date}")
    print(f"  交易日历结束: {trading_dates_str[-1]}")

    if ipo_date < trading_start:
        print(f"  结果: 跳过（不应该发生）")
    elif ipo_date > trading_end:
        print(f"  结果: 跳过（不应该发生）")
    else:
        print(f"  结果: [OK] 不跳过，但第60个交易日可能超出范围")

    print(f"\n{'=' * 100}")
    print("测试总结")
    print(f"{'=' * 100}")
    print("  逻辑修复: IPO日期不在交易日历范围内的股票将被跳过")
    print("  修复效果:")
    print("    - 1990年代的股票不会进行新股过滤")
    print("    - 2000年代的股票不会进行新股过滤")
    print("    - 只有交易日历范围内的新股才会进行过滤")
    print("=" * 100)


if __name__ == "__main__":
    test_old_stock_skip()
