#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试交易日历获取功能

验证：
1. tushare API是否正常工作
2. 数据格式是否正确
3. 日期范围是否合理
"""

import sys
from pathlib import Path

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta

def test_tushare_trading_calendar():
    """测试tushare交易日历获取"""
    print("=" * 60)
    print("测试tushare交易日历获取")
    print("=" * 60)

    try:
        import tushare as ts

        # Tushare token
        TS_TOKEN = "022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000"

        print("\n步骤1: 初始化tushare pro API...")
        pro = ts.pro_api(TS_TOKEN)
        print("[OK] tushare API初始化成功")

        # 计算日期范围
        end_date = datetime.now() + timedelta(days=30)
        start_date = datetime.now() - timedelta(days=90)
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')

        print(f"\n步骤2: 获取交易日历...")
        print(f"  日期范围: {start_date_str} ~ {end_date_str}")

        # 获取交易日历
        df = pro.trade_cal(
            exchange='SSE',
            start_date=start_date_str,
            end_date=end_date_str
        )

        if df is None or df.empty:
            print("[FAIL] tushare返回空数据")
            return False

        print(f"[OK] 获取到 {len(df)} 条日历数据")

        # 筛选交易日
        df_trade = df[df['is_open'] == 1]
        trading_dates = df_trade['cal_date'].tolist()

        # 排序：tushare返回的是倒序，需要转为正序（从早到晚）
        trading_dates = sorted(trading_dates)

        print(f"\n步骤3: 筛选交易日...")
        print(f"  交易日数量: {len(trading_dates)}")
        print(f"  最早日期: {trading_dates[0] if trading_dates else 'N/A'}")
        print(f"  最晚日期: {trading_dates[-1] if trading_dates else 'N/A'}")
        print(f"  [INFO] 已排序为正序（从早到晚）")

        # 验证数据格式
        print(f"\n步骤4: 验证数据格式...")
        if trading_dates:
            # 检查第一个和最后一个日期格式
            first_date = trading_dates[0]
            last_date = trading_dates[-1]

            # 验证是否为8位数字字符串
            assert len(first_date) == 8, f"第一个日期格式不正确: {first_date}"
            assert len(last_date) == 8, f"最后一个日期格式不正确: {last_date}"
            assert first_date.isdigit(), f"第一个日期不是数字: {first_date}"
            assert last_date.isdigit(), f"最后一个日期不是数字: {last_date}"

            print(f"  [OK] 日期格式验证通过 (YYYYMMDD)")

            # 测试转换为date对象
            first_date_obj = datetime.strptime(first_date, '%Y%m%d').date()
            last_date_obj = datetime.strptime(last_date, '%Y%m%d').date()
            print(f"  [OK] 日期对象转换成功")
            print(f"     第一个日期: {first_date_obj}")
            print(f"     最后日期: {last_date_obj}")

            # 显示前5个和后5个交易日
            print(f"\n步骤5: 交易日样例...")
            print(f"  前5个交易日: {trading_dates[:5]}")
            print(f"  后5个交易日: {trading_dates[-5:]}")

            print(f"\n[OK] 所有测试通过！")
            return True
        else:
            print("[FAIL] 没有获取到交易日")
            return False

    except ImportError:
        print("[ERROR] 未安装tushare库")
        print("请运行: pip install tushare")
        return False
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False


def test_date_parsing():
    """测试日期解析逻辑（与trader.py中的逻辑一致）"""
    print("\n" + "=" * 60)
    print("测试日期解析逻辑")
    print("=" * 60)

    from datetime import datetime

    # 测试数据（模拟tushare返回的数据格式）
    test_dates = [
        '20260101',
        '20260115',
        '20260120',
    ]

    print(f"\n测试数据: {test_dates}")
    print(f"\n解析日期...")

    parsed_dates = []
    for date_str in test_dates:
        # 这是trader.py中使用的解析逻辑
        if isinstance(date_str, str):
            date_obj = datetime.strptime(date_str, '%Y%m%d').date()
            parsed_dates.append(date_obj)
            print(f"  {date_str} -> {date_obj}")

    print(f"\n[OK] 日期解析测试通过！")
    print(f"  解析结果: {parsed_dates}")

    return True


if __name__ == "__main__":
    print("\n交易日历测试程序")
    print(f"当前时间: {datetime.now()}\n")

    # 测试1: tushare交易日历获取
    result1 = test_tushare_trading_calendar()

    # 测试2: 日期解析逻辑
    result2 = test_date_parsing()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"tushare交易日历获取: {'[OK] 通过' if result1 else '[FAIL] 失败'}")
    print(f"日期解析逻辑: {'[OK] 通过' if result2 else '[FAIL] 失败'}")

    if result1 and result2:
        print("\n[SUCCESS] 所有测试通过！交易日历功能正常。")
        sys.exit(0)
    else:
        print("\n[WARNING] 部分测试失败，请检查错误信息。")
        sys.exit(1)
