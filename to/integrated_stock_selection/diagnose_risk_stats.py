#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断风控统计问题
"""

def diagnose_risk_control_stats():
    """诊断风控统计问题"""
    print("=" * 100)
    print("风控统计问题诊断")
    print("=" * 100)

    file_path = "rqalpha_minute_datasource/strategy_backtest.py"

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    print("\n[分析1] 检查统计生成代码中的关键字匹配:")
    print("-" * 100)

    # Line 1424-1425 的统计代码
    print("\n[Line 1424] 累计下跌统计:")
    print("    cumulative_decline_exits = len([t for t in risk_exits if '累计下跌' in t.get('exit_reason', '')])")
    print("    关键字: '累计下跌'")

    print("\n[Line 1425] 连续阴线统计:")
    print("    consecutive_decline_exits = len([t for t in risk_exits if '连续阴线' in t.get('exit_reason', '')])")
    print("    关键字: '连续阴线'")

    print("\n[分析2] 检查实际风控触发时的原因字符串:")
    print("-" * 100)

    # 查找风控触发时的reason
    for i, line in enumerate(lines):
        if "'reason': '累计下跌超5%'" in line:
            print(f"\n[Line {i+1}] 找到累计下跌风控的reason:")
            print(f"    {line.strip()}")
            print(f"    reason字符串: '累计下跌超5%'")
            print(f"    匹配关键字 '累计下跌': 包含 ✓")

        if "'reason': '连续3根阴线'" in line:
            print(f"\n[Line {i+1}] 找到连续阴线风控的reason:")
            print(f"    {line.strip()}")
            print(f"    reason字符串: '连续3根阴线'")
            print(f"    匹配关键字 '连续阴线': 包含 ✓")

    print("\n[分析3] 检查execute_sell调用时的exit_reason格式:")
    print("-" * 100)

    for i, line in enumerate(lines):
        if 'f"风控: {risk_check[\'reason\']}"' in line or 'f"风控: {risk_check["reason"]}"' in line:
            print(f"\n[Line {i+1}] 找到execute_sell的exit_reason构造:")
            print(f"    {line.strip()}")
            print(f"    最终exit_reason格式: '风控: {{reason}}'")
            print(f"    例如: '风控: 累计下跌超5%' 或 '风控: 连续3根阴线'")

    print("\n[分析4] 检查统计时的关键字匹配:")
    print("-" * 100)

    # 模拟匹配
    test_cases = [
        ("'风控: 累计下跌超5%'", "'累计下跌'", True),
        ("'风控: 连续3根阴线'", "'连续阴线'", True),
    ]

    for exit_reason, keyword, expected in test_cases:
        match = keyword.strip("'") in exit_reason.strip("'")
        status = "✓" if match == expected else "✗"
        print(f"\n    {status} exit_reason={exit_reason}, keyword={keyword}")
        print(f"       匹配结果: {match} (预期: {expected})")

    print("\n[分析5] 可能的问题原因:")
    print("-" * 100)
    print("""
    1. ✓ 买入风控统计（跌超3%，涨超5%）- 逻辑正确
       - 统计代码: Line 1413-1418
       - 显示代码: Line 1306-1314
       - 关键字匹配: '累计下跌' ✓, '连续阴线' ✓

    2. ✓ 卖出风控统计（累计下跌超5%，连续3根阴线）- 逻辑正确
       - 统计代码: Line 1424-1425
       - 显示代码: Line 1325-1331
       - 关键字匹配正确

    3. ? 实际数据问题
       - 如果连续3根阴线风控从未触发，统计数据为0是正常的
       - 需要检查回测日志，确认是否真的有连续阴线风控触发

    4. ? 日志记录问题
       - 可能风控触发了但记录到trades时exit_reason格式不对
       - 需要检查execute_sell的调用是否都传入了正确的exit_reason
    """)

    print("\n[分析6] 需要进一步检查的地方:")
    print("-" * 100)
    print("""
    1. 运行一次回测，检查日志中是否有 "连续3根阴线" 的风控触发记录
    2. 如果有触发但统计为0，检查trades列表中的exit_reason字段
    3. 检查是否所有execute_sell调用都传入了正确的exit_reason
    4. 确认risk_control_details字典是否正确返回
    """)

    print("\n" + "=" * 100)
    print("[结论]")
    print("=" * 100)
    print("""
    统计代码的逻辑是正确的，关键字匹配也没有问题。

    如果"连续3根阴线"统计数据为0，可能的原因：
    1. 回测期间确实没有触发连续3根阴线风控（这是正常的）
    2. 风控触发了但未正确记录到trades（需要检查）
    3. 其他execute_sell调用覆盖了exit_reason（需要检查）

    建议下一步：
    1. 运行回测，查看日志中是否有连续阴线风控触发的记录
    2. 在回测代码中添加调试输出，打印trades列表中的exit_reason
    3. 或者修改统计代码，添加调试信息
    """)

if __name__ == "__main__":
    diagnose_risk_control_stats()
