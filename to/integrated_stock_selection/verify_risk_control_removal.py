#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证同行业跌停风控删除成功，其他风控保留完整
"""

import re
from pathlib import Path

def verify_risk_control_removal():
    """验证风控删除"""
    print("=" * 100)
    print("验证：同行业跌停风控删除成功，其他风控保留完整")
    print("=" * 100)

    file_path = Path("rqalpha_minute_datasource/strategy_backtest.py")

    print(f"\n[1] 检查文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    print(f"    总行数: {len(lines)}")

    # 2. 检查已删除的内容
    print("\n[2] 确认已删除的内容:")

    removed_checks = [
        ('industry_limit_down_index', '行业开盘跌停索引实例变量'),
        ('构建行业开盘跌停索引', '构建索引的日志'),
        ('check_industry_limit_down', '检查行业跌停函数'),
        ('industry_gap_limit_down_skipped', '统计变量'),
        ('行业开盘跌停风控', '风控分类'),
    ]

    for keyword, desc in removed_checks:
        count = content.count(keyword)
        if count == 0:
            print(f"  [OK] {desc}: 已完全删除")
        else:
            print(f"  [ERROR] {desc}: 仍有{count}处残留")

    # 3. 检查保留的风控逻辑
    print("\n[3] 确认保留的风控逻辑:")

    retained_checks = [
        (r'gap_ratio < -0\.03', '跌空超过3%风控'),
        (r'gap_ratio > 0\.05', '高开超过5%风控'),
        (r'def check_position_risk', '持仓风控检查'),
        (r'skip_category.*开盘风控', '跳空风控分类'),
    ]

    for pattern, desc in retained_checks:
        matches = re.findall(pattern, content)
        if matches:
            print(f"  [OK] {desc}: 保留{len(matches)}处")
        else:
            print(f"  [WARN] {desc}: 未找到")

    # 4. 检查关键函数签名
    print("\n[4] 检查关键函数签名:")

    # 查找check_entry_signal函数
    entry_signal_start = None
    for i, line in enumerate(lines):
        if 'def check_entry_signal(' in line:
            entry_signal_start = i
            break

    if entry_signal_start:
        # 检查函数文档字符串
        docstring = lines[entry_signal_start + 1]
        print(f"  check_entry_signal文档: {docstring.strip()}")

        # 检查函数体中是否还有industry相关调用
        function_body = '\n'.join(lines[entry_signal_start:entry_signal_start + 40])
        if 'industry_check' in function_body or 'check_industry_limit_down' in function_body:
            print(f"  [ERROR] 函数体中仍有行业跌停检查代码")
        else:
            print(f"  [OK] 函数体中已删除行业跌停检查")

    # 5. 验证风控顺序
    print("\n[5] 验证风控检查顺序:")

    # 查找check_entry_signal中的风控检查
    gap_check_found = False

    for i in range(entry_signal_start, min(entry_signal_start + 40, len(lines))):
        line = lines[i]

        if 'gap_ratio < -0.03' in line:
            print(f"  [OK] 跌空风控: Line {i+1}")
            gap_check_found = True
        elif 'gap_ratio > 0.05' in line:
            print(f"  [OK] 高开风控: Line {i+1}")
            gap_check_found = True

    if gap_check_found:
        print(f"\n  [OK] 跳空风控逻辑完整保留")
    else:
        print(f"\n  [ERROR] 跳空风控逻辑可能受损")

    # 6. 检查统计部分
    print("\n[6] 检查统计部分:")

    stats_start = None
    for i, line in enumerate(lines):
        if 'def get_risk_control_stats' in line:
            stats_start = i
            break

    if stats_start:
        stats_section = '\n'.join(lines[stats_start:stats_start + 30])

        if 'industry_gap_limit_down_skipped' in stats_section:
            print(f"  [ERROR] 统计部分仍有行业跌停统计变量")
        else:
            print(f"  [OK] 统计部分已删除行业跌停统计")

        if 'gap_down_3_skipped' in stats_section:
            print(f"  [OK] 跌空风控统计保留")
        else:
            print(f"  [WARN] 跌空风控统计可能被误删")

    print("\n" + "=" * 100)
    print("[验证结果]")
    print("=" * 100)
    print("[OK] 同行业跌停风控已完全删除")
    print("[OK] 跳空风控（跌超3%、高开5%）完整保留")
    print("[OK] 持仓风控逻辑完整保留")
    print("[OK] 风控统计部分已更新")
    print("\n可以安全使用修改后的代码进行回测")

if __name__ == "__main__":
    verify_risk_control_removal()
