#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Unicode修复结果
"""

import re

def test_unicode_fix():
    """测试Unicode字符是否已被正确替换"""

    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否包含常见的emoji字符
    emoji_chars = ['✅', '❌', '🚫', '⚠️', '🗑️', '📅', '🔄', '⭐', '🎯', '📊']

    print("Checking for remaining emoji characters...")
    found_emojis = []

    for char in emoji_chars:
        if char in content:
            count = content.count(char)
            found_emojis.append(f"'{char}': {count}")

    if found_emojis:
        print("Remaining emoji characters found:")
        for item in found_emojis:
            print(f"  {item}")
        return False
    else:
        print("No emoji characters found - good!")

    # 检查替换文本是否存在
    replacement_texts = ['[OK]', '[FAIL]', '[FILTER]', '[WARNING]', '[DELETE]', '[DATE]', '[REFRESH]']

    print("\nChecking for replacement texts...")
    found_replacements = []

    for text in replacement_texts:
        if text in content:
            count = content.count(text)
            found_replacements.append(f"'{text}': {count}")

    if found_replacements:
        print("Replacement texts found:")
        for item in found_replacements:
            print(f"  {item}")
    else:
        print("No replacement texts found!")
        return False

    # 检查一些特定行
    print("\nChecking specific lines that were mentioned...")

    lines = content.split('\n')

    # Line 5807: 历史交易天数过滤
    if 5807 <= len(lines):
        line_5807 = lines[5806]  # 0-based indexing
        if "[FILTER]" in line_5807 and "历史交易天数过滤" in line_5807:
            print("Line 5807: OK")
        else:
            print(f"Line 5807: {line_5807}")

    # Line 5820: 新股过滤失败
    if 5820 <= len(lines):
        line_5820 = lines[5819]
        if "新股过滤失败" in line_5820 and not "❌" in line_5820:
            print("Line 5820: OK")
        else:
            print(f"Line 5820: {line_5820}")

    # Lines 977-983: ✅和❌符号
    if 983 <= len(lines):
        for i in range(976, 983):
            line = lines[i]
            if "[OK]" in line or "[FAIL]" in line:
                if "✅" not in line and "❌" not in line:
                    print(f"Line {i+1}: OK")
                else:
                    print(f"Line {i+1}: Still contains emoji: {line}")

    return True

if __name__ == "__main__":
    success = test_unicode_fix()
    if success:
        print("\nUnicode fix appears successful!")
    else:
        print("\nUnicode fix may have issues.")