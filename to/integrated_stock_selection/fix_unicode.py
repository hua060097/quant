#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复main.py中的Unicode字符编码问题
将所有emoji和特殊字符替换为适合GBK编码的纯文本
"""

import re
import os

def fix_unicode_in_file(file_path):
    """修复文件中的Unicode字符"""

    # 读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Failed to read file {file_path}")
        return False

    original_content = content

    # Unicode字符替换映射表
    replacements = {
        # 替换emoji为纯文本
        '✅': '[OK]',
        '❌': '[FAIL]',
        '🚫': '[FILTER]',
        '⚠️': '[WARNING]',
        '🗑️': '[DELETE]',
        '📅': '[DATE]',
        '🔄': '[REFRESH]',
        '⭐': '[STAR]',
        '🎯': '[TARGET]',
        '📊': '[CHART]',
        '📈': '[UP]',
        '📉': '[DOWN]',
        '🔍': '[SEARCH]',
        '💹': '[PROFIT]',
        '🔥': '[HOT]',
        '❗️': '[IMPORTANT]',
        '✨': '[SPARKLE]',
        '🏆': '[TROPHY]',
        '🛠️': '[TOOL]',
        '💡': '[IDEA]',
        '🔔': '[BELL]',
        '📢': '[ANNOUNCE]',
        '🎉': '[CELEBRATE]',
        '🎊': '[PARTY]',
        '🎁': '[GIFT]',
        '🎈': '[BALLOON]',
        '🎀': '[RIBBON]',
        '🔝': '[TOP]',
        '🔟': '[NUMBER10]',
        '🔢': '[NUMBERS]',
        '🔣': '[SYMBOLS]',
        '🔤': '[LETTERS]',
        '🔡': '[LOWER]',
        '🔠': '[UPPER]',
        '🔗': '[LINK]',
        '🔌': '[PLUGIN]',
        '🔎': '[MAGNIFY]',
        '🔓': '[UNLOCK]',
        '🔒': '[LOCK]',
        '🔏': '[LOCKED]',
        '🔐': '[SECURE]',
        '🔑': '[KEY]',
        '🔖': '[BOOKMARK]',
        '📋': '[CLIPBOARD]',
        '📌': '[PIN]',
        '📍': '[LOCATION]',
        '📎': '[PAPERCLIP]',
        '📏': '[RULER]',
        '📐': '[TRIANGLE]',
        '💾': '[SAVE]',
        '🔧': '[WRENCH]',

        # 替换其他特殊符号
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v',
        '•': '*',
        '…': '...',
        '—': '-',
        '–': '-',

        # 中文状态词替换（可选，如果需要的话）
        # 这里保留中文，因为GBK支持中文
    }

    # 应用替换
    changes_made = 0
    for unicode_char, replacement in replacements.items():
        if unicode_char in content:
            count = content.count(unicode_char)
            content = content.replace(unicode_char, replacement)
            changes_made += count
            print(f"Replaced unicode char with '{replacement}' ({count} times)")

    # 如果有修改，写回文件
    if changes_made > 0:
        # 创建备份
        backup_path = file_path + '.unicode_backup'
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            print(f"Created backup file: {backup_path}")
        except Exception as e:
            print(f"Failed to create backup: {e}")

        # 写入修复后的内容
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed file: {file_path} (replaced {changes_made} characters)")
            return True
        except Exception as e:
            print(f"Failed to write file: {e}")
            return False
    else:
        print(f"File {file_path} needs no fix")
        return True

def main():
    """主函数"""
    # 目标文件路径
    file_path = "C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/main.py"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Starting to fix file: {file_path}")
    success = fix_unicode_in_file(file_path)

    if success:
        print("Fix completed!")
    else:
        print("Fix failed!")

if __name__ == "__main__":
    main()