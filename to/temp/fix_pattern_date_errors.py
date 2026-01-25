#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 修复在pattern_details中使用'date'字段的问题
pattern1 = r"'date': pattern\.get\('date', None\)"
replacement1 = r"'date': pattern.get('date_str', None)"

content = re.sub(pattern1, replacement1, content)

# 修复pattern中直接引用'date'的问题
pattern2 = r"pattern\['date'\]"
replacement2 = r"pattern['date_str']"

content = re.sub(pattern2, replacement2, content)

# 修复col检查中排除'date'的情况
pattern3 = r"col not in \['date', 'stock_code', 'stock_name'\]"
replacement3 = r"col not in ['date_str', 'stock_code', 'stock_name']"

content = re.sub(pattern3, replacement3, content)

# 修复pattern.get('date')的调用
pattern4 = r"pattern\.get\('date'"
replacement4 = r"pattern.get('date_str'"

content = re.sub(pattern4, replacement4, content)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修复模式详情中的date字段引用") 