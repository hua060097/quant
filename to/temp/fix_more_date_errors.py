#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 修复在_analyze_stock_change_point_patterns方法中的date和date_str混用问题
pattern1 = r"rel_data = relations_df\[\(relations_df\['stock_code'\] == stock_code\) & \(relations_df\['date'\] == cp_date\)\]"
replacement1 = r"rel_data = relations_df[(relations_df['stock_code'] == stock_code) & (relations_df['date_str'] == cp_date)]"

content = re.sub(pattern1, replacement1, content)

# 修复其他可能的date和date_str混用问题
pattern2 = r"df\['date'\]"
replacement2 = r"df['date_str']"

content = re.sub(pattern2, replacement2, content)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修复更多可能的date列访问错误") 