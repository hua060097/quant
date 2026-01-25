#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 修复尝试访问'date'列的代码
pattern = r"latest_date = predict_features\['date'\]\.max\(\)"
replacement = r"latest_date = predict_features['date_str'].max()"

# 使用re.sub替换匹配的模式
content = re.sub(pattern, replacement, content)

# 检查并修复_predict_stocks_with_change_points方法中所有对'date'列的引用
pattern2 = r"predict_features\['date'\]"
replacement2 = r"predict_features['date_str']"

# 替换其他可能的date引用
content = re.sub(pattern2, replacement2, content)

# 检查relations_df中的date访问
pattern3 = r"relations_df\['date'\]"
replacement3 = r"relations_df['date_str']"

# 替换relations_df中date引用
content = re.sub(pattern3, replacement3, content)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修复_predict_stocks_with_change_points方法中的date列访问错误") 