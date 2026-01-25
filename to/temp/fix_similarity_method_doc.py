#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 修复_calculate_weighted_similarity方法的文档注释
pattern = r"Returns:\s+tuple: \(相似度分数, 匹配的特征列表\)"
replacement = r"Returns:\n            tuple: (相似度分数, 匹配的特征列表, 特征贡献度字典)"

# 使用re.sub替换匹配的模式
content = re.sub(pattern, replacement, content)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修复_calculate_weighted_similarity方法的文档注释") 