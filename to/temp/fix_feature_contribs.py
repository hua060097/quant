#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 修复计算特征贡献度可能的除零问题
pattern = r'''        # 计算每个特征对相似度的贡献
        feature_contributions = {}
        if similarity > 0:
            for i, feature in enumerate\(common_features\):
                # 计算特征贡献度
                contribution = weights\[i\] \* vec1\[i\] \* vec2\[i\] / \(similarity \* denominator\)
                feature_contributions\[feature\] = contribution
            
            # 按贡献度排序
            feature_contributions = {k: v for k, v in sorted\(
                feature_contributions\.items\(\), key=lambda item: abs\(item\[1\]\), reverse=True
            \)}'''

replacement = r'''        # 计算每个特征对相似度的贡献
        feature_contributions = {}
        if similarity > 0 and denominator > 0:  # 确保分母大于0
            for i, feature in enumerate(common_features):
                try:
                    # 安全计算特征贡献度
                    contribution = weights[i] * vec1[i] * vec2[i] / (similarity * denominator)
                    feature_contributions[feature] = contribution
                except (ZeroDivisionError, RuntimeWarning):
                    # 处理可能的除零错误
                    feature_contributions[feature] = 0.0
            
            # 按贡献度排序
            feature_contributions = {k: v for k, v in sorted(
                feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True
            )}'''

# 使用re.sub替换匹配的模式
content = re.sub(pattern, replacement, content)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修复_calculate_weighted_similarity方法中计算特征贡献度的除零问题") 