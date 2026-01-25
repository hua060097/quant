#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 修复在计算技术指标预测相关性时可能的除零或NaN问题
pattern1 = r"corr = np\.corrcoef\(price_changes\[\:-1\], tech_values\[1:\]\)\[0, 1\]"
replacement1 = r"""# 安全计算相关系数，避免无效值
                                            if not np.isnan(price_changes).any() and not np.isnan(tech_values).any() and np.std(price_changes[:-1]) > 0 and np.std(tech_values[1:]) > 0:
                                                corr = np.corrcoef(price_changes[:-1], tech_values[1:])[0, 1]
                                                pattern[f'{tech_feature}_predictive_corr'] = corr
                                            else:
                                                pattern[f'{tech_feature}_predictive_corr'] = 0.0"""

content = re.sub(pattern1, replacement1, content)

# 修复板块特征相关性计算中可能的除零或NaN问题
pattern2 = r"corr = np\.corrcoef\(board_values\[\:-1\], price_changes\[1:\]\)\[0, 1\]"
replacement2 = r"""# 安全计算板块特征与个股表现相关性，避免无效值
                                            if not np.isnan(board_values).any() and not np.isnan(price_changes).any() and np.std(board_values[:-1]) > 0 and np.std(price_changes[1:]) > 0:
                                                corr = np.corrcoef(board_values[:-1], price_changes[1:])[0, 1]
                                                pattern[f'{board_feature}_lead_correlation'] = corr
                                            else:
                                                pattern[f'{board_feature}_lead_correlation'] = 0.0"""

content = re.sub(pattern2, replacement2, content)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修复np.corrcoef计算中可能的除零或NaN问题") 