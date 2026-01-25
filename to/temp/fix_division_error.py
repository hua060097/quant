#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 修复除零错误的代码
pattern = r'pattern\[\'regime_volume_trend\'\] = np\.mean\(segment_df\[\'volume\'\]\) / np\.median\(segment_df\[\'volume\'\]\)'
replacement = r'''# 安全计算成交量趋势，避免除以零和NaN值
                                median_volume = np.median(segment_df[\'volume\'])
                                mean_volume = np.mean(segment_df[\'volume\'])
                                if median_volume is not None and median_volume > 0 and not np.isnan(mean_volume) and not np.isnan(median_volume):
                                    pattern[\'regime_volume_trend\'] = mean_volume / median_volume
                                else:
                                    pattern[\'regime_volume_trend\'] = 1.0  # 默认值为1.0表示无变化'''

# 使用re.sub替换匹配的模式
content = re.sub(pattern, replacement, content)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修复成交量趋势计算中的除零错误") 