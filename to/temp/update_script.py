#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import pickle
import pandas as pd

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 1. 修改函数返回值注释
pattern1 = r'(Returns:\s+)添加了相对特征的DataFrame'
replacement1 = r'\1tuple: (添加了相对特征的DataFrame, 股票板块关联结果DataFrame)'
content = re.sub(pattern1, replacement1, content)

# 2. 修改函数空数据返回处理
pattern2 = r'if stock_features_df\.empty or sector_features_df\.empty:\s+return stock_features_df'
replacement2 = 'if stock_features_df.empty or sector_features_df.empty:\n            return stock_features_df, pd.DataFrame()'
content = re.sub(pattern2, replacement2, content)

# 3. 添加结果缓存和修改返回值
pattern3 = r'(        # 保存计算结果到缓存\s+if use_cache and cache_file:\s+try:.*?\s+except Exception as e:\s+self\.logger\.warning\(f"保存缓存失败: {e}"\)\s+\s+)return result_df'
replacement3 = r'\1        # 保存板块关联结果为单独的DataFrame\n        relations_df = pd.DataFrame.from_dict(results, orient=\'index\').reset_index(drop=True) if results else pd.DataFrame()\n        \n        # 创建关联结果缓存文件\n        if use_cache and cache_file:\n            try:\n                relations_cache_file = cache_file.parent / f"relations_{cache_file.name}"\n                self.logger.info(f"保存板块关联结果到缓存文件: {relations_cache_file}")\n                with open(relations_cache_file, \'wb\') as f:\n                    pickle.dump(relations_df, f)\n            except Exception as e:\n                self.logger.warning(f"保存关联结果缓存失败: {e}")\n        \n        return result_df, relations_df'
content = re.sub(pattern3, replacement3, content, flags=re.DOTALL)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修改 calculate_stock_sector_relative_features 方法") 