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

# 修改缓存加载逻辑
pattern = r'(                # 如果缓存存在，直接加载并返回.*?\s+if cache_file\.exists\(\):.*?\s+)self\.logger\.info\(f"从缓存加载相对特征: {cache_file}"\).*?\s+try:.*?\s+with open\(cache_file, \'rb\'\) as f:.*?\s+return pickle\.load\(f\).*?\s+except Exception as e:.*?\s+self\.logger\.warning\(f"加载缓存失败: {e}，将重新计算特征"\)'
replacement = r'\1self.logger.info(f"从缓存加载相对特征: {cache_file}")\n                    try:\n                        # 加载特征缓存\n                        with open(cache_file, \'rb\') as f:\n                            result_df = pickle.load(f)\n                            \n                        # 尝试加载关系数据缓存\n                        relations_cache_file = cache_file.parent / f"relations_{cache_file.name}"\n                        relations_df = pd.DataFrame()\n                        \n                        if relations_cache_file.exists():\n                            self.logger.info(f"从缓存加载板块关联结果: {relations_cache_file}")\n                            try:\n                                with open(relations_cache_file, \'rb\') as f:\n                                    relations_df = pickle.load(f)\n                                self.logger.info(f"成功加载板块关联结果缓存，包含 {len(relations_df)} 条记录")\n                            except Exception as e:\n                                self.logger.warning(f"加载板块关联结果缓存失败: {e}，将使用空DataFrame")\n                                relations_df = pd.DataFrame()\n                        \n                        return result_df, relations_df\n                    except Exception as e:\n                        self.logger.warning(f"加载缓存失败: {e}，将重新计算特征")'
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# 修改空数据返回处理
pattern2 = r'(        if stock_features_df\.empty or sector_features_df\.empty:.*?\s+)return stock_features_df, pd\.DataFrame\(\)'
replacement2 = r'\1self.logger.warning("特征数据为空，返回空DataFrame")\n            return stock_features_df, pd.DataFrame()'
content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修改calculate_stock_sector_relative_features方法的缓存加载逻辑") 