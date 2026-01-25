#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 修复函数末尾的return语句
pattern = r'''        # 保存板块关联结果为单独的DataFrame
        relations_df = pd\.DataFrame\.from_dict\(results, orient='index'\)\.reset_index\(drop=True\) if results else pd\.DataFrame\(\)
        
        # 创建关联结果缓存文件
        if use_cache and cache_file:
            try:
                relations_cache_file = cache_file\.parent / f"relations_{cache_file\.name}"
                self\.logger\.info\(f"保存板块关联结果到缓存文件: {relations_cache_file}"\)
                with open\(relations_cache_file, 'wb'\) as f:
                    pickle\.dump\(relations_df, f\)
            except Exception as e:
                self\.logger\.warning\(f"保存关联结果缓存失败: {e}"\)
        
        return result_df, relations_df'''

replacement = r'''        # 如果前面的代码没有创建relations_df（比如在结果为空的情况下），则创建一个空的DataFrame
        if 'relations_df' not in locals():
            self.logger.info("没有找到有效的股票板块关联数据，创建空的relations_df")
            relations_df = pd.DataFrame()
        
        # 创建关联结果缓存文件
        if use_cache and cache_file:
            try:
                relations_cache_file = cache_file.parent / f"relations_{cache_file.name}"
                self.logger.info(f"保存板块关联结果到缓存文件: {relations_cache_file}")
                with open(relations_cache_file, 'wb') as f:
                    pickle.dump(relations_df, f)
            except Exception as e:
                self.logger.warning(f"保存关联结果缓存失败: {e}")
        
        return result_df, relations_df'''

# 使用DOTALL模式匹配多行文本
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修复calculate_stock_sector_relative_features方法的return语句") 