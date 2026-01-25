#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 修改耗时的for循环更新操作，直接将update_map中的内容保存到relations_df
pattern = r'''(            try:
                results_df = pd\.DataFrame\.from_dict\(results, orient='index'\)
                
                # 重建索引便于合并
                results_df = results_df\.reset_index\(drop=True\)
                
                self\.logger\.info\(f"更新索引完成"\)
                # 创建一个映射用于更新原始DataFrame
                update_map = {.*?}
                
)                # 批量更新原始DataFrame以提高性能
                update_count = 0
                for \(stock_code, date_str\), values in update_map\.items\(\):
                    try:
                        # 使用布尔索引，避免链式索引
                        mask = \(result_df\['stock_code'\] == stock_code\) & \(result_df\['date_str'\] == date_str\)
                        
                        # 只在有匹配的行时更新
                        if mask\.any\(\):
                            for col, val in values\.items\(\):
                                result_df\.loc\[mask, col\] = val
                            update_count \+= 1
                            self\.logger\.info\(f"更新股票 {stock_code} 在 {date_str} 的特征数据"\)
                    except Exception as e:
                        self\.logger\.error\(f"更新结果DataFrame时出错: {str\(e\)}"\)
                
                self\.logger\.info\(f"已更新 {update_count} 个股票\-日期组合的特征数据"\)'''

replacement = r'''\1                # 不再将update_map的数据合并到result_df中，避免耗时操作
                # 将update_map保存到relations_df中，供后续方法直接使用
                self.logger.info("跳过耗时的DataFrame更新操作，直接使用relations_df进行查询")
                
                # 为了保持与原始数据相似的结构，我们将update_map转换为DataFrame
                relations_rows = []
                for (stock_code, date_str), values in update_map.items():
                    row = {'stock_code': stock_code, 'date': date_str}
                    row.update(values)
                    relations_rows.append(row)
                
                # 创建relations_df
                relations_df = pd.DataFrame(relations_rows)
                self.logger.info(f"创建板块关联DataFrame，包含 {len(relations_df)} 条记录")'''

# 使用DOTALL模式匹配多行文本
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功优化calculate_stock_sector_relative_features方法，去掉耗时的for循环更新操作") 