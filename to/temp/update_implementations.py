#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 1. 修改_analyze_stock_change_point_patterns方法的实现来使用relations_df
pattern1 = r'(    def _analyze_stock_change_point_patterns\(self, stock_features_df, change_point_results, relations_df=None\):.*?\s+)key_sector_features = \[(.*?)\]'
replacement1 = r'\1# 检查是否提供了股票板块关联数据\n        if relations_df is not None and not relations_df.empty:\n            self.logger.info(f"使用单独的股票板块关联数据(包含{len(relations_df)}条记录)进行分析")\n        \n        key_sector_features = [\2]'
content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)

# 2. 修改_predict_stocks_with_change_points方法的实现来使用relations_df
pattern2 = r'(    def _predict_stocks_with_change_points\(self, predict_features, feature_patterns, relations_df=None\):.*?\s+)self\.logger\.info\(f"开始预测 {len\(predict_features\)} 只股票的表现\.\.\."\)'
replacement2 = r'\1self.logger.info(f"开始预测 {len(predict_features)} 只股票的表现...")\n        \n        # 检查是否提供了股票板块关联数据\n        if relations_df is not None and not relations_df.empty:\n            self.logger.info(f"使用单独的股票板块关联数据(包含{len(relations_df)}条记录)进行预测")'
content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

# 3. 在_analyze_stock_change_point_patterns中使用relations_df
pattern3 = r'(                    # 创建一个pattern字典，包含股票代码、日期和所有可用的关键特征.*?\s+)pattern = {.*?\s+\'stock_code\': stock_code,.*?\s+\'date\': cp_date,.*?\s+\'future_return_2d\': cp_return.*?\s+}'
replacement3 = r'\1pattern = {\n                        \'stock_code\': stock_code,\n                        \'date\': cp_date,\n                        \'future_return_2d\': cp_return\n                    }\n                    \n                    # 如果有单独的板块关联数据，从中获取相关信息\n                    if relations_df is not None and not relations_df.empty:\n                        # 查找对应的板块关联数据\n                        rel_data = relations_df[(relations_df[\'stock_code\'] == stock_code) & (relations_df[\'date\'] == cp_date)]\n                        if not rel_data.empty:\n                            # 添加板块关联信息\n                            for col in rel_data.columns:\n                                if col not in [\'stock_code\', \'date\']:\n                                    pattern[f\'sector_{col}\'] = rel_data.iloc[0][col]'
content = re.sub(pattern3, replacement3, content, flags=re.DOTALL)

# 4. 在_predict_stocks_with_change_points中使用relations_df
pattern4 = r'(                    # 提取股票特征.*?\s+)stock_features = {}.*?\s+for col in stock_row\.index:.*?\s+if col not in \[\'date\', \'stock_code\', \'stock_name\'\]:.*?\s+# 过滤非数值特征.*?\s+try:.*?\s+value = float\(stock_row\[col\]\).*?\s+stock_features\[col\] = value.*?\s+except \(ValueError, TypeError\):.*?\s+pass'
replacement4 = r'\1stock_features = {}\n            for col in stock_row.index:\n                if col not in [\'date\', \'stock_code\', \'stock_name\']:\n                    # 过滤非数值特征\n                    try:\n                        value = float(stock_row[col])\n                        stock_features[col] = value\n                    except (ValueError, TypeError):\n                        pass\n            \n            # 如果有单独的板块关联数据，从中获取相关信息\n            if relations_df is not None and not relations_df.empty:\n                # 查找对应的板块关联数据\n                rel_data = relations_df[(relations_df[\'stock_code\'] == stock_code) & (relations_df[\'date\'] == latest_date)]\n                if not rel_data.empty:\n                    # 添加板块关联信息\n                    for col in rel_data.columns:\n                        if col not in [\'stock_code\', \'date\']:\n                            stock_features[f\'sector_{col}\'] = rel_data.iloc[0][col]\n                            self.logger.debug(f"添加板块关联特征 {col} = {rel_data.iloc[0][col]}")'
content = re.sub(pattern4, replacement4, content, flags=re.DOTALL)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修改_analyze_stock_change_point_patterns和_predict_stocks_with_change_points方法的实现") 