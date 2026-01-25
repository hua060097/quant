#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

# 当前文件路径
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_sector_models.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 修改在select_stocks_by_change_points方法中的调用
# 1. 修改第一个调用 - 训练数据
pattern1 = r'(        # 步骤5: 计算个股相对板块的特征（仅针对变化点）.*?\s+self\.logger\.info\("计算个股相对板块的特征（仅针对变化点）\.\.\."\).*?\s+)train_features = self\.calculate_stock_sector_relative_features\(.*?\)'
replacement1 = r'\1train_features, train_relations = self.calculate_stock_sector_relative_features(\n            train_features, sector_features_df, stock_change_points, is_training=True, use_cache=use_cache)'
content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)

# 2. 修改第二个调用 - 预测数据
pattern2 = r'(        # 对预测数据也进行相对特征计算（全部日期）.*?\s+self\.logger\.info\("计算预测数据的个股相对板块特征\.\.\."\).*?\s+)predict_features = self\.calculate_stock_sector_relative_features\(.*?\)'
replacement2 = r'\1predict_features, predict_relations = self.calculate_stock_sector_relative_features(\n            predict_features, sector_features_df, is_training=False, use_cache=use_cache)'
content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

# 3. 在分析阶段使用板块关联数据
pattern3 = r'(        # 步骤6: 分析变化点特征模式.*?\s+self\.logger\.info\("分析个股变化点特征模式\.\.\."\).*?\s+)stock_feature_patterns = self\._analyze_stock_change_point_patterns\(train_features, stock_change_points\)'
replacement3 = r'\1# 传递训练数据和板块关联结果\nstock_feature_patterns = self._analyze_stock_change_point_patterns(train_features, stock_change_points, relations_df=train_relations)'
content = re.sub(pattern3, replacement3, content, flags=re.DOTALL)

# 4. 在预测阶段使用板块关联数据  
pattern4 = r'(        # 步骤7: 预测个股表现.*?\s+self\.logger\.info\("预测个股表现\.\.\."\).*?\s+)stock_predictions = self\._predict_stocks_with_change_points\(predict_features, stock_feature_patterns\)'
replacement4 = r'\1# 传递预测数据和板块关联结果\nstock_predictions = self._predict_stocks_with_change_points(predict_features, stock_feature_patterns, relations_df=predict_relations)'
content = re.sub(pattern4, replacement4, content, flags=re.DOTALL)

# 5. 修改_analyze_stock_change_point_patterns方法接受relations_df参数
pattern5 = r'(    def _analyze_stock_change_point_patterns\(self, stock_features_df, change_point_results\):.*?\s+""".*?)\s+Args:\s+(.*?)\s+Returns:'
replacement5 = r'\1\n        Args:\n            stock_features_df: 个股特征DataFrame\n            change_point_results: 变化点检测结果\n            relations_df: 股票板块关联结果DataFrame，可选参数\n            \n        Returns:'
content = re.sub(pattern5, replacement5, content, flags=re.DOTALL)

pattern6 = r'    def _analyze_stock_change_point_patterns\(self, stock_features_df, change_point_results\):'
replacement6 = r'    def _analyze_stock_change_point_patterns(self, stock_features_df, change_point_results, relations_df=None):'
content = re.sub(pattern6, replacement6, content)

# 6. 修改_predict_stocks_with_change_points方法接受relations_df参数
pattern7 = r'(    def _predict_stocks_with_change_points\(self, predict_features, feature_patterns\):.*?\s+""".*?)\s+Args:\s+(.*?)\s+Returns:'
replacement7 = r'\1\n        Args:\n            predict_features: 预测期特征数据\n            feature_patterns: 历史特征模式数据\n            relations_df: 股票板块关联结果DataFrame，可选参数\n            \n        Returns:'
content = re.sub(pattern7, replacement7, content, flags=re.DOTALL)

pattern8 = r'    def _predict_stocks_with_change_points\(self, predict_features, feature_patterns\):'
replacement8 = r'    def _predict_stocks_with_change_points(self, predict_features, feature_patterns, relations_df=None):'
content = re.sub(pattern8, replacement8, content)

# 写入文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("成功修改调用calculate_stock_sector_relative_features方法的代码") 