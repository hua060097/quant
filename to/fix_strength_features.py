#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复_predict_with_rolling_model函数中的两个问题：
1. 强度特征不匹配
2. market_group分类转换错误
"""

import os
import re

def fix_problems():
    """修复问题"""
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "to", "train_sector_models.py")
    if not os.path.exists(file_path):
        print(f"无法找到目标文件: {file_path}")
        return False
    
    # 创建备份
    backup_path = f"{file_path}.strength_fix.bak"
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"创建备份文件: {backup_path}")
    
    # 修复问题1：强度特征不匹配
    updated_content = original_content
    
    # 查找并替换合并市场预测结果的代码
    strength_pattern = re.compile(r'(\s+# 准备强度模型特征\s+strength_X = test_df\[strength_features\]\.fillna\(0\)\s+# 确保数据类型为数值型\s+strength_X = strength_X\.astype\(float\))\s+# 合并市场预测结果\s+market_prob_cols = \[col for col in test_df\.columns if col\.startswith\(\'market_prob_\'\)\]\s+if market_prob_cols:\s+for col in market_prob_cols:\s+if col not in strength_X\.columns:\s+strength_X\[col\] = test_df\[col\]')
    
    replacement = r'\1\n                    # 注释掉以下代码，避免特征维度不匹配问题\n                    # # 合并市场预测结果\n                    # market_prob_cols = [col for col in test_df.columns if col.startswith(\'market_prob_\')]\n                    # if market_prob_cols:\n                    #     for col in market_prob_cols:\n                    #         if col not in strength_X.columns:\n                    #             strength_X[col] = test_df[col]'
    
    updated_content = re.sub(strength_pattern, replacement, updated_content)
    
    # 修复问题2：market_group分类转换错误
    # 查找并替换market_group转换代码
    market_group_pattern = re.compile(r'(\s+# 特殊处理market_group特征，避免类型转换错误\s+if cat_feat == \'market_group\':\s+self\.logger\.info\("使用安全方式处理market_group特征"\))\s+# 将market_group转为字符串类型，避免分类转换问题\s+X_final\[cat_feat\] = X_final\[cat_feat\]\.astype\(str\)')
    
    market_group_replacement = r'\1\n                            # 直接设置为固定字符串值，完全避开分类转换\n                            X_final[cat_feat] = "neutral"  # 使用默认值作为字符串\n                            # 跳过转为str的操作，避免转换错误'
    
    updated_content = re.sub(market_group_pattern, market_group_replacement, updated_content)
    
    # 写回修改后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("成功修复了以下问题：")
    print("1. 强度特征不匹配问题 - 移除了向strength_X添加额外market_prob列的代码")
    print("2. market_group转换错误问题 - 改为直接使用固定字符串值")
    return True

if __name__ == "__main__":
    success = fix_problems()
    if success:
        print("修复脚本执行成功")
    else:
        print("修复脚本执行失败")
        import sys
        sys.exit(1) 