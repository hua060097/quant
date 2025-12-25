#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手动修复train_sector_models.py中的缩进和变量引用问题
"""

def fix_indentation_and_vars():
    # 读取原始文件
    filepath = "to/train_sector_models.py"
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复缩进和变量引用问题
    # 修改 " if not sector_rows.empty and sequence_feature.shape[0] >= seq_length:" 行的缩进
    content = content.replace(" if not sector_rows.empty and sequence_feature.shape[0] >= seq_length:", "                        if not sector_rows.empty and sequence_feature.shape[0] >= seq_length:")
    
    # 修复 recent_sequence = sequence[-seq_length:] 为 recent_sequence = sequence_feature[-seq_length:]
    content = content.replace("                            recent_sequence = sequence[-seq_length:]", "                            recent_sequence = sequence_feature[-seq_length:]")
    
    # 修复验证数据部分的缩进和变量引用
    content = content.replace(" if not sector_rows.empty and sequence_feature.shape[0] >= seq_length:", "                                    if not sector_rows.empty and sequence_feature.shape[0] >= seq_length:")
    content = content.replace("                                        recent_sequence = sequence[-seq_length:]", "                                        recent_sequence = sequence_feature[-seq_length:]")
    
    # 修复try-except语句中的缺少except子句问题
    # 添加exception处理到函数中的try语句
    content = content.replace("        try:\n            # 获取模型组件并进行健壮性检查", 
                             "        try:\n            # 获取模型组件并进行健壮性检查\n        except Exception as e:\n            self.logger.error(f\"预测时出错: {e}\")\n            self.logger.error(traceback.format_exc())\n            return None\n\n        try:")
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"文件修复完成: {filepath}")

if __name__ == "__main__":
    fix_indentation_and_vars() 