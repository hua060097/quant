#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复to/train_sector_models.py中的错误
1. 'dict' object has no attribute 'shape'错误
2. index out of bounds错误
"""

import os
import re
import sys

def fix_sequence_shape_error():
    """修复sequence.shape错误，处理sequence可能是字典的情况"""
    filename = "to/train_sector_models.py"
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复LNN训练过程中的sequence.shape错误
    pattern = r"(for sector_code, sequence in sequence_train_data\.items\(\):\s+sector_rows = train_df_split\[train_df_split\['sector_code'\] == sector_code\]\s+)(\s+if not sector_rows\.empty and sequence\.shape\[0\] >= seq_length:)"
    replacement = r"\1        # 处理sequence_data，根据新格式从中提取sequence矩阵\n        sequence_feature = None\n        if isinstance(sequence, dict) and 'features' in sequence:\n            # 新格式：字典包含'features'键\n            sequence_feature = sequence['features']\n            print(f\"使用新格式序列特征，板块 {sector_code}\")\n        elif hasattr(sequence, 'shape'):\n            # 旧格式：直接是特征矩阵\n            sequence_feature = sequence\n            print(f\"使用旧格式序列特征，板块 {sector_code}\")\n        else:\n            print(f\"板块 {sector_code} 的序列特征格式无效，类型: {type(sequence)}\")\n            continue\n\n\2"
    
    # 替换原始代码中的模式
    modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # 修复验证数据中的sequence.shape错误
    pattern2 = r"(for sector_code, sequence in sequence_val_data\.items\(\):\s+sector_rows = val_df\[val_df\['sector_code'\] == sector_code\]\s+)(\s+if not sector_rows\.empty and sequence\.shape\[0\] >= seq_length:)"
    replacement2 = r"\1        # 处理sequence_data，根据新格式从中提取sequence矩阵\n        sequence_feature = None\n        if isinstance(sequence, dict) and 'features' in sequence:\n            # 新格式：字典包含'features'键\n            sequence_feature = sequence['features']\n            print(f\"使用新格式序列特征，板块 {sector_code}\")\n        elif hasattr(sequence, 'shape'):\n            # 旧格式：直接是特征矩阵\n            sequence_feature = sequence\n            print(f\"使用旧格式序列特征，板块 {sector_code}\")\n        else:\n            print(f\"板块 {sector_code} 的序列特征格式无效，类型: {type(sequence)}\")\n            continue\n\n\2"
    
    modified_content = re.sub(pattern2, replacement2, modified_content, flags=re.DOTALL)
    
    # 修复_predict_with_sequence中的索引越界错误
    pattern3 = r"(for i, row in test_df\.iterrows\(\):\s+sector_code = row\['sector_code'\]\s+# 获取强度模型预测\s+if strength_pred is not None:\s+# 找到当前板块在strength_pred中的索引\s+idx = test_df\[test_df\['sector_code'\] == sector_code\]\.index\.get_loc\(i\)\s+strength_score = strength_pred\[idx\])"
    replacement3 = r"\1\n            # 安全地获取索引，避免越界\n            try:\n                if idx < len(strength_pred):\n                    strength_score = strength_pred[idx]\n                else:\n                    self.logger.warning(f\"索引 {idx} 超出strength_pred范围 {len(strength_pred)}\")\n                    strength_score = 0.0\n            except Exception as e:\n                self.logger.warning(f\"获取强度预测时出错: {e}\")\n                strength_score = 0.0"
    
    modified_content = re.sub(pattern3, replacement3, modified_content, flags=re.DOTALL)
    
    # 写回文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"文件 {filename} 已修复")

if __name__ == "__main__":
    fix_sequence_shape_error()
    print("修复完成！") 