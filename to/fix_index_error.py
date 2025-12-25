#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复train_sector_models.py中的索引越界问题
"""

def fix_index_out_of_bounds():
    # 读取原始文件
    filepath = "to/train_sector_models.py"
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到并修复以下模式的代码:
    # idx = test_df[test_df['sector_code'] == sector_code].index.get_loc(i)
    # strength_score = strength_pred[idx]
    target_code = "idx = test_df[test_df['sector_code'] == sector_code].index.get_loc(i)\n                            strength_score = strength_pred[idx]"
    
    replacement_code = """idx = test_df[test_df['sector_code'] == sector_code].index.get_loc(i)
                            # 安全地获取索引，避免越界
                            try:
                                if idx < len(strength_pred):
                                    strength_score = strength_pred[idx]
                                else:
                                    self.logger.warning(f"索引 {idx} 超出strength_pred范围 {len(strength_pred)}")
                                    strength_score = 0.0
                            except Exception as e:
                                self.logger.warning(f"获取强度预测时出错: {e}")
                                strength_score = 0.0"""
    
    # 替换代码
    new_content = content.replace(target_code, replacement_code)
    
    # 处理else分支中的相同问题
    target_code2 = "idx = test_df[test_df['sector_code'] == sector_code].index.get_loc(i)\n                        strength_score = strength_pred[idx]"
    
    replacement_code2 = """idx = test_df[test_df['sector_code'] == sector_code].index.get_loc(i)
                        # 安全地获取索引，避免越界
                        try:
                            if idx < len(strength_pred):
                                strength_score = strength_pred[idx]
                            else:
                                self.logger.warning(f"索引 {idx} 超出strength_pred范围 {len(strength_pred)}")
                                strength_score = 0.0
                        except Exception as e:
                            self.logger.warning(f"获取强度预测时出错: {e}")
                            strength_score = 0.0"""
    
    # 替换代码
    new_content = new_content.replace(target_code2, replacement_code2)
    
    # 处理LNN预测过程中可能的索引越界问题
    lnn_target = "sequence = sequence_data[-seq_length:]"
    lnn_replacement = """# 安全地获取序列数据，避免越界
                                try:
                                    sequence = sequence_data[-seq_length:]
                                except Exception as e:
                                    self.logger.error(f"获取序列数据时出错: {e}")
                                    continue"""
    
    # 替换LNN预测代码
    new_content = new_content.replace(lnn_target, lnn_replacement)
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"索引越界问题修复完成: {filepath}")

if __name__ == "__main__":
    fix_index_out_of_bounds() 