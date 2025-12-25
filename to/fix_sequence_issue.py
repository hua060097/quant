#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手动修复train_sector_models.py中的序列特征处理和索引越界问题
"""

def fix_train_model_file():
    # 读取原始文件
    filepath = "to/train_sector_models.py"
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 修复_train_rolling_day_model函数中的sequence.shape错误
    # 找到行 "if not sector_rows.empty and sequence.shape[0] >= seq_length:"
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 修复LNN训练过程中的sequence.shape错误
        if "for sector_code, sequence in sequence_train_data.items():" in line:
            new_lines.append(line)
            next_line = lines[i+1]
            new_lines.append(next_line)  # 添加sector_rows那一行
            
            # 添加sequence_feature变量的处理代码
            new_lines.append("                        # 处理sequence_data，根据新格式从中提取sequence矩阵\n")
            new_lines.append("                        sequence_feature = None\n")
            new_lines.append("                        if isinstance(sequence, dict) and 'features' in sequence:\n")
            new_lines.append("                            # 新格式：字典包含'features'键\n")
            new_lines.append("                            sequence_feature = sequence['features']\n")
            new_lines.append("                            self.logger.info(f\"使用新格式序列特征，板块 {sector_code}\")\n")
            new_lines.append("                        elif hasattr(sequence, 'shape'):\n")
            new_lines.append("                            # 旧格式：直接是特征矩阵\n")
            new_lines.append("                            sequence_feature = sequence\n")
            new_lines.append("                            self.logger.info(f\"使用旧格式序列特征，板块 {sector_code}\")\n")
            new_lines.append("                        else:\n")
            new_lines.append("                            self.logger.warning(f\"板块 {sector_code} 的序列特征格式无效，类型: {type(sequence)}\")\n")
            new_lines.append("                            continue\n\n")
            
            # 替换下一个if语句中的sequence为sequence_feature
            i += 2
            while i < len(lines) and "if not sector_rows.empty and sequence.shape[0] >= seq_length:" not in lines[i]:
                i += 1
                
            if i < len(lines):
                new_line = lines[i].replace("sequence.shape", "sequence_feature.shape")
                new_lines.append(new_line)
                i += 1
                
                # 替换下一行中的sequence为sequence_feature
                if i < len(lines) and "recent_sequence = sequence[-seq_length:]" in lines[i]:
                    new_line = lines[i].replace("sequence[-seq_length:]", "sequence_feature[-seq_length:]")
                    new_lines.append(new_line)
                    i += 1
                else:
                    new_lines.append(lines[i])
                    i += 1
            else:
                i += 1
                continue
        
        # 修复验证数据中的sequence.shape错误
        elif "for sector_code, sequence in sequence_val_data.items():" in line:
            new_lines.append(line)
            next_line = lines[i+1]
            new_lines.append(next_line)  # 添加sector_rows那一行
            
            # 添加sequence_feature变量的处理代码
            new_lines.append("                                    # 处理sequence_data，根据新格式从中提取sequence矩阵\n")
            new_lines.append("                                    sequence_feature = None\n")
            new_lines.append("                                    if isinstance(sequence, dict) and 'features' in sequence:\n")
            new_lines.append("                                        # 新格式：字典包含'features'键\n")
            new_lines.append("                                        sequence_feature = sequence['features']\n")
            new_lines.append("                                        self.logger.info(f\"使用新格式序列特征，板块 {sector_code}\")\n")
            new_lines.append("                                    elif hasattr(sequence, 'shape'):\n")
            new_lines.append("                                        # 旧格式：直接是特征矩阵\n")
            new_lines.append("                                        sequence_feature = sequence\n")
            new_lines.append("                                        self.logger.info(f\"使用旧格式序列特征，板块 {sector_code}\")\n")
            new_lines.append("                                    else:\n")
            new_lines.append("                                        self.logger.warning(f\"板块 {sector_code} 的序列特征格式无效，类型: {type(sequence)}\")\n")
            new_lines.append("                                        continue\n\n")
            
            # 替换下一个if语句中的sequence为sequence_feature
            i += 2
            while i < len(lines) and "if not sector_rows.empty and sequence.shape[0] >= seq_length:" not in lines[i]:
                i += 1
                
            if i < len(lines):
                new_line = lines[i].replace("sequence.shape", "sequence_feature.shape")
                new_lines.append(new_line)
                i += 1
                
                # 替换下一行中的sequence为sequence_feature
                if i < len(lines) and "recent_sequence = sequence[-seq_length:]" in lines[i]:
                    new_line = lines[i].replace("sequence[-seq_length:]", "sequence_feature[-seq_length:]")
                    new_lines.append(new_line)
                    i += 1
                else:
                    new_lines.append(lines[i])
                    i += 1
            else:
                i += 1
                continue
                
        # 修复idx超出范围错误
        elif "idx = test_df[test_df['sector_code'] == sector_code].index.get_loc(i)" in line:
            new_lines.append(line)
            next_line = lines[i+1]
            if "strength_score = strength_pred[idx]" in next_line:
                new_lines.append(next_line)
                new_lines.append("                            # 安全地获取索引，避免越界\n")
                new_lines.append("                            try:\n")
                new_lines.append("                                if idx < len(strength_pred):\n")
                new_lines.append("                                    strength_score = strength_pred[idx]\n")
                new_lines.append("                                else:\n")
                new_lines.append("                                    self.logger.warning(f\"索引 {idx} 超出strength_pred范围 {len(strength_pred)}\")\n")
                new_lines.append("                                    strength_score = 0.0\n")
                new_lines.append("                            except Exception as e:\n")
                new_lines.append("                                self.logger.warning(f\"获取强度预测时出错: {e}\")\n")
                new_lines.append("                                strength_score = 0.0\n")
                i += 2
            else:
                new_lines.append(next_line)
                i += 2
        else:
            new_lines.append(line)
            i += 1
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"文件修复完成: {filepath}")
    
if __name__ == "__main__":
    fix_train_model_file() 