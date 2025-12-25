#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接修复LNN特征生成代码中的缩进问题

这个脚本采用直接修改的方式，而非正则表达式，避免复杂的匹配和替换导致的问题
"""

import os
import sys
import re

def fix_train_sector_models():
    """
    直接修改文件，解决try/except缺失和缩进问题
    """
    
    # 文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(script_dir, "train_sector_models.py")
    
    # 读取原文件
    with open(target_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 保存备份
    with open(target_file + '.bak', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    # 修复第3744行左右的try语句问题
    try_start_line = 3743  # 近似行号
    for i in range(try_start_line-10, try_start_line+10):
        if i < len(lines) and "try:" in lines[i] and "# 在训练集上的预测" in lines[i+1]:
            # 找到try语句开始位置
            try_start = i
            # 找到相应的except语句位置
            for j in range(try_start+1, len(lines)):
                if "except Exception" in lines[j]:
                    # 已经有except子句，不需要修复
                    break
                if j - try_start > 20:  # 如果超过20行还没找到except，则需要添加
                    # 找到try语句块结束位置（根据缩进判断）
                    try_indent = len(lines[try_start]) - len(lines[try_start].lstrip())
                    for k in range(try_start+1, len(lines)):
                        line_indent = len(lines[k]) - len(lines[k].lstrip())
                        if line_indent <= try_indent and k > try_start + 2:  # 至少要有2行内容
                            # 在这里插入except子句
                            except_line = ' ' * try_indent + 'except Exception as e_metrics_train:\n'
                            except_block = [
                                except_line,
                                ' ' * (try_indent + 4) + 'self.logger.warning(f"计算训练集指标时出错: {e_metrics_train}")\n',
                                ' ' * (try_indent + 4) + 'metrics = {\n',
                                ' ' * (try_indent + 8) + "'train_error': str(e_metrics_train),\n",
                                ' ' * (try_indent + 8) + "'train_date': train_date\n",
                                ' ' * (try_indent + 4) + '}\n'
                            ]
                            lines[k:k] = except_block
                            break
                    break
    
    # 修复第3761行左右的try语句问题
    try_start_line = 3760  # 近似行号
    for i in range(try_start_line-10, try_start_line+10):
        if i < len(lines) and "try:" in lines[i] and "# 首先，尝试为验证集生成LNN特征" in lines[i+1]:
            # 找到try语句开始位置
            try_start = i
            # 检查是否已有对应的except子句
            has_except = False
            try_indent = len(lines[try_start]) - len(lines[try_start].lstrip())
            for j in range(try_start+1, len(lines)):
                if "except Exception" in lines[j] and len(lines[j]) - len(lines[j].lstrip()) == try_indent:
                    has_except = True
                    break
                
                # 检查是否已经到了if块的结束位置
                if j < len(lines) and "if not val_df.empty" in lines[j-10:j] and len(lines[j]) - len(lines[j].lstrip()) < try_indent:
                    # 在这里需要插入except子句
                    except_line = ' ' * try_indent + 'except Exception as e_val:\n'
                    except_block = [
                        except_line,
                        ' ' * (try_indent + 4) + 'self.logger.warning(f"验证集评估出错: {e_val}")\n',
                        ' ' * (try_indent + 4) + 'self.logger.warning(traceback.format_exc())\n',
                        ' ' * (try_indent + 4) + "# 添加错误信息到指标\n",
                        ' ' * (try_indent + 4) + "metrics['val_error'] = str(e_val)\n"
                    ]
                    lines[j:j] = except_block
                    break
    
    # 修复第3922行左右的else语句和缩进问题
    else_start_line = 3921  # 近似行号
    for i in range(else_start_line-10, else_start_line+10):
        if i < len(lines) and "else:" in lines[i].strip() and "# 如果没有特征列信息" in lines[i+1]:
            # 检查前面是否有匹配的if语句
            else_line = i
            else_indent = len(lines[i]) - len(lines[i].lstrip())
            # 向上查找匹配的if语句
            for j in range(else_line-1, max(0, else_line-50), -1):
                if "if " in lines[j] and len(lines[j]) - len(lines[j].lstrip()) == else_indent:
                    # 找到了匹配的if，不需要修复
                    break
                if j == max(0, else_line-50) + 1:
                    # 未找到匹配的if，需要将else改为if
                    lines[else_line] = lines[else_line].replace("else:", "if 'final_feature_columns' not in model_metadata or not model_metadata['final_feature_columns']:")
    
    # 修复LNN模型创建序列特征部分的else语句
    lnn_model_line = 3950  # 近似行号
    for i in range(lnn_model_line-30, lnn_model_line+30):
        if i < len(lines) and "if lnn_model is not None and model_package.get('seq_length', 0) > 0:" in lines[i]:
            lnn_if_line = i
            lnn_if_indent = len(lines[i]) - len(lines[i].lstrip())
            # 查找错误的else缩进
            for j in range(lnn_if_line+1, min(len(lines), lnn_if_line+30)):
                if "else:" in lines[j]:
                    if len(lines[j]) - len(lines[j].lstrip()) != lnn_if_indent:
                        # 不正确的else缩进，修复它
                        lines[j] = ' ' * lnn_if_indent + "else:\n"
                    break
    
    # 修复强度模型的else语句
    strength_model_line = 4000  # 近似行号
    for i in range(strength_model_line-30, strength_model_line+30):
        if i < len(lines) and "if strength_model is not None and strength_features:" in lines[i]:
            strength_if_line = i
            strength_if_indent = len(lines[i]) - len(lines[i].lstrip())
            # 查找错误的else缩进
            for j in range(strength_if_line+1, min(len(lines), strength_if_line+50)):
                if "else:" in lines[j] and "strength_model不可用" in lines[j+1]:
                    if len(lines[j]) - len(lines[j].lstrip()) != strength_if_indent:
                        # 不正确的else缩进，修复它
                        lines[j] = ' ' * strength_if_indent + "else:\n"
                    break
    
    # 写入修复后的文件
    with open(target_file, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print(f"成功修复了train_sector_models.py文件中的问题，原文件已备份为 {target_file}.bak")
    return True

if __name__ == "__main__":
    if fix_train_sector_models():
        print("修复完成!")
    else:
        print("修复失败!") 