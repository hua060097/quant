#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复_predict_with_rolling_model函数中的返回值问题
"""
import os
import re
import sys

def fix_predict_with_rolling_model():
    """修复_predict_with_rolling_model函数中的返回值问题"""
    
    # 找到目标文件
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "to", "train_sector_models.py")
    if not os.path.exists(file_path):
        print(f"无法找到目标文件: {file_path}")
        return False
    
    # 创建备份
    backup_path = f"{file_path}.rollbak"
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"创建备份文件: {backup_path}")
    
    # 使用简单的替换方法，基于行号
    lines = original_content.split('\n')
    
    # 查找_predict_with_rolling_model函数
    start_line = -1
    for i, line in enumerate(lines):
        if "def _predict_with_rolling_model" in line:
            start_line = i
            break
    
    if start_line == -1:
        print("无法找到_predict_with_rolling_model函数")
        return False
    
    # 查找函数中返回model_package的部分
    return_model_package_index = -1
    for i in range(start_line, len(lines)):
        if "return model_package" in lines[i]:
            return_model_package_index = i
            break
    
    if return_model_package_index == -1:
        print("无法找到return model_package语句")
        return False
    
    # 查找创建prediction_results的行
    prediction_results_index = -1
    for i in range(start_line, return_model_package_index):
        if "prediction_results = {" in lines[i]:
            prediction_results_index = i
            break
    
    if prediction_results_index == -1:
        print("无法找到prediction_results定义")
        return False
    
    # 查找创建model_package的起始行
    model_package_start = -1
    for i in range(prediction_results_index, return_model_package_index):
        if "model_package = {" in lines[i]:
            model_package_start = i
            break
    
    if model_package_start == -1:
        print("无法找到model_package定义")
        return False
    
    # 删除model_package定义和元数据部分，直接返回prediction_results
    lines[model_package_start:return_model_package_index+1] = ['            return prediction_results']
    
    # 写回修改后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print("成功修复_predict_with_rolling_model函数的返回值问题")
    return True

if __name__ == "__main__":
    success = fix_predict_with_rolling_model()
    if success:
        print("修复脚本执行成功")
    else:
        print("修复脚本执行失败")
        sys.exit(1) 