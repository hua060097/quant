#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def read_log_range(file_path, start_line, end_line):
    """读取指定行数范围的日志内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 提取指定范围的行（注意Python索引从0开始）
    selected_lines = lines[start_line-1:end_line]
    
    print(f"读取日志文件: {file_path}")
    print(f"行数范围: {start_line} - {end_line}")
    print(f"实际读取行数: {len(selected_lines)}")
    print("=" * 80)
    
    # 输出内容
    for i, line in enumerate(selected_lines, start=start_line):
        print(f"{i}→{line.rstrip()}")

if __name__ == "__main__":
    log_file = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\stock_selection.log"
    start = 1253391
    end = 1255277
    
    try:
        read_log_range(log_file, start, end)
    except Exception as e:
        print(f"读取日志失败: {e}")