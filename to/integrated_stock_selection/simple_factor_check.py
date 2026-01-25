#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单检查因子列表的语法和基本正确性
"""

def check_main_factors():
    """简单检查 main.py 中的因子列表"""
    try:
        # 尝试导入 main.py 中的函数
        import sys
        sys.path.append('.')

        # 直接从文件中提取因子列表，避免导入问题
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找 _get_stock_factors 函数
        import re
        function_pattern = r'def _get_stock_factors\(.*?\):(.*?)(?=\ndef |\nclass |\Z)'
        match = re.search(function_pattern, content, re.DOTALL)

        if match:
            function_content = match.group(1)

            # 提取 all_factors 列表中的字符串
            string_pattern = r"'([^']+)'"
            factors = re.findall(string_pattern, function_content)

            print(f"从 main.py 中提取到 {len(factors)} 个因子:")
            for i, factor in enumerate(factors, 1):
                print(f"  {i:2d}. {factor}")

            return factors
        else:
            print("未找到 _get_stock_factors 函数")
            return []

    except Exception as e:
        print(f"检查失败: {e}")
        return []

def check_existing_factor_files():
    """检查实际存在的因子文件"""
    import os

    definitions_dir = 'factor_system/factor_lab/definitions'
    if not os.path.exists(definitions_dir):
        print(f"目录 {definitions_dir} 不存在")
        return []

    factor_files = []
    for file_name in os.listdir(definitions_dir):
        if file_name.endswith('.py') and file_name != '__init__.py':
            factor_files.append(file_name)

    print(f"发现 {len(factor_files)} 个因子文件:")
    for file_name in sorted(factor_files):
        print(f"  - {file_name}")

    return factor_files

def main():
    print("简单因子检查工具")
    print("=" * 40)

    # 检查 main.py 中的因子
    print("\n1. 检查 main.py 中的因子列表:")
    main_factors = check_main_factors()

    # 检查因子文件
    print("\n2. 检查实际存在的因子文件:")
    factor_files = check_existing_factor_files()

    # 基本统计
    print(f"\n3. 统计信息:")
    print(f"  main.py 中的因子数量: {len(main_factors)}")
    print(f"  实际因子文件数量: {len(factor_files)}")

    if main_factors:
        print(f"  main.py 中的前10个因子: {main_factors[:10]}")
        print(f"  main.py 中的后10个因子: {main_factors[-10:]}")

if __name__ == "__main__":
    main()