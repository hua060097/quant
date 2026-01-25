#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成 __init__.py 中的 __all__ 列表
基于导入语句自动生成，避免手动维护不一致
"""

import re
import os

def parse_imports_from_init():
    """解析 __init__.py 中的导入语句，提取所有因子名称"""
    init_file = 'factor_system/factor_lab/definitions/__init__.py'

    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()

    imported_factors = []

    # 按行处理，处理多行导入
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 跳过注释行
        if line.startswith('#') or not line:
            i += 1
            continue

        # 单行导入
        if line.startswith('from .') and 'import' in line and '(' not in line:
            # from .moving_average import ma_ratio_5_10, ma_convergence_score  # noqa: F401
            if '#' in line:
                line = line.split('#')[0].strip()

            if 'import' in line:
                imports_part = line.split('import')[1].strip()
                factors = [f.strip() for f in imports_part.split(',') if f.strip()]
                # 过滤掉包含空格或特殊字符的无效因子名
                factors = [f for f in factors if f and f.replace('_', '').replace('-', '').isalnum() or '_' in f]
                imported_factors.extend(factors)

        # 多行导入开始
        elif line.startswith('from .') and 'import' in line and '(' in line:
            # from .candlestick import (
            # 收集多行导入的因子
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()

                # 结束符
                if next_line == ')':
                    i += 1
                    break

                # 跳过空行和注释
                if not next_line or next_line.startswith('#'):
                    i += 1
                    continue

                # 移除行内注释
                if '#' in next_line:
                    next_line = next_line.split('#')[0].strip()

                # 提取因子名称
                factors = [f.strip() for f in next_line.split(',') if f.strip()]
                # 过滤掉包含空格或特殊字符的无效因子名
                factors = [f for f in factors if f and (f.replace('_', '').replace('-', '').isalnum() or '_' in f)]
                imported_factors.extend(factors)

                i += 1

        i += 1

    # 去重并排序
    imported_factors = list(set(imported_factors))
    imported_factors.sort()

    return imported_factors

def generate_new_all_content():
    """生成新的 __all__ 定义"""
    factors = parse_imports_from_init()

    # 生成 __all__ 列表
    all_lines = ["__all__ = ["]

    for i, factor in enumerate(factors):
        comma = "," if i < len(factors) - 1 else ""
        all_lines.append(f"    '{factor}'{comma}")

    all_lines.append("]")

    return '\n'.join(all_lines), len(factors)

def update_init_file():
    """更新 __init__.py 文件"""
    init_file = 'factor_system/factor_lab/definitions/__init__.py'

    # 读取原文件
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 生成新的 __all__ 定义
    new_all_content, factor_count = generate_new_all_content()

    # 查找并替换原有的 __all__ 定义
    if '__all__' in content:
        # 找到 __all__ 的开始和结束
        start = content.find('__all__ = [')
        if start != -1:
            # 找到对应的结束括号
            bracket_count = 0
            pos = start
            while pos < len(content):
                if content[pos] == '[':
                    bracket_count += 1
                elif content[pos] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end = pos + 1
                        break
                pos += 1

            if bracket_count == 0:
                # 替换原有的 __all__ 定义
                new_content = content[:start] + new_all_content + content[end:]
            else:
                print(f"[ERROR] 无法找到 __all__ 的结束括号")
                return False
        else:
            print(f"[ERROR] 无法找到 __all__ 的开始位置")
            return False
    else:
        # 在文件末尾添加 __all__ 定义
        new_content = content + '\n\n' + new_all_content

    # 备份原文件
    backup_file = init_file + '.backup'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已备份原文件到: {backup_file}")

    # 写入新内容
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"已更新 __all__，包含 {factor_count} 个因子")
    return True

def check_consistency():
    """检查导入语句与当前 __all__ 的一致性"""
    init_file = 'factor_system/factor_lab/definitions/__init__.py'

    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 获取导入的因子
    imported_factors = set(parse_imports_from_init())

    # 获取 __all__ 中的因子
    all_factors = set()
    if '__all__' in content:
        start = content.find('__all__ = [')
        if start != -1:
            # 找到对应的结束括号
            bracket_count = 0
            pos = start
            while pos < len(content):
                if content[pos] == '[':
                    bracket_count += 1
                elif content[pos] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end = pos + 1
                        break
                pos += 1

            if bracket_count == 0:
                all_definition = content[start:end]
                # 提取字符串内容
                import ast
                try:
                    all_list = ast.literal_eval(all_definition.split('=')[1].strip())
                    all_factors = set(all_list)
                except Exception as e:
                    print(f"[ERROR] 解析 __all__ 失败: {e}")
                    return False

    # 比较差异
    only_in_imports = imported_factors - all_factors
    only_in_all = all_factors - imported_factors

    if only_in_imports:
        print(f"在导入中但不在 __all__ 中 ({len(only_in_imports)} 个):")
        for factor in sorted(only_in_imports):
            print(f"  - {factor}")

    if only_in_all:
        print(f"在 __all__ 中但不在导入中 ({len(only_in_all)} 个):")
        for factor in sorted(only_in_all):
            print(f"  - {factor}")

    if not only_in_imports and not only_in_all:
        print("✅ 导入语句与 __all__ 完全一致")
        return True
    else:
        print("❌ 导入语句与 __all__ 不一致")
        return False

def main():
    """主函数"""
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'check':
            print("检查导入语句与 __all__ 的一致性:")
            check_consistency()
        elif command == 'generate':
            print("自动生成 __all__:")
            update_init_file()
        elif command == 'demo':
            print("演示: 显示将要生成的 __all__ 内容:")
            new_all_content, factor_count = generate_new_all_content()
            print(f"\n因子数量: {factor_count}")
            print(f"生成的 __all__:\n{new_all_content}")
        else:
            print("用法:")
            print("  python auto_generate_all.py check     - 检查一致性")
            print("  python auto_generate_all.py generate  - 自动生成")
            print("  python auto_generate_all.py demo      - 演示生成内容")
    else:
        print("因子系统 __all__ 自动生成工具")
        print("\n建议的操作流程:")
        print("1. python auto_generate_all.py check      # 检查当前状态")
        print("2. python auto_generate_all.py demo       # 查看将要生成的内容")
        print("3. python auto_generate_all.py generate   # 执行自动生成")
        print("\n或者直接运行删除 __all__，因为代码中没有使用 import *")

if __name__ == "__main__":
    main()