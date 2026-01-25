#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单方案：直接删除 __all__，因为代码中没有使用 import *
"""

def remove_all_from_init():
    """直接从 __init__.py 中删除 __all__ 定义"""
    init_file = 'factor_system/factor_lab/definitions/__init__.py'

    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()

    if '__all__' not in content:
        print("__init__.py 中没有 __all__ 定义")
        return True

    # 备份原文件
    backup_file = init_file + '.backup_before_all_removal'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已备份原文件到: {backup_file}")

    # 删除 __all__ 定义
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
            # 删除 __all__ 定义，包括前后的空行
            # 找到 __all__ 之前的换行符
            line_start = start
            while line_start > 0 and content[line_start-1] != '\n':
                line_start -= 1

            # 找到 __all__ 之后的换行符
            line_end = end
            while line_end < len(content) and content[line_end] != '\n':
                line_end += 1
            if line_end < len(content):
                line_end += 1  # 包含换行符

            new_content = content[:line_start] + content[line_end:]

            # 写回文件
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print("✅ 已成功删除 __all__ 定义")
            return True
        else:
            print("❌ 无法找到 __all__ 的结束括号")
            return False
    else:
        print("❌ 无法找到 __all__ 的开始位置")
        return False

def check_all_usage():
    """检查代码中是否有使用 __all__ 或 import * 的地方"""
    import os

    # 检查 import * 的使用
    import_star_count = 0
    all_usage_count = 0

    for root, dirs, files in os.walk('.'):
        # 跳过一些目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if 'import *' in content or 'from *' in content:
                        import_star_count += 1
                        print(f"发现 import * 使用: {file_path}")

                    if '__all__' in content:
                        all_usage_count += 1
                        if file_path.endswith('__init__.py'):
                            print(f"__all__ 定义: {file_path}")
                        else:
                            print(f"__all__ 使用: {file_path}")

                except:
                    continue

    print(f"\n检查结果:")
    print(f"  import * 使用次数: {import_star_count}")
    print(f"  __all__ 使用次数: {all_usage_count}")

    return import_star_count == 0

def main():
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'check':
            print("检查代码中 __all__ 和 import * 的使用情况:")
            check_all_usage()
        elif command == 'remove':
            print("删除 __all__ 定义:")
            if remove_all_from_init():
                print("操作完成")
            else:
                print("操作失败")
        elif command == 'suggest':
            print("建议方案:")
            print("1. 由于代码中没有使用 import *, __all__ 可以安全删除")
            print("2. 删除 __all__ 可以简化维护，避免手动同步问题")
            print("3. 如果未来需要，可以用脚本自动生成")
        else:
            print("用法:")
            print("  python simple_all_generator.py check   - 检查使用情况")
            print("  python simple_all_generator.py remove  - 删除 __all__")
            print("  python simple_all_generator.py suggest - 查看建议")
    else:
        print("__all__ 简化处理工具")
        print("\n基于分析结果，建议直接删除 __all__，原因:")
        print("1. 代码中没有使用 'from module import *'")
        print("2. __all__ 主要作用是控制 import * 的行为")
        print("3. 手动维护 __all__ 容易出错，当前已不一致")
        print("4. 删除后可以简化代码维护")
        print("\n运行 'python simple_all_generator.py remove' 执行删除")

if __name__ == "__main__":
    main()