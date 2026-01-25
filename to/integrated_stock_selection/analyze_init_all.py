#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 __init__.py 中 __all__ 的使用情况和自动生成可能性
"""

def analyze_all_usage():
    """分析 __all__ 的实际使用情况"""
    print("=" * 80)
    print("分析 __init__.py 中的 __all__ 使用情况")
    print("=" * 80)

    # 1. 检查 __init__.py 中的导入语句
    print("\n1. 检查 __init__.py 中的导入语句:")
    try:
        with open('factor_system/factor_lab/definitions/__init__.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取所有 import 语句
        import_lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('from .') and 'import' in line:
                import_lines.append(line)

        print(f"   找到 {len(import_lines)} 个导入语句:")
        for i, line in enumerate(import_lines, 1):
            print(f"   {i:2d}. {line}")

        # 提取导入的因子名称
        imported_factors = []
        for line in import_lines:
            # 从 "from .xxx import a, b, c" 中提取 a, b, c
            if 'import' in line:
                imports_part = line.split('import')[1].strip()
                # 处理多行导入
                factors = [f.strip() for f in imports_part.split(',')]
                # 移除注释
                factors = [f.split('#')[0].strip() for f in factors if f.strip() and not f.strip().startswith('#')]
                imported_factors.extend(factors)

        print(f"\n   导入的因子总数: {len(imported_factors)}")
        print(f"   导入的因子: {imported_factors[:10]}...")

    except Exception as e:
        print(f"   [ERROR] 读取 __init__.py 失败: {e}")

    # 2. 检查 __all__ 的内容
    print("\n2. 检查 __all__ 的内容:")
    try:
        with open('factor_system/factor_lab/definitions/__init__.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找 __all__ 定义
        if '__all__' in content:
            # 提取 __all__ 列表
            start = content.find('__all__ = [')
            if start != -1:
                end = content.find(']', start) + 1
                all_definition = content[start:end]
                print(f"   __all__ 定义:")
                print(f"   {all_definition}")

                # 提取 __all__ 中的因子名称
                import ast
                try:
                    all_list = ast.literal_eval(all_definition.split('=')[1].strip())
                    print(f"   __all__ 中的因子数: {len(all_list)}")
                    print(f"   __all__ 中的因子: {all_list[:10]}...")
                except:
                    print("   [ERROR] 无法解析 __all__ 列表")
            else:
                print("   [ERROR] 找不到 __all__ 列表结束符")
        else:
            print("   [INFO] __init__.py 中没有 __all__ 定义")

    except Exception as e:
        print(f"   [ERROR] 分析 __all__ 失败: {e}")

    # 3. 检查代码中是否有使用 import * 的地方
    print("\n3. 检查代码中是否有使用 import * 的地方:")

    import os
    python_files = []
    for root, dirs, files in os.walk('.'):
        # 跳过一些目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    import_star_usage = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查是否有 import * 相关的语句
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if 'import *' in line or 'from *' in line:
                    import_star_usage.append((file_path, i, line))
        except:
            continue

    if import_star_usage:
        print(f"   发现 {len(import_star_usage)} 处 import * 使用:")
        for file_path, line_num, line in import_star_usage[:5]:
            print(f"     {file_path}:{line_num} - {line}")
        if len(import_star_usage) > 5:
            print(f"     ... 还有 {len(import_star_usage) - 5} 处")
    else:
        print("   [INFO] 代码中没有发现 import * 的使用")

    # 4. 分析 __all__ 的实际作用
    print("\n4. 分析 __all__ 的实际作用:")
    print("   __all__ 的主要作用:")
    print("   a) 控制 'from module import *' 的行为")
    print("   b) 作为模块公共API的文档")
    print("   c) 防止 linter 报告未使用的导入")

    if not import_star_usage:
        print("   [结论] 由于代码中没有使用 import *, __all__ 实际上不起作用")

    # 5. 检查不一致性
    print("\n5. 检查导入语句与 __all__ 的不一致性:")
    try:
        # 重新分析导入和 __all__
        imported_factors = set()
        with open('factor_system/factor_lab/definitions/__init__.py', 'r', encoding='utf-8') as f:
            content = f.read()

        for line in content.split('\n'):
            if line.strip().startswith('from .') and 'import' in line:
                imports_part = line.split('import')[1].strip()
                factors = [f.strip().split('#')[0].strip() for f in imports_part.split(',') if f.strip()]
                imported_factors.update(factors)

        # 查找 __all__
        all_factors = set()
        if '__all__' in content:
            start = content.find('__all__ = [')
            if start != -1:
                end = content.find(']', start) + 1
                all_definition = content[start:end]
                try:
                    import ast
                    all_list = ast.literal_eval(all_definition.split('=')[1].strip())
                    all_factors = set(all_list)
                except:
                    pass

        # 比较差异
        only_in_imports = imported_factors - all_factors
        only_in_all = all_factors - imported_factors

        if only_in_imports:
            print(f"   在导入中但不在 __all__ 中 ({len(only_in_imports)} 个):")
            for factor in sorted(list(only_in_imports)[:10]):
                print(f"     - {factor}")
            if len(only_in_imports) > 10:
                print(f"     ... 还有 {len(only_in_imports) - 10} 个")

        if only_in_all:
            print(f"   在 __all__ 中但不在导入中 ({len(only_in_all)} 个):")
            for factor in sorted(list(only_in_all)[:10]):
                print(f"     - {factor}")
            if len(only_in_all) > 10:
                print(f"     ... 还有 {len(only_in_all) - 10} 个")

        if not only_in_imports and not only_in_all:
            print("   [GOOD] 导入语句和 __all__ 完全一致")

    except Exception as e:
        print(f"   [ERROR] 比较失败: {e}")

def suggest_auto_generation():
    """建议自动生成方案"""
    print("\n" + "=" * 80)
    print("自动生成 __all__ 的建议方案")
    print("=" * 80)

    print("\n建议:")
    print("1. 由于代码中没有使用 import *, __all__ 可以删除")
    print("2. 如果为了代码文档化，可以创建自动生成脚本")
    print("3. 在 CI/CD 中加入一致性检查")

    print("\n自动生成脚本示例:")
    script_code = '''
#!/usr/bin/env python3
"""
自动生成 __init__.py 中的 __all__ 列表
"""

import os
import ast

def generate_all():
    """自动生成 __all__ 列表"""

    # 读取当前 __init__.py
    init_file = 'factor_system/factor_lab/definitions/__init__.py'
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取导入的因子名称
    imported_factors = []
    for line in content.split('\\n'):
        if line.strip().startswith('from .') and 'import' in line:
            imports_part = line.split('import')[1].strip()
            factors = [f.strip().split('#')[0].strip() for f in imports_part.split(',') if f.strip()]
            imported_factors.extend(factors)

    # 生成新的 __all__ 定义
    all_definition = "__all__ = [\\n"
    for i, factor in enumerate(sorted(imported_factors)):
        comma = "," if i < len(imported_factors) - 1 else ""
        all_definition += f"    '{factor}'{comma}\\n"
    all_definition += "]"

    # 替换原有的 __all__ 定义
    if '__all__' in content:
        start = content.find('__all__ = [')
        end = content.find(']', start) + 1
        new_content = content[:start] + all_definition + content[end:]
    else:
        new_content = content + "\\n\\n" + all_definition

    # 写回文件
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"已更新 __all__，包含 {len(imported_factors)} 个因子")

if __name__ == "__main__":
    generate_all()
'''

    print(script_code)

    print("\n一致性检查脚本:")
    check_script = '''
#!/usr/bin/env python3
"""
检查导入语句与 __all__ 的一致性
"""

def check_consistency():
    """检查一致性并报告差异"""
    # 实现类似上面的比较逻辑
    # 如果发现不一致，返回错误码1
    pass

if __name__ == "__main__":
    if not check_consistency():
        print("导入语句与 __all__ 不一致！")
        exit(1)
'''

    print(check_script)

if __name__ == "__main__":
    analyze_all_usage()
    suggest_auto_generation()