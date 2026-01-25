#!/usr/bin/env python3
"""
批量修复因子计算中的索引和分组排名错误

这个脚本将自动检测并修复所有因子定义中的类似错误模式。
"""

import re
import os

def detect_and_fix_factor_errors(file_path):
    """检测并修复单个文件中的因子错误"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        original_content = content

    # 错误模式1: data.groupby('date')['column'].transform(lambda s: pd.Series(other_series.loc[s.index]).rank(...))
    pattern1 = r"rank = data\.groupby\('date'\)\[('[^']+'\)\]\.transform\(\s*lambda s: pd\.Series\(([^)]+)\.loc\[s\.index\]\)\.rank\(pct=True, method='min'\)\s*\)"

    # 错误模式2: 简化版本 - 使用不同列名进行分组
    pattern2 = r"data\.groupby\('date'\)\[('[^']+'\)\]\.transform\(\s*lambda s: pd\.Series\(([^)]+)\.loc\[s\.index\]\)\.rank\(pct=True, method='min'\)\s*\)"

    # 查找所有匹配项
    matches = re.finditer(pattern1, content)
    fixes_applied = 0

    for match in matches:
        original_line = match.group(0)
        grouped_column = match.group(1)  # 分组的列名
        factor_series = match.group(2)   # 因子Series名

        # 生成修复后的代码
        fixed_code = f"""# 标准化到[0, 1]区间 - 修复索引错误
    # 创建包含{factor_series}和date的DataFrame用于正确分组
    df_for_rank = pd.DataFrame({{
        'factor_value': {factor_series},
        'date': data['date'].values
    }, index=data.index)

    # 按日期分组，对factor_value进行排名归一化
    rank = df_for_rank.groupby('date')['factor_value'].transform(
        lambda x: x.rank(pct=True, method='min')
    )

    return rank.fillna(0.5).astype('float64')"""

        content = content.replace(original_line, fixed_code)
        fixes_applied += 1
        print(f"修复了 {file_path} 中的一个错误 (分组列: {grouped_column}, 因子: {factor_series})")

    # 检查pattern2
    matches2 = re.finditer(pattern2, content)
    for match in matches2:
        original_line = match.group(0)
        grouped_column = match.group(1)  # 分组的列名
        factor_series = match.group(2)   # 因子Series名

        # 生成修复后的代码
        fixed_code = f"""# 创建包含{factor_series}和date的DataFrame用于正确分组
    df_for_rank = pd.DataFrame({{
        'factor_value': {factor_series},
        'date': data['date'].values
    }, index=data.index)

    rank = df_for_rank.groupby('date')['factor_value'].transform(
        lambda x: x.rank(pct=True, method='min')
    )

    return rank.fillna(0.5).astype('float64')"""

        content = content.replace(original_line, fixed_code)
        fixes_applied += 1
        print(f"修复了 {file_path} 中的一个错误 (分组列: {grouped_column}, 因子: {factor_series})")

    # 如果有修改，写回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {file_path}: 修复了 {fixes_applied} 个错误")
        return fixes_applied
    else:
        print(f"✅ {file_path}: 无需修复")
        return 0

def main():
    """主函数：遍历所有因子定义文件进行修复"""

    factor_definitions_dir = "factor_system/factor_lab/definitions"

    if not os.path.exists(factor_definitions_dir):
        print(f"错误: 目录 {factor_definitions_dir} 不存在")
        return

    total_fixes = 0
    processed_files = 0

    # 遍历所有Python文件
    for filename in os.listdir(factor_definitions_dir):
        if filename.endswith('.py') and filename != '__init__.py':
            file_path = os.path.join(factor_definitions_dir, filename)
            print(f"\n处理文件: {filename}")
            fixes = detect_and_fix_factor_errors(file_path)
            total_fixes += fixes
            processed_files += 1

    print(f"\n🎉 修复完成!")
    print(f"处理文件数: {processed_files}")
    print(f"总修复数: {total_fixes}")

if __name__ == "__main__":
    main()