#!/usr/bin/env python3
"""
删除冗余的IntelligentClipper相关代码
"""

def cleanup_intelligent_clipper():
    """清理IntelligentClipper相关代码"""
    file_path = "factor_system/factor_lab/pipeline.py"

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 找到需要删除的代码段
    start_delete = None
    end_delete = None

    for i, line in enumerate(lines):
        if "class IntelligentClipper:" in line:
            start_delete = i
        elif start_delete is not None and line.strip().startswith("class FactorPipeline:"):
            end_delete = i
            break

    if start_delete is not None and end_delete is not None:
        # 删除IntelligentClipper类和PipelineResult
        cleaned_lines = lines[:start_delete] + lines[end_delete:]

        # 删除_intelligent_winsorize方法
        start_method = None
        end_method = None
        for i, line in enumerate(cleaned_lines):
            if "def _intelligent_winsorize(" in line:
                start_method = i
            elif start_method is not None and line.strip().startswith("def _daily_winsorize("):
                end_method = i
                break

        if start_method is not None and end_method is not None:
            cleaned_lines = cleaned_lines[:start_method] + cleaned_lines[end_method:]

        # 删除相关引用
        final_lines = []
        for line in cleaned_lines:
            if "self.intelligent_clipper" not in line:
                final_lines.append(line)

        # 保存修改后的文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(final_lines)

        print(f"成功清理IntelligentClipper相关代码")
    else:
        print("未找到需要删除的代码段")

if __name__ == "__main__":
    cleanup_intelligent_clipper()