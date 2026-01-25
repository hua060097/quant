#!/usr/bin/env python3
import ast

def check_syntax(file_path):
    """检查Python文件的语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # 尝试解析AST
        ast.parse(source)
        print(f"✅ {file_path} 语法检查通过")
        return True
        
    except SyntaxError as e:
        print(f"❌ {file_path} 语法错误: {e}")
        print(f"   行 {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ {file_path} 检查失败: {e}")
        return False

if __name__ == "__main__":
    main_file = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\main.py"
    check_syntax(main_file)