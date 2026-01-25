#!/usr/bin/env python3
"""测试导入"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # 尝试导入main中的特定函数
    from main import load_sw_sector_mapping
    print("[OK] load_sw_sector_mapping 导入成功")

    # 检查函数签名
    import inspect
    sig = inspect.signature(load_sw_sector_mapping)
    print(f"[OK] 函数签名: {sig}")

except Exception as e:
    print(f"[FAIL] 导入失败: {e}")
    import traceback
    traceback.print_exc()