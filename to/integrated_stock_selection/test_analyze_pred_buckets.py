"""
测试预测分桶分析模式
"""
import sys
import os

# 确保UTF-8编码输出
sys.stdout.reconfigure(encoding='utf-8')

# 添加项目路径
project_path = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection"
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# 导入main_alpha
from main_alpha原始 import run_analyze_pred_buckets_mode
import argparse

# 创建模拟args
class Args:
    def __init__(self):
        self.mode = 'analyze_pred_buckets'
        self.output_dir = './results'

args = Args()

try:
    print("开始测试预测分桶分析模式...")
    success = run_analyze_pred_buckets_mode(args)
    if success:
        print("测试成功完成！")
    else:
        print("测试失败！")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
