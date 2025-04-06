#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块特征生成器启动脚本
简化用户使用流程，直接调用集成版特征生成器
"""

import os
import sys
import subprocess

def main():
    """主函数，转发命令行参数到集成版特征生成器"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建集成版特征生成器脚本路径
    integrated_script = os.path.join(current_dir, "to", "sector_features", "run_integrated_generator.py")
    
    # 检查脚本是否存在
    if not os.path.exists(integrated_script):
        print(f"错误: 集成版特征生成器脚本不存在: {integrated_script}")
        return 1
    
    # 构建命令，转发所有命令行参数
    cmd = [sys.executable, integrated_script] + sys.argv[1:]
    
    # 执行命令
    try:
        print(f"启动集成版板块特征生成器...")
        return subprocess.call(cmd)
    except Exception as e:
        print(f"执行集成版特征生成器时出错: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 