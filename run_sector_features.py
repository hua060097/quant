#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块特征生成启动脚本

该脚本用于从项目根目录启动板块特征生成模块，正确处理导入路径问题。

使用示例:
    python run_sector_features.py --start-date 20230101 --end-date 20230331
    
    或者:
    python run_sector_features.py --start_date 20230101 --end_date 20230331

支持的参数:
    --start-date/--start_date: 开始日期，格式为YYYYMMDD
    --end-date/--end_date: 结束日期，格式为YYYYMMDD
    --sectors: 板块列表，用逗号分隔，如"TGN人工智能,TGN新能源"
    --force-update/--force_update: 强制重新生成所有特征
    --max-workers/--max_workers: 最大工作进程数
    --batch-size/--batch_size: 每批处理的日期数量
    --output-dir/--output_dir: 输出目录
    --use-parquet/--use_parquet: 使用Parquet格式保存所有特征数据(默认启用)
    --csv-only/--csv_only: 只使用CSV格式保存特征(调试用，不建议用于大量数据)
"""

import os
import sys
from pathlib import Path

# 确保当前工作目录是项目根目录
if 'to' not in os.listdir('.'):
    print("警告: 当前目录不是项目根目录，可能导致导入错误!")
    print(f"当前工作目录: {os.getcwd()}")
    print("请从项目根目录运行此脚本")

# 添加当前目录到sys.path，确保能找到to模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 尝试导入模块
try:
    # 直接导入板块特征生成模块的main函数
    from to.generate_sector_features import main
    
    # 执行主函数
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保从项目根目录运行此脚本")
    sys.exit(1) 