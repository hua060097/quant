#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
清理日志文件脚本
将根目录下的旧日志文件移动到to/log目录，
并删除to目录下的日志文件
"""

import os
import shutil
import glob
import logging
from datetime import datetime

# 配置日志
os.makedirs('to/log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/cleanup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cleanup")

def main():
    """主函数"""
    logger.info("开始清理日志文件")
    
    # 获取当前日期和时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 清理根目录下的日志文件
    root_log_files = glob.glob("*.log")
    logger.info(f"在根目录下找到 {len(root_log_files)} 个日志文件")
    
    for log_file in root_log_files:
        try:
            target_path = os.path.join("to/log", log_file)
            # 如果目标文件已存在，添加时间戳
            if os.path.exists(target_path):
                base_name, ext = os.path.splitext(log_file)
                target_path = os.path.join("to/log", f"{base_name}_{current_time}{ext}")
            
            # 移动文件
            shutil.move(log_file, target_path)
            logger.info(f"已移动: {log_file} -> {target_path}")
        except Exception as e:
            logger.error(f"移动 {log_file} 时出错: {e}")
    
    # 清理to目录下的日志文件
    to_log_files = glob.glob("to/*.log")
    logger.info(f"在to目录下找到 {len(to_log_files)} 个日志文件")
    
    for log_file in to_log_files:
        try:
            file_name = os.path.basename(log_file)
            target_path = os.path.join("to/log", file_name)
            # 如果目标文件已存在，添加时间戳
            if os.path.exists(target_path):
                base_name, ext = os.path.splitext(file_name)
                target_path = os.path.join("to/log", f"{base_name}_{current_time}{ext}")
            
            # 移动文件
            shutil.move(log_file, target_path)
            logger.info(f"已移动: {log_file} -> {target_path}")
        except Exception as e:
            logger.error(f"移动 {log_file} 时出错: {e}")
    
    logger.info("日志文件清理完成")

if __name__ == "__main__":
    main() 