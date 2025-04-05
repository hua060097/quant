import logging
import os
import sys
import time
from datetime import datetime
import traceback

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logger = logging.getLogger('analysis_monitor')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('to/log/analysis_monitor.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def monitor_log_file(log_file_path, interval=5, max_runtime=300):
    """
    监控分析日志文件并报告进度
    
    参数:
        log_file_path: 要监控的日志文件路径
        interval: 检查间隔(秒)
        max_runtime: 最大运行时间(秒)
    """
    start_time = time.time()
    last_size = 0
    last_position = 0
    
    logger.info(f"开始监控日志文件: {log_file_path}")
    
    try:
        while time.time() - start_time < max_runtime:
            if os.path.exists(log_file_path):
                current_size = os.path.getsize(log_file_path)
                
                if current_size > last_size:
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        f.seek(last_position)
                        new_content = f.read()
                        last_position = f.tell()
                    
                    logger.info(f"日志文件增加了 {current_size - last_size} 字节")
                    
                    # 分析日志内容
                    analyze_log_content(new_content)
                    
                    last_size = current_size
            else:
                logger.warning(f"日志文件不存在: {log_file_path}")
            
            time.time()
            time.sleep(interval)
    
    except KeyboardInterrupt:
        logger.info("监控被用户中断")
    except Exception as e:
        logger.error(f"监控过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info("监控结束")

def analyze_log_content(content):
    """分析日志内容并提取关键信息"""
    # 查找关键进度指标
    progress_indicators = [
        "初始化高级量化模型",
        "加载板块数据",
        "加载交易日期",
        "特征提取",
        "模型训练",
        "模型评估",
        "预测结果",
        "生成交易信号",
        "回测结果",
        "完成量化交易研究流程"
    ]
    
    error_indicators = [
        "ERROR",
        "Exception",
        "失败",
        "错误"
    ]
    
    for line in content.split('\n'):
        # 检查错误
        if any(indicator in line for indicator in error_indicators):
            logger.error(f"检测到错误: {line.strip()}")
        
        # 检查进度
        for indicator in progress_indicators:
            if indicator in line:
                logger.info(f"进度更新: {line.strip()}")

def main():
    logger.info("=" * 50)
    logger.info("分析监控工具启动")
    logger.info("=" * 50)
    
    # 设置要监控的日志文件路径
    # 如果有自定义日志路径，可以在这里修改
    log_file_path = "run_analysis.log"
    
    try:
        # 启动监控
        monitor_log_file(log_file_path)
    except Exception as e:
        logger.error(f"主程序发生错误: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 