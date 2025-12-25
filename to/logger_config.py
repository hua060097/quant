import logging
import os
from pathlib import Path

def configure_logger(name, log_file_name):
    """配置logger，统一输出到to/generate_sector_features.log
    
    Args:
        name: logger名称，通常使用__name__
        
    Returns:
        配置好的logger实例
    """
    # 确保日志目录存在
    log_dir = Path('to/log')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志文件路径
    log_file = log_dir / log_file_name
    
    # 获取logger实例
    logger = logging.getLogger(name)
    
    # 如果logger已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置formatter
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.propagate = False 
    
    return logger 