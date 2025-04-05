#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单测试加载指定日期的模型
"""

import os
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型加载函数
from advanced_quant_model import _load_models_for_date

def main():
    """主函数"""
    # 测试日期
    test_dates = ["20230401", "20230402"]
    
    for date_str in test_dates:
        logger.info(f"测试加载日期 {date_str} 的模型")
        
        # 使用函数加载模型
        models = _load_models_for_date(date_str)
        
        if models:
            logger.info(f"成功加载 {len(models)} 个模型")
            logger.info("模型列表:")
            for name in models:
                logger.info(f"  - {name}")
        else:
            logger.info(f"未找到日期 {date_str} 的模型")

if __name__ == "__main__":
    logger.info("开始测试")
    main()
    logger.info("测试完成") 