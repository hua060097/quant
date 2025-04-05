#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # 尝试导入需要的模块
    from market_status_processor import MarketStatusProcessor
    
    try:
        from to.advanced_quant_data import ParquetDataStorage
    except ImportError:
        logger.warning("无法导入ParquetDataStorage，将仅使用外部数据源")
        ParquetDataStorage = None
except ImportError as e:
    logger.error(f"导入模块失败: {str(e)}")
    logger.error("请确保已安装所有依赖并且目录结构正确")
    sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="大盘状态处理工具")
    parser.add_argument("--start_date", type=str, default="20230101", 
                        help="开始日期，格式: YYYYMMDD")
    parser.add_argument("--end_date", type=str, 
                        default=datetime.now().strftime("%Y%m%d"),
                        help="结束日期，格式: YYYYMMDD，默认为今天")
    parser.add_argument("--data_dir", type=str, default="local_data/market_status", 
                        help="市场状态数据存储目录")
    parser.add_argument("--raw_data_dir", type=str, default="local_data", 
                        help="原始股票数据存储目录")
    parser.add_argument("--parallel", action="store_true", default=True,
                        help="是否启用并行处理")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="最大并行工作进程数，默认为CPU核心数")
    parser.add_argument("--index_only", action="store_true", default=False,
                        help="只处理指数数据，不使用本地股票数据")
    parser.add_argument("--report", action="store_true", default=False,
                        help="生成分析报告")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 初始化原始数据存储
        raw_data_storage = None
        if not args.index_only and ParquetDataStorage is not None:
            try:
                logger.info(f"初始化原始数据存储: {args.raw_data_dir}")
                raw_data_storage = ParquetDataStorage(args.raw_data_dir)
            except Exception as e:
                logger.warning(f"初始化原始数据存储失败: {str(e)}")
                logger.warning("将尝试从外部数据源获取数据")
        
        # 初始化市场状态处理器
        logger.info(f"初始化市场状态处理器: {args.data_dir}")
        processor = MarketStatusProcessor(
            data_dir=args.data_dir,
            raw_data_storage=raw_data_storage,
            parallel=args.parallel,
            max_workers=args.max_workers
        )
        
        # 处理市场状态
        start_time = datetime.now()
        logger.info(f"开始处理 {args.start_date} 至 {args.end_date} 的市场状态数据")
        processed_count = processor.process_market_status(args.start_date, args.end_date)
        end_time = datetime.now() 
        
        # 处理完成，打印统计信息
        duration = end_time - start_time
        logger.info(f"市场状态数据处理完成，共处理 {processed_count} 个交易日")
        logger.info(f"处理耗时: {duration.total_seconds():.2f} 秒")
        
        # 生成报告
        if args.report:
            report_file = os.path.join(args.data_dir, f"market_status_report_{args.start_date}_{args.end_date}.md")
            logger.info(f"生成市场状态报告: {report_file}")
            processor.generate_market_status_report(
                start_date=args.start_date,
                end_date=args.end_date,
                output_file=report_file
            )
            logger.info(f"报告已保存到: {report_file}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        return 1
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 