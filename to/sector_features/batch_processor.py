#!/usr/bin/env python
# -*- coding: utf-8 -*
import os
import time
import logging
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from datetime import datetime
from .feature_generator import SectorFeatureGenerator
from multiprocessing import Pool

try:
    from to.path_utils import get_data_path, ensure_dir
except ImportError:
    # 如果导入失败，提供简单的替代实现
    def get_data_path(relative_path=None):
        from pathlib import Path
        data_root = Path('to') / 'local_data'
        if relative_path:
            return data_root / relative_path
        return data_root
    
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)
        return path

logger = logging.getLogger(__name__)

# 定义一个全局函数，用于多进程调用
def _process_sector_wrapper(args):
    """处理单个板块的包装函数，用于多进程
    
    Args:
        args: 包含处理参数的元组 (sector_name, start_date, end_date, output_dir, sector_stocks, weighting_method)
        
    Returns:
        元组 (板块名称, 成功标志, 错误消息)
    """
    try:
        sector_name, start_date, end_date, output_dir, sector_stocks, weighting_method = args
        
        logger.info(f"开始处理板块 {sector_name} 的特征生成")
        
        # 在进程内部创建数据加载器
        from to.data_loaders import LocalDataLoader
        
        # 创建数据加载器
        logger.info(f"为板块 {sector_name} 创建数据加载器")
        data_loader = LocalDataLoader()
        
        # 创建板块映射
        sector_mapping = {sector_name: sector_stocks}
        
        # 创建特征生成器
        logger.info(f"为板块 {sector_name} 创建特征生成器")
        generator = SectorFeatureGenerator(
            data_provider=data_loader,
            sector_mapping=sector_mapping,
            output_dir=output_dir,
            weighting_method=weighting_method
        )
        
        # 处理板块特征
        logger.info(f"生成板块 {sector_name} 的特征数据")
        features_df = generator.process_sector(sector_name, start_date, end_date, True)
        
        # 检查结果
        if features_df is None or features_df.empty:
            logger.warning(f"板块 {sector_name} 生成的特征为空")
            return sector_name, False, "生成的特征为空"
        
        logger.info(f"成功生成板块 {sector_name} 的特征数据，共 {len(features_df)} 行")
        return sector_name, True, None
        
    except Exception as e:
        import traceback
        error_msg = f"处理板块特征时出错: {str(e)}"
        logging.error(f"板块 {args[0] if len(args) > 0 else 'unknown'} {error_msg}")
        logging.error(f"错误详情: {traceback.format_exc()}")
        return args[0] if len(args) > 0 else "unknown", False, error_msg

class SectorFeatureBatchProcessor:
    """板块特征批量处理器
    
    负责批量处理多个板块的特征生成，支持多进程并行处理。
    """
    
    def __init__(self, 
                 data_loader: Any, 
                 sector_mapping: Dict[str, List[str]],
                 output_dir: str = None,
                 weighting_method: str = "equal",
                 max_workers: int = None):
        """初始化批量处理器
        
        Args:
            data_loader: 数据加载器
            sector_mapping: 板块映射字典 {板块名称: [股票代码列表]}
            output_dir: 特征输出目录，默认使用get_data_path('sector_features')
            weighting_method: 权重计算方法
            max_workers: 最大工作进程数，默认为CPU核心数-1
        """
        self.data_loader = data_loader
        self.sector_mapping = sector_mapping
        self.output_dir = output_dir if output_dir is not None else get_data_path('sector_features')
        self.weighting_method = weighting_method
        
        # 设置最大工作进程数
        if max_workers is None:
            self.max_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.max_workers = max(1, min(max_workers, multiprocessing.cpu_count()))
            
        # 创建特征生成器
        self.feature_generator = SectorFeatureGenerator(
            data_provider=data_loader,
            sector_mapping=sector_mapping,
            output_dir=output_dir,
            weighting_method=weighting_method
        )
        
        # 确保输出目录存在
        ensure_dir(self.output_dir)
        
        logger.info(f"初始化板块特征批量处理器，最大工作进程数: {self.max_workers}")
    
    def _process_single_sector(self, sector_name, start_date, end_date):
        """处理单个板块的特征生成
        
        Args:
            sector_name: 板块名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict: 处理结果
        """
        start_time = time.time()
        result = {
            'sector': sector_name,
            'start_date': start_date,
            'end_date': end_date,
            'success': False,
            'row_count': 0,
            'time_taken': 0
        }
        
        try:
            logger.info(f"开始处理板块 {sector_name} 的特征")
            
            # 使用特征生成器生成特征
            features_df = self.feature_generator.process_sector(
                sector_name=sector_name,
                start_date=start_date,
                end_date=end_date
            )
            
            # 更新结果
            result['success'] = True
            result['row_count'] = len(features_df)
            
            # 记录处理时间
            elapsed_time = time.time() - start_time
            result['time_taken'] = elapsed_time
            
            logger.info(f"成功处理板块 {sector_name} 的特征，行数: {result['row_count']}，耗时: {elapsed_time:.2f}秒")
            
        except Exception as e:
            # 处理失败
            logger.error(f"处理板块 {sector_name} 时出错: {e}")
            import traceback
            logger.error(f"{traceback.format_exc()}")
            
            # 更新结果
            result['success'] = False
            result['error'] = str(e)
            
            # 记录处理时间
            elapsed_time = time.time() - start_time
            result['time_taken'] = elapsed_time
            
        return result
    
    def _process_sector_wrapper(self, args):
        """多进程处理包装函数
        
        Args:
            args: 元组 (sector_name, start_date, end_date)
            
        Returns:
            处理结果字典
        """
        sector_name, start_date, end_date = args
        return self._process_single_sector(sector_name, start_date, end_date)
    
    def process_sectors(self, 
                       sector_names: List[str], 
                       start_date: str, 
                       end_date: str,
                       use_multiprocessing: bool = True) -> Dict[str, Dict[str, Any]]:
        """批量处理多个板块的特征生成
        
        Args:
            sector_names: 板块名称列表
            start_date: 开始日期
            end_date: 结束日期
            use_multiprocessing: 是否使用多进程
            
        Returns:
            处理结果字典 {板块名称: 结果字典}
        """
        total_sectors = len(sector_names)
        logger.info(f"开始批量处理 {total_sectors} 个板块的特征，时间段: {start_date}-{end_date}")
        
        # 用于记录结果的字典
        results = {}
        
        # 构建参数列表
        args_list = [(sector, start_date, end_date) for sector in sector_names]
        
        # 使用多进程处理
        if use_multiprocessing and total_sectors > 1 and self.max_workers > 1:
            logger.info(f"使用多进程处理，进程数: {min(self.max_workers, total_sectors)}")
            
            try:
                # 创建进程池
                with Pool(processes=min(self.max_workers, total_sectors)) as pool:
                    # 提交所有任务
                    results_list = pool.map(self._process_sector_wrapper, args_list)
                    
                    # 整理结果
                    for result in results_list:
                        results[result['sector']] = result
            except Exception as e:
                logger.error(f"多进程处理出错: {str(e)}")
                logger.warning("切换到单进程模式")
                
                # 如果多进程失败，回退到单进程模式
                for sector in sector_names:
                    if sector not in results:
                        result = self._process_single_sector(sector, start_date, end_date)
                        results[result['sector']] = result
        else:
            # 单进程顺序处理
            logger.info("使用单进程顺序处理")
            for sector in sector_names:
                result = self._process_single_sector(sector, start_date, end_date)
                results[result['sector']] = result
                
                # 每处理10个板块输出一次进度
                processed = len(results)
                if processed % 10 == 0 or processed == total_sectors:
                    success_count = sum(1 for r in results.values() if r['success'])
                    logger.info(f"进度: {processed}/{total_sectors} 板块 ({processed/total_sectors*100:.1f}%)，成功: {success_count}，失败: {processed-success_count}")
        
        # 计算统计信息
        success_count = sum(1 for r in results.values() if r['success'])
        total_rows = sum(r['row_count'] for r in results.values() if r['success'])
        total_duration = sum(r['time_taken'] for r in results.values())
        
        logger.info(f"批量处理完成，共 {total_sectors} 个板块，成功: {success_count}，失败: {total_sectors-success_count}")
        logger.info(f"总行数: {total_rows}，总耗时: {total_duration:.2f}秒，平均每板块耗时: {total_duration/total_sectors:.2f}秒")
        
        return results
    
    def save_report(self, results: Dict[str, Dict[str, Any]], report_file: Optional[str] = None) -> str:
        """保存处理报告
        
        Args:
            results: 处理结果字典
            report_file: 报告文件路径，默认为None，将自动生成文件名
            
        Returns:
            报告文件路径
        """
        if report_file is None:
            # 自动生成报告文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.output_dir, f"processing_report_{timestamp}.json")
        
        # 创建报告内容
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_sectors': len(results),
            'success_count': sum(1 for r in results.values() if r.get('success', False)),
            'fail_count': sum(1 for r in results.values() if not r.get('success', False)),
            'total_rows': sum(r.get('row_count', 0) for r in results.values()),
            'total_duration': sum(r.get('time_taken', 0) for r in results.values()),
            'details': results
        }
        
        # 将报告保存为JSON文件
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"处理报告已保存至: {report_file}")
        except Exception as e:
            logger.error(f"保存处理报告失败: {str(e)}")
        
        return report_file