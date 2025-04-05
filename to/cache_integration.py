#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
缓存集成模块 - 将数据缓存系统无缝集成到回测框架中

该模块负责将数据缓存系统与现有回测框架集成，提供透明的数据访问接口和
钩子函数，使回测框架能够自动使用缓存系统优化数据获取性能。

主要功能:
- 透明地替换原始数据获取函数
- 提供回测框架初始化和清理钩子
- 跟踪和报告缓存性能

Authors: AI Assistant
Date:    2023-11-15
"""

import os
import logging
import time
import functools
from pathlib import Path
import sys
import importlib
import inspect
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
import threading
import traceback

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/cache_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cache_integration")

# 添加当前目录到路径，确保能够导入market_data_cache
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 尝试导入数据缓存管理器
try:
    from data_cache_manager import get_data_cache_manager
except ImportError:
    logger.error("无法导入数据缓存管理器，请确保data_cache_manager.py文件存在")
    get_data_cache_manager = None

# 尝试导入原始数据API
try:
    import xtquant.xtdata as xtdata
    XTDATA_AVAILABLE = True
except ImportError:
    logger.warning("无法导入xtdata模块，将使用模拟模式")
    xtdata = None
    XTDATA_AVAILABLE = False

class CacheIntegration:
    """
    缓存集成类 - 负责将缓存系统集成到回测框架中
    
    提供透明的数据访问层，劫持原始API调用并通过缓存系统提供数据，
    同时保持与原始API相同的接口，确保上层应用无需修改。
    """
    def __init__(self, cache_dir=None, memory_limit_mb=2000):
        """
        初始化缓存集成
        
        参数:
            cache_dir (str, optional): 缓存目录路径
            memory_limit_mb (int, optional): 内存缓存大小限制(MB)
        """
        self.enabled = True
        self.original_functions = {}
        self.patched = False
        self.performance_tracking = {}
        
        # 初始化缓存管理器
        self._init_cache(cache_dir, memory_limit_mb)
    
    def _init_cache(self, cache_dir=None, memory_limit_mb=2000):
        """
        初始化数据缓存管理器
        
        参数:
            cache_dir (str, optional): 缓存目录路径
            memory_limit_mb (int, optional): 内存缓存大小限制(MB)
        """
        if get_data_cache_manager:
            self.cache_manager = get_data_cache_manager(cache_dir, memory_limit_mb)
            logger.info(f"数据缓存管理器初始化成功: 缓存目录={cache_dir}, 内存限制={memory_limit_mb}MB")
        else:
            self.cache_manager = None
            logger.error("数据缓存管理器初始化失败")
    
    def integrate(self):
        """
        集成缓存系统到回测框架
        
        将原始的数据获取函数替换为使用缓存的版本，实现透明的缓存集成。
        
        返回:
            bool: 是否成功集成
        """
        if not self.cache_manager:
            logger.error("未初始化缓存管理器，无法集成")
            return False
        
        if self.patched:
            logger.warning("缓存系统已经集成，无需重复操作")
            return True
        
        try:
            # 保存原始函数
            if XTDATA_AVAILABLE:
                self.original_functions['get_market_data'] = xtdata.get_market_data
                self.original_functions['get_trading_dates'] = xtdata.get_trading_dates
                
                # 替换为缓存版本
                xtdata.get_market_data = self._patched_get_market_data
                xtdata.get_trading_dates = self._patched_get_trading_dates
                
                # 设置为已修补
                self.patched = True
                logger.info("缓存系统成功集成到数据API")
                return True
            else:
                logger.warning("xtdata模块不可用，无法进行集成")
                return False
        except Exception as e:
            logger.error(f"集成缓存系统时发生错误: {e}")
            self.restore()  # 出错时恢复原始函数
            return False
    
    def restore(self):
        """
        恢复原始函数
        
        取消对数据API的修补，恢复原始函数。
        
        返回:
            bool: 是否成功恢复
        """
        if not self.patched:
            return True
        
        try:
            if XTDATA_AVAILABLE:
                # 恢复原始函数
                for func_name, original_func in self.original_functions.items():
                    setattr(xtdata, func_name, original_func)
                
                # 重置状态
                self.original_functions = {}
                self.patched = False
                logger.info("已恢复原始数据API函数")
                return True
            return False
        except Exception as e:
            logger.error(f"恢复原始函数时发生错误: {e}")
            return False
    
    def _patched_get_market_data(self, stock_list, field_list, period='1d', 
                                start_time='', end_time='', count=-1, 
                                dividend_type='none', fill_data=True):
        """
        修补后的市场数据获取函数
        
        与原始API保持相同的接口，但使用缓存版本获取数据。
        
        参数:
            与原始xtdata.get_market_data相同
            
        返回:
            pd.DataFrame: 市场数据
        """
        if not self.enabled:
            # 如果缓存被禁用，直接调用原始函数
            return self.original_functions['get_market_data'](
                stock_list, field_list, period, start_time, end_time, 
                count, dividend_type, fill_data
            )
        
        # 检查是否在递归调用中（使用线程局部存储避免递归）
        thread_local = threading.local()
        if getattr(thread_local, 'in_get_market_data', False):
            # 如果已经在递归调用中，直接使用原始函数
            return self.original_functions['get_market_data'](
                stock_list, field_list, period, start_time, end_time, 
                count, dividend_type, fill_data
            )
        
        # 设置递归标记
        thread_local.in_get_market_data = True
        
        try:
            # 记录开始时间
            start_perf = time.time()
            
            # 转换为列表
            if isinstance(stock_list, str):
                stock_list = [stock_list]
            if isinstance(field_list, str):
                field_list = [field_list]
            
            # 仅使用基本参数调用缓存版本
            # 注意：某些高级参数如count/dividend_type在缓存版本可能不支持
            try:
                result = self.cache_manager.get_market_data(
                    stock_list=stock_list,
                    field_list=field_list,
                    period=period,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # 记录性能
                elapsed = time.time() - start_perf
                self._add_performance_record('get_market_data', elapsed)
                
                return result
            except Exception as e:
                logger.error(f"使用缓存获取市场数据失败: {e}，回退到原始API")
                
                # 发生错误时回退到原始API
                return self.original_functions['get_market_data'](
                    stock_list, field_list, period, start_time, end_time, 
                    count, dividend_type, fill_data
                )
        finally:
            # 无论成功与否，都要清除递归标记
            thread_local.in_get_market_data = False
    
    def _patched_get_trading_dates(self, market='SH', start_time='', end_time='', count=-1):
        """
        修补后的交易日期获取函数
        
        与原始API保持相同的接口，但使用缓存版本获取数据。
        
        参数:
            与原始xtdata.get_trading_dates相同
            
        返回:
            list: 交易日期列表
        """
        if not self.enabled:
            # 如果缓存被禁用，直接调用原始函数
            logger.debug("缓存系统未启用，使用原始API函数获取交易日期")
            return self.original_functions['get_trading_dates'](
                market, start_time, end_time, count
            )
        
        # 检查是否在递归调用中（使用线程局部存储避免递归）
        thread_local = threading.local()
        if getattr(thread_local, 'in_get_trading_dates', False):
            # 如果已经在递归调用中，直接使用原始函数
            logger.debug("检测到递归调用，使用原始API函数获取交易日期")
            return self.original_functions['get_trading_dates'](
                market, start_time, end_time, count
            )
        
        # 设置递归标记
        thread_local.in_get_trading_dates = True
        
        try:
            # 记录开始时间
            start_perf = time.time()
            
            # 记录调用参数
            logger.debug(f"调用缓存版本的get_trading_dates: market={market}, start_time={start_time}, end_time={end_time}, count={count}")
            
            try:
                # 确保参数处理与原始API一致
                # 特别是处理空字符串和None的差异
                cache_start_time = None if start_time == '' else start_time
                cache_end_time = None if end_time == '' else end_time
                
                logger.debug(f"缓存参数处理: start_time={'None' if cache_start_time is None else cache_start_time}, "
                             f"end_time={'None' if cache_end_time is None else cache_end_time}")
                
                result = self.cache_manager.get_trading_dates(
                    market=market,
                    start_time=cache_start_time,
                    end_time=cache_end_time
                )
                
                # 如果缓存返回None或空列表，则调用原始API
                if result is None or len(result) == 0:
                    logger.debug("缓存返回空结果，回退到原始API")
                    result = self.original_functions['get_trading_dates'](
                        market, start_time, end_time, count
                    )
                    
                    # 如果API返回有效结果，则更新缓存
                    if result and len(result) > 0:
                        # 尝试保存到缓存
                        try:
                            self.cache_manager.save_trading_dates(market=market, trading_dates=result)
                            logger.debug(f"已将API结果({len(result)}项)保存到缓存")
                        except Exception as e:
                            logger.warning(f"保存交易日期到缓存时出错: {e}")
                
                # 处理count参数
                if count > 0 and result and len(result) > count:
                    logger.debug(f"根据count={count}截取结果，原始长度={len(result)}")
                    result = result[-count:]
                
                # 记录性能
                elapsed = time.time() - start_perf
                self._add_performance_record('get_trading_dates', elapsed)
                
                logger.debug(f"get_trading_dates成功返回{len(result) if result else 0}个结果，耗时={elapsed:.3f}秒")
                return result
            except Exception as e:
                logger.error(f"使用缓存获取交易日期失败: {e}，回退到原始API")
                logger.debug(f"异常详情: {traceback.format_exc()}")
                
                # 发生错误时回退到原始API
                return self.original_functions['get_trading_dates'](
                    market, start_time, end_time, count
                )
        finally:
            # 无论成功与否，都要清除递归标记
            thread_local.in_get_trading_dates = False
    
    def _add_performance_record(self, function_name, elapsed):
        """
        添加性能记录
        
        参数:
            function_name (str): 函数名称
            elapsed (float): 执行时间(秒)
        """
        if function_name not in self.performance_tracking:
            self.performance_tracking[function_name] = {
                'calls': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0
            }
        
        stats = self.performance_tracking[function_name]
        stats['calls'] += 1
        stats['total_time'] += elapsed
        stats['min_time'] = min(stats['min_time'], elapsed)
        stats['max_time'] = max(stats['max_time'], elapsed)
    
    def get_performance_report(self):
        """
        获取性能报告
        
        返回:
            dict: 性能统计信息
        """
        report = {
            'cache_stats': self.cache_manager.get_stats() if self.cache_manager else {},
            'function_performance': {}
        }
        
        # 计算每个函数的平均执行时间
        for func_name, stats in self.performance_tracking.items():
            avg_time = stats['total_time'] / stats['calls'] if stats['calls'] > 0 else 0
            
            report['function_performance'][func_name] = {
                'calls': stats['calls'],
                'total_time': stats['total_time'],
                'avg_time': avg_time,
                'min_time': stats['min_time'] if stats['min_time'] != float('inf') else 0,
                'max_time': stats['max_time']
            }
        
        return report
    
    def generate_report(self):
        """
        生成完整性能报告
        
        返回:
            str: 性能报告文本
        """
        if not self.cache_manager:
            return "缓存管理器未初始化，无法生成报告"
        
        # 获取缓存系统的详细报告
        cache_report = self.cache_manager.generate_performance_report()
        
        # 获取函数性能报告
        func_report = ["", "=" * 60, "函数性能统计", "=" * 60]
        
        for func_name, stats in self.performance_tracking.items():
            if stats['calls'] > 0:
                avg_time = stats['total_time'] / stats['calls']
                func_report.append(f"函数: {func_name}")
                func_report.append(f"  调用次数: {stats['calls']}")
                func_report.append(f"  总耗时: {stats['total_time']:.4f}秒")
                func_report.append(f"  平均耗时: {avg_time:.4f}秒")
                func_report.append(f"  最小耗时: {stats['min_time'] if stats['min_time'] != float('inf') else 0:.4f}秒")
                func_report.append(f"  最大耗时: {stats['max_time']:.4f}秒")
                func_report.append("-" * 40)
        
        # 合并报告
        full_report = cache_report + "\n" + "\n".join(func_report)
        
        # 保存报告
        try:
            log_dir = Path("to/log")
            log_dir.mkdir(exist_ok=True, parents=True)
            report_path = log_dir / f"cache_integration_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(full_report)
            
            logger.info(f"性能报告已保存到: {report_path}")
        except Exception as e:
            logger.error(f"保存性能报告失败: {e}")
        
        return full_report
    
    def preload_data(self, stock_list, start_date, end_date, fields=None, periods=None):
        """
        预加载数据到缓存
        
        参数:
            stock_list (list): 股票代码列表
            start_date (str): 开始日期
            end_date (str): 结束日期
            fields (list, optional): 字段列表
            periods (list, optional): 周期列表
            
        返回:
            dict: 预加载统计信息
        """
        if not self.cache_manager:
            logger.error("缓存管理器未初始化，无法预加载数据")
            return None
        
        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        if periods is None:
            periods = ['1d']
        
        return self.cache_manager.preload_data(
            stock_list=stock_list,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
            periods=periods
        )
    
    def parallel_preload(self, stock_list, start_date, end_date, fields=None, 
                         periods=None, batch_size=50, max_workers=None):
        """
        并行预加载数据到缓存
        
        参数:
            stock_list (list): 股票代码列表
            start_date (str): 开始日期
            end_date (str): 结束日期
            fields (list, optional): 字段列表
            periods (list, optional): 周期列表
            batch_size (int): 每批处理的股票数量
            max_workers (int, optional): 最大工作线程数
            
        返回:
            dict: 预加载统计信息
        """
        if not self.cache_manager:
            logger.error("缓存管理器未初始化，无法预加载数据")
            return None
        
        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        if periods is None:
            periods = ['1d']
        
        return self.cache_manager.parallel_preload(
            stock_list=stock_list,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
            periods=periods,
            batch_size=batch_size,
            max_workers=max_workers
        )
    
    def clear_cache(self, older_than_days=None):
        """
        清理缓存数据
        
        参数:
            older_than_days (int, optional): 清理指定天数前的缓存
            
        返回:
            dict: 清理统计信息
        """
        if not self.cache_manager:
            logger.error("缓存管理器未初始化，无法清理缓存")
            return None
        
        return self.cache_manager.clear_cache(older_than_days)
    
    def enable(self):
        """启用缓存系统"""
        self.enabled = True
        logger.info("缓存系统已启用")
    
    def disable(self):
        """禁用缓存系统"""
        self.enabled = False
        logger.info("缓存系统已禁用")


# 全局缓存集成实例
_CACHE_INTEGRATION_INSTANCE = None

def get_cache_integration(cache_dir=None, memory_limit_mb=2000, create_new=False):
    """
    获取全局缓存集成实例(单例模式)
    
    参数:
        cache_dir (str, optional): 缓存目录路径
        memory_limit_mb (int, optional): 内存缓存大小限制(MB)
        create_new (bool, optional): 是否创建新实例而不使用单例
        
    返回:
        CacheIntegration: 缓存集成实例
    """
    global _CACHE_INTEGRATION_INSTANCE
    
    if create_new:
        # 如果需要，创建新实例
        return CacheIntegration(
            cache_dir=cache_dir,
            memory_limit_mb=memory_limit_mb
        )
    elif _CACHE_INTEGRATION_INSTANCE is None:
        # 首次创建单例
        _CACHE_INTEGRATION_INSTANCE = CacheIntegration(
            cache_dir=cache_dir,
            memory_limit_mb=memory_limit_mb
        )
    
    return _CACHE_INTEGRATION_INSTANCE


def integrate_cache_system():
    """
    集成缓存系统到回测框架
    
    这是一个便捷函数，用于在回测脚本中快速集成缓存系统。
    
    返回:
        CacheIntegration: 缓存集成实例
    """
    integration = get_cache_integration()
    if integration.integrate():
        logger.info("缓存系统已成功集成到回测框架")
    else:
        logger.warning("缓存系统集成失败")
    
    return integration


def restore_original_functions():
    """
    恢复原始函数
    
    在回测结束时调用，恢复原始的数据获取函数。
    
    返回:
        bool: 是否成功恢复
    """
    integration = get_cache_integration()
    return integration.restore()


if __name__ == "__main__":
    # 简单的测试代码
    print("缓存集成模块")
    
    # 配置基本日志
    logging.basicConfig(level=logging.INFO)
    
    # 检查所需的模块是否可用
    print(f"数据缓存管理器: {'可用' if get_data_cache_manager else '不可用'}")
    print(f"xtdata API: {'可用' if XTDATA_AVAILABLE else '不可用'}")
    
    if get_data_cache_manager and XTDATA_AVAILABLE:
        # 集成缓存系统
        integration = integrate_cache_system()
        
        # 简单测试
        if integration.patched:
            try:
                # 测试交易日期获取
                dates = xtdata.get_trading_dates('SH', '20230101', '20230131')
                print(f"获取到{len(dates)}个交易日期")
                
                # 测试市场数据获取
                data = xtdata.get_market_data(
                    stock_list=['000001.SZ', '600000.SH'],
                    field_list=['open', 'close'],
                    period='1d',
                    start_time='20230101',
                    end_time='20230110'
                )
                print(f"获取到市场数据，形状: {data.shape if hasattr(data, 'shape') else '无数据'}")
                
                # 生成性能报告
                report = integration.generate_report()
                print("\n性能报告片段:")
                print("\n".join(report.split("\n")[:10]) + "\n...")
            finally:
                # 恢复原始函数
                integration.restore()
    else:
        print("缺少必要的模块，无法进行测试") 