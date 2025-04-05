#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
向量化特征集成模块
用于将向量化特征提取器集成到现有回测系统中
"""

import os
import sys
import logging
import time
import functools
from pathlib import Path

# 配置日志
os.makedirs('to/log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/vectorized_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vectorized_integration")

# 添加当前目录到路径，确保能够导入依赖模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入向量化特征提取模块
try:
    from vectorized_features import VectorizedFeatureExtractor, create_vectorized_extractor
    logger.info("成功导入向量化特征提取模块")
except ImportError as e:
    logger.error(f"导入向量化特征提取模块失败: {e}")
    logger.error("请确保vectorized_features.py文件位于当前目录下")
    raise

# 尝试导入基础特征提取相关模块
try:
    from xtquant import xtdata
    logger.info("成功导入xtquant.xtdata模块")
except ImportError:
    logger.warning("无法导入xtquant.xtdata模块，某些功能可能受限")

class VectorizedIntegration:
    """向量化特征集成类，用于替换原始特征提取方法"""
    
    def __init__(self, benchmark_mode=False):
        """
        初始化向量化特征集成
        
        参数:
            benchmark_mode: 是否启用基准测试模式（同时运行原始方法和向量化方法进行对比）
        """
        self.benchmark_mode = benchmark_mode
        self._original_extract_features = None
        self._patched = False
        self._vectorized_extractor = create_vectorized_extractor()
        self._performance_data = {
            'original_calls': 0,
            'original_time': 0,
            'vectorized_calls': 0,
            'vectorized_time': 0,
            'start_time': time.time()
        }
        
        logger.info(f"向量化特征集成初始化完成，基准测试模式: {benchmark_mode}")
    
    def integrate(self, target_class=None, target_methods=None, debug_mode=False):
        """
        集成向量化特征提取到现有系统
        
        参数:
            target_class: 目标类，默认为AdvancedQuantModel
            target_methods: 要替换的目标方法列表，默认为['_extract_features']
            debug_mode: 是否启用调试模式，输出更多日志信息
        """
        if self._patched:
            logger.warning("向量化特征提取已经集成，不需要重复操作")
            return True
        
        if debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("已启用调试模式")
        
        # 如果未指定target_methods，则默认替换_extract_features方法
        if target_methods is None:
            target_methods = ['_extract_features']
            
        if debug_mode:
            logger.debug(f"目标方法列表: {target_methods}")
        
        # 导入目标类
        try:
            # 如果未指定target_class，则默认使用AdvancedQuantModel
            if target_class is None:
                from advanced_quant_model import AdvancedQuantModel
                target_class = AdvancedQuantModel
                
            logger.info(f"目标类: {target_class.__name__}")
            
            # 检查并替换每个目标方法
            for method_name in target_methods:
                if hasattr(target_class, method_name):
                    # 保存原始方法
                    if method_name == '_extract_features':
                        self._original_extract_features = getattr(target_class, method_name)
                        # 替换为向量化版本
                        setattr(target_class, method_name, self._vectorized_extract_features)
                        logger.info(f"成功替换{target_class.__name__}.{method_name}方法")
                    else:
                        logger.warning(f"目前只支持替换_extract_features方法，跳过{method_name}")
                else:
                    logger.warning(f"{target_class.__name__}没有{method_name}方法")
            
            self._patched = True
            logger.info("向量化特征提取已成功集成到现有系统")
            return True
            
        except ImportError as e:
            logger.error(f"导入目标类失败: {e}")
            return False
        except Exception as e:
            logger.error(f"集成向量化特征提取时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _vectorized_extract_features(self, stocks, date, lookback_days, forward_days=5):
        """
        向量化特征提取方法，替代原始的特征提取方法
        
        参数与原始方法保持一致，确保兼容性
        """
        start_time = time.time()
        self._performance_data['vectorized_calls'] += 1
        
        try:
            # 记录开始时间和基本参数
            logger.info(f"向量化特征提取: {len(stocks)}只股票, 日期={date}, 回溯={lookback_days}天")
            
            # 执行基准测试模式
            if self.benchmark_mode and self._original_extract_features:
                original_start = time.time()
                original_features, original_labels = self._original_extract_features(
                    self, stocks, date, lookback_days, forward_days
                )
                original_time = time.time() - original_start
                self._performance_data['original_time'] += original_time
                self._performance_data['original_calls'] += 1
                logger.info(f"原始方法执行耗时: {original_time:.4f}秒")
            
            # 检查参数有效性
            if not stocks:
                logger.warning("股票列表为空，无法提取特征")
                return None, None
            
            # 获取交易日期
            if not hasattr(self, 'trading_dates') or not self.trading_dates:
                logger.warning("交易日期列表为空，尝试加载交易日期")
                if hasattr(self, 'load_trading_dates'):
                    if not self.load_trading_dates():
                        logger.error("无法加载交易日期列表，特征提取失败")
                        return None, None
                else:
                    logger.error("实例缺少load_trading_dates方法")
                    return None, None
            
            # 获取最接近的交易日
            if hasattr(self, '_get_closest_trading_date'):
                current_date = self._get_closest_trading_date(date)
                if current_date != date:
                    logger.info(f"日期 {date} 不是交易日，使用最接近的交易日 {current_date}")
                    date = current_date
            
            # 计算开始日期
            if hasattr(self, '_get_start_date'):
                start_date = self._get_start_date(date, lookback_days)
                if not start_date:
                    logger.error(f"无法确定开始日期，当前日期: {date}, 回溯天数: {lookback_days}")
                    return None, None
            else:
                # 简单实现开始日期计算
                date_idx = self.trading_dates.index(date)
                start_idx = max(0, date_idx - lookback_days)
                start_date = self.trading_dates[start_idx]
            
            # 计算结束日期
            date_idx = self.trading_dates.index(date)
            if date_idx + forward_days < len(self.trading_dates):
                end_date = self.trading_dates[date_idx + forward_days]
            else:
                end_date = self.trading_dates[-1]
            
            logger.info(f"获取市场数据: 日期范围={start_date}至{end_date}")
            
            # 获取市场数据
            try:
                fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
                market_data = xtdata.get_market_data(
                    field_list=fields, 
                    stock_list=stocks, 
                    period='1d',
                    start_time=start_date,
                    end_time=end_date
                )
                
                if market_data is None or not all(field in market_data for field in fields[:5]):
                    logger.error("获取市场数据失败或数据缺失关键字段")
                    return None, None
                
                # 检查数据有效性
                valid_stocks = self._check_data_validity(market_data, stocks, fields)
                if not valid_stocks:
                    logger.warning("没有有效的股票数据")
                    return None, None
                
                # 调用向量化特征提取器
                features_matrix, labels_vector, final_valid_stocks = self._vectorized_extractor.extract_features_batch(
                    market_data=market_data,
                    stocks=valid_stocks,
                    date_idx=date_idx,
                    lookback_days=lookback_days,
                    forward_days=forward_days
                )
                
                # 检查结果有效性
                if features_matrix is None or labels_vector is None or not final_valid_stocks:
                    logger.warning("特征提取结果无效")
                    return None, None
                
                # 记录结果统计
                vectorized_time = time.time() - start_time
                self._performance_data['vectorized_time'] += vectorized_time
                
                # 如果在基准测试模式下，比较结果
                if self.benchmark_mode and 'original_features' in locals():
                    self._compare_results(
                        original_features, original_labels, 
                        features_matrix, labels_vector,
                        original_time, vectorized_time
                    )
                
                logger.info(f"向量化特征提取完成: {features_matrix.shape}, 有效股票数量={len(final_valid_stocks)}, 耗时={vectorized_time:.4f}秒")
                return features_matrix, labels_vector
                
            except Exception as e:
                logger.error(f"获取市场数据或特征提取过程中出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None, None
                
        except Exception as e:
            logger.error(f"向量化特征提取过程中发生未处理异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 如果向量化方法失败，尝试回退到原始方法
            if self._original_extract_features:
                logger.warning("向量化方法失败，回退到原始特征提取方法")
                return self._original_extract_features(self, stocks, date, lookback_days, forward_days)
            else:
                return None, None
    
    def _check_data_validity(self, market_data, stocks, fields):
        """检查市场数据的有效性，返回有效的股票列表"""
        valid_stocks = []
        
        # 获取第一个日期，用于检查
        first_date = list(market_data['close'].index)[0] if 'close' in market_data else None
        
        if first_date is None:
            logger.error("市场数据中没有收盘价数据")
            return []
        
        # 检查每只股票的数据完整性
        for stock in stocks:
            is_valid = True
            
            # 检查每个必要字段
            for field in fields[:5]:  # 前5个字段是必须的
                if field not in market_data or stock not in market_data[field].loc[first_date]:
                    is_valid = False
                    break
            
            if is_valid:
                valid_stocks.append(stock)
        
        logger.info(f"有效股票数量: {len(valid_stocks)} / {len(stocks)}")
        return valid_stocks
    
    def _compare_results(self, original_features, original_labels, 
                         vectorized_features, vectorized_labels,
                         original_time, vectorized_time):
        """比较原始方法和向量化方法的结果"""
        import numpy as np
        
        try:
            # 检查原始结果是否有效
            if original_features is None or original_labels is None:
                logger.warning("原始方法结果无效，无法进行比较")
                return
            
            # 计算形状差异
            shape_diff = False
            if original_features.shape != vectorized_features.shape:
                logger.warning(f"特征矩阵形状不同: 原始={original_features.shape}, 向量化={vectorized_features.shape}")
                shape_diff = True
            
            if not shape_diff:
                # 计算特征差异
                feature_diff = np.abs(original_features - vectorized_features).mean()
                logger.info(f"特征平均绝对差异: {feature_diff:.6f}")
                
                # 计算标签差异
                label_match = np.mean(original_labels == vectorized_labels)
                logger.info(f"标签匹配率: {label_match:.2%}")
            
            # 计算性能提升
            speedup = original_time / vectorized_time if vectorized_time > 0 else float('inf')
            logger.info(f"性能提升: {speedup:.2f}倍 (原始={original_time:.4f}秒, 向量化={vectorized_time:.4f}秒)")
            
        except Exception as e:
            logger.error(f"比较结果时出错: {e}")
    
    def restore(self):
        """恢复原始特征提取方法"""
        if not self._patched or self._original_extract_features is None:
            logger.warning("向量化特征提取尚未集成或原始方法未保存，无需恢复")
            return
        
        try:
            from advanced_quant_model import AdvancedQuantModel
            
            # 恢复原始方法
            AdvancedQuantModel._extract_features = self._original_extract_features
            logger.info("已恢复原始特征提取方法")
            
            # 标记为未集成
            self._patched = False
            
        except ImportError as e:
            logger.error(f"导入AdvancedQuantModel失败: {e}")
    
    def get_performance_report(self):
        """获取性能报告数据"""
        if not self._patched:
            logger.warning("向量化特征提取尚未集成，无性能数据")
            return {}
        
        # 计算运行时间
        runtime = time.time() - self._performance_data['start_time']
        
        # 计算平均时间
        avg_original_time = (self._performance_data['original_time'] / 
                            self._performance_data['original_calls'] 
                            if self._performance_data['original_calls'] > 0 else 0)
        
        avg_vectorized_time = (self._performance_data['vectorized_time'] / 
                              self._performance_data['vectorized_calls'] 
                              if self._performance_data['vectorized_calls'] > 0 else 0)
        
        # 计算加速比
        speedup = (avg_original_time / avg_vectorized_time 
                  if avg_vectorized_time > 0 else float('inf'))
        
        # 组装报告
        report = {
            'runtime_seconds': runtime,
            'original_calls': self._performance_data['original_calls'],
            'vectorized_calls': self._performance_data['vectorized_calls'],
            'total_original_time': self._performance_data['original_time'],
            'total_vectorized_time': self._performance_data['vectorized_time'],
            'avg_original_time': avg_original_time,
            'avg_vectorized_time': avg_vectorized_time,
            'speedup': speedup,
            'time_saved_pct': (1 - avg_vectorized_time / avg_original_time) * 100 if avg_original_time > 0 else 0
        }
        
        return report
    
    def print_performance_report(self):
        """打印性能报告"""
        report = self.get_performance_report()
        
        if not report:
            return
        
        logger.info("\n===== 向量化特征提取性能报告 =====")
        logger.info(f"总运行时间: {report['runtime_seconds']:.2f}秒")
        logger.info(f"调用次数: 原始={report['original_calls']}, 向量化={report['vectorized_calls']}")
        
        if report['original_calls'] > 0:
            logger.info(f"平均耗时: 原始={report['avg_original_time']:.4f}秒, 向量化={report['avg_vectorized_time']:.4f}秒")
            logger.info(f"加速比: {report['speedup']:.2f}倍")
            logger.info(f"节省时间: {report['time_saved_pct']:.2f}%")
        
        logger.info(f"总耗时: 原始={report['total_original_time']:.2f}秒, 向量化={report['total_vectorized_time']:.2f}秒")
        logger.info("====================================")

# 创建全局向量化集成实例
_GLOBAL_VECTORIZED_INTEGRATION = None

def get_vectorized_integration(benchmark_mode=False, create_new=True):
    """
    获取向量化集成实例
    
    参数:
        benchmark_mode: 是否启用基准测试模式
        create_new: 是否创建新实例，默认为True
    
    返回:
        VectorizedIntegration实例
    """
    global _GLOBAL_VECTORIZED_INTEGRATION
    
    # 如果要求新实例或全局实例不存在，则创建新实例
    if create_new or _GLOBAL_VECTORIZED_INTEGRATION is None:
        integration = VectorizedIntegration(benchmark_mode=benchmark_mode)
        # 如果全局实例不存在，同时将新实例保存为全局实例
        if _GLOBAL_VECTORIZED_INTEGRATION is None:
            _GLOBAL_VECTORIZED_INTEGRATION = integration
        return integration
    
    # 返回现有全局实例
    return _GLOBAL_VECTORIZED_INTEGRATION

# 测试代码
if __name__ == "__main__":
    # 简单测试代码
    vectorized_integration = get_vectorized_integration(benchmark_mode=True)
    
    # 集成向量化特征提取
    vectorized_integration.integrate(
        target_class=None,
        target_methods=['_extract_features'],
        debug_mode=True
    )
    
    # 打印原始特征提取方法信息
    logger.info(f"原始特征提取方法: {vectorized_integration._original_extract_features}")
    
    # 这里可以添加更多测试代码
    logger.info("向量化特征集成模块加载成功") 