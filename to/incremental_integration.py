#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增量训练集成模块
将增量训练功能无缝集成到现有回测框架中，通过替换原有训练方法实现性能优化。
"""

import os
import sys
import time
import logging
import inspect
import functools
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

# 配置日志
os.makedirs('to/log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/incremental_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("incremental_integration")

# 尝试导入增量训练管理器
try:
    from incremental_training import get_incremental_manager
except ImportError:
    try:
        # 尝试从相对路径导入
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from incremental_training import get_incremental_manager
    except ImportError:
        logger.error("无法导入增量训练管理器模块，请确保incremental_training.py在正确的路径")
        get_incremental_manager = None

# 单例模式，确保只有一个增量训练集成实例
_INCREMENTAL_INTEGRATION_INSTANCE = None

def get_incremental_integration():
    """获取全局增量训练集成实例"""
    global _INCREMENTAL_INTEGRATION_INSTANCE
    if _INCREMENTAL_INTEGRATION_INSTANCE is None:
        _INCREMENTAL_INTEGRATION_INSTANCE = IncrementalIntegration()
    return _INCREMENTAL_INTEGRATION_INSTANCE

class IncrementalIntegration:
    """增量训练集成类，负责集成增量训练到现有模型"""
    
    def __init__(self, model_dir: str = 'to/models/incremental', 
                 version_control: bool = True,
                 max_versions: int = 5,
                 performance_monitoring: bool = True):
        """
        初始化增量训练集成
        
        参数:
            model_dir: 模型存储目录
            version_control: 是否启用版本控制
            max_versions: 最大保存版本数
            performance_monitoring: 是否启用性能监控
        """
        self.model_dir = Path(model_dir)
        self.version_control = version_control
        self.max_versions = max_versions
        self.performance_monitoring = performance_monitoring
        
        # 获取增量训练管理器
        if get_incremental_manager:
            self.manager = get_incremental_manager(
                model_dir=model_dir,
                version_control=version_control,
                max_versions=max_versions
            )
        else:
            self.manager = None
            logger.error("增量训练管理器初始化失败")
            return
        
        # 原始方法备份
        self.original_methods = {}
        
        # 性能统计
        self.performance_stats = {
            'original_training_time': 0,
            'incremental_training_time': 0,
            'training_count': 0,
            'time_saved': 0
        }
        
        # 尝试导入必要模块
        try:
            import xgboost
            import lightgbm
            import catboost
            import torch
            self.all_dependencies_met = True
        except ImportError as e:
            logger.warning(f"部分依赖模块未安装: {e}，某些功能可能受限")
            self.all_dependencies_met = False

        logger.info(f"增量训练集成初始化完成，模型存储目录: {self.model_dir}")
        
    def integrate(self, quant_model_class=None):
        """将增量训练集成到量化模型类中
        
        参数:
            quant_model_class: 量化模型类，如果为None则尝试从AdvancedQuantModel导入
            
        返回:
            bool: 是否成功集成
        """
        logger.info("开始集成增量训练系统...")
        
        # 如果未指定类，尝试导入
        if quant_model_class is None:
            try:
                logger.info("尝试导入默认AdvancedQuantModel类")
                from advanced_quant_model import AdvancedQuantModel
                quant_model_class = AdvancedQuantModel
                logger.info("成功导入AdvancedQuantModel类")
            except ImportError:
                logger.error("无法导入AdvancedQuantModel类")
                return False
        
        # 保存类引用
        self.quant_model_class = quant_model_class
        logger.info(f"将为{quant_model_class.__name__}类集成增量训练功能")
        
        # 先检查是否有_load_sector_data方法
        if not hasattr(quant_model_class, '_load_sector_data'):
            logger.error(f"{quant_model_class.__name__}类缺少_load_sector_data方法，无法集成")
            return False
        
        # 检查是否有sector_stocks属性
        test_instance = quant_model_class()
        try:
            test_instance._load_sector_data()
            if not hasattr(test_instance, 'sector_stocks'):
                logger.error(f"加载行业数据后，{quant_model_class.__name__}实例仍然没有sector_stocks属性")
                return False
            logger.info(f"成功加载行业数据，共 {len(test_instance.sector_stocks)} 个行业")
        except Exception as e:
            logger.error(f"测试加载行业数据失败: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 查找训练方法
        train_methods = self._find_train_methods(quant_model_class)
        if not train_methods:
            logger.warning("未找到标准训练方法")
        else:
            logger.info(f"找到 {len(train_methods)} 个标准训练方法: {', '.join(train_methods)}")
            
            # 保存原始方法
            self._save_original_methods(quant_model_class, train_methods)
            
            # 打补丁到训练方法
            if not self._patch_train_methods(quant_model_class, train_methods):
                logger.error("打补丁到标准训练方法失败")
                return False
        
        # 特别处理train_sector_models方法
        if hasattr(quant_model_class, 'train_sector_models'):
            logger.info("找到train_sector_models方法，正在集成增量训练")
            if not self._patch_train_sector_models(quant_model_class):
                logger.error("打补丁到train_sector_models方法失败")
                self.restore_original_methods(quant_model_class)
                return False
        else:
            logger.warning("未找到train_sector_models方法，无法集成行业模型增量训练")
        
        logger.info("增量训练系统集成完成")
        self.patched = True
        return True
    
    def _find_train_methods(self, cls) -> List[str]:
        """查找类中的训练相关方法"""
        train_methods = []
        
        for attr_name in dir(cls):
            if attr_name.startswith('_') and 'train' in attr_name.lower():
                # 获取方法
                method = getattr(cls, attr_name)
                
                # 检查是否为方法
                if callable(method):
                    # 检查方法签名
                    sig = inspect.signature(method)
                    params = sig.parameters
                    
                    # 查找包含features和labels参数的方法
                    param_names = [p.lower() for p in params.keys()]
                    if any(name in param_names for name in ['features', 'x', 'x_train', 'data']) and \
                       any(name in param_names for name in ['labels', 'y', 'y_train', 'target']):
                        train_methods.append(attr_name)
                        logger.info(f"发现训练方法: {attr_name}")
        
        return train_methods
    
    def _save_original_methods(self, cls, method_names: List[str]):
        """保存原始方法便于还原"""
        for method_name in method_names:
            self.original_methods[method_name] = getattr(cls, method_name)
        
        # 单独保存train_sector_models
        if hasattr(cls, 'train_sector_models'):
            self.original_methods['train_sector_models'] = cls.train_sector_models
    
    def _patch_train_methods(self, cls, method_names: List[str]) -> bool:
        """替换训练方法为增量训练版本"""
        for method_name in method_names:
            original_method = getattr(cls, method_name)
            
            @functools.wraps(original_method)
            def incremental_train_wrapper(self, *args, **kwargs):
                # 提取方法名
                current_method = inspect._findmethodbyname(self.__class__, method_name)[0]
                
                # 标识这次训练的唯一ID
                method_fullname = f"{self.__class__.__name__}.{method_name}"
                
                # 构建由参数计算的散列值作为模型ID
                # 注意：这个散列值的计算可能需要根据实际方法参数调整
                params_str = str(args) + str(sorted(kwargs.items()))
                import hashlib
                model_id = hashlib.md5(params_str.encode()).hexdigest()[:10]
                full_model_id = f"{method_fullname}_{model_id}"
                
                # 获取特征和标签参数
                # 根据方法签名可能需要调整参数的获取方式
                sig = inspect.signature(current_method)
                params = sig.parameters
                param_names = list(params.keys())
                
                # 尝试识别特征和标签参数
                X, y = None, None
                
                if len(args) >= 2:  # 假设前两个位置参数是特征和标签
                    X, y = args[0], args[1]
                else:
                    # 尝试从kwargs中提取
                    feature_params = ['features', 'x', 'x_train', 'data']
                    label_params = ['labels', 'y', 'y_train', 'target']
                    
                    for p in feature_params:
                        if p in kwargs:
                            X = kwargs[p]
                            break
                    
                    for p in label_params:
                        if p in kwargs:
                            y = kwargs[p]
                            break
                
                # 如果无法识别特征和标签，使用原始方法
                if X is None or y is None:
                    logger.warning(f"无法识别方法 {method_name} 的特征和标签参数，使用原始方法")
                    return original_method(self, *args, **kwargs)
                
                # 确定模型类型
                if 'xgboost' in str(current_method).lower():
                    model_type = 'xgboost'
                elif 'lightgbm' in str(current_method).lower():
                    model_type = 'lightgbm'
                elif 'catboost' in str(current_method).lower():
                    model_type = 'catboost'
                elif 'lnn' in str(current_method).lower() or 'liquid' in str(current_method).lower():
                    model_type = 'lnn'
                else:
                    model_type = 'xgboost'  # 默认使用XGBoost
                
                # 记录开始时间
                start_time = time.time()
                
                # 尝试使用增量训练
                try:
                    # 调用增量训练
                    model = self._incremental_train_model(model_type, full_model_id, X, y, **kwargs)
                    
                    # 记录结束时间
                    end_time = time.time()
                    
                    # 更新性能统计
                    training_time = end_time - start_time
                    self.performance_stats['incremental_training_time'] += training_time
                    self.performance_stats['training_count'] += 1
                    
                    # 记录日志
                    logger.info(f"增量训练 {model_type} 模型 {full_model_id} 完成，耗时: {training_time:.2f}秒")
                    
                    return model
                    
                except Exception as e:
                    logger.error(f"增量训练出错，回退到原始方法: {e}")
                    return original_method(self, *args, **kwargs)
            
            # 将方法绑定到类
            setattr(cls, method_name, incremental_train_wrapper)
            logger.info(f"方法 {method_name} 已被替换为增量训练版本")
        
        return True
    
    def _patch_train_sector_models(self, cls):
        """特殊处理板块模型训练方法"""
        if not hasattr(cls, 'train_sector_models'):
            logger.warning("类中没有train_sector_models方法，跳过")
            return
        
        original_method = cls.train_sector_models
        
        @functools.wraps(original_method)
        def incremental_train_sector_wrapper(self, date=None, lookback_short=30, lookback_long=90, 
                                           test_size=0.2, forward_days=5, parallel="auto"):
            """增量训练版本的板块模型训练方法"""
            # 记录开始时间
            start_time = time.time()
            
            # 确保行业数据已加载
            logger.info("确保行业数据已加载...")
            try:
                # 检查self.sectors是否存在，这是_load_sector_data设置的属性
                if not hasattr(self, 'sectors') or not self.sectors:
                    self._load_sector_data()
                    logger.info(f"成功加载行业数据，共 {len(self.sectors)} 个行业")
                else:
                    logger.info(f"行业数据已存在，共 {len(self.sectors)} 个行业")
                
                # 初始化sector_stocks属性，这是增量训练用到的属性名
                self.sector_stocks = self.sectors
            except Exception as e:
                logger.error(f"加载行业数据失败: {e}")
                raise RuntimeError(f"无法加载行业数据: {e}")
            
            # 使用对应的并行方法
            if parallel == "ray" and hasattr(self, '_train_with_ray'):
                return self._incremental_train_with_ray(date, lookback_short, lookback_long, test_size, forward_days)
            elif parallel == "process_pool" or (parallel == "auto" and sys.platform == "win32"):
                return self._incremental_train_with_process_pool(date, lookback_short, lookback_long, test_size, forward_days)
            else:
                return self._incremental_train_sequential(date, lookback_short, lookback_long, test_size, forward_days)
        
        # 添加增量训练版本的方法
        setattr(cls, '_incremental_train_sequential', self._create_incremental_train_sequential(cls))
        setattr(cls, '_incremental_train_with_process_pool', self._create_incremental_train_with_process_pool(cls))
        setattr(cls, '_incremental_train_with_ray', self._create_incremental_train_with_ray(cls))
        setattr(cls, '_incremental_train_sector_process', self._create_incremental_train_sector_process(cls))
        
        # 将方法绑定到类
        setattr(cls, 'train_sector_models', incremental_train_sector_wrapper)
        logger.info("方法 train_sector_models 已被替换为增量训练版本")
    
    def _create_incremental_train_sequential(self, cls):
        """创建序列方式的增量训练方法"""
        def _incremental_train_sequential(self, date, lookback_short, lookback_long, test_size, forward_days):
            """顺序处理各板块的增量训练"""
            logger.info(f"开始顺序增量训练板块模型，日期: {date}")
            
            # 确保行业数据已加载
            if not hasattr(self, 'sectors') or not self.sectors:
                logger.info("行业数据未加载，正在加载...")
                try:
                    self._load_sector_data()
                    logger.info(f"成功加载行业数据，共 {len(self.sectors)} 个行业")
                except Exception as e:
                    logger.error(f"加载行业数据失败: {e}")
                    raise RuntimeError(f"无法加载行业数据: {e}")
            
            # 确保行业数据存在
            if not hasattr(self, 'sectors') or not self.sectors:
                error_msg = "加载行业数据后sectors仍不存在"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 初始化sector_stocks属性，这是增量训练用到的属性名
            self.sector_stocks = self.sectors
            
            # 记录开始时间
            start_time = time.time()
            
            # 获取板块列表
            sector_names = list(self.sector_stocks.keys())
            logger.info(f"将处理 {len(sector_names)} 个板块")
            
            # 创建结果字典
            results = {}
            
            # 顺序处理每个板块
            for i, sector_name in enumerate(sector_names):
                logger.info(f"开始训练板块 [{i+1}/{len(sector_names)}]: {sector_name}")
                try:
                    result = self._incremental_train_sector_process(
                        sector_name, date, lookback_short, lookback_long, test_size, forward_days
                    )
                    results[sector_name] = result
                    logger.info(f"板块 {sector_name} 训练完成")
                except Exception as e:
                    logger.error(f"训练板块 {sector_name} 时出错: {e}")
                    logger.error(traceback.format_exc())
                    results[sector_name] = None
            
            # 记录总时间
            total_time = time.time() - start_time
            
            # 统计成功数量
            success_count = sum(1 for r in results.values() if r is not None)
            success_rate = success_count / len(sector_names) if sector_names else 0
            logger.info(f"顺序增量训练完成，成功率: {success_rate:.2%}，总耗时: {total_time:.2f}秒")
            
            return results
        
        return _incremental_train_sequential
    
    def _create_incremental_train_with_process_pool(self, cls):
        """创建进程池方式的增量训练方法"""
        def _incremental_train_with_process_pool(self, date, lookback_short, lookback_long, test_size, forward_days):
            """使用进程池并行处理各板块的增量训练"""
            import multiprocessing as mp
            from functools import partial
            
            logger.info(f"开始使用进程池并行增量训练板块模型，日期: {date}")
            
            # 确保行业数据已加载
            if not hasattr(self, 'sectors') or not self.sectors:
                logger.info("行业数据未加载，正在加载...")
                try:
                    self._load_sector_data()
                    logger.info(f"成功加载行业数据，共 {len(self.sectors)} 个行业")
                except Exception as e:
                    logger.error(f"加载行业数据失败: {e}")
                    raise RuntimeError(f"无法加载行业数据: {e}")
            
            # 确保行业数据存在
            if not hasattr(self, 'sectors') or not self.sectors:
                error_msg = "加载行业数据后sectors仍不存在"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 初始化sector_stocks属性，这是增量训练用到的属性名
            self.sector_stocks = self.sectors
            
            # 记录开始时间
            start_time = time.time()
            
            # 获取板块列表
            sector_names = list(self.sector_stocks.keys())
            logger.info(f"将使用进程池处理 {len(sector_names)} 个板块")
            
            # 创建进程池训练函数
            def process_sector(sector_name, date, lookback_short, lookback_long, test_size, forward_days):
                try:
                    model_instance = cls()
                    return sector_name, model_instance._incremental_train_sector_process(
                        sector_name, date, lookback_short, lookback_long, test_size, forward_days
                    )
                except Exception as e:
                    error_msg = f"训练板块 {sector_name} 时出错: {e}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    return sector_name, None
            
            # 创建部分函数
            worker_func = partial(
                process_sector,
                date=date,
                lookback_short=lookback_short,
                lookback_long=lookback_long,
                test_size=test_size,
                forward_days=forward_days
            )
            
            # 计算合适的进程数
            cpu_count = mp.cpu_count()
            process_count = min(cpu_count - 1, len(sector_names)) or 1
            logger.info(f"使用 {process_count} 个进程进行增量训练 (总CPU: {cpu_count})")
            
            # 创建进程池并执行
            try:
                with mp.Pool(processes=process_count) as pool:
                    # 提交任务
                    results_iter = pool.map(worker_func, sector_names)
                    
                    # 收集结果
                    results = dict(results_iter)
                    
                    # 统计成功数量
                    success_count = sum(1 for r in results.values() if r is not None)
                    success_rate = success_count / len(sector_names) if sector_names else 0
                    
                    # 计算总时间
                    total_time = time.time() - start_time
                    logger.info(f"进程池增量训练完成，成功率: {success_rate:.2%}，总耗时: {total_time:.2f}秒")
                    
                    return results
            except Exception as e:
                logger.error(f"进程池执行时出错: {e}")
                logger.error(traceback.format_exc())
                logger.warning("回退到顺序处理方式")
                return self._incremental_train_sequential(date, lookback_short, lookback_long, test_size, forward_days)
        
        return _incremental_train_with_process_pool
    
    def _create_incremental_train_with_ray(self, cls):
        """创建Ray方式的增量训练方法"""
        def _incremental_train_with_ray(self, date, lookback_short, lookback_long, test_size, forward_days):
            """使用Ray并行处理各板块的增量训练"""
            import ray
            
            logger.info(f"开始使用Ray并行增量训练板块模型，日期: {date}")
            
            # 确保行业数据已加载
            if not hasattr(self, 'sectors') or not self.sectors:
                logger.info("行业数据未加载，正在加载...")
                try:
                    self._load_sector_data()
                    logger.info(f"成功加载行业数据，共 {len(self.sectors)} 个行业")
                except Exception as e:
                    logger.error(f"加载行业数据失败: {e}")
                    raise RuntimeError(f"无法加载行业数据: {e}")
            
            # 确保行业数据存在
            if not hasattr(self, 'sectors') or not self.sectors:
                error_msg = "加载行业数据后sectors仍不存在"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 初始化sector_stocks属性，这是增量训练用到的属性名
            self.sector_stocks = self.sectors
            
            # 检查Ray是否已初始化
            ray_initialized = False
            try:
                if not ray.is_initialized():
                    logger.info("初始化Ray...")
                    ray.init(ignore_reinit_error=True)
                ray_initialized = True
                logger.info("Ray已成功初始化")
            except Exception as e:
                logger.error(f"Ray初始化失败: {e}")
                logger.error(traceback.format_exc())
                logger.warning("将使用进程池作为备选方案")
                return self._incremental_train_with_process_pool(date, lookback_short, lookback_long, test_size, forward_days)
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 获取板块列表
                sector_names = list(self.sector_stocks.keys())
                logger.info(f"将使用Ray处理 {len(sector_names)} 个板块")
                
                # 创建远程训练函数
                @ray.remote
                def train_sector_remote(sector_name, date, lookback_short, lookback_long, test_size, forward_days):
                    try:
                        model_instance = cls()
                        return model_instance._incremental_train_sector_process(
                            sector_name, date, lookback_short, lookback_long, test_size, forward_days
                        )
                    except Exception as e:
                        error_msg = f"训练板块 {sector_name} 时出错: {e}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        return None
                
                # 提交远程任务
                futures = [
                    train_sector_remote.remote(
                        sector_name, date, lookback_short, lookback_long, test_size, forward_days
                    )
                    for sector_name in sector_names
                ]
                
                # 设置超时时间（分钟）
                timeout_minutes = 60
                timeout_seconds = timeout_minutes * 60
                logger.info(f"设置任务等待超时时间为 {timeout_minutes} 分钟")
                
                # 收集结果
                results = {}
                pending = list(futures)
                
                # 使用超时机制收集结果
                while pending:
                    # 设置每批次等待时间
                    wait_time = 10.0  # 秒
                    
                    # 等待部分结果完成
                    finished, pending = ray.wait(
                        pending, 
                        num_returns=len(pending), 
                        timeout=wait_time
                    )
                    
                    if finished:
                        # 获取已完成的结果
                        for i, future in enumerate(finished):
                            try:
                                result = ray.get(future, timeout=10.0)
                                sector_name = sector_names[futures.index(future)]
                                results[sector_name] = result
                                logger.info(f"板块 {sector_name} 训练完成 [{len(results)}/{len(sector_names)}]")
                            except Exception as e:
                                logger.error(f"获取板块训练结果时出错: {e}")
                    
                    # 检查是否超时
                    elapsed_time = time.time() - start_time
                    if elapsed_time > timeout_seconds:
                        logger.warning(f"已超过设定的 {timeout_minutes} 分钟，停止等待其余任务")
                        break
                    
                    logger.info(f"已完成 {len(results)}/{len(sector_names)} 个板块，剩余 {len(pending)} 个任务")
                
                # 统计成功数量
                success_count = sum(1 for r in results.values() if r is not None)
                success_rate = success_count / len(sector_names) if sector_names else 0
                
                # 计算总时间
                total_time = time.time() - start_time
                logger.info(f"Ray并行增量训练完成，成功率: {success_rate:.2%}，总耗时: {total_time:.2f}秒")
                
                return results
                
            except Exception as e:
                logger.error(f"Ray执行过程中出错: {e}")
                logger.error(traceback.format_exc())
                logger.warning("将使用进程池作为备选方案")
                return self._incremental_train_with_process_pool(date, lookback_short, lookback_long, test_size, forward_days)
            
            finally:
                # 如果由本方法初始化Ray，则关闭Ray
                if ray_initialized and ray.is_initialized():
                    try:
                        logger.info("关闭Ray...")
                        ray.shutdown()
                        logger.info("Ray已关闭")
                    except Exception as e:
                        logger.error(f"关闭Ray时出错: {e}")
        
        return _incremental_train_with_ray
    
    def _create_incremental_train_sector_process(self, cls):
        """创建单个板块的增量训练处理方法"""
        def _incremental_train_sector_process(self, sector_name, date, lookback_short, lookback_long, test_size, forward_days):
            """处理单个板块的增量训练"""
            try:
                logger.info(f"开始处理板块 {sector_name} 的增量训练")
                
                # 获取板块股票
                stocks = self.sector_stocks.get(sector_name, [])
                if not stocks:
                    logger.warning(f"板块 {sector_name} 没有股票")
                    return False
                
                # 提取特征
                features, labels = self._extract_features(stocks, date, lookback_short, forward_days)
                if features.empty or labels.empty:
                    logger.warning(f"板块 {sector_name} 特征或标签为空")
                    return False
                
                # 使用增量训练
                model_id = f"sector_{sector_name}_short"
                model_short = self._incremental_train_model('xgboost', model_id, features, labels, test_size=test_size)
                
                if not model_short:
                    logger.warning(f"板块 {sector_name} 短期模型训练失败")
                    return False
                
                # 如果指定了长期回溯期，也训练长期模型
                model_long = None
                if lookback_long > 0:
                    features_long, labels_long = self._extract_features(stocks, date, lookback_long, forward_days)
                    if not features_long.empty and not labels_long.empty:
                        model_id = f"sector_{sector_name}_long"
                        model_long = self._incremental_train_model('xgboost', model_id, features_long, labels_long, test_size=test_size)
                
                # 存储模型到内存中
                if model_short and model_long:
                    # 合并模型
                    model = self._combine_models(model_short, model_long)
                    self.sector_models[sector_name] = {
                        'model': model,
                        'short_model': model_short,
                        'long_model': model_long,
                        'stocks': stocks,
                        'date': date
                    }
                elif model_short:
                    self.sector_models[sector_name] = {
                        'model': model_short,
                        'short_model': model_short,
                        'long_model': None,
                        'stocks': stocks,
                        'date': date
                    }
                
                logger.info(f"板块 {sector_name} 增量训练完成")
                return True
                
            except Exception as e:
                logger.error(f"处理板块 {sector_name} 增量训练时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
        
        return _incremental_train_sector_process
    
    def _incremental_train_model(self, model_type, model_id, features, labels, **kwargs):
        """调用增量训练管理器训练模型"""
        if self.manager is None:
            logger.error("增量训练管理器未初始化")
            return None
        
        try:
            # 转换为numpy数组
            import numpy as np
            import pandas as pd
            
            if isinstance(features, pd.DataFrame):
                X = features.values
            else:
                X = np.array(features)
                
            if isinstance(labels, pd.Series):
                y = labels.values
            else:
                y = np.array(labels)
            
            # 调用增量训练管理器
            model = self.manager.train(model_type, model_id, X, y, **kwargs)
            return model
            
        except Exception as e:
            logger.error(f"增量训练模型出错: {e}")
            return None
    
    def restore_original_methods(self, cls=None):
        """还原原始训练方法"""
        if not self.original_methods:
            logger.warning("没有保存的原始方法，无法还原")
            return False
        
        # 如果未指定类，尝试自动查找
        if cls is None:
            try:
                if 'advanced_quant_model' in sys.modules:
                    cls = sys.modules['advanced_quant_model'].AdvancedQuantModel
                else:
                    logger.error("未指定类，且无法自动查找AdvancedQuantModel")
                    return False
            except Exception as e:
                logger.error(f"自动查找AdvancedQuantModel失败: {e}")
                return False
        
        # 还原所有方法
        for method_name, original_method in self.original_methods.items():
            setattr(cls, method_name, original_method)
            logger.info(f"方法 {method_name} 已还原为原始版本")
        
        self.original_methods.clear()
        logger.info("所有方法已还原为原始版本")
        return True
    
    def generate_performance_report(self):
        """生成性能报告"""
        if not self.performance_stats['training_count']:
            logger.warning("没有训练记录，无法生成性能报告")
            return
        
        # 获取增量训练管理器的统计数据
        if self.manager:
            manager_stats = self.manager.get_performance_stats()
            
            # 合并统计数据
            self.performance_stats.update({
                'full_training_count': manager_stats.get('training_count', 0) - manager_stats.get('incremental_count', 0),
                'incremental_count': manager_stats.get('incremental_count', 0),
                'avg_full_time': manager_stats.get('avg_full_training_time', 0),
                'avg_incremental_time': manager_stats.get('avg_incremental_training_time', 0),
                'time_saved': manager_stats.get('time_saved', 0),
                'time_saved_pct': manager_stats.get('time_saved_pct', 0)
            })
        
        # 生成报告
        logger.info("\n===== 增量训练性能报告 =====")
        logger.info(f"总训练次数: {self.performance_stats['training_count']}")
        logger.info(f"全量训练次数: {self.performance_stats.get('full_training_count', 0)}")
        logger.info(f"增量训练次数: {self.performance_stats.get('incremental_count', 0)}")
        
        if self.performance_stats.get('avg_full_time', 0) > 0:
            logger.info(f"\n平均全量训练时间: {self.performance_stats['avg_full_time']:.2f}秒/次")
            logger.info(f"平均增量训练时间: {self.performance_stats['avg_incremental_time']:.2f}秒/次")
            
            speedup = self.performance_stats['avg_full_time'] / self.performance_stats['avg_incremental_time'] if self.performance_stats['avg_incremental_time'] > 0 else 0
            logger.info(f"平均加速比: {speedup:.2f}倍")
            
            logger.info(f"估算节省时间: {self.performance_stats['time_saved']:.2f}秒 ({self.performance_stats['time_saved_pct']:.2f}%)")
        
        logger.info("=============================")
        
        # 同时从增量训练管理器获取详细报告
        if self.manager:
            self.manager.print_performance_report()
        
        return self.performance_stats

# 自动执行集成
try:
    logger.info("尝试自动集成增量训练...")
    integration = get_incremental_integration()
    integration.integrate()
    logger.info("增量训练集成模块加载成功")
except Exception as e:
    logger.error(f"自动集成增量训练失败: {e}")
    import traceback
    logger.error(traceback.format_exc())

# 测试代码
if __name__ == "__main__":
    # 测试增量训练集成
    logger.info("测试增量训练集成功能")
    
    # 获取增量训练集成
    integration = get_incremental_integration()
    
    # 尝试集成到AdvancedQuantModel
    try:
        from advanced_quant_model import AdvancedQuantModel
        success = integration.integrate(AdvancedQuantModel)
        if success:
            logger.info("成功集成到AdvancedQuantModel")
        else:
            logger.error("集成到AdvancedQuantModel失败")
    except ImportError:
        logger.warning("无法导入AdvancedQuantModel，跳过测试")
    
    logger.info("增量训练集成模块测试完成") 