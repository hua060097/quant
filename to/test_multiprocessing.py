#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多进程池序列化功能
"""
import os
import sys
import time
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("test_multiprocessing")

def worker_function(data):
    """测试在子进程中执行的函数"""
    import os
    import time
    
    # 创建一个子进程专用的logger
    worker_logger = logging.getLogger(f"worker_{os.getpid()}")
    worker_logger.setLevel(logging.INFO)
    
    # 获取进程信息
    pid = os.getpid()
    
    # 模拟处理时间
    worker_logger.info(f"进程 {pid} 开始处理数据: {data}")
    time.sleep(1)
    
    try:
        # 创建一个简单的模型（不包含RLock）
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # 创建一个简单的数据集
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)
        
        # 创建并训练模型
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        # 返回进程ID、数据和模型
        worker_logger.info(f"进程 {pid} 成功训练模型")
        result = {
            'pid': pid,
            'data': data,
            'model': model,
            'accuracy': model.score(X, y)
        }
        return result
        
    except Exception as e:
        worker_logger.error(f"进程 {pid} 处理数据时出错: {e}")
        print(traceback.format_exc())
        return {
            'pid': pid,
            'data': data,
            'error': str(e)
        }

def test_multiprocessing():
    """测试多进程池功能"""
    print("==== 开始测试多进程池序列化 ====")
    
    try:
        # 创建进程池
        num_workers = 2
        data_items = list(range(4))
        
        logger.info(f"创建 {num_workers} 个进程的进程池")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交任务
            logger.info(f"提交 {len(data_items)} 个任务到进程池")
            futures = [executor.submit(worker_function, item) for item in data_items]
            
            # 获取结果
            results = []
            for i, future in enumerate(futures):
                try:
                    logger.info(f"获取任务 {i} 的结果")
                    result = future.result(timeout=10)
                    logger.info(f"任务 {i} 完成: 进程 {result.get('pid')} 处理了数据 {result.get('data')}")
                    
                    # 检查模型
                    if 'model' in result:
                        model = result['model']
                        logger.info(f"模型类型: {type(model)}")
                        logger.info(f"模型精度: {result.get('accuracy'):.4f}")
                    
                    results.append(result)
                except Exception as e:
                    logger.error(f"获取任务 {i} 结果时出错: {e}")
                    print(traceback.format_exc())
        
        logger.info(f"所有任务完成，获得 {len(results)} 个结果")
        print("==== 多进程池序列化测试完成 ====")
        return True
    
    except Exception as e:
        logger.error(f"多进程池测试失败: {e}")
        print(traceback.format_exc())
        print("==== 多进程池测试失败 ====")
        return False

if __name__ == "__main__":
    # 确保使用独立进程启动方式
    multiprocessing.set_start_method('spawn', force=True)
    test_multiprocessing() 