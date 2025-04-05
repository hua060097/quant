#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试_combine_models方法在多进程环境下的序列化
"""

import logging
import os
import sys
import time
import multiprocessing as mp
import pickle

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('to/log/model_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('test_combine')

def worker_function(model_obj):
    """测试序列化和执行一个模型对象"""
    try:
        logger.info(f"Worker进程PID: {os.getpid()}")
        logger.info(f"接收到模型对象: {type(model_obj)}")
        
        # 尝试使用模型进行一些操作
        if hasattr(model_obj, 'predict'):
            logger.info("模型有predict方法")
            
        if hasattr(model_obj, 'predict_proba'):
            logger.info("模型有predict_proba方法")
        
        if hasattr(model_obj, 'short_model'):
            logger.info("模型有short_model属性")
            
        # 尝试再次序列化
        pickled_data = pickle.dumps(model_obj)
        logger.info(f"模型重新序列化成功，大小: {len(pickled_data)} 字节")
        
        return True, "模型序列化和传递测试成功"
    except Exception as e:
        logger.error(f"Worker进程出错: {str(e)}")
        return False, str(e)

def test_multiprocessing():
    """测试在多进程环境下模型的序列化和传递"""
    logger.info("=" * 50)
    logger.info("开始多进程序列化测试")
    logger.info("=" * 50)
    
    try:
        # 添加'to'目录到模块搜索路径
        to_dir = os.path.join(os.getcwd(), 'to')
        sys.path.insert(0, to_dir)
        logger.info(f"添加模块路径: {to_dir}")
        
        # 导入AdvancedQuantModel
        from advanced_quant_model import AdvancedQuantModel
        
        # 创建模型实例
        logger.info("创建模型实例...")
        model = AdvancedQuantModel()
        
        # 模拟创建两个简单的模型
        from sklearn.ensemble import RandomForestClassifier
        
        # 创建一些简单的模型
        logger.info("创建测试模型...")
        short_model = RandomForestClassifier(n_estimators=10, random_state=42)
        long_model = RandomForestClassifier(n_estimators=5, random_state=21)
        
        # 使用随机数据训练模型
        import numpy as np
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        logger.info("训练测试模型...")
        short_model.fit(X, y)
        long_model.fit(X, y)
        
        # 创建组合模型
        logger.info("创建组合模型...")
        combined_model = model._combine_models(short_model, long_model)
        logger.info(f"组合模型类型: {type(combined_model)}")
        
        # 测试组合模型的方法
        logger.info("测试组合模型方法...")
        test_pred = combined_model.predict(X[:5])
        logger.info(f"预测结果: {test_pred}")
        
        # 测试序列化
        logger.info("测试序列化...")
        pickled_data = pickle.dumps(combined_model)
        logger.info(f"序列化成功，数据大小: {len(pickled_data)} 字节")
        
        # 使用multiprocessing测试
        logger.info("创建进程池...")
        with mp.Pool(processes=2) as pool:
            logger.info("启动Worker进程...")
            result = pool.apply_async(worker_function, (combined_model,))
            
            logger.info("等待进程完成...")
            success, message = result.get(timeout=60)
            
            if success:
                logger.info(f"多进程测试成功: {message}")
            else:
                logger.error(f"多进程测试失败: {message}")
        
        logger.info("=" * 50)
        logger.info("多进程序列化测试完成")
        logger.info("=" * 50)
        
        return success
        
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_multiprocessing() 