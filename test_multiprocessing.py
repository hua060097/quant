#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试修复后的模型在多进程环境中的使用
"""

import os
import sys
import logging
import pickle
import multiprocessing as mp
import numpy as np
import time

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/multiprocessing_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multiprocessing_test")

def train_model_process(process_id, model_data, X, y, result_queue):
    """在子进程中训练模型"""
    try:
        # 反序列化模型
        model = pickle.loads(model_data)
        logger.info(f"进程 {process_id} 成功反序列化模型")
        
        # 使用模型进行预测
        predictions = model.predict(X)
        logger.info(f"进程 {process_id} 成功进行预测: {predictions[:3]}...")
        
        # 返回结果
        result_queue.put((process_id, True, "预测成功"))
        
    except Exception as e:
        logger.error(f"进程 {process_id} 出错: {str(e)}")
        result_queue.put((process_id, False, str(e)))

def main():
    """主函数：测试多进程环境下的模型使用"""
    logger.info("=" * 50)
    logger.info("开始多进程测试")
    logger.info("=" * 50)
    
    try:
        # 添加'to'目录到模块搜索路径
        to_dir = os.path.join(os.getcwd(), 'to')
        sys.path.insert(0, to_dir)
        logger.info(f"添加模块路径: {to_dir}")
        
        # 导入AdvancedQuantModel
        from advanced_quant_model import AdvancedQuantModel
        
        # 创建模型实例
        logger.info("创建AdvancedQuantModel实例...")
        model = AdvancedQuantModel()
        
        # 创建测试模型
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info("创建和训练测试子模型...")
        # 创建随机数据
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # 训练两个简单模型
        short_model = RandomForestClassifier(n_estimators=5, random_state=42)
        long_model = RandomForestClassifier(n_estimators=5, random_state=21)
        
        short_model.fit(X, y)
        long_model.fit(X, y)
        
        # 创建组合模型
        logger.info("创建组合模型...")
        combined_model = model._combine_models(short_model, long_model)
        
        # 序列化模型
        logger.info("序列化模型...")
        model_data = pickle.dumps(combined_model)
        logger.info(f"序列化成功，数据大小: {len(model_data)} 字节")
        
        # 创建测试数据
        test_X = np.random.rand(10, 5)
        
        # 创建多进程
        logger.info("创建多进程...")
        processes = []
        result_queue = mp.Queue()
        
        # 启动多个进程
        num_processes = 3
        for i in range(num_processes):
            p = mp.Process(
                target=train_model_process,
                args=(i, model_data, test_X, None, result_queue)
            )
            processes.append(p)
            p.start()
            logger.info(f"进程 {i} 已启动")
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        # 收集结果
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # 分析结果
        success_count = sum(1 for _, success, _ in results if success)
        logger.info(f"总进程数: {num_processes}, 成功数: {success_count}")
        
        for process_id, success, message in results:
            if success:
                logger.info(f"进程 {process_id} 成功: {message}")
            else:
                logger.error(f"进程 {process_id} 失败: {message}")
        
        logger.info("=" * 50)
        logger.info("多进程测试完成")
        logger.info("=" * 50)
        
        return success_count == num_processes
        
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main() 