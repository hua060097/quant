#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试_combine_models方法的序列化
"""

import os
import sys
import pickle
import logging

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/test_pickle.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_pickle")

def main():
    """测试序列化功能"""
    logger.info("开始测试序列化功能")
    
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
        import numpy as np
        
        logger.info("创建测试子模型...")
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
        
        # 测试序列化
        logger.info("测试序列化...")
        try:
            pickled_data = pickle.dumps(combined_model)
            logger.info(f"序列化成功，数据大小: {len(pickled_data)} 字节")
            
            # 反序列化
            logger.info("测试反序列化...")
            unpickled_model = pickle.loads(pickled_data)
            logger.info(f"反序列化成功，对象类型: {type(unpickled_model)}")
            
            # 尝试使用反序列化的模型
            logger.info("使用反序列化后的模型预测...")
            test_x = np.random.rand(5, 5)
            predictions = unpickled_model.predict(test_x)
            logger.info(f"预测结果: {predictions}")
            
            logger.info("序列化测试通过!")
            return True
        
        except Exception as e:
            logger.error(f"序列化失败: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main() 