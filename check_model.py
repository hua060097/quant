#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查训练模型的内容
"""

import os
import pickle
import logging
import sys
from pathlib import Path
import pprint
import glob

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_model(model_file):
    print(f"Checking model file: {model_file}")
    try:
        file_size = os.path.getsize(model_file)
        print(f"File size: {file_size} bytes")
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            
        print(f"Model type: {type(model)}")
        
        if isinstance(model, dict):
            print(f"Model keys: {list(model.keys())}")
            
            for key, value in model.items():
                print(f"  {key}: {type(value)}")
                
                if hasattr(value, 'feature_importances_'):
                    print(f"  {key} feature importances: {value.feature_importances_[:5]}...")
                    
                if hasattr(value, 'n_features_in_'):
                    print(f"  {key} number of features: {value.n_features_in_}")
        else:
            print("Model is not a dictionary")
    except Exception as e:
        print(f"Error processing model file: {e}")
    
    print()

def main():
    """主函数"""
    # 检查TGN区块链的模型
    sector = "TGN区块链"
    date = "20230401"
    
    model_path = os.path.join("models", sector, f"{date}.pkl")
    
    if os.path.exists(model_path):
        check_model(model_path)
    else:
        logger.error(f"模型文件不存在: {model_path}")

    # 获取一些模型文件路径
    model_files = glob.glob('models/*/20230602.pkl')[:3]  # 只取前3个文件
    for model_file in model_files:
        check_model(model_file)

    # 尝试加载和测试一个特定的模型
    specific_model_file = os.path.join('models', 'TGN共享单车', '20230605.pkl')
    if os.path.exists(specific_model_file):
        check_model(specific_model_file)
    else:
        print(f"Specific model file not found: {specific_model_file}")

if __name__ == "__main__":
    logger.info("开始检查模型")
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/TGN区块链/20230401.pkl"
    
    check_model(model_path)
    logger.info("检查完成")

    print("Model check completed") 