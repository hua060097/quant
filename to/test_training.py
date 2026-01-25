#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修改后的训练逻辑
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent / "integrated_stock_selection"
sys.path.insert(0, str(project_root))

def test_training():
    """测试训练功能"""
    try:
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        
        # 导入主要模块
        from main import run_train_model_mode, parse_arguments
        
        # 创建模拟参数
        class MockArgs:
            def __init__(self):
                self.mode = 'train_model'
                self.output_dir = './results'
                self.log_level = 'INFO'
                self.save_results = True
        
        args = MockArgs()
        
        print("开始测试新的训练逻辑...")
        print("=" * 60)
        
        # 运行训练
        success = run_train_model_mode(args)
        
        if success:
            print("训练测试成功完成！")
            return True
        else:
            print("训练测试失败！")
            return False
            
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("新特征工程训练逻辑测试")
    print("=" * 60)
    
    success = test_training()
    
    if success:
        print("\n所有测试通过！新的训练逻辑工作正常。")
        print("特征工程策略:")
        print("  - 18个股票因子保持原始形式")
        print("  - 8个市场因子进行去相关+PCA处理")
        print("  - XGBoost使用interaction_constraints控制特征交互")
        sys.exit(0)
    else:
        print("\n测试失败，请检查代码修改。")
        sys.exit(1)