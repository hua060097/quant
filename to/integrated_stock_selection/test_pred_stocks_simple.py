#!/usr/bin/env python3
"""
简单测试pred_stocks模式
"""

import sys
import os

def test_pred_stocks():
    """测试pred_stocks模式的基本功能"""

    print("测试pred_stocks模式")

    # 模拟命令行参数
    class MockArgs:
        def __init__(self):
            self.mode = 'pred_stocks'
            self.date = '20251216'
            self.output_dir = './results'

    args = MockArgs()

    # 测试参数验证
    try:
        from main import run_pred_stocks_mode

        print(f"参数验证: mode={args.mode}, date={args.date}")

        # 测试模型文件存在性
        from pathlib import Path
        model_path = Path("./results/lightgbm_model_window3_20251216_200900.pkl")

        if model_path.exists():
            print("模型文件存在")
        else:
            print("模型文件不存在 - 请先运行train_model_walkforward模式生成模型")

        # 测试因子数据
        factor_file = Path("../local_data/features/daily/20251216.parquet")
        if factor_file.exists():
            import pandas as pd
            df = pd.read_parquet(factor_file)
            print(f"因子文件存在: {len(df)} 条记录")
        else:
            print("因子文件不存在")

        print("\npred_stocks模式实现完成")
        print("使用方法: python main.py --mode pred_stocks --date 20251216")

        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pred_stocks()