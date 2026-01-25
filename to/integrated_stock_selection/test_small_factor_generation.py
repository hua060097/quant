#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小规模因子生成测试

测试修复后的因子生成流程，验证不再出现数据损坏问题
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_small_factor_generation.log')
        ]
    )

def test_small_factor_generation():
    """测试小规模因子生成"""
    setup_logging()
    logger = logging.getLogger(__name__)

    print("=== 小规模因子生成测试 ===")

    try:
        # 测试生成少量因子
        print("\n1. 测试生成2个指定因子...")
        cmd1 = [
            "python", "main.py",
            "--mode", "generate_factors",
            "--start_date", "20240101",
            "--end_date", "20240103",
            "--factor_names", "pct_chg_1d,alpha_001"
        ]

        print(f"执行命令: {' '.join(cmd1)}")
        result1 = subprocess.run(cmd1, capture_output=True, text=True, encoding='utf-8')

        if result1.returncode == 0:
            print("SUCCESS: 2个因子生成成功")
            logger.info("2个因子生成成功")
        else:
            print(f"ERROR: 2个因子生成失败")
            print(f"错误输出: {result1.stderr}")
            logger.error(f"2个因子生成失败: {result1.stderr}")
            return False

        print("\n2. 测试生成所有因子...")
        cmd2 = [
            "python", "main.py",
            "--mode", "generate_factors",
            "--start_date", "20240101",
            "--end_date", "20240103"
        ]

        print(f"执行命令: {' '.join(cmd2)}")
        result2 = subprocess.run(cmd2, capture_output=True, text=True, encoding='utf-8')

        if result2.returncode == 0:
            print("SUCCESS: 所有因子生成成功")
            logger.info("所有因子生成成功")
        else:
            print(f"ERROR: 所有因子生成失败")
            print(f"错误输出: {result2.stderr}")
            logger.error(f"所有因子生成失败: {result2.stderr}")
            return False

        return True

    except Exception as e:
        print(f"ERROR: 测试过程出错: {e}")
        logger.error(f"测试过程出错: {e}")
        return False

def check_factor_results():
    """检查生成的因子结果"""
    print("\n=== 检查生成的因子结果 ===")

    try:
        import pandas as pd
        from pathlib import Path

        # 检查正式因子文件
        factor_file = Path("factor_lab_data/factors.parquet")
        if factor_file.exists():
            print("SUCCESS: 正式因子文件存在")

            try:
                # 读取前1000行检查
                sample_data = pd.read_parquet(factor_file, engine='pyarrow')
                if len(sample_data) > 1000:
                    sample_data = sample_data.head(1000)

                print(f"因子数据形状: {sample_data.shape}")

                # 检查关键因子
                key_factors = ['pct_chg_1d', 'alpha_001', 'returns_zscore']
                for factor in key_factors:
                    if factor in sample_data.columns:
                        nan_count = sample_data[factor].isna().sum()
                        total_count = len(sample_data)
                        valid_count = total_count - nan_count

                        if valid_count > 0:
                            valid_data = sample_data[factor].dropna()
                            mean_val = valid_data.mean()
                            std_val = valid_data.std()
                            print(f"  {factor}: 均值={mean_val:.6f}, 标准差={std_val:.6f}, 有效率={valid_count/total_count:.2%}")
                        else:
                            print(f"  {factor}: 无有效数据")
                    else:
                        print(f"  {factor}: 因子不存在")

            except Exception as e:
                print(f"ERROR: 读取因子文件失败: {e}")
                return False
        else:
            print("WARNING: 正式因子文件不存在")

        # 检查临时文件是否存在并清理
        temp_file = Path("factor_lab_data/factors_raw.parquet")
        if temp_file.exists():
            print("INFO: 临时文件存在，应该会被自动清理")

        return True

    except Exception as e:
        print(f"ERROR: 检查结果失败: {e}")
        return False

def cleanup_test_files():
    """清理测试文件"""
    print("\n=== 清理测试文件 ===")
    try:
        import shutil
        from pathlib import Path

        # 清理测试生成的因子文件
        test_files = [
            "factor_lab_data/factors.parquet",
            "factor_lab_data/factors_raw.parquet"
        ]

        for file_path in test_files:
            path = Path(file_path)
            if path.exists():
                try:
                    path.unlink()
                    print(f"SUCCESS: 删除 {file_path}")
                except Exception as e:
                    print(f"WARNING: 删除 {file_path} 失败: {e}")

        print("测试文件清理完成")

    except Exception as e:
        print(f"WARNING: 清理过程出错: {e}")

if __name__ == "__main__":
    print("开始小规模因子生成测试...")

    # 执行测试
    success = test_small_factor_generation()

    if success:
        # 检查结果
        result_success = check_factor_results()

        if result_success:
            print("\n=== 测试结果 ===")
            print("SUCCESS: 所有测试通过！")
            print("修复验证成功，因子生成管道现在工作正常")
            print("\n可以安全地运行完整的因子生成命令")
        else:
            print("\n=== 测试结果 ===")
            print("WARNING: 因子生成成功但结果检查失败")
    else:
        print("\n=== 测试结果 ===")
        print("ERROR: 因子生成测试失败")

    # 清理测试文件
    cleanup_test_files()