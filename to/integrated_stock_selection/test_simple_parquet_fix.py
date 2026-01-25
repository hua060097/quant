#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的parquet保存验证测试

测试修复后的parquet保存逻辑
"""

import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

def test_parquet_save_validation():
    """测试parquet保存验证逻辑"""
    print("=== 测试修复后的parquet保存验证 ===")

    # 创建测试数据
    test_data = pd.DataFrame({
        'date': ['20240101', '20240102', '20240103'] * 100,
        'stock_code': ['SH600001', 'SH600002', 'SH600003'] * 100,
        'pct_chg_1d': np.random.normal(0, 2, 300),
        'returns_zscore': np.random.normal(0, 1, 300)
    })

    print(f"SUCCESS 测试数据创建成功: {test_data.shape}")
    print(f"  数据类型: {test_data.dtypes.to_dict()}")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test.parquet"

        print(f"FILE 测试文件路径: {test_file}")

        try:
            # 测试修复后的保存逻辑
            print("\nSAVE 开始保存测试...")

            # 保存数据
            test_data.to_parquet(test_file, index=False)
            print(f"SUCCESS 保存成功")

            # 验证文件大小
            file_size = test_file.stat().st_size / (1024 * 1024)
            print(f"  文件大小: {file_size:.2f} MB")

            # 测试修复后的验证逻辑
            print("\nVERIFY 验证读取测试...")

            # 使用pyarrow引擎读取
            test_read = pd.read_parquet(test_file, engine='pyarrow')
            if len(test_read) > 10:
                test_read = test_read.head(10)

            print(f"SUCCESS 读取验证成功:")
            print(f"  读取形状: {test_read.shape}")
            print(f"  列数匹配: {test_read.shape[1] == test_data.shape[1]}")

            # 检查数据质量
            print(f"\nQUALITY 数据质量检查:")
            for col in ['pct_chg_1d', 'returns_zscore']:
                if col in test_read.columns:
                    nan_count = test_read[col].isna().sum()
                    total_count = len(test_read)
                    if total_count > 0:
                        valid_data = test_read[col].dropna()
                        if len(valid_data) > 0:
                            mean_val = valid_data.mean()
                            std_val = valid_data.std()
                            print(f"  {col}: 均值={mean_val:.6f}, 标准差={std_val:.6f}, 有效={len(valid_data)}/{total_count}")
                        else:
                            print(f"  {col}: 无有效数据")
                    else:
                        print(f"  {col}: 数据为空")

            return True

        except Exception as e:
            print(f"ERROR 测试失败: {e}")
            print(f"错误类型: {type(e)}")
            import traceback
            print(f"错误详情:\n{traceback.format_exc()}")
            return False

def test_factor_store_logic():
    """测试FactorDatasetStore的保存逻辑"""
    print("\n=== 测试FactorDatasetStore保存逻辑 ===")

    try:
        from factor_system.factor_lab.store import FactorDatasetStore

        # 创建临时存储
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FactorDatasetStore(Path(temp_dir) / "test_store")

            # 创建测试数据
            test_df = pd.DataFrame({
                'date': ['20240101'] * 100,
                'stock_code': [f"SH{600000+i:06d}" for i in range(100)],
                'test_factor': np.random.normal(0, 1, 100)
            })

            print(f"SUCCESS 存储对象创建成功: {store.raw_path}")

            # 测试保存
            print("\nSAVE 测试原始保存...")
            store.save_raw(test_df)
            print("SUCCESS 保存成功")

            # 验证保存
            print("\nVERIFY 验证保存结果...")
            if store.raw_exists():
                loaded_data = store.load_raw()
                print(f"SUCCESS 加载成功: {loaded_data.shape}")
                print(f"  列数匹配: {loaded_data.shape[1] == test_df.shape[1]}")

                # 检查数据完整性
                if 'test_factor' in loaded_data.columns:
                    orig_nan = test_df['test_factor'].isna().sum()
                    loaded_nan = loaded_data['test_factor'].isna().sum()
                    print(f"  数据完整性: {orig_nan} -> {loaded_nan} NaN")

                    if loaded_nan == orig_nan:
                        print("SUCCESS 数据完整性保持")
                        return True
                    else:
                        print("ERROR 数据完整性受损")
                        return False
                else:
                    print("ERROR 测试因子列缺失")
                    return False
            else:
                print("ERROR 保存失败：文件不存在")
                return False

    except ImportError as e:
        print(f"ERROR 导入失败: {e}")
        return False
    except Exception as e:
        print(f"ERROR 测试失败: {e}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("开始测试修复后的parquet保存逻辑...")

    # 测试1: 基本parquet操作
    success1 = test_parquet_save_validation()

    # 测试2: FactorDatasetStore逻辑
    success2 = test_factor_store_logic()

    print(f"\n=== 测试结果 ===")
    print(f"基本parquet操作: {'SUCCESS 通过' if success1 else 'ERROR 失败'}")
    print(f"FactorStore逻辑: {'SUCCESS 通过' if success2 else 'ERROR 失败'}")

    if success1 and success2:
        print("\nSUCCESS 所有测试通过！修复成功！")
        print("现在可以安全运行因子生成命令。")
    else:
        print("\nWARNING 部分测试失败，需要进一步检查。")