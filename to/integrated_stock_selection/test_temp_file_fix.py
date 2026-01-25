#!/usr/bin/env python3
"""
测试临时文件数据损坏修复

验证修复后的临时文件保存和加载机制是否能正确处理大量因子的数据
"""

import logging
import sys
import os
from pathlib import Path

# 添加当前目录到路径
sys.path.append('.')

def setup_logging():
    """设置详细日志"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_temp_file_fix.log')
        ]
    )

def test_temp_file_merge():
    """测试临时文件合并机制"""
    setup_logging()
    logger = logging.getLogger(__name__)

    print("=== 测试临时文件合并机制修复 ===")

    try:
        from factor_system.factor_lab.store import FactorDatasetStore
        import pandas as pd
        import numpy as np

        # 创建测试存储
        test_store_path = "test_factor_lab_data"
        store = FactorDatasetStore(test_store_path)

        # 清理旧测试文件
        if store.raw_exists():
            store.raw_path.unlink()
            logger.info("清理旧测试文件")

        print(f"✅ 存储对象创建成功: {store.raw_path}")

        # 模拟分批生成因子（类似生成所有76个因子的过程）
        n_stocks = 1000
        n_dates = 30
        total_factors = 76  # 模拟所有因子

        # 创建测试数据索引
        dates = pd.date_range('20240101', periods=n_dates)
        stock_codes = [f"SH{600000+i:06d}" for i in range(n_stocks)]
        index_data = []
        for date in dates:
            for stock in stock_codes:
                index_data.append((date.strftime('%Y%m%d'), stock))

        test_df = pd.DataFrame(index_data, columns=['date', 'stock_code'])
        print(f"✅ 测试索引创建成功: {test_df.shape}")

        # 分批模拟因子生成过程
        batch_size = 10  # 每批10个因子
        factor_names = [f"alpha_{i:03d}" for i in range(1, total_factors + 1)]

        print(f"\n📊 开始分批生成 {total_factors} 个因子，每批 {batch_size} 个")

        for batch_idx in range(0, total_factors, batch_size):
            batch_factors = factor_names[batch_idx:batch_idx + batch_size]
            print(f"\n--- 批次 {batch_idx//batch_size + 1}: {batch_factors} ---")

            # 创建批次因子数据
            batch_data = test_df.copy()
            for factor_name in batch_factors:
                # 生成有意义的随机因子数据
                np.random.seed(hash(factor_name) % 10000)  # 确保可重现
                factor_values = np.random.normal(0, 1, len(batch_data))

                # 模拟一些真实世界的因子特性
                if 'alpha_001' in factor_name:
                    # 模拟价格变化因子（类似pct_chg_1d）
                    factor_values = np.random.normal(0, 0.02, len(batch_data)) * 100
                elif 'alpha_002' in factor_name:
                    # 模拟收益率因子
                    factor_values = np.random.normal(0.001, 0.015, len(batch_data))
                elif 'returns_zscore' in factor_name:
                    # 模拟标准化收益率
                    factor_values = np.random.standard_normal(len(batch_data))

                batch_data[factor_name] = factor_values

            print(f"  批次数据形状: {batch_data.shape}")

            # 检查批次数据质量
            batch_factor_cols = [c for c in batch_data.columns if c not in store.index_columns]
            batch_nan_count = batch_data[batch_factor_cols].isna().sum().sum()
            total_values = len(batch_data) * len(batch_factor_cols)
            batch_nan_ratio = batch_nan_count / total_values if total_values > 0 else 0

            print(f"  批次质量 - NaN比率: {batch_nan_ratio:.4f} ({batch_nan_count}/{total_values})")

            # 保存到临时文件（使用修复后的逻辑）
            try:
                logger.info(f"[测试] 开始保存批次 {batch_idx//batch_size + 1}")
                store.merge_raw_incremental(batch_data)

                # 验证保存后的文件
                if store.raw_exists():
                    file_size = store.raw_path.stat().st_size / (1024 * 1024)  # MB
                    print(f"  ✅ 保存成功 - 文件大小: {file_size:.2f} MB")

                    # 验证加载的数据
                    loaded_data = store.load_raw()
                    print(f"  ✅ 加载验证成功 - 形状: {loaded_data.shape}")

                    # 检查加载后的数据质量
                    loaded_factor_cols = [c for c in loaded_data.columns if c not in store.index_columns]
                    loaded_nan_count = loaded_data[loaded_factor_cols].isna().sum().sum()
                    loaded_total_values = len(loaded_data) * len(loaded_factor_cols)
                    loaded_nan_ratio = loaded_nan_count / loaded_total_values if loaded_total_values > 0 else 0

                    print(f"  加载质量 - NaN比率: {loaded_nan_ratio:.4f} ({loaded_nan_count}/{loaded_total_values})")

                    # 检查是否出现数据损坏
                    if loaded_nan_count > batch_nan_count:
                        print(f"  ❌ 数据损坏警告：NaN数量增加 ({batch_nan_count} -> {loaded_nan_count})")

                        # 检查具体哪些因子出现问题
                        problem_factors = []
                        for col in loaded_factor_cols[:10]:  # 检查前10个因子
                            if col in batch_data.columns:
                                orig_nan = batch_data[col].isna().sum()
                                loaded_nan = loaded_data[col].isna().sum()
                                if loaded_nan > orig_nan:
                                    problem_factors.append(col)
                                    logger.error(f"因子 {col} 损坏: {orig_nan} -> {loaded_nan} NaN")

                        if problem_factors:
                            print(f"  ❌ 发现问题因子: {problem_factors}")
                            return False
                    else:
                        print(f"  ✅ 数据完整性保持良好")

                else:
                    print(f"  ❌ 保存失败：文件不存在")
                    return False

            except Exception as e:
                print(f"  ❌ 保存失败: {e}")
                logger.error(f"批次保存失败: {e}")
                import traceback
                logger.error(f"错误详情: {traceback.format_exc()}")
                return False

        # 最终验证：模拟完整的阶段2加载过程
        print(f"\n🔄 最终验证：完整数据加载")
        try:
            final_data = store.load_raw()
            final_factor_cols = [c for c in final_data.columns if c not in store.index_columns]

            print(f"  最终数据形状: {final_data.shape}")
            print(f"  最终因子数: {len(final_factor_cols)}")

            # 检查每个因子的最终质量
            final_problematic = []
            for col in final_factor_cols:
                nan_count = final_data[col].isna().sum()
                if nan_count == len(final_data):
                    final_problematic.append(f"{col}: 全部NaN")
                else:
                    valid_data = final_data[col].dropna()
                    if len(valid_data) > 0:
                        mean_val = valid_data.mean()
                        std_val = valid_data.std()
                        if std_val == 0:
                            final_problematic.append(f"{col}: 常数列")
                        else:
                            logger.info(f"✅ 因子 {col}: 均值={mean_val:.6f}, 标准差={std_val:.6f}, 有效={len(valid_data)}")
                    else:
                        final_problematic.append(f"{col}: 无有效数据")

            if final_problematic:
                print(f"\n❌ 最终发现 {len(final_problematic)} 个问题因子:")
                for problem in final_problematic[:10]:  # 只显示前10个
                    print(f"  {problem}")
                return False
            else:
                print(f"\n✅ 所有 {len(final_factor_cols)} 个因子质量检查通过！")
                return True

        except Exception as e:
            print(f"❌ 最终验证失败: {e}")
            logger.error(f"最终验证失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return False

def cleanup_test_data():
    """清理测试数据"""
    print("\n🧹 清理测试数据...")
    try:
        import shutil
        test_path = Path("test_factor_lab_data")
        if test_path.exists():
            shutil.rmtree(test_path)
            print("✅ 测试数据清理完成")
    except Exception as e:
        print(f"⚠️ 清理失败: {e}")

if __name__ == "__main__":
    success = test_temp_file_merge()

    print(f"\n=== 测试结果 ===")
    if success:
        print("✅ 临时文件修复测试通过！")
        print("现在可以安全地生成所有因子而不担心数据损坏")
        print("\n建议运行以下命令测试完整因子生成:")
        print("python main.py --mode generate_factors --start_date 20240101 --end_date 20240110")
    else:
        print("❌ 测试失败，需要进一步检查问题")
        print("请查看 test_temp_file_fix.log 获取详细信息")

    cleanup_test_data()