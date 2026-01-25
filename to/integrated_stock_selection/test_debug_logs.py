#!/usr/bin/env python3
"""
测试因子生成调试日志

验证添加的调试日志是否正常工作
"""

import logging
import sys

# 添加当前目录到路径
sys.path.append('.')

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_factor_generation.log')
        ]
    )

def test_logging():
    """测试日志系统"""
    setup_logging()
    logger = logging.getLogger(__name__)

    print("=== 测试调试日志系统 ===")
    logger.info("INFO级别日志测试")
    logger.debug("DEBUG级别日志测试")
    logger.error("ERROR级别日志测试")
    logger.warning("WARNING级别日志测试")

    print("日志测试完成，请检查输出和控制台")

def import_factor_libraries():
    """导入因子相关库"""
    setup_logging()
    logger = logging.getLogger(__name__)

    print("\n=== 导入因子库测试 ===")

    try:
        from factor_system.factor_lab.commands import generate_and_store
        from factor_system.factor_lab.store import FactorDatasetStore
        from factor_system.factor_lab.pipeline import FactorPipeline

        print("✅ 所有因子库导入成功")
        logger.info("因子库导入测试成功")

        # 创建一个测试存储对象
        store = FactorDatasetStore("test_factor_lab_data")
        print(f"✅ 存储对象创建成功: {store.path}")
        logger.info(f"存储对象创建成功: {store.path}")

        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        logger.error(f"导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        logger.error(f"其他错误: {e}")
        return False

if __name__ == "__main__":
    test_logging()
    success = import_factor_libraries()

    if success:
        print("\n✅ 所有测试通过")
        print("现在可以运行因子生成命令来查看详细的调试日志")
        print("例如：python main.py --mode generate_factors --factor_names alpha_001,alpha_002 --start_date 20250101 --end_date 20250110")
    else:
        print("\n❌ 测试失败，请检查环境配置")