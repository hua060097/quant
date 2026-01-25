"""
测试修改后的select模式

验证:
1. LightGBM单模型加载
2. 数据加载（复用main.py逻辑）
3. 过滤逻辑（与main.py pred_stocks一致）
4. 预测逻辑
5. 结果保存
"""

import sys
from pathlib import Path

# 确保能导入模块
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from live_trading.config import PathConfig
from live_trading.model_loader import ModelLoader
from live_trading.stock_selector import StockSelector
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_model_loader():
    """测试LightGBM单模型加载"""
    logger.info("\n" + "=" * 60)
    logger.info("测试1: LightGBM模型加载")
    logger.info("=" * 60)

    model_loader = ModelLoader()
    success = model_loader.load_lightgbm_model()

    if success:
        logger.info("✅ 模型加载成功")
        logger.info(f"  模型数量: {len(model_loader.models)}")
        logger.info(f"  特征数量: {len(model_loader.feature_names)}")
        logger.info(f"  模型类型: {model_loader.model_types}")

        # 获取模型信息
        info = model_loader.get_model_info()
        logger.info(f"  模型信息: {info}")
        return True
    else:
        logger.error("❌ 模型加载失败")
        return False


def test_data_loading(date: str):
    """测试数据加载（复用main.py逻辑）"""
    logger.info("\n" + "=" * 60)
    logger.info(f"测试2: 加载{date}的因子数据")
    logger.info("=" * 60)

    try:
        from live_trading.main import load_factor_data

        factor_data = load_factor_data(date)

        if factor_data is not None and not factor_data.empty:
            logger.info("✅ 数据加载成功")
            logger.info(f"  股票数量: {len(factor_data)}")
            logger.info(f"  数据列数: {len(factor_data.columns)}")

            # 检查关键列
            key_columns = ['stock_code', 'date', 'is_limit_up', 'is_limit_down', 'amount', 'list_date']
            missing = [col for col in key_columns if col not in factor_data.columns]
            if missing:
                logger.warning(f"  缺失列: {missing}")
            else:
                logger.info("  ✅ 所有关键列都存在")

            return True, factor_data
        else:
            logger.error("❌ 数据加载失败或为空")
            return False, None

    except Exception as e:
        logger.error(f"❌ 数据加载异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None


def test_stock_selector(date: str, factor_data, model_loader):
    """测试选股流程"""
    logger.info("\n" + "=" * 60)
    logger.info("测试3: 选股流程")
    logger.info("=" * 60)

    try:
        selector = StockSelector(model_loader)
        num_stocks = 3

        logger.info(f"开始选股: 日期={date}, 数量={num_stocks}")
        result = selector.select_stocks(date, num_stocks, factor_data)

        if result is not None:
            logger.info("✅ 选股成功")
            logger.info(f"  选股数量: {len(result['stocks'])}")

            if result['stocks']:
                logger.info("  选股结果:")
                for stock in result['stocks']:
                    logger.info(f"    {stock['rank']}. {stock['code']} - 分数: {stock['score']:.3f}")
            else:
                logger.warning("  ⚠️ 无符合条件的股票")

            return True
        else:
            logger.error("❌ 选股失败")
            return False

    except Exception as e:
        logger.error(f"❌ 选股异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_result_file():
    """测试结果文件"""
    logger.info("\n" + "=" * 60)
    logger.info("测试4: 检查结果文件")
    logger.info("=" * 60)

    result_file = PathConfig.STOCK_SELECTIONS_FILE

    if result_file.exists():
        logger.info(f"✅ 结果文件存在: {result_file}")

        import json
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                selections = json.load(f)

            logger.info(f"  选股记录数: {len(selections)}")
            logger.info(f"  日期列表: {list(selections.keys())[-5:]}")  # 显示最近5次

            # 显示最新一次的选股结果
            latest_date = max(selections.keys())
            latest_result = selections[latest_date]
            logger.info(f"\n  最新选股 ({latest_date}):")
            if latest_result['stocks']:
                for stock in latest_result['stocks']:
                    logger.info(f"    {stock['rank']}. {stock['code']} - 分数: {stock['score']:.3f}")

            return True

        except Exception as e:
            logger.error(f"❌ 读取结果文件失败: {e}")
            return False
    else:
        logger.warning(f"⚠️ 结果文件不存在: {result_file}")
        return False


def main():
    """主测试函数"""
    logger.info("\n" + "=" * 80)
    logger.info("开始测试live_trading的select模式修改")
    logger.info("=" * 80)

    # 测试日期（使用最近的交易日）
    test_date = "20250108"  # 请根据实际情况修改

    results = []

    # 测试1: 模型加载
    result1 = test_model_loader()
    results.append(("模型加载", result1))

    if not result1:
        logger.error("模型加载失败，停止后续测试")
        return

    # 测试2: 数据加载
    result2, factor_data = test_data_loading(test_date)
    results.append(("数据加载", result2))

    if not result2:
        logger.error("数据加载失败，停止后续测试")
        return

    # 测试3: 选股流程
    model_loader = ModelLoader()
    model_loader.load_lightgbm_model()
    result3 = test_stock_selector(test_date, factor_data, model_loader)
    results.append(("选股流程", result3))

    # 测试4: 结果文件
    result4 = test_result_file()
    results.append(("结果文件", result4))

    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, r in results if r)
    logger.info(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        logger.info("\n🎉 所有测试通过！")
    else:
        logger.warning(f"\n⚠️ {total - passed}个测试失败")


if __name__ == '__main__':
    main()
