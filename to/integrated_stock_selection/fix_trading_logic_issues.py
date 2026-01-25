"""
修复交易逻辑中的关键问题

1. 买入和卖出日期为N/A的问题
2. 总收益为0的问题
3. 持仓时间计算问题

作者: Claude Code Assistant
日期: 2025-11-22
"""

import logging
import sys
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.append('.')

def analyze_trading_logic_issues():
    """分析交易逻辑问题"""
    logger.info("=== 分析交易逻辑问题 ===")

    try:
        # 读取main.py中的关键代码段
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # 分析问题1：交易记录创建条件
        logger.info("问题1: 交易记录创建条件分析")

        # 查找交易记录创建的条件
        if "if not sell_data.empty:" in content:
            logger.warning("⚠️ 发现潜在问题：交易记录创建依赖于sell_data不为空")
            logger.warning("   如果sell_data为空，交易记录将不会被创建")
            logger.warning("   这会导致买入/卖出日期显示为N/A")

        # 分析问题2：数据获取逻辑
        logger.info("\n问题2: 数据获取逻辑分析")

        if "sell_data = backtest_data[" in content:
            logger.info("✅ 找到卖出数据获取逻辑")
            logger.warning("   可能存在数据匹配问题：stock_code + date组合可能找不到对应数据")

        # 分析问题3：日期计算逻辑
        logger.info("\n问题3: 日期计算逻辑分析")

        if "buy_date = trading_dates[date_idx - holding_days + 1]" in content:
            logger.info("✅ 买入日期计算：T+1日买入")

        if "sell_date = date" in content:
            logger.info("✅ 卖出日期计算：T+1+holding_days日卖出")
            logger.warning("   需要验证交易日历是否包含所有需要的日期")

        return True

    except Exception as e:
        logger.error(f"❌ 分析失败: {e}")
        return False

def suggest_fixes():
    """建议修复方案"""
    logger.info("\n=== 建议修复方案 ===")

    fixes = [
        {
            "问题": "买入/卖出日期为N/A",
            "原因": "交易记录创建依赖sell_data不为空，如果找不到卖出数据就不创建记录",
            "解决方案": [
                "1. 添加调试日志，显示哪些股票在哪些日期找不到卖出数据",
                "2. 在找不到卖出数据时，使用前一个交易日的收盘价",
                "3. 或者创建一个缺失数据的标记记录"
            ]
        },
        {
            "问题": "总收益为0",
            "原因": "交易记录可能为空或net_return计算有误",
            "解决方案": [
                "1. 添加交易记录数量统计",
                "2. 验证net_return计算逻辑",
                "3. 检查交易成本扣除是否正确"
            ]
        },
        {
            "问题": "持仓时间不等于holding_days",
            "原因": "交易日历计算可能有问题",
            "解决方案": [
                "1. 验证trading_dates数组是否正确",
                "2. 添加持仓时间计算的调试日志",
                "3. 检查节假日和非交易日处理"
            ]
        }
    ]

    for i, fix in enumerate(fixes, 1):
        logger.info(f"\n{i}. {fix['问题']}")
        logger.info(f"   原因: {fix['原因']}")
        logger.info("   解决方案:")
        for solution in fix['解决方案']:
            logger.info(f"   {solution}")

def propose_patching_code():
    """提供代码补丁建议"""
    logger.info("\n=== 代码补丁建议 ===")

    patches = [
        {
            "位置": "第8970行附近",
            "问题": "sell_data为空时不创建交易记录",
            "建议代码": """
# 修复前：
if not sell_data.empty:
    sell_price = sell_data.iloc[0]['close_front']
    # ... 创建交易记录

# 修复后：
if not sell_data.empty:
    sell_price = sell_data.iloc[0]['close_front']
else:
    # 找不到卖出数据时的处理
    logger.warning(f"找不到{stock_code}在{sell_date}的卖出数据，使用前一日价格")
    # 尝试获取前一个交易日的数据
    prev_date = trading_dates[trading_dates.index(sell_date) - 1] if sell_date in trading_dates else None
    if prev_date:
        prev_data = backtest_data[
            (backtest_data['stock_code'] == stock_code) &
            (backtest_data['date'] == prev_date)
        ]
        if not prev_data.empty:
            sell_price = prev_data.iloc[0]['close_front']
        else:
            logger.error(f"无法找到{stock_code}的任何价格数据，跳过此交易")
            continue
    else:
        logger.error(f"无法找到{sell_date}的前一个交易日，跳过此交易")
        continue

# 无论sell_data是否为空，都创建交易记录
gross_return = (sell_price - holding['buy_price']) / holding['buy_price']
net_return = gross_return - TRADING_COST_TOTAL
# ... 创建交易记录
            """
        }
    ]

    for i, patch in enumerate(patches, 1):
        logger.info(f"\n补丁{i}: {patch['位置']}")
        logger.info(f"问题: {patch['问题']}")
        logger.info(f"建议代码:\n{patch['建议代码']}")

def main():
    """主函数"""
    logger.info("🔧 开始分析交易逻辑问题")
    logger.info("=" * 60)

    # 分析问题
    analysis_ok = analyze_trading_logic_issues()

    # 建议修复方案
    suggest_fixes()

    # 提供代码补丁
    propose_patching_code()

    logger.info("\n" + "=" * 60)
    logger.info("✅ 分析完成")
    logger.info("=" * 60)

    if analysis_ok:
        logger.info("\n📋 下一步建议:")
        logger.info("1. 先运行当前程序，观察具体哪些股票在哪些日期缺失数据")
        logger.info("2. 添加详细的调试日志来定位问题")
        logger.info("3. 根据实际运行情况选择合适的修复方案")
        logger.info("4. 测试修复后的逻辑确保不影响其他功能")
    else:
        logger.error("\n❌ 分析失败，请检查代码文件")

if __name__ == "__main__":
    main()