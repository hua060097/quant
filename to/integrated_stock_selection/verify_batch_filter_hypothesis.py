#!/usr/bin/env python3
"""
验证分批过滤导致数据截断的假设
"""

def verify_hypothesis():
    """验证假设"""

    print("=== 验证分批过滤假设 ===")

    print("\n假设：问题出现在因子管道的line 567/584")
    print("batch_filtered = batch_result.loc[target_mask].copy()")

    print("\n数据流分析：")
    print("1. institutional_activity函数接收完整数据（121天/股票）")
    print("2. 函数计算完整结果（约50万个有效值）")
    print("3. 但在分批保存时，target_mask过滤结果（只剩5,399个）")
    print("4. 因子质量检查看到的是过滤后的结果，不是完整结果")

    print("\n证据：")
    print("- 数据加载正确：653,279条记录，121天/股票")
    print("- 函数逻辑正确：应该产生92天 × 5,399股票 = 496,708有效结果")
    print("- 实际结果：只有5,399有效结果（正好是目标日期的股票数）")
    print("- 比例：0.83%，正好是目标日期占总日期的比例")

    print("\n这解释了为什么：")
    print("1. 之前运行成功：目标日期874天，有效率82.81%")
    print("2. 当前运行失败：目标日期56天，有效率0.83%")
    print("3. near_support_count_60d成功：可能它不受target_mask过滤影响")

    print("\n真正的解决方案：")
    print("不是修改institutional_activity函数")
    print("而是修改分批保存机制，让因子质量检查看到完整结果")
    print("同时保持最终保存时使用target_mask过滤")

    print("\n关键问题：")
    print("因子质量检查应该在什么时候进行？")
    print("- 选项A：在target_mask过滤之前检查完整结果")
    print("- 选项B：修改质量检查逻辑，理解时间序列因子的特殊性")

if __name__ == "__main__":
    verify_hypothesis()