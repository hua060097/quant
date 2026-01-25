#!/usr/bin/env python3
"""
验证标签配置参数修改的测试脚本
"""

import numpy as np

def test_label_configuration():
    """测试修改后的标签分配逻辑"""

    # 模拟修改后的allocate_labels_daily函数的关键部分
    def allocate_labels_daily_test(ranks, t1_returns, t2_returns, N_candidates):
        """
        简化版的标签分配函数，用于测试配置参数
        """
        # 🔧 可配置的标签样本数量参数
        # 可以直接修改这些数值来调整各标签对应的样本数量：
        TARGET_COUNT_30 = 20  # 标签30对应的最大样本数量（推荐范围：10-30）
        TARGET_COUNT_20 = 40  # 标签20对应的最大样本数量（推荐范围：30-50）
        TARGET_COUNT_10 = 60  # 标签10对应的最大样本数量（推荐范围：50-80）
        # 注意：建议保持 TARGET_COUNT_30 <= TARGET_COUNT_20 <= TARGET_COUNT_10 的关系

        n_stocks = len(ranks)
        labels = np.full(n_stocks, 2, dtype=int)  # 默认标签2

        # 简化模拟：假设没有排除任何股票，按排名直接分配标签
        sorted_indices = np.argsort(ranks)  # 按排名排序

        # 分配标签30
        target_30 = min(TARGET_COUNT_30, len(sorted_indices))
        selected_30_indices = sorted_indices[:target_30]
        labels[selected_30_indices] = 30

        # 分配标签20
        remaining_after_30 = sorted_indices[target_30:]
        target_20 = min(TARGET_COUNT_20, len(remaining_after_30))
        selected_20_indices = remaining_after_30[:target_20]
        labels[selected_20_indices] = 20

        # 分配标签10
        remaining_after_20 = remaining_after_30[target_20:]
        target_10 = min(TARGET_COUNT_10, len(remaining_after_20))
        selected_10_indices = remaining_after_20[:target_10]
        labels[selected_10_indices] = 10

        return labels

    # 测试用例1：充足候选股票
    print("=== 测试用例1：充足候选股票 (200只) ===")
    n_stocks = 200
    ranks = np.arange(1, n_stocks + 1)  # 排名1-200
    t1_returns = np.random.normal(0, 0.02, n_stocks)
    t2_returns = np.random.normal(0, 0.02, n_stocks)

    labels = allocate_labels_daily_test(ranks, t1_returns, t2_returns, n_stocks)

    count_30 = np.sum(labels == 30)
    count_20 = np.sum(labels == 20)
    count_10 = np.sum(labels == 10)
    count_other = np.sum((labels != 30) & (labels != 20) & (labels != 10))

    print(f"标签30: {count_30}只 (期望: 20)")
    print(f"标签20: {count_20}只 (期望: 40)")
    print(f"标签10: {count_10}只 (期望: 60)")
    print(f"其他标签: {count_other}只")
    print(f"总股票数: {len(labels)}")
    print()

    # 测试用例2：候选股票不足
    print("=== 测试用例2：候选股票不足 (80只) ===")
    n_stocks = 80
    ranks = np.arange(1, n_stocks + 1)
    t1_returns = np.random.normal(0, 0.02, n_stocks)
    t2_returns = np.random.normal(0, 0.02, n_stocks)

    labels = allocate_labels_daily_test(ranks, t1_returns, t2_returns, n_stocks)

    count_30 = np.sum(labels == 30)
    count_20 = np.sum(labels == 20)
    count_10 = np.sum(labels == 10)
    count_other = np.sum((labels != 30) & (labels != 20) & (labels != 10))

    print(f"标签30: {count_30}只 (期望: 20, 实际按min)")
    print(f"标签20: {count_20}只 (期望: 40, 实际按min)")
    print(f"标签10: {count_10}只 (期望: 60, 实际按min)")
    print(f"其他标签: {count_other}只")
    print(f"总股票数: {len(labels)}")
    print()

    # 测试用例3：自定义配置
    print("=== 测试用例3：自定义配置 (标签30=15, 标签20=30, 标签10=45) ===")

    def test_custom_config():
        """测试自定义配置"""
        # 自定义配置
        TARGET_COUNT_30 = 15
        TARGET_COUNT_20 = 30
        TARGET_COUNT_10 = 45

        n_stocks = 150
        ranks = np.arange(1, n_stocks + 1)
        t1_returns = np.random.normal(0, 0.02, n_stocks)
        t2_returns = np.random.normal(0, 0.02, n_stocks)

        # 使用自定义配置的分配逻辑
        sorted_indices = np.argsort(ranks)
        labels = np.full(n_stocks, 2, dtype=int)

        target_30 = min(TARGET_COUNT_30, len(sorted_indices))
        selected_30_indices = sorted_indices[:target_30]
        labels[selected_30_indices] = 30

        remaining_after_30 = sorted_indices[target_30:]
        target_20 = min(TARGET_COUNT_20, len(remaining_after_30))
        selected_20_indices = remaining_after_30[:target_20]
        labels[selected_20_indices] = 20

        remaining_after_20 = remaining_after_30[target_20:]
        target_10 = min(TARGET_COUNT_10, len(remaining_after_20))
        selected_10_indices = remaining_after_20[:target_10]
        labels[selected_10_indices] = 10

        count_30 = np.sum(labels == 30)
        count_20 = np.sum(labels == 20)
        count_10 = np.sum(labels == 10)
        count_other = np.sum((labels != 30) & (labels != 20) & (labels != 10))

        print(f"自定义配置 - 标签30: {TARGET_COUNT_30}, 标签20: {TARGET_COUNT_20}, 标签10: {TARGET_COUNT_10}")
        print(f"实际分配 - 标签30: {count_30}只, 标签20: {count_20}只, 标签10: {count_10}只")
        print(f"其他标签: {count_other}只")
        print(f"总股票数: {len(labels)}")

    test_custom_config()

    print("\n✅ 测试完成！")
    print("\n📝 使用说明：")
    print("1. 在main.py中搜索 'TARGET_COUNT_' 可以找到配置参数")
    print("2. 直接修改这三个变量的值即可调整各标签的样本数量")
    print("3. 建议保持 TARGET_COUNT_30 <= TARGET_COUNT_20 <= TARGET_COUNT_10")
    print("4. 修改后重新运行模型训练即可生效")

if __name__ == "__main__":
    test_label_configuration()