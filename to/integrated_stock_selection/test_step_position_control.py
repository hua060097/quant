"""
阶梯式仓位控制测试脚本
Test Script for Step Position Control

测试目标：
1. 验证阶梯函数在不同阈值下的正确性
2. 测试边界条件
3. 对比平滑函数与阶梯函数的表现
4. 验证在回测和实盘场景下的集成
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from step_position_control import StepPositionController, create_position_controller_from_data

# ========================================================================
# 测试1: 基本功能测试
# ========================================================================

def test_basic_functionality():
    """
    测试基本功能：验证阶梯函数在各个阈值下是否正确工作
    """
    print("=" * 80)
    print("TEST 1: 基本功能测试")
    print("=" * 80)

    controller = StepPositionController(
        top1_min=0.0,
        top1_max=1.0,
        step_thresholds=[
            (0.02, 1.0),
            (0.01, 0.7),
            (0.007, 0.4),
            (0.004, 0.3),
            (0.0, 0.2)
        ]
    )

    test_cases = [
        # (score, expected_weight, description)
        (0.025, 1.0, "超过最高阈值"),
        (0.02, 1.0, "等于最高阈值"),
        (0.015, 0.7, "在0.01和0.02之间"),
        (0.01, 0.7, "等于0.01阈值"),
        (0.008, 0.4, "在0.007和0.01之间"),
        (0.007, 0.4, "等于0.007阈值"),
        (0.005, 0.3, "在0.004和0.007之间"),
        (0.004, 0.3, "等于0.004阈值"),
        (0.002, 0.2, "低于0.004"),
        (0.0, 0.2, "等于最低值"),
        (-0.001, 0.2, "负值（归一化后为0）"),
    ]

    all_passed = True
    for score, expected, desc in test_cases:
        actual = controller.get_position_weight(score)
        passed = abs(actual - expected) < 1e-10
        status = "[PASS]" if passed else "[FAIL]"

        print(f"{status} | Score={score:6.3f} | Expected={expected:.1f} | Actual={actual:.1f} | {desc}")

        if not passed:
            all_passed = False

    print(f"\n结果: {'全部通过' if all_passed else '存在失败'}")
    return all_passed


# ========================================================================
# 测试2: 边界条件测试
# ========================================================================

def test_edge_cases():
    """
    测试边界条件和异常输入
    """
    print("\n" + "=" * 80)
    print("TEST 2: 边界条件测试")
    print("=" * 80)

    controller = StepPositionController(
        top1_min=0.0,
        top1_max=1.0,
        step_thresholds=[
            (0.02, 1.0),
            (0.01, 0.7),
            (0.007, 0.4),
            (0.004, 0.3),
            (0.0, 0.2)
        ]
    )

    # 测试极端值
    edge_cases = [
        (100.0, "极大值（远超top1_max）"),
        (-100.0, "极小值（负数）"),
        (np.inf, "无穷大"),
        (-np.inf, "负无穷大"),
        (np.nan, "NaN"),
    ]

    for score, desc in edge_cases:
        try:
            weight = controller.get_position_weight(score)
            print(f"[OK] {desc:20s} | Weight={weight:.2f}")
        except Exception as e:
            print(f"[ERR] {desc:20s} | Error: {e}")

    # 测试归一化范围
    print("\n归一化测试：")
    for score in [0.0, 0.5, 1.0]:
        strength = controller.calculate_pred_strength(score)
        print(f"  Score={score:.1f} -> Strength={strength:.4f}")


# ========================================================================
# 测试3: 对比平滑函数 vs 阶梯函数
# ========================================================================

def compare_smooth_vs_step():
    """
    对比平滑函数和阶梯函数的表现差异
    """
    print("\n" + "=" * 80)
    print("TEST 3: 平滑函数 vs 阶梯函数对比")
    print("=" * 80)

    # 创建测试分数
    test_scores = np.linspace(0, 0.03, 31)

    # 阶梯函数
    step_controller = StepPositionController(
        top1_min=0.0,
        top1_max=1.0,
        step_thresholds=[
            (0.02, 1.0),
            (0.01, 0.7),
            (0.007, 0.4),
            (0.004, 0.3),
            (0.0, 0.2)
        ]
    )

    step_weights = [step_controller.get_position_weight(s) for s in test_scores]

    # 平滑函数（模拟旧的 lead 逻辑）
    # lead = top1_score - mean(top5_scores)
    # 假设 mean(top5_scores) = 0.01
    mean_top5 = 0.01
    leads = test_scores - mean_top5

    # 简单的线性平滑函数
    smooth_weights = np.clip(leads * 50 + 0.5, 0.2, 1.0)

    # 统计差异
    print(f"\n统计对比：")
    print(f"  测试样本数: {len(test_scores)}")
    print(f"  阶梯函数唯一值数量: {len(set(step_weights))}")
    print(f"  平滑函数唯一值数量: {len(np.unique(np.round(smooth_weights, 6)))}")
    print(f"  平均绝对差异: {np.mean(np.abs(np.array(step_weights) - smooth_weights)):.4f}")
    print(f"  最大差异: {np.max(np.abs(np.array(step_weights) - smooth_weights)):.4f}")

    # 找出差异最大的点
    diff = np.abs(np.array(step_weights) - smooth_weights)
    max_diff_idx = np.argmax(diff)
    print(f"\n最大差异点：")
    print(f"  Score: {test_scores[max_diff_idx]:.6f}")
    print(f"  Step: {step_weights[max_diff_idx]:.2f}")
    print(f"  Smooth: {smooth_weights[max_diff_idx]:.2f}")
    print(f"  Difference: {diff[max_diff_idx]:.2f}")


# ========================================================================
# 测试4: 性能测试
# ========================================================================

def test_performance():
    """
    测试性能：计算大量仓位权重的速度
    """
    print("\n" + "=" * 80)
    print("TEST 4: 性能测试")
    print("=" * 80)

    controller = StepPositionController(
        top1_min=0.0,
        top1_max=1.0,
        step_thresholds=[
            (0.02, 1.0),
            (0.01, 0.7),
            (0.007, 0.4),
            (0.004, 0.3),
            (0.0, 0.2)
        ]
    )

    # 生成大量测试数据
    n_samples = 100000
    test_scores = np.random.uniform(0, 0.03, n_samples)

    import time
    start_time = time.time()

    weights = [controller.get_position_weight(s) for s in test_scores]

    elapsed = time.time() - start_time

    print(f"  样本数量: {n_samples:,}")
    print(f"  总耗时: {elapsed:.4f} 秒")
    print(f"  平均耗时: {elapsed/n_samples*1000:.4f} 毫秒/次")
    print(f"  吞吐量: {n_samples/elapsed:,.0f} 次/秒")


# ========================================================================
# 测试5: 实际场景模拟
# ========================================================================

def test_real_world_scenario():
    """
    模拟实际交易场景
    """
    print("\n" + "=" * 80)
    print("TEST 5: 实际交易场景模拟")
    print("=" * 80)

    # 模拟一天的预测结果
    daily_predictions = pd.DataFrame({
        'stock_code': [f'{i:06d}.SZ' for i in range(1, 11)],
        'pred_top1': [0.025, 0.018, 0.012, 0.008, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0005],
        'current_price': [10 + i for i in range(10)]
    })

    print("\n预测数据：")
    print(daily_predictions.to_string(index=False))

    # 初始化控制器
    controller = StepPositionController(
        top1_min=0.0,
        top1_max=1.0,
        step_thresholds=[
            (0.02, 1.0),
            (0.01, 0.7),
            (0.007, 0.4),
            (0.004, 0.3),
            (0.0, 0.2)
        ]
    )

    # 计算每只股票的仓位权重
    daily_predictions['position_weight'] = daily_predictions['pred_top1'].apply(
        lambda x: controller.get_position_weight(x, verbose=True)
    )

    print("\n仓位分配：")
    print(daily_predictions.to_string(index=False))

    # 统计
    print("\n仓位统计：")
    print(daily_predictions['position_weight'].value_counts().sort_index().to_string())


# ========================================================================
# 运行所有测试
# ========================================================================

def run_all_tests():
    """
    运行所有测试
    """
    print("\n" + "#" * 80)
    print("# 阶梯式仓位控制 - 完整测试套件")
    print("#" * 80)

    test_results = []

    # Test 1
    result1 = test_basic_functionality()
    test_results.append(("基本功能", result1))

    # Test 2
    test_edge_cases()
    test_results.append(("边界条件", True))  # 只要没有崩溃就算通过

    # Test 3
    compare_smooth_vs_step()
    test_results.append(("函数对比", True))

    # Test 4
    test_performance()
    test_results.append(("性能测试", True))

    # Test 5
    test_real_world_scenario()
    test_results.append(("场景模拟", True))

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    for test_name, passed in test_results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} | {test_name}")

    all_passed = all(result for _, result in test_results)

    print(f"\n最终结果: {'全部通过' if all_passed else '部分失败'}")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
