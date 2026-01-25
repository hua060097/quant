"""
验证clip逻辑：clip不会放大边界内的值
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("验证 Clip 逻辑")
print("=" * 80)

# 测试数据
test_values = pd.Series([-10, -6, -5, 0, 2, 5, 6, 8, 10])
print(f"\n原始值: {test_values.tolist()}")

# 固定clip
clipped_fixed = test_values.clip(-6, 6)
print(f"clip(-6, 6)后: {clipped_fixed.tolist()}")

# 自适应clip（假设median=0, MAD=1）
median = 0
mad = 1
threshold = 3 * mad * 1.4826  # 约4.45
lower = median - threshold  # 约-4.45
upper = median + threshold  # 约4.45

clipped_adaptive = test_values.clip(lower, upper)
print(f"\n自适应clip({lower:.2f}, {upper:.2f})后: {clipped_adaptive.tolist()}")

print("\n" + "=" * 80)
print("关键验证: 值=2，clip范围=-6~6")
print("=" * 80)

val = 2.0
clipped = pd.Series([val]).clip(-6, 6)[0]
print(f"原始值: {val}")
print(f"clip(-6, 6)后: {clipped}")
print(f"是否相等: {val == clipped}")
print(f"结论: clip不会放大边界内的值，{val}仍然是{clipped}")

print("\n" + "=" * 80)
print("关键验证: 值=8，clip范围=-6~6")
print("=" * 80)

val = 8.0
clipped = pd.Series([val]).clip(-6, 6)[0]
print(f"原始值: {val}")
print(f"clip(-6, 6)后: {clipped}")
print(f"是否被截断: {val != clipped}")
print(f"结论: clip会截断超出边界的值，{val}被截断为{clipped}")
