"""
测试ST股票过滤优化的正确性和性能
"""
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def filter_st_stocks_old(data: pd.DataFrame, stock_names: dict, logger=None):
    """旧实现：使用 iterrows"""
    st_filter = pd.Series(True, index=data.index)
    st_stocks = []

    for idx, row in data.iterrows():
        stock_code = row['stock_code']
        stock_name = stock_names.get(stock_code, "")

        if "ST" in stock_name or "st" in stock_name or stock_name.endswith("退") or "上证" in stock_name or "*" in stock_name:
            st_filter.loc[idx] = False
            st_stocks.append(f"{stock_code}({stock_name})")

    return st_filter, st_stocks


def filter_st_stocks_new(data: pd.DataFrame, stock_names: dict, logger=None):
    """新实现：向量化操作"""
    # 步骤1: 映射股票名称
    data['_stock_name'] = data['stock_code'].map(stock_names).fillna("")

    # 步骤2: 向量化判断是否为ST股票
    st_mask = (
        data['_stock_name'].str.contains('ST', case=False, na=False) |
        data['_stock_name'].str.contains('退', case=False, na=False) |
        data['_stock_name'].str.contains('上证', case=False, na=False) |
        data['_stock_name'].str.contains('\\*', case=False, na=False)
    )

    # 步骤3: 创建过滤结果
    st_filter = ~st_mask

    # 步骤4: 收集ST股票信息用于日志显示
    st_stocks_data = data[st_mask][['stock_code', '_stock_name']].head(100)
    st_stocks = [f"{row['stock_code']}({row['_stock_name']})" for _, row in st_stocks_data.iterrows()]

    # 清理临时列
    del data['_stock_name']

    return st_filter, st_stocks


def test_st_filter_optimization():
    """测试ST股票过滤优化"""
    print("=" * 100)
    print("ST股票过滤优化验证测试")
    print("=" * 100)

    # 创建测试数据
    n_records = 100_000  # 使用较小的数据集进行快速测试
    n_stocks = 1_000

    print(f"\n[测试数据]")
    print(f"  记录数: {n_records:,}")
    print(f"  股票数: {n_stocks:,}")

    # 生成股票代码
    stock_codes = [f"{str(i).zfill(6)}.SZ" for i in range(n_stocks)]

    # 创建DataFrame
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    data = {
        'stock_code': np.random.choice(stock_codes, n_records),
        'date': np.random.choice(dates, n_records),
    }
    df = pd.DataFrame(data)

    # 创建股票名称字典（包含各种ST股票）
    stock_names = {}
    st_patterns = [
        ("ST", "ST股票"),
        ("*ST", "*ST股票"),
        ("退", "退市股票"),
        ("上证", "上证指数"),
        ("普通", "普通股票")
    ]

    for i, code in enumerate(stock_codes):
        # 20%的股票是ST股票
        if i < n_stocks // 5:
            pattern_type = i % 4
            if pattern_type == 0:
                stock_names[code] = f"ST{code}"
            elif pattern_type == 1:
                stock_names[code] = f"*ST{code}"
            elif pattern_type == 2:
                stock_names[code] = f"{code}退"
            else:
                stock_names[code] = f"上证{code}"
        else:
            stock_names[code] = f"普通股票{code}"

    print(f"  ST股票比例: 20%")

    # 测试旧实现
    print(f"\n{'=' * 100}")
    print("测试旧实现（iterrows）")
    print(f"{'=' * 100}")

    df_old = df.copy()
    start_time = time.time()
    st_filter_old, st_stocks_old = filter_st_stocks_old(df_old, stock_names, logger=None)
    old_time = time.time() - start_time

    print(f"  耗时: {old_time:.4f} 秒")
    print(f"  过滤掉ST股票: {(~st_filter_old).sum():,} 只")
    print(f"  剩余股票: {st_filter_old.sum():,} 只")

    # 测试新实现
    print(f"\n{'=' * 100}")
    print("测试新实现（向量化）")
    print(f"{'=' * 100}")

    df_new = df.copy()
    start_time = time.time()
    st_filter_new, st_stocks_new = filter_st_stocks_new(df_new, stock_names, logger=None)
    new_time = time.time() - start_time

    print(f"  耗时: {new_time:.4f} 秒")
    print(f"  过滤掉ST股票: {(~st_filter_new).sum():,} 只")
    print(f"  剩余股票: {st_filter_new.sum():,} 只")

    # 验证正确性
    print(f"\n{'=' * 100}")
    print("正确性验证")
    print(f"{'=' * 100}")

    # 检查过滤结果是否一致
    results_match = (st_filter_old == st_filter_new).all()

    print(f"  过滤结果一致性: {'[OK] 通过' if results_match else '[FAIL] 失败'}")

    if not results_match:
        diff_count = (st_filter_old != st_filter_new).sum()
        print(f"  不一致的记录数: {diff_count}")
        print(f"  旧实现过滤数: {(~st_filter_old).sum()}")
        print(f"  新实现过滤数: {(~st_filter_new).sum()}")

        # 显示一些不一致的例子
        diff_indices = (st_filter_old != st_filter_new)[st_filter_old != st_filter_new].index
        print(f"\n  不一致示例（前10个）:")
        for idx in diff_indices[:10]:
            row = df.loc[idx]
            old_val = st_filter_old.loc[idx]
            new_val = st_filter_new.loc[idx]
            print(f"    {row['stock_code']} - 旧: {old_val}, 新: {new_val}")

    # 性能对比
    print(f"\n{'=' * 100}")
    print("性能对比")
    print(f"{'=' * 100}")

    print(f"  旧实现（iterrows）: {old_time:.4f} 秒")
    print(f"  新实现（向量化）: {new_time:.4f} 秒")

    if new_time < old_time:
        speedup = old_time / new_time
        improvement = (old_time - new_time) / old_time * 100
        print(f"\n  性能提升: {speedup:.2f}x 倍")
        print(f"  时间节省: {improvement:.1f}%")
        print(f"  节省时间: {old_time - new_time:.4f} 秒")

        # 估算大规模数据的性能提升
        estimated_large_time = new_time * (1_000_000 / n_records)
        estimated_old_large_time = old_time * (1_000_000 / n_records)
        print(f"\n  在100万条数据上的预估:")
        print(f"    旧实现: {estimated_old_large_time:.2f} 秒")
        print(f"    新实现: {estimated_large_time:.2f} 秒")
        print(f"    节省时间: {estimated_old_large_time - estimated_large_time:.2f} 秒")

    # 显示ST股票示例
    print(f"\n{'=' * 100}")
    print("ST股票示例")
    print(f"{'=' * 100}")

    if st_stocks_new:
        display_count = min(5, len(st_stocks_new))
        print(f"  总共发现 {len(st_stocks_new)} 只ST股票")
        print(f"  前{display_count}只ST股票:")
        for i in range(display_count):
            print(f"    {i+1}. {st_stocks_new[i]}")

    print(f"\n{'=' * 100}")
    print("测试结果")
    print(f"{'=' * 100}")

    if results_match:
        print(f"  状态: [OK] 所有测试通过")
        print(f"  功能: 正确性验证通过")
        print(f"  性能: 提升 {old_time / new_time:.2f}x 倍")
    else:
        print(f"  状态: [FAIL] 测试失败")
        print(f"  原因: 过滤结果不一致")

    print("=" * 100)


if __name__ == "__main__":
    test_st_filter_optimization()
