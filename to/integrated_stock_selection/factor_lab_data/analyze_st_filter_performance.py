"""
ST股票过滤性能瓶颈分析和优化
"""
import pandas as pd
import numpy as np
import time

def analyze_st_filter_bottleneck():
    """分析ST股票过滤的性能瓶颈"""

    print("=" * 100)
    print("ST股票过滤性能瓶颈分析")
    print("=" * 100)

    # 模拟数据（假设有100万条记录，5000只股票）
    n_records = 1_000_000
    n_stocks = 5000

    print(f"\n[模拟数据]")
    print(f"  记录数: {n_records:,}")
    print(f"  股票数: {n_stocks:,}")

    # 生成模拟数据
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')
    stock_codes = [f"{str(i).zfill(6)}.SZ" for i in range(n_stocks)]

    # 创建DataFrame
    np.random.seed(42)
    data = {
        'stock_code': np.random.choice(stock_codes, n_records),
        'date': np.random.choice(dates, n_records),
    }
    df = pd.DataFrame(data)

    print(f"  DataFrame形状: {df.shape}")
    print(f"  内存占用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # 模拟股票名称字典
    stock_names = {}
    st_count = 0
    for code in stock_codes:
        # 10%的股票是ST股票
        if np.random.random() < 0.1:
            stock_names[code] = f"ST{code}"
            st_count += 1
        else:
            stock_names[code] = f"普通股票{code}"

    print(f"  ST股票数量: {st_count}")
    print(f"  股票名称字典大小: {len(stock_names)}")

    # 测试当前实现（旧逻辑）
    print(f"\n{'=' * 100}")
    print("测试当前实现（使用 iterrows）")
    print(f"{'=' * 100}")

    start_time = time.time()

    # 当前的低效实现
    st_filter = pd.Series(True, index=df.index)
    st_stocks = []

    iter_start = time.time()
    for idx, row in df.iterrows():
        stock_code = row['stock_code']
        stock_name = stock_names.get(stock_code, "")

        if "ST" in stock_name or "st" in stock_name or stock_name.endswith("退") or "上证" in stock_name or "*" in stock_name:
            st_filter.loc[idx] = False
            st_stocks.append(f"{stock_code}({stock_name})")

    iter_time = time.time() - iter_start
    total_time = time.time() - start_time

    print(f"  循环标记耗时: {iter_time:.4f} 秒")
    print(f"  总耗时: {total_time:.4f} 秒")
    print(f"  过滤掉ST股票: {len(st_stocks)} 只")

    # 测试优化方案
    print(f"\n{'=' * 100}")
    print("测试优化方案（向量化操作）")
    print(f"{'=' * 100}")

    start_time = time.time()

    # 优化方案：向量化操作
    # 1. 映射股票名称
    map_start = time.time()
    df['_stock_name'] = df['stock_code'].map(stock_names).fillna("")
    map_time = time.time() - map_start

    # 2. 向量化判断
    judge_start = time.time()
    st_mask = (
        df['_stock_name'].str.contains('ST', case=False, na=False) |
        df['_stock_name'].str.contains('退', case=False, na=False) |
        df['_stock_name'].str.contains('上证', case=False, na=False) |
        df['_stock_name'].str.contains('\\*', case=False, na=False)
    )
    judge_time = time.time() - judge_start

    # 3. 创建结果
    result_time = time.time() - start_time

    # 清理临时列
    del df['_stock_name']

    print(f"  步骤1 - 映射股票名称: {map_time:.4f} 秒")
    print(f"  步骤2 - 向量化判断: {judge_time:.4f} 秒")
    print(f"  总耗时: {result_time:.4f} 秒")
    print(f"  过滤掉ST股票: {st_mask.sum()} 只")

    # 对比结果
    print(f"\n{'=' * 100}")
    print("性能对比")
    print(f"{'=' * 100}")
    print(f"  旧实现（iterrows）: {iter_time:.4f} 秒")
    print(f"  新实现（向量化）: {result_time:.4f} 秒")

    if result_time < iter_time:
        speedup = iter_time / result_time
        improvement = (iter_time - result_time) / iter_time * 100
        print(f"\n  性能提升: {speedup:.2f}x 倍")
        print(f"  时间节省: {improvement:.1f}%")
        print(f"  节省时间: {iter_time - result_time:.4f} 秒")

    print(f"\n{'=' * 100}")
    print("瓶颈分析")
    print(f"{'=' * 100}")
    print(f"  [问题1] iterrows() 是 pandas 最慢的迭代方式")
    print(f"    - 每次迭代都要创建 Series 对象")
    print(f"    - 无法利用向量化优化")
    print(f"  [问题2] 逐行赋值 st_filter.loc[idx] = False 效率低")
    print(f"    - 每次赋值都需要索引查找")
    print(f"  [问题3] 字典查询 stock_names.get(stock_code) 在循环中重复")
    print(f"    - 可以提前映射到DataFrame")
    print(f"  [问题4] 多个字符串 'in' 操作效率低")
    print(f"    - 可以用 str.contains() 向量化")

    return df, stock_names, st_filter, iter_time, result_time

if __name__ == "__main__":
    analyze_st_filter_bottleneck()
