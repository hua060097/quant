"""
新股过滤性能瓶颈分析和优化
"""
import pandas as pd
import numpy as np
import time
from pathlib import Path

def analyze_bottleneck():
    """分析新股过滤的性能瓶颈"""

    print("=" * 100)
    print("新股过滤性能瓶颈分析")
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
    data = {
        'stock_code': np.random.choice(stock_codes, n_records),
        'date': np.random.choice(dates, n_records),
    }
    stock_data = pd.DataFrame(data)

    print(f"  DataFrame形状: {stock_data.shape}")
    print(f"  内存占用: {stock_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # 模拟新股过滤数据
    filter_stock_date_pairs = []
    for i in range(100):  # 假设有100只新股
        stock_code = stock_codes[i]
        ipo_date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 100))
        filter_end_date = ipo_date + pd.Timedelta(days=25)
        filter_stock_date_pairs.append({
            'stock_code': stock_code,
            'ipo_date': ipo_date,
            'filter_end_date': filter_end_date
        })

    print(f"  新股数量: {len(filter_stock_date_pairs)}")

    # 测试当前实现（旧逻辑）
    print(f"\n{'=' * 100}")
    print("测试当前实现（旧逻辑）")
    print(f"{'=' * 100}")

    start_time = time.time()

    # 5.1 构建需要过滤的股票集合（冗余代码！）
    filter_stocks = {pair['stock_code'] for pair in filter_stock_date_pairs}
    print(f"  步骤1 - 构建filter_stocks集合: {len(filter_stocks)} 只股票")

    # 创建临时列
    temp_data = stock_data.copy()
    temp_data['_need_filter'] = False  # 明确初始化为布尔类型
    copy_time = time.time() - start_time
    print(f"  步骤2 - 复制DataFrame: {copy_time:.4f} 秒")

    # 循环标记
    loop_start = time.time()
    for pair in filter_stock_date_pairs:
        stock_code = pair['stock_code']
        ipo_date = pair['ipo_date']
        filter_end_date = pair['filter_end_date']

        mask = (
            (temp_data['stock_code'] == stock_code) &
            (temp_data['date'] >= ipo_date) &
            (temp_data['date'] <= filter_end_date)
        )
        temp_data.loc[mask, '_need_filter'] = True

    loop_time = time.time() - loop_start
    print(f"  步骤3 - 循环标记: {loop_time:.4f} 秒")
    print(f"    平均每次迭代: {loop_time/len(filter_stock_date_pairs):.6f} 秒")

    # 执行过滤
    filter_start = time.time()
    filtered_data = stock_data[~temp_data['_need_filter']].copy()
    filter_time = time.time() - filter_start
    print(f"  步骤4 - 执行过滤: {filter_time:.4f} 秒")

    total_time = time.time() - start_time
    print(f"\n  总耗时: {total_time:.4f} 秒")

    # 分析瓶颈
    print(f"\n{'=' * 100}")
    print("瓶颈分析")
    print(f"{'=' * 100}")

    bottlenecks = []

    # 瓶颈1: DataFrame复制
    if copy_time > total_time * 0.1:
        bottlenecks.append({
            'step': 'DataFrame复制',
            'time': copy_time,
            'pct': copy_time / total_time * 100,
            'issue': '复制整个DataFrame非常耗时，占用大量内存'
        })

    # 瓶颈2: 循环中的多次布尔索引
    if loop_time > total_time * 0.3:
        bottlenecks.append({
            'step': '循环标记',
            'time': loop_time,
            'pct': loop_time / total_time * 100,
            'issue': f'对{len(filter_stock_date_pairs)}只新股循环，每次都扫描整个DataFrame'
        })

    # 瓶颈3: 冗余的filter_stocks
    print(f"\n  [冗余代码分析]")
    print(f"    filter_stocks 被创建但从未使用")
    print(f"    占用内存: {len(filter_stocks)} 个字符串对象")
    print(f"    是否冗余: 是 [OK]")

    if bottlenecks:
        print(f"\n  [性能瓶颈]")
        for i, b in enumerate(bottlenecks, 1):
            print(f"    {i}. {b['step']}")
            print(f"       耗时: {b['time']:.4f} 秒 ({b['pct']:.1f}% 总时间)")
            print(f"       问题: {b['issue']}")
    else:
        print(f"\n  [未发现明显瓶颈]")

    return stock_data, filter_stock_date_pairs, total_time

def test_optimized_implementation(stock_data, filter_stock_date_pairs):
    """测试优化后的实现"""

    print(f"\n{'=' * 100}")
    print("测试优化后实现")
    print(f"{'=' * 100}")

    start_time = time.time()

    # 优化方案1: 使用向量化操作，避免循环和复制
    # 1. 将新股过滤条件转换为DataFrame
    new_stock_df = pd.DataFrame(filter_stock_date_pairs)

    prep_time = time.time() - start_time

    # 2. 使用merge快速匹配
    merge_start = time.time()

    # 将日期转换为数值便于比较
    stock_data['_date_num'] = stock_data['date'].astype(np.int64)
    new_stock_df['_ipo_num'] = new_stock_df['ipo_date'].astype(np.int64)
    new_stock_df['_filter_end_num'] = new_stock_df['filter_end_date'].astype(np.int64)

    # 执行merge
    merged = stock_data.merge(
        new_stock_df,
        on='stock_code',
        how='left'
    )

    # 向量化判断：是否需要过滤
    # 条件：匹配到新股 且 日期在IPO和过滤结束日期之间
    merged['_need_filter'] = (
        merged['_ipo_num'].notna() &
        (merged['_date_num'] >= merged['_ipo_num']) &
        (merged['_date_num'] <= merged['_filter_end_num'])
    ).fillna(False)

    merge_time = time.time() - merge_start

    # 3. 执行过滤
    filter_start = time.time()
    filtered_data = stock_data[~merged['_need_filter']].copy()
    filter_time = time.time() - filter_start

    total_time = time.time() - start_time

    print(f"  步骤1 - 准备工作: {prep_time:.4f} 秒")
    print(f"  步骤2 - Merge操作: {merge_time:.4f} 秒")
    print(f"  步骤3 - 执行过滤: {filter_time:.4f} 秒")
    print(f"\n  总耗时: {total_time:.4f} 秒")

    # 清理临时列
    if '_date_num' in stock_data.columns:
        del stock_data['_date_num']

    return filtered_data, total_time

if __name__ == "__main__":
    # 运行分析
    stock_data, filter_stock_date_pairs, old_time = analyze_bottleneck()

    # 测试优化版本
    filtered_new, new_time = test_optimized_implementation(stock_data, filter_stock_date_pairs)

    # 对比结果
    print(f"\n{'=' * 100}")
    print("性能对比")
    print(f"{'=' * 100}")
    print(f"  旧实现: {old_time:.4f} 秒")
    print(f"  新实现: {new_time:.4f} 秒")

    if new_time < old_time:
        speedup = old_time / new_time
        improvement = (old_time - new_time) / old_time * 100
        print(f"\n  性能提升: {speedup:.2f}x 倍")
        print(f"  时间节省: {improvement:.1f}%")
        print(f"  节省时间: {old_time - new_time:.4f} 秒")
    else:
        print(f"\n  优化版本未能提升性能（可能是因为数据规模较小）")
