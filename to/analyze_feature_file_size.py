"""
分析特征文件大小差异

对比 20260108.parquet 和 20260109.parquet
找出为什么 20260109.parquet 文件明显过大
"""

import pandas as pd
from pathlib import Path
import os

def analyze_parquet_file(file_path):
    """分析单个parquet文件"""
    print(f"\n{'='*80}")
    print(f"分析文件: {file_path.name}")
    print(f"{'='*80}")

    # 1. 文件大小
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"\n[SIZE] 文件大小: {file_size_mb:.2f} MB")
    print(f"       字节数: {file_path.stat().st_size:,}")

    # 2. 读取数据
    print(f"\n[READ] 读取数据...")
    df = pd.read_parquet(file_path)

    # 3. 基本信息
    print(f"\n[BASIC] 基本信息:")
    print(f"  行数: {len(df):,}")
    print(f"  列数: {len(df.columns)}")
    print(f"  内存占用: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # 4. 股票数量
    stock_count = df['stock_code'].nunique()
    print(f"\n[STOCK] 股票数量: {stock_count}")

    # 5. 列名列表
    print(f"\n[COLUMNS] 列名列表 (共{len(df.columns)}列):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    # 6. 每列的数据类型和内存占用
    print(f"\n[DTYPE] 每列数据类型和内存占用:")
    mem_usage = df.memory_usage(deep=True)
    total_mem = mem_usage.sum()

    for col in df.columns:
        dtype = df[col].dtype
        mem_mb = mem_usage[col] / (1024**2)
        mem_pct = (mem_usage[col] / total_mem) * 100
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100

        print(f"  {col:30s} | {str(dtype):15s} | {mem_mb:8.2f} MB ({mem_pct:5.2f}%) | 空值: {null_count:6d} ({null_pct:5.2f}%)")

    # 7. 检查是否有重复数据
    print(f"\n[DUPLICATE] 重复数据检查:")
    # 检查是否有完全重复的行
    dup_rows = df.duplicated().sum()
    print(f"  完全重复的行: {dup_rows} ({dup_rows/len(df)*100:.2f}%)")

    # 检查stock_code和date组合的重复
    if 'stock_code' in df.columns and 'date' in df.columns:
        key_dup = df.duplicated(subset=['stock_code', 'date']).sum()
        print(f"  stock_code+date重复: {key_dup} ({key_dup/len(df)*100:.2f}%)")

    # 8. 数据样本
    print(f"\n[SAMPLE] 前3行数据样本:")
    print(df.head(3).to_string())

    return {
        'file_size_mb': file_size_mb,
        'rows': len(df),
        'cols': len(df.columns),
        'stock_count': stock_count,
        'memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'df': df
    }


def compare_two_files(file1_path, file2_path):
    """对比两个文件"""
    print(f"\n{'='*80}")
    print(f"[COMPARE] 对比分析")
    print(f"{'='*80}")

    # 读取数据
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    # 1. 大小对比
    size1 = file1_path.stat().st_size
    size2 = file2_path.stat().st_size
    size_ratio = size2 / size1

    print(f"\n[SIZE] 文件大小对比:")
    print(f"  {file1_path.name}: {size1/1024/1024:.2f} MB")
    print(f"  {file2_path.name}: {size2/1024/1024:.2f} MB")
    print(f"  比值: {size_ratio:.2f}x (20260109 是 20260108 的 {size_ratio:.2f} 倍)")

    # 2. 行数对比
    rows1 = len(df1)
    rows2 = len(df2)
    rows_ratio = rows2 / rows1

    print(f"\n[ROWS] 行数对比:")
    print(f"  {file1_path.name}: {rows1:,} 行")
    print(f"  {file2_path.name}: {rows2:,} 行")
    print(f"  比值: {rows_ratio:.2f}x")

    # 3. 列数对比
    cols1 = len(df1.columns)
    cols2 = len(df2.columns)

    print(f"\n[COLS] 列数对比:")
    print(f"  {file1_path.name}: {cols1} 列")
    print(f"  {file2_path.name}: {cols2} 列")

    # 4. 列名差异
    cols1_set = set(df1.columns)
    cols2_set = set(df2.columns)

    print(f"\n[COLS_DIFF] 列名差异:")
    only_in_1 = cols1_set - cols2_set
    only_in_2 = cols2_set - cols1_set

    if only_in_1:
        print(f"  只在 {file1_path.name} 中: {only_in_1}")
    else:
        print(f"  只在 {file1_path.name} 中: 无")

    if only_in_2:
        print(f"  只在 {file2_path.name} 中: {only_in_2}")
    else:
        print(f"  只在 {file2_path.name} 中: 无")

    if cols1_set == cols2_set:
        print(f"  列名完全相同")

    # 5. 每列内存占用对比
    print(f"\n[MEM_COMPARE] 每列内存占用对比:")
    mem1 = df1.memory_usage(deep=True)
    mem2 = df2.memory_usage(deep=True)

    for col in sorted(set(df1.columns) | set(df2.columns)):
        if col in df1.columns and col in df2.columns:
            mem1_mb = mem1[col] / (1024**2)
            mem2_mb = mem2[col] / (1024**2)
            ratio = mem2_mb / mem1_mb if mem1_mb > 0 else 0

            print(f"  {col:30s} | 文件1: {mem1_mb:8.2f} MB | 文件2: {mem2_mb:8.2f} MB | 比值: {ratio:.2f}x")
        elif col in df1.columns:
            mem1_mb = mem1[col] / (1024**2)
            print(f"  {col:30s} | 文件1: {mem1_mb:8.2f} MB | 文件2: 缺失")
        elif col in df2.columns:
            mem2_mb = mem2[col] / (1024**2)
            print(f"  {col:30s} | 文件1: 缺失 | 文件2: {mem2_mb:8.2f} MB")

    # 6. 检查是否有异常的大对象
    print(f"\n[LARGE_OBJ] 检查大对象:")
    for col in df2.columns:
        if df2[col].dtype == 'object':
            unique_count = df2[col].nunique()
            print(f"  {col:30s} | 类型: object | 唯一值数量: {unique_count}")

            # 显示几个样本
            samples = df2[col].dropna().head(3).tolist()
            print(f"    样本: {samples}")


def main():
    base_path = Path('local_data/features/daily')

    file1 = base_path / '20260108.parquet'
    file2 = base_path / '20260109.parquet'

    print(f"{'='*80}")
    print(f"特征文件大小差异分析")
    print(f"{'='*80}")

    # 检查文件是否存在
    if not file1.exists():
        print(f"\n[ERROR] 文件不存在: {file1}")
        return
    if not file2.exists():
        print(f"\n[ERROR] 文件不存在: {file2}")
        return

    # 分析文件1
    info1 = analyze_parquet_file(file1)

    # 分析文件2
    info2 = analyze_parquet_file(file2)

    # 对比分析
    compare_two_files(file1, file2)

    # 总结
    print(f"\n{'='*80}")
    print(f"[SUMMARY] 分析总结")
    print(f"{'='*80}")
    print(f"\n文件1 ({file1.name}):")
    print(f"  大小: {info1['file_size_mb']:.2f} MB")
    print(f"  行数: {info1['rows']:,}")
    print(f"  列数: {info1['cols']}")
    print(f"  内存: {info1['memory_mb']:.2f} MB")

    print(f"\n文件2 ({file2.name}):")
    print(f"  大小: {info2['file_size_mb']:.2f} MB")
    print(f"  行数: {info2['rows']:,}")
    print(f"  列数: {info2['cols']}")
    print(f"  内存: {info2['memory_mb']:.2f} MB")

    size_diff = info2['file_size_mb'] - info1['file_size_mb']
    size_ratio = info2['file_size_mb'] / info1['file_size_mb']

    print(f"\n差异:")
    print(f"  大小差异: {size_diff:.2f} MB ({size_ratio:.2f}x)")
    print(f"  行数差异: {info2['rows'] - info1['rows']:,}")
    print(f"  列数差异: {info2['cols'] - info1['cols']}")


if __name__ == "__main__":
    main()
