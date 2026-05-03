"""
查看 factors.parquet 文件中的因子信息和日期范围
"""

import pandas as pd
import os

def check_parquet_info():
    """检查 parquet 文件的基本信息"""

    parquet_path = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data\factors.parquet"

    print("=" * 80)
    print("Factors Parquet 文件信息检查")
    print("=" * 80)

    # 检查文件是否存在
    if not os.path.exists(parquet_path):
        print(f"\n[错误] 文件不存在: {parquet_path}")
        return

    # 文件大小
    file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"\n文件路径: {parquet_path}")
    print(f"文件大小: {file_size_mb:.2f} MB")

    # 读取文件
    print("\n正在读取文件...")
    try:
        df = pd.read_parquet(parquet_path)
        print("[OK] 文件读取成功")
    except Exception as e:
        print(f"[ERROR] 读取失败: {e}")
        return

    # 基本信息
    print(f"\n{'='*80}")
    print("基本信息")
    print(f"{'='*80}")
    print(f"总行数: {len(df):,}")
    print(f"总列数: {len(df.columns):,}")

    # 日期范围
    print(f"\n{'='*80}")
    print("日期范围")
    print(f"{'='*80}")

    if 'date' in df.columns:
        dates = df['date']
        date_count = len(dates)
        unique_dates = dates.dropna().unique()

        print(f"日期列名: date")
        print(f"总日期记录数: {date_count:,}")
        print(f"唯一交易日数: {len(unique_dates):,}")

        # 开始和结束日期
        min_date = unique_dates.min()
        max_date = unique_dates.max()

        print(f"\n开始日期: {min_date}")
        print(f"结束日期: {max_date}")

        # 计算时间跨度
        if hasattr(min_date, 'strftime') and hasattr(max_date, 'strftime'):
            time_span = (max_date - min_date).days
            print(f"时间跨度: {time_span} 天")
    else:
        print("[WARNING] 未找到 'date' 列")
        print(f"\n所有列名: {df.columns.tolist()}")

    # 检查是否有其他日期相关列
    date_like_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_like_cols and 'date' not in date_like_cols:
        print(f"\n发现可能的日期列: {date_like_cols}")

    # 股票数量
    print(f"\n{'='*80}")
    print("股票信息")
    print(f"{'='*80}")

    if 'stock_code' in df.columns:
        unique_stocks = df['stock_code'].dropna().unique()
        print(f"股票代码列名: stock_code")
        print(f"唯一股票数: {len(unique_stocks):,}")
    else:
        print("[WARNING] 未找到 'stock_code' 列")

    # 因子列表
    print(f"\n{'='*80}")
    print("因子列表")
    print(f"{'='*80}")

    # 排除常见的非因子列
    exclude_cols = {'date', 'stock_code', 'datetime', 'time', 'index'}
    factor_columns = [col for col in df.columns if col not in exclude_cols]

    print(f"因子总数: {len(factor_columns)}")
    print(f"\n所有因子列名:")

    # 按每行5个因子显示
    cols_per_row = 5
    for i in range(0, len(factor_columns), cols_per_row):
        row_factors = factor_columns[i:i+cols_per_row]
        factor_str = "  ".join([f"{j+1:3d}. {factor:<30}" for j, factor in enumerate(row_factors, start=i)])
        print(factor_str)

    # 数据质量检查
    print(f"\n{'='*80}")
    print("数据质量检查")
    print(f"{'='*80}")

    # 检查缺失值
    missing_count = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    missing_ratio = missing_count / total_cells * 100

    print(f"缺失值总数: {missing_count:,} / {total_cells:,} ({missing_ratio:.2f}%)")

    # 每个因子的缺失值统计
    print(f"\n因子缺失值统计 (前20个缺失最多的因子):")
    factor_missing = df[factor_columns].isnull().sum()
    factor_missing = factor_missing.sort_values(ascending=False).head(20)

    for factor, missing in factor_missing.items():
        if missing > 0:
            missing_pct = missing / len(df) * 100
            print(f"  {factor:<30}: {missing:>8,} ({missing_pct:>5.2f}%)")

    # 数据类型
    print(f"\n{'='*80}")
    print("数据类型分布")
    print(f"{'='*80}")

    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {str(dtype):<15}: {count:>5} 列")

    # 内存使用
    print(f"\n{'='*80}")
    print("内存使用")
    print(f"{'='*80}")

    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"DataFrame 内存占用: {memory_mb:.2f} MB")

    # 显示前几行数据预览
    print(f"\n{'='*80}")
    print("数据预览 (前3行)")
    print(f"{'='*80}")

    # 选择关键列显示
    preview_cols = ['date', 'stock_code'] if 'stock_code' in df.columns else ['date']
    preview_cols = preview_cols + factor_columns[:5]  # 只显示前5个因子

    print(df[preview_cols].head(3).to_string())

    print(f"\n{'='*80}")
    print("检查完成")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    check_parquet_info()
