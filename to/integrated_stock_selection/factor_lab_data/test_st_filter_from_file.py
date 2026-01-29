"""
测试从 st.parquet 文件过滤 ST 股票的功能
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
# main.py 在 integrated_stock_selection/ 目录
# 测试脚本在 integrated_stock_selection/factor_lab_data/ 目录
current_dir = Path(__file__).parent
integrated_stock_selection_dir = current_dir.parent
project_root = integrated_stock_selection_dir.parent  # to 目录

# 添加 integrated_stock_selection 到 path 以导入 main
if str(integrated_stock_selection_dir) not in sys.path:
    sys.path.append(str(integrated_stock_selection_dir))

# 导入主函数
from main import _filter_st_stocks
import logging

# 创建logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("=" * 80)
print("测试 ST 股票过滤功能（从 st.parquet 文件读取）")
print("=" * 80)

# 1. 创建测试数据
print("\n[1] 创建测试数据...")

# 模拟股票数据
test_data = pd.DataFrame({
    'stock_code': [
        '000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ',
        '399001.SZ', '399006.SZ', '600000.SH', '600001.SH', '600002.SH'
    ] * 10,  # 100条记录
    'date': pd.date_range('2024-01-15', '2024-01-24').repeat(10),  # 10天
    'factor1': np.random.randn(100),
    'factor2': np.random.randn(100)
})

print(f"  测试数据量: {len(test_data)} 条记录")
print(f"  股票数量: {test_data['stock_code'].nunique()} 只")
print(f"  日期范围: {test_data['date'].min()} ~ {test_data['date'].max()}")

# 2. 读取 ST 数据
print("\n[2] 读取 ST 数据文件...")

st_file = project_root / "local_data" / "st.parquet"
print(f"  ST文件路径: {st_file}")

if st_file.exists():
    st_df = pd.read_parquet(st_file)
    print(f"  ST数据量: {len(st_df)} 条记录")
    print(f"  ST日期范围: {st_df['trade_date'].min()} ~ {st_df['trade_date'].max()}")

    # 显示部分ST股票
    print(f"\n  ST股票示例:")
    sample_st = st_df.groupby('trade_date').head(2).sort_values('trade_date')
    for _, row in sample_st.head(10).iterrows():
        print(f"    {row['trade_date']}: {row['ts_code']}")
else:
    print(f"  [ERROR] ST数据文件不存在: {st_file}")
    sys.exit(1)

# 3. 执行过滤
print("\n[3] 执行 ST 股票过滤...")

st_filter = _filter_st_stocks(test_data, logger)

print(f"\n  过滤结果:")
print(f"    总记录数: {len(test_data)}")
print(f"    非ST股票: {st_filter.sum()}")
print(f"    ST股票: {(~st_filter).sum()}")

# 4. 验证结果
print("\n[4] 验证过滤结果...")

# 应用过滤
filtered_data = test_data[st_filter].copy()

# 找出被过滤的记录
removed_records = test_data[~st_filter]

if len(removed_records) > 0:
    print(f"  被过滤的记录:")
    print(f"    数量: {len(removed_records)}")
    print(f"    股票代码: {removed_records['stock_code'].unique()}")
    print(f"    日期分布:")
    for date, group in removed_records.groupby('date'):
        print(f"      {date.strftime('%Y-%m-%d')}: {len(group)} 条 ({', '.join(group['stock_code'].unique())})")

    # 验证：检查这些股票是否真的在 ST 数据中
    print(f"\n  验证被过滤股票是否在 ST 数据中:")
    verification_passed = True
    for _, row in removed_records.iterrows():
        date_str = row['date'].strftime('%Y%m%d')
        stock_code = row['stock_code']

        # 在ST数据中查找
        in_st = st_df[
            (st_df['trade_date'] == date_str) &
            (st_df['ts_code'] == stock_code)
        ]

        if len(in_st) > 0:
            print(f"    [OK] {date_str} {stock_code}: 确认是ST股票")
        else:
            print(f"    [FAIL] {date_str} {stock_code}: 误报，不在ST数据中")
            verification_passed = False

    if verification_passed:
        print(f"\n  [OK] 验证通过：所有被过滤的股票都在ST数据中")
    else:
        print(f"\n  [FAIL] 验证失败：存在误报")
else:
    print(f"  没有记录被过滤（可能测试数据中的股票都不在ST列表中）")

# 5. 性能测试
print("\n[5] 性能测试...")

import time

# 大数据集测试
large_data = pd.DataFrame({
    'stock_code': [f"{str(i).zfill(6)}.SZ" for i in range(3000)] * 100,  # 30万条
    'date': pd.date_range('2024-01-01', '2024-05-10')[:100].repeat(3000),
    'factor1': np.random.randn(300000)
})

print(f"  大数据集: {len(large_data):,} 条记录")

start_time = time.time()
st_filter_large = _filter_st_stocks(large_data, logger)
elapsed_time = time.time() - start_time

print(f"  过滤耗时: {elapsed_time:.3f} 秒")
print(f"  性能: {len(large_data)/elapsed_time:,.0f} 条/秒")

if elapsed_time < 1.0:
    print(f"  [OK] 性能优秀（< 1秒）")
elif elapsed_time < 3.0:
    print(f"  [WARN] 性能一般（1-3秒）")
else:
    print(f"  [FAIL] 性能较差（> 3秒）")

print(f"\n{'=' * 80}")
print("测试完成")
print("=" * 80)
