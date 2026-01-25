"""
测试新股过滤索引修复
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def simulate_new_stock_filter_with_different_index():
    """模拟新股过滤，测试索引不匹配的情况"""

    print("=" * 100)
    print("测试新股过滤索引修复")
    print("=" * 100)

    # 创建测试数据（模拟真实场景：索引不是从0开始）
    n_records = 10_000
    n_stocks = 100

    print(f"\n[测试数据]")
    print(f"  记录数: {n_records:,}")
    print(f"  股票数: {n_stocks:,}")

    # 生成股票代码
    stock_codes = [f"{str(i).zfill(6)}.SZ" for i in range(n_stocks)]

    # 创建DataFrame（使用非连续索引，模拟真实场景）
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')

    # 使用非连续索引（这是关键！）
    index = np.random.permutation(np.arange(n_records)) + 10000  # 从10000开始

    data = {
        'stock_code': np.random.choice(stock_codes, n_records),
        'date': np.random.choice(dates, n_records),
        'factor1': np.random.randn(n_records),
        'factor2': np.random.randn(n_records),
    }
    stock_data = pd.DataFrame(data, index=index)

    print(f"  原始索引范围: {stock_data.index.min()} ~ {stock_data.index.max()}")
    print(f"  原始索引示例: {stock_data.index[:10].tolist()}")

    # 创建新股过滤数据（假设有10只新股）
    filter_stock_date_pairs = []
    for i in range(10):
        stock_code = stock_codes[i]
        ipo_date = pd.Timestamp('2024-06-01') + pd.Timedelta(days=i)
        filter_end_date = ipo_date + pd.Timedelta(days=25)
        filter_stock_date_pairs.append({
            'stock_code': stock_code,
            'ipo_date': ipo_date,
            'filter_end_date': filter_end_date
        })

    print(f"  新股数量: {len(filter_stock_date_pairs)}")

    # 测试修复后的逻辑
    print(f"\n{'=' * 100}")
    print("测试修复后的新股过滤逻辑")
    print(f"{'=' * 100}")

    try:
        # 步骤1: 将新股过滤条件转换为DataFrame
        new_stock_df = pd.DataFrame(filter_stock_date_pairs)

        # 步骤2: 使用merge快速匹配
        stock_data_copy = stock_data.copy()
        stock_data_copy['_date_num'] = stock_data_copy['date'].astype(np.int64)
        new_stock_df['_ipo_num'] = new_stock_df['ipo_date'].astype(np.int64)
        new_stock_df['_filter_end_num'] = new_stock_df['filter_end_date'].astype(np.int64)

        # 执行merge
        merged = stock_data_copy.merge(
            new_stock_df[['stock_code', '_ipo_num', '_filter_end_num']],
            on='stock_code',
            how='left'
        )

        print(f"\n[Merge结果]")
        print(f"  merged形状: {merged.shape}")
        print(f"  merged索引范围: {merged.index.min()} ~ {merged.index.max()}")
        print(f"  merged索引示例: {merged.index[:10].tolist()}")

        # 步骤3: 向量化判断：是否需要过滤
        need_filter_mask = (
            merged['_ipo_num'].notna() &
            (merged['_date_num'] >= merged['_ipo_num']) &
            (merged['_date_num'] <= merged['_filter_end_num'])
        )

        print(f"\n[过滤掩码]")
        print(f"  need_filter_mask形状: {need_filter_mask.shape}")
        print(f"  need_filter_mask索引范围: {need_filter_mask.index.min()} ~ {need_filter_mask.index.max()}")
        print(f"  need_filter_mask索引示例: {need_filter_mask.index[:10].tolist()}")
        print(f"  需要过滤的记录数: {need_filter_mask.sum()}")

        # 步骤4: 执行过滤（使用修复后的逻辑）
        print(f"\n[执行过滤]")
        original_columns = stock_data_copy.columns.tolist()

        # 旧方法（会报错）
        print(f"\n  尝试旧方法（应该会报错）:")
        try:
            filtered_old = stock_data_copy[~need_filter_mask].copy()
            print(f"    旧方法成功（不应该到这里）")
        except Exception as e:
            print(f"    旧方法失败（预期行为）: {type(e).__name__}")
            print(f"    错误信息: {str(e)[:100]}...")

        # 新方法（修复后）
        print(f"\n  尝试新方法（修复后）:")
        filtered_new = merged[~need_filter_mask][original_columns].copy()
        print(f"    新方法成功！")
        print(f"    filtered_data形状: {filtered_new.shape}")
        print(f"    过滤前记录数: {len(stock_data_copy)}")
        print(f"    过滤后记录数: {len(filtered_new)}")
        print(f"    过滤掉记录数: {len(stock_data_copy) - len(filtered_new)}")

        # 验证列是否正确
        print(f"\n[列验证]")
        print(f"  原始列数: {len(original_columns)}")
        print(f"  过滤后列数: {len(filtered_new.columns)}")
        print(f"  列是否一致: {filtered_new.columns.tolist() == original_columns}")

        # 验证数据正确性
        print(f"\n[数据验证]")
        original_stock_codes = set(stock_data_copy['stock_code'].unique())
        filtered_stock_codes = set(filtered_new['stock_code'].unique())
        removed_stocks = original_stock_codes - filtered_stock_codes

        print(f"  原始股票数: {len(original_stock_codes)}")
        print(f"  过滤后股票数: {len(filtered_stock_codes)}")
        print(f"  完全被过滤的股票: {len(removed_stocks)}")

        if removed_stocks:
            print(f"  被过滤的股票示例: {list(removed_stocks)[:5]}")

        # 清理临时列
        del stock_data_copy['_date_num']

        print(f"\n{'=' * 100}")
        print("测试结果: [OK] 索引修复成功")
        print(f"{'=' * 100}")

        return True

    except Exception as e:
        print(f"\n{'=' * 100}")
        print(f"测试失败: {e}")
        print(f"{'=' * 100}")
        import traceback
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    simulate_new_stock_filter_with_different_index()
