"""
获取差异超过1%的因子列表
"""
import pandas as pd
import numpy as np
from pathlib import Path

def get_diff_factors(file1_path, file2_path, threshold=1.0):
    """获取差异超过阈值的因子"""

    # 读取文件
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    # 日期过滤
    start_date = pd.Timestamp('2023-06-01')
    end_date = pd.Timestamp('2025-12-26')

    df1['_date'] = pd.to_datetime(df1['date'])
    df2['_date'] = pd.to_datetime(df2['date'])

    df1_filtered = df1[(df1['_date'] >= start_date) & (df1['_date'] <= end_date)].copy()
    df2_filtered = df2[(df2['_date'] >= start_date) & (df2['_date'] <= end_date)].copy()

    # 创建复合键对齐
    df1_filtered['_key'] = df1_filtered['stock_code'].astype(str) + '_' + df1_filtered['_date'].astype(str)
    df2_filtered['_key'] = df2_filtered['stock_code'].astype(str) + '_' + df2_filtered['_date'].astype(str)

    common_keys = set(df1_filtered['_key']) & set(df2_filtered['_key'])

    df1_aligned = df1_filtered[df1_filtered['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)
    df2_aligned = df2_filtered[df2_filtered['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)

    # 找共同因子
    exclude_cols = {'stock_code', 'date'}
    cols1 = set(df1.columns) - exclude_cols
    cols2 = set(df2.columns) - exclude_cols
    common_cols = sorted(list(cols1 & cols2))

    # 分析每个因子
    diff_factors = []

    for factor in common_cols:
        col1 = pd.to_numeric(df1_aligned[factor], errors='coerce')
        col2 = pd.to_numeric(df2_aligned[factor], errors='coerce')

        valid_mask = ~(col1.isna() | col2.isna())

        if valid_mask.sum() > 0:
            abs_diff = np.abs(col2 - col1)
            pct_diff = (abs_diff[valid_mask] > 0.01).sum() / valid_mask.sum() * 100

            if pct_diff > threshold:
                max_diff = abs_diff[valid_mask].max()
                mean_diff = abs_diff[valid_mask].mean()
                exact_match_pct = ((col2[valid_mask] - col1[valid_mask]) == 0).sum() / valid_mask.sum() * 100

                diff_factors.append({
                    'factor': factor,
                    'pct_diff': pct_diff,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'exact_match_pct': exact_match_pct
                })

    # 排序
    diff_factors.sort(key=lambda x: x['pct_diff'], reverse=True)

    print("=" * 100)
    print(f"差异超过{threshold}%的因子（共{len(diff_factors)}个）:")
    print("=" * 100)
    print(f"{'因子名':<50} {'差异>0.01占比':>15} {'最大差异':>12} {'平均差异':>12} {'完全相同占比':>15}")
    print("-" * 105)

    for item in diff_factors:
        print(f"{item['factor']:<50} {item['pct_diff']:>14.2f}% {item['max_diff']:>12.6f} {item['mean_diff']:>12.6f} {item['exact_match_pct']:>14.2f}%")

    return diff_factors

if __name__ == "__main__":
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")
    file1 = base_path / "factors.parquet"
    file2 = base_path / "factors-correct.parquet"

    diff_factors = get_diff_factors(file1, file2, threshold=1.0)
