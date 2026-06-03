"""
对比 factors.parquet 与 factors_20260513.parquet 最后120个交易日的因子差异
"""
import pandas as pd
import numpy as np

# 读取数据
print("读取 factors.parquet ...")
f1 = pd.read_parquet('factor_lab_data/factors.parquet')
print("读取 factors_20260513.parquet ...")
f2 = pd.read_parquet('factor_lab_data/factors_20260513.parquet')

# 获取最后120个交易日
all_dates = sorted(f1['date'].unique())
last_120_dates = all_dates[-120:]
print(f"\n最后120个交易日: {last_120_dates[0].date()} ~ {last_120_dates[-1].date()}")

# 筛选最后120个交易日
f1_tail = f1[f1['date'].isin(last_120_dates)].copy()
f2_tail = f2[f2['date'].isin(last_120_dates)].copy()

print(f"f1_tail shape: {f1_tail.shape}, f2_tail shape: {f2_tail.shape}")

# 获取因子列（排除 date 和 stock_code）
factor_cols = [c for c in f1.columns if c not in ('date', 'stock_code')]
print(f"因子列数: {len(factor_cols)}")

# 按 date + stock_code 对齐
f1_tail = f1_tail.set_index(['date', 'stock_code']).sort_index()
f2_tail = f2_tail.set_index(['date', 'stock_code']).sort_index()

# 确保索引完全一致
common_idx = f1_tail.index.intersection(f2_tail.index)
print(f"共同索引行数: {len(common_idx)}")

f1_aligned = f1_tail.loc[common_idx, factor_cols]
f2_aligned = f2_tail.loc[common_idx, factor_cols]

# ============ 1. 逐因子差异分析 ============
print("\n" + "="*80)
print("1. 逐因子差异分析（最后120个交易日）")
print("="*80)

results = []
for col in factor_cols:
    s1 = f1_aligned[col]
    s2 = f2_aligned[col]

    # 都是NaN的位置
    both_nan = s1.isna() & s2.isna()
    one_nan = s1.isna() ^ s2.isna()

    # 有效差异（非NaN的值不同）
    valid_mask = ~(s1.isna() | s2.isna())
    s1_valid = s1[valid_mask]
    s2_valid = s2[valid_mask]

    diff = (s1_valid - s2_valid).abs()
    max_diff = diff.max() if len(diff) > 0 else 0
    mean_diff = diff.mean() if len(diff) > 0 else 0
    median_diff = diff.median() if len(diff) > 0 else 0
    n_diff_nonzero = (diff > 1e-10).sum() if len(diff) > 0 else 0
    pct_diff = n_diff_nonzero / len(diff) * 100 if len(diff) > 0 else 0

    # 相对差异（相对于f2的绝对值均值）
    abs_mean_f2 = s2_valid.abs().mean() if len(s2_valid) > 0 else 1
    rel_diff_pct = mean_diff / abs_mean_f2 * 100 if abs_mean_f2 > 1e-10 else 0

    # NaN差异
    n_one_nan = one_nan.sum()
    n_both_nan = both_nan.sum()

    results.append({
        'factor': col,
        'max_abs_diff': max_diff,
        'mean_abs_diff': mean_diff,
        'median_abs_diff': median_diff,
        'n_nonzero_diff': n_diff_nonzero,
        'pct_nonzero_diff': pct_diff,
        'relative_diff_pct': rel_diff_pct,
        'n_one_nan_diff': n_one_nan,
        'n_both_nan': n_both_nan,
        'n_valid_rows': len(diff),
    })

df_results = pd.DataFrame(results)

# 按最大绝对差异排序
df_results_sorted = df_results.sort_values('max_abs_diff', ascending=False)

print("\n--- 有差异的因子（非零差异行数 > 0） ---")
has_diff = df_results_sorted[df_results_sorted['n_nonzero_diff'] > 0]
if len(has_diff) == 0:
    print("所有因子完全一致！")
else:
    print(f"有差异的因子数: {len(has_diff)} / {len(factor_cols)}")
    print(f"\nTop 20 最大绝对差异因子:")
    print(has_diff.head(20).to_string(index=False))

    print(f"\nTop 20 最大相对差异因子（相对均值）:")
    has_diff_rel = df_results_sorted[df_results_sorted['n_nonzero_diff'] > 0].sort_values('relative_diff_pct', ascending=False)
    print(has_diff_rel.head(20)[['factor', 'relative_diff_pct', 'mean_abs_diff', 'max_abs_diff', 'pct_nonzero_diff']].to_string(index=False))

    print(f"\nTop 20 差异行占比最高的因子:")
    has_diff_pct = df_results_sorted[df_results_sorted['n_nonzero_diff'] > 0].sort_values('pct_nonzero_diff', ascending=False)
    print(has_diff_pct.head(20)[['factor', 'pct_nonzero_diff', 'mean_abs_diff', 'max_abs_diff', 'n_nonzero_diff', 'n_valid_rows']].to_string(index=False))

# ============ 2. NaN差异分析 ============
print("\n" + "="*80)
print("2. NaN差异分析")
print("="*80)
nan_diff = df_results[df_results['n_one_nan_diff'] > 0].sort_values('n_one_nan_diff', ascending=False)
if len(nan_diff) == 0:
    print("没有NaN差异！")
else:
    print(f"有NaN差异的因子数: {len(nan_diff)}")
    print(nan_diff[['factor', 'n_one_nan_diff', 'n_valid_rows', 'n_both_nan']].to_string(index=False))

# ============ 3. 完全一致的因子 ============
print("\n" + "="*80)
print("3. 完全一致的因子")
print("="*80)
no_diff = df_results[df_results['n_nonzero_diff'] == 0]
print(f"完全一致的因子数: {len(no_diff)} / {len(factor_cols)}")

# ============ 4. 最大差异因子的具体值示例 ============
print("\n" + "="*80)
print("4. 最大差异因子具体值示例")
print("="*80)
if len(has_diff) > 0:
    top_factor = has_diff.iloc[0]['factor']
    print(f"\n差异最大的因子: {top_factor}")
    s1_top = f1_aligned[top_factor]
    s2_top = f2_aligned[top_factor]
    diff_top = (s1_top - s2_top).abs()
    # 取差异最大的5行
    top5_idx = diff_top.nlargest(5).index
    compare_df = pd.DataFrame({
        'factors.parquet': s1_top.loc[top5_idx].values,
        'factors_20260513.parquet': s2_top.loc[top5_idx].values,
        'diff': diff_top.loc[top5_idx].values,
    }, index=[f"{idx[0].date()}_{idx[1]}" for idx in top5_idx])
    print(compare_df.to_string())

# ============ 5. 汇总统计 ============
print("\n" + "="*80)
print("5. 汇总统计")
print("="*80)
print(f"总因子数: {len(factor_cols)}")
print(f"完全一致: {len(no_diff)}")
print(f"有数值差异: {len(has_diff)}")
print(f"有NaN差异: {len(nan_diff)}")
if len(has_diff) > 0:
    print(f"最大绝对差异: {has_diff['max_abs_diff'].max():.8f} (因子: {has_diff.iloc[0]['factor']})")
    print(f"平均差异占比: {has_diff['pct_nonzero_diff'].mean():.2f}%")
