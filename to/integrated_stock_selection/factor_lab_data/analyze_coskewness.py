"""
详细分析 coskewness 因子的差异原因
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_coskewness_factor(file1_path, file2_path):
    """详细分析 coskewness 因子"""

    print("=" * 100)
    print("coskewness 因子详细分析")
    print("=" * 100)

    # 读取文件
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    print(f"\n[原始文件信息]")
    print(f"  文件1 (factors.parquet): {df1.shape[0]:,} 行 × {df1.shape[1]} 列")
    print(f"  文件2 (factors-correct.parquet): {df2.shape[0]:,} 行 × {df2.shape[1]} 列")

    # 日期过滤：2023-06-01 至 2025-12-26
    start_date = pd.Timestamp('2023-06-01')
    end_date = pd.Timestamp('2025-12-26')

    df1['_date'] = pd.to_datetime(df1['date'])
    df2['_date'] = pd.to_datetime(df2['date'])

    df1_filtered = df1[(df1['_date'] >= start_date) & (df1['_date'] <= end_date)].copy()
    df2_filtered = df2[(df2['_date'] >= start_date) & (df2['_date'] <= end_date)].copy()

    print(f"\n[日期范围过滤]")
    print(f"  目标日期范围: 2023-06-01 至 2025-12-26")
    print(f"  文件1过滤后: {df1_filtered.shape[0]:,} 行")
    print(f"  文件2过滤后: {df2_filtered.shape[0]:,} 行")

    # 创建复合键对齐
    df1_filtered['_key'] = df1_filtered['stock_code'].astype(str) + '_' + df1_filtered['_date'].astype(str)
    df2_filtered['_key'] = df2_filtered['stock_code'].astype(str) + '_' + df2_filtered['_date'].astype(str)

    common_keys = set(df1_filtered['_key']) & set(df2_filtered['_key'])

    df1_aligned = df1_filtered[df1_filtered['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)
    df2_aligned = df2_filtered[df2_filtered['_key'].isin(common_keys)].sort_values('_key').reset_index(drop=True)

    print(f"\n[对齐后数据]")
    print(f"  对齐行数: {len(df1_aligned):,}")

    # 分析 coskewness 因子
    factor = 'coskewness'

    if factor not in df1_aligned.columns or factor not in df2_aligned.columns:
        print(f"\n[ERROR] 因子 {factor} 不存在")
        return

    print("\n" + "=" * 100)
    print(f"因子: {factor}")
    print("=" * 100)

    col1 = pd.to_numeric(df1_aligned[factor], errors='coerce')
    col2 = pd.to_numeric(df2_aligned[factor], errors='coerce')

    valid_mask = ~(col1.isna() | col2.isna())

    if valid_mask.sum() == 0:
        print(f"\n[WARNING] 没有有效数据")
        return

    diff = col2 - col1
    abs_diff = np.abs(diff)

    diff_valid = diff[valid_mask]
    abs_diff_valid = abs_diff[valid_mask]

    print(f"\n[基本统计]")
    print(f"  有效比较行数: {valid_mask.sum():,}")

    print(f"\n  文件1:")
    print(f"    均值: {col1.mean():.6f}")
    print(f"    标准差: {col1.std():.6f}")
    print(f"    最小值: {col1.min():.6f}")
    print(f"    最大值: {col1.max():.6f}")
    print(f"    中位数: {col1.median():.6f}")

    print(f"\n  文件2:")
    print(f"    均值: {col2.mean():.6f}")
    print(f"    标准差: {col2.std():.6f}")
    print(f"    最小值: {col2.min():.6f}")
    print(f"    最大值: {col2.max():.6f}")
    print(f"    中位数: {col2.median():.6f}")

    print(f"\n[差异分析]")
    print(f"  完全相同: {(diff_valid == 0).sum():,} ({(diff_valid == 0).sum()/len(diff_valid)*100:.4f}%)")
    print(f"  最大差异: {abs_diff_valid.max():.6f}")
    print(f"  平均差异: {abs_diff_valid.mean():.6f}")
    print(f"  中位数差异: {abs_diff_valid.median():.6f}")

    # 差异分布
    print(f"\n[差异分布]")
    bins = [0, 1e-9, 0.001, 0.01, 0.1, 1.0, float('inf')]
    labels = ["=0", "0~0.001", "0.001~0.01", "0.01~0.1", "0.1~1.0", ">1.0"]

    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        if i == len(bins)-2:
            count = (abs_diff_valid >= low).sum()
        else:
            count = ((abs_diff_valid >= low) & (abs_diff_valid < high)).sum()
        pct = count / len(diff_valid) * 100
        print(f"  {labels[i]:<15} {count:>10,} ({pct:>8.4f}%)")

    # 分析极端值
    extreme_threshold = 5.0
    extreme1 = (col1.abs() > extreme_threshold).sum()
    extreme2 = (col2.abs() > extreme_threshold).sum()

    print(f"\n[极端值分析 (阈值={extreme_threshold})]")
    print(f"  文件1极端值数量: {extreme1} ({extreme1/valid_mask.sum()*100:.2f}%)")
    print(f"  文件2极端值数量: {extreme2} ({extreme2/valid_mask.sum()*100:.2f}%)")

    # 分析NaN情况
    nan1 = col1.isna().sum()
    nan2 = col2.isna().sum()

    print(f"\n[NaN值分析]")
    print(f"  文件1 NaN数量: {nan1} ({nan1/len(col1)*100:.2f}%)")
    print(f"  文件2 NaN数量: {nan2} ({nan2/len(col2)*100:.2f}%)")

    # 最大差异样本分析
    print(f"\n[最大差异的前20个样本]")
    top_indices = abs_diff_valid.nlargest(20).index
    print(f"{'排名':<6} {'索引':<10} {'股票代码':<15} {'日期':<12} {'文件1值':<12} {'文件2值':<12} {'差异':<12}")
    print("-" * 100)

    for rank, idx in enumerate(top_indices, 1):
        stock_code = df1_aligned.loc[idx, 'stock_code']
        date = df1_aligned.loc[idx, 'date']
        val1 = col1[idx]
        val2 = col2[idx]
        diff_val = diff[idx]

        print(f"{rank:<6} {idx:<10} {stock_code:<15} {str(date):<12} {val1:<12.6f} {val2:<12.6f} {diff_val:<12.6f}")

    # 按日期分析差异
    print(f"\n[按日期的差异统计]")
    df1_aligned['_diff'] = abs_diff
    df1_aligned['_valid'] = valid_mask

    date_stats = df1_aligned[df1_aligned['_valid']].groupby('_date').agg({
        '_diff': ['mean', 'max', 'count'],
        factor: ['mean', 'std']
    }).round(6)

    date_stats.columns = ['mean_diff', 'max_diff', 'count', 'mean_val', 'std_val']
    date_stats = date_stats.sort_values('mean_diff', ascending=False)

    print(f"\n  平均差异最大的前20个日期:")
    print(f"{'日期':<12} {'平均差异':<12} {'最大差异':<12} {'样本数':<10} {'因子均值':<12} {'因子标准差':<12}")
    print("-" * 80)

    for date, row in date_stats.head(20).iterrows():
        date_str = str(date.date())
        print(f"{date_str:<12} {row['mean_diff']:<12.6f} {row['max_diff']:<12.6f} {int(row['count']):<10} {row['mean_val']:<12.6f} {row['std_val']:<12.6f}")

    # 相关性分析
    print(f"\n[相关性分析]")
    correlation = col1[valid_mask].corr(col2[valid_mask])
    print(f"  相关系数: {correlation:.8f}")

    # 尝试线性拟合
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        col1[valid_mask].dropna(),
        col2[valid_mask].dropna()
    )
    print(f"  线性拟合: col2 = {slope:.8f} * col1 + {intercept:.8f}")
    print(f"  R^2: {r_value**2:.8f}")
    print(f"  P值: {p_value:.2e}")

    # 检查是否有系统性偏差
    bias = diff_valid.mean()
    print(f"\n[系统性偏差]")
    print(f"  平均偏差 (col2 - col1): {bias:.6f}")
    if abs(bias) > 0.01:
        print(f"  结论: 存在系统性偏差")
    else:
        print(f"  结论: 无显著系统性偏差")

    print(f"\n[分析完成]")

if __name__ == "__main__":
    # 文件路径
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")
    file1 = base_path / "factors.parquet"
    file2 = base_path / "factors-correct.parquet"

    analyze_coskewness_factor(file1, file2)
