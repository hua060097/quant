"""
对比2025-08-08两个因子文件中的股票数量
"""
import pandas as pd
from pathlib import Path

def compare_stock_count_on_date(file1_path, file2_path, target_date='2025-08-08'):
    """对比指定日期的股票数量"""

    print("=" * 100)
    print(f"对比日期: {target_date}")
    print("=" * 100)

    # 读取文件
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    print(f"\n[文件基本信息]")
    print(f"  文件1 (factors.parquet): {df1.shape[0]:,} 行 × {df1.shape[1]} 列")
    print(f"  文件2 (factors-correct.parquet): {df2.shape[0]:,} 行 × {df2.shape[1]} 列")

    # 转换日期列
    df1['_date'] = pd.to_datetime(df1['date'])
    df2['_date'] = pd.to_datetime(df2['date'])

    # 过滤目标日期
    target_date_parsed = pd.Timestamp(target_date)

    df1_date = df1[df1['_date'] == target_date_parsed].copy()
    df2_date = df2[df2['_date'] == target_date_parsed].copy()

    print(f"\n[{target_date} 的数据统计]")
    print(f"  文件1股票数量: {len(df1_date):,}")
    print(f"  文件2股票数量: {len(df2_date):,}")
    print(f"  差异: {len(df2_date) - len(df1_date):+,}")

    # 获取股票列表
    stocks1 = set(df1_date['stock_code'].unique()) if len(df1_date) > 0 else set()
    stocks2 = set(df2_date['stock_code'].unique()) if len(df2_date) > 0 else set()

    print(f"\n[股票列表分析]")
    print(f"  文件1独有股票: {len(stocks1 - stocks2)}")
    print(f"  文件2独有股票: {len(stocks2 - stocks1)}")
    print(f"  共同股票: {len(stocks1 & stocks2)}")

    # 分析 market_trend_adaptive_beta 因子
    factor = 'market_trend_adaptive_beta'

    if factor in df1_date.columns and factor in df2_date.columns:
        print(f"\n[{factor} 因子分析]")

        # 统计有效值
        col1 = pd.to_numeric(df1_date[factor], errors='coerce')
        col2 = pd.to_numeric(df2_date[factor], errors='coerce')

        valid1 = col1.notna().sum()
        valid2 = col2.notna().sum()

        print(f"  文件1有效值数量: {valid1} / {len(df1_date)}")
        print(f"  文件2有效值数量: {valid2} / {len(df2_date)}")

        # 统计值分布
        print(f"\n  文件1值分布:")
        print(f"    均值: {col1.mean():.6f}")
        print(f"    标准差: {col1.std():.6f}")
        print(f"    最小值: {col1.min():.6f}")
        print(f"    最大值: {col1.max():.6f}")
        print(f"    中位数: {col1.median():.6f}")

        print(f"\n  文件2值分布:")
        print(f"    均值: {col2.mean():.6f}")
        print(f"    标准差: {col2.std():.6f}")
        print(f"    最小值: {col2.min():.6f}")
        print(f"    最大值: {col2.max():.6f}")
        print(f"    中位数: {col2.median():.6f}")

        # 统计特殊值
        val1_counts = col1.value_counts()
        val2_counts = col2.value_counts()

        print(f"\n  文件1高频值 (Top 5):")
        for val, count in val1_counts.head(5).items():
            pct = count / valid1 * 100
            print(f"    {val:>12.6f}: {count:>6} ({pct:>5.2f}%)")

        print(f"\n  文件2高频值 (Top 5):")
        for val, count in val2_counts.head(5).items():
            pct = count / valid2 * 100
            print(f"    {val:>12.6f}: {count:>6} ({pct:>5.2f}%)")

        # 检查异常值（>50或<-50）
        abnormal_threshold = 50
        abnormal1 = (col1.abs() > abnormal_threshold).sum()
        abnormal2 = (col2.abs() > abnormal_threshold).sum()

        print(f"\n  异常值统计 (绝对值>{abnormal_threshold}):")
        print(f"    文件1异常值数量: {abnormal1} ({abnormal1/valid1*100:.2f}%)")
        print(f"    文件2异常值数量: {abnormal2} ({abnormal2/valid2*100:.2f}%)")

        if abnormal1 > 0:
            print(f"\n    文件1异常值样本 (前10个):")
            abnormal_indices1 = col1[col1.abs() > abnormal_threshold].index
            for idx in list(abnormal_indices1)[:10]:
                stock = df1_date.loc[idx, 'stock_code']
                val = col1[idx]
                print(f"      {stock}: {val:.6f}")

        if abnormal2 > 0:
            print(f"\n    文件2异常值样本 (前10个):")
            abnormal_indices2 = col2[col2.abs() > abnormal_threshold].index
            for idx in list(abnormal_indices2)[:10]:
                stock = df2_date.loc[idx, 'stock_code']
                val = col2[idx]
                print(f"      {stock}: {val:.6f}")

        # 对比共同股票的值
        common_stocks = stocks1 & stocks2
        if len(common_stocks) > 0:
            print(f"\n[共同股票对比 ({len(common_stocks)}只)]")

            # 创建对比DataFrame
            df1_common = df1_date[df1_date['stock_code'].isin(common_stocks)].set_index('stock_code')
            df2_common = df2_date[df2_date['stock_code'].isin(common_stocks)].set_index('stock_code')

            compare_df = pd.DataFrame({
                'file1_value': col1.loc[df1_common.index],
                'file2_value': col2.loc[df2_common.index]
            })

            compare_df['diff'] = compare_df['file2_value'] - compare_df['file1_value']
            compare_df['abs_diff'] = compare_df['diff'].abs()

            print(f"  完全相同: {(compare_df['diff'] == 0).sum()} ({(compare_df['diff'] == 0).sum()/len(compare_df)*100:.2f}%)")
            print(f"  有差异: {(compare_df['diff'] != 0).sum()} ({(compare_df['diff'] != 0).sum()/len(compare_df)*100:.2f}%)")
            print(f"  最大差异: {compare_df['abs_diff'].max():.6f}")
            print(f"  平均差异: {compare_df['abs_diff'].mean():.6f}")

            # 找出差异最大的股票
            top_diff = compare_df.nlargest(10, 'abs_diff')
            print(f"\n  差异最大的10只股票:")
            for stock, row in top_diff.iterrows():
                print(f"    {stock}: 文件1={row['file1_value']:.6f}, 文件2={row['file2_value']:.6f}, 差异={row['diff']:.6f}")

    # 检查Beta计算依赖的变量
    beta_related_cols = ['pct_chg', 'beta_60d']

    print(f"\n[Beta相关变量检查]")

    for col in beta_related_cols:
        if col in df1.columns and col in df2.columns:
            col1_data = pd.to_numeric(df1_date[col], errors='coerce')
            col2_data = pd.to_numeric(df2_date[col], errors='coerce')

            print(f"\n  {col}:")
            print(f"    文件1有效值: {col1_data.notna().sum()} / {len(df1_date)}")
            print(f"    文件2有效值: {col2_data.notna().sum()} / {len(df2_date)}")

            if col == 'beta_60d' and col1_data.notna().sum() > 0 and col2_data.notna().sum() > 0:
                print(f"    文件1统计: 均值={col1_data.mean():.6f}, 标准差={col1_data.std():.6f}, 范围=[{col1_data.min():.6f}, {col1_data.max():.6f}]")
                print(f"    文件2统计: 均值={col2_data.mean():.6f}, 标准差={col2_data.std():.6f}, 范围=[{col2_data.min():.6f}, {col2_data.max():.6f}]")

                # 统计极端Beta值
                extreme_threshold = 10
                extreme1 = (col1_data.abs() > extreme_threshold).sum()
                extreme2 = (col2_data.abs() > extreme_threshold).sum()
                print(f"    文件1极端Beta(>{extreme_threshold}): {extreme1} ({extreme1/col1_data.notna().sum()*100:.2f}%)")
                print(f"    文件2极端Beta(>{extreme_threshold}): {extreme2} ({extreme2/col2_data.notna().sum()*100:.2f}%)")

    print(f"\n[分析完成]")

if __name__ == "__main__":
    # 文件路径
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")
    file1 = base_path / "factors.parquet"
    file2 = base_path / "factors-correct.parquet"

    compare_stock_count_on_date(file1, file2, '2025-08-08')
