"""
查看生成的分桶分析结果
"""
import pandas as pd
import sys
import os

# 设置UTF-8编码
sys.stdout.reconfigure(encoding='utf-8')

results_dir = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\results\pred_bucket_analysis"

try:
    # 读取汇总文件
    summary_file = os.path.join(results_dir, "bucket_summary_20260213_232408.csv")
    df = pd.read_csv(summary_file)

    print("=" * 80)
    print("预测分桶分析汇总统计")
    print("=" * 80)
    print(f"\n总记录数: {len(df)}")
    print(f"\n列名: {df.columns.tolist()}")
    print(f"\n前10行数据:")
    print(df.head(10).to_string())

    # 按持有期分组统计
    print("\n按持有期分组:")
    print("-" * 80)
    for day_offset in range(1, 4):
        day_data = df[df['holding_day'] == day_offset]
        if not day_data.empty:
            print(f"\nT+{day_offset} 统计:")
            print(f"  记录数: {len(day_data)}")

            # 按桶分组
            for bucket_name in day_data['bucket'].unique():
                bucket_day_data = day_data[day_data['bucket'] == bucket_name]
                print(f"\n  {bucket_name}:")
                print(f"    样本数: {len(bucket_day_data)}")
                print(f"    平均收益: {bucket_day_data['mean_return'].mean():.4f}")
                print(f"    中位数收益: {bucket_day_data['median_return'].median():.4f}")

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
