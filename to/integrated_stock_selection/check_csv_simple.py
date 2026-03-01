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
    # 列出目录中的文件
    files = os.listdir(results_dir)
    csv_files = [f for f in files if f.endswith('.csv')]

    print(f"目录中找到 {len(csv_files)} 个CSV文件：")
    for f in csv_files[:5]:
        print(f"  - {f}")

    # 读取第一个详细文件
    if csv_files:
        first_file = os.path.join(results_dir, csv_files[0])
        df = pd.read_csv(first_file)

        print(f"\n读取文件: {csv_files[0]}")
        print(f"总行数: {len(df)}")
        print(f"\n列名: {df.columns.tolist()}")
        print(f"\n前10行数据:")
        print(df.head(10).to_string())

        # 按持有期统计
        print("\n按holding_day分组统计:")
        print("-" * 80)
        for day_offset in range(1, 4):
            day_data = df[df['holding_day'] == day_offset]
            if not day_data.empty:
                print(f"\nT+{day_offset}:")
                print(f"  记录数: {len(day_data)}")
                print(f"  平均收益: {day_data['actual_return'].mean():.4f}")

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
