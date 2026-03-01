"""
读取生成的分桶分析结果CSV
"""
import pandas as pd
import sys
import os

# 设置UTF-8编码
sys.stdout.reconfigure(encoding='utf-8')

results_dir = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\results\pred_bucket_analysis"

try:
    # 读取第一个详细文件
    files = [f for f in os.listdir(results_dir) if 'P0~P20' in f and 'detailed' in f]
    if files:
        first_file = os.path.join(results_dir, files[0])
        df = pd.read_csv(first_file)

        print("=" * 80)
        print(f"文件: {os.path.basename(first_file)}")
        print(f"总行数: {len(df)}")
        print(f"列名: {df.columns.tolist()}")
        print(f"\n前10行数据:")
        print(df.head(10).to_string())

        # 按持有期分组
        print("\n按holding_day分组统计:")
        print("-" * 80)
        for day_offset in range(1, 4):
            day_data = df[df['holding_day'] == day_offset]
            if not day_data.empty:
                print(f"\nT+{day_offset}期:")
                print(f"  记录数: {len(day_data)}")
                print(f"  平均收益: {day_data['actual_return'].mean():.4f}")

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
