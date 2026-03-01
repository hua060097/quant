import csv

# 读取文件
with open(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\results\pred_bucket_analysis\bucket_P0~P20_detailed_20260213_232011.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

    print("CSV文件前10行:")
    for i, row in enumerate(rows[:10]):
        print(f"行{i+1}: {row}")

print(f"\n总共 {len(rows)} 行")
