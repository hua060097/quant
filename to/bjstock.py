import pandas as pd
import json
import os

# 确保目标目录存在
os.makedirs('to/local_data', exist_ok=True)

# 读取Excel文件
excel_path = 'to/docs/北交所全部股票.xlsx'
# df = pd.read_excel(excel_path)

# # 假设股票代码列的列名为"股票代码"，如果不是请替换为实际列名
# # 如果不知道列名，可以先打印df.columns查看
# stock_codes = df['股票代码'].tolist()

# # 将股票代码列表保存为JSON文件
json_path = 'to/local_data/北交所股票代码.json'
# with open(json_path, 'w', encoding='utf-8') as f:
#     json.dump(stock_codes, f, ensure_ascii=False, indent=4)
# print(f"已成功将{len(stock_codes)}个北交所股票代码保存至{json_path}")    

# 读取JSON文件中的股票代码列表
with open(json_path, 'r', encoding='utf-8') as f:
    stock_codes_from_json = json.load(f)

print(f"\n从JSON文件读取的北交所股票代码列表:")
print(f"共{len(stock_codes_from_json)}只股票:")
print(type(stock_codes_from_json))
print(stock_codes_from_json)

# 为每个股票代码添加.BJ后缀
stock_codes_with_suffix = [f"{code}.BJ" for code in stock_codes_from_json]

# 将带后缀的股票代码列表写入JSON文件
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(stock_codes_with_suffix, f, ensure_ascii=False, indent=4)

print(f"\n已将添加.BJ后缀的{len(stock_codes_with_suffix)}个股票代码保存至{json_path}")
print(f"示例: {stock_codes_with_suffix[:5]}")

# 更新stock_codes_from_json为带后缀的版本
stock_codes_from_json = stock_codes_with_suffix


# 将列表转换为字符串
stock_codes_str = ', '.join(str(code) for code in stock_codes_from_json)

print(f"\n转换后的股票代码字符串:")
print(stock_codes_str)


