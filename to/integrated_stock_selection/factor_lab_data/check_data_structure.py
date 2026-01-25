"""
检查daily_data.parquet的数据结构
"""
import pandas as pd
from pathlib import Path

def check_data_structure():
    """检查数据文件的结构"""

    data_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\daily\202305\daily_data.parquet")

    print("=" * 100)
    print(f"检查数据文件结构: {data_path}")
    print("=" * 100)

    # 读取数据
    df = pd.read_parquet(data_path)

    print(f"\n[基本信息]")
    print(f"  数据形状: {df.shape}")
    print(f"  列名: {list(df.columns)}")
    print(f"  索引名: {df.index.name}")
    print(f"  索引类型: {type(df.index)}")

    print(f"\n[索引示例（前10个）]")
    print(df.index[:10])

    print(f"\n[列数据类型]")
    print(df.dtypes)

    # 检查是否有股票代码相关的列
    print(f"\n[查找股票代码相关列]")
    for col in df.columns:
        if 'code' in col.lower() or 'stock' in col.lower():
            print(f"  列 '{col}' 的唯一值数量: {df[col].nunique()}")
            print(f"  示例值: {df[col].head(5).tolist()}")

    # 检查索引中是否包含股票代码
    print(f"\n[检查索引]")
    print(f"  索引唯一值数量: {df.index.nunique()}")
    print(f"  索引示例: {df.index[:10].tolist()}")

    # 尝试查找 688469.SH
    print(f"\n[查找股票 688469.SH]")
    if 'stock_code' in df.columns:
        if '688469.SH' in df['stock_code'].values:
            print(f"  ✓ 在 stock_code 列中找到")
            stock_rows = df[df['stock_code'] == '688469.SH']
            print(f"  行数: {len(stock_rows)}")
            if 'date' in stock_rows.columns:
                print(f"  日期范围: {stock_rows['date'].min()} 至 {stock_rows['date'].max()}")
        else:
            print(f"  ✗ 不在 stock_code 列中")
            # 搜索类似的代码
            similar = df[df['stock_code'].str.contains('688469', na=False)]
            if len(similar) > 0:
                print(f"  找到类似的代码: {similar['stock_code'].unique()}")

    # 如果索引是MultiIndex，检查
    if isinstance(df.index, pd.MultiIndex):
        print(f"\n[MultiIndex信息]")
        print(f"  层级数量: {df.index.nlevels}")
        print(f"  层级名称: {df.index.names}")
        for i, name in enumerate(df.index.names):
            print(f"    层级 {i} ({name}): {df.index.get_level_values(i).nunique()} 个唯一值")

if __name__ == "__main__":
    check_data_structure()
