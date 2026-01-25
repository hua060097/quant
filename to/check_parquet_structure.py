#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查Parquet文件结构
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def check_parquet_structure():
    """检查parquet文件的结构"""
    
    # 检查一个具体的parquet文件
    parquet_file = Path("local_data/daily/202412/daily_data.parquet")
    
    if not parquet_file.exists():
        print(f"文件不存在: {parquet_file}")
        return
    
    try:
        # 使用pyarrow读取schema
        print("=== 使用PyArrow检查Schema ===")
        table = pq.read_table(parquet_file)
        print(f"Schema: {table.schema}")
        print(f"列名: {table.column_names}")
        print(f"数据行数: {len(table)}")
        
        # 转换为pandas并检查
        print("\n=== 使用Pandas检查数据 ===")
        df = table.to_pandas()
        print(f"DataFrame形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"数据类型:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # 检查前几行数据
        print(f"\n=== 前3行数据 ===")
        print(df.head(3))
        
        # 检查是否有date_str列
        if 'date_str' in df.columns:
            print(f"\n=== date_str列信息 ===")
            print(f"date_str唯一值数量: {df['date_str'].nunique()}")
            print(f"date_str样例值: {df['date_str'].head(5).tolist()}")
        else:
            print(f"\n=== 没有date_str列 ===")
            # 检查是否有其他日期相关列
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            print(f"包含'date'的列: {date_cols}")
            
        # 检查是否有stock_code列
        if 'stock_code' in df.columns:
            print(f"\n=== stock_code列信息 ===")
            print(f"stock_code唯一值数量: {df['stock_code'].nunique()}")
            print(f"stock_code样例值: {df['stock_code'].head(5).tolist()}")
        else:
            print(f"\n=== 没有stock_code列 ===")
            # 检查是否有其他股票相关列
            stock_cols = [col for col in df.columns if 'stock' in col.lower() or 'code' in col.lower()]
            print(f"包含'stock'或'code'的列: {stock_cols}")
            
    except Exception as e:
        print(f"检查文件时出错: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    check_parquet_structure()
