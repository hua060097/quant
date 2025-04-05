#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成样本数据脚本

为测试创建板块特征样本数据
"""

import os
import csv
from pathlib import Path

# 确保目录存在
def ensure_dir(path):
    path_obj = Path(path)
    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {path_obj}")
    return path_obj

# 创建板块特征数据
def create_sector_features():
    sector_name = "TGN人工智能"
    directory = f"to/local_data/sector_features/{sector_name}"
    ensure_dir(directory)
    
    file_path = f"{directory}/{sector_name}_features.csv"
    
    # 样本数据
    data = [
        ["date", "open_avg", "high_avg", "low_avg", "close_avg", "volume_avg", "amount_avg", "change_avg", "active_count", "ma5_avg", "ma10_avg", "ma20_avg", "ma60_avg", "price_above_ma5_ratio", "price_above_ma20_ratio", "total_amount", "sector_name", "stock_count"],
        ["20230103", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "TGN人工智能", "6"],
        ["20230104", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "TGN人工智能", "6"],
        ["20230105", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "TGN人工智能", "6"],
        ["20230106", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "TGN人工智能", "6"],
        ["20230109", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "TGN人工智能", "6"],
        ["20230110", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "TGN人工智能", "6"],
        ["20230111", "12.5", "12.8", "12.2", "12.6", "345000", "4347000", "0.02", "4", "12.5", "12.3", "12.1", "11.8", "0.67", "0.83", "26082000", "TGN人工智能", "6"],
        ["20230112", "12.6", "12.9", "12.4", "12.8", "378000", "4838400", "0.016", "5", "12.55", "12.35", "12.15", "11.85", "0.7", "0.85", "29030400", "TGN人工智能", "6"],
        ["20230113", "12.8", "13.2", "12.7", "13.1", "425000", "5567500", "0.023", "6", "12.62", "12.4", "12.2", "11.9", "0.83", "0.9", "33405000", "TGN人工智能", "6"],
        ["20230116", "13.1", "13.4", "12.9", "13.2", "482000", "6362400", "0.008", "5", "12.74", "12.45", "12.25", "11.95", "0.85", "0.92", "38174400", "TGN人工智能", "6"],
        ["20230117", "13.2", "13.5", "13.0", "13.3", "510000", "6783000", "0.008", "6", "12.9", "12.5", "12.3", "12.0", "0.87", "0.95", "40698000", "TGN人工智能", "6"],
        ["20230118", "13.3", "13.6", "13.1", "13.4", "535000", "7169000", "0.008", "6", "13.02", "12.55", "12.35", "12.05", "0.9", "0.97", "43014000", "TGN人工智能", "6"],
        ["20230119", "13.4", "13.7", "13.2", "13.5", "520000", "7020000", "0.007", "5", "13.1", "12.6", "12.4", "12.1", "0.92", "0.98", "42120000", "TGN人工智能", "6"],
        ["20230120", "13.5", "13.8", "13.3", "13.7", "550000", "7535000", "0.015", "6", "13.22", "12.65", "12.45", "12.15", "0.95", "1.0", "45210000", "TGN人工智能", "6"]
    ]
    
    # 写入CSV文件
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    
    print(f"创建了板块特征样本数据: {file_path}")

# 创建板块映射数据
def create_sector_mapping():
    mapping_dir = "to/local_data/sector_mapping"
    ensure_dir(mapping_dir)
    
    import json
    from datetime import datetime
    
    mapping = {
        "TGN人工智能": [
            "000001.SZ", "000002.SZ", "600000.SH", 
            "600001.SH", "600519.SH", "002415.SZ"
        ],
        "TGN新能源": [
            "300750.SZ", "300014.SZ", "002594.SZ",
            "601012.SH", "002460.SZ", "300274.SZ"
        ],
        "TGN半导体": [
            "688981.SH", "603501.SH", "600584.SH",
            "002049.SZ", "688012.SH", "603986.SH"
        ]
    }
    
    # 使用当前日期作为文件名
    today = datetime.now().strftime('%Y%m%d')
    file_path = f"{mapping_dir}/mapping_{today}.json"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"创建了板块映射样本数据: {file_path}")

if __name__ == "__main__":
    # 创建板块特征数据
    create_sector_features()
    
    # 创建板块映射数据
    create_sector_mapping() 