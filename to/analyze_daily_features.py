#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析daily特征文件结构的脚本
分析to/local_data/features/daily目录下的parquet文件内容和结构
"""

import os
import pandas as pd
import logging
from pathlib import Path
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_parquet_file(file_path):
    """
    分析一个parquet文件的内容和结构
    
    Args:
        file_path: parquet文件路径
        
    Returns:
        None, 直接打印分析结果
    """
    logger.info(f"分析文件: {file_path}")
    
    # 读取parquet文件
    try:
        df = pd.read_parquet(file_path)
        
        # 基本信息
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"内存使用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # 列信息
        logger.info("列名和数据类型:")
        for col, dtype in df.dtypes.items():
            logger.info(f"  {col}: {dtype}")
        
        # 唯一值信息
        logger.info("唯一值统计:")
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                logger.info(f"  {col}: {df[col].nunique()} 个唯一值")
        
        # 样本数据
        logger.info("样本数据 (前5行):")
        logger.info(df.head().to_string())
        
        # 如果有stock_code列，分析股票数量
        if 'stock_code' in df.columns:
            stock_count = df['stock_code'].nunique()
            logger.info(f"股票数量: {stock_count}")
            logger.info(f"股票代码示例: {df['stock_code'].unique()[:5].tolist()}")
        
        # 如果有date列，分析日期范围
        if 'date' in df.columns:
            date_col = 'date'
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                date_min = df[date_col].min().strftime('%Y-%m-%d')
                date_max = df[date_col].max().strftime('%Y-%m-%d')
            else:
                date_min = str(df[date_col].min())
                date_max = str(df[date_col].max())
            logger.info(f"日期范围: {date_min} 到 {date_max}")
        
        return df
    
    except Exception as e:
        logger.error(f"分析文件失败: {e}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分析daily特征文件结构")
    parser.add_argument("--date", type=str, help="要分析的日期 (YYYYMMDD 格式)")
    parser.add_argument("--feature_dir", type=str, default="to/local_data/features/daily",
                        help="特征目录路径")
    
    args = parser.parse_args()
    
    # 构建特征目录路径
    feature_dir = Path(args.feature_dir)
    
    # 检查目录是否存在
    if not feature_dir.exists():
        logger.error(f"特征目录不存在: {feature_dir}")
        return
    
    # 如果提供了日期，分析该日期的文件
    if args.date:
        file_path = feature_dir / f"{args.date}.parquet"
        if file_path.exists():
            analyze_parquet_file(file_path)
        else:
            logger.error(f"日期 {args.date} 的特征文件不存在")
            
            # 尝试列出目录中的所有parquet文件
            parquet_files = list(feature_dir.glob("*.parquet"))
            if parquet_files:
                logger.info(f"目录中有 {len(parquet_files)} 个parquet文件:")
                for file in parquet_files[:5]:  # 只显示前5个
                    logger.info(f"  {file.name}")
                if len(parquet_files) > 5:
                    logger.info(f"  ... 等共 {len(parquet_files)} 个文件")
            else:
                logger.info("目录中没有找到parquet文件")
    else:
        # 没有提供日期，尝试分析目录中的第一个parquet文件
        parquet_files = list(feature_dir.glob("*.parquet"))
        if parquet_files:
            logger.info(f"目录中有 {len(parquet_files)} 个parquet文件，分析第一个:")
            analyze_parquet_file(parquet_files[0])
        else:
            logger.error("目录中没有找到parquet文件")

if __name__ == "__main__":
    main()