#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试特征数据加载

测试从保存的特征文件中加载数据，验证数据结构是否适合模型训练
"""
import os
import sys
import logging
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到系统路径
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent
sys.path.append(str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_feature_loading(start_date, end_date, feature_dir="to/local_data/sector_features"):
    """测试从特征文件加载数据
    
    Args:
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD
        feature_dir: 特征数据目录
    """
    # 导入特征生成器
    try:
        from to.sector_features.feature_generator import SectorFeatureGenerator
    except ImportError:
        try:
            from sector_features.feature_generator import SectorFeatureGenerator
        except ImportError:
            logger.error("无法导入特征生成器模块")
            return
    
    logger.info(f"测试从 {start_date} 到 {end_date} 的特征数据加载")
    
    # 创建空的特征生成器实例（仅用于读取数据）
    feature_generator = SectorFeatureGenerator(
        data_provider=None, 
        sector_mapping={},
        output_dir=feature_dir
    )
    
    # 加载特征数据
    features_df = feature_generator.get_all_features_for_date_range(start_date, end_date)
    
    if features_df.empty:
        logger.error(f"未找到 {start_date} 到 {end_date} 的特征数据")
        return
    
    # 打印数据统计信息
    logger.info(f"成功加载特征数据，形状: {features_df.shape}")
    logger.info(f"包含的日期: {features_df['date'].nunique()} 个")
    logger.info(f"包含的板块: {features_df['sector_name'].nunique()} 个")
    
    # 打印部分数据样例
    logger.info("数据样例:")
    print(features_df.head())
    
    # 打印特征列
    logger.info(f"特征列 ({len(features_df.columns)} 个):")
    print(features_df.columns.tolist())
    
    # 如果需要，检查特征是否有缺失值
    null_counts = features_df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning("发现缺失值:")
        for col, count in null_counts[null_counts > 0].items():
            logger.warning(f"  {col}: {count} 个缺失值")
    else:
        logger.info("所有特征都没有缺失值，数据完整")
    
    # 检查数据类型
    logger.info("数据类型:")
    for col, dtype in features_df.dtypes.items():
        logger.info(f"  {col}: {dtype}")
    
    # 检查可用于模型训练的示例数量
    logger.info(f"可用于模型训练的样本数量: {len(features_df)}")
    
    # 返回加载的数据
    return features_df

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试特征数据加载")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期，格式为YYYYMMDD")
    parser.add_argument("--end_date", type=str, required=True, help="结束日期，格式为YYYYMMDD")
    parser.add_argument("--feature_dir", type=str, default="to/local_data/sector_features", help="特征数据目录")
    
    args = parser.parse_args()
    
    # 运行测试
    test_feature_loading(args.start_date, args.end_date, args.feature_dir)

if __name__ == "__main__":
    main() 