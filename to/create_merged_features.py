#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建合并特征文件

将一段时间内的所有板块特征合并为一个文件，方便模型训练
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
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))

# 导入路径工具
try:
    from to.path_utils import get_data_path, ensure_dir
except ImportError:
    # 如果path_utils.py不存在，则提供兼容性处理
    def get_data_path(relative_path=None):
        """返回相对于数据目录的绝对路径"""
        project_root = Path(__file__).resolve().parent.parent
        data_root = project_root / "to" / "local_data"
        if relative_path:
            return data_root / relative_path
        return data_root
    
    def ensure_dir(path):
        """确保目录存在，如果不存在则创建"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_merged_features(start_date, end_date, feature_dir=None):
    """创建合并特征文件
    
    Args:
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD
        feature_dir: 特征数据目录，如果为None则使用默认值
        
    Returns:
        合并文件的路径
    """
    # 设置特征目录
    if feature_dir is None:
        feature_dir = get_data_path('sector_features')
    
    # 确保特征目录存在
    feature_dir = ensure_dir(feature_dir)
    
    # 导入特征生成器
    try:
        from to.sector_features.feature_generator import SectorFeatureGenerator
    except ImportError:
        try:
            from sector_features.feature_generator import SectorFeatureGenerator
        except ImportError:
            logger.error("无法导入特征生成器模块")
            return None
    
    logger.info(f"为 {start_date} 到 {end_date} 的特征数据创建合并文件")
    logger.info(f"使用特征目录: {feature_dir}")
    
    # 创建空的特征生成器实例（仅用于读取和保存数据）
    feature_generator = SectorFeatureGenerator(
        data_provider=None, 
        sector_mapping={},
        output_dir=feature_dir
    )
    
    # 创建合并文件
    merged_file_path = feature_generator.save_merged_features_for_period(start_date, end_date)
    
    if merged_file_path and os.path.exists(merged_file_path):
        # 读取合并文件来验证
        merged_df = pd.read_parquet(merged_file_path)
        
        # 显示统计信息
        logger.info(f"已创建合并特征文件: {merged_file_path}")
        logger.info(f"特征形状: {merged_df.shape}")
        logger.info(f"包含的日期: {merged_df['date'].nunique()} 个")
        logger.info(f"包含的板块: {merged_df['sector_name'].nunique()} 个")
        
        return merged_file_path
    else:
        logger.error("创建合并特征文件失败")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="创建合并特征文件")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期，格式为YYYYMMDD")
    parser.add_argument("--end_date", type=str, required=True, help="结束日期，格式为YYYYMMDD")
    parser.add_argument("--feature_dir", type=str, default=None, help="特征数据目录")
    
    args = parser.parse_args()
    
    # 创建合并文件
    merged_file_path = create_merged_features(args.start_date, args.end_date, args.feature_dir)
    
    if merged_file_path:
        print(f"\n特征文件已成功合并并保存到: {merged_file_path}")
        print("此文件可以直接用于模型训练。")
    else:
        print("合并特征文件失败，请检查日志获取详细信息。")

if __name__ == "__main__":
    main() 