#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征数据压缩模块

提供特征数据的高效压缩和存储功能，优化内存和磁盘使用。
支持多种存储格式和压缩方法。
"""

import os
import logging
import json
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

# 获取logger
logger = logging.getLogger(__name__)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """优化DataFrame的数据类型，减少内存占用
    
    对不同类型的列应用最合适的数据类型，降低内存占用。
    
    Args:
        df: 输入DataFrame
        
    Returns:
        优化后的DataFrame
    """
    result = df.copy()
    
    if result.empty:
        return result
    
    # 处理数值型列
    for col in result.select_dtypes(include=['int64', 'float64']).columns:
        # 检查是否有缺失值
        has_nan = result[col].isna().any()
        col_min = result[col].min()
        col_max = result[col].max()
        
        # 对于整型
        if not has_nan and pd.api.types.is_integer_dtype(result[col]):
            if col_min >= 0:
                if col_max < 2**8:
                    result[col] = result[col].astype(np.uint8)
                elif col_max < 2**16:
                    result[col] = result[col].astype(np.uint16)
                elif col_max < 2**32:
                    result[col] = result[col].astype(np.uint32)
            else:
                if col_min > -2**7 and col_max < 2**7:
                    result[col] = result[col].astype(np.int8)
                elif col_min > -2**15 and col_max < 2**15:
                    result[col] = result[col].astype(np.int16)
                elif col_min > -2**31 and col_max < 2**31:
                    result[col] = result[col].astype(np.int32)
        # 对于浮点型
        elif pd.api.types.is_float_dtype(result[col]):
            if not has_nan and col_min >= 0 and col_max < 65535:
                # 如果值都是整数且在合适的范围内
                if np.all(result[col] == result[col].astype(int)):
                    result[col] = result[col].astype(np.uint16)
                else:
                    result[col] = result[col].astype(np.float32)
            else:
                result[col] = result[col].astype(np.float32)
    
    # 处理时间类型列，尤其是'date'列
    for col in ['date', 'datetime', 'time']:
        if col in result.columns and pd.api.types.is_string_dtype(result[col]):
            try:
                # 尝试转换为日期
                result[col] = pd.to_datetime(result[col])
            except:
                # 如果转换失败，保留原样
                pass
    
    # 处理字符串列
    for col in result.select_dtypes(include=['object']).columns:
        # 如果字符串列的唯一值数量小于总行数的一半，考虑使用类别类型
        if result[col].nunique() < len(result) * 0.5:
            result[col] = result[col].astype('category')
    
    return result


def save_features(
    df: pd.DataFrame, 
    file_path: Union[str, Path], 
    format: str = 'parquet',
    compress: bool = True,
    optimize: bool = True,
    **kwargs
) -> bool:
    """保存特征数据到文件
    
    支持多种格式和压缩选项，优化存储效率。
    
    Args:
        df: 特征DataFrame
        file_path: 输出文件路径
        format: 输出格式，支持'parquet', 'csv', 'pickle'
        compress: 是否压缩数据
        optimize: 是否优化数据类型
        **kwargs: 传递给底层保存函数的额外参数
        
    Returns:
        是否成功保存
    """
    if df.empty:
        logger.warning(f"DataFrame为空，不保存: {file_path}")
        return False
    
    # 创建目录
    path_obj = Path(file_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # 优化数据类型
    if optimize:
        df = optimize_dtypes(df)
    
    try:
        # 根据格式选择保存方法
        if format.lower() == 'parquet':
            compression = 'snappy' if compress else None
            df.to_parquet(path_obj, compression=compression, **kwargs)
        elif format.lower() == 'csv':
            compression = 'gzip' if compress else None
            df.to_csv(path_obj, compression=compression, index=False, **kwargs)
        elif format.lower() == 'pickle':
            compression = 'gzip' if compress else None
            df.to_pickle(path_obj, compression=compression, **kwargs)
        else:
            logger.warning(f"不支持的存储格式: {format}")
            return False
        
        logger.debug(f"成功保存特征数据: {path_obj}, 格式: {format}, 压缩: {compress}")
        return True
    except Exception as e:
        logger.error(f"保存特征数据失败: {path_obj}, 错误: {e}")
        return False


def load_features(
    file_path: Union[str, Path], 
    **kwargs
) -> pd.DataFrame:
    """加载特征数据
    
    自动检测文件格式并加载。
    
    Args:
        file_path: 文件路径
        **kwargs: 传递给底层加载函数的额外参数
        
    Returns:
        加载的DataFrame
    """
    path_obj = Path(file_path)
    
    if not path_obj.exists():
        logger.warning(f"文件不存在: {path_obj}")
        return pd.DataFrame()
    
    try:
        # 根据文件扩展名选择加载方法
        suffix = path_obj.suffix.lower()
        
        if suffix == '.parquet':
            return pd.read_parquet(path_obj, **kwargs)
        elif suffix == '.csv':
            return pd.read_csv(path_obj, **kwargs)
        elif suffix in ['.pkl', '.pickle']:
            return pd.read_pickle(path_obj, **kwargs)
        else:
            logger.warning(f"不支持的文件格式: {suffix}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"加载特征数据失败: {path_obj}, 错误: {e}")
        return pd.DataFrame()


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """获取文件信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件信息字典
    """
    path_obj = Path(file_path)
    
    if not path_obj.exists():
        return {
            'exists': False,
            'size': 0,
            'modified': None,
            'format': None
        }
    
    suffix = path_obj.suffix.lower()
    format_map = {
        '.parquet': 'parquet',
        '.csv': 'csv',
        '.pkl': 'pickle',
        '.pickle': 'pickle'
    }
    
    # 尝试加载数据获取行数和列数
    shape = (0, 0)
    try:
        df = load_features(path_obj)
        shape = df.shape
    except:
        pass
    
    return {
        'exists': True,
        'size': path_obj.stat().st_size,
        'modified': path_obj.stat().st_mtime,
        'format': format_map.get(suffix, 'unknown'),
        'shape': shape
    }


def compress_feature_files(
    directory: Union[str, Path],
    pattern: str = "*.csv",
    output_format: str = 'parquet'
) -> int:
    """压缩目录中的特征文件
    
    将目录中的特征文件转换为更高效的格式，并可选删除原始文件。
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        output_format: 输出格式
        
    Returns:
        成功压缩的文件数量
    """
    dir_path = Path(directory)
    
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning(f"目录不存在: {dir_path}")
        return 0
    
    # 查找匹配的文件
    files = list(dir_path.glob(pattern))
    
    if not files:
        logger.info(f"目录 {dir_path} 中没有找到匹配 {pattern} 的文件")
        return 0
    
    success_count = 0
    
    for file_path in files:
        try:
            # 加载数据
            df = load_features(file_path)
            
            if df.empty:
                logger.warning(f"文件 {file_path} 的数据为空")
                continue
            
            # 设置输出路径
            output_path = file_path.with_suffix(f".{output_format}")
            
            # 保存为新格式
            success = save_features(df, output_path, format=output_format, compress=True, optimize=True)
            
            if success:
                success_count += 1
                logger.info(f"成功压缩文件: {file_path} → {output_path}")
                
                # 删除原始文件
                if output_path != file_path:
                    file_path.unlink()
                    logger.debug(f"删除原始文件: {file_path}")
        except Exception as e:
            logger.error(f"压缩文件 {file_path} 失败: {e}")
    
    return success_count


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.DEBUG)
    
    # 测试特征压缩功能
    test_dir = Path("test_features")
    test_dir.mkdir(exist_ok=True)
    
    # 创建测试数据
    test_df = pd.DataFrame({
        'date': pd.date_range('20230101', periods=100),
        'int_col': np.random.randint(0, 100, 100),
        'float_col': np.random.randn(100),
        'category_col': np.random.choice(['A', 'B', 'C'], 100),
        'big_float': np.random.randn(100) * 1000,
        'small_int': np.random.randint(0, 10, 100)
    })
    
    # 保存为不同格式
    csv_path = test_dir / "test_features.csv"
    parquet_path = test_dir / "test_features.parquet"
    
    # 测试保存和加载
    save_features(test_df, csv_path, format='csv')
    save_features(test_df, parquet_path, format='parquet')
    
    # 加载并验证
    loaded_csv = load_features(csv_path)
    loaded_parquet = load_features(parquet_path)
    
    # 对比大小
    csv_info = get_file_info(csv_path)
    parquet_info = get_file_info(parquet_path)
    
    print(f"CSV文件大小: {csv_info['size'] / 1024:.2f} KB")
    print(f"Parquet文件大小: {parquet_info['size'] / 1024:.2f} KB")
    print(f"压缩比: {csv_info['size'] / parquet_info['size']:.2f}x")
    
    # 测试目录压缩
    compress_feature_files(test_dir, pattern="*.csv", output_format='parquet') 