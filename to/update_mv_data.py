#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新市值数据脚本
从 to/local_data/daily 读取 total_mv, circ_mv, turnover_rate
更新到 to/local_data/features/daily 对应的特征文件中
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_data_structure():
    """检查原始数据和特征文件的结构"""
    
    logger.info("🔍 检查数据结构...")
    
    # 检查原始数据结构
    daily_dir = Path("to/local_data/daily")
    if daily_dir.exists():
        sample_month = list(daily_dir.iterdir())[0]
        sample_file = sample_month / "daily_data.parquet"
        
        if sample_file.exists():
            logger.info(f"📂 检查原始数据样本: {sample_file}")
            raw_df = pd.read_parquet(sample_file)
            logger.info(f"   原始数据列: {list(raw_df.columns)}")
            logger.info(f"   原始数据记录数: {len(raw_df)}")
            
            # 检查市值字段
            mv_fields = ['total_mv', 'circ_mv', 'turnover_rate']
            for field in mv_fields:
                if field in raw_df.columns:
                    valid_count = (raw_df[field].notna() & (raw_df[field] > 0)).sum()
                    logger.info(f"   {field}: 存在, 有效数据 {valid_count}/{len(raw_df)}")
                else:
                    logger.info(f"   {field}: 不存在")
    
    # 检查特征文件结构
    features_dir = Path("to/local_data/features/daily")
    if features_dir.exists():
        sample_files = list(features_dir.glob("*.parquet"))[:1]
        if sample_files:
            sample_file = sample_files[0]
            logger.info(f"📂 检查特征文件样本: {sample_file}")
            feature_df = pd.read_parquet(sample_file)
            logger.info(f"   特征文件列: {list(feature_df.columns)}")
            logger.info(f"   特征文件记录数: {len(feature_df)}")
            
            # 检查市值字段
            mv_fields = ['total_mv', 'circ_mv', 'turnover_rate']
            for field in mv_fields:
                if field in feature_df.columns:
                    valid_count = (feature_df[field].notna() & (feature_df[field] > 0)).sum()
                    logger.info(f"   {field}: 存在, 有效数据 {valid_count}/{len(feature_df)}")
                else:
                    logger.info(f"   {field}: 不存在")

def get_date_from_filename(filename):
    """从文件名提取日期"""
    try:
        # 特征文件名格式: YYYYMMDD.parquet
        date_str = filename.stem
        if len(date_str) == 8 and date_str.isdigit():
            return date_str
    except:
        pass
    return None

def get_month_from_date(date_str):
    """从日期字符串获取月份"""
    try:
        return date_str[:6]  # YYYYMM
    except:
        return None

def update_mv_data(start_date=None, end_date=None, dry_run=True):
    """更新市值数据
    
    Args:
        start_date: 开始日期 YYYYMMDD，None表示处理所有
        end_date: 结束日期 YYYYMMDD，None表示处理所有
        dry_run: 是否为试运行模式，True时不实际保存文件
    """
    
    logger.info("🚀 开始更新市值数据...")
    logger.info(f"   模式: {'试运行' if dry_run else '正式运行'}")
    logger.info(f"   日期范围: {start_date or '全部'} - {end_date or '全部'}")
    
    # 路径设置
    daily_dir = Path("to/local_data/daily")
    features_dir = Path("to/local_data/features/daily")
    
    if not daily_dir.exists():
        logger.error(f"❌ 原始数据目录不存在: {daily_dir}")
        return False
    
    if not features_dir.exists():
        logger.error(f"❌ 特征文件目录不存在: {features_dir}")
        return False
    
    # 获取所有特征文件
    feature_files = list(features_dir.glob("*.parquet"))
    logger.info(f"📊 找到 {len(feature_files)} 个特征文件")
    
    # 过滤日期范围
    if start_date or end_date:
        filtered_files = []
        for file in feature_files:
            date_str = get_date_from_filename(file)
            if date_str:
                if start_date and date_str < start_date:
                    continue
                if end_date and date_str > end_date:
                    continue
                filtered_files.append(file)
        feature_files = filtered_files
        logger.info(f"📊 日期过滤后: {len(feature_files)} 个文件")
    
    # 按月份分组原始数据
    monthly_data_cache = {}
    
    def load_monthly_data(month_key):
        """加载月度原始数据"""
        if month_key in monthly_data_cache:
            return monthly_data_cache[month_key]
        
        month_dir = daily_dir / month_key
        month_file = month_dir / "daily_data.parquet"
        
        if not month_file.exists():
            logger.warning(f"⚠️ 月度数据文件不存在: {month_file}")
            return None
        
        try:
            df = pd.read_parquet(month_file)
            
            # 检查必需字段
            required_fields = ['stock_code', 'date', 'total_mv', 'circ_mv', 'turnover_rate']
            missing_fields = [f for f in required_fields if f not in df.columns]
            
            if missing_fields:
                logger.warning(f"⚠️ 月度数据缺少字段 {month_key}: {missing_fields}")
                return None
            
            # 转换日期格式
            if 'date_str' not in df.columns:
                df['date_str'] = df['date'].dt.strftime('%Y%m%d')
            
            monthly_data_cache[month_key] = df
            logger.info(f"📂 加载月度数据 {month_key}: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"❌ 加载月度数据失败 {month_key}: {e}")
            return None
    
    # 统计信息
    stats = {
        'processed_files': 0,
        'updated_files': 0,
        'total_records': 0,
        'updated_records': 0,
        'errors': 0
    }
    
    start_time = time.time()
    
    # 处理每个特征文件
    for i, feature_file in enumerate(feature_files, 1):
        try:
            date_str = get_date_from_filename(feature_file)
            if not date_str:
                logger.warning(f"⚠️ 无法从文件名提取日期: {feature_file.name}")
                continue
            
            month_key = get_month_from_date(date_str)
            if not month_key:
                logger.warning(f"⚠️ 无法提取月份: {date_str}")
                continue
            
            logger.info(f"📋 处理文件 {i}/{len(feature_files)}: {feature_file.name} (日期: {date_str})")
            
            # 加载特征文件
            feature_df = pd.read_parquet(feature_file)
            stats['total_records'] += len(feature_df)
            
            # 加载对应月份的原始数据
            raw_df = load_monthly_data(month_key)
            if raw_df is None:
                stats['errors'] += 1
                continue
            
            # 筛选对应日期的原始数据
            daily_raw = raw_df[raw_df['date_str'] == date_str].copy()
            
            if daily_raw.empty:
                logger.warning(f"⚠️ 原始数据中没有找到日期 {date_str} 的数据")
                stats['errors'] += 1
                continue
            
            logger.info(f"   原始数据记录: {len(daily_raw)}")
            logger.info(f"   特征数据记录: {len(feature_df)}")
            
            # 准备更新数据
            updated_count = 0
            mv_fields = ['total_mv', 'circ_mv', 'turnover_rate']
            
            # 创建原始数据的索引映射
            raw_dict = daily_raw.set_index('stock_code')[mv_fields].to_dict('index')
            
            # 🔧 预先转换DataFrame列的数据类型以避免兼容性警告
            dtype_converted = False
            for field in mv_fields:
                if field in feature_df.columns:
                    current_dtype = feature_df[field].dtype

                    if field in ['total_mv', 'circ_mv']:
                        # 市值字段：确保使用足够大的数据类型
                        if current_dtype in ['uint32', 'float32']:
                            feature_df[field] = feature_df[field].astype('float64')
                            dtype_converted = True
                            logger.debug(f"   转换 {field} 从 {current_dtype} 到 float64")
                    elif field == 'turnover_rate':
                        # 换手率字段：确保使用合适的浮点类型
                        if current_dtype == 'float16':
                            feature_df[field] = feature_df[field].astype('float32')
                            dtype_converted = True
                            logger.debug(f"   转换 {field} 从 {current_dtype} 到 float32")

            if dtype_converted:
                logger.debug(f"   已预转换数据类型以避免兼容性警告")

            # 更新特征数据
            for idx, row in feature_df.iterrows():
                stock_code = row['stock_code']

                if stock_code in raw_dict:
                    raw_data = raw_dict[stock_code]

                    # 检查是否需要更新
                    need_update = False
                    for field in mv_fields:
                        if field in feature_df.columns:
                            current_val = row[field]
                            new_val = raw_data[field]

                            # 如果原始数据有有效值，且与当前值不同
                            if pd.notna(new_val) and new_val != 0:
                                if pd.isna(current_val) or current_val == 0 or current_val != new_val:
                                    # 🔧 根据目标列的数据类型进行正确转换
                                    target_dtype = feature_df[field].dtype

                                    if target_dtype == 'uint64':
                                        # uint64类型：确保为正整数
                                        converted_val = np.uint64(max(0, int(new_val)))
                                    elif target_dtype == 'float64':
                                        converted_val = np.float64(new_val)
                                    elif target_dtype == 'float32':
                                        converted_val = np.float32(new_val)
                                    elif target_dtype == 'float16':
                                        converted_val = np.float16(new_val)
                                    else:
                                        # 其他类型直接赋值
                                        converted_val = new_val

                                    feature_df.at[idx, field] = converted_val
                                    need_update = True

                    if need_update:
                        updated_count += 1
            
            stats['updated_records'] += updated_count
            logger.info(f"   更新记录数: {updated_count}")
            
            # 保存更新后的文件
            if updated_count > 0:
                if not dry_run:
                    feature_df.to_parquet(feature_file, index=False)
                    logger.info(f"   ✅ 已保存更新")
                else:
                    logger.info(f"   🔍 试运行模式，未实际保存")
                
                stats['updated_files'] += 1
            else:
                logger.info(f"   ℹ️ 无需更新")
            
            stats['processed_files'] += 1
            
        except Exception as e:
            logger.error(f"❌ 处理文件失败 {feature_file.name}: {e}")
            stats['errors'] += 1
            import traceback
            logger.error(traceback.format_exc())
    
    # 输出统计信息
    elapsed_time = time.time() - start_time
    logger.info(f"\n📊 更新完成统计:")
    logger.info(f"   处理文件数: {stats['processed_files']}")
    logger.info(f"   更新文件数: {stats['updated_files']}")
    logger.info(f"   总记录数: {stats['total_records']}")
    logger.info(f"   更新记录数: {stats['updated_records']}")
    logger.info(f"   错误数: {stats['errors']}")
    logger.info(f"   耗时: {elapsed_time:.2f} 秒")
    
    return stats['errors'] == 0

if __name__ == "__main__":
    # 首先检查数据结构
    check_data_structure()
    
    print("\n" + "="*60)
    print("🚀 市值数据更新脚本")
    print("="*60)
    
    # 用户选择
    print("请选择运行模式:")
    print("1. 试运行 (检查但不保存)")
    print("2. 正式运行 (实际更新文件)")
    print("3. 指定日期范围运行")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        logger.info("🔍 开始试运行...")
        update_mv_data(dry_run=True)
    elif choice == "2":
        confirm = input("⚠️ 确认要正式更新文件吗? (y/N): ").strip().lower()
        if confirm == 'y':
            logger.info("🚀 开始正式运行...")
            update_mv_data(dry_run=False)
        else:
            logger.info("❌ 用户取消操作")
    elif choice == "3":
        start_date = input("请输入开始日期 (YYYYMMDD, 回车跳过): ").strip()
        end_date = input("请输入结束日期 (YYYYMMDD, 回车跳过): ").strip()
        
        start_date = start_date if start_date else None
        end_date = end_date if end_date else None
        
        dry_run = input("是否试运行? (Y/n): ").strip().lower() != 'n'
        
        logger.info(f"🚀 开始{'试运行' if dry_run else '正式运行'}...")
        update_mv_data(start_date=start_date, end_date=end_date, dry_run=dry_run)
    else:
        logger.info("❌ 无效选择")
