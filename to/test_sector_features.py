#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块特征生成测试脚本

用于测试增强版特征生成器的功能
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

# 导入路径工具
try:
    from to.path_utils import get_data_path, ensure_dir, normalize_path
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
        
    def normalize_path(path_str):
        """标准化路径字符串"""
        return path_str

def test_generate_features(sector="TGN人工智能", date=None, output_dir=None, save_csv=True):
    """测试板块特征生成功能
    
    Args:
        sector: 要处理的板块，默认为TGN人工智能
        date: 要处理的日期，格式为YYYYMMDD，默认为最近一个交易日
        output_dir: 输出目录，默认为数据目录下的sector_features
        save_csv: 是否同时保存CSV格式，默认为True以方便查看结果
    """
    # 导入数据加载器
    try:
        import xtquant.xtdata as xtdata
    except ImportError:
        logger.error("无法导入xtquant，请确保已安装xtquant包")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_dir = get_data_path('sector_features')
    
    # 确保输出目录存在
    output_dir = ensure_dir(output_dir)
    
    logger.info(f"使用输出目录: {output_dir}")
    
    # 导入必要的模块
    try:
        from sector_features.data_provider import XtDataProvider
        
        # 可能的增强版特征生成器导入路径
        try:
            # 直接导入
            from sector_features.enhanced_feature_generator import EnhancedSectorFeatureGenerator
            logger.info("成功导入增强版特征生成器")
        except ImportError:
            try:
                # 从to包导入
                from to.sector_features.enhanced_feature_generator import EnhancedSectorFeatureGenerator
                logger.info("从to包导入增强版特征生成器")
            except ImportError:
                # 显示导入路径，以便调试
                logger.error(f"导入增强版特征生成器失败，sys.path: {sys.path}")
                # 导入基本版特征生成器
                try:
                    from sector_features.feature_generator import SectorFeatureGenerator as EnhancedSectorFeatureGenerator
                    logger.warning("使用基本版特征生成器代替增强版")
                except ImportError:
                    logger.error("无法导入任何特征生成器")
                    return
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        return
    
    # 连接数据源
    try:
        if not xtdata.get_client().is_connected():
            xtdata.connect()
            logger.info("成功连接到数据源")
    except Exception as e:
        logger.error(f"连接数据源失败: {e}")
        return
    
    # 如果未指定日期，则使用最近一个交易日
    if date is None:
        today = datetime.now().strftime("%Y%m%d")
        # 获取最近10天的交易日历，取最新的一天
        try:
            trading_dates = xtdata.get_trading_dates("SH", "20230101", today)
            if trading_dates:
                date = datetime.fromtimestamp(trading_dates[-1]/1000).strftime("%Y%m%d")
                logger.info(f"使用最近交易日: {date}")
            else:
                logger.error("获取交易日历失败")
                return
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return
    
    # 获取板块股票列表
    try:
        stocks = xtdata.get_stock_list_in_sector(sector)
        if not stocks:
            logger.error(f"板块 {sector} 不存在或没有成分股")
            return
        logger.info(f"成功获取板块 {sector} 的 {len(stocks)} 只成分股")
    except Exception as e:
        logger.error(f"获取板块成分股失败: {e}")
        return
    
    # 创建数据提供器和特征生成器
    data_provider = XtDataProvider()
    sector_mapping = {sector: stocks}
    
    # 创建增强特征生成器
    feature_generator = EnhancedSectorFeatureGenerator(
        data_provider=data_provider,
        sector_mapping=sector_mapping,
        output_dir=output_dir,
        weighting_method="equal"
    )
    
    # 生成特征
    logger.info(f"开始生成板块 {sector} 在 {date} 的特征")
    features = feature_generator.generate_sector_features_for_date(sector, date)
    
    # 输出结果
    if features:
        # 打印特征
        logger.info(f"成功生成板块 {sector} 在 {date} 的特征:")
        for key, value in features.items():
            logger.info(f"{key}: {value}")
        
        # 保存特征，根据参数决定是否同时保存CSV
        feature_generator.save_features(features, sector, date, save_csv=save_csv)
        
        # 指示保存的文件位置
        parquet_path = os.path.join(output_dir, sector, f"features_{date}.parquet")
        logger.info(f"特征已保存到: {parquet_path}")
        
        if save_csv:
            csv_path = os.path.join(output_dir, sector, f"features_{date}.csv")
            logger.info(f"CSV格式特征已保存到: {csv_path}")
        
        # 将特征转换为DataFrame并打印
        features_df = pd.DataFrame([features])
        pd.set_option('display.max_columns', None)
        print(f"\n板块 {sector} 在 {date} 的特征表格:")
        print(features_df)
    else:
        logger.error(f"未能生成板块 {sector} 在 {date} 的特征")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试板块特征生成")
    parser.add_argument("--sector", type=str, default="TGN人工智能", help="要处理的板块名称")
    parser.add_argument("--date", type=str, default=None, help="要处理的日期，格式为YYYYMMDD，默认为最近交易日")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--no_csv", action="store_true", help="不保存CSV格式(默认保存CSV便于查看)")
    
    args = parser.parse_args()
    
    # 运行测试
    test_generate_features(args.sector, args.date, args.output_dir, not args.no_csv)

if __name__ == "__main__":
    main() 