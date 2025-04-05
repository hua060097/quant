#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
概念板块分析工具

功能：
1. 获取所有tng概念板块
2. 计算每个概念板块包含的股票数量
3. 将结果保存到CSV和JSON文件中
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
import xtquant.xtdata as xtdata
from stock_sectors_mapping import get_tng_sectors, get_sector_stocks

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/analyze_tng_sectors.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("analyze_tng_sectors")

def connect_xtdata():
    """连接xtdata服务"""
    try:
        # 尝试连接xtdata服务
        xtdata.connect()
        logger.info("成功连接xtdata服务")
        return True
    except Exception as e:
        logger.error(f"连接xtdata服务失败: {e}")
        return False

def analyze_tng_sectors():
    """分析tng概念板块信息"""
    try:
        # 连接xtdata服务
        if not connect_xtdata():
            logger.error("无法连接xtdata服务，终止分析")
            return False
        
        # 获取所有概念板块
        all_sectors = get_tng_sectors()
        if not all_sectors:
            logger.error("获取概念板块列表失败")
            return False
        
        logger.info(f"成功获取 {len(all_sectors)} 个概念板块")
        
        # 收集每个板块的信息
        sector_stats = []
        sector_stocks = {}
        
        # 处理每个概念板块
        for i, sector in enumerate(all_sectors):
            logger.info(f"处理板块 {i+1}/{len(all_sectors)}: {sector}")
            
            # 直接获取概念板块包含的股票列表
            stocks_in_sector = get_sector_stocks(sector)
            
            if stocks_in_sector:
                # 存储板块股票列表
                sector_stocks[sector] = stocks_in_sector
                
                # 添加板块统计信息
                sector_stats.append({
                    'sector_name': sector,
                    'stock_count': len(stocks_in_sector)
                })
                
                logger.info(f"板块 '{sector}' 包含 {len(stocks_in_sector)} 只股票")
            else:
                logger.warning(f"板块 '{sector}' 没有获取到股票")
                # 添加空列表，保持数据完整性
                sector_stocks[sector] = []
                sector_stats.append({
                    'sector_name': sector,
                    'stock_count': 0
                })
        
        # 按股票数量排序
        sector_stats.sort(key=lambda x: x['stock_count'], reverse=True)
        
        # 创建结果目录
        result_dir = 'sector_analysis'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        
        # 保存为JSON
        json_path = os.path.join(result_dir, f'tng_sectors_stats.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sector_stats, f, ensure_ascii=False, indent=4)
        logger.info(f"板块统计信息已保存到JSON: {json_path}")
        
        # 保存详细的板块-股票映射
        mapping_path = os.path.join(result_dir, f'tng_sectors_mapping.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(sector_stocks, f, ensure_ascii=False, indent=4)
        logger.info(f"板块-股票映射已保存到: {mapping_path}")
        
        # 生成统计报告
        report = f"""
        概念板块分析报告 (生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        ----------------------------------------------------------------------
        概念板块总数: {len(sector_stats)}
        
        按股票数量排名前20的板块:
        {pd.DataFrame([{'板块名称': s['sector_name'], '股票数量': s['stock_count']} for s in sector_stats[:20]]).to_string(index=False)}
        
        股票数量最少的10个板块:
        {pd.DataFrame([{'板块名称': s['sector_name'], '股票数量': s['stock_count']} for s in sorted(sector_stats, key=lambda x: x['stock_count'])[:10]]).to_string(index=False)}
        
        分析完成，结果文件:
        - CSV文件: {csv_path}
        - JSON文件: {json_path}
        - 映射文件: {mapping_path}
        ----------------------------------------------------------------------
        """
        
        logger.info(report)
        
        # 保存报告
        report_path = os.path.join(result_dir, f'tng_sectors_report_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"分析报告已保存到: {report_path}")
        
        return True
    except Exception as e:
        logger.error(f"分析概念板块时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    logger.info("开始分析tng概念板块...")
    success = analyze_tng_sectors()
    if success:
        logger.info("概念板块分析完成")
    else:
        logger.error("概念板块分析失败")

if __name__ == "__main__":
    main()