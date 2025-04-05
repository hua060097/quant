#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易日历获取测试脚本

此脚本用于验证交易日历的获取是否正常工作。
包括直接API调用和通过回测模型获取两种方式。

作者: AI助手
日期: 2025-03-22
"""

import os
import sys
import logging
import traceback
from datetime import datetime, timedelta

# 配置日志
os.makedirs('to/log', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/calendar_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("calendar_test")

# 确保当前目录在路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    logger.info(f"添加当前目录到路径: {current_dir}")

def test_direct_api():
    """直接从API获取交易日历"""
    logger.info("==== 测试直接从API获取交易日历 ====")
    try:
        import xtquant.xtdata as xtdata
        logger.info("成功导入xtdata模块")
        
        # 确保连接
        xtdata.connect()
        logger.info("已连接到数据服务")
        
        # 获取交易日期
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = '20220101'
        logger.info(f"尝试获取交易日期范围: {start_date} 至 {end_date}")
        
        trading_dates = xtdata.get_trading_dates(
            market='SH', 
            start_time=start_date, 
            end_time=end_date
        )

        print(trading_dates)
        
        if trading_dates and len(trading_dates) > 0:
            logger.info(f"成功获取 {len(trading_dates)} 个交易日期")
            logger.info(f"第一个日期: {trading_dates[0]}, 类型: {type(trading_dates[0])}")
            logger.info(f"最后一个日期: {trading_dates[-1]}, 类型: {type(trading_dates[-1])}")
            
            # 如果是时间戳，转换为可读日期
            if isinstance(trading_dates[0], int):
                readable_dates = [datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d') for ts in trading_dates[:5]]
                logger.info(f"前5个可读日期: {readable_dates}")
            
            return trading_dates
        else:
            logger.error("API返回空的交易日期列表")
            return None
    except Exception as e:
        logger.error(f"直接API获取交易日期失败: {e}")
        logger.error(traceback.format_exc())
        return None

def test_model_calendar():
    """通过回测模型获取交易日历"""
    logger.info("==== 测试通过回测模型获取交易日历 ====")
    try:
        # 先检查缓存集成是否可用
        try:
            from cache_integration import integrate_cache_system
            logger.info("尝试初始化缓存系统...")
            
            cache_integration = integrate_cache_system()
            if cache_integration and cache_integration.patched:
                logger.info("缓存系统已成功集成")
            else:
                logger.warning("缓存系统集成失败或未完成")
        except Exception as e:
            logger.warning(f"初始化缓存系统时出错: {e}")
            logger.error(traceback.format_exc())
        
        # 导入模型
        from advanced_quant_model import AdvancedQuantModel
        logger.info("成功导入AdvancedQuantModel模块")
        
        # 创建模型实例
        model = AdvancedQuantModel()
        logger.info("已创建模型实例")
        
        # 加载交易日期
        logger.info("尝试加载交易日期...")
        model.load_trading_dates()
        
        if model.trading_dates and len(model.trading_dates) > 0:
            logger.info(f"成功通过模型加载 {len(model.trading_dates)} 个交易日期")
            logger.info(f"第一个日期: {model.trading_dates[0]}, 类型: {type(model.trading_dates[0])}")
            logger.info(f"最后一个日期: {model.trading_dates[-1]}, 类型: {type(model.trading_dates[-1])}")
            
            # 显示前5个日期
            logger.info(f"前5个日期: {model.trading_dates[:5]}")
            
            return model.trading_dates
        else:
            logger.error("模型返回空的交易日期列表")
            return None
    except Exception as e:
        logger.error(f"通过模型获取交易日期失败: {e}")
        logger.error(traceback.format_exc())
        return None

def compare_results(api_dates, model_dates):
    """比较两种方式获取的结果"""
    logger.info("==== 比较两种获取方式的结果 ====")
    
    if not api_dates or not model_dates:
        logger.error("无法比较结果，至少有一种方式获取失败")
        return
    
    # 处理API返回的时间戳格式
    if isinstance(api_dates[0], int):
        api_dates_str = [datetime.fromtimestamp(ts/1000).strftime('%Y%m%d') for ts in api_dates]
    else:
        api_dates_str = api_dates
    
    # 计算交集和差集
    api_set = set(api_dates_str)
    model_set = set(model_dates)
    
    common = api_set.intersection(model_set)
    api_only = api_set - model_set
    model_only = model_set - api_set
    
    logger.info(f"API获取的日期数量: {len(api_dates_str)}")
    logger.info(f"模型获取的日期数量: {len(model_dates)}")
    logger.info(f"两者共有日期数量: {len(common)}")
    
    if api_only:
        logger.warning(f"仅在API结果中存在的日期数量: {len(api_only)}")
        logger.warning(f"示例: {list(api_only)[:5]}")
    
    if model_only:
        logger.warning(f"仅在模型结果中存在的日期数量: {len(model_only)}")
        logger.warning(f"示例: {list(model_only)[:5]}")
    
    if len(common) == len(api_set) and len(common) == len(model_set):
        logger.info("两种方式获取的结果完全一致")
    else:
        logger.warning("两种方式获取的结果存在差异")

def main():
    """主函数"""
    try:
        logger.info("===== 开始测试交易日历获取 =====")
        
        # 测试直接API获取
        api_dates = test_direct_api()
        
        # 测试模型获取
        model_dates = test_model_calendar()
        
        # 比较结果
        if api_dates and model_dates:
            compare_results(api_dates, model_dates)
        
        logger.info("===== 测试完成 =====")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 