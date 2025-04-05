#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试交易日期获取问题
对比直接调用API和通过缓存系统调用的不同
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_cache_issue")

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 直接导入xtdata
try:
    from xtquant import xtdata
    logger.info("成功导入xtdata模块")
except ImportError as e:
    logger.error(f"无法导入xtdata模块: {e}")
    sys.exit(1)

# 测试连接
def test_connection():
    try:
        logger.info("尝试连接到数据源...")
        xtdata.connect()
        logger.info("成功连接到数据源")
        return True
    except Exception as e:
        logger.error(f"连接数据源失败: {e}")
        return False

# 直接测试获取交易日期
def test_direct_trading_dates():
    try:
        logger.info("直接调用API测试获取交易日期...")
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        logger.info(f"时间范围: {start_date} 至 {end_date}")
        # 使用xtdata直接获取交易日历
        trading_dates = xtdata.get_trading_dates(market='SH', start_time=start_date, end_time=end_date)
        
        if trading_dates and len(trading_dates) > 0:
            logger.info(f"直接API调用成功获取到 {len(trading_dates)} 个交易日期")
            logger.info(f"第一个日期: {trading_dates[0]}, 类型: {type(trading_dates[0])}")
            logger.info(f"最后一个日期: {trading_dates[-1]}")
            return True
        else:
            logger.warning("直接API调用未获取到交易日期")
            return False
    except Exception as e:
        logger.error(f"直接API调用获取交易日期失败: {e}")
        return False

# 测试通过缓存系统获取交易日期
def test_cached_trading_dates():
    try:
        logger.info("通过缓存系统测试获取交易日期...")
        
        # 导入数据缓存管理器
        try:
            from data_cache_manager import get_data_cache_manager
            logger.info("成功导入数据缓存管理器")
        except ImportError as e:
            logger.error(f"无法导入数据缓存管理器: {e}")
            return False
        
        # 创建缓存管理器实例
        cache_manager = get_data_cache_manager()
        if not cache_manager:
            logger.error("无法创建缓存管理器实例")
            return False
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        logger.info(f"时间范围: {start_date} 至 {end_date}")
        # 使用缓存管理器获取交易日历
        trading_dates = cache_manager.get_trading_dates(
            market='SH', 
            start_time=start_date,
            end_time=end_date
        )
        
        if trading_dates and len(trading_dates) > 0:
            logger.info(f"缓存系统成功获取到 {len(trading_dates)} 个交易日期")
            logger.info(f"第一个日期: {trading_dates[0]}, 类型: {type(trading_dates[0])}")
            logger.info(f"最后一个日期: {trading_dates[-1]}")
            return True
        else:
            logger.warning("缓存系统未获取到交易日期")
            return False
    except Exception as e:
        logger.error(f"缓存系统获取交易日期失败: {e}")
        logger.exception(e)  # 打印详细堆栈信息
        return False

# 测试通过集成系统获取交易日期
def test_integration_trading_dates():
    try:
        logger.info("通过集成系统测试获取交易日期...")
        
        # 导入缓存集成系统
        try:
            from cache_integration import get_cache_integration
            logger.info("成功导入缓存集成系统")
        except ImportError as e:
            logger.error(f"无法导入缓存集成系统: {e}")
            return False
        
        # 创建集成系统实例
        integration = get_cache_integration(create_new=True)
        if not integration:
            logger.error("无法创建集成系统实例")
            return False
        
        # 先记录原始函数
        original_get_trading_dates = xtdata.get_trading_dates
        
        # 集成缓存系统
        success = integration.integrate()
        if not success:
            logger.error("集成缓存系统失败")
            return False
        
        logger.info("缓存系统集成成功，测试获取交易日期...")
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        logger.info(f"时间范围: {start_date} 至 {end_date}")
        
        # 使用集成后的函数获取交易日历
        try:
            trading_dates = xtdata.get_trading_dates(
                market='SH', 
                start_time=start_date,
                end_time=end_date
            )
            
            if trading_dates and len(trading_dates) > 0:
                logger.info(f"集成系统成功获取到 {len(trading_dates)} 个交易日期")
                logger.info(f"第一个日期: {trading_dates[0]}, 类型: {type(trading_dates[0])}")
                logger.info(f"最后一个日期: {trading_dates[-1]}")
                result = True
            else:
                logger.warning("集成系统未获取到交易日期")
                result = False
        except Exception as e:
            logger.error(f"集成系统获取交易日期出错: {e}")
            logger.exception(e)
            result = False
        finally:
            # 恢复原始函数
            integration.restore()
            logger.info("已恢复原始函数")
            # 确认已恢复
            logger.info(f"函数是否已恢复: {xtdata.get_trading_dates is original_get_trading_dates}")
            
        return result
    except Exception as e:
        logger.error(f"测试集成系统获取交易日期失败: {e}")
        logger.exception(e)
        return False

# 主函数
def main():
    logger.info("===== 开始测试交易日期获取问题 =====")
    
    # 测试连接
    if not test_connection():
        logger.error("无法连接到数据源，测试终止")
        return
    
    # 直接测试获取交易日期
    direct_result = test_direct_trading_dates()
    
    # 测试通过缓存系统获取交易日期
    cached_result = test_cached_trading_dates()
    
    # 测试通过集成系统获取交易日期
    integration_result = test_integration_trading_dates()
    
    # 显示结果
    logger.info("\n===== 测试结果 =====")
    logger.info(f"直接API调用: {'成功' if direct_result else '失败'}")
    logger.info(f"缓存系统调用: {'成功' if cached_result else '失败'}")
    logger.info(f"集成系统调用: {'成功' if integration_result else '失败'}")
    
    # 分析问题
    if direct_result and not cached_result:
        logger.info("\n问题分析: 直接API调用成功但缓存系统调用失败，可能是缓存管理器配置错误或参数不一致")
    elif direct_result and not integration_result:
        logger.info("\n问题分析: 直接API调用成功但集成系统调用失败，可能是集成过程有问题或发生递归调用")
    elif not direct_result:
        logger.info("\n问题分析: 直接API调用都失败，可能是API本身配置有问题或网络连接问题")
    else:
        logger.info("\n问题分析: 所有方式都成功，可能问题出在其他地方，如初始化顺序或线程安全问题")
    
    logger.info("===== 测试结束 =====")

if __name__ == "__main__":
    main() 