#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易日期获取问题调试脚本

该脚本用于调试交易日期获取相关的问题，测试各种途径获取交易日期的差异。
包括:
1. 直接从API获取
2. 通过数据缓存管理器获取
3. 通过回测模型加载

作者: AI助手
日期: 2024-03-22
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
import threading
import traceback

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/trading_dates_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_dates_test")

# 确保当前目录在路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    logger.info(f"添加当前目录到路径: {current_dir}")

# 测试函数 - 直接使用API获取交易日期
def test_direct_api():
    """直接从API获取交易日期"""
    logger.info("===== 测试直接从API获取交易日期 =====")
    
    try:
        import xtquant.xtdata as xtdata
        logger.info("成功导入xtdata模块")
        
        # 确保连接
        xtdata.connect()
        logger.info("已连接到数据服务")
        
        # 获取交易日期
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        logger.info(f"尝试获取交易日期范围: {start_date} 至 {end_date}")
        
        start_time = time.time()
        trading_dates = xtdata.get_trading_dates(
            market='SH', 
            start_time=start_date, 
            end_time=end_date
        )
        elapsed = time.time() - start_time
        
        if trading_dates:
            logger.info(f"成功获取 {len(trading_dates)} 个交易日期，耗时: {elapsed:.3f}秒")
            logger.info(f"第一个日期: {trading_dates[0]}, 类型: {type(trading_dates[0])}")
            logger.info(f"最后一个日期: {trading_dates[-1]}, 类型: {type(trading_dates[-1])}")
        else:
            logger.error("API返回空的交易日期列表")
        
        return trading_dates
    except Exception as e:
        logger.error(f"直接API获取交易日期失败: {e}")
        logger.error(traceback.format_exc())
        return None

# 测试函数 - 通过数据缓存管理器获取
def test_cache_manager():
    """通过数据缓存管理器获取交易日期"""
    logger.info("===== 测试通过数据缓存管理器获取交易日期 =====")
    
    try:
        from data_cache_manager import get_data_cache_manager
        logger.info("成功导入数据缓存管理器")
        
        # 获取缓存管理器实例
        cache_manager = get_data_cache_manager()
        if not cache_manager:
            logger.error("无法获取缓存管理器实例")
            return None
        
        logger.info("已获取缓存管理器实例")
        
        # 获取交易日期
        start_time = time.time()
        trading_dates = cache_manager.get_trading_dates(market='SH')
        elapsed = time.time() - start_time
        
        if trading_dates:
            logger.info(f"成功通过缓存管理器获取 {len(trading_dates)} 个交易日期，耗时: {elapsed:.3f}秒")
            logger.info(f"第一个日期: {trading_dates[0]}, 类型: {type(trading_dates[0])}")
            logger.info(f"最后一个日期: {trading_dates[-1]}, 类型: {type(trading_dates[-1])}")
        else:
            logger.error("缓存管理器返回空的交易日期列表")
        
        return trading_dates
    except Exception as e:
        logger.error(f"通过缓存管理器获取交易日期失败: {e}")
        logger.error(traceback.format_exc())
        return None

# 测试函数 - 通过回测模型加载
def test_quant_model():
    """通过回测模型加载交易日期"""
    logger.info("===== 测试通过回测模型加载交易日期 =====")
    
    try:
        from advanced_quant_model import AdvancedQuantModel
        logger.info("成功导入回测模型")
        
        # 创建模型实例
        model = AdvancedQuantModel()
        logger.info("已创建回测模型实例")
        
        # 加载交易日期
        start_time = time.time()
        model.load_trading_dates()
        elapsed = time.time() - start_time
        
        if model.trading_dates:
            logger.info(f"成功通过回测模型加载 {len(model.trading_dates)} 个交易日期，耗时: {elapsed:.3f}秒")
            logger.info(f"第一个日期: {model.trading_dates[0]}, 类型: {type(model.trading_dates[0])}")
            logger.info(f"最后一个日期: {model.trading_dates[-1]}, 类型: {type(model.trading_dates[-1])}")
        else:
            logger.error("回测模型返回空的交易日期列表")
        
        return model.trading_dates
    except Exception as e:
        logger.error(f"通过回测模型加载交易日期失败: {e}")
        logger.error(traceback.format_exc())
        return None

# 测试函数 - 通过缓存集成测试
def test_cache_integration():
    """测试缓存集成对交易日期获取的影响"""
    logger.info("===== 测试缓存集成对交易日期获取的影响 =====")
    
    try:
        from cache_integration import integrate_cache_system
        logger.info("成功导入缓存集成模块")
        
        # 获取集成实例
        integration = integrate_cache_system()
        if not integration:
            logger.error("无法获取缓存集成实例")
            return None
        
        logger.info("已获取缓存集成实例，集成状态：" + ("已集成" if integration.patched else "未集成"))
        
        # 如果尚未集成，先进行集成
        if not integration.patched:
            success = integration.integrate()
            logger.info(f"执行集成操作: {'成功' if success else '失败'}")
        
        # 测试集成后的交易日期获取
        import xtquant.xtdata as xtdata
        
        start_time = time.time()
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        trading_dates = xtdata.get_trading_dates(
            market='SH', 
            start_time=start_date, 
            end_time=end_date
        )
        elapsed = time.time() - start_time
        
        if trading_dates:
            logger.info(f"成功通过集成后的API获取 {len(trading_dates)} 个交易日期，耗时: {elapsed:.3f}秒")
            logger.info(f"第一个日期: {trading_dates[0]}, 类型: {type(trading_dates[0])}")
            logger.info(f"最后一个日期: {trading_dates[-1]}, 类型: {type(trading_dates[-1])}")
        else:
            logger.error("集成后的API返回空的交易日期列表")
        
        # 恢复原始函数
        integration.restore()
        logger.info("已恢复原始API函数")
        
        return trading_dates
    except Exception as e:
        logger.error(f"测试缓存集成获取交易日期失败: {e}")
        logger.error(traceback.format_exc())
        return None

# 测试多线程场景
def test_threading():
    """测试多线程环境下的交易日期获取"""
    logger.info("===== 测试多线程环境下的交易日期获取 =====")
    
    results = {}
    
    def thread_func(method_name, method):
        try:
            logger.info(f"线程 {method_name} 开始执行")
            result = method()
            logger.info(f"线程 {method_name} 执行完成")
            results[method_name] = result
        except Exception as e:
            logger.error(f"线程 {method_name} 执行失败: {e}")
            logger.error(traceback.format_exc())
            results[method_name] = None
    
    # 创建线程
    threads = [
        threading.Thread(target=thread_func, args=("direct_api", test_direct_api)),
        threading.Thread(target=thread_func, args=("cache_manager", test_cache_manager)),
        threading.Thread(target=thread_func, args=("cache_integration", test_cache_integration))
    ]
    
    # 启动线程
    for t in threads:
        t.start()
        time.sleep(0.5)  # 间隔启动以避免竞争
    
    # 等待线程完成
    for t in threads:
        t.join()
    
    # 检查结果
    logger.info(f"多线程测试结果:")
    for name, result in results.items():
        if result:
            logger.info(f"  - {name}: 成功获取 {len(result)} 个交易日期")
        else:
            logger.info(f"  - {name}: 获取失败")
    
    return results

# 运行所有测试
def run_all_tests():
    """顺序运行所有测试"""
    logger.info("===== 开始运行所有测试 =====")
    
    # 顺序运行所有测试函数
    direct_result = test_direct_api()
    cache_result = test_cache_manager()
    model_result = test_quant_model()
    integration_result = test_cache_integration()
    
    # 分析结果
    logger.info("===== 测试结果分析 =====")
    results = {
        "直接API": direct_result,
        "缓存管理器": cache_result,
        "回测模型": model_result,
        "缓存集成": integration_result
    }
    
    for name, result in results.items():
        if result:
            logger.info(f"{name}: 成功 ({len(result)} 条数据)")
        else:
            logger.info(f"{name}: 失败")
    
    # 检查差异
    if direct_result and cache_result:
        if len(direct_result) == len(cache_result):
            logger.info("API和缓存管理器返回的交易日期数量一致")
        else:
            logger.warning(f"API和缓存管理器返回的交易日期数量不一致: {len(direct_result)} vs {len(cache_result)}")
    
    if model_result:
        if direct_result and len(direct_result) == len(model_result):
            logger.info("API和回测模型返回的交易日期数量一致")
        else:
            logger.warning(f"API和回测模型返回的交易日期数量不一致: {len(direct_result if direct_result else [])} vs {len(model_result)}")
    
    # 多线程测试
    logger.info("开始多线程测试...")
    thread_results = test_threading()
    
    logger.info("===== 测试全部完成 =====")

if __name__ == "__main__":
    try:
        logger.info("===== 交易日期获取问题调试脚本 =====")
        logger.info(f"当前工作目录: {os.getcwd()}")
        
        # 运行所有测试
        run_all_tests()
    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("脚本执行结束") 