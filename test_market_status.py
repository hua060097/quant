from to.market_status.market_status_processor import MarketStatusProcessor
import logging
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_data(start_date, end_date, index_codes):
    """创建模拟市场数据，仅用于测试
    Args:
        start_date: str，开始日期，格式YYYYMMDD
        end_date: str，结束日期，格式YYYYMMDD
        index_codes: list，指数代码列表
    Returns:
        pd.DataFrame: 模拟数据
    """
    try:
        # 将日期字符串转换为日期对象
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        # 生成日期范围（只保留工作日）
        curr_date = start
        date_range = []
        
        while curr_date <= end:
            # 只保留周一至周五
            if curr_date.weekday() < 5:
                date_range.append(curr_date)
            curr_date += timedelta(days=1)
        
        logger.info(f"生成了 {len(date_range)} 个工作日")
        
        # 生成基础数据
        data = []
        for date in date_range:
            for code in index_codes:
                # 生成随机价格和成交量
                base_price = 3000 if code in ['000001.SH', '399001.SZ'] else 1000
                price_range = base_price * 0.05  # 价格波动范围为基准价格的5%
                
                close = base_price + np.random.uniform(-price_range, price_range)
                open_price = close * (1 + np.random.uniform(-0.02, 0.02))
                high = max(close, open_price) * (1 + np.random.uniform(0, 0.02))
                low = min(close, open_price) * (1 - np.random.uniform(0, 0.02))
                volume = np.random.uniform(1000000, 10000000)
                
                data.append({
                    'date': date,
                    'code': code,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 设置日期索引
        df = df.set_index('date')
        
        # 按日期和代码排序
        df = df.sort_index().sort_values('code')
        
        logger.info(f"成功创建模拟数据: {len(index_codes)} 个指数, {len(date_range)} 个交易日")
        return df
        
    except Exception as e:
        logger.error(f"创建模拟数据时出错: {str(e)}")
        return None

def main():
    try:
        # 创建处理器实例
        logger.info("创建MarketStatusProcessor实例")
        processor = MarketStatusProcessor()
        
        # 设置测试日期和指数代码
        test_date = '20230327'
        # 计算30天前的日期作为起始日期
        start_date = '20230226'  # 约30天前
        index_codes = ['399006.SZ', '000001.SH', '000688.SH', '899050.BJ']
        
        logger.info(f"准备处理日期 {start_date} 至 {test_date} 的市场状态，指数: {index_codes}")
        print(f"处理日期范围: {start_date} 至 {test_date}")
        print("="*80)
        
        # 创建模拟数据（使用30天的历史数据以便计算技术指标）
        logger.info(f"创建模拟市场数据，时间范围: {start_date} 至 {test_date}")
        mock_data = create_mock_data(start_date, test_date, index_codes)
        if mock_data is None:
            logger.error("创建模拟数据失败")
            return
        
        # 测试方法1: 首先测试analyze_market_status方法
        logger.info("测试1: 调用analyze_market_status方法")
        try:
            market_status_dict = processor.analyze_market_status(mock_data)
            
            # 检查返回结果
            if market_status_dict and isinstance(market_status_dict, dict):
                logger.info(f"成功分析了 {len(market_status_dict)} 个交易日的市场状态")
                print(f"成功分析了 {len(market_status_dict)} 个交易日的市场状态")
                
                # 显示最后一个日期的市场状态作为示例
                if market_status_dict:
                    last_date = sorted(market_status_dict.keys())[-1]
                    market_status = market_status_dict[last_date]
                    
                    print(f"\n日期 {last_date.strftime('%Y%m%d')} 的市场状态分析结果:")
                    print("-"*40)
                    print(f"市场类型: {market_status.get('market_type', '未知')}")
                    print(f"市场周期: {market_status.get('market_cycle', '未知')}")
                    print(f"趋势强度: {market_status.get('trend_strength', 0.00):.2f}\n")
                    
                    # 获取交易建议
                    print(f"交易建议: {market_status.get('trading_advice', '观望为主，等待明确信号')}")
                    print(f"风险等级: {market_status.get('risk_level', 'low')}")
                    print(f"信号强度: {market_status.get('signal_strength', 0)}\n")
            else:
                logger.error("市场状态分析失败")
                print("\n市场状态分析失败")
        except Exception as e:
            logger.error(f"analyze_market_status方法测试出错: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"\nanalyze_market_status方法测试出错: {str(e)}")
            
        # 测试方法2: 测试process_market_status方法
        print("\n" + "="*80)
        logger.info("测试2: 调用process_market_status方法")
        try:
            # 将模拟数据存储到处理器中供测试使用
            processor._mock_data = mock_data  
            
            # 测试process_market_status
            result = processor.process_market_status(start_date, test_date, index_codes)
            
            # 显示处理结果
            if result['success']:
                logger.info(f"处理成功: {result['message']}")
                print(f"处理成功: {result['message']}")
            else:
                logger.error(f"处理失败: {result['message']}")
                print(f"处理失败: {result['message']}")
                
            # 查询保存的数据
            logger.info("查询已保存的市场状态数据")
            available_dates = processor.get_all_available_dates()
            logger.info(f"可用日期数量: {len(available_dates)}")
            print(f"可用日期数量: {len(available_dates)}")
            
            if available_dates:
                # 显示第一个和最后一个日期的市场状态
                first_date = available_dates[0]
                last_date = available_dates[-1]
                
                first_status = processor.get_market_status(first_date)
                last_status = processor.get_market_status(last_date)
                
                print(f"\n第一个日期 {first_date} 的市场类型: {first_status.get('market_type', '未知')}")
                print(f"最后一个日期 {last_date} 的市场类型: {last_status.get('market_type', '未知')}")
        except Exception as e:
            logger.error(f"process_market_status方法测试出错: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"\nprocess_market_status方法测试出错: {str(e)}")
            
        # 测试方法3: 测试process_market_data函数
        print("\n" + "="*80)
        logger.info("测试3: 调用process_market_data函数")
        try:
            # 临时替换_get_market_data方法，使用模拟数据
            original_get_market_data = processor._get_market_data
            
            def mock_get_market_data(*args, **kwargs):
                return mock_data
                
            processor._get_market_data = mock_get_market_data
            
            # 导入并测试process_market_data函数
            from to.market_status.market_status_processor import process_market_data
            
            result = process_market_data(processor, start_date, test_date, index_codes)
            
            # 恢复原始方法
            processor._get_market_data = original_get_market_data
            
            # 显示处理结果
            if result['success']:
                logger.info(f"处理成功: {result['message']}")
                print(f"处理成功: {result['message']}")
            else:
                logger.error(f"处理失败: {result['message']}")
                print(f"处理失败: {result['message']}")
                
        except Exception as e:
            logger.error(f"process_market_data函数测试出错: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"\nprocess_market_data函数测试出错: {str(e)}")
            
    except Exception as e:
        logger.error(f"测试过程中发生异常: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"测试过程中发生异常: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    logger.info("开始执行测试脚本")
    main()
    logger.info("测试脚本执行完毕") 