#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强回测系统使用示例
展示如何使用增强的回测引擎进行更真实的市场模拟
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置日志
os.makedirs('to/log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/enhanced_backtest_example.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_backtest_example")

# 尝试导入补丁模块
try:
    from backtest_engine_patch import (
        patch_backtest_engine, create_enhanced_backtest_engine,
        HAS_MARKET_SIMULATOR, HAS_BACKTEST_ENGINE
    )
    logger.info("成功导入回测引擎补丁模块")
except ImportError:
    logger.error("无法导入回测引擎补丁模块，请确保backtest_engine_patch.py在正确的路径下")
    HAS_PATCH_MODULE = False
    sys.exit(1)

# 检查必要的模块是否可用
if not (HAS_MARKET_SIMULATOR and HAS_BACKTEST_ENGINE):
    logger.error("缺少必要的模块，无法运行增强的回测示例")
    sys.exit(1)

# 应用回测引擎补丁
result = patch_backtest_engine()
if not result:
    logger.error("应用回测引擎补丁失败，无法继续运行")
    sys.exit(1)

logger.info("应用回测引擎补丁成功")

def load_trading_signals(start_date, end_date):
    """
    加载或生成交易信号
    这里仅为示例，实际应用中应该使用真实的信号生成逻辑
    
    参数:
        start_date: 开始日期
        end_date: 结束日期
        
    返回:
        交易信号字典
    """
    logger.info(f"生成交易信号示例: {start_date} 至 {end_date}")
    
    # 这里仅生成一些示例信号
    # 实际应用中应该替换为实际的信号生成逻辑
    
    # 从advanced_quant_model获取交易日历
    try:
        from advanced_quant_model import AdvancedQuantModel
        model = AdvancedQuantModel()
        model.load_trading_dates()
        trading_dates = model.trading_dates
        
        # 获取回测区间内的日期
        start_idx = trading_dates.index(start_date) if start_date in trading_dates else 0
        end_idx = trading_dates.index(end_date) if end_date in trading_dates else len(trading_dates) - 1
        dates = trading_dates[start_idx:end_idx+1]
        
        # 获取股票列表示例
        stock_list = ['000001.SZ', '000002.SZ', '000063.SZ', '000568.SZ', 
                     '000625.SZ', '000651.SZ', '000725.SZ', '000768.SZ',
                     '600000.SH', '600519.SH', '600036.SH', '601318.SH',
                     '601166.SH', '601688.SH', '601888.SH', '603288.SH']
    except (ImportError, Exception) as e:
        logger.warning(f"无法从量化模型获取数据，使用示例数据: {str(e)}")
        
        # 使用示例数据
        from datetime import datetime, timedelta
        start_date_dt = datetime.strptime(start_date, '%Y%m%d')
        end_date_dt = datetime.strptime(end_date, '%Y%m%d')
        
        dates = []
        current_date = start_date_dt
        while current_date <= end_date_dt:
            # 跳过周末
            if current_date.weekday() < 5:
                dates.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
        
        stock_list = ['000001.SZ', '000002.SZ', '000063.SZ', '000568.SZ', 
                     '000625.SZ', '000651.SZ', '000725.SZ', '000768.SZ',
                     '600000.SH', '600519.SH', '600036.SH', '601318.SH',
                     '601166.SH', '601688.SH', '601888.SH', '603288.SH']
    
    # 生成交易信号
    signals = {}
    
    # 为每个交易日生成信号
    for date in dates:
        signals[date] = {}
        
        # 每5天随机选择一些股票买入和卖出
        day_idx = dates.index(date)
        
        # 买入信号
        if day_idx % 5 == 0:
            buy_stocks = np.random.choice(stock_list, size=min(5, len(stock_list)), replace=False)
            
            for stock in buy_stocks:
                signals[date][stock] = {
                    'buy': True,
                    'score': np.random.random() * 100,
                    'reason': '示例买入信号'
                }
        
        # 卖出信号
        if day_idx % 7 == 0:
            sell_stocks = np.random.choice(stock_list, size=min(3, len(stock_list)), replace=False)
            
            for stock in sell_stocks:
                if stock in signals[date]:
                    signals[date][stock]['sell'] = True
                else:
                    signals[date][stock] = {
                        'sell': True,
                        'reason': '示例卖出信号'
                    }
    
    logger.info(f"生成了 {len(dates)} 个交易日的信号")
    return signals

def run_comparison_test(start_date='20240101', end_date='20240331'):
    """
    运行不同配置的回测比较测试
    
    参数:
        start_date: 回测开始日期
        end_date: 回测结束日期
    """
    logger.info(f"开始回测比较测试: {start_date} 至 {end_date}")
    
    # 创建不同配置的回测引擎
    engine_configs = [
        {
            "name": "原始回测引擎",
            "use_enhanced_simulator": False,
            "color": "blue"
        },
        {
            "name": "零滑点模型",
            "use_enhanced_simulator": True,
            "slippage_model_type": "fixed",
            "slippage_param": 0.0,
            "color": "green"
        },
        {
            "name": "固定滑点模型(0.05%)",
            "use_enhanced_simulator": True,
            "slippage_model_type": "fixed",
            "slippage_param": 0.0005,
            "color": "red"
        },
        {
            "name": "基于成交量的滑点模型",
            "use_enhanced_simulator": True,
            "slippage_model_type": "volume",
            "slippage_param": 0.0001,
            "color": "purple"
        },
        {
            "name": "随机滑点模型(0-0.1%)",
            "use_enhanced_simulator": True,
            "slippage_model_type": "random",
            "slippage_param": 0.001,
            "color": "orange"
        }
    ]
    
    # 生成交易信号
    signals = load_trading_signals(start_date, end_date)
    
    # 运行不同配置的回测
    results = []
    
    for config in engine_configs:
        logger.info(f"运行 {config['name']} 回测...")
        
        # 创建回测引擎
        if config["use_enhanced_simulator"]:
            engine = create_enhanced_backtest_engine(
                start_date=start_date,
                end_date=end_date,
                initial_capital=1000000,
                position_size=0.1,
                max_positions=10,
                slippage_model_type=config.get("slippage_model_type", "volume"),
                slippage_param=config.get("slippage_param", 0.0001),
                commission_rate=0.00025,
                min_commission=5.0,
                price_limit_pct=0.1,
                t_plus_1=True
            )
        else:
            # 使用原始引擎
            from backtest_signals import BacktestEngine
            engine = BacktestEngine(
                start_date=start_date,
                end_date=end_date,
                initial_capital=1000000,
                position_size=0.1,
                max_positions=10
            )
        
        # 加载交易信号
        for date, date_signals in signals.items():
            if not hasattr(engine, 'trading_signals'):
                engine.trading_signals = {}
            engine.trading_signals[date] = date_signals
        
        # 运行回测
        start_time = time.time()
        result = engine.run_backtest()
        end_time = time.time()
        
        if result:
            # 收集回测结果
            equity_curve_df = pd.DataFrame(engine.equity_curve)
            equity_curve_df['date'] = pd.to_datetime(equity_curve_df['date'])
            equity_curve_df.set_index('date', inplace=True)
            
            # 计算回测性能指标
            total_return = (equity_curve_df['equity'].iloc[-1] / equity_curve_df['equity'].iloc[0]) - 1
            
            # 添加结果
            results.append({
                "name": config["name"],
                "color": config["color"],
                "equity_curve": equity_curve_df,
                "total_return": total_return,
                "sharpe_ratio": engine.sharpe_ratio if hasattr(engine, 'sharpe_ratio') else None,
                "max_drawdown": engine.max_drawdown if hasattr(engine, 'max_drawdown') else None,
                "annual_return": engine.annual_return if hasattr(engine, 'annual_return') else None,
                "trade_count": len(engine.trade_history),
                "runtime": end_time - start_time
            })
            
            logger.info(f"{config['name']} 回测完成，总收益率: {total_return:.2%}, 夏普比率: {engine.sharpe_ratio if hasattr(engine, 'sharpe_ratio') else 'N/A'}")
        else:
            logger.error(f"{config['name']} 回测失败")
    
    # 可视化比较结果
    visualize_comparison(results, start_date, end_date)
    
    return results

def visualize_comparison(results, start_date, end_date):
    """
    可视化不同回测配置的比较结果
    
    参数:
        results: 回测结果列表
        start_date: 回测开始日期
        end_date: 回测结束日期
    """
    if not results:
        logger.error("没有有效的回测结果可供可视化")
        return
    
    # 设置图形大小
    plt.figure(figsize=(15, 10))
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 绘制净值曲线
    for result in results:
        equity_curve = result["equity_curve"]
        normalized_equity = equity_curve['equity'] / equity_curve['equity'].iloc[0]
        ax1.plot(equity_curve.index, normalized_equity, label=result["name"], color=result["color"])
    
    ax1.set_title('净值曲线比较', fontsize=14)
    ax1.set_ylabel('净值 (初始值=1)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # 绘制性能指标比较
    names = [result["name"] for result in results]
    total_returns = [result["total_return"] * 100 for result in results]
    sharpe_ratios = [result["sharpe_ratio"] if result["sharpe_ratio"] is not None else 0 for result in results]
    
    # 绘制总收益率
    ax2.bar(names, total_returns, color=[result["color"] for result in results])
    ax2.set_title('总收益率比较 (%)', fontsize=14)
    ax2.set_ylabel('收益率 (%)', fontsize=12)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    
    # 为每个柱形添加标签
    for i, v in enumerate(total_returns):
        ax2.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    
    # 绘制夏普比率
    ax3.bar(names, sharpe_ratios, color=[result["color"] for result in results])
    ax3.set_title('夏普比率比较', fontsize=14)
    ax3.set_ylabel('夏普比率', fontsize=12)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    
    # 为每个柱形添加标签
    for i, v in enumerate(sharpe_ratios):
        ax3.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    file_path = f"to/backtest_comparison_{start_date}_to_{end_date}.png"
    plt.savefig(file_path)
    logger.info(f"比较结果图表已保存至 {file_path}")
    
    # 显示图形
    plt.close()
    
    # 创建性能指标表格
    performance_data = []
    for result in results:
        performance_data.append([
            result["name"],
            f"{result['total_return']:.2%}",
            f"{result['annual_return']:.2%}" if result['annual_return'] is not None else "N/A",
            f"{result['sharpe_ratio']:.2f}" if result['sharpe_ratio'] is not None else "N/A",
            f"{result['max_drawdown']:.2%}" if result['max_drawdown'] is not None else "N/A",
            result["trade_count"],
            f"{result['runtime']:.2f}秒"
        ])
    
    performance_df = pd.DataFrame(
        performance_data,
        columns=["配置", "总收益率", "年化收益率", "夏普比率", "最大回撤", "交易次数", "运行时间"]
    )
    
    # 保存表格
    table_path = f"to/backtest_comparison_{start_date}_to_{end_date}.csv"
    performance_df.to_csv(table_path, index=False, encoding='utf-8-sig')
    logger.info(f"性能指标表格已保存至 {table_path}")
    
    # 打印表格
    logger.info("\n" + performance_df.to_string(index=False))

def run_t_plus_1_test(start_date='20240101', end_date='20240331'):
    """
    T+1规则影响测试
    比较开启和关闭T+1规则对回测结果的影响
    
    参数:
        start_date: 回测开始日期
        end_date: 回测结束日期
    """
    logger.info(f"开始T+1规则影响测试: {start_date} 至 {end_date}")
    
    # 创建不同T+1配置的回测引擎
    engine_configs = [
        {
            "name": "T+1规则开启",
            "t_plus_1": True,
            "color": "blue"
        },
        {
            "name": "T+1规则关闭",
            "t_plus_1": False,
            "color": "red"
        }
    ]
    
    # 生成交易信号
    signals = load_trading_signals(start_date, end_date)
    
    # 运行不同配置的回测
    results = []
    
    for config in engine_configs:
        logger.info(f"运行 {config['name']} 回测...")
        
        # 创建回测引擎
        engine = create_enhanced_backtest_engine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=1000000,
            position_size=0.1,
            max_positions=10,
            slippage_model_type="volume",
            slippage_param=0.0001,
            commission_rate=0.00025,
            min_commission=5.0,
            price_limit_pct=0.1,
            t_plus_1=config["t_plus_1"]
        )
        
        # 加载交易信号
        for date, date_signals in signals.items():
            if not hasattr(engine, 'trading_signals'):
                engine.trading_signals = {}
            engine.trading_signals[date] = date_signals
        
        # 运行回测
        start_time = time.time()
        result = engine.run_backtest()
        end_time = time.time()
        
        if result:
            # 收集回测结果
            equity_curve_df = pd.DataFrame(engine.equity_curve)
            equity_curve_df['date'] = pd.to_datetime(equity_curve_df['date'])
            equity_curve_df.set_index('date', inplace=True)
            
            # 计算回测性能指标
            total_return = (equity_curve_df['equity'].iloc[-1] / equity_curve_df['equity'].iloc[0]) - 1
            
            # 添加结果
            results.append({
                "name": config["name"],
                "color": config["color"],
                "equity_curve": equity_curve_df,
                "total_return": total_return,
                "sharpe_ratio": engine.sharpe_ratio if hasattr(engine, 'sharpe_ratio') else None,
                "max_drawdown": engine.max_drawdown if hasattr(engine, 'max_drawdown') else None,
                "annual_return": engine.annual_return if hasattr(engine, 'annual_return') else None,
                "trade_count": len(engine.trade_history),
                "runtime": end_time - start_time
            })
            
            logger.info(f"{config['name']} 回测完成，总收益率: {total_return:.2%}, 交易次数: {len(engine.trade_history)}")
        else:
            logger.error(f"{config['name']} 回测失败")
    
    # 可视化比较结果
    plt.figure(figsize=(12, 6))
    
    for result in results:
        equity_curve = result["equity_curve"]
        normalized_equity = equity_curve['equity'] / equity_curve['equity'].iloc[0]
        plt.plot(equity_curve.index, normalized_equity, label=result["name"], color=result["color"])
    
    plt.title('T+1规则影响测试', fontsize=14)
    plt.ylabel('净值 (初始值=1)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # 保存图形
    file_path = f"to/t_plus_1_test_{start_date}_to_{end_date}.png"
    plt.savefig(file_path)
    logger.info(f"T+1测试结果图表已保存至 {file_path}")
    
    # 打印对比结果
    logger.info("\nT+1规则影响测试结果:")
    logger.info(f"T+1开启 - 总收益率: {results[0]['total_return']:.2%}, 交易次数: {results[0]['trade_count']}")
    logger.info(f"T+1关闭 - 总收益率: {results[1]['total_return']:.2%}, 交易次数: {results[1]['trade_count']}")
    
    trade_count_diff_pct = (results[1]['trade_count'] - results[0]['trade_count']) / results[0]['trade_count'] * 100
    return_diff_pct = (results[1]['total_return'] - results[0]['total_return']) / abs(results[0]['total_return']) * 100
    
    logger.info(f"交易次数增加: {trade_count_diff_pct:.2f}%")
    logger.info(f"收益率变化: {return_diff_pct:.2f}%")
    
    return results

def run_price_limit_test(start_date='20240101', end_date='20240331'):
    """
    涨跌停限制影响测试
    比较不同涨跌停限制对回测结果的影响
    
    参数:
        start_date: 回测开始日期
        end_date: 回测结束日期
    """
    logger.info(f"开始涨跌停限制影响测试: {start_date} 至 {end_date}")
    
    # 创建不同涨跌停限制配置的回测引擎
    engine_configs = [
        {
            "name": "无涨跌停限制",
            "price_limit_pct": 1.0,  # 100%，实际上相当于无限制
            "color": "blue"
        },
        {
            "name": "10%涨跌停限制",
            "price_limit_pct": 0.1,
            "color": "green"
        },
        {
            "name": "5%涨跌停限制",
            "price_limit_pct": 0.05,
            "color": "red"
        }
    ]
    
    # 生成交易信号
    signals = load_trading_signals(start_date, end_date)
    
    # 运行不同配置的回测
    results = []
    
    for config in engine_configs:
        logger.info(f"运行 {config['name']} 回测...")
        
        # 创建回测引擎
        engine = create_enhanced_backtest_engine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=1000000,
            position_size=0.1,
            max_positions=10,
            slippage_model_type="volume",
            slippage_param=0.0001,
            commission_rate=0.00025,
            min_commission=5.0,
            price_limit_pct=config["price_limit_pct"],
            t_plus_1=True
        )
        
        # 加载交易信号
        for date, date_signals in signals.items():
            if not hasattr(engine, 'trading_signals'):
                engine.trading_signals = {}
            engine.trading_signals[date] = date_signals
        
        # 运行回测
        result = engine.run_backtest()
        
        if result:
            # 收集回测结果
            equity_curve_df = pd.DataFrame(engine.equity_curve)
            equity_curve_df['date'] = pd.to_datetime(equity_curve_df['date'])
            equity_curve_df.set_index('date', inplace=True)
            
            # 计算回测性能指标
            total_return = (equity_curve_df['equity'].iloc[-1] / equity_curve_df['equity'].iloc[0]) - 1
            
            # 添加结果
            results.append({
                "name": config["name"],
                "color": config["color"],
                "equity_curve": equity_curve_df,
                "total_return": total_return,
                "sharpe_ratio": engine.sharpe_ratio if hasattr(engine, 'sharpe_ratio') else None,
                "max_drawdown": engine.max_drawdown if hasattr(engine, 'max_drawdown') else None,
                "annual_return": engine.annual_return if hasattr(engine, 'annual_return') else None,
                "trade_count": len(engine.trade_history),
                "rejected_orders": engine.market_simulator.rejected_count if hasattr(engine.market_simulator, 'rejected_count') else 0
            })
            
            logger.info(f"{config['name']} 回测完成，总收益率: {total_return:.2%}, "
                       f"交易次数: {len(engine.trade_history)}, "
                       f"被拒订单数: {engine.market_simulator.rejected_count if hasattr(engine.market_simulator, 'rejected_count') else 0}")
        else:
            logger.error(f"{config['name']} 回测失败")
    
    # 可视化比较结果
    plt.figure(figsize=(12, 6))
    
    for result in results:
        equity_curve = result["equity_curve"]
        normalized_equity = equity_curve['equity'] / equity_curve['equity'].iloc[0]
        plt.plot(equity_curve.index, normalized_equity, label=result["name"], color=result["color"])
    
    plt.title('涨跌停限制影响测试', fontsize=14)
    plt.ylabel('净值 (初始值=1)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # 保存图形
    file_path = f"to/price_limit_test_{start_date}_to_{end_date}.png"
    plt.savefig(file_path)
    logger.info(f"涨跌停限制测试结果图表已保存至 {file_path}")
    
    # 打印对比结果
    logger.info("\n涨跌停限制影响测试结果:")
    for result in results:
        logger.info(f"{result['name']} - 总收益率: {result['total_return']:.2%}, "
                   f"交易次数: {result['trade_count']}, "
                   f"被拒订单数: {result.get('rejected_orders', 0)}")
    
    return results

def main():
    """主函数"""
    logger.info("开始运行增强回测系统示例")
    
    # 设置回测参数
    start_date = '20240101'
    end_date = '20240331'
    
    # 运行对比测试
    logger.info("\n===== 运行不同滑点模型的比较测试 =====")
    comparison_results = run_comparison_test(start_date, end_date)
    
    # 运行T+1规则测试
    logger.info("\n===== 运行T+1规则影响测试 =====")
    t_plus_1_results = run_t_plus_1_test(start_date, end_date)
    
    # 运行涨跌停限制测试
    logger.info("\n===== 运行涨跌停限制影响测试 =====")
    price_limit_results = run_price_limit_test(start_date, end_date)
    
    logger.info("\n===== 增强回测系统示例运行完毕 =====")
    logger.info("回测结果和分析图表已保存到以下文件:")
    logger.info(f"- to/backtest_comparison_{start_date}_to_{end_date}.png")
    logger.info(f"- to/backtest_comparison_{start_date}_to_{end_date}.csv")
    logger.info(f"- to/t_plus_1_test_{start_date}_to_{end_date}.png")
    logger.info(f"- to/price_limit_test_{start_date}_to_{end_date}.png")

if __name__ == "__main__":
    main() 