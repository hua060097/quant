#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于交易信号的回测系统
用于测试高级量化模型生成的交易信号的有效性

主要功能：
1. 加载模型生成的历史交易信号
2. 模拟根据信号执行交易
3. 计算回测绩效
4. 生成回测报告
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import json
import logging
import xtquant.xtdata as xtdata
from advanced_quant_model import AdvancedQuantModel
import ray
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
import platform
import concurrent.futures

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backtest")

# 顶级函数，用于进程池并行处理
def generate_trading_signals_for_date(date):
    """
    为指定日期生成交易信号的独立函数（模块级别）
    
    Args:
        date: 需要生成信号的交易日期
        
    Returns:
        tuple: (日期, 信号列表, 错误信息)
    """
    try:
        logger.info(f"子进程开始为日期 {date} 生成交易信号")
        
        # 重新创建模型实例而不是尝试序列化现有实例
        from advanced_quant_model import AdvancedQuantModel
        
        # 每个进程创建自己的模型实例
        temp_model = AdvancedQuantModel()
        
        # 加载必要的数据
        try:
            logger.info(f"为日期 {date} 加载数据")
            temp_model.load_all_data()
        except Exception as e:
            logger.error(f"为日期 {date} 加载数据时出错: {str(e)}")
            return date, None, str(e)
        
        # 生成交易信号
        try:
            logger.info(f"为日期 {date} 生成信号")
            signals = temp_model.generate_trading_signals(date)
            if signals:
                logger.info(f"日期 {date} 生成了 {len(signals)} 个交易信号")
                # 只返回可序列化的结果，不返回模型
                return date, signals, None
            else:
                logger.warning(f"日期 {date} 未生成任何交易信号")
                return date, None, "未生成任何信号"
        except Exception as e:
            logger.error(f"为日期 {date} 生成信号时出错: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"错误详情: {error_details}")
            return date, None, str(e)
        
    finally:
        # 清理资源
        if 'temp_model' in locals():
            try:
                del temp_model
            except:
                pass

# Ray远程函数，需要在模块级别定义
@ray.remote
def generate_trading_signals_ray(date):
    """Ray远程函数，为指定日期生成交易信号"""
    try:
        # 复用顶级函数中的逻辑，避免代码重复
        return generate_trading_signals_for_date(date)
    except Exception as e:
        import traceback
        logger.error(f"Ray处理日期 {date} 时出错: {e}")
        logger.error(traceback.format_exc())
        return date, None, f"Ray处理错误: {str(e)}"

# 字体设置，用于显示中文
try:
    # 配置中文字体
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows 下的黑体字体
    if os.path.exists(font_path):
        font = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font.get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用内置黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except Exception as e:
    logger.warning(f"配置中文字体失败: {e}")

# 回测参数
DEFAULT_INITIAL_CAPITAL = 1000000  # 初始资金 100万
DEFAULT_COMMISSION = 0.0003  # 佣金费率 万三
DEFAULT_SLIPPAGE = 0.0002  # 滑点
DEFAULT_POSITION_SIZE = 0.1  # 单个股票仓位比例
MAX_POSITIONS = 10  # 最大持仓数量

class BacktestEngine:
    """交易信号回测引擎"""
    
    def __init__(self, start_date='20230601', end_date=None, initial_capital=DEFAULT_INITIAL_CAPITAL,
                 commission=DEFAULT_COMMISSION, slippage=DEFAULT_SLIPPAGE,
                 position_size=DEFAULT_POSITION_SIZE, max_positions=MAX_POSITIONS):
        """
        初始化回测引擎
        
        参数:
            start_date: 回测开始日期，格式'YYYYMMDD'
            end_date: 回测结束日期，格式'YYYYMMDD'，默认为当前日期
            initial_capital: 初始资金
            commission: 佣金费率
            slippage: 滑点
            position_size: 单个股票仓位比例
            max_positions: 最大持仓数量
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y%m%d')
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.max_positions = max_positions
        
        # 回测数据
        self.capital = initial_capital  # 当前资金
        self.positions = {}  # 当前持仓 {股票代码: {数量, 成本价, 当前价, 市值, 收益率}}
        self.position_history = []  # 持仓历史
        self.trade_history = []  # 交易历史
        self.daily_capital = []  # 每日资金记录
        self.daily_portfolio_value = []  # 每日组合价值
        self.daily_returns = []  # 每日收益率
        
        # 回测结果
        self.total_return = 0  # 总收益率
        self.annual_return = 0  # 年化收益率
        self.max_drawdown = 0  # 最大回撤
        self.sharpe_ratio = 0  # 夏普比率
        self.volatility = 0  # 波动率
        self.win_rate = 0  # 胜率
        self.profit_loss_ratio = 0  # 盈亏比
        
        # 交易信号数据
        self.trading_signals = {}  # {日期: 信号列表}
        
        # 交易日历
        self.trading_dates = self._get_trading_dates()
        
        # 模型
        self.model = None  # 初始化为None，需要时加载
        
        # 并行计算设置
        self._init_parallel_computing()
        
        # 连接数据源
        self._connect_data_source()
        
        logger.info(f"回测引擎初始化完成: {start_date} 到 {end_date}")
        logger.info(f"初始资金: {initial_capital}, 佣金: {commission}, 滑点: {slippage}")
        logger.info(f"单个股票仓位比例: {position_size}, 最大持仓数量: {max_positions}")
    
    def _init_parallel_computing(self):
        """初始化并行计算环境"""
        # 确定CPU核心数量
        self.num_workers = max(1, cpu_count() - 1)  # 保留一个核心给系统
        logger.info(f"设置并行计算工作线程数: {self.num_workers}")
        
        # 初始化线程池和进程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.num_workers)
        
        # 初始化Ray (如果可用)
        self.ray_initialized = False
        try:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True,local_mode=True)
            self.ray_initialized = True
            logger.info("Ray分布式计算框架初始化成功")
        except Exception as e:
            logger.warning(f"Ray初始化失败: {e}, 将使用标准并行计算")
    
    def __del__(self):
        """析构函数，清理资源"""
        # 关闭线程池和进程池
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=False)
        
        # 关闭Ray
        if hasattr(self, 'ray_initialized') and self.ray_initialized:
            try:
                ray.shutdown()
            except:
                pass
    
    def _connect_data_source(self):
        """连接数据源"""
        try:
            xtdata.connect()
            logger.info("数据源连接成功")
        except Exception as e:
            logger.error(f"数据源连接失败: {e}")
            raise ConnectionError("无法连接到数据源")
    
    def _get_trading_dates(self):
        """获取交易日期"""
        try:
            # 获取上交所交易日历
            trading_dates = xtdata.get_trading_dates('SH', self.start_date, self.end_date)
            
            # 转换为标准日期格式
            trading_dates = [datetime.fromtimestamp(ts/1000).strftime('%Y%m%d') for ts in trading_dates]
            
            logger.info(f"获取到 {len(trading_dates)} 个交易日")
            return trading_dates
        except Exception as e:
            logger.error(f"获取交易日期失败: {e}")
            return []
    
    def _get_previous_trading_date(self, current_date, n_days=1):
        """获取前n个交易日"""
        if current_date not in self.trading_dates:
            # 找最近的交易日
            for date in sorted(self.trading_dates, reverse=True):
                if date <= current_date:
                    current_date = date
                    break
        
        try:
            current_index = self.trading_dates.index(current_date)
            if current_index < n_days:
                return None
            
            return self.trading_dates[current_index - n_days]
        except ValueError:
            logger.error(f"日期 {current_date} 不在交易日历中")
            return None
    
    def _get_stock_price(self, stock_code, date):
        """获取股票在指定日期的价格"""
        try:
            data = xtdata.get_market_data(
                field_list=['open', 'high', 'low', 'close', 'volume'],
                stock_list=[stock_code],
                period='1d',
                start_time=date,
                end_time=date
            )
            
            # 检查数据是否返回并且包含所需字段
            if data and all(field in data for field in ['open', 'high', 'low', 'close', 'volume']):
                # 创建数据结构
                try:
                    # 检查数据结构
                    if stock_code in data['close']:
                        # 转换日期格式以便正确访问数据
                        date_str = date
                        date_obj = None
                        
                        # 尝试获取datetime格式的日期
                        for date_format in data['close'].index:
                            if isinstance(date_format, str) and date_format == date:
                                date_str = date_format
                                break
                            elif hasattr(date_format, 'strftime') and date_format.strftime('%Y%m%d') == date:
                                date_obj = date_format
                                break
                        
                        # 使用找到的日期格式访问数据
                        if date_obj is not None:
                            # 如果找到了datetime格式的日期对象
                            return {
                                'open': float(data['open'][stock_code].get(date_obj, 0)),
                                'high': float(data['high'][stock_code].get(date_obj, 0)),
                                'low': float(data['low'][stock_code].get(date_obj, 0)),
                                'close': float(data['close'][stock_code].get(date_obj, 0)),
                                'volume': float(data['volume'][stock_code].get(date_obj, 0))
                            }
                        else:
                            # 使用字符串格式日期
                            return {
                                'open': float(data['open'][stock_code].get(date_str, 0)),
                                'high': float(data['high'][stock_code].get(date_str, 0)),
                                'low': float(data['low'][stock_code].get(date_str, 0)),
                                'close': float(data['close'][stock_code].get(date_str, 0)),
                                'volume': float(data['volume'][stock_code].get(date_str, 0))
                            }
                    else:
                        logger.warning(f"股票 {stock_code} 在返回数据中未找到")
                except Exception as e:
                    logger.error(f"处理股票 {stock_code} 在日期 {date} 的数据时出错: {e}")
                    # 输出数据结构以便调试
                    for field in data:
                        if field == 'close':
                            logger.info(f"close数据结构: {type(data[field])}")
                            if hasattr(data[field], 'keys'):
                                logger.info(f"close键列表: {list(data[field].keys())}")
            
            # 如果上面的处理失败或数据不完整，尝试使用备用方法
            logger.warning(f"使用备用方法获取股票 {stock_code} 在日期 {date} 的价格")
            # 尝试直接获取单只股票的当天数据
            prev_date = self._get_previous_trading_date(date, 1)
            data = xtdata.get_market_data(
                field_list=['open', 'high', 'low', 'close', 'volume'],
                stock_list=[stock_code],
                period='1d',
                start_time=prev_date if prev_date else date,
                end_time=date,
                count=2  # 获取多一天的数据，避免日期匹配问题
            )
            
            if data and all(field in data for field in ['open', 'high', 'low', 'close', 'volume']):
                # 如果数据存在，取最后一条（最近日期）
                try:
                    # 如果数据结构是字典形式
                    if isinstance(data['close'], dict) and stock_code in data['close']:
                        # 获取最后一个日期的数据
                        latest_date = list(data['close'][stock_code].keys())[-1]
                        return {
                            'open': float(data['open'][stock_code][latest_date]),
                            'high': float(data['high'][stock_code][latest_date]),
                            'low': float(data['low'][stock_code][latest_date]),
                            'close': float(data['close'][stock_code][latest_date]),
                            'volume': float(data['volume'][stock_code][latest_date])
                        }
                except Exception as e:
                    logger.error(f"备用方法处理股票 {stock_code} 数据时出错: {e}")
            
            return None
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 在日期 {date} 的价格时出错: {e}")
            return None
    
    def load_trading_signals(self, signals_file=None):
        """
        加载交易信号
        
        参数:
            signals_file: 交易信号文件路径，默认为None表示从模型生成
        """
        if signals_file and os.path.exists(signals_file):
            try:
                with open(signals_file, 'rb') as f:
                    self.trading_signals = pickle.load(f)
                logger.info(f"从文件加载了 {len(self.trading_signals)} 个日期的交易信号")
                return True
            except Exception as e:
                logger.error(f"从文件加载交易信号失败: {e}")
        
        # 使用模型生成交易信号
        logger.info("使用模型生成交易信号")
        
        if self.model is None:
            logger.info("初始化模型")
            self.model = AdvancedQuantModel()
            self.model.load_all_data()
            # 使用并行训练
            logger.info("开始训练板块模型")
            self.model.train_sector_models(parallel="auto")
        
        # 为每个交易日生成信号 - 使用并行计算
        return self._generate_trading_signals_parallel(signals_file)
    
    def _generate_trading_signals_parallel(self, signals_file):
        """并行生成交易信号"""
        # 检查是否有交易日期
        if not self.trading_dates:
            logger.error("没有交易日期，无法生成信号")
            return False
        
        # 检测操作系统类型
        is_windows = platform.system() == 'Windows'
        
        # 确定并行计算方式
        if not is_windows and self.ray_initialized:
            return self._generate_signals_ray(signals_file)
        else:
            return self._generate_signals_process_pool(signals_file)
    
    def _generate_signals_ray(self, signals_file):
        """使用Ray并行生成交易信号"""
        try:
            if not self.trading_dates:
                logger.error("没有交易日期，无法生成信号")
                return False
            
            # 确保Ray已初始化
            if not ray.is_initialized():
                try:
                    ray.init(ignore_reinit_error=True)
                    logger.info("Ray分布式计算框架初始化成功")
                except Exception as e:
                    logger.warning(f"Ray初始化失败: {e}, 将使用标准进程池")
                    return self._generate_signals_process_pool(signals_file)
            
            # 创建批次以减少并发量
            batch_size = 20  # 每批最多20个日期
            trading_date_batches = [self.trading_dates[i:i+batch_size] 
                                   for i in range(0, len(self.trading_dates), batch_size)]
            
            logger.info(f"将 {len(self.trading_dates)} 个交易日拆分为 {len(trading_date_batches)} 个批次进行Ray处理")
            
            # 存储所有生成的信号
            self.trading_signals = {}
            
            # 批次处理，避免内存压力
            for batch_idx, date_batch in enumerate(trading_date_batches):
                logger.info(f"开始处理Ray批次 {batch_idx+1}/{len(trading_date_batches)}, 包含 {len(date_batch)} 个交易日")
                
                # 创建任务
                futures = [generate_trading_signals_ray.remote(date) for date in date_batch]
                
                # 显示进度条
                with tqdm(total=len(futures), desc=f"批次 {batch_idx+1} 进度") as pbar:
                    # 获取结果并更新进度条
                    batch_signals_count = 0
                    for result in ray.get(futures):
                        date, signals, error = result
                        if signals:
                            self.trading_signals[date] = signals
                            batch_signals_count += 1
                            logger.info(f"成功获取日期 {date} 的交易信号: {len(signals)} 个")
                        else:
                            logger.warning(f"未能获取日期 {date} 的交易信号: {error}")
                        pbar.update(1)
                
                logger.info(f"Ray批次 {batch_idx+1} 处理完成，成功生成 {batch_signals_count}/{len(date_batch)} 个日期的信号")
            
            # 保存信号到文件
            if self.trading_signals:
                try:
                    if signals_file is None:
                        signals_file = 'trading_signals.pkl'
                    logger.info(f"保存 {len(self.trading_signals)} 个日期的交易信号到 {signals_file}")
                    with open(signals_file, 'wb') as f:
                        pickle.dump(self.trading_signals, f)
                    logger.info(f"交易信号已保存到 {signals_file}")
                except Exception as e:
                    logger.error(f"保存交易信号到文件时出错: {e}")
                    import traceback
                    logger.error(f"错误详情: {traceback.format_exc()}")
            
            return len(self.trading_signals) > 0
            
        except Exception as e:
            logger.error(f"使用Ray生成交易信号时出错: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 如果Ray失败，回退到进程池
            logger.info("回退到进程池方式生成信号")
            return self._generate_signals_process_pool(signals_file)
    
    def _generate_signals_process_pool(self, signals_file):
        """使用进程池生成交易信号"""
        try:
            if not self.trading_dates:
                logger.error("没有交易日期，无法生成信号")
                return False
            
            # 在Windows上，进程池启动新进程需要更多资源
            # 因此我们使用更大的批处理大小来减少进程创建次数
            is_windows = platform.system() == 'Windows'
            batch_size = 30 if not is_windows else 50
            
            # 将交易日期分批
            trading_date_batches = [self.trading_dates[i:i+batch_size]
                                  for i in range(0, len(self.trading_dates), batch_size)]
            
            logger.info(f"将 {len(self.trading_dates)} 个交易日拆分为 {len(trading_date_batches)} 个批次进行处理")
            
            # 创建进程池
            # 在Windows上限制进程数，避免资源过度消耗
            max_workers = min(self.num_workers, cpu_count() - 1) if not is_windows else max(1, min(4, cpu_count() // 2))
            logger.info(f"使用 {max_workers} 个工作进程进行并行处理")
            
            all_signals = {}
            total_processed = 0
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交批处理任务
                futures = [executor.submit(generate_signals_for_dates_batch, batch)
                          for batch in trading_date_batches]
                
                # 处理结果
                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch_signals = future.result()
                        if batch_signals:
                            batch_size = len(batch_signals)
                            logger.info(f"完成一批 {batch_size} 个交易日的信号生成")
                            total_processed += batch_size
                            # 合并结果
                            all_signals.update(batch_signals)
                    except Exception as e:
                        logger.error(f"处理交易信号批次时出错: {e}")
            
            if all_signals:
                # 保存交易信号
                if signals_file:
                    self._save_signals_to_file(all_signals, signals_file)
                
                # 设置信号
                self.trading_signals = all_signals
                signal_count = sum(len(signals) for signals in all_signals.values())
                logger.info(f"总共生成 {len(all_signals)} 个交易日的信号，共 {signal_count} 个交易信号")
                
                return True
            else:
                logger.error("未能生成任何交易信号")
                return False
        except Exception as e:
            logger.exception(f"生成交易信号时出错: {e}")
            return False
    
    def _save_signals_to_file(self, signals, filepath):
        """
        保存交易信号到文件
        
        参数:
            signals: 交易信号字典
            filepath: 文件路径
        """
        try:
            logger.info(f"保存 {len(signals)} 个日期的交易信号到 {filepath}")
            with open(filepath, 'wb') as f:
                pickle.dump(signals, f)
            logger.info(f"交易信号已保存到 {filepath}")
        except Exception as e:
            logger.error(f"保存交易信号到文件时出错: {e}")
            logger.exception("错误详情")
    
    def run_backtest(self):
        """运行回测"""
        if not self.trading_signals:
            logger.error("没有交易信号，无法进行回测")
            return False
        
        if not self.trading_dates:
            logger.error("没有交易日期，无法进行回测")
            return False
        
        logger.info(f"开始回测: {self.start_date} 到 {self.end_date}")
        
        # 初始化回测数据
        self.capital = self.initial_capital
        self.positions = {}
        self.position_history = []
        self.trade_history = []
        self.daily_capital = []
        self.daily_portfolio_value = []
        self.daily_returns = []
        
        # 记录初始资金
        self.daily_capital.append({
            'date': self.trading_dates[0],
            'capital': self.capital
        })
        
        self.daily_portfolio_value.append({
            'date': self.trading_dates[0],
            'portfolio_value': self.capital
        })
        
        # 使用并行方式进行回测
        self._run_backtest_parallel()
        
        # 计算回测结果
        self._calculate_performance_metrics()
        
        logger.info("回测完成")
        return True
    
    def _run_backtest_parallel(self):
        """并行执行回测"""
        # 创建交易日期的批次，以减少进程间通信开销
        batch_size = min(30, len(self.trading_dates) // (self.num_workers * 2) + 1)
        date_batches = [self.trading_dates[i:i+batch_size] for i in range(0, len(self.trading_dates), batch_size)]
        
        # 预加载股票价格数据以提高性能
        self._preload_price_data()
        
        # 顺序处理批次，但每个批次内并行处理日期
        for batch_idx, date_batch in enumerate(date_batches):
            logger.info(f"处理批次 {batch_idx+1}/{len(date_batches)}")
            
            # 使用线程池并行处理批次内的日期
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # 提交所有日期的任务
                future_to_date = {
                    executor.submit(self._process_backtest_day, date): date 
                    for date in date_batch
                }
                
                # 处理完成的任务结果
                for future in tqdm(as_completed(future_to_date), total=len(future_to_date), desc=f"批次 {batch_idx+1} 进度"):
                    date = future_to_date[future]
                    try:
                        result = future.result()
                        # 这里我们不需要结果，因为函数直接修改了对象状态
                    except Exception as e:
                        logger.error(f"处理日期 {date} 时出错: {e}")
            
            # 在每个批次处理完毕后整合结果
            self._consolidate_batch_results(date_batch)
    
    def _preload_price_data(self):
        """预加载价格数据以提高回测性能"""
        logger.info("预加载价格数据...")
        
        # 获取所有唯一的股票代码
        all_stocks = set()
        for date, signals in self.trading_signals.items():
            for signal in signals:
                all_stocks.add(signal['stock_code'])
        
        # 添加当前持仓的股票
        for stock in self.positions:
            all_stocks.add(stock)
        
        # 如果股票数量太多，只预加载部分数据
        if len(all_stocks) > 500:
            logger.info(f"股票数量过多 ({len(all_stocks)}), 仅预加载部分数据")
            return
        
        # 预加载价格数据
        stock_list = list(all_stocks)
        logger.info(f"预加载 {len(stock_list)} 只股票的价格数据")
        
        # 使用线程池并行加载数据
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for stock in stock_list:
                futures.append(executor.submit(
                    xtdata.get_market_data,
                    field_list=['open', 'high', 'low', 'close', 'volume'],
                    stock_list=[stock],
                    period='1d',
                    start_time=self.trading_dates[0],
                    end_time=self.trading_dates[-1]
                ))
            
            # 等待所有数据加载完成
            for future in tqdm(as_completed(futures), total=len(futures), desc="加载价格数据"):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"预加载价格数据时出错: {e}")
    
    def _process_backtest_day(self, date):
        """处理单个回测日的交易和市场行为"""
        try:
            # 更新持仓价格和市值
            self._update_positions(date)
            
            # 执行交易
            if date in self.trading_signals:
                self._execute_trades(date, self.trading_signals[date])
            
            # 返回处理结果
            return {
                'date': date,
                'positions': {k: v.copy() for k, v in self.positions.items()},
                'capital': self.capital,
                'portfolio_value': self._calculate_portfolio_value()
            }
        except Exception as e:
            logger.error(f"处理日期 {date} 时出错: {e}")
            return None
    
    def _consolidate_batch_results(self, date_batch):
        """整合批次处理结果"""
        for date in date_batch:
            portfolio_value = self._calculate_portfolio_value()
            
            # 记录当日持仓
            self.position_history.append({
                'date': date,
                'positions': {k: v.copy() for k, v in self.positions.items()}
            })
            
            # 计算收益率
            if len(self.daily_portfolio_value) > 0:
                prev_value = self.daily_portfolio_value[-1]['portfolio_value']
                daily_return = (portfolio_value / prev_value) - 1 if prev_value > 0 else 0
                self.daily_returns.append({
                    'date': date,
                    'return': daily_return
                })
            
            # 记录资金和组合价值
            self.daily_capital.append({
                'date': date,
                'capital': self.capital
            })
            
            self.daily_portfolio_value.append({
                'date': date,
                'portfolio_value': portfolio_value
            })
    
    def _get_batch_stock_prices(self, stock_list, date):
        """批量获取股票价格数据"""
        if not stock_list:
            return {}
        
        try:
            # 使用xtdata批量获取数据
            data = xtdata.get_market_data(
                field_list=['open', 'high', 'low', 'close', 'volume'],
                stock_list=stock_list,
                period='1d',
                start_time=date,
                end_time=date
            )
            
            # 整理结果，使用安全的数据访问方式
            result = {}
            if data and all(field in data for field in ['open', 'high', 'low', 'close', 'volume']):
                for stock in stock_list:
                    try:
                        # 检查数据结构
                        if stock in data['close']:
                            # 转换日期格式以便正确访问数据
                            date_str = date
                            date_obj = None
                            
                            # 尝试获取正确的索引格式
                            if isinstance(data['close'][stock], pd.Series) and not data['close'][stock].empty:
                                # 获取第一个可用的日期
                                available_date = data['close'][stock].index[0]
                                result[stock] = {
                                    'open': float(data['open'][stock].get(available_date, 0)),
                                    'high': float(data['high'][stock].get(available_date, 0)),
                                    'low': float(data['low'][stock].get(available_date, 0)),
                                    'close': float(data['close'][stock].get(available_date, 0)),
                                    'volume': float(data['volume'][stock].get(available_date, 0))
                                }
                            elif isinstance(data['close'][stock], dict):
                                # 如果是字典，直接获取日期键
                                if date in data['close'][stock]:
                                    result[stock] = {
                                        'open': float(data['open'][stock][date]),
                                        'high': float(data['high'][stock][date]),
                                        'low': float(data['low'][stock][date]),
                                        'close': float(data['close'][stock][date]),
                                        'volume': float(data['volume'][stock][date])
                                    }
                                elif len(data['close'][stock]) > 0:
                                    # 获取第一个可用日期
                                    first_date = list(data['close'][stock].keys())[0]
                                    result[stock] = {
                                        'open': float(data['open'][stock][first_date]),
                                        'high': float(data['high'][stock][first_date]),
                                        'low': float(data['low'][stock][first_date]),
                                        'close': float(data['close'][stock][first_date]),
                                        'volume': float(data['volume'][stock][first_date])
                                    }
                    except Exception as e:
                        logger.warning(f"处理股票 {stock} 批量价格数据时出错: {e}")
            
            if not result:
                logger.warning(f"无法批量获取股票价格，将逐个获取")
                # 如果批量获取失败，尝试逐个获取
                for stock in stock_list:
                    price_data = self._get_stock_price(stock, date)
                    if price_data:
                        result[stock] = price_data
            
            return result
            
        except Exception as e:
            logger.error(f"批量获取股票价格时出错: {e}")
            return {}
    
    def _update_positions(self, date):
        """并行更新持仓价格和市值"""
        # 如果没有持仓，直接返回
        if not self.positions:
            return
        
        # 获取所有持仓股票的代码
        stock_list = list(self.positions.keys())
        
        # 批量获取股票价格数据
        price_data = self._get_batch_stock_prices(stock_list, date)
        
        # 更新每个持仓的价格和市值
        for stock_code in list(self.positions.keys()):
            if stock_code in price_data:
                # 更新价格和市值
                self.positions[stock_code]['current_price'] = price_data[stock_code]['close']
                self.positions[stock_code]['market_value'] = self.positions[stock_code]['quantity'] * price_data[stock_code]['close']
                self.positions[stock_code]['return'] = (price_data[stock_code]['close'] / self.positions[stock_code]['cost_price']) - 1
            else:
                logger.warning(f"无法获取股票 {stock_code} 在日期 {date} 的价格，将从持仓中移除")
                # 如果无法获取价格，则假设以最后已知价格卖出
                last_price = self.positions[stock_code]['current_price']
                self._record_trade(
                    date=date,
                    stock_code=stock_code,
                    action='sell',
                    quantity=self.positions[stock_code]['quantity'],
                    price=last_price,
                    amount=self.positions[stock_code]['quantity'] * last_price,
                    commission=self.positions[stock_code]['quantity'] * last_price * self.commission,
                    reason='price_not_available'
                )
                
                # 更新资金
                self.capital += self.positions[stock_code]['quantity'] * last_price * (1 - self.commission)
                del self.positions[stock_code]
    
    def _execute_trades(self, date, signals):
        """
        执行交易
        
        参数:
            date: 交易日期
            signals: 交易信号列表
        """
        # 先检查是否需要卖出
        self._check_sell_signals(date)
        
        # 如果持仓数量已达上限，无法买入新的股票
        if len(self.positions) >= self.max_positions:
            logger.info(f"日期 {date} - 持仓数量已达上限 {self.max_positions}，无法买入新的股票")
            return
        
        # 计算可用于买入的资金
        available_capital = self.capital * 0.95  # 保留5%现金作为缓冲
        
        # 获取市场总体情况
        market_status = self._get_market_status(date)
        logger.info(f"日期 {date} - 市场状态: {market_status['condition']}, 波动率: {market_status['volatility']:.2%}")
        
        # 根据市场状态调整买入策略
        if market_status['condition'] == 'bearish':
            # 熊市减少仓位配置
            max_allocation = min(self.max_positions - len(self.positions), 2)
            position_size_factor = 0.7  # 减少每个仓位的大小
            logger.info(f"熊市状态，减少仓位配置，最多买入 {max_allocation} 只股票")
        elif market_status['condition'] == 'volatile':
            # 波动市场谨慎配置
            max_allocation = min(self.max_positions - len(self.positions), 3)
            position_size_factor = 0.8
            logger.info(f"高波动市场，谨慎配置，最多买入 {max_allocation} 只股票")
        elif market_status['condition'] == 'bullish':
            # 牛市加大仓位
            max_allocation = self.max_positions - len(self.positions)
            position_size_factor = 1.1  # 略微增加仓位大小
            logger.info(f"牛市状态，积极配置，最多买入 {max_allocation} 只股票")
        else:
            # 正常市场
            max_allocation = self.max_positions - len(self.positions)
            position_size_factor = 1.0
            logger.info(f"正常市场状态，标准配置，最多买入 {max_allocation} 只股票")
        
        # 对信号进行多因素评分
        scored_signals = []
        for signal in signals:
            # 跳过已持仓的股票
            if signal['stock_code'] in self.positions:
                continue
                
            # 获取股票价格和成交量数据
            price_data = self._get_stock_price(signal['stock_code'], date)
            if not price_data:
                logger.warning(f"无法获取股票 {signal['stock_code']} 的价格，跳过买入")
                continue
            
            # 分析流动性和价格波动
            vol_data = self._get_stock_vol_data(signal['stock_code'], date, lookback=5)
            if not vol_data:
                # 如果没有足够的历史数据，使用简单评分
                liquidity_score = 0.5
                volatility_score = 0.5
            else:
                # 基于平均成交量评估流动性
                avg_volume = np.mean(vol_data['volume'])
                liquidity_score = min(avg_volume / 10000000, 1.0)  # 流动性评分，最大1.0
                
                # 评估波动率
                price_volatility = np.std(vol_data['returns']) if len(vol_data['returns']) > 0 else 0
                volatility_score = 1.0 - min(price_volatility * 10, 0.7)  # 波动率越低分数越高，最低0.3
            
            # 计算综合评分
            confidence = signal.get('confidence', 0.6)
            estimated_rr_ratio = signal.get('estimated_rr_ratio', 1.5)
            sector_score = signal.get('sector_score', 0.5)
            
            # 综合评分计算 (各因素加权)
            total_score = (
                confidence * 0.35 + 
                estimated_rr_ratio * 0.25 + 
                liquidity_score * 0.15 + 
                volatility_score * 0.15 + 
                sector_score * 0.1
            )
            
            # 保存评分和相关数据
            scored_signals.append({
                'signal': signal,
                'score': total_score,
                'price_data': price_data,
                'liquidity': liquidity_score,
                'volatility': volatility_score,
                'avg_volume': vol_data.get('avg_volume', price_data['volume'])
            })
        
        # 根据评分排序
        scored_signals.sort(key=lambda x: x['score'], reverse=True)
        
        # 限制最大买入数量
        scored_signals = scored_signals[:max_allocation]
        
        if scored_signals:
            logger.info(f"候选交易股票: {len(scored_signals)}只，进行风险平衡分配")
            
            # 计算每个信号应分配的资金比例
            total_score = sum(s['score'] for s in scored_signals)
            for signal_data in scored_signals:
                # 按照评分比例分配资金
                allocation_percent = signal_data['score'] / total_score if total_score > 0 else 1.0 / len(scored_signals)
                signal_data['allocation'] = allocation_percent
            
            # 执行买入
            for signal_data in scored_signals:
                signal = signal_data['signal']
                price_data = signal_data['price_data']
                allocation = signal_data['allocation']
                liquidity = signal_data['liquidity']
                avg_volume = signal_data['avg_volume']
                
                # 计算买入数量和金额
                target_amount = available_capital * allocation * position_size_factor
                
                # 如果目标金额太小，跳过
                if target_amount < 5000:  # 最小5000元
                    logger.info(f"股票 {signal['stock_code']} 的目标金额 {target_amount:.2f} 小于5000元，跳过买入")
                    continue
                
                # 基于流动性动态计算滑点
                # 流动性越高，滑点越小；交易金额与成交量比例越高，滑点越大
                base_slippage = self.slippage
                if avg_volume > 0:
                    volume_ratio = min(target_amount / (avg_volume * price_data['close']), 0.1)  # 交易占比，最高10%
                    adjusted_slippage = base_slippage * (1 + volume_ratio * 10) * (1.5 - liquidity)
                else:
                    adjusted_slippage = base_slippage * 1.5
                
                # 计算买入价格（考虑动态滑点）
                price = price_data['close'] * (1 + adjusted_slippage)
                
                # 考虑实际市场买入情况，价格介于开盘和收盘之间
                if market_status['condition'] == 'volatile':
                    # 高波动市场，价格可能接近最高价
                    price = min(price, price_data['high'])
                
                # 计算买入数量，考虑整手(100股)交易
                quantity = int(target_amount / price / 100) * 100
                
                # 如果买入数量为0，跳过
                if quantity == 0:
                    logger.info(f"股票 {signal['stock_code']} 的买入数量为0，跳过买入")
                    continue
                
                # 计算实际买入金额
                actual_amount = quantity * price
                commission = actual_amount * self.commission
                total_cost = actual_amount + commission
                
                # 检查资金是否足够
                if total_cost > self.capital:
                    logger.info(f"资金不足，无法买入股票 {signal['stock_code']}")
                    continue
                
                # 记录交易
                self._record_trade(
                    date=date,
                    stock_code=signal['stock_code'],
                    action='buy',
                    quantity=quantity,
                    price=price,
                    amount=actual_amount,
                    commission=commission,
                    reason=signal['signal_type']
                )
                
                # 更新持仓
                self.positions[signal['stock_code']] = {
                    'quantity': quantity,
                    'cost_price': price,
                    'current_price': price,
                    'market_value': actual_amount,
                    'entry_date': date,
                    'signal_type': signal['signal_type'],
                    'stop_loss_price': signal.get('stop_loss_price', price * 0.9),  # 默认止损价
                    'take_profit_price': signal.get('take_profit_price', price * 1.2),  # 默认止盈价
                    'return': 0,
                    'max_return': 0,
                    'adjusted_slippage': adjusted_slippage
                }
                
                # 更新资金
                self.capital -= total_cost
                available_capital -= total_cost
                
                logger.info(f"买入 {signal['stock_code']} {quantity}股，价格 {price:.2f}，金额 {actual_amount:.2f}，佣金 {commission:.2f}，滑点 {adjusted_slippage:.4f}")
        else:
            logger.info(f"没有符合条件的股票信号，本日不买入")
    
    def _check_sell_signals(self, date):
        """检查卖出信号"""
        for stock_code in list(self.positions.keys()):
            position = self.positions[stock_code]
            
            # 获取最新价格
            price_data = self._get_stock_price(stock_code, date)
            if not price_data:
                logger.warning(f"无法获取股票 {stock_code} 的价格，暂不卖出")
                continue
            
            current_price = price_data['close']
            
            # 更新最大收益率
            current_return = (current_price / position['cost_price']) - 1
            position['max_return'] = max(position['max_return'], current_return)
            
            # 检查止损
            if current_price <= position['stop_loss_price']:
                self._sell_stock(date, stock_code, current_price, 'stop_loss')
                continue
            
            # 检查动态止盈
            # 如果当前收益率从最高点回撤超过一定比例，则卖出
            if position['max_return'] > 0.1:  # 最大收益率超过10%
                # 从最高点回撤超过30%则卖出
                if current_return <= position['max_return'] * 0.7:
                    self._sell_stock(date, stock_code, current_price, 'dynamic_take_profit')
                    continue
            
            # 检查持仓时间
            entry_date = position['entry_date']
            hold_days = self.trading_dates.index(date) - self.trading_dates.index(entry_date)
            
            # 持仓超过20个交易日且收益率为负，考虑卖出
            if hold_days >= 20 and current_return < 0:
                self._sell_stock(date, stock_code, current_price, 'time_stop')
                continue
            
            # 持仓超过50个交易日，无论盈亏都卖出
            if hold_days >= 50:
                self._sell_stock(date, stock_code, current_price, 'max_hold_time')
                continue
    
    def _sell_stock(self, date, stock_code, price, reason):
        """卖出股票"""
        position = self.positions[stock_code]
        quantity = position['quantity']
        amount = quantity * price
        commission = amount * self.commission
        
        # 记录交易
        self._record_trade(
            date=date,
            stock_code=stock_code,
            action='sell',
            quantity=quantity,
            price=price,
            amount=amount,
            commission=commission,
            reason=reason
        )
        
        # 更新资金
        self.capital += amount - commission
        
        # 移除持仓
        del self.positions[stock_code]
        
        logger.info(f"卖出 {stock_code} {quantity}股，价格 {price:.2f}，金额 {amount:.2f}，佣金 {commission:.2f}，原因: {reason}")
    
    def _record_trade(self, date, stock_code, action, quantity, price, amount, commission, reason):
        """记录交易"""
        self.trade_history.append({
            'date': date,
            'stock_code': stock_code,
            'action': action,
            'quantity': quantity,
            'price': price,
            'amount': amount,
            'commission': commission,
            'reason': reason
        })
    
    def _calculate_portfolio_value(self):
        """计算投资组合价值"""
        positions_value = sum(position['market_value'] for position in self.positions.values())
        return self.capital + positions_value
    
    def _calculate_performance_metrics(self):
        """计算绩效指标"""
        if not self.daily_portfolio_value or len(self.daily_portfolio_value) < 2:
            logger.error("没有足够的数据计算绩效指标")
            return
        
        # 计算总收益率
        initial_value = self.daily_portfolio_value[0]['portfolio_value']
        final_value = self.daily_portfolio_value[-1]['portfolio_value']
        self.total_return = (final_value / initial_value) - 1
        
        # 计算年化收益率
        days = len(self.daily_portfolio_value) - 1
        self.annual_return = (1 + self.total_return) ** (252 / days) - 1
        
        # 计算最大回撤
        self.max_drawdown = 0
        peak_value = initial_value
        drawdown_duration = 0
        max_drawdown_duration = 0
        current_drawdown_start = None
        max_drawdown_start = None
        max_drawdown_end = None
        
        for i, daily in enumerate(self.daily_portfolio_value):
            if daily['portfolio_value'] > peak_value:
                peak_value = daily['portfolio_value']
                # 如果设置了回撤开始日期，说明处于回撤期内，现在已恢复，计算回撤持续时间
                if current_drawdown_start:
                    drawdown_duration = i - current_drawdown_start
                    current_drawdown_start = None
                    drawdown_duration = 0
            else:
                drawdown = (peak_value - daily['portfolio_value']) / peak_value
                # 如果是新的回撤开始
                if current_drawdown_start is None:
                    current_drawdown_start = i
                
                # 统计回撤持续时间
                drawdown_duration = i - current_drawdown_start if current_drawdown_start is not None else 0
                
                # 如果是最大回撤，记录相关信息
                if drawdown > self.max_drawdown:
                    self.max_drawdown = drawdown
                    max_drawdown_start = current_drawdown_start
                    max_drawdown_end = i
                
                # 如果回撤持续时间创新高
                if drawdown_duration > max_drawdown_duration:
                    max_drawdown_duration = drawdown_duration
        
        self.max_drawdown_duration = max_drawdown_duration
        if max_drawdown_start is not None and max_drawdown_end is not None:
            self.max_drawdown_start_date = self.daily_portfolio_value[max_drawdown_start]['date']
            self.max_drawdown_end_date = self.daily_portfolio_value[max_drawdown_end]['date']
        
        # 计算波动率和夏普比率
        if len(self.daily_returns) > 1:
            returns = [r['return'] for r in self.daily_returns]
            self.volatility = np.std(returns) * np.sqrt(252)
            avg_return = np.mean(returns) * 252
            risk_free_rate = 0.03  # 假设无风险收益率为3%
            self.sharpe_ratio = (avg_return - risk_free_rate) / self.volatility if self.volatility > 0 else 0
            
            # 计算下行波动率（仅考虑负收益）
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                self.downside_volatility = np.std(negative_returns) * np.sqrt(252)
                # 计算索提诺比率（使用下行波动率而非总波动率）
                self.sortino_ratio = (avg_return - risk_free_rate) / self.downside_volatility if self.downside_volatility > 0 else 0
            else:
                self.downside_volatility = 0
                self.sortino_ratio = float('inf')  # 如果没有负收益，设为无穷大
        
        # 计算最大连续盈利/亏损天数
        if len(self.daily_returns) > 0:
            current_streak = 1
            max_win_streak = 0
            max_loss_streak = 0
            
            for i in range(1, len(self.daily_returns)):
                current_return = self.daily_returns[i]['return']
                prev_return = self.daily_returns[i-1]['return']
                
                if (current_return > 0 and prev_return > 0) or (current_return < 0 and prev_return < 0):
                    current_streak += 1
                else:
                    # 记录最大连续盈利/亏损
                    if prev_return > 0:
                        max_win_streak = max(max_win_streak, current_streak)
                    else:
                        max_loss_streak = max(max_loss_streak, current_streak)
                    current_streak = 1
            
            # 检查最后一个streak
            if len(self.daily_returns) > 0:
                last_return = self.daily_returns[-1]['return']
                if last_return > 0:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)
            
            self.max_consecutive_wins = max_win_streak
            self.max_consecutive_losses = max_loss_streak
        
        # 计算信息比率（相对于基准的超额收益与超额波动率之比）
        if hasattr(self, 'benchmark_returns') and len(self.benchmark_returns) > 0:
            # 确保长度一致
            min_length = min(len(returns), len(self.benchmark_returns))
            returns = returns[:min_length]
            benchmark_returns = self.benchmark_returns[:min_length]
            
            # 计算超额收益
            excess_returns = [r - b for r, b in zip(returns, benchmark_returns)]
            
            if len(excess_returns) > 1:
                excess_volatility = np.std(excess_returns) * np.sqrt(252)
                avg_excess_return = np.mean(excess_returns) * 252
                self.information_ratio = avg_excess_return / excess_volatility if excess_volatility > 0 else 0
            else:
                self.information_ratio = 0
        else:
            self.information_ratio = None
        
        # 计算VaR（历史法）
        if len(returns) > 20:  # 确保有足够的数据点
            self.var_95 = np.percentile(returns, 5) * np.sqrt(252)  # 95% VaR（年化）
            self.var_99 = np.percentile(returns, 1) * np.sqrt(252)  # 99% VaR（年化）
        else:
            self.var_95 = None
            self.var_99 = None
        
        # 计算胜率和盈亏比
        winning_trades = [t for t in self.trade_history if t['action'] == 'sell' and 
                          t['amount'] > self._find_buy_amount(t['stock_code'], t['date'])]
        
        losing_trades = [t for t in self.trade_history if t['action'] == 'sell' and 
                         t['amount'] <= self._find_buy_amount(t['stock_code'], t['date'])]
        
        self.win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if (len(winning_trades) + len(losing_trades)) > 0 else 0
        
        # 计算平均盈利和平均亏损
        avg_profit = np.mean([t['amount'] - self._find_buy_amount(t['stock_code'], t['date']) for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([self._find_buy_amount(t['stock_code'], t['date']) - t['amount'] for t in losing_trades]) if losing_trades else 1
        
        self.profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
        
        # 计算平均持仓时间
        holding_days = []
        for t in self.trade_history:
            if t['action'] == 'sell':
                # 找到对应的买入记录
                buy_trade = None
                for bt in self.trade_history:
                    if bt['action'] == 'buy' and bt['stock_code'] == t['stock_code'] and bt['date'] < t['date']:
                        buy_trade = bt
                        break
                
                if buy_trade:
                    buy_date = datetime.strptime(buy_trade['date'], '%Y%m%d')
                    sell_date = datetime.strptime(t['date'], '%Y%m%d')
                    days = (sell_date - buy_date).days
                    holding_days.append(days)
        
        self.avg_holding_days = np.mean(holding_days) if holding_days else 0
        
        # 输出主要指标
        logger.info(f"======== 回测绩效指标 ========")
        logger.info(f"回测区间: {self.daily_portfolio_value[0]['date']} 至 {self.daily_portfolio_value[-1]['date']}")
        logger.info(f"总收益率: {self.total_return:.2%}")
        logger.info(f"年化收益率: {self.annual_return:.2%}")
        logger.info(f"最大回撤: {self.max_drawdown:.2%}")
        logger.info(f"最大回撤持续天数: {self.max_drawdown_duration}")
        if hasattr(self, 'max_drawdown_start_date') and hasattr(self, 'max_drawdown_end_date'):
            logger.info(f"最大回撤区间: {self.max_drawdown_start_date} 至 {self.max_drawdown_end_date}")
        logger.info(f"波动率(年化): {self.volatility:.2%}")
        logger.info(f"夏普比率: {self.sharpe_ratio:.2f}")
        logger.info(f"索提诺比率: {self.sortino_ratio:.2f}")
        if self.information_ratio is not None:
            logger.info(f"信息比率: {self.information_ratio:.2f}")
        logger.info(f"胜率: {self.win_rate:.2%}")
        logger.info(f"盈亏比: {self.profit_loss_ratio:.2f}")
        logger.info(f"最大连续盈利天数: {self.max_consecutive_wins}")
        logger.info(f"最大连续亏损天数: {self.max_consecutive_losses}")
        logger.info(f"平均持仓时间: {self.avg_holding_days:.1f}天")
        if self.var_95 is not None:
            logger.info(f"95% 风险价值(VaR): {abs(self.var_95):.2%}")
        if self.var_99 is not None:
            logger.info(f"99% 风险价值(VaR): {abs(self.var_99):.2%}")
        logger.info(f"===============================")
    
    def _find_buy_amount(self, stock_code, sell_date):
        """根据股票代码和卖出日期找到对应的买入金额"""
        for trade in self.trade_history:
            if trade['stock_code'] == stock_code and trade['action'] == 'buy' and trade['date'] <= sell_date:
                return trade['amount']
        return 0
    
    def plot_portfolio_performance(self):
        """绘制投资组合表现"""
        if not self.daily_portfolio_value or len(self.daily_portfolio_value) < 2:
            logger.error("没有足够的数据绘制图表")
            return
        
        # 准备数据
        dates = [datetime.strptime(d['date'], '%Y%m%d') for d in self.daily_portfolio_value]
        portfolio_values = [d['portfolio_value'] for d in self.daily_portfolio_value]
        initial_value = portfolio_values[0]
        portfolio_returns = [v / initial_value - 1 for v in portfolio_values]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制投资组合价值
        ax1.plot(dates, portfolio_values, label='投资组合价值', color='blue')
        ax1.set_title('投资组合价值变化')
        ax1.set_ylabel('价值 (元)')
        ax1.grid(True)
        ax1.legend()
        
        # 格式化x轴日期
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        fig.autofmt_xdate()
        
        # 绘制收益率
        ax2.plot(dates, portfolio_returns, label='收益率', color='green')
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax2.set_title('收益率变化')
        ax2.set_ylabel('收益率 (%)')
        ax2.set_xlabel('日期')
        ax2.grid(True)
        ax2.legend()
        
        # 格式化y轴为百分比
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图表
        plt.savefig('portfolio_performance.png', dpi=300)
        plt.close()
        
        logger.info("投资组合表现图表已保存为 portfolio_performance.png")
    
    def plot_monthly_returns(self):
        """绘制月度收益热图"""
        if not self.daily_returns or len(self.daily_returns) < 30:
            logger.error("没有足够的数据绘制月度收益热图")
            return
        
        # 准备数据
        df = pd.DataFrame(self.daily_returns)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df.set_index('date', inplace=True)
        
        # 计算月度收益
        monthly_returns = df['return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns.to_frame()
        
        # 添加年份和月份列
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        
        # 数据透视为表格形式
        pivot_table = monthly_returns.pivot_table(index='year', columns='month', values='return')
        
        # 设置月份名称
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
        
        # 创建热图
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='RdYlGn', 
                   linewidths=.5, center=0, vmin=-0.1, vmax=0.1)
        plt.title('月度收益热图')
        
        # 保存图表
        plt.savefig('monthly_returns_heatmap.png', dpi=300)
        plt.close()
        
        logger.info("月度收益热图已保存为 monthly_returns_heatmap.png")
    
    def generate_report(self):
        """生成回测报告"""
        if not self.daily_portfolio_value or len(self.daily_portfolio_value) < 2:
            logger.error("没有足够的数据生成报告")
            return
        
        # 生成绘图
        self.plot_portfolio_performance()
        self.plot_monthly_returns()
        
        # 生成报告数据
        report = {
            "回测信息": {
                "开始日期": self.start_date,
                "结束日期": self.end_date,
                "初始资金": self.initial_capital,
                "结束资金": self.daily_portfolio_value[-1]['portfolio_value'],
                "交易日数": len(self.trading_dates)
            },
            "绩效指标": {
                "总收益率": f"{self.total_return:.2%}",
                "年化收益率": f"{self.annual_return:.2%}",
                "最大回撤": f"{self.max_drawdown:.2%}",
                "波动率": f"{self.volatility:.2%}",
                "夏普比率": f"{self.sharpe_ratio:.2f}",
                "胜率": f"{self.win_rate:.2%}",
                "盈亏比": f"{self.profit_loss_ratio:.2f}"
            },
            "交易统计": {
                "总交易次数": len(self.trade_history),
                "买入次数": len([t for t in self.trade_history if t['action'] == 'buy']),
                "卖出次数": len([t for t in self.trade_history if t['action'] == 'sell']),
                "平均持仓时间": self._calculate_avg_hold_days(),
                "总佣金": sum(t['commission'] for t in self.trade_history)
            }
        }
        
        # 输出报告
        logger.info("\n===== 回测报告 =====")
        for section, data in report.items():
            logger.info(f"\n{section}:")
            for key, value in data.items():
                logger.info(f"  {key}: {value}")
        
        # 保存报告为JSON
        try:
            with open('backtest_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=4)
            logger.info("回测报告已保存为 backtest_report.json")
        except Exception as e:
            logger.error(f"保存回测报告时出错: {e}")
        
        # 保存交易历史为CSV
        try:
            trades_df = pd.DataFrame(self.trade_history)
            trades_df.to_csv('trade_history.csv', index=False)
            logger.info("交易历史已保存为 trade_history.csv")
        except Exception as e:
            logger.error(f"保存交易历史时出错: {e}")
        
        # 保存每日净值为CSV
        try:
            daily_df = pd.DataFrame(self.daily_portfolio_value)
            daily_df.to_csv('daily_portfolio_value.csv', index=False)
            logger.info("每日净值已保存为 daily_portfolio_value.csv")
        except Exception as e:
            logger.error(f"保存每日净值时出错: {e}")
        
        return report
    
    def _calculate_avg_hold_days(self):
        """计算平均持仓天数"""
        hold_days = []
        
        # 准备买入记录字典
        buy_records = {}
        for trade in self.trade_history:
            if trade['action'] == 'buy':
                buy_records[trade['stock_code']] = trade['date']
        
        # 计算每个卖出记录的持仓天数
        for trade in self.trade_history:
            if trade['action'] == 'sell' and trade['stock_code'] in buy_records:
                buy_date = buy_records[trade['stock_code']]
                sell_date = trade['date']
                
                try:
                    buy_index = self.trading_dates.index(buy_date)
                    sell_index = self.trading_dates.index(sell_date)
                    hold_days.append(sell_index - buy_index)
                except ValueError:
                    continue
        
        return np.mean(hold_days) if hold_days else 0
    
    def _get_market_status(self, date):
        """分析市场整体状态"""
        try:
            # 获取指数数据（上证、深证、创业板）
            indices = ['000001.SH', '399001.SZ', '399006.SZ']
            lookback = 10  # 回看天数
            
            # 获取前N天数据
            prev_date = self._get_previous_trading_date(date, lookback)
            if not prev_date:
                return {'condition': 'normal', 'volatility': 0.01}
                
            data = xtdata.get_market_data(
                field_list=['close'],
                stock_list=indices,
                period='1d',
                start_time=prev_date,
                end_time=date
            )
            
            if data is None or 'close' not in data:
                return {'condition': 'normal', 'volatility': 0.01}
            
            # 分析指数表现
            market_analysis = {}
            for index in indices:
                if index in data['close'].index:
                    prices = data['close'].loc[index]
                    if len(prices) >= 2:
                        # 计算收益率序列
                        price_list = [p for d, p in prices.items()]
                        returns = [(price_list[i] / price_list[i-1] - 1) for i in range(1, len(price_list))]
                        
                        # 计算指标
                        current_return = price_list[-1] / price_list[0] - 1
                        volatility = np.std(returns) if len(returns) > 1 else 0.01
                        
                        # 趋势判断
                        positive_days = sum(1 for r in returns if r > 0)
                        trend_ratio = positive_days / len(returns) if returns else 0.5
                        
                        market_analysis[index] = {
                            'return': current_return,
                            'volatility': volatility,
                            'trend_ratio': trend_ratio,
                        }
            
            # 如果没有得到任何数据
            if not market_analysis:
                return {'condition': 'normal', 'volatility': 0.01}
            
            # 计算组合指标
            avg_return = np.mean([ma['return'] for ma in market_analysis.values()])
            avg_volatility = np.mean([ma['volatility'] for ma in market_analysis.values()])
            avg_trend = np.mean([ma['trend_ratio'] for ma in market_analysis.values()])
            
            # 判断市场状态
            condition = 'normal'
            if avg_return > 0.05 and avg_trend > 0.6:
                condition = 'bullish'  # 牛市
            elif avg_return < -0.05 and avg_trend < 0.4:
                condition = 'bearish'  # 熊市
            elif avg_volatility > 0.02:
                condition = 'volatile'  # 波动市场
            
            return {
                'condition': condition,
                'volatility': avg_volatility,
                'return': avg_return,
                'trend': avg_trend,
                'details': market_analysis
            }
            
        except Exception as e:
            logger.error(f"分析市场状态时出错: {e}")
            return {'condition': 'normal', 'volatility': 0.01}
    
    def _get_stock_vol_data(self, stock_code, date, lookback=5):
        """获取股票的成交量和价格波动数据"""
        try:
            # 获取前N天数据
            prev_date = self._get_previous_trading_date(date, lookback)
            if not prev_date:
                return None
                
            data = xtdata.get_market_data(
                field_list=['close', 'volume'],
                stock_list=[stock_code],
                period='1d',
                start_time=prev_date,
                end_time=date
            )
            
            if data is None or 'close' not in data or 'volume' not in data:
                return None
                
            if stock_code in data['close'].index and stock_code in data['volume'].index:
                prices = data['close'].loc[stock_code]
                volumes = data['volume'].loc[stock_code]
                
                if len(prices) >= 2:
                    # 整理成列表
                    price_list = [p for d, p in prices.items()]
                    volume_list = [v for d, v in volumes.items()]
                    
                    # 计算收益率
                    returns = [(price_list[i] / price_list[i-1] - 1) for i in range(1, len(price_list))]
                    
                    return {
                        'prices': price_list,
                        'volume': volume_list,
                        'returns': returns,
                        'avg_volume': np.mean(volume_list) if volume_list else 0
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 成交量数据时出错: {e}")
            return None

def generate_signals_for_dates_batch(dates_batch):
    """
    为一批日期生成交易信号
    
    参数:
        dates_batch: 日期列表
        
    返回:
        dict: 日期到信号的映射 {date: signals}
    """
    batch_signals = {}
    for date in dates_batch:
        try:
            # 复用现有的单日期信号生成函数
            date_str, signals, error = generate_trading_signals_for_date(date)
            if signals:
                batch_signals[date_str] = signals
        except Exception as e:
            logger.error(f"为日期 {date} 生成信号时出错: {e}")
    
    return batch_signals

# 主函数
def main():
    """主函数"""
    logger.info("开始回测交易信号")
    
    # 初始化回测引擎
    engine = BacktestEngine(
        start_date='20230701',
        end_date='20240301',
        initial_capital=1000000,
        position_size=0.1,
        max_positions=10
    )
    
    # 加载或生成交易信号
    logger.info("加载交易信号")
    if not engine.load_trading_signals():
        logger.error("无法加载或生成交易信号")
        return
    
    # 运行回测
    logger.info("运行回测")
    if not engine.run_backtest():
        logger.error("回测失败")
        return
    
    # 生成报告
    logger.info("生成回测报告")
    engine.generate_report()
    
    logger.info("回测完成")


if __name__ == "__main__":
    main() 