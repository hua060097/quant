import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import concurrent.futures
import logging
import json
from tqdm import tqdm
from pathlib import Path
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# 添加项目根目录到导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入市场分析器 - 修正导入路径
try:
    from market_analyzer import MarketAnalyzer  # 优先使用相对导入
except ImportError:
    try:
        from to.market_status.market_analyzer import MarketAnalyzer  # 尝试从包导入
    except ImportError:
        from .market_analyzer import MarketAnalyzer  # 尝试相对导入

# 尝试预先导入数据存储模块
ParquetDataStorage = None
try:
    from to.advanced_quant_data import ParquetDataStorage
    print("成功预加载to.advanced_quant_data模块")
except ImportError:
    try:
        sys.path.insert(0, os.path.dirname(current_dir))
        from advanced_quant_data import ParquetDataStorage
        print("成功预加载advanced_quant_data模块（相对路径）")
    except ImportError:
        print("无法预加载ParquetDataStorage模块，将在需要时尝试导入")

# 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('to/log/market_status_processor.log', encoding='utf-8')
#     ])
logger = logging.getLogger(__name__)

class MarketStatusProcessor:
    """大盘状态处理器 - 预处理并存储大盘状态数据
    
    功能特点:
    1. 批量预处理市场状态数据并存储为高效格式
    2. 支持按日期范围快速检索市场状态
    3. 多进程并行处理提高性能
    4. 与现有回测框架无缝集成
    """
    
    def __init__(self, data_dir="local_data/market_status"):
        """初始化市场状态处理器"""
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 获取当前文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))  # 获取项目根目录
        
        # 设置数据目录 - 使用项目根目录作为基准
        self.data_dir = data_dir
        self.daily_status_dir = os.path.normpath(os.path.join(project_root, "to", "local_data", "market_status", "daily"))
        self.metadata_dir = os.path.normpath(os.path.join(project_root, "to", "local_data", "market_status", "metadata"))
        
        # 创建必要的目录
        os.makedirs(self.daily_status_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # 验证目录权限
        self._validate_directories()
        
        # 初始化市场分析器
        self.analyzer = MarketAnalyzer()
        
        # 初始化原始数据存储
        self.raw_data_storage = None
        
        # 设置模拟数据（用于测试）
        self._mock_data = None
        
        # 加载市场状态映射
        self.MARKET_STATUS = {
            11: "强势突破上涨",
            12: "加速上涨",
            13: "稳健上涨",
            21: "盘升上涨",
            22: "回调上涨",
            23: "整固上涨",
            31: "强势盘整",
            32: "中性盘整",
            33: "弱势盘整",
            41: "盘跌下行",
            42: "回抽下跌",
            43: "整固下跌",
            51: "加速下跌",
            52: "恐慌下跌",
            53: "超跌反弹"
        }
        
        # 加载市场周期映射
        self.MARKET_CYCLE = {
            1: "积累期",
            2: "上升期",
            3: "分配期",
            4: "下降期"
        }
        
        # 加载指数权重配置
        self.INDEX_WEIGHTS = {
            "399006.SZ": 0.9,  # 创业板指
            '899050.BJ': 0.1, 
        }
        
        # 设置默认参数
        self.default_start_date = '20230101'  # 使用2023年1月1日作为默认开始日期
        self.default_end_date = '20230331'   # 使用2023年3月31日作为默认结束日期
        self.default_index_codes = ['399006.SZ','899050.BJ']
        
        # 初始化元数据
        self._init_metadata()
    
    def _init_metadata(self):
        """初始化元数据"""
        # 初始化市场状态数据
        self.market_status_data = {}
        
        # 初始化元数据文件
        self.metadata_file = os.path.join(self.metadata_dir, "status_metadata.json")
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "available_dates": [],
                "last_update": "",
                "index_codes": self.default_index_codes,
                "data_schema": {
                    "version": "1.0",
                    "format": "json"
                }
            }
            self._save_metadata()
            
        # 加载已有的市场状态数据
        status_file = os.path.join(self.daily_status_dir, "market_status.json")
        if os.path.exists(status_file):
            try:
                # ✅ 修复：检查文件是否为空
                if os.path.getsize(status_file) == 0:
                    self.logger.warning(f"市场状态文件为空: {status_file}，将初始化为空字典")
                    self.market_status_data = {}
                else:
                    with open(status_file, "r", encoding='utf-8') as f:
                        self.market_status_data = json.load(f)
                    self.logger.info(f"成功加载市场状态数据，包含 {len(self.market_status_data)} 个日期的数据")
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON 解析失败: {str(e)}，文件可能损坏或格式错误")
                self.logger.info(f"将备份并重新创建市场状态文件")
                # 备份损坏的文件
                backup_file = status_file + ".backup"
                try:
                    import shutil
                    shutil.copy2(status_file, backup_file)
                    self.logger.info(f"已备份损坏的文件到: {backup_file}")
                except Exception as backup_error:
                    self.logger.warning(f"备份文件失败: {backup_error}")
                self.market_status_data = {}
            except Exception as e:
                self.logger.error(f"加载市场状态数据时出错: {str(e)}")
                self.market_status_data = {}
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=4)
    
    def _date_to_str(self, date):
        """转换日期为字符串格式"""
        if isinstance(date, str):
            return date
        elif isinstance(date, (datetime, pd.Timestamp)):
            return date.strftime("%Y%m%d")
        else:
            raise ValueError(f"不支持的日期格式: {type(date)}")
    
    def _str_to_date(self, date_str):
        """转换字符串为日期格式"""
        if isinstance(date_str, (datetime, pd.Timestamp)):
            return date_str
        return pd.Timestamp(date_str)
    
    def _get_date_range(self, start_date, end_date):
        """获取日期范围"""
        start_date = self._str_to_date(start_date)
        end_date = self._str_to_date(end_date)
        
        # 从原始数据存储获取交易日历
        if self.raw_data_storage:
            try:
                # 尝试使用get_trading_calendar方法
                if hasattr(self.raw_data_storage, 'get_trading_calendar') and callable(getattr(self.raw_data_storage, 'get_trading_calendar')):
                    logger.info(f"使用get_trading_calendar获取交易日历: {start_date} - {end_date}")
                    calendar_df = self.raw_data_storage.get_trading_calendar(
                        self._date_to_str(start_date), 
                        self._date_to_str(end_date)
                    )
                    if calendar_df is not None and not calendar_df.empty:
                        # 根据DataFrame的结构提取日期列
                        date_column = None
                        for col in ['date', 'date_str', 'trade_date']:
                            if col in calendar_df.columns:
                                date_column = col
                                break
                        
                        if date_column:
                            trade_dates = calendar_df[date_column].tolist()
                            logger.info(f"成功获取 {len(trade_dates)} 个交易日")
                            return [self._str_to_date(d) for d in trade_dates]
                
                # 尝试使用get_trade_dates方法
                if hasattr(self.raw_data_storage, 'get_trade_dates') and callable(getattr(self.raw_data_storage, 'get_trade_dates')):
                    logger.info(f"使用get_trade_dates获取交易日历: {start_date} - {end_date}")
                    trade_dates = self.raw_data_storage.get_trade_dates(
                        self._date_to_str(start_date), 
                        self._date_to_str(end_date)
                    )
                    
                    if trade_dates:
                        logger.info(f"成功获取 {len(trade_dates)} 个交易日")
                        return [self._str_to_date(d) for d in trade_dates]
            except Exception as e:
                logger.error(f"获取交易日历失败: {str(e)}")
                logger.info("将使用连续日期范围作为备选方案")
        
        # 如果没有交易日历或获取失败，生成连续日期
        logger.info(f"使用连续日期范围: {start_date} - {end_date}")
        date_range = []
        curr_date = start_date
        while curr_date <= end_date:
            date_range.append(curr_date)
            curr_date += timedelta(days=1)
        
        logger.info(f"生成了 {len(date_range)} 个连续日期")
        return date_range
    
    def _get_file_path(self, date):
        """获取指定日期的文件路径"""
        date_str = self._date_to_str(date)
        year_month = date_str[:6]
        
        # 构建目录路径并规范化
        dir_path = os.path.normpath(os.path.join(self.daily_status_dir, year_month))
        os.makedirs(dir_path, exist_ok=True)
        
        # 返回规范化的绝对路径
        return os.path.normpath(os.path.join(dir_path, f"market_status_{date_str}.parquet"))
    
    def _load_trading_calendar(self):
        """
        加载交易日历

        返回:
            pd.DataFrame: 交易日历，包含 date_str, date, market 列
            None: 加载失败
        """
        try:
            import pandas as pd
            from pathlib import Path

            # 构建交易日历文件路径
            # 当前文件: to/market_status/market_status_processor.py
            # 交易日历: to/local_data/calendar/trading_calendar.parquet
            current_dir = Path(__file__).resolve().parent  # to/market_status/
            project_root = current_dir.parent.parent  # path/
            calendar_file = project_root / "to" / "local_data" / "calendar" / "trading_calendar.parquet"

            self.logger.debug(f"加载交易日历: {calendar_file}")

            if not calendar_file.exists():
                self.logger.error(f"交易日历文件不存在: {calendar_file}")
                return None

            # 读取交易日历
            trading_calendar = pd.read_parquet(calendar_file)

            # 验证必要列
            if 'date_str' not in trading_calendar.columns:
                self.logger.error("交易日历缺少 date_str 列")
                return None

            self.logger.info(f"成功加载交易日历: {len(trading_calendar)} 个交易日")
            self.logger.debug(f"交易日历范围: {trading_calendar['date_str'].min()} ~ {trading_calendar['date_str'].max()}")

            return trading_calendar

        except Exception as e:
            self.logger.error(f"加载交易日历失败: {str(e)}")
            return None

    def _get_history_start_date_by_trading_days(self, start_date_str, trading_days_back=180):
        """
        根据交易日历回溯指定数量的交易日

        参数:
            start_date_str: str, 开始日期字符串 (YYYYMMDD)
            trading_days_back: int, 回溯的交易日数量，默认180

        返回:
            str: 回溯后的日期字符串 (YYYYMMDD)
            None: 失败或交易日历不足
        """
        try:
            # 加载交易日历
            trading_calendar = self._load_trading_calendar()
            if trading_calendar is None:
                self.logger.warning("无法加载交易日历，使用自然日回溯（备用方案）")
                return None

            # 找到 start_date_str 在交易日历中的位置
            matching_rows = trading_calendar[trading_calendar['date_str'] == start_date_str]

            if matching_rows.empty:
                self.logger.warning(f"开始日期 {start_date_str} 不在交易日历中")
                # 尝试找到最接近的交易日
                # 将 start_date_str 转换为 datetime
                from datetime import datetime
                start_date_dt = datetime.strptime(start_date_str, '%Y%m%d')

                # 找到最接近的交易日（之后或等于）
                future_dates = trading_calendar[trading_calendar['date'] >= start_date_dt]
                if not future_dates.empty:
                    closest_date_str = future_dates.iloc[0]['date_str']
                    self.logger.info(f"开始日期 {start_date_str} 不是交易日，使用最接近的交易日 {closest_date_str}")
                    start_date_str = closest_date_str
                    matching_rows = trading_calendar[trading_calendar['date_str'] == start_date_str]
                else:
                    self.logger.error(f"无法找到接近 {start_date_str} 的交易日")
                    return None

            # 获取索引
            start_index = matching_rows.index[0]

            # 回溯交易日
            history_index = start_index - trading_days_back

            # 检查边界
            if history_index < 0:
                self.logger.warning(
                    f"交易日历不足 {trading_days_back} 个交易日 "
                    f"(可用: {start_index + 1} 个)，将使用最早交易日"
                )
                history_index = 0

            # 获取历史开始日期
            history_start_date_str = trading_calendar.iloc[history_index]['date_str']

            self.logger.info(
                f"交易日历回溯: {start_date_str} (索引 {start_index}) "
                f"-> {history_start_date_str} (索引 {history_index}), "
                f"回溯 {start_index - history_index} 个交易日"
            )

            return history_start_date_str

        except Exception as e:
            self.logger.error(f"根据交易日历回溯日期失败: {str(e)}")
            return None

    def _get_market_data(self, start_date, end_date, index_codes=None):
        """
        获取指定日期范围内的市场数据
        
        参数:
            start_date: 开始日期，可以是日期对象或字符串
            end_date: 结束日期，可以是日期对象或字符串
            index_codes: 指数代码列表，默认为创业板指数
            
        返回:
            dict: 包含市场数据的字典
        """
        try:
            from to.market_status.market_analyzer import MarketAnalyzer
            # 直接使用MarketAnalyzer中定义的创业板指数代码
            if index_codes is None:
                index_codes = [MarketAnalyzer.INDEX_CODE]
            
            # 将日期转换为字符串格式
            start_date_str = self._date_to_str(start_date) if start_date else None
            end_date_str = self._date_to_str(end_date) if end_date else None
            
            self.logger.info(f"开始处理市场状态: start_date={start_date_str}, end_date={end_date_str}, index_codes={index_codes}")
            
            # 导入xtquant数据接口
            try:
                import xtquant.xtdata as xtdata
                self.logger.info("成功导入xtquant.xtdata模块")
            except ImportError:
                self.logger.error("导入xtquant模块失败，请确保已安装xtquant包")
                return None
            
            # 获取市场数据
            fields = ['open', 'high', 'low', 'close', 'volume', 'preClose']
            
            # 计算历史数据开始日期 (需要至少120天 + 一些缓冲)
            required_history_days = 200
            try:
                # ✅ 修复：使用交易日历回溯，而非自然日
                history_start_date_str = self._get_history_start_date_by_trading_days(
                    start_date_str,
                    trading_days_back=required_history_days
                )

                # 如果交易日历方法失败，回退到自然日回溯（备用方案）
                if history_start_date_str is None:
                    self.logger.warning("交易日历回溯失败，使用自然日回溯（备用方案）")
                    request_start_date = self._str_to_date(start_date_str)
                    history_start_date = request_start_date - timedelta(days=required_history_days)
                    history_start_date_str = self._date_to_str(history_start_date)

            except (ValueError, TypeError):
                 self.logger.warning(f"无法解析请求的开始日期 {start_date_str}，将使用默认历史回溯期")
                 # 备用方案：使用自然日回溯
                 history_start_date_str = (datetime.now() - timedelta(days=required_history_days*2)).strftime('%Y%m%d')

            print(f"start_date_str: {start_date_str}, end_date_str: {end_date_str}, history_start_date_str: {history_start_date_str}")
            # 获取市场数据
            try:
                raw_data = xtdata.get_market_data(
                    field_list=fields,
                    stock_list=index_codes,
                    start_time=history_start_date_str, # 使用计算出的历史开始日期
                    end_time=end_date_str,
                    dividend_type='front'
                )
                
                self.logger.info(f"原始数据获取完成，开始处理...") # 简化日志，raw_data可能很大
                # self.logger.info(f"raw_data: {raw_data}") # 避免打印大量原始数据
                
                if raw_data is None or all(df.empty for df in raw_data.values()):
                    self.logger.error("获取市场数据失败或数据为空")
                    return None
                
                # 转换为DataFrame格式
                market_data = {}
                for field in fields:
                    # 确保返回的数据结构符合预期
                    if field in raw_data and isinstance(raw_data[field], pd.DataFrame):
                        market_data[field] = raw_data[field]
                    else:
                        self.logger.error(f"字段 '{field}' 的数据格式不正确或缺失")
                        return None 
                
                # 检查数据完整性
                if 'close' not in market_data or market_data['close'].empty:
                    self.logger.error("获取到的收盘价数据为空或缺失")
                    return None
                
                # 计算实际数据的时间范围
                actual_first_date = None
                actual_last_date = None
                # 修正：指数是索引，日期是列
                num_indices = len(market_data['close'].index)
                data_shape = market_data['close'].shape
                date_columns = market_data['close'].columns

                if not date_columns.empty:
                    try:
                        # 尝试将列名（日期字符串）转换为日期对象
                        actual_first_date = pd.to_datetime(date_columns[0])
                        actual_last_date = pd.to_datetime(date_columns[-1])
                        actual_first_date_str = actual_first_date.strftime('%Y%m%d')
                        actual_last_date_str = actual_last_date.strftime('%Y%m%d')
                        print(f"actual_first_date_str:{actual_first_date_str} and actual_last_date_str:{actual_last_date_str}")
                    except (ValueError, TypeError):
                        self.logger.warning(f"无法从列名解析实际日期范围: {date_columns[0]} - {date_columns[-1]}")
                        # 使用请求范围作为备选
                        actual_first_date_str = start_date_str
                        actual_last_date_str = end_date_str
                else:
                    self.logger.warning("数据列为空，无法确定实际日期范围")
                    actual_first_date_str = start_date_str
                    actual_last_date_str = end_date_str

                self.logger.info(f"成功获取市场数据: {num_indices} 个指数, 请求范围: {start_date_str}至{end_date_str}")
                if actual_first_date_str and actual_last_date_str:
                     self.logger.info(f"实际数据日期范围: {actual_first_date_str} 至 {actual_last_date_str}")
                self.logger.info(f"数据形状 (指数行数, 日期列数): {data_shape}")
                
                # 数据清洗：仅保留请求日期范围内的数据给后续步骤？
                # 注意：这里需要小心，因为analyze_market_status需要历史数据
                # 也许应该在analyze_market_status内部处理日期过滤
                # filtered_market_data = {}
                # request_start = self._str_to_date(start_date_str)
                # request_end = self._str_to_date(end_date_str)
                # for field, df in market_data.items():
                #     cols_to_keep = [col for col in df.columns if request_start <= pd.to_datetime(col) <= request_end]
                #     filtered_market_data[field] = df[cols_to_keep]
                # return filtered_market_data
                
                return market_data # 返回包含历史数据的完整数据
                
            except Exception as e:
                self.logger.error(f"获取市场数据时出错: {str(e)}")
                traceback.print_exc()
                return None
                
        except Exception as e:
            self.logger.error(f"_get_market_data出错: {str(e)}")
            traceback.print_exc()
            return None

    def process_market_status(self, start_date=None, end_date=None, index_codes=None):
        """
        处理指定日期范围内的市场状态数据
        
        参数:
            start_date: 开始日期，可以是日期对象或字符串
            end_date: 结束日期，可以是日期对象或字符串
            index_codes: 指数代码列表，默认使用创业板指数
            
        返回:
            bool: 处理成功返回True，否则返回False
        """
        try:
            from to.market_status.market_analyzer import MarketAnalyzer
            
            self.logger.info(f"处理市场状态数据: 开始日期={start_date}, 结束日期={end_date}")
            
            # 使用MarketAnalyzer中定义的创业板指数代码
            if index_codes is None:
                index_codes = [MarketAnalyzer.INDEX_CODE]
            
            # 获取日期范围
            date_range = self._get_date_range(start_date, end_date)
            if not date_range:
                self.logger.error("获取日期范围失败")
                return False
            print(f"date:{date_range[-1]}")
            # 获取市场数据
            market_data_dict = self._get_market_data(date_range[0], date_range[-1], index_codes)
            if market_data_dict is None:
                self.logger.error("获取市场数据失败")
                return False

            # --- 新增转置逻辑 ---
            market_data_transposed = {}
            try:
                if not isinstance(market_data_dict, dict) or 'close' not in market_data_dict:
                    self.logger.error("获取的市场数据格式不正确或缺少'close'字段，无法转置")
                    return False
                
                for field, df in market_data_dict.items():
                    # 检查 df 是否为 DataFrame 并且非空
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # 将列（日期字符串）转换为 DatetimeIndex
                        try:
                            df.columns = pd.to_datetime(df.columns)
                        except (ValueError, TypeError) as dt_err:
                             self.logger.warning(f"字段'{field}'的列名无法转换为日期时间: {dt_err}. 跳过转置此字段.")
                             market_data_transposed[field] = pd.DataFrame() # 放空DF
                             continue # 处理下一个字段
                        
                        # 转置，使得日期成为索引，股票代码成为列
                        market_data_transposed[field] = df.T 
                        # 重命名索引和列名
                        market_data_transposed[field].index.name = 'date'
                        market_data_transposed[field].columns.name = 'code'
                    else:
                        self.logger.warning(f"字段 '{field}' 的数据为空或格式不正确，跳过转置")
                        market_data_transposed[field] = pd.DataFrame() # 放空DF
                
                if 'close' not in market_data_transposed or market_data_transposed['close'].empty:
                    self.logger.error("转置后的收盘价数据为空或不存在")
                    return False
                
                self.logger.info(f"成功将获取的数据转置为 (日期 x 指数) 格式，形状: {market_data_transposed['close'].shape}")

            except Exception as e:
                self.logger.error(f"转置市场数据时出错: {str(e)}")
                traceback.print_exc()
                return False
            # --- 结束转置逻辑 ---

            # 进行市场状态处理
            self.logger.info(f"开始串行处理 {self._date_to_str(date_range[0])} 至 {self._date_to_str(date_range[-1])} 的市场状态数据 (使用转置后数据)")
            
            # 计算技术指标 (传入转置后的字典)
            indicators = self._calculate_indicators(market_data_transposed)
            
            # 分析市场状态 (传入转置后的字典)
            analyzer = MarketAnalyzer()
            market_status = analyzer.analyze_market_status(market_data_transposed, indicators)
            # print(f"market_status: {market_status}")
            
            # 保存分析结果 - 只保存用户指定日期范围内的数据
            process_dates = []

            # 将用户指定的日期范围转换为日期对象，用于比较
            user_start_date = self._str_to_date(start_date)
            user_end_date = self._str_to_date(end_date)

            for date in market_status.keys():
                if market_status[date].get('market_type') != 0:  # 只处理有效的市场状态
                    date_str = date.strftime('%Y%m%d') if isinstance(date, datetime) else date
                    current_date = self._str_to_date(date_str)

                    # ✅ 只保存用户指定日期范围内的数据
                    if user_start_date <= current_date <= user_end_date:
                        self._save_market_status(market_status[date], date_str)
                        process_dates.append(date_str)
                    else:
                        # 历史数据只用于计算，不存储
                        self.logger.debug(f"跳过历史数据日期: {date_str} (仅用于计算指标)")
            
            self.logger.info(f"处理完成，市场状态处理成功，共保存 {len(process_dates)} 个交易日的数据")
            print(f"处理完成，市场状态处理成功，共保存 {len(process_dates)} 个交易日的数据")
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理市场状态数据时出错: {str(e)}")
            traceback.print_exc()
            return False
    
    def _calculate_indicators(self, market_data_transposed):
        """计算技术指标
        
        参数:
            market_data_transposed: 转置后的市场数据字典 (日期x指数)
            
        返回:
            dict: 计算好的指标数据 {index_code: {indicator_name: array}}
        """
        indicators = {}
        
        # 检查输入格式
        if not isinstance(market_data_transposed, dict) or 'close' not in market_data_transposed:
            self.logger.error("输入的market_data_transposed格式不正确，需要字典格式且包含'close'字段")
            return {}
        
        # 检查是否有数据
        if market_data_transposed['close'].empty:
            self.logger.warning("输入的转置后收盘价数据为空")
            return {}
        
        # 指数代码现在是列名
        stock_codes = market_data_transposed['close'].columns.tolist()
        # 日期现在是索引
        dates = market_data_transposed['close'].index
        
        self.logger.info(f"开始为 {len(stock_codes)} 个指数计算指标，日期范围: {dates[0]} - {dates[-1]}")

        for index_code in stock_codes:
            self.logger.debug(f"正在计算指数 {index_code} 的指标")
            try:
                # 按列（指数代码）提取数据序列，日期是索引
                close_series = market_data_transposed['close'][index_code]
                # 获取其他字段的数据，如果存在的话
                open_series = market_data_transposed.get('open', pd.DataFrame()).get(index_code, pd.Series(dtype=float))
                high_series = market_data_transposed.get('high', pd.DataFrame()).get(index_code, pd.Series(dtype=float))
                low_series = market_data_transposed.get('low', pd.DataFrame()).get(index_code, pd.Series(dtype=float))
                volume_series = market_data_transposed.get('volume', pd.DataFrame()).get(index_code, pd.Series(dtype=float))
                preclose_series = market_data_transposed.get('preClose', pd.DataFrame()).get(index_code, pd.Series(dtype=float))
                
                # 检查序列是否为空
                if close_series.empty or high_series.empty or low_series.empty:
                    self.logger.warning(f"指数 {index_code} 的必要价格数据 (close/high/low) 为空，跳过指标计算")
                    continue

                # 将 Series 转换为 NumPy 数组进行计算
                close = close_series.values
                # 对于可选数据，如果为空则创建nan数组
                open_price = open_series.values if not open_series.empty else np.full_like(close, np.nan)
                high = high_series.values
                low = low_series.values
                volume = volume_series.values if not volume_series.empty else np.full_like(close, np.nan)
                
                # 确保有足够的数据进行计算（例如，至少需要比最长MA周期多1）
                min_required_data = 121 # 假设最长MA是120
                if len(close) < min_required_data:
                    self.logger.warning(f"指数 {index_code} 的历史数据不足 ({len(close)} < {min_required_data} 天)，跳过指标计算")
                    continue
                    
                self.logger.debug(f"指数 {index_code} 数据长度 {len(close)}，开始计算各项指标")

                # --- 保留原有的指标计算逻辑 --- 
                # 确保这些计算函数能处理 NaN 值，或者在传入前处理
                ma5 = self._calculate_ma(close, 5)
                ma10 = self._calculate_ma(close, 10)
                ma20 = self._calculate_ma(close, 20)
                ma60 = self._calculate_ma(close, 60)
                ma120 = self._calculate_ma(close, 120)
             
                macd, macd_signal, macd_hist = self._calculate_macd(close)
                
                rsi6 = self._calculate_rsi(close, 6)
                rsi12 = self._calculate_rsi(close, 12)
                rsi24 = self._calculate_rsi(close, 24)
                
                # KDJ 和 Bollinger Bands 需要 high, low, close
                kdj_k, kdj_d, kdj_j = self._calculate_kdj(high, low, close)
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)
                
                # 计算布林带宽度和位置 (处理潜在的NaN和零值)
                bb_width = np.full_like(close, np.nan)
                bb_position = np.full_like(close, np.nan)
                valid_bb = ~np.isnan(bb_middle) & (np.abs(bb_middle) > 1e-10)
                bb_width[valid_bb] = (bb_upper[valid_bb] - bb_lower[valid_bb]) / bb_middle[valid_bb]
                valid_width = ~np.isnan(bb_upper) & ~np.isnan(bb_lower) & (np.abs(bb_upper - bb_lower) > 1e-10)
                bb_position[valid_width] = (close[valid_width] - bb_lower[valid_width]) / (bb_upper[valid_width] - bb_lower[valid_width])

                # --- 其他依赖多日窗口的复杂计算 --- 
                # 实际计算支撑位和阻力位，而不是简单使用默认值
                support_level = self._calculate_support_resistance(low, close, mode='support')
                resistance_level = self._calculate_support_resistance(high, close, mode='resistance')

                # 计算盈亏比指标
                profit_loss_ratio_14 = self._calculate_profit_loss_ratio(close, support_level, resistance_level, n=14)
                profit_loss_ratio_30 = self._calculate_profit_loss_ratio(close, support_level, resistance_level, n=30)

                # print(f"profit_loss_ratio_14: {profit_loss_ratio_14}")
                # print(f"profit_loss_ratio_30: {profit_loss_ratio_30}")
            
                
                # 分析价格结构
                price_structure = self._analyze_price_structure(high, low, close)
                
                # 识别市场周期
                market_cycle = self._identify_market_cycle(close, volume, macd)
                
                self.logger.debug(f"指数 {index_code}: 复杂指标计算完成")

                # --- 结束指标计算 ---
                
                # 存储指标结果
                indicators[index_code] = {
                    'dates': dates.to_list(), # 存储日期索引
                    'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60, 'ma120': ma120,
                    'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist,
                    'rsi6': rsi6, 'rsi12': rsi12, 'rsi24': rsi24,
                    'kdj_k': kdj_k, 'kdj_d': kdj_d, 'kdj_j': kdj_j,
                    'bollinger_upper': bb_upper, 'bollinger_middle': bb_middle, 'bollinger_lower': bb_lower,
                    'bollinger_width': bb_width, 'bollinger_position': bb_position,
                    'support_level': support_level,
                    'resistance_level': resistance_level,
                    'price_structure': price_structure,
                    'market_cycle': market_cycle,
                    # 存储原始Series方便后续访问
                    'close_series': close_series,
                    'open_series': open_series,
                    'high_series': high_series,
                    'low_series': low_series,
                    'volume_series': volume_series,
                    'preclose_series': preclose_series,  # ✅ 添加 preClose 系列
                    'profit_loss_ratio_14': profit_loss_ratio_14,
                    'profit_loss_ratio_30': profit_loss_ratio_30
                }
                self.logger.info(f"成功计算指数 {index_code} 的指标")

            except KeyError as e:
                self.logger.error(f"处理指数 {index_code} 时缺少必要字段: {e}。检查 market_data_transposed 字典内容。")
            except Exception as e:
                self.logger.error(f"计算指数 {index_code} 的指标时发生未知错误: {str(e)}")
                traceback.print_exc()
        
        self.logger.info(f"指标计算完成，共处理 {len(indicators)} 个指数")
        return indicators
 
    
    def _calculate_ma(self, data, n):
        """计算移动平均线
        Args:
            data: numpy.array，价格数据
            n: int，移动平均周期
        Returns:
            numpy.array: 移动平均值
        """
        try:
            if len(data) < n:
                return np.array([np.nan])
                
            ma = np.zeros_like(data)
            for i in range(n - 1, len(data)):
                ma[i] = np.mean(data[i-n+1:i+1])
            
            # 处理开始的n-1个值
            ma[:n-1] = ma[n-1]
            
            return ma
            
        except Exception as e:
            self.logger.error(f"计算移动平均线时出错: {str(e)}")
            return np.array([np.nan])
            
    def _calculate_profit_loss_ratio(self, data, support_level, resistance_level, n=14):
        """计算盈亏比 - 基于支撑位和阻力位的风险收益比率分析
        
        专业量化分析方法：
        1. 充分利用完整历史数据，无需设置初始默认值
        2. 使用真实波动范围(ATR)进行风险评估
        3. 基于价格在支撑位和阻力位之间的相对位置调整盈亏比
        4. 应用专业平滑算法提高指标稳定性
        
        Args:
            data: numpy.array，价格数据（收盘价）
            support_level: numpy.array，支撑位
            resistance_level: numpy.array，阻力位
            n: int，计算周期（用于平滑处理）
            
        Returns:
            numpy.array: 专业计算的盈亏比，基于历史数据
        """
        try:
            # 初始化结果数组
            profit_loss_ratio = np.zeros_like(data)
            
            # 数据有效性检查
            if len(data) < 2:
                self.logger.warning("计算盈亏比的数据点数不足")
                return profit_loss_ratio
            
            # 计算真实波动范围(TR)
            tr_array = np.zeros_like(data)
            
            # 利用完整历史数据计算TR，不使用默认初始值
            for i in range(1, len(data)):
                # 真实波动范围定义：当日最高价与最低价的差值、当日最高价与前收的差值、当日最低价与前收的差值，三者中的最大值
                # 由于我们只有收盘价，因此使用相邻两天收盘价的差值作为真实波动范围的近似
                tr_array[i] = abs(data[i] - data[i-1])
            
            # 计算ATR - 使用完整历史数据
            atr = np.zeros_like(data)
            
            # 从第n天开始，前n-1天设为NaN表示无足够数据计算
            for i in range(n, len(data)):
                atr[i] = np.mean(tr_array[i-n+1:i+1])
            
            # 为前n-1天设置合理的ATR值 - 使用可用的最早ATR值
            if n < len(data):
                first_valid_atr = atr[n]
                atr[:n] = first_valid_atr
            
            # 计算每个时间点的盈亏比
            for i in range(len(data)):
                current_price = data[i]
                current_support = support_level[i]
                current_resistance = resistance_level[i]
                current_atr = atr[i]
                
                # 验证支撑位和阻力位的有效性
                # 支撑位必须低于当前价格，阻力位必须高于当前价格
                if current_support >= current_price or current_resistance <= current_price:
                    # 无效的支撑位或阻力位 - 使用基于ATR的动态计算
                    price_range = current_atr * 2
                    current_support = current_price - price_range
                    current_resistance = current_price + price_range
                
                # 计算风险和收益空间
                upside_potential = max(0.001, current_resistance - current_price)
                downside_risk = max(0.001, current_price - current_support)
                
                # 根据专业的风险收益分析计算盈亏比
                # 参考: https://www.investopedia.com/terms/p/profit_loss_ratio.asp
                raw_ratio = upside_potential / downside_risk
                
                # 根据ATR进行风险调整
                atr_factor = current_atr / current_price
                
                # 价格在支撑位和阻力位间的相对位置
                if current_resistance > current_support:  # 避免除以零
                    price_position = (current_price - current_support) / (current_resistance - current_support)
                    
                    # 动态调整盈亏比 - 基于价格位置的专业风险调整
                    if price_position < 0.25:  # 接近支撑位
                        position_factor = 1.2  # 上涨空间大，提高盈亏比
                    elif price_position > 0.75:  # 接近阻力位
                        position_factor = 0.8  # 下跌风险大，降低盈亏比
                    else:  # 位于中间区域
                        position_factor = 1.0
                else:
                    # 支撑位和阻力位重合的极端情况
                    position_factor = 1.0
                
                # 将ATR和位置因素纳入计算
                adjusted_ratio = raw_ratio * position_factor * (1 + atr_factor)
                
                # 根据专业交易标准限制盈亏比范围
                profit_loss_ratio[i] = min(max(adjusted_ratio, 0.2), 5.0)
            
            # 使用专业的移动平均平滑处理
            if n > 1:
                # 保存原始计算结果
                raw_result = np.copy(profit_loss_ratio)
                
                # 应用加权移动平均平滑
                for i in range(n, len(profit_loss_ratio)):
                    # 使用线性加权，最近的数据权重更大
                    weights = np.linspace(1, 2, n)
                    weighted_sum = np.sum(raw_result[i-n+1:i+1] * weights)
                    profit_loss_ratio[i] = weighted_sum / np.sum(weights)
            
            # 确保前n日的盈亏比也有合理的值 - 使用后向填充
            if n < len(profit_loss_ratio):
                # 计算最早n个有效值的平均值
                early_avg = np.mean(profit_loss_ratio[n:min(n*2, len(profit_loss_ratio))])
                # 应用到前n个值
                profit_loss_ratio[:n] = early_avg
            
            return profit_loss_ratio
            
        except Exception as e:
            self.logger.error(f"计算盈亏比时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return np.ones_like(data)  # 出错时返回中性值
    
    def _calculate_macd(self, data):
        """计算MACD指标
        Args:
            data: numpy.array，价格数据
        Returns:
            tuple: (MACD线, 信号线, 柱状图)
        """
        try:
            if len(data) < 26:
                return np.array([0.0]), np.array([0.0]), np.array([0.0])
                
            # 计算EMA
            ema12 = np.zeros_like(data)
            ema26 = np.zeros_like(data)
            
            # 初始化
            ema12[0] = data[0]
            ema26[0] = data[0]
            
            # EMA权重
            k1 = 2.0 / (12 + 1)
            k2 = 2.0 / (26 + 1)
            
            # 计算EMA
            for i in range(1, len(data)):
                ema12[i] = (data[i] - ema12[i-1]) * k1 + ema12[i-1]
                ema26[i] = (data[i] - ema26[i-1]) * k2 + ema26[i-1]
            
            # 计算DIF（快线-慢线，即MACD线）
            macd_line = ema12 - ema26
            
            # 计算DEA (9日EMA，即信号线)
            signal_line = np.zeros_like(macd_line)
            signal_line[0] = macd_line[0]
            k3 = 2.0 / (9 + 1)
            
            for i in range(1, len(macd_line)):
                signal_line[i] = (macd_line[i] - signal_line[i-1]) * k3 + signal_line[i-1]
            
            # 计算MACD柱状图
            histogram = 2 * (macd_line - signal_line)
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            self.logger.error(f"计算MACD时出错: {str(e)}")
            return np.array([0.0]), np.array([0.0]), np.array([0.0])
    
    def _calculate_rsi(self, data, n=14):
        """计算RSI指标
        Args:
            data: numpy.array，价格数据
            n: int，RSI周期
        Returns:
            numpy.array: RSI值
        """
        try:
            if len(data) < n + 1:
                return np.array([50.0])  # 返回中性值
                
            # 计算价格变化
            deltas = np.diff(data)
            
            # 分离上涨和下跌
            up = np.where(deltas > 0, deltas, 0)
            down = np.where(deltas < 0, -deltas, 0)
            
            # 计算移动平均
            roll_up = np.zeros_like(data)
            roll_down = np.zeros_like(data)
            
            roll_up[n] = np.mean(up[:n])
            roll_down[n] = np.mean(down[:n])
            
            for i in range(n + 1, len(data)):
                roll_up[i] = (roll_up[i-1] * (n-1) + up[i-1]) / n
                roll_down[i] = (roll_down[i-1] * (n-1) + down[i-1]) / n
            
            # 计算RS和RSI
            rs = roll_up / roll_down
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            # 处理无穷大和NaN
            rsi = np.nan_to_num(rsi, nan=50.0, posinf=100.0, neginf=0.0)
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"计算RSI时出错: {str(e)}")
            return np.array([50.0])  # 返回中性值
    
    def _calculate_kdj(self, high, low, close, n=9, m1=3, m2=3):
        """计算KDJ指标"""
        return self.analyzer._calculate_kdj(high, low, close, n, m1, m2)
    
    def _calculate_bollinger_bands(self, data, n=20, k=2):
        """计算布林带
        Args:
            data: numpy.array，价格数据
            n: int，移动平均周期
            k: int，标准差倍数
        Returns:
            tuple: (上轨, 中轨, 下轨)
        """
        try:
            if len(data) < n:
                return np.array([data[0]]), np.array([data[0]]), np.array([data[0]])
                
            # 计算移动平均线
            ma = self._calculate_ma(data, n)
            
            # 计算标准差
            std = np.zeros_like(data)
            for i in range(n - 1, len(data)):
                std[i] = np.std(data[i-n+1:i+1])
            
            # 计算布林带
            upper = ma + k * std
            lower = ma - k * std
            
            # 处理开始的n-1个值
            upper[:n-1] = upper[n-1]
            lower[:n-1] = lower[n-1]
            
            return upper, ma, lower
            
        except Exception as e:
            self.logger.error(f"计算布林带时出错: {str(e)}")
            return np.array([data[0]]), np.array([data[0]]), np.array([data[0]])
            
    def _calculate_position_score(self, data, indicators):
        """计算位置得分
        Args:
            data: pd.DataFrame，市场数据
            indicators: dict，技术指标
        Returns:
            float: 位置得分
        """
        try:
            if data.empty or not indicators:
                return 0.0
                
            # 获取布林带指标
            bb_upper = indicators.get('bollinger_upper', [])
            bb_lower = indicators.get('bollinger_lower', [])
            bb_middle = indicators.get('bollinger_middle', [])
            
            if len(bb_upper) == 0 or len(bb_lower) == 0 or len(bb_middle) == 0:
                return 0.0
                
            # 计算当前价格在布林带中的位置
            current_price = data['close'].iloc[-1]
            band_width = bb_upper[-1] - bb_lower[-1]
            if band_width == 0:
                return 0.5  # 返回中性值
                
            position = (current_price - bb_lower[-1]) / band_width
            
            # 归一化得分
            return max(min(position, 1.0), 0.0)
            
        except Exception as e:
            self.logger.error(f"计算位置得分时出错: {str(e)}")
            return 0.0
            
    def _determine_market_type(self, total_score):
        """根据总分确定市场类型
        Args:
            total_score: float，总分
        Returns:
            str: 市场类型
        """
        if np.isnan(total_score):
            return "中性盘整"
            
        if total_score >= 0.8:
            return "强势上涨"
        elif total_score >= 0.6:
            return "温和上涨"
        elif total_score >= 0.4:
            return "中性盘整"
        elif total_score >= 0.2:
            return "温和下跌"
        else:
            return "强势下跌"
    
    def _calculate_support_resistance(self, price_data, close_data, mode='support'):
        """计算支撑位或阻力位"""
        return self.analyzer._calculate_support_resistance(price_data, close_data, mode)
    
    def _analyze_price_structure(self, high, low, close):
        """分析价格结构"""
        return self.analyzer._analyze_price_structure(high, low, close)
    
    def _identify_market_cycle(self, close, volume, macd):
        """识别市场周期"""
        return self.analyzer._identify_market_cycle(close, volume, macd)
    
    def _save_market_status(self, market_status, save_date=None):
        """保存市场状态数据
        Args:
            market_status: dict，市场状态数据
            save_date: str，要保存的日期，格式YYYYMMDD，默认为当前日期
        Returns:
            bool: 保存是否成功
        """
        try:
            if market_status is None:
                self.logger.error("市场状态数据为空，无法保存")
                return False
                
            # 获取保存日期
            if save_date is None:
                # 仅在未指定日期时使用当前日期
                save_date = datetime.now().strftime('%Y%m%d')
                self.logger.info(f"未指定保存日期，使用当前日期: {save_date}")
            
            # self.logger.info(f"保存市场状态数据到日期: {save_date}")

            # print(f"market_status: {market_status}")
            
            # 构建要保存的数据结构
            save_data = {
                'market_type': market_status.get('market_type', '中性盘整'),
                'market_description': market_status.get('market_type', '中性盘整'),
                'market_cycle': market_status.get('market_cycle', '蓄势'),
                'cycle_description': market_status.get('market_cycle', '蓄势'),
                'trend_strength':  round(float(market_status.get('trend_strength', 0.0)), 2),  # 保持trend_strength的原始值
                'signals': {
                    'trading_advice': market_status.get('trading_advice', '观望为主，等待明确信号'),
                    'risk_level': market_status.get('risk_level', 'low'),
                    'strength': market_status.get('signal_strength', 0)
                },
                'individual_status': {}
            }

            # print(f"save_data: {save_data}")
            
            # 处理各指数的状态
            for index_code, status in market_status.get('individual_status', {}).items():
                # 获取market_status_info中的数据
                market_status_info = status.get('market_status_info', {})
                technical_data = status.get('technical_data', {})
                market_data = status.get('market_data', {})
                
                # 构建保存结构
                save_data['individual_status'][index_code] = {
                    # 主要评分指标
                    'trend_score': round(float(status.get('trend_score', 0.0)), 2),
                    'position_score': round(float(status.get('position_score', 0.0)), 2),
                    'volume_score': round(float(status.get('volume_score', 0.0)), 2),
                    'momentum_score': round(float(status.get('momentum_score', 0.0)), 2),
                    'volatility_score': round(float(status.get('volatility_score', 0.0)), 2),
                    'structure_score': round(float(status.get('structure_score', 0.0)), 2),
                    'cycle_score': round(float(status.get('cycle_score', 0.0)), 2),
                    'total_score': round(float(status.get('total_score', 0.0)), 2),
                    
                    # 市场状态信息
                    'market_status_info': {
                        'type_code': market_status_info.get('type_code', 32),
                        'type_desc': market_status_info.get('type_desc', '中性盘整'),
                        'cycle_code': market_status_info.get('cycle_code', 0),
                        'cycle_desc': market_status_info.get('cycle_desc', '未知周期')
                    },
                    
                    # 市场描述 (为了兼容性保留)
                    'market_description': market_status_info.get('type_desc', '中性盘整'),
                    'cycle_description': market_status_info.get('cycle_desc', '未知周期'),
                    
                    # 技术数据
                    'technical_data': {
                        'close': self._round_float(technical_data.get('close')),
                        'prev_close': self._round_float(technical_data.get('prev_close')),
                        'ma5': self._round_float(technical_data.get('ma5')),
                        'ma10': self._round_float(technical_data.get('ma10')),
                        'ma20': self._round_float(technical_data.get('ma20')),
                        'ma60': self._round_float(technical_data.get('ma60')),
                        'rsi6': self._round_float(technical_data.get('rsi6')),
                        'rsi12': self._round_float(technical_data.get('rsi12')),
                        'macd': self._round_float(technical_data.get('macd')),
                        'macd_signal': self._round_float(technical_data.get('macd_signal')),
                        'macd_hist': self._round_float(technical_data.get('macd_hist')),
                        'kdj_k': self._round_float(technical_data.get('kdj_k')),
                        'kdj_d': self._round_float(technical_data.get('kdj_d')),
                        'bb_width': self._round_float(technical_data.get('bb_width')),
                        'bb_position': self._round_float(technical_data.get('bb_position'))
                    },
                    
                    # 市场数据
                    'market_data': {
                        'close': self._round_float(market_data.get('close')),
                        'open': self._round_float(market_data.get('open')),
                        'high': self._round_float(market_data.get('high')),
                        'low': self._round_float(market_data.get('low')),
                        'volume': self._round_float(market_data.get('volume')),
                        'daily_change_pct': self._round_float(market_data.get('daily_change_pct'))
                    },
                    
                    # 单独的支撑位和阻力位
                    'support': self._round_float(technical_data.get('support')),
                    'resistance': self._round_float(technical_data.get('resistance')),
                    'profit_loss_ratio_14': self._round_float(technical_data.get('profit_loss_ratio_14')),
                    'profit_loss_ratio_30': self._round_float(technical_data.get('profit_loss_ratio_30')),
                    'change_percent': self._round_float(market_data.get('daily_change_pct', 0.0))
                }
            
            # 保存数据
            self.market_status_data[save_date] = save_data
            
            # 保存到文件
            if not self._save_market_status_data():
                self.logger.error("保存市场状态数据到文件失败")
                return False
            
            # 更新元数据
            if save_date not in self.metadata["available_dates"]:
                self.metadata["available_dates"].append(save_date)
                self.metadata["available_dates"].sort()
                # 使用保存日期作为最后更新时间，而不是使用当前系统时间
                formatted_date = f"{save_date[:4]}-{save_date[4:6]}-{save_date[6:]} 00:00:00"
                self.metadata["last_update"] = formatted_date
                self._save_metadata()
                self.logger.info(f"成功新增 {save_date} 的市场状态数据")
            else:
                # 即使是已存在的日期，也需要更新最后更新时间
                formatted_date = f"{save_date[:4]}-{save_date[4:6]}-{save_date[6:]} 00:00:00"
                self.metadata["last_update"] = formatted_date
                self._save_metadata()
                self.logger.info(f"成功更新 {save_date} 的市场状态数据")
            
            self.logger.info(f"成功保存 {save_date} 的市场状态数据")
            return True
            
        except Exception as e:
            self.logger.error(f"保存市场状态数据时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _round_float(self, value, digits=2):
        """将浮点数四舍五入到指定小数位数
        
        Args:
            value: 要四舍五入的值
            digits: 小数位数，默认为2
            
        Returns:
            float: 四舍五入后的值，如果输入不是数字则返回原值
        """
        try:
            if value is None:
                return None
            return round(float(value), digits)
        except (TypeError, ValueError):
            return value
    
    def get_market_status(self, query_date=None):
        """获取市场状态数据
        Args:
            query_date: str，查询日期，格式YYYYMMDD
        Returns:
            dict: 市场状态数据，修改为适合模型训练的结构
        """
        try:
            if query_date is None:
                query_date = self.default_end_date
                
            # 检查数据是否存在
            if query_date not in self.market_status_data:
                self.logger.warning(f"未找到日期 {query_date} 的市场状态数据")
                return None
            
            # 获取原始市场状态数据
            market_status = self.market_status_data[query_date]
            
            # 从market_analyzer引入索引代码
            from to.market_status.market_analyzer import MarketAnalyzer
            
            # 只保留创业板指数数据
            if 'individual_status' in market_status and '899050.BJ' in market_status['individual_status']:
                # 删除北证50指数数据
                del market_status['individual_status']['899050.BJ']
                
            # 设置创业板指数数据
            if 'individual_status' in market_status and MarketAnalyzer.INDEX_CODE in market_status['individual_status']:
                cyb_data = market_status['individual_status'][MarketAnalyzer.INDEX_CODE]
                # 更新市场状态数据，只基于创业板指数
                market_status['market_type'] = cyb_data.get('market_description', '')
                market_status['market_description'] = cyb_data.get('market_description', '')
                market_status['market_cycle'] = cyb_data.get('cycle_description', '')
                market_status['cycle_description'] = cyb_data.get('cycle_description', '')
                market_status['trend_strength'] = cyb_data.get('total_score', 0)  # 使用实际的total_score作为trend_strength
                
                # 添加支撑位、压力位和涨跌幅数据
                market_status['support_level'] = cyb_data.get('support', 0)
                market_status['resistance_level'] = cyb_data.get('resistance', 0)
                market_status['change_percent'] = cyb_data.get('change_percent', 0)
                
                # 确保得分指标都存在，并提供默认值
                market_status['trend_score'] = cyb_data.get('trend_score', 0)  # 趋势得分：0-5，值越大表示上升趋势越强
                market_status['position_score'] = cyb_data.get('position_score', 0)  # 位置得分：0-5，值越大表示位置越接近上轨
                market_status['volume_score'] = cyb_data.get('volume_score', 0)  # 量能得分：0-5，值越大表示成交量越活跃
                market_status['momentum_score'] = cyb_data.get('momentum_score', 0)  # 动能得分：0-5，值越大表示上升动能越强
                market_status['volatility_score'] = cyb_data.get('volatility_score', 0)  # 波动得分：0-5，值越大表示波动越剧烈
                market_status['structure_score'] = cyb_data.get('structure_score', 0)  # 结构得分：0-5，值越大表示上涨结构越健康
                market_status['cycle_score'] = cyb_data.get('cycle_score', 0)  # 周期得分：0-5，值越大表示处于上升周期
                market_status['total_score'] = cyb_data.get('total_score', 0)  # 总分：0-5，值越大表示市场整体向好
                
            return market_status
            
        except Exception as e:
            self.logger.error(f"获取市场状态数据时出错: {str(e)}")
            return None
    
    def get_all_available_dates(self):
        """获取所有可用的日期列表"""
        return sorted(self.metadata["available_dates"])
    
    def get_latest_market_status(self):
        """获取最新的市场状态数据"""
        available_dates = self.get_all_available_dates()
        if available_dates:
            latest_date = available_dates[-1]
            return self.get_market_status(query_date=latest_date)
        return None
    
    def generate_market_status_report(self, start_date=None, end_date=None, output_file=None):
        """
        生成市场状态报告
        
        参数:
            start_date: 报告开始日期
            end_date: 报告结束日期
            output_file: 输出文件路径
            
        返回:
            str: 报告内容
        """
        # 获取日期范围
        if start_date is None or end_date is None:
            available_dates = self.get_all_available_dates()
            if not available_dates:
                return "没有可用的市场状态数据"
            
            if start_date is None:
                start_date = available_dates[0]
            if end_date is None:
                end_date = available_dates[-1]
        
        # 获取市场状态数据 - 修复此处的调用
        market_status = self.get_market_status(query_date=end_date)
        
        if market_status is None:
            return f"未找到 {end_date} 的市场状态数据"
            
        # 转换为DataFrame以便进行统计分析
        market_status_df = pd.DataFrame([market_status])  # 只处理单个日期的数据
        
        if market_status_df.empty:
            return f"未找到 {start_date} 至 {end_date} 的市场状态数据"
        
        # 生成报告
        report = f"# 市场状态报告 ({start_date} - {end_date})\n\n"
        
        # 市场状态统计
        market_type_counts = market_status_df["market_type"].value_counts().sort_index()
        market_cycle_counts = market_status_df["market_cycle"].value_counts().sort_index()
        
        report += "## 市场状态统计\n\n"
        report += "### 市场类型分布\n\n"
        report += "| 市场类型 | 描述 | 天数 | 占比 |\n"
        report += "| --- | --- | --- | --- |\n"
        
        for market_type, count in market_type_counts.items():
            description = self.analyzer.MARKET_STATUS.get(market_type, "未知状态")
            percentage = count / len(market_status_df) * 100
            report += f"| {market_type} | {description} | {count} | {percentage:.2f}% |\n"
        
        report += "\n### 市场周期分布\n\n"
        report += "| 市场周期 | 描述 | 天数 | 占比 |\n"
        report += "| --- | --- | --- | --- |\n"
        
        for cycle, count in market_cycle_counts.items():
            description = self.analyzer.MARKET_CYCLE.get(cycle, "未知周期")
            percentage = count / len(market_status_df) * 100
            report += f"| {cycle} | {description} | {count} | {percentage:.2f}% |\n"
        
        # 市场评分统计
        report += "\n## 市场评分统计\n\n"
        
        score_fields = [
            "trend_score", "position_score", "volume_score", 
            "momentum_score", "volatility_score", "structure_score", 
            "cycle_score", "total_score"
        ]
        
        stats = {}
        for field in score_fields:
            stats[field] = {
                "mean": market_status_df[field].mean(),
                "std": market_status_df[field].std(),
                "min": market_status_df[field].min(),
                "max": market_status_df[field].max()
            }
        
        report += "| 评分指标 | 平均值 | 标准差 | 最小值 | 最大值 |\n"
        report += "| --- | --- | --- | --- | --- |\n"
        
        for field, values in stats.items():
            report += f"| {field} | {values['mean']:.2f} | {values['std']:.2f} | {values['min']:.2f} | {values['max']:.2f} |\n"
        
        # 最近状态摘要
        recent_days = min(7, len(market_status_df))
        recent_status = market_status_df.tail(recent_days)
        
        report += f"\n## 最近 {recent_days} 天市场状态\n\n"
        report += "| 日期 | 市场类型 | 描述 | 周期 | 趋势得分 | 总分 | 交易建议 |\n"
        report += "| --- | --- | --- | --- | --- | --- | --- |\n"
        
        for _, row in recent_status.iterrows():
            date = row["date"]
            market_type = row["market_type"]
            description = row["market_description"]
            cycle = row["cycle_description"]
            trend_score = row["trend_score"]
            total_score = row["total_score"]
            advice = row["trading_advice"]
            
            report += f"| {date} | {market_type} | {description} | {cycle} | {trend_score:.2f} | {total_score:.2f} | {advice} |\n"
        
        # 保存报告
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
        
        return report
    

    def test_file_paths(self):
        """测试文件路径处理"""
        self.logger.info("开始测试文件路径处理...")
        
        # 获取当前文件的绝对路径
        current_file = os.path.normpath(os.path.abspath(__file__))
        self.logger.info(f"当前文件路径: {current_file}")
        
        # 测试数据目录
        self.logger.info(f"数据目录: {self.data_dir}")
        self.logger.info(f"日线数据目录: {self.daily_status_dir}")
        self.logger.info(f"元数据目录: {self.metadata_dir}")
        
        # 测试文件路径生成
        test_date = datetime.now()
        file_path = self._get_file_path(test_date)
        self.logger.info(f"测试日期 {test_date.strftime('%Y%m%d')} 的文件路径: {file_path}")
        
        # 验证目录是否存在和权限
        self.logger.info("验证目录:")
        self.logger.info(f"日线数据目录存在: {os.path.exists(self.daily_status_dir)}")
        self.logger.info(f"日线数据目录可写: {os.access(self.daily_status_dir, os.W_OK)}")
        self.logger.info(f"元数据目录存在: {os.path.exists(self.metadata_dir)}")
        self.logger.info(f"元数据目录可写: {os.access(self.metadata_dir, os.W_OK)}")
        
        # 测试目录创建
        test_date_str = test_date.strftime('%Y%m')
        test_dir = os.path.normpath(os.path.join(self.daily_status_dir, test_date_str))
        os.makedirs(test_dir, exist_ok=True)
        self.logger.info(f"测试目录创建: {test_dir}")
        self.logger.info(f"测试目录存在: {os.path.exists(test_dir)}")
        self.logger.info(f"测试目录可写: {os.access(test_dir, os.W_OK)}")
        
        return {
            'current_file': current_file,
            'data_dir': self.data_dir,
            'daily_status_dir': self.daily_status_dir,
            'metadata_dir': self.metadata_dir,
            'test_file_path': file_path,
            'test_dir': test_dir,
            'permissions': {
                'daily_dir_writable': os.access(self.daily_status_dir, os.W_OK),
                'metadata_dir_writable': os.access(self.metadata_dir, os.W_OK),
                'test_dir_writable': os.access(test_dir, os.W_OK)
            }
        }

    def _validate_directories(self):
        """验证目录权限和可访问性"""
        try:
            # 验证日线数据目录
            if not os.path.exists(self.daily_status_dir):
                self.logger.error(f"日线数据目录不存在: {self.daily_status_dir}")
                raise IOError(f"日线数据目录不存在: {self.daily_status_dir}")
            
            # 验证元数据目录
            if not os.path.exists(self.metadata_dir):
                self.logger.error(f"元数据目录不存在: {self.metadata_dir}")
                raise IOError(f"元数据目录不存在: {self.metadata_dir}")
            
            # 测试写入权限
            test_file = os.path.join(self.daily_status_dir, ".test_write")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                self.logger.error(f"目录写入权限测试失败: {str(e)}")
                raise IOError(f"目录写入权限测试失败: {str(e)}")
            
            self.logger.info("目录验证通过")
            return True
        except Exception as e:
            self.logger.error(f"目录验证失败: {str(e)}")
            raise

    # def analyze_market_status(self, market_data):
    #     """分析市场状态
    #     Args:
    #         market_data: DataFrame，包含市场数据
    #     Returns:
    #         dict: 以日期为键的市场状态分析结果字典
    #     """
    #     try:
    #         if market_data is None or market_data.empty:
    #             self.logger.error("市场数据为空，无法进行分析")
    #             return None
                
    #         # 检查数据完整性
    #         required_fields = ['open', 'high', 'low', 'close', 'volume']
    #         for field in required_fields:
    #             if field not in market_data.columns:
    #                 self.logger.error(f"缺少必要的数据字段: {field}")
    #                 return None
            
    #         # 计算技术指标
    #         indicators = self._calculate_indicators(market_data)
    #         if not indicators:
    #             self.logger.error("计算技术指标失败")
    #             return None
            
    #         # 获取数据中的所有日期
    #         all_dates = sorted(market_data.index.unique())
    #         self.logger.info(f"开始分析 {len(all_dates)} 个交易日的市场状态")
            
    #         # 创建以日期为键的结果字典
    #         date_results = {}
            
    #         # 为每个日期分析市场状态
    #         for current_date in all_dates:
    #             # 过滤出当前日期的数据
    #             current_data = market_data.loc[current_date]
                
    #             # 如果当前日期只有一条记录，需要转换为DataFrame
    #             if not isinstance(current_data, pd.DataFrame):
    #                 current_data = pd.DataFrame([current_data])
                
    #             # 确保有足够的数据进行分析
    #             if len(current_data) == 0 or 'code' not in current_data.columns:
    #                 continue
                
    #             # 初始化当前日期的结果字典
    #             result = {
    #                 'market_type': '中性盘整',  # 默认市场类型
    #                 'market_cycle': '蓄势',     # 默认市场周期
    #                 'trend_strength': 0.0,      # 趋势强度
    #                 'trading_advice': '观望为主，等待明确信号',
    #                 'risk_level': 'low',
    #                 'signal_strength': 0,
    #                 'individual_status': {}
    #             }
                
    #             # 分析每个指数的状态
    #             for index_code in current_data['code'].unique():
    #                 # 获取当前指数的历史数据用于计算指标
    #                 index_historical_data = market_data[market_data['code'] == index_code]
    #                 index_historical_data = index_historical_data.loc[:current_date]  # 只使用当前日期及之前的数据
                    
    #                 if len(index_historical_data) < 20:  # 确保有足够的历史数据
    #                     self.logger.warning(f"日期 {current_date} 的指数 {index_code} 历史数据不足，跳过分析")
    #                     continue
                    
    #                 # 获取当前日期的指数数据
    #                 index_current_data = current_data[current_data['code'] == index_code]
                    
    #                 if len(index_current_data) == 0:
    #                     continue
                    
    #                 # 计算各项得分
    #                 trend_score = self._calculate_trend_score(index_historical_data, indicators.get(index_code, {}))
    #                 position_score = self._calculate_position_score(index_historical_data, indicators.get(index_code, {}))
    #                 volume_score = self._calculate_volume_score(index_historical_data)
    #                 momentum_score = self._calculate_momentum_score(index_historical_data, indicators.get(index_code, {}))
                    
    #                 # 计算总分
    #                 total_score = (trend_score + position_score + volume_score + momentum_score) / 4
                    
    #                 # 确定市场类型
    #                 market_type = self._determine_market_type(total_score)
                    
    #                 # 获取技术指标值（使用当前日期的值）
    #                 close_value = None
    #                 ma5_value = None
    #                 ma20_value = None
    #                 rsi6_value = None
    #                 macd_value = None
                    
    #                 try:
    #                     # 直接从当前数据获取收盘价
    #                     if not index_current_data.empty:
    #                         close_value = float(index_current_data['close'].values[0])
                        
    #                     # 从技术指标中获取MA5、MA20、RSI6、MACD
    #                     if index_code in indicators:
    #                         index_indicators = indicators[index_code]
                            
    #                         # 如果存在ma5数据并且长度足够，获取最新值
    #                         if 'ma5' in index_indicators and len(index_indicators['ma5']) > 0:
    #                             ma5_value = float(index_indicators['ma5'][-1])
                                
    #                         # 同理获取其他指标
    #                         if 'ma20' in index_indicators and len(index_indicators['ma20']) > 0:
    #                             ma20_value = float(index_indicators['ma20'][-1])
                                
    #                         if 'rsi6' in index_indicators and len(index_indicators['rsi6']) > 0:
    #                             rsi6_value = float(index_indicators['rsi6'][-1])
                                
    #                         if 'macd' in index_indicators and len(index_indicators['macd']) > 0:
    #                             macd_value = float(index_indicators['macd'][-1])
    #                 except Exception as e:
    #                     self.logger.warning(f"获取技术指标值时出错: {e}")
                    
    #                 # 构建技术指标字典
    #                 tech_indicators = {
    #                     '收盘价': close_value,
    #                     'MA5': ma5_value,
    #                     'MA20': ma20_value,
    #                     'RSI6': rsi6_value,
    #                     'MACD': macd_value
    #                 }

    #                 print(f"技术指标: {tech_indicators}")
                    
    #                 # 获取支撑位和阻力位
    #                 support = indicators.get(index_code, {}).get('support_level', [])[-len(index_historical_data):][0] if indicators.get(index_code, {}).get('support_level') is not None and len(indicators.get(index_code, {}).get('support_level', [])) > len(index_historical_data) else None
    #                 resistance = indicators.get(index_code, {}).get('resistance_level', [])[-len(index_historical_data):][0] if indicators.get(index_code, {}).get('resistance_level') is not None and len(indicators.get(index_code, {}).get('resistance_level', [])) > len(index_historical_data) else None
    #                 profit_loss_ratio_14 = indicators.get(index_code, {}).get('profit_loss_ratio_14', [])[-len(index_historical_data):][0] if indicators.get(index_code, {}).get('profit_loss_ratio_14') is not None and len(indicators.get(index_code, {}).get('profit_loss_ratio_14', [])) > len(index_historical_data) else None
    #                 profit_loss_ratio_30 = indicators.get(index_code, {}).get('profit_loss_ratio_30', [])[-len(index_historical_data):][0] if indicators.get(index_code, {}).get('profit_loss_ratio_30') is not None and len(indicators.get(index_code, {}).get('profit_loss_ratio_30', [])) > len(index_historical_data) else None
                    
    #                 # 计算涨跌幅
    #                 change_percent = 0
    #                 if len(index_historical_data) >= 2:
    #                     prev_close = index_historical_data['close'].values[-2] if len(index_historical_data) > 1 else index_historical_data['close'].values[-1]
    #                     curr_close = index_historical_data['close'].values[-1]
    #                     change_percent = 100 * (curr_close - prev_close) / prev_close if prev_close != 0 else 0
                    
    #                 # 保存指数状态
    #                 result['individual_status'][index_code] = {
    #                     'market_type': market_type,
    #                     'trend_score': trend_score,
    #                     'position_score': position_score,
    #                     'volume_score': volume_score,
    #                     'momentum_score': momentum_score,
    #                     'total_score': total_score,
    #                     'technical_indicators': tech_indicators,
    #                     'support': support,
    #                     'resistance': resistance,
    #                     'profit_loss_ratio_14': profit_loss_ratio_14,
    #                     'profit_loss_ratio_30': profit_loss_ratio_30,
    #                     'change_percent': change_percent
    #                 }
                    
    #                 # 更新整体市场状态 - 修复trend_strength计算逻辑
    #                 # 使用实际total_score值而不是负值
    #                 if total_score > result['trend_strength']:
    #                     result['trend_strength'] = total_score
                
    #             # 如果没有分析任何指数，则跳过当前日期
    #             if not result['individual_status']:
    #                 continue
                
    #             # 确定整体市场类型和建议
    #             result['market_type'] = self._determine_composite_market_type(result['individual_status'])
    #             result['trading_advice'] = self._generate_trading_advice(result)
    #             result['market_cycle'] = self._determine_market_cycle(result['individual_status'])
    #             result['signal_strength'] = self._calculate_signal_strength(result)
                
    #             # 将当前日期的分析结果添加到结果字典
    #             date_results[current_date] = result
            
    #         self.logger.info(f"成功分析 {len(date_results)} 个交易日的市场状态")
    #         return date_results
            
    #     except Exception as e:
    #         self.logger.error(f"分析市场状态时出错: {str(e)}")
    #         import traceback
    #         self.logger.error(traceback.format_exc())
    #         return None
    
    def _calculate_trend_score(self, data, indicators):
        """计算趋势得分"""
        try:
            if data.empty or not indicators:
                return 0.0
                
            # 计算趋势相关指标
            close_prices = data['close'].values
            ma5 = indicators.get('ma5', [])
            ma20 = indicators.get('ma20', [])
            
            if len(close_prices) < 2 or len(ma5) == 0 or len(ma20) == 0:
                return 0.0
                
            # 计算价格趋势
            price_trend = (close_prices[-1] - close_prices[0]) / close_prices[0]
            
            # 计算均线趋势
            ma_trend = 0.0
            if len(ma5) > 1 and len(ma20) > 1:
                ma5_trend = (ma5[-1] - ma5[0]) / ma5[0]
                ma20_trend = (ma20[-1] - ma20[0]) / ma20[0]
                ma_trend = (ma5_trend + ma20_trend) / 2
            
            # 综合得分
            trend_score = (price_trend + ma_trend) / 2
            return max(min(trend_score, 1.0), 0.0)  # 归一化到0-1之间
            
        except Exception as e:
            self.logger.error(f"计算趋势得分时出错: {str(e)}")
            return 0.0
            
    def _calculate_volume_score(self, data):
        """计算量能得分"""
        try:
            if data.empty:
                return 0.0
                
            # 获取最近的成交量数据
            volumes = data['volume'].values
            if len(volumes) < 5:
                return 0.0
                
            # 计算最近5日平均成交量
            recent_avg = np.mean(volumes[-5:])
            if recent_avg == 0:
                return 0.0
                
            # 计算相对于前期的量能变化
            if len(volumes) >= 20:
                previous_avg = np.mean(volumes[-20:-5])
                volume_change = (recent_avg - previous_avg) / previous_avg
            else:
                volume_change = 0.0
                
            # 归一化得分
            return max(min((volume_change + 1) / 2, 1.0), 0.0)
            
        except Exception as e:
            self.logger.error(f"计算量能得分时出错: {str(e)}")
            return 0.0
            
    def _calculate_momentum_score(self, data, indicators):
        """计算动能得分"""
        try:
            if data.empty or not indicators:
                return 0.0
                
            # 获取RSI和MACD指标
            rsi = indicators.get('rsi6', [])
            macd = indicators.get('macd', [])
            
            if len(rsi) == 0 or len(macd) == 0:
                return 0.0
                
            # 计算RSI得分
            rsi_score = (rsi[-1] - 50) / 50  # 归一化到[-1, 1]
            
            # 计算MACD得分
            if len(macd) >= 2:
                macd_score = 1 if macd[-1] > macd[-2] else -1
            else:
                macd_score = 0
                
            # 综合得分
            momentum_score = (rsi_score + macd_score) / 2
            
            # 归一化到[0, 1]
            return max(min((momentum_score + 1) / 2, 1.0), 0.0)
            
        except Exception as e:
            self.logger.error(f"计算动能得分时出错: {str(e)}")
            return 0.0
            
    def _determine_composite_market_type(self, index_status):
        """确定综合市场类型"""
        # 计算各指数的权重
        weights = {
            '399006.SZ': 0.9,  # 创业板指
            '899050.BJ': 0.1   # 北证50
        }
        
        # 计算加权得分
        weighted_score = 0.0
        for index_code, status in index_status.items():
            weight = weights.get(index_code, 0.25)  # 默认权重0.25
            weighted_score += status['total_score'] * weight
            
        # 根据加权得分确定市场类型
        return self._determine_market_type(weighted_score)
        
    def _determine_market_cycle(self, index_status):
        """确定市场周期"""
        # 统计各指数的趋势和位置得分
        trend_scores = []
        position_scores = []
        
        for status in index_status.values():
            trend_scores.append(status['trend_score'])
            position_scores.append(status['position_score'])
            
        # 计算平均得分
        avg_trend = np.mean(trend_scores)
        avg_position = np.mean(position_scores)
        
        # 根据趋势和位置确定周期
        if avg_trend > 0.6 and avg_position > 0.6:
            return "上升"
        elif avg_trend < 0.4 and avg_position < 0.4:
            return "下降"
        elif avg_trend > 0.5 and avg_position < 0.5:
            return "回调"
        elif avg_trend < 0.5 and avg_position > 0.5:
            return "反弹"
        else:
            return "蓄势"
            
    def _generate_trading_advice(self, result):
        """
        根据市场状态生成交易建议 - 短线交易版本
        
        参数:
            result (dict): 市场状态分析结果
            
        返回:
            str: 交易建议
        """
        try:
            market_type = result.get('market_type', 32)
            signal_strength = result.get('signal_strength', 0)
            
            # 检查是否有创业板指数数据，因为短线交易主要关注创业板
            has_gem_index = '399006.SZ' in result.get('individual_status', {})
            gem_index_data = result.get('individual_status', {}).get('399006.SZ', {})
            
            # 获取创业板的单日涨跌幅
            daily_change = None
            if has_gem_index:
                # 直接从指数状态中获取涨跌幅
                daily_change = gem_index_data.get('change_percent')
                if daily_change is None:
                    # 如果没有直接的涨跌幅数据，尝试从技术指标中获取
                    tech_indicators = gem_index_data.get('technical_indicators', {})
                    daily_change = tech_indicators.get('涨跌幅')
            
            # 根据市场类型和信号强度生成交易建议
            if market_type in [11, 12]:  # 强势上涨、加速上涨
                if signal_strength >= 8:
                    return "短线可积极做多，关注创业板领先板块，高开可追"
                elif signal_strength >= 6:
                    return "短线偏多，盘中回调可考虑入场，设好止损"
                else:
                    return "可择机短多，注意盘中节奏，防止回撤"
                    
            elif market_type == 13:  # 稳健上涨
                if has_gem_index and daily_change and daily_change > 3.0:
                    return "短线可继续持有多单，关注量能配合，注意盘中分歧"
                else:
                    return "可小仓位试多，分批建仓，严格设置止损"
                    
            elif market_type in [21, 22, 23]:  # 盘升上涨、回调上涨、整固上涨
                if has_gem_index and daily_change and daily_change > 1.5:
                    return "高抛低吸，注意日内波动，逢低介入为主"
                else:
                    return "观望为主，等待市场明确方向，日内参与为主"
                    
            elif market_type in [31, 32, 33]:  # 盘整类型
                return "区间震荡对待，以日内短线为主，不宜隔夜持仓"
                    
            elif market_type in [41, 42, 43]:  # 轻微下跌
                if has_gem_index and daily_change and daily_change < -2.0:
                    return "暂不介入，等待企稳信号，注意关键位支撑"
                else:
                    return "回避为主，减少操作频率，不追空"
                    
            elif market_type in [51, 52, 53]:  # 强势下跌
                if signal_strength >= 8:
                    return "空仓观望，回避风险，等待企稳再入场"
                else:
                    return "谨慎防守，注意止损，不盲目抄底"
            
            # 默认建议
            return "观望为主，等待明确信号"
            
        except Exception as e:
            self.logger.error(f"生成交易建议时出错: {str(e)}")
            return "观望为主，等待明确信号"

    def _calculate_signal_strength(self, result):
        """
        计算信号强度 - 短线交易增强版
        
        参数:
            result (dict): 市场状态分析结果
            
        返回:
            int: 信号强度，范围0-10
        """
        try:
            market_type = result.get('market_type', 32)
            index_status = result.get('individual_status', {})
            
            # 基础信号强度
            base_strength = 0
            
            # 根据市场类型设置基础信号强度
            if market_type in [11, 12, 52]:  # 极强上涨和极度下跌
                base_strength = 8
            elif market_type in [13, 51]:  # 稳健上涨和加速下跌
                base_strength = 6
            elif market_type in [21, 22, 23, 41, 42, 43]:  # 中等上涨和中等下跌
                base_strength = 4
            else:  # 盘整类型
                base_strength = 2
            
            # 调整因子：检查创业板指数的单日涨跌幅
            adjustment = 0
            if '399006.SZ' in index_status:
                # 直接从指数状态中获取涨跌幅
                daily_change = index_status.get('399006.SZ', {}).get('change_percent')
                if daily_change is None:
                    # 如果没有直接的涨跌幅数据，尝试从技术指标中获取
                    tech_indicators = index_status.get('399006.SZ', {}).get('technical_indicators', {})
                    daily_change = tech_indicators.get('涨跌幅')
                
                if daily_change is not None:
                    if abs(daily_change) > 7:  # 大幅波动
                        adjustment += 2
                    elif abs(daily_change) > 3:  # 明显波动
                        adjustment += 1
                    
                    # 方向一致性增强信号
                    if (market_type in [11, 12, 13, 21, 22, 23] and daily_change > 0) or \
                       (market_type in [41, 42, 43, 51, 52, 53] and daily_change < 0):
                        adjustment += 1
            
            # 确保最终得分在0-10范围内
            return max(0, min(10, base_strength + adjustment))
            
        except Exception as e:
            self.logger.error(f"计算信号强度时出错: {str(e)}")
            return 5  # 出错时返回中等信号强度

    def _save_market_status_data(self):
        """保存市场状态数据到文件"""
        try:
            status_file = os.path.join(self.daily_status_dir, "market_status.json")
            
            # 增量更新逻辑：先检查文件是否存在并加载现有数据
            existing_data = {}
            if os.path.exists(status_file):
                try:
                    # ✅ 修复：检查文件是否为空
                    if os.path.getsize(status_file) == 0:
                        self.logger.warning(f"市场状态文件为空: {status_file}，将创建新文件")
                        existing_data = {}
                    else:
                        with open(status_file, "r", encoding='utf-8') as f:
                            existing_data = json.load(f)
                        self.logger.info(f"加载现有市场状态数据，共 {len(existing_data)} 个日期")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON 解析失败: {str(e)}，文件可能损坏，将创建新文件")
                    existing_data = {}
                except Exception as e:
                    self.logger.warning(f"加载现有市场状态数据时出错: {str(e)}，将创建新文件")
            
            # 合并数据，新数据会覆盖同日期的旧数据
            merged_data = {**existing_data, **self.market_status_data}
            
            # 保存合并后的数据
            with open(status_file, "w", encoding='utf-8') as f:
                # 使用自定义的序列化函数处理特殊类型
                json.dump(merged_data, f, indent=4, default=str, ensure_ascii=False)
            
            # 更新内存中的数据为合并后的数据
            self.market_status_data = merged_data
            
            self.logger.info(f"成功保存市场状态数据，包含 {len(merged_data)} 个日期的数据")
            return True
        except Exception as e:
            # 记录更详细的错误信息，包括堆栈跟踪
            import traceback
            error_msg = f"保存市场状态数据到文件时出错: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return False

    def query_market_status(self, query_date=None):
        """
        查询指定日期的市场状态，并生成专业的市场分析报告
        
        参数:
            query_date: 查询日期，格式为YYYYMMDD，默认为最新日期
            
        返回:
            bool: 成功返回True，失败返回False
        """
        try:
            # 获取市场状态数据
            market_status = self.get_market_status(query_date)
            if market_status is None:
                self.logger.error(f"未找到日期 {query_date} 的市场状态数据")
                return False
            
            # 打印市场分析报告标题
            print("\n" + "="*80)
            print(f"创业板市场状态分析报告 ({query_date})")
            print("="*80)
            
            # 1. 市场概述
            print("\n[市场状态摘要]")
            print(f"市场状态: {market_status['market_description']}")
            print(f"综合评分: {market_status['trend_strength']:.2f}")
            print(f"市场周期: {market_status['cycle_description']}")
            
            # 2. 创业板指数状态
            if '399006.SZ' in market_status['individual_status']:
                cyb_status = market_status['individual_status']['399006.SZ']
                tech_data = cyb_status.get('technical_data', {})
                
                print("\n[创业板指数详情]")
                print(f"当前状态: {cyb_status.get('market_description', '未知')}")
                print(f"收盘价: {tech_data.get('close', 0):.2f}")
                
                # 获取涨跌幅
                change_percent = cyb_status.get('change_percent', 0)
                if change_percent != 0:
                    change_color = "红" if change_percent > 0 else "绿"
                    print(f"涨跌幅: {change_color}{change_percent:.2f}%")
                
                # 移动均线状态
                ma5 = tech_data.get('ma5', 0)
                ma20 = tech_data.get('ma20', 0)
                close = tech_data.get('close', 0)
                
                print("\n[趋势分析]")
                # 短期趋势 (基于MA5与收盘价)
                short_trend = "上涨" if close > ma5 else "下跌" if close < ma5 else "盘整"
                short_strength = abs((close - ma5) / ma5 * 100) if ma5 > 0 else 0
                if short_strength > 5:
                    short_trend = "强势" + short_trend
                elif short_strength > 2:
                    short_trend = "温和" + short_trend
                
                # 中期趋势 (基于MA20与收盘价)
                mid_trend = "上涨" if close > ma20 else "下跌" if close < ma20 else "盘整"
                mid_strength = abs((close - ma20) / ma20 * 100) if ma20 > 0 else 0
                if mid_strength > 10:
                    mid_trend = "强势" + mid_trend
                elif mid_strength > 5:
                    mid_trend = "温和" + mid_trend
                
                # 长期趋势 (基于MA60与收盘价)
                ma60 = tech_data.get('ma60', 0)
                long_trend = "上涨" if close > ma60 else "下跌" if close < ma60 else "盘整"
                long_strength = abs((close - ma60) / ma60 * 100) if ma60 > 0 else 0
                if long_strength > 15:
                    long_trend = "强势" + long_trend
                elif long_strength > 7:
                    long_trend = "温和" + long_trend
                
                print(f"短期趋势: {short_trend} (收盘价 vs MA5: {(close - ma5) / ma5 * 100:.2f}%" if ma5 > 0 else "短期趋势: 数据不足")
                print(f"中期趋势: {mid_trend} (收盘价 vs MA20: {(close - ma20) / ma20 * 100:.2f}%" if ma20 > 0 else "中期趋势: 数据不足")
                if ma60 > 0:
                    print(f"长期趋势: {long_trend} (收盘价 vs MA60: {(close - ma60) / ma60 * 100:.2f}%)")
                
                # 技术数据展示
                print("\n[技术指标]")
                print(f"均线数据:  MA5={ma5:.2f}, MA20={ma20:.2f}" + (f", MA60={ma60:.2f}" if ma60 > 0 else ""))
                
                # 动量指标
                rsi6 = tech_data.get('rsi6', 0)
                print(f"动能指标:  RSI6={rsi6:.2f}")
                
                # MACD指标
                macd = tech_data.get('macd', 0)
                macd_signal = tech_data.get('macd_signal', 0)
                macd_hist = tech_data.get('macd_hist', 0)
                print(f"MACD指标:  DIFF={macd:.4f}, DEA={macd_signal:.4f}, HIST={macd_hist:.4f}")
                
                # KDJ指标
                kdj_k = tech_data.get('kdj_k', 0)
                kdj_d = tech_data.get('kdj_d', 0)
                if kdj_k > 0 and kdj_d > 0:
                    print(f"KDJ指标:   K={kdj_k:.2f}, D={kdj_d:.2f}")
                
                # 布林带指标
                if 'bb_width' in tech_data and 'bb_position' in tech_data:
                    bb_width = tech_data.get('bb_width', 0)
                    bb_position = tech_data.get('bb_position', 0)
                    print(f"布林带:     宽度={bb_width:.4f}, 位置={bb_position:.4f}")
                
                # 支撑与阻力
                support = cyb_status.get('support', 0)
                resistance = cyb_status.get('resistance', 0)
                if support > 0 and resistance > 0:
                    print(f"价格区间:   支撑位={support:.2f}, 阻力位={resistance:.2f}")
                else:
                    # 尝试从market_status中获取
                    support = market_status.get('support_level', 0)
                    resistance = market_status.get('resistance_level', 0)
                    if support > 0 and resistance > 0:
                        print(f"价格区间:   支撑位={support:.2f}, 阻力位={resistance:.2f}")
            
            # 3. 多因素评分
            print("\n[多因素评分]")
            if '399006.SZ' in market_status['individual_status']:
                cyb_status = market_status['individual_status']['399006.SZ']
                print(f"趋势因子:   {cyb_status.get('trend_score', 0):.2f}")
                print(f"位置因子:   {cyb_status.get('position_score', 0):.2f}")
                print(f"量能因子:   {cyb_status.get('volume_score', 0):.2f}")
                print(f"动能因子:   {cyb_status.get('momentum_score', 0):.2f}")
                print(f"波动因子:   {cyb_status.get('volatility_score', 0):.2f}")
                print(f"结构因子:   {cyb_status.get('structure_score', 0):.2f}")
                print(f"周期因子:   {cyb_status.get('cycle_score', 0):.2f}")
                print(f"综合评分:   {cyb_status.get('total_score', 0):.2f}")
            
            # 4. 交易信号
            print("\n[交易信号]")
            signals = market_status.get('signals', {})
            print(f"交易信号:   {signals.get('trend_signal', 'neutral')}")
            print(f"信号强度:   {signals.get('strength', 0)}")
            print(f"交易建议:   {signals.get('trading_advice', '观望')}")
            print(f"风险等级:   {signals.get('risk_level', 'medium')}")
            
            print("\n" + "="*80)
            return True
            
        except Exception as e:
            self.logger.error(f"查询市场状态时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# 功能注解系统
PROCESSOR_FUNCTIONS = {}

def register_function(name, description):
    """注册函数装饰器，用于添加到可执行的功能列表"""
    def decorator(func):
        PROCESSOR_FUNCTIONS[name] = {
            "function": func,
            "description": description
        }
        return func
    return decorator

@register_function("process", "处理和存储大盘状态数据")
def process_market_data(processor, start_date=None, end_date=None, index_codes=None, parallel=False):
    """处理大盘状态数据并保存
    
    参数:
        processor (MarketStatusProcessor): 处理器实例
        start_date (str): 开始日期，格式YYYYMMDD，默认为过去30天
        end_date (str): 结束日期，格式YYYYMMDD，默认为当前日期
        index_codes (list): 指数代码列表，默认为['399006.SZ']
        parallel (bool): 是否使用并行处理，默认为False
        
    返回:
        dict: 处理结果
    """
    import datetime
    
    logger = logging.getLogger(__name__)
    
    # 设置默认日期
    if start_date is None:
        # 使用2023年1月1日作为默认开始日期
        start_date = processor.default_start_date
        logger.info(f"未指定开始日期，使用默认值: {start_date}")
    
    if end_date is None:
        # 使用2023年3月31日作为默认结束日期
        end_date = processor.default_end_date
        logger.info(f"未指定结束日期，使用默认值: {end_date}")
    
    # 记录实际处理的日期范围，确保用户清楚知道处理的是什么日期
    logger.info(f"处理市场状态数据: 开始日期={start_date}, 结束日期={end_date}")
    
    # 设置默认指数代码
    if index_codes is None:
        index_codes = processor.default_index_codes
        
    # 串行处理
    if not parallel:
        print(f"开始串行处理 {start_date} 至 {end_date} 的市场状态数据")
        logger.info(f"开始串行处理 {start_date} 至 {end_date} 的市场状态数据")
        success = processor.process_market_status(start_date, end_date, index_codes)
        
        # 适应process_market_status返回布尔值的新情况
        result = {
            'success': success,
            'message': f"处理{'成功' if success else '失败'}, 时间范围: {start_date} 至 {end_date}",
            'data': None
        }
        
        if result['success']:
            logger.info(f"处理完成，{result['message']}")
            print(f"处理完成，{result['message']}")
        else:
            logger.error(f"处理失败: {result['message']}")
            print(f"处理失败: {result['message']}")
        return result
    # 并行处理
    else:
        logger.info(f"开始并行处理 {start_date} 至 {end_date} 的市场状态数据")
        # 获取日期范围
        date_range = processor._get_date_range(start_date, end_date)
        
        # 创建进程池
        with ProcessPoolExecutor(max_workers=15) as executor:
            # 提交任务
            futures = {executor.submit(processor.process_market_status, processor._date_to_str(date), processor._date_to_str(date), index_codes): date for date in date_range}
            
            # 处理结果
            success_count = 0
            error_count = 0
            for future in as_completed(futures):
                date = futures[future]
                try:
                    success = future.result()  # 现在这里返回的是布尔值
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        logger.error(f"处理 {date} 失败")
                except Exception as e:
                    error_count += 1
                    logger.error(f"处理 {date} 时出错: {str(e)}")
            
            logger.info(f"并行处理完成，成功: {success_count}, 失败: {error_count}")
            print(f"并行处理完成，成功: {success_count}, 失败: {error_count}")
            
            return {
                'success': error_count == 0,
                'message': f"处理完成，成功: {success_count}, 失败: {error_count}",
                'data': {
                    'success_count': success_count,
                    'error_count': error_count
                }
            }

@register_function("query", "查询市场状态数据")
def query_market_status(processor, date=None):
    """查询市场状态数据并提供专业量化分析
    
    参数:
        processor (MarketStatusProcessor): 处理器实例
        date (str): 查询日期，格式YYYYMMDD，默认为None表示查询最新日期
        
    返回:
        dict: 市场状态数据
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"查询市场状态数据: 日期={date if date else '最新'}")
    
    # 调用处理器的get_market_status方法
    market_status = processor.get_market_status(date)
    print(f"market_status: {market_status}")
    
    if market_status:
        if date:
            logger.info(f"成功查询到 {date} 的市场状态数据")
            display_date = date
        else:
            logger.info(f"成功查询到最新市场状态数据")
            # 当查询最新数据时，尝试从数据中获取实际日期
            if 'date' in market_status:
                display_date = market_status['date']
            else:
                # 如果没有日期信息，从处理器获取最新可用日期
                available_dates = processor.get_all_available_dates()
                display_date = available_dates[-1] if available_dates else "未知日期"
        
        # 打印市场状态的详细信息
        print(f"\n============ 市场状态量化分析报告 ({display_date}) ============")
        
        # 市场综合状态
        if 'market_description' in market_status:
            market_desc = market_status.get('market_description', '未知')
            total_score = market_status.get('total_score', 0)
            print(f"市场状态: {market_desc}  综合评分: {total_score:.2f}")
            
        # 市场周期
        if 'cycle_description' in market_status:
            print(f"市场周期: {market_status.get('cycle_description', '未知')}")
        
        # 分割线
        print("-" * 60)
        
        # 各指数状态 - 重点关注创业板
        if 'individual_status' in market_status:
            ind_status = market_status['individual_status']
            
            # 特别关注创业板指数
            if '399006.SZ' in ind_status:
                cyb_status = ind_status['399006.SZ']
                print("\n【创业板指数】- 主要参考指标")
                print(f"当前状态: {cyb_status.get('market_description', '未知')}  评分: {cyb_status.get('total_score', 0):.2f}")
                
                # 技术指标数据
                if 'technical_data' in cyb_status:
                    tech_data = cyb_status['technical_data']
                    
                    # 基本价格信息
                    print(f"收盘价: {tech_data.get('close', 0):.2f}")
                    
                    # 计算前一日价格和涨跌幅
                    if 'prev_close' in tech_data:
                        prev_close = tech_data.get('prev_close', 0)
                        price_change = tech_data.get('close', 0) - prev_close
                        price_change_pct = (price_change / prev_close * 100) if prev_close > 0 else 0
                        change_sign = "+" if price_change >= 0 else ""
                        print(f"涨跌: {change_sign}{price_change:.2f} ({change_sign}{price_change_pct:.2f}%)")
                    
                    # 均线系统
                    print(f"均线系统: MA5={tech_data.get('ma5', 0):.2f} MA20={tech_data.get('ma20', 0):.2f} MA60={tech_data.get('ma60', 0):.2f}")
                    
                    # 通过均线关系判断趋势
                    close = tech_data.get('close', 0)
                    ma5 = tech_data.get('ma5', 0)
                    ma20 = tech_data.get('ma20', 0)
                    ma60 = tech_data.get('ma60', 0)
                    
                    # 短期趋势 (基于MA5和价格关系)
                    short_term = "上涨" if close > ma5 else "下跌" if close < ma5 else "盘整"
                    if close > ma5 * 1.05:
                        short_term = "强势上涨"
                    elif close < ma5 * 0.95:
                        short_term = "强势下跌"
                    
                    # 中期趋势 (基于MA20和价格/MA5关系)
                    medium_term = "上涨" if close > ma20 and ma5 > ma20 else "下跌" if close < ma20 and ma5 < ma20 else "盘整"
                    if close > ma20 * 1.1:
                        medium_term = "强势上涨"
                    elif close < ma20 * 0.9:
                        medium_term = "强势下跌"
                    
                    # 长期趋势 (基于MA60和价格/MA20关系)
                    long_term = "上涨" if close > ma60 and ma20 > ma60 else "下跌" if close < ma60 and ma20 < ma60 else "盘整"
                    if close > ma60 * 1.15:
                        long_term = "强势上涨"
                    elif close < ma60 * 0.85:
                        long_term = "强势下跌"
                    
                    print(f"趋势分析: 短期-{short_term} | 中期-{medium_term} | 长期-{long_term}")
                    
                    # 动能和波动指标
                    print(f"动能指标: RSI6={tech_data.get('rsi6', 0):.2f} | RSI12={tech_data.get('rsi12', 0):.2f}")
                    print(f"MACD: DIFF={tech_data.get('macd', 0):.4f} | DEA={tech_data.get('macd_signal', 0):.4f} | HIST={tech_data.get('macd_hist', 0):.4f}")
                    
                    # KDJ指标
                    if 'kdj_k' in tech_data and 'kdj_d' in tech_data:
                        print(f"KDJ: K={tech_data.get('kdj_k', 0):.2f} | D={tech_data.get('kdj_d', 0):.2f}")
                    
                    # 布林带
                    if 'bb_width' in tech_data:
                        print(f"布林带: 宽度={tech_data.get('bb_width', 0):.4f} | 位置={tech_data.get('bb_position', 0):.4f}")
                    
                    # 支撑与阻力
                    if 'support' in tech_data and 'resistance' in tech_data:
                        print(f"支撑位: {tech_data.get('support', 0):.2f} | 阻力位: {tech_data.get('resistance', 0):.2f}")
                    
                # 得分情况
                print("\n多因子评分:")
                print(f"  趋势得分: {cyb_status.get('trend_score', 0):.2f}  位置得分: {cyb_status.get('position_score', 0):.2f}")
                print(f"  量能得分: {cyb_status.get('volume_score', 0):.2f}  动能得分: {cyb_status.get('momentum_score', 0):.2f}")
                print(f"  波动得分: {cyb_status.get('volatility_score', 0):.2f}  结构得分: {cyb_status.get('structure_score', 0):.2f}")
                print(f"  周期得分: {cyb_status.get('cycle_score', 0):.2f}")
            
            # 简要显示其他指数
            other_indices = [idx for idx in ind_status.keys() if idx != '399006.SZ']
            if other_indices:
                print("\n其他指数状态:")
                for idx in other_indices:
                    status = ind_status[idx]
                    if 'technical_data' in status:
                        tech = status['technical_data']
                        if 'close' in tech and 'prev_close' in tech:
                            close = tech.get('close', 0)
                            prev = tech.get('prev_close', 0)
                            chg_pct = (close - prev) / prev * 100 if prev > 0 else 0
                            sign = "+" if chg_pct >= 0 else ""
                            print(f"  {idx}: {status.get('market_description', '未知')} ({sign}{chg_pct:.2f}%)")
                        else:
                            print(f"  {idx}: {status.get('market_description', '未知')}")
                    else:
                        print(f"  {idx}: {status.get('market_description', '未知')}")
        
        # 分割线
        print("-" * 60)
        
        # 交易信号
        if 'signals' in market_status:
            signals = market_status['signals']
            print("\n【交易量化信号】")
            
            # 趋势信号
            trend_signal = signals.get('trend_signal', 'neutral')
            trend_text = "多头" if trend_signal == 'bullish' else "空头" if trend_signal == 'bearish' else "中性"
            print(f"主导趋势: {trend_text}  信号强度: {signals.get('strength', 0)}/10")
            
            # 风险水平
            risk_level = signals.get('risk_level', 'medium')
            risk_text = "高" if risk_level == 'high' else "中" if risk_level == 'medium' else "低"
            print(f"风险等级: {risk_text}  预警级别: {signals.get('alert_level', 0)}/5")
            
            # 交易建议
            print(f"\n交易建议: {signals.get('trading_advice', '观望')}")
            
            # 如果有机会板块，显示
            if 'opportunity_sectors' in signals and signals['opportunity_sectors']:
                sectors = signals['opportunity_sectors']
                print(f"重点关注板块: {', '.join(sectors)}")
    else:
        if date:
            logger.warning(f"未找到 {date} 的市场状态数据")
            print(f"未找到 {date} 的市场状态数据")
        else:
            logger.warning(f"未找到市场状态数据")
            print(f"未找到市场状态数据")
    
    return market_status

@register_function("update", "增量更新市场状态数据")
def update_market_data(processor, start_date=None, end_date=None, index_codes=None, force=False, parallel=False):
    """
    增量更新市场状态数据
    
    参数:
        processor: MarketStatusProcessor实例
        start_date: 开始日期，如果为None则使用30天前
        end_date: 结束日期，如果为None则使用当前日期
        index_codes: 指数列表，如果为None则使用默认列表
        force: 是否强制更新（即使已存在数据）
        parallel: 是否并行处理
    """
    import sys
    from datetime import datetime, timedelta
    
    # 处理日期参数
    current_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        # 默认处理前30天的数据
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        
    if end_date is None:
        end_date = current_date
    
    # 确保processor实例使用正确的并行设置
    processor.parallel = parallel
    
    print(f"增量更新 {start_date} 至 {end_date} 的市场状态数据...")
    # import time
    # time.sleep(10000)
    # 检查已有数据，只处理未处理的日期
    available_dates = processor.metadata["available_dates"]
    target_dates = processor._get_date_range(start_date, end_date)
    target_date_strs = [processor._date_to_str(date) for date in target_dates]
    
    # 记录更新前的市场状态数据数量
    original_data_count = len(processor.market_status_data)
    # 记录已存在但将被更新的日期
    existing_dates_to_update = []
    
    if force:
        # 强制更新所有日期
        dates_to_process = target_date_strs
        # 找出已存在但将被更新的日期
        existing_dates_to_update = [date for date in dates_to_process if date in processor.market_status_data]
    else:
        # 只处理未处理的日期
        dates_to_process = [date for date in target_date_strs if date not in available_dates]
    
    if not dates_to_process:
        print("所有日期已处理，无需更新")
        return 0
    
    print(f"需要处理 {len(dates_to_process)} 个日期: {dates_to_process[0]} 至 {dates_to_process[-1]}")
    if force and existing_dates_to_update:
        print(f"其中 {len(existing_dates_to_update)} 个日期的数据将被覆盖更新")
    
    # 重新设置处理范围
    start_date = dates_to_process[0]
    end_date = dates_to_process[-1]
    
    processed_count = processor.process_market_status(start_date, end_date, index_codes=index_codes)
    
    # 计算更新统计信息
    updated_data_count = len(processor.market_status_data)
    new_data_count = updated_data_count - original_data_count
    replaced_data_count = len(existing_dates_to_update)
    
    # 输出更详细的更新状态
    print(f"处理完成，市场状态数据更新结果:")
    print(f"  - 总数据量: {updated_data_count} 条")
    print(f"  - 新增数据: {new_data_count} 条")
    if force:
        print(f"  - 覆盖更新: {replaced_data_count} 条")
    print(f"  - 处理日期范围: {start_date} 至 {end_date}")
    
    return processed_count

@register_function("report", "生成市场状态报告")
def generate_market_report(processor, start_date=None, end_date=None, output_file=None):
    """
    生成市场状态报告
    
    参数:
        processor: MarketStatusProcessor实例
        start_date: 开始日期，如果为None则使用可用的第一个日期或30天前
        end_date: 结束日期，如果为None则使用当前日期
        output_file: 输出文件路径，如果为None则使用默认路径
    """
    from datetime import datetime, timedelta
    import os
    
    # 处理日期参数
    current_date = datetime.now().strftime("%Y%m%d")
    
    if start_date is None:
        # 检查是否有可用日期
        available_dates = processor.get_all_available_dates()
        if available_dates:
            start_date = available_dates[0]
        else:
            # 默认使用30天前
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    
    if end_date is None:
        end_date = current_date
    
    if output_file is None:
        # 默认输出到数据目录
        output_file = os.path.join(processor.data_dir, f"market_report_{start_date}_{end_date}.md")
    
    print(f"生成 {start_date} 至 {end_date} 的市场状态报告...")
    report = processor.generate_market_status_report(
        start_date=start_date, 
        end_date=end_date,
        output_file=output_file
    )
    
    if report.startswith("未找到") or report.startswith("没有可用"):
        print(f"生成报告失败: {report}")
        return None
    else:
        print(f"报告已保存到: {output_file}")
        return report

def show_available_functions():
    """显示所有可用的功能"""
    print("\n可用功能列表:")
    for name, info in PROCESSOR_FUNCTIONS.items():
        print(f"  {name}: {info['description']}")
    print("\n使用方法示例:")
    print("  # 处理最近30天的数据")
    print("  process_market_data(processor)")
    print("  # 处理特定日期范围的数据")
    print("  process_market_data(processor, '20230101', '20230331')")
    print("  # 查询最新市场状态")
    print("  query_market_status(processor)")
    print("  # 生成报告")
    print("  generate_market_report(processor)")
    print("\n详细参数说明请查看各函数的文档字符串。")

# 主函数
if __name__ == "__main__":
    import sys
    from datetime import datetime
    import os
    
    # 默认数据目录
    data_dir = "local_data/market_status"
    
    # 创建处理器实例
    processor = MarketStatusProcessor(data_dir=data_dir)
    
    # 显示功能列表
    show_available_functions()
    
    # 导出函数到全局命名空间，方便交互式使用
    globals().update({
        name: lambda func=info["function"], *args, **kwargs: func(processor, *args, **kwargs)
        for name, info in PROCESSOR_FUNCTIONS.items()
    })
    
    # 为了方便直接使用函数名而不是字典访问
    process_market_data = lambda *args, **kwargs: PROCESSOR_FUNCTIONS["process"]["function"](processor, *args, **kwargs)
    query_market_status = lambda *args, **kwargs: PROCESSOR_FUNCTIONS["query"]["function"](processor, *args, **kwargs)
    update_market_data = lambda *args, **kwargs: PROCESSOR_FUNCTIONS["update"]["function"](processor, *args, **kwargs)
    generate_market_report = lambda *args, **kwargs: PROCESSOR_FUNCTIONS["report"]["function"](processor, *args, **kwargs)
    
    # 导出实例便于交互式访问
    print("\n示例代码:")
    print("# 示例1: 处理最近30天的数据")
    print("result = process_market_data()")
    print("\n# 示例2: 查询最新市场状态")
    print("status = query_market_status()")
    print("\n# 示例3: 生成市场报告")
    print("report = generate_market_report()")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='市场状态处理器')
    parser.add_argument('--start_date', '-s', type=str, default='20260113',
                        help='开始日期 (YYYYMMDD格式，默认: 20260113)')
    parser.add_argument('--end_date', '-e', type=str, default='20260113',
                        help='结束日期 (YYYYMMDD格式，默认: 20260113)')
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    # 可以在这里添加示例调用
    # process_market_data(start_date='20230101', end_date='20251021', index_codes=['399006.SZ'])
    process_market_data(start_date=start_date, end_date=end_date, index_codes=['399006.SZ']) 
    # process_market_data(start_date='20230101', end_date='20251231', index_codes=['399006.SZ'])  20250630
    # process_market_data(start_date='20230101', end_date='20230601', index_codes=['399006.SZ'])
    # process_market_data(start_date='20260121', end_date='20260126', index_codes=['399006.SZ'])

    # update_market_data(start_date='20250424', end_date='20250425', index_codes=['399006.SZ'], force =True) 
    # query_market_status(date='20230103')
    # 添加main代码块，只有直接运行这个文件时才会执行
    # if __name__ == "__main__":
    #     # 初始化处理器实例
    #     processor = MarketStatusProcessor()
        
    #     # 运行测试代码
    #     print("直接运行测试:")
    #     process_market_data(processor, start_date='20250322', end_date='20250327')
        
    #     # 测试查询功能
    #     print("\n测试查询最新市场状态:")
    #     query_market_status(processor)
        
    #     print("\n测试查询特定日期市场状态:")
    #     query_market_status(processor, date='20240927')
        
        # 路径测试等其他测试...
    # # 测试文件路径处理
