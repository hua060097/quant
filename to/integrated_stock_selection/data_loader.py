#!/usr/bin/env python3
"""
一体化选股系统数据加载器
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "clean_ml_system"))

# 默认配置
class DefaultSystemConfig:
    def __init__(self):
        self.min_market_cap = 50
        self.max_volatility = 0.1
        # 添加缺失的属性
        project_root = Path(__file__).parent.parent
        self.cache_dir = project_root / "integrated_stock_selection/cache"
        self.results_dir = project_root / "integrated_stock_selection/results"
        self.stock_features_dir = project_root / "local_data/daily"
        self.sector_features_dir = project_root / "local_data/sector_features"
        self.market_status_dir = project_root / "local_data/market_status"
        self.board_limit_dir = project_root / "local_data/sector_features_board"

        # 创建必要目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


system_config = DefaultSystemConfig()

# 默认特征列表
STOCK_FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg',
    'total_mv', 'circ_mv', 'turnover_rate'
]

EXCLUDED_FEATURES = ['date', 'stock_code', 'sector_code']

# 尝试导入基础数据加载器
try:
    from clean_ml_system.data_loader import DataLoader as BaseDataLoader
except ImportError:

    # 如果导入失败，创建一个简化版本
    class BaseDataLoader:
        def __init__(self):
            # 初始化交易日历
            self.trading_calendar = None
            self._load_trading_calendar()

        def _load_trading_calendar(self):
            """严格使用advanced_quant_data加载交易日历"""
            try:
                import sys
                from pathlib import Path

                # 添加to目录到路径
                project_root = Path(__file__).parent.parent.parent
                to_dir = str(project_root / "to")
                if to_dir not in sys.path:
                    sys.path.append(to_dir)

                from advanced_quant_data import ParquetDataStorage
                storage = ParquetDataStorage()

                # 严格使用get_trading_calendar方法
                calendar_df = storage.get_trading_calendar('20220101', '20251231')

                if calendar_df.empty:
                    raise Exception("交易日历数据为空")

                if 'date_str' not in calendar_df.columns:
                    raise Exception("交易日历缺少date_str列")

                # 提取交易日列表
                self.trading_calendar = calendar_df['date_str'].tolist()

                if not self.trading_calendar:
                    raise Exception("交易日历列表为空")

                print(f"IntegratedDataLoader加载交易日历成功: {len(self.trading_calendar)} 个交易日")
                print(f"交易日历范围: {min(self.trading_calendar)} - {max(self.trading_calendar)}")

            except Exception as e:
                error_msg = f"严格加载交易日历失败: {e}"
                print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)



        def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
            """获取交易日期列表"""
            if not self.trading_calendar:
                print("警告：交易日历为空")
                return []

            import pandas as pd

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            valid_dates = []
            for date_str in self.trading_calendar:
                date_obj = pd.to_datetime(date_str)
                if start_dt <= date_obj <= end_dt:
                    valid_dates.append(date_str)

            print(f"获取交易日 {start_date} - {end_date}: {valid_dates}")
            return valid_dates

        def load_sector_features(self, start_date: str, end_date: str):
            return None

        def load_market_features(self, start_date: str, end_date: str):
            return None

class IntegratedDataLoader:
    """一体化数据加载器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_loader = BaseDataLoader()
        self.cache_dir = system_config.cache_dir
        # self._load_trading_calendar()

    def _load_trading_calendar(self, start: str, end: str):
        """严格使用advanced_quant_data加载交易日历"""
        try:
            import sys
            from pathlib import Path
            # 添加to目录到路径
            project_root = Path(__file__).parent.parent.parent
            to_dir = str(project_root / "to")
            if to_dir not in sys.path:
                sys.path.append(to_dir)
            from advanced_quant_data import ParquetDataStorage
            storage = ParquetDataStorage()
            # 严格使用get_trading_calendar方法
            calendar_df = storage.get_trading_calendar(start, end)
            if calendar_df.empty:
                raise Exception("交易日历数据为空")
            if 'date_str' not in calendar_df.columns:
                raise Exception("交易日历缺少date_str列")
            # 提取交易日列表
            self.trading_calendar = calendar_df['date_str'].tolist()
            if not self.trading_calendar:
                raise Exception("交易日历列表为空")
            print(f"IntegratedDataLoader加载交易日历成功: {len(self.trading_calendar)} 个交易日")
            print(f"交易日历范围: {min(self.trading_calendar)} - {max(self.trading_calendar)}")
        except Exception as e:
            error_msg = f"严格加载交易日历失败: {e}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        
        try:
            """获取交易日期列表"""
            # if not self.trading_calendar:
            self._load_trading_calendar(start_date, end_date)

            import pandas as pd

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            valid_dates = []
            for date_str in self.trading_calendar:
                date_obj = pd.to_datetime(date_str)
                if start_dt <= date_obj <= end_dt:
                    valid_dates.append(date_str)

            print(f"获取交易日 {start_date} - {end_date}: {valid_dates}")
            return valid_dates
            
        except Exception as e:
            self.logger.error(f"获取交易日失败: {e}")
            # return self.base_loader.get_trading_dates(start_date, end_date)
            return []
            
    
    def load_stock_features(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载个股特征数据"""
        try:
            self.logger.info(f"加载个股特征数据: {start_date} - {end_date}")
            
            # 检查缓存
            cache_key = f"stock_features_{start_date}_{end_date}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                self.logger.info(f"从缓存加载: {cache_file}")
                return pd.read_parquet(cache_file)
            
            # 使用本地实现以确保正确的数据路径
            # 获取交易日
            trading_dates = self.get_trading_dates(start_date, end_date)
            if not trading_dates:
                self.logger.error("没有找到交易日")
                return None

            all_data = []

            # 逐日加载数据文件
            # 使用相对于当前文件的路径，确保路径正确
            current_file_dir = Path(__file__).parent  # integrated_stock_selection目录
            features_dir = current_file_dir.parent / "local_data" / "features" / "daily"

            for date_str in trading_dates:
                # 构建日度文件路径：local_data/features/daily/YYYYMMDD.parquet
                file_path = features_dir / f"{date_str}.parquet"

                if not file_path.exists():
                    self.logger.warning(f"日度文件不存在: {file_path}")
                    continue

                try:
                    # 加载日度数据
                    daily_data = pd.read_parquet(file_path)

                    # 确保有日期列
                    if 'date' not in daily_data.columns:
                        # 如果没有date列，添加日期
                        daily_data['date'] = pd.to_datetime(date_str)

                    all_data.append(daily_data)
                    self.logger.info(f"加载{date_str}数据: {len(daily_data)}条记录")

                except Exception as e:
                    self.logger.error(f"加载日度文件失败 {file_path}: {e}")
                    continue
            
            if not all_data:
                self.logger.error("没有成功加载任何数据")
                self.logger.error(f"交易日总数: {len(trading_dates)}")
                self.logger.error(f"特征文件目录: {features_dir}")
                self.logger.error(f"目录是否存在: {features_dir.exists()}")
                if features_dir.exists():
                    existing_files = list(features_dir.glob("*.parquet"))
                    self.logger.error(f"目录中的文件数: {len(existing_files)}")
                    if existing_files:
                        self.logger.error(f"前5个文件: {[f.name for f in existing_files[:5]]}")
                return None

            self.logger.info(f"准备合并数据，共{len(all_data)}个文件的数据")

            # 合并所有数据
            try:
                result_df = pd.concat(all_data, ignore_index=True)
                self.logger.info(f"数据合并成功，总记录数: {len(result_df)}")
            except Exception as e:
                self.logger.error(f"数据合并失败: {e}")
                self.logger.error(f"all_data长度: {len(all_data)}")
                for i, df in enumerate(all_data[:3]):  # 只检查前3个
                    self.logger.error(f"DataFrame {i}: shape={df.shape}, columns={len(df.columns)}")
                raise

            self.logger.info(f"成功加载个股特征: {len(result_df)}条记录, {len(result_df.columns)}个特征")

            # 🚀 修复: 保存到缓存时明确指定schema,防止pyarrow自动降级数据类型导致溢出
            # 问题: pyarrow会将float64的amount自动降级为int32,导致大于2,147,483,647的值溢出
            import pyarrow as pa
            schema = pa.Schema.from_pandas(result_df, preserve_index=False)
            result_df.to_parquet(cache_file, schema=schema, engine='pyarrow')

            return result_df
            
        except Exception as e:
            self.logger.error(f"加载个股特征失败: {e}")
            return None

    def _post_process_sector_data(
        self,
        data: Optional[pd.DataFrame],
        start_date: str,
        end_date: str,
        default_sector_code: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """保证板块数据具备因子计算所需字段"""
        if data is None:
            return None

        if data.empty:
            return data

        try:
            df = data.copy()
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            if 'date' not in df.columns:
                if 'date_str' in df.columns:
                    df['date'] = pd.to_datetime(df['date_str'], errors='coerce')
                else:
                    self.logger.warning("板块数据缺少 date 字段，无法使用")
                    return None
            else:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            df = df.dropna(subset=['date'])
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            if df.empty:
                return df

            if 'sector_code' not in df.columns:
                if 'sector' in df.columns:
                    df['sector_code'] = df['sector']
                elif 'sector_name' in df.columns:
                    df['sector_code'] = df['sector_name']
                elif default_sector_code is not None:
                    df['sector_code'] = default_sector_code
                else:
                    self.logger.warning("板块数据缺少 sector_code，且无法推断")
                    return None

            # 确保sector_code是字符串类型，处理可能的list类型
            df['sector_code'] = df['sector_code'].apply(lambda x: str(x) if not isinstance(x, list) else str(x[0]) if x else 'UNKNOWN')
            if 'sector_name' not in df.columns and 'sector' in df.columns:
                df['sector_name'] = df['sector']

            df['date_str'] = df['date'].dt.strftime('%Y%m%d')
            df = df.sort_values(['sector_code', 'date']).reset_index(drop=True)

            if 'pct_chg' in df.columns:
                df['return_1d'] = df['pct_chg'].astype(float)
            elif 'close' in df.columns:
                df['return_1d'] = df.groupby('sector_code')['close'].pct_change() * 100
            else:
                df['return_1d'] = 0.0

            df['return'] = df['return_1d']
            df['daily_return'] = df['return_1d']

            if 'close' in df.columns:
                group_close = df.groupby('sector_code')['close']
                for window in (3, 5, 10, 20):
                    col = f'return_{window}d'
                    if col not in df.columns:
                        df[col] = group_close.transform(lambda s, w=window: (s / s.shift(w) - 1.0) * 100)
            else:
                self.logger.warning("板块数据缺少 close 列，无法计算多日收益")

            for col in ['return_1d', 'return', 'daily_return', 'return_3d', 'return_5d', 'return_10d', 'return_20d']:
                if col in df.columns:
                    df[col] = df[col].astype(float).fillna(0.0)

            return df

        except Exception as exc:
            self.logger.warning(f"板块数据整理失败: {exc}")
            return None

    def _load_sector_features_basic(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """简化的板块数据加载逻辑，不强依赖额外衍生文件"""
        try:
            sector_root = system_config.sector_features_dir
            if not sector_root.exists():
                self.logger.warning(f"板块特征目录不存在: {sector_root}")
                return None

            frames: List[pd.DataFrame] = []
            for sector_dir in sector_root.iterdir():
                if not sector_dir.is_dir():
                    continue
                sector_code = sector_dir.name

                candidates = [
                    sector_dir / f"{sector_code}_features.parquet",
                    sector_dir / f"{sector_code}.parquet",
                ]
                feature_file = next((p for p in candidates if p.exists()), None)
                if feature_file is None:
                    continue

                try:
                    df = pd.read_parquet(feature_file)
                except Exception as read_exc:
                    self.logger.debug(f"读取板块文件失败 {feature_file}: {read_exc}")
                    continue

                processed = self._post_process_sector_data(df, start_date, end_date, default_sector_code=sector_code)
                if processed is not None and not processed.empty:
                    frames.append(processed)

            if not frames:
                self.logger.warning(f"简化加载仍未得到板块数据: {start_date} - {end_date}")
                return None

            sector_data = pd.concat(frames, ignore_index=True)
            sector_data = sector_data.sort_values(['sector_code', 'date']).reset_index(drop=True)
            self.logger.info(f"简化板块加载成功: {len(sector_data)} 条记录, {sector_data['sector_code'].nunique()} 个板块")
            return sector_data

        except Exception as exc:
            self.logger.error(f"简化加载板块数据失败: {exc}")
            return None


    def load_sector_features(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载板块特征数据"""
        try:
            self.logger.info(f"加载板块特征数据: {start_date} - {end_date}")

            sector_data: Optional[pd.DataFrame] = None
            try:
                sector_data = self.base_loader.load_sector_features(start_date, end_date)
            except Exception as inner_exc:
                self.logger.warning(f"专业板块加载失败，将尝试简化逻辑: {inner_exc}")

            sector_data = self._post_process_sector_data(sector_data, start_date, end_date)

            if sector_data is None or sector_data.empty:
                self.logger.warning("板块数据为空或缺少关键字段，启用简化加载逻辑")
                sector_data = self._load_sector_features_basic(start_date, end_date)

            if sector_data is None or sector_data.empty:
                self.logger.error("板块数据加载失败")
                return None

            sector_count = (
                sector_data['sector_code'].nunique()
                if 'sector_code' in sector_data.columns else '未知'
            )
            self.logger.info(f"成功加载板块数据: {len(sector_data)} 条记录, {sector_count} 个板块")

            return sector_data

        except Exception as e:
            self.logger.error(f"板块数据加载失败: {e}")
            return None

    def load_market_features(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载市场特征数据"""
        try:
            self.logger.info(f"加载市场特征数据: {start_date} - {end_date}")

            # 尝试使用基础加载器加载市场数据
            market_data = None
            try:
                market_data = self.base_loader.load_market_features(start_date, end_date)
            except Exception as inner_exc:
                self.logger.warning(f"市场数据加载失败: {inner_exc}")

            # 如果加载失败，创建默认市场数据
            if market_data is None or market_data.empty:
                self.logger.warning("市场数据加载失败，将使用默认值")
                trading_dates = self.get_trading_dates(start_date, end_date)
                market_data = pd.DataFrame({
                    'date': pd.to_datetime(trading_dates, format='%Y%m%d'),
                    'daily_change_pct': 0.0,  # 修正：使用正确的字段名
                    'volume': 1.0,  # 修正：使用正确的字段名
                })

            # 添加日期范围日志
            if market_data is not None and not market_data.empty and 'date' in market_data.columns:
                date_min = market_data['date'].min()
                date_max = market_data['date'].max()
                self.logger.info(f"成功加载市场数据: {len(market_data)} 条记录")
                self.logger.info(f"市场数据日期范围: {date_min} 至 {date_max}")
            else:
                self.logger.info(f"成功加载市场数据: {len(market_data)} 条记录")

            return market_data

        except Exception as e:
            self.logger.error(f"加载市场数据失败: {e}")
            return None

    def load_all_data(
        self,
        start_date: str,
        end_date: str,
        extend_history_days: int = 0
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        加载所有需要的数据 - 包含个股、板块和市场特征

        Args:
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
            extend_history_days: 扩展历史数据天数，用于计算需要历史数据的特征

        Returns:
            Tuple: (stock_data, sector_data, market_data)
        """
        try:
            self.logger.info(f"开始加载所有数据: {start_date} - {end_date}, 扩展历史天数: {extend_history_days}")

            # 计算扩展后的开始日期
            if extend_history_days > 0:
                extended_start_date = self._calculate_extended_start_date(start_date, end_date, extend_history_days)
            else:
                extended_start_date = start_date

            # 加载个股特征数据
            stock_data = self.load_stock_features(extended_start_date, end_date)

            # 加载板块特征数据
            sector_data = self.load_sector_features(extended_start_date, end_date)

            # 加载市场特征数据
            market_data = self.load_market_features(extended_start_date, end_date)

            self.logger.info(f"数据加载完成:")
            if stock_data is not None:
                self.logger.info(f"  个股数据: {len(stock_data)}条")
            else:
                self.logger.warning(f"  个股数据: 加载失败")

            if sector_data is not None:
                self.logger.info(f"  板块数据: {len(sector_data)}条")
            else:
                self.logger.warning(f"  板块数据: 加载失败")

            if market_data is not None:
                self.logger.info(f"  市场数据: {len(market_data)}条")
            else:
                self.logger.warning(f"  市场数据: 加载失败")

            return stock_data, sector_data, market_data

        except Exception as e:
            self.logger.error(f"加载所有数据失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None, None

    def _calculate_extended_start_date(self, start_date: str, end_date: str, extend_history_days: int) -> str:
        """
        计算扩展后的开始日期，确保有足够的历史交易日

        Args:
            start_date: 原始开始日期
            end_date: 结束日期
            extend_history_days: 需要扩展的交易日数量

        Returns:
            扩展后的开始日期
        """
        try:
            # 【修复】计算足够大的历史数据范围，确保包含所需的交易日
            # 估算：120个交易日大约需要6个月的自然日，为安全起见，扩展到8个月
            from datetime import datetime, timedelta

            start_dt = datetime.strptime(start_date, '%Y%m%d')
            # 向前扩展足够的自然日以确保有足够的历史交易日
            # 120个交易日约等于6个月，使用2倍安全边际
            safe_start_dt = start_dt - timedelta(days=extend_history_days * 2)
            safe_start_date = safe_start_dt.strftime('%Y%m%d')

            # 获取足够大范围的交易日历
            all_trading_dates = self.get_trading_dates(safe_start_date, end_date)

            if not all_trading_dates:
                self.logger.warning("无法获取交易日历，使用原始开始日期")
                return start_date

            self.logger.info(f"获取交易日历范围: {safe_start_date} - {end_date}, 共{len(all_trading_dates)}个交易日")

            # 查找原始开始日期在交易日历中的位置
            start_index = None
            for i, date in enumerate(all_trading_dates):
                if date >= start_date:  # 使用>=确保找到第一个不小于start_date的交易日
                    start_index = i
                    break

            if start_index is None:
                self.logger.warning(f"未找到开始日期{start_date}在交易日历中的位置，使用原始范围")
                return start_date

            # 向前扩展extend_history_days个交易日
            extended_start_index = max(0, start_index - extend_history_days)
            extended_start_date = all_trading_dates[extended_start_index]

            # 计算实际扩展的交易日数量
            actual_extended_days = start_index - extended_start_index

            self.logger.info(f"📅 数据范围扩展详情:")
            self.logger.info(f"   原始范围: {start_date} - {end_date}")
            self.logger.info(f"   扩展范围: {extended_start_date} - {end_date}")
            self.logger.info(f"   请求扩展: {extend_history_days} 个交易日")
            self.logger.info(f"   实际扩展: {actual_extended_days} 个交易日")
            self.logger.info(f"   原始开始日期在交易日历中的位置: 第{start_index + 1}个交易日")

            # 验证是否满足扩展要求
            if actual_extended_days < extend_history_days:
                self.logger.warning(f"⚠️ 实际扩展交易日({actual_extended_days})少于请求数量({extend_history_days})")
                self.logger.warning(f"   这可能是因为历史数据不足，已扩展到最早可用的交易日: {extended_start_date}")
            else:
                self.logger.info(f"✅ 成功扩展{actual_extended_days}个交易日，满足{extend_history_days}个交易日的要求")

            return extended_start_date

        except Exception as e:
            self.logger.error(f"计算扩展开始日期失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return start_date

    def load_stock_sector_mapping(self) -> Optional[Dict[str, List[str]]]:
        """
        加载股票-板块映射数据

        Returns:
            Dict[str, List[str]]: 股票代码到板块列表的映射，格式为 {"股票代码": ["板块1", "板块2", ...]}
            如果加载失败返回 None
        """
        try:
            import json
            from pathlib import Path

            # 构建JSON文件路径（智能路径解析：支持多种执行环境）
            # 尝试多个可能的路径
            possible_paths = [
                Path("../to/local_data/sector_mapping/stock_sector_mapping.json"),  # 从integrated_stock_selection目录
                Path("to/local_data/sector_mapping/stock_sector_mapping.json"),     # 从根目录
                Path(__file__).parent.parent / "to" / "local_data" / "sector_mapping" / "stock_sector_mapping.json"  # 基于当前文件位置
            ]

            json_file_path = None
            for path in possible_paths:
                if path.exists():
                    json_file_path = path
                    break

            if json_file_path is None:
                json_file_path = possible_paths[0]  # 默认使用第一个路径

            if not json_file_path.exists():
                self.logger.warning(f"股票板块映射文件不存在: {json_file_path}")
                return None

            # 读取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)

            # 验证数据格式
            if not isinstance(mapping_data, dict):
                self.logger.error("股票板块映射文件格式错误：根对象不是字典")
                return None

            # 验证数据内容
            valid_mapping = {}
            for stock_code, sectors in mapping_data.items():
                if isinstance(sectors, list) and all(isinstance(s, str) for s in sectors):
                    valid_mapping[stock_code] = sectors
                else:
                    self.logger.warning(f"股票 {stock_code} 的板块数据格式错误，已跳过")

            self.logger.info(f"✅ 股票板块映射加载成功: {len(valid_mapping)} 只股票")
            return valid_mapping

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON文件解析失败: {e}")
            return None
        except Exception as e:
            self.logger.error(f"加载股票板块映射失败: {e}")
            return None
