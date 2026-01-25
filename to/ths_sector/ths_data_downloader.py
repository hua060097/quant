#!/usr/bin/env python3
"""
同花顺行业概念板块数据下载器
基于tushare接口下载同花顺概念板块相关数据

主要功能：
1. 下载同花顺概念板块基础信息 (ths_index)
2. 下载概念板块成分股信息 (ths_member)  
3. 下载概念板块历史行情数据 (ths_daily)
4. 支持增量更新
"""

import os
import json
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
from typing import Optional, List, Dict, Any

class THSSectorDataDownloader:
    """同花顺行业概念板块数据下载器"""
    
    def __init__(self, token: str = '981cd3a1452111f36a31f73193866ad6df35fa26c1b980a73a061fd5'):
        """
        初始化下载器
        
        Args:
            token: tushare API token
        """
        self.token = token
        self.pro = ts.pro_api(token)
        
        # 设置数据保存路径
        self.base_path = Path("local_data/ths_sector")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # API调用间隔（秒）
        self.api_interval = 0.3

    def download_hot_sectors(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        下载指定日期范围内的每日最强板块数据

        Args:
            start_date: 开始日期 (YYYYMMDD格式)
            end_date: 结束日期 (YYYYMMDD格式)

        Returns:
            最强板块数据DataFrame
        """
        self.logger.info(f"开始下载最强板块数据: {start_date} - {end_date}")

        all_data = []

        # 生成日期范围
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        current_date = start_dt
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y%m%d')

            try:
                self.logger.info(f"正在下载 {date_str} 的最强板块数据...")

                # 调用API获取数据
                df = self.pro.limit_cpt_list(trade_date=date_str)

                if df is not None and not df.empty:
                    all_data.append(df)
                    self.logger.info(f"✅ {date_str}: 获取到 {len(df)} 条板块数据")
                else:
                    self.logger.warning(f"⚠️ {date_str}: 没有获取到数据")

                # API调用间隔
                time.sleep(self.api_interval)

            except Exception as e:
                self.logger.error(f"❌ {date_str}: 下载失败 - {e}")
                time.sleep(1)  # 出错时等待更长时间

            current_date += timedelta(days=1)

        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data, ignore_index=True)

            # 保存到parquet文件
            filename = f"hot_sectors_{start_date}_{end_date}.parquet"
            file_path = self.base_path / filename
            combined_df.to_parquet(file_path, index=False)

            # 保存元数据
            metadata = {
                'file_type': 'hot_sectors',
                'start_date': start_date,
                'end_date': end_date,
                'total_records': len(combined_df),
                'unique_dates': combined_df['trade_date'].nunique(),
                'created_at': datetime.now().isoformat()
            }

            metadata_file = self.base_path / f"hot_sectors_{start_date}_{end_date}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.logger.info(f"✅ 最强板块数据已保存: {file_path}")
            self.logger.info(f"📊 总计 {len(combined_df)} 条记录，覆盖 {combined_df['trade_date'].nunique()} 个交易日")

            return combined_df
        else:
            self.logger.warning("❌ 没有获取到任何数据")
            return pd.DataFrame()

    def update_hot_sectors(self, days_back: int = 30) -> pd.DataFrame:
        """
        增量更新最强板块数据

        Args:
            days_back: 向前更新的天数，默认30天

        Returns:
            更新的最强板块数据DataFrame
        """
        self.logger.info(f"开始增量更新最强板块数据 (向前{days_back}天)")

        # 计算更新日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')

        # 检查是否已有数据文件
        existing_files = list(self.base_path.glob("hot_sectors_*.parquet"))

        if existing_files:
            # 找到最新的数据文件
            latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"找到现有数据文件: {latest_file}")

            # 读取现有数据
            existing_df = pd.read_parquet(latest_file)

            # 获取现有数据的最新日期
            if not existing_df.empty:
                latest_date = existing_df['trade_date'].max()
                # 从最新日期的下一天开始更新
                start_dt = datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)
                start_date = start_dt.strftime('%Y%m%d')
                self.logger.info(f"现有数据最新日期: {latest_date}, 从 {start_date} 开始更新")

            # 下载新数据
            new_df = self.download_hot_sectors(start_date, end_date)

            if not new_df.empty:
                # 合并数据
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                # 去重并排序
                combined_df = combined_df.drop_duplicates(subset=['ts_code', 'trade_date'])
                combined_df = combined_df.sort_values(['trade_date', 'rank'])

                # 保存合并后的数据
                overall_start = combined_df['trade_date'].min()
                overall_end = combined_df['trade_date'].max()
                filename = f"hot_sectors_{overall_start}_{overall_end}.parquet"
                file_path = self.base_path / filename
                combined_df.to_parquet(file_path, index=False)

                # 删除旧文件
                if latest_file != file_path:
                    latest_file.unlink()
                    self.logger.info(f"已删除旧文件: {latest_file}")

                # 保存元数据
                metadata = {
                    'file_type': 'hot_sectors',
                    'start_date': overall_start,
                    'end_date': overall_end,
                    'total_records': len(combined_df),
                    'unique_dates': combined_df['trade_date'].nunique(),
                    'updated_at': datetime.now().isoformat()
                }

                metadata_file = self.base_path / f"hot_sectors_{overall_start}_{overall_end}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                self.logger.info(f"✅ 数据更新完成: {file_path}")
                return combined_df
            else:
                self.logger.info("没有新数据需要更新")
                return existing_df
        else:
            # 没有现有数据，直接下载
            self.logger.info("没有找到现有数据文件，开始全量下载")
            return self.download_hot_sectors(start_date, end_date)

    def get_stocks_by_sector_code(self, ts_code: str) -> list:
        """
        通过板块ts_code获取该板块内所有股票列表

        Args:
            ts_code: 板块代码 (如: 885728.TI)

        Returns:
            股票代码列表
        """
        try:
            # 读取概念成分文件
            members_file = self.base_path / "ths_concept_members.json"

            if not members_file.exists():
                self.logger.error(f"概念成分文件不存在: {members_file}")
                return []

            with open(members_file, 'r', encoding='utf-8') as f:
                members_data = json.load(f)

            # 从数据数组中筛选对应的板块
            stocks = []
            if 'data' in members_data:
                for item in members_data['data']:
                    if item.get('ts_code') == ts_code:
                        stocks.append(item.get('con_code'))

                if stocks:
                    self.logger.info(f"板块 {ts_code} 包含 {len(stocks)} 只股票")
                    return stocks
                else:
                    self.logger.warning(f"未找到板块 {ts_code} 的成分股信息")
                    return []
            else:
                self.logger.error("成分股文件格式错误，缺少data字段")
                return []

        except Exception as e:
            self.logger.error(f"获取板块成分股失败: {e}")
            return []

    def get_top_hot_sectors(self, trade_date: str, top_n: int = 6) -> list:
        """
        获取指定日期的最强板块前top_n名的ts_code列表

        Args:
            trade_date: 交易日期 (YYYYMMDD格式)
            top_n: 返回前几名，默认6名

        Returns:
            板块ts_code列表，按rank排序
        """
        try:
            # 查找最强板块数据文件
            hot_sector_files = list(self.base_path.glob("hot_sectors_*.parquet"))

            if not hot_sector_files:
                self.logger.error("未找到最强板块数据文件")
                return []

            # 找到最新的数据文件
            latest_file = max(hot_sector_files, key=lambda x: x.stat().st_mtime)

            # 读取数据
            df = pd.read_parquet(latest_file)

            # 筛选指定日期的数据
            date_data = df[df['trade_date'] == trade_date]

            if date_data.empty:
                self.logger.warning(f"未找到日期 {trade_date} 的最强板块数据")
                return []

            # 按rank排序，取前top_n名
            top_sectors = date_data.sort_values('rank').head(top_n)
            ts_codes = top_sectors['ts_code'].tolist()

            self.logger.info(f"获取到 {trade_date} 前{top_n}强板块: {ts_codes}")

            return ts_codes

        except Exception as e:
            self.logger.error(f"获取最强板块失败: {e}")
            return []


    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def download_ths_index(self, type_filter: str = 'N', exchange: str = 'A') -> pd.DataFrame:
        """
        下载同花顺概念板块基础信息
        
        Args:
            type_filter: 指数类型，N-概念指数 I-行业指数 R-地域指数 S-同花顺特色指数 等
            exchange: 市场类型，A-a股 HK-港股 US-美股
            
        Returns:
            概念板块基础信息DataFrame
        """
        self.logger.info(f"开始下载同花顺概念板块基础信息 (type={type_filter}, exchange={exchange})")
        
        try:
            # 调用tushare接口
            df = self.pro.ths_index(type=type_filter, exchange=exchange)
            
            if df.empty:
                self.logger.warning("未获取到概念板块数据")
                return df
                
            self.logger.info(f"成功获取 {len(df)} 个概念板块")
            
            # 保存到JSON文件
            output_file = self.base_path / f"ths_concept_index_{type_filter}_{exchange}.json"
            
            # 转换为适合保存的格式
            data_to_save = {
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': type_filter,
                'exchange': exchange,
                'count': len(df),
                'data': df.to_dict('records')
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"概念板块基础信息已保存到: {output_file}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"下载概念板块基础信息失败: {e}")
            raise
            
    def download_ths_members(self, concept_codes: List[str] = None) -> pd.DataFrame:
        """
        下载概念板块成分股信息
        
        Args:
            concept_codes: 概念板块代码列表，如果为None则从已保存的概念板块文件中读取
            
        Returns:
            概念板块成分股信息DataFrame
        """
        self.logger.info("开始下载概念板块成分股信息")
        
        # 如果没有提供概念代码，从文件中读取
        if concept_codes is None:
            concept_codes = self._load_concept_codes()
            
        if not concept_codes:
            self.logger.error("没有找到概念板块代码")
            return pd.DataFrame()
            
        all_members = []
        
        for i, ts_code in enumerate(concept_codes):
            try:
                self.logger.info(f"下载概念板块 {ts_code} 成分股 ({i+1}/{len(concept_codes)})")
                
                # 调用tushare接口
                df = self.pro.ths_member(ts_code=ts_code)
                
                if not df.empty:
                    all_members.append(df)
                    self.logger.info(f"概念板块 {ts_code} 包含 {len(df)} 只股票")
                else:
                    self.logger.warning(f"概念板块 {ts_code} 没有成分股数据")
                    
                # API调用间隔
                time.sleep(self.api_interval)
                
            except Exception as e:
                self.logger.error(f"下载概念板块 {ts_code} 成分股失败: {e}")
                continue
                
        if not all_members:
            self.logger.warning("未获取到任何成分股数据")
            return pd.DataFrame()
            
        # 合并所有数据
        result_df = pd.concat(all_members, ignore_index=True)
        
        self.logger.info(f"成功获取 {len(result_df)} 条成分股记录")
        
        # 保存数据
        self._save_members_data(result_df)
        
        return result_df
        
    def _load_concept_codes(self) -> List[str]:
        """从已保存的概念板块文件中加载概念代码"""
        concept_file = self.base_path / "ths_concept_index_N_A.json"
        
        if not concept_file.exists():
            self.logger.error(f"概念板块文件不存在: {concept_file}")
            return []
            
        try:
            with open(concept_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            codes = [item['ts_code'] for item in data['data']]
            self.logger.info(f"从文件中加载了 {len(codes)} 个概念板块代码")
            return codes
            
        except Exception as e:
            self.logger.error(f"加载概念板块代码失败: {e}")
            return []
            
    def _save_members_data(self, df: pd.DataFrame):
        """保存成分股数据"""
        # 保存为CSV格式，便于查看和处理
        csv_file = self.base_path / "ths_concept_members.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 同时保存为JSON格式
        json_file = self.base_path / "ths_concept_members.json"
        data_to_save = {
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_records': len(df),
            'unique_concepts': df['ts_code'].nunique(),
            'unique_stocks': df['con_code'].nunique(),
            'data': df.to_dict('records')
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"成分股数据已保存到: {csv_file} 和 {json_file}")
        
    def download_ths_daily(self, start_date: str = '20230101', end_date: str = None, 
                          concept_codes: List[str] = None) -> pd.DataFrame:
        """
        下载概念板块历史行情数据
        
        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD，默认为今天
            concept_codes: 概念板块代码列表，如果为None则从已保存的概念板块文件中读取
            
        Returns:
            概念板块历史行情数据DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        self.logger.info(f"开始下载概念板块历史行情数据 ({start_date} - {end_date})")
        
        # 如果没有提供概念代码，从文件中读取
        if concept_codes is None:
            concept_codes = self._load_concept_codes()
            
        if not concept_codes:
            self.logger.error("没有找到概念板块代码")
            return pd.DataFrame()
            
        all_daily_data = []
        
        for i, ts_code in enumerate(concept_codes):
            try:
                self.logger.info(f"下载概念板块 {ts_code} 行情数据 ({i+1}/{len(concept_codes)})")
                
                # 调用tushare接口
                df = self.pro.ths_daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not df.empty:
                    all_daily_data.append(df)
                    self.logger.info(f"概念板块 {ts_code} 获取 {len(df)} 条行情记录")
                else:
                    self.logger.warning(f"概念板块 {ts_code} 没有行情数据")
                    
                # API调用间隔
                time.sleep(self.api_interval)
                
            except Exception as e:
                self.logger.error(f"下载概念板块 {ts_code} 行情数据失败: {e}")
                continue
                
        if not all_daily_data:
            self.logger.warning("未获取到任何行情数据")
            return pd.DataFrame()
            
        # 合并所有数据
        result_df = pd.concat(all_daily_data, ignore_index=True)
        
        self.logger.info(f"成功获取 {len(result_df)} 条行情记录")
        
        # 保存数据
        self._save_daily_data(result_df, start_date, end_date)
        
        return result_df

    def _save_daily_data(self, df: pd.DataFrame, start_date: str, end_date: str):
        """保存行情数据"""
        # 保存为parquet格式，便于大数据量处理
        parquet_file = self.base_path / f"ths_concept_daily_{start_date}_{end_date}.parquet"
        df.to_parquet(parquet_file, index=False)

        # 保存元数据
        metadata = {
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_date': start_date,
            'end_date': end_date,
            'total_records': len(df),
            'unique_concepts': df['ts_code'].nunique(),
            'date_range': {
                'min_date': df['trade_date'].min(),
                'max_date': df['trade_date'].max()
            },
            'file_path': str(parquet_file)
        }

        metadata_file = self.base_path / f"ths_concept_daily_{start_date}_{end_date}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.logger.info(f"行情数据已保存到: {parquet_file}")
        self.logger.info(f"元数据已保存到: {metadata_file}")

    def update_daily_data(self, days_back: int = 30) -> pd.DataFrame:
        """
        增量更新行情数据

        Args:
            days_back: 向前更新的天数，默认30天

        Returns:
            更新的行情数据DataFrame
        """
        self.logger.info(f"开始增量更新行情数据 (向前{days_back}天)")

        # 计算更新日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')

        # 检查是否已有数据文件
        existing_files = list(self.base_path.glob("ths_concept_daily_*.parquet"))

        if existing_files:
            # 找到最新的数据文件
            latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"找到最新数据文件: {latest_file}")

            # 读取最新数据的日期范围
            try:
                existing_df = pd.read_parquet(latest_file)
                if not existing_df.empty:
                    latest_date = existing_df['trade_date'].max()
                    # 从最新日期的下一天开始更新
                    start_date = (datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
                    self.logger.info(f"从 {start_date} 开始增量更新")
            except Exception as e:
                self.logger.warning(f"读取现有数据失败，将进行全量更新: {e}")

        # 如果开始日期大于等于结束日期，说明数据已是最新
        if start_date >= end_date:
            self.logger.info("数据已是最新，无需更新")
            return pd.DataFrame()

        # 下载增量数据
        new_data = self.download_ths_daily(start_date=start_date, end_date=end_date)

        if not new_data.empty and existing_files:
            # 合并新旧数据
            try:
                old_data = pd.read_parquet(latest_file)
                combined_data = pd.concat([old_data, new_data], ignore_index=True)

                # 去重并排序
                combined_data = combined_data.drop_duplicates(subset=['ts_code', 'trade_date'])
                combined_data = combined_data.sort_values(['ts_code', 'trade_date'])

                # 保存合并后的数据
                full_start = old_data['trade_date'].min()
                self._save_daily_data(combined_data, full_start, end_date)

                self.logger.info(f"增量更新完成，合并数据共 {len(combined_data)} 条记录")
                return combined_data

            except Exception as e:
                self.logger.error(f"合并数据失败: {e}")
                return new_data

        return new_data

    def get_concept_summary(self) -> Dict[str, Any]:
        """
        获取概念板块数据摘要

        Returns:
            数据摘要字典
        """
        summary = {
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'concept_index': {},
            'concept_members': {},
            'daily_data': {}
        }

        # 概念板块基础信息摘要
        concept_file = self.base_path / "ths_concept_index_N_A.json"
        if concept_file.exists():
            try:
                with open(concept_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                summary['concept_index'] = {
                    'file_exists': True,
                    'update_time': data.get('update_time'),
                    'count': data.get('count'),
                    'type': data.get('type'),
                    'exchange': data.get('exchange')
                }
            except Exception as e:
                summary['concept_index'] = {'file_exists': True, 'error': str(e)}
        else:
            summary['concept_index'] = {'file_exists': False}

        # 成分股信息摘要
        members_file = self.base_path / "ths_concept_members.json"
        if members_file.exists():
            try:
                with open(members_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                summary['concept_members'] = {
                    'file_exists': True,
                    'update_time': data.get('update_time'),
                    'total_records': data.get('total_records'),
                    'unique_concepts': data.get('unique_concepts'),
                    'unique_stocks': data.get('unique_stocks')
                }
            except Exception as e:
                summary['concept_members'] = {'file_exists': True, 'error': str(e)}
        else:
            summary['concept_members'] = {'file_exists': False}

        # 行情数据摘要
        daily_files = list(self.base_path.glob("ths_concept_daily_*.parquet"))
        if daily_files:
            latest_file = max(daily_files, key=lambda x: x.stat().st_mtime)
            try:
                df = pd.read_parquet(latest_file)
                summary['daily_data'] = {
                    'file_exists': True,
                    'latest_file': latest_file.name,
                    'total_records': len(df),
                    'unique_concepts': df['ts_code'].nunique(),
                    'date_range': {
                        'min_date': df['trade_date'].min(),
                        'max_date': df['trade_date'].max()
                    }
                }
            except Exception as e:
                summary['daily_data'] = {'file_exists': True, 'error': str(e)}
        else:
            summary['daily_data'] = {'file_exists': False}

        return summary
