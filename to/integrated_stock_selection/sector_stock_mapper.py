#!/usr/bin/env python3
"""
板块-个股映射器
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class SectorStockMapper:
    """板块-个股映射器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._stock_sector_cache = {}  # 缓存股票-板块映射
        self._sector_stock_cache = {}  # 缓存板块-股票映射
        self._data_cache = None  # 缓存原始数据
        self._data_loader_imported = False  # 标记是否已导入数据加载器
        self._get_stock_sectors_func = None  # 缓存函数引用
        self._get_stock_list_in_sector_func = None  # 缓存函数引用
        self._global_stock_mapping = None  # 全局股票映射缓存
        self._mapping_loaded = False  # 标记是否已加载全局映射
        
    def _ensure_data_loader_imported(self):
        """确保数据加载器已导入"""
        if not self._data_loader_imported:
            try:
                sys.path.append(str(project_root))
                from data_loaders import get_stock_sectors, get_stock_list_in_sector
                self._get_stock_sectors_func = get_stock_sectors
                self._get_stock_list_in_sector_func = get_stock_list_in_sector
                self._data_loader_imported = True
                self.logger.info("数据加载器导入成功")
            except ImportError as e:
                self.logger.error(f"导入数据加载器失败: {e}")
                raise

    def _load_sector_data_once(self):
        """一次性加载所有板块数据"""
        if self._data_cache is None:
            try:
                # 导入并加载原始数据
                sys.path.append(str(project_root / "to"))
                from stock_sectors_mapping import load_data
                self._data_cache = load_data()
                if self._data_cache:
                    self.logger.info("板块数据一次性加载完成")
                else:
                    self.logger.warning("板块数据加载失败")
            except Exception as e:
                self.logger.error(f"加载板块数据失败: {e}")
                self._data_cache = {}

    def _load_global_mapping_once(self):
        """一次性加载全局股票-板块映射（程序运行期间只加载一次）"""
        if not self._mapping_loaded:
            try:
                self.logger.info("开始加载全局股票-板块映射...")

                # 确保数据加载器已导入
                self._ensure_data_loader_imported()

                # 获取所有股票的板块信息
                if self._get_stock_sectors_func:
                    # 从现有数据中获取股票代码，避免遍历所有可能的代码
                    all_stock_codes = self._get_existing_stock_codes()

                    self.logger.info(f"开始构建{len(all_stock_codes)}只股票的全局板块映射")

                    global_mapping = {}
                    failed_count = 0

                    for i, stock_code in enumerate(all_stock_codes):
                        try:
                            sectors = self._get_stock_sectors_func(stock_code)
                            if sectors:
                                global_mapping[stock_code] = sectors
                            else:
                                failed_count += 1
                        except Exception as e:
                            failed_count += 1
                            if failed_count <= 10:  # 只记录前10个错误
                                self.logger.debug(f"获取股票{stock_code}板块失败: {e}")

                        # 每1000只股票显示一次进度
                        if (i + 1) % 1000 == 0:
                            self.logger.info(f"全局映射进度: {i + 1}/{len(all_stock_codes)}")

                    self._global_stock_mapping = global_mapping
                    self._mapping_loaded = True

                    self.logger.info(f"全局股票-板块映射加载完成: 成功{len(global_mapping)}只, 失败{failed_count}只")
                else:
                    self.logger.error("数据加载器函数未初始化")
                    self._global_stock_mapping = {}
                    self._mapping_loaded = True

            except Exception as e:
                self.logger.error(f"加载全局映射失败: {e}")
                self._global_stock_mapping = {}
                self._mapping_loaded = True

    def _get_existing_stock_codes(self) -> List[str]:
        """从现有数据源获取股票代码列表"""
        try:
            # 尝试从数据文件中获取实际存在的股票代码
            stock_codes = set()

            # 方法1：从缓存的板块数据中获取
            if self._data_cache:
                for sector_data in self._data_cache.values():
                    if isinstance(sector_data, pd.DataFrame) and 'stock_code' in sector_data.columns:
                        stock_codes.update(sector_data['stock_code'].unique())

            # 方法2：如果缓存为空，尝试从数据加载器获取一个样本
            if not stock_codes:
                try:
                    # 导入数据加载器获取股票列表
                    sys.path.append(str(project_root / "to"))
                    from data_loaders import get_all_stock_codes
                    stock_codes = set(get_all_stock_codes())
                except ImportError:
                    # 如果无法导入，使用一个较小的常用股票代码范围
                    self.logger.warning("无法获取完整股票列表，使用默认范围")
                    for i in range(1, 1000):  # 减少范围以提高性能
                        stock_codes.add(f"{i:06d}.SZ")
                    for i in range(600000, 601000):
                        stock_codes.add(f"{i}.SH")

            return list(stock_codes)

        except Exception as e:
            self.logger.error(f"获取股票代码列表失败: {e}")
            return []

    def get_stock_sectors(self, stock_code: str) -> List[str]:
        """获取股票所属的板块列表 - 优化版本"""
        try:
            # 检查缓存
            if stock_code in self._stock_sector_cache:
                return self._stock_sector_cache[stock_code]

            # 确保数据已加载
            self._load_sector_data_once()

            # 直接从缓存数据中查询，避免重复加载
            if self._data_cache and "stock_to_sectors" in self._data_cache:
                stock_info = self._data_cache["stock_to_sectors"].get(stock_code)
                if stock_info:
                    sectors = stock_info.get("sectors", [])
                else:
                    sectors = []
            else:
                # 回退到原始方法
                self._ensure_data_loader_imported()
                sectors = self._get_stock_sectors_func(stock_code) if self._get_stock_sectors_func else []

            # 缓存结果
            self._stock_sector_cache[stock_code] = sectors
            return sectors

        except Exception as e:
            self.logger.error(f"获取股票{stock_code}的板块失败: {e}")
            return []

    def _assign_default_sectors(self, stock_code: str) -> List[str]:
        """为新股分配默认板块"""
        try:
            # 根据股票代码前缀分配默认板块
            if stock_code.startswith('688'):
                # 科创板
                return ['TGN科创板', 'TGN新股']
            elif stock_code.startswith('301'):
                # 创业板注册制
                return ['TGN创业板', 'TGN新股']
            elif stock_code.startswith('003'):
                # 创业板新股
                return ['TGN创业板', 'TGN新股']
            elif stock_code.startswith('001'):
                # 深市主板新股
                return ['TGN深市主板', 'TGN新股']
            elif stock_code.startswith('603') or stock_code.startswith('605'):
                # 沪市主板新股
                return ['TGN沪市主板', 'TGN新股']
            elif stock_code.startswith('920') or stock_code.endswith('.BJ'):
                # 北交所
                return ['TGN北交所', 'TGN新股']
            else:
                # 其他新股
                return ['TGN新股']

        except Exception as e:
            self.logger.error(f"分配默认板块失败 {stock_code}: {e}")
            return []
    
    def get_sector_stocks(self, sector_code: str) -> List[str]:
        """获取板块包含的股票列表 - 优化版本"""
        try:
            # 检查缓存
            if sector_code in self._sector_stock_cache:
                return self._sector_stock_cache[sector_code]

            # 确保数据已加载
            self._load_sector_data_once()

            # 直接从缓存数据中查询，避免重复加载
            if self._data_cache and "sector_to_stocks" in self._data_cache:
                stocks = self._data_cache["sector_to_stocks"].get(sector_code, [])
            else:
                # 回退到原始方法
                self._ensure_data_loader_imported()
                stocks = self._get_stock_list_in_sector_func(sector_code) if self._get_stock_list_in_sector_func else []

            # 缓存结果
            self._sector_stock_cache[sector_code] = stocks
            return stocks

        except Exception as e:
            self.logger.error(f"获取板块{sector_code}的股票失败: {e}")
            return []
    
    def build_stock_sector_mapping(self, stock_codes: List[str]) -> Dict[str, List[str]]:
        """批量构建股票-板块映射 - 使用全局缓存优化版本"""
        try:
            # 确保全局映射已加载（只在第一次调用时加载）
            self._load_global_mapping_once()

            # 如果全局映射可用，直接从中提取需要的股票
            if self._global_stock_mapping:
                mapping = {}
                for stock_code in stock_codes:
                    if stock_code in self._global_stock_mapping:
                        mapping[stock_code] = self._global_stock_mapping[stock_code]

                self.logger.info(f"从全局缓存获取映射: 成功{len(mapping)}只, 总需求{len(stock_codes)}只")
                return mapping

            # 如果全局映射不可用，回退到原有逻辑
            self.logger.warning("全局映射不可用，使用原有逻辑")
            self.logger.info(f"构建{len(stock_codes)}只股票的板块映射")

            # 一次性加载所有数据
            self._load_sector_data_once()

            mapping = {}
            failed_count = 0

            # 如果有缓存数据，直接批量处理
            if self._data_cache and "stock_to_sectors" in self._data_cache:
                stock_to_sectors = self._data_cache["stock_to_sectors"]

                for i, stock_code in enumerate(stock_codes):
                    if i % 1000 == 0:
                        self.logger.info(f"处理进度: {i}/{len(stock_codes)}")

                    # 直接从缓存查询
                    if stock_code in stock_to_sectors:
                        sectors = stock_to_sectors[stock_code].get("sectors", [])
                        if sectors:
                            mapping[stock_code] = sectors
                            # 同时更新本地缓存
                            self._stock_sector_cache[stock_code] = sectors
                        else:
                            # 为没有板块信息的股票分配默认板块
                            default_sectors = self._assign_default_sectors(stock_code)
                            if default_sectors:
                                mapping[stock_code] = default_sectors
                                self._stock_sector_cache[stock_code] = default_sectors
                            else:
                                failed_count += 1
                    else:
                        # 为新股分配默认板块
                        default_sectors = self._assign_default_sectors(stock_code)
                        if default_sectors:
                            mapping[stock_code] = default_sectors
                            self._stock_sector_cache[stock_code] = default_sectors
                        else:
                            failed_count += 1
            else:
                # 回退到逐个查询
                for i, stock_code in enumerate(stock_codes):
                    if i % 1000 == 0:
                        self.logger.info(f"处理进度: {i}/{len(stock_codes)}")

                    sectors = self.get_stock_sectors(stock_code)
                    if sectors:
                        mapping[stock_code] = sectors
                    else:
                        # 为新股分配默认板块
                        default_sectors = self._assign_default_sectors(stock_code)
                        if default_sectors:
                            mapping[stock_code] = default_sectors
                        else:
                            failed_count += 1

            self.logger.info(f"映射构建完成: 成功{len(mapping)}只, 失败{failed_count}只")
            return mapping

        except Exception as e:
            self.logger.error(f"构建股票板块映射失败: {e}")
            return {}
    
    def build_sector_stock_mapping(self, sector_codes: List[str]) -> Dict[str, List[str]]:
        """批量构建板块-股票映射 - 优化版本"""
        try:
            self.logger.info(f"构建{len(sector_codes)}个板块的股票映射")

            # 一次性加载所有数据
            self._load_sector_data_once()

            mapping = {}
            failed_count = 0

            # 如果有缓存数据，直接批量处理
            if self._data_cache and "sector_to_stocks" in self._data_cache:
                sector_to_stocks = self._data_cache["sector_to_stocks"]

                for i, sector_code in enumerate(sector_codes):
                    if i % 100 == 0:
                        self.logger.info(f"处理进度: {i}/{len(sector_codes)}")

                    # 直接从缓存查询
                    if sector_code in sector_to_stocks:
                        stocks = sector_to_stocks[sector_code]
                        if stocks:
                            mapping[sector_code] = stocks
                            # 同时更新本地缓存
                            self._sector_stock_cache[sector_code] = stocks
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
            else:
                # 回退到逐个查询
                for i, sector_code in enumerate(sector_codes):
                    if i % 100 == 0:
                        self.logger.info(f"处理进度: {i}/{len(sector_codes)}")

                    stocks = self.get_sector_stocks(sector_code)
                    if stocks:
                        mapping[sector_code] = stocks
                    else:
                        failed_count += 1

            self.logger.info(f"映射构建完成: 成功{len(mapping)}个, 失败{failed_count}个")
            return mapping

        except Exception as e:
            self.logger.error(f"构建板块股票映射失败: {e}")
            return {}
    
    def calculate_sector_overlap_scores(self, stock_data: pd.DataFrame, 
                                      selected_sectors: List[str]) -> pd.DataFrame:
        """计算个股的板块重叠评分"""
        try:
            self.logger.info(f"计算板块重叠评分，选中板块数: {len(selected_sectors)}")
            
            # 获取所有股票代码
            stock_codes = stock_data['stock_code'].unique().tolist()
            
            # 构建股票-板块映射
            stock_sector_mapping = self.build_stock_sector_mapping(stock_codes)
            
            # 计算重叠评分
            overlap_scores = {}
            selected_sector_set = set(selected_sectors)
            
            for stock_code, sectors in stock_sector_mapping.items():
                stock_sector_set = set(sectors)
                
                # 计算重叠数量
                overlap_count = len(stock_sector_set & selected_sector_set)
                
                # 计算重叠比例
                if len(stock_sector_set) > 0:
                    overlap_ratio = overlap_count / len(stock_sector_set)
                else:
                    overlap_ratio = 0
                
                # 综合评分：重叠数量 + 重叠比例
                overlap_score = overlap_count * 10 + overlap_ratio * 20
                
                overlap_scores[stock_code] = {
                    'overlap_count': overlap_count,
                    'overlap_ratio': overlap_ratio,
                    'overlap_score': overlap_score,
                    'total_sectors': len(stock_sector_set),
                    'sectors': sectors
                }
            
            # 转换为DataFrame
            overlap_df = pd.DataFrame.from_dict(overlap_scores, orient='index')
            overlap_df.reset_index(inplace=True)
            overlap_df.rename(columns={'index': 'stock_code'}, inplace=True)
            
            self.logger.info(f"重叠评分计算完成: {len(overlap_df)}只股票")
            
            return overlap_df
            
        except Exception as e:
            self.logger.error(f"计算板块重叠评分失败: {e}")
            return pd.DataFrame()
    
    def get_sector_quality_weights(self, sector_scores: Dict[str, float]) -> Dict[str, float]:
        """根据板块评分计算质量权重"""
        try:
            if not sector_scores:
                return {}
            
            # 计算权重（评分越高权重越大）
            max_score = max(sector_scores.values())
            min_score = min(sector_scores.values())
            
            if max_score == min_score:
                # 所有板块评分相同，等权重
                return {sector: 1.0 for sector in sector_scores.keys()}
            
            # 归一化权重
            weights = {}
            for sector, score in sector_scores.items():
                # 线性归一化到0.5-1.5范围
                normalized_weight = 0.5 + (score - min_score) / (max_score - min_score)
                weights[sector] = normalized_weight
            
            return weights
            
        except Exception as e:
            self.logger.error(f"计算板块质量权重失败: {e}")
            return {}
    
    def enhance_stock_features_with_sectors(self, stock_data: pd.DataFrame,
                                          sector_results: List,
                                          top_n: int = 10) -> pd.DataFrame:
        """
        基于板块过滤个股 - 只保留与前N个板块有重叠的股票

        修改逻辑：
        1. 提取前N个优质板块
        2. 过滤掉与这些板块没有重叠的个股
        3. 不再计算复杂的板块特征，简化为过滤逻辑
        """
        try:
            self.logger.info(f"基于前{top_n}个板块过滤个股")

            # 提取前N个板块
            selected_sectors = [result.sector_code for result in sector_results[:top_n]]
            self.logger.info(f"选中的板块: {selected_sectors}")

            # 过滤股票：只保留与选中板块有重叠的股票
            filtered_stocks = []
            filtered_count = 0

            for _, stock in stock_data.iterrows():
                stock_code = stock['stock_code']
                stock_sectors = self.get_stock_sectors(stock_code)

                # 检查是否与选中板块有重叠
                has_overlap = any(sector in selected_sectors for sector in stock_sectors)

                if has_overlap:
                    filtered_stocks.append(stock)
                else:
                    filtered_count += 1

            if not filtered_stocks:
                self.logger.warning("没有股票与选中板块重叠，返回原始数据")
                return stock_data

            # 转换为DataFrame
            filtered_data = pd.DataFrame(filtered_stocks)

            self.logger.info(f"板块过滤完成: {len(stock_data)} -> {len(filtered_data)} (过滤{filtered_count}只)")

            return filtered_data

        except Exception as e:
            self.logger.error(f"板块过滤失败: {e}")
            return stock_data
