#!/usr/bin/env python3
"""
选股引擎
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config import system_config, sector_config
from data_loader import IntegratedDataLoader
from sector_stock_mapper import SectorStockMapper
from feature_engineer import FeatureEngineer
from multi_factor_scorer import MultiFactorScorer
from enhanced_stock_scorer import EnhancedStockScorer
from sector_backtester import SectorBacktester

# 导入板块分析器
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "clean_ml_system"))

# 导入板块分析器 - 使用ultimate_multi_day_analysis.py
try:
    # 导入ultimate_multi_day_analysis模块
    from ultimate_multi_day_analysis import UltimateMultiDayAnalyzer

except Exception as e:
    # 如果导入失败，创建一个简化版本
    print(f"导入UltimateMultiDayAnalyzer失败: {e}")
    class UltimateMultiDayAnalyzer:
        def analyze_enhanced_daily_potential(self, daily_sectors, market_features):
            return []

@dataclass
class StockSelection:
    """选股结果"""
    date: str
    stock_code: str
    stock_name: str
    composite_score: float
    risk_adjusted_score: float
    technical_score: float
    fundamental_score: float
    sector_score: float
    capital_flow_score: float
    selected_sectors: List[str]
    confidence: float
    expected_return: float

class StockSelector:
    """选股引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_loader = IntegratedDataLoader()
        self.sector_mapper = SectorStockMapper()
        self.feature_engineer = FeatureEngineer()
        self.multi_factor_scorer = MultiFactorScorer()
        self.enhanced_scorer = EnhancedStockScorer()
        self.sector_backtester = SectorBacktester()
        self.sector_analyzer = UltimateMultiDayAnalyzer()
    
    def select_daily_stocks(self, date: str, stock_data: pd.DataFrame,
                          sector_data: pd.DataFrame, market_data: pd.DataFrame) -> List[StockSelection]:
        """每日选股"""
        try:
            self.logger.info(f"执行{date}选股")

            # 检查数据完整性：确保有当日和前一日数据
            current_date_obj = pd.to_datetime(date)
            available_dates = sorted(stock_data['date'].unique())

            if current_date_obj not in available_dates:
                self.logger.error(f"没有{date}的股票数据")
                return []

            # 检查是否有前一日数据（用于K线形态特征和强势股相似性计算）
            current_date_index = None
            for i, d in enumerate(available_dates):
                if d == current_date_obj:
                    current_date_index = i
                    break

            if current_date_index is None or current_date_index == 0:
                self.logger.warning(f"没有{date}的前一交易日数据，可能影响特征计算质量")

            # 1. 板块潜力分析
            daily_sectors = sector_data[sector_data['date'] == pd.to_datetime(date)]
            daily_market = market_data[market_data['date'] == pd.to_datetime(date)]

            if len(daily_sectors) == 0:
                self.logger.warning(f"{date}没有板块数据")
                return []
            
            # 处理市场数据格式
            if len(daily_market) > 0:
                market_dict = daily_market.iloc[0].to_dict()
            else:
                market_dict = {}

            sector_results = self.sector_analyzer.analyze_enhanced_daily_potential(
                daily_sectors, market_dict
            )
            
            if not sector_results:
                self.logger.warning(f"{date}板块分析失败")
                return []
            
            # 2. 个股特征增强
            daily_stocks = stock_data[stock_data['date'] == pd.to_datetime(date)]
            
            if len(daily_stocks) == 0:
                self.logger.warning(f"{date}没有个股数据")
                return []
            
            enhanced_stocks = self.sector_mapper.enhance_stock_features_with_sectors(
                daily_stocks, sector_results, top_n=system_config.sector_filter_count
            )
            
            # 3. 特征工程（stock_data已包含完整的K线形态特征）
            engineered_stocks = self.feature_engineer.engineer_all_features(enhanced_stocks)

            # 4. 增强个股评分（强势股相似性和涨幅惩罚）
            # 注意：stock_data现在已经是特征完整的数据，包含历史数据和K线形态特征
            enhanced_scored_stocks = self.enhanced_scorer.calculate_enhanced_scores(
                stock_data, sector_results, date
            )

            # 5. 合并工程化特征和增强评分到当日数据
            current_date_obj = pd.to_datetime(date)
            current_enhanced_scores = enhanced_scored_stocks[enhanced_scored_stocks['date'] == current_date_obj].copy()

            # 确保关键评分列存在
            required_score_columns = ['limit_up_similarity_score', 'consecutive_limit_penalty', 'cumulative_return_10d']

            if len(current_enhanced_scores) > 0 and len(engineered_stocks) > 0:
                # 获取所有评分相关列
                available_score_columns = []
                for col in current_enhanced_scores.columns:
                    if col in required_score_columns or 'score' in col.lower() or 'penalty' in col.lower() or 'cumulative_return' in col.lower():
                        available_score_columns.append(col)

                self.logger.info(f"合并评分列: {available_score_columns}")

                # 检查关键列是否存在
                missing_columns = [col for col in required_score_columns if col not in current_enhanced_scores.columns]
                if missing_columns:
                    self.logger.error(f"增强评分数据中缺少关键列: {missing_columns}")
                    # 在enhanced_scores中添加缺失列
                    for col in missing_columns:
                        current_enhanced_scores[col] = 0
                        available_score_columns.append(col)

                # 检查merge前的数据
                self.logger.info(f"merge前 - engineered_stocks: {len(engineered_stocks)}只股票")
                self.logger.info(f"merge前 - current_enhanced_scores: {len(current_enhanced_scores)}只股票")

                # 检查stock_code重叠情况
                eng_codes = set(engineered_stocks['stock_code'].unique())
                enh_codes = set(current_enhanced_scores['stock_code'].unique())
                overlap_codes = eng_codes & enh_codes
                self.logger.info(f"stock_code重叠: {len(overlap_codes)}/{len(eng_codes)}只股票")

                # 合并数据
                merge_columns = ['stock_code'] + available_score_columns
                self.logger.info(f"准备merge的列: {merge_columns}")

                # 确保merge列都存在
                actual_merge_columns = ['stock_code']
                for col in available_score_columns:
                    if col in current_enhanced_scores.columns:
                        actual_merge_columns.append(col)
                    else:
                        self.logger.warning(f"列{col}在enhanced_scores中不存在")

                # 检查列名冲突
                eng_columns = set(engineered_stocks.columns)
                enh_columns = set(current_enhanced_scores.columns)
                conflict_columns = eng_columns & enh_columns - {'stock_code'}
                if conflict_columns:
                    self.logger.warning(f"发现列名冲突: {conflict_columns}")

                # 执行merge，确保增强评分列不被覆盖
                merged_stocks = engineered_stocks.merge(
                    current_enhanced_scores[actual_merge_columns],
                    on='stock_code', how='left', suffixes=('_original', '_enhanced')
                )

                self.logger.info(f"merge后: {len(merged_stocks)}只股票, {len(merged_stocks.columns)}列")

                # 检查merge后的列
                merge_result_columns = set(merged_stocks.columns)
                self.logger.info(f"merge后包含的评分列: {[col for col in merge_result_columns if 'score' in col.lower() or 'penalty' in col.lower()]}")

                # 验证合并结果并处理列名
                for col in required_score_columns:
                    # 检查可能的列名：原列名、带enhanced后缀、带original后缀
                    possible_names = [col, f"{col}_enhanced", f"{col}_original"]
                    found_col = None
                    best_col = None

                    # 优先选择_enhanced列（包含最新的增强评分）
                    for name in [f"{col}_enhanced", col, f"{col}_original"]:
                        if name in merged_stocks.columns:
                            found_col = name
                            # 检查数据质量
                            non_null_count = merged_stocks[name].notna().sum()
                            non_zero_count = (merged_stocks[name] > 0).sum()
                            if non_zero_count > 0:  # 优先选择有非零数据的列
                                best_col = name
                                break
                            elif found_col is None:
                                found_col = name

                    if best_col:
                        final_col = best_col
                    elif found_col:
                        final_col = found_col
                    else:
                        final_col = None

                    if final_col:
                        non_null_count = merged_stocks[final_col].notna().sum()
                        non_zero_count = (merged_stocks[final_col] > 0).sum()
                        avg_val = merged_stocks[final_col].mean()
                        self.logger.info(f"评分列{col}(使用:{final_col}): {non_null_count}/{len(merged_stocks)}只股票有数据, {non_zero_count}只>0, 平均{avg_val:.2f}")

                        # 确保使用正确的列名
                        if final_col != col:
                            merged_stocks[col] = merged_stocks[final_col]
                            # 清理其他相关列
                            for name in possible_names:
                                if name != col and name in merged_stocks.columns:
                                    merged_stocks.drop(columns=[name], inplace=True)
                    else:
                        self.logger.error(f"合并后仍缺少评分列{col}")
                        merged_stocks[col] = 0
            else:
                self.logger.warning("增强评分数据为空，使用默认值")
                merged_stocks = engineered_stocks.copy()
                # 设置默认值
                for col in required_score_columns:
                    merged_stocks[col] = 0

            # 最终检查和填充NaN值
            for col in required_score_columns:
                if col in merged_stocks.columns:
                    merged_stocks[col] = merged_stocks[col].fillna(0)
                else:
                    merged_stocks[col] = 0

            # 6. 多因子评分（特征已完整，直接计算评分）
            scored_stocks = self.multi_factor_scorer.calculate_scores_only(merged_stocks)
            
            # 6. 过滤和排序
            filtered_stocks = self.multi_factor_scorer.filter_stocks_by_criteria(scored_stocks, date)
            ranked_stocks = self.multi_factor_scorer.rank_stocks_by_score(filtered_stocks)

            # 7. 选择最优股票
            selected_stocks = self._select_top_stocks(
                ranked_stocks, sector_results, date, system_config.daily_stock_count
            )
            
            self.logger.info(f"{date}选股完成: 选中{len(selected_stocks)}只股票")
            
            return selected_stocks
            
        except Exception as e:
            self.logger.error(f"{date}选股失败: {e}")
            return []
    
    def _select_top_stocks(self, ranked_stocks: pd.DataFrame, sector_results: List,
                          date: str, count: int) -> List[StockSelection]:
        """
        选择最优股票 - 适配板块过滤后的逻辑

        注意：此时所有股票都已经通过板块过滤，都与前N个优质板块相关
        重点是在这些优质股票中进行多样化选择，避免过度集中在某个板块

        参数:
        - sector_results: 保留用于兼容性，但在当前逻辑中不再使用
        """
        try:
            # sector_results参数保留用于兼容性，但不再使用
            _ = sector_results  # 避免未使用变量警告

            if len(ranked_stocks) == 0:
                return []

            # 按风险调整评分排序
            top_stocks = ranked_stocks.nlargest(count * 3, 'risk_adjusted_score')

            # 多样化选择（在优质板块内进行均衡分配）
            selected = []
            used_sectors = set()

            for _, stock in top_stocks.iterrows():
                if len(selected) >= count:
                    break

                # 获取股票所属板块
                stock_sectors = self.sector_mapper.get_stock_sectors(stock['stock_code'])

                # 计算与已选股票的板块重叠度
                sector_overlap = len(set(stock_sectors) & used_sectors)

                # 调整选择逻辑：由于都是优质板块的股票，重点是避免过度集中
                # 允许适度重叠（<=2），或者是评分特别高的股票
                if sector_overlap <= 2 or stock['risk_adjusted_score'] > 75:

                    # 计算置信度
                    confidence = self._calculate_confidence(stock)

                    # 计算预期收益
                    expected_return = self._calculate_expected_return(stock)
                    
                    selection = StockSelection(
                        date=date,
                        stock_code=stock['stock_code'],
                        stock_name=stock.get('stock_name', stock['stock_code']),
                        composite_score=stock.get('composite_score', 0),
                        risk_adjusted_score=stock.get('risk_adjusted_score', 0),
                        technical_score=stock.get('technical_score', 0),
                        fundamental_score=stock.get('fundamental_score', 0),
                        sector_score=stock.get('sector_score', 0),
                        capital_flow_score=stock.get('capital_flow_score', 0),
                        selected_sectors=stock_sectors[:3],  # 前3个板块
                        confidence=confidence,
                        expected_return=expected_return
                    )
                    
                    selected.append(selection)
                    used_sectors.update(stock_sectors[:2])  # 记录前2个板块
            
            # 如果选择不足，补充高评分股票
            if len(selected) < count:
                remaining_stocks = top_stocks[~top_stocks['stock_code'].isin([s.stock_code for s in selected])]
                
                for _, stock in remaining_stocks.iterrows():
                    if len(selected) >= count:
                        break
                    
                    stock_sectors = self.sector_mapper.get_stock_sectors(stock['stock_code'])
                    confidence = self._calculate_confidence(stock)
                    expected_return = self._calculate_expected_return(stock)
                    
                    selection = StockSelection(
                        date=date,
                        stock_code=stock['stock_code'],
                        stock_name=stock.get('stock_name', stock['stock_code']),
                        composite_score=stock.get('composite_score', 0),
                        risk_adjusted_score=stock.get('risk_adjusted_score', 0),
                        technical_score=stock.get('technical_score', 0),
                        fundamental_score=stock.get('fundamental_score', 0),
                        sector_score=stock.get('sector_score', 0),
                        capital_flow_score=stock.get('capital_flow_score', 0),
                        selected_sectors=stock_sectors[:3],
                        confidence=confidence,
                        expected_return=expected_return
                    )
                    
                    selected.append(selection)
            
            return selected
            
        except Exception as e:
            self.logger.error(f"选择最优股票失败: {e}")
            return []
    
    def _calculate_confidence(self, stock: pd.Series) -> float:
        """
        计算选股置信度 - 适配板块过滤后的逻辑

        注意：不再使用overlap_score，因为所有股票都已通过板块过滤
        改为使用sector_score来评估板块相关性
        """
        try:
            confidence_factors = []

            # 评分置信度
            risk_adjusted_score = stock.get('risk_adjusted_score', 0)
            if risk_adjusted_score > 80:
                confidence_factors.append(0.9)
            elif risk_adjusted_score > 60:
                confidence_factors.append(0.7)
            elif risk_adjusted_score > 40:
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.3)

            # 板块置信度 - 使用sector_score替代overlap_score
            sector_score = stock.get('sector_score', 0)
            if sector_score > 60:
                confidence_factors.append(0.8)
            elif sector_score > 40:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)

            # 技术面置信度
            technical_score = stock.get('technical_score', 0)
            if technical_score > 70:
                confidence_factors.append(0.8)
            elif technical_score > 50:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)

            # 资金面置信度
            capital_flow_score = stock.get('capital_flow_score', 0)
            if capital_flow_score > 70:
                confidence_factors.append(0.8)
            elif capital_flow_score > 50:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)

            # 计算综合置信度
            confidence = np.mean(confidence_factors)

            return min(0.95, max(0.1, confidence))

        except Exception as e:
            self.logger.error(f"计算置信度失败: {e}")
            return 0.5
    
    def _calculate_expected_return(self, stock: pd.Series) -> float:
        """计算预期收益率"""
        try:
            # 基于历史数据和评分计算预期收益
            base_return = 0.01  # 基础预期收益1%
            
            # 评分加成
            risk_adjusted_score = stock.get('risk_adjusted_score', 0)
            score_bonus = (risk_adjusted_score - 50) * 0.0002  # 评分每高1分，预期收益增加0.02%
            
            # 技术面加成
            technical_score = stock.get('technical_score', 0)
            technical_bonus = (technical_score - 50) * 0.0001
            
            # 板块加成
            sector_score = stock.get('sector_score', 0)
            sector_bonus = (sector_score - 50) * 0.0001
            
            # 动量加成
            momentum_5d = stock.get('price_momentum_5d', 0)
            momentum_bonus = momentum_5d * 0.5  # 动量直接影响预期收益
            
            expected_return = base_return + score_bonus + technical_bonus + sector_bonus + momentum_bonus
            
            # 限制在合理范围内
            return min(0.1, max(-0.05, expected_return))
            
        except Exception as e:
            self.logger.error(f"计算预期收益失败: {e}")
            return 0.01
    

    def run_stock_selection(self, start_date: str, end_date: str) -> List[StockSelection]:
        """运行选股流程 - 优化版本，预先计算所有特征"""
        try:
            self.logger.info(f"运行选股流程: {start_date} - {end_date}")

            # 加载数据（自动扩展历史数据范围）
            stock_data, sector_data, market_data = self.data_loader.load_all_data(start_date, end_date, extend_history_days=30)

            if stock_data is None or sector_data is None:
                self.logger.error("数据加载失败")
                return []

            # 获取交易日
            trading_dates = self.data_loader.get_trading_dates(start_date, end_date)

            self.logger.info(f"数据加载完成: 股票数据{len(stock_data)}条，交易日{len(trading_dates)}个")

            # 🔥 关键修复：预先计算所有K线形态特征（包含历史数据）
            self.logger.info("开始预计算K线形态特征...")
            feature_complete_stock_data = self.multi_factor_scorer.feature_calculator.calculate_missing_features(stock_data)
            self.logger.info(f"K线形态特征计算完成: {len(feature_complete_stock_data)}条记录")

            self.logger.info(f"开始多线程选股，共{len(trading_dates)}个交易日")

            all_selections = []

            # 使用多线程处理选股
            max_workers = min(8, len(trading_dates))  # 最多8个线程，避免过度并发

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务（传入特征完整的数据）
                future_to_date = {
                    executor.submit(
                        self._select_stocks_for_date,
                        date_str, feature_complete_stock_data, sector_data, market_data
                    ): date_str
                    for date_str in trading_dates
                }

                # 收集结果
                completed_count = 0
                for future in as_completed(future_to_date):
                    date_str = future_to_date[future]
                    completed_count += 1

                    try:
                        daily_selections = future.result()
                        all_selections.extend(daily_selections)

                        if daily_selections:
                            self.logger.info(f"{date_str}: 选中 {[s.stock_code for s in daily_selections]}")

                        # 显示进度
                        if completed_count % 10 == 0 or completed_count == len(trading_dates):
                            progress = completed_count / len(trading_dates) * 100
                            self.logger.info(f"选股进度: {completed_count}/{len(trading_dates)} ({progress:.1f}%)")

                    except Exception as e:
                        self.logger.error(f"日期{date_str}选股失败: {e}")

            # 按日期排序结果
            all_selections.sort(key=lambda x: x.date)

            self.logger.info(f"多线程选股流程完成: 总计{len(all_selections)}次选股")

            return all_selections

        except Exception as e:
            self.logger.error(f"选股流程失败: {e}")
            return []

    def _select_stocks_for_date(self, date_str: str, stock_data: pd.DataFrame,
                               sector_data: pd.DataFrame, market_data: pd.DataFrame) -> List[StockSelection]:
        """
        为单个日期执行选股 - 线程安全版本
        """
        try:
            # 执行当日选股
            daily_selections = self.select_daily_stocks(
                date_str, stock_data, sector_data, market_data
            )

            return daily_selections

        except Exception as e:
            self.logger.error(f"日期{date_str}选股失败: {e}")
            return []
