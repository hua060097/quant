#!/usr/bin/env python3
"""
板块策略回测器
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class SectorBacktestResult:
    """板块回测结果"""
    sector_code: str
    sector_name: str
    total_return: float
    win_rate: float
    avg_return: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    profitable_trades: int
    
    # 排名信息
    avg_rank: float
    top3_rate: float  # 进入前3的比例
    top5_rate: float  # 进入前5的比例

class SectorBacktester:
    """板块策略回测器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def backtest_sector_strategy(self, sector_data: pd.DataFrame, 
                                start_date: str, end_date: str,
                                top_n: int = 5) -> List[SectorBacktestResult]:
        """回测板块策略"""
        try:
            self.logger.info(f"开始板块策略回测: {start_date} - {end_date}")
            
            # 数据预处理
            data = sector_data.copy()
            data['date'] = pd.to_datetime(data['date'])
            
            # 过滤日期范围
            start_date_obj = pd.to_datetime(start_date)
            end_date_obj = pd.to_datetime(end_date)
            
            filtered_data = data[
                (data['date'] >= start_date_obj) & 
                (data['date'] <= end_date_obj)
            ]
            
            if len(filtered_data) == 0:
                self.logger.error("没有有效的板块数据")
                return []
            
            # 获取所有交易日
            trading_dates = sorted(filtered_data['date'].unique())
            
            # 模拟板块选择和收益计算
            sector_performance = self._simulate_sector_selection(
                filtered_data, trading_dates, top_n
            )
            
            # 计算回测结果
            backtest_results = self._calculate_sector_backtest_results(
                sector_performance, filtered_data
            )
            
            self.logger.info(f"板块回测完成: {len(backtest_results)}个板块")
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"板块回测失败: {e}")
            return []
    
    def _simulate_sector_selection(self, sector_data: pd.DataFrame, 
                                 trading_dates: List, top_n: int) -> Dict:
        """模拟板块选择过程"""
        try:
            sector_performance = {}
            
            # 导入板块分析器
            try:
                import sys
                from pathlib import Path
                project_root = Path(__file__).parent.parent
                sys.path.append(str(project_root / "clean_ml_system"))
                from enhanced_sector_analyzer import EnhancedSectorAnalyzer
                
                analyzer = EnhancedSectorAnalyzer()
            except ImportError:
                self.logger.error("无法导入板块分析器")
                return {}
            
            for i, date in enumerate(trading_dates[:-1]):  # 排除最后一天
                try:
                    # 获取当日板块数据
                    daily_sectors = sector_data[sector_data['date'] == date]
                    
                    if len(daily_sectors) == 0:
                        continue
                    
                    # 创建虚拟市场数据
                    daily_market = pd.DataFrame({
                        'date': [date],
                        'market_return': [0.0],
                        'market_volume': [1.0]
                    })
                    
                    # 分析板块潜力
                    sector_results = analyzer.analyze_enhanced_daily_potential(
                        daily_sectors, daily_market
                    )
                    
                    if not sector_results:
                        continue
                    
                    # 选择前N个板块
                    selected_sectors = sector_results[:top_n]
                    
                    # 计算次日收益
                    next_date = trading_dates[i + 1]
                    next_day_data = sector_data[sector_data['date'] == next_date]
                    
                    for j, sector_result in enumerate(selected_sectors):
                        sector_code = sector_result.sector_code
                        
                        # 获取次日该板块数据
                        sector_next = next_day_data[
                            next_day_data['sector_code'] == sector_code
                        ]
                        
                        if len(sector_next) == 0:
                            continue
                        
                        # 计算收益率
                        next_return = sector_next.iloc[0].get('change_percent', 0) / 100
                        
                        # 记录表现
                        if sector_code not in sector_performance:
                            sector_performance[sector_code] = {
                                'returns': [],
                                'ranks': [],
                                'selections': 0,
                                'dates': []
                            }
                        
                        sector_performance[sector_code]['returns'].append(next_return)
                        sector_performance[sector_code]['ranks'].append(j + 1)  # 排名从1开始
                        sector_performance[sector_code]['selections'] += 1
                        sector_performance[sector_code]['dates'].append(date)
                
                except Exception as e:
                    self.logger.warning(f"处理日期{date}失败: {e}")
                    continue
            
            return sector_performance
            
        except Exception as e:
            self.logger.error(f"模拟板块选择失败: {e}")
            return {}
    
    def _calculate_sector_backtest_results(self, sector_performance: Dict, 
                                         sector_data: pd.DataFrame) -> List[SectorBacktestResult]:
        """计算板块回测结果"""
        try:
            results = []
            
            for sector_code, performance in sector_performance.items():
                returns = performance['returns']
                ranks = performance['ranks']
                selections = performance['selections']
                
                if len(returns) == 0:
                    continue
                
                # 获取板块名称
                sector_info = sector_data[sector_data['sector_code'] == sector_code]
                sector_name = sector_info.iloc[0].get('sector_name', sector_code) if len(sector_info) > 0 else sector_code
                
                # 计算收益指标
                total_return = np.sum(returns)
                avg_return = np.mean(returns)
                win_rate = sum(1 for r in returns if r > 0) / len(returns)
                
                # 计算最大回撤
                cumulative_returns = np.cumsum(returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = cumulative_returns - running_max
                max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
                
                # 计算夏普比率
                return_std = np.std(returns) if len(returns) > 1 else 0
                sharpe_ratio = avg_return / return_std if return_std > 0 else 0
                
                # 计算排名指标
                avg_rank = np.mean(ranks)
                top3_rate = sum(1 for r in ranks if r <= 3) / len(ranks)
                top5_rate = sum(1 for r in ranks if r <= 5) / len(ranks)
                
                # 盈利交易数
                profitable_trades = sum(1 for r in returns if r > 0)
                
                result = SectorBacktestResult(
                    sector_code=sector_code,
                    sector_name=sector_name,
                    total_return=total_return,
                    win_rate=win_rate,
                    avg_return=avg_return,
                    max_drawdown=max_drawdown,
                    sharpe_ratio=sharpe_ratio,
                    total_trades=selections,
                    profitable_trades=profitable_trades,
                    avg_rank=avg_rank,
                    top3_rate=top3_rate,
                    top5_rate=top5_rate
                )
                
                results.append(result)
            
            # 按总收益率排序
            results.sort(key=lambda x: x.total_return, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"计算板块回测结果失败: {e}")
            return []
    
    def analyze_sector_rank_decay(self, sector_results: List, top_n: int = 5) -> Dict:
        """分析板块排名衰减效应"""
        try:
            rank_analysis = {}
            
            for i in range(1, top_n + 1):
                rank_sectors = [s for s in sector_results if s.avg_rank <= i + 0.5 and s.avg_rank >= i - 0.5]
                
                if rank_sectors:
                    avg_return = np.mean([s.avg_return for s in rank_sectors])
                    avg_win_rate = np.mean([s.win_rate for s in rank_sectors])
                    count = len(rank_sectors)
                    
                    rank_analysis[f'rank_{i}'] = {
                        'avg_return': avg_return,
                        'avg_win_rate': avg_win_rate,
                        'count': count,
                        'sectors': [s.sector_code for s in rank_sectors]
                    }
            
            return rank_analysis
            
        except Exception as e:
            self.logger.error(f"分析排名衰减失败: {e}")
            return {}
    
    def generate_sector_report(self, sector_results: List[SectorBacktestResult], 
                             rank_analysis: Dict) -> str:
        """生成板块回测报告"""
        try:
            report = []
            report.append("=" * 60)
            report.append("板块策略回测报告")
            report.append("=" * 60)
            
            # 整体统计
            if sector_results:
                total_sectors = len(sector_results)
                avg_return = np.mean([s.avg_return for s in sector_results])
                avg_win_rate = np.mean([s.win_rate for s in sector_results])
                
                report.append(f"总板块数: {total_sectors}")
                report.append(f"平均收益率: {avg_return:.2%}")
                report.append(f"平均胜率: {avg_win_rate:.1%}")
                report.append("")
            
            # 前10个板块表现
            report.append("前10个板块表现:")
            for i, sector in enumerate(sector_results[:10]):
                report.append(f"  {i+1:2d}. {sector.sector_code}: "
                            f"收益{sector.total_return:.2%}, "
                            f"胜率{sector.win_rate:.1%}, "
                            f"交易{sector.total_trades}次, "
                            f"平均排名{sector.avg_rank:.1f}")
            
            report.append("")
            
            # 排名衰减分析
            if rank_analysis:
                report.append("排名衰减分析:")
                for rank, stats in rank_analysis.items():
                    rank_num = rank.split('_')[1]
                    report.append(f"  排名{rank_num}: "
                                f"平均收益{stats['avg_return']:.2%}, "
                                f"胜率{stats['avg_win_rate']:.1%}, "
                                f"板块数{stats['count']}")
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"生成板块报告失败: {e}")
            return "板块报告生成失败"
