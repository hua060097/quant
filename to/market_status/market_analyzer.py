#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场分析器 - 用于分析大盘状态
"""

import os
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

# 尝试导入Numba加速模块
try:
    from .numba_market_analyzer import patch_market_analyzer, HAS_NUMBA
except ImportError:
    try:
        from numba_market_analyzer import patch_market_analyzer, HAS_NUMBA
    except ImportError:
        HAS_NUMBA = False
        patch_market_analyzer = None

# 配置日志
logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """大盘分析器 - 专注于创业板指数分析
    
    功能特点：
    1. 创业板指数专注分析：专注于创业板指数分析，更精准把握短线行情
    2. 多维度评分体系：从趋势、位置、量能、动能、波动等多维度评估市场
    3. 精细化市场分类：15种细分市场状态，提供更精准的市场判断
    4. 周期识别：自动识别市场所处周期阶段
    5. 信号生成：基于市场状态生成交易信号和风险预警
    6. 高效计算：使用向量化实现提高性能
    """
    
    # 指数代码 - 只关注创业板指数
    INDEX_CODE = '399006.SZ'  # 创业板指数
    
    # 修改市场状态分类为更细致的划分
    MARKET_STATUS = {
        # 强势上涨
        11: "强势突破上涨",    # 突破重要压力位，量价配合，持续性强
        12: "加速上涨",       # 快速上涨，但未到过热
        13: "稳健上涨",       # 均线多头排列，量价良好
        
        # 温和上涨
        21: "盘升上涨",       # 缓慢向上，成交量温和
        22: "回调上涨",       # 短期回调后继续上行
        23: "整固上涨",       # 高位整理后继续上行
        
        # 盘整
        31: "强势盘整",       # 高位盘整，支撑强
        32: "中性盘整",       # 箱体震荡，方向不明
        33: "弱势盘整",       # 低位盘整，压力大
        
        # 温和下跌
        41: "盘跌下行",       # 缓慢向下，暂无明显企稳
        42: "回抽下跌",       # 反弹后继续下行
        43: "整固下跌",       # 低位整理后继续下行
        
        # 快速下跌
        51: "加速下跌",       # 快速下跌，但未到恐慌
        52: "恐慌下跌",       # 快速大幅下跌，成交量萎缩
        53: "超跌反弹",       # 大幅下跌后的反弹修复
    }
    
    # 市场周期定义
    MARKET_CYCLE = {
        1: "积累期",      # 底部区域，价格低迷，成交量低，但有机构悄悄建仓
        2: "上升期",      # 突破上行，成交量逐渐放大，趋势确立
        3: "分配期",      # 高位区域，成交量显著放大，波动加剧，机构悄悄卖出
        4: "下降期",      # 明确下跌，成交量先放大后萎缩，趋势向下
    }
    
    def __init__(self, use_numba=True):
        """初始化大盘分析器
        
        Args:
            use_numba: 是否使用Numba加速
        """
        self.logger = logging.getLogger(__name__)
        self._has_numba_accelerator = False
        
        # 保存原始计算函数的引用
        self._calculate_ma_original = self._calculate_ma
        self._calculate_ema_original = self._calculate_ema
        self._calculate_macd_original = self._calculate_macd
        self._calculate_rsi_original = self._calculate_rsi
        self._calculate_bollinger_bands_original = self._calculate_bollinger_bands
        self._calculate_kdj_original = self._calculate_kdj
        
        # 如果可用且用户请求，应用Numba加速
        if use_numba and HAS_NUMBA and patch_market_analyzer:
            try:
                success = patch_market_analyzer(self.__class__)
                if success:
                    self._has_numba_accelerator = True
                    self.logger.info("成功应用Numba加速到市场分析器")
                else:
                    self.logger.warning("应用Numba加速失败，将使用原始计算方法")
            except Exception as e:
                self.logger.error(f"初始化Numba加速器失败: {e}")

    def _calculate_ma(self, data, n):
        """
        计算简单移动平均线 - 向量化实现
        
        参数:
            data (np.array): 价格数据
            n (int): 移动平均窗口大小
            
        返回:
            np.array: 移动平均线
        """
        ma = np.zeros_like(data)
        for i in range(len(data)):
            if i < n-1:
                ma[i] = np.nan
            else:
                ma[i] = np.mean(data[i-n+1:i+1])
        return ma
    
    def _calculate_ema(self, data, n):
        """
        计算指数移动平均线 - 向量化实现
        
        参数:
            data (np.array): 价格数据
            n (int): 移动平均窗口大小
            
        返回:
            np.array: 指数移动平均线
        """
        ema = np.zeros_like(data)
        ema[0] = data[0]  # 初始值
        
        # EMA权重
        k = 2 / (n + 1)
        
        # 计算EMA
        for i in range(1, len(data)):
            ema[i] = data[i] * k + ema[i-1] * (1 - k)
            
        return ema

    def _generate_market_signals(self, current_status, previous_status=None):
        """
        根据市场状态生成交易信号
        
        参数:
            current_status (dict): 当前市场状态
            previous_status (dict): 前一日市场状态
            
        返回:
            dict: 交易信号
        """
        # 添加日志记录传入的状态
        logger.debug(f"_generate_market_signals received: type={current_status.get('market_status_info', {}).get('type_code', 'N/A')}, score={current_status.get('total_score', 'N/A')}")
        
        signals = {
            'trend_signal': 'neutral',    # 趋势信号：bullish/neutral/bearish
            'strength': 0,                # 信号强度：0-10
            'risk_level': 'medium',       # 风险等级：low/medium/high
            'trading_advice': '观望',      # 交易建议
            'opportunity_sectors': [],    # 机会板块
            'alert_level': 0              # 预警级别：0-5
        }
        
        # 趋势信号判断
        market_type = current_status.get('market_type', 0)
        total_score = current_status.get('total_score', 0)
        
        # 根据市场类型确定基本趋势信号
        if market_type in [11, 12, 13, 21, 22, 23]:  # 上涨类型
            signals['trend_signal'] = 'bullish'
            signals['strength'] = min(10, int(abs(total_score) * 2))
        elif market_type in [41, 42, 43, 51, 52, 53]:  # 下跌类型
            signals['trend_signal'] = 'bearish'
            signals['strength'] = min(10, int(abs(total_score) * 2))
        else:  # 盘整类型
            signals['trend_signal'] = 'neutral'
            signals['strength'] = min(5, int(abs(total_score) * 1.5))
        
        # 风险等级评估
        volatility_score = current_status.get('volatility_score', 0)
        momentum_score = current_status.get('momentum_score', 0)
        
        if volatility_score > 2 or abs(momentum_score) > 2:
            signals['risk_level'] = 'high'
        elif volatility_score > 1 or abs(momentum_score) > 1:
            signals['risk_level'] = 'medium'
        else:
            signals['risk_level'] = 'low'
        
        # 交易建议生成
        if signals['trend_signal'] == 'bullish':
            if signals['strength'] >= 8:
                signals['trading_advice'] = '积极做多，逢低吸纳'
                signals['alert_level'] = 1
            elif signals['strength'] >= 5:
                signals['trading_advice'] = '适度做多，波段操作'
                signals['alert_level'] = 2
            else:
                signals['trading_advice'] = '谨慎做多，高抛低吸'
                signals['alert_level'] = 3
        elif signals['trend_signal'] == 'bearish':
            if signals['strength'] >= 8:
                signals['trading_advice'] = '适当减仓，控制风险'
                signals['alert_level'] = 4
            elif signals['strength'] >= 5:
                signals['trading_advice'] = '谨慎观望，暂停做多'
                signals['alert_level'] = 3
            else:
                signals['trading_advice'] = '等待市场企稳，逢高减仓'
                signals['alert_level'] = 2
        else:
            signals['trading_advice'] = '观望为主，等待明确信号'
            signals['alert_level'] = 2
        
        # 状态转换信号检测
        if previous_status:
            prev_type = previous_status.get('market_type', 0)
            curr_type = current_status.get('market_type', 0)
            
            # 重大转折点检测
            if (prev_type in [41, 42, 43, 51, 52, 53] and curr_type in [11, 12, 13, 21, 22, 23]):
                # 由空转多
                signals['trading_advice'] = '市场由空转多，可适度布局'
                signals['alert_level'] = 5
            elif (prev_type in [11, 12, 13, 21, 22, 23] and curr_type in [41, 42, 43, 51, 52, 53]):
                # 由多转空
                signals['trading_advice'] = '市场由多转空，注意风险控制'
                signals['alert_level'] = 5
        
        return signals

    def analyze_market_status(self, market_data_transposed, indicators_dict):
        """
        分析市场状态 - 专注于创业板指数分析
        
        参数:
            market_data_transposed (dict): 转置后的市场数据字典 (日期x指数)
            indicators_dict (dict): 技术指标字典 {index_code: {indicator_name: array}}
            
        返回:
            dict: 市场状态字典 {date_str: {status_info}}
        """
        logger.info("开始分析市场状态 (处理转置后数据)")
        
        # 初始化结果字典
        market_status = {}
        # 获取创业板指数代码
        index_code = self.INDEX_CODE # '399006.SZ'

        if index_code not in indicators_dict:
            logger.warning(f"指标数据中缺少指数 {index_code}，无法进行分析")
            return {}
            
        # 获取该指数的指标和数据
        indicators = indicators_dict[index_code]
        # 获取日期列表 (确保它是列表或可以转换为列表)
        dates = indicators.get('dates') 
        if not dates or not isinstance(dates, (list, pd.Index, np.ndarray)):
            logger.error(f"指数 {index_code} 的指标数据中缺少有效的日期列表 ('dates')")
            return {}
            
        # 将日期转换为 Timestamp 对象列表，方便处理
        try:
            dates_ts = pd.to_datetime(dates).to_list()
        except Exception as e:
            logger.error(f"无法将指标中的日期列表转换为时间戳: {e}")
            return {}

        # 获取必要的原始数据 Series (确保它们存在)
        close_series = indicators.get('close_series')
        volume_series = indicators.get('volume_series')
        open_series = indicators.get('open_series')
        high_series = indicators.get('high_series')
        low_series = indicators.get('low_series')
        if close_series is None or volume_series is None or open_series is None or high_series is None or low_series is None:
            logger.error(f"指数 {index_code} 的指标数据中缺少必要的原始数据 Series (close/volume/open/high/low)")
            return {}
            
        # 验证所有指标数组/列表的长度是否与日期数量一致
        num_dates = len(dates_ts)
        for key, value in indicators.items():
            # 跳过非数组/列表或Series类型的值以及原始Series
            if not isinstance(value, (np.ndarray, list)) or key.endswith('_series') or key == 'dates': 
                continue
            if len(value) != num_dates:
                logger.error(f"指标 '{key}' 的长度 ({len(value)}) 与日期数量 ({num_dates}) 不匹配")
                return {}
        logger.debug(f"指数 {index_code}: 所有指标长度验证通过 ({num_dates} 条记录)")

        # --- 按日期迭代进行分析 --- 
        for i, current_date_ts in enumerate(dates_ts):
            date_str = current_date_ts.strftime('%Y%m%d') # 使用 YYYYMMDD 格式字符串作为字典键
            
            # 初始化当日状态
            market_status[date_str] = {
                'date': current_date_ts, 
                'individual_status': {}, 'composite_status': {},
                'market_type': 0, 'market_description': "未分析", # 初始状态为未分析
                'trend_strength': 0.0, 'market_cycle': 0, 'cycle_description': ""
            }

            logger.debug(f"开始分析日期: {date_str} (索引 {i})")
            # --- 提取当日指标值 --- 
            try:
                # 使用索引 i 获取指标数组/列表中的值
                ma5_val = indicators['ma5'][i]
                ma10_val = indicators['ma10'][i]
                ma20_val = indicators['ma20'][i]
                ma60_val = indicators['ma60'][i]
                macd_val = indicators['macd'][i]
                macd_signal_val = indicators['macd_signal'][i]
                macd_hist_val = indicators['macd_hist'][i]
                rsi6_val = indicators['rsi6'][i]
                rsi12_val = indicators['rsi12'][i]
                kdj_k_val = indicators['kdj_k'][i]
                kdj_d_val = indicators['kdj_d'][i]
                bb_width_val = indicators['bollinger_width'][i]
                bb_position_val = indicators['bollinger_position'][i]
                # 复杂指标暂时使用默认值
                support_level_val = indicators['support_level'][i]
                resistance_level_val = indicators['resistance_level'][i]
                price_structure_val = indicators['price_structure'][i]
                market_cycle_val = indicators['market_cycle'][i]
                
                # 获取当日和前一日收盘价 (使用 .iloc 访问 Series)
                current_close = close_series.iloc[i]
                prev_close = close_series.iloc[i-1] if i > 0 else current_close # 安全获取前一日
                # 获取当日其他市场数据
                current_open = open_series.iloc[i]
                current_high = high_series.iloc[i]
                current_low = low_series.iloc[i]
                current_volume = volume_series.iloc[i]
                
                # 检查关键值是否为 NaN (可能影响评分)
                nan_indicators = []
                if np.isnan(current_close): nan_indicators.append("close")
                if np.isnan(prev_close): nan_indicators.append("prev_close")
                if np.isnan(ma60_val): nan_indicators.append("ma60")
                if np.isnan(rsi6_val): nan_indicators.append("rsi6")

                if nan_indicators:
                    logger.warning(f"日期 {date_str} 缺少关键指标数据 ({', '.join(nan_indicators)})，无法进行完整评分。将使用默认中性状态。")
                    # 分配一个默认/中性市场类型而不是跳过
                    market_status[date_str]['market_type'] = 32 # 中性盘整
                    market_status[date_str]['market_description'] = f"关键指标缺失 ({', '.join(nan_indicators)})"
                    market_status[date_str]['total_score'] = 0 # 或者设为NaN？暂定为0
                    # 仍然需要跳过后续的评分计算，但该日期的数据不会因为market_type=0而被过滤掉
                    continue # 跳过本轮循环的后续评分计算

            except IndexError:
                logger.warning(f"日期 {date_str} 索引 {i} 超出指标数组范围，跳过分析")
                continue
            except KeyError as e:
                 logger.warning(f"日期 {date_str} 分析时缺少指标键 '{e}'，跳过分析")
                 continue
            except Exception as e:
                 logger.error(f"提取日期 {date_str} 指标时发生未知错误: {e}")
                 traceback.print_exc()
                 continue

            # --- 使用提取的当日指标值进行评分 --- 
            # (这部分逻辑基本不变，但需要确保传入的是数值而非Series/DataFrame)
            
            # 计算每日涨跌幅
            daily_change_pct = 0.0
            if prev_close != 0 and not np.isnan(prev_close):
                daily_change_pct = (current_close - prev_close) / prev_close * 100
            else:
                logger.debug(f"日期 {date_str}: 前一日收盘价为0或NaN，无法计算涨跌幅")

            # 1. 趋势强度评估 (需要一个接受单日数值的版本)
            # 假设已创建 _calculate_trend_score_single 方法
            trend_score = self._calculate_trend_score_single(
                 current_close, prev_close,
                 ma5_val, ma10_val, ma20_val, ma60_val,
                 macd_val, macd_signal_val, macd_hist_val, rsi6_val
            )
            
            # 2. 位置状态评估
            position_score = 0
            # 确保 ma60_val 不是 NaN
            if not np.isnan(ma60_val):
                price = current_close
                if price > ma60_val * 1.2: position_score += 3
                elif price > ma60_val * 1.1: position_score += 2
                # ... (其他条件) ...
                elif price < ma60_val: position_score -= 1

            # 3. 量能评估 (改进：使用 nanmean 并调整逻辑)
            volume_score = 0
            # 需要获取历史成交量窗口 (i 必须大于等于 20 才有 20 天历史)
            if i >= 19: # 索引从0开始，需要20个点，即 i-19 到 i (包含i)
                try:
                    # 使用 .iloc 获取 Series 的切片，确保索引存在
                    # 截取 [i-5+1:i+1] -> 5天; [i-20+1:i+1] -> 20天
                    recent_vol_slice = volume_series.iloc[max(0, i-5+1):i+1]
                    long_vol_slice = volume_series.iloc[max(0, i-20+1):i+1]
                    
                    # 使用 nanmean 安全计算均值
                    recent_vol_avg = np.nanmean(recent_vol_slice.values)
                    long_vol_avg = np.nanmean(long_vol_slice.values)
                    
                    # 检查计算结果是否有效 (非 NaN 且长期均量大于0)
                    if not np.isnan(recent_vol_avg) and not np.isnan(long_vol_avg) and long_vol_avg > 1e-9:
                        ratio = recent_vol_avg / long_vol_avg
                        if ratio > 1.5: volume_score = 2 # 显著放量
                        elif ratio > 1.0: volume_score = 1 # 温和放量
                        elif ratio < 0.7: volume_score = -2 # 显著缩量
                        elif ratio < 1.0: volume_score = -1 # 温和缩量
                    else:
                         logger.debug(f"日期 {date_str} 无法计算有效成交量均值，跳过量能评分")
                except Exception as vol_err:
                    logger.warning(f"日期 {date_str} 计算量能评分时出错: {vol_err}")
            else:
                 logger.debug(f"日期 {date_str} 成交量历史数据不足20天，跳过量能评分")
            
            # 4. 动能评估
            momentum_score = 0
            # 确保指标不是 NaN
            if not np.isnan(rsi6_val) and not np.isnan(macd_val) and not np.isnan(macd_signal_val):
                if rsi6_val > 70 and macd_val > macd_signal_val: momentum_score += 3
                # ... (其他条件保持不变) ...
                elif rsi6_val < 50 and macd_val < macd_signal_val: momentum_score -= 1

            # 5. 波动性评估
            volatility_score = 0
            if not np.isnan(bb_width_val):
                if bb_width_val > 0.1: volatility_score = 3
                elif bb_width_val > 0.05: volatility_score = 2
                else: volatility_score = 1

            # 6. 价格结构评估 (实现简单逻辑)
            structure_score = 0
            if i >= 9: # 需要至少10天数据 (包括当天)
                try:
                    recent_highs = high_series.iloc[max(0, i-10+1):i+1].values
                    recent_lows = low_series.iloc[max(0, i-10+1):i+1].values
                    # 检查是否有 NaN
                    if not np.isnan(recent_highs).any() and not np.isnan(recent_lows).any():
                        # 简单比较首尾高低点趋势 (更复杂可以用线性回归斜率等)
                        if recent_highs[-1] > recent_highs[0] and recent_lows[-1] > recent_lows[0]:
                            structure_score = 1 # 上升结构
                        elif recent_highs[-1] < recent_highs[0] and recent_lows[-1] < recent_lows[0]:
                            structure_score = -1 # 下降结构
                except Exception as struct_err:
                     logger.warning(f"日期 {date_str} 计算结构评分时出错: {struct_err}")
            else:
                logger.debug(f"日期 {date_str} 历史数据不足10天，跳过结构评分")

            # 7. 周期评估 (实现简单逻辑)
            market_cycle_val = 0 # 初始化为0 (未知)
            cycle_score = 0
            cycle_description = "未知周期" # 初始化
            slope_period = 5 # 计算 MA20 的 5 日斜率
            ma20_slope_threshold = 0.1 # MA20斜率阈值 (需要调整)
            
            if i >= max(19 + slope_period -1, 59): # 确保有MA20和MA60，并足够计算斜率
                try:
                    ma20_recent = indicators['ma20'][i-slope_period+1:i+1]
                    ma60_val = indicators['ma60'][i] # 当日MA60
                    current_close = close_series.iloc[i] # 当日收盘价
                    
                    # 检查是否有 NaN
                    if not np.isnan(ma20_recent).any() and not np.isnan(ma60_val) and not np.isnan(current_close):
                        # 简单计算MA20斜率 (可以用 linregress 获得更准确的)
                        ma20_slope = (ma20_recent[-1] - ma20_recent[0]) / slope_period 
                        
                        if current_close > ma60_val:
                            if ma20_slope > ma20_slope_threshold:
                                market_cycle_val = 2 # 上升期
                                cycle_score = 1
                            else:
                                market_cycle_val = 3 # 分配期
                        else: # current_close <= ma60_val
                            if ma20_slope < -ma20_slope_threshold:
                                market_cycle_val = 4 # 下降期
                                cycle_score = -1
                            else:
                                market_cycle_val = 1 # 积累期
                        
                        cycle_description = self.MARKET_CYCLE.get(market_cycle_val, "未知周期")
                    else:
                        logger.debug(f"日期 {date_str} 缺少计算周期所需的指标(MA20/MA60/Close)，跳过周期评分")
                except Exception as cycle_err:
                    logger.warning(f"日期 {date_str} 计算周期评分时出错: {cycle_err}")
            else:
                 logger.debug(f"日期 {date_str} 历史数据不足无法计算周期，跳过周期评分")
            
            # 综合评分计算
            # 调整权重，增加趋势和动能权重，降低未完全实现的结构/周期权重
            total_score = (
                trend_score * 0.30 + position_score * 0.15 + volume_score * 0.10 + 
                momentum_score * 0.20 + volatility_score * 0.10 + 
                structure_score * 0.05 + cycle_score * 0.10 
            )

            # 市场状态判断 (基于更新后的 total_score)
            calculated_market_type = self._determine_market_type(total_score)
            calculated_market_description = self.MARKET_STATUS.get(calculated_market_type, "未知状态")
            # 使用计算出的 cycle_description
            # cycle_description = self.MARKET_CYCLE.get(market_cycle_val, "未知周期")
            trend_strength = abs(total_score)

            # --- 保存当日分析结果 --- 
            # 准备详细状态信息
            individual_status = {
                'trend_score': trend_score, 'position_score': position_score, 'volume_score': volume_score,
                'momentum_score': momentum_score, 'volatility_score': volatility_score,
                'structure_score': structure_score, 'cycle_score': cycle_score, 'total_score': total_score,
                # 明确区分代码和描述
                'market_status_info': {
                    'type_code': calculated_market_type, # 使用计算出的类型
                    'type_desc': calculated_market_description, 
                    'cycle_code': market_cycle_val,
                    'cycle_desc': cycle_description
                },
                'technical_data': { # 保存当日的数值
                    'close': current_close, 'prev_close': prev_close,
                    'ma5': ma5_val, 'ma20': ma20_val, 'ma60': ma60_val,
                    'rsi6': rsi6_val, 'rsi12': rsi12_val, # 确保 rsi12_val 已提取
                    'macd': macd_val, 'macd_signal': macd_signal_val, 'macd_hist': macd_hist_val,
                    'kdj_k': kdj_k_val, 'kdj_d': kdj_d_val, # 确保 kdj 值已提取
                    'bb_width': bb_width_val, 'bb_position': bb_position_val, # 确保 bb 值已提取
                    'support': support_level_val, 'resistance': resistance_level_val
                },
                'market_data': { # 保存当日的数值
                    'close': current_close, 'open': current_open, 'high': current_high,
                    'low': current_low, 'volume': current_volume,
                    'daily_change_pct': round(daily_change_pct, 2) # 添加每日涨跌幅
                }
            }
            
            # 更新当日的整体状态字典
            current_status_dict = market_status[date_str] 
            current_status_dict['market_type'] = calculated_market_type # <--- 修正：更新外层 market_type
            current_status_dict['market_description'] = calculated_market_description # <--- 修正：更新外层 description
            current_status_dict['trend_strength'] = trend_strength # 整体趋势强度保留在外层
            current_status_dict['market_cycle'] = market_cycle_val # 更新外层 cycle_code
            current_status_dict['cycle_description'] = cycle_description # 更新外层 cycle_description
            current_status_dict['individual_status'][index_code] = individual_status
            # 因为只分析一个指数，composite status 可以直接复制 individual status 的副本
            current_status_dict['composite_status'] = individual_status.copy()

            # 生成市场信号 (需要前一日的状态)
            previous_date_str = dates_ts[i-1].strftime('%Y%m%d') if i > 0 else None
            previous_day_status = market_status.get(previous_date_str, {}).get('composite_status')
            current_status_dict['signals'] = self._generate_market_signals(
                 current_status_dict['composite_status'], 
                 previous_day_status
            )
            logger.debug(f"完成分析日期: {date_str}")

        # --- 循环结束后 --- 
        # 修正日志计数和返回过滤逻辑：基于外层 market_type 是否非零
        analyzed_dates = [
            d for d, s in market_status.items() 
            if s.get('market_type', 0) != 0
        ]
        logger.info(f"市场状态分析完成，共分析 {len(analyzed_dates)} 个有效交易日")
        
        # 返回只包含有效分析结果的字典
        return {date_key: status for date_key, status in market_status.items() if date_key in analyzed_dates}

    # 确保这个辅助方法存在于 MarketAnalyzer 类中
    def _calculate_trend_score_single(self, close_price, prev_close, ma5, ma10, ma20, ma60, macd, macd_signal, macd_hist, rsi6):
        trend_score = 0.0
        # 1. 单日涨跌幅
        if prev_close != 0 and not np.isnan(prev_close): # 避免除零和NaN
            daily_change_pct = (close_price - prev_close) / prev_close * 100
            if daily_change_pct >= 8: trend_score += 5.0
            elif daily_change_pct >= 5: trend_score += 4.0
            elif daily_change_pct >= 3: trend_score += 3.0
            elif daily_change_pct >= 1.5: trend_score += 2.0
            elif daily_change_pct >= 0.5: trend_score += 1.0
            elif daily_change_pct <= -8: trend_score -= 5.0
            elif daily_change_pct <= -5: trend_score -= 4.0
            elif daily_change_pct <= -3: trend_score -= 3.0
            elif daily_change_pct <= -1.5: trend_score -= 2.0
            elif daily_change_pct <= -0.5: trend_score -= 1.0

        # 2. 均线多空排列评分 (检查 NaN)
        if not np.isnan([ma5, ma10, ma20, ma60]).any():
            if ma5 > ma10 > ma20 > ma60: trend_score += 3.0
            elif ma5 > ma10 > ma20: trend_score += 2.0
            elif ma5 > ma10: trend_score += 1.0
            elif ma5 < ma10 < ma20 < ma60: trend_score -= 3.0
            elif ma5 < ma10 < ma20: trend_score -= 2.0
            elif ma5 < ma10: trend_score -= 1.0

        # 3. 均线位置关系评分 (检查 NaN)
        if not np.isnan(ma5):
            if close_price > ma5 * 1.05: trend_score += 1.5
            elif close_price > ma5: trend_score += 0.8
        if not np.isnan(ma20):
            if close_price > ma20 * 1.1: trend_score += 1.5
            elif close_price > ma20: trend_score += 1.0
        if not np.isnan(ma60):
            if close_price > ma60 * 1.15: trend_score += 2.0
            elif close_price > ma60: trend_score += 1.2

        # 4. 动能指标 (检查 NaN)
        if not np.isnan(macd_hist):
            if macd_hist > 0:
                if macd_hist > 0.15: trend_score += 1.5
                elif macd_hist > 0.05: trend_score += 0.8
            else:
                if macd_hist < -0.15: trend_score -= 1.5
                elif macd_hist < -0.05: trend_score -= 0.8
        # MACD 金叉死叉判断需要前一日数据，在 _calculate_trend_score_single 中无法直接实现

        # 结合RSI指标 (检查 NaN)
        if not np.isnan(rsi6):
            if rsi6 > 80: trend_score += 1.5
            elif rsi6 > 65: trend_score += 1.0
            elif rsi6 < 20: trend_score -= 1.5
            elif rsi6 < 35: trend_score -= 1.0

        # 将得分限制在合理范围，例如 -10 到 10
        return max(-10.0, min(10.0, trend_score))

    def _calculate_macd(self, data, fast=12, slow=26, signal=9):
        """
        计算MACD指标 - 向量化实现
        
        参数:
            data (np.array): 价格数据
            fast (int): 快线周期
            slow (int): 慢线周期
            signal (int): 信号线周期
            
        返回:
            tuple: (MACD线, 信号线, 柱状图)
        """
        # 计算EMA
        ema_fast = self._calculate_ema(data, fast)
        ema_slow = self._calculate_ema(data, slow)
        
        # 计算MACD线
        macd_line = ema_fast - ema_slow
        
        # 计算信号线
        signal_line = self._calculate_ema(macd_line, signal)
        
        # 计算柱状图
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_rsi(self, data, n=14):
        """
        计算RSI指标 - 向量化实现
        
        参数:
            data (np.array): 价格数据
            n (int): RSI周期
            
        返回:
            np.array: RSI值
        """
        # 计算价格变化
        deltas = np.zeros_like(data)
        deltas[1:] = data[1:] - data[:-1]
        
        # 分离上涨和下跌
        up = np.zeros_like(deltas)
        down = np.zeros_like(deltas)
        
        up[deltas > 0] = deltas[deltas > 0]
        down[deltas < 0] = -deltas[deltas < 0]
        
        # 计算平均上涨和下跌
        roll_up = np.zeros_like(data)
        roll_down = np.zeros_like(data)
        
        # 初始平均值
        roll_up[n] = np.mean(up[1:n+1])
        roll_down[n] = np.mean(down[1:n+1])
        
        # 计算后续值
        for i in range(n+1, len(data)):
            roll_up[i] = (roll_up[i-1] * (n-1) + up[i]) / n
            roll_down[i] = (roll_down[i-1] * (n-1) + down[i]) / n
        
        # 计算RS和RSI
        rs = np.zeros_like(data)
        rsi = np.zeros_like(data)
        
        # 避免除以零
        rs[n:] = np.divide(roll_up[n:], roll_down[n:], 
                           out=np.ones_like(roll_up[n:]) * 100, 
                           where=roll_down[n:] != 0)
        
        rsi[n:] = 100 - (100 / (1 + rs[n:]))
        rsi[:n] = np.nan
        
        return rsi
    
    def _calculate_kdj(self, high, low, close, n=9, m1=3, m2=3):
        """
        计算KDJ指标 - 向量化实现
        
        参数:
            high (np.array): 最高价数据
            low (np.array): 最低价数据
            close (np.array): 收盘价数据
            n (int): 计算周期
            m1 (int): K值平滑系数
            m2 (int): D值平滑系数
            
        返回:
            tuple: (K值, D值, J值)
        """
        # 计算n日内的最高价和最低价
        high_n = np.zeros_like(high)
        low_n = np.zeros_like(low)
        
        for i in range(len(high)):
            if i < n - 1:
                high_n[i] = np.nan
                low_n[i] = np.nan
            else:
                high_n[i] = np.max(high[i-n+1:i+1])
                low_n[i] = np.min(low[i-n+1:i+1])
        
        # 计算未成熟随机值RSV
        rsv = np.zeros_like(close)
        for i in range(n-1, len(close)):
            if high_n[i] == low_n[i]:
                rsv[i] = 50
            else:
                rsv[i] = 100 * (close[i] - low_n[i]) / (high_n[i] - low_n[i])
        
        # 计算K值
        k = np.zeros_like(close)
        k[n-1] = 50  # 初始值
        
        # 计算D值
        d = np.zeros_like(close)
        d[n-1] = 50  # 初始值
        
        # 平滑计算K和D
        for i in range(n, len(close)):
            k[i] = (k[i-1] * (m1-1) + rsv[i]) / m1
            d[i] = (d[i-1] * (m2-1) + k[i]) / m2
        
        # 计算J值
        j = 3 * k - 2 * d
        
        return k, d, j
    
    def _calculate_bollinger_bands(self, data, n=20, k=2):
        """
        计算布林带指标 - 向量化实现
        
        参数:
            data (np.array): 价格数据
            n (int): 移动平均窗口大小
            k (float): 标准差倍数
            
        返回:
            tuple: (上轨, 中轨, 下轨)
        """
        # 计算中轨(简单移动平均线)
        middle = self._calculate_ma(data, n)
        
        # 计算标准差
        std = np.zeros_like(data)
        for i in range(len(data)):
            if i < n-1:
                std[i] = np.nan
            else:
                std[i] = np.std(data[i-n+1:i+1])
        
        # 计算上轨和下轨
        upper = middle + k * std
        lower = middle - k * std
        
        return upper, middle, lower
    
    def _calculate_atr(self, high, low, close, n=14):
        """
        计算平均真实波幅(ATR) - 向量化实现
        
        参数:
            high (np.array): 最高价数据
            low (np.array): 最低价数据  
            close (np.array): 收盘价数据
            n (int): 计算周期
            
        返回:
            np.array: ATR值
        """
        # 计算真实波幅(TR)
        tr = np.zeros_like(high)
        
        # 第一个TR仅考虑当日高低价差
        tr[0] = high[0] - low[0]
        
        # 后续TR为三种价差的最大值
        for i in range(1, len(high)):
            hl = high[i] - low[i]  # 当日高低价差
            hc = abs(high[i] - close[i-1])  # 当日最高价与前一日收盘价差
            lc = abs(low[i] - close[i-1])  # 当日最低价与前一日收盘价差
            tr[i] = max(hl, hc, lc)
        
        # 计算ATR
        atr = np.zeros_like(high)
        atr[:n-1] = np.nan
        
        # 初始ATR值为前n日TR的平均值
        if len(tr) >= n:
            atr[n-1] = np.mean(tr[:n])
            
            # 计算后续值
            for i in range(n, len(high)):
                atr[i] = (atr[i-1] * (n-1) + tr[i]) / n
        
        return atr
    
    def _calculate_returns(self, data):
        """
        计算收益率 - 向量化实现
        
        参数:
            data (np.array): 价格数据
            
        返回:
            np.array: 收益率序列
        """
        returns = np.zeros_like(data)
        returns[1:] = (data[1:] - data[:-1]) / data[:-1]
        returns[0] = 0  # 第一个元素没有收益率
        return returns
    
    def _calculate_cumulative_return(self, data, n):
        """
        计算累积收益率 - 向量化实现
        
        参数:
            data (np.array): 价格数据
            n (int): 计算周期
            
        返回:
            np.array: 累积收益率
        """
        cum_returns = np.zeros_like(data)
        cum_returns[:n] = np.nan
        
        for i in range(n, len(data)):
            cum_returns[i] = (data[i] / data[i-n]) - 1
            
        return cum_returns
    
    def _calculate_up_down_ratio(self, data, n):
        """
        计算上涨下跌比率 - 向量化实现
        
        参数:
            data (np.array): 价格数据
            n (int): 计算周期
            
        返回:
            np.array: 上涨下跌比率
        """
        # 计算日收益率
        returns = self._calculate_returns(data)
        
        # 计算上涨下跌比率
        ratio = np.zeros_like(data)
        ratio[:n] = np.nan
        
        for i in range(n, len(data)):
            win_days = np.sum(returns[i-n+1:i+1] > 0)
            lose_days = np.sum(returns[i-n+1:i+1] < 0)
            
            if lose_days == 0:
                ratio[i] = win_days  # 避免除以零
            else:
                ratio[i] = win_days / lose_days
                
        return ratio
    
    def _calculate_volatility(self, data, window=20):
        """
        计算波动率 - 向量化实现
        
        参数:
            data (np.array): 价格数据
            window (int): 计算窗口大小
            
        返回:
            np.array: 波动率序列
        """
        # 计算收益率
        returns = self._calculate_returns(data)
        
        # 计算滚动标准差
        volatility = np.zeros_like(data)
        for i in range(window, len(data)):
            volatility[i] = np.std(returns[i-window+1:i+1])
        
        return volatility
    
    def _calculate_support_resistance(self, price_data, close_data, mode='support'):
        """
        计算支撑位或阻力位 - 向量化实现
        
        参数:
            price_data (np.array): 价格数据(最低价/最高价)
            close_data (np.array): 收盘价数据
            mode (str): 模式，'support'表示支撑位，'resistance'表示阻力位
            
        返回:
            np.array: 支撑位或阻力位
        """
        result = np.zeros_like(price_data)
        
        # 默认值使用当前价格
        result[:] = close_data
        
        # 窗口大小
        window = 20
        
        if len(price_data) < window:
            return result
            
        for i in range(window, len(price_data)):
            window_prices = price_data[i-window:i]
            
            if mode == 'support':
                # 支撑位：找出窗口内的低点
                min_indices = np.where(np.diff(np.signbit(np.diff(window_prices))) > 0)[0] + 1
                if len(min_indices) > 0:
                    min_prices = window_prices[min_indices]
                    # 找出最显著的（频率最高或最近的）支撑位
                    if len(min_prices) > 0:
                        result[i] = min_prices[-1]  # 使用最近的低点
            else:
                # 阻力位：找出窗口内的高点
                max_indices = np.where(np.diff(np.signbit(np.diff(window_prices))) < 0)[0] + 1
                if len(max_indices) > 0:
                    max_prices = window_prices[max_indices]
                    # 找出最显著的（频率最高或最近的）阻力位
                    if len(max_prices) > 0:
                        result[i] = max_prices[-1]  # 使用最近的高点
                
        return result
        
    def _analyze_price_structure(self, high, low, close):
        """
        分析价格结构 - 向量化实现
        
        参数:
            high (np.array): 最高价数据
            low (np.array): 最低价数据
            close (np.array): 收盘价数据
            
        返回:
            np.array: 价格结构评分 (正值表示看涨结构，负值表示看跌结构)
        """
        # 计算价格结构评分
        structure_score = np.zeros_like(close)
        
        # 检查数据是否足够
        if len(close) < 20:
            return structure_score
            
        # 使用布林带判断价格通道
        upper, middle, lower = self._calculate_bollinger_bands(close, 20, 2)
        
        # 初始化看涨和看跌计数
        bullish_count = np.zeros_like(close)
        bearish_count = np.zeros_like(close)
        
        # 定义窗口大小
        window = 10
        
        for i in range(window, len(close)):
            # 计算窗口内的高点和低点
            window_high = high[i-window:i]
            window_low = low[i-window:i]
            window_close = close[i-window:i]
            
            # 获取高低点索引
            max_indices = np.where(np.diff(np.signbit(np.diff(window_high))) < 0)[0] + 1
            min_indices = np.where(np.diff(np.signbit(np.diff(window_low))) > 0)[0] + 1
            
            # 检查是否有足够的高低点
            if len(max_indices) >= 2 and len(min_indices) >= 2:
                # 提取高低点价格
                highs = window_high[max_indices]
                lows = window_low[min_indices]
                
                # 判断高点趋势 (高点抬升为看涨, 高点下降为看跌)
                if len(highs) >= 2:
                    if highs[-1] > highs[-2]:
                        bullish_count[i] += 1
                    elif highs[-1] < highs[-2]:
                        bearish_count[i] += 1
                
                # 判断低点趋势 (低点抬升为看涨, 低点下降为看跌)
                if len(lows) >= 2:
                    if lows[-1] > lows[-2]:
                        bullish_count[i] += 1
                    elif lows[-1] < lows[-2]:
                        bearish_count[i] += 1
            
            # 判断布林带位置
            if close[i] > upper[i]:
                bullish_count[i] += 1  # 价格突破上轨，看涨
            elif close[i] < lower[i]:
                bearish_count[i] += 1  # 价格跌破下轨，看跌
                
            # 计算结构得分 (-5到5)
            structure_score[i] = min(5, bullish_count[i]) - min(5, bearish_count[i])
        
        return structure_score
        
    def _identify_market_cycle(self, close, volume, macd):
        """
        识别市场周期 - 向量化实现
        
        参数:
            close (np.array): 收盘价数据
            volume (np.array): 成交量数据
            macd (np.array): MACD指标数据
            
        返回:
            np.array: 市场周期值 (1-积累期, 2-上升期, 3-分配期, 4-下降期)
        """
        # 初始化周期值
        cycle = np.zeros_like(close, dtype=int)
        
        # 计算周期
        for i in range(1, len(close)):
            # 默认为当前周期(延续性)
            cycle[i] = cycle[i-1] if cycle[i-1] > 0 else 1
            
            # 积累期 -> 上升期
            if cycle[i] == 1 and i > 20:
                # 判断条件：MACD由负转正 + 价格突破压力位 + 成交量放大
                if (macd[i] > 0 and macd[i-1] <= 0 and 
                    close[i] > close[i-10:i].max() and 
                    volume[i] > np.mean(volume[i-10:i]) * 1.3):
                    cycle[i] = 2  # 进入上升期
            
            # 上升期 -> 分配期
            elif cycle[i] == 2 and i > 20:
                # 判断条件：价格创新高 + 成交量显著放大 + MACD柱状图缩窄
                if (close[i] > close[i-20:i].max() and 
                    volume[i] > np.mean(volume[i-5:i]) * 1.5 and 
                    abs(macd[i]) < abs(macd[i-1])):
                    cycle[i] = 3  # 进入分配期
            
            # 分配期 -> 下降期
            elif cycle[i] == 3 and i > 20:
                # 判断条件：MACD由正转负 + 价格跌破支撑位
                if (macd[i] < 0 and macd[i-1] >= 0 and 
                    close[i] < close[i-10:i].min()):
                    cycle[i] = 4  # 进入下降期
            
            # 下降期 -> 积累期
            elif cycle[i] == 4 and i > 20:
                # 判断条件：成交量萎缩 + 价格波动减小 + MACD柱状图收窄
                if (volume[i] < np.mean(volume[i-10:i]) * 0.7 and 
                    abs(close[i] - close[i-1]) < np.mean(np.abs(np.diff(close[i-10:i]))) * 0.5 and
                    abs(macd[i]) < abs(macd[i-5:i]).mean()):
                    cycle[i] = 1  # 进入积累期
        
        return cycle

    def _calculate_trend_score(self, data, indicators):
        """计算趋势得分，强化对单日大幅变动的敏感性
        
        参数:
            data: 市场数据
            indicators: 指标数据
            
        返回:
            float: 趋势得分
        """
        # 获取价格和均线数据
        close_price = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2] if len(data['close']) > 1 else close_price
        
        # 获取均线数据
        ma5 = indicators['ma5'][-1]
        ma10 = indicators['ma10'][-1]
        ma20 = indicators['ma20'][-1]
        ma60 = indicators['ma60'][-1]
        
        # 初始化得分
        trend_score = 0
        
        # 1. 单日涨跌幅评分 - 提高对单日大涨的敏感度
        daily_change_pct = (close_price - prev_close) / prev_close * 100
        
        # 单日涨跌得分 - 强化单日大涨影响
        if daily_change_pct >= 8:  # 单日涨幅超过8%，给予更高分数
            trend_score += 5.0  # 大幅加分
        elif daily_change_pct >= 5:  # 5-8%
            trend_score += 4.0
        elif daily_change_pct >= 3:  # 3-5%
            trend_score += 3.0
        elif daily_change_pct >= 1.5:  # 1.5-3%
            trend_score += 2.0
        elif daily_change_pct >= 0.5:  # 0.5-1.5%
            trend_score += 1.0
        elif daily_change_pct <= -8:  # 单日跌幅超过8%
            trend_score -= 5.0
        elif daily_change_pct <= -5:
            trend_score -= 4.0
        elif daily_change_pct <= -3:
            trend_score -= 3.0
        elif daily_change_pct <= -1.5:
            trend_score -= 2.0
        elif daily_change_pct <= -0.5:
            trend_score -= 1.0
            
        # 2. 均线多空排列评分
        if ma5 > ma10 > ma20 > ma60:
            trend_score += 3  # 完美多头排列
        elif ma5 > ma10 > ma20:
            trend_score += 2  # 中期多头排列
        elif ma5 > ma10:
            trend_score += 1  # 短期多头排列
        elif ma5 < ma10 < ma20 < ma60:
            trend_score -= 3  # 完美空头排列
        elif ma5 < ma10 < ma20:
            trend_score -= 2  # 中期空头排列
        elif ma5 < ma10:
            trend_score -= 1  # 短期空头排列
            
        # 3. 均线位置关系评分
        if close_price > ma5 * 1.05:
            trend_score += 1.5  # 大幅站上短期均线
        elif close_price > ma5:
            trend_score += 0.8  # 站上短期均线
        
        if close_price > ma20 * 1.1:
            trend_score += 1.5  # 大幅站上中期均线
        elif close_price > ma20:
            trend_score += 1.0  # 站上中期均线
            
        if close_price > ma60 * 1.15:
            trend_score += 2.0  # 大幅站上长期均线
        elif close_price > ma60:
            trend_score += 1.2  # 站上长期均线
        
        # 4. 动能指标
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd'][-1]
            macd_signal = indicators['macd_signal'][-1]
            macd_hist = indicators['macd_hist'][-1] if 'macd_hist' in indicators else macd - macd_signal
            
            # MACD柱状图放大
            if macd_hist > 0:
                if macd_hist > 0.15:  # 柱状图明显放大
                    trend_score += 1.5
                elif macd_hist > 0.05:
                    trend_score += 0.8
            else:
                if macd_hist < -0.15:
                    trend_score -= 1.5
                elif macd_hist < -0.05:
                    trend_score -= 0.8
                    
            # MACD金叉死叉判断
            if len(indicators['macd']) > 1 and len(indicators['macd_signal']) > 1:
                prev_macd = indicators['macd'][-2]
                prev_signal = indicators['macd_signal'][-2]
                
                # 金叉
                if macd > macd_signal and prev_macd <= prev_signal:
                    trend_score += 2.0  # 加强金叉信号影响
                # 死叉
                elif macd < macd_signal and prev_macd >= prev_signal:
                    trend_score -= 2.0  # 加强死叉信号影响
        
        # 结合RSI指标
        if 'rsi6' in indicators:
            rsi6 = indicators['rsi6'][-1]
            
            if rsi6 > 80:  # 极度超买
                trend_score += 1.5  # 短期动能强
            elif rsi6 > 65:  # 超买
                trend_score += 1.0
            elif rsi6 < 20:  # 极度超卖
                trend_score -= 1.5
            elif rsi6 < 35:  # 超卖
                trend_score -= 1.0
                
        return trend_score
        
    def _determine_market_type(self, total_score):
        """基于总分决定市场类型
        
        特别强化对短期剧烈变化的识别
        
        参数:
            total_score: 总评分
            
        返回:
            int: 市场类型代码
        """
        # 根据总分判断市场类型
        if total_score >= 4.0:  # 提高强势判断的阈值敏感度
            return 11  # 强势突破上涨
        elif total_score >= 3.0:
            return 12  # 加速上涨
        elif total_score >= 2.0:
            return 13  # 稳健上涨
        elif total_score >= 1.0:
            return 21  # 盘升上涨
        elif total_score >= 0.5:
            return 22  # 回调上涨
        elif total_score > 0:
            return 23  # 整固上涨
        elif total_score >= -0.5:
            return 31  # 强势盘整
        elif total_score >= -1.0:
            return 32  # 中性盘整
        elif total_score >= -2.0:
            return 33  # 弱势盘整
        elif total_score >= -3.0:
            return 41  # 盘跌下行
        elif total_score >= -4.0:
            return 42  # 回抽下跌
        elif total_score >= -5.0:
            return 43  # 整固下跌
        elif total_score >= -6.0:
            return 51  # 加速下跌
        elif total_score >= -7.0:
            return 52  # 恐慌下跌
        else:
            return 53  # 超跌反弹
            
    def _determine_composite_market_type(self, index_status):
        """
        简化版本 - 直接返回创业板指数的市场类型
        """
        # 直接返回创业板指数的市场类型
        if self.INDEX_CODE in index_status:
            return index_status[self.INDEX_CODE].get('market_type', 32)  # 默认中性盘整
        
        return 32  # 默认中性盘整