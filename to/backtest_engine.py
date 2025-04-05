import os
import pickle
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    # ... 这里是类的其他方法 ...
    
    def load_rolling_signals(self, long_cycle_update=10, short_cycle_update=1):
        """加载滚动训练模式下的交易信号
        
        参数:
            long_cycle_update: 长周期模型更新间隔(天)
            short_cycle_update: 短周期模型更新间隔(天)
        """
        logger.info("开始加载滚动训练模式下的交易信号")
        
        try:
            # 创建高级量化模型实例
            from advanced_quant_model import AdvancedQuantModel
            model = AdvancedQuantModel()
            
            # 加载交易日历
            logger.info("加载交易日历数据...")
            model.load_trading_dates()
            
            # 找到开始和结束日期的索引
            try:
                start_idx = model.trading_dates.index(self.start_date)
                end_idx = model.trading_dates.index(self.end_date)
            except ValueError:
                logger.error(f"无法在交易日历中找到开始日期或结束日期")
                return False
            
            # 初始化自适应更新参数
            last_long_update = None
            last_short_update = None
            model_performance_history = []
            market_condition = 'normal'  # 市场状态：normal, volatile, trending
            
            # 批量处理大小
            batch_size = 20
            
            # 计算总批次数
            total_batches = (end_idx - start_idx + 1) // batch_size + (1 if (end_idx - start_idx + 1) % batch_size > 0 else 0)
            logger.info(f"信号生成将分 {total_batches} 批次执行")
            
            # 创建信号字典
            all_signals = {}
            
            # 加载最佳参数 (如果有)
            best_params = {}
            if os.path.exists('best_params.pkl'):
                try:
                    with open('best_params.pkl', 'rb') as f:
                        loaded_params = pickle.load(f)
                        # 参数名称映射
                        param_mapping = {
                            'short': 'lookback_short',
                            'long': 'lookback_long'
                        }
                        # 转换参数名称
                        best_params = {}
                        for key, value in loaded_params.items():
                            # 如果有对应的映射名称，使用映射名称
                            mapped_key = param_mapping.get(key, key)
                            # 确保数字类型参数被正确转换
                            if mapped_key in ['lookback_short', 'lookback_long', 'forward_days']:
                                best_params[mapped_key] = int(value) if isinstance(value, str) else value
                            elif mapped_key == 'test_size':
                                best_params[mapped_key] = float(value) if isinstance(value, str) else value
                            else:
                                best_params[mapped_key] = value
                    logger.info(f"加载最佳参数: {best_params}")
                except Exception as e:
                    logger.warning(f"加载最佳参数失败: {e}")
            
            # 分批处理交易日期
            for batch_idx in range(total_batches):
                batch_start_idx = start_idx + batch_idx * batch_size
                batch_end_idx = min(batch_start_idx + batch_size - 1, end_idx)
                
                batch_start_date = model.trading_dates[batch_start_idx]
                batch_end_date = model.trading_dates[batch_end_idx]
                
                logger.info(f"处理批次 {batch_idx+1}/{total_batches}: {batch_start_date} 至 {batch_end_date}")
                
                # 处理这个批次中的每一天
                for day_idx in range(batch_start_idx, batch_end_idx + 1):
                    current_date = model.trading_dates[day_idx]
                    
                    # 跟踪已经处理过的交易日数量
                    processed_trading_days = day_idx - start_idx + 1
                    
                    # 分析当前市场状态
                    if day_idx > start_idx + 20:  # 需要一些历史数据
                        market_features = model._extract_market_features(current_date, lookback_days=20)
                        
                        # 根据市场波动性和趋势强度评估市场状态
                        if market_features is not None:
                            market_volatility = market_features.get('volatility', 0)
                            market_trend = market_features.get('trend_strength', 0)
                            
                            if market_volatility > 0.02:  # 高波动
                                market_condition = 'volatile'
                            elif market_trend > 0.05:  # 明显趋势
                                market_condition = 'trending'
                            else:
                                market_condition = 'normal'
                                
                            logger.info(f"日期 {current_date} 市场状态: {market_condition}, 波动率: {market_volatility:.4f}, 趋势强度: {market_trend:.4f}")
                    
                    # 根据市场状态和模型性能决定是否更新模型
                    need_long_update = False
                    need_short_update = False
                    
                    # 检查上次长周期更新后的表现
                    if len(model_performance_history) >= 5:
                        recent_performance = sum(model_performance_history[-5:]) / 5
                        if recent_performance < 0.3:  # 表现不佳
                            logger.info(f"最近5次预测表现不佳 (准确率: {recent_performance:.2f})，触发模型更新")
                            need_long_update = True
                    
                    # 根据市场状态动态调整更新策略
                    days_since_long_update = ((day_idx - model.trading_dates.index(last_long_update)) 
                                           if last_long_update else long_cycle_update)
                    
                    days_since_short_update = ((day_idx - model.trading_dates.index(last_short_update)) 
                                            if last_short_update else short_cycle_update)
                    
                    # 高波动市场需要更频繁更新
                    if market_condition == 'volatile' and days_since_short_update >= 1:
                        need_short_update = True
                        if days_since_long_update >= 5:  # 高波动市场缩短长周期更新间隔
                            need_long_update = True
                    
                    # 趋势市场可以较少更新
                    elif market_condition == 'trending' and days_since_short_update >= 3:
                        need_short_update = True
                        if days_since_long_update >= 15:  # 趋势市场可以延长更新间隔
                            need_long_update = True
                    
                    # 普通市场使用标准间隔
                    elif days_since_short_update >= short_cycle_update:
                        need_short_update = True
                        if days_since_long_update >= long_cycle_update:
                            need_long_update = True
                    
                    # 执行长周期模型更新
                    if need_long_update:
                        logger.info(f"执行长周期模型更新，日期: {current_date}，处理进度: {processed_trading_days}/{end_idx-start_idx+1}")
                        
                        # 使用最佳参数或默认参数
                        lookback_long = int(best_params.get('lookback_long', 90))
                        lookback_short = int(best_params.get('lookback_short', 30))
                        parallel = best_params.get('parallel', 'process_pool')
                        
                        # 长周期训练使用较大的数据窗口
                        success = model.train_sector_models(
                            date=current_date,
                            lookback_short=lookback_short,
                            lookback_long=lookback_long,
                            parallel=parallel
                        )
                        
                        if success:
                            logger.info(f"长周期模型更新成功")
                            last_long_update = current_date
                            last_short_update = current_date  # 长周期更新也算作短周期更新
                        else:
                            logger.error(f"长周期模型更新失败")
                    
                    # 执行短周期模型更新
                    elif need_short_update:
                        logger.info(f"执行短周期模型更新，日期: {current_date}，处理进度: {processed_trading_days}/{end_idx-start_idx+1}")
                        
                        # 短周期训练只更新短周期参数
                        lookback_short = int(best_params.get('lookback_short', 30))
                        
                        # 短周期训练使用较小的数据窗口
                        success = model.train_sector_models(
                            date=current_date,
                            lookback_short=lookback_short,
                            lookback_long=0,  # 不使用长周期数据
                            parallel="sequential"  # 使用简单的顺序处理即可
                        )
                        
                        if success:
                            logger.info(f"短周期模型更新成功")
                            last_short_update = current_date
                        else:
                            logger.warning(f"短周期模型更新失败")
                    
                    # 生成当天的交易信号
                    signals = model.generate_trading_signals(date=current_date)
                    if signals:
                        logger.info(f"日期 {current_date} 生成 {len(signals)} 个交易信号")
                        all_signals[current_date] = signals
                        
                        # 如果已有历史数据，评估上次信号的准确性
                        if len(model_performance_history) > 0 and day_idx > start_idx + 5:
                            prev_date = model.trading_dates[day_idx - 5]  # 5个交易日前
                            if prev_date in all_signals:
                                # 评估5天前生成的信号准确性
                                prev_signals = all_signals[prev_date]
                                correct_predictions = 0
                                total_predictions = len(prev_signals)
                                
                                if total_predictions > 0:
                                    # 获取现在与5天前的价格数据比较
                                    for signal in prev_signals:
                                        stock_code = signal['stock_code']
                                        price_data_prev = model._get_stock_price(stock_code, prev_date)
                                        price_data_current = model._get_stock_price(stock_code, current_date)
                                        
                                        if price_data_prev and price_data_current:
                                            prev_price = price_data_prev['close']
                                            current_price = price_data_current['close']
                                            actual_return = (current_price - prev_price) / prev_price
                                            
                                            # 检查预测是否正确
                                            if (signal['signal_type'] == 'buy' and actual_return > 0) or \
                                               (signal['signal_type'] == 'sell' and actual_return < 0):
                                                correct_predictions += 1
                                    
                                    accuracy = correct_predictions / total_predictions
                                    model_performance_history.append(accuracy)
                                    logger.info(f"5天前 ({prev_date}) 信号准确率: {accuracy:.2f}, 正确: {correct_predictions}, 总数: {total_predictions}")
                    else:
                        logger.warning(f"日期 {current_date} 未生成交易信号")
            
            # 设置信号字典
            self.trading_signals = all_signals
            logger.info(f"成功加载 {len(all_signals)} 个交易日的信号")
            return True
            
        except Exception as e:
            logger.exception(f"加载滚动训练信号时发生异常: {e}")
            return False
    
    # ... 这里是类的其他方法 ... 