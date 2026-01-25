"""
实盘交易系统主程序

支持两种模式:
1. 选股模式: python main.py --mode select --date 20250109 --num_stocks 3
2. 交易模式: python main.py --mode trade --date 20250109 --total_amount 30000 --num_stocks 3
3. 手动模式: python main.py --mode manual --operation buy --stock 600000.SH --amount 50000
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加必要的路径到sys.path
def _setup_import_paths():
    """
    设置导入路径，确保能找到所有必要的模块
    """
    current_file = Path(__file__).resolve()

    # 1. 添加项目根目录 (to/integrated_stock_selection/)
    project_root = current_file.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 2. 添加xtquant所在目录 (to/)
    to_root = project_root.parent
    if str(to_root) not in sys.path:
        sys.path.insert(0, str(to_root))

    # 3. 检查xtquant是否可访问
    xtquant_path = to_root / 'xtquant'
    if xtquant_path.exists():
        # 确保能找到xtquant模块
        if str(to_root) not in sys.path:
            sys.path.insert(0, str(to_root))

# 执行路径设置
_setup_import_paths()

from live_trading.config import PathConfig, StopLossConfig
from live_trading.model_loader import ModelLoader
from live_trading.stock_selector import StockSelector


def setup_logging():
    """配置日志"""
    log_file = PathConfig.LOG_DIR / f"live_trading_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_factor_data(date: str):
    """
    加载因子数据（加载完整的alpha因子数据）

    Args:
        date: 日期字符串 (YYYYMMDD)

    Returns:
        pd.DataFrame: 包含因子和基础行情的数据
    """
    import sys
    import pandas as pd

    logger = logging.getLogger(__name__)

    try:
        # 添加父目录到sys.path
        parent_dir_str = str(PathConfig.PROJECT_ROOT)
        if parent_dir_str not in sys.path:
            sys.path.insert(0, parent_dir_str)

        # 使用FactorDatasetStore加载因子数据（与main.py一致）
        from factor_system.factor_lab.store import FactorDatasetStore
        from data_loader import IntegratedDataLoader

        # 1. 加载alpha因子数据
        logger.info(f"步骤1: 加载 {date} 的alpha因子...")
        # 注意：PathConfig.PROJECT_ROOT 是 Path 对象，可以直接使用 /
        factor_store_path = PathConfig.PROJECT_ROOT / 'factor_lab_data' / 'factors.parquet'
        store = FactorDatasetStore(factor_store_path)

        if not store.exists():
            logger.error(f"因子数据文件不存在: {factor_store_path}")
            logger.error(f"请先运行 'python main.py --mode generate_factors' 生成因子数据")
            return None

        factor_df = store.load()
        if factor_df is None or factor_df.empty:
            logger.error("因子数据文件为空")
            return None

        # 筛选指定日期的数据
        factor_df['date'] = pd.to_datetime(factor_df['date'])
        factor_df['date_str'] = factor_df['date'].dt.strftime('%Y%m%d')
        prediction_data = factor_df[factor_df['date_str'] == date].copy()

        if prediction_data.empty:
            logger.error(f"因子数据中没有找到 {date} 的数据")
            return None

        logger.info(f"✅ 成功加载alpha因子: {len(prediction_data)}只股票, {len([col for col in prediction_data.columns if col.startswith('alpha_')])}个alpha因子")

        # 2. 加载基础行情数据
        logger.info(f"步骤2: 加载 {date} 的基础行情数据...")
        loader = IntegratedDataLoader()
        stock_data = loader.load_stock_features(date, date)

        if stock_data is None or stock_data.empty:
            logger.error(f"无法加载 {date} 的基础行情数据")
            return None

        stock_data['date_str'] = stock_data['date'].dt.strftime('%Y%m%d')

        logger.info(f"✅ 成功加载基础行情: {len(stock_data)}只股票")

        # 3. 合并数据
        logger.info(f"步骤3: 合并alpha因子和基础行情...")
        # 注意：基础行情中没有 list_date 字段
        base_columns = ['stock_code', 'date', 'is_limit_up', 'is_limit_down', 'amount', 'pct_chg', 'is_zhaban']
        merged_data = pd.merge(
            prediction_data,
            stock_data[base_columns],
            on=['stock_code', 'date'],
            how='left'
        )

        logger.info(f"✅ 成功加载 {date} 的因子数据: {len(merged_data)}只股票")
        logger.info(f"  总列数: {len(merged_data.columns)}")
        logger.info(f"  alpha因子数: {len([col for col in merged_data.columns if col.startswith('alpha_')])}")

        # 打印关键列是否存在
        key_columns = ['stock_code', 'date', 'is_limit_up', 'is_limit_down', 'amount', 'pct_chg']
        missing_columns = [col for col in key_columns if col not in merged_data.columns]
        if missing_columns:
            logger.warning(f"  缺失的关键列: {missing_columns}")

        return merged_data

    except ImportError as e:
        logger.error(f"无法导入模块: {e}")
        logger.error(f"请确保在integrated_stock_selection目录下运行")
        return None
    except Exception as e:
        logger.error(f"加载因子数据失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None


def run_select_mode(args):
    """
    运行选股模式（使用LightGBM单模型，与main.py pred_stocks逻辑一致）

    Args:
        args: 命令行参数
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("选股模式（与main.py pred_stocks逻辑一致）")
    logger.info("=" * 60)
    logger.info(f"日期: {args.date}")
    logger.info(f"选股数量: {args.num_stocks}")

    # 1. 加载LightGBM模型
    logger.info("\n步骤1: 加载LightGBM模型...")
    model_loader = ModelLoader()
    if not model_loader.load_lightgbm_model():
        logger.error("模型加载失败")
        return False

    # 2. 加载因子数据（使用main.py的逻辑）
    logger.info(f"\n步骤2: 加载 {args.date} 的因子数据...")
    factor_data = load_factor_data(args.date)
    if factor_data is None:
        logger.error("因子数据加载失败")
        return False

    # 3. 选股
    logger.info("\n步骤3: 开始选股...")
    selector = StockSelector(model_loader)
    result = selector.select_stocks(args.date, args.num_stocks, factor_data)

    if result is None:
        logger.error("选股失败")
        return False

    # 4. 显示结果
    logger.info("\n" + "=" * 60)
    logger.info("选股结果")
    logger.info("=" * 60)

    if result['stocks']:
        # 加载SW行业映射用于显示
        try:
            import json
            from pathlib import Path

            # 直接加载 SW 行业映射文件
            # 注意：正确路径是 PROJECT_ROOT 的父目录/local_data/sw_sector2/
            sw_sector_file = PathConfig.PROJECT_ROOT.parent / 'local_data' / 'sw_sector2' / 'sw_sector.json'
            if sw_sector_file.exists():
                with open(sw_sector_file, 'r', encoding='utf-8') as f:
                    sw_sector_data = json.load(f)
                stock_to_industry = sw_sector_data

                logger.info(f"{'排名':<4} {'股票代码':<12} {'预测值(ATR)':<15} {'SW行业':<20}")
                logger.info("-" * 60)
                for stock in result['stocks']:
                    industry = stock_to_industry.get(stock['code'], '未知')
                    logger.info(f"{stock['rank']:<4} {stock['code']:<12} {stock['score']:<15.3f} {industry:<20}")
            else:
                logger.warning(f"SW行业映射文件不存在: {sw_sector_file}")
                for stock in result['stocks']:
                    logger.info(f"  {stock['rank']}. {stock['code']} - 分数: {stock['score']:.3f}")
        except Exception as e:
            logger.warning(f"无法加载SW行业映射: {e}")
            for stock in result['stocks']:
                logger.info(f"  {stock['rank']}. {stock['code']} - 分数: {stock['score']:.3f}")
    else:
        logger.warning("  无符合条件的股票")

    logger.info(f"\n结果已保存到: {PathConfig.STOCK_SELECTIONS_FILE}")
    logger.info("=" * 60)

    return True


def run_trade_mode(args):
    """
    运行交易模式（状态机版本）

    状态机逻辑：
    1. 有持仓第2天+ → 进入5分钟监控模式
    2. 有持仓第1天 → 提示无需监控，退出
    3. 无持仓 + 今天是买入日 → 执行买入，退出
    4. 无持仓 + 今天不是买入日 → 退出

    Args:
        args: 命令行参数
    """
    import pandas as pd
    from live_trading.trader import XtQuantTrader, TradingScheduler
    from live_trading.monitor import PositionMonitor5Min

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("交易模式（状态机版本）")
    logger.info("=" * 60)
    logger.info(f"选股日期(T日): {args.date}")

    # 1. 连接迅投API
    logger.info("\n步骤1: 连接迅投API...")
    trader = XtQuantTrader()
    if not trader.connect():
        logger.error("连接失败")
        return False

    scheduler = TradingScheduler(trader)

    # 2. 查询账户持仓
    logger.info("\n步骤2: 查询账户持仓...")
    positions = trader.query_stock_positions(trader.account)

    # 调试：检查返回值
    if positions is None:
        logger.warning("⚠️ query_stock_positions返回None，可能是连接问题")
        trader.disconnect()
        return False

    logger.info(f"📊 查询到持仓数量: {len(positions)}")
    if positions:
        for i, pos in enumerate(positions, 1):
            logger.info(f"  持仓{i}: {pos.stock_code}, 持仓数量: {pos.volume}股, 可用数量: {pos.can_use_volume}股")

    # 过滤出有效持仓（持仓数量 >= 100股 = 1手）
    # 注意：使用volume而不是can_use_volume，因为买入当天可用数量为0（T+1交易制度）
    valid_positions = [pos for pos in positions if pos.volume >= 100]

    # 用于收集所有启动的监控器，以便在主线程退出前等待它们结束
    active_monitors = []

    if valid_positions:
        # ========================================
        # 情况1&2: 有有效持仓
        # ========================================
        logger.info(f"发现有效持仓: {len(valid_positions)}只股票")

        for pos in valid_positions:
            stock_code = pos.stock_code
            holding_volume = pos.volume
            volume = pos.can_use_volume

            logger.info(f"  {stock_code}: {holding_volume}股")

            logger.info(f"\n{'='*60}")
            logger.info(f"处理持仓: {stock_code}")
            logger.info(f"{'='*60}")
            logger.info(f"  持仓数量: {holding_volume}股")
            logger.info(f"  可用数量: {volume}股")

            # 加载买入记录
            buy_record = scheduler.load_latest_buy_record(stock_code)

            if not buy_record:
                logger.warning(f"  未找到{stock_code}的买入记录，无法监控")
                logger.warning(f"  请手动管理此持仓")
                continue

            entry_date = buy_record['entry_date']
            entry_price = buy_record.get('entry_price', 0)
            record_volume = buy_record.get('volume', 0)

            logger.info(f"  买入日期: {entry_date}")
            logger.info(f"  买入价格: {entry_price:.2f}元" if entry_price > 0 else "  买入价格: 未知")
            logger.info(f"  买入数量: {record_volume}股")

            # 计算持仓天数
            holding_days = scheduler.calculate_holding_days(entry_date)
            logger.info(f"  持仓天数: {holding_days}天")

            # 判断持仓天数
            if holding_days == 0:
                # ========================================
                # 情况2: 持仓第1天（当天买入）
                # ========================================
                logger.info(f"\n✅ {stock_code}: 当天买入（T+0），无法卖出，无需监控")
                logger.info("=" * 60)
                continue

            elif holding_days == 1:
                # ========================================
                # 情况1a: 持仓第2天，启动5分钟监控（无尾盘卖出）
                # ========================================
                logger.info(f"\n🛡️ {stock_code}: 持仓第2天，启动5分钟风控监控（无尾盘卖出）")

                # 创建监控器
                monitor_5m = PositionMonitor5Min(trader, scheduler)
                monitor_5m.enable_closing_sell = False  # 第2天不尾盘卖出

                # 添加持仓
                monitor_5m.add_position_5m(
                    stock_code,
                    entry_price,
                    entry_date
                )

                # 启动监控
                monitor_5m.start_5m_monitoring()

                # 将监控器添加到列表，用于后续等待
                active_monitors.append(monitor_5m)

                logger.info(f"✅ 5分钟K线监控已启动")
                logger.info(f"  监控股票: {stock_code}")
                logger.info(f"  买入价格: {entry_price:.2f}元")
                logger.info(f"  买入日期: {entry_date}")
                logger.info(f"  持仓天数: {holding_days}天")
                logger.info(f"  风控规则: 累计下跌5% OR 连续3根阴线")
                logger.info(f"  尾盘卖出: 否")
                logger.info("")

                # 监控在后台线程运行，主线程继续处理下一个持仓
                # 不再需要等待监控结束（删除了阻塞的while循环）
                continue

            elif holding_days == 2:
                # ========================================
                # 情况1b: 持仓第3天，启动5分钟监控（14:55尾盘卖出）
                # ========================================
                logger.info(f"\n🛡️ {stock_code}: 持仓第3天，启动5分钟风控监控（14:55尾盘卖出）")

                # 创建监控器
                monitor_5m = PositionMonitor5Min(trader, scheduler)
                monitor_5m.enable_closing_sell = True  # 第3天启用尾盘卖出

                # 添加持仓
                monitor_5m.add_position_5m(
                    stock_code,
                    entry_price,
                    entry_date
                )

                # 启动监控
                monitor_5m.start_5m_monitoring()

                # 将监控器添加到列表，用于后续等待
                active_monitors.append(monitor_5m)

                logger.info(f"✅ 5分钟K线监控已启动")
                logger.info(f"  监控股票: {stock_code}")
                logger.info(f"  买入价格: {entry_price:.2f}元")
                logger.info(f"  买入日期: {entry_date}")
                logger.info(f"  持仓天数: {holding_days}天")
                logger.info(f"  风控规则: 累计下跌5% OR 连续3根阴线")
                logger.info(f"  尾盘卖出: 14:55")
                logger.info("")

                # 监控在后台线程运行，主线程继续处理下一个持仓
                # 不再需要等待监控结束（删除了阻塞的while循环）
                continue

            elif holding_days >= 3:
                # ========================================
                # 情况1c: 持仓满3天（异常情况），立即卖出
                # ========================================
                logger.warning(f"\n⚠️ {stock_code}: 持仓异常（{holding_days}天），立即卖出")

                # 获取最新价
                current_price = trader.get_current_price(stock_code)
                if current_price and entry_price > 0:
                    profit_pct = (current_price - entry_price) / entry_price * 100
                    logger.info(f"  买入价: {entry_price:.2f}, 卖出价: {current_price:.2f}, 收益率: {profit_pct:+.2f}%")
                else:
                    profit_pct = None

                # 执行卖出
                order_id = trader.order_manager.smart_sell_full_position(stock_code, volume)

                if order_id:
                    logger.info(f"✅ 卖出成功: {stock_code}, 订单号={order_id}")

                    # 保存记录
                    scheduler.save_trade_record(
                        args.date,
                        'sell',
                        [stock_code],
                        {stock_code: order_id},
                        exit_reason='持仓异常',
                        profit_pct=profit_pct
                    )
                else:
                    logger.error(f"❌ 卖出失败: {stock_code}")

                continue

        # 所有持仓处理完成
        logger.info("\n" + "=" * 60)
        logger.info("持仓处理完成")
        logger.info("=" * 60)

        # 等待所有监控线程结束
        if active_monitors:
            logger.info(f"\n⏳ 等待 {len(active_monitors)} 个监控线程结束...")
            logger.info("💡 提示: 按 Ctrl+C 可随时停止监控")

            try:
                for idx, monitor in enumerate(active_monitors, 1):
                    logger.info(f"  [{idx}/{len(active_monitors)}] 等待监控结束...")

                    # 等待监控线程结束（主线程阻塞在这里）
                    if monitor.monitor_thread and monitor.monitor_thread.is_alive():
                        monitor.monitor_thread.join()
                        logger.info(f"  [{idx}/{len(active_monitors)}] ✅ 监控线程已结束")

                logger.info("\n✅ 所有监控线程已结束")

            except KeyboardInterrupt:
                logger.info("\n\n⚠️ 用户手动停止监控")
                logger.info("正在停止所有监控线程...")

                # 停止所有监控
                for monitor in active_monitors:
                    monitor.stop_5m_monitoring()

                logger.info("✅ 所有监控已停止")
                logger.info("程序即将退出...")

    else:
        # ========================================
        # 情况3&4: 无持仓
        # ========================================
        logger.info("账户无持仓")

        # 计算交易日期
        logger.info("\n步骤3: 计算交易日期...")
        # args.date 是 T+1日（买入日）
        buy_date = pd.to_datetime(args.date, format='%Y%m%d')
        logger.info(f"T+1日(买入日): {buy_date.strftime('%Y-%m-%d')}")

        # 反推 T日（选股日）：往前推最近的交易日
        t_date = buy_date - pd.Timedelta(days=1)
        while t_date.weekday() >= 5:  # 跳过周末
            t_date -= pd.Timedelta(days=1)
        logger.info(f"T日(选股日): {t_date.strftime('%Y-%m-%d')}")

        today = datetime.now().date()
        buy_date_date = buy_date.date()

        if today != buy_date_date:
            # ========================================
            # 情况4: 今天不是买入日
            # ========================================
            logger.info(f"\n当前日期{today}不是买入日{buy_date_date}，无需操作")
            trader.disconnect()
            return True

        # ========================================
        # 情况3: 今天是买入日，执行买入
        # ========================================
        logger.info(f"\n步骤4: 加载选股结果...")

        if not PathConfig.STOCK_SELECTIONS_FILE.exists():
            logger.error(f"选股文件不存在: {PathConfig.STOCK_SELECTIONS_FILE}")
            trader.disconnect()
            return False

        try:
            with open(PathConfig.STOCK_SELECTIONS_FILE, 'r', encoding='utf-8') as f:
                selections = json.load(f)
        except Exception as e:
            logger.error(f"加载选股文件失败: {e}")
            trader.disconnect()
            return False

        # 使用选股日（t_date）查找选股记录
        t_date_str = t_date.strftime('%Y%m%d')
        if t_date_str not in selections:
            logger.error(f"未找到日期{t_date_str}的选股结果")
            logger.info(f"可用的选股日期: {list(selections.keys())}")
            trader.disconnect()
            return False

        selected_stocks = selections[t_date_str]['stocks']
        if not selected_stocks:
            logger.warning("选股结果为空,无法交易")
            trader.disconnect()
            return False

        # 提取top1股票
        top1_stocks = [s for s in selected_stocks if s['rank'] == 1]
        if not top1_stocks:
            logger.error("未找到rank=1的股票")
            trader.disconnect()
            return False

        top1_stock = top1_stocks[0]
        stock_code = top1_stock['code']
        score = top1_stock['score']

        logger.info(f"✅ 找到top1股票: {stock_code}, 分数: {score:.3f}")

        # 执行买入
        logger.info(f"\n步骤5: 执行买入...")
        logger.info(f"目标股票: {stock_code}")

        order_id = trader.order_manager.smart_buy_full_position(
            stock_code,
            buy_date.strftime('%Y-%m-%d')
        )

        if order_id:
            logger.info(f"✅ 买入成功: {stock_code}, 订单号={order_id}")

            # 获取买入价格和数量
            entry_price = trader.get_current_price(stock_code)
            if not entry_price and order_id in trader.order_manager.orders:
                entry_price = trader.order_manager.orders[order_id]['price']

            # 获取买入数量
            volume = 0
            if order_id in trader.order_manager.orders:
                volume = trader.order_manager.orders[order_id].get('volume', 0)

            # 保存记录（增强版）
            scheduler.save_trade_record(
                args.date,
                'buy',
                [stock_code],
                {stock_code: order_id},
                entry_date=buy_date.strftime('%Y-%m-%d'),
                entry_price=entry_price,
                volume=volume
            )

            logger.info(f"买入记录已保存:")
            logger.info(f"  股票代码: {stock_code}")
            logger.info(f"  买入日期: {buy_date.strftime('%Y-%m-%d')}")
            logger.info(f"  买入价格: {entry_price:.2f}元" if entry_price else "  买入价格: 未知")
            logger.info(f"  买入数量: {volume}股")

            logger.info(f"\nℹ️  买入完成，程序退出")
            logger.info(f"ℹ️  明天再次运行程序将启动5分钟风控监控")

        else:
            logger.error(f"❌ 买入失败: {stock_code}")
            trader.disconnect()
            return False

    # 断开连接
    trader.disconnect()

    logger.info("\n" + "=" * 60)
    logger.info("交易完成")
    logger.info("=" * 60)

    return True


def run_manual_mode(args):
    """
    手动交易模式

    支持命令:
    - 买入: --mode manual --operation buy --stock 600000.SH --amount 50000
    - 卖出: --mode manual --operation sell --stock 600000.SH --lots 10
    - 全仓卖出: --mode manual --operation sell --stock 600000.SH --full

    Args:
        args: 命令行参数

    Returns:
        bool: 执行是否成功
    """
    from live_trading.trader import XtQuantTrader, TradingScheduler

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("手动交易模式")
    logger.info("=" * 60)
    logger.info(f"操作类型: {'买入' if args.operation == 'buy' else '卖出'}")
    logger.info(f"目标股票: {args.stock}")

    if args.operation == 'buy':
        logger.info(f"最大金额: {args.amount:.2f}元")
    else:
        if args.lots:
            logger.info(f"卖出数量: {args.lots}手")
        else:
            logger.info(f"卖出方式: 全仓")

    # 1. 连接迅投API
    logger.info("\n步骤1: 连接迅投API...")
    trader = XtQuantTrader()
    if not trader.connect():
        logger.error("连接失败")
        return False

    scheduler = TradingScheduler(trader)

    try:
        # 2. 根据操作类型执行
        if args.operation == 'buy':
            success = _execute_manual_buy(args, trader, scheduler)
        elif args.operation == 'sell':
            success = _execute_manual_sell(args, trader, scheduler)
        else:
            logger.error(f"未知操作类型: {args.operation}")
            success = False

        if success:
            logger.info("\n" + "=" * 60)
            logger.info(f"手动{'买入' if args.operation == 'buy' else '卖出'}完成")
            logger.info("=" * 60)

        return success

    finally:
        # 断开连接
        trader.disconnect()


def _execute_manual_buy(args, trader, scheduler):
    """
    执行手动买入

    买入逻辑：
    1. 查询账户可用资金
    2. 获取股票当前价格
    3. 根据当前价格计算最大可买手数（考虑资金限制）
    4. 根据用户指定金额计算计划买入手数
    5. 取两者的最小值作为最终买入手数
    6. 执行买入

    Args:
        args: 命令行参数
        trader: XtQuantTrader实例
        scheduler: TradingScheduler实例

    Returns:
        bool: 执行是否成功
    """
    logger = logging.getLogger(__name__)

    stock_code = args.stock
    user_max_amount = args.amount

    # 2. 查询账户资金
    logger.info("\n步骤2: 查询账户资金...")

    # account是StockAccount对象，需要调用query_stock_asset来获取资产
    asset = trader.query_stock_asset(trader.account)

    if not asset:
        logger.error("无法查询账户资产")
        return False

    available_cash = asset.cash

    if available_cash is None:
        logger.error("无法获取可用资金")
        return False

    logger.info(f"  可用资金: {available_cash:.2f}元")
    logger.info(f"  用户指定金额: {user_max_amount:.2f}元")

    # 验证资金
    if user_max_amount > available_cash:
        logger.error(f"❌ 资金不足！可用资金{available_cash:.2f}元，小于指定金额{user_max_amount:.2f}元")
        return False

    # 3. 获取股票当前价格
    logger.info("\n步骤3: 获取股票当前价格...")
    current_price = trader.get_current_price(stock_code)

    if not current_price:
        logger.error(f"无法获取{stock_code}的当前价格")
        return False

    logger.info(f"  当前价格: {current_price:.2f}元")

    # 4. 计算可买手数
    logger.info("\n步骤4: 计算可买手数...")

    # 计算最大可买手数（基于可用资金）
    max_lots_by_cash = int(available_cash / (current_price * 100))
    max_amount_by_cash = max_lots_by_cash * 100 * current_price

    # 计算计划买入手数（基于用户指定金额）
    planned_lots = int(user_max_amount / (current_price * 100))
    planned_amount = planned_lots * 100 * current_price

    # 最终买入手数（取最小值）
    buy_lots = min(max_lots_by_cash, planned_lots)

    logger.info(f"  资金最大可买: {max_lots_by_cash}手（{max_amount_by_cash:.2f}元）")
    logger.info(f"  计划买入: {planned_lots}手（{planned_amount:.2f}元）")
    logger.info(f"  最终买入: {buy_lots}手")

    # 验证至少能买1手
    if buy_lots < 1:
        logger.error(f"❌ 资金不足！当前价格{current_price:.2f}元，1手需要{current_price * 100:.2f}元")
        return False

    buy_volume = buy_lots * 100
    buy_amount = buy_volume * current_price

    logger.info(f"  买入数量: {buy_volume}股")
    logger.info(f"  预计金额: {buy_amount:.2f}元")

    # 5. 执行买入
    logger.info("\n步骤5: 执行买入...")
    logger.info(f"目标股票: {stock_code}")
    logger.info(f"买入方式: 限价单（价格={current_price:.2f}元）")

    order_id = trader.order_manager.smart_buy_full_position(
        stock_code,
        datetime.now().strftime('%Y-%m-%d'),
        target_volume=buy_volume  # 指定买入数量
    )

    if order_id:
        logger.info(f"✅ 买入成功: {stock_code}")
        logger.info(f"  订单号: {order_id}")
        logger.info(f"  买入价格: {current_price:.2f}元")
        logger.info(f"  买入数量: {buy_volume}股")
        logger.info(f"  实际金额: {buy_amount:.2f}元")

        # 6. 保存交易记录
        entry_date = datetime.now().strftime('%Y-%m-%d')
        scheduler.save_trade_record(
            datetime.now().strftime('%Y%m%d'),
            'manual_buy',
            [stock_code],
            {stock_code: order_id},
            entry_date=entry_date,
            entry_price=current_price,
            volume=buy_volume
        )

        logger.info(f"\n买入记录已保存:")
        logger.info(f"  股票代码: {stock_code}")
        logger.info(f"  买入日期: {entry_date}")
        logger.info(f"  买入价格: {current_price:.2f}元")
        logger.info(f"  买入数量: {buy_volume}股")
        logger.info(f"  买入金额: {buy_amount:.2f}元")
        logger.info(f"  操作模式: 手动交易")

        return True
    else:
        logger.error(f"❌ 买入失败: {stock_code}")
        return False


def _execute_manual_sell(args, trader, scheduler):
    """
    执行手动卖出

    Args:
        args: 命令行参数
        trader: XtQuantTrader实例
        scheduler: TradingScheduler实例

    Returns:
        bool: 执行是否成功
    """
    logger = logging.getLogger(__name__)

    stock_code = args.stock

    # 2. 查询持仓
    logger.info("\n步骤2: 查询持仓...")
    positions = trader.query_stock_positions(trader.account)

    if not positions:
        logger.error(f"账户无持仓或未找到{stock_code}的持仓")
        return False

    # 查找目标股票的持仓
    target_position = None
    for pos in positions:
        if pos.stock_code == stock_code:
            target_position = pos
            break

    if not target_position:
        logger.error(f"未找到{stock_code}的持仓")
        return False

    available_volume = target_position.can_use_volume
    logger.info(f"  持仓数量: {available_volume}股")

    if available_volume < 100:
        logger.error(f"可用数量不足1手，无法卖出")
        return False

    # 3. 计算卖出数量
    logger.info("\n步骤3: 计算卖出数量...")

    if args.full:
        # 全仓卖出
        sell_volume = available_volume
        sell_lots = sell_volume // 100
        logger.info(f"  卖出方式: 全仓")
        logger.info(f"  卖出数量: {sell_volume}股（{sell_lots}手）")
    else:
        # 指定手数卖出
        sell_lots = args.lots
        sell_volume = sell_lots * 100

        if sell_volume > available_volume:
            logger.error(f"❌ 持仓不足！持仓{available_volume}股，小于指定数量{sell_volume}股")
            return False

        logger.info(f"  卖出方式: 指定手数")
        logger.info(f"  卖出数量: {sell_volume}股（{sell_lots}手）")

    # 4. 获取当前价格
    logger.info("\n步骤4: 获取当前价格...")
    current_price = trader.get_current_price(stock_code)

    if not current_price:
        logger.warning(f"无法获取{stock_code}的当前价格，使用市价卖出")
    else:
        logger.info(f"  当前价格: {current_price:.2f}元")

    # 5. 执行卖出
    logger.info("\n步骤5: 执行卖出...")
    logger.info(f"目标股票: {stock_code}")

    order_id = trader.order_manager.smart_sell_full_position(
        stock_code,
        sell_volume
    )

    if order_id:
        logger.info(f"✅ 卖出成功: {stock_code}")
        logger.info(f"  订单号: {order_id}")

        if current_price:
            sell_amount = sell_volume * current_price
            logger.info(f"  卖出价格: {current_price:.2f}元")
            logger.info(f"  卖出数量: {sell_volume}股")
            logger.info(f"  卖出金额: {sell_amount:.2f}元")

            # 计算收益率
            buy_record = scheduler.load_latest_buy_record(stock_code)
            if buy_record and 'entry_price' in buy_record:
                entry_price = buy_record['entry_price']
                profit_pct = (current_price - entry_price) / entry_price * 100
                logger.info(f"  买入价格: {entry_price:.2f}元")
                logger.info(f"  收益率: {profit_pct:+.2f}%")
            else:
                profit_pct = None

        # 6. 保存交易记录
        scheduler.save_trade_record(
            datetime.now().strftime('%Y%m%d'),
            'manual_sell',
            [stock_code],
            {stock_code: order_id},
            exit_reason='手动卖出',
            profit_pct=profit_pct
        )

        logger.info(f"\n卖出记录已保存:")
        logger.info(f"  股票代码: {stock_code}")
        logger.info(f"  卖出日期: {datetime.now().strftime('%Y-%m-%d')}")
        if current_price:
            logger.info(f"  卖出价格: {current_price:.2f}元")
        if profit_pct is not None:
            logger.info(f"  收益率: {profit_pct:+.2f}%")
        logger.info(f"  卖出数量: {sell_volume}股")
        if args.full:
            logger.info(f"  卖出方式: 全仓卖出")
        else:
            logger.info(f"  卖出方式: {sell_lots}手")
        logger.info(f"  操作模式: 手动交易")

        return True
    else:
        logger.error(f"❌ 卖出失败: {stock_code}")
        return False



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实盘交易系统')

    parser.add_argument('--mode', type=str, required=True, choices=['select', 'trade', 'manual'],
                       help='运行模式: select-选股模式, trade-交易模式, manual-手动交易模式')

    parser.add_argument('--date', type=str,
                       help='日期(格式: YYYYMMDD)，select和trade模式必需')

    parser.add_argument('--num_stocks', type=int, default=3,
                       help='选股数量（select模式，默认3）')

    # manual模式参数
    parser.add_argument('--operation', type=str, choices=['buy', 'sell'],
                       help='操作类型: buy-买入, sell-卖出（manual模式必需）')

    parser.add_argument('--stock', type=str,
                       help='股票代码（manual模式必需）')

    parser.add_argument('--amount', type=float,
                       help='最大买入金额（元），manual买入模式必需')

    parser.add_argument('--lots', type=int,
                       help='卖出手数（manual卖出模式可选）')

    parser.add_argument('--full', action='store_true',
                       help='全仓卖出（manual卖出模式可选，与--lots互斥）')

    args = parser.parse_args()

    # 参数验证
    if args.mode in ['select', 'trade'] and not args.date:
        parser.error(f"{args.mode}模式需要--date参数")

    if args.mode == 'manual':
        if not all([args.operation, args.stock]):
            parser.error("manual模式需要--operation和--stock参数")

        if args.operation == 'buy' and not args.amount:
            parser.error("manual买入模式需要--amount参数")

        if args.operation == 'sell' and not any([args.lots, args.full]):
            parser.error("manual卖出模式需要--lots或--full参数")

        if args.operation == 'sell' and args.lots and args.full:
            parser.error("--lots和--full参数互斥")

    # 配置日志
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info(f"实盘交易系统启动 - 模式: {args.mode}")

    try:
        if args.mode == 'select':
            success = run_select_mode(args)
        elif args.mode == 'trade':
            success = run_trade_mode(args)
        elif args.mode == 'manual':
            success = run_manual_mode(args)
        else:
            logger.error(f"未知模式: {args.mode}")
            return

        if success:
            logger.info("✅ 执行成功")
        else:
            logger.error("❌ 执行失败")

    except Exception as e:
        logger.error(f"执行异常: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")


if __name__ == '__main__':
    main()

