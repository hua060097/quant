"""
实盘交易系统主程序

支持多种模式:
1. 选股模式: python main.py --mode select --date 20250109 --num_stocks 3
2. 交易模式: python main.py --mode trade --date 20250109
3. 手动模式: python main.py --mode manual --operation buy --stock 600000.SH --amount 50000
4. 同步模式: python main.py --mode sync --date 20260422
"""

import sys
import json
import time
import argparse
import logging
import os
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

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

from live_trading.config import PathConfig, StopLossConfig, StrategyConfig
from live_trading.model_loader import ModelLoader
from live_trading.stock_selector import StockSelector


_SHUTDOWN_EVENT = threading.Event()
_SIGINT_COUNT = 0


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


def _install_interrupt_handlers(logger: logging.Logger) -> None:
    """让 Ctrl+C 具备可预测的退出路径。第二次中断直接强退。"""
    def _handler(signum, frame):
        global _SIGINT_COUNT
        _SIGINT_COUNT += 1
        if _SIGINT_COUNT >= 2:
            logger.warning("收到重复中断信号，强制退出")
            os._exit(130)
        logger.warning("收到中断信号，开始停止流程；再次按 Ctrl+C 将强制退出")
        _SHUTDOWN_EVENT.set()
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except Exception as e:
        logger.warning(f"安装中断处理失败: {e}")


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


def _load_daily_gate_factor_history():
    import pandas as pd

    factor_store_path = PathConfig.PROJECT_ROOT / 'factor_lab_data' / 'factors.parquet'
    if not factor_store_path.exists():
        raise FileNotFoundError(f"因子数据文件不存在: {factor_store_path}")

    factor_names = ['mkt_ma5_followthrough_v3', 'market_60d_new_high_ratio']
    df = pd.read_parquet(factor_store_path, columns=['date', *factor_names])
    df['date'] = pd.to_datetime(df['date'])
    return df.groupby('date', as_index=False)[factor_names].first().sort_values('date').reset_index(drop=True)


def _build_select_scheme_choices(select_date: str, factor_data, selected_stocks: list[dict]) -> dict:
    import pandas as pd

    if factor_data is None or factor_data.empty or len(selected_stocks) < 2:
        return {}

    current_date = pd.to_datetime(select_date)
    current_row = factor_data.iloc[0]
    mkt_value = float(current_row.get('mkt_ma5_followthrough_v3')) if pd.notna(current_row.get('mkt_ma5_followthrough_v3')) else None
    nh_value = float(current_row.get('market_60d_new_high_ratio')) if pd.notna(current_row.get('market_60d_new_high_ratio')) else None

    if mkt_value is None:
        return {}

    hist = _load_daily_gate_factor_history()
    train_mask = (
        (hist['date'] >= pd.Timestamp(StrategyConfig.Q40_TRAIN_START))
        & (hist['date'] <= pd.Timestamp(StrategyConfig.Q40_TRAIN_END))
    )

    train_mkt = hist.loc[train_mask, 'mkt_ma5_followthrough_v3'].dropna()
    train_nh = hist.loc[train_mask, 'market_60d_new_high_ratio'].dropna()
    if train_mkt.empty:
        return {}

    mkt_q40 = float(train_mkt.quantile(0.4))
    mkt_q30 = float(train_mkt.quantile(0.3))
    nh_q70 = float(train_nh.quantile(0.7)) if not train_nh.empty else None

    hist_before_today = hist.loc[hist['date'] < current_date, 'mkt_ma5_followthrough_v3'].dropna()
    if len(hist_before_today) >= 20:
        mkt_exp_q30 = float(hist_before_today.quantile(0.3))
    else:
        mkt_exp_q30 = mkt_q30

    top1 = selected_stocks[0]
    top2 = selected_stocks[1]
    top1_code = top1.get('code')
    top2_code = top2.get('code')

    reg_choice = None
    reg_pct_threshold = 0.98
    reg_raw_threshold = 0.40
    if 'severe_abnormal_pressure_10d' in factor_data.columns and top1_code is not None:
        reg_panel = factor_data[['stock_code', 'severe_abnormal_pressure_10d']].copy()
        reg_panel['severe_abnormal_pressure_10d'] = pd.to_numeric(
            reg_panel['severe_abnormal_pressure_10d'], errors='coerce'
        )
        reg_panel = reg_panel.dropna(subset=['severe_abnormal_pressure_10d'])
        if not reg_panel.empty:
            reg_panel['reg_pct_rank'] = reg_panel['severe_abnormal_pressure_10d'].rank(
                pct=True, method='average'
            )
            reg_map = dict(zip(reg_panel['stock_code'], reg_panel['reg_pct_rank']))
            top1_reg_pct = reg_map.get(top1_code)
            top2_reg_pct = reg_map.get(top2_code)
            if top1_reg_pct is not None:
                reg_choice = {
                    'top1_reg_raw': float(reg_panel.loc[reg_panel['stock_code'] == top1_code, 'severe_abnormal_pressure_10d'].iloc[0]),
                    'top2_reg_raw': float(reg_panel.loc[reg_panel['stock_code'] == top2_code, 'severe_abnormal_pressure_10d'].iloc[0]) if top2_code in set(reg_panel['stock_code']) else None,
                    'top1_reg_pct': float(top1_reg_pct),
                    'top2_reg_pct': float(top2_reg_pct) if top2_reg_pct is not None else None,
                }

    top_factor_choice = None
    if {'cpv_factor', 'high_to_close_drawdown'}.issubset(set(factor_data.columns)):
        top_factor_panel = factor_data[['stock_code', 'cpv_factor', 'high_to_close_drawdown']].copy()
        top_factor_panel['cpv_factor'] = pd.to_numeric(top_factor_panel['cpv_factor'], errors='coerce')
        top_factor_panel['high_to_close_drawdown'] = pd.to_numeric(top_factor_panel['high_to_close_drawdown'], errors='coerce')
        top_factor_panel = top_factor_panel.dropna(subset=['cpv_factor', 'high_to_close_drawdown'])
        if not top_factor_panel.empty:
            factor_map = top_factor_panel.set_index('stock_code')[['cpv_factor', 'high_to_close_drawdown']].to_dict('index')
            top1_factor = factor_map.get(top1_code)
            top2_factor = factor_map.get(top2_code)
            if top1_factor is not None:
                top_factor_choice = {
                    'top1_cpv': float(top1_factor['cpv_factor']),
                    'top1_h2c': float(top1_factor['high_to_close_drawdown']),
                    'top2_cpv': float(top2_factor['cpv_factor']) if top2_factor is not None else None,
                    'top2_h2c': float(top2_factor['high_to_close_drawdown']) if top2_factor is not None else None,
                }

    def _make_choice(
        name: str,
        choose_top1: bool | None,
        threshold: float | None,
        signal_value: float | None,
        extra: dict | None = None,
    ):
        if choose_top1 is None:
            chosen = None
        else:
            chosen = top1 if choose_top1 else top2
        payload = {
            'name': name,
            'chosen_rank': int(chosen.get('rank', 1 if choose_top1 else 2)) if chosen else 0,
            'selected_code': chosen.get('code') if chosen else None,
            'threshold': threshold,
            'signal_value': signal_value,
            'chosen_by_top1': bool(choose_top1) if choose_top1 is not None else False,
            'action': 'buy' if chosen is not None else 'cash',
        }
        if extra:
            payload.update(extra)
        return payload

    choices = {
        'mkt_q40': _make_choice(
            name='mkt_ma5_followthrough_v3_q40',
            choose_top1=mkt_value >= mkt_q40,
            threshold=mkt_q40,
            signal_value=mkt_value,
        ),
        'mkt_exp_q30_shift1': _make_choice(
            name='mkt_ma5_followthrough_v3_expanding_q30_shift1',
            choose_top1=mkt_value >= mkt_exp_q30,
            threshold=mkt_exp_q30,
            signal_value=mkt_value,
        ),
    }

    if reg_choice is not None:
        reg_raw_choose_top1 = (mkt_value >= mkt_exp_q30) and (reg_choice['top1_reg_raw'] < reg_raw_threshold)
        choices['mkt_exp_q30_reg10_raw_0p40_to_top2'] = _make_choice(
            name='mkt_exp_q30_reg10_raw_0.40_to_top2',
            choose_top1=reg_raw_choose_top1,
            threshold=mkt_exp_q30,
            signal_value=mkt_value,
            extra={
                'second_factor_name': 'top1_severe_abnormal_pressure_10d',
                'second_threshold': reg_raw_threshold,
                'second_signal_value': reg_choice['top1_reg_raw'],
                'top2_second_signal_value': reg_choice.get('top2_reg_raw'),
            },
        )

        reg_choose_top1 = (mkt_value >= mkt_exp_q30) and (reg_choice['top1_reg_pct'] < reg_pct_threshold)
        if mkt_value < mkt_exp_q30:
            reg_choose_top1 = False
        choices['mkt_exp_q30_top1_hi98_to_top2'] = _make_choice(
            name='mkt_exp_q30_top1_hi98_to_top2',
            choose_top1=reg_choose_top1,
            threshold=mkt_exp_q30,
            signal_value=mkt_value,
            extra={
                'second_factor_name': 'top1_severe_abnormal_pressure_10d_pct_rank',
                'second_threshold': reg_pct_threshold,
                'second_signal_value': reg_choice['top1_reg_pct'],
                'top2_second_signal_value': reg_choice.get('top2_reg_pct'),
            },
        )

        reg_pct_state_threshold = 0.95 if mkt_value >= mkt_exp_q30 else 0.85
        pct_state_choose_top1 = reg_choice['top1_reg_pct'] < reg_pct_state_threshold
        choices['switch_reg_pct_mktstate_hi95_lo85'] = _make_choice(
            name='switch_reg_pct_mktstate_hi95_lo85',
            choose_top1=pct_state_choose_top1,
            threshold=reg_pct_state_threshold,
            signal_value=reg_choice['top1_reg_pct'],
            extra={
                'state_factor_name': 'mkt_ma5_followthrough_v3_expanding_q30_shift1',
                'state_threshold': mkt_exp_q30,
                'state_signal_value': mkt_value,
                'second_factor_name': 'selected_severe_abnormal_pressure_10d_pct_rank',
                'second_threshold': reg_pct_state_threshold,
                'second_signal_value': reg_choice['top1_reg_pct'] if pct_state_choose_top1 else reg_choice.get('top2_reg_pct'),
                'top1_second_signal_value': reg_choice['top1_reg_pct'],
                'top2_second_signal_value': reg_choice.get('top2_reg_pct'),
            },
        )

        if top_factor_choice is not None:
            selected_cpv = top_factor_choice['top1_cpv'] if pct_state_choose_top1 else top_factor_choice.get('top2_cpv')
            selected_h2c = top_factor_choice['top1_h2c'] if pct_state_choose_top1 else top_factor_choice.get('top2_h2c')
            veto_triggered = (
                selected_cpv is not None and selected_h2c is not None
                and selected_cpv >= 1.5
                and selected_h2c <= -0.015
            )
            choices['switch_reg_pct_mktstate_hi95_lo85_cpv_h2c_veto'] = _make_choice(
                name='switch_reg_pct_mktstate_hi95_lo85_plus_cpv1.5_h2c-0.015_veto',
                choose_top1=(None if veto_triggered else pct_state_choose_top1),
                threshold=reg_pct_state_threshold,
                signal_value=reg_choice['top1_reg_pct'],
                extra={
                    'state_factor_name': 'mkt_ma5_followthrough_v3_expanding_q30_shift1',
                    'state_threshold': mkt_exp_q30,
                    'state_signal_value': mkt_value,
                    'second_factor_name': 'selected_severe_abnormal_pressure_10d_pct_rank',
                    'second_threshold': reg_pct_state_threshold,
                    'second_signal_value': reg_choice['top1_reg_pct'] if pct_state_choose_top1 else reg_choice.get('top2_reg_pct'),
                    'top1_second_signal_value': reg_choice['top1_reg_pct'],
                    'top2_second_signal_value': reg_choice.get('top2_reg_pct'),
                    'veto_factor_name': 'selected_cpv_factor_and_high_to_close_drawdown',
                    'veto_cpv_threshold': 1.5,
                    'veto_h2c_threshold': -0.015,
                    'veto_cpv_value': selected_cpv,
                    'veto_h2c_value': selected_h2c,
                    'veto_triggered': veto_triggered,
                    'action': 'cash' if veto_triggered else 'buy',
                },
            )

    if nh_value is not None and nh_q70 is not None:
        dual_choose_top1 = (mkt_value >= mkt_q40) and (nh_value >= nh_q70)
        choices['dual_gate'] = _make_choice(
            name='mkt_q40_plus_new_high_ratio_q70',
            choose_top1=dual_choose_top1,
            threshold=mkt_q40,
            signal_value=mkt_value,
            extra={
                'second_factor_name': 'market_60d_new_high_ratio',
                'second_threshold': nh_q70,
                'second_signal_value': nh_value,
            },
        )

    return choices


def _persist_select_scheme_choices(select_date: str, result: dict, scheme_choices: dict) -> None:
    if not result or not scheme_choices:
        return

    preferred_key = (
        'switch_reg_pct_mktstate_hi95_lo85_cpv_h2c_veto'
        if scheme_choices.get('switch_reg_pct_mktstate_hi95_lo85_cpv_h2c_veto')
        else (
            'switch_reg_pct_mktstate_hi95_lo85'
            if scheme_choices.get('switch_reg_pct_mktstate_hi95_lo85')
            else (
                'mkt_exp_q30_reg10_raw_0p40_to_top2'
                if scheme_choices.get('mkt_exp_q30_reg10_raw_0p40_to_top2')
                else (
                    'mkt_exp_q30_top1_hi98_to_top2'
                    if scheme_choices.get('mkt_exp_q30_top1_hi98_to_top2')
                    else ('mkt_exp_q30_shift1' if scheme_choices.get('mkt_exp_q30_shift1') else 'mkt_q40')
                )
            )
        )
    )
    preferred_info = scheme_choices.get(preferred_key)
    if not preferred_info:
        return

    for stock in result.get('stocks', []):
        stock['q40_selected'] = stock.get('code') == preferred_info.get('selected_code')

    result['gates'] = scheme_choices
    result['preferred_gate'] = preferred_key
    # 兼容现有 trade 模式：继续复用 q40 字段，但实际写入当前首选方案
    result['q40'] = {
        'enabled': True,
        'chosen_rank': preferred_info.get('chosen_rank', 1),
        'threshold': preferred_info.get('threshold'),
        'signal_value': preferred_info.get('signal_value'),
        'selected_code': preferred_info.get('selected_code'),
        'scheme_name': preferred_info.get('name'),
        'action': preferred_info.get('action', 'buy'),
        'second_factor_name': preferred_info.get('second_factor_name'),
        'second_threshold': preferred_info.get('second_threshold'),
        'second_signal_value': preferred_info.get('second_signal_value'),
        'veto_factor_name': preferred_info.get('veto_factor_name'),
        'veto_cpv_threshold': preferred_info.get('veto_cpv_threshold'),
        'veto_h2c_threshold': preferred_info.get('veto_h2c_threshold'),
        'veto_cpv_value': preferred_info.get('veto_cpv_value'),
        'veto_h2c_value': preferred_info.get('veto_h2c_value'),
        'veto_triggered': preferred_info.get('veto_triggered', False),
        'state_threshold': preferred_info.get('state_threshold'),
        'state_signal_value': preferred_info.get('state_signal_value'),
    }

    selections = _load_selections()
    selections[select_date] = result
    with open(PathConfig.STOCK_SELECTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(selections, f, ensure_ascii=False, indent=2)


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

                logger.info(f"{'排名':<4} {'股票代码':<12} {'预测值(ATR)':<15} {'SW行业':<20} {'可交易':<8}")
                logger.info("-" * 60)
                for stock in result['stocks']:
                    industry = stock_to_industry.get(stock['code'], '未知')
                    tradeable_tag = 'Y' if stock.get('tradeable', True) else 'N'
                    logger.info(f"{stock['rank']:<4} {stock['code']:<12} {stock['score']:<15.3f} {industry:<20} {tradeable_tag:<8}")
                chosen_stock = result['stocks'][0]
                logger.info(
                    f"Top1交易判断: {chosen_stock['code']} score={chosen_stock['score']:.3f} "
                    f"tradeable={chosen_stock.get('tradeable', True)} "
                    f"strategy={StrategyConfig.EXHAUSTION_FILTER_STRATEGY} "
                    f"result={chosen_stock.get('exhaustion_filter_reason', 'PASS') if chosen_stock.get('exhaustion_filtered') else 'PASS'} "
                    f"detail={chosen_stock.get('exhaustion_filter_detail', 'NA')}"
                )
            else:
                logger.warning(f"SW行业映射文件不存在: {sw_sector_file}")
                for stock in result['stocks']:
                    markers = []
                    if not stock.get('tradeable', True):
                        markers.append('BLOCK')
                    marker_text = f" <<{','.join(markers)}" if markers else ''
                    logger.info(f"  {stock['rank']}. {stock['code']} - 分数: {stock['score']:.3f}{marker_text}")
                chosen_stock = result['stocks'][0]
                logger.info(
                    f"Top1交易判断: {chosen_stock['code']} score={chosen_stock['score']:.3f} "
                    f"tradeable={chosen_stock.get('tradeable', True)} "
                    f"strategy={StrategyConfig.EXHAUSTION_FILTER_STRATEGY} "
                    f"result={chosen_stock.get('exhaustion_filter_reason', 'PASS') if chosen_stock.get('exhaustion_filtered') else 'PASS'} "
                    f"detail={chosen_stock.get('exhaustion_filter_detail', 'NA')}"
                )
        except Exception as e:
            logger.warning(f"无法加载SW行业映射: {e}")
            for stock in result['stocks']:
                markers = []
                if not stock.get('tradeable', True):
                    markers.append('BLOCK')
                marker_text = f" <<{','.join(markers)}" if markers else ''
                logger.info(f"  {stock['rank']}. {stock['code']} - 分数: {stock['score']:.3f}{marker_text}")
            chosen_stock = result['stocks'][0]
            logger.info(
                f"Top1交易判断: {chosen_stock['code']} score={chosen_stock['score']:.3f} "
                f"tradeable={chosen_stock.get('tradeable', True)} "
                f"strategy={StrategyConfig.EXHAUSTION_FILTER_STRATEGY} "
                f"result={chosen_stock.get('exhaustion_filter_reason', 'PASS') if chosen_stock.get('exhaustion_filtered') else 'PASS'} "
                f"detail={chosen_stock.get('exhaustion_filter_detail', 'NA')}"
            )
    else:
        logger.warning("  无符合条件的股票")

    logger.info(f"\n结果已保存到: {PathConfig.STOCK_SELECTIONS_FILE}")
    logger.info("=" * 60)

    return True


def _load_selections() -> Dict:
    if not PathConfig.STOCK_SELECTIONS_FILE.exists():
        return {}
    with open(PathConfig.STOCK_SELECTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def _get_trade_candidate(selection_entry: Dict) -> Optional[Dict]:
    stocks = selection_entry.get('stocks', [])
    if not stocks:
        return None

    chosen = dict(stocks[0])
    if chosen.get('exhaustion_filtered'):
        chosen['blocked_by_default_filter'] = True
        return chosen
    chosen['target_rank'] = 1
    chosen['is_weak_market_env'] = False
    return chosen


def _get_strategy_position_snapshot(valid_positions, scheduler) -> tuple[Dict[int, Dict], Dict[str, Dict]]:
    occupied_slots: Dict[int, Dict] = {}
    position_records: Dict[str, Dict] = {}
    fallback_slot = 0

    for pos in valid_positions:
        record = scheduler.load_latest_buy_record(pos.stock_code)
        if not record:
            continue
        slot_index = record.get('slot_index')
        if slot_index is None or int(slot_index) in occupied_slots:
            while fallback_slot in occupied_slots and fallback_slot < StrategyConfig.MAX_CONCURRENT_POSITIONS:
                fallback_slot += 1
            slot_index = fallback_slot
        slot_index = int(slot_index)
        if slot_index >= StrategyConfig.MAX_CONCURRENT_POSITIONS:
            continue
        occupied_slots[slot_index] = {
            'stock_code': pos.stock_code,
            'position': pos,
            'record': record,
        }
        position_records[pos.stock_code] = record
    return occupied_slots, position_records


def _calculate_target_volume(trader, stock_code: str, target_amount: float) -> tuple[int, Optional[float]]:
    price = trader.get_current_price(stock_code)
    if price is None or price <= 0:
        return 0, None
    volume = int(target_amount / price / 100) * 100
    return volume, price


def run_trade_mode(args):
    """运行实盘交易模式：只负责T+1开仓，监控由monitor模式独立承担。"""
    from live_trading.trader import XtQuantTrader, TradingScheduler

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("交易模式（仅T+1买入）")
    logger.info("=" * 60)
    logger.info(f"选股日期(T日): {args.date}")

    trader = XtQuantTrader()
    if not trader.connect():
        logger.error("连接失败")
        return False

    scheduler = TradingScheduler(trader)
    today_str = datetime.now().strftime('%Y-%m-%d')
    today_compact = datetime.now().strftime('%Y%m%d')

    selections = _load_selections()
    selection_entry = selections.get(args.date)
    if not selection_entry:
        logger.error(f"未找到 {args.date} 的选股结果: {PathConfig.STOCK_SELECTIONS_FILE}")
        trader.disconnect()
        return False

    positions = trader.query_stock_positions(trader.account) or []
    valid_positions = [pos for pos in positions if pos.volume >= 100]
    occupied_slots, position_records = _get_strategy_position_snapshot(valid_positions, scheduler)

    logger.info(f"策略总资金上限: {StrategyConfig.STRATEGY_CAPITAL:.2f}")
    logger.info(f"当前槽位配置: {StrategyConfig.MAX_CONCURRENT_POSITIONS}槽位, {StrategyConfig.POSITION_SCHEDULE}")
    logger.info(f"当前有效持仓: {len(valid_positions)}只, 已占槽位: {sorted(occupied_slots.keys())}")

    candidate = _get_trade_candidate(selection_entry)
    if candidate:
        logger.info(
            f"今日候选: {candidate.get('code')} rank={candidate.get('target_rank', 1)} score={candidate.get('score', 0):.3f}"
        )
        if candidate.get('blocked_by_default_filter'):
            logger.warning(
                f"默认过滤规则阻止交易: {candidate.get('code')} "
                f"strategy={StrategyConfig.EXHAUSTION_FILTER_STRATEGY} "
                f"reason={candidate.get('exhaustion_filter_reason', 'NA')}"
            )
            candidate = None
    else:
        logger.warning("今日无可交易候选")

    free_slots = [idx for idx in range(StrategyConfig.MAX_CONCURRENT_POSITIONS) if idx not in occupied_slots]
    if candidate and free_slots:
        candidate_code = candidate['code']
        if candidate_code in position_records:
            logger.info(f"{candidate_code} 已在持仓中，跳过重复开仓")
        else:
            trader.ensure_market_quote_subscription([candidate_code])
            slot_index = free_slots[0]
            target_pct = float(StrategyConfig.POSITION_SCHEDULE[slot_index])
            planned_amount = float(StrategyConfig.STRATEGY_CAPITAL * target_pct)
            target_rank = int(candidate.get('target_rank', 1))
            is_weak_market_env = bool(candidate.get('is_weak_market_env', False))
            target_volume, ref_price = _calculate_target_volume(trader, candidate_code, planned_amount)

            logger.info(
                f"尝试开仓: {candidate_code}, 槽位={slot_index + 1}/{StrategyConfig.MAX_CONCURRENT_POSITIONS}, "
                f"目标仓位={target_pct:.0%}, 计划金额={planned_amount:.2f}, 参考价={ref_price if ref_price else 0:.2f}, "
                f"目标股数={target_volume}"
            )

            if target_volume >= 100:
                execution_result = trader.order_manager.smart_buy_full_position(
                    candidate_code,
                    today_str,
                    target_volume=target_volume,
                    wait_for_open=True,
                    apply_opening_risk=True,
                    target_rank=target_rank,
                    is_weak_market_env=is_weak_market_env,
                )
                if execution_result:
                    filled_volume = int(execution_result.get('filled_volume', 0) or 0)
                    entry_price = float(execution_result.get('avg_price', 0.0) or 0.0)
                    open_price = float(execution_result.get('open_price', 0.0) or 0.0)
                    entry_date = execution_result.get('first_fill_date') or today_str
                    last_order_id = execution_result.get('last_order_id')
                    order_ids = {candidate_code: last_order_id} if last_order_id else {}
                    scheduler.save_trade_record(
                        today_compact,
                        'buy',
                        [candidate_code],
                        order_ids,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        volume=filled_volume,
                        slot_index=slot_index,
                        target_pct=target_pct,
                        planned_amount=planned_amount,
                        strategy_capital=float(StrategyConfig.STRATEGY_CAPITAL),
                        target_rank=target_rank,
                        open_price=open_price,
                    )
                    logger.info(
                        f"✅ 买入成功: {candidate_code}, filled={filled_volume}/{target_volume}, "
                        f"avg_price={entry_price:.4f}, first_fill_date={entry_date}, order_id={last_order_id}"
                    )
                else:
                    logger.warning(f"买入未成交或被风控跳过: {candidate_code}")
            else:
                logger.warning(f"{candidate_code} 目标股数不足1手，跳过开仓")
    elif not free_slots:
        logger.info("当前无空闲槽位，跳过新开仓")

    logger.info("交易模式只负责本次买入流程，不再启动持仓监控；请独立运行 --mode monitor 负责风控")
    trader.disconnect()
    return True


def _run_monitoring_session(trader, scheduler, logger, session_name: str = "监控模式") -> bool:
    """仅恢复当前持仓的5分钟监控，不执行任何开仓逻辑。"""
    from live_trading.monitor import PositionMonitor5Min

    positions = trader.query_stock_positions(trader.account) or []
    valid_positions = [pos for pos in positions if pos.volume >= 100]
    if not valid_positions:
        logger.info(f"{session_name}: 当前无有效持仓，结束")
        trader.disconnect()
        return True

    if hasattr(trader, "ensure_market_quote_subscription"):
        trader.ensure_market_quote_subscription([pos.stock_code for pos in valid_positions])

    monitor_5m = PositionMonitor5Min(trader, scheduler)
    monitored_count = 0

    for pos in valid_positions:
        buy_record = scheduler.load_latest_buy_record(pos.stock_code)
        if not buy_record:
            logger.warning(f"{pos.stock_code} 未找到买入记录，跳过监控")
            continue

        entry_date = buy_record['entry_date']
        entry_price = buy_record.get('entry_price', 0)
        open_price = float(buy_record.get('open_price', 0) or 0.0)
        slot_index = buy_record.get('slot_index')
        target_pct = buy_record.get('target_pct')
        holding_days = scheduler.calculate_holding_days(entry_date)

        logger.info(
            f"持仓监控准备: {pos.stock_code}, entry_date={entry_date}, entry_price={entry_price}, "
            f"open_price={open_price if open_price > 0 else 'N/A'}, holding_days={holding_days}, slot={slot_index}"
        )

        if holding_days == 0:
            logger.info(f"{pos.stock_code}: 当天买入（T+0），不启动监控")
            continue

        if holding_days >= 3:
            logger.warning(f"{pos.stock_code}: 持仓异常（{holding_days}天），交由监控器立即处理")

        monitor_5m.add_position_5m(
            pos.stock_code,
            entry_price,
            entry_date,
            slot_index=slot_index,
            target_pct=target_pct,
            enable_closing_sell=(holding_days >= 2),
            open_price=open_price if open_price > 0 else None,
        )
        monitored_count += 1

    if monitored_count == 0:
        logger.info(f"{session_name}: 当前无需要监控的持仓，结束")
        trader.disconnect()
        return True

    monitor_5m.start_5m_monitoring()
    logger.info(f"✅ 5分钟监控已启动，监控持仓数: {monitored_count}")

    try:
        while monitor_5m.is_monitoring and not _SHUTDOWN_EVENT.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("收到中断信号，停止监控")
    finally:
        monitor_5m.stop_5m_monitoring()
        trader.disconnect()

    return True


def run_monitor_mode(args):
    """仅启动当前持仓的5分钟监控，不执行任何开仓逻辑。"""
    from live_trading.trader import XtQuantTrader, TradingScheduler

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("监控模式")
    logger.info("=" * 60)
    logger.info(f"监控参考日期: {args.date}")

    trader = XtQuantTrader()
    if not trader.connect():
        logger.error("连接失败")
        return False

    scheduler = TradingScheduler(trader)
    return _run_monitoring_session(trader, scheduler, logger, session_name="监控模式")


def run_sync_mode(args):
    """同步当前持仓到本地买入记录。

    适用场景：
    - 在券商客户端手动买入后，希望让程序后续能正确识别槽位和监控起点。
    - 只补“当前有持仓但本地没有买入记录”的股票，不覆盖已有记录。
    """
    from live_trading.trader import XtQuantTrader, TradingScheduler

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("持仓同步模式")
    logger.info("=" * 60)
    logger.info(f"同步买入日期: {args.date}")

    sync_date = datetime.strptime(args.date, '%Y%m%d').strftime('%Y-%m-%d')

    trader = XtQuantTrader()
    if not trader.connect():
        logger.error("连接失败")
        return False

    scheduler = TradingScheduler(trader)
    today_compact = datetime.now().strftime('%Y%m%d')

    try:
        positions = trader.query_stock_positions(trader.account) or []
        valid_positions = [pos for pos in positions if getattr(pos, 'volume', 0) >= 100]

        if not valid_positions:
            logger.info("当前无有效持仓，无需同步")
            return True

        occupied_slots, _ = _get_strategy_position_snapshot(valid_positions, scheduler)
        free_slots = [idx for idx in range(StrategyConfig.MAX_CONCURRENT_POSITIONS) if idx not in occupied_slots]

        synced = []
        skipped_existing = []
        skipped_invalid = []

        for pos in valid_positions:
            stock_code = pos.stock_code
            existing = scheduler.load_latest_buy_record(stock_code)
            if existing:
                skipped_existing.append(stock_code)
                logger.info(
                    f"{stock_code} 已存在买入记录，跳过同步: "
                    f"entry_date={existing.get('entry_date')} entry_price={existing.get('entry_price')} volume={existing.get('volume')}"
                )
                continue

            volume = int(getattr(pos, 'can_use_volume', 0) or 0)
            if volume < 100:
                volume = int(getattr(pos, 'volume', 0) or 0)
            volume = (volume // 100) * 100
            if volume < 100:
                skipped_invalid.append(stock_code)
                logger.warning(f"{stock_code} 当前可同步数量不足1手，跳过")
                continue

            entry_price = float(getattr(pos, 'open_price', 0) or 0.0)
            if entry_price <= 0:
                entry_price = float(getattr(pos, 'avg_price', 0) or 0.0)
            if entry_price <= 0:
                current_price = trader.get_current_price(stock_code)
                entry_price = float(current_price or 0.0)
            if entry_price <= 0:
                skipped_invalid.append(stock_code)
                logger.warning(f"{stock_code} 无法确定买入成本价，跳过")
                continue

            slot_index = None
            target_pct = None
            planned_amount = None
            strategy_capital = float(StrategyConfig.STRATEGY_CAPITAL)

            if free_slots:
                slot_index = free_slots.pop(0)
                if slot_index < len(StrategyConfig.POSITION_SCHEDULE):
                    target_pct = float(StrategyConfig.POSITION_SCHEDULE[slot_index])
                    planned_amount = strategy_capital * target_pct

            synthetic_order_id = f"SYNC_{stock_code}_{today_compact}"
            scheduler.save_trade_record(
                today_compact,
                'manual_buy',
                [stock_code],
                {stock_code: synthetic_order_id},
                entry_date=sync_date,
                entry_price=entry_price,
                volume=volume,
                slot_index=slot_index,
                target_pct=target_pct,
                planned_amount=planned_amount,
                strategy_capital=strategy_capital,
            )
            synced.append({
                'stock_code': stock_code,
                'entry_date': sync_date,
                'entry_price': entry_price,
                'volume': volume,
                'slot_index': slot_index,
                'target_pct': target_pct,
            })
            logger.info(
                f"✅ 已同步持仓: {stock_code}, entry_date={sync_date}, entry_price={entry_price:.4f}, "
                f"volume={volume}, slot={slot_index}, target_pct={target_pct}"
            )

        logger.info("=" * 60)
        logger.info(
            f"同步完成: synced={len(synced)} skipped_existing={len(skipped_existing)} skipped_invalid={len(skipped_invalid)}"
        )
        if synced:
            for item in synced:
                logger.info(
                    f"  {item['stock_code']} -> date={item['entry_date']} price={item['entry_price']:.4f} "
                    f"volume={item['volume']} slot={item['slot_index']}"
                )
        return True
    finally:
        trader.disconnect()


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

    execution_result = trader.order_manager.smart_buy_full_position(
        stock_code,
        datetime.now().strftime('%Y-%m-%d'),
        target_volume=buy_volume  # 指定买入数量
    )

    if execution_result:
        filled_volume = int(execution_result.get('filled_volume', 0) or 0)
        entry_price = float(execution_result.get('avg_price', 0.0) or 0.0)
        open_price = float(execution_result.get('open_price', 0.0) or 0.0)
        entry_date = execution_result.get('first_fill_date') or datetime.now().strftime('%Y-%m-%d')
        last_order_id = execution_result.get('last_order_id')
        logger.info(f"✅ 买入成功: {stock_code}")
        logger.info(f"  订单号: {last_order_id}")
        logger.info(f"  买入价格: {entry_price:.2f}元")
        logger.info(f"  买入数量: {filled_volume}股")
        logger.info(f"  实际金额: {entry_price * filled_volume:.2f}元")

        # 6. 保存交易记录
        scheduler.save_trade_record(
            datetime.now().strftime('%Y%m%d'),
            'manual_buy',
            [stock_code],
            ({stock_code: last_order_id} if last_order_id else {}),
            entry_date=entry_date,
            entry_price=entry_price,
            volume=filled_volume,
            open_price=open_price,
        )

        logger.info(f"\n买入记录已保存:")
        logger.info(f"  股票代码: {stock_code}")
        logger.info(f"  买入日期: {entry_date}")
        logger.info(f"  买入价格: {entry_price:.2f}元")
        logger.info(f"  买入数量: {filled_volume}股")
        logger.info(f"  买入金额: {entry_price * filled_volume:.2f}元")
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

    parser.add_argument('--mode', type=str, required=True, choices=['select', 'trade', 'monitor', 'manual', 'sync'],
                       help='运行模式: select-选股模式, trade-交易模式, monitor-仅监控模式, manual-手动交易模式, sync-持仓同步模式')

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
    if args.mode in ['select', 'trade', 'monitor', 'sync'] and not args.date:
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
    _install_interrupt_handlers(logger)
    logger.info(f"实盘交易系统启动 - 模式: {args.mode}")

    try:
        if args.mode == 'select':
            success = run_select_mode(args)
        elif args.mode == 'trade':
            success = run_trade_mode(args)
        elif args.mode == 'monitor':
            success = run_monitor_mode(args)
        elif args.mode == 'sync':
            success = run_sync_mode(args)
        elif args.mode == 'manual':
            success = run_manual_mode(args)
        else:
            logger.error(f"未知模式: {args.mode}")
            return

        if success:
            logger.info("✅ 执行成功")
        else:
            logger.error("❌ 执行失败")

    except KeyboardInterrupt:
        logger.info("收到中断信号，程序退出")
        success = False
    except Exception as e:
        logger.error(f"执行异常: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
    finally:
        if _SHUTDOWN_EVENT.is_set():
            try:
                time.sleep(0.2)
            finally:
                os._exit(130)


if __name__ == '__main__':
    main()
