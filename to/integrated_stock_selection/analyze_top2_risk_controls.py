import argparse
import contextlib
import io
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from rqalpha_minute_datasource.strategy_backtest_lead_fusion_rank_binary_v2 import (
    BacktestStrategy,
    EntryRiskControlStrategy,
    RiskControlStrategy,
    StrategyConfig,
    StrategyType,
)

WINDOW3 = {
    "name": "Window 3",
    "train_start": "2023-06-01",
    "train_end": "2025-03-01",
    "val_start": "2025-03-07",
    "val_end": "2025-07-25",
    "test_start": "2025-08-05",
    "test_end": "2026-04-22",
}


def _load_main_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("cls_main_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def _build_backtest_data(main_script: Path, model_path: Path) -> pd.DataFrame:
    cache_path = Path("results") / f"top2_risk_scan_input_{model_path.stem}.pkl"
    if cache_path.exists():
        return pd.read_pickle(cache_path)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mod = _load_main_module(main_script)
        factor_data = mod._load_factor_cache_data(include_label_generation_features=True)
        test_data = factor_data[
            (factor_data["date"] >= pd.to_datetime(WINDOW3["test_start"])) &
            (factor_data["date"] <= pd.to_datetime(WINDOW3["test_end"]))
        ].copy()
        if test_data.empty:
            raise ValueError("Window 3 测试集为空")

        import pickle

        with open(model_path, "rb") as f:
            lgb_data = pickle.load(f)

        model_result = {
            "models": [lgb_data["model"]],
            "scalers": [lgb_data["scaler"]],
            "feature_names": lgb_data["feature_names"],
            "winsorization_thresholds": {},
            "model_files": [model_path],
            "model_performances": [1.0],
            "model_types": ["lightgbm"],
        }

        feature_columns = model_result["feature_names"]
        X_test, _ = mod._prepare_feature_data(test_data, feature_columns, data_type="Top2风控扫描测试集")
        if X_test is None:
            raise ValueError("测试集特征准备失败")

        model = model_result["models"][0]
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        test_data_with_pred = test_data.copy()
        test_data_with_pred["predicted_return"] = y_pred
        test_data_with_pred["predicted_return_atr_raw"] = y_pred
        test_data_with_pred["std_pred"] = np.zeros_like(y_pred)
    required_defaults = {
        "future_return_2d": 0.0,
        "is_can_buy": 1,
        "pct_chg": 0.0,
        "volume": 0.0,
        "amount": 0.0,
        "atr14": 0.0,
        "open": 0.0,
        "high": 0.0,
        "low": 0.0,
        "close": 0.0,
        "price_ema20_ratio": 1.0,
        "price_ema50_ratio": 1.0,
        "ema20_ema50_ratio": 1.0,
        "ema50_slope": 0.0,
        "log_price_regression_slope": 0.0,
        "log_price_regression_r2": 0.0,
        "adx_14": 10.0,
        "drawdown_from_20d_high_pct": 0.0,
        "drawdown_atr_ratio": 0.0,
        "atr20": 0.5,
        "negative_days_5d": 1,
        "large_red_candle_5d": 1,
        "distance_to_support_atr": 0.5,
        "bollinger_percent_b": 0.5,
        "rsi_14": 50.0,
        "is_limit_up": 0,
        "is_limit_down": 0,
        "is_zhaban": 0,
        "limit_strength": 0.0,
        "limit_times": 0,
        "consecutive_limit_up": 0,
        "consecutive_limit_down": 0,
        "near_support": 0,
        "near_resistance": 0,
        "short_reversal": 0,
    }
    for col, default_value in required_defaults.items():
        if col not in test_data_with_pred.columns:
            test_data_with_pred[col] = default_value

    test_data_with_pred = test_data_with_pred.sort_values("date").copy()
    if "stock_code" in test_data_with_pred.columns:
        test_data_with_pred = test_data_with_pred[
            ~test_data_with_pred["stock_code"].astype(str).str.endswith((".BJ", ".bj"), na=False)
        ].copy()
    if hasattr(mod, "_filter_new_stocks_batch"):
        test_data_with_pred = mod._filter_new_stocks_batch(test_data_with_pred, logging.getLogger(__name__))
    if hasattr(mod, "_filter_st_stocks"):
        st_mask = mod._filter_st_stocks(test_data_with_pred, logging.getLogger(__name__))
        test_data_with_pred = test_data_with_pred[st_mask].copy()
    test_data_with_pred = test_data_with_pred[~test_data_with_pred["predicted_return"].isna()].copy()
    test_data_with_pred["date"] = pd.to_datetime(test_data_with_pred["date"]).dt.date
    test_data_with_pred.to_pickle(cache_path)
    return test_data_with_pred


def _score_result(total_return: float, max_drawdown: float, sharpe: float) -> float:
    dd = abs(float(max_drawdown))
    calmar_like = float(total_return) / max(dd, 1e-6)
    return (
        0.90 * float(total_return)
        - 0.60 * dd
        + 18.0 * float(sharpe)
        + 12.0 * calmar_like
    )


def _build_entry_schemes() -> List[Dict]:
    return [
        {
            "entry_name": "gap_dn5_up5",
            "entry_strategy": EntryRiskControlStrategy.GAP_BASED,
            "entry_risk_up_threshold": 0.05,
            "entry_risk_down_threshold": -0.06,
            "entry_risk_down_threshold_non_top1": -0.05,
            "entry_risk_ma5_up_threshold": 0.05,
            "entry_risk_ma5_down_threshold": -0.03,
        },
        {
            "entry_name": "gap_dn4_up5",
            "entry_strategy": EntryRiskControlStrategy.GAP_BASED,
            "entry_risk_up_threshold": 0.05,
            "entry_risk_down_threshold": -0.06,
            "entry_risk_down_threshold_non_top1": -0.04,
            "entry_risk_ma5_up_threshold": 0.05,
            "entry_risk_ma5_down_threshold": -0.03,
        },
        {
            "entry_name": "gap_dn3_up5",
            "entry_strategy": EntryRiskControlStrategy.GAP_BASED,
            "entry_risk_up_threshold": 0.05,
            "entry_risk_down_threshold": -0.06,
            "entry_risk_down_threshold_non_top1": -0.03,
            "entry_risk_ma5_up_threshold": 0.05,
            "entry_risk_ma5_down_threshold": -0.03,
        },
        {
            "entry_name": "ma5_open_ge_ma5_gapup5",
            "entry_strategy": EntryRiskControlStrategy.MA5_BASED,
            "entry_risk_up_threshold": 0.05,
            "entry_risk_down_threshold": -0.06,
            "entry_risk_down_threshold_non_top1": -0.05,
            "entry_risk_ma5_up_threshold": 0.05,
            "entry_risk_ma5_down_threshold": 0.0,
        },
    ]


def _build_exit_schemes() -> List[Dict]:
    return [
        {
            "exit_name": "sl5_pb3",
            "stop_loss_threshold": -0.05,
            "profit_bearish_required_count": 3,
        },
        {
            "exit_name": "sl4_pb3",
            "stop_loss_threshold": -0.04,
            "profit_bearish_required_count": 3,
        },
        {
            "exit_name": "sl4_pb4",
            "stop_loss_threshold": -0.04,
            "profit_bearish_required_count": 4,
        },
        {
            "exit_name": "sl5_pb4",
            "stop_loss_threshold": -0.05,
            "profit_bearish_required_count": 4,
        },
    ]


def _run_single_scheme(
    df: pd.DataFrame,
    minute_data_path: str,
    holding_days: int,
    entry_scheme: Dict,
    exit_scheme: Dict,
) -> Dict:
    cfg = StrategyConfig(
        strategy_type=StrategyType.TOP_2,
        stock_count=2,
        enable_dynamic_rank_switch=False,
        holding_days=holding_days,
        max_concurrent_positions=3,
        fixed_position_pct=1.0,
        position_schedule=[1.0],
        risk_control_strategy=RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN,
        stop_loss_threshold=exit_scheme["stop_loss_threshold"],
        profit_bearish_required_count=exit_scheme["profit_bearish_required_count"],
        entry_risk_control_strategy=entry_scheme["entry_strategy"],
        entry_risk_ma5_up_threshold=entry_scheme["entry_risk_ma5_up_threshold"],
        entry_risk_ma5_down_threshold=entry_scheme["entry_risk_ma5_down_threshold"],
        entry_risk_up_threshold=entry_scheme["entry_risk_up_threshold"],
        entry_risk_down_threshold=entry_scheme["entry_risk_down_threshold"],
        entry_risk_down_threshold_non_top1=entry_scheme["entry_risk_down_threshold_non_top1"],
    )

    trading_dates = sorted(df["date"].unique().tolist())
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        previous_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            strategy = BacktestStrategy(cfg, df, minute_data_path)
            result = strategy.run_backtest(trading_dates)
        finally:
            logging.disable(previous_disable)

    summary = result.get("summary_stats", {}) or {}
    risk_details = summary.get("risk_control_details", {}) or {}

    total_return = float(result.get("total_return", 0.0))
    max_drawdown = float(result.get("max_drawdown", 0.0))
    sharpe = float(result.get("sharpe_ratio", 0.0))
    score = _score_result(total_return, max_drawdown, sharpe)

    return {
        "entry_name": entry_scheme["entry_name"],
        "exit_name": exit_scheme["exit_name"],
        "scheme_name": f"{entry_scheme['entry_name']}__{exit_scheme['exit_name']}",
        "total_return": total_return,
        "annual_return": float(result.get("annual_return", 0.0)),
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "total_trades": int(result.get("total_trades", 0)),
        "win_rate": float(result.get("win_rate", 0.0)),
        "avg_return": float(result.get("avg_return", 0.0)),
        "score": score,
        "skipped_positions": int(summary.get("skipped_positions", 0)),
        "gap_down_skipped": int(summary.get("gap_down_3_skipped", 0)),
        "gap_up_skipped": int(summary.get("gap_up_5_skipped", 0)),
        "ma5_gap_down_skipped": int(summary.get("ma5_gap_down_skipped", 0)),
        "ma5_gap_up_skipped": int(summary.get("ma5_gap_up_skipped", 0)),
        "risk_exits": int(risk_details.get("total_risk_exits", 0)),
        "stoploss_exits": int(risk_details.get("cumulative_decline_exits", 0)),
        "bearish_exits": int(risk_details.get("profit_bearish_exits", 0)),
        "t1_low_break_exits": int(risk_details.get("t1_low_break_exits", 0)),
        "risk_exit_rate": float(risk_details.get("risk_exit_rate", 0.0)),
    }


def _print_summary(result_df: pd.DataFrame) -> None:
    display_cols = [
        "scheme_name",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "score",
        "skipped_positions",
        "risk_exit_rate",
        "stoploss_exits",
        "bearish_exits",
    ]
    print(result_df[display_cols].head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描 Top2 更优风控方案")
    parser.add_argument(
        "--main_script",
        default="main_alpha_二分类_原因子_with_test_metrics_妖股_copy.py",
        help="二分类主脚本路径",
    )
    parser.add_argument(
        "--model_path",
        default="results/lightgbm_model_window3_20260427_231053.pkl",
        help="模型文件路径",
    )
    parser.add_argument(
        "--minute_data_path",
        default="../local_data/minute/",
        help="5分钟数据目录",
    )
    parser.add_argument(
        "--holding_days",
        type=int,
        default=3,
        help="持仓天数",
    )
    parser.add_argument(
        "--entry_names",
        default="",
        help="逗号分隔，仅运行指定entry_name",
    )
    parser.add_argument(
        "--exit_names",
        default="",
        help="逗号分隔，仅运行指定exit_name",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("rqalpha_minute_datasource.strategy_backtest_lead_fusion_rank_binary_v2").setLevel(logging.WARNING)
    logging.getLogger("rqalpha_minute_datasource.minute_backtest").setLevel(logging.WARNING)
    main_script = Path(args.main_script)
    model_path = Path(args.model_path)
    df = _build_backtest_data(main_script, model_path)

    rows = []
    entry_schemes = _build_entry_schemes()
    exit_schemes = _build_exit_schemes()
    if args.entry_names.strip():
        wanted_entry = {x.strip() for x in args.entry_names.split(",") if x.strip()}
        entry_schemes = [x for x in entry_schemes if x["entry_name"] in wanted_entry]
    if args.exit_names.strip():
        wanted_exit = {x.strip() for x in args.exit_names.split(",") if x.strip()}
        exit_schemes = [x for x in exit_schemes if x["exit_name"] in wanted_exit]
    if not entry_schemes or not exit_schemes:
        raise ValueError("筛选后的 entry/exit 组合为空")
    total = len(entry_schemes) * len(exit_schemes)
    idx = 0
    for entry_scheme in entry_schemes:
        for exit_scheme in exit_schemes:
            idx += 1
            print(f"[{idx}/{total}] {entry_scheme['entry_name']} + {exit_scheme['exit_name']}")
            rows.append(
                _run_single_scheme(
                    df=df,
                    minute_data_path=args.minute_data_path,
                    holding_days=args.holding_days,
                    entry_scheme=entry_scheme,
                    exit_scheme=exit_scheme,
                )
            )

    result_df = pd.DataFrame(rows).sort_values(
        ["score", "total_return", "sharpe_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    out_path = Path("results") / f"top2_risk_control_scan_{model_path.stem}.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")
    _print_summary(result_df)


if __name__ == "__main__":
    main()
