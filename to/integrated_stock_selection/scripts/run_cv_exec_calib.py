# -*- coding: utf-8 -*-
"""阶段3：用 A2（旧84 raw + old valid_mask）11折 CV 的 Rank1 预测，
在 2023-04~2025-12 的 5 分钟引擎上标定执行层配置（严禁看 2026）。

输入：stage15 CV 的 cv_top5_details.csv（dedup/rank==1/param/cp）。
输出：cv_exec_calib.csv（总收益/年化/MDD/Sharpe/胜率/Calmar）+ 每个配置的 pkl。

用法：
  python scripts/run_cv_exec_calib.py \
    --cv_top5 results/stage15_cv_A2_old84_yvalid_20261021/cv_top5_details.csv \
    --out_dir results/cv_exec_calib_20261021
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage15_v5_pipeline import (  # noqa: E402
    OOS_MINUTE_OVERRIDES, find_best_price_cache, run_minute_backtest_oos,
)

_EXEC_COMMON = {
    "entry_risk_down_threshold": -0.03,
    "entry_risk_up_threshold": 0.08,
    "enable_first_bar_down_veto": True,
    "first_bar_down_threshold": -0.02,
}
_ATR_COMMON = {
    "enable_atr_risk_budget": True,
    "atr_risk_per_trade": 0.025,
    "atr_window": 20,
    "atr_position_floor": 0.20,
    "atr_position_cap": 0.50,
}

CONFIGS = [
    {"name": "Baseline_80", "schedule": [0.8, 0.2], "slots": 2, "overrides": {}},
    {"name": "Baseline_50", "schedule": [0.5, 0.5], "slots": 2, "overrides": {}},
    {"name": "Baseline_50_FastBear1", "schedule": [0.5, 0.5], "slots": 2,
     "overrides": {"small_profit_bearish_required_count": 1}},
    {"name": "Exec_Confirm_50", "schedule": [0.5, 0.5], "slots": 2,
     "overrides": dict(_EXEC_COMMON)},
    {"name": "Exec_ATR_2Slot", "schedule": [0.5, 0.5], "slots": 2,
     "overrides": {**_EXEC_COMMON, **_ATR_COMMON}},
    {"name": "Exec_ATR_3Slot", "schedule": [0.4, 0.3, 0.3], "slots": 3,
     "overrides": {**_EXEC_COMMON, **_ATR_COMMON}},
]


def build_configs(only: set[str] | None = None) -> list[dict]:
    out = []
    for c in CONFIGS:
        if only and c["name"] not in only:
            continue
        ov = dict(OOS_MINUTE_OVERRIDES)
        ov.update(c["overrides"])
        out.append({**c, "overrides": ov})
    return out


def load_rank1(path: Path, param: str, cp: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"param", "cp", "date", "rank", "stock_code", "pred", "universe"}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"[CV-EXEC] cv_top5 缺列: {sorted(missing)}")
    df = df[(df["param"].astype(str) == str(param))
            & (df["cp"].astype(int) == int(cp))
            & (df["rank"].astype(int) == 1)
            & (df["universe"].astype(str) == "dedup")].copy()
    if "scheme" in df.columns:
        df = df[df["scheme"].astype(str) == "lgb"]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["stock_code"] = df["stock_code"].astype(str)
    out = df[["date", "stock_code", "pred"]].rename(columns={"pred": "predicted_return"})
    out = out.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    if out.empty:
        raise SystemExit("[CV-EXEC] Rank1 预测为空，检查 param/cp/universe")
    return out


def perf_row(name: str, res: dict) -> dict:
    perf = res.get("Top-1") if isinstance(res, dict) else None
    if not isinstance(perf, dict):
        perf = next((v for v in (res or {}).values() if isinstance(v, dict)), {})
    ann = float(perf.get("annual_return", np.nan))
    mdd = float(perf.get("max_drawdown", np.nan))
    calmar = (ann / (abs(mdd) / 100.0)) if np.isfinite(ann) and np.isfinite(mdd) and abs(mdd) > 1e-9 else np.nan
    return {
        "config": name,
        "total_return_pct": float(perf.get("total_return", np.nan)),
        "annual_return_pct": ann * 100.0 if np.isfinite(ann) else np.nan,
        "max_drawdown_pct": mdd,
        "sharpe": float(perf.get("sharpe_ratio", np.nan)),
        "win_rate_pct": float(perf.get("win_rate", np.nan)),
        "trades": int(perf.get("total_trades", 0) or 0),
        "calmar": calmar,
    }


def _child_logger(out: Path) -> logging.Logger:
    lg = logging.getLogger(f"cv_exec_calib.{out}")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fh = logging.FileHandler(out / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    lg.addHandler(fh)
    return lg


def _run_one(payload: dict) -> dict:
    """子进程：跑单个执行配置，返回一行指标。"""
    c = payload["config"]
    lg = _child_logger(payload["out_dir"])
    lg.info(f"[CV-EXEC] === {c['name']} schedule={c['schedule']} slots={c['slots']} ===")
    res = run_minute_backtest_oos(
        pred_df=payload["pred"].copy(), price_path=payload["price_path"],
        test_start=payload["test_start"], test_end=payload["test_end"],
        out_dir=payload["out_dir"], logger=lg,
        strategy_overrides=c["overrides"],
        out_name=f"minute_bt_{c['name']}.pkl",
        position_schedule=c["schedule"],
        max_concurrent_positions=c["slots"],
    )
    row = perf_row(c["name"], res or {})
    lg.info(f"[CV-EXEC] {row}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_top5", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--param", default="H23")
    ap.add_argument("--cp", type=int, default=170)
    ap.add_argument("--test_start", default="2023-03-31")
    ap.add_argument("--test_end", default="2025-12-31")
    ap.add_argument("--price_path", default="")
    ap.add_argument("--only", default="", help="逗号分隔配置名；空=全部")
    ap.add_argument("--jobs", type=int, default=1, help="并行配置数（每个引擎单线程；服务器可 6）")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cv_exec_calib")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(out / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)

    only = {x.strip() for x in str(args.only).split(",") if x.strip()} or None
    pred = load_rank1(Path(args.cv_top5), args.param, args.cp)
    logger.info(f"[CV-EXEC] Rank1 天数={len(pred)} 区间={pred['date'].min().date()}~{pred['date'].max().date()}")
    price_path = Path(args.price_path) if args.price_path else find_best_price_cache(args.test_start)
    logger.info(f"[CV-EXEC] price_path={price_path}")

    payloads = [{
        "config": c, "pred": pred, "price_path": price_path,
        "test_start": args.test_start, "test_end": args.test_end, "out_dir": out,
    } for c in build_configs(only)]
    _nj = max(1, min(int(args.jobs or 1), len(payloads)))
    logger.info(f"[CV-EXEC] 并行配置数={_nj}/{len(payloads)}")
    if _nj <= 1:
        rows = [_run_one(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=_nj) as ex:
            rows = list(ex.map(_run_one, payloads))
    df = pd.DataFrame(rows)
    df.to_csv(out / "cv_exec_calib.csv", index=False, encoding="utf-8-sig")
    logger.info("\n" + df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())