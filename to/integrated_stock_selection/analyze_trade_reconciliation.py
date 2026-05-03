import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "results" / "daily_top5_details_Window_3_20260323_203408.json"

LOG_CONFIGS = [
    {
        "name": "serial_top1",
        "path": ROOT / "stock_selection_test_mode_walkforward_排序模型_原始因子_实盘_overlay_新市场因子_原日期03231942.log",
        "mode": "serial",
        "dynamic_rank": False,
        "position_pct": 1.0,
    },
    {
        "name": "parallel_top1",
        "path": ROOT / "stock_selection_test_mode_walkforward_排序模型_原始因子_实盘_overlay_新市场因子_并行03241134.log",
        "mode": "parallel",
        "dynamic_rank": False,
        "position_pct": 0.5,
    },
    {
        "name": "serial_switch",
        "path": ROOT / "stock_selection_test_mode_walkforward_排序模型_原始因子_实盘_overlay_新市场因子_原日期03232028.log",
        "mode": "serial",
        "dynamic_rank": True,
        "position_pct": 1.0,
    },
    {
        "name": "parallel_switch",
        "path": ROOT / "stock_selection_test_mode_walkforward_排序模型_原始因子_实盘_overlay_新市场因子_并行03241055.log",
        "mode": "parallel",
        "dynamic_rank": True,
        "position_pct": 0.5,
    },
]

RE_RANK = re.compile(r"处理Top-1策略: 买入排名第(\d+)的股票")
RE_BUY_LOGIC = re.compile(
    r"📅\s+([0-9A-Z.]+)\s+T\+1买入逻辑: T日=(\d{4}-\d{2}-\d{2})\(预测日\), T\+1日=(\d{4}-\d{2}-\d{2})\(执行日\)"
)
RE_SELL = re.compile(
    r"💰 卖出: ([0-9A-Z.]+), 价格: ([0-9.]+), 收益: ([+-]?[0-9.]+) \(([+-]?[0-9.]+)%\), 原因: (.+)$"
)
RE_TOTAL_RETURN = re.compile(r"总收益:\s*([+-]?[0-9.]+)%")
RE_MAX_DD = re.compile(r"最大回撤:\s*([+-]?[0-9.]+)%")
RE_WIN_RATE = re.compile(r"胜率:\s*([+-]?[0-9.]+)%")
RE_TRADE_COUNT = re.compile(r"交易次数:\s*(\d+)")


@dataclass
class Trade:
    stock_code: str
    t_date: str
    buy_date: str
    rank: int
    position_pct: float
    actual_return: Optional[float] = None
    sell_reason: Optional[str] = None
    json_real_return: Optional[float] = None


def load_json_map() -> Dict[str, Dict[int, float]]:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    out: Dict[str, Dict[int, float]] = {}
    for date_key, payload in data.items():
        rank_map = {}
        for i in range(1, 6):
            rank_key = f"top{i}"
            if rank_key in payload and payload[rank_key] is not None:
                rank_map[i] = float(payload[rank_key]["real_return"])
        out[date_key] = rank_map
    return out


def extract_log_summary(text: str) -> Dict[str, Optional[float]]:
    def last_float(pattern: re.Pattern[str]) -> Optional[float]:
        matches = pattern.findall(text)
        return float(matches[-1]) if matches else None

    def last_int(pattern: re.Pattern[str]) -> Optional[int]:
        matches = pattern.findall(text)
        return int(matches[-1]) if matches else None

    return {
        "total_return_pct": last_float(RE_TOTAL_RETURN),
        "max_drawdown_pct": last_float(RE_MAX_DD),
        "win_rate_pct": last_float(RE_WIN_RATE),
        "trade_count": last_int(RE_TRADE_COUNT),
    }


def compute_path_stats(returns: List[float]) -> Dict[str, float]:
    if not returns:
        return {
            "mean_pct": float("nan"),
            "cum_pct": float("nan"),
            "win_rate_pct": float("nan"),
            "max_drawdown_pct": float("nan"),
        }

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    wins = 0
    for r in returns:
        if r > 0:
            wins += 1
        equity *= 1.0 + r
        if equity > peak:
            peak = equity
        dd = equity / peak - 1.0
        if dd < max_dd:
            max_dd = dd
    return {
        "mean_pct": sum(returns) / len(returns) * 100.0,
        "cum_pct": (equity - 1.0) * 100.0,
        "win_rate_pct": wins / len(returns) * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
    }


def parse_trades(config: Dict[str, object], json_map: Dict[str, Dict[int, float]]) -> Dict[str, object]:
    text = Path(config["path"]).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    open_trades: Dict[str, List[Trade]] = defaultdict(list)
    finished: List[Trade] = []
    current_rank = 1

    for line in lines:
        rank_match = RE_RANK.search(line)
        if rank_match:
            current_rank = int(rank_match.group(1))
            continue

        buy_match = RE_BUY_LOGIC.search(line)
        if buy_match:
            stock_code = buy_match.group(1)
            t_date = buy_match.group(2).replace("-", "")
            buy_date = buy_match.group(3).replace("-", "")
            rank = current_rank if config["dynamic_rank"] else 1
            trade = Trade(
                stock_code=stock_code,
                t_date=t_date,
                buy_date=buy_date,
                rank=rank,
                position_pct=float(config["position_pct"]),
                json_real_return=json_map.get(t_date, {}).get(rank),
            )
            open_trades[stock_code].append(trade)
            continue

        sell_match = RE_SELL.search(line)
        if sell_match:
            stock_code = sell_match.group(1)
            actual_return = float(sell_match.group(3))
            sell_reason = sell_match.group(5)
            if not open_trades[stock_code]:
                continue
            trade = open_trades[stock_code].pop(0)
            trade.actual_return = actual_return
            trade.sell_reason = sell_reason
            finished.append(trade)

    finished.sort(key=lambda x: (x.buy_date, x.stock_code))
    actual_weighted = [t.position_pct * float(t.actual_return) for t in finished if t.actual_return is not None]
    json_weighted = [
        t.position_pct * float(t.json_real_return)
        for t in finished
        if t.actual_return is not None and t.json_real_return is not None
    ]

    return {
        "summary": extract_log_summary(text),
        "trades": finished,
        "actual_trade_path": compute_path_stats(actual_weighted),
        "json_trade_path": compute_path_stats(json_weighted),
        "missing_json_count": sum(1 for t in finished if t.actual_return is not None and t.json_real_return is None),
    }


def main() -> None:
    json_map = load_json_map()
    report = {}
    for cfg in LOG_CONFIGS:
        report[cfg["name"]] = parse_trades(cfg, json_map)

    print(json.dumps(report, ensure_ascii=False, indent=2, default=lambda o: o.__dict__))


if __name__ == "__main__":
    main()
