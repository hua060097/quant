import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

RUN_FILTER = RESULTS / "walkforward_metrics_20260520_190404.json"
RUN_NO_FILTER = RESULTS / "walkforward_metrics_20260520_210454.json"
SHADOW = RESULTS / "walkforward_shadow_skipped_opportunities_20260520_190404.csv"
LOG_FILTER = BASE / "stock_selection_test_mode_walkforward_分类模型_原因子184_optuna.log"


def load_metrics(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    first = data[0]
    result = first["all_results"]["1"]
    return first, result, result["trades"]


def parse_dt(value):
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def trade_key(t):
    return t["stock_code"], t["buy_datetime"]


def equity_curve(trades, initial):
    equity = initial
    peak = initial
    max_dd = 0.0
    peak_date = None
    trough_date = None
    peak_equity = initial
    trough_equity = initial
    rows = []

    for t in sorted(trades, key=lambda x: (parse_dt(x["sell_datetime"]), x["stock_code"])):
        equity += float(t["profit_loss"])
        if equity > peak:
            peak = equity
            peak_date = t["sell_datetime"]
            peak_equity = equity
        dd = (peak - equity) / peak if peak else 0.0
        rows.append(
            {
                "date": t["sell_datetime"],
                "stock_code": t["stock_code"],
                "profit_loss": float(t["profit_loss"]),
                "profit_loss_pct": float(t["profit_loss_pct"]),
                "equity": equity,
                "drawdown": dd,
            }
        )
        if dd > max_dd:
            max_dd = dd
            trough_date = t["sell_datetime"]
            trough_equity = equity
    return rows, {
        "closed_trade_max_dd": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "peak_equity": peak_equity,
        "trough_equity": trough_equity,
    }


def summarize_trades(name, top, result, trades):
    wins = [t for t in trades if float(t["profit_loss"]) > 0]
    losses = [t for t in trades if float(t["profit_loss"]) <= 0]
    print(f"\n== {name} ==")
    print(f"period={top['test_period']}")
    print(
        "metrics: total_return={:.2f}%, max_drawdown={:.2f}%, sharpe={:.2f}, "
        "trades={}, win_rate={:.2f}%, avg_return={:.2%}".format(
            float(result["total_return"]),
            float(result["max_drawdown"]),
            float(result["sharpe_ratio"]),
            int(result["total_trades"]),
            float(result["win_rate"]),
            float(result["avg_return"]),
        )
    )
    print(
        "closed P/L: sum={:.0f}, wins={}, losses={}, avg_loss={:.2%}, avg_win={:.2%}".format(
            sum(float(t["profit_loss"]) for t in trades),
            len(wins),
            len(losses),
            sum(float(t["actual_return"]) for t in losses) / len(losses) if losses else 0.0,
            sum(float(t["actual_return"]) for t in wins) / len(wins) if wins else 0.0,
        )
    )


def print_top_losses(title, trades, n=12):
    print(f"\n-- {title} --")
    for t in sorted(trades, key=lambda x: float(x["profit_loss"]))[:n]:
        print(
            "{buy} {code} ret={ret:.2%} pnl={pnl:.0f} sell={sell} exit={exit}".format(
                buy=t["buy_datetime"][:10],
                code=t["stock_code"],
                ret=float(t["actual_return"]),
                pnl=float(t["profit_loss"]),
                sell=t["sell_datetime"],
                exit=t.get("exit_reason", ""),
            )
        )


def compare_runs(filter_trades, no_filter_trades):
    f_map = {trade_key(t): t for t in filter_trades}
    nf_map = {trade_key(t): t for t in no_filter_trades}
    only_filter = [f_map[k] for k in sorted(set(f_map) - set(nf_map), key=lambda x: x[1])]
    only_no_filter = [nf_map[k] for k in sorted(set(nf_map) - set(f_map), key=lambda x: x[1])]
    common = [f_map[k] for k in set(f_map) & set(nf_map)]

    def agg(rows):
        if not rows:
            return "n=0"
        return "n={}, pnl={:.0f}, avg_ret={:.2%}, win={:.1%}".format(
            len(rows),
            sum(float(t["profit_loss"]) for t in rows),
            sum(float(t["actual_return"]) for t in rows) / len(rows),
            sum(1 for t in rows if float(t["actual_return"]) > 0) / len(rows),
        )

    print("\n== 184(filter) vs 185(no filter) trade-set diff ==")
    print(f"common: {agg(common)}")
    print(f"only_filter: {agg(only_filter)}")
    print(f"only_no_filter: {agg(only_no_filter)}")
    print_top_losses("only_no_filter worst trades", only_no_filter, 10)
    print_top_losses("filter-run worst trades", filter_trades, 12)


def analyze_shadow(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row["actual_return"] = float(row["actual_return"])
            row["profit_loss"] = float(row["profit_loss"])
            row["exhaustion_blocked_bool"] = str(row.get("exhaustion_blocked", "")).lower() == "true"
            rows.append(row)

    by_status = defaultdict(list)
    for row in rows:
        by_status[row.get("shadow_status", "")].append(row)

    print("\n== shadow skipped opportunities 20260520_190404 ==")
    print(
        "total={}, pnl={:.0f}, avg_ret={:.2%}, win={:.1%}, blocked_by_v3={}".format(
            len(rows),
            sum(r["profit_loss"] for r in rows),
            sum(r["actual_return"] for r in rows) / len(rows) if rows else 0.0,
            sum(1 for r in rows if r["actual_return"] > 0) / len(rows) if rows else 0.0,
            sum(1 for r in rows if r["exhaustion_blocked_bool"]),
        )
    )
    for status, items in sorted(by_status.items()):
        print(
            "status={} n={} pnl={:.0f} avg_ret={:.2%} win={:.1%}".format(
                status,
                len(items),
                sum(r["profit_loss"] for r in items),
                sum(r["actual_return"] for r in items) / len(items),
                sum(1 for r in items if r["actual_return"] > 0) / len(items),
            )
        )

    print("\n-- shadow worst PASS opportunities --")
    pass_rows = [r for r in rows if not r["exhaustion_blocked_bool"]]
    for r in sorted(pass_rows, key=lambda x: x["actual_return"])[:10]:
        print(
            "{date} {code} ret={ret:.2%} status={status} gap={gap:.2%} "
            "runup={runup} cpv={cpv} blowoff={blowoff} sector_accel={sector_accel} "
            "liq={liq} sector_rank={sector_rank} reason={reason}".format(
                date=r["buy_datetime"][:10],
                code=r["stock_code"],
                ret=r["actual_return"],
                status=r["shadow_status"],
                gap=float(r.get("entry_gap_ratio") or 0.0),
                runup=r.get("runup_ma5_exhaustion", ""),
                cpv=r.get("cpv_factor", ""),
                blowoff=r.get("prev_day_blowoff_risk", ""),
                sector_accel=r.get("sector_momentum_accel_rank", ""),
                liq=r.get("market_liquidity_change", ""),
                sector_rank=r.get("intra_sector_momentum_rank", ""),
                reason=r.get("entry_reason_detail", ""),
            )
        )

    print("\n-- shadow v3-blocked opportunities --")
    for r in sorted([r for r in rows if r["exhaustion_blocked_bool"]], key=lambda x: x["actual_return"])[:12]:
        print(
            "{date} {code} ret={ret:.2%} pnl={pnl:.0f} reason={reason}".format(
                date=r["buy_datetime"][:10],
                code=r["stock_code"],
                ret=r["actual_return"],
                pnl=r["profit_loss"],
                reason=r.get("exhaustion_reason", ""),
            )
        )
    return rows


def analyze_log_blocks(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocked = re.findall(r"exhaustion BLOCKED \[three_rule_v3\]: (.*)", text)
    counts = Counter()
    for reason in blocked:
        if "runup(" in reason:
            counts["rule1_runup_cpv"] += 1
        if "blowoff(" in reason or "XR2" in reason:
            counts["rule2_blowoff_sector"] += 1
        if "sector_leadership(" in reason:
            counts["rule3_leadership_ret20"] += 1
        if "liq(" in reason:
            counts["rule4_liq_combo"] += 1
        if "disp(" in reason:
            counts["rule5_market_circuit"] += 1
        if "limit_down_3d" in reason:
            counts["rule6_limit_down_3d"] += 1
    print("\n== log blocked-rule hit counts ==")
    print(f"blocked_lines={len(blocked)}")
    for key, value in counts.most_common():
        print(f"{key}: {value}")


def main():
    top_f, res_f, trades_f = load_metrics(RUN_FILTER)
    top_nf, res_nf, trades_nf = load_metrics(RUN_NO_FILTER)
    summarize_trades("184 / three_rule_v3 ON", top_f, res_f, trades_f)
    summarize_trades("185 / exhaustion filter OFF", top_nf, res_nf, trades_nf)

    for name, res, trades in [
        ("184 / three_rule_v3 ON", res_f, trades_f),
        ("185 / exhaustion filter OFF", res_nf, trades_nf),
    ]:
        curve, dd = equity_curve(trades, float(res["initial_capital"]))
        print(
            "\n{} closed-trade DD approx: {:.2%}, peak={}, trough={}, peak_equity={:.0f}, trough_equity={:.0f}".format(
                name,
                dd["closed_trade_max_dd"],
                dd["peak_date"],
                dd["trough_date"],
                dd["peak_equity"],
                dd["trough_equity"],
            )
        )
        if dd["peak_date"] and dd["trough_date"]:
            p = parse_dt(dd["peak_date"])
            q = parse_dt(dd["trough_date"])
            window_trades = [t for t in trades if p <= parse_dt(t["sell_datetime"]) <= q]
            print_top_losses(f"{name} trades sold during closed-DD window", window_trades, 12)

    compare_runs(trades_f, trades_nf)
    analyze_shadow(SHADOW)
    analyze_log_blocks(LOG_FILTER)


if __name__ == "__main__":
    main()
