from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OLD = Path("results/walkforward_metrics_20260524_131700.json")
NEW = Path("results/walkforward_metrics_20260524_174751.json")


def load(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for w in data:
        for t in w["all_results"]["1"]["trades"]:
            rows.append({
                "stock_code": t["stock_code"],
                "buy_datetime": pd.Timestamp(t["buy_datetime"]),
                "buy_date": pd.Timestamp(str(t["buy_datetime"])[:10]),
                "sell_datetime": pd.Timestamp(t["sell_datetime"]),
                "actual_return": float(t["actual_return"]),
                "profit_loss": float(t["profit_loss"]),
                "exit_reason": t.get("exit_reason", ""),
                "buy_price": float(t["buy_price"]),
                "sell_price": float(t["sell_price"]),
            })
    df = pd.DataFrame(rows)
    df["key"] = df["buy_date"].dt.strftime("%Y%m%d") + "|" + df["stock_code"]
    return df


def summary(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))[0]["all_results"]["1"]
    return {
        "total_return": r.get("total_return"),
        "max_drawdown": r.get("max_drawdown"),
        "win_rate": r.get("win_rate"),
        "total_trades": r.get("total_trades"),
        "final_value": r.get("final_value"),
    }


def buckets(df: pd.DataFrame) -> dict:
    return {
        "n": len(df),
        "pnl": float(df.profit_loss.sum()),
        "<=-8": int((df.actual_return <= -0.08).sum()),
        "<=-5": int((df.actual_return <= -0.05).sum()),
        "<=-4": int((df.actual_return <= -0.04).sum()),
        "-4~0": int(((df.actual_return > -0.04) & (df.actual_return < 0)).sum()),
        "3~8": int(((df.actual_return >= 0.03) & (df.actual_return < 0.08)).sum()),
        ">=8": int((df.actual_return >= 0.08).sum()),
    }


def main() -> None:
    old = load(OLD)
    new = load(NEW)
    print("193 summary", summary(OLD))
    print("194 summary", summary(NEW))
    print("193 buckets", buckets(old))
    print("194 buckets", buckets(new))

    print("\n194 exit reason counts")
    print(new.exit_reason.value_counts().to_string())

    open_stop = new[new.exit_reason.str.contains("09:30开盘", na=False)].copy()
    print("\n194 09:30 open stop trades")
    print(open_stop[[
        "stock_code", "buy_date", "buy_datetime", "sell_datetime", "buy_price", "sell_price",
        "actual_return", "profit_loss", "exit_reason"
    ]].sort_values("actual_return").to_string(index=False))

    m = old.merge(new, on="key", how="outer", suffixes=("_193", "_194"), indicator=True)
    both = m[m["_merge"] == "both"].copy()
    both["ret_diff"] = both["actual_return_194"] - both["actual_return_193"]
    both["pnl_diff"] = both["profit_loss_194"] - both["profit_loss_193"]
    changed = both[(both.exit_reason_193 != both.exit_reason_194) | (both.ret_diff.abs() > 1e-9)].copy()
    print("\nchanged same buy-date/stock trades")
    cols = [
        "stock_code_193", "buy_date_193", "actual_return_193", "exit_reason_193",
        "actual_return_194", "exit_reason_194", "profit_loss_193", "profit_loss_194", "pnl_diff",
    ]
    print(changed[cols].sort_values("pnl_diff").to_string(index=False))

    print("\nold only")
    print(m[m._merge == "left_only"][["stock_code_193", "buy_date_193", "actual_return_193", "profit_loss_193", "exit_reason_193"]].to_string(index=False))
    print("\nnew only")
    print(m[m._merge == "right_only"][["stock_code_194", "buy_date_194", "actual_return_194", "profit_loss_194", "exit_reason_194"]].to_string(index=False))


if __name__ == "__main__":
    main()
