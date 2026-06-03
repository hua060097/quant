from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OLD = Path("results/walkforward_metrics_20260522_013300.json")
NEW = Path("results/walkforward_metrics_20260524_131700.json")
OLD_SEL = Path("results/walkforward_selections_20260522_013300.json")
NEW_SEL = Path("results/walkforward_selections_20260524_131700.json")
NEW_SHADOW = Path("results/walkforward_shadow_skipped_opportunities_20260524_131700.csv")


def load_trades(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for wi, win in enumerate(data):
        result = win.get("all_results", {}).get("1", {})
        for t in result.get("trades", []):
            rows.append({
                "window": win.get("window_name", f"w{wi}"),
                "stock_code": t.get("stock_code"),
                "buy_datetime": pd.Timestamp(str(t.get("buy_datetime"))),
                "buy_date": pd.Timestamp(str(t.get("buy_datetime"))[:10]),
                "sell_datetime": pd.Timestamp(str(t.get("sell_datetime"))),
                "actual_return": float(t.get("actual_return", 0)),
                "profit_loss": float(t.get("profit_loss", 0)),
                "buy_price": float(t.get("buy_price", 0)),
                "sell_price": float(t.get("sell_price", 0)),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["key"] = df["buy_date"].dt.strftime("%Y%m%d")
    return df


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Usually one window in this run.
    result = data[0].get("all_results", {}).get("1", {})
    keys = [
        "total_return", "annual_return", "max_drawdown", "sharpe_ratio",
        "calmar_ratio", "win_rate", "total_trades", "final_value",
    ]
    return {k: result.get(k) for k in keys}


def bucket_counts(df: pd.DataFrame) -> dict:
    return {
        "n": len(df),
        "pnl": float(df["profit_loss"].sum()) if len(df) else 0.0,
        "ret_sum": float(df["actual_return"].sum()) if len(df) else 0.0,
        "<=-8": int((df["actual_return"] <= -0.08).sum()),
        "<=-4": int((df["actual_return"] <= -0.04).sum()),
        "-4~0": int(((df["actual_return"] > -0.04) & (df["actual_return"] < 0)).sum()),
        "0~3": int(((df["actual_return"] >= 0) & (df["actual_return"] < 0.03)).sum()),
        "3~8": int(((df["actual_return"] >= 0.03) & (df["actual_return"] < 0.08)).sum()),
        ">=8": int((df["actual_return"] >= 0.08).sum()),
    }


def main() -> None:
    old = load_trades(OLD)
    new = load_trades(NEW)
    print("OLD summary", load_summary(OLD))
    print("NEW summary", load_summary(NEW))
    print("\nOLD buckets", bucket_counts(old))
    print("NEW buckets", bucket_counts(new))

    merged = old.merge(
        new,
        on="key",
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )
    same = merged[(merged["_merge"] == "both") & (merged["stock_code_old"] == merged["stock_code_new"])].copy()
    switched = merged[(merged["_merge"] == "both") & (merged["stock_code_old"] != merged["stock_code_new"])].copy()
    old_only = merged[merged["_merge"] == "left_only"].copy()
    new_only = merged[merged["_merge"] == "right_only"].copy()
    print("\ncounts same/switched/old_only/new_only", len(same), len(switched), len(old_only), len(new_only))
    print("same pnl old/new", same["profit_loss_old"].sum(), same["profit_loss_new"].sum())
    print("switched pnl old/new/diff", switched["profit_loss_old"].sum(), switched["profit_loss_new"].sum(), (switched["profit_loss_new"] - switched["profit_loss_old"]).sum())
    print("old_only pnl", old_only["profit_loss_old"].sum(), "new_only pnl", new_only["profit_loss_new"].sum())

    if not switched.empty:
        switched["pnl_diff"] = switched["profit_loss_new"] - switched["profit_loss_old"]
        cols = [
            "key", "stock_code_old", "actual_return_old", "profit_loss_old",
            "stock_code_new", "actual_return_new", "profit_loss_new", "pnl_diff",
        ]
        print("\nWorst switched days by pnl_diff")
        print(switched[cols].sort_values("pnl_diff").head(30).to_string(index=False))
        print("\nBest switched days by pnl_diff")
        print(switched[cols].sort_values("pnl_diff", ascending=False).head(20).to_string(index=False))

    # infer Rule8 actual filtered candidates: old trade absent or replaced in new on same day.
    if NEW_SHADOW.exists():
        shadow = pd.read_csv(NEW_SHADOW)
        if not shadow.empty:
            shadow["buy_date"] = pd.to_datetime(shadow["buy_datetime"].astype(str).str[:10])
            shadow["key"] = shadow["buy_date"].dt.strftime("%Y%m%d")
            print("\nNEW shadow buckets", bucket_counts(pd.DataFrame({
                "actual_return": pd.to_numeric(shadow["actual_return"], errors="coerce"),
                "profit_loss": pd.to_numeric(shadow["profit_loss"], errors="coerce"),
            })))
            print("\nNEW shadow rows")
            show = ["key", "stock_code", "actual_return", "profit_loss", "skip_reason"] if "skip_reason" in shadow.columns else ["key", "stock_code", "actual_return", "profit_loss"]
            print(shadow[show].sort_values("actual_return").to_string(index=False))


if __name__ == "__main__":
    main()
