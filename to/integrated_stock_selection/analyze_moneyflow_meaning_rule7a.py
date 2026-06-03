import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FACTOR_FILE = BASE / "factor_lab_data" / "factors.parquet"
RUN_FILES = {
    "old162_off": RESULTS / "walkforward_metrics_20260512_194610.json",
    "latest184_v3": RESULTS / "walkforward_metrics_20260520_190404.json",
    "latest185_off": RESULTS / "walkforward_metrics_20260520_210454.json",
}
FACTORS = [
    "top10_10d_strength_level_v3",
    "buy_sm_amount_rate",
    "buy_lg_amount_rate",
    "net_d5_amount_to_turnover",
    "money_flow_pressure",
    "money_flow_pressure_pct",
]


def load_events():
    rows = []
    for ds, path in RUN_FILES.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for t in data[0]["all_results"]["1"]["trades"]:
            rows.append(
                {
                    "dataset": ds,
                    "stock_code": t["stock_code"],
                    "buy_date": pd.Timestamp(t["buy_datetime"][:10]),
                    "actual_return": float(t["actual_return"]),
                    "profit_loss": float(t["profit_loss"]),
                }
            )
    events = pd.DataFrame(rows)
    events["key"] = events["stock_code"] + "|" + events["buy_date"].dt.strftime("%Y-%m-%d")
    return events


def enrich(events):
    df = pd.read_parquet(FACTOR_FILE, columns=["date", "stock_code"] + FACTORS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"])
    grouped = {
        stock: (g["date"].to_numpy(dtype="datetime64[ns]"), g.reset_index(drop=True))
        for stock, g in df.groupby("stock_code", sort=False)
    }
    out = []
    for _, e in events.iterrows():
        rec = e.to_dict()
        item = grouped.get(e["stock_code"])
        if item is not None:
            dates, g = item
            pos = np.searchsorted(dates, np.datetime64(e["buy_date"]), side="left") - 1
            if pos >= 0:
                r = g.iloc[int(pos)]
                rec["factor_date"] = r["date"]
                for c in FACTORS:
                    rec[c] = r[c]
        out.append(rec)
    x = pd.DataFrame(out)
    x["lg_minus_sm"] = x["buy_lg_amount_rate"] - x["buy_sm_amount_rate"]
    x["lg_plus_sm"] = x["buy_lg_amount_rate"] + x["buy_sm_amount_rate"]
    return x


def make_ref(df):
    priority = {"latest185_off": 0, "old162_off": 1, "latest184_v3": 2}
    x = df.copy()
    x["priority"] = x["dataset"].map(priority)
    return x.sort_values(["key", "priority"]).drop_duplicates("key", keep="first").drop(columns=["priority"])


def describe(name, d, cols):
    print(f"\n== {name} n={len(d)} ==")
    for c in cols:
        s = d[c].dropna()
        if s.empty:
            continue
        print(
            "{}: mean={:.3f}, med={:.3f}, p25={:.3f}, p75={:.3f}, min={:.3f}, max={:.3f}".format(
                c, s.mean(), s.median(), s.quantile(0.25), s.quantile(0.75), s.min(), s.max()
            )
        )


def main():
    df = enrich(load_events())
    ref = make_ref(df)
    recent = df[
        df["dataset"].eq("latest184_v3")
        & (df["buy_date"] >= pd.Timestamp("2026-04-01"))
        & (df["actual_return"] <= -0.05)
    ].copy()
    hist = ref[~ref["key"].isin(recent["key"])].copy()
    wins = hist[hist["actual_return"] >= 0.10].copy()
    losses = hist[hist["actual_return"] <= -0.05].copy()
    neutral = hist[(hist["actual_return"] > -0.05) & (hist["actual_return"] < 0.10)].copy()

    cols = FACTORS + ["lg_minus_sm", "lg_plus_sm"]
    describe("recent_losses", recent, cols)
    describe("historical_big_wins", wins, cols)
    describe("historical_big_losses", losses, cols)
    describe("historical_neutral", neutral, cols)

    print("\n== recent loss rows ==")
    print(
        recent[
            [
                "stock_code",
                "buy_date",
                "actual_return",
                "top10_10d_strength_level_v3",
                "buy_sm_amount_rate",
                "buy_lg_amount_rate",
                "lg_minus_sm",
                "net_d5_amount_to_turnover",
                "money_flow_pressure",
                "money_flow_pressure_pct",
            ]
        ]
        .sort_values("actual_return")
        .to_string(index=False)
    )

    print("\n== candidate moneyflow variants, rounded thresholds ==")
    variants = {
        "sm_outflow_only": lambda x: (x["top10_10d_strength_level_v3"] >= 0.70) & (x["buy_sm_amount_rate"] <= -1.25),
        "sm_outflow_lg_not_bad": lambda x: (x["top10_10d_strength_level_v3"] >= 0.70)
        & (x["buy_sm_amount_rate"] <= -1.25)
        & (x["buy_lg_amount_rate"] >= -1.0),
        "sm_outflow_lg_positive": lambda x: (x["top10_10d_strength_level_v3"] >= 0.70)
        & (x["buy_sm_amount_rate"] <= -1.25)
        & (x["buy_lg_amount_rate"] > 0.0),
        "sm_outflow_lg_divergence": lambda x: (x["top10_10d_strength_level_v3"] >= 0.70)
        & (x["buy_sm_amount_rate"] <= -1.25)
        & (x["lg_minus_sm"] >= 1.5),
        "both_sm_lg_outflow": lambda x: (x["top10_10d_strength_level_v3"] >= 0.70)
        & (x["buy_sm_amount_rate"] <= -1.25)
        & (x["buy_lg_amount_rate"] <= -0.5),
    }
    groups = {"recent": recent, "wins": wins, "losses": losses, "neutral": neutral}
    for name, fn in variants.items():
        print(f"\n{name}")
        for gname, g in groups.items():
            m = fn(g)
            h = g[m]
            print(
                "  {}: n={}, avg={:.2%}, loss={}, bigwin={}, pos={}".format(
                    gname,
                    len(h),
                    h["actual_return"].mean() if len(h) else 0.0,
                    int((h["actual_return"] <= -0.05).sum()),
                    int((h["actual_return"] >= 0.10).sum()),
                    int((h["actual_return"] > 0).sum()),
                )
            )
        hits = df[fn(df)].sort_values(["actual_return", "buy_date"])
        print("  hit rows:")
        for _, r in hits.head(12).iterrows():
            print(
                "    {ds} {date} {code} ret={ret:.2%} strength={st:.2f} sm={sm:.2f} lg={lg:.2f} diff={diff:.2f}".format(
                    ds=r["dataset"],
                    date=r["buy_date"].date(),
                    code=r["stock_code"],
                    ret=r["actual_return"],
                    st=r["top10_10d_strength_level_v3"],
                    sm=r["buy_sm_amount_rate"],
                    lg=r["buy_lg_amount_rate"],
                    diff=r["lg_minus_sm"],
                )
            )


if __name__ == "__main__":
    main()
