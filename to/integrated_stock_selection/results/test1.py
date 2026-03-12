import json
import numpy as np
import pandas as pd

path = "daily_gate_stats_Window_3_gateN80_20260305_225809.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

records = []

for d in dates:
    rec = data[d]

    serial_ret = None
    if rec.get("serial_top1"):
        serial_ret = rec["serial_top1"].get("real_return")

    records.append({
        "date": d,
        "serial_return": serial_ret,
        "conditional_gap": rec.get("conditional_gap"),
        "unconditional_gap": rec.get("unconditional_gap"),
        "score_margin": rec.get("conditional_score_margin_top1_top2")
    })

df = pd.DataFrame(records)
df = df.sort_values("date")

# rolling metrics
df["rolling10_return"] = df["serial_return"].rolling(10).mean()
df["rolling10_gap"] = df["conditional_gap"].rolling(10).mean()

def describe_series(name, s):
    s = s.dropna()
    print(f"\n{name}")
    print("count:", len(s))
    print("mean:", s.mean())
    print("std:", s.std())
    print("p5:", np.percentile(s,5))
    print("p10:", np.percentile(s,10))
    print("p25:", np.percentile(s,25))
    print("median:", np.percentile(s,50))
    print("p75:", np.percentile(s,75))
    print("p90:", np.percentile(s,90))

describe_series("serial_return", df["serial_return"])
describe_series("rolling10_return", df["rolling10_return"])
describe_series("conditional_gap", df["conditional_gap"])
describe_series("rolling10_gap", df["rolling10_gap"])
describe_series("score_margin", df["score_margin"])