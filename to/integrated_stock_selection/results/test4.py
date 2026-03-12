import json
import numpy as np
import pandas as pd

path = "daily_gate_stats_Window_3_gateN80_20260305_225809.json"
path = "daily_gate_stats_Window_3_gateN80_20260308_234514.json"
ROLL = 10

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

rows = []
for d in dates:
    rec = data[d]
    cls = rec.get("cls_topn") or []
    serial = rec.get("serial_top1") or {}

    # 串行Top1（你现在实际用的：候选内按rank_score选1）
    serial_ret = serial.get("real_return")

    # 删除排序模型后的Top1：候选内按p_cls选1
    cls_top1_ret = None
    if len(cls) > 0:
        cls_top1 = max(cls, key=lambda x: x.get("p_cls", float("-inf")))
        cls_top1_ret = cls_top1.get("real_return")

    # 候选域 baseline：Top5（按p_cls）等权均值（可选）
    cls_sorted_by_p = sorted(cls, key=lambda x: x.get("p_cls", float("-inf")), reverse=True)
    pcls_top5 = [x.get("real_return") for x in cls_sorted_by_p[:5] if x.get("real_return") is not None]
    pcls_top5_mean = float(np.mean(pcls_top5)) if pcls_top5 else None

    rows.append({
        "date": d,
        "serial_ret": serial_ret,
        "cls_top1_ret": cls_top1_ret,
        "diff_serial_minus_cls": (serial_ret - cls_top1_ret) if (serial_ret is not None and cls_top1_ret is not None) else None,
        "pcls_top5_mean": pcls_top5_mean,
    })

df = pd.DataFrame(rows).sort_values("date")

def smean(s):
    s = pd.Series(s).dropna()
    return float(s.mean()) if len(s) else None

def sratio_pos(s):
    s = pd.Series(s).dropna()
    return float((s > 0).mean()) if len(s) else None

def describe(name, s):
    s = pd.Series(s).dropna().values
    if len(s) == 0:
        print(name, "EMPTY")
        return
    ps = [5,10,25,50,75,90,95]
    q = np.percentile(s, ps)
    print(f"\n{name}: count={len(s)} mean={s.mean():.6f} std={s.std(ddof=1):.6f}")
    for p, v in zip(ps, q):
        print(f"  p{p}: {v:.6f}")

print("=== Core comparison (delete ranker) ===")
print("Serial mean:", smean(df["serial_ret"]))
print("ClsTop1 mean:", smean(df["cls_top1_ret"]))
print("Serial-ClsTop1 mean:", smean(df["diff_serial_minus_cls"]))
print("P(Serial > ClsTop1):", sratio_pos(df["diff_serial_minus_cls"]))

describe("serial_ret", df["serial_ret"])
describe("cls_top1_ret", df["cls_top1_ret"])
describe("diff_serial_minus_cls", df["diff_serial_minus_cls"])

# Rolling regime bucket: based on rolling10 serial_ret
df["rolling_serial"] = df["serial_ret"].rolling(ROLL).mean()

# 3 buckets: bad / mid / good regime by rolling10 serial
valid = df.dropna(subset=["rolling_serial", "diff_serial_minus_cls"])
if len(valid) > 0:
    q1 = valid["rolling_serial"].quantile(0.33)
    q2 = valid["rolling_serial"].quantile(0.66)

    def bucket(x):
        if x <= q1: return "bad"
        if x <= q2: return "mid"
        return "good"

    valid["regime"] = valid["rolling_serial"].apply(bucket)

    print("\n=== Serial - ClsTop1 by regime (rolling10 serial) ===")
    for g in ["bad", "mid", "good"]:
        sub = valid[valid["regime"] == g]
        print(f"{g}: n={len(sub)} mean_diff={sub['diff_serial_minus_cls'].mean():.6f}  P(>0)={(sub['diff_serial_minus_cls']>0).mean():.3f}")

# Optional: compare both vs candidate pcls-top5 mean (is top1 selection adding value over top5 average?)
print("\n=== Value over candidate pcls-Top5 mean (optional) ===")
df["serial_minus_pcls5"] = df["serial_ret"] - df["pcls_top5_mean"]
df["cls1_minus_pcls5"] = df["cls_top1_ret"] - df["pcls_top5_mean"]
print("Serial - pclsTop5 mean:", smean(df["serial_minus_pcls5"]), " P(>0):", sratio_pos(df["serial_minus_pcls5"]))
print("ClsTop1 - pclsTop5 mean:", smean(df["cls1_minus_pcls5"]), " P(>0):", sratio_pos(df["cls1_minus_pcls5"]))