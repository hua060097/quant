import json
import numpy as np
import pandas as pd

path = "daily_gate_stats_Window_3_gateN80_20260308_234514.json"

ROLL_SHORT = 10
ROLL_LONG = 20

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

rows = []
for d in dates:
    rec = data[d]
    cls = rec.get("cls_topn") or []
    if len(cls) == 0:
        continue

    # 按分类排序
    cls_by_p = sorted(cls, key=lambda x: x.get("p_cls", float("-inf")), reverse=True)

    # 候选域 Top5
    top5 = cls_by_p[:5]
    if len(top5) == 0:
        continue

    # 候选域均值（分类层 alpha）
    top5_rets = [x.get("real_return") for x in top5 if x.get("real_return") is not None]
    if len(top5_rets) == 0:
        continue
    candidate_alpha = float(np.mean(top5_rets))

    # 分类Top1（对照）
    cls_top1_ret = cls_by_p[0].get("real_return")

    # 排序 Top1（在 Top5 内）
    top5_by_rank = sorted(top5, key=lambda x: x.get("rank_score", float("-inf")), reverse=True)
    rank_top1_ret = top5_by_rank[0].get("real_return")

    # 排序 alpha = 排序选出的Top1 - 候选域均值
    rerank_alpha = None
    if rank_top1_ret is not None:
        rerank_alpha = rank_top1_ret - candidate_alpha

    # 排序分离度
    score_margin = None
    if len(top5_by_rank) >= 2:
        s1 = top5_by_rank[0].get("rank_score")
        s2 = top5_by_rank[1].get("rank_score")
        if s1 is not None and s2 is not None:
            score_margin = float(s1 - s2)

    rows.append({
        "date": d,
        "candidate_alpha": candidate_alpha,
        "rerank_alpha": rerank_alpha,
        "cls_top1_ret": cls_top1_ret,
        "rank_top1_ret": rank_top1_ret,
        "score_margin": score_margin,
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# rolling 指标
for col in ["candidate_alpha", "rerank_alpha", "score_margin"]:
    df[f"r{ROLL_SHORT}_{col}"] = df[col].rolling(ROLL_SHORT).mean()
    df[f"r{ROLL_LONG}_{col}"] = df[col].rolling(ROLL_LONG).mean()

def describe(name, s):
    x = pd.Series(s).dropna().values
    if len(x) == 0:
        print(f"\n{name}: EMPTY")
        return
    print(f"\n{name}: count={len(x)} mean={x.mean():.6f} std={x.std(ddof=1):.6f}")
    for p in [5,10,25,50,75,90,95]:
        print(f"  p{p}: {np.percentile(x, p):.6f}")

print("=== Raw layer distributions ===")
describe("candidate_alpha", df["candidate_alpha"])
describe("rerank_alpha", df["rerank_alpha"])
describe("score_margin", df["score_margin"])

print("\n=== Rolling distributions ===")
describe(f"r{ROLL_SHORT}_candidate_alpha", df[f"r{ROLL_SHORT}_candidate_alpha"])
describe(f"r{ROLL_SHORT}_rerank_alpha", df[f"r{ROLL_SHORT}_rerank_alpha"])
describe(f"r{ROLL_SHORT}_score_margin", df[f"r{ROLL_SHORT}_score_margin"])

describe(f"r{ROLL_LONG}_candidate_alpha", df[f"r{ROLL_LONG}_candidate_alpha"])
describe(f"r{ROLL_LONG}_rerank_alpha", df[f"r{ROLL_LONG}_rerank_alpha"])
describe(f"r{ROLL_LONG}_score_margin", df[f"r{ROLL_LONG}_score_margin"])

# -------------------------
# 状态机：离线样例
# -------------------------
# 阈值来自历史分位，不拍脑袋
cand_bad_th = np.nanpercentile(df[f"r{ROLL_SHORT}_candidate_alpha"], 10)
rerank_bad_th = np.nanpercentile(df[f"r{ROLL_SHORT}_rerank_alpha"], 10)
margin_bad_th = np.nanpercentile(df[f"r{ROLL_LONG}_score_margin"], 25)

cand_good_th = np.nanpercentile(df[f"r{ROLL_SHORT}_candidate_alpha"], 50)
rerank_good_th = np.nanpercentile(df[f"r{ROLL_SHORT}_rerank_alpha"], 50)

print("\n=== Thresholds (from history) ===")
print("cand_bad_th   =", cand_bad_th)
print("rerank_bad_th =", rerank_bad_th)
print("margin_bad_th =", margin_bad_th)
print("cand_good_th  =", cand_good_th)
print("rerank_good_th=", rerank_good_th)

def classify_state(row):
    c = row[f"r{ROLL_SHORT}_candidate_alpha"]
    r = row[f"r{ROLL_SHORT}_rerank_alpha"]
    m = row[f"r{ROLL_LONG}_score_margin"]

    if pd.isna(c) or pd.isna(r) or pd.isna(m):
        return "WARMUP"

    # 市场坏：候选域整体都不赚钱/明显偏弱
    if c <= cand_bad_th and r <= 0 and m > margin_bad_th:
        return "BAD_MARKET"

    # 模型坏：排序 alpha 差，且分离度也塌
    if r <= rerank_bad_th and m <= margin_bad_th:
        return "MODEL_DECAY"

    # alpha变弱：没到坏市场/坏模型，但明显低于正常
    if c < cand_good_th or r < rerank_good_th:
        return "WEAK_ALPHA"

    return "GOOD"

df["state"] = df.apply(classify_state, axis=1)

print("\n=== State counts ===")
print(df["state"].value_counts(dropna=False).to_string())

print("\n=== Recent 20 days state trace ===")
print(df[["date",
          f"r{ROLL_SHORT}_candidate_alpha",
          f"r{ROLL_SHORT}_rerank_alpha",
          f"r{ROLL_LONG}_score_margin",
          "state"]].tail(20).to_string(index=False))

# 各状态的均值
print("\n=== State-wise averages ===")
for s, g in df.groupby("state"):
    print(f"\n[{s}] n={len(g)}")
    print("candidate_alpha mean =", g["candidate_alpha"].mean())
    print("rerank_alpha mean    =", g["rerank_alpha"].mean())
    print("score_margin mean    =", g["score_margin"].mean())
    print("rank_top1_ret mean   =", g["rank_top1_ret"].mean())
    print("cls_top1_ret mean    =", g["cls_top1_ret"].mean())

print("\nDone.")