import json
import numpy as np
import pandas as pd

# ========= 配置 =========
path = "daily_gate_stats_Window_3_gateN80_20260305_225809.json"
ROLL = 10  # rolling窗口，用于分布与启停信号统计
TOPK_FOR_BASELINE = 5  # 候选域 baseline 用 top5 平均
# =======================

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

rows = []
for d in dates:
    rec = data[d]
    cls = rec.get("cls_topn") or []
    serial = rec.get("serial_top1") or {}
    rank_all = rec.get("rank_all_top1") or {}

    # --- 三类Top1 ---
    rank_all_code = rank_all.get("code")
    rank_all_ret = rank_all.get("real_return")
    rank_all_score = rank_all.get("predicted_return")

    serial_code = serial.get("code")
    serial_ret = serial.get("real_return")
    serial_rank_pos_all = serial.get("rank_pos_all")
    serial_pcls = serial.get("p_cls")
    serial_score = serial.get("predicted_return")

    # 分类Top1（在cls_topn内按p_cls最大）
    cls_top1 = None
    if len(cls) > 0:
        cls_top1 = max(cls, key=lambda x: x.get("p_cls", float("-inf")))
    cls_top1_code = cls_top1.get("code") if cls_top1 else None
    cls_top1_ret = cls_top1.get("real_return") if cls_top1 else None
    cls_top1_pcls = cls_top1.get("p_cls") if cls_top1 else None
    cls_top1_rank_pos_all = cls_top1.get("rank_pos_all") if cls_top1 else None
    cls_top1_rank_score = cls_top1.get("rank_score") if cls_top1 else None

    # 候选域按排序分数排序（rank_score）
    cls_sorted_by_rank = sorted(cls, key=lambda x: x.get("rank_score", float("-inf")), reverse=True)
    # 候选域 Top2 / Top5 baseline
    cls_rank2_ret = cls_sorted_by_rank[1].get("real_return") if len(cls_sorted_by_rank) >= 2 else None
    cls_rank_top5_rets = [x.get("real_return") for x in cls_sorted_by_rank[:TOPK_FOR_BASELINE] if x.get("real_return") is not None]
    cls_rank_top5_mean_ret = float(np.mean(cls_rank_top5_rets)) if len(cls_rank_top5_rets) > 0 else None

    # 候选域按分类分数排序（p_cls）
    cls_sorted_by_pcls = sorted(cls, key=lambda x: x.get("p_cls", float("-inf")), reverse=True)
    cls_pcls_top5_rets = [x.get("real_return") for x in cls_sorted_by_pcls[:TOPK_FOR_BASELINE] if x.get("real_return") is not None]
    cls_pcls_top5_mean_ret = float(np.mean(cls_pcls_top5_rets)) if len(cls_pcls_top5_rets) > 0 else None

    # --- 你新增的分离度指标 ---
    score_margin = rec.get("conditional_score_margin_top1_top2")  # 你加的
    cond_gap = rec.get("conditional_gap")
    uncond_gap = rec.get("unconditional_gap")

    # --- 门控一致性：rank_all_top1 是否在 cls_topn 内 ---
    cls_codes = {x.get("code") for x in cls if x.get("code")}
    recall_rankall_in_cls = None
    if rank_all_code and len(cls_codes) > 0:
        recall_rankall_in_cls = 1.0 if rank_all_code in cls_codes else 0.0

    # --- 三者谁更好（当日） ---
    # serial vs cls_top1
    serial_minus_cls = (serial_ret - cls_top1_ret) if (serial_ret is not None and cls_top1_ret is not None) else None
    # serial vs rank_all
    serial_minus_rankall = (serial_ret - rank_all_ret) if (serial_ret is not None and rank_all_ret is not None) else None
    # rank_all vs cls_top1
    rankall_minus_cls = (rank_all_ret - cls_top1_ret) if (rank_all_ret is not None and cls_top1_ret is not None) else None

    # 排序Top1是否真的优于候选域Top2（你之前问过）
    top1_minus_top2 = (serial_ret - cls_rank2_ret) if (serial_ret is not None and cls_rank2_ret is not None) else None

    # 排序Top1相对候选域Top5平均（更重要）
    top1_minus_top5rankmean = (serial_ret - cls_rank_top5_mean_ret) if (serial_ret is not None and cls_rank_top5_mean_ret is not None) else None

    # 分类Top1相对分类Top5平均
    cls1_minus_cls5pclsmean = (cls_top1_ret - cls_pcls_top5_mean_ret) if (cls_top1_ret is not None and cls_pcls_top5_mean_ret is not None) else None

    rows.append({
        "date": d,
        "gate_n": rec.get("gate_n"),
        "universe_size": rec.get("universe_size"),

        "rank_all_code": rank_all_code,
        "rank_all_ret": rank_all_ret,
        "rank_all_score": rank_all_score,

        "cls_top1_code": cls_top1_code,
        "cls_top1_ret": cls_top1_ret,
        "cls_top1_pcls": cls_top1_pcls,
        "cls_top1_rank_pos_all": cls_top1_rank_pos_all,
        "cls_top1_rank_score": cls_top1_rank_score,

        "serial_code": serial_code,
        "serial_ret": serial_ret,
        "serial_pcls": serial_pcls,
        "serial_score": serial_score,
        "serial_rank_pos_all": serial_rank_pos_all,

        "cond_gap": cond_gap,
        "uncond_gap": uncond_gap,
        "score_margin": score_margin,
        "recall_rankall_in_cls": recall_rankall_in_cls,

        "serial_minus_cls": serial_minus_cls,
        "serial_minus_rankall": serial_minus_rankall,
        "rankall_minus_cls": rankall_minus_cls,

        "top1_minus_top2": top1_minus_top2,
        "top1_minus_top5rankmean": top1_minus_top5rankmean,
        "cls1_minus_cls5pclsmean": cls1_minus_cls5pclsmean,

        "cls_rank_top5_mean_ret": cls_rank_top5_mean_ret,
        "cls_pcls_top5_mean_ret": cls_pcls_top5_mean_ret,
    })

df = pd.DataFrame(rows).sort_values("date")

# ============ 基础汇总 ============
def smean(x):
    x = pd.Series(x).dropna()
    return float(x.mean()) if len(x) else None

def sratio_pos(x):
    x = pd.Series(x).dropna()
    return float((x > 0).mean()) if len(x) else None

print("\n=== Overall Means (label returns) ===")
print("Rank-All Top1 mean:", smean(df["rank_all_ret"]))
print("Cls-Top1(mean within cls_topn, p_cls max) mean:", smean(df["cls_top1_ret"]))
print("Serial-Top1 mean:", smean(df["serial_ret"]))

print("\n=== Pairwise (Serial as final decision) ===")
print("Serial - ClsTop1 mean:", smean(df["serial_minus_cls"]), " | P(>0):", sratio_pos(df["serial_minus_cls"]))
print("Serial - RankAllTop1 mean:", smean(df["serial_minus_rankall"]), " | P(>0):", sratio_pos(df["serial_minus_rankall"]))
print("RankAllTop1 - ClsTop1 mean:", smean(df["rankall_minus_cls"]), " | P(>0):", sratio_pos(df["rankall_minus_cls"]))

print("\n=== Does ranking HELP inside the candidate set? ===")
print("SerialTop1 - candidateRankTop2 mean:", smean(df["top1_minus_top2"]), " | P(>0):", sratio_pos(df["top1_minus_top2"]))
print("SerialTop1 - candidateRankTop5Mean mean:", smean(df["top1_minus_top5rankmean"]), " | P(>0):", sratio_pos(df["top1_minus_top5rankmean"]))

print("\n=== Does classification ranking HELP inside candidate set? ===")
print("ClsTop1 - candidatePclsTop5Mean mean:", smean(df["cls1_minus_cls5pclsmean"]), " | P(>0):", sratio_pos(df["cls1_minus_cls5pclsmean"]))

print("\n=== Gating alignment ===")
print("Recall(rank_all_top1 in cls_topn):", smean(df["recall_rankall_in_cls"]))

print("\n=== Structure signals ===")
print("conditional_gap mean:", smean(df["cond_gap"]))
print("unconditional_gap mean:", smean(df["uncond_gap"]))
print("score_margin mean:", smean(df["score_margin"]))

# ============ 分布（全样本 + rolling） ============
df["rolling{}_serial_ret".format(ROLL)] = df["serial_ret"].rolling(ROLL).mean()
df["rolling{}_cond_gap".format(ROLL)] = df["cond_gap"].rolling(ROLL).mean()

def describe(name, series):
    x = pd.Series(series).dropna().values
    if len(x) == 0:
        print(name, "EMPTY")
        return
    ps = [5,10,25,50,75,90,95]
    q = np.percentile(x, ps)
    print(f"\n{name}: count={len(x)} mean={x.mean():.6f} std={x.std(ddof=1):.6f}")
    for p, v in zip(ps, q):
        print(f"  p{p}: {v:.6f}")

print("\n=== Distributions ===")
describe("serial_ret", df["serial_ret"])
describe("rank_all_ret", df["rank_all_ret"])
describe("cls_top1_ret", df["cls_top1_ret"])
describe("serial_minus_cls", df["serial_minus_cls"])
describe("serial_minus_rankall", df["serial_minus_rankall"])
describe("top1_minus_top5rankmean", df["top1_minus_top5rankmean"])
describe("score_margin", df["score_margin"])
describe(f"rolling{ROLL}_serial_ret", df[f"rolling{ROLL}_serial_ret"])
describe(f"rolling{ROLL}_cond_gap", df[f"rolling{ROLL}_cond_gap"])

print("\nDone.")