# -*- coding: utf-8 -*-
import re
import math
import numpy as np
import pandas as pd

LOG_PATH = r"stock_selection_test_mode_walkforward_排序模型_原始因子_实盘7.log"

ROLLING_N = 60
LEAD_Z_TH = -0.4  # 你可以改成 0 / 0.5 / 1.0 等
MIN_HIST = 10      # rolling 窗口内至少多少天才开始算 lead_z
EPS = 1e-9

pred_date_re = re.compile(r"INFO - 📊 (\d{4}-\d{2}-\d{2}) Top-K预测数据: (\d+)只股票")
pred_line_re = re.compile(r"INFO -\s+📈\s+([0-9A-Z]{6}\.(?:SH|SZ))[:：]\s+预测收益=([0-9\.\-]+)")
buy_signal_re = re.compile(r"买入信号:\s+(True|False)")
buy_success_re = re.compile(r"✅ 买入成功:\s+([0-9A-Z]{6}\.(?:SH|SZ))")
sell_re = re.compile(r"💰 卖出:\s+([0-9A-Z]{6}\.(?:SH|SZ)),.*收益:\s*([0-9\.\-]+)\s*\(")

def parse_log(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    rows = []
    current_pred_date = None
    current_topk = []
    current_topk_n = None
    active_trade_row = None

    for line in lines:
        m = pred_date_re.search(line)
        if m:
            current_pred_date = m.group(1)
            current_topk = []
            current_topk_n = int(m.group(2))
            continue

        m = pred_line_re.search(line)
        if m and current_pred_date is not None:
            current_topk.append((m.group(1), float(m.group(2))))
            if current_topk_n and len(current_topk) >= current_topk_n:
                top1_code, top1_score = current_topk[0]
                topk_scores = [s for _, s in current_topk]
                topk_mean = float(np.mean(topk_scores)) if topk_scores else np.nan
                lead_vs_top5mean = float(top1_score - topk_mean) if math.isfinite(topk_mean) else np.nan
                gap = float(topk_scores[0] - topk_scores[1]) if len(topk_scores) >= 2 else np.nan

                rows.append({
                    "date": pd.to_datetime(current_pred_date).date(),
                    "top1_code": top1_code,
                    "top1_score": top1_score,
                    "topk_mean": topk_mean,
                    "lead_vs_top5mean": lead_vs_top5mean,
                    "gap": gap,
                    "buy_allowed": None,
                    "realized_ret": np.nan,
                })

                current_pred_date = None
                current_topk = []
                current_topk_n = None
            continue

        m = buy_signal_re.search(line)
        if m and rows:
            if rows[-1]["buy_allowed"] is None:
                rows[-1]["buy_allowed"] = (m.group(1) == "True")
            continue

        m = buy_success_re.search(line)
        if m:
            code = m.group(1)
            # 将 active_trade_row 绑定到最近一条同 code 的 row
            for r in reversed(rows):
                if r["top1_code"] == code and pd.isna(r["realized_ret"]):
                    active_trade_row = r
                    break
            continue

        m = sell_re.search(line)
        if m:
            code = m.group(1)
            ret = float(m.group(2))
            if active_trade_row is not None and active_trade_row["top1_code"] == code and pd.isna(active_trade_row["realized_ret"]):
                active_trade_row["realized_ret"] = ret
                active_trade_row = None
            else:
                # 兜底：匹配最近未填 realized_ret 的同 code 行
                for r in reversed(rows):
                    if r["top1_code"] == code and pd.isna(r["realized_ret"]):
                        r["realized_ret"] = ret
                        break
            continue

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

def add_lead_z(df: pd.DataFrame, n=10, min_hist=5, eps=1e-9) -> pd.DataFrame:
    leads = df["lead_vs_top5mean"].astype(float).values
    lead_z = []
    for i in range(len(df)):
        hist = leads[max(0, i - n): i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < min_hist or (not np.isfinite(leads[i])):
            lead_z.append(np.nan)
            continue
        med = np.median(hist)
        mad = np.median(np.abs(hist - med))
        denom = 1.4826 * mad + eps
        lead_z.append((leads[i] - med) / denom)
    df = df.copy()
    df["lead_z"] = lead_z
    return df

def summarize(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "win_rate": np.nan}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "win_rate": float((x > 0).mean()),
    }

if __name__ == "__main__":
    df = parse_log(LOG_PATH)
    print(f"[OK] 解析到 {len(df)} 个预测日")

    df = add_lead_z(df, n=ROLLING_N, min_hist=MIN_HIST, eps=EPS)

    base = df[(df["buy_allowed"] == True) & df["realized_ret"].notna()]
    print("\n=== 基线（不做 lead_z 过滤） ===")
    print(summarize(base["realized_ret"]))

    eval_df = df[(df["buy_allowed"] == True) & df["realized_ret"].notna() & df["lead_z"].notna()]
    keep = eval_df[eval_df["lead_z"] >= LEAD_Z_TH]
    drop = eval_df[eval_df["lead_z"] < LEAD_Z_TH]

    print(f"\n=== lead_z 过滤（rolling={ROLLING_N}, th={LEAD_Z_TH}）===")
    print("✅ 保留:", summarize(keep["realized_ret"]))
    print("🚫 过滤:", summarize(drop["realized_ret"]))
