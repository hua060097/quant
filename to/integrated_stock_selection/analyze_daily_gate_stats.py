import argparse
import json
from pathlib import Path

import numpy as np


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _max_drawdown(returns):
    if not returns:
        return 0.0
    equity = np.cumprod(1.0 + np.array(returns, dtype=float))
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.where(peak == 0, 1.0, peak) - 1.0
    return float(np.min(drawdown))


def _summarize_series(name, rows):
    real_returns = [_safe_float(r.get("real_return")) for r in rows]
    real_returns = [r for r in real_returns if r is not None]
    pred_scores = [_safe_float(r.get("rank_score", r.get("predicted_return"))) for r in rows]
    pred_scores = [r for r in pred_scores if r is not None]
    rank_pct_all = [_safe_float(r.get("rank_pct_all")) for r in rows]
    rank_pct_all = [r for r in rank_pct_all if r is not None]
    p_cls = [_safe_float(r.get("p_cls")) for r in rows]
    p_cls = [r for r in p_cls if r is not None]

    if not real_returns:
        return {
            "name": name,
            "count": 0,
        }

    return {
        "name": name,
        "count": len(real_returns),
        "mean_real_return": float(np.mean(real_returns)),
        "median_real_return": float(np.median(real_returns)),
        "win_rate": float(np.mean(np.array(real_returns) > 0)),
        "cum_return": float(np.prod(1.0 + np.array(real_returns)) - 1.0),
        "max_drawdown": _max_drawdown(real_returns),
        "mean_predicted_return": float(np.mean(pred_scores)) if pred_scores else None,
        "mean_rank_pct_all": float(np.mean(rank_pct_all)) if rank_pct_all else None,
        "mean_p_cls": float(np.mean(p_cls)) if p_cls else None,
    }


def _summarize_daily_basket(name, daily_items):
    daily_returns = []
    daily_pred = []
    daily_rank_pct = []
    daily_p_cls = []
    for item in daily_items:
        members = item["members"]
        member_returns = [_safe_float(x.get("real_return")) for x in members]
        member_returns = [x for x in member_returns if x is not None]
        if not member_returns:
            continue
        daily_returns.append(float(np.mean(member_returns)))

        member_pred = [_safe_float(x.get("rank_score", x.get("predicted_return"))) for x in members]
        member_pred = [x for x in member_pred if x is not None]
        if member_pred:
            daily_pred.append(float(np.mean(member_pred)))

        member_rank_pct = [_safe_float(x.get("rank_pct_all")) for x in members]
        member_rank_pct = [x for x in member_rank_pct if x is not None]
        if member_rank_pct:
            daily_rank_pct.append(float(np.mean(member_rank_pct)))

        member_p_cls = [_safe_float(x.get("p_cls")) for x in members]
        member_p_cls = [x for x in member_p_cls if x is not None]
        if member_p_cls:
            daily_p_cls.append(float(np.mean(member_p_cls)))

    if not daily_returns:
        return {"name": name, "days": 0}

    return {
        "name": name,
        "days": len(daily_returns),
        "mean_daily_return": float(np.mean(daily_returns)),
        "median_daily_return": float(np.median(daily_returns)),
        "win_rate": float(np.mean(np.array(daily_returns) > 0)),
        "cum_return": float(np.prod(1.0 + np.array(daily_returns)) - 1.0),
        "max_drawdown": _max_drawdown(daily_returns),
        "mean_predicted_return": float(np.mean(daily_pred)) if daily_pred else None,
        "mean_rank_pct_all": float(np.mean(daily_rank_pct)) if daily_rank_pct else None,
        "mean_p_cls": float(np.mean(daily_p_cls)) if daily_p_cls else None,
    }


def _print_summary(title, stats):
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in stats.items():
        if key == "name":
            continue
        if isinstance(value, float):
            if "return" in key or "drawdown" in key or "rate" in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="分析 daily_gate_stats JSON 的前半段/后半段表现")
    parser.add_argument("json_path", help="daily_gate_stats_xxx.json 路径")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dates = sorted(data.keys())
    if not dates:
        raise ValueError("JSON 中没有交易日数据")

    mid = len(dates) // 2
    segments = {
        "first_half": dates[:mid],
        "second_half": dates[mid:],
        "full_period": dates,
    }

    for seg_name, seg_dates in segments.items():
        rank_topn_daily = []
        cls_topn_daily = []
        rank_top1_rows = []
        cls_top1_rows = []
        serial_top1_rows = []

        for d in seg_dates:
            day = data[d]
            rank_topn = day.get("rank_topn", [])
            cls_topn = day.get("cls_topn", [])
            cls_top1_obj = day.get("cls_top1")
            rank_all_top1 = day.get("rank_all_top1")
            serial_top1 = day.get("serial_top1")

            if rank_topn:
                rank_topn_daily.append({"date": d, "members": rank_topn})
                rank_top1 = rank_topn[0]
                if isinstance(rank_top1, dict) and rank_top1.get("real_return") is not None:
                    rank_top1_rows.append(rank_top1)
            elif isinstance(rank_all_top1, dict) and rank_all_top1.get("real_return") is not None:
                rank_top1_rows.append(rank_all_top1)

            if cls_topn:
                cls_topn_daily.append({"date": d, "members": cls_topn})
                cls_top1 = cls_topn[0]
                if isinstance(cls_top1, dict) and cls_top1.get("real_return") is not None:
                    cls_top1_rows.append(cls_top1)
            elif isinstance(cls_top1_obj, dict) and cls_top1_obj.get("real_return") is not None:
                cls_top1_rows.append(cls_top1_obj)
            if isinstance(serial_top1, dict) and serial_top1.get("real_return") is not None:
                serial_top1_rows.append(serial_top1)

        print(f"\n{'=' * 80}")
        print(f"{json_path.name} | {seg_name} | days={len(seg_dates)}")
        print(f"{'=' * 80}")

        _print_summary("排序Top1", _summarize_series("rank_top1", rank_top1_rows))
        _print_summary("排序TopN(日均篮子收益)", _summarize_daily_basket("rank_topn", rank_topn_daily))
        _print_summary("分类Top1", _summarize_series("cls_top1", cls_top1_rows))
        _print_summary("分类TopN(日均篮子收益)", _summarize_daily_basket("cls_topn", cls_topn_daily))
        _print_summary("分类TopN+排序Top1", _summarize_series("serial_top1", serial_top1_rows))


if __name__ == "__main__":
    main()
