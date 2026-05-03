import json, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd


def calc_stats(arr):
    s = pd.Series(arr, dtype='float64').dropna()
    if len(s) == 0:
        return {}
    nav = (1 + s).cumprod()
    dd = nav / nav.cummax() - 1
    std = s.std(ddof=0)
    sharpe = (s.mean() / std) * math.sqrt(252) if std > 0 else np.nan
    cvar10 = s[s <= s.quantile(0.10)].mean() if len(s) else np.nan
    return {
        'count': int(len(s)),
        'mean': float(s.mean()),
        'median': float(s.median()),
        'std': float(std),
        'win_rate': float((s > 0).mean()),
        'big_loss_3pct_rate': float((s <= -0.03).mean()),
        'big_loss_5pct_rate': float((s <= -0.05).mean()),
        'big_loss_8pct_rate': float((s <= -0.08).mean()),
        'cvar10': float(cvar10),
        'max_drawdown': float(dd.min()),
        'sharpe': float(sharpe),
        'final_nav': float(nav.iloc[-1]),
        'p5': float(s.quantile(0.05)),
        'p10': float(s.quantile(0.10)),
        'p25': float(s.quantile(0.25)),
        'p75': float(s.quantile(0.75)),
        'p90': float(s.quantile(0.90)),
        'p95': float(s.quantile(0.95)),
    }


def fmt_pct(x):
    return f"{x*100:.2f}%"


def print_stats(title, st):
    print("=" * 120)
    print(title)
    print("=" * 120)
    for k, v in st.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print()


def weighted_rank_ret(day, weights):
    rank_topn = day.get('rank_topn', [])
    k = len(weights)
    if len(rank_topn) < k:
        return np.nan
    rets = np.array([rank_topn[i]['real_return'] for i in range(k)], dtype=float)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return float((rets * w).sum())


def analyze(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    dates = sorted(data.keys())
    half = len(dates) // 2
    parts = {
        'full': dates,
        'first_half': dates[:half],
        'second_half': dates[half:],
    }

    out_rows = []

    for part_name, dts in parts.items():
        rank_top1 = []
        serial_top1 = []
        cls_top1 = []
        rank_eq = {k: [] for k in range(2, 11)}
        weighted = {
            'w_50_50': [],
            'w_60_40': [],
            'w_70_30': [],
            'w_80_20': [],
            'w_50_30_20': [],
            'w_40_30_30': [],
            'w_eq3': [],
        }

        same_pick = []
        rank_top1_in_cls = []
        serial_in_rank_top20 = []
        rank_top1_cls_rank = []
        serial_rank_rank = []

        for dt in dts:
            day = data[dt]
            rank_top1.append(day['rank_all_top1']['real_return'])
            serial_top1.append(day['serial_top1']['real_return'])
            cls_top1.append(day['cls_top1']['real_return'])

            rtopn = day.get('rank_topn', [])
            for k in range(2, 11):
                if len(rtopn) >= k:
                    rank_eq[k].append(np.mean([x['real_return'] for x in rtopn[:k]]))

            weighted['w_50_50'].append(weighted_rank_ret(day, [0.5, 0.5]))
            weighted['w_60_40'].append(weighted_rank_ret(day, [0.6, 0.4]))
            weighted['w_70_30'].append(weighted_rank_ret(day, [0.7, 0.3]))
            weighted['w_80_20'].append(weighted_rank_ret(day, [0.8, 0.2]))
            weighted['w_50_30_20'].append(weighted_rank_ret(day, [0.5, 0.3, 0.2]))
            weighted['w_40_30_30'].append(weighted_rank_ret(day, [0.4, 0.3, 0.3]))
            weighted['w_eq3'].append(weighted_rank_ret(day, [1/3, 1/3, 1/3]))

            rank_top1_code = day['rank_all_top1']['code']
            serial_code = day['serial_top1']['code']
            cls_codes = [x['code'] for x in day.get('cls_topn', [])]
            rank_codes = [x['code'] for x in day.get('rank_topn', [])[:20]]

            same_pick.append(int(rank_top1_code == serial_code))
            if rank_top1_code in cls_codes:
                rank_top1_in_cls.append(1)
                rank_top1_cls_rank.append(cls_codes.index(rank_top1_code) + 1)
            else:
                rank_top1_in_cls.append(0)
                rank_top1_cls_rank.append(np.nan)

            if serial_code in rank_codes:
                serial_in_rank_top20.append(1)
                serial_rank_rank.append(rank_codes.index(serial_code) + 1)
            else:
                serial_in_rank_top20.append(0)
                serial_rank_rank.append(np.nan)

        print(f"Loaded part={part_name}, days={len(dts)}, start={dts[0]}, end={dts[-1]}")
        print(f"same_pick_rate: {np.mean(same_pick):.6f}")
        print(f"rank_top1_in_cls_top20_rate: {np.mean(rank_top1_in_cls):.6f}")
        print(f"serial_top1_in_rank_top20_rate: {np.mean(serial_in_rank_top20):.6f}")
        print('rank_top1_cls_rank distribution:')
        print(pd.Series(rank_top1_cls_rank).value_counts(dropna=False).sort_index())
        print('serial_top1_rank_rank distribution:')
        print(pd.Series(serial_rank_rank).value_counts(dropna=False).sort_index())
        print()

        strategy_map = {
            'rank_top1': rank_top1,
            'serial_top1': serial_top1,
            'cls_top1': cls_top1,
            **{f'rank_eq{k}': v for k, v in rank_eq.items()},
            **weighted,
        }
        for strategy, arr in strategy_map.items():
            st = calc_stats(arr)
            out_rows.append({'part': part_name, 'strategy': strategy, **st})
            print_stats(f'{part_name} | {strategy}', st)

    out = pd.DataFrame(out_rows)
    print('\n' + '#' * 120)
    print('SUMMARY TABLE')
    print('#' * 120)
    print(out[['part','strategy','mean','median','win_rate','big_loss_5pct_rate','big_loss_8pct_rate','max_drawdown','sharpe','final_nav']].sort_values(['part','mean'], ascending=[True, False]).to_string(index=False))

    csv_path = path.with_suffix('.execution_scan.csv')
    out.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\nSaved csv: {csv_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('json_path', type=str)
    args = ap.parse_args()
    analyze(Path(args.json_path))
