"""log 政策 A/B 实验（0 算力）：水平类字段 raw vs log 的 SiA/ICIR/胜率对比。

专业惯例对照：市值一律 log；换手/成交额/成交量水平普遍 log 或 rank/zscore。
判定以 SiA 为主（含 Pearson ICIR/胜率分量）；RankIC 理论上应基本不变（单调变换免疫），
若 RankIC 明显变化说明窗口/NaN 处理引入差异，需检查。

用法：
  python experiments_log_ab.py                 # 快速样本（800股）
  python experiments_log_ab.py --full          # 全量面板
  python experiments_log_ab.py --windows 20,60,120
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import _paths  # noqa: F401
from evaluator import load_panel
from fitness import DataContext, build_fast_context, tier1
from grammar import Node, field, ts


def _mean_log(name: str, w: int, use_log: bool) -> Node:
    base: Node = field(name)
    if use_log:
        base = Node('LOG', (base,))
    return ts('TS_MEAN', base, w)


def build_cases(windows: tuple) -> dict:
    cases = {}
    for w in windows:
        cases[f'TURNOVER_mean{w}_raw'] = _mean_log('TURNOVER', w, False)
        cases[f'TURNOVER_mean{w}_log'] = _mean_log('TURNOVER', w, True)
        cases[f'VOLUME_mean{w}_raw'] = _mean_log('VOLUME', w, False)
        cases[f'VOLUME_mean{w}_log'] = _mean_log('VOLUME', w, True)
    cases['MARKETCAP_raw'] = field('MARKET_CAP')
    cases['MARKETCAP_log'] = Node('LOG', (field('MARKET_CAP'),))
    cases['AMOUNT_mean60_raw'] = _mean_log('AMOUNT', 60, False)
    cases['AMOUNT_mean60_log'] = _mean_log('AMOUNT', 60, True)
    return cases


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='log 政策 A/B（0 算力）')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--stocks', type=int, default=800)
    ap.add_argument('--windows', default='20,60,120')
    args = ap.parse_args()
    windows = tuple(int(x) for x in args.windows.split(',') if x.strip())
    frame = load_panel()
    ctx = DataContext(frame=frame) if args.full else build_fast_context(
        frame, n_stocks=args.stocks)
    print(f'ctx rows={ctx.n_rows:,} sections={len(ctx.ud)} label={ctx.label_name}')
    cases = build_cases(windows)
    out = {}
    for name, node in cases.items():
        m = ctx.screen(node, name=name)
        out[name] = tier1(m)
    header = f'{"case":24s} {"dir":>3s} {"sia":>8s} {"rank_ic":>8s} {"ic":>8s} ' \
             f'{"ic_ir":>7s} {"win":>5s} {"turn":>6s} {"net":>7s}'
    print(header)
    for name in cases:
        t = out[name]
        print(f'{name:24s} {t["sia"]:8.4f} {(t["rank_ic"] or 0):8.4f} '
              f'{(t["ic"] or 0):8.4f} {(t["ic_ir"] or 0):7.3f} {(t["win"] or 0):5.2f} '
              f'{((t["turnover"] or -1) * 100):6.1f} {(t["net"] or -9):+7.2f}')
    print('\npair deltas（log - raw）:')
    for base in sorted({k.rsplit("_", 1)[0] for k in cases}):
        raw, lg = f'{base}_raw', f'{base}_log'
        if raw in out and lg in out:
            dr = out[lg]['rank_ic'] - out[raw]['rank_ic']
            ds = out[lg]['sia'] - out[raw]['sia']
            print(f'  {base:22s} Δsia={ds:+.4f} Δrank_ic={dr:+.4f} '
                  f'(sia: {out[raw]["sia"]:.4f} -> {out[lg]["sia"]:.4f})')


if __name__ == '__main__':
    main()
