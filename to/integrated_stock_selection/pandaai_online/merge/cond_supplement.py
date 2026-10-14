"""条件IC补捞（通道C，2026-09-29）：对 aligned 双榜扫描候选（非入选者）批量算
栖息地/市况条件 RankIC，捞"条件显著、无条件一般"的漏网因子。

输入：out/ALIGNED_TOP2_*_flat.csv（默认最新）
范围：vwap 按 LGB 降序前 N + c2c 按 A_official 降序前 N（默认300），(lib,factor_id)去重
判定：cond_t_max = 全部tag×双标签上 |cond_*_rank_ic_tstat| 最大值（要求对应 ndays>=10）；
      旗标 = cond_t_max>=2.0 且 双标签无条件|rank_ic|最大值<0.05；按 cond_t_max 降序取前15
产物：out/cond_supplement_<ts>.csv + 控制台旗标清单
口径：与 build_top.cond_ic_table 完全一致（cond_ic.build_cond_prep/cond_ic_metrics，
      mined 公式全面板求值，refs 走 FactorBuilder）。
用法：python cond_supplement.py [--flat PATH] [--top 300]（merge 目录下运行）
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from cond_ic import build_cond_prep, cond_ic_metrics
from decorrelate_top import load_all_specs
from evaluator import Evaluator, load_panel
from fitness import DataContext
from parser import parse_formula
from refs import FactorBuilder

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'out'
COND_T_FLAG = 2.0
UNCOND_WEAK = 0.05
FLAG_QUOTA = 15


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='条件IC补捞（通道C）')
    ap.add_argument('--flat', default='', help='ALIGNED_TOP2_*_flat.csv，默认最新')
    ap.add_argument('--top', type=int, default=300, help='每标签扫描深度')
    ap.add_argument('--resume', default='',
                    help='已算出的 cond_supplement CSV，跳过其中已有 factor_id（断点续算）')
    ap.add_argument('--min-trace', type=float, default=0.0,
                    help='仅计算双标签 max|rank_ic|>=此值的因子（扩展轮过滤纯噪声）')
    args = ap.parse_args()
    t0 = time.time()

    if args.flat:
        flat_path = Path(args.flat)
    else:
        cands = sorted(OUT_DIR.glob('ALIGNED_TOP2_*_flat.csv'),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit('no ALIGNED_TOP2_*_flat.csv in out/')
        flat_path = cands[-1]
    df = pd.read_csv(flat_path, encoding='utf-8-sig')
    print(f'flat: {flat_path.name} rows={len(df)}', flush=True)

    v = df[df['label'] == 'vwap'].sort_values('LGB', ascending=False).head(args.top)
    c = df[df['label'] == 'c2c'].sort_values('A_official', ascending=False).head(args.top)
    uni = pd.concat([v, c]).drop_duplicates(subset=['lib', 'factor_id'])
    if args.resume:
        _done = pd.read_csv(args.resume, encoding='utf-8-sig')
        _before = len(uni)
        uni = uni[~uni['factor_id'].isin(set(_done['factor_id'].astype(str)))]
        print(f'resume: skip {(_before - len(uni))} computed -> {len(uni)} todo', flush=True)
    if args.min_trace > 0:
        _tr = df.groupby('factor_id')['rank_ic'].apply(
            lambda s: float(np.nanmax(np.abs(pd.to_numeric(
                s, errors='coerce').to_numpy(dtype=float)))))
        _before = len(uni)
        uni = uni[uni['factor_id'].map(_tr) >= args.min_trace]
        print(f'min-trace {args.min_trace}: {_before} -> {len(uni)} todo', flush=True)
    print(f'scan universe: vwap{len(v)} + c2c{len(c)} -> unique {len(uni)}', flush=True)

    frame = load_panel()
    ev = Evaluator(frame)
    builder = FactorBuilder(frame)
    specs_by_id = load_all_specs()
    prep = build_cond_prep(frame)
    ctxs = {lab: DataContext(frame=frame, label=lab, evaluator=ev)
            for lab in ('c2c', 'vwap')}

    rows = []
    n_skip = 0
    for i, (_, r) in enumerate(uni.iterrows()):
        lib, fid = r['lib'], str(r['factor_id'])
        try:
            if lib == 'mined':
                node = parse_formula(str(r['formula_cn']))
                f_full = np.asarray(ev.values(node), dtype=float)
            else:
                spec = specs_by_id.get(fid)
                if spec is None:
                    n_skip += 1
                    continue
                f_full = builder.get_factor(spec).to_numpy(dtype=float)
        except Exception:
            n_skip += 1
            continue
        rec = {'factor_id': fid, 'lib': lib, 'family': r.get('family', ''),
               'formula': ('' if lib != 'mined' else str(r.get('formula_cn', '')))}
        for lab, ctx in ctxs.items():
            try:
                per = cond_ic_metrics(ctx, f_full[ctx._pos], prep, name=fid)
            except Exception:
                continue
            for k, val in per.items():
                rec[f'{lab}_{k}'] = val
        rec['uncond_rank_ic_vwap'] = r.get('rank_ic') if r['label'] == 'vwap' else np.nan
        rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f'  [{i + 1}/{len(uni)}] {time.time() - t0:.0f}s', flush=True)

    out = pd.DataFrame(rows)
    tcols = [c for c in out.columns if c.endswith('_rank_ic_tstat')]
    ncol = [c for c in out.columns if c.endswith('_ndays')]
    # cond_t_max 仅统计样本充足的 tag（ndays>=10）
    tmax = np.zeros(len(out))
    for tc in tcols:
        tag = tc[:-len('_rank_ic_tstat')]
        nd = f'{tag}_ndays'
        t = pd.to_numeric(out[tc], errors='coerce').to_numpy(dtype=float)
        if nd in out.columns:
            nn = pd.to_numeric(out[nd], errors='coerce').to_numpy(dtype=float)
            t = np.where(nn >= 10, t, np.nan)
        tmax = np.fmax(tmax, np.abs(np.where(np.isfinite(t), t, 0.0)))
    out['cond_t_max'] = tmax
    # 无条件强度：双标签 rank_ic 绝对值最大（flat.csv 每行自带本标签 rank_ic）
    ur = out.groupby('factor_id')['uncond_rank_ic_vwap'].apply(
        lambda s: float(np.nanmax(np.abs(pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)))))
    out['uncond_abs_max'] = out['factor_id'].map(ur)
    out['flag'] = (out['cond_t_max'] >= COND_T_FLAG) & (out['uncond_abs_max'] < UNCOND_WEAK)
    out = out.sort_values('cond_t_max', ascending=False)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = OUT_DIR / f'cond_supplement_{ts}.csv'
    out.to_csv(p, index=False, encoding='utf-8-sig')
    flags = out[out['flag']].head(FLAG_QUOTA)
    print(f'done: computed={len(out)} skipped={n_skip} flagged={(out["flag"]).sum()} '
          f'total={time.time() - t0:.0f}s -> {p.name}', flush=True)
    print('--- channel-C flags (cond_t>=2 & uncond|rank_ic|<0.05) ---', flush=True)
    for _, r in flags.iterrows():
        print(f'  {r["factor_id"]} [{r["lib"]}/{r["family"]}] cond_t={r["cond_t_max"]:.2f} '
              f'uncond={r["uncond_abs_max"]:.4f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
