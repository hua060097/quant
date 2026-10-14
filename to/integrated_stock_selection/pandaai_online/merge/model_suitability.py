"""模型适配筛选（2026-09-27）：在 top 因子中找最适合进 CURATED 训练池的候选。

与 build_top（竞赛口径榜单）分离：本脚本是纯消费者，只读最新 flat +
value_cache + factors.parquet 池列，不重跑构建，可对历史 flat 重算。

判定链（文献依据见 nonlinear.py 与报告）：
  1. twin 否决：候选 vs CURATED 全员，逐日截面 Spearman 跨日均值 max|ρ|>thr → 出局
     （build_top 正交规则在现池上的延伸；mRMR 的 redundancy 项）
  2. CMIM 否决：min over top-k 池代表元 I(标签;候选|代表元) / I(标签;候选)
     < ratio → 出局（单个池成员解释掉 90%+ 即无条件增量；Fleuret CMIM）
  3. JMI 排序：幸存者按 Σ_topk I((候选,代表元);标签) 降序（互补性；Brown 2012）
  4. 成本标注：net/turnover 只做标注与合取参考，不单独否决（文献：单因子净值
     不决定去留，决定的是组合边际；见 Novy-Marx/Velikov、TCA 2024）
最终进 CURATED 仍需模型 add-one（≥2 窗口），本脚本只产出候选短名单。

用法：
  python model_suitability.py [--flat PATH] [--top 150] [--twin 0.7]
                             [--cmim-ratio 0.1] [--topk 5] [--limit 0]
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import _paths  # noqa: F401
from decorrelate_top import load_all_specs
from evaluator import Evaluator, load_panel
from fitness import START, END as FIT_END, build_label_series
from grammar import expression_hash
from nonlinear import cond_mi_monthly, joint_mi_monthly
from parser import parse_formula
from refs import FactorBuilder

import build_top as bt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'out'
CURATED_SRC = HERE.parent.parent / ('main_alpha_二分类_原因子_with_test_metrics_妖股_copy_原始_'
                                     '20260720_regression_rankgauss_binary_labelround.py')
FACTORS_PARQUET = HERE.parent.parent / 'factor_lab_data' / 'factors.parquet'


def curated_names() -> list[str]:
    """AST 解析 CURATED（按 skill 要求不 import 主文件）。"""
    src = io.open(CURATED_SRC, encoding='utf-8').read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'CURATED_MODEL_FACTOR_GROUPS':
                    out, seen = [], set()
                    for _k, _v in zip(node.value.keys, node.value.values):
                        for e in _v.elts:
                            if isinstance(e, ast.Constant) and e.value not in seen:
                                seen.add(e.value)
                                out.append(e.value)
                    return out
    raise RuntimeError('CURATED_MODEL_FACTOR_GROUPS not found')


def latest_flat(explicit: str = '') -> Path:
    if explicit:
        return Path(explicit)
    files = sorted(OUT_DIR.glob('ALIGNED_TOP2_*_flat.csv'))
    if not files:
        raise FileNotFoundError('no ALIGNED_TOP2 flat found')
    return files[-1]


def monthly_spearman(x: np.ndarray, Y: np.ndarray, months: np.ndarray) -> np.ndarray:
    """x 与 Y 每列的月均截面 Spearman（pairwise 完备行）。"""
    x = np.asarray(x, dtype=float)
    out = np.full(Y.shape[1], np.nan)
    ug = np.unique(months)
    acc = np.zeros(Y.shape[1])
    cnt = np.zeros(Y.shape[1])
    for g in ug:
        idx = np.flatnonzero(months == g)
        xv = x[idx]
        okx = np.isfinite(xv)
        if okx.sum() < 50:
            continue
        rx = rankdata(xv[okx])
        for j in range(Y.shape[1]):
            yv = np.asarray(Y[idx, j], dtype=float)
            ok = okx & np.isfinite(yv)
            if ok.sum() < 50:
                continue
            ry = rankdata(yv[ok])
            rxx = rx[ok[okx]]
            rxx = rxx - rxx.mean()
            ryy = ry - ry.mean()
            den = float(np.sqrt((rxx * rxx).sum() * (ryy * ryy).sum()))
            if den > 0:
                acc[j] += float((rxx * ryy).sum() / den)
                cnt[j] += 1
    with np.errstate(invalid='ignore'):
        out = acc / np.maximum(cnt, 1)
    out[cnt == 0] = np.nan
    return out


def main(argv=None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='模型适配筛选：top因子→CURATED候选短名单')
    ap.add_argument('--flat', default='')
    ap.add_argument('--top', type=int, default=150)
    ap.add_argument('--twin', type=float, default=0.7)
    ap.add_argument('--cmim-ratio', type=float, default=0.1)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0,
                    help='冒烟：只处理前 K 个候选（0=全量）')
    args = ap.parse_args(argv)
    t0 = time.time()

    flat_path = latest_flat(args.flat)
    print(f'flat: {flat_path}', flush=True)
    flat = pd.read_csv(flat_path)
    v = flat[flat.label == 'vwap'].sort_values('LGB', ascending=False).head(args.top)
    v = v.reset_index(drop=True)
    if args.limit:
        v = v.head(args.limit).reset_index(drop=True)
    print(f'candidates: {len(v)}', flush=True)

    pool_names = curated_names()
    print(f'CURATED pool: {len(pool_names)}', flush=True)

    frame = load_panel()
    ev = Evaluator(frame)
    builder = FactorBuilder(frame)
    mask_d = ((frame['date'] >= pd.Timestamp(START))
              & (frame['date'] <= pd.Timestamp(FIT_END))).to_numpy()
    # [CORR-CALIBRATION 2026-10-02] twin 否决的相关性截面由「每月最后 1 个交易日」
    # 改为「月内全部交易日」，与 build_top / decorrelate_top / dedup 同步。
    pos_me = np.flatnonzero(mask_d)
    day_group = frame['date'].to_numpy()[pos_me]
    y_me = build_label_series(frame, 'vwap', 3).to_numpy(dtype=float)[pos_me]
    cache_dir = bt._value_cache_dir(frame, pos_me)
    specs = load_all_specs()

    # 候选值：缓存命中，否则现算并回填缓存
    cand_vals, cand_ok = {}, []
    n_eval = 0
    for j in range(len(v)):
        r = v.loc[j]
        try:
            if r.lib == 'mined':
                key = expression_hash(parse_formula(str(r.formula_cn)))
            else:
                key = bt._ref_spec_key(specs[str(r.factor_id)])
            arr = bt._cache_load(cache_dir, key, len(pos_me))
            if arr is None:
                if r.lib == 'mined':
                    arr = np.asarray(ev.values(parse_formula(str(r.formula_cn))),
                                     dtype=float)[pos_me].astype(np.float32)
                else:
                    arr = builder.get_factor(specs[str(r.factor_id)]) \
                        .to_numpy(dtype=float)[pos_me].astype(np.float32)
                bt._cache_save(cache_dir, key, arr)
                n_eval += 1
            cand_vals[j] = np.asarray(arr, dtype=float)
            cand_ok.append(j)
        except Exception as exc:  # noqa: BLE001
            print(f'  skip {r.factor_id}: {exc!r}'[:120], flush=True)
    print(f'candidate values ready: {len(cand_ok)}/{len(v)} '
          f'(fresh eval {n_eval}) {time.time() - t0:.0f}s', flush=True)

    # 池值：factors.parquet 按逐日截面日期过滤后读（只读一次）
    me_dates = pd.Series(frame['date'].to_numpy()[pos_me])
    me_keys = set(zip(frame['stock_code'].astype(str).to_numpy()[pos_me],
                      me_dates.dt.strftime('%Y%m%d').to_numpy()))
    _pf = pd.read_parquet(FACTORS_PARQUET,
                          columns=['stock_code', 'date'] + pool_names)
    _pf['stock_code'] = _pf['stock_code'].astype(str)
    _pf['d8'] = pd.to_datetime(_pf['date']).dt.strftime('%Y%m%d')
    _pf = _pf[_pf.set_index(['stock_code', 'd8']).index.isin(me_keys)].copy()
    print(f'pool rows after month-end filter: {len(_pf):,} {time.time() - t0:.0f}s',
          flush=True)
    me_frame = pd.DataFrame({
        'stock_code': frame['stock_code'].astype(str).to_numpy()[pos_me],
        'd8': me_dates.dt.strftime('%Y%m%d').to_numpy(),
        'month': day_group,
        'y': y_me,
    })
    for j in cand_ok:
        me_frame[f'__c{j}'] = cand_vals[j]
    m = me_frame.merge(_pf, on=['stock_code', 'd8'], how='inner')
    print(f'common rows: {len(m):,} months: {m.month.nunique()} {time.time() - t0:.0f}s',
          flush=True)
    months = m['month'].to_numpy()
    yv = m['y'].to_numpy(dtype=float)

    rows = []
    for j in cand_ok:
        r = v.loc[j]
        f = m[f'__c{j}'].to_numpy(dtype=float)
        P = m[pool_names].to_numpy(dtype=float)
        rho = monthly_spearman(f, P, months)
        # 先滤掉 NaN 再取 topk：NaN 占位排序会把无效列顶到前面（2026-09-27 实测
        # 广播型池因子每日期内常数 → den=0 → NaN，必须后置）
        finite_idx = [k for k in range(len(rho)) if np.isfinite(rho[k])]
        order = sorted(finite_idx, key=lambda k: -abs(float(rho[k])))
        top = [(pool_names[k], float(rho[k])) for k in order[:args.topk]]
        max_rho = top[0][1] if top else float('nan')
        max_who = top[0][0] if top else ''
        # CMIM 否决 + JMI：只对 topk 代表元算
        cmi_min, cmi_by, jmi_sum = float('nan'), '', 0.0
        mi_plain = float('nan')
        if top:
            from nonlinear import mi_monthly
            mi_plain = mi_monthly(f, yv, months)['mi']
            best = 1.0
            for pname, _ in top:
                z = m[pname].to_numpy(dtype=float)
                from nonlinear import cond_mi_monthly, joint_mi_monthly
                d = cond_mi_monthly(f, yv, z, months)
                j = joint_mi_monthly(f, z, yv, months)
                c = d.get('cmi', float('nan'))
                if np.isfinite(c) and c < best:
                    best, cmi_by = c, pname
                if np.isfinite(j.get('jmi')):
                    jmi_sum += j['jmi']
            cmi_min = best if best <= 1.0 else float('nan')
        ratio = (cmi_min / mi_plain) if (np.isfinite(cmi_min) and np.isfinite(mi_plain)
                                         and mi_plain > 0) else float('nan')
        twin_out = bool(np.isfinite(max_rho) and abs(max_rho) > args.twin)
        cmim_out = bool(np.isfinite(ratio) and ratio < args.cmim_ratio)
        rows.append({
            'factor_id': r.factor_id, 'lib': r.lib, 'LGB': float(r.LGB),
            'mono': r.mono, 'turnover': r.turnover, 'net': r.get('net'),
            'mi': float(r.get('mi')) if r.get('mi') == r.get('mi') else float('nan'),
            'twin_max_rho': float(max_rho) if np.isfinite(max_rho) else float('nan'),
            'twin_who': max_who,
            'twin_veto': twin_out,
            'cmim_min': float(cmi_min) if np.isfinite(cmi_min) else float('nan'),
            'cmim_by': cmi_by,
            'cmim_ratio': float(ratio) if np.isfinite(ratio) else float('nan'),
            'cmim_veto': cmim_out,
            'jmi': float(jmi_sum),
            'verdict': 'OUT-twin' if twin_out else ('OUT-cmim' if cmim_out else 'KEEP'),
        })
        if (len(rows) % 25 == 0) or len(rows) == len(cand_ok):
            print(f'  scored {len(rows)}/{len(cand_ok)} {time.time() - t0:.0f}s', flush=True)

    out = pd.DataFrame(rows)
    keep = out[out.verdict == 'KEEP'].sort_values('jmi', ascending=False).reset_index(drop=True)
    # 短名单内部去重（2026-09-27）：候选两两 |ρ|>thr 只留 JMI 高者。
    # 教训：此前短名单同时 chứa 29fa/18e2、a030/9f77 等多组精确否定对
    # （|ρ|=1.0），只对池去重不够，短名单内部也必须去重。
    if len(keep) > 1:
        _kid = keep.factor_id.astype(str).tolist()
        _j2fid = {j: str(v.loc[j, 'factor_id']) for j in cand_ok}
        _fid2val = {_j2fid[j]: cand_vals[j] for j in cand_ok
                    if _j2fid[j] in set(_kid)}
        _drop = set()
        for _a in range(len(_kid)):
            if _kid[_a] in _drop or _kid[_a] not in _fid2val:
                continue
            for _b in range(_a + 1, len(_kid)):
                if _kid[_b] in _drop or _kid[_b] not in _fid2val:
                    continue
                _r = monthly_spearman(
                    _fid2val[_kid[_a]],
                    np.asarray(_fid2val[_kid[_b]], dtype=float).reshape(-1, 1),
                    months)
                if np.isfinite(_r[0]) and abs(float(_r[0])) > args.twin:
                    _drop.add(_kid[_b])
        if _drop:
            keep = keep[~keep.factor_id.astype(str).isin(_drop)] \
                .reset_index(drop=True)
            print(f'短名单内部去重：{len(_drop)} 个出局 {sorted(_drop)}', flush=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out.to_csv(OUT_DIR / f'MODEL_SUITABILITY_{ts}_full.csv', index=False,
               encoding='utf-8-sig')
    keep.to_csv(OUT_DIR / f'MODEL_SUITABILITY_{ts}_shortlist.csv', index=False,
                encoding='utf-8-sig')
    with open(OUT_DIR / f'MODEL_SUITABILITY_{ts}.json', 'w',
              encoding='utf-8') as fh:
        json.dump({'generated_at': datetime.now().isoformat(timespec='seconds'),
                   'flat': str(flat_path), 'top': args.top,
                   'twin_thr': args.twin, 'cmim_ratio_thr': args.cmim_ratio,
                   'topk': args.topk, 'elapsed_s': round(time.time() - t0, 1),
                   'veto_counts': out.verdict.value_counts().to_dict()},
                  fh, ensure_ascii=False)
    lines = ['# 模型适配筛选（top因子 → CURATED 候选）', '',
             f'生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  '
             f'flat={flat_path.name} 候选={len(v)}（vwap LGB top{args.top}） '
             f'池={len(pool_names)} 因子',
             f'规则：twin 否决（vs 池 max|ρ|>{args.twin}）→ CMIM 否决'
             f'（min 条件MI/无条件MI<{args.cmim_ratio}）→ JMI 排序；net/turnover 仅标注。',
             f'否决统计：{out.verdict.value_counts().to_dict()}', '',
             '## 短名单（JMI 降序；进 CURATED 仍需 add-one ≥2 窗口验证）', '',
             '| 排名 | 因子 | 来源 | LGB | mono | JMI | twin_max | CMIM_ratio | 换手 | 净超额 |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for i, r in keep.head(20).iterrows():
        lines.append('| %d | %s | %s | %.5f | %s | %.4f | %.3f(%s) | %.3f | %s | %s |' % (
            i + 1, r.factor_id, r.lib, r.LGB, r.mono, r.jmi,
            r.twin_max_rho, str(r.twin_who)[:24], r.cmim_ratio,
            r.turnover, r.net))
    lines += ['',
              '> 注：本名单是 add-one 实验的候选输入，不是进池结论；'
              '单因子净值为负 + 低冗余者保留送组合测试（文献：组合边际决定去留）。']
    md_path = OUT_DIR / f'MODEL_SUITABILITY_{ts}.md'
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'saved {md_path}  total {time.time() - t0:.0f}s', flush=True)
    print(f'verdicts: {out.verdict.value_counts().to_dict()}', flush=True)
    if len(keep):
        print('shortlist top5:', ', '.join(keep.head(5).factor_id.astype(str).tolist()),
              flush=True)
    else:
        print('shortlist EMPTY: no candidate survives twin+CMIM vetoes', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
