"""去相关双榜单（aligned_v1 口径，dedup.py 同判据：|ρ|>0.7=高相关）。

贪心：每标签按指标（c2c→A_official；vwap→LGB）降序遍历候选，
与已选成员的"逐日截面 Spearman 跨日均值"最大|ρ|>0.7 → 跳过（高相关只留指标最优者），
取满 20 个为止。

候选池：ALIGNED_TOP2_*_flat.csv 每标签前 N（默认150，保证去相关后仍够 20 个）。
因子值：refs 候选经 FactorBuilder（从原始 factor_index.json/dir_adapt_specs 重建完整
params）；mined 候选经 parse_formula→Evaluator（节点级缓存）。

用法：python decorrelate_top.py [--top 150] [--keep 20] [--flat xxx_flat.csv]
产物：out/DECORR_TOP2_<ts>.md + _flat.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from evaluator import Evaluator, load_panel
from parser import parse_formula
from refs import FactorBuilder, dir_adapt_specs, uploaded_map

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'out'
PON = HERE.parent
REFS = PON / 'refs'
LIB_REPOS = {
    'directional': REFS / 'skill-quant-factor-directional-alpha' / 'factor_index.json',
    'risk-pattern': REFS / 'skill-quant-factor-risk-pattern-alpha' / 'factor_index.json',
    'volume-stat': REFS / 'skill-quant-factor-volume-stat-alpha' / 'factor_index.json',
}
START = '20210901'
END = '20260821'
CORR_THRESHOLD = 0.7


def load_all_specs() -> dict:
    by_id: dict = {}
    for lib, path in LIB_REPOS.items():
        for s in json.loads(path.read_text(encoding='utf-8')):
            by_id[str(s['factor_id'])] = s
    for s in dir_adapt_specs():
        by_id.setdefault(str(s['factor_id']), s)
    return by_id


def factor_full_values(lib: str, fid: str, formula_cn: str,
                       frame: pd.DataFrame, ev: Evaluator, builder,
                       specs_by_id: dict,
                       store_map: dict | None = None) -> np.ndarray | None:
    """四源统一求值分发（V4.3c 收敛点）：mined/custom/store/refs 各一分支。

    全仓 6 处手抄的收敛替代（build_top 扫描/cond/dom/面板、select_specialists
    斜率/面板导出）。返回与 frame 同长的 float64 数组，失败返回 None
    （调用方跳过；store 缺列同样跳过，不伪造）。
    store_map：{fid: 与 frame 对齐的全窗数组}（调用方批量预读，如
    build_top._store_aligned 或 select_specialists 的单次批量映射）。
    """
    try:
        if lib == 'mined':
            return np.asarray(ev.values(parse_formula(str(formula_cn))), dtype=float)
        if lib == 'custom':
            from custom_registry import evaluate_custom_factor as _ecf
            from custom_registry import load_all_custom_specs as _lcs
            _spec = _lcs().get(str(fid))
            if _spec is None:
                return None
            return np.asarray(_ecf(_spec, frame, ev), dtype=float)
        if lib == 'store':
            if not store_map or str(fid) not in store_map:
                return None
            return np.asarray(store_map[str(fid)], dtype=float)
        _spec = (specs_by_id or {}).get(str(fid))
        if _spec is None:
            return None
        return builder.get_factor(_spec).to_numpy(dtype=float)
    except Exception:
        return None


def month_end_spearman_matrix(vals: np.ndarray, month_group: np.ndarray) -> np.ndarray | None:
    """vals: (n_rows, n_cand) float32；month_group: 每行月标签。逐日 rank→相关→跨日均值。

    2026-09-24 修复：
    * 某列当日常数（零方差）→ 涉及该列的日度相关不计入（此前 sd==0 置 1.0 使常数因子
      与所有列相关=0，静默通过去重）；
    * 相关矩阵按"有效日数"逐对平均，无有效日的对为 NaN；
    * 无任何有效日截面时返回 None（此前返回全零矩阵，去重静默失效为 no-op）。

    [CORR-CALIBRATION 2026-10-02] group_key 语义由「月」改为「交易日」：调用方原先只把
    每月最后一个交易日传入（每月 1 个截面，ρ 估计噪声过大），现改为传入月内全部交易日，
    每日一个截面后跨日平均。函数名与 month_group 形参沿用旧名以保持兼容。
    """
    n_c = vals.shape[1]
    acc = np.zeros((n_c, n_c))
    valid = np.zeros((n_c, n_c), dtype=np.int32)
    cnt = 0
    for g in np.unique(month_group):
        idx = np.flatnonzero(month_group == g)
        block = vals[idx].astype(np.float64)
        # 2026-10-21：去掉 `np.isfinite(block).all(axis=1)` 的全列连坐过滤
        # （任一低覆盖因子（如 id_morning_keep）会把当日样本交集成片抽干，
        # 多列同时稀疏时直接让相关矩阵变 None）。改为逐列独立处理：
        # 有效样本<50 → good[j]=False；否则用当日该列有限值中位数填缺失，
        # 中位数行在秩空间≈0.5（中性），既不放大也不删除该列。
        n_rows = block.shape[0]
        if n_rows < 50:
            continue
        ranks = np.empty_like(block, dtype=np.float64)
        good = np.ones(n_c, dtype=bool)
        for j in range(n_c):
            col = block[:, j]
            fin = np.isfinite(col)
            if int(fin.sum()) < 50:
                good[j] = False
                ranks[:, j] = 0.0
                continue
            med = float(np.median(col[fin]))
            filled = np.where(fin, col, med)
            ranks[:, j] = pd.Series(filled).rank().to_numpy()
            if len(np.unique(filled)) <= 1:
                good[j] = False
        ranks -= ranks.mean(axis=0)
        sd = ranks.std(axis=0)
        sd[~good] = 1.0  # 坏列不参与除法（其相关已被逐对掩码剔除）
        sd[sd == 0] = 1.0
        ranks /= sd
        pair_ok = good[:, None] & good[None, :]
        contrib = ranks.T @ ranks / n_rows
        acc += np.where(pair_ok, contrib, 0.0)
        valid += pair_ok.astype(np.int32)
        cnt += 1
    if cnt == 0:
        return None
    corr = np.full((n_c, n_c), np.nan)
    has = valid > 0
    corr[has] = (acc / valid)[has]
    return corr


def pct(x) -> str:
    try:
        v = float(x)
        return f'{v * 100:.1f}%' if np.isfinite(v) else '-'
    except (TypeError, ValueError):
        return '-'


def fmt(x, spec='%.4f') -> str:
    try:
        v = float(x)
        return spec % v if np.isfinite(v) else '-'
    except (TypeError, ValueError):
        return '-'


def base_row(r, lab, rank, sort_key, max_rho=None) -> dict:
    return {'label': lab, 'rank': rank, 'lib': r['lib'], 'factor_id': r['factor_id'],
            'family': r.get('family', ''), 'name_cn': r['name_cn'],
            'formula_cn': r['formula_cn'], 'uploaded': r['uploaded'],
            'direction': r.get('direction'), 'metric': float(r[sort_key]),
            'sort_key': sort_key, 'A_official': r.get('A_official'), 'LGB': r.get('LGB'),
            'rank_ic': r.get('rank_ic'), 'rank_ic_ir': r.get('rank_ic_ir'),
            'win': r.get('win'), 'rank_ic_win': r.get('rank_ic_win'),
            'ic_ir': r.get('ic_ir'), 'ic_mean': r.get('ic_mean'), 'mono': r.get('mono'),
            'turnover': r.get('turnover'), 'coverage': r.get('coverage'),
            'net': r.get('net'), 'sharpe': r.get('sharpe'), 'mdd': r.get('mdd'),
            'max_rho_vs_kept': max_rho}


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='去相关双榜单')
    ap.add_argument('--top', type=int, default=150)
    ap.add_argument('--keep', type=int, default=20)
    ap.add_argument('--flat', default='')
    args = ap.parse_args()
    t0 = time.time()

    if args.flat:
        flat_path = Path(args.flat)
    else:
        files = sorted(OUT_DIR.glob('ALIGNED_TOP2_*_flat.csv'))
        if not files:
            raise SystemExit('找不到 ALIGNED_TOP2_*_flat.csv')
        flat_path = files[-1]
    df = pd.read_csv(flat_path)
    df['uploaded'] = df['uploaded'].fillna('')
    print(f'flat={flat_path.name} rows={len(df):,}', flush=True)

    specs_by_id = load_all_specs()
    frame = load_panel()
    ev = Evaluator(frame)
    builder = FactorBuilder(frame)
    uploaded = uploaded_map()

    # [CORR-CALIBRATION 2026-10-02] 相关性截面由「每月最后 1 个交易日」改为「月内全部交易日」。
    # 原口径每月仅 1 个截面，ρ 估计的月内标准误过大；改全日后每月约 21 个截面，
    # 标准误下降约 √21 倍，且与 dedup.py / pool_matrix.py / tier2_info_audit 口径统一。
    mask_d = ((frame['date'] >= pd.Timestamp(START))
              & (frame['date'] <= pd.Timestamp(END))).to_numpy()
    pos_me = np.flatnonzero(mask_d)
    day_group = frame['date'].to_numpy()[pos_me]
    n_days = int(pd.Series(day_group).nunique())
    print(f'逐日截面: {n_days} 个交易日 × {pos_me.size:,} 行', flush=True)

    rows = []
    for lab, sort_key in (('c2c', 'A_official'), ('vwap', 'LGB')):
        sub = df[df['label'] == lab].sort_values(sort_key, ascending=False).head(args.top)
        cols = []
        row_idx = []
        for i_row, (_, r) in enumerate(sub.iterrows()):
            if r['lib'] == 'mined':
                try:
                    node = parse_formula(str(r['formula_cn']))
                    v = np.asarray(ev.values(node), dtype=float)
                except Exception:
                    continue
            else:
                spec = specs_by_id.get(str(r['factor_id']))
                if spec is None:
                    continue
                try:
                    v = builder.get_factor(spec).to_numpy(dtype=float)
                except Exception:
                    continue
            cols.append(v[pos_me].astype(np.float32))
            row_idx.append(i_row)
        if not cols:
            print(f'[{lab}] 无候选值', flush=True)
            continue
        vals = np.column_stack(cols)
        print(f'[{lab}] 候选 {vals.shape[1]}（池 {len(sub)}，{len(sub) - len(cols)} 个求值失败），'
              '计算逐日截面相关矩阵 ...', flush=True)
        corr = month_end_spearman_matrix(vals, day_group)
        if corr is None:
            raise RuntimeError(
                f'[{lab}] correlation matrix is empty: {vals.shape[1]} candidates, '
                f'{len(np.unique(day_group))} day groups; refusing to skip dedup')
        kept: list[int] = []
        dropped = []
        for j, i_row in enumerate(row_idx):
            if not kept:
                kept.append(j)
                continue
            pair = np.abs(corr[j, kept])
            finite = pair[np.isfinite(pair)]
            # 相关不可评估（零方差/无有效月）→ 保守按高相关剔除，但显式记录 NaN
            mx = float(finite.max()) if finite.size else float('nan')
            if finite.size and mx <= CORR_THRESHOLD:
                kept.append(j)
            else:
                dropped.append((j, mx))
            if len(kept) >= args.keep:
                break
        print(f'[{lab}] 候选 {len(row_idx)} → 去相关保留 {len(kept)}，剔除 {len(dropped)}', flush=True)
        for rank, j in enumerate(kept, 1):
            rows.append(base_row(sub.iloc[row_idx[j]], lab, rank, sort_key))
        for j, mx in dropped:
            rows.append(base_row(sub.iloc[row_idx[j]], lab, -1, sort_key, max_rho=mx))
        print(f'[{lab}] done {time.time() - t0:.0f}s', flush=True)

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    md_path = OUT_DIR / f'DECORR_TOP2_{ts}.md'
    lines = ['# 去相关双榜单（aligned_v1；高相关判据 |ρ|>0.7，同轴只留指标最优）', '',
             f'生成：{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}  '
             f'候选池=每标签前 {args.top}（按 A_official / LGB 降序）；'
             '相关=逐日截面 Spearman 跨日均值（与 dedup.py 判据一致）。', '']
    for lab, title in (('c2c', '榜1-去相关：官方A top20（c2c_3d，上传决策）'),
                       ('vwap', '榜2-去相关：LGB 复合分 top20（vwap_3d，自用 LightGBM）')):
        sel = sorted([r for r in rows if r['label'] == lab and r['rank'] > 0],
                     key=lambda r: r['rank'])
        if lab == 'c2c':
            lines += [f'## {title}', '',
                      '| 排名 | 因子 | 来源 | 方向 | RankIC | ICIR(P) | IC胜率 | 换手/次 | 净超额 | 夏普 | 回撤 | A官方 | 已上传 |',
                      '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
            for r in sel:
                lines.append('| %d | %s (%s) %s | %s | %s | %+.4f | %+.3f | %s | %s | %s | %s | %s | %s | %s |' % (
                    r['rank'], r['factor_id'], r['family'], str(r['formula_cn'])[:70],
                    r['lib'], int(r['direction']) if pd.notna(r['direction']) else '-',
                    r['rank_ic'], r['ic_ir'], pct(r['win']), pct(r['turnover']),
                    pct(r['net']), fmt(r['sharpe']), pct(r['mdd']),
                    '%.5f' % float(r['A_official']), r['uploaded'] or ''))
        else:
            lines += [f'## {title}', '',
                      '| 排名 | 因子 | 来源 | 方向 | RankIC | RankICIR | rank胜率 | 单调性 | 换手/次 | 覆盖度 | LGB | (参考)A官方 | 已上传 |',
                      '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
            for r in sel:
                lines.append('| %d | %s (%s) %s | %s | %s | %+.4f | %+.3f | %s | %s | %s | %s | %.5f | %s | %s |' % (
                    r['rank'], r['factor_id'], r['family'], str(r['formula_cn'])[:70],
                    r['lib'], int(r['direction']) if pd.notna(r['direction']) else '-',
                    r['rank_ic'], r['rank_ic_ir'], pct(r['rank_ic_win']),
                    fmt(r['mono'], '%+.2f'), pct(r['turnover']), pct(r['coverage']),
                    float(r['LGB']), fmt(r.get('A_official'), '%.5f'), r['uploaded'] or ''))
        lines.append('')
    for lab in ('c2c', 'vwap'):
        kept_n = sum(1 for r in rows if r['label'] == lab and r['rank'] > 0)
        dropped = [r for r in rows if r['label'] == lab and r['rank'] < 0]
        fams: dict = {}
        for r in dropped:
            k = f"{r['lib']}:{(r['family'] or 'x')[:14]}"
            fams[k] = fams.get(k, 0) + 1
        top_fam = sorted(fams.items(), key=lambda x: -x[1])[:6]
        lines += [f'### {lab} 去重统计：候选 {kept_n + len(dropped)} → 保留 {kept_n}，'
                  f'高相关剔除 {len(dropped)}',
                  '被剔除最多来源: ' + (', '.join(f'{k}×{n}' for k, n in top_fam) or '无'), '']
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUT_DIR / f'DECORR_TOP2_{ts}_flat.csv', index=False,
                              encoding='utf-8-sig')
    print(f'saved {md_path}  total {time.time() - t0:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
