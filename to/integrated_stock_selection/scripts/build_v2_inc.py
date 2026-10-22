#!/usr/bin/env python3
"""Build an additive v2-inc matrix: frozen base manifest + quality/rho-passed new custom factors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MERGE = ROOT / 'pandaai_online' / 'merge'
if str(MERGE) not in sys.path:
    sys.path.insert(0, str(MERGE))

from decorrelate_top import month_end_spearman_matrix
from export_lgb_dataset import cs_mad_norm, cs_rank_centered

RAW_CUSTOM_IDS = frozenset({'ts_amt_exp_5_60_raw', 'ts_mkt_stock_amt_z60_raw'})


def _is_raw_candidate(fid: str) -> bool:
    return str(fid) in RAW_CUSTOM_IDS or str(fid).endswith('_raw')


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _candidate_ids(candidate_json: Path, base_features: set[str],
                   override: str) -> list[str]:
    if override:
        return [x.strip() for x in override.split(',') if x.strip()]
    data = _load_json(candidate_json)
    out: list[str] = []
    for item in data.get('base20') or []:
        fid = str(item.get('factor') or '')
        lib = str(item.get('lib') or '')
        if fid and lib == 'custom' and fid not in base_features and fid not in out:
            out.append(fid)
    return out


def _metrics(flat_path: Path, ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not flat_path.exists():
        return out
    df = pd.read_csv(flat_path, usecols=['factor_id', 'label', 'rank_ic', 'LGB'])
    df = df[df['label'] == 'vwap'].drop_duplicates('factor_id')
    for _, r in df[df['factor_id'].isin(ids)].iterrows():
        out[str(r['factor_id'])] = {'rank_ic': float(r['rank_ic']),
                                    'LGB': float(r['LGB'])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base-dir', default='results/merge_v2/lgb_ready',
                    help='old 84 matrix dir (lgb_features.parquet + features_manifest.json)')
    ap.add_argument('--base-parquet', default='',
                    help='override base parquet path')
    ap.add_argument('--base-manifest', default='',
                    help='override base manifest path')
    ap.add_argument('--candidate-panel', default='results/merge_v2/lgb_ready_v2/vwap_top20_factors.parquet',
                    help='new candidate factor value panel')
    ap.add_argument('--candidate-json', default='results/merge_v2/vwap_top20.json',
                    help='new vwap_top20.json (custom lib discovery)')
    ap.add_argument('--candidate-flat', default='pandaai_online/merge/out/ALIGNED_TOP2_20261022_015602_flat.csv',
                    help='new flat CSV for quality metrics')
    ap.add_argument('--candidate-ids', default='',
                    help='comma-separated override; default = custom IDs in candidate JSON not in base')
    ap.add_argument('--compute-missing', action='store_true',
                    help='compute custom candidates not present in candidate panel from custom_registry')
    ap.add_argument('--data-cut', default='',
                    help='data cut for --compute-missing; default from base manifest')
    ap.add_argument('--out-dir', default='results/merge_v2/lgb_ready_v2inc')
    ap.add_argument('--max-rho', type=float, default=0.70)
    ap.add_argument('--min-abs-rankic', type=float, default=0.012)
    ap.add_argument('--min-lgb', type=float, default=0.0015)
    ap.add_argument('--cs-norm', default='mad', choices=('mad', 'rank', 'none'),
                    help='normalization for non-raw candidates (default mad = cs_mad_norm)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    base_parquet = Path(args.base_parquet) if args.base_parquet else base_dir / 'lgb_features.parquet'
    base_manifest = Path(args.base_manifest) if args.base_manifest else base_dir / 'features_manifest.json'
    candidate_panel = Path(args.candidate_panel)
    candidate_json = Path(args.candidate_json)
    candidate_flat = Path(args.candidate_flat)
    out_dir = Path(args.out_dir)

    manifest = _load_json(base_manifest)
    base_features = [str(x) for x in (manifest.get('features_all') or [])]
    base = pd.read_parquet(base_parquet)
    missing_base = [x for x in base_features if x not in base.columns]
    if missing_base:
        raise SystemExit(f'base manifest columns missing in parquet: {missing_base[:10]}')
    base['stock_code'] = base['stock_code'].astype(str)
    base['date'] = pd.to_datetime(base['date'])
    base = base[['stock_code', 'date'] + base_features].copy()

    ids = _candidate_ids(candidate_json, set(base_features), args.candidate_ids)
    if not ids:
        raise SystemExit('no new custom candidate IDs discovered')
    panel = pd.read_parquet(candidate_panel)
    panel['stock_code'] = panel['stock_code'].astype(str)
    panel['date'] = pd.to_datetime(panel['date'])
    missing_ids = [x for x in ids if x not in panel.columns]
    if missing_ids:
        if not args.compute_missing:
            raise SystemExit(f'candidate IDs missing from panel: {missing_ids}; '
                             f'pass --compute-missing or add them to the panel')
        from custom_registry import evaluate_custom_factor, load_all_custom_specs
        from evaluator import Evaluator, load_panel
        cut = args.data_cut or str(manifest.get('data_cut') or '2025-12-31')
        frame = load_panel(data_cut=cut)
        ev = Evaluator(frame)
        specs = load_all_custom_specs()
        p = frame[['stock_code', 'date']].copy()
        p['stock_code'] = p['stock_code'].astype(str)
        p['date'] = pd.to_datetime(p['date'])
        for fid in missing_ids:
            spec = specs.get(fid)
            if spec is None:
                continue
            try:
                p[fid] = np.asarray(evaluate_custom_factor(spec, frame, ev), dtype=float)
            except Exception:
                continue
        panel = panel.merge(p, on=['stock_code', 'date'], how='outer')
    ids = [x for x in ids if x in panel.columns]
    if not ids:
        raise SystemExit('candidate IDs not present in candidate panel')
    cand = panel[['stock_code', 'date'] + ids].copy()
    merged = base.merge(cand, on=['stock_code', 'date'], how='inner')
    if merged.empty:
        raise SystemExit('base/candidate merge produced zero rows')

    non_raw_ids = [fid for fid in ids if not _is_raw_candidate(fid)]
    if non_raw_ids and args.cs_norm != 'none':
        if args.cs_norm == 'rank':
            norm = cs_rank_centered(merged, non_raw_ids)
        else:
            norm = cs_mad_norm(merged, non_raw_ids)
        for fid in non_raw_ids:
            merged[fid] = norm[fid].to_numpy(dtype=float)

    metrics = _metrics(candidate_flat, ids)
    accepted: list[str] = []
    report: dict = {'base_features': len(base_features), 'candidates': ids,
                    'max_rho': args.max_rho, 'min_abs_rankic': args.min_abs_rankic,
                    'min_lgb': args.min_lgb, 'accepted': [], 'rejected': []}
    for fid in ids:
        m = metrics.get(fid)
        if m is None:
            report['rejected'].append({'factor': fid, 'reason': 'missing_metrics'})
            continue
        if abs(m['rank_ic']) < args.min_abs_rankic or m['LGB'] < args.min_lgb:
            report['rejected'].append({'factor': fid, 'reason': 'quality_gate',
                                       'rank_ic': m['rank_ic'], 'LGB': m['LGB']})
            continue
        cols = base_features + [fid]
        vals = merged[cols].to_numpy(dtype=np.float32)
        corr = month_end_spearman_matrix(vals, merged['date'].to_numpy())
        if corr is None:
            report['rejected'].append({'factor': fid, 'reason': 'corr_none'})
            continue
        rho = np.abs(corr[-1, :-1])
        if not np.isfinite(rho).any():
            report['rejected'].append({'factor': fid, 'reason': 'corr_all_nan'})
            continue
        j = int(np.nanargmax(rho))
        max_rho = float(rho[j])
        base_fid = base_features[j]
        item = {'factor': fid, 'max_rho': round(max_rho, 6),
                'max_rho_vs': base_fid, 'rank_ic': m['rank_ic'], 'LGB': m['LGB']}
        if max_rho <= args.max_rho:
            accepted.append(fid)
            report['accepted'].append(item)
        else:
            item['reason'] = 'rho_challenge'
            report['rejected'].append(item)
        del vals, corr, rho

    new_features = base_features + accepted
    new_manifest = dict(manifest)
    new_manifest['features_all'] = new_features
    new_manifest['n_total_features'] = len(new_features)
    by = {k: list(v) for k, v in (manifest.get('by_class') or {}).items()}
    by.setdefault('class1_stock_alpha', [])
    for fid in accepted:
        if fid not in by['class1_stock_alpha']:
            by['class1_stock_alpha'].append(fid)
    new_manifest['by_class'] = by
    skip = list(manifest.get('skip_daily_cs_norm_cols') or [])
    for fid in accepted:
        if fid not in skip:
            skip.append(fid)
    new_manifest['skip_daily_cs_norm_cols'] = skip
    bounded = list(manifest.get('bounded_raw_cols') or [])
    for fid in accepted:
        if _is_raw_candidate(fid) and fid not in bounded:
            bounded.append(fid)
    new_manifest['bounded_raw_cols'] = bounded
    new_manifest['cs_norm'] = args.cs_norm

    report['cs_norm'] = args.cs_norm
    report['raw_candidates'] = [x for x in accepted if _is_raw_candidate(x)]
    report['normalized_candidates'] = [x for x in accepted if not _is_raw_candidate(x)]
    report['n_base'] = len(base_features)
    report['n_accepted'] = len(accepted)
    report['n_total'] = len(new_features)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    out = base.copy()
    for fid in accepted:
        out[fid] = merged[fid].to_numpy()
    out.to_parquet(out_dir / 'lgb_features.parquet', index=False)
    (out_dir / 'features_manifest.json').write_text(
        json.dumps(new_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'v2inc_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {out_dir} ({len(new_features)} features)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
