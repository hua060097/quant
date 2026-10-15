# -*- coding: utf-8 -*-
"""LightGBM 训练特征矩阵与 FeatureContract 一键组装导出器（V4.1 Stage6，merge内部版）。

用法：
    python export_lgb_dataset.py --max-alpha 80 --max-interact 25 --out-dir ../../results/merge_v2/lgb_ready

红线（PLAN D7/D8，§21）：
- 纯市场广播（1xT）永远为空，不入模；
- 纯行业广播（SxT）默认不入模（--max-sector 默认 0；显式 >0 需Owner豁免，见PLAN §21）；
- Class4 交互列与行业列写入前不得再做单日截面标准化（见 skip_daily_cs_norm_cols）。
- Stage15桥接（factors_with_merge_cand.parquet / bridge_stage15.py）本期不实施，
  详见 refactor_doc.md §六待定问题。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import windows as W
from evaluator import ensure_local_fields, load_panel

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / 'results' / 'merge_v2'

#: Class 2 个股栖息地/风险状态固定底座（ensure_local_fields 原4字段 + 扩展2字段按需）
HABITAT_COLS = ('BETA60', 'BETA20', 'IVOL20', 'AMIHUD20', 'TURNOVER', 'MARKET_CAP')


def cs_rank_centered(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """对 Class 1 个股 Alpha 做逐日截面居中百分位秩 [-0.5, 0.5] 标准化。"""
    out = df[['stock_code', 'date']].copy()
    g = df.groupby('date', sort=False)
    for c in cols:
        out[c] = (g[c].rank(pct=True) - 0.5).astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-cut', default=W.DEFAULT_DATA_CUT)
    ap.add_argument('--max-alpha', type=int, default=80, help='Class 1 正交个股Alpha上限')
    ap.add_argument('--max-interact', type=int, default=25, help='Class 4 门控交互因子上限')
    ap.add_argument('--max-sector', type=int, default=0,
                    help='Class 3 行业广播状态上限（默认0=不入模；>0仅诊断，需单独审批）')
    ap.add_argument('--alpha-panel', default=str(RESULTS / 'vwap_top20_factors.parquet'),
                    help='Class 1 候选面板（默认读build_top动态面板；缺失则空跑并警告）')
    ap.add_argument('--out-dir', default=str(RESULTS / 'lgb_ready'))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载底层面板与 Class 2 个股栖息地底座（只读，不改历史指纹语义）
    panel = load_panel(data_cut=args.data_cut)
    ensure_local_fields(panel)
    try:
        from evaluator import ensure_extended_fields
        ensure_extended_fields(panel)
    except Exception as e:
        print(f'warning: extended fields unavailable {str(e)[:120]}')
    avail_hab = [c for c in HABITAT_COLS if c in panel.columns]
    base_df = panel[['stock_code', 'date'] + avail_hab].copy()
    if 'MARKET_CAP' in base_df.columns:
        base_df['LOG_MV'] = np.log(np.maximum(
            base_df.pop('MARKET_CAP').to_numpy(dtype=float), 1.0)).astype(np.float32)
        habitat_features = [c for c in avail_hab if c != 'MARKET_CAP'] + ['LOG_MV']
    else:
        habitat_features = list(avail_hab)

    # 2. 合并 Class 1 正交个股 Alpha（缺失则空跑警告，不伪造）
    # V4.3c 席位配额：habitat_seats.json 席位列优先强制收录（专属配额，不占 max_alpha）。
    alpha_features = []
    seat_cols = []
    _seat_path = RESULTS / 'habitat_seats.json'
    if _seat_path.exists():
        try:
            seat_cols = [str(x) for x in
                         json.load(open(_seat_path, encoding='utf-8')).get('seats', [])]
        except Exception:
            seat_cols = []
    alpha_path = Path(args.alpha_panel)
    if alpha_path.exists():
        adf = pd.read_parquet(alpha_path)
        adf['date'] = pd.to_datetime(adf['date'])
        adf = adf[adf['date'] <= pd.Timestamp(args.data_cut)]
        _all = [c for c in adf.columns if c not in ('stock_code', 'date')]
        _seats = [c for c in seat_cols if c in _all]
        _rest = [c for c in _all if c not in set(_seats)][:args.max_alpha]
        acols = _seats + _rest
        adf_norm = cs_rank_centered(adf, acols)
        base_df = base_df.merge(adf_norm, on=['stock_code', 'date'], how='left')
        alpha_features = acols
        if _seats:
            print(f'Class1 席位强制收录 {len(_seats)} 列', flush=True)
    else:
        print(f'warning: alpha panel missing {alpha_path}（Class1为空跑，不伪造列）')

    # 3. Class 3 行业状态（默认关闭；显式开启仅诊断）
    sector_features = []
    if args.max_sector and args.max_sector > 0:
        print('warning: --max-sector>0 需Owner豁免（PLAN §21 D8 clarification）；'
              '纯市场广播恒为0')
        sec_path = RESULTS / 'gates_search' / 'direct_sector_states.parquet'
        if not sec_path.exists():
            sec_path = RESULTS / 'gates_search' / 'approved_sector_states.parquet'
        if sec_path.exists():
            sdf = pd.read_parquet(sec_path)
            sdf['date'] = pd.to_datetime(sdf['date'])
            sdf = sdf[sdf['date'] <= pd.Timestamp(args.data_cut)]
            scols = [c for c in sdf.columns
                     if c not in ('stock_code', 'date', 'sector')][:args.max_sector]
            if 'sector' in sdf.columns and 'stock_code' not in sdf.columns:
                from field_catalog import sw2_mapping
                base_df['_sec'] = base_df['stock_code'].map(sw2_mapping())
                base_df = base_df.merge(sdf[['date', 'sector'] + scols],
                                        left_on=['date', '_sec'],
                                        right_on=['date', 'sector'], how='left')
                base_df.drop(columns=['_sec', 'sector'], errors='ignore', inplace=True)
            else:
                base_df = base_df.merge(sdf[['stock_code', 'date'] + scols],
                                        on=['stock_code', 'date'], how='left')
            sector_features = scols

    # 4. 合并 Class 4 显式门控交互因子（严禁再做单日横截面 Z-score/Rank!）
    inter_path = RESULTS / 'approved_interactions.parquet'
    interact_features = []
    if inter_path.exists():
        idf = pd.read_parquet(inter_path)
        idf['date'] = pd.to_datetime(idf['date'])
        idf = idf[idf['date'] <= pd.Timestamp(args.data_cut)]
        icols = [c for c in idf.columns if c not in ('stock_code', 'date')][:args.max_interact]
        base_df = base_df.merge(idf[['stock_code', 'date'] + icols],
                                on=['stock_code', 'date'], how='left')
        interact_features = icols
    else:
        print(f'note: {inter_path} 不存在（Class4为空跑；先跑 --form gain 并过FDR）')

    # 2b. Class2 股性榜单（V4.3c）：vwap_habitat_pool 并入并做截面居中秩。
    # 6 个固定底座保持原值（历史可比冻结）；新增池列统一居中秩（树分裂不变）。
    habitat_pool_features = []
    _hp_path = RESULTS / 'vwap_habitat_pool.parquet'
    if _hp_path.exists():
        try:
            hdf = pd.read_parquet(_hp_path)
            hdf['date'] = pd.to_datetime(hdf['date'])
            hdf = hdf[hdf['date'] <= pd.Timestamp(args.data_cut)]
            hcols = [c for c in hdf.columns if c not in ('stock_code', 'date')]
            hdf_norm = cs_rank_centered(hdf, hcols)
            base_df = base_df.merge(hdf_norm, on=['stock_code', 'date'], how='left')
            habitat_pool_features = hcols
            print(f'Class2 股性池并入 {len(hcols)} 列', flush=True)
        except Exception as e:
            print(f'warning: habitat pool 合并失败 {str(e)[:120]}', flush=True)

    all_features = alpha_features + interact_features + habitat_features \
        + habitat_pool_features + sector_features
    feat_file = out_dir / 'lgb_features.parquet'
    base_df.to_parquet(feat_file, index=False)

    manifest = {
        'data_cut': args.data_cut,
        'n_total_features': len(all_features),
        'features_all': all_features,
        'by_class': {
            'class1_stock_alpha': alpha_features,
            'class2_stock_habitat': habitat_features,
            'class2_habitat_pool': habitat_pool_features,
            'class3_sector_state': sector_features,
            'class4_gated_interact': interact_features,
            'market_broadcast_1xT': [],  # 铁律：永远为空！
        },
        # 供下游训练脚本读取：以下列严禁在训练前做逐日横截面标准化
        'skip_daily_cs_norm_cols': interact_features + sector_features
        + habitat_pool_features + alpha_features,
    }
    man_file = out_dir / 'features_manifest.json'
    man_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"导出完成: {len(all_features)} 列特征 -> {feat_file}")
    print(f"Manifest -> {man_file}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
