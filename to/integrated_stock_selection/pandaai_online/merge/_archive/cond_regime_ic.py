"""条件 IC 诊断（2026-09-26）：行业/市场状态因子的高低分组对个股因子 IC 的影响。

核心问题：`sector_disp_20`、`sector_mom_60` 等 regime 变量的高低，是否导致
动量/反转等个股因子的截面 IC 产生显著差异？如果是，该 regime 变量对 LGB
有条件价值（帮助模型学习"什么时候信动量、什么时候信反转"）。

方法：
  对每个 regime 变量，按截面中位数将交易日分为高低两组，
  在各组内分别计算目标因子的截面 Spearman RankIC，报告两组差异和 Welch t 检验。

用法：
  python cond_regime_ic.py --factor "TS_MEAN(RET1,20)" --regime sector_disp_20
  python cond_regime_ic.py --factor "TS_MEAN(RET1,20)" --regime ALL
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import _paths  # noqa: F401
from evaluator import load_panel, ensure_local_fields
from field_catalog import sw2_mapping
from parser import parse_formula, ParseError

REGIME_VARIABLES = [
    'sector_disp_20', 'sector_disp_5',
    'sector_mom_20', 'sector_mom_60',
    'sector_vol_20',
    'mkt_mom_20', 'mkt_vol_20',
]


def compute_regime_variables(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """计算 regime 变量（每个都是 date 级 Series：同日全市场同值）。"""
    codes = frame['stock_code']
    ret = frame.groupby(codes, sort=False)['CLOSE'].pct_change(fill_method=None)
    ind = codes.map(sw2_mapping()).fillna('__NA__')

    # 行业日收益：(date, industry) → mean RET1
    ind_ret_daily = ret.groupby([frame['date'], ind], sort=False).mean()

    # 行业内分化：同日同行业 RET1 的截面 std（逐日），跨行业均值 → 全局日频
    disp_daily = ret.groupby([frame['date'], ind], sort=False).std()
    disp_global = disp_daily.groupby(level=0).mean()

    # 行业动量：行业日收益 rolling sum，跨行业均值 → 全局日频
    ind_mom_daily = ind_ret_daily.groupby(level=0).sum()
    mom_global = ind_mom_daily.groupby(level=0).mean()

    # 行业波动：行业日收益 rolling std，跨行业均值
    ind_vol_daily = ind_ret_daily.groupby(level=0).std()
    vol_global = ind_vol_daily.groupby(level=0).mean()

    # 市场动量/波动
    mkt_ret = ret.groupby(frame['date'], sort=False).mean()

    out = {}
    for w in (5, 20):
        out[f'sector_disp_{w}'] = disp_global.rolling(w, min_periods=max(2, w // 2)).mean()
    for w in (20, 60):
        out[f'sector_mom_{w}'] = mom_global.rolling(w, min_periods=max(2, w // 2)).mean()
    out['sector_vol_20'] = ind_vol_daily.groupby(level=0).mean().rolling(
        20, min_periods=10).mean()
    out['mkt_mom_20'] = mkt_ret.rolling(20, min_periods=10).mean()
    out['mkt_vol_20'] = mkt_ret.rolling(20, min_periods=10).std()
    return out


def conditional_ic(ctx_di, ctx_labels, factor_values, reg_series, regime_name,
                   dates, min_stocks=30):
    """按 regime 高/低分组计算截面 RankIC。

    regime 是 date 级 broadcast（同日全市场同值）。
    分组：regime > 全期中位数 → high；≤ → low。
    每组内逐截面 Spearman RankIC（rankdata），报告 mean/t-stat。
    """
    reg_by_date = reg_series.reindex(pd.to_datetime(dates)).to_numpy(dtype=float)
    fin_reg = np.isfinite(reg_by_date)
    if fin_reg.sum() < 20:
        return {'error': 'regime 覆盖不足'}

    med_reg = np.nanmedian(reg_by_date[fin_reg])

    order_all = np.argsort(ctx_di, kind="stable")
    d_sorted = ctx_di[order_all]
    v_sorted = np.asarray(factor_values, dtype=float)[order_all]
    y_sorted = np.asarray(ctx_labels, dtype=float)[order_all]
    D = int(d_sorted.max()) + 1
    n_day = np.bincount(d_sorted, minlength=D)
    seg = np.concatenate([[0], np.cumsum(n_day)])

    reg_sorted = reg_by_date[order_all]
    ics_high, ics_low = [], []
    for k in range(D):
        a, b = int(seg[k]), int(seg[k + 1])
        if b - a < min_stocks:
            continue
        reg_k = reg_sorted[a:b]
        reg_fin = np.isfinite(reg_k)
        if reg_fin.sum() == 0:
            continue
        is_high_k = bool(np.nanmedian(reg_k[reg_fin]) > med_reg)

        vv = v_sorted[a:b]
        yy = y_sorted[a:b]
        ok = np.isfinite(vv) & np.isfinite(yy)
        if ok.sum() < min_stocks:
            continue
        rv = rankdata(vv[ok])
        ry = rankdata(yy[ok])
        rv = rv - rv.mean()
        ry = ry - ry.mean()
        den = float(np.sqrt((rv * rv).sum() * (ry * ry).sum()))
        ic = float((rv * ry).sum() / den) if den > 0 else np.nan
        if np.isfinite(ic):
            if is_high_k:
                ics_high.append(ic)
            else:
                ics_low.append(ic)

    ics_high = np.array(ics_high)
    ics_low = np.array(ics_low)
    if len(ics_high) < 10 or len(ics_low) < 10:
        return {'error': '分组样本不足'}

    mean_h = float(ics_high.mean())
    mean_l = float(ics_low.mean())
    diff = mean_h - mean_l
    se_h = ics_high.std(ddof=1) / np.sqrt(len(ics_high))
    se_l = ics_low.std(ddof=1) / np.sqrt(len(ics_low))
    t_stat = diff / np.sqrt(se_h ** 2 + se_l ** 2) if (se_h + se_l) > 0 else np.nan

    return {
        'n_high': len(ics_high), 'n_low': len(ics_low),
        'ic_high': mean_h, 'ic_low': mean_l,
        'diff': diff, 't_stat': t_stat,
    }


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='条件 IC：regime 高/低分组对因子 IC 的影响')
    ap.add_argument('--factor', required=True, help='因子公式')
    ap.add_argument('--label', default='vwap', choices=['c2c', 'vwap'])
    ap.add_argument('--regimes', default='ALL', help='逗号分隔的 regime 变量名，ALL=全部')
    ap.add_argument('--full', action='store_true', help='全量面板（默认快评 800 股）')
    ap.add_argument('--stocks', type=int, default=800)
    args = ap.parse_args()

    try:
        node = parse_formula(args.factor)
    except ParseError as exc:
        raise SystemExit(f'parse failed: {exc}')

    from fitness import DataContext, build_fast_context, tier1
    frame = load_panel()
    ensure_local_fields(frame)
    if args.full:
        ctx = DataContext(frame=frame, label=args.label)
    else:
        ctx = build_fast_context(frame, n_stocks=args.stocks, label=args.label)

    regime_vars = compute_regime_variables(frame)
    factor_vals = ctx.values(node)
    t1 = tier1(ctx.screen(node, name='target'))

    print(f'factor: {args.factor}')
    print(f'label: {args.label}  rank_ic={t1["rank_ic"]:+.4f}  sia={t1["sia"]:.5f}  '
          f'rows={ctx.n_rows:,} sections={len(ctx.ud)}')
    print(f'\n{"regime":24s} {"IC_high":>8s} {"IC_low":>8s} {"diff":>8s} {"t":>7s} {"判定"}')
    print('-' * 78)

    selected = (list(regime_vars.keys()) if args.regimes == 'ALL'
                else [x.strip() for x in args.regimes.split(',') if x.strip() in regime_vars])

    for name in selected:
        reg_series = regime_vars[name]
        result = conditional_ic(
            ctx.di, ctx.labels, factor_vals, reg_series, name, ctx.row_dates)
        if 'error' in result:
            print(f'  {name:24s} {result["error"]}')
            continue
        verdict = ('*** 高低显著' if abs(result['t_stat']) > 2 else
                   '** 边际' if abs(result['t_stat']) > 1.5 else '')
        print(f'  {name:24s} high={result["ic_high"]:+.4f}  low={result["ic_low"]:+.4f}  '
              f'diff={result["diff"]:+.4f}  t={result["t_stat"]:+.2f}  {verdict}')


if __name__ == '__main__':
    main()
