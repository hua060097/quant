#!/usr/bin/env python3
"""检查factors.parquet中的因子状态"""

import pandas as pd

def check_factors_status():
    df = pd.read_parquet('C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/factor_lab_data/factors.parquet')

    # 失败的因子列表
    failed_factors = [
        'consecutive_up_severity', 'corr_20d_mom5', 'corr_5d', 'corr_slope_20_120',
        'distance_from_52w_high', 'distance_from_52w_low', 'idiosyncratic_volatility_sector',
        'institutional_activity_score', 'lower_shadow_ratio', 'ma60_slope', 'ma_convergence_score',
        'ma_ratio_5_10', 'market_correlation', 'near_support_count_60d', 'obv_momentum_factor',
        'open_to_high_ratio', 'pct_chg_1d', 'pct_chg_2d', 'price_impact', 'price_ma60_ratio',
        'pullback_volume_ratio', 'resistance_distance', 'return_3d', 'support_distance',
        'support_risk_reward_ratio', 'turnover_rate_strength', 'volatility_change_rate',
        'volatility_correlation', 'volume_price_divergence'
    ]

    print('修复的因子在factors.parquet中的状态:')
    print('='*60)

    for factor in failed_factors:
        z_col = f'{factor}_z'
        rank_col = f'{factor}_rank'
        intra_z_col = f'{factor}_intra_z'
        intra_rank_col = f'{factor}_intra_rank'

        status = []
        if z_col in df.columns:
            status.append(f'[OK] {z_col}')
        if rank_col in df.columns:
            status.append(f'[OK] {rank_col}')
        if intra_z_col in df.columns:
            status.append(f'[OK] {intra_z_col}')
        if intra_rank_col in df.columns:
            status.append(f'[OK] {intra_rank_col}')

        if factor in df.columns:
            status.append(f'[RAW] {factor}')

        if status:
            print(f'{factor}: {" ".join(status)}')
        else:
            print(f'{factor}: [MISSING] 未找到')

    print('\n' + '='*60)
    print('检查几个具体因子的情况:')

    # 检查几个具体的因子
    test_factors = ['ma_ratio_5_10', 'corr_slope_20_120', 'ma_convergence_score']
    for factor in test_factors:
        if factor in df.columns:
            print(f'\n{factor} (原始):')
            series = df[factor]
            print(f'  数量: {len(series):,}')
            print(f'  均值: {series.mean():.6f}')
            print(f'  标准差: {series.std():.6f}')
            print(f'  范围: [{series.min():.6f}, {series.max():.6f}]')

        z_col = f'{factor}_z'
        if z_col in df.columns:
            print(f'\n{factor}_z (z-score归一化):')
            series = df[z_col]
            print(f'  数量: {len(series):,}')
            print(f'  均值: {series.mean():.6f}')
            print(f'  标准差: {series.std():.6f}')
            print(f'  范围: [{series.min():.6f}, {series.max():.6f}]')

            # 检查是否有极端值
            extreme_count = ((series.abs() > 3).sum())
            extreme_ratio = extreme_count / len(series)
            print(f'  超过±3的极端值比例: {extreme_ratio:.2%}')

if __name__ == "__main__":
    check_factors_status()