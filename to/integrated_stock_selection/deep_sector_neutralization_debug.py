#!/usr/bin/env python3
"""深度调试行业中性化，找出异常值的根本原因"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def deep_sector_neutralization_debug():
    """深度调试行业中性化过程"""
    print("=== 深度行业中性化调试 ===\n")

    try:
        # 1. 模拟实际的数据结构（多个交易日）
        print("步骤1: 模拟实际的多交易日数据结构")

        # 读取多日数据
        dates_to_check = ['20241113', '20241114', '20241115']
        all_data = []

        for date_str in dates_to_check:
            try:
                df = pd.read_parquet(f"C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/{date_str}.parquet")
                df['date_str'] = date_str
                all_data.append(df)
                print(f"  {date_str}: {df.shape}")
            except Exception as e:
                print(f"  {date_str}: 文件不存在或读取失败 - {e}")

        if not all_data:
            print("没有找到有效的数据文件")
            return

        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"合并后数据形状: {combined_df.shape}")

        # 2. 加载行业映射
        print("\n步骤2: 加载行业映射")
        from main import load_sw_sector_mapping
        sector_mapping = load_sw_sector_mapping()
        print(f"行业映射长度: {len(sector_mapping)}")

        # 3. 添加行业信息
        print("\n步骤3: 添加行业信息")
        combined_df['sector'] = combined_df['stock_code'].map(sector_mapping)

        # 4. 创建pct_chg_1d_raw（模拟原始的昨日涨幅）
        print("\n步骤4: 创建模拟的昨日涨幅数据")
        # 这里简化处理，实际应该是前一日数据
        combined_df['pct_chg_1d_raw'] = combined_df['pct_chg']

        # 5. 检查数据质量
        print("\n步骤5: 数据质量检查")
        print(f"总数据量: {len(combined_df)}")
        print(f"有行业信息的数据量: {combined_df['sector'].notna().sum()}")
        print(f"无行业信息的数据量: {combined_df['sector'].isna().sum()}")

        # 检查每个日期的数据分布
        date_stats = combined_df.groupby('date_str').agg({
            'stock_code': 'count',
            'sector': lambda x: x.notna().sum(),
            'pct_chg_1d_raw': ['mean', 'std', 'min', 'max']
        })
        print("各日期数据统计:")
        print(date_stats)

        # 6. 模拟完整的行业中性化过程
        print("\n步骤6: 模拟完整的行业中性化过程")

        # 准备工作数据
        work_df = combined_df[['stock_code', 'date_str', 'pct_chg_1d_raw', 'sector']].copy()
        work_df = work_df.dropna(subset=['sector', 'pct_chg_1d_raw']).reset_index(drop=True)

        print(f"工作数据形状: {work_df.shape}")

        # 创建结果Series
        result = pd.Series(0.0, index=work_df.index)

        # 对每个交易日进行截面回归
        extreme_values = []
        regression_errors = []

        for date_str, date_group in work_df.groupby('date_str'):
            print(f"\n处理日期: {date_str}")
            print(f"  股票数量: {len(date_group)}")

            if len(date_group) <= 1:
                print(f"  股票数量太少，跳过")
                continue

            # 检查行业分布
            sector_counts = date_group['sector'].value_counts()
            single_stock_sectors = sector_counts[sector_counts == 1]

            if len(single_stock_sectors) > 0:
                print(f"  警告: 发现 {len(single_stock_sectors)} 个单股票行业")
                for sector in single_stock_sectors.index:
                    stock = date_group[date_group['sector'] == sector]['stock_code'].iloc[0]
                    pct_val = date_group[date_group['sector'] == sector]['pct_chg_1d_raw'].iloc[0]
                    print(f"    {sector}: {stock} ({pct_val:.6f})")

            try:
                # 生成行业哑变量
                X = pd.get_dummies(date_group['sector'], drop_first=False)
                y = date_group['pct_chg_1d_raw'].values.reshape(-1, 1)

                print(f"  回归矩阵: X={X.shape}, y={y.shape}")

                # 检查回归矩阵的条件数（可能的多重共线性问题）
                if hasattr(np.linalg, 'cond'):
                    cond_num = np.linalg.cond(X.values.astype(float))
                    print(f"  回归矩阵条件数: {cond_num:.2f}")
                    if cond_num > 1e10:
                        print(f"  警告: 矩阵条件数过大，可能存在数值不稳定")

                # 线性回归
                model = LinearRegression(fit_intercept=True)
                model.fit(X, y)

                # 预测和残差计算
                y_pred = model.predict(X)
                residual = (y - y_pred).flatten()

                print(f"  回归R²: {model.score(X, y):.6f}")
                print(f"  残差统计: 均值={np.mean(residual):.6f}, 标准差={np.std(residual):.6f}")
                print(f"  残差范围: [{np.min(residual):.6f}, {np.max(residual):.6f}]")

                # 检查极端残差
                extreme_mask = np.abs(residual) > 500
                extreme_count = np.sum(extreme_mask)

                if extreme_count > 0:
                    print(f"  发现 {extreme_count} 个极端残差 (>500):")
                    extreme_indices = np.where(extreme_mask)[0]

                    for idx in extreme_indices:
                        original = y[idx, 0]
                        predicted = y_pred[idx, 0]
                        resid = residual[idx]
                        stock_code = date_group.iloc[idx]['stock_code']
                        sector = date_group.iloc[idx]['sector']

                        extreme_values.append({
                            'date_str': date_str,
                            'stock_code': stock_code,
                            'sector': sector,
                            'original': original,
                            'predicted': predicted,
                            'residual': resid
                        })

                        print(f"    {stock_code} ({sector}): {original:.6f} -> {predicted:.6f}, 残差={resid:.6f}")

                # 将残差赋值给结果
                date_indices = date_group.index
                if hasattr(date_indices, 'values'):
                    date_indices = date_indices.values

                for i, idx in enumerate(date_indices):
                    result.loc[idx] = residual[i]

            except Exception as e:
                print(f"  回归失败: {e}")
                regression_errors.append({
                    'date_str': date_str,
                    'error': str(e),
                    'stock_count': len(date_group),
                    'sector_count': len(sector_counts)
                })

        # 7. 汇总极端值分析
        print(f"\n步骤7: 极端值汇总分析")
        print(f"回归错误数量: {len(regression_errors)}")
        print(f"极端残差数量: {len(extreme_values)}")

        if extreme_values:
            print("\n所有极端残差详情:")
            extreme_df = pd.DataFrame(extreme_values)
            print(extreme_df.sort_values('residual', ascending=False))

            # 特别检查是否有可能导致1729.304679的情况
            very_large_residuals = extreme_df[abs(extreme_df['residual']) > 1000]
            if len(very_large_residuals) > 0:
                print(f"\n极大残差 (|值| > 1000):")
                print(very_large_residuals)

        # 8. 检查001279.SZ的完整处理过程
        print(f"\n步骤8: 检查001279.SZ的处理过程")
        stock_001279 = work_df[work_df['stock_code'] == '001279.SZ']

        if len(stock_001279) > 0:
            for _, stock_row in stock_001279.iterrows():
                date_str = stock_row['date_str']
                original = stock_row['pct_chg_1d_raw']
                sector = stock_row['sector']

                print(f"  日期: {date_str}")
                print(f"  行业: {sector}")
                print(f"  原始值: {original:.6f}")

                # 在极端值中查找
                matching_extreme = [ev for ev in extreme_values if ev['stock_code'] == '001279.SZ' and ev['date_str'] == date_str]
                if matching_extreme:
                    extreme_info = matching_extreme[0]
                    print(f"  预测值: {extreme_info['predicted']:.6f}")
                    print(f"  残差: {extreme_info['residual']:.6f}")
                    print(f"  *** 发现异常! ***")
                else:
                    print(f"  未发现极端残差")
        else:
            print("  未找到001279.SZ的数据")

        # 9. 检查是否是数据处理顺序问题
        print(f"\n步骤9: 检查数据处理顺序问题")
        print("注意: 当前分析使用的是当日pct_chg作为pct_chg_1d_raw")
        print("实际的pct_chg_1d应该是前一日的pct_chg")
        print("如果数据映射或时间对齐有问题，可能导致异常")

        # 10. 结论和建议
        print(f"\n步骤10: 分析结论")
        if len(extreme_values) > 0:
            print("1. 发现了极端残差，说明行业中性化过程中存在问题")
            print("2. 可能的原因:")
            print("   - 单股票行业的回归计算不稳定")
            print("   - 行业哑变量矩阵的数值稳定性问题")
            print("   - 数据时间对齐问题")
            print("3. 建议:")
            print("   - 对单股票行业进行特殊处理")
            print("   - 改进数值稳定性")
            print("   - 检查数据时间对齐逻辑")
        else:
            print("1. 当前的简化测试未发现极端残差")
            print("2. 异常值可能来自:")
            print("   - 数据预处理阶段的其他问题")
            print("   - 时间对齐的具体实现")
            print("   - 其他数据处理步骤")

    except Exception as e:
        print(f"深度调试失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    deep_sector_neutralization_debug()