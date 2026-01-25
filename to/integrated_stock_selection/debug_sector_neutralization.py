#!/usr/bin/env python3
"""深入分析行业中性化实现逻辑和数据质量"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def debug_sector_neutralization():
    """调试行业中性化过程，找出异常值的根本原因"""
    print("=== 行业中性化调试分析 ===\n")

    try:
        # 1. 读取20241114的数据
        df = pd.read_parquet("C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/20241114.parquet")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 2. 创建pct_chg_1d原始数据（昨日涨幅）
        print("\n=== 步骤1: 创建pct_chg_1d原始数据 ===")
        df['pct_chg_1d_raw'] = df['pct_chg']  # 注意：这里应该是前一日数据，但为了调试先用当日

        # 3. 检查001279.SZ的基础数据
        print("\n=== 步骤2: 检查001279.SZ的基础数据 ===")
        stock_mask = df['stock_code'] == '001279.SZ'
        stock_data = df[stock_mask].copy()

        if len(stock_data) > 0:
            print(f"001279.SZ基础数据:")
            print(f"  原始pct_chg: {stock_data['pct_chg'].iloc[0]:.6f}")
            print(f"  pct_chg_1d_raw: {stock_data['pct_chg_1d_raw'].iloc[0]:.6f}")

        # 4. 加载行业映射数据
        print("\n=== 步骤3: 加载行业映射数据 ===")
        try:
            from main import load_sw_sector_mapping
            sector_mapping = load_sw_sector_mapping()
            print(f"行业映射数据类型: {type(sector_mapping)}")
            print(f"行业映射数据长度: {len(sector_mapping)}")

            # 检查001279.SZ的行业
            if '001279.SZ' in sector_mapping:
                stock_sector = sector_mapping['001279.SZ']
                print(f"001279.SZ行业: {stock_sector}")
            else:
                print("001279.SZ不在行业映射中")
                stock_sector = None
                # 尝试查找相似代码
                similar_codes = [code for code in sector_mapping.keys() if code.startswith('001279')]
                print(f"相似代码: {similar_codes[:10]}")

        except Exception as e:
            print(f"加载行业映射失败: {e}")
            stock_sector = None
            # 创建虚拟行业映射进行测试
            sector_mapping = {}

        # 5. 模拟行业中性化过程
        print("\n=== 步骤4: 模拟行业中性化过程 ===")

        # 添加行业信息到数据
        df['sector'] = df['stock_code'].map(sector_mapping)
        print(f"行业映射结果:")
        print(f"  有行业信息的股票数: {df['sector'].notna().sum()}")
        print(f"  无行业信息的股票数: {df['sector'].isna().sum()}")

        print(f"001279.SZ行业: {stock_sector}")

        # 6. 手动进行行业中性化
        print("\n=== 步骤5: 手动进行行业中性化 ===")

        # 删除没有行业数据的行
        work_df = df[['stock_code', 'date', 'pct_chg_1d_raw', 'sector']].copy()
        work_df = work_df.dropna(subset=['sector', 'pct_chg_1d_raw'])

        print(f"工作数据形状: {work_df.shape}")
        print(f"行业分布:")
        sector_counts = work_df['sector'].value_counts().head(10)
        for sector, count in sector_counts.items():
            print(f"  {sector}: {count}")

        # 检查001279.SZ所在行业的数据
        if stock_sector:
            sector_data = work_df[work_df['sector'] == stock_sector]
            print(f"\n{stock_sector}行业数据:")
            print(f"  股票数量: {len(sector_data)}")
            print(f"  pct_chg_1d_raw统计:")
            print(f"    均值: {sector_data['pct_chg_1d_raw'].mean():.6f}")
            print(f"    标准差: {sector_data['pct_chg_1d_raw'].std():.6f}")
            print(f"    范围: [{sector_data['pct_chg_1d_raw'].min():.6f}, {sector_data['pct_chg_1d_raw'].max():.6f}]")

            # 显示该行业的部分股票
            print(f"  该行业的部分股票:")
            sample_stocks = sector_data[['stock_code', 'pct_chg_1d_raw']].head(10)
            for idx, row in sample_stocks.iterrows():
                print(f"    {row['stock_code']}: {row['pct_chg_1d_raw']:.6f}")

        # 7. 尝试简单的行业中性化
        print("\n=== 步骤6: 尝试简单的行业中性化 ===")

        # 方法1: 行业均值调整
        sector_means = work_df.groupby('sector')['pct_chg_1d_raw'].mean()
        print(f"行业均值统计:")
        print(f"  均值: {sector_means.mean():.6f}")
        print(f"  标准差: {sector_means.std():.6f}")
        print(f"  范围: [{sector_means.min():.6f}, {sector_means.max():.6f}]")

        # 检查异常的行业均值
        extreme_sectors = sector_means[abs(sector_means) > 10]
        if len(extreme_sectors) > 0:
            print(f"异常行业均值 (|值| > 10):")
            for sector, mean_val in extreme_sectors.items():
                count = sector_counts.get(sector, 0)
                print(f"  {sector}: {mean_val:.6f} (股票数: {count})")

        # 方法2: 手动回归行业中性化（模拟_sector_neutralize函数）
        print(f"\n=== 步骤7: 手动回归行业中性化 ===")

        try:
            # 创建行业哑变量
            sector_dummies = pd.get_dummies(work_df['sector'], prefix='sector', drop_first=True)

            # 准备回归数据
            X = sector_dummies
            y = work_df['pct_chg_1d_raw'].values

            print(f"回归数据形状: X={X.shape}, y={y.shape}")
            print(f"行业哑变量数量: {len(sector_dummies.columns)}")

            # 线性回归
            model = LinearRegression()
            model.fit(X, y)

            # 计算残差
            y_pred = model.predict(X)
            residuals = y - y_pred

            print(f"回归结果:")
            print(f"  R²: {model.score(X, y):.6f}")
            print(f"  残差统计:")
            print(f"    均值: {np.mean(residuals):.6f}")
            print(f"    标准差: {np.std(residuals):.6f}")
            print(f"    范围: [{np.min(residuals):.6f}, {np.max(residuals):.6f}]")

            # 检查极端残差
            extreme_residuals = np.abs(residuals) > 100
            extreme_count = np.sum(extreme_residuals)
            print(f"  极端残差数量 (|残差| > 100): {extreme_count}")

            if extreme_count > 0:
                extreme_indices = np.where(extreme_residuals)[0]
                print(f"  极端残差详情:")
                for idx in extreme_indices[:10]:  # 只显示前10个
                    stock_code = work_df.iloc[idx]['stock_code']
                    sector = work_df.iloc[idx]['sector']
                    original = y[idx]
                    predicted = y_pred[idx]
                    residual = residuals[idx]
                    print(f"    {stock_code} ({sector}): 原始={original:.6f}, 预测={predicted:.6f}, 残差={residual:.6f}")

            # 特别检查001279.SZ
            if len(stock_data) > 0:
                stock_idx = work_df[work_df['stock_code'] == '001279.SZ'].index
                if len(stock_idx) > 0:
                    stock_idx = stock_idx[0]
                    if stock_idx in range(len(residuals)):
                        stock_residual = residuals[stock_idx]
                        print(f"\n001279.SZ行业中性化结果:")
                        print(f"  原始值: {y[stock_idx]:.6f}")
                        print(f"  预测值: {y_pred[stock_idx]:.6f}")
                        print(f"  残差: {stock_residual:.6f}")
                        print(f"  行业: {work_df.iloc[stock_idx]['sector']}")

        except Exception as e:
            print(f"手动回归失败: {e}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")

        # 8. 检查数据质量问题
        print(f"\n=== 步骤8: 检查数据质量问题 ===")

        # 检查pct_chg_1d_raw的分布
        print(f"pct_chg_1d_raw质量检查:")
        print(f"  NaN数量: {work_df['pct_chg_1d_raw'].isna().sum()}")
        print(f"  无穷值数量: {np.isinf(work_df['pct_chg_1d_raw']).sum()}")
        print(f"  极端值 (|值| > 50) 数量: {(abs(work_df['pct_chg_1d_raw']) > 50).sum()}")

        # 检查行业映射问题
        print(f"\n行业映射质量检查:")
        unique_sectors = work_df['sector'].unique()
        print(f"  总行业数: {len(unique_sectors)}")

        # 检查空行业名
        empty_sectors = [s for s in unique_sectors if pd.isna(s) or str(s).strip() == '']
        print(f"  空行业名数量: {len(empty_sectors)}")

        # 检查异常行业名
        weird_sectors = [s for s in unique_sectors if isinstance(s, str) and len(s) > 20]
        if weird_sectors:
            print(f"  异常行业名: {weird_sectors}")

    except Exception as e:
        print(f"调试失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_sector_neutralization()