#!/usr/bin/env python3
"""
测试pred_stocks模式的实现
"""

def test_pred_stocks_mode():
    """测试pred_stocks模式的各项功能"""

    print("=== 测试pred_stocks模式 ===")

    # 测试参数解析
    print("\n1. 测试参数解析:")
    try:
        import sys
        sys.argv = ['main.py', '--mode', 'pred_stocks', '--date', '20251216']

        from main import parse_arguments
        args = parse_arguments()

        print(f"   模式: {args.mode}")
        print(f"   日期: {args.date}")
        print("   ✅ 参数解析正常")

    except Exception as e:
        print(f"   ❌ 参数解析失败: {e}")

    # 测试模型文件是否存在
    print("\n2. 检查模型文件:")
    try:
        from pathlib import Path
        model_path = Path("./results/lightgbm_model_window3_20251216_200900.pkl")

        if model_path.exists():
            print(f"   ✅ 模型文件存在: {model_path}")

            # 尝试加载模型文件
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            print(f"   模型对象: {type(model_data.get('model', 'unknown'))}")
            print(f"   特征数量: {len(model_data.get('feature_columns', []))}")
            print("   ✅ 模型文件可正常加载")

        else:
            print(f"   ❌ 模型文件不存在: {model_path}")
            print("   请先运行 run_test_model_walkforward_mode 来生成模型")

    except Exception as e:
        print(f"   ❌ 检查模型文件失败: {e}")

    # 测试因子数据是否存在
    print("\n3. 检查因子数据:")
    try:
        from pathlib import Path
        factor_file = Path("../local_data/features/daily/20251216.parquet")

        if factor_file.exists():
            import pandas as pd
            df = pd.read_parquet(factor_file)
            print(f"   ✅ 因子文件存在: {len(df):,} 条记录, {df['stock_code'].nunique():,} 只股票")
        else:
            print(f"   ❌ 因子文件不存在: {factor_file}")

    except Exception as e:
        print(f"   ❌ 检查因子文件失败: {e}")

    # 测试SW行业映射
    print("\n4. 检查SW行业映射:")
    try:
        from main import load_sw_sector_mapping
        sw_mapping = load_sw_sector_mapping()

        if sw_mapping:
            print(f"   ✅ SW行业映射加载成功: {len(sw_mapping)} 只股票")
        else:
            print("   ⚠️ SW行业映射为空")

    except Exception as e:
        print(f"   ❌ 检查SW行业映射失败: {e}")

    print("\n=== 测试完成 ===")
    print("如需完整测试，请运行:")
    print("python main.py --mode pred_stocks --date 20251216")

if __name__ == "__main__":
    test_pred_stocks_mode()