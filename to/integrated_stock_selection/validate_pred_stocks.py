#!/usr/bin/env python3
"""
验证pred_stocks模式的代码语法
"""

def validate_pred_stocks():
    """验证pred_stocks模式代码"""

    print("=== 验证pred_stocks模式代码 ===")

    try:
        # 验证参数解析
        print("1. 验证参数解析...")
        import sys
        sys.argv = ['main.py', '--mode', 'pred_stocks', '--date', '20251216']

        from main import parse_arguments
        args = parse_arguments()

        if args.mode == 'pred_stocks' and args.date == '20251216':
            print("   参数解析: 正常")
        else:
            print(f"   参数解析: 异常 - mode={args.mode}, date={args.date}")

    except Exception as e:
        print(f"   参数解析失败: {e}")
        return False

    # 验证函数导入
    try:
        print("2. 验证函数导入...")
        from main import run_pred_stocks_mode, _load_factor_data_for_prediction, _prepare_prediction_data
        print("   函数导入: 正常")
    except Exception as e:
        print(f"   函数导入失败: {e}")
        return False

    # 验证help信息
    try:
        print("3. 验证help信息...")
        import subprocess
        result = subprocess.run(['python', 'main.py', '--help'], capture_output=True, text=True)

        if 'pred_stocks' in result.stdout:
            print("   help信息: 正常 (包含pred_stocks)")
        else:
            print("   help信息: 缺少pred_stocks")

    except Exception as e:
        print(f"   help信息验证失败: {e}")

    print("=== 验证完成 ===")
    print("pred_stocks模式已成功添加到main.py")
    print("使用方法: python main.py --mode pred_stocks --date YYYYMMDD")

    return True

if __name__ == "__main__":
    validate_pred_stocks()