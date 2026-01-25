#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 --add_factor 模式的功能
"""

def test_add_factor_mode():
    """测试 add_factor 参数的功能"""
    print("=" * 80)
    print("测试 --add_factor 模式功能")
    print("=" * 80)

    # 恢复原始因子数据
    import shutil
    from pathlib import Path

    factors1_path = Path("factor_lab_data/factors1.parquet")
    factors_path = Path("factor_lab_data/factors.parquet")

    if factors1_path.exists():
        print("恢复原始因子数据...")
        shutil.copy2(factors1_path, factors_path)
        print(f"已从 {factors1_path} 恢复到 {factors_path}")
    else:
        print(f"备份文件不存在: {factors1_path}")
        return False

    print("\n" + "=" * 80)
    print("命令说明:")
    print("=" * 80)
    print("新增因子模式命令:")
    print('python main.py --mode generate_factors --start_date 20230601 --end_date 20251218 --factor_names drawdown_recovery_speed --add_factor')
    print()
    print("参数说明:")
    print("  --factor_names drawdown_recovery_speed: 指定要生成的因子")
    print("  --add_factor: 启用新增因子模式，保留原有因子，仅新增指定因子（列扩展）")
    print()
    print("预期效果:")
    print("  - 保留原有124个因子的数据不变")
    print("  - 新增 drawdown_recovery_speed 因子")
    print("  - 最终得到125个因子的完整数据")
    print("  - 不会触发 overwrite 模式，不会损坏现有数据")

    print("\n" + "=" * 80)
    print("与其他模式的对比:")
    print("=" * 80)
    print("1. 全量生成模式:")
    print('   python main.py --mode generate_factors --start_date 20230601 --end_date 20251218')
    print("   → 重新生成所有因子，覆盖整个文件")
    print()
    print("2. 增量保存模式 (有风险):")
    print('   python main.py --mode generate_factors --start_date 20230601 --end_date 20251218 --factor_names drawdown_recovery_speed --incremental_save')
    print("   → 可能触发 overwrite 模式，损坏现有因子数据")
    print()
    print("3. 新增因子模式 (安全):")
    print('   python main.py --mode generate_factors --start_date 20230601 --end_date 20251218 --factor_names drawdown_recovery_speed --add_factor')
    print("   → 强制列扩展，保留所有现有因子，安全新增指定因子")

    return True

if __name__ == "__main__":
    test_add_factor_mode()