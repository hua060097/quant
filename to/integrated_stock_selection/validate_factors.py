#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 main.py 中的 _get_stock_factors 函数是否正确包含了所有定义的因子
"""

import sys
import os
sys.path.append('.')

def get_factors_from_registry():
    """从因子注册表获取所有已注册的因子"""
    try:
        from factor_system.factor_lab.registry import registry
        return sorted(registry.names())
    except Exception as e:
        print(f"从注册表获取因子失败: {e}")
        return []

def get_factors_from_main():
    """从 main.py 获取 _get_stock_factors 返回的因子"""
    try:
        from main import _get_stock_factors
        return sorted(_get_stock_factors())
    except Exception as e:
        print(f"从 main.py 获取因子失败: {e}")
        return []

def main():
    print("因子验证工具 - 检查 main.py 中的因子是否完整")
    print("=" * 60)

    # 获取两边的因子列表
    registry_factors = get_factors_from_registry()
    main_factors = get_factors_from_main()

    print(f"\n注册表中的因子数量: {len(registry_factors)}")
    print(f"main.py 中的因子数量: {len(main_factors)}")

    # 比较差异
    registry_set = set(registry_factors)
    main_set = set(main_factors)

    missing_in_main = registry_set - main_set
    extra_in_main = main_set - registry_set

    if not missing_in_main and not extra_in_main:
        print("\n✅ 验证通过！")
        print("main.py 中的因子列表与注册表完全一致")
        return True
    else:
        print("\n❌ 验证失败！发现不一致：")

        if missing_in_main:
            print(f"\n在注册表中但不在 main.py 中的因子 ({len(missing_in_main)} 个):")
            for factor in sorted(missing_in_main):
                print(f"  - {factor}")

        if extra_in_main:
            print(f"\n在 main.py 中但不在注册表中的因子 ({len(extra_in_main)} 个):")
            for factor in sorted(extra_in_main):
                print(f"  - {factor}")

        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 因子同步成功！")
        sys.exit(0)
    else:
        print("\n⚠️  请检查因子同步是否完整")
        sys.exit(1)