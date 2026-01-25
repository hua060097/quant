#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析因子系统的加载和注册机制
"""

def analyze_factor_system():
    """分析因子系统的工作原理"""
    print("=" * 80)
    print("因子系统分析")
    print("=" * 80)

    # 1. 检查模块导入
    print("1. 模块导入测试:")
    try:
        from factor_system.factor_lab.definitions import moving_average
        print("   [SUCCESS] moving_average 模块导入成功")

        # 检查因子对象是否存在
        has_ma60 = hasattr(moving_average, 'price_ma60_ratio')
        has_ma5 = hasattr(moving_average, 'price_ma5_ratio')
        has_ma10 = hasattr(moving_average, 'price_ma10_ratio')

        print(f"   price_ma60_ratio 对象存在: {has_ma60}")
        print(f"   price_ma5_ratio 对象存在: {has_ma5}")
        print(f"   price_ma10_ratio 对象存在: {has_ma10}")

        if has_ma60:
            ma60_obj = moving_average.price_ma60_ratio
            print(f"   price_ma60_ratio 对象: {ma60_obj}")
            print(f"   price_ma60_ratio 名称: {ma60_obj.name if hasattr(ma60_obj, 'name') else 'N/A'}")

    except Exception as e:
        print(f"   [ERROR] 导入失败: {e}")
        import traceback
        traceback.print_exc()

    print()

    # 2. 检查注册表状态
    print("2. 注册表状态:")
    try:
        from factor_system.factor_lab.registry import registry

        print(f"   注册表中的因子总数: {len(registry.names())}")
        print(f"   注册表是否为空: {len(registry.names()) == 0}")

        if len(registry.names()) > 0:
            print(f"   注册表中的前5个因子: {registry.names()[:5]}")
        else:
            print("   注册表中没有任何因子")

    except Exception as e:
        print(f"   [ERROR] 注册表检查失败: {e}")

    print()

    # 3. 检查definitions/__init__.py的导入
    print("3. definitions/__init__.py 导入检查:")
    try:
        from factor_system.factor_lab import definitions
        print("   [SUCCESS] definitions 包导入成功")

        # 检查__all__属性
        if hasattr(definitions, '__all__'):
            all_factors = definitions.__all__
            print(f"   __all__ 中的因子数量: {len(all_factors)}")

            # 检查我们的新因子
            new_factors = ['price_ma5_ratio', 'price_ma10_ratio', 'price_ma60_ratio']
            for factor in new_factors:
                is_in_all = factor in all_factors
                print(f"   {factor} 在 __all__ 中: {is_in_all}")
        else:
            print("   [WARNING] definitions 没有 __all__ 属性")

    except Exception as e:
        print(f"   [ERROR] definitions 检查失败: {e}")

    print()

    # 4. 模拟因子系统初始化
    print("4. 模拟完整的因子系统初始化:")
    try:
        # 模拟导入所有定义模块
        print("   导入 definitions.__init__...")

        # 这里会触发所有 from .xxx import yyy 语句
        import factor_system.factor_lab.definitions

        # 检查注册表状态
        from factor_system.factor_lab.registry import registry
        print(f"   导入后注册表因子总数: {len(registry.names())}")

        if len(registry.names()) > 0:
            print("   注册表中的因子:")
            for i, name in enumerate(registry.names()[:10]):
                print(f"     {i+1}. {name}")
            if len(registry.names()) > 10:
                print(f"     ... 还有 {len(registry.names()) - 10} 个因子")
        else:
            print("   [WARNING] 注册表仍然为空")

    except Exception as e:
        print(f"   [ERROR] 模拟初始化失败: {e}")

    print()

    # 5. 核心分析
    print("5. 核心发现:")
    print("   [发现1] definitions/__init__.py 负责因子注册")
    print("           - 当 from .xxx import yyy 执行时，会调用 register_factor()")
    print("           - register_factor() 将因子添加到全局注册表中")
    print()
    print("   [发现2] price_ma60_ratio 未在 __init__.py 中导入")
    print("           - 所以 register_factor() 没有被调用")
    print("           - 因子不在全局注册表中")
    print()
    print("   [发现3] 系统仍然能够生成因子")
    print("           - 说明系统有备用机制")
    print("           - 可能直接使用模块中的因子对象")
    print()

    # 6. 检查备用机制
    print("6. 检查备用机制:")
    try:
        # 直接从模块获取因子
        from factor_system.factor_lab.definitions.moving_average import price_ma60_ratio
        print(f"   [SUCCESS] 直接从 moving_average 模块获取: {price_ma60_ratio.name}")

        # 检查因子的可调用性
        if hasattr(price_ma60_ratio, 'calculator'):
            print(f"   [SUCCESS] 因子计算器可用: {price_ma60_ratio.calculator}")

    except Exception as e:
        print(f"   [ERROR] 备用机制测试失败: {e}")

    print("=" * 80)
    print("分析结论:")
    print("=" * 80)
    print("1. __init__.py 的作用是集中管理和注册所有因子")
    print("2. 未在 __init__.py 中导入的因子不会进入全局注册表")
    print("3. 系统可能有备用机制直接使用模块中的因子对象")
    print("4. 推荐做法: 将新因子添加到 __init__.py 中以确保一致性")

if __name__ == "__main__":
    analyze_factor_system()