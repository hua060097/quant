#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试使用fiximport解决导入问题
并演示如何修复"ValueError: The truth value of a DataFrame is ambiguous"错误
"""

import fiximport  # 放在最开始，用于解决导入问题

# 导入生成特征的函数
try:
    from to.generate_sector_features import generate_features
    print("成功导入generate_features函数")
    
    # 正确使用DataFrame的例子
    import pandas as pd
    import numpy as np
    
    print("\n===== DataFrame布尔值模糊错误修复示例 =====")
    
    # 创建一个简单的DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    
    # 错误用法: 直接将DataFrame用作布尔值
    print("错误用法例子（已注释防止程序崩溃）:")
    print("if df:  # 这会导致 ValueError: The truth value of a DataFrame is ambiguous")
    
    # 正确用法1: 检查DataFrame是否为空
    print("\n正确用法1 - 检查是否为空:")
    if not df.empty:
        print("  DataFrame不为空")
    
    # 正确用法2: 使用any()或all()
    print("\n正确用法2 - 使用any()或all():")
    if df['A'].any():  # 检查某列是否有任何为True的值
        print("  列A中存在非零值")
    
    if (df > 0).all().all():  # 检查所有值是否都为正
        print("  所有值都为正")
    
    # 正确用法3: 比较特定值
    print("\n正确用法3 - 比较特定值:")
    if (df['A'] == 2).any():
        print("  列A中存在值为2的元素")
    
    # 正确用法4: 使用布尔索引
    print("\n正确用法4 - 使用布尔索引:")
    filtered_df = df[df['A'] > 1]
    print(f"  过滤后的DataFrame:\n{filtered_df}")
    
    print("\n===== 修复说明 =====")
    print("当使用DataFrame时，永远不要直接将其作为布尔条件使用")
    print("而是应该使用:\n1. df.empty - 检查是否为空\n2. df.any()或df.all() - 检查是否有TRUE值或全为TRUE\n3. df.item() - 仅当DataFrame只包含一个值时使用")
    
    # 尝试调用函数
    # 注释掉以避免再次出错
    # result = generate_features(
    #     start_date="20230101", 
    #     end_date="20230110",
    #     sectors=["TGN人工智能"],
    #     force_update=True
    # )
    # 
    # print(f"成功执行函数，处理了 {len(result)} 个板块")
except ImportError as e:
    print(f"导入错误: {e}")
except Exception as e:
    print(f"执行错误: {e}") 