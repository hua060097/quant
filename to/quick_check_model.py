#!/usr/bin/env python3

import pickle
from pathlib import Path

# 检查最新的模型文件
model_file = Path('integrated_stock_selection/results/xgboost_model_20250825_113113.pkl')

print(f'检查模型文件: {model_file}')

try:
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f'模型文件包含的键: {list(model_data.keys())}')
    
    # 检查是否有market_multiplier
    if 'market_multiplier' in model_data:
        mm = model_data['market_multiplier']
        print(f'✅ 找到 market_multiplier')
        print(f'类型: {type(mm)}')
        if isinstance(mm, dict):
            print(f'组件: {list(mm.keys())}')
            print(f'fallback状态: {mm.get("fallback", "未定义")}')
            
            # 检查关键组件
            for key in ['scaler', 'pca', 'ridge']:
                value = mm.get(key)
                if value is not None:
                    print(f'  {key}: {type(value).__name__}')
                else:
                    print(f'  {key}: None')
        else:
            print(f'⚠️ market_multiplier不是字典类型: {mm}')
    else:
        print('❌ 未找到 market_multiplier 字段')
        
except Exception as e:
    print(f'❌ 检查失败: {e}')
    import traceback
    traceback.print_exc()