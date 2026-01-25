#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量删除main.py中所有XGBoost相关代码
"""

import re

def remove_xgboost_code():
    """删除所有XGBoost相关代码"""

    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 1. 删除函数参数中的use_xgboost
    # _train_final_model_for_window函数定义
    pattern1 = r'def _train_final_model_for_window\([^)]*?use_xgboost[^)]*?\):'
    replacement1 = 'def _train_final_model_for_window(train_data, val_data, best_params, window_idx, use_lightgbm=True, market_data=None, optimize_thresholds=False, sw_sector_mapping=None):'
    content = re.sub(pattern1, replacement1, content)

    # 2. 删除所有调用_train_final_model_for_window时的use_xgboost参数
    pattern2 = r',\s*use_xgboost=[^,\)]*?(?=\))'
    content = re.sub(pattern2, '', content)

    # 3. 删除run_train_model_mode函数参数中的use_xgboost
    pattern3 = r'def run_train_model_mode\([^)]*?use_xgboost=None[^)]*?\):'
    replacement3 = 'def run_train_model_mode(args, train_end_date=None, is_regression=True, use_lightgbm=None):'
    content = re.sub(pattern3, replacement3, content)

    # 4. 删除所有use_xgboost相关的变量赋值和判断
    # 删除 "use_xgboost = 'xgboost' in selected_models"
    content = re.sub(r"use_xgboost = 'xgboost' in selected_models.*?\n", '', content)

    # 删除 "use_xgboost = args.use_xgboost"
    content = re.sub(r"use_xgboost = args\.use_xgboost.*?\n", '', content)

    # 删除 "if use_xgboost is None:" 相关的判断
    pattern4 = r'if use_xgboost is None or use_lightgbm is None:.*?use_xgboost = \'xgboost\' in selected_models.*?\n.*?if use_xgboost is None:.*?\n.*?use_xgboost = \'xgboost\' in selected_models.*?\n'
    replacement4 = '''if use_lightgbm is None:
        use_lightgbm = 'lightgbm' in selected_models or 'lightgbm' in selected_models.lower()
'''
    content = re.sub(pattern4, replacement4, content, flags=re.DOTALL)

    # 5. 删除 "if not use_xgboost and not use_lightgbm:" 判断，改为只检查lightgbm
    pattern5 = r'if not use_xgboost and not use_lightgbm:.*?logger\.error.*?\n.*?return None\n'
    replacement5 = '''if not use_lightgbm:
        logger.error("[FAIL] 必须至少选择一个模型（LightGBM）")
        return None
'''
    content = re.sub(pattern5, replacement5, content, flags=re.DOTALL)

    # 6. 删除XGBoost日志输出
    content = re.sub(r"logger\.info\(f\"   XGBoost: \{'\[OK\]'\[^\"]*\"\)  # noqa: W605.*?\n", '', content)
    content = re.sub(r"logger\.info\(f\"   XGBoost: \{'\[FAIL\]'\[^\"]*\"\)  # noqa: W605.*?\n", '', content)

    # 7. 删除XGBoost训练相关的代码块（从"if use_xgboost:"到对应的结束）
    # 这需要更复杂的处理，暂时跳过，手动处理

    if content != original_content:
        # 备份原文件
        with open('main.py.backup_before_xgboost_removal', 'w', encoding='utf-8') as f:
            f.write(original_content)

        # 写入修改后的内容
        with open('main.py', 'w', encoding='utf-8') as f:
            f.write(content)

        print("已批量删除use_xgboost参数")
        print("原文件已备份到: main.py.backup_before_xgboost_removal")
        return True
    else:
        print("未发现需要修改的内容")
        return False

if __name__ == "__main__":
    print("开始批量删除XGBoost相关代码...")
    success = remove_xgboost_code()
    if success:
        print("批量删除完成！")
    else:
        print("没有需要修改的内容")
