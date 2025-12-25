#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复_train_rolling_day_model函数中model_package返回问题的脚本
"""

import os
import sys
import re

def fix_train_rolling_day_model_return():
    """修复_train_rolling_day_model函数的返回值，确保返回包含模型的model_package"""
    
    # 定位主程序文件
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "to", "train_sector_models.py")
    if not os.path.exists(file_path):
        print(f"无法找到目标文件: {file_path}")
        return False
    
    # 创建备份
    backup_path = f"{file_path}.bak"
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"创建备份文件: {backup_path}")
    
    # 定义需要替换的代码部分的模式
    pattern = r"(# 返回更丰富的结果信息\s+)return\s+\{\s*'predictions'.*?'date':\s*date_obj\s*\}"
    
    # 替换为新的返回代码
    replacement = r"""\1# 创建预测结果字典
            prediction_results = {
                'predictions': result_df,
                'test_df': test_df,
                'metrics': metrics,
                'date': date_obj
            }
            
            # 创建完整的模型包
            model_package = {
                # 模型组件
                'market_classifier': market_model,
                'strength_predictor': strength_model,
                'lgb_model': final_model,  # 使用lgb_model键名，确保与_predict_with_rolling_model函数兼容
                'final_model': final_model,  # 额外添加，提供冗余
                'final_ensemble': final_model,  # 额外添加，提供冗余
                'model': final_model,  # 额外添加，提供冗余
                'lnn': lnn_model,
                
                # 市场子模型
                **market_submodels,
                
                # 模型元数据
                'metadata': {
                    'final_feature_columns': final_model_input_actual_cols,
                    'categorical_features': categorical_feature,
                    'trained_cat_dtypes': trained_cat_dtypes_map,
                    'label_encoders': trained_label_encoders,
                    'training_date': train_date,
                    'market_features': market_env_feats_train if 'market_env_feats_train' in locals() else [],
                    'strength_features': X_meta_train.columns.tolist() if 'X_meta_train' in locals() else []
                },
                
                # 序列模型参数
                'seq_length': seq_length if 'seq_length' in locals() else 0,
                
                # 预测结果(可选，用于调试)
                'prediction_results': prediction_results
            }
            
            self.logger.info(f"成功创建模型包，包含关键组件: market_classifier, strength_predictor, lgb_model, lnn")
            return model_package"""
    
    # 执行替换
    modified_content = re.sub(pattern, replacement, original_content, flags=re.DOTALL)
    
    # 检查是否有变化
    if modified_content == original_content:
        print("没有找到匹配的代码段进行替换")
        
        # 尝试另一种方法：直接定位返回语句
        simple_pattern = r"(# 返回更丰富的结果信息\s+)return\s+\{([^}]*?)}'predictions'([^}]*?)}"
        modified_content = re.sub(simple_pattern, replacement, original_content, flags=re.DOTALL)
        
        if modified_content == original_content:
            print("两种方法都无法匹配代码段，尝试更简单的方法")
            
            # 最简单的方法：定位关键行周围的代码
            lines = original_content.split('\n')
            for i, line in enumerate(lines):
                if '# 返回更丰富的结果信息' in line and i + 6 < len(lines):
                    # 找到目标行后，替换接下来的几行
                    replacement_lines = replacement.split('\n')
                    lines[i:i+6] = replacement_lines
                    modified_content = '\n'.join(lines)
                    print(f"使用直接替换行的方法，从第{i}行开始替换")
                    break
    
    # 如果内容有变化，写入文件
    if modified_content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print("修复成功！_train_rolling_day_model函数现在返回完整的model_package")
        return True
    else:
        print("修复失败：无法找到要替换的代码段")
        return False

if __name__ == "__main__":
    success = fix_train_rolling_day_model_return()
    if success:
        print("修复脚本执行成功")
    else:
        print("修复脚本执行失败")
        sys.exit(1) 