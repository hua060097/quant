#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复LNN特征生成代码中的缩进问题。

这个脚本读取train_sector_models.py，彻底替换LNN特征生成的问题部分，
避免重复代码和缩进问题。
"""

import os
import re
import sys

def fix_lnn_features():
    """修复train_sector_models.py中的LNN特征生成代码"""
    
    # 文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(script_dir, "train_sector_models.py")
    
    # 读取原文件
    with open(target_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 保存备份
    with open(target_file + '.bak2', 'w', encoding='utf-8') as file:
        file.write(content)
    
    # 1. 按行读取文件
    lines = content.split('\n')
    
    # 2. 识别需要替换的代码区域 - 从开始标记到结束标记
    start_idx = -1
    end_idx = -1
    
    # 查找开始位置 - LNN特征处理代码块的开始
    for i, line in enumerate(lines):
        if "                                    # 使用与训练集相同的逻辑处理序列数据" in line:
            next_line = lines[i+1] if i+1 < len(lines) else ""
            if "recent_sequence = sequence[-seq_length:]" in next_line:
                start_idx = i
                break
    
    # 查找结束位置 - 找到LNN处理块后的首个非嵌套级别的except异常处理
    if start_idx != -1:
        for i in range(start_idx, len(lines)):
            if "                                except Exception as e:" in lines[i]:
                if "测试数据: 处理板块" in lines[i+1]:
                    end_idx = i
                    break
    
    if start_idx == -1 or end_idx == -1:
        print(f"未找到需要替换的代码区域: start_idx={start_idx}, end_idx={end_idx}")
        return False
    
    # 3. 构建用于替换的新代码块
    new_code = """                                    # 使用与训练集相同的逻辑处理序列数据
                                    recent_sequence = sequence[-seq_length:]
                                    
                                    # 确保序列至少是2D
                                    if len(recent_sequence.shape) < 2:
                                        try:
                                            recent_sequence = recent_sequence.reshape(len(recent_sequence), 1)
                                        except Exception as e:
                                            self.logger.error(f"测试数据: 重塑序列数据时出错: {e}")
                                            continue
                                    
                                    # 创建tensor变量
                                    seq_tensor = None
                                    
                                    # 应用scaler - 独立于if块，解决缩进问题
                                    if scaler is not None:
                                        try:
                                            # 确保维度正确后再应用scaler
                                            seq_flat = recent_sequence.reshape(-1, recent_sequence.shape[-1])
                                            seq_scaled = scaler.transform(seq_flat)
                                            seq_tensor = torch.FloatTensor(seq_scaled.reshape(recent_sequence.shape)).unsqueeze(0)
                                        except Exception as e:
                                            self.logger.warning(f"测试数据: 缩放序列数据时出错: {e}")
                                            # 失败则使用未缩放的数据
                                            seq_tensor = torch.FloatTensor(recent_sequence).unsqueeze(0)
                                    else:
                                        # 没有scaler时直接创建tensor
                                        seq_tensor = torch.FloatTensor(recent_sequence).unsqueeze(0)
                                    
                                    # 使用LNN模型预测 - 独立于if块，解决缩进问题
                                    try:
                                        with torch.no_grad():
                                            try:
                                                # 尝试使用predict方法
                                                lnn_pred = lnn_model.predict(seq_tensor.cpu().numpy())
                                                
                                                # 确保输出格式正确
                                                if isinstance(lnn_pred, np.ndarray):
                                                    if lnn_pred.ndim > 1:
                                                        test_lnn_outputs[sector_code] = float(lnn_pred[0][0])
                                                    else:
                                                        test_lnn_outputs[sector_code] = float(lnn_pred[0])
                                                else:
                                                    lnn_pred_np = lnn_pred.cpu().numpy() if hasattr(lnn_pred, 'cpu') else np.array(lnn_pred)
                                                    test_lnn_outputs[sector_code] = float(lnn_pred_np[0][0] if lnn_pred_np.ndim > 1 else lnn_pred_np[0])
                                            
                                            except Exception as e_predict:
                                                # 如果predict方法失败，尝试直接调用模型
                                                self.logger.error(f"测试数据: 使用predict方法预测失败: {e_predict}，尝试使用模型直接调用")
                                                
                                                # 尝试备用方法
                                                if hasattr(lnn_model, 'model'):
                                                    outputs, _ = lnn_model.model(seq_tensor)
                                                    test_lnn_outputs[sector_code] = float(outputs.cpu().numpy()[0][0])
                                                else:
                                                    outputs, _ = lnn_model(seq_tensor)
                                                    test_lnn_outputs[sector_code] = float(outputs.cpu().numpy()[0][0])
                                    except Exception as e:
                                        self.logger.error(f"测试数据: LNN模型预测出错，板块 {sector_code}: {e}")
                                        continue"""
    
    # 4. 替换代码
    new_lines = lines[:start_idx] + new_code.split('\n') + lines[end_idx:]
    
    # 5. 添加market_group特殊处理代码
    market_group_marker = "                    # 为其他缺失的列添加默认值0"
    market_group_code = """                    # 特殊处理market_group特征
                    if 'market_group' in missing_cols:
                        self.logger.info("为测试数据添加market_group特征...")
                        # 使用字符串值而不是分类对象，避免转换错误
                        X_final['market_group'] = 'neutral'  # 使用默认值'neutral'
                        # 从missing_cols中移除market_group
                        missing_cols.remove('market_group')
                    
                    """
    
    # 将列表转换回文本
    modified_content = '\n'.join(new_lines)
    
    # 查找并添加market_group处理代码
    modified_content = modified_content.replace(market_group_marker, market_group_code + market_group_marker)
    
    # 6. 保存修改后的文件
    with open(target_file, 'w', encoding='utf-8') as file:
        file.write(modified_content)
    
    print(f"成功修复了LNN特征生成代码，原文件已备份为 {target_file}.bak2")
    return True

if __name__ == "__main__":
    if fix_lnn_features():
        print("修复完成！")
    else:
        print("修复失败！请手动检查文件。") 