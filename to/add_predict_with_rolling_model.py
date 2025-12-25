#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
恢复丢失的_predict_with_rolling_model函数
"""

import re

def add_function_to_file():
    # 文件路径
    file_path = "train_sector_models.py"
    
    # 要添加的函数
    function_to_add = '''
    def _predict_with_rolling_model(self, model_package, features_df, prediction_date):
        """使用滚动模型进行预测
        
        Args:
            model_package: 模型包
            features_df: 特征DataFrame
            prediction_date: 预测日期
            
        Returns:
            预测结果DataFrame
        """
        try:
            # 1. 准备数据
            if not isinstance(prediction_date, pd.Timestamp):
                date_obj = pd.to_datetime(prediction_date)
            else:
                date_obj = prediction_date
            
            # 过滤出预测日期的特征数据
            test_df = features_df[features_df['date'] == date_obj].copy()
            
            if test_df.empty:
                self.logger.warning(f"日期 {prediction_date} 没有特征数据")
                return pd.DataFrame()  # 返回空DataFrame
                
            # 2. 获取模型组件
            market_model = model_package.get('market_classifier')
            strength_model = model_package.get('strength_predictor')
            
            # 尝试获取final_model，通过多种可能的键名
            final_model = None
            for key_name in ['final_ensemble', 'model', 'final_model', 'lgb_model']:
                if key_name in model_package:
                    final_model = model_package[key_name]
                    self.logger.info(f"通过键名 '{key_name}' 找到final_model")
                    break
                    
            if final_model is None:
                self.logger.error("无法找到最终模型，请检查模型包键名")
                return pd.DataFrame()
                
            lnn_model = model_package.get('lnn')
            market_submodels = {k: v for k, v in model_package.items() if '_submodel' in k}
            
            # 获取模型元数据
            model_metadata = model_package.get('metadata', {})
            final_feature_columns = model_metadata.get('final_feature_columns', [])
            categorical_features = model_metadata.get('categorical_features', [])
            trained_cat_dtypes = model_metadata.get('trained_cat_dtypes', {})
            label_encoders = model_metadata.get('label_encoders', {})
            
            # 备份date列，以防在处理过程中丢失
            if 'date' in test_df.columns:
                date_backup = test_df['date'].copy()
            
            # 3. 测试数据预处理
            # 3.1 检查测试数据是否包含必要的标签列
            target_col = 'is_top_performer'
            if target_col not in test_df.columns and 'future_return_2d' in test_df.columns:
                self.logger.info("测试数据中缺少is_top_performer标签，但有future_return_2d列，尝试创建标签...")
                try:
                    # 创建标签
                    labeled_test_df = self._create_rolling_top_performer_labels(test_df, 'future_return_2d')
                    test_df = labeled_test_df.copy()
                    has_labels = True
                    self.logger.info(f"成功为测试数据创建is_top_performer标签: {target_col in test_df.columns}")
                except Exception as e:
                    self.logger.warning(f"无法创建标签: {str(e)}")
                    has_labels = False
            else:
                has_labels = target_col in test_df.columns
            
            # 确保date列被保留
            if 'date' not in test_df.columns and 'date_backup' in locals():
                test_df['date'] = date_backup
                self.logger.info("从备份中恢复了date列")
            
            # 3.2 如果有LNN模型和序列长度大于0，创建序列特征
            seq_features = None
            if lnn_model is not None and model_package.get('seq_length', 0) > 0:
                try:
                    seq_length = model_package.get('seq_length', 10)
                    sectors = test_df['sector_code'].unique()
                    seq_features = self._create_sequence_features(
                        features_df, 
                        sector_list=sectors, 
                        end_date=date_obj, 
                        seq_length=seq_length
                    )
                except Exception as e:
                    self.logger.warning(f"创建序列特征失败: {str(e)}")
                    seq_features = None
            else:
                self.logger.warning("LNN模型不可用或序列长度为0，跳过序列特征创建")
            
            # 4. 应用市场环境模型
            self.logger.info("应用市场环境子模型...")
            market_preds = None
            
            # 获取市场特征
            market_features = model_metadata.get('market_features', [])
            if market_model is not None and market_features:
                try:
                    # 确保所有市场特征都存在于测试数据中
                    missing_features = [f for f in market_features if f not in test_df.columns]
                    if missing_features:
                        self.logger.warning(f"测试数据中缺少以下市场特征: {missing_features}")
                        # 添加缺失的特征列，填充为0
                        for feat in missing_features:
                            test_df[feat] = 0
                    
                    # 准备市场特征数据
                    market_X = test_df[market_features].fillna(0)
                    # 确保数据类型为数值型
                    market_X = market_X.astype(float)
                    
                    # 预测市场类别概率
                    market_preds = market_model.predict_proba(market_X)
                    
                    # 将市场预测添加到测试数据
                    for i in range(market_preds.shape[1]):
                        test_df[f'market_prob_{i}'] = market_preds[:, i]
                        
                    self.logger.info(f"市场模型预测成功，形状: {market_preds.shape}")
                except Exception as e:
                    self.logger.warning(f"市场模型预测失败: {str(e)}")
                    self.logger.warning(traceback.format_exc())
                    # 创建默认预测
                    market_pred_shape = (test_df.shape[0], 3)  # 假设有3个市场类别
                    market_preds = np.zeros(market_pred_shape)
                    market_preds[:, 1] = 1.0  # 默认中性市场
                    
                    # 添加到测试数据
                    for i in range(market_preds.shape[1]):
                        test_df[f'market_prob_{i}'] = market_preds[:, i]
            else:
                self.logger.warning("市场模型不可用或无市场特征")
                # 创建默认预测
                market_pred_shape = (test_df.shape[0], 3)  # 假设有3个市场类别
                market_preds = np.zeros(market_pred_shape)
                market_preds[:, 1] = 1.0  # 默认中性市场
                
                # 添加到测试数据
                for i in range(market_preds.shape[1]):
                    test_df[f'market_prob_{i}'] = market_preds[:, i]
            
            # 5. 应用强度预测模型
            strength_pred = None
            strength_features = model_metadata.get('strength_features', [])
            if strength_model is not None and strength_features:
                try:
                    # 确保所有强度特征都存在于测试数据中
                    missing_features = [f for f in strength_features if f not in test_df.columns]
                    if missing_features:
                        self.logger.warning(f"测试数据中缺少以下强度特征: {missing_features}")
                        # 添加缺失的特征列，填充为0
                        for feat in missing_features:
                            test_df[feat] = 0
                    
                    # 准备强度模型特征
                    strength_X = test_df[strength_features].fillna(0)
                    # 确保数据类型为数值型
                    strength_X = strength_X.astype(float)
                    
                    # 合并市场预测结果
                    market_prob_cols = [col for col in test_df.columns if col.startswith('market_prob_')]
                    if market_prob_cols:
                        for col in market_prob_cols:
                            if col not in strength_X.columns:
                                strength_X[col] = test_df[col]
                    
                    # 预测强度
                    strength_pred = strength_model.predict(strength_X)
                    test_df['predicted_strength'] = strength_pred
                    self.logger.info(f"强度模型预测成功，样本数: {len(strength_pred)}")
                except Exception as e:
                    self.logger.warning(f"强度模型预测失败: {str(e)}")
                    self.logger.warning(traceback.format_exc())
                    # 使用默认值
                    strength_pred = np.ones(len(test_df)) * 0.5
                    test_df['predicted_strength'] = strength_pred
            else:
                self.logger.warning("强度模型不可用或无强度特征，使用默认强度值0.5")
                strength_pred = np.ones(len(test_df)) * 0.5
                test_df['predicted_strength'] = strength_pred
            
            # 6. 准备最终模型的输入特征
            X_final = test_df.copy()
            
            # 获取模型元数据中的特征列，确保特征顺序和数量与训练时一致
            if final_feature_columns:
                # 检查是否有缺失的特征列
                missing_cols = [col for col in final_feature_columns if col not in X_final.columns]
                if missing_cols:
                    self.logger.warning(f"测试数据中缺少以下特征列: {missing_cols}")
                    
                    # 特殊处理LNN特征
                    if 'lnn_output_feature' in missing_cols and lnn_model is not None and seq_features is not None:
                        self.logger.info("尝试为测试数据生成真实的LNN特征...")
                        
                        # 创建一个字典来存储板块的LNN预测结果
                        test_lnn_outputs = {}
                        
                        # 遍历测试数据中的每个板块
                        for sector_code in test_df['sector_code'].unique():
                            if sector_code in seq_features:
                                try:
                                    # 提取序列特征
                                    seq_data = seq_features[sector_code]
                                    
                                    # 检查序列特征格式
                                    if isinstance(seq_data, dict) and 'features' in seq_data:
                                        sequence = seq_data['features']
                                    elif hasattr(seq_data, 'shape'):
                                        sequence = seq_data
                                    else:
                                        self.logger.warning(f"测试数据: 板块 {sector_code} 的序列特征格式无效")
                                        continue
                                        
                                    # 确保序列长度足够
                                    seq_length = model_package.get('seq_length', 10)
                                    if not hasattr(sequence, 'shape') or len(sequence) < seq_length:
                                        self.logger.warning(f"测试数据: 板块 {sector_code} 的序列特征长度不足")
                                        continue
                                        
                                    # 获取模型的Scaler
                                    scaler = model_package.get('scaler')
                                        
                                    # 使用与训练集相同的逻辑处理序列数据
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
                                    
                                    # 应用scaler
                                    if scaler is not None:
                                        try:
                                            seq_flat = recent_sequence.reshape(-1, recent_sequence.shape[-1])
                                            seq_scaled = scaler.transform(seq_flat)
                                            seq_tensor = torch.FloatTensor(seq_scaled.reshape(recent_sequence.shape)).unsqueeze(0)
                                        except Exception as e:
                                            self.logger.warning(f"测试数据: 缩放序列数据时出错: {e}")
                                            seq_tensor = torch.FloatTensor(recent_sequence).unsqueeze(0)
                                    else:
                                        seq_tensor = torch.FloatTensor(recent_sequence).unsqueeze(0)
                                        
                                    # 使用LNN模型预测
                                    try:
                                        with torch.no_grad():
                                            # 使用与训练集相同的预测逻辑
                                            try:
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
                                        continue
                                except Exception as e:
                                    self.logger.error(f"测试数据: 处理板块 {sector_code} 的LNN特征时出错: {e}")
                                    continue
                        
                        # 如果成功生成了LNN预测值，将其添加到测试数据
                        if test_lnn_outputs:
                            # 使用映射函数将LNN值添加到测试数据
                            def test_map_lnn_value(row):
                                sector = row['sector_code']
                                if sector in test_lnn_outputs:
                                    return test_lnn_outputs[sector]
                                return 0.0
                            
                            # 添加LNN特征列到测试数据
                            X_final['lnn_output_feature'] = test_df.apply(test_map_lnn_value, axis=1)
                            self.logger.info(f"成功为测试数据中的 {len(test_lnn_outputs)} 个板块添加真实LNN特征")
                            # 从missing_cols中移除lnn_output_feature
                            missing_cols.remove('lnn_output_feature')
                        else:
                            self.logger.warning("未能为测试数据生成任何LNN特征，将使用默认值0")
                            X_final['lnn_output_feature'] = 0.0
                    
                    # 特殊处理market_group特征
                    if 'market_group' in missing_cols:
                        self.logger.info("为测试数据添加market_group特征...")
                        # 使用字符串值而不是分类对象，避免转换错误
                        X_final['market_group'] = 'neutral'  # 使用默认值'neutral'
                        # 从missing_cols中移除market_group
                        missing_cols.remove('market_group')
                    
                    # 为其他缺失的列添加默认值0
                    for col in missing_cols:
                        X_final[col] = 0
                
                # 确保只使用模型需要的特征列，且顺序一致
                try:
                    X_final = X_final[final_feature_columns]
                    self.logger.info(f"使用模型指定的 {len(final_feature_columns)} 个特征列")
                except KeyError as e:
                    self.logger.error(f"获取特征列时出错: {e}")
                    # 如果出错，尝试使用所有非标签列作为特征
                    non_feature_cols = ['date', 'sector_code', 'is_top_performer', 'future_return_2d']
                    feature_cols = [col for col in X_final.columns if col not in non_feature_cols]
                    X_final = X_final[feature_cols]
                    self.logger.warning(f"使用所有可用特征列: {len(feature_cols)} 个")
            else:
                # 如果没有特征列信息，使用所有非标签列
                non_feature_cols = ['date', 'sector_code', 'is_top_performer', 'future_return_2d']
                non_feature_cols += [col for col in X_final.columns if col.startswith('future_return')]
                feature_cols = [col for col in X_final.columns if col not in non_feature_cols]
                X_final = X_final[feature_cols]
                self.logger.warning(f"模型元数据中缺少特征列信息，使用所有可用特征列: {len(feature_cols)} 个")
                
            # 确保分类特征的数据类型与训练时一致
            for cat_feat in categorical_features:
                if cat_feat in X_final.columns and cat_feat in trained_cat_dtypes:
                    try:
                        # 特殊处理market_group特征，避免类型转换错误
                        if cat_feat == 'market_group':
                            self.logger.info("使用安全方式处理market_group特征")
                            # 将market_group转为字符串类型，避免分类转换问题
                            X_final[cat_feat] = X_final[cat_feat].astype(str)
                            continue
                            
                        dtype = trained_cat_dtypes[cat_feat]
                        X_final[cat_feat] = X_final[cat_feat].astype(dtype)
                    except Exception as e:
                        self.logger.warning(f"无法转换分类特征 {cat_feat} 的数据类型: {str(e)}")
            
            # 应用标签编码器
            for col, encoder in label_encoders.items():
                if col in X_final.columns:
                    try:
                        # 确保处理未见过的值
                        unique_vals = X_final[col].unique()
                        for val in unique_vals:
                            if val not in encoder.classes_:
                                # 对未见过的值进行特殊处理，例如使用最常见的类别
                                self.logger.warning(f"列 {col} 中存在未见过的值: {val}，将替换为最常见类别")
                                X_final.loc[X_final[col] == val, col] = encoder.classes_[0]
                        
                        X_final[col] = encoder.transform(X_final[col])
                    except Exception as e:
                        self.logger.warning(f"标签编码失败 {col}: {str(e)}")
            
            # 7. 使用最终模型进行预测
            final_pred = None
            try:
                # 1. 检查特征列是否与训练时一致
                if final_feature_columns and len(final_feature_columns) > 0:
                    # 确保预测数据中包含所有需要的特征列，顺序与训练时一致
                    missing_cols = [col for col in final_feature_columns if col not in X_final.columns]
                    if missing_cols:
                        self.logger.warning(f"预测数据中缺少 {len(missing_cols)}/{len(final_feature_columns)} 特征列")
                        # 如果缺失的列太多，可能会影响预测质量
                        if len(missing_cols) > len(final_feature_columns) * 0.3:  # 超过30%特征缺失
                            self.logger.error(f"缺失特征过多，可能会严重影响预测质量")
                        
                        # 为缺失的列添加默认值0
                        for col in missing_cols:
                            X_final[col] = 0
                    
                    # 确保按照训练时的列顺序选择特征
                    X_final = X_final[final_feature_columns]
                
                # 2. 处理分类特征编码
                if isinstance(categorical_features, list) and categorical_features:
                    for cat_feat in categorical_features:
                        if cat_feat in X_final.columns:
                            # 尝试将分类特征转换为符合训练时的格式
                            if cat_feat in trained_cat_dtypes:
                                try:
                                    # 记录转换前的数据类型
                                    before_dtype = str(X_final[cat_feat].dtype)
                                    
                                    # 获取训练时的数据类型
                                    train_dtype = trained_cat_dtypes[cat_feat]
                                    
                                    # 转换类型
                                    X_final[cat_feat] = X_final[cat_feat].astype(train_dtype)
                                    self.logger.info(f"成功转换特征 {cat_feat} 的数据类型: {before_dtype} -> {train_dtype}")
                                except Exception as e:
                                    self.logger.warning(f"转换特征 {cat_feat} 的数据类型时出错: {e}")
                
                # 3. 确保数据类型一致，对所有列进行类型检查
                for col in X_final.columns:
                    # 检查是否为空值，如果是则填充为0
                    if X_final[col].isnull().any():
                        X_final[col] = X_final[col].fillna(0)
                    
                    # 确保是数值类型
                    if not pd.api.types.is_numeric_dtype(X_final[col]):
                        try:
                            X_final[col] = pd.to_numeric(X_final[col], errors='coerce').fillna(0)
                        except Exception as e:
                            self.logger.warning(f"将列 {col} 转换为数值类型时出错: {e}")
                
                # 4. 尝试不同的预测方法，增加健壮性
                # 尝试直接使用booster进行预测，设置predict_disable_shape_check=True
                if hasattr(final_model, 'booster_') and hasattr(final_model.booster_, 'predict'):
                    self.logger.info("使用LightGBM booster直接预测，设置predict_disable_shape_check=True")
                    try:
                        final_pred = final_model.booster_.predict(X_final.values, 
                                                            pred_leaf=False, 
                                                            pred_contrib=False,
                                                            predict_disable_shape_check=True)
                    except Exception as e1:
                        self.logger.warning(f"使用booster.predict方法失败: {e1}, 尝试其他方法")
                
                # 如果上面的方法失败，尝试使用predict_proba
                if final_pred is None and hasattr(final_model, 'predict_proba'):
                    self.logger.info("使用predict_proba方法预测")
                    try:
                        # 对于LightGBM模型，先修改_Booster的predict参数，再调用predict_proba
                        if hasattr(final_model, '_Booster'):
                            # 临时保存原方法
                            original_predict = final_model._Booster.predict
                            # 创建带有predict_disable_shape_check的包装方法
                            def wrapped_predict(*args, **kwargs):
                                kwargs['predict_disable_shape_check'] = True
                                return original_predict(*args, **kwargs)
                            # 替换方法
                            final_model._Booster.predict = wrapped_predict
                            
                        # 尝试预测
                        final_pred = final_model.predict_proba(X_final)
                        
                        # 恢复原方法
                        if hasattr(final_model, '_Booster'):
                            final_model._Booster.predict = original_predict
                            
                        # 对于二分类问题，取正类的概率
                        if hasattr(final_pred, 'shape') and final_pred.shape[1] > 1:
                            final_pred = final_pred[:, 1]
                    except Exception as e2:
                        self.logger.warning(f"使用predict_proba方法失败: {e2}, 尝试其他方法")
                
                # 如果前两种方法都失败，尝试使用标准predict方法
                if final_pred is None:
                    self.logger.info("使用predict方法预测")
                    try:
                        final_pred = final_model.predict(X_final)
                    except Exception as e3:
                        self.logger.error(f"所有预测方法都失败: {e3}")
                        # 使用默认值
                        final_pred = np.zeros(len(X_final))
            except Exception as e:
                self.logger.error(f"最终模型预测过程中发生错误: {str(e)}")
                self.logger.error(traceback.format_exc())
                # 使用默认值
                final_pred = np.zeros(len(X_final))
            
            # 8. 结果整合与评估
            result_df = pd.DataFrame()
            result_df['sector_code'] = test_df['sector_code']
            result_df['prediction'] = final_pred
            
            # 如果有sector_name列，也包含它
            if 'sector_name' in test_df.columns:
                result_df['sector_name'] = test_df['sector_name']
            
            # 排序并添加排名
            result_df = result_df.sort_values(by='prediction', ascending=False)
            result_df['rank'] = range(1, len(result_df) + 1)
            
            # 计算评估指标（如果有实际标签）
            metrics = {}
            if has_labels:
                try:
                    # 计算预测的精确度
                    result_df['actual'] = test_df[target_col]
                    
                    # 计算准确率(二分类)
                    accuracy = accuracy_score(result_df['actual'], result_df['prediction'] > 0.5)
                    # 计算命中率
                    hit_rate = np.mean((result_df['prediction'] > 0.5) & result_df['actual'])
                    
                    # 计算其他评估指标
                    metrics['accuracy'] = accuracy
                    metrics['hit_rate'] = hit_rate
                    
                    # 计算前N名的命中情况
                    actual_top = test_df[test_df[target_col] == 1]['sector_code'].unique()
                    n_actual_top = len(actual_top)
                    predicted_top = result_df.head(n_actual_top)['sector_code'].tolist()
                    
                    # 计算交集数量
                    correct = len(set(actual_top) & set(predicted_top))
                    
                    # 计算精确率、召回率等
                    precision = correct / len(predicted_top) if len(predicted_top) > 0 else 0
                    recall = correct / len(actual_top) if len(actual_top) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    metrics['precision'] = precision
                    metrics['recall'] = recall
                    metrics['f1'] = f1
                    metrics['top_n_correct'] = correct
                    metrics['top_n_total'] = n_actual_top
                    
                    # 记录正确预测的板块
                    metrics['correct_sectors'] = list(set(actual_top) & set(predicted_top))
                    metrics['predicted_top_sectors'] = predicted_top
                    metrics['actual_top_sectors'] = list(actual_top)
                    
                    self.logger.info(f"预测评估指标: 准确率={accuracy:.4f}, 命中率={hit_rate:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}")
                    self.logger.info(f"前{n_actual_top}名板块中正确预测了{correct}个")
                except Exception as e:
                    self.logger.error(f"计算评估指标时出错: {e}")
                    self.logger.error(traceback.format_exc())
            
            # 返回更丰富的结果信息
            return {
                'predictions': result_df,
                'test_df': test_df,
                'metrics': metrics,
                'date': date_obj
            }
            
        except Exception as e:
            self.logger.error(f"预测过程中发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
'''
    
    # 读取原始文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找添加位置
    # 插入到_create_sequence_features函数前面
    pattern = r'def _create_sequence_features\('
    match = re.search(pattern, content)
    
    if match:
        # 获取匹配的起始位置
        pos = match.start()
        # 在该位置之前插入函数
        new_content = content[:pos] + function_to_add + "\n\n    " + content[pos:]
        
        # 写入新文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("函数添加成功！")
    else:
        print("未找到插入位置，请手动添加函数。")

if __name__ == "__main__":
    add_function_to_file() 