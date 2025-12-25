"""
序列数据结构问题修复代码示例

此文件展示了如何修复train_sector_models.py中的序列数据结构问题
请手动将此代码段替换到train_sector_models.py的相应位置
"""

# 在train_sector_models.py文件中找到以下代码段:
"""
if lnn_model is not None and sequence_train_data is not None:
    self.logger.info("尝试为最终模型生成LNN特征...")
    try:
        # 判断序列数据的结构
        if 'features' in sequence_train_data and 'labels' in sequence_train_data:
            # 使用新的结构格式
            # 使用序列模型进行预测
            lnn_predictions = lnn_model.predict(sequence_train_data['features'])
            
            # ... 现有代码 ...
            
        else:
            self.logger.warning("序列数据结构不包含'features'和'labels'，无法生成LNN特征")
    except Exception as e:
        self.logger.error(f"生成LNN特征时出错: {e}\n{traceback.format_exc()}")
"""

# 将其替换为:
"""
if lnn_model is not None and sequence_train_data is not None:
    self.logger.info("尝试为最终模型生成LNN特征...")
    try:
        # 判断序列数据的结构
        if 'features' in sequence_train_data and 'labels' in sequence_train_data:
            # 使用新的结构格式
            # 使用序列模型进行预测
            lnn_predictions = lnn_model.predict(sequence_train_data['features'])
            
            if isinstance(lnn_predictions, np.ndarray):
                final_lnn_feat_val = lnn_predictions.flatten() if lnn_predictions.ndim > 1 else lnn_predictions
                
                # 通过日期和板块代码匹配LNN预测结果和原始DataFrame
                sequence_dates = sequence_train_data['dates']
                sequence_sector_codes = sequence_train_data['sector_codes']
                
                # 为所有在序列数据中的行设置LNN特征
                lnn_results_df = pd.DataFrame({
                    'date': sequence_dates,
                    'sector_code': sequence_sector_codes,
                    lnn_out_feat_name: final_lnn_feat_val
                })
                
                # 如果日期是字符串格式，转换回datetime以便合并
                if lnn_results_df['date'].dtype == 'object':
                    try:
                        lnn_results_df['date'] = pd.to_datetime(lnn_results_df['date'])
                    except Exception as e:
                        self.logger.warning(f"无法将日期列转换为datetime: {e}, 将尝试直接合并")
                
                # 确保temp_X_for_final_df中的date列是datetime类型
                if 'date' in temp_X_for_final_df.columns and temp_X_for_final_df['date'].dtype != 'datetime64[ns]':
                    try:
                        temp_X_for_final_df['date'] = pd.to_datetime(temp_X_for_final_df['date'])
                    except Exception as e:
                        self.logger.warning(f"无法将temp_X_for_final_df中的日期列转换为datetime: {e}")
                
                # 合并到原始数据
                temp_X_for_final_df = temp_X_for_final_df.merge(
                    lnn_results_df, 
                    on=['date', 'sector_code'], 
                    how='left'
                )
                
                # 填充缺失值
                if f"{lnn_out_feat_name}_x" in temp_X_for_final_df.columns:
                    temp_X_for_final_df[lnn_out_feat_name] = temp_X_for_final_df[f"{lnn_out_feat_name}_y"].fillna(temp_X_for_final_df[f"{lnn_out_feat_name}_x"])
                    temp_X_for_final_df.drop([f"{lnn_out_feat_name}_x", f"{lnn_out_feat_name}_y"], axis=1, inplace=True)
            else:
                self.logger.warning("LNN预测结果不是numpy数组")
        # 检查是否为嵌套字典结构（即_create_sequence_features的输出格式）
        elif isinstance(sequence_train_data, dict) and len(sequence_train_data) > 0 and all(isinstance(v, dict) and 'features' in v for v in sequence_train_data.values()):
            self.logger.info("检测到嵌套字典结构的序列数据，尝试合并处理...")
            # 收集所有板块的特征、标签和日期
            all_features = []
            all_labels = []
            all_dates = []
            all_sector_codes = []
            
            for sector_code, sector_data in sequence_train_data.items():
                if 'features' in sector_data and 'labels' in sector_data and 'dates' in sector_data:
                    features = sector_data['features']
                    labels = sector_data['labels']
                    dates = sector_data['dates']
                    
                    # 确保数据格式正确
                    if isinstance(features, np.ndarray) and len(features) > 0:
                        n_samples = len(features)
                        all_features.append(features)
                        all_labels.append(labels)
                        all_dates.append(dates)
                        all_sector_codes.extend([sector_code] * n_samples)
            
            if all_features:
                # 合并所有板块的数据
                combined_features = np.vstack(all_features)
                combined_labels = np.concatenate(all_labels)
                combined_dates = np.concatenate(all_dates)
                
                self.logger.info(f"合并后的序列特征形状: {combined_features.shape}")
                
                # 使用合并后的数据进行预测
                lnn_predictions = lnn_model.predict(combined_features)
                
                if isinstance(lnn_predictions, np.ndarray):
                    final_lnn_feat_val = lnn_predictions.flatten() if lnn_predictions.ndim > 1 else lnn_predictions
                    
                    # 创建结果DataFrame
                    lnn_results_df = pd.DataFrame({
                        'date': combined_dates,
                        'sector_code': all_sector_codes,
                        lnn_out_feat_name: final_lnn_feat_val
                    })
                    
                    # 如果日期是字符串格式，转换回datetime以便合并
                    if lnn_results_df['date'].dtype == 'object':
                        try:
                            lnn_results_df['date'] = pd.to_datetime(lnn_results_df['date'])
                        except Exception as e:
                            self.logger.warning(f"无法将日期列转换为datetime: {e}, 将尝试直接合并")
                    
                    # 确保temp_X_for_final_df中的date列是datetime类型
                    if 'date' in temp_X_for_final_df.columns and temp_X_for_final_df['date'].dtype != 'datetime64[ns]':
                        try:
                            temp_X_for_final_df['date'] = pd.to_datetime(temp_X_for_final_df['date'])
                        except Exception as e:
                            self.logger.warning(f"无法将temp_X_for_final_df中的日期列转换为datetime: {e}")
                    
                    # 合并到原始数据
                    temp_X_for_final_df = temp_X_for_final_df.merge(
                        lnn_results_df, 
                        on=['date', 'sector_code'], 
                        how='left'
                    )
                    
                    # 填充缺失值
                    if f"{lnn_out_feat_name}_x" in temp_X_for_final_df.columns:
                        temp_X_for_final_df[lnn_out_feat_name] = temp_X_for_final_df[f"{lnn_out_feat_name}_y"].fillna(temp_X_for_final_df[f"{lnn_out_feat_name}_x"])
                        temp_X_for_final_df.drop([f"{lnn_out_feat_name}_x", f"{lnn_out_feat_name}_y"], axis=1, inplace=True)
                else:
                    self.logger.warning("LNN预测结果不是numpy数组")
            else:
                self.logger.warning("合并后的序列特征为空，无法生成LNN特征")
        else:
            self.logger.warning("序列数据结构不包含'features'和'labels'，无法生成LNN特征")
    except Exception as e:
        self.logger.error(f"生成LNN特征时出错: {e}\n{traceback.format_exc()}")
""" 