# 在_train_rolling_day_model函数中创建LNN模型的代码

# 在sequence_flat_scaled = scaler.transform(sequence_flat)和sequence_features_scaled = sequence_flat_scaled.reshape(sequence_features.shape)之后添加：

# 在验证数据准备完成后，fit_kwargs_lnn创建前，添加以下代码
"""
# 在这里初始化LNN模型
try:
    # 获取特征维度
    input_size = sequence_features.shape[2]  # 特征维度
    output_size = 1  # 输出维度 (通常是1，代表预测值)
    hidden_size = 64  # 隐藏层大小
    num_layers = 2  # 层数
    
    # 检测CUDA是否可用
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建LNN模型实例
    lnn_model = self.LiquidNetModel(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device
    )
    
    self.logger.info(f"成功创建LNN模型，输入维度: {input_size}, 输出维度: {output_size}, 设备: {device}")
except Exception as e_lnn_init:
    self.logger.error(f"创建LNN模型时出错: {e_lnn_init}")
    self.logger.error(traceback.format_exc())
    lnn_model = None
    use_lnn = False
"""
