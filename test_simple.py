import pickle
import os
import numpy as np
import pandas as pd

# 加载模型
model_path = 'models/TGN区块链/20230401.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 创建测试特征
feature_cols = model['feature_columns']
num_stocks = 5
stocks = [f'00000{i}.SZ' for i in range(1, num_stocks+1)]
data = np.random.randn(num_stocks, len(feature_cols))
features = pd.DataFrame(data, index=stocks, columns=feature_cols)

print("创建的测试特征:")
print(features.head())

# 使用模型预测
model_short = model['model_short']
short_probs = model_short.predict_proba(features.values)[:, 1]

model_long = model['model_long']
long_probs = model_long.predict_proba(features.values)[:, 1]

# 打印结果
print('\n股票预测概率:')
print('股票代码  短期概率  长期概率')
for i, stock in enumerate(stocks):
    print(f'{stock}: {short_probs[i]:.4f}  {long_probs[i]:.4f}') 