"""
Debug script for analyze_pred_buckets mode
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle

print("=" * 80)
print("调试脚本 - 预测分桶分析")
print("=" * 80)

# Step 1: Load data
print("\n[步骤 1] 加载数据...")
cache_file = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\cache\stock_features_20230601_20260212.parquet")
df = pd.read_parquet(cache_file)
print(f"加载的完整数据: {len(df)} 条记录")

# Filter test set
test_start = pd.to_datetime('20250605')
test_end = pd.to_datetime('20260203')
prediction_data = df[df['date'] >= test_start].copy()
print(f"测试集数据: {len(prediction_data)} 条记录")
print(f"日期范围: {prediction_data['date'].min()} 到 {prediction_data['date'].max()}")
print(f"列名: {prediction_data.columns.tolist()}")

# Step 2: Load model
print("\n[步骤 2] 加载模型...")
model_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\results\lightgbm_model_window3_20260107_033514.pkl")
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_columns = model_data.get('feature_names', [])
if not feature_columns:
    if hasattr(model, 'feature_name'):
        feature_columns = model.feature_name()
    else:
        print("ERROR: 无法获取特征列")
        exit(1)

print(f"模型加载成功，特征数: {len(feature_columns)}")

# Step 3: Generate predictions
print("\n[步骤 3] 生成预测...")
X_pred = prediction_data[feature_columns]
predictions = model.predict(X_pred)
prediction_data['pred_top1'] = predictions
print(f"预测完成，pred_top1 范围: {predictions.min():.4f} ~ {predictions.max():.4f}")

# Step 4: Create buckets
print("\n[步骤 4] 创建分桶...")
bin_edges = np.percentile(prediction_data['pred_top1'], [0, 20, 40, 60, 80, 100])
bin_labels = ['P0~P20', 'P20~P40', 'P40~P60', 'P60~P80', 'P80~P100']
prediction_data['pred_bucket'] = pd.cut(prediction_data['pred_top1'], bins=bin_edges, labels=bin_labels, include_lowest=True)

for label in bin_labels:
    count = len(prediction_data[prediction_data['pred_bucket'] == label])
    print(f"  {label}: {count} 条股票")

# Step 5: Check one bucket
print("\n[步骤 5] 检查 P0~P20 桶...")
label = 'P0~P20'
bucket_data = prediction_data[prediction_data['pred_bucket'] == label].copy()
print(f"P0~P20 桶: {len(bucket_data)} 条股票")
print(f"列名: {bucket_data.columns.tolist()}")

# 按日期和pred_top1排序
bucket_data_sorted = bucket_data.sort_values(['date', 'pred_top1'], ascending=[False, False])

# 按日期分组，每天取Top1
daily_top1 = bucket_data_sorted.groupby('date').head(1)
print(f"daily_top1: {len(daily_top1)} 条记录")

# 打印前5条
print("\n前5条 daily_top1 记录:")
print(daily_top1[['date', 'stock_code', 'pred_top1', 'close']].head(5))

# Step 6: Test date matching for first record
print("\n[步骤 6] 测试日期匹配...")
holding_days = 3

first_record = daily_top1.iloc[0]
buy_date = first_record.name  # This is the index (date)
stock_code = first_record['stock_code']
buy_close = first_record['close']
pred_score = first_record['pred_top1']

print(f"第一条记录:")
print(f"  buy_date: {buy_date} (type: {type(buy_date)})")
print(f"  stock_code: {stock_code}")
print(f"  buy_close: {buy_close}")
print(f"  pred_score: {pred_score}")

# 转换buy_date为datetime
if isinstance(buy_date, pd.Timestamp):
    buy_date_dt = buy_date.to_pydatetime()
elif isinstance(buy_date, datetime):
    buy_date_dt = buy_date
else:
    print(f"ERROR: 无效的buy_date类型: {type(buy_date)}")
    exit(1)

# 测试 T+1, T+2, T+3
for day_offset in range(1, holding_days + 1):
    target_date = buy_date_dt + timedelta(days=day_offset)
    target_date_ts = pd.Timestamp(target_date)

    print(f"\nT+{day_offset}:")
    print(f"  target_date (datetime): {target_date}")
    print(f"  target_date_ts (Timestamp): {target_date_ts}")
    print(f"  target_date_ts in prediction_data['date'].values: {target_date_ts in prediction_data['date'].values}")

    if target_date_ts in prediction_data['date'].values:
        target_row = prediction_data[prediction_data['date'] == target_date_ts]
        print(f"  ✓ 目标日期存在!")
        print(f"  target_row 长度: {len(target_row)}")
        print(f"  stock_code in target_row['stock_code'].values: {stock_code in target_row['stock_code'].values}")

        if stock_code in target_row['stock_code'].values:
            print(f"  ✓ 股票 {stock_code} 存在于目标日期!")
            stock_target_row = target_row[target_row['stock_code'] == stock_code].iloc[0]
            target_close = stock_target_row['close']
            actual_return = (target_close - buy_close) / buy_close
            print(f"  ✓ 可以计算收益! return={actual_return:.4f}")
        else:
            print(f"  ✗ 股票 {stock_code} 不在目标日期的数据中")
            print(f"  目标日期的股票数: {len(target_row)}")
            print(f"  前5个stock_code: {target_row['stock_code'].head(5).tolist()}")
    else:
        print(f"  ✗ 目标日期不在测试集中")
        print(f"  测试集最后5个日期: {sorted(prediction_data['date'].unique())[-5:]}")

print("\n" + "=" * 80)
print("调试完成")
print("=" * 80)
