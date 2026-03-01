"""
Fix analyze_pred_buckets mode to use correct global Top1 logic
"""
import re

# Read the file
with open(r'main_alpha原始.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the section to replace - looking for the bucket calculation part
# Pattern: find the section starting with "# 为每个桶的股票计算收益" or similar
pattern_start = r'# 为每个桶的股票计算收益\s+#    bucket_results = \{\}'
pattern_end = r'bucket_results\[label\] = pd\.DataFrame\(results\)'

if not re.search(pattern_start, content):
    print("Pattern start not found")
    exit(1)

if not re.search(pattern_end, content):
    print("Pattern end not found")
    exit(1)

# Extract the section before the bucket calculation
sections = content.split(pattern_start)
if len(sections) != 2:
    print(f"Expected 2 sections, got {len(sections)}")
    exit(1)

before_section = sections[0]
after_section = sections[1]

# Build the new section
new_section = '''    # ========================================================================
    # 步骤4: 计算T+1/T+2/T+3实际收益（整体Top1逻辑）
    # ========================================================================
    logger.info("\\n[REFRESH] [步骤 4/7] 计算T+1/T+2/T+3实际收益（整体Top1）...")

    holding_days = 3  # 默认3天（T+1, T+2, T+3）

    # 获取测试集的最后日期，用于过滤买入日期
    test_end_date = prediction_data['date'].max()
    logger.info(f"  [INFO] 测试集实际结束日期: {test_end_date.strftime('%Y-%m-%d')}")
    logger.info(f"  [INFO] 将只计算买入日期 <= {test_end_date - timedelta(days=holding_days)} 的记录")

    # **关键步骤1**: 在整个测试集上按日期分组，每天取pred_top1最高的股票（整体Top1）
    logger.info("\\n[SELECT] 步骤4.1: 在整个测试集上选择每日Top1股票...")
    daily_top1_overall = prediction_data.sort_values(['date', 'pred_top1'], ascending=[False, False]).groupby('date').head(1)
    logger.info(f"  [OK] 测试集共有 {len(daily_top1_overall)} 个每日Top1股票")

    # **关键步骤2**: 为这些Top1股票计算T+1/T+2/T+3收益
    logger.info("\\n[SELECT] 步骤4.2: 为Top1股票计算T+1/T+2/T+3收益...")

    all_top1_results = []
    skipped_count = 0

    for buy_date, top1_row in daily_top1_overall.iterrows():
        stock_code = top1_row['stock_code']
        pred_score = top1_row['pred_top1']
        pred_bucket = top1_row['pred_bucket']
        buy_close = top1_row['close']

        # 跳过买入日期过晚的记录（目标日期会超出测试集）
        if buy_date > test_end_date - timedelta(days=holding_days):
            skipped_count += 1
            continue

        # 转换buy_date为datetime类型
        if isinstance(buy_date, pd.Timestamp):
            buy_date_dt = buy_date.to_pydatetime()
        elif isinstance(buy_date, datetime):
            buy_date_dt = buy_date
        else:
            continue

        # 计算各持有期的收益
        for day_offset in range(1, holding_days + 1):  # T+1, T+2, T+3
            target_date = buy_date_dt + timedelta(days=day_offset)
            target_date_ts = pd.Timestamp(target_date)

            # 检查目标日期是否在测试集中（使用pd.Timestamp进行比较）
            if target_date_ts in prediction_data['date'].values:
                target_row = prediction_data[prediction_data['date'] == target_date_ts]

                if stock_code in target_row['stock_code'].values:
                    stock_target_row = target_row[target_row['stock_code'] == stock_code].iloc[0]
                    target_close = stock_target_row['close']

                    # 计算该期的实际收益
                    actual_return = (target_close - buy_close) / buy_close

                    all_top1_results.append({
                        'date': buy_date,
                        'stock_code': stock_code,
                        'pred_top1': pred_score,
                        'pred_bucket': pred_bucket,
                        'holding_day': day_offset,  # 1=T+1, 2=T+2, 3=T+3
                        'target_date': target_date_ts,
                        'actual_return': actual_return
                    })
                    break  # 找到T+1就跳出，不再计算T+2/T+3（与run_pred_stocks逻辑一致）

    logger.info(f"  [OK] 成功计算 {len(all_top1_results)} 条Top1股票的收益记录")
    logger.info(f"  [INFO] 跳过 {skipped_count} 条记录（买入日期过晚）")

    # **关键步骤3**: 按桶分组统计这些Top1股票的表现
    logger.info("\\n[SELECT] 步骤4.3: 按桶分组统计Top1股票表现...")

    bucket_results = {}
    for label in bin_labels:
        # 筛选该桶的Top1股票记录
        bucket_df = pd.DataFrame([r for r in all_top1_results if r['pred_bucket'] == label])
        bucket_results[label] = bucket_df
        logger.info(f"     [OK] {label} 桶: {len(bucket_df)} 条Top1股票记录")

    logger.info("=" * 80)
'''

# Replace the old section with new section
new_content = before_section + new_section + after_section

# Write back
with open(r'main_alpha原始.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
    print('Successfully updated main_alpha原始.py with correct global Top1 logic')
