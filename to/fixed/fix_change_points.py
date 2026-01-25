"""
修复变化点数据验证和处理相关问题
"""

# 以下是需要替换的代码，位于to/train_sector_models.py中calculate_stock_sector_relative_features函数内
# 替换原有的变化点验证逻辑，从3820行左右开始

# 原代码:
'''
if change_points:
    # 只处理变化点对应的日期
    for stock_code, cp_data in change_points.items():
        if cp_data and 'dates' in cp_data:
            dates = cp_data['dates']
            if isinstance(dates, np.ndarray) and dates.any() or dates:
                for date_str in dates:
                    process_combinations.append((stock_code, date_str))
    
    self.logger.info(f"将处理 {len(process_combinations)} 个变化点对应的股票-日期组合")
else:
    # 如果没有提供变化点，则处理所有组合
    self.logger.warning("未提供变化点数据，将处理所有股票-日期组合")
    stock_groups = result_df.groupby(['stock_code', 'date_str'])
    process_combinations = [(stock_code, date_str) for (stock_code, date_str) in stock_groups.groups.keys()]
'''

# 修复后的代码:
'''
# 确定需要处理的股票-日期组合
process_combinations = []

# 修复问题1: 详细检查变化点数据的有效性
has_valid_change_points = False
if change_points is not None:
    self.logger.info(f"开始检查变化点数据 - 变化点类型: {type(change_points)}, 长度: {len(change_points) if hasattr(change_points, '__len__') else '未知'}")
    # 输出变化点的数据结构，以便调试
    try:
        example_keys = list(change_points.keys())[:3]  # 获取最多3个键用于示例
        self.logger.info(f"变化点数据格式示例 (前{len(example_keys)}个键): {example_keys}")
        
        for key in example_keys:
            cp_item = change_points.get(key, {})
            self.logger.info(f"股票 {key} 的变化点数据: {type(cp_item)}, 键: {list(cp_item.keys()) if isinstance(cp_item, dict) else '非字典'}")
            if isinstance(cp_item, dict) and 'dates' in cp_item:
                date_example = cp_item['dates']
                self.logger.info(f"dates 数据类型: {type(date_example)}, 长度: {len(date_example) if hasattr(date_example, '__len__') else '未知'}")
                if isinstance(date_example, list) or isinstance(date_example, np.ndarray):
                    self.logger.info(f"前几个日期示例: {date_example[:3] if len(date_example) > 0 else '空'}")
    except Exception as e:
        self.logger.error(f"检查变化点数据结构时出错: {str(e)}")
        self.logger.error(traceback.format_exc())
    
    # 检查是否有任何有效的变化点
    valid_cp_count = 0
    empty_cp_count = 0
    invalid_cp_count = 0
    for stock_code, cp_data in change_points.items():
        if cp_data and isinstance(cp_data, dict) and 'dates' in cp_data:
            dates = cp_data['dates']
            if isinstance(dates, list) and dates:
                has_valid_change_points = True
                valid_cp_count += 1
                for date_str in dates:
                    process_combinations.append((stock_code, date_str))
            elif isinstance(dates, np.ndarray) and dates.size > 0:
                has_valid_change_points = True
                valid_cp_count += 1
                for date_str in dates:
                    process_combinations.append((stock_code, date_str))
            else:
                empty_cp_count += 1
        else:
            invalid_cp_count += 1
    
    self.logger.info(f"变化点数据统计: 有效变化点股票: {valid_cp_count}, 空日期列表: {empty_cp_count}, 无效格式: {invalid_cp_count}")
    
    if has_valid_change_points:
        self.logger.info(f"将处理 {len(process_combinations)} 个变化点对应的股票-日期组合")
    else:
        self.logger.warning("未找到有效的变化点数据，将处理所有股票-日期组合")
        has_valid_change_points = False
else:
    self.logger.warning("未提供变化点参数，将处理所有股票-日期组合")

if not has_valid_change_points:
    # 如果没有提供变化点，则处理所有组合
    self.logger.warning("将处理所有股票-日期组合")
    stock_groups = result_df.groupby(['stock_code', 'date_str'])
    process_combinations = [(stock_code, date_str) for (stock_code, date_str) in stock_groups.groups.keys()]
'''

# ===============================================================
# 批处理优化及进度日志改进代码
# ===============================================================

# 原代码:
'''
# 批处理，每批处理1000个组合
batch_size = 1000
total_groups = len(process_combinations)
processed = 0

# 创建结果字典
results = {}

# 按批次处理
for batch_start in range(0, total_groups, batch_size):
    batch_end = min(batch_start + batch_size, total_groups)
    batch = process_combinations[batch_start:batch_end]
    
    # 使用线程池并行处理批次
    with ThreadPoolExecutor(max_workers=min(20, os.cpu_count() * 5)) as executor:
        # 提交所有任务
        futures_dict = {}
        for stock_code, date_str in batch:
            future = executor.submit(
                self._process_single_stock_date, 
                stock_code, 
                date_str, 
                result_df_idx, 
                stock_sector_mapping, 
                sector_features_df_idx
            )
            futures_dict[future] = (stock_code, date_str)
        
        # 收集结果
        for future in as_completed(futures_dict):
            try:
                result = future.result()
                if result:
                    key = (result['stock_code'], result['date'])
                    results[key] = result
            except Exception as e:
                stock_code, date_str = futures_dict[future]
                self.logger.error(f"处理股票 {stock_code} 在 {date_str} 的特征时出错: {e}")
    
    # 更新处理进度
    processed += len(batch)
    self.logger.info(f"已处理 {processed}/{total_groups} 个股票-日期组合 ({processed/total_groups:.1%})")
'''

# 修复后的代码:
'''
# 批处理，每批处理1000个组合
batch_size = 1000
total_groups = len(process_combinations)
processed = 0

# 创建结果字典
results = {}

# 添加进度日志 - 修复问题3
self.logger.info(f"开始处理 {total_groups} 个股票-日期组合，每批次 {batch_size} 个...")
start_time = time.time()

# 按批次处理
for batch_start in range(0, total_groups, batch_size):
    batch_end = min(batch_start + batch_size, total_groups)
    batch = process_combinations[batch_start:batch_end]
    
    batch_start_time = time.time()
    self.logger.info(f"开始处理第 {batch_start//batch_size + 1} 批，共 {len(batch)} 个组合...")
    
    # 使用线程池并行处理批次
    with ThreadPoolExecutor(max_workers=min(20, os.cpu_count() * 5)) as executor:
        # 提交所有任务
        futures_dict = {}
        for stock_code, date_str in batch:
            future = executor.submit(
                self._process_single_stock_date, 
                stock_code, 
                date_str, 
                result_df_idx, 
                stock_sector_mapping, 
                sector_features_df_idx
            )
            futures_dict[future] = (stock_code, date_str)
        
        # 收集结果
        batch_success = 0
        batch_errors = 0
        for future in as_completed(futures_dict):
            try:
                result = future.result()
                if result:
                    key = (result['stock_code'], result['date'])
                    results[key] = result
                    batch_success += 1
            except Exception as e:
                stock_code, date_str = futures_dict[future]
                self.logger.error(f"处理股票 {stock_code} 在 {date_str} 的特征时出错: {str(e)}")
                batch_errors += 1
    
    # 更新处理进度
    processed += len(batch)
    batch_time = time.time() - batch_start_time
    self.logger.info(f"第 {batch_start//batch_size + 1} 批处理完成：成功 {batch_success}，失败 {batch_errors}，用时 {batch_time:.2f} 秒")
    self.logger.info(f"总进度: 已处理 {processed}/{total_groups} 个股票-日期组合 ({processed/total_groups:.1%})")

total_time = time.time() - start_time
self.logger.info(f"所有批次处理完成，总用时 {total_time:.2f} 秒，平均每个组合 {total_time/total_groups:.4f} 秒")
'''

# ===============================================================
# _process_single_stock_date函数中的Series truth value问题修复
# ===============================================================

'''
# 修复问题2：在处理股票与板块的相关性时，确保所有比较的值都是标量
# 原代码中的问题部分（注意修改前）:
if not np.isnan(stock_change) and not np.isnan(sector_change):
    return_diff = abs(stock_change - sector_change)
    return_correlation = 1.0 / (1.0 + return_diff)

# 修复后的代码:
if not (np.isnan(stock_change) or np.isnan(sector_change)):  # 修复：使用逻辑运算符而不是 and
    return_diff = abs(stock_change - sector_change)
    return_correlation = 1.0 / (1.0 + return_diff)

# 其他需要添加的处理代码（例如确保所有输入都是标量）:
# 确保值为标量
if isinstance(stock_change, pd.Series):
    stock_change = stock_change.iloc[0] if not stock_change.empty else 0
if isinstance(sector_change, pd.Series):
    sector_change = sector_change.iloc[0] if not sector_change.empty else 0

# 对于布尔值的处理:
is_limit_up = stock_row.get('is_limit_up', False)
if isinstance(is_limit_up, pd.Series):
    is_limit_up = is_limit_up.iloc[0] if not is_limit_up.empty else False
'''

# ===============================================================
# 如何应用这些修改
# ===============================================================

'''
1. 问题1: 变化点数据验证逻辑有问题
   - 原因: 原代码中的判断条件 "dates.any() or dates" 可能导致错误判断
   - 解决方案: 更详细地检查变化点数据格式，区分处理list和numpy.ndarray类型

2. 问题2: Series truth value错误
   - 原因: pandas Series对象在逻辑条件判断中会产生歧义
   - 解决方案: 
     a) 改用 "(not np.isnan(a) or not np.isnan(b))" 代替 "not np.isnan(a) and not np.isnan(b)"
     b) 确保比较前所有值都转换为标量

3. 问题3: 处理过程卡住没有进度更新
   - 原因: 处理大量数据时缺少详细进度日志
   - 解决方案: 添加更详细的进度日志，包括批次处理时间、成功/失败统计等
''' 