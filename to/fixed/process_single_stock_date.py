"""
修复_process_single_stock_date函数中的Series truth value问题
"""

# 以下是_process_single_stock_date函数中需要添加的代码：

"""
在函数开头处，添加日志输出：
"""
'''
self.logger.debug(f"开始处理股票 {stock_code} 在 {date_str} 的特征相关性")
'''

"""
获取股票特征后，添加类型检查：
"""
'''
# 优化计算：预先获取股票特征，避免重复访问
stock_price = stock_row.get('close', 0)
stock_change = stock_row.get('pct_chg', 0)
stock_volume = stock_row.get('vol', 0)
stock_turnover = stock_row.get('turnover_rate', 0)
stock_amplitude = stock_row.get('amplitude', 0)

# 确保值为标量，修复 Series truth value 问题
if isinstance(stock_change, pd.Series):
    stock_change = stock_change.iloc[0] if not stock_change.empty else 0
if isinstance(stock_volume, pd.Series):
    stock_volume = stock_volume.iloc[0] if not stock_volume.empty else 0
if isinstance(stock_turnover, pd.Series):
    stock_turnover = stock_turnover.iloc[0] if not stock_turnover.empty else 0
if isinstance(stock_amplitude, pd.Series):
    stock_amplitude = stock_amplitude.iloc[0] if not stock_amplitude.empty else 0
'''

"""
计算涨跌幅相关性时：
"""
'''
# 1. 涨跌幅相关性
sector_change = sector_row.get('pct_chg', 0)
if isinstance(sector_change, pd.Series):
    sector_change = sector_change.iloc[0] if not sector_change.empty else 0
    
return_correlation = 0.0
if not (np.isnan(stock_change) or np.isnan(sector_change)):  # 修复：使用逻辑运算符而不是 and
    return_diff = abs(stock_change - sector_change)
    return_correlation = 1.0 / (1.0 + return_diff)
'''

"""
类似地，对于其他值的获取都需要添加类型检查：
"""
'''
# 2. 成交量相关性
volume_correlation = 0.0
sector_volume = sector_row.get('vol', 0)
if isinstance(sector_volume, pd.Series):
    sector_volume = sector_volume.iloc[0] if not sector_volume.empty else 0
    
# 3. 换手率相关性
turnover_correlation = 0.0
sector_turnover = sector_row.get('turnover_rate', 0)
if isinstance(sector_turnover, pd.Series):
    sector_turnover = sector_turnover.iloc[0] if not sector_turnover.empty else 0
    
# 4. 波动率相关性
volatility_correlation = 0.0
sector_amplitude = sector_row.get('amplitude', 0)
if isinstance(sector_amplitude, pd.Series):
    sector_amplitude = sector_amplitude.iloc[0] if not sector_amplitude.empty else 0
'''

"""
技术指标相关性计算部分：
"""
'''
# 5. 技术指标相关性
tech_correlation = 0.0
tech_indicators = ['rsi14', 'ma5_slope', 'momentum_10d']
matching_indicators = []

# 修复：检查存在的技术指标
for ind in tech_indicators:
    has_stock_ind = ind in stock_row.index
    has_sector_ind = ind in sector_row.index
    if has_stock_ind and has_sector_ind:
        matching_indicators.append(ind)

if matching_indicators:
    tech_diffs = []
    for ind in matching_indicators:
        stock_val = stock_row[ind]
        sector_val = sector_row[ind]
        
        # 确保值为标量
        if isinstance(stock_val, pd.Series):
            stock_val = stock_val.iloc[0] if not stock_val.empty else 0
        if isinstance(sector_val, pd.Series):
            sector_val = sector_val.iloc[0] if not sector_val.empty else 0
            
        # 确保是数值，避免比较错误
        if pd.notna(stock_val) and pd.notna(sector_val):
            tech_diffs.append(abs(stock_val - sector_val))
            
    avg_diff = sum(tech_diffs) / len(tech_diffs) if tech_diffs else 1.0
    tech_correlation = 1.0 / (1.0 + avg_diff)
'''

"""
龙头特征处理部分：
"""
'''
# 6. 个股在板块中的地位
leadership_score = 0.0
is_leader = False

# 使用已计算的龙头特征
consecutive_days = stock_row.get('consecutive_limit_days', 0)
if isinstance(consecutive_days, pd.Series):
    consecutive_days = consecutive_days.iloc[0] if not consecutive_days.empty else 0

if consecutive_days >= 2:
    leadership_score += min(1.0, consecutive_days / 3.0)
    is_leader = True

# 确保布尔值是标量
is_one_word_limit = stock_row.get('is_one_word_limit', False)
if isinstance(is_one_word_limit, pd.Series):
    is_one_word_limit = is_one_word_limit.iloc[0] if not is_one_word_limit.empty else False
    
if is_one_word_limit:
    leadership_score += 0.5
    is_leader = True

is_high_elasticity = stock_row.get('is_high_elasticity', False)
if isinstance(is_high_elasticity, pd.Series):
    is_high_elasticity = is_high_elasticity.iloc[0] if not is_high_elasticity.empty else False
    
if is_high_elasticity:
    leadership_score += 0.3

market_cap_type = stock_row.get('market_cap_type', '')
if isinstance(market_cap_type, pd.Series):
    market_cap_type = market_cap_type.iloc[0] if not market_cap_type.empty else ''

is_limit_up = stock_row.get('is_limit_up', False)
if isinstance(is_limit_up, pd.Series):
    is_limit_up = is_limit_up.iloc[0] if not is_limit_up.empty else False
    
if market_cap_type == 'large' and (is_limit_up or consecutive_days > 0):
    leadership_score += 0.4
    is_leader = True
'''

"""
涨跌停结构相似性部分：
"""
'''
# 7. 涨跌停结构相似性
limit_structure_score = 0.0

# 修复：安全地检查字段是否存在
limit_up_count_exists = 'limit_up_count' in sector_row.index
limit_down_count_exists = 'limit_down_count' in sector_row.index

has_limit_up_count = False
has_limit_down_count = False

if limit_up_count_exists:
    limit_up_count = sector_row['limit_up_count']
    if isinstance(limit_up_count, pd.Series):
        limit_up_count = limit_up_count.iloc[0] if not limit_up_count.empty else 0
    has_limit_up_count = limit_up_count > 0
    
if limit_down_count_exists:
    limit_down_count = sector_row['limit_down_count']
    if isinstance(limit_down_count, pd.Series):
        limit_down_count = limit_down_count.iloc[0] if not limit_down_count.empty else 0
    has_limit_down_count = limit_down_count > 0

if has_limit_up_count or has_limit_down_count:
    is_stock_limit_up = stock_row.get('is_limit_up', False)
    if isinstance(is_stock_limit_up, pd.Series):
        is_stock_limit_up = is_stock_limit_up.iloc[0] if not is_stock_limit_up.empty else False
        
    is_stock_limit_down = stock_row.get('is_limit_down', False)
    if isinstance(is_stock_limit_down, pd.Series):
        is_stock_limit_down = is_stock_limit_down.iloc[0] if not is_stock_limit_down.empty else False
    
    if is_stock_limit_up and has_limit_up_count:
        limit_structure_score += 0.5
    if is_stock_limit_down and has_limit_down_count:
        limit_structure_score += 0.5
'''

"""
最后在异常处理部分添加详细错误跟踪：
"""
'''
except Exception as e:
    self.logger.error(f"处理股票 {stock_code} 在 {date_str} 的特征时出错: {str(e)}")
    # 添加详细的错误跟踪以便更好地调试
    import traceback
    self.logger.error(f"错误详情: {traceback.format_exc()}")
    return None
''' 