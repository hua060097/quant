#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同步 factor_lab/definitions/ 目录下所有因子到 main.py 的 _get_stock_factors 函数中
"""

import os
import re
import ast

def extract_factors_from_file(file_path):
    """从单个因子文件中提取所有因子名称"""
    factors = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找所有 register_factor 调用
        # 模式: factor_name = register_factor(FactorDefinition(...))
        pattern = r'(\w+)\s*=\s*register_factor\s*\('
        matches = re.findall(pattern, content)

        for match in matches:
            factor_name = match.strip()
            # 跳过明显不是因子的变量名
            if factor_name and not factor_name.startswith('_') and len(factor_name) > 2:
                factors.append(factor_name)

        # 也查找 __all__ 中的因子（如果存在）
        if '__all__' in content:
            # 简单提取 __all__ 列表中的字符串
            all_pattern = r'__all__\s*=\s*\[([^\]]+)\]'
            all_match = re.search(all_pattern, content, re.DOTALL)
            if all_match:
                all_content = all_match.group(1)
                # 提取单引号或双引号中的字符串
                string_pattern = r'["\']([^"\']+)["\']'
                all_factors = re.findall(string_pattern, all_content)
                factors.extend(all_factors)

    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")

    return factors

def scan_all_factor_files():
    """扫描 factor_lab/definitions/ 目录下所有因子文件"""
    definitions_dir = 'factor_system/factor_lab/definitions'

    if not os.path.exists(definitions_dir):
        print(f"错误: 目录 {definitions_dir} 不存在")
        return []

    all_factors = []
    factor_files = []

    # 遍历目录中的所有 .py 文件
    for file_name in os.listdir(definitions_dir):
        if file_name.endswith('.py') and file_name != '__init__.py':
            file_path = os.path.join(definitions_dir, file_name)
            factor_files.append((file_name, file_path))

    print(f"发现 {len(factor_files)} 个因子文件:")
    for file_name, file_path in factor_files:
        print(f"  - {file_name}")

    # 从每个文件中提取因子
    for file_name, file_path in factor_files:
        print(f"\n正在处理 {file_name}...")
        factors = extract_factors_from_file(file_path)
        if factors:
            print(f"  发现因子: {factors}")
            all_factors.extend(factors)
        else:
            print(f"  未发现因子")

    # 去重并排序
    unique_factors = list(set(all_factors))
    unique_factors.sort()

    print(f"\n总共发现 {len(unique_factors)} 个唯一因子:")
    for i, factor in enumerate(unique_factors, 1):
        print(f"  {i:2d}. {factor}")

    return unique_factors

def read_current_get_stock_factors():
    """读取当前 main.py 中的 _get_stock_factors 函数"""
    main_file = 'main.py'

    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找 _get_stock_factors 函数
        function_pattern = r'def _get_stock_factors\(\):(.*?)(?=\n\ndef|\nclass|\n# [A-Z]|\Z)'
        match = re.search(function_pattern, content, re.DOTALL)

        if match:
            return content, match.group(0)
        else:
            print("错误: 未找到 _get_stock_factors 函数")
            return None, None

    except Exception as e:
        print(f"读取 main.py 时出错: {e}")
        return None, None

def generate_new_all_factors_section(factors):
    """生成新的 all_factors 列表定义"""

    # 按类别分组因子
    categorized_factors = {
        'momentum_reversal': [],
        'moving_average': [],
        'candlestick': [],
        'market_corr': [],
        'sector': [],
        'support': [],
        'kdj': [],
        'technical_indicators': [],
        'statistical_moments': [],
        'cross_sectional': [],
        'liquidity': [],
        'risk_control': [],
        'low_volatility': [],
        'behavioral': [],
        'intraday': [],
        'worldquant': [],
        'style_rotation': [],
        'interaction': [],
        'support_resistance': [],
        'market_context': [],
        'returns': [],
        'rankings': [],
        'microstructure': [],
        'other': []
    }

    # 根据因子名称尝试分类
    for factor in factors:
        if any(keyword in factor for keyword in ['pct_chg', 'return', 'momentum', 'reversal', 'trend', 'decay']):
            categorized_factors['momentum_reversal'].append(factor)
        elif any(keyword in factor for keyword in ['ma', 'moving', 'average']):
            categorized_factors['moving_average'].append(factor)
        elif any(keyword in factor for keyword in ['candle', 'shadow', 'market_structure', 'pullback', 'volume_strength', 'near_support']):
            categorized_factors['candlestick'].append(factor)
        elif any(keyword in factor for keyword in ['corr', 'correlation']):
            categorized_factors['market_corr'].append(factor)
        elif any(keyword in factor for keyword in ['sector', 'sw_', 'intra_sector']):
            categorized_factors['sector'].append(factor)
        elif any(keyword in factor for keyword in ['support', 'resistance', 'bollinger', 'obv', 'institutional']):
            categorized_factors['support'].append(factor)
        elif any(keyword in factor for keyword in ['kdj', 'macd', 'atr']):
            categorized_factors['kdj'].append(factor)
        elif any(keyword in factor for keyword in ['ppo', 'mfi', 'vortex', 'mass_index']):
            categorized_factors['technical_indicators'].append(factor)
        elif any(keyword in factor for keyword in ['skew', 'kurtosis', 'coskew', 'downside_beta', 'idiosyncratic']):
            categorized_factors['statistical_moments'].append(factor)
        elif any(keyword in factor for keyword in ['rank_', 'cross_sectional']):
            categorized_factors['cross_sectional'].append(factor)
        elif any(keyword in factor for keyword in ['turnover', 'amihud', 'liquidity', 'volume']):
            categorized_factors['liquidity'].append(factor)
        elif any(keyword in factor for keyword in ['drawdown', 'ulcer', 'efficiency', 'risk', 'var_', 'cvar_']):
            categorized_factors['risk_control'].append(factor)
        elif any(keyword in factor for keyword in ['beta', 'volatility', 'daily_return']):
            categorized_factors['low_volatility'].append(factor)
        elif any(keyword in factor for keyword in ['52w', 'acceleration']):
            categorized_factors['behavioral'].append(factor)
        elif any(keyword in factor for keyword in ['overnight', 'intraday', 'open_to']):
            categorized_factors['intraday'].append(factor)
        elif any(keyword in factor for keyword in ['alpha_']):
            categorized_factors['worldquant'].append(factor)
        elif any(keyword in factor for keyword in ['sector_', 'size_', 'style', 'rotation']):
            categorized_factors['style_rotation'].append(factor)
        elif any(keyword in factor for keyword in ['interaction']):
            categorized_factors['interaction'].append(factor)
        elif any(keyword in factor for keyword in ['support_distance', 'resistance_distance']):
            categorized_factors['support_resistance'].append(factor)
        elif any(keyword in factor for keyword in ['market_', 'trend_', 'sentiment', 'volatility_context', 'liquidity_change']):
            categorized_factors['market_context'].append(factor)
        elif any(keyword in factor for keyword in ['returns_zscore']):
            categorized_factors['returns'].append(factor)
        elif any(keyword in factor for keyword in ['rank']):
            categorized_factors['rankings'].append(factor)
        elif any(keyword in factor for keyword in ['price_impact']):
            categorized_factors['microstructure'].append(factor)
        else:
            categorized_factors['other'].append(factor)

    # 生成代码
    code_lines = ["    # ===== 自动同步的所有因子 ====="]

    category_names = {
        'momentum_reversal': '动量与反转因子',
        'moving_average': '移动平均因子',
        'candlestick': '蜡烛图因子',
        'market_corr': '市场相关性因子',
        'sector': '行业板块因子',
        'support': '支撑阻力因子',
        'kdj': 'KDJ技术指标因子',
        'technical_indicators': '技术指标因子',
        'statistical_moments': '统计矩因子',
        'cross_sectional': '横截面排名因子',
        'liquidity': '流动性因子',
        'risk_control': '风险控制因子',
        'low_volatility': '低波动异常因子',
        'behavioral': '行为金融因子',
        'intraday': '日内微观结构因子',
        'worldquant': 'WorldQuant Alpha因子',
        'style_rotation': '风格轮动因子',
        'interaction': '交互特征因子',
        'support_resistance': '支撑压力位因子',
        'market_context': '市场环境因子',
        'returns': '收益率因子',
        'rankings': '排名因子',
        'microstructure': '微观结构因子',
        'other': '其他因子'
    }

    total_factors = 0
    for category, factors in categorized_factors.items():
        if factors:
            code_lines.append(f"    # ===== {category_names.get(category, category)} =====")
            for factor in sorted(factors):
                code_lines.append(f"        '{factor}',")
            code_lines.append("")
            total_factors += len(factors)

    code_lines.append(f"    # 总计: {total_factors} 个因子")

    return '\n'.join(code_lines)

def update_get_stock_factors():
    """更新 main.py 中的 _get_stock_factors 函数"""

    # 1. 扫描所有因子
    print("正在扫描所有因子文件...")
    all_factors = scan_all_factor_files()

    if not all_factors:
        print("未发现任何因子，退出")
        return False

    # 2. 读取当前函数
    print("\n正在读取当前的 _get_stock_factors 函数...")
    content, function_content = read_current_get_stock_factors()

    if not content:
        print("无法读取 main.py，退出")
        return False

    # 3. 生成新的 all_factors 定义
    print("\n正在生成新的 all_factors 列表...")
    new_all_factors_code = generate_new_all_factors_section(all_factors)

    # 4. 替换函数内容
    print("正在更新 main.py...")

    # 查找函数开始位置
    function_start_pattern = r'def _get_stock_factors\(\):.*?\n'
    function_start_match = re.search(function_start_pattern, content)

    if not function_start_match:
        print("无法找到函数开始位置")
        return False

    function_start = function_start_match.end()

    # 查找函数结束位置（下一个函数或类开始）
    function_end_pattern = r'\n\n(?:def |class |# [A-Z]|Z$)'
    function_end_match = re.search(function_end_pattern, content[function_start:])

    if function_end_match:
        function_end = function_start + function_end_match.start()
    else:
        function_end = len(content)

    # 构建新的函数内容
    new_function = content[:function_start] + '\n' + new_all_factors_code + '\n\n    return all_factors\n'

    # 如果有剩余内容，添加回来
    if function_end < len(content):
        remaining_content = content[function_end:]
        # 确保剩余内容以换行符开始
        if not remaining_content.startswith('\n'):
            remaining_content = '\n' + remaining_content
        new_function += remaining_content

    # 5. 备份原文件
    backup_file = 'main.py.backup_before_factor_sync'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已备份原文件到: {backup_file}")

    # 6. 写入新文件
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(new_function)

    print(f"\n✅ 更新完成！")
    print(f"  - 总共发现 {len(all_factors)} 个因子")
    print(f"  - 已写入 main.py 的 _get_stock_factors 函数")

    return True

if __name__ == "__main__":
    print("因子同步工具 - 将所有定义的因子同步到 main.py")
    print("=" * 60)

    success = update_get_stock_factors()

    if success:
        print("\n🎉 同步成功！现在 main.py 中的 _get_stock_factors 函数包含了所有定义的因子。")
        print("\n建议:")
        print("1. 检查生成的因子列表是否符合预期")
        print("2. 运行测试确保系统能正常工作")
        print("3. 可以将此脚本集成到 CI/CD 流程中自动更新")
    else:
        print("\n❌ 同步失败，请检查错误信息并重试")