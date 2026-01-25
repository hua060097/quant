#!/usr/bin/env python3
"""
SW申万行业编码生成器

生成标准化的SW行业编码映射，确保模型训练、回测、预测使用一致的编码。
生成的编码文件包含：
1. 股票代码 -> 行业名称映射
2. 行业名称 -> 编码映射
3. 编码 -> 行业名称映射
4. 元数据（生成时间、版本等）
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np

# 导入现有函数
import sys
sys.path.append('.')
from main import load_sw_sector_mapping


def generate_sw_encoding(
    sw_sector_mapping: Dict[str, str],
    output_dir: str = "./config",
    encoding_name: str = "sw_industry_encoding"
) -> Dict:
    """
    生成标准化的SW行业编码

    Args:
        sw_sector_mapping: 股票代码到行业名称的映射字典
        output_dir: 输出目录
        encoding_name: 编码文件名称前缀

    Returns:
        Dict: 完整的编码信息
    """
    print("=" * 80)
    print("SW申万行业编码生成器")
    print("=" * 80)

    # 1. 基础数据统计
    print(f"\n[统计] 基础数据统计:")
    print(f"  股票数量: {len(sw_sector_mapping):,}")

    # 获取所有行业名称并去重
    all_industries = set(sw_sector_mapping.values())
    print(f"  行业数量: {len(all_industries)}")

    # 2. 检查数据质量
    print(f"\n[检查] 数据质量检查:")
    missing_industry = {code: industry for code, industry in sw_sector_mapping.items()
                       if not industry or industry == '未知'}
    if missing_industry:
        print(f"  ⚠️ 发现 {len(missing_industry)} 只股票没有行业信息或为'未知'")
        print(f"  前10个示例: {list(missing_industry.keys())[:10]}")
    else:
        print(f"  ✅ 所有股票都有有效的行业信息")

    # 3. 生成行业编码（按照行业名称排序，确保稳定）
    print(f"\n🔧 生成行业编码:")

    # 排序行业名称，确保编码顺序稳定
    sorted_industries = sorted(all_industries)

    # 生成编码映射
    industry_name_to_code = {industry_name: idx for idx, industry_name in enumerate(sorted_industries)}
    code_to_industry_name = {idx: industry_name for industry_name, idx in industry_name_to_code.items()}

    print(f"  编码方式: 顺序编码 (0-{len(sorted_industries)-1})")
    print(f"  行业排序: 按行业名称拼音排序")

    # 4. 统计每个行业的股票数量
    print(f"\n📈 行业分布统计:")
    industry_stock_counts = {}
    for industry in all_industries:
        stock_count = sum(1 for sector in sw_sector_mapping.values() if sector == industry)
        industry_stock_counts[industry] = stock_count

    # 显示行业分布
    sorted_by_count = sorted(industry_stock_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"  股票最多的前10个行业:")
    for i, (industry, count) in enumerate(sorted_by_count[:10]):
        code = industry_name_to_code[industry]
        print(f"    {i+1:2d}. {industry:20s} (编码:{code:2d}) - {count:4d}只股票")

    # 5. 生成完整的编码数据结构
    encoding_data = {
        "metadata": {
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "description": "SW申万行业标准化编码映射",
            "total_stocks": len(sw_sector_mapping),
            "total_industries": len(all_industries),
            "encoding_method": "sequential_by_name",
            "missing_stocks": len(missing_industry)
        },
        "mappings": {
            "stock_to_industry": sw_sector_mapping,
            "industry_to_code": industry_name_to_code,
            "code_to_industry": code_to_industry_name
        },
        "statistics": {
            "industry_stock_counts": industry_stock_counts,
            "industry_distribution": {
                industry: {"code": code, "stock_count": count}
                for industry, count in industry_stock_counts.items()
                for code in [industry_name_to_code[industry]]
            }
        }
    }

    # 6. 保存编码文件
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 保存JSON格式（便于查看和版本控制）
    json_file = output_path / f"{encoding_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(encoding_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 编码文件已保存:")
    print(f"  JSON格式: {json_file}")

    # 保存Pickle格式（便于快速加载）
    pickle_file = output_path / f"{encoding_name}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(encoding_data, f)
    print(f"  Pickle格式: {pickle_file}")

    # 7. 生成使用示例代码
    example_file = output_path / f"{encoding_name}_usage_example.py"
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(f'''#!/usr/bin/env python3
"""
SW行业编码使用示例

演示如何加载和使用标准化的SW行业编码
"""

import pickle
from pathlib import Path

def load_sw_encoding(encoding_file: str = "./config/sw_industry_encoding.pkl"):
    """加载SW行业编码"""
    with open(encoding_file, 'rb') as f:
        encoding_data = pickle.load(f)
    return encoding_data

def get_industry_code(stock_code: str, encoding_data):
    """获取股票对应的行业编码"""
    mappings = encoding_data['mappings']

    # 1. 股票代码 -> 行业名称
    industry_name = mappings['stock_to_industry'].get(stock_code, '未知')

    # 2. 行业名称 -> 编码
    industry_code = mappings['industry_to_code'].get(industry_name, -1)

    return industry_code, industry_name

def main():
    # 加载编码数据
    encoding_data = load_sw_encoding()

    # 显示元数据
    metadata = encoding_data['metadata']
    print(f"SW行业编码版本: {{metadata['version']}}")
    print(f"生成时间: {{metadata['generated_at']}}")
    print(f"总股票数: {{metadata['total_stocks']:,}}")
    print(f"总行业数: {{metadata['total_industries']}}")

    # 使用示例
    test_stocks = ['000001.SZ', '000002.SZ', '600000.SH']

    for stock in test_stocks:
        code, industry = get_industry_code(stock, encoding_data)
        print(f"{{stock}} -> {{industry}} (编码: {{code}})")

if __name__ == "__main__":
    main()
''')
    print(f"  使用示例: {example_file}")

    # 8. 验证编码完整性
    print(f"\n✅ 编码完整性验证:")

    # 验证双向映射
    invalid_mappings = 0
    for stock_code, industry_name in sw_sector_mapping.items():
        if industry_name in industry_name_to_code:
            code = industry_name_to_code[industry_name]
            reconstructed_industry = code_to_industry_name.get(code)
            if reconstructed_industry != industry_name:
                invalid_mappings += 1
        else:
            invalid_mappings += 1

    if invalid_mappings == 0:
        print(f"  ✅ 双向映射验证通过")
    else:
        print(f"  ❌ 发现 {invalid_mappings} 个无效映射")

    print(f"\n🎉 SW行业编码生成完成！")
    print(f"📁 文件位置: {output_path}")
    print(f"📄 主要文件: {json_file}")

    return encoding_data


def validate_existing_encoding(encoding_file: str) -> bool:
    """
    验证现有编码文件是否有效

    Args:
        encoding_file: 编码文件路径

    Returns:
        bool: 编码文件是否有效
    """
    try:
        with open(encoding_file, 'rb') as f:
            encoding_data = pickle.load(f)

        # 检查必要字段
        required_fields = ['metadata', 'mappings']
        for field in required_fields:
            if field not in encoding_data:
                print(f"❌ 缺少必要字段: {field}")
                return False

        mappings = encoding_data['mappings']
        required_mappings = ['stock_to_industry', 'industry_to_code', 'code_to_industry']
        for mapping in required_mappings:
            if mapping not in mappings:
                print(f"❌ 缺少必要映射: {mapping}")
                return False

        print(f"✅ 编码文件验证通过")
        print(f"   版本: {encoding_data['metadata']['version']}")
        print(f"   生成时间: {encoding_data['metadata']['generated_at']}")
        print(f"   股票数: {encoding_data['metadata']['total_stocks']:,}")
        print(f"   行业数: {encoding_data['metadata']['total_industries']}")

        return True

    except Exception as e:
        print(f"❌ 编码文件验证失败: {e}")
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="SW申万行业编码生成器")
    parser.add_argument("--output-dir", default="./config", help="输出目录 (默认: ./config)")
    parser.add_argument("--name", default="sw_industry_encoding", help="编码文件名前缀")
    parser.add_argument("--validate", help="验证现有编码文件")
    parser.add_argument("--force", action="store_true", help="强制重新生成，覆盖现有文件")

    args = parser.parse_args()

    # 验证模式
    if args.validate:
        print(f"验证编码文件: {args.validate}")
        is_valid = validate_existing_encoding(args.validate)
        return 0 if is_valid else 1

    # 检查现有文件
    encoding_file = Path(args.output_dir) / f"{args.name}.pkl"
    if encoding_file.exists() and not args.force:
        print(f"⚠️ 编码文件已存在: {encoding_file}")
        print("使用 --force 参数强制重新生成，或使用 --validate 验证现有文件")

        # 验证现有文件
        is_valid = validate_existing_encoding(str(encoding_file))
        return 0 if is_valid else 1

    # 生成编码
    try:
        # 加载SW行业映射
        print("加载SW行业映射数据...")
        sw_sector_mapping = load_sw_sector_mapping()

        if not sw_sector_mapping:
            print("❌ 无法加载SW行业映射数据")
            return 1

        # 生成编码
        encoding_data = generate_sw_encoding(sw_sector_mapping, args.output_dir, args.name)

        print(f"\n✅ 成功生成SW行业编码！")
        return 0

    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())