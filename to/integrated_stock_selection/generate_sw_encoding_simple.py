#!/usr/bin/env python3
"""
SW申万行业编码生成器 - 简化版

生成标准化的SW行业编码映射，确保模型训练、回测、预测使用一致的编码。
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict

def generate_sw_encoding_simple(sw_sector_mapping: Dict[str, str], output_dir: str = "./config") -> Dict:
    """
    生成标准化的SW行业编码 - 简化版

    Args:
        sw_sector_mapping: 股票代码到行业名称的映射字典
        output_dir: 输出目录

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

    # 2. 生成行业编码（按照行业名称排序，确保稳定）
    print(f"\n[编码] 生成行业编码:")

    # 排序行业名称，确保编码顺序稳定
    sorted_industries = sorted(all_industries)

    # 生成编码映射
    industry_name_to_code = {industry_name: idx for idx, industry_name in enumerate(sorted_industries)}
    code_to_industry_name = {idx: industry_name for industry_name, idx in industry_name_to_code.items()}

    print(f"  编码方式: 顺序编码 (0-{len(sorted_industries)-1})")
    print(f"  行业排序: 按行业名称拼音排序")

    # 3. 统计每个行业的股票数量
    print(f"\n[分布] 行业分布统计:")
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

    # 4. 生成完整的编码数据结构
    encoding_data = {
        "metadata": {
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "description": "SW申万行业标准化编码映射",
            "total_stocks": len(sw_sector_mapping),
            "total_industries": len(all_industries),
            "encoding_method": "sequential_by_name"
        },
        "mappings": {
            "stock_to_industry": sw_sector_mapping,
            "industry_to_code": industry_name_to_code,
            "code_to_industry": code_to_industry_name
        },
        "statistics": {
            "industry_stock_counts": industry_stock_counts
        }
    }

    # 5. 保存编码文件
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 保存JSON格式（便于查看和版本控制）
    json_file = output_path / "sw2_industry_encoding.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(encoding_data, f, indent=2, ensure_ascii=False)

    # 保存Pickle格式（便于快速加载）
    pickle_file = output_path / "sw2_industry_encoding.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(encoding_data, f)

    print(f"\n[保存] 编码文件已保存:")
    print(f"  JSON格式: {json_file}")
    print(f"  Pickle格式: {pickle_file}")

    print(f"\n[完成] SW行业编码生成完成！")
    print(f"  文件位置: {output_path}")

    return encoding_data


def load_sw_sector_mapping() -> Dict[str, str]:
    """
    从绝对路径加载SW行业映射数据

    Returns:
        Dict[str, str]: 股票代码到行业名称的映射字典
    """
    # 获取当前脚本所在目录的父目录（to目录）
    current_file = Path(__file__).resolve()  # 当前脚本的绝对路径
    integrated_stock_selection_dir = current_file.parent  # integrated_stock_selection目录
    to_dir = integrated_stock_selection_dir.parent  # to目录

    # 构建sw_sector.json的绝对路径
    sw_sector_json_path = to_dir / "local_data" / "sw_sector2" / "sw_sector.json"

    print(f"从以下路径加载SW行业映射: {sw_sector_json_path}")

    # 检查文件是否存在
    if not sw_sector_json_path.exists():
        raise FileNotFoundError(f"SW行业映射文件不存在: {sw_sector_json_path}")

    # 读取JSON文件
    with open(sw_sector_json_path, 'r', encoding='utf-8') as f:
        sw_sector_mapping = json.load(f)

    print(f"成功加载 {len(sw_sector_mapping):,} 只股票的行业映射数据")

    return sw_sector_mapping


def main():
    """主函数"""
    try:
        # 加载SW行业映射（从绝对路径读取JSON文件）
        print("加载SW行业映射数据...")
        sw_sector_mapping = load_sw_sector_mapping()

        if not sw_sector_mapping:
            print("错误: 无法加载SW行业映射数据")
            return 1

        # 生成编码
        encoding_data = generate_sw_encoding_simple(sw_sector_mapping)

        print(f"\n[成功] SW行业编码生成成功！")
        return 0

    except Exception as e:
        print(f"错误: 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())