"""
计算指定时间段内 daily_change_pct 的均值
"""
import json
from pathlib import Path
from datetime import datetime

# 文件路径
market_status_file = Path(__file__).parent.parent.parent / "local_data" / "market_status" / "daily" / "market_status.json"

print("=" * 80)
print("计算 daily_change_pct 均值")
print("=" * 80)

# 时间范围
start_date = "20220705"
end_date = "20260126"

print(f"\n目标时间段: {start_date} ~ {end_date}")
print(f"读取文件: {market_status_file}")

# 读取 JSON 文件
try:
    with open(market_status_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"文件读取成功")

    # 筛选指定时间范围内的 daily_change_pct
    daily_changes = []

    for date_str, date_data in data.items():
        # 检查日期是否在范围内
        if start_date <= date_str <= end_date:
            if isinstance(date_data, dict) and 'individual_status' in date_data:
                individual_status = date_data['individual_status']

                # 获取市场指数的 daily_change_pct (通常只有一个指数，如 399006.SZ)
                for stock_code, stock_data in individual_status.items():
                    if isinstance(stock_data, dict) and 'market_data' in stock_data:
                        market_data = stock_data['market_data']
                        if isinstance(market_data, dict) and 'daily_change_pct' in market_data:
                            change_pct = market_data['daily_change_pct']
                            daily_changes.append({
                                'date': date_str,
                                'daily_change_pct': change_pct,
                                'index_code': stock_code
                            })
                            break  # 每天只取第一个指数

    if not daily_changes:
        print(f"\n错误: 在 {start_date}~{end_date} 范围内未找到数据")
        print(f"可用的日期范围: {min(data.keys())} ~ {max(data.keys())}")
    else:
        # 按日期排序
        daily_changes.sort(key=lambda x: x['date'])

        # 计算整体均值
        change_values = [item['daily_change_pct'] for item in daily_changes]
        mean_change = sum(change_values) / len(change_values)

        print(f"\n{'=' * 80}")
        print(f"统计结果")
        print(f"{'=' * 80}")
        print(f"数据天数: {len(daily_changes)} 天")
        print(f"daily_change_pct 均值: {mean_change:.6f} ({mean_change*100:.4f}%)")
        print(f"最大值: {max(change_values):.6f}")
        print(f"最小值: {min(change_values):.6f}")
        print(f"标准差: {(sum((x - mean_change) ** 2 for x in change_values) / len(change_values)) ** 0.5:.6f}")

        # 显示前5条和后5条数据
        print(f"\n{'=' * 80}")
        print(f"数据预览")
        print(f"{'=' * 80}")
        print(f"\n前5条:")
        for item in daily_changes[:5]:
            print(f"  {item['date']}: {item['daily_change_pct']:.6f} (指数: {item['index_code']})")

        print(f"\n后5条:")
        for item in daily_changes[-5:]:
            print(f"  {item['date']}: {item['daily_change_pct']:.6f} (指数: {item['index_code']})")

except FileNotFoundError:
    print(f"\n错误: 文件不存在 {market_status_file}")
except json.JSONDecodeError as e:
    print(f"\n错误: JSON 解析失败 - {e}")
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'=' * 80}")
print("计算完成")
print("=" * 80)
