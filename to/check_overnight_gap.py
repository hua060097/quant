"""
检查20251220~20260107期间所有股票的隔夜涨跌幅
隔夜涨跌幅 = (t+1日开盘价 - t日收盘价) / t日收盘价
如果绝对值大于25%，则输出该股票的日期和stock_code
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# 数据目录
DATA_DIR = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\features\daily")

# 检查的日期范围
START_DATE = "20251220"
END_DATE = "20260107"

# 需要加载的日期列表
def get_date_list(start_date, end_date):
    """获取日期列表"""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return date_list

# 加载数据
def load_data(date_list):
    """加载指定日期范围的数据"""
    all_data = []
    for date_str in date_list:
        file_path = DATA_DIR / f"{date_str}.parquet"
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                all_data.append(df)
                print(f"已加载: {date_str}, 数据量: {len(df)}")
            except Exception as e:
                print(f"加载失败: {date_str}, 错误: {e}")
        else:
            print(f"文件不存在: {date_str}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# 计算隔夜涨跌幅
def check_overnight_gap(df, threshold=0.25):
    """
    检查隔夜涨跌幅
    隔夜涨跌幅 = (t+1日开盘价 - t日收盘价) / t日收盘价
    """
    if df.empty:
        print("没有数据可检查")
        return

    # 按stock_code和date排序
    df = df.sort_values(['stock_code', 'date']).copy()

    results = []

    # 按股票分组处理
    for stock_code in df['stock_code'].unique():
        stock_df = df[df['stock_code'] == stock_code].copy()
        stock_df = stock_df.sort_values('date').reset_index(drop=True)

        # 计算隔夜涨跌幅
        for i in range(len(stock_df) - 1):
            current_row = stock_df.iloc[i]
            next_row = stock_df.iloc[i + 1]

            close_price = current_row['close']
            open_price = next_row['open']

            # 检查价格有效性
            if pd.isna(close_price) or pd.isna(open_price) or close_price == 0:
                continue

            # 计算隔夜涨跌幅
            overnight_return = (open_price - close_price) / close_price

            # 检查是否超过阈值
            if abs(overnight_return) > threshold:
                results.append({
                    'stock_code': stock_code,
                    'date': current_row['date'],
                    't_date': current_row['date'],
                    't_close': close_price,
                    't1_date': next_row['date'],
                    't1_open': open_price,
                    'overnight_return': overnight_return,
                    'overnight_return_pct': overnight_return * 100
                })

    return results

# 主函数
def main():
    print("=" * 80)
    print("检查隔夜涨跌幅异常股票")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print(f"阈值: 25%")
    print("=" * 80)

    # 获取日期列表
    date_list = get_date_list(START_DATE, END_DATE)
    print(f"\n需要检查的日期数量: {len(date_list)}")

    # 加载数据
    print("\n开始加载数据...")
    df = load_data(date_list)

    if df.empty:
        print("没有加载到任何数据")
        return

    print(f"\n总共加载 {len(df)} 条数据")
    print(f"股票数量: {df['stock_code'].nunique()}")
    print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")

    # 检查隔夜涨跌幅
    print("\n开始检查隔夜涨跌幅...")
    abnormal_results = check_overnight_gap(df, threshold=0.25)

    # 输出结果
    print("\n" + "=" * 80)
    print(f"检查完成！共发现 {len(abnormal_results)} 条异常记录")
    print("=" * 80)

    if abnormal_results:
        print("\n异常股票详情：")
        print("-" * 80)
        for i, result in enumerate(abnormal_results, 1):
            print(f"\n异常 #{i}")
            print(f"  股票代码: {result['stock_code']}")
            print(f"  T日日期: {result['t_date']}, 收盘价: {result['t_close']:.2f}")
            print(f"  T+1日日期: {result['t1_date']}, 开盘价: {result['t1_open']:.2f}")
            print(f"  隔夜涨跌幅: {result['overnight_return_pct']:.2f}%")

        # 保存到CSV
        output_file = "overnight_gap_abnormal.csv"
        results_df = pd.DataFrame(abnormal_results)
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")
    else:
        print("\n未发现隔夜涨跌幅超过25%的异常股票")

if __name__ == "__main__":
    main()
