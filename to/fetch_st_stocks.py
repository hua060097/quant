"""
获取ST股票列表数据，支持增量更新
用法: python fetch_st_stocks.py -s <start_date> -e <end_date>
示例: python fetch_st_stocks.py -s 20240101 -e 20240131
"""
import sys
import pandas as pd
import tushare as ts
from pathlib import Path
from datetime import datetime
import time
import argparse

# Tushare token
# TS_TOKEN = '022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000'
TS_TOKEN = 'gg99b01a244311f7742512463d9e1588954d174b86d695f8a9f1cc4b'

# 接口限流配置 (stock_st: 500次/分钟)
RATE_LIMIT_PER_MIN = 500
SLEEP_INTERVAL = 0.15   # 每次请求后 sleep, 留安全余量, 实际约 400 次/分钟
MAX_RETRIES = 5         # 单日最大重试次数
RATE_LIMIT_SLEEP = 65   # 触发限流时等待秒数 (>60s 跨过限流窗口)

# 输出文件路径
OUTPUT_FILE = Path(__file__).parent / "local_data" / "st.parquet"
FAILED_DATES_FILE = Path(__file__).parent / "local_data" / "st_failed_dates.txt"


def parse_date_range(start_date, end_date):
    """解析日期范围，从交易日历中获取交易日列表"""
    # 交易日历文件路径
    calendar_file = Path(__file__).parent / "local_data" / "calendar" / "trading_calendar.parquet"

    if not calendar_file.exists():
        print(f"[ERROR] 交易日历文件不存在: {calendar_file}")
        raise FileNotFoundError(f"交易日历文件不存在: {calendar_file}")

    # 读取交易日历
    try:
        calendar_df = pd.read_parquet(calendar_file)

        # 获取日期列（可能是 'date_str' 或 'date'）
        if 'date_str' in calendar_df.columns:
            trading_dates = calendar_df['date_str'].tolist()
        elif 'date' in calendar_df.columns:
            # 如果是 datetime 类型，转换为字符串
            if pd.api.types.is_datetime64_any_dtype(calendar_df['date']):
                trading_dates = calendar_df['date'].dt.strftime('%Y%m%d').tolist()
            else:
                trading_dates = calendar_df['date'].astype(str).tolist()
        else:
            raise ValueError(f"交易日历文件中未找到日期列，可用列: {calendar_df.columns.tolist()}")

        # 筛选指定日期范围内的交易日
        date_list = [d for d in trading_dates if start_date <= d <= end_date]

        if not date_list:
            print(f"[WARNING] 指定日期范围 {start_date}~{end_date} 内没有交易日")
            print(f"[INFO] 交易日历范围: {trading_dates[0]} ~ {trading_dates[-1]}")

        return date_list

    except Exception as e:
        print(f"[ERROR] 读取交易日历失败: {e}")
        raise


def load_existing_data():
    """加载已有的parquet文件"""
    if not OUTPUT_FILE.exists():
        return None

    try:
        df = pd.read_parquet(OUTPUT_FILE)
        print(f"[INFO] 已加载现有数据: {len(df)} 条记录")
        if 'trade_date' in df.columns:
            existing_dates = df['trade_date'].unique()
            print(f"[INFO] 现有数据日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
            print(f"[INFO] 现有数据包含 {len(existing_dates)} 个交易日")
        return df
    except Exception as e:
        print(f"[ERROR] 加载现有数据失败: {e}")
        return None


def fetch_st_stocks(start_date, end_date):
    """获取指定日期范围的ST股票数据"""
    print("=" * 80)
    print("ST股票数据获取工具")
    print("=" * 80)

    # 设置token
    ts.set_token(TS_TOKEN)
    pro = ts.pro_api()

    # 解析日期范围
    date_list = parse_date_range(start_date, end_date)
    print(f"\n[INFO] 目标日期范围: {start_date} ~ {end_date}")
    print(f"[INFO] 交易日数量: {len(date_list)} 天")

    # 加载现有数据
    existing_df = load_existing_data()

    # 确定需要获取的日期（排除已存在的日期）
    if existing_df is not None and 'trade_date' in existing_df.columns:
        existing_dates = set(existing_df['trade_date'].astype(str).tolist())
        new_dates = [d for d in date_list if d not in existing_dates]
        overwritten_dates = [d for d in date_list if d in existing_dates]

        if overwritten_dates:
            print(f"[INFO] 将覆盖 {len(overwritten_dates)} 个已有日期的数据")
            print(f"[INFO] 覆盖日期: {overwritten_dates[:5]}{'...' if len(overwritten_dates) > 5 else ''}")

        if not new_dates and not overwritten_dates:
            print("\n[INFO] 指定日期范围内的数据已存在，无需获取")
            return

        # 对于增量更新，只需要获取新日期；对于覆盖，需要获取所有指定日期
        dates_to_fetch = date_list  # 获取所有日期，然后合并
    else:
        dates_to_fetch = date_list
        new_dates = date_list
        overwritten_dates = []

    if new_dates:
        print(f"[INFO] 需要获取 {len(new_dates)} 个新日期的数据")

    print(f"\n[INFO] 开始获取数据...")

    # 循环获取数据
    all_data = []
    success_dates = []      # 实际成功获取(含空返回)的日期, 作为后续删除旧数据依据
    failed_dates = []       # 重试后仍失败的日期, 旧数据保持不动
    success_count = 0
    skip_count = 0
    error_count = 0

    for i, trade_date in enumerate(dates_to_fetch, 1):
        print(f"\r[{i}/{len(dates_to_fetch)}] 获取 {trade_date} 的ST股票列表...", end='', flush=True)

        df_st = None
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df_st = pro.stock_st(trade_date=trade_date)
                last_error = None
                break
            except Exception as e:
                last_error = e
                err_msg = str(e)
                # 限流错误: 长 sleep 跨过限流窗口
                if '频率超限' in err_msg or 'rate' in err_msg.lower() or '每分钟' in err_msg:
                    wait = RATE_LIMIT_SLEEP
                    print(f"\r[{i}/{len(dates_to_fetch)}] {trade_date} 第{attempt}次触发限流, 等待 {wait}s 后重试...")
                    time.sleep(wait)
                else:
                    # 其它错误: 指数退避
                    wait = min(30, 2 ** attempt)
                    print(f"\r[{i}/{len(dates_to_fetch)}] {trade_date} 第{attempt}次失败({err_msg[:60]}), 等待 {wait}s 后重试...")
                    time.sleep(wait)

        if last_error is not None:
            # 重试耗尽仍失败
            error_count += 1
            failed_dates.append(trade_date)
            print(f"\r[{i}/{len(dates_to_fetch)}] 获取 {trade_date} 最终失败({MAX_RETRIES}次重试均失败): {last_error}")
            continue

        # 成功 (含空返回)
        success_dates.append(trade_date)
        if df_st is not None and len(df_st) > 0:
            all_data.append(df_st)
            success_count += 1
            print(f"\r[{i}/{len(dates_to_fetch)}] 获取 {trade_date} 成功: {len(df_st)} 只ST股票")
        else:
            skip_count += 1
            print(f"\r[{i}/{len(dates_to_fetch)}] 获取 {trade_date}: 无ST股票")

        # API限流: 主动 sleep 避免触发限流
        time.sleep(SLEEP_INTERVAL)

    print(f"\n\n{'=' * 80}")
    print("数据获取完成")
    print(f"{'=' * 80}")
    print(f"成功: {success_count} 天")
    print(f"跳过(无ST): {skip_count} 天")
    print(f"失败: {error_count} 天")

    # 持久化失败日期, 便于后续单独重跑
    if failed_dates:
        try:
            FAILED_DATES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(FAILED_DATES_FILE, 'w', encoding='utf-8') as f:
                for d in failed_dates:
                    f.write(f"{d}\n")
            print(f"[INFO] 失败日期已写入: {FAILED_DATES_FILE}")
            print(f"[INFO] 可使用 -s {failed_dates[0]} -e {failed_dates[-1]} 重跑失败区间")
        except Exception as e:
            print(f"[WARNING] 写入失败日期文件出错: {e}")

    if not all_data and not success_dates:
        print("\n[WARNING] 未获取到任何数据, 不更新文件")
        return

    # 合并数据 (可能为空, 即所有成功日期均无 ST 股票)
    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)
    else:
        new_df = pd.DataFrame()
    print(f"\n[INFO] 获取数据总量: {len(new_df)} 条记录")

    # 保存或更新数据 (仅以 success_dates 作为删除依据, 失败日期旧数据保持不动)
    save_data(new_df, existing_df, success_dates)


def save_data(new_df, existing_df, success_dates):
    """保存数据到parquet文件

    Args:
        new_df: 本次成功获取的数据 (可能为空 DataFrame)
        existing_df: 已有数据
        success_dates: 本次确认成功获取的日期列表 (含空返回); 仅这些日期的旧数据会被替换
    """
    print(f"\n[INFO] 保存数据到: {OUTPUT_FILE}")

    try:
        if existing_df is None:
            # 没有现有数据
            if new_df is None or len(new_df) == 0:
                print("[INFO] 无现有数据且本次未获取到 ST 记录, 不创建文件")
                return
            new_df.to_parquet(OUTPUT_FILE, index=False)
            print(f"[SUCCESS] 数据已保存: {len(new_df)} 条记录")
        else:
            # 有现有数据, 仅删除本次确认成功获取的日期, 失败日期旧数据保留
            success_dates_set = set(str(d) for d in success_dates)

            existing_df_filtered = existing_df[
                ~existing_df['trade_date'].astype(str).isin(success_dates_set)
            ]

            # 合并数据 (new_df 可能为空)
            if new_df is not None and len(new_df) > 0:
                merged_df = pd.concat([existing_df_filtered, new_df], ignore_index=True)
            else:
                merged_df = existing_df_filtered.reset_index(drop=True)

            # 按日期排序
            if 'trade_date' in merged_df.columns:
                merged_df = merged_df.sort_values('trade_date').reset_index(drop=True)

            # 保存
            merged_df.to_parquet(OUTPUT_FILE, index=False)

            removed = len(existing_df) - len(existing_df_filtered)
            added = len(new_df) if new_df is not None else 0
            print(f"[SUCCESS] 数据已更新:")
            print(f"  - 替换了 {removed} 条旧记录 (来自 {len(success_dates_set)} 个成功日期)")
            print(f"  - 新增了 {added} 条记录")
            print(f"  - 总记录数: {len(merged_df)} 条")
            if len(merged_df) > 0 and 'trade_date' in merged_df.columns:
                print(f"  - 日期范围: {merged_df['trade_date'].min()} ~ {merged_df['trade_date'].max()}")

    except Exception as e:
        print(f"[ERROR] 保存数据失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='获取ST股票列表数据，支持增量更新',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python fetch_st_stocks.py -s 20240101 -e 20240131
  python fetch_st_stocks.py --start 20240101 --end 20240131
  python fetch_st_stocks.py -s 20240101 -e 20240131  # 增量更新
  python fetch_st_stocks.py -s 20240115 -e 20240120  # 覆盖已有数据
        '''
    )

    # 添加参数
    parser.add_argument(
        '-s', '--start',
        type=str,
        required=True,
        help='开始日期，格式: YYYYMMDD (例如: 20240101)'
    )

    parser.add_argument(
        '-e', '--end',
        type=str,
        required=True,
        help='结束日期，格式: YYYYMMDD (例如: 20240131)'
    )

    # 解析参数
    try:
        args = parser.parse_args()
        start_date = args.start
        end_date = args.end
    except SystemExit as e:
        # argparse 会自动处理 -h/--help，其他错误也会退出
        if e.code != 0:
            print("\n[ERROR] 参数解析失败")
        return

    # 验证日期格式
    try:
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
    except Exception as e:
        print(f"[ERROR] 日期格式错误: {e}")
        print("请使用格式: YYYYMMDD (例如: 20240101)")
        return

    # 检查日期范围
    if start_date > end_date:
        print(f"[ERROR] 开始日期({start_date}) 不能晚于结束日期({end_date})")
        return

    # 执行获取
    fetch_st_stocks(start_date, end_date)


if __name__ == "__main__":
    main()
