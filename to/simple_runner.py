#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块轮动模型运行脚本
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# 导入路径工具
try:
    from to.path_utils import get_project_path, get_model_path, normalize_path, ensure_dir
except ImportError:
    # 如果path_utils.py不存在，则提供兼容性处理
    def get_project_path(relative_path=None):
        """返回相对于项目根目录的绝对路径"""
        project_root = Path(__file__).resolve().parent.parent
        if relative_path:
            return project_root / relative_path
        return project_root
    
    def get_model_path(relative_path=None):
        """返回相对于模型目录的绝对路径"""
        model_root = get_project_path('to/models')
        if relative_path:
            return model_root / relative_path
        return model_root
    
    def normalize_path(path_str):
        """标准化路径字符串"""
        return path_str
        
    def ensure_dir(path):
        """确保目录存在，如果不存在则创建"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

# 将项目根目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 导入项目模块
from to.train_models import train_sector_rotation_model, generate_stock_signals_from_sector_prediction

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_sector_rotation_model(date, lookback_short=30, lookback_long=90, topk=3, stocks_per_sector=3,
                             use_lnn=True, save_predictions=True, verbose=True):
    """运行板块轮动模型

    Args:
        date: 目标日期，格式为YYYYMMDD
        lookback_short: 短期回溯天数
        lookback_long: 长期回溯天数
        topk: 选择的板块数量
        stocks_per_sector: 每个板块选择的股票数量
        use_lnn: 是否使用液体神经网络
        save_predictions: 是否保存预测结果
        verbose: 是否打印详细信息
    """
    # 导入训练模块
    from to import train_models
    
    # 设置模型和信号保存目录
    model_dir = ensure_dir(get_model_path('sector_rotation'))
    signal_dir = ensure_dir(get_project_path('to/signals'))
    
    # 运行模型训练和预测
    if verbose:
        print(f"正在运行板块轮动模型，目标日期: {date}")
        print(f"使用参数: 短期回溯={lookback_short}天, 长期回溯={lookback_long}天, 使用LNN={use_lnn}")
    
    signals = train_models.run_sector_rotation_for_date(
        target_date=date,
        lookback_short=lookback_short,
        lookback_long=lookback_long,
        top_sectors=topk,
        stocks_per_sector=stocks_per_sector,
        use_lnn=use_lnn,
        save_model=True,
        model_dir=model_dir
    )
    
    if save_predictions and signals:
        # 保存预测结果
        signal_file = os.path.join(signal_dir, f"signals_{date}.json")
        with open(signal_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(signals, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"交易信号已保存至: {signal_file}")
    
    return signals

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行板块轮动模型")
    
    parser.add_argument("--date", type=str, help="指定日期，格式为YYYYMMDD，默认为当前日期")
    parser.add_argument("--start_date", type=str, help="开始日期，格式为YYYYMMDD")
    parser.add_argument("--end_date", type=str, help="结束日期，格式为YYYYMMDD")
    parser.add_argument("--lookback_short", type=int, default=30, help="短期回溯天数，默认30")
    parser.add_argument("--lookback_long", type=int, default=90, help="长期回溯天数，默认90")
    parser.add_argument("--forward_days", type=int, default=5, help="预测未来几天，默认5")
    parser.add_argument("--feature_dir", type=str, help="特征数据目录")
    parser.add_argument("--model_dir", type=str, help="模型保存目录")
    parser.add_argument("--signal_dir", type=str, help="交易信号保存目录")
    parser.add_argument("--use_lnn", action="store_true", help="使用LNN模型进行时间序列建模")
    parser.add_argument("--top_sectors", type=int, default=3, help="选择排名靠前的板块数量")
    parser.add_argument("--stocks_per_sector", type=int, default=3, help="每个板块选择的股票数量")
    parser.add_argument("--no_signals", action="store_true", help="不生成交易信号")
    
    return parser.parse_args()

def generate_date_list(start_date, end_date):
    """生成日期列表"""
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    
    date_list = []
    current = start
    
    while current <= end:
        date_list.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    
    return date_list

def main():
    """主函数"""
    args = parse_args()
    
    # 如果指定了开始和结束日期，则处理日期范围
    if args.start_date and args.end_date:
        dates = generate_date_list(args.start_date, args.end_date)
        logger.info(f"将处理 {len(dates)} 个日期，从 {args.start_date} 到 {args.end_date}")
        
        success_count = 0
        for date in dates:
            logger.info(f"\n{'='*50}\n处理日期: {date}\n{'='*50}")
            success = run_sector_rotation_model(
                date=date,
                lookback_short=args.lookback_short,
                lookback_long=args.lookback_long,
                topk=args.top_sectors,
                stocks_per_sector=args.stocks_per_sector,
                use_lnn=args.use_lnn,
                save_predictions=not args.no_signals
            )
            
            if success:
                success_count += 1
        
        logger.info(f"\n总结: 成功处理 {success_count}/{len(dates)} 个日期")
        
    else:
        # 处理单个日期
        date = args.date if args.date else datetime.now().strftime('%Y%m%d')
        logger.info(f"处理日期: {date}")
        
        success = run_sector_rotation_model(
            date=date,
            lookback_short=args.lookback_short,
            lookback_long=args.lookback_long,
            topk=args.top_sectors,
            stocks_per_sector=args.stocks_per_sector,
            use_lnn=args.use_lnn,
            save_predictions=not args.no_signals
        )
        
        exit_code = 0 if success else 1
        sys.exit(exit_code)

if __name__ == "__main__":
    main() 