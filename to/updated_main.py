def main():
    """主函数，用于命令行调用"""
    parser = argparse.ArgumentParser(description="板块轮动模型训练与预测")
    parser.add_argument('--mode', type=str, choices=['train', 'select_stocks'], 
                      help='操作模式：train (训练模型), select_stocks (选股)')
    parser.add_argument('--start_date', type=str, help='开始日期，格式为YYYYMMDD')
    parser.add_argument('--end_date', type=str, help='结束日期，格式为YYYYMMDD')
    parser.add_argument('--date', type=str, help='预测日期，格式为YYYYMMDD')
    parser.add_argument('--top_n', type=int, default=50, help='每个交易日选择前N只股票')
    parser.add_argument('--max_workers', type=int, help='并行训练的进程数')
    parser.add_argument('--sector_list', type=str, help='板块列表文件路径，用于板块轮动模型训练')
    parser.add_argument('--model_dir', type=str, help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='sector_rotation_model', help='模型名称')
    parser.add_argument('--save_results', type=str, help='保存结果的文件路径')
    parser.add_argument('--stock_list', type=str, help='股票列表文件路径，用于选股')
    parser.add_argument('--use_cache', action='store_true', help='是否使用特征缓存，默认为True')
    parser.add_argument('--no_cache', dest='use_cache', action='store_false', help='不使用特征缓存')
    parser.add_argument('--show_date', type=str, help='查看特定日期的选股结果')
    parser.set_defaults(use_cache=True)
    
    args = parser.parse_args()
    
    logger.setLevel(logging.DEBUG)
    # 创建SectorModelTrainer实例
    trainer = SectorModelTrainer(model_dir=args.model_dir)
    
    # 加载板块列表
    sector_list = []
    try:
        # 优先使用参数指定的板块列表文件
        if args.sector_list and os.path.exists(args.sector_list):
            with open(args.sector_list, 'r', encoding='utf-8') as f:
                sector_list = [line.strip() for line in f if line.strip()]
            logger.info(f"从文件 {args.sector_list} 加载了 {len(sector_list)} 个板块")
        else:
            # 尝试从stock_sectors_mapping模块获取
            try:
                # 将当前工作目录添加到sys.path
                sys.path.append(os.getcwd())
                from stock_sectors_mapping import get_tng_sectors
                sector_list = get_tng_sectors()
                logger.info(f"从get_tng_sectors加载了 {len(sector_list)} 个板块")
            except ImportError:
                # 如果导入失败，尝试从to目录导入
                try:
                    sys.path.append(str(get_project_root()))
                    from to.stock_sectors_mapping import get_tng_sectors
                    sector_list = get_tng_sectors()
                    logger.info(f"从to.stock_sectors_mapping加载了 {len(sector_list)} 个板块")
                except ImportError:
                    logger.error("无法导入get_tng_sectors函数，请提供板块列表文件")
    except Exception as e:
        logger.error(f"获取板块列表时出错: {e}")
        logger.error(traceback.format_exc())
    
    # 加载股票列表（如果需要）
    stock_list = None
    if args.stock_list and os.path.exists(args.stock_list):
        try:
            with open(args.stock_list, 'r', encoding='utf-8') as f:
                stock_list = [line.strip() for line in f if line.strip()]
            logger.info(f"从文件 {args.stock_list} 加载了 {len(stock_list)} 只股票")
        except Exception as e:
            logger.error(f"加载股票列表时出错: {e}")
    
    # 运行模式
    if args.mode == 'train':
        # 训练板块轮动模型（如果已存在模型，会先尝试加载）
        if not sector_list:
            print("错误: 训练板块轮动模型需要提供板块列表")
            return
                    
        # 调用训练函数 - 内部已包含了2025年2月生命周期预测和结果保存
        model_result = trainer.train_rotation_model(
                sector_list=sector_list,
                start_date=args.start_date,
                end_date=args.end_date,
                model_name=args.model_name)
            
        if model_result:
            print(f"板块轮动模型训练成功，2025年2月的生命周期阶段预测结果已保存在results目录下")
        else:
            print("板块轮动模型训练失败")
    
    elif args.mode == 'select_stocks':
        # 使用变化点方法进行选股
        top_n = args.top_n if args.top_n else 50
        model_name = args.model_name if args.model_name else "stock_selector"
        
        # 设置日期范围
        start_date = args.start_date
        end_date = args.end_date if args.end_date else datetime.now().strftime("%Y%m%d")
        
        print(f"开始运行基于变化点检测的智能选股框架...")
        print(f"- 时间范围: {start_date} 至 {end_date}")
        print(f"- 选出数量: 每个交易日前 {top_n} 只股票")
        print(f"- 模型名称: {model_name}")
        print(f"- 使用缓存: {'是' if args.use_cache else '否'}")
        print("-" * 80)
        
        # 调用选股函数
        selected_stocks = trainer.select_stocks_by_change_points(
            start_date=start_date,
            end_date=end_date,
            model_name=model_name,
            top_n=top_n,
            use_cache=args.use_cache
        )
        
        if selected_stocks:
            # 是否查看特定日期的结果
            if args.show_date and 'selected_stocks_by_date' in selected_stocks and args.show_date in selected_stocks['selected_stocks_by_date']:
                date_to_show = args.show_date
                stocks_to_show = selected_stocks['selected_stocks_by_date'][date_to_show]
                
                print(f"\n日期 {date_to_show} 的选股结果 (共选出 {len(stocks_to_show)} 只股票):")
                print("-" * 80)
                print(f"{'排名':<6}{'股票代码':<12}{'股票名称':<15}{'预期收益率':>12}{'概率':>10}{'置信度':>10}{'得分':>10}")
                print("-" * 80)
                
                for stock in stocks_to_show:
                    print(f"{stock.get('rank', 0):<6}{stock['stock_code']:<12}{stock.get('stock_name', ''):<15}{stock['expected_return']*100:>12.2f}%{stock['probability']*100:>9.1f}%{stock['confidence']*100:>9.1f}%{stock['score']*100:>9.2f}")
            
            # 查看按日期选股的总体情况
            elif 'selected_stocks_by_date' in selected_stocks:
                dates = selected_stocks.get('dates', [])
                if not dates and 'selected_stocks_by_date' in selected_stocks:
                    dates = sorted(selected_stocks['selected_stocks_by_date'].keys())
                
                print(f"\n按日期选股结果总览 (共 {len(dates)} 个交易日):")
                print("-" * 80)
                print(f"{'日期':<12}{'选出股票数':>12}")
                print("-" * 80)
                
                for date in dates:
                    if date in selected_stocks['selected_stocks_by_date']:
                        stocks_count = len(selected_stocks['selected_stocks_by_date'][date])
                        print(f"{date:<12}{stocks_count:>12}")
                
                print("\n要查看特定日期的选股结果，请使用 --show_date 参数")
                print(f"例如: python train_sector_models.py --mode select_stocks --show_date {dates[0] if dates else '20250101'}")
            
            # 显示默认结果 (最新日期或全局)
            else:
                print(f"\n选股结果 (日期: {selected_stocks.get('date', '')}, 共选出 {len(selected_stocks.get('selected_stocks', []))} 只股票):")
                print("-" * 80)
                print(f"{'排名':<6}{'股票代码':<12}{'股票名称':<15}{'预期收益率':>12}{'概率':>10}{'置信度':>10}{'得分':>10}")
                print("-" * 80)
                
                for i, stock in enumerate(selected_stocks.get('selected_stocks', [])):
                    if i < 20:  # 只显示前20只
                        print(f"{i+1:<6}{stock['stock_code']:<12}{stock.get('stock_name', ''):<15}{stock['expected_return']*100:>12.2f}%{stock['probability']*100:>9.1f}%{stock['confidence']*100:>9.1f}%{stock['score']*100:>9.2f}")
                
                if len(selected_stocks.get('selected_stocks', [])) > 20:
                    print(f"... 还有 {len(selected_stocks['selected_stocks'])-20} 只股票未显示")
            
            # 显示评估结果
            if 'evaluation' in selected_stocks:
                eval_results = selected_stocks['evaluation']
                if 'overall_summary' in eval_results:
                    summary = eval_results['overall_summary']
                    if 'avg_actual_return' in summary:
                        print("\n整体评估结果:")
                        print(f"- 平均实际收益率: {summary['avg_actual_return']:.2f}%")
                        print(f"- 方向准确率: {summary['direction_accuracy']*100:.1f}%")
                        print(f"- 胜率: {summary['win_rate']*100:.1f}%")
                        if 'performance_rating' in summary:
                            print(f"- 表现评级: {summary['performance_rating']}")
                
                # 显示日平均表现
                if 'daily_summaries' in eval_results and eval_results['daily_summaries']:
                    daily_returns = [s['avg_actual_return'] for s in eval_results['daily_summaries'] 
                                    if 'avg_actual_return' in s]
                    if daily_returns:
                        avg_daily_return = sum(daily_returns) / len(daily_returns)
                        print(f"- 日均收益率: {avg_daily_return:.2f}%")
                
            print(f"\n选股结果已保存到 {trainer.model_dir / f'{model_name}_results.json'}")
            
            if 'selected_stocks_by_date' in selected_stocks:
                daily_results_dir = trainer.model_dir / f"{model_name}_daily_results"
                print(f"各日期选股结果已保存到目录: {daily_results_dir}")
        else:
            print("选股失败，请检查日志获取详细信息")
    else:
        print("请指定有效的运行模式: --mode train 或 --mode select_stocks")
        print("\n示例用法:")
        print("训练板块轮动模型:")
        print("  python train_sector_models.py --mode train --start_date 20230101 --end_date 20231231 --model_name sector_rotation_model")
        print("\n智能选股:")
        print("  python train_sector_models.py --mode select_stocks --start_date 20230101 --end_date 20231231 --top_n 50 --model_name stock_selector")
        print("  python train_sector_models.py --mode select_stocks --start_date 20230101 --end_date 20231231 --top_n 50 --no_cache  # 不使用缓存")
        print("  python train_sector_models.py --mode select_stocks --show_date 20250101  # 查看特定日期的选股结果") 