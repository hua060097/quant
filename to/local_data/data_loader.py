import pyarrow.dataset as ds
import pyarrow as pa
import pandas as pd

class DataLoader:
    def _read_parquet_data(self, dataset_path, date_range=None, stock_codes=None, fields=None):
        """从Parquet文件读取数据"""
        logger.info(f"尝试使用 pyarrow.dataset 从 {dataset_path} 读取数据，日期范围: {date_range[0] if date_range else None} 到 {date_range[1] if date_range else None}")
        
        filters = []
        
        # 添加日期过滤条件
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            
            # 尝试两种可能的日期字段名
            date_field_options = ['date_str', 'trade_date']
            
            # 首先检查哪个日期字段存在
            try:
                # 获取数据集的字段信息
                dataset = ds.dataset(dataset_path, format="parquet")
                schema = dataset.schema
                field_names = [field.name for field in schema]
                
                # 确定使用哪个日期字段
                date_field = None
                for option in date_field_options:
                    if option in field_names:
                        date_field = option
                        break
                
                if date_field:
                    logger.info(f"使用字段 '{date_field}' 作为日期过滤条件")
                    filters.append((ds.field(date_field) >= start_date))
                    filters.append((ds.field(date_field) <= end_date))
                else:
                    # 如果没有日期字段，尝试使用日期类型字段
                    if 'date' in field_names:
                        logger.info(f"使用字段 'date' 作为日期过滤条件")
                        # 将日期字符串转换为datetime格式进行过滤
                        filters.append((ds.field('date') >= pd.to_datetime(start_date)))
                        filters.append((ds.field('date') <= pd.to_datetime(end_date)))
                    else:
                        logger.warning(f"数据集中没有找到可用的日期字段: {field_names}")
            except Exception as e:
                logger.error(f"检查数据集字段时出错: {str(e)}")
                # 使用默认字段
                logger.info(f"使用默认字段 'date_str' 作为日期过滤条件")
                filters.append((ds.field('date_str') >= start_date))
                filters.append((ds.field('date_str') <= end_date)) 