"""
选股模块

T日收盘后运行,基于模型预测进行选股
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import PathConfig, TradingConfig, FilterConfig
from .model_loader import ModelLoader


class StockSelector:
    """选股器"""
    
    def __init__(self, model_loader: ModelLoader):
        self.logger = logging.getLogger(__name__)
        self.model_loader = model_loader
        self.selections_file = PathConfig.STOCK_SELECTIONS_FILE
    
    def select_stocks(self, date: str, num_stocks: int, factor_data: pd.DataFrame) -> Dict[str, Any]:
        """
        选股主流程
        
        Args:
            date: 选股日期(格式: YYYYMMDD)
            num_stocks: 选股数量(1-5)
            factor_data: 因子数据
        
        Returns:
            Dict: 选股结果
        """
        try:
            self.logger.info(f"开始选股: 日期={date}, 数量={num_stocks}")
            
            # 1. 提取当日数据
            date_dt = pd.to_datetime(date, format='%Y%m%d')
            daily_data = factor_data[factor_data['date'] == date_dt].copy()
            
            if daily_data.empty:
                self.logger.error(f"未找到日期{date}的数据")
                return None
            
            self.logger.info(f"当日数据: {len(daily_data)}只股票")
            
            # 2. 基础过滤
            filtered_data = self._apply_filters(daily_data, date)
            
            if filtered_data.empty:
                self.logger.warning("过滤后无可选股票")
                return self._create_empty_result(date, num_stocks)
            
            self.logger.info(f"过滤后剩余: {len(filtered_data)}只股票")
            
            # 3. 模型预测
            predictions = self._predict_stocks(filtered_data)
            
            if predictions is None:
                self.logger.error("预测失败")
                return None
            
            # 4. 选股排序
            selected_stocks = self._rank_and_select(filtered_data, predictions, num_stocks)
            
            # 5. 保存结果
            result = self._create_result(date, selected_stocks, num_stocks)
            self._save_selection(date, result)
            
            self.logger.info(f"选股完成: 选出{len(selected_stocks)}只股票")
            return result
            
        except Exception as e:
            self.logger.error(f"选股失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _apply_filters(self, data: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        应用过滤条件（与main.py pred_stocks完全一致）

        过滤规则:
        1. ST股票
        2. T日涨停股票 (is_limit_up == 1)
        3. T日跌停股票 (is_limit_down == 1)
        4. 新股 (上市不足30天)
        5. 流动性不足 (amount < 1000万)

        Args:
            data: 股票数据
            date: 日期字符串

        Returns:
            pd.DataFrame: 过滤后的数据
        """
        filtered = data.copy()
        original_count = len(filtered)

        # 1. 过滤ST股票
        if 'stock_code' in filtered.columns:
            before_st = len(filtered)
            st_filter = ~filtered['stock_code'].str.contains('ST', case=False, na=False)
            filtered = filtered[st_filter].copy()
            st_filtered = before_st - len(filtered)
            if st_filtered > 0:
                self.logger.info(f"  [过滤] ST股过滤: {before_st} -> {len(filtered)} (过滤掉 {st_filtered})")

        # 2. 过滤T日涨停股票（使用is_limit_up字段）
        if 'is_limit_up' in filtered.columns:
            before_limit_up = len(filtered)
            filtered = filtered[filtered['is_limit_up'] != 1].copy()
            limit_up_filtered = before_limit_up - len(filtered)
            if limit_up_filtered > 0:
                self.logger.info(f"  [过滤] 涨停股过滤: {before_limit_up} -> {len(filtered)} (过滤掉 {limit_up_filtered})")

        # 3. 过滤T日跌停股票（使用is_limit_down字段）
        if 'is_limit_down' in filtered.columns:
            before_limit_down = len(filtered)
            filtered = filtered[filtered['is_limit_down'] != 1].copy()
            limit_down_filtered = before_limit_down - len(filtered)
            if limit_down_filtered > 0:
                self.logger.info(f"  [过滤] 跌停股过滤: {before_limit_down} -> {len(filtered)} (过滤掉 {limit_down_filtered})")

        if 'is_zhaban' in filtered.columns:
            before_limit_down = len(filtered)
            filtered = filtered[filtered['is_zhaban'] != 1].copy()
            limit_down_filtered = before_limit_down - len(filtered)
            if limit_down_filtered > 0:
               self.logger.info(f"  [过滤] 炸板股过滤: {before_limit_down} -> {len(filtered)} (过滤掉 {limit_down_filtered})")
            else:
               self.logger.info(f"无炸板")    
        else:
            self.logger.info(f"无炸板字段，{filtered.columns}")              

        # 4. 过滤新股（上市不足30天）
        if 'list_date' in filtered.columns:
            before_new_stock = len(filtered)
            date_dt = pd.to_datetime(date, format='%Y%m%d')
            filtered['list_date'] = pd.to_datetime(filtered['list_date'], format='%Y%m%d', errors='coerce')
            filtered['days_since_listing'] = (date_dt - filtered['list_date']).dt.days

            # 上市30天及以上
            new_stock_mask = filtered['days_since_listing'] >= 30
            filtered = filtered[new_stock_mask].copy()
            new_stock_filtered = before_new_stock - len(filtered)
            if new_stock_filtered > 0:
                self.logger.info(f"  [过滤] 新股过滤(<30天): {before_new_stock} -> {len(filtered)} (过滤掉 {new_stock_filtered})")

        # 5. 流动性过滤（成交额 < 1000万）
        if 'amount' in filtered.columns:
            before_liquidity = len(filtered)
            liquidity_threshold = 10_000_000  # 1000万
            filtered = filtered[filtered['amount'] >= liquidity_threshold].copy()
            liquidity_filtered = before_liquidity - len(filtered)
            if liquidity_filtered > 0:
                self.logger.info(f"  [过滤] 流动性过滤(amount<1000万): {before_liquidity} -> {len(filtered)} (过滤掉 {liquidity_filtered})")

        total_filtered = original_count - len(filtered)
        filter_rate = total_filtered / original_count * 100 if original_count > 0 else 0
        self.logger.info(f"  [过滤] 总计: {original_count} -> {len(filtered)} (过滤率 {filter_rate:.1f}%)")

        return filtered
    
    def _predict_stocks(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        模型预测（使用LightGBM单模型，与main.py pred_stocks一致）

        Args:
            data: 股票数据

        Returns:
            np.ndarray: 预测分数（ATR单位）
        """
        try:
            # 1. 加载模型和参数
            model = self.model_loader.models[0]  # LightGBM单模型
            scaler = self.model_loader.scalers[0]
            label_transformer = self.model_loader.label_transformers[0]
            feature_names = self.model_loader.feature_names

            # 2. 特征对齐
            X = self._align_features(data, feature_names)
            if X is None:
                return None

            # 3. 标准化（如果scaler存在）
            if scaler is not None:
                X_scaled = scaler.transform(X)
                self.logger.info("使用scaler进行特征标准化")
            else:
                X_scaled = X
                self.logger.warning("scaler为None，跳过标准化步骤")

            # 4. LightGBM预测
            y_pred_raw = model.predict(X_scaled, num_iteration=model.best_iteration)

            # 5. 逆变换（如果有label_transformer）
            if label_transformer is not None:
                y_pred = label_transformer.inverse_transform(y_pred_raw.reshape(-1, 1)).flatten()
            else:
                y_pred = y_pred_raw

            self.logger.info(f"✅ LightGBM预测完成: [{y_pred.min():.3f}, {y_pred.max():.3f}] ATR")

            return y_pred

        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _align_features(self, data: pd.DataFrame, feature_names: List[str]) -> Optional[np.ndarray]:
        """
        特征对齐
        
        Args:
            data: 股票数据
            feature_names: 模型特征名称列表
        
        Returns:
            np.ndarray: 对齐后的特征矩阵
        """
        try:
            aligned_data = []
            missing_features = []
            
            for feature in feature_names:
                if feature in data.columns:
                    aligned_data.append(data[feature].values)
                else:
                    aligned_data.append(np.zeros(len(data)))
                    missing_features.append(feature)
            
            if missing_features:
                self.logger.warning(f"缺失特征(用0填充): {len(missing_features)}个")
            
            X = np.column_stack(aligned_data)
            self.logger.info(f"特征对齐完成: {X.shape}")
            
            return X
            
        except Exception as e:
            self.logger.error(f"特征对齐失败: {e}")
            return None
    
    def _rank_and_select(self, data: pd.DataFrame, predictions: np.ndarray, num_stocks: int) -> List[Dict]:
        """
        排序并选股（与main.py pred_stocks一致，不使用阈值过滤）

        Args:
            data: 股票数据
            predictions: 预测分数
            num_stocks: 选股数量

        Returns:
            List[Dict]: 选中的股票列表
        """
        # 添加预测分数
        data = data.copy()
        data['predicted_score'] = predictions

        # 排序（按预测值降序）
        data = data.sort_values('predicted_score', ascending=False)

        # 选取Top-N（与main.py pred_stocks一致，不使用阈值过滤）
        selected = data.head(num_stocks)

        # 构建结果
        stocks = []
        for idx, (_, row) in enumerate(selected.iterrows(), 1):
            stocks.append({
                'code': row['stock_code'],
                'score': float(row['predicted_score']),
                'rank': idx
            })

        return stocks
    
    def _create_result(self, date: str, stocks: List[Dict], num_stocks: int) -> Dict:
        """创建选股结果"""
        return {
            'stocks': stocks,
            'num_stocks': num_stocks,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _create_empty_result(self, date: str, num_stocks: int) -> Dict:
        """创建空结果"""
        return {
            'stocks': [],
            'num_stocks': num_stocks,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _save_selection(self, date: str, result: Dict):
        """
        保存选股结果到JSON文件
        
        Args:
            date: 日期
            result: 选股结果
        """
        try:
            # 读取现有数据
            if self.selections_file.exists():
                with open(self.selections_file, 'r', encoding='utf-8') as f:
                    selections = json.load(f)
            else:
                selections = {}
            
            # 更新数据
            selections[date] = result
            
            # 保存
            with open(self.selections_file, 'w', encoding='utf-8') as f:
                json.dump(selections, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"选股结果已保存: {self.selections_file}")
            
        except Exception as e:
            self.logger.error(f"保存选股结果失败: {e}")

