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

from .config import PathConfig, TradingConfig, FilterConfig, StrategyConfig
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
            filtered_data = self._attach_prediction_meta(filtered_data, predictions)
            selected_stocks = self._rank_and_select(filtered_data, num_stocks)

            # 5. 默认过滤规则：先排序，再判断候选是否可交易
            self._annotate_exhaustion_flags(selected_stocks, filtered_data)

            # 6. q40 动态切换标注（与 pred_stocks 口径一致，仅新增 q40，不加 h2c）
            q40_info = self._compute_q40_pick(filtered_data, selected_stocks)

            # 7. 保存结果
            result = self._create_result(date, selected_stocks, len(selected_stocks), q40_info)
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
    
    def _attach_prediction_meta(self, data: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        """附加模型预测分数及当日候选池内的预测z-score。"""
        data = data.copy()
        data['predicted_score'] = predictions
        scores = pd.to_numeric(data['predicted_score'], errors='coerce')
        score_std = float(np.nanstd(scores, ddof=0))
        if score_std > 0:
            data['score_z_in_day'] = (scores - float(np.nanmean(scores))) / score_std
        else:
            data['score_z_in_day'] = np.nan
        return data

    def _rank_and_select(self, data: pd.DataFrame, num_stocks: int) -> List[Dict]:
        """
        排序并选股（与main.py pred_stocks一致，不使用阈值过滤）

        Args:
            data: 股票数据
            num_stocks: 选股数量

        Returns:
            List[Dict]: 选中的股票列表
        """
        data = data.copy()

        # 排序（按预测值降序）
        data = data.sort_values('predicted_score', ascending=False)

        # 选取Top-N（保留至少Top5，保证 q40 选到 Top2 时也能落到结果里）
        topn = max(int(num_stocks), int(StrategyConfig.PRED_TOPN), 2)
        selected = data.head(topn)

        # 构建结果
        stocks = []
        for idx, (_, row) in enumerate(selected.iterrows(), 1):
            stocks.append({
                'code': row['stock_code'],
                'score': float(row['predicted_score']),
                'score_z_in_day': float(row['score_z_in_day']) if pd.notna(row.get('score_z_in_day', np.nan)) else None,
                'rank': idx,
                'q40_selected': False,
                'exhaustion_filtered': False,
                'exhaustion_filter_reason': '',
                'tradeable': True,
            })

        return stocks

    def _is_exhaustion_filtered_row(self, row: pd.Series) -> tuple[bool, str, str]:
        """与主脚本默认过滤规则保持一致，并返回详细值。"""
        if not StrategyConfig.ENABLE_EXHAUSTION_BUY_FILTER:
            return False, '', 'default_filter_disabled'

        strategy = StrategyConfig.EXHAUSTION_FILTER_STRATEGY

        if strategy == 'three_rule_v3':
            rma_val = pd.to_numeric(row.get('runup_ma5_exhaustion', np.nan), errors='coerce')
            cpv_val = pd.to_numeric(row.get('cpv_factor', np.nan), errors='coerce')
            blowoff_val = pd.to_numeric(row.get('prev_day_blowoff_risk', np.nan), errors='coerce')
            sector_val = pd.to_numeric(row.get('sector_momentum_accel_rank', np.nan), errors='coerce')
            sl_val = pd.to_numeric(row.get('sector_leadership_persistence_v3', np.nan), errors='coerce')
            ret20_val = pd.to_numeric(row.get('return_20d_intra_z', np.nan), errors='coerce')
            liq_val = pd.to_numeric(row.get('market_liquidity_change', np.nan), errors='coerce')
            disp_val = pd.to_numeric(row.get('mkt_dispersion_shock_v2', np.nan), errors='coerce')
            adr_val = pd.to_numeric(row.get('market_advance_decline_ratio', np.nan), errors='coerce')
            ulcer_val = pd.to_numeric(row.get('ulcer_index', np.nan), errors='coerce')
            mom_peak_val = pd.to_numeric(row.get('momentum_peak_decay_3d', np.nan), errors='coerce')
            sector_rank_val = pd.to_numeric(row.get('intra_sector_momentum_rank', np.nan), errors='coerce')
            top10_val = pd.to_numeric(row.get('top10_10d_strength_level_v3', np.nan), errors='coerce')
            buy_sm_val = pd.to_numeric(row.get('buy_sm_amount_rate', np.nan), errors='coerce')
            non_first_pct20_val = pd.to_numeric(row.get('prev_non_first_limit_up_performance_percentile_20d', np.nan), errors='coerce')
            kdj_j_val = pd.to_numeric(row.get('kdj_j_9', np.nan), errors='coerce')
            reg_severe_pct_val = pd.to_numeric(row.get('reg_severe_up_10d_ratio_pct', np.nan), errors='coerce')
            ma5_raw_val = pd.to_numeric(row.get('price_ma5_ratio_raw', np.nan), errors='coerce')
            ma_conv_val = pd.to_numeric(row.get('ma_convergence_score', np.nan), errors='coerce')
            ma5_pullback_val = pd.to_numeric(row.get('ma5_pullback_depth_10d', np.nan), errors='coerce')
            ret_accel_val = pd.to_numeric(row.get('ret_3d_vs_10d_accel', np.nan), errors='coerce')
            detail = (
                f"runup={rma_val:.2f}" if pd.notna(rma_val) else "runup=NaN"
            ) + ", " + (
                f"cpv={cpv_val:.2f}" if pd.notna(cpv_val) else "cpv=NaN"
            ) + ", " + (
                f"blowoff={blowoff_val:.2f}" if pd.notna(blowoff_val) else "blowoff=NaN"
            ) + ", " + (
                f"sector_accel={sector_val:.2f}" if pd.notna(sector_val) else "sector_accel=NaN"
            ) + ", " + (
                f"sector_leadership={sl_val:.2f}" if pd.notna(sl_val) else "sector_leadership=NaN"
            ) + ", " + (
                f"return_20d_intra_z={ret20_val:.2f}" if pd.notna(ret20_val) else "return_20d_intra_z=NaN"
            ) + ", " + (
                f"liq={liq_val:.2f}" if pd.notna(liq_val) else "liq=NaN"
            ) + ", " + (
                f"ulcer={ulcer_val:.2f}" if pd.notna(ulcer_val) else "ulcer=NaN"
            ) + ", " + (
                f"mom_peak={mom_peak_val:.2f}" if pd.notna(mom_peak_val) else "mom_peak=NaN"
            ) + ", " + (
                f"sector_rank={sector_rank_val:.2f}" if pd.notna(sector_rank_val) else "sector_rank=NaN"
            ) + ", " + (
                f"top10={top10_val:.2f}" if pd.notna(top10_val) else "top10=NaN"
            ) + ", " + (
                f"buy_sm={buy_sm_val:.2f}" if pd.notna(buy_sm_val) else "buy_sm=NaN"
            ) + ", " + (
                f"disp={disp_val:.2f}" if pd.notna(disp_val) else "disp=NaN"
            ) + ", " + (
                f"adr={adr_val:.2f}" if pd.notna(adr_val) else "adr=NaN"
            ) + ", " + (
                f"non_first_limit_pct20={non_first_pct20_val:.2f}" if pd.notna(non_first_pct20_val) else "non_first_limit_pct20=NaN"
            ) + ", " + (
                f"kdj_j_9_z={kdj_j_val:.2f}" if pd.notna(kdj_j_val) else "kdj_j_9_z=NaN"
            ) + ", " + (
                f"reg_severe_pct={reg_severe_pct_val:.4f}" if pd.notna(reg_severe_pct_val) else "reg_severe_pct=NaN"
            ) + ", " + (
                f"ma5_raw={ma5_raw_val:.2f}" if pd.notna(ma5_raw_val) else "ma5_raw=NaN"
            ) + ", " + (
                f"ma_conv={ma_conv_val:.4f}" if pd.notna(ma_conv_val) else "ma_conv=NaN"
            ) + ", " + (
                f"ma5_pullback={ma5_pullback_val:.4f}" if pd.notna(ma5_pullback_val) else "ma5_pullback=NaN"
            ) + ", " + (
                f"ret_accel={ret_accel_val:.2f}" if pd.notna(ret_accel_val) else "ret_accel=NaN"
            )

            rule1_hit = (
                pd.notna(rma_val) and pd.notna(cpv_val)
                and rma_val > StrategyConfig.EXHAUSTION_RUNUP_MA5_THRESHOLD
                and cpv_val <= StrategyConfig.EXHAUSTION_V3_CPV_THRESHOLD
            )
            rule2_hit = (
                pd.notna(blowoff_val) and pd.notna(sector_val)
                and (
                    (blowoff_val >= StrategyConfig.EXHAUSTION_V3_BLOWOFF_THRESHOLD and sector_val <= StrategyConfig.EXHAUSTION_V3_SECTOR_ACCEL_THRESHOLD)
                    or (blowoff_val > 5.0 and sector_val <= 0.50)   # XR2a: blowoff extreme
                )
            )
            rule3_hit = (
                pd.notna(sl_val) and pd.notna(ret20_val)
                and sl_val <= StrategyConfig.EXHAUSTION_V3_SECTOR_LEADERSHIP_THRESHOLD
                and ret20_val <= StrategyConfig.EXHAUSTION_V3_RETURN_20D_INTRA_Z_THRESHOLD
            )
            rule4_hit = (
                pd.notna(liq_val)
                and liq_val < StrategyConfig.EXHAUSTION_V3_MARKET_LIQ_THRESHOLD
                and (
                    (pd.notna(ulcer_val) and ulcer_val < StrategyConfig.EXHAUSTION_V3_ULCER_THRESHOLD)
                    or (pd.notna(mom_peak_val) and mom_peak_val > StrategyConfig.EXHAUSTION_V3_MOM_PEAK_THRESHOLD)
                    or (pd.notna(sector_rank_val) and sector_rank_val < StrategyConfig.EXHAUSTION_V3_INTRA_SECTOR_RANK_THRESHOLD)
                )
            )
            rule5_hit = (
                pd.notna(disp_val) and pd.notna(adr_val)
                and disp_val < StrategyConfig.EXHAUSTION_V3_MKT_DISP_THRESHOLD
                and adr_val < StrategyConfig.EXHAUSTION_V3_MKT_ADR_THRESHOLD
            )

            # Rule 6: 近3日有过跌停
            rule6_hit = (
                StrategyConfig.ENABLE_EXHAUSTION_LIMIT_DOWN_3D
                and row.get('limit_down_3d', 0) == 1
            )
            rule7_hit = (
                pd.notna(top10_val) and pd.notna(buy_sm_val)
                and top10_val >= StrategyConfig.EXHAUSTION_V3_TOP10_STRENGTH_THRESHOLD
                and buy_sm_val <= StrategyConfig.EXHAUSTION_V3_BUY_SM_AMOUNT_THRESHOLD
            )
            rule8_hit = (
                pd.notna(non_first_pct20_val) and pd.notna(kdj_j_val)
                and non_first_pct20_val >= StrategyConfig.EXHAUSTION_V3_NON_FIRST_LIMIT_UP_PERF_PCT20_THRESHOLD
                and kdj_j_val <= StrategyConfig.EXHAUSTION_V3_KDJ_J_Z_THRESHOLD
            )
            rule10_hit = (
                pd.notna(reg_severe_pct_val) and pd.notna(adr_val)
                and reg_severe_pct_val >= StrategyConfig.EXHAUSTION_V3_REG_SEVERE_UP_10D_PCT_THRESHOLD
                and adr_val >= StrategyConfig.EXHAUSTION_V3_REG_HOT_ADR_THRESHOLD
            )
            # Rule11: MA5乖离率 >= 9% + 均线收敛度 >= 0.20
            rule11_hit = (
                pd.notna(ma5_raw_val) and pd.notna(ma_conv_val)
                and ma5_raw_val >= StrategyConfig.EXHAUSTION_V3_MA5_RAW_THRESHOLD
                and ma_conv_val >= StrategyConfig.EXHAUSTION_V3_MA_CONVERGENCE_THRESHOLD
            )
            # Rule12: MA5回撤深度极深(<=-0.05) + 短期加速上涨(>=6) → 高位回撤加速衰竭
            rule12_hit = (
                pd.notna(ma5_pullback_val) and pd.notna(ret_accel_val)
                and ma5_pullback_val <= StrategyConfig.EXHAUSTION_V3_MA5_PULLBACK_DEPTH_THRESHOLD
                and ret_accel_val >= StrategyConfig.EXHAUSTION_V3_RET_3D_VS_10D_ACCEL_THRESHOLD
            )

            if rule1_hit or rule2_hit or rule3_hit or rule4_hit or rule5_hit or rule6_hit or rule7_hit or rule8_hit or rule10_hit or rule11_hit or rule12_hit:
                parts = []
                if rule1_hit:
                    parts.append(
                        f"runup({rma_val:.2f})>{StrategyConfig.EXHAUSTION_RUNUP_MA5_THRESHOLD}"
                        f" & cpv({cpv_val:.2f})<={StrategyConfig.EXHAUSTION_V3_CPV_THRESHOLD}"
                    )
                if rule2_hit:
                    if blowoff_val >= StrategyConfig.EXHAUSTION_V3_BLOWOFF_THRESHOLD and sector_val <= StrategyConfig.EXHAUSTION_V3_SECTOR_ACCEL_THRESHOLD:
                        r2r = f"blowoff({blowoff_val:.2f})>={StrategyConfig.EXHAUSTION_V3_BLOWOFF_THRESHOLD} & sector_accel({sector_val:.2f})<={StrategyConfig.EXHAUSTION_V3_SECTOR_ACCEL_THRESHOLD}"
                    else:
                        r2r = f"XR2a:blowoff({blowoff_val:.2f})>5.0 & sector_accel({sector_val:.2f})<=0.50"
                    parts.append(r2r)
                if rule3_hit:
                    parts.append(
                        f"sector_leadership({sl_val:.2f})<={StrategyConfig.EXHAUSTION_V3_SECTOR_LEADERSHIP_THRESHOLD}"
                        f" & return_20d_intra_z({ret20_val:.2f})<={StrategyConfig.EXHAUSTION_V3_RETURN_20D_INTRA_Z_THRESHOLD}"
                    )
                if rule4_hit:
                    r4_reasons = []
                    if pd.notna(ulcer_val) and ulcer_val < StrategyConfig.EXHAUSTION_V3_ULCER_THRESHOLD:
                        r4_reasons.append('ulcer=%.2f' % ulcer_val)
                    if pd.notna(mom_peak_val) and mom_peak_val > StrategyConfig.EXHAUSTION_V3_MOM_PEAK_THRESHOLD:
                        r4_reasons.append('mom_peak=%.1f' % mom_peak_val)
                    if pd.notna(sector_rank_val) and sector_rank_val < StrategyConfig.EXHAUSTION_V3_INTRA_SECTOR_RANK_THRESHOLD:
                        r4_reasons.append('sector_rank=%.2f' % sector_rank_val)
                    parts.append(
                        f"liq({liq_val:.2f})<{StrategyConfig.EXHAUSTION_V3_MARKET_LIQ_THRESHOLD}"
                        f" & ({'+'.join(r4_reasons)})"
                    )
                if rule5_hit:
                    parts.append(
                        f"disp({disp_val:.2f})<{StrategyConfig.EXHAUSTION_V3_MKT_DISP_THRESHOLD}"
                        f" & adr({adr_val:.2f})<{StrategyConfig.EXHAUSTION_V3_MKT_ADR_THRESHOLD}"
                    )
                if rule6_hit:
                    parts.append("limit_down_3d=1")
                if rule7_hit:
                    parts.append(
                        f"top10_strength({top10_val:.2f})>={StrategyConfig.EXHAUSTION_V3_TOP10_STRENGTH_THRESHOLD}"
                        f" & buy_sm_amount({buy_sm_val:.2f})<={StrategyConfig.EXHAUSTION_V3_BUY_SM_AMOUNT_THRESHOLD}"
                    )
                if rule8_hit:
                    parts.append(
                        f"non_first_limit_pct20({non_first_pct20_val:.2f})>={StrategyConfig.EXHAUSTION_V3_NON_FIRST_LIMIT_UP_PERF_PCT20_THRESHOLD}"
                        f" & kdj_j_9_z({kdj_j_val:.2f})<={StrategyConfig.EXHAUSTION_V3_KDJ_J_Z_THRESHOLD}"
                    )
                if rule10_hit:
                    parts.append(
                        f"reg_severe_up_10d_ratio_pct({reg_severe_pct_val:.4f})>={StrategyConfig.EXHAUSTION_V3_REG_SEVERE_UP_10D_PCT_THRESHOLD}"
                        f" & adr({adr_val:.2f})>={StrategyConfig.EXHAUSTION_V3_REG_HOT_ADR_THRESHOLD}"
                    )
                if rule11_hit:
                    parts.append(
                        f"ma5_raw({ma5_raw_val:.2f})>={StrategyConfig.EXHAUSTION_V3_MA5_RAW_THRESHOLD}"
                        f" & ma_conv({ma_conv_val:.2f})>={StrategyConfig.EXHAUSTION_V3_MA_CONVERGENCE_THRESHOLD}"
                    )
                if rule12_hit:
                    parts.append(
                        f"ma5_pullback({ma5_pullback_val:.4f})<={StrategyConfig.EXHAUSTION_V3_MA5_PULLBACK_DEPTH_THRESHOLD}"
                        f" & ret_accel({ret_accel_val:.2f})>={StrategyConfig.EXHAUSTION_V3_RET_3D_VS_10D_ACCEL_THRESHOLD}"
                    )
                return True, " | ".join(parts), detail
            return False, '', detail

        if strategy in ('runup_ma5', 'runup_or_blowoff'):
            runup_val = pd.to_numeric(row.get('runup_ma5_exhaustion', np.nan), errors='coerce')
            top10_val = pd.to_numeric(row.get('top10_10d_strength_level_v3', np.nan), errors='coerce')
            blowoff_val = pd.to_numeric(row.get('prev_day_blowoff_risk', np.nan), errors='coerce')
            sector_accel_val = pd.to_numeric(row.get('sector_momentum_accel_rank', np.nan), errors='coerce')
            market_ad_val = pd.to_numeric(row.get('market_advance_decline_ratio', np.nan), errors='coerce')
            detail = (
                f"runup_ma5_exhaustion={runup_val:.2f}" if pd.notna(runup_val) else "runup_ma5_exhaustion=NaN"
            ) + ", " + (
                f"top10_10d_strength_level_v3={top10_val:.2f}" if pd.notna(top10_val) else "top10_10d_strength_level_v3=NaN"
            ) + ", " + (
                f"prev_day_blowoff_risk={blowoff_val:.2f}" if pd.notna(blowoff_val) else "prev_day_blowoff_risk=NaN"
            ) + ", " + (
                f"sector_momentum_accel_rank={sector_accel_val:.2f}" if pd.notna(sector_accel_val) else "sector_momentum_accel_rank=NaN"
            ) + ", " + (
                f"market_advance_decline_ratio={market_ad_val:.2f}" if pd.notna(market_ad_val) else "market_advance_decline_ratio=NaN"
            )
            runup_hit = (
                pd.notna(runup_val)
                and pd.notna(top10_val)
                and runup_val > StrategyConfig.EXHAUSTION_RUNUP_MA5_THRESHOLD
                and top10_val >= StrategyConfig.EXHAUSTION_TOP10_10D_STRENGTH_THRESHOLD
            )

            if strategy == 'runup_ma5' and runup_hit:
                return (
                    True,
                    f"runup_ma5_exhaustion({runup_val:.2f})>{StrategyConfig.EXHAUSTION_RUNUP_MA5_THRESHOLD}"
                    f" AND top10_10d_strength_level_v3({top10_val:.2f})>={StrategyConfig.EXHAUSTION_TOP10_10D_STRENGTH_THRESHOLD}",
                    detail,
                )

            if strategy == 'runup_or_blowoff':
                blowoff_hit = (
                    pd.notna(blowoff_val)
                    and pd.notna(sector_accel_val)
                    and blowoff_val >= StrategyConfig.EXHAUSTION_PREV_DAY_BLOWOFF_THRESHOLD
                    and sector_accel_val <= StrategyConfig.EXHAUSTION_SECTOR_MOMENTUM_ACCEL_WEAK_THRESHOLD
                )
                market_ad_hit = (
                    pd.notna(market_ad_val)
                    and market_ad_val >= StrategyConfig.EXHAUSTION_MARKET_ADVANCE_DECLINE_RATIO_THRESHOLD
                )
                if runup_hit or blowoff_hit or market_ad_hit:
                    parts = []
                    if runup_hit:
                        parts.append(
                            f"runup_ma5_exhaustion({runup_val:.2f})>{StrategyConfig.EXHAUSTION_RUNUP_MA5_THRESHOLD}"
                            f" AND top10_10d_strength_level_v3({top10_val:.2f})>={StrategyConfig.EXHAUSTION_TOP10_10D_STRENGTH_THRESHOLD}"
                        )
                    if blowoff_hit:
                        parts.append(
                            f"prev_day_blowoff_risk({blowoff_val:.2f})>={StrategyConfig.EXHAUSTION_PREV_DAY_BLOWOFF_THRESHOLD}"
                            f" AND sector_momentum_accel_rank({sector_accel_val:.2f})<={StrategyConfig.EXHAUSTION_SECTOR_MOMENTUM_ACCEL_WEAK_THRESHOLD}"
                        )
                    if market_ad_hit:
                        parts.append(
                            f"market_advance_decline_ratio({market_ad_val:.2f})>={StrategyConfig.EXHAUSTION_MARKET_ADVANCE_DECLINE_RATIO_THRESHOLD}"
                        )
                    return True, " OR ".join(parts), detail
            return False, '', detail

        if strategy == 'close_below_ma5':
            close_below = row.get('close_below_ma5_after_runup', 0)
            turnover_z = pd.to_numeric(row.get('turnover_z_20', np.nan), errors='coerce')
            detail = (
                f"close_below_ma5_after_runup={int(close_below or 0)}, "
                + (f"turnover_z_20={turnover_z:.2f}" if pd.notna(turnover_z) else "turnover_z_20=NaN")
            )
            if int(close_below or 0) == 1 and pd.notna(turnover_z):
                if turnover_z > StrategyConfig.EXHAUSTION_TURNOVER_Z20_THRESHOLD:
                    return (
                        True,
                        f"close_below_ma5_after_runup=1 AND turnover_z_20({turnover_z:.2f})>{StrategyConfig.EXHAUSTION_TURNOVER_Z20_THRESHOLD}",
                        detail,
                    )
            return False, '', detail

        return False, '', 'strategy_not_matched'

    def _annotate_exhaustion_flags(self, selected_stocks: List[Dict[str, Any]], filtered_data: pd.DataFrame) -> None:
        if not selected_stocks:
            return
        row_map = filtered_data.set_index('stock_code')
        for stock in selected_stocks:
            code = stock.get('code')
            if code not in row_map.index:
                continue
            row = row_map.loc[code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            filtered, reason, detail = self._is_exhaustion_filtered_row(row)
            stock['exhaustion_filtered'] = bool(filtered)
            stock['exhaustion_filter_reason'] = reason
            stock['exhaustion_filter_detail'] = detail
            stock['tradeable'] = not bool(filtered)
    
    def _compute_q40_pick(self, filtered_data: pd.DataFrame, selected_stocks: List[Dict]) -> Dict[str, Any]:
        """按 main_alpha pred_stocks 的口径计算 q40 选股结果。"""
        default = {
            'enabled': False,
            'chosen_rank': 1,
            'threshold': None,
            'signal_value': None,
            'selected_code': selected_stocks[0]['code'] if selected_stocks else None,
        }

        if not StrategyConfig.ENABLE_DYNAMIC_RANK_SWITCH:
            return default
        if not selected_stocks or 'mkt_ma5_followthrough_v3' not in filtered_data.columns:
            return default

        try:
            factor_path = PathConfig.PROJECT_ROOT / 'factor_lab_data' / 'factors.parquet'
            ref = pd.read_parquet(factor_path, columns=['date', 'mkt_ma5_followthrough_v3'])
            ref['date'] = pd.to_datetime(ref['date'], errors='coerce')
            ref = (
                ref.dropna(subset=['date'])
                .groupby('date', as_index=False)
                .first()
                .sort_values('date')
                .reset_index(drop=True)
            )

            train_mask = (
                (ref['date'] >= pd.Timestamp(StrategyConfig.Q40_TRAIN_START)) &
                (ref['date'] <= pd.Timestamp(StrategyConfig.Q40_TRAIN_END))
            )
            ref_switch = pd.to_numeric(ref.loc[train_mask, 'mkt_ma5_followthrough_v3'], errors='coerce').dropna()
            if len(ref_switch) < 20:
                return default

            signal_value = pd.to_numeric(filtered_data['mkt_ma5_followthrough_v3'], errors='coerce').dropna()
            if signal_value.empty:
                return default

            threshold = float(ref_switch.quantile(0.4))
            chosen_rank = 1 if float(signal_value.iloc[0]) >= threshold else 2
            chosen_rank = min(chosen_rank, len(selected_stocks))
            selected_code = selected_stocks[chosen_rank - 1]['code']

            for stock in selected_stocks:
                stock['q40_selected'] = stock['rank'] == chosen_rank

            return {
                'enabled': True,
                'chosen_rank': chosen_rank,
                'threshold': threshold,
                'signal_value': float(signal_value.iloc[0]),
                'selected_code': selected_code,
            }
        except Exception as e:
            self.logger.warning(f"q40 标注失败，回退 Top1: {e}")
            return default

    def _create_result(self, date: str, stocks: List[Dict], num_stocks: int, q40_info: Dict[str, Any]) -> Dict:
        """创建选股结果"""
        return {
            'stocks': stocks,
            'num_stocks': num_stocks,
            'q40': q40_info,
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
