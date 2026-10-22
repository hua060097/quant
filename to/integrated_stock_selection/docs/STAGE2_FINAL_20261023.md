# Stage2 收敛报告（2026-10-23）

## 结论

1. **v2-inc 不采纳，A2 84 列冻结**
   - `v2_inc_85`：84 + `ts_amt_exp_5_60_raw`（85 列）。
   - cp170：A2 `1.3069%` vs v2 `1.0299%`，Δ `-0.2770%`，95% CI `[-0.6561%, +0.0562%]`；4/11 折胜，plateau 不达标。
   - cp200：A2 `1.1595%` vs v2 `0.9102%`；cp260：`1.2639%` vs `0.8292%`。
   - `v2_inc_all` 实际 86 列：`ts_amt`、`wq012` 通过；`cyc_profit_ratio_20` max|ρ|=0.7559（vs `price_ma60_ratio`）被拒。

2. **Q5 SHAP：不是 11 个 GM gain 门控列**
   - Q5 135 天，A2 84 列、H23@170。
   - 主要正贡献差：
     - `log_price_slope_20d`：Q5 `0.2576` vs Q1-Q4 `0.0630`（+0.1947）
     - `ma_convergence_score`：`0.5095` vs `0.3934`（+0.1161）
     - `ma5_pullback_depth_10d`：`0.2020` vs `0.1518`（+0.0502）
     - `IVOL20`、`TURNOVER`：各约 +0.023
   - 11 个 `*__GM_newhigh_ratio_chg__gain` 合计：Q5 `0.3669` vs Q1-Q4 `0.3734`，**Q5 反而略低**。

3. **真实 5 分钟引擎：A_skip 优于 A_half**
   - CV 2023-03-31~2025-12-31：
     - Exec_Confirm_50：`1250.50% / 157.31% / 29.65% / 2.12`
     - A_skip：`1363.89% / 164.95% / 30.05% / 2.24`
     - A_half：`1224.58% / 155.50% / 29.64% / 2.16`
   - 2026 OOS：
     - Exec_Confirm_50：`69.43% / 112.81% / 40.51% / 1.69`
     - A_skip：`76.45% / 125.55% / 30.79% / 1.90`
     - A_half：`75.28% / 123.41% / 35.09% / 1.85`
   - OOS 19 个 A_skip 触发日：5 月 2 天、7 月 5 天、8 月 7 天、9 月 5 天；17/19 在 Q3，H1 与 baseline 相同。

4. **2026 Q3 是全市场正样本正常、模型 Rank1 精度衰减**
   - 全市场 y=1：Q1 `2.99%`、Q2 `4.27%`、Q3 `3.13%`（全样本核心正类率约 `3.07%`）。
   - Rank1 Precision@1：Q1 `10.71%`、Q2 `23.33%`、Q3 `5.45%`；CV 最差 Fold05 为 `9.52%`，Q3 更低。
   - Q3 Rank1 日均 `-4.23%`，分钟 MDD `-38.67%`；CV 最差分钟 MDD 约 `-25.45%`。
   - **结论：Q3 不是正样本率枯竭，而是模型在 Q3 的 Rank1 特征/排序失效。**

5. **Q3 排名结构：真牛股在 Rank3~12，不在 Rank1**
   - Q3 55 个交易日，rank1-15 Precision@7%/平均 r3：
     - Rank1 `9.09% / -4.23%`
     - Rank2 `9.09% / -2.72%`
     - Rank3 `18.18% / -0.96%`
     - Rank4 `18.18% / -1.70%`
     - Rank5 `5.45% / -2.52%`
     - Rank6 `12.73% / -1.58%`
     - Rank7 `10.91% / -2.17%`
     - Rank8 `18.18% / +0.28%`
     - Rank9 `14.55% / -1.43%`
     - Rank10 `18.18% / +0.46%`
     - Rank11 `16.36% / -0.91%`
     - Rank12 `20.00% / +0.37%`
     - Rank13 `12.73% / -1.56%`
     - Rank14 `16.36% / -0.51%`
     - Rank15 `14.55% / -0.78%`
   - Top1-30 整体 Precision@7% = `13.21%`。Q2 Rank1 为 `23.33%`，Q3 `5.45%`。
   - **典型“高分拥挤陷阱”：Rank1/Rank2 被过度透支票占据，真正突破龙头集中在 Rank3~12。**

6. **wq_gtja_elite10 缓存风险已确认并修复**
   - `CustomFactorSpec.spec_hash` 原只哈希 Python wrapper 源码；wq wrapper 只是 `return _ensure_computed(...)`，helper 修改不会改变 hash。
   - 已改为纳入所在 `.py` 文件全文哈希，避免 helper 改动后命中旧 `value_cache`。
   - A2 84 不含 wq，本次 A2 结论不受影响；后续 build_top 评估 wq 需使用新 hash/清旧缓存。

## 未完成

- **2026 Q2 vs Q3 Rank1 SHAP 对比**：本地有 A2 booster（`a2_models`），但本地没有 84 列特征矩阵 `lgb_features.parquet`；无法在本地对 Q2/Q3 Rank1 样本调用 `pred_contrib`。需要该矩阵（服务器数据盘 2.8G）或由服务器导出 Q2/Q3 Rank1 特征切片。该诊断尚未闭环。
- 全市场标签审计已完成：`results/server_save_20261022/full_market_label_audit_20261023.csv`。
- Q3 全市场 MDD/Precision 已与分钟 pkl 对齐；分钟 pkl 本地已保存。

## 关键文件

- 代码：`scripts/build_v2_inc.py`、`scripts/compare_v2inc_cv.py`、`scripts/shap_q5_rank1.py`、`scripts/run_cv_exec_overlay.py`、`scripts/full_market_label_audit.py`、`scripts/step4_quarter_attribution.py`、`stage15_v5_pipeline.py`、`pandaai_online/merge/custom_registry.py`。
- 报告：`results/server_save_20261022/`（v2 对拍、SHAP、全市场标签）、`results/exec_overlay_cv_fixed3_askip.pkl`、`results/exec_overlay_cv_fixed3_ahalf.pkl`、`results/exec_overlay_oos_20261022/*.pkl`。
