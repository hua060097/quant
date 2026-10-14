# Tier2 批次实现方案（2026-09-30 夜，待执行）

## 现状
- tier2 = 73 个 merge 候选（通道A/B 20+20 正交保留、通道C 12 refs+10 mined、通道D 8 refs+10 mined，去重后），
  证据已在注册表（JMI 短名单 / full-KEEP JMI / cond diff_t / c2c），但**值未进 factors.parquet**。
- 2026-09-30 夜的 one-shot 裁决（pool_select.py）已覆盖 tier1 全量 228。tier2 是**扩展问题**（增测），不是成员资格问题。

## 为什么不能直接搬值
merge 引擎的求值面板（evaluator.load_panel）与 factor_lab 的训练面板在行集/日期范围/ST 处理上不一致。
`--features_file` 只认 factors.parquet 里的列 → 每个候选必须**重新实现为 factor_lab 定义**并在
nofrozen 管线下生成，数值对拍通过后才能进池测试。无捷径。

## 两类来源的实现路径
1. **refs**（R###、efficiency_*_confirm_volume 等）：merge/refs.py 的 FactorBuilder spec 已有本地逻辑，
   翻译成 factor_lab definitions（读 base 列计算）。
2. **mined**（hex 哈希）：平台 DSL 公式在 flat.csv 的 formula_cn 列，evaluator.parse_formula 的 AST
   逐算子映射到 pandas/numpy（TS_MEAN/IF/RANK 等语义对照 evaluator.py）。

## Batch 1 建议（8-12 个，证据最强 + 机制互补 + 公式最简 + 正负孪生只取一）
| 来源 | 候选 | 理由 |
|---|---|---|
| JMI 短名单 top5 | 1f21bad9efa5、R346、reversal_120_smooth、f82fa54b7c3d、9c3b053a3bae | twin+CMIM 双否决幸存且 JMI 最高 |
| 通道C mined diff_t top3 | 3eab67e9daf8、09c016d6154c、2d1c55bf42ee | diff_t +5.7/+5.7/+5.3（M8_gate，彼此高相关→取1-2个即可） |
| 通道D mined top2 | 09daa73656ba、92ec5ef6d160 | nmi 最高（M9_composite） |
| 通道C refs 无孪生 | R234、R576、R616 | cond t 19.6/16.3/13.9，剔除 sign-twin 后独立 |

## 单个因子的标准流程（用户指定路径）
1. 在 `factor_system/factor_lab/definitions/<合适模块>.py` 实现 + `register_factor`
2. `definitions/__init__.py` 加 import
3. nofrozen 主脚本路由：原始值因子进 `_get_original_factors`（用户指定），z-score 类进 `_get_zscore_factors`
4. 数值对拍：与 merge evaluator 在重叠日期截面 Spearman≈1
5. `python main_alpha原始_overlay_origin_nofrozen.py --mode generate_factors --add_factor --factor_names <名> --start_date 20210601 --end_date 20260918 --handle_st --skip_minute_agg`
6. 卫生抽查（覆盖率/常数）→ `frozen_runner.py update`

## Batch 1 增测臂
`胜者池(来自 POOL_SELECTED) + batch1` vs `胜者池`，一轮 quick（H15，或 H14+H15），
判据沿用 pool_select 的预注册容差（champ/min_fold/q10/mdd_min 四指标，去增量不亏才留）。
工时估计：8-12 个 × 30-60min ≈ 1-1.5 个工作日（不含生成与 quick）。

## 批次推进规则
- Batch 1 通过 → Batch 2 按证据降序补位，族配额 ≤3，与已在池因子做月末相关 |ρ|>0.97 剔除
- Batch 1 失败 → 停止扩展，定位原因（证据泄漏 or 实现语义偏差）再议
