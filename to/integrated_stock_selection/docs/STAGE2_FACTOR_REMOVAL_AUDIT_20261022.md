# Stage2 因子被挤掉审计（84 → 67，27 个移除 / 10 个新增）

- 生成时间：2026-10-22
- 旧：`ALIGNED_TOP2_20261019_230912`（2026-10-19 22:20，`vwap_orth_scan=250`，class1=46，pool=10，seats=10）
- 新：`ALIGNED_TOP2_20261022_015602`（2026-10-22 01:34:53，`vwap_top20_new.json n=29`，pool=10，seats=10）
- 新矩阵（服务器数据盘，本地无该 parquet）：`/mnt/data/integrated_stock_selection/results/merge_v2/lgb_ready_v2/lgb_features.parquet`，67 列（旧 84 − 27 + 10）
- 旧数据：`results/from_server_20261020/ALIGNED_TOP2_20261019_230912_flat.csv`、`vwap_top20.json`、`vwap_habitat_pool.json`、`habitat_seats.json`、`features_manifest.json`
- 新数据：`results/from_server_20261022/vwap_top20_new.json`、`vwap_habitat_pool_new.json`、`habitat_seats_new.json`、`features_manifest_v2.json`

> **口径校正（重要）**：本文旧 `vwap_rank` 一律用旧 flat 的 `label='vwap'` 行按 `LGB` 降序、1 起计算（共 1249 行），与 `build_top` 的 `vwap_df.sort_values('LGB')` 扫描顺序一致。此前口头汇总里的 `22f1=107 / 17d=132 / 28ddd=94` 等名次有误；以本文表格为准。由此，27 个移除的归因从“15 scan + 3 dedup”校正为 **“18 scan + 4 seat + 5 pool”**，`22f1`、`17d`、`28ddd` 均是 scan 截断，不是 dedup。

## 0. 结论（先看这段）

1. **27 个移除不是一个原因，也不是“新因子都是坏因子”**。按可核查证据只有 3 类：

   | 类别 | 数量 | 说明 |
   |---|---:|---|
   | 扫描深度截断（250 → 150） | 18 | 18 个旧 top20 非席位因子，旧 vwap LGB rank 150~243；旧 run `vwap_orth_scan=250` 纳入，新 run 未传 `--scan`、走默认 150 |
   | 旧栖息地席位被替换 | 4 | 旧 seats 4 个 → 新 seats 4 个；通道本身不按 150 扫描选，但入口受 `_hab_vals` 扩面预算影响 |
   | 旧栖息地池成员被替换 | 5 | 旧 pool 中 5 个被替换，另 5 个保留；池按 habitat_score 重排，入口同受扩面预算影响 |
   | **合计** | **27** | **18 + 4 + 5** |

2. **18 个 scan 中，17 个是硬截断，1 个是边界**：
   - 17 个旧 vwap rank = 158~243，**全部 >150**；即使没有新候选也进不了新 scan 150。
   - `8a5d629f29a7` 旧 vwap rank = **150**；新 150 窗口本应包含它，但新 run 又加入了新候选/新相关性，它是否仍在新 150 内需新 flat 复核。保守归类为“scan/候选重排边界”。

3. **新因子真正入选的只有 1 个**：`ts_amt_exp_5_60_raw`（Top20/新矩阵）。其余新自定义因子（`ts_mkt_stock_amt_z60_raw`、`id_morning_keep`、`id_close_vwap_prem`、3 个 alpha_supplement、10 个 wq_gtja_elite10）全部未进 Top20/池/席位；`paper_idio_mom_20d` PIT 审计未通过。

4. **本轮不是增量制**：`build_top` 是全量重选，在新候选、新相关性矩阵、席位/池规则下重排，必然有人进有人出。**旧 84 vs 新 67 不是干净的“加法 A/B”**，至少混杂：
   - `--scan` 250 → 150（**命令遗漏，18/27 的主因**）
   - 新自定义候选进入贪婪正交扫描（旧 meta `n_custom=17`；新 run 输出 custom specs=34，server 日志；本地新 JSON 不含该字段）
   - `decorrelate_top.month_end_spearman_matrix` 的 Stage1 修复（旧 `.all(axis=1)` → 新逐列中位数填充）改变了 rho/席位/池分数
   - 席位/池通道按新候选与新 rho 重排

5. **2026 没有加入评估窗口**。报告里的“2021-09-01~2026-08-21”是 `build_md` 写死的展示字符串（新旧报告逐字相同，`build_top.py` 第 1767 行）；实际评估掩码是 `min(FIT_END='20260821', data_cut='2025-12-31') = 2025-12-31`。2026 一天都没进 build_top 的筛选；旧 84 也是同一窗口，窗口未变。`features_manifest_v2.json` 的 `data_cut=2025-12-31` 可佐证。

## 0.1 对 Gemini "全部 27 都是 scan 150 + [:150]" 的复核

**已验证成立的部分：**

- 新 meta 明确 `vwap_orth_scan=150`、`c2c_orth_scan=150`、`n_custom=34`、`n_total=2532`；旧 meta 为 250 / 17 / 2498。
- `_habitat_expand` 确有硬截断：旧代码 `42afdbc` 与当前代码均为 `_extra_all[:150]`；新 run 日志为 `habitat 扩面候选：pick=33 fam=340 other=706 -> 算150 个`，随后 `habitat 估值扩面：+150 个`。
- 新 run 只对 150 个 habitat 额外候选求值；18 个 class1 非席位因子的旧 vwap rank 为 150~243，其中 17 个 >150，确实被 scan 150 直接截断。
- 新 run 的 `custom specs=34`，旧 run 为 17；Gemini 说“本轮 15/16 个新自定义”不准确，实际是 **17 个新自定义**（34−17）。新 17 个的 vwap rank：`ts_amt_exp_5_60_raw` 144、`cyc_profit_ratio_20` 205、`wq012_vol_ret_asym5` 284、`res_sec_upvol_ratio10` 386、`ts_mkt_stock_amt_z60_raw` 585、`id_close_vwap_prem` 698、`id_morning_keep` 711、`gtja019_trend_eff5` 608、`gtja047_high_clv_anchor` 860、`wq083_coil_compress` 636、`wq055_stoch_turn_sync10` 470、`sec_down_resilience_20` 499、`id_intraday_over_overnight_5d` 1051、`res_sec_alpha5_ir` 551、`res_sec_up_down_cap10` 474、`res_sec_clv_lead5` 540、`wq054_body_top_decay3` 956。只有 `ts_amt` 在 150 内；`cyc`/`wq012` 会在 400 扫描内，其余多数 >400。`paper_idio_mom_20d` 是旧 17 之一，PIT 审计失败；不是本轮新因子。

**不能直接成立/被夸大的部分：**

- “旧 scan 250 时 `_habitat_expand` 能覆盖到第 251~800 名”不成立。`_extra_all` 上限是 **150 个额外候选**；即使旧 scan 250 把 1~250 放进 `val_store`，额外预算最多再覆盖约 251~400，不可能覆盖 552/678/707/806。
- 因此 `high_level_churn_risk_10d`（旧 vwap 707）、`price_impact`（678）、`5840f8c1f79f`（552）、`9118514c0a20`（806）、`5beb4eba2fb5`（462）的旧选中/新移除 **不能只用 scan 150 + `[:150]` 解释**；它们要么在旧 run 走了 `habitat_pick`/不同 `val_store`/不同代码版本，要么是席位/池分数重排。旧 build log 未保留，当前无法闭环。
- 只有 `t2_mined_a71e1027f89e_pct`（347）、`R740`（346）、`e45066282d89`（328）、`191b2f2c49b2`（368）落在旧 250 的额外预算 251~400 内，**可能**被 151~250 的 100 个因子挤掉；其余 5 个不支持该多米诺链条。
- 所以 27 的准确表述是：**18 个 direct scan（17 硬截断 + `8a5d` 边界）+ 4 个席位 + 5 个池；后 9 个与 scan/扩面预算部分耦合，但并非全部由 `[:150]` 证明。** 审计报告上一版把 9 个写成“完全独立”也过强；正确说法是“通道独立，但入口受 `_hab_vals` 扩面预算影响”。

**对 Gemini 建议的评价：**

- “去 `[:150]` 硬截断 + 过线全扫”方向正确；直接改 `--scan 400` 仍是新的名次截断，建议用 **质量门 + 可选 full scan**，而不是只把 150 改成 400。
- “`--incremental` 锁死旧 84、只追加新自定义”方向正确；但 build_top 只产出 class1/pool/seats，**class2 habitat 6 + class3 sector 2 + class4 interact 20 不在 build_top 内**，增量必须配合 `export_lgb_dataset` 的旧 manifest/矩阵，不能只靠 build_top 输出。
- “新因子被换手惩罚挡在 fast trigger 前 30”需要单独验证；本轮 fast_trigger 只选出 `id_late30_ret`，没有证据说新自定义因子都因此落选。

## 1. 27 个被移除因子

### 1.1 18 个：scan 250 → 150（全部为旧 top20 非席位）

| 因子 | 旧 c2c rank | 旧 vwap rank | 旧 A_official | 旧 LGB | 说明 |
|---|---:|---:|---:|---:|---|
| `8a5d629f29a7` | 266 | **150** | 0.009592 | 0.003712 | 边界：旧 rank=150，新 150 窗口本应包含；新候选/keep 后是否出局待新 flat |
| `17d85010d6d8` | 356 | 158 | 0.006341 | 0.003526 | 硬截断：旧 rank>150 |
| `22f1b27a3fdf` | 130 | 169 | 0.018398 | 0.003361 | 硬截断；不是被 `ts_amt` dedup 挤掉，旧 rank 本身已 >150 |
| `bdf8c3aee48d` | 25 | 170 | 0.024648 | 0.003319 | 硬截断 |
| `cdd9487db7fa` | 310 | 172 | 0.007513 | 0.003297 | 硬截断 |
| `b43f685674d0` | 197 | 181 | 0.013347 | 0.003168 | 硬截断 |
| `99c24edeab96` | 180 | 189 | 0.015025 | 0.003089 | 硬截断 |
| `28ddd4bb12b6` | 312 | 198 | 0.007328 | 0.002946 | 硬截断；不是 dedup |
| `log_price_slope_20d` | 443 | 203 | 0.004852 | 0.002864 | 硬截断 |
| `2727b0b343ed` | 301 | 206 | 0.007884 | 0.002771 | 硬截断 |
| `R345` | 361 | 207 | 0.006273 | 0.002766 | 硬截断 |
| `2e7e20eff971` | 150 | 210 | 0.017247 | 0.002708 | 硬截断（旧 c2c 150，但 vwap 210） |
| `4b1ef2152b54` | 109 | 214 | 0.019731 | 0.002665 | 硬截断 |
| `a03ac08d0b02` | 274 | 216 | 0.009002 | 0.002653 | 硬截断 |
| `alpha_001` | 270 | 221 | 0.009374 | 0.002603 | 硬截断 |
| `mass_index_reversal` | 398 | 228 | 0.005418 | 0.002560 | 硬截断 |
| `33c0e8b07c17` | 182 | 241 | 0.014838 | 0.002298 | 硬截断 |
| `breakout_250_vol_scaled` | 469 | 243 | 0.004372 | 0.002273 | 硬截断 |

> 这 18 个都在旧 top20 的 36 个非席位中；旧 top20 另有 10 个席位，见 1.2。

### 1.2 4 个：旧席位移除（通道不按 scan 选，但入口受 `hab_vals` 扩面预算影响）

| 旧席位 | 旧 c2c rank | 旧 vwap rank | 新席位 | 新席位旧 c2c rank | 新席位旧 vwap rank |
|---|---:|---:|---|---:|---:|
| `5840f8c1f79f` | 608 | 552 | `211acadc4aa8` | 173 | 401 |
| `high_level_churn_risk_10d` | 253 | 707 | `cond_cum5_ma_ret5` | 239 | 167 |
| `price_impact` | 985 | 678 | `qs_range_ratio_cv7` | 153 | 400 |
| `t2_mined_a71e1027f89e_pct` | 306 | 347 | `sector_relative_volatility` | 200 | 282 |

- 新 seats 4 个的旧 vwap rank 全部 >150（401/167/400/282），说明**席位不是按 150 扫描直接选的**；但席位评分只读取 `hab_vals`，而 `hab_vals` 的扩面入口受 `_habitat_expand` 预算影响，因此不能仅凭“通道独立”断言旧 4 席位与 scan 完全无关。
- `cond_cum5_ma_ret5` 旧榜已在 top20（旧 c2c 239 / vwap 167），本轮从普通成员升级为席位，因此**不计入 class1 新增 5 个**；它是“保留成员 + 新席位”。

### 1.3 5 个：旧 pool 移除（池通道不按 scan 选，但入口受 `hab_vals` 扩面预算影响）

| 旧 pool 成员 | 旧 c2c rank | 旧 vwap rank | 旧 habitat_score | 新 pool 成员 | 新 habitat_score |
|---|---:|---:|---:|---|---:|
| `R740` | 209 | 346 | 0.0999 | `mi_post_high_drawdown` | 0.1018 |
| `e45066282d89` | 177 | 328 | 0.0746 | `qs_atr_ratio_cv10` | 0.0985 |
| `9118514c0a20` | 151 | 806 | 0.0613 | `downside_beta` | 0.0785 |
| `191b2f2c49b2` | 140 | 368 | 0.0399 | `178ca3cde2d3` | 0.0730 |
| `5beb4eba2fb5` | 242 | 462 | 0.0378 | `93ad66d60250` | 0.0564 |

- 旧 pool 10 个中另 5 个保留：`upper_shadow_ratio_3d_mean`（0.0647）、`ma_convergence_score`（0.0597）、`ma5_pullback_depth_10d`（0.0593）、`distance_from_52w_low`（0.0589）、`lower_shadow_absorption_5d`（0.0513）。
- 5 个新 pool 成员全部是**旧候选**（不是本轮新增自定义），旧 flat c2c/vwap 排名在 111~387 / 251~377；旧池未收录，新池按新 habitat_score 前 10 入选。
- **口径注意**：表中旧 score 是旧 run 的值；旧成员在新 run 的 habitat_score 未写入新 pool JSON（只写 top10），因此不能直接用旧 score 与新 score 比“谁更高”。可以确认的是：新 run 的 top10 里这 5 个新成员分数为 0.1018/0.0985/0.0785/0.0730/0.0564；旧成员的新分数需完整 pool 中间结果，当前未闭环。
- 旧 pool 被移除的 5 个旧 vwap rank 为 346/328/806/368/462；其中 346/328/368 落在旧 250 扩面预算 251~400 内，806/462 超出。故“池通道完全独立”也不严谨：入口受 `hab_vals` 扩面预算影响，但 806/462 不能用 `[:150]` 单独解释。

## 2. 新增的 10 个

### 2.1 class1 新增 5 个（Top20/面板）

| 因子 | lib/family | 旧 c2c rank | 旧 vwap rank | 通道 | 说明 |
|---|---|---:|---:|---|---|
| `ts_amt_exp_5_60_raw` | custom / custom_liquidity | 新因子 | 新因子 | 正交扫描 | **本轮唯一入选的新自定义因子**（5日/60日成交额对数扩张比，raw） |
| `665d5fe29859` | mined / M14_context | 282 | 141 | 正交扫描 | 旧候选；vwap 141 ≤150，新扫描内入选；不是新自定义 |
| `211acadc4aa8` | mined / M16_interaction | 173 | 401 | 新席位 | 席位通道 |
| `sector_relative_volatility` | store | 200 | 282 | 新席位 | 席位通道 |
| `qs_range_ratio_cv7` | store | 153 | 400 | 新席位 | 席位通道 |

> 新 top20 29 = 旧 top20 保留 24 + 新增 5。新增 5 中 2 个是非席位（`ts_amt`、`665d`），3 个是新席位（`211`、`sector`、`qs`）；`cond_cum5` 是保留成员升级为席位，不在这 5 个里。

### 2.2 pool 新增 5 个（均为旧候选）

`mi_post_high_drawdown`、`qs_atr_ratio_cv10`、`downside_beta`、`178ca3cde2d3`、`93ad66d60250`（分数见 1.3）。

### 2.3 本轮新增自定义因子落选清单

- `ts_mkt_stock_amt_z60_raw`：未进 Top20/池/席位
- `id_morning_keep`、`id_close_vwap_prem`：未进
- `alpha_supplement` 3 个（`id_intraday_over_overnight_5d`、`cyc_profit_ratio_20`、`sec_down_resilience_20`）：未进
- `wq_gtja_elite10` 10 个（`wq054_body_top_decay3`、`wq012_vol_ret_asym5`、`wq055_stoch_turn_sync10`、`gtja019_trend_eff5`、`gtja047_high_clv_anchor`、`wq083_coil_compress`、`res_sec_alpha5_ir`、`res_sec_up_down_cap10`、`res_sec_clv_lead5`、`res_sec_upvol_ratio10`）：未进
- `paper_idio_mom_20d`：PIT 审计未通过（2023-09-27 / 2024-02-06 / 2025-08-29 三个断面 n_diff 13.9万~23.1万，仅告警不阻断，但影响可信度）

## 3. 为什么会被移除：按因果强度排序

### 3.1 最大机械原因：`--scan` 250 → 150（18/27）

- 旧 meta（`ALIGNED_TOP2_20261019_230912.json` 第 1 行）明确：
  - `vwap_orth_scan = 250`
  - `c2c_orth_scan = 250`
  - `n_custom = 17`
- 新 run 的启动命令**没有 `--scan`**；审计时 `build_top.py` 第 1852 行默认 `--scan 150`（现已改为 400 + 质量门）。
- `build_top.py` 第 1976 行：
  `vwap_df = df[df['label']=='vwap'].sort_values('LGB', ascending=False)`
  第 2014-2016 行把 `df_vwap.head(max_scan)` 交给 `vwap_orthogonal_indices`；所以旧 250 能看到 vwap rank≤250 的候选，新 150 只能看到≤150。
- 18 个被移除的非席位旧 rank 为 150~243：旧 250 全部覆盖，新 150 只有 `8a5d`（rank=150）在边界，其余 17 个直接落在窗口外。
- 但 `_habitat_expand` 另有 `_extra_all[:150]`：新 150 时 151~250 的 100 个候选涌入扩面队列，挤占 150 个额外名额；新日志为 `pick=33 fam=340 other=706 -> 算150`。这会间接影响席位/池入口，但不能解释 rank 552~806 的旧成员。
- **修复**：用 `--scan 250 --no-quality-gate` 复现旧口径，确认 17 个硬截断回归；新代码默认 `--scan 400` + 质量门 + `--habitat-max-extra 0`（不截断），`8a5d` 仍需看新名次。

### 3.2 新候选进入同一贪婪正交扫描

- 本轮新增自定义中只有 `ts_amt_exp_5_60_raw` 入选；`665d5fe29859` 是旧候选（旧 vwap 141）在新扫描内被选中。
- `22f1b27a3fdf`（旧 vwap 169）与本轮 `ts_amt` 同属 AMOUNT 家族，但 **`22f1` 在 150 窗口外，根本没有进入新 run 的 vwap 扫描**；因此没有证据说“`ts_amt` 把 `22f1` 挤掉”。只有在 `--scan 250` 重跑后，两者同时在场，才需要比较 |ρ|>0.7 的替换关系。
- 同理，`17d85010d6d8`（158）、`28ddd4bb12b6`（198）也是 scan 截断，不是 dedup。

### 3.3 席位通道重排（4/27）

- 旧 4 席位（`5840f8c1f79f`、`high_level_churn_risk_10d`、`price_impact`、`t2_mined_a71e1027f89e_pct`）全部被新 4 席替换。
- 新席位中 `cond_cum5_ma_ret5` 是保留成员升级；`211acadc4aa8`、`qs_range_ratio_cv7`、`sector_relative_volatility` 是新入选席位。
- 新席位旧 vwap rank 401/167/400/282，全部 >150，说明席位不是按 150 扫描直接选；但席位评分只读 `hab_vals`，其入口受 `_habitat_expand[:150]` 预算影响。4 个旧席位出局与新 run 的 habitat/条件 RankIC 重排有关，不能断言与 scan 无关。

### 3.4 栖息地池通道重排（5/27）

- 旧 pool 10 个中 5 个被替换、5 个保留；新 pool 5 个旧候选补位。
- 旧 pool 被移除的 5 个旧 vwap rank 346/328/806/368/462，全部 >150；新 pool 补位的 5 个旧 vwap rank 251~377，也全部 >150。池不是按 150 扫描直接选，但池成员同样只从 `hab_vals` 读取，入口受扩面预算影响。
- 其中 346/328/368 落在旧 250 扩面预算 251~400 内，可能被 151~250 的 100 个因子挤掉；806/462 超出旧额外预算，不能由 `[:150]` 单独解释。旧成员的新分数未在 JSON 中，精确“谁超过谁”待完整中间结果。

### 3.5 相关性修复 / P4 的全局混杂

- Stage1 修复了 `decorrelate_top.month_end_spearman_matrix`：旧 `ok = np.isfinite(block).all(axis=1)` 会因任一列 NaN 抽掉整行；新实现逐列有限值中位数填充（有效样本 <50 才置 bad，见 `decorrelate_top.py` 第 108-135 行）。
- `build_top` 的正交扫描、席位、池都使用这个矩阵；旧 84 用旧矩阵，新 67 用修复后矩阵，**即使候选集完全相同，rho 也会变**。
- 因此旧 84 vs 新 67 不是受控 A/B；相关性修复主要影响“谁在席位/池/去重中胜出”，不改变 18 个 scan 截断的机械事实。
- 旧 84 由 P4 前代码（`42afdbc`）生成；新 67 使用 P4 `994998f` + 本轮修复；P4 声称串并逐位一致，但旧/新仍不是同一代码版本，需在干净 B0/B1 中隔离。

### 3.6 窗口：没有变化，2026 未参与

- `fitness.py`：`START='20210901'`、`END='20260821'`、`HOLDOUT_START='20250701'`（长期硬编码）。
- `build_top.py` 实际掩码：
  `_fit_end = min(str(FIT_END), str(args.data_cut))`
  默认 `args.data_cut = W.DEFAULT_DATA_CUT = '2025-12-31'` → `min('20260821','2025-12-31') = 2025-12-31`
- 因此 build_top 的真实评估窗口是 **2021-09-01 ~ 2025-12-31**；2026 数据不在筛选里（2026 只在 `lgb_features_plus2026` / OOS 使用）。
- 报告里的“2021-09-01~2026-08-21”是 `build_md` 第 1767 行写死的展示字符串；旧报告（2026-10-19 23:09）逐字相同。它不是本轮改的，也不代表实际评估终点。
- 建议：改为按 `START ~ min(FIT_END, data_cut)` 动态生成，避免继续误导。

## 4. 67 列的账目（核对用）

| class | 旧 84 | 新 67 | 变化 |
|---|---:|---:|---|
| class1_stock_alpha | 46 | 29 | −22 +5 |
| class2_stock_habitat | 6 | 6 | 0 |
| class2_habitat_pool | 10 | 10 | −5 +5 |
| class3_sector_state | 2 | 2 | 0 |
| class4_gated_interact | 20 | 20 | 0 |
| **合计** | **84** | **67** | **−27 +10** |

- 27 移除 = **18 scan + 4 seat + 5 pool**；其中 22 个来自 class1（18 非席位 + 4 席位，席位也在 top20 内），5 个来自 habitat_pool。
- 10 新增 = **5 class1 + 5 pool**；class1 的 5 个是 `ts_amt`、`665d`、`211`、`sector`、`qs`；`cond_cum5` 是旧 class1 保留、升级席位，不计入新增。
- 旧 top20 46 = 10 个席位 + 36 个非席位；22 个移除 = 18 个非席位 + 4 个席位；旧 top20 保留 24 = 6 个旧席位 + 18 个非席位。新 top20 29 = 24 保留 + 5 新增。
- 旧 pool 10 = 5 保留 + 5 移除；新 pool 10 = 5 旧保留 + 5 旧候选补位。注意 pool 成员不在 class1 的 46 里，因此 27 是 22 class1 + 5 pool。

## 5. 尚未闭环的证据（需要新实例 / 新 flat）

当前 server 已 Stopped，新 flat 只在数据盘；以下精确归因待新实例：

1. 新 run 的 `vwap_orth_scan` 实际值（预期 150；新 meta `ALIGNED_TOP2_20261022_015602.json` 第 1 行可确认）。
2. `8a5d629f29a7` 在新 run 的 vwap rank/keep 结果：旧 rank=150 是边界，若新 150 包含它却仍未入选，则是新候选/相关性/keep 的作用，不应算硬 scan。
3. 旧 pool 5 个成员在新 run 的完整 habitat_score；旧 seats 4 个成员在新 run 的 seat 分数；新 seats 的 hot5_q80_t。新 JSON 只写最终 top10，需 `habitat_pool` / `seat_detail` 中间输出。
4. `22f1b27a3fdf` 与 `ts_amt_exp_5_60_raw` 的实测 |ρ|：**仅在 `--scan 250` 重跑、两者同时在场时才有意义**；当前 150 口径下 22f1 根本没进扫描。
5. 新 flat 的逐候选 rank/rho：`/mnt/data/integrated_stock_selection/pandaai_online/merge/out/ALIGNED_TOP2_20261022_015602.flat.csv`（及同前缀 JSON）。
6. **旧 build log 未保留**：因此 `high_level/price_impact/5840/911/5beb` 在旧 run 是走 `habitat_pick`、`val_store` 还是与当前不同的代码路径，无法从现有文件闭环；不能用 `[:150]` 直接断言这 5 个（或全部 9 个）都被挤掉。

## 6. 推荐动作

1. **先复跑旧口径**：`--scan 250 --no-quality-gate --habitat-max-extra 150`，确认 17 个硬 scan 截断回归；`8a5d` 单独看新名次。18/27 = 66.7%，是最大单一原因。
2. **再做新口径验证**：`--scan 400`（质量门默认开）或 `--full-scan`，并设 `--habitat-max-extra 0`，确认 4 席位/5 池是否因扩面入口恢复；若恢复，则 Gemini 的多米诺判断在 9 个上部分成立。
3. **做 v2-inc 加法实验**：冻结旧 84 + 仅 `ts_amt_exp_5_60_raw`，H23 A/B 对 A2（1.307%）。若担心与 `22f1` 相关，同时测：
   - `84 + ts_amt`（追加版）
   - `84 − 22f1 + ts_amt`（替换版）
   分别看 CV，不要用全量重选的 67 结果回答“新增因子是否加分”。
4. **build_top 的 `--incremental <features_manifest.json>` 已加入保护逻辑**：只保留旧 84 的 class1/pool/seats，并追加本轮选中的新 custom；class2 habitat 6 + class3 sector 2 + class4 interact 20 仍需 `export_lgb_dataset` 的旧 manifest 承接，不能只靠 build_top 输出。
5. **修正报告窗口字符串**：已改为动态 `START ~ min(FIT_END, data_cut)`。
6. **v2 全网格 CV 的解读**：它回答“全量重选 67 列能否打赢旧 84”，不是“新增因子是否加分”；当前 v2 CV 因实例停止而中断，任务文件在数据盘，可断点续跑。

## 7. 附录：关键文件与参数

- 旧 meta：`results/from_server_20261020/ALIGNED_TOP2_20261019_230912.json` 第 1 行
  - `vwap_orth_scan=250`、`c2c_orth_scan=250`、`n_total=2498`、`n_custom=17`、`n_store=136`、`keep_pool=80`、`keep_base=20`
- 新 top20：`results/from_server_20261022/vwap_top20_new.json`
  - `generated_at=2026-10-22T01:34:53`、`n=29`、`base20` 29 个
- 新 pool：`results/from_server_20261022/vwap_habitat_pool_new.json`，`members=10`，前 5 分数 0.1018/0.0985/0.0785/0.0730/0.0564
- 新 seats：`results/from_server_20261022/habitat_seats_new.json`，`seats=10`
- 新 manifest：`results/from_server_20261022/features_manifest_v2.json`
  - `n_total_features=67`、`data_cut=2025-12-31`；by_class: 29/6/10/2/20
- 旧/新榜单：
  - 旧：`results/from_server_20261020/vwap_top20.json`（46）、`vwap_habitat_pool.json`（10）、`habitat_seats.json`（10）
  - 新：`results/from_server_20261022/vwap_top20_new.json`（29）、`vwap_habitat_pool_new.json`（10）、`habitat_seats_new.json`（10）
- 旧 flat 排名口径：`label='vwap'` 按 `LGB` 降序、`label='c2c'` 按 `A_official` 降序，1 起；旧 flat 1249 个 vwap 候选 / 1249 个 c2c 候选。
- 代码常量：
  - `pandaai_online/merge/fitness.py`: `START='20210901'`, `END='20260821'`, `HOLDOUT_START='20250701'`
  - `pandaai_online/merge/windows.py`: `DEFAULT_DATA_CUT='2025-12-31'`
  - `pandaai_online/merge/build_top.py`: 审计时 `--scan` 默认 150（旧代码第 1852 行；现已改为 400 + 质量门）；`_extra_all[:150]` 硬截断（旧代码第 1059 行；现已可配置，默认不截断）；窗口字符串硬编码（旧代码第 1767 行；现已动态化）
  - `pandaai_online/merge/decorrelate_top.py`: Stage1 逐列中位数填充（第 108-135 行）
- 新视图（服务器数据盘，本地无该 parquet）：`/mnt/data/integrated_stock_selection/results/merge_v2/lgb_ready_v2/lgb_features.parquet`（67 列，6,666,938 行）；`lgb_features_plus2026.parquet`（7,555,041 行，67 列）
- v2-inc 构建器：`scripts/build_v2_inc.py`（旧 84 + 通过质量门且 max|ρ|≤0.70 的新 custom；输出矩阵 + manifest + challenge report）。
- Rank 断崖诊断：`scripts/diagnose_rank_cliff.py`；报告 `results/rank_cliff_diag_20261022/diagnostics.md` / `.json`（CV 669 日 + 2026 OOS 171 日）。
- 服务器数据盘新 flat：`/mnt/data/integrated_stock_selection/pandaai_online/merge/out/ALIGNED_TOP2_20261022_015602.*`
