# 2026-07 市场门控 / merge 漏斗 / 模式 A 穿透核查

## 0. 本地文件与口径

- `market_gates_search.parquet`: **36** 个市场门控；`gate_search_market.parquet`: **40** 个；本地重建 48 规格 `gate_search_market.parquet`（qwin=252）。
- 重建 48 规格 sha12 = `e9322fc3e169`，与 server bank-v2 `market_sha12` **完全一致**。
- 2026-07 月均值 p 为日级滚动分位 p 的月均值；`g=2p-1`。
- 48 个门控中 2026-07 均值极端（p<.15 或 >.85）的共 13 个。

## 1. 2026-07 极端读数

### 低位（退潮方向）

| gate | July p | July g | 低分位天数 | 23日中 |
|---|---:|---:|---:|---:|
| GM_newhigh_ratio_level | 0.0806 | -0.839 | 17 | 23 |
| GM_nh60_prem_chg | 0.1291 | -0.742 | 19 | 23 |

### 高位（恐慌/扩散方向）

| gate | July p | July g | 高分位天数 | 23日中 |
|---|---:|---:|---:|---:|
| GM_sec_disp_ratio | 0.8777 | +0.755 | 17 | 23 |
| GM_vol_med_ratio | 0.9132 | +0.826 | 23 | 23 |
| GM_amp_med_ratio | 0.9420 | +0.884 | 22 | 23 |
| GM_xs_disp_ratio | 0.9683 | +0.937 | 22 | 23 |
| GM_xs_iqr_ratio | 0.9757 | +0.951 | 23 | 23 |
| GM_sec_disp_level | 0.9758 | +0.952 | 23 | 23 |
| GM_xs_disp_level | 0.9810 | +0.962 | 23 | 23 |
| GM_amp_med_level | 0.9907 | +0.981 | 23 | 23 |
| GM_xs_iqr_level | 0.9912 | +0.982 | 23 | 23 |
| GM_amt_hhi_level | 0.9960 | +0.992 | 23 | 23 |
| GM_vol_med_level | 0.9984 | +0.997 | 23 | 23 |

### 重点门控

| gate | July p | July g | July低/高天数 | Q3 p | Q3 g |
|---|---:|---:|---:|---:|---:|
| GM_newhigh_ratio_accel | 0.4493 | -0.101 | 1/0 | 0.4547 | -0.091 |
| GM_newhigh_ratio_chg | 0.3435 | -0.313 | 2/0 | 0.5765 | +0.153 |
| GM_newhigh_ratio_level | 0.0806 | -0.839 | 17/0 | 0.1653 | -0.669 |
| GM_newhigh_ratio_ratio | 0.1648 | -0.670 | 12/0 | 0.3419 | -0.316 |
| GM_nh60_prem_accel | 0.4933 | -0.013 | 10/11 | 0.4488 | -0.102 |
| GM_nh60_prem_chg | 0.1291 | -0.742 | 19/1 | 0.4521 | -0.096 |
| GM_nh60_prem_level | 0.4386 | -0.123 | 11/5 | 0.2223 | -0.555 |
| GM_nh60_prem_ratio | 0.6213 | +0.243 | 6/13 | 0.7813 | +0.563 |
| GM_sec_top5_persist_accel | 0.4739 | -0.052 | 8/7 | 0.5303 | +0.061 |
| GM_sec_top5_persist_chg | 0.2786 | -0.443 | 9/0 | 0.4448 | -0.110 |
| GM_sec_top5_persist_level | 0.3301 | -0.340 | 5/0 | 0.3520 | -0.296 |
| GM_sec_top5_persist_ratio | 0.4957 | -0.009 | 3/0 | 0.5427 | +0.085 |
| GM_turn_med_accel | 0.4455 | -0.109 | 4/2 | 0.4672 | -0.066 |
| GM_turn_med_chg | 0.4400 | -0.120 | 2/0 | 0.4180 | -0.164 |
| GM_turn_med_level | 0.5526 | +0.105 | 0/0 | 0.2552 | -0.490 |
| GM_turn_med_ratio | 0.4997 | -0.001 | 0/0 | 0.2759 | -0.448 |

- **明确退潮**：`GM_newhigh_ratio_level`（p=.0806，17/23 天低分位）、`GM_nh60_prem_chg`（p=.1291，19/23 天低分位）。
- **边缘**：`GM_newhigh_ratio_ratio` p=.1648（12/23 天低分位）。
- **没有强退潮**：`GM_sec_top5_persist_*` p=.279~.496；`GM_turn_med_*` p=.440~.553；`GM_newhigh_ratio_chg` p=.3435。
- 高位集中在波动/振幅/离散度/成交集中度：`vol_med_level` .998、`amp_med_level` .991、`xs_iqr_level` .991、`xs_disp_level` .981、`amt_hhi_level` .996；这是恐慌/扩散，不是温和退潮。
- Q3 月均值仅 `GM_amt_hhi_level` 仍 >.85（.866）；7 月退潮门控在 Q3 已回落。

## 2. Merge 漏斗：谁掉了、掉在哪一步

- 本地主 bank-v1：market 3 个；server bank-v2：market 10 个。**2026-10-18 的 specialist 产物用的是 bank-v2，不是本地主 bank-v1。**
- bank-v2 包含 `GM_nh60_prem_chg`、`GM_sec_top5_persist_level/ratio`、`GM_newhigh_ratio_chg/ratio`、`GM_turn_med_accel` 等；不包含 `GM_newhigh_ratio_level`。
- specialist pool：400 个，selected 211 个；suggested_form={'gain': 400}。
- 最终 market cands：gain 20 / flip 0 / switch 4；gain 中 newhigh_ratio_chg 13、limit_ratio_ratio 6、turn_med_accel 1。

| gate | bank-v2 | pool best | selected | gain cand | flip | switch cand | F1 gain pass | 84 gate-pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GM_newhigh_ratio_chg | 1 | 77 | 53 | 13 | 0 | 2 | 11 | 13 |
| GS_sec_xs_disp_level | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| GM_limit_ratio_ratio | 1 | 30 | 27 | 6 | 0 | 2 | 0 | 2 |
| GM_turn_med_accel | 1 | 3 | 2 | 1 | 0 | 0 | 1 | 1 |
| GM_sec_top5_persist_level | 1 | 150 | 79 | 0 | 0 | 0 | 0 | 0 |
| GM_sec_top5_persist_ratio | 1 | 106 | 34 | 0 | 0 | 0 | 0 | 0 |
| GM_nh60_prem_chg | 1 | 25 | 15 | 0 | 0 | 0 | 0 | 0 |
| GM_amt_hhi_chg | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| GM_hl_spread_level | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_limit_ratio_level | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_newhigh_ratio_accel | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_newhigh_ratio_level | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_newhigh_ratio_ratio | 1 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_nh60_prem_accel | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_nh60_prem_level | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_nh60_prem_ratio | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_sec_top5_persist_accel | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_sec_top5_persist_chg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_turn_med_chg | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_turn_med_level | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GM_turn_med_ratio | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| GS_sec_amp_med_level | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- **`GM_newhigh_ratio_chg` 不是 raw pool 垄断**：pool selected 53/211；但 20 个 gain 名额里占 13 个（65%），24 个 market cands 里占 15 个（62.5%）。
- **`GM_sec_top5_persist_level/ratio` 才是 pool 最大簇**（79+34=113/211 selected），但 `select_specialists --max-market-cands 20` 按 `|t|*|hi-lo|` 截断，**两者均 0 个 gain cands**。
- **`GM_nh60_prem_chg`**：bank-v2 有、selected 15 个，但最高分 0.0768 低于 Top20 门槛，**0 个 cand**。
- **`GM_newhigh_ratio_level`**：7 月最强退潮信号（p=.0806），但**没进 bank-v2**，在 48→bank 构建阶段就掉队。
- **`GM_newhigh_ratio_chg`**：7 月 p=.3435，并非强退潮信号，却拿到最多名额并最终进入 84。

### F1 淘汰原因

- market gain：20 个，h3_pass 12，FDR 15；switch 4 个全部通过；flip 0 个（根本没有生成候选）。
- market gain 失败 8 个：`ok_ci` 失败 6、`ok_blocks` 失败 6、`ok_placebo` 失败 5。
- 失败不是 `delta_mean` 不够：8 个 delta_mean 均为正；是 **CI 下界>0 / nblocks_pos>=4 / placebo_p<.05**。
- 2 个 newhigh gain 仅 placebo 失败（p=.0896/.0995）；6 个 limit gain 因 CI/block/placebo 失败。
- 84 名单：20/84 个门控配对特征；按门控：{'GM_newhigh_ratio_chg': 13, 'GM_turn_med_accel': 1, 'GM_limit_ratio_ratio': 2, 'GS_sec_xs_disp_level': 4}。
- 其中 market gain 12（newhigh 11 + turn 1）、market switch 4、sector switch 4；limit gain 0 个进入 84。

## 3. 模式 A（GP 内嵌 MKT_GATE1/2, SEC_GATE1/2）

- 84 列名单中 `MKT_GATE1/2/SEC_GATE1/2` 字面字段：**0**；门控配对特征 **20/84**（这些是 F1 specialist，不是模式 A GP）。
- ledger：15,551 条事件，含四个 GATE 字段的公式 **0** 条；最后事件 `2026-10-15T17:17:55`。
- V4.3d `GATE_FIELDS` 提交时间：**2026-10-16 15:08**；ledger 最后事件 **2026-10-15 17:17**。
- 因此不是“跑过但轮数不够”，而是 **V4.3d 之后没有跑过 run_search.py**；模式 A 使用率为 0 是时间顺序问题，不是随机命中率问题。
- 另外，84 的 20 个门控配对来自 F1 specialist（GM/GS 名称），与 GP 内嵌 `MKT_GATE1/2` 是两条不同路径。

## 4. 结论

- 7 月真实退潮信号在 **newhigh_ratio_level / nh60_prem_chg**；sec_top5_persist 只是偏弱，turn_med 没有退潮信号。
- 能识别退潮的 gate：newhigh_level 死在 bank 构建；nh60_prem_chg 死在 select_specialists Top20 截断；sec_top5 也死在 Top20 截断。
- 最终 84 的 20 个门控配对由 newhigh_chg/limit/turn 主导，其中 newhigh_chg 13/20；这不是因为它是 7 月最强退潮门控，而是评分/配额机制。
- F1 不是全灭：market gain 12/20、switch 4/4、sector switch 4/4 通过；flip 0 是因为没有生成，不是被 judge 淘汰。
- 模式 A 目前 0 使用；V4.3d 后尚未跑 run_search，需至少一轮新搜索才能检验。







本地有这个文件，但不是完整 48 门控版本，而且版本链路有明显错位：
- results/merge_v2/gates_search/market_gates_search.parquet：只有 36 个市场门控。
- results/merge_v2/gates_search/gate_search_market.parquet：只有 40 个。
- 我按 gate_search_space.py 本地重建了完整 48 规格，market_sha12=e9322fc3e169，与 server bank-v2 的 hash 完全一致。
- 当前本地主 certified_gate_bank.json 是旧的 bank-v1（3 个市场门控）；2026-10-18 的 specialist 产物用的是 bank-v2（10 个市场门控）。
核查 1：2026-07 真实读数
48 个门控中，7 月月均 p 达极端（p<.15 或 >.85）共 13 个。
明确低位/退潮方向：
门控	July p	July g	低分位天数
GM_newhigh_ratio_level	0.0806	-0.839	17/23
GM_nh60_prem_chg	0.1291	-0.742	19/23
边缘低位：
- GM_newhigh_ratio_ratio p=.1648，12/23 天低分位。
重点门控结论：
- GM_nh60_prem_*：只有 chg 强退潮；level .439、ratio .621、accel .493，均不强。
- GM_sec_top5_persist_*：level .330、ratio .496、chg .279，没有强退潮信号。
- GM_turn_med_*：July p=.440~.553，没有退潮信号；Q3 level/ratio 才降到 .255/.276。
- GM_newhigh_ratio_*：level/ratio 强退潮，chg/accel 不強；chg p=.3435。
高位不是温和退潮，而是恐慌/扩散：
- vol_med_level .998、amt_hhi_level .996、xs_iqr_level .991、amp_med_level .991、xs_disp_level .981、sec_disp_level .976 等。
- 2026 Q3 月均值仅 GM_amt_hhi_level 仍 >.85（.866）；7 月退潮信号到 Q3 已明显回落。
核查 2：退潮门控在 merge 漏斗哪里掉队
关键版本事实：bank-v2 有 10 个市场门控；本地主 bank-v1 只有 3 个。
门控	bank-v2	pool selected	gain 候选	switch 候选	进入 84
GM_newhigh_ratio_chg	是	53	13	2	13
GM_limit_ratio_ratio	是	27	6	2	2
GM_turn_med_accel	是	2	1	0	1
GM_sec_top5_persist_level	是	79	0	0	0
GM_sec_top5_persist_ratio	是	34	0	0	0
GM_nh60_prem_chg	是	15	0	0	0
GM_newhigh_ratio_level	否	0	0	0	0
- GM_newhigh_ratio_level：7 月最强退潮信号，但 没进 bank-v2，在 48 → bank 构建阶段就掉了。
- GM_nh60_prem_chg：bank-v2 里有，selected 15 个；但 select_specialists --max-market-cands 20 按 |t|*|hi-lo| 截断，最高分仅 0.0768，低于 Top20 门槛 0.1680，所以 0 个候选。
- GM_sec_top5_persist_level/ratio：pool 里合计 113/211 selected，是最大簇；但最高分 .1297/.153，仍低于 .168，0 个候选。
- GM_newhigh_ratio_chg：并不是 raw pool 垄断（53/211 selected），但在 20 个 gain 名额里占 13 个（65%）；24 个 market 候选里占 15 个（62.5%），最终 84 的 20 个门控配对里占 13 个。它 7 月 p=.3435，并不是最强退潮门控。
- Flip 0：specialist_pool.json 400 个因子全部 suggested_form='gain'，specialist_cands_flip.json 为空；flip 根本没生成候选，不是被 F1 淘汰。
- Gain 淘汰原因：20 个 gain 中 12 个 h3 通过、15 个 FDR 通过；8 个失败：
- 6 个 limit_ratio_ratio gain 全部失败：CI 下界 ≤0、nblocks_pos=3，其中 3 个 placebo 也失败。
- 2 个 newhigh_ratio_chg gain 仅 placebo 失败（p=.0896/.0995）。
- 不是 delta_mean 不够；所有 delta_mean 都为正。失败集中在 delta_ci_lo>0、nblocks_pos>=4、placebo_p<.05。
- Switch：4 个 market switch 全部通过；4 个 sector switch 也全部通过。
- 84 名单实际包含 20/84 个门控配对特征：market gain 12（newhigh 11 + turn 1）、market switch 4、sector switch 4；limit gain 0 个进入 84。
核查 3：模式 A（MKT_GATE1/2, SEC_GATE1/2）使用率
- 84 列名单中字面 MKT_GATE1/2 或 SEC_GATE1/2：0 个。
- 84 里的 20 个 __GM_/__GS_ 特征是 F1 specialist 门控配对，不是 GP 内嵌的 MKT_GATE1/2 模式 A。
- ledger.jsonl：15,551 条事件，含四个 GATE 字段的公式 0 条，最后事件 2026-10-15T17:17:55。
- V4.3d GATE_FIELDS 提交时间：2026-10-16 15:08。
- 所以不是“跑过但轮数不够”，而是 V4.3d 之后还没跑过 run_search.py，实际是 0 轮。模式 A 使用率 0 是时间顺序问题，不是随机命中问题。
结论
7 月真实退潮信号是 GM_newhigh_ratio_level + GM_nh60_prem_chg；sec_top5_persist 只是偏弱，turn_med 没有退潮信号。
但前者没进 bank-v2，后者和 sec_top5 都在 select_specialists 的 Top20 候选截断处被挤掉；最终 84 的门控配对由 GM_newhigh_ratio_chg 主导，而它并不是 7 月最强退潮门控。
文件：
- results/gate_audit_20261023/summary_gate.md
- gate48_monthly_2026.csv、gate48_july_extremes.csv、gate48_focus_july.csv
- funnel_by_gate.csv、f1_gain_failures.csv
- 48 门控重建：results/merge_v2/gates_search_audit_20261023/gate_search_market.parquet
- 脚本：scripts/gate_audit_20261023.py
