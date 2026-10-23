# Check 3: CV-only gate conditioning / truncated specialists

## 1. T 日门控 vs future 3D TURNOVER RankIC / A2 Rank1

- 目标：未来 T+1..T+3 日 turnover 截面 RankIC 均值；A2 H23@170 Rank1 的 r3 收益。
- 信号：48 个市场门控 g=2p-1，加 past20 日 turnover RankIC；block bootstrap 1000 次（20 日块），BH-FDR q<=0.10。
- CV 仅 <=2025-12-31；A2 Rank1 数据从 2023-03-31 起。

### 与 future3_turnover_ic 显著（q<=0.10）

| signal | rho | p_boot | q_bh | n |
|---|---:|---:|---:|---:|

### 与 A2 Rank1 显著（q<=0.10）

| signal | rho | p_boot | q_bh | n |
|---|---:|---:|---:|---:|

### future3 IC 最负 5 个

| signal | rho | q_bh |
|---|---:|---:|
| GM_limit_ratio_level | -0.137 | 0.8547 |
| GM_limit_ratio_ratio | -0.114 | 0.8547 |
| GM_sec_top5_persist_level | -0.106 | 0.8547 |
| GM_newhigh_ratio_level | -0.105 | 0.8547 |
| GM_newhigh_ratio_ratio | -0.094 | 0.8547 |

## 2. 被 Top20 截断的 nh60/sec 候选

- `specialist_pool` 中有 15 个 `GM_nh60_prem_chg`、79 个 `GM_sec_top5_persist_level`、34 个 ratio 的 selected 因子；
  但 `select_specialists --max-market-cands 20` 只保留全局 score Top20，因此它们 0 个进入 gain cands。
- 本地没有 `specialist_factors.parquet`，也没有这些门控的 F1 `delta_mean/CI/nblocks/placebo` 记录，
  无法在不重算因子值的情况下诚实地判定 H3；**结论是未验证，而不是通过/不通过**。

| gate | selected | max score | 最高候选是否在全局 Top20 |
|---|---:|---:|---:|
| GM_nh60_prem_chg | 15 | 0.0768 | 否 |
| GM_sec_top5_persist_level | 79 | 0.1297 | 否 |
| GM_sec_top5_persist_ratio | 34 | 0.1530 | 否 |

- Top20 全局截断分位：约 0.1680；nh60 最高 0.0768，sec_top5 level/ratio 最高 .1297/.1530。
- 由于这些候选没有 F1 记录，不能把“未进 cands”误写成“F1 不通过”。
