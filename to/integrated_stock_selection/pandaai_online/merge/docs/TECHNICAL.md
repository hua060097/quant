# Merge 引擎技术总文档

> 本文档是 `pandaai_online/merge/` 的唯一总技术文档。
> 维护规则：**按流水线阶段 + 模块**组织，不按版本流水。每处功能点标注引入版本（如 `[v2.8]`）。
> 版本号与 `docs/MERGE_V2_PLAN.md` 的计划版本对齐。
> 更新日期：**2026-10-11（第三轮修订）**。状态：P0 完成；P1 主体完成；P2 冒烟闸门已建立（14/14）。
>
> - 第二轮按三方审查修订：**实施清单改四态矩阵**（PLAN §17.9）、**S8 拆两条子状态机**、
>   **状态生命周期**（§0.3.1）、**三层证据**（§0.3.2）、**孤儿模块核查**（§0.5.1）、
>   **三种评估级别**（§8.1）；新增 PLAN **§9.6**（FDR 推断目标与数据独立性，**实测为 Case B**）、
>   **§18**（跨模块接口契约规格）、**§19**（目标一致性诊断设计）。
> - **第四轮**为**指纹数据截止日实现**（2026-10-11 用户指令）：新增 **§8.5**；
>   新增模块 `migrate_data_cut.py`（遗留标记迁移，备份+校验+删 sidecar）与
>   `selftest_cut.py`（**10/10 PASS**）；
>   `run_search.py --data-cut`（默认 `windows.DEFAULT_DATA_CUT=2025-12-31`）
>   **同时是评估窗上界**——指纹记的截止日 = 实际用到的数据上界（否则记录说谎）；
>   `ledger.py` 摘要升级 `SUMMARY_SCHEMA=2`，按截止日分列 `cuts`/`tested_pairs`/`vfps`。
>   实测（seed 993）：快评 `n_periods=42`×3=**126 天**（2025-07-01~2025-12-31）、
>   全量 `350`×3=**1050 天**（2021-09-01~2025-12-31），24 条记录 `data_cut=2025-12-31`，
>   遗留截止日已测=0（隔离生效）。
>   **副产物**：个股线排除集/腿库/精英回流**默认清空**（此前无任何因子在新窗下评估过）
>   → 搜索冷启动；这是正确语义，不是缺陷（`--use-legacy-evidence` 可显式复用旧 winner）。
>   **（已裁决，2026-10-11，PLAN §20.10 方案 B）**：`--cut`（holdout 切点）与用户原则
>   「holdout 不得越过数据截止日（cut_date），**切点可落在 2025 年内**」——
>   此前文中曾误记为「只能在 2025 年之前」，系引用笔误；正解与代码实现
>   （`cut < data_cut` 强制检查）均以更正版为准。

> - 第三轮为**数据接触审计**（三方审查第二轮 + 用户原则）：新增 **§8.3 两套窗口两套纪律**、
>   **§8.4 结论等级边界**、**§0.5.1 生命周期核查**已含搜索线缺口；新增 PLAN **§20**
>   （审计事实 + 治理规则；**治理规则 §20.6 已于 2026-10-11 批准**，
>   20.6.4 已实施——本节为记录，不含独立代码）；
>   审计报告 `results/merge_v2/data_contact_audit_2026-10-11.md`。
>   **关键更正**：`windows.py` 的"窗口单一事实源"**只对门控线成立**——个股线硬编码于 `fitness.py`
>   （`2021-09-01 ~ 2026-08-21`，holdout `2025-07-01` 起）；其 holdout **实测评估了 2026 年数据**
>   （`monthly_excess` 覆盖至 2026-07），衰减监控段 **154/169 天（91%）已被消耗**。

---

## 0. 定位、流水线与模块清单

### 0.1 定位

本引擎是 PandaAI 因子挖掘的**本地端**：候选生成 → 本地评估 → 去重 → 入池 → 导出平台批次。

- **数据只读**：`factor_system/`、`factor_lab_data/`、`cache/` 原始数据不改（项目硬约束）。
- **平台执行外置**：只走 `run_search.py upload/export` 经原 skill `batch.py` 校验，不在本引擎内发请求。
- **三条候选线**（研究对象的划分，见 §0.2）：
  - **个股线（STOCK）**：进化搜索生成公式候选，走 H2 → POOL_S3。
  - **门控线（GATE）**：生成市场/行业状态条件，先过 G0–G4 资格，再与个股因子组成交互候选，走 F1/ΔIC。
  - **行业侧（SECTOR）**：门控线的一支，逐 (股,行业) 掩码，**专用后端**。
- **两条标签线**（口径不可混用）：
  - **平台线**：以 c2c 标签对齐平台读数，用于上传决策（`fitness.py`）。
  - **本地生产线**：以 tp 口径（`prodlabel.py`）服务自用训练与 merge 内部池决策。

### 0.2 统一流水线视图（九阶段 S0–S8）

> 设计规格见 `docs/MERGE_V2_PLAN.md` **§17 统一候选流水线架构**。
> 原则：**统一骨架，不统一科学性**——阶段划分/指纹/台账/数据版本/多重检验分母/停止规则/产物格式统一；
> 各线的科学假设与判定规则各自独立。

| 阶段 | 职责 | 承担模块 | 产物 | 门禁 |
|---|---|---|---|---|
| **S0 候选摄入** | 规格标准化 | `gate_search_space`（门控规格元组）、`grammar`（公式树）、`ix_build`（登记册）、`intake`/`llm_client`（提案） | `gate_search_*.parquet`、`hypothesis_registry.csv` | 无 |
| **S1 身份与指纹** | 精确去重 / 近克隆家族 / 值等价 | 个股：`grammar.structural_fingerprint` + `ledger`；门控：`gate_ledger`（`candidate_fp`/`family_fp`/`value_fp`） | `ledger.jsonl`、`gate_ledger.jsonl` | 无 |
| **S2 廉价预筛链** | 逐层留痕淘汰 | 个股：`lint` / `dust`（`run_search.evaluate_fast` 内 9 层）；门控：`gate_search_screen`（G0 + 方向中性度）+ 规格合法性 / 尘埃 / 稀疏（`run_gate_pipeline`） | `screen.csv` | G0、方向中性度（\|ρ\|<0.6） |
| **S3 快层评估** | 粗指标，**仅用于排序截断** | `fitness.build_fast_context`（个股，1800 股×24 月）；门控**不启用快层**（抽样实测否决，见 §3.11） | — | 无（不得作判定） |
| **S4 排序与停止** | 排序分 + 停止规则 | `fitness.rank_score`（个股）；`run_gate_pipeline.gate_score` + `gate_stop`（门控） | `gate_stop_state.json` | 硬预算 / `--stall` 早停 / 连续 2 批 keeper<10% 暂停 |
| **S5 全层评估** | **唯一判定层**（个股线**只记录**，不删） | 个股：`evaluate_full`（`fitness`）→ **仅计算并记录 H2 指标**；门控：`gate_g2g3`（市场）/ `gate_g2g3_sector`（行业）；交互：`h3_eval` + `f1_batch`/`f1_sector` | `h2_*.csv`、`f1_*.csv`、`g2g3*.csv` | 见 §0.3 |
| **S6 去冗余与校正** | 家族内比较 + 多重检验 | `dedup`、`gate_validate.g4_independence`、`eval_v2.bh_fdr`、`judge_f1`/`judge_gate_search` | `*_judged.csv` | FDR q≤0.10；**推断目标与数据独立性约束见 PLAN §9.6（本项目为 Case B）** |
| **S7 短名单与台账** | 记录与回写 | `report`、`ledger`、`gate_ledger`、`confirm_candidate` | shortlist JSON / `pipeline_shortlist.csv` | — |
| **S8a 个股池（`S8_Stock_Pool`）** | 按冻结的 POOL_S3 规则**决定**入池/删除；ΔComb；版本冻结与导出 | `pool_v2`（POOL_S3）、`pool_legacy`（ΔComb）、`executor` | `pool_state.json`、`candidates.txt` | POOL_S3（唯一删除判据）；删除数 ≤20% 熔断 |
| **S8b 门控交互训练（`S8_Gate_Interaction_Train`）** | 统计资格 → **模型级增量（外部生产引擎对照）** → 生产特征准入 → 训练池冻结 | **已裁定不在 merge 内实现**（2026-10-13：模型级对照由仓库根生产引擎承担；准入规则见 PLAN §17.9 项 13） | 训练池 manifest（待定，见 PLAN §18） | **F1 通过 ≠ 训练池准入**；须过外部生产对照 + 冻结流程 |

> **S8 必须分叉**：两条线的 S8 业务语义与数学本质不同（个股线是"按池规则删除/保留"；
> 门控线是"通过模型增量检验后成为训练特征"）。**统计候选资格 ≠ 生产特征资格**，
> 不得共用一个含义模糊的 `passed` 状态。

### 0.3 三道门的边界（不得混为一谈）

| 门 | 测什么 | 粒度 | 判据 | 成本 |
|---|---|---|---|---|
| **H2 → POOL_S3** | **个股因子**能否入池 | 每个因子 | POOL_S3：\|RankIC\|<0.015 **且** \|t_NW\|<2 **且** null 不显著 → 删除 | 秒级/因子 |
| **G0–G4** | **门控**有没有资格生成交互候选 | 每个门控 | G0 持续性 / G1 描述 / G2 组级对齐 / G3 块置换 / G4 独立性 | 秒级/门控（G2/G3 约 2.2s） |
| **F1/ΔIC** | **门控×因子交互**能否入池 | 每个 (门控,因子) 对 | §8.4：CI 下界>0 **且** ≥4/6 块 **且** placebo p<0.05，再过 FDR | 秒级（市场）/ ~2 分钟（行业） |
| 模型级增量（外部生产对照） | **模型对基线的增量** | 冻结池之后 | 喂/不喂候选特征的生产引擎双运行 | 不参与初筛（merge 内不实现） |

**三条不同的门，判据与粒度都不同**：H2 只记录不删（POOL_S3 才是唯一删除判据）；
G0–G4 是**门控资格**；F1/ΔIC 是**交互候选入池门**。

**"记录"与"决定"分属不同阶段**（此前表述不清，已修正）：

```
S5：计算并记录 H2 指标（个股线**不删除**任何因子）
S6：家族去冗余 + FDR 校正
S8a：按冻结的 POOL_S3 规则**决定**入池或删除
```

门控交互线同理：

```
S5：G0–G4 与 F1 分层评估
S6：适用的冗余处理 + 预注册检验族校正（约束见 PLAN §9.6）
S7：形成"**统计候选**短名单"
S8b：经**外部生产引擎对照（模型级增量）**+ 池构建规则 + 版本冻结后，才成为**训练可用特征**
```

### 0.3.1 状态生命周期（禁止用一个 `passed` 代表两件事）

| 状态 | 含义 | 能否用于"确认有效" |
|---|---|---|
| `candidate` | 已登记，未完成正式检验 | 否 |
| `qualified_stat` | 达预设**统计**筛选条件，允许进入模型级验证 | **否** |
| `tested_increment` | 完成模型增量测试（须记录基线与增强的对照结果、时间段、标签、交易指标） | 否 |
| `frozen_prod` | 通过规定验收流程，并**冻结**特征定义、数据版本与训练配置 | 否（仍须前向确认） |
| `rejected` / `quarantined` | 未通过，或因数据/版本/证据问题暂时隔离 | 否 |
| **`E3 confirmed`** | **前向确认通过**（数据未参与任何选择） | **是** |

> **`G2/G3/F1 passed` ≠ `E3 confirmed`。** 术语上必须分开：前者是"通过指定统计门槛"，
> 后者是"前向确认"。

### 0.3.2 三层证据不可互相替代

| 层 | 回答 | 不能证明 |
|---|---|---|
| **工程证据** | 代码能否运行、数据契约是否满足、产物是否完整 | 因子有效 |
| **统计证据** | 预先规定的检验是否通过、效果是否稳定 | 交易收益改善 |
| **经济证据** | 在实际交易约束下，样本外是否获得增量收益 | 未来持续有效 |

冒烟测试 14/14 通过提供的是**工程证据**：说明管道畅通、接口匹配、台账可写、错误可捕获。
它**不提供统计证据，也不提供经济证据**。（外部意见中"统计学上等价于零信息"的表述过头了——
工程信息是有价值的，只是不能用来暗示预测能力。）

### 0.4 数据流图

```
原始字段(cache/stock_features) → evaluator.load_panel
        │
        ├─ S0/S1 候选生成 + 指纹 ────────────────────────────────┐
        │   ├ 个股: grammar 进化 / refs / intake 提案            │
        │   ├ 门控: gate_search_space 规格元组(10基础量×4操作化)  │
        │   └ 登记: ix_build → hypothesis_registry.csv           │
        │                      │                                 │
        │            S2 预筛链（逐层留痕）                        │
        │              ├ 个股: lint/dust/腿冗余/vfp/…（9 层）      │
        │              └ 门控: 规格合法性/尘埃/值等价/G0+中性度    │
        │                      │                                 │
        │            S3/S4 快层(仅排序) + 排序分 + 停止规则        │
        │                      │                                 │
        │            S5 全层评估（唯一判定层）                     │
        │              ├ 个股: H2(rankic/t_NW/null) → POOL_S3     │
        │              ├ 门控: G2/G3 组级资格                     │
        │              └ 交互: F1/ΔIC + placebo + 块一致性         │
        │                      │                                 │
        │            S6 去冗余(dedup/G4) + FDR(bh_fdr)            │
        │                      │                                 │
        │            S7 短名单 + 台账回写                          │
        │                      │                                 │
        └─ S8 池(pool_legacy ΔComb / pool_v2 POOL_S3) → executor → 平台
                               ↓
                  台账(ledger / gate_ledger) / 报告(report) / 知识回流(knowledge)
```

### 0.5 模块清单（87 个 `.py`，合计 22,998 行）

| 类别 | 模块 | 行数 | 最后修改 |
|---|---|---|---|
| 入口与执行 | `run_search.py` | 2094 | 2026-10-10 |
|  | `smoke_pipeline.py` | 193 | 2026-10-11 |
|  | `executor.py` | 90 | 2026-09-15 |
|  | `_paths.py` | 11 | 2026-09-15 |
| 数据与标签 | `evaluator.py` | 583 | 2026-09-25 |
|  | `fitness.py` | 469 | 2026-09-25 |
|  | `windows.py` | 193 | 2026-10-10 |
|  | `field_catalog.py` | 164 | 2026-09-25 |
|  | `prodlabel.py` | 122 | 2026-10-08 |
|  | `boundary.py` | 76 | 2026-10-09 |
| 候选生成 | `grammar.py` | 1151 | 2026-09-26 |
|  | `knowledge.py` | 687 | 2026-09-24 |
|  | `refs.py` | 532 | 2026-09-26 |
|  | `refs_cond.py` | 329 | 2026-09-21 |
|  | `lint.py` | 306 | 2026-09-25 |
|  | `gates.py` | 272 | 2026-10-10 |
|  | `parser.py` | 262 | 2026-09-24 |
|  | `refs_cond_t1.py` | 258 | 2026-09-21 |
|  | `gate_search_space.py` | 260 | 2026-10-11 |
|  | `intake.py` | 236 | 2026-09-24 |
|  | `ix_build.py` | 202 | 2026-10-09 |
|  | `llm_client.py` | 147 | 2026-09-15 |
|  | `explain.py` | 112 | 2026-09-15 |
|  | `fixtures.py` | 94 | 2026-09-16 |
|  | `make_f1_cands.py` | 53 | 2026-10-11 |
| 评估（个股） | `stability_diag.py` | 574 | 2026-10-10 |
|  | `eval_v2.py` | 255 | 2026-10-10 |
|  | `h2_rerun.py` | 165 | 2026-10-10 |
|  | `fastranks.py` | 147 | 2026-10-10 |
|  | `dedup.py` | 66 | 2026-10-02 |
| 评估（交互/门控） | `f1_sector.py` | 247 | 2026-10-12（重写为搜索长表 schema） |
|  | `h3_eval.py` | 201 | 2026-10-10 |
|  | `f1_batch.py` | 165 | 2026-10-12（默认门控源改为搜索面板；旧门控删除） |
|  | `corr_compare.py` | 127 | 2026-10-10 |

> **已删除模块（2026-10-12，随旧手工门控退役）**：`gate_eval_v2.py`(202) /
> `gate_form_sweep.py`(209) / `m4_confound.py`(181) / `f1_gate_reeval.py`(150) /
> `gate_confound_check.py` / `gate_tercile_check.py` / `bench_sample_precision.py`。
> 它们的形式对比/混杂检验等问题已由 §2.7 的搜索空间 + `gate_g4_clusters.py`
> （series/cross 双口径）+ PLAN §16.24 的正交化证据取代。
> （历史行数记录以删除前版本为准。）
|  | `gate_confound_check.py` | 122 | 2026-10-11 |
|  | `bench_sample_precision.py` | 122 | 2026-10-11 |
|  | `gate_tercile_check.py` | 117 | 2026-10-11 |
|  | `judge_f1.py` | 88 | 2026-10-11 |
|  | `judge_gate_search.py` | 66 | 2026-10-11 |
|  | `gate_g2g3.py` | 374 | 2026-10-11 |
|  | `gate_g2g3_sector.py` | 360 | 2026-10-11 |
|  | `gate_validate.py` | 287 | 2026-10-10 |
| 门控流水线 | `run_gate_pipeline.py` | 312 | 2026-10-11 |
|  | `gate_ledger.py` | 302 | 2026-10-11 |
|  | `gate_stop.py` | 183 | 2026-10-11 |
|  | `verify_sector_inputs.py` | 157 | 2026-10-11 |
|  | `gate_search_screen.py` | 154 | 2026-10-11 |
|  | `confirm_candidate.py` | 141 | 2026-10-11 |
|  | `gate_search_prepare_f1.py` | 72 | 2026-10-11 |
| 池与榜单 | `build_top.py` | 704 | 2026-10-02 |
|  | `pool_legacy.py` | 448 | 2026-10-10 |
|  | `decorrelate_top.py` | 277 | 2026-10-02 |
|  | `pool_matrix_oct.py` | 209 | 2026-10-10 |
|  | `pool_v2.py` | 162 | 2026-10-09 |
|  | `pool_matrix.py` | 138 | 2026-10-10 |
|  | `pool_matrix15.py` | 98 | 2026-10-10 |
|  | `pool_matrix15b.py` | 62 | 2026-10-10 |
|  | `pool_matrix_oct_final.py` | 61 | 2026-10-10 |
| 台账/报告/窗口 | `ledger.py` | 304 | 2026-09-25 |
|  | `tag_search_windows.py` | 252 | 2026-10-10 |
|  | `report.py` | 239 | 2026-09-26 |
| 诊断（只作诊断） | `model_suitability.py` | 327 | 2026-10-02 |
|  | `nonlinear.py` | 264 | 2026-10-02 |
|  | `resid.py` | 212 | 2026-09-24 |
|  | `mkt_sector_test.py` | 209 | 2026-09-23 |
|  | `cond_ic.py` | 206 | 2026-09-25 |
|  | `cond_regime_ic.py` | 195 | 2026-09-26 |
|  | `offset_diag.py` | 157 | 2026-09-25 |
|  | `cond_supplement.py` | 154 | 2026-09-29 |
|  | `experiments_log_ab.py` | 83 | 2026-09-16 |
| 自测 | `selftest_supplements.py` | 600 | 2026-10-10 |
|  | `selftest_phase_a.py` | 288 | 2026-10-10 |
|  | `selftest_b3.py` | 268 | 2026-09-16 |
|  | `selftest_b2.py` | 262 | 2026-09-16 |
|  | `selftest_gates.py` | 170 | 2026-10-10 |
|  | `selftest_b0.py` | 168 | 2026-09-16 |
|  | `selftest_b1.py` | 130 | 2026-09-20 |
|  | `selftest_stop.py` | 90 | 2026-10-11 |

### 0.5.1 孤儿模块核查（回应"按文件列清单不等于证明没有孤儿模块"）

> 上表按文件类别列出，**不能证明每个模块都有明确的生产调用路径**。
> 本节按「生命周期 / 调用入口 / 输出消费者 / 退出条件」四字段核查。
>
> **生命周期取值**：`生产`（在正式链路上）/ `实验`（人工运行的对照或一次性分析）/
> `历史`（保留备查、新流程不用）/ `诊断`（只出报告，不产判定）/ `自测` / `待建`

| 生命周期 | 模块 | 调用入口 | 输出消费者 | 退出条件 |
|---|---|---|---|---|
| **生产** | `run_search.py` / `executor.py` | 人 / CLI | 平台批次、台账 | — |
| **生产** | `grammar` `parser` `lint` `explain` `intake` `llm_client` `knowledge` `refs` `ix_build` | `run_search.py` | `ledger`、登记册 | — |
| **生产** | `gates`（**共享基础设施**） `gate_validate`（G0） `gate_search_space` `gate_search_screen` `gate_search_prepare_f1` `gate_g2g3` `gate_g2g3_sector` `gate_ledger` `gate_stop` `run_gate_pipeline` | `run_gate_pipeline.py` / CLI | `gate_ledger`、短名单、PLAN §17.9 | — |
| **生产** | `eval_v2` `fastranks` `h3_eval` `f1_batch` `f1_sector` `judge_f1` `dedup` `eval`(fitness) `boundary` `windows` `field_catalog` `prodlabel` | 各评估脚本 | `h2_*.csv`、`f1_*.csv`、`*_judged.csv` | — |
| **生产** | `pool_legacy` `pool_v2` `ledger` `report` `tag_search_windows` | `run_search.py` / CLI | `pool_state.json`、报告 | — |
| **闸门** | `smoke_pipeline.py` | 人 / CI | `out/smoke/` 日志 | 上服务器前必须全绿 |
| **实验（人工运行）** | `corr_compare` `h2_rerun` `verify_sector_inputs` `confirm_candidate` `make_f1_cands` `judge_gate_search` `gate_g4_clusters` | **仅人工 CLI**（无自动调用者） | 报告 MD / CSV | **候选集体失效后可归档** |
| **诊断（只出报告）** | `stability_diag` `cond_ic` `cond_regime_ic` `cond_supplement` `offset_diag` `nonlinear` `resid` `model_suitability` `mkt_sector_test` `experiments_log_ab` | **仅人工 CLI** | 报告 / 候选清单（非进池结论） | 结论被采纳或否证后可归档 |
| **历史** | `pool_matrix` `pool_matrix15` `pool_matrix15b` `pool_matrix_oct` `pool_matrix_oct_final` `refs_cond` `refs_cond_t1` | 人 / 历史流程 | — | 已由 legacy 恢复的 API 覆盖；**不再用于新流程** |
| **自测** | `selftest_phase_a` `selftest_supplements` `selftest_b0..b3` `selftest_gates` `selftest_stop` `fixtures` | 人 / `smoke_pipeline.py` | PASS/FAIL | — |
| **基础设施** | `_paths` | import 时注入 sys.path | — | — |
| **待建** | `folds.py` `contract.py` `run_version.py` | — | — | 见 PLAN §13.1；**契约规格见 PLAN §18** |

**核查结论（诚实声明）**：

1. **不存在"无人调用"的死模块**——每个模块至少有一个调用入口（生产链、人工 CLI、或自测）。
2. **但存在 13 个"仅人工 CLI 调用"的实验/诊断模块**：它们不在任何自动化链路上，
   产物是报告而非判定。**这是设计选择**（诊断层不应自动跑），但意味着：
   **它们的产物不会被自动纳入台账，`selection_provenance` 也无法自动追溯。**
   → 若有诊断结论被用于决策，必须**手工登记来源**（PLAN §19.3 实验纪律要求）。
3. `refs_cond.py` / `refs_cond_t1.py` 归为**历史**：其功能已被 `refs.py` 的第 3 路径
   与条件补充覆盖，且不在 v2 链路上。**建议归档说明，但不删除。**

### 0.6 与其余文档的关系（防止架构分裂）

| 文档 | 视角 | 权威范围 |
|---|---|---|
| **本文（TECHNICAL.md）** | **模块与流水线视角** | merge 目录内所有模块的职责、接口、实现教训 |
| `docs/MERGE_V2_PLAN.md` | **规格视角** | v2 实施**唯一依据**；含 §17 八阶段流水线设计、§8 门控/交互判据、§9 评估规则。**本文的 v2 部分不得与其冲突** |
| `docs/CODEMAP.md` | 项目级 | 整个项目的代码-职责对照（不限于 merge） |
| `results/merge_v2/arch_review_v1.md` | 审查视角 | 三路径现状与模块职责核对结论（含代码/测试/文档三态矩阵） |
| `results/merge_v2/arch_gap_review.md` | 审查视角 | 现状—目标差距表（A 能力清单+代码依据 / B 差异表） |
| `results/merge_v2/search_arch_design.md` | 设计提案 | 薄编排层 + 三后端（**处置 2026-10-13：部分被实现取代，残余进 backlog，未整体批准**——文件头有处置标记） |
| `results/merge_v2/*_report.md` | 证据 | 各专题的 E1 证据与结论（见 §9 索引） |

**不新建架构文档**：本文是 merge 架构的唯一落点；新功能先更新本文对应模块节，再在 PLAN 修订摘要登记。

---

## 1. 数据与标签口径

### 1.0 数据文件与两条线的取用边界（2026-10-11 实测）

> 回答三个问题：个股搜索只用基础行情吗？门控线直接用了现成因子吗？「个股数据」在哪份文件里？

#### 1.0.1 三份数据文件

| 文件 | 列数 | 内容 | 起点 | 行数 |
|---|---|---|---|---|
| `cache/stock_features_20200101_20260918.parquet` | **41** | **股票基础行情**：open/high/low/close/volume/amount/pct_chg、total_mv/circ_mv、turnover_rate、涨跌停/ST/一字板标记、`future_return_1d~5d` | 2020-01-02 | 7,555,041 |
| `factor_lab_data/factors.parquet` | **262** | **因子值**：`date`+`stock_code`+**260 个因子列**（alpha_*、sector_*、mkt_*、t2_mined_* 等） | 2021-06-01 | 6,272,523 |
| `results/merge_v2/labels_prod.parquet` | — | 生产标签（`y_prod` / `r3` / `tradable` 等） | 2020-01-02 | — |

**两份行情/因子文件只共用 `date` + `stock_code` 两个键，无重叠列**：

- cache 独有 **39 列**（全部是基础行情/状态）；
- `factors.parquet` 独有 **260 列**（全部是因子值）；
- ⇒ **`factors.parquet` 不含任何 OHLCV / 市值 / 状态列**，因此它**无法**求值任何含
  `CLOSE`/`OPEN`/`TURNOVER` 的公式——两边的数据不可互换。

#### 1.0.2 各线的取用

| 线 | 主数据 | 用途 | 是否读 `factors.parquet` |
|---|---|---|---|
| **个股（进化搜索）** | **cache** | 候选公式求值：8 个平台字段（`OPEN/HIGH/LOW/CLOSE/VOLUME/AMOUNT/MARKET_CAP/TURNOVER`）+ 4 个本地字段（`MKT_RET/IND_RET/BETA60/DISP`，本地计算） | **仅池诊断**：12 个池成员中只有 1 个（`distance_from_52w_high`）的 `panel == 'factors'`；另 3 个读 `qs_*_panel.parquet`、1 个从 cache 的 `total_mv` 现算（`cache_log_total_mv`）、7 个 `inline`（用 evaluator 从 cache 现算）。用途是**池相关惩罚 / ΔComb**，**不参与候选生成与 IC 评估** |
| **门控** | **cache + factors.parquet** | ① 门控序列 `g`：`gates.py:23` 读 **cache** 的 `pct_chg/amount/open/high/low`（+ `moneyflow` 目录）② 个股因子 `x`：`boundary.read_factors` 读 **factors.parquet** 的 124 个 role=STOCK 列 ③ 标签：`labels_prod.parquet` | **是（核心输入）** |

**⇒ 回答「个股搜索需要 factors.parquet 吗」**：

- **候选生成与评估完全不需要**（只用 cache 的 8+4 个字段构造公式）；
- **池相关与 ΔComb 需要**，但只为取那 1 个 `panel='factors'` 成员的因子值。

#### 1.0.3 「门控线的个股数据」有两个含义，来源不同

| 含义 | 来源 | 内容 |
|---|---|---|
| ① 构造**门控序列**所需的基础行情 | **cache**（与个股线同一文件） | `pct_chg` / `amount` / `open` / `high` / `low`（+ moneyflow） |
| ② 交互检验所需的**个股因子 x** | **factors.parquet** | **124 个 `role=STOCK` 因子列**（由 `results/merge_v2/roles.csv` 定义） |

#### 1.0.4 `roles.csv` 的角色分类（174 行）

| role | 个数 | 说明 | 门控线是否用 |
|---|---|---|---|
| **STOCK** | **124** | 个股级因子（逐股逐日有值 → 可算日 IC） | **是（全部）** |
| SECTOR_STATE | 19 | 行业状态（日内取值数 ≈ 行业数） | 否（不进个股 IC） |
| MARKET_BROADCAST | 18 | 市场广播（全市场同日同值） | 否 |
| CROSS_SEC | 7 | 横截面类 | 否 |
| STOCK_REL | 5 | 个股相对（如 `stock_turnover_sector_resid_5d_pct`） | 否 |
| CROSS_MKT | 1 | 跨市场 | 否 |

#### 1.0.5 `ic_stock_2025.parquet`（门控线的核心输入之一）

- 内容：上述 **124 个 STOCK 因子的日 IC 矩阵**，**1115 日 × 124 列**（2021-06-01 ~ 2025-12-31）；
- 列集合与 `roles.csv` 的 STOCK 集**完全一致**（实测：双向差集均为空）；
- 构建：`stability_diag.ic_matrix` 经 `boundary.read_factors` 读 **factors.parquet**；
- 消费：`gate_g2g3.py` / `run_gate_pipeline.py` / `verify_sector_inputs.py` 直接读该矩阵，
  不重复读 factors.parquet。

> 更正：此前 PROGRESS 有一条把该因子数写成「123 个 STOCK」——实测为 **124**。

#### 1.0.6 F1 里的「个股因子 x」是什么、作用是什么

**F1 的形式**（`f1_batch.py:46-62`、`f1_sector.py`、`gate_search_space` 的候选定义）：

```
F1 = d · x · max(s·g, 0)          # x = 个股因子, g = 门控序列
```

实现层（`f1_batch.run_one_market`）：

```python
g   = gate_series                                  # 门控 p 序列（0~1）
on  = isfinite(g) & (s * g > 0)                    # 门控『开』的交易日
f1_ic = where(on, ic_z, nan)                       # 只在 on-day 保留个股因子的日 IC
delta = H.delta_ic(dates, f1_ic, ic_z)             # on-day 平均 IC − 全期平均 IC
```

其中 **`ic_z` = 个股因子的日 IC 序列**，来自
`E._daily_corr(xr, yr, day, nd)`；而 `x`（`base_cols`）由
`boundary.read_factors(...)` 从 **factors.parquet 的 124 个 `role=STOCK` 列**读取。

**⇒ 所以 F1 度量的不是『门控的行情 × 市场/行业数据』，而是：
「门控打开的那些交易日里，这个个股因子的 IC 是否显著高于它的全期水平」。**

**为什么必须挂个股因子**（回答「交互检验的个股因子是做什么的」）：

- **门控自身不能选股**——它只回答「当前处于什么状态」（`g` 是 0~1 的标量序列，
  对全市场/全行业广播），不产生任何横截面排序；
- 要变成可交易信号，门控必须**调制一个已有的 alpha**：门控开 → 用 `x` 选股；
  门控关 → 弃权（置 NaN）。这就是 `max(s·g, 0)` 的语义；
- 因此 `x` 的**作用**是：**提供横截面选股排序的被门控对象**。门控没有独立的股票层面含义。

**两层不要混**（用户设想的「门控序列 g 的行情 × 市场/行业数据」属于**构造层**）：

| 层 | 做什么 | 位置 |
|---|---|---|
| **构造层** | 由市场/行业/资金流数据构造门控序列 `g` | `gates.py`：`load_market_daily`→`build_market_gates`；
`load_sector_daily`→`build_sector_gates`；`load_moneyflow_panel`→`build_s3` |
| **检验层（F1）** | 门控状态 × **个股因子日 IC** 的条件增量 | `f1_batch.py` / `f1_sector.py` / `h3_eval.delta_ic` |

两层的输入文件也不同：构造层读 **cache**（行情）与 moneyflow；检验层读 **factors.parquet**（个股因子）。
### 1.1 evaluator.py — 本地面板与向量化求值 [既有]

- `load_panel()`：读 cache（`cache_utils.find_stock_features` 自动发现，禁止硬编码文件名），映射为平台字段名，AMOUNT 保持原始元。
  **实际文件（2026-10-11 实测）**：`cache/stock_features_20200101_20260918.parquet`
  （7,555,041 行 / 5126 只 / 2020-01-02 ~ 2026-09-18 / 1629 交易日）。
  **与门控线的 `factor_lab_data/factors.parquet` 不是同一张**（后者 6,272,523 行 / 2021-06-01 起）；
  但**重叠区逐项完全相同**（行数差 0、`(stock_code,date)` 键集合相同、日历相同、股票池相同）
  → 差异仅为 cache 多出的 2020-01-02 ~ 2021-05-31 前段（被 `fitness.START` 裁掉）。见 §8.10。
- 本地求值器只支持 VERIFIED/DOCUMENTED 分级算子（见 `field_catalog.py`）。
- 本地专用字段（MKT_RET / IND_RET / BETA60 等）惰性计算；含本地字段的公式禁止上平台（`formula_uses_local` 闸门）。
- `value_fingerprint(values, n_q=9)`：值分布指纹——`(v−median)/IQR` 归一 + 镜像规范化 + 分位量化 + sha1 前 12 位。
  用于"数学/仿射/镜像等价 → 同指纹"，被个股线（vfp 去重）与门控线（`gate_ledger.gate_value_fp`）共用。

### 1.2 fitness.py — 平台线评估 [既有]

- `DataContext`：c2c_Nd 标签（平台口径，已校准）、周期采样、aligned_v1 宇宙。
- 主排序分 SiA = |RankIC|×|ICIR|×胜率；官方 A（`sia_of`）用于上传决策。
- 方向只在训练窗冻结；holdout 冻结方向。
- 多标签：`--label c2c --label vwap` 同一次运行内因子值只算一次；生成排除按（公式，标签）对计数。
- `build_fast_context(frame, n_stocks=1800, months=24)`：**快层面板**（S3），仅用于排序截断。。
  **快评抽样实测（2026-10-11）**：截面抽 `2500 / 5126 = 48.77%`；**时间上不抽样**，
  快评评估窗 = `data_cut − 26 月` = **2023-10-31 ~ 2025-12-31（529 交易日）**；
  与全量窗（2021-09-01 ~ 2025-12-31，1050 交易日）叠加后，快评**实际只算全量的 25.57% 行数**
  （≈3.9× 压缩）。详见 §8.9。
  **注意**：快评窗**不是** 2021~2025——2021~2025 是**全量评估窗**。
  抽样为确定性（`seed=11`）；1800 只抽样股票全部在窗内有数据（有效样本 = 名义样本）。
  另：`--platform-prior` 与 `calibration.json` **只作用于 c2c**，vwap 无关（§8.8）。

### 1.3 prodlabel.py — 本地生产标签 [v2.8/P0 新增]

- 与主训练链路 `_generate_training_labels_for_all_data` 同口径：
  TP=(O+H+L+C)/4；buy=open(T+1)；sell3=0.7·TP(T+2)+0.3·TP(T+3)；sell5=0.7·TP(T+4)+0.3·TP(T+5)；
  R3、R5；DD3=min(low T+2,T+3)（T+1 不纳入，A 股 T+1 不可卖出）。
- 输出列：`y_prod`（R3≥7% 且 R5≥R3 且 DD3>−5%，即部署口径 multihorizon_35）、`y_r3dd`（去持续性对照）、`y_fade`（窗内回落约束候选）、`tradable` 可交易掩码。
- `check_vs_atr`：与 `daily_atr_label.build_atr_label` 的 R3 逐行对照。P0 实测 543,589 行，max 差 1.9e-08（门 ≤1e-6），通过。
- `label_stats`：正类基础率、零正类日占比、头部 5% 日占比、逐年基础率。全样本 y_prod 2.11%（`results/merge_v2/labels_prod.parquet`，本地重生成用时 9 秒）。
- 注意：cache 中的 `future_return_1d~5d` 口径未核验，禁止用于评估，以本模块重算为准。

### 1.4 boundary.py — 数据边界封装 [v2.8/P1 新增]

- 所有 v2 流程的数据读取必须经过本模块：`read_factors` / `read_labels` / `read_cache`。
- 强制 `DATA_MAX`（非标签）与 `LABEL_MAX`（标签型）两个上界：只返回上界之前的数据行，同时记录文件最大日期与返回最大日期（`file_max_date` / `signal_max_date`），写入 `results/merge_v2/boundary_log.jsonl`；返回为空则抛错。
- 实现教训 [v2.8/P1]：初版对整文件断言（文件最大日超上界即抛错），导致任何带截止日的版本读取都失败。已改为"过滤后返回"，§10.4 泄漏测试核对的是返回帧的最大日期。
- §10.4 边界泄漏测试与 §10.5 外层不泄漏测试依赖本机制。

### 1.5 windows.py — 门控线的窗口事实源 [v2.8.18 新增]

> **范围更正（2026-10-11 审计）**：本模块**不是全项目的窗口单一事实源**。
> 它被 18 个门控/池/诊断模块 import，**个股进化线不 import 它**
> （其窗口硬编码于 `fitness.py:29-31`）。详见 **§8.3**。

- 三个关键日期不可混用（PLAN §10.4）：`SIGNAL_MAX_DATE=2026-09-11`（数据上界，标签完整）、
  `LABEL_MAX_DATE=2026-09-18`（标签完整可得日）、`FORWARD_FIRST_SIGNAL_DATE=2026-09-21`（E3 起点）。
- **选择/监控分界**（PLAN §16.21）：`FACTOR_SCREEN_MAX` / `CV_MAX` / `TRAIN_SIGNAL_MAX` =
  **2025-12-31**（一切「做决定」的步骤只用这段）；`DECAY_MONITOR_START` = **2026-01-05**。
- `VERSION_WINDOWS` 固化 V1–V8 与 P 的 D_k / 外层窗 / 严格边界 S_k（C17：冻结后不得改动）。
- API：`prod_window()` / `decay_monitor_window()` / `legacy_search_window()` /
  `version_window(v)` / `refresh_prod_window()`（D13 向前滚动）/ `check_window()`。
- `check_window` 对 `evidence` 以 `E1` 开头的选择窗强制校验 `end < 2026-01-05`，越界即报违规
  ——**但该函数未被个股线调用**，故对个股线不构成实际约束。
- **审计认定的已消耗区段**：`CONSUMED_BY_STOCK_SEARCH = ("2025-07-01","2026-08-21")`
  （进化线 holdout 实测评估至 2026-07）；`UNTOUCHED_2026_REMNANT = ("2026-08-22","2026-09-11")`
  （仅 15 交易日，是否可用待审计 §20.7-2）。
- `f1_batch.py` / `f1_sector.py` 的硬编码 `LABEL_MAX` 已改为引用 `windows.FACTOR_SCREEN_MAX`，
  可用 `--label-max 2024-12-26` 复现旧结果。

> **已撤销的错误记述**：本节此前写"全部 P0/P1 搜索的 `label_max` 都是 2024-12-26
> （boundary_log 153 条记录反查）……2025 全年 246 个交易日**从未用于因子筛选**"。
> 该记述**不成立**：`boundary_log` 中 `data_max=2024-12-26` 的记录仅 10 条且
> **全部来自门控/诊断线**（label 为 `h3calib`/`m4diag`/`calib2`/`dbg*`），
> 该日志不含搜索侧记录；而个股线 holdout 覆盖 2025-07 起（含 2025 全年下半年与 2026 上半年）。
> `windows.py:44-47` 的 `LEGACY_SEARCH_*` 注释已就地加注更正。

### 1.6 field_catalog.py — 字段与算子白名单 [既有]

- `PLATFORM_FIELDS`（平台可表达的 8 字段）、`LOCAL_ONLY_FIELDS`（MKT_RET/IND_RET/BETA60/DISP）、
  `LOCAL_ONLY_OPS`（TS_REGRESS_RESID/BETA）、`VERIFIED_OPS`、`ARITY`、`CONST_WHITELIST`。
- `sw2_mapping`：静态申万二级归属（5466 股，131 行业）——**D5 已记录"静态映射的前视风险"**。

---

## 2. 候选生成

### 2.1 grammar.py — 模板与进化 [既有]

- 机制模板族 M1–M8，v2 增加 M9 复合 / M10 形状 / M11 状态门 / M12 文档；M14 上下文（6 形）、M16 跨机制交互、M17 纯上下文广播。
- 变异 / 交叉 / 种群生成；UCB 机制配额；精英回流；腿库（SiA 加权抽样 + 结构指纹多样性）。
- `expression_hash`（精确）+ `structural_fingerprint`（近克隆：字段多重集 + 带窗口算子序列 + 常数集）。
- **v2 规划**：候选语法收敛到 F1/F2/F3 与 STOCK_REL 相对化；M14/M16/M17 降为 legacy，不用于 v2（见 MERGE_V2_PLAN §9.6、§7）。

### 2.2 parser.py / lint.py / explain.py [既有]

- parser：平台公式字符串 → AST（提案与 LLM 输出入口）。
- lint：单位一致性、数学域、稀疏并列、MEAN 陷阱、组合期类型约束、数值尘埃、腿冗余三层闸门。
- explain：AST → 中文机制解释。

### 2.3 intake.py / llm_client.py / knowledge.py [既有]

- intake：提案闭环 parse → validate → lint → 小样本冒烟 → 去重 → 台账。
- llm_client：OpenAI 兼容提案通道，无 key 时用 inbox 模板。
- knowledge：机制知识库（`knowledge/mechanisms.jsonl`，append-only）；search 结束 auto_reflow（低产出 family 聚合为 failed 条目）。

### 2.4 refs.py / refs_cond.py / refs_cond_t1.py — 外部已有因子筛选 [既有]

- quantskills 三库 + dir-adapt；32 base × 8 transform 本地算子库；aligned_v1 直评入台账。
- `refs_cond*.py`、`cond_supplement.py` 为条件补充脚本（第 3 路径扩展，aligned_v1 口径）。

### 2.5 ix_build.py — v2 候选生成 [v2.8/P1 已实现并自测]

- 实现 MERGE_V2_PLAN §7：F1 门控乘积 / F2 行业相对化 / F3 门控选择，预算 F1≤140、F2=40、F3≤10、纯个股≤60、总数≤200（超限抛错）。
- `register` / `register_f2`：假设登记册 `results/merge_v2/hypothesis_registry.csv`，ID 全局唯一递增（HYP-0001…）。
- `build_f2`：行业相对化（sub/div），用静态 SW2 归属；自测确认组内均值恒为 0。
- `build_f1` / `build_f3`：需要门控序列 artifact，缺失时明确抛错。
- `load_sector_map`：路径修正记录——初版曾拼出 `to\to\local_data`，已改为 `dirname(ROOT)/local_data`。
- 联合评估结论 [v2.8/P1]：F2（sub/div）相对方向对齐后的基因子 delta 全负（均值 −0.0167，0/40 为正）。行业相对化移除了绝对收益信号而非提纯——base 因子的行业 beta 本身就是绝对涨幅的载体。div 形式另有构造缺陷（分母近 0 时比值爆炸），已从模板移除。**F2 不进入池**。~~见 `results/merge_v2/joint_eval_report.md`~~（**该报告于 2026-10-12 被误删**——删除批次按"旧门控联合报告"处理，实为 F2 证据；文件未被 git 跟踪，无法恢复。结论与数字以本行为准，F2 台账行与 `f2_panel.parquet` 保留。）
- 禁止事项：笛卡尔积、无假设候选、用门控切分训练集。

### 2.6 旧手工门控（M1–M4 / S1–S3'）—— **已删除**（2026-10-12，用户裁定）

> 门控定义现由**搜索空间生成**（§2.7）。本节正文保留为**历史记录**（定义与构造
> 的唯一权威文本在 PLAN §8.1），代码与产物已删：`gates.py` 拆分为只含共享基础设施
> （`thresh_form`/`_roll_quantile`/`load_market_daily`/`load_sector_map`/THRESH/QWIN/QMIN）；
> `gates/{market,sector}_gates.parquet` 与 6 个孤儿诊断工具、76 个旧结果文件删除；
> `f1_batch` 默认门控源改为 `gates_search/market_gates_search.parquet`，
> `f1_sector` 重写为搜索长表 schema（date/sector/gate_id/raw/p）。
> 取证报告（`gate_audit_M1M2S1S3.md`/`m4_diag_report.md`/`gate_forensics_*.md`）已删，
> 结论保留于 PLAN §16.24。

- **历史定义**（保留为记录）：M1–M4（市场）：`load_market_daily`（cache 的 tradable 非 ST 等权聚合）→ M1 20 日波动 / M2 60 日累计收益 / M3 成交额 log(mean20/mean120) / M4 上涨股占比 20 日均值 → 各取 252 日滚动分位。S1/S2（行业）：S1 行业 20 日收益 − 市场 20 日收益 / S2 四子指标（turn/vol/amp/share）各取 252 日分位后等权。S3'（行业资金）：moneyflow 原始列重算 → 行业等权均值 → 20 日均（min_periods=1）→ 行业自身 252 日分位（min_periods=63）。
- **门控形式**（§2.6.1）：`_g` 连续形式 `g=2p−1`（on = g>0，≈中位切分，保留约 50% 天）；
  `_t` THRESH 形式（**p≥0.7 → +1，p≤0.3 → −1，中间 NaN**，保留约 30% 天，[v2.8.19] 新增）。
  THRESH 的阈值 0.7/0.3 直接取自 PLAN §8.1 既有「on/off（p≥0.7 / p≤0.3）」定义——归一化
  形式继续存在于 `gates.thresh_form`（搜索线在用），已不再引用旧门控列。
- **实测（2026-10-10/11，选择窗 ≤2025-12-31）**：7 个手工门控在新窗 + THRESH + 完整 F1 下**全部需改或删**——
  M4 通过 16–17/20 但**是市场方向代理**（corr(M4_raw, 20日累计市场收益)=0.887，正交化 R²=0.791，
  正交化后 ΔIC→−0.0008 t=0.59；方向分层中**低方向层 on-day=0**）；M1 反向；M2/M3/S1/S2/S3' 均 h3_pass=0
  （M3 为弱效应：4 个过名义判据、0 过 FDR）。
- **自测（2026-10-12 重写）**：`selftest_gates.py` 夹具全部改为**合成数据**（不再依赖任何真实门控产物）——
  T1 共享基础设施性质 / T2 G0 覆盖率口径（合成预热序列复现 v2.8.18）/ T3 OPS 因果性 / T4 掩码等价性，全 PASS。

### 2.7 门控搜索空间与粗筛 [v2.8.19 新增]

- **`gate_search_space.py`** — 门控从手工定义扩展为**空间生成**：
  - 市场级 **10 个方向中性基础量**（截面收益离散度/IQR、跨行业离散度、大小盘分化、涨跌停比、
    成交额 HHI、换手/振幅/波动中位数、新高占比）× 4 操作化（level/chg/ratio/accel）；
  - 行业级 5 个基础量 × 受限操作化，**归一化用跨行业截面排名**（S1/S2/S3' 的失效教训）；
  - 共 **54 个候选**（market 40 / sector 14）。
  - 设计依据逐条对应 7 个手工门控的失效（M4 是方向 → 排除方向代理；M2 是累计收益 → 排除该构造；
    M1 水平反向 → 改用变化/加速度；S1/S2/S3' 自身历史分位全零 → 行业侧改截面排名）。
  - **实现教训**：行业分支曾写成 `transform(lambda x: fn(pd.Series(x.to_numpy())))`——**numpy 往返丢索引**，
    `transform` 按索引对齐时几乎全部错位为 NaN（raw 非空仅 0.75%、p 恒 1.0），
    导致 14 个行业候选全部被误判 `insufficient`。修正为 `fn(x)` 后 **G0 通过 40/54 → 54/54**、行业 eligible **0 → 14**。
- **`gate_search_screen.py`** — 第一级粗筛（秒级）：G0 资格 + **方向中性度**
  （\|corr(p, 市场方向分位)\| < 0.6）。自动抓出 4 个方向代理（`GM_newhigh_ratio_level` corr=0.826 等）。
- **`gate_search_prepare_f1.py`** / **`make_f1_cands.py`** — 搜索门控 → f1_batch 输入转换 / 候选 JSON 生成。

### 2.8 gate_stop.py — 门控搜索停止规则 [v2.8.23 新增]

- **与进化搜索同构的简单口径**（已核实 `run_search`：`--n` 撒种预算 / `--full-cap` 全评上限 / `--stall` 可选早停）：
  - **硬预算**：`n_seeds`（一批生成数）、`full_cap`（进昂贵层数）；
  - **可选早停**：`stall`（连续无改进），默认 0=关；
  - **方向级暂停**：连续 `keeper_rate_batches` 批 keeper 率 < `keeper_rate_min`(0.10)
    → **暂停该方向并诊断**（非永久终止；批次 <20 不按比例判定）。
- **keeper 定义分线**（不得混用）：门控线 = 通过 G2/G3 **且**方向中性；**正式入池候选 ≠ keeper**。
- `selftest_stop.py` **10/10 PASS**（stall 触发与重置、stall=0 关闭、预算扣减、连续两批暂停、
  仅一批不触发、批次过小保护、非连续低率不触发、持久化往返、缺文件回落）。

---

## 3. 评估体系

### 3.1 fitness rank_score（平台线） [既有]

- 综合排序分 = min(train, holdout) SiA × 复杂度因子 × 池相关惩罚（|ρ|>0.4 降权）× 残差掉幅惩罚（可选）× 平台先验。
  **平台先验是 c2c-only**：`--platform-prior` 为 opt-in（默认 `False`），只在 `evaluate_full` 的
  **c2c 主口径**（`run_search.py:575-583`）传入；快评的 `rank_score`（`509`）**不传**。
  **vwap 榜不用它**——按 `vwap_*` 排序（`1581-1584, 1633-1642`），其 `score` tie-breaker 由
  `_rescore_rows`（`1492`）从本地 c2c 派生 ⇒ **调整 vwap 无需处理平台先验**。详见 §8.8 / PLAN §2.5.1。
- LGB 复合分（榜2，自用）= |RankIC|×|RankICIR|×rank 胜率 × 单调性 × 换手软惩罚。

### 3.2 eval_v2.py — v2 评估引擎 [v2.8/P1 新增]

- 口径：tradable==1 样本；每日截面有效样本 <200 的日期剔除；RankIC 对 tp 口径 R3；t_NW 用 Newey–West lag=3。
- 函数：
  - `rankic_series`：逐日 Spearman IC（bincount 向量化）。
  - `nw_t`：Bartlett 核 Newey–West t。
  - `perm_null_mean`：日内打乱因子的 RankIC 均值零分布，双侧 p（有限置换校正 `(1+k)/(B+1)`）。
  - `topk_lift`（K=5/20/50）、`auc_day`（Mann-Whitney 秩公式）：**仅诊断**。
  - `h2_record`：子检验 A（RankIC_mean、t_NW、null p）+ 诊断列 B–E（hot5/hot2/mkt_up/mkt_dn，只作标注）。
  - `pool_s3_delete`：**POOL_S3 删除判据（唯一）**：|RankIC|<0.015 且 |t|<2 且 null 不显著，三者同时满足才删除。
  - `bh_fdr`：BH-FDR，q ≤ 0.10。
- 自测：已知因子 `pullback_depth_from_10d_high` 全历史 RankIC 均值 0.0172（与独立探针一致）、t 2.6、null_p 0.0。
- 实现教训：初版逐日 Python 循环 20 分钟超时，改为 bincount 向量化；`_within_rank` 曾返回全局排名
  （RankIC 碰巧正确、AUC 越界），已修正为真正的日内平均秩。

### 3.3 诊断集（只作诊断，不作门） [既有]

- `cond_ic.py`：条件 RankIC（hot5/hot2_q 系列 + mkt_up/mkt_dn），与主管线同口径。
- `cond_regime_ic.py`、`refs_cond*.py`：regime/条件补充。
- `offset_diag.py`：offset 0/1/2 错位截面 RankIC 一致性（调仓起点稳健性）。
- `nonlinear.py`：月度 MI / 尾部 JSD / 条件 MI / 联合 MI（见 `docs/nonlinear_selection.md`）。MI/JSD/CMIM 只产候选，不产结论。
- `resid.py`：风险轴残差 IC（MARKET_CAP→TURNOVER→VOL20 顺序正交化）与行业中性化诊断（仅诊断）。
- `model_suitability.py`：twin 否决 → CMIM 否决 → JMI 排序，产 add-one 候选短名单（非进池结论）。
- `mkt_sector_test.py`：市场层/行业层因子枚举评估。
- `experiments_log_ab.py`：log 政策 A/B（市值一律 LOG 等结论）。

### 3.4 fastranks.py — 评估加速层 [v2.8.19 新增]

- **三处优化**（profile 驱动，单因子 **36.9s → 4.5s，8.2×**，与 `eval_v2.h2_record` **逐位一致**）：
  1. `blockwise_rank`：逐块 `scipy.stats.rankdata`（块已连续，免双键 lexsort），较 `eval_v2._within_rank` 快 **2.4×**，逐值一致；
  2. `PanelRanker`：缓存 day 结构与 **y（标签 r3）日内排名**——该排名对所有因子相同，旧实现每因子重复算 2 次；
  3. `perm_null`：利用"有限值置换下 n/Σx/Σx²/Σy/Σy² 均为不变量、**只有 Σxy 变**"→ 每次置换 1 次 bincount 而非 7 次。
- `h2_core(factor)`：产出 H2 核心列，与 `eval_v2.h2_record` 同 seed **逐位一致**。
- `h2_rerun.py`：新选择窗下的 H2 重跑工具（多进程；`--verify` 对拍原实现）。

### 3.5 stability_diag.py — 因子时间稳定性诊断 [纯诊断]

- 诊断量：`slope_t`（全样本时间趋势 NW t）、`max_consec_neg_blk`、`delta_<blk>_t`、
  `jack10/jack05`（丢弃 top q |IC| 天后的 IC 均值比）、`ic_acf`（acf1/acf5 + n_eff，Bartlett 型）、
  `sc_*`（符号一致性）、`H`（块间异质性）。
- 零分布校准：**块内置换**（block=5 覆盖 R3 的 3 日标签重叠）。仅对**顺序相关**统计量有效——
  `conc_top10` 对时间顺序完全不变（置换后变动 0.00e+00），`top10_contrib` 分母在 H0 下退化（已弃用）。
- **结论为负**：124 个 STOCK 因子的 `slope_t<−2` 为 **0/124**（零分布校准后 0.0%）；
  集中度命中项与 POOL_S3 完全重合（61 个 \|t\|≥5 强因子中命中 0）→ **不新增稳定性门**。
- 结构性成因：acf1 中位 **+0.458**（有效样本 435/869 天）、PC1 28.4%、46.2% 因子对 |ρ|>0.3。
- 默认窗口为设计窗（`windows.LEGACY_SEARCH_END`），**2025-01-02~2026-09-11 的 415 个交易日保持封存**。
- 见 `results/merge_v2/stability_diag_report.md`。

### 3.6 h3_eval.py — 条件效应评估（检验 A） [v2.8.7 新增，v2.8.9 修正]

- `build_f1`：`F1 = d·x·max(s·g_c, 0)`。
- `delta_ic`：**弃权语义**——分开取均值再相减（**不得**用 `nanmean(F1−base)`）；
  块自助（块长 5）每次重算两侧均值；6 块一致性计数。
- `slope_test`：`RankIC ~ g` OLS + NW（**仅解释，不作门**，见 PLAN §8.4）。
- `block_placebo`：循环块置换（块长 20）× 200 次，`p=(1+k)/201`；**NaN 位置固定**，只在有限值间置换。
- `level_leak`：F1 日截面 spread 与 |g| 的**时间序列**相关（>0.8 且 ΔIC 不显著 → 判 leak）。
- 自测：合成真效应检出（p≈0.02）、零效应 p≈0.96、30 个 AR1 随机门控 p 均匀（mean≈0.49）。
- **实现教训**：`delta_ic` 与 `block_placebo` 初版用 `nanmean(F1−base)`（共同有限日配对差），
  在 on-day 上 F1 与 base 排名完全相同导致差恒为 0、p 全为 1.0。已改为分开取均值再相减。

### 3.7 F1 评估（候选级）：f1_batch / f1_sector / judge_* [v2.8.9 起]

- **`f1_batch.py`**（市场门控）：**掩码捷径**——on-day 上 `F1=d·x·max(s·g,0)` 与 `d·x` 日内秩相同
  （已证明 + `selftest_gates.py` T4 钉住），故只置换掩码、不重排。**秒级/候选**。
  新增 `--form g|t`（连续/THRESH 形式）与 `--gates-parquet`（指向搜索空间候选池）。
- **`f1_sector.py`**（行业门控）：**全重算**——门控因股而异，必须重建 F1 分数后排名。
  已接入 `fastranks.blockwise_rank`；新增 `--form`；**修复"只接受 S1/S3"的硬断言**（S2 首次可评）；
  新增 `--sample-stocks`（**仅粗筛，实测不得用于判定**，见 §3.11）。
- **`judge_f1.py` / `judge_gate_search.py`**：判定口径 = PLAN §8.4 三项
  **`(a) CI 下界>0` 且 `(b) nblocks≥4` 且 `(c) placebo p<0.05`** + BH-FDR(q≤0.10)。
  **实现教训**：初版误用 `nblocks>=5`（由 M4 旧数据反推——旧批次 2 个失败者 nblocks=4，
  但其主因是 p=0.164，块数非判别因素，故 100/100 吻合掩盖差异）。修正后仅 M3-t 结论改变
  （"0 通过" → "4 个过名义判据、0 过 FDR"，弱效应而非无效应）。
- **`f1_gate_reeval.py`**：市场门控 F1 补测（按与 f1_batch 完全相同口径，对拍至 3.99e-17）。
- **F1 联合判定（历史）**：旧门控族的 `f1_joint_judged.csv`（100 候选联合 BH-FDR；
  h3_pass 18/100、FDR 12/100，全 M4——该 12 个**无效**，M4 是方向代理）与
  `f1_joint_report.md` 已随 2026-10-12 旧门控删除；**现行族判定走
  `f1_family.py` 清单 + `judge_f1.py --family` 硬约束**（见 §3.11/§8.12）。

### 3.8 门控评估工具集 [2026-10-10/11 新增；2026-10-12 收缩]

> **2026-10-12 随旧手工门控退役删除**：`gate_eval_v2`（正式评估，被 `f1_batch`/`f1_sector`
> + `judge_f1` 取代）、`gate_form_sweep`（阈值扫描，THRESH 阈值已冻结于 §8.1/`thresh_form`）、
> `gate_confound_check`/`gate_tercile_check`/`m4_confound`（M4 正交化混杂检验——结论已
> 收录 PLAN §16.24：M4 正交化后 ΔIC→−0.0008 (t=0.59)、低方向层 on-day=0 ⇒ 方向代理）、
> `f1_gate_reeval`（旧 M3 复评）。** surviving 工具**见下。

| 模块 | 作用 |
|---|---|
| `corr_compare.py` | 因子值相关 vs IC 序列相关的口径辨析与对照矩阵 |

**组合形式的关键结论**：THRESH 形式在**新旧两个窗口**上一致把四个市场门控的效应量放大 **1.4~3.9 倍**
（M2 +0.0029→+0.0110、M3 +0.0057→+0.0124、M1 −0.0067→−0.0160、M4 +0.0222→+0.0312），
M2 一致性升至 20/20。**但阈值的选取须预注册**（τ=0.7 取自 PLAN §8.1 既有定义，非事后挑选）。

**方向混杂的决定性判据是分层检验**：M4 在低方向层 **on-day = 0**（是择时开关而非条件增强）；
`GM_hl_spread_level` 出现**符号翻转**（低 −0.0082 / 高 +0.0409）；`GM_limit_ratio_ratio` **三层全正**。

### 3.9 gate_validate.py — 门控资格 G0–G4 [v2.8/P1 已实现；v2.8.18 修订]

- **G0.1 二值资格**：on 段长 ≥20 日、年翻转 ≤8、自相关半衰期 ≥20 日；状态三值 pass/fail/insufficient。
- **G0.2 连续资格** [v2.8.18]：覆盖率 ≥0.8、std ≥0.2、\|中位数\| ≤0.8。
  **覆盖率分母改为评估窗**（新增 `eval_start` 参数）——旧口径用全历史行数，把评估窗之前的 warm-up
  计入缺失：M3 全历史 cov=0.7729<0.8 判 fail，评估窗口径 cov=0.9767 **本应通过**，导致
  **M3 与 S2 被误杀、从未进入 F1**。按 §8.2.4 作为**新规则版本**生效，**不回溯修改旧判定**
  （`gate_g0.json` 保留，新判定在 `gate_g0_v2818.json`）。
- **G1**：门控与未来 5 日收益的分年相关（**仅描述**）。
- **G2/G3** [v2.8.22 实现]：见 §3.10。
- **G4 独立性**：门控间 |ρ|≤0.8 → 保留 G0–G3 更优者。**注意 G4 不是单门控固有属性**——
  它依赖当前门控集合，新门控加入后须重算。
- 记录字段：`form` / `g0_status` / `evidence_status` / `eligible_for_pool` / `reason`。
- **实现教训**：G4 初版把同门控 `(p,g)` 对标为相关（`g=2p−1` 单调变换必然 ρ=1），已排除；
  行业跨行业汇总规则（≥50% pass）先写进 §8.2.5 再实现。
- **门控审计（历史，2026-10-10）**：曾产 `results/merge_v2/gate_audit_M1M2S1S3.md`
  （通读旧 `gates.py` + H3 对齐点核对：M1/M2/S1/S3' 实现无系统性错误；S1 的
  groupby-rolling-reset_index 写法依赖输入排序，已验证正确；S3' vendor 数据质量
  为残余风险）。**该报告与旧门控定义代码已于 2026-10-12 一并删除**；
  定义的唯一权威文本在 PLAN §8.1，淘汰结论在 PLAN §16.24。
- `coverage_diag` [v2.8.5]（已删除，2026-10-12）：g0 覆盖率诊断只解释失败来源；
  其职责由 `gate_search_screen.py` 输出的 cov 字段承接（同作诊断不作门）。

### 3.10 gate_g2g3.py / gate_g2g3_sector.py — 组级门控资格 [v2.8.22 实现]

- **口径**（严格按 PLAN §8.2 原文，**不做自由度折算**）：G2 = 条件斜率中位数 +
  **4 折同号** + **95% 区间不含 0** + **同号因子占比 ≥60%**；G3 = 块置换 200 次超零分布 95 分位。
- **95% 区间用「块自助法（块长 5）」对时间块重采样**——同一次重采样应用于全部因子，
  因子间相关性由时间块结构**自然保留**，无需 n_eff 折算（`n_eff≈1.5` 时 `df=n_eff−1` 无合理解释）。
- **方向对齐**：按 §8.4.1 对全体因子定向（`sign(全样本 RankIC 均值)`）。
  **实测 bug**：未定向时因 16/20 的 base20 方向为 −1，组中位统计量符号整体翻转
  （曾导致 M4/`GM_limit_ratio_ratio` 的中位 ΔIC 由正变负）。
- **`gate_g2g3_sector.py`（行业专用后端）**：市场版门控是每日标量 → 掩码捷径；
  行业版门控**因股而异** → 必须按 `f1 = d·x·max(s·g,0)` 重建后排名。
  G3 用**受限因子集 × 受限置换**（成本 = n_perm × n_factors，全量 200×123 ≈10h 不可行），
  仅构造零分布标度；**双侧检验**（单侧会把 M1 式负向效应误判为"未超零分布"）。
  **性能**：`prep_row_index` 预计算逐行 `(date_idx, sector_col_idx)`——原实现把 500 万行的
  Python 级字典查找放在每次 `f1_ic_series` 内（≈8s/次，累计约 464s）→ **427s → 79s（5.4×）**。
- **⚠ 2026-10-12 结论更正：G2 属 bug 修复，G3 属规格变更（不可混称）。**
  **G2**：§8.2 冻结原文即「95% 区间**不含 0**」（双侧）⇒ 原 `ci_lo > 0` 是**规格违背**，
  修复 = **恢复规格**（`ci_excludes_zero(lo,hi)`，4 处统一）。
  **G3**：§8.2 冻结原文是「真实值**超过**零分布 95% 分位」（单侧）⇒ 原 `obs > q95` **符合规格**；
  sector 版此前已擅自双侧（其注释说明理由），把市场版对齐它 = **规格变更**（§8.2 已改为 v2.8.28 双侧）。
  **方向语义已用合成数据验收**（`selftest_gate_direction.py` **8/8**：POS/NEG/NULL × pos/neg）。
  **G4 已跑**（`gate_g4_clusters.py`）：行业 14→12 簇、市场 36→34 簇（中位 |ρ| 0.126/0.064）——
  **否证了「行业门控高度近克隆」的推断**；但 G4 只答独立性，**不答有效性**。
  效果：行业 G2 **0/14→14/14**、市场短名单 **1→3**；`GM_amt_hhi_chg__neg` **0/20→18/20**。
  **全部结论仍为 E1**（全族 FDR 0/1440，Case B 未解）。详见 PLAN §20.20。

- **（已废弃的旧表述）G2/G3 判定原为单侧，已改双侧。**
  原 G2 的 `ci_lo > 0`（4 处）只认正向，把 ΔIC 稳定为负的门控全判 fail；
  市场 G3 的 `obs > q95` 同病。新增 `ci_excludes_zero(lo,hi)`（`lo>0 or hi<0`）统一四处；
  G3 双侧 + `p_two_sided`。**效果**：行业 G2 **0/14→14/14**、市场短名单 **1→3**。
  另：`gate_search_prepare_f1.py` 原**硬编码 `s=1`**（只出 `__pos`），
  使负向门控**从未在正确方向被测过**；现默认双向（`--pos-only` 退回）。
  修复后 `GM_amt_hhi_chg__neg` **18/20（全场最强）**。详见 PLAN §20.19。

- **G2 口径分歧（待裁决）**：PLAN §8.2 写的是「线性条件斜率」，而 F1 组合用「THRESH 阈值对比」，
  实测二者可符号相反。本实现**并列输出两种口径**。
- **结果**：40 个市场门控 → G2【斜率】2/40、G2【阈值】3/40、G3 4/40；通过的 3 个
  （M4、`GM_limit_ratio_ratio`、`GM_hl_spread_level`）**恰好等于 F1 找到的集合**，
  而耗时 89 秒 vs F1 的 8'51" → **验证了"先 G2 组级筛门控 → 再跑候选级 F1"的顺序**。
- **G2 不控制方向混杂**：M4 通过 G2/G3 但仍是方向代理 → G2 是门控资格检验，
  方向混杂须由独立的分层检验（§3.8）处理，**两层缺一不可**。

### 3.11 bench_sample_precision.py — 抽样精度基准 [v2.8.23 新增]

- **结论：1500 只抽样被实测否决**。对 3 个行业候选（按 \|ΔIC\| 取大/中/小）分别用全量与三种子抽样：
  **符号一致率仅 1/3**；误差中位 0.0026、最大 0.0066，与 ΔIC 本身同量级；
  且抽样值系统性偏正（说明不只噪声、还有偏置）。
- **⇒ 抽样截面路线否决**，`f1_sector --sample-stocks` 保留但标注"**不得用于任何判定**"。
  两级评估的正确形态是 **G2 前置**（便宜、用全量面板与全部因子、无抽样偏差），而非抽样面板。

---

## 4. 去重、池与榜单

### 4.1 dedup.py [既有]

- 月末截面 Spearman + 轴判定；结构指纹跨 run 落盘（sidecar）。
- 稀疏并列拦截（截面独立值中位数 <100 不进全量）。
- `monthly_spearman(d, start='2023-06-01', end='2026-07-10', min_obs=50)`：月内**全部交易日**逐日 Spearman 再跨日取均值。

### 4.2 pool_legacy.py [B1 重建] / pool_v2.py

**pool_legacy.py**（原 `pool.py` 被 v2 覆盖后按调用方实测用法重建，报告见
`results/merge_v2/b1_restore_report.md`）：

- 官方三段式 `Comb = 0.20·NA + 0.35·NB + 0.45·NC`；`NA = clamp(rawA/0.08)×Decay`（受 cap），
  `NB = clamp(rawB/0.06)`，两者与平台积分敏感度（+0.001 rawA=+100 分、rawB=+233 分）自洽。
- API（10 个）：`load_pool_state` / `all_entries` / `build_extra_panel` / `comb_proxy` /
  `comb_table` / `marginal_sets` / `leave_one_out` / `greedy_forward`（只加不换）/
  `refresh_member_sia` / `_Universe`+`_plan`（内部）。
- C 代理复用 `pool_sim.pool_backtest`（`build_panel_daily` 为唯一 construction 路径）。
- `mktcap` 无落盘面板，按 `local_mktcap.py` 口径 `log(clip(total_mv,1))` 由 cache 现算。
- **`refresh_member_sia` 按各成员自身日期栅格评估**（非 cache 派生栅格）：面板只落调仓日
  且与 cache 栅格零重叠；ICIR 取**非年化** `mean/std`（与 screen 落盘同量级）。
- 调用方 9 个：`run_search` / `selftest_phase_a` / `selftest_supplements` / `pool_matrix` /
  `pool_matrix15` / `pool_matrix15b` / `pool_matrix_oct` / `pool_matrix_oct_final` /
  `pandaai_online/diag_pool_amihud`。
- **未决**：`G4_EXPECTED`/`S6` 为 2026-09 七人 P7 池的过期标定常数（G3 已精确复现
  ρ=−0.8129）；`refresh_member_sia` 与存量 `sia` 残余口径差 0.46×–1.9×。裁决前不改常数。

**pool_v2.py**（原 `pool.py` 的 v2 内容）：`hygiene_report` / `redundancy_groups` /
`pool_s3_decide` / `check_delete_frac`（超 20% 熔断）/ `gates_impact_report`。
自测：卫生门正确拦截离散型因子（`trend_consistency` nunique=6）；近重复列 ρ=1.0 正确成组；
20% 熔断抛错。

### 4.3 build_top.py / decorrelate_top.py [既有]

- 双榜单聚合：榜1官方 A/c2c（上传决策）、榜2 LGB 复合分/vwap（自用）。
- **已知泄漏风险**：全窗排序后用于选择构成选择偏差（用户笔记 20260929.txt 已标记）。
  v2 要求任何排序或选择只能用折训练截止日之前的数据（as-of 纪律）。

### 4.4 pool_matrix*.py [既有，历史]

- pool_matrix / pool_matrix15 / pool_matrix15b / pool_matrix_oct*：池矩阵各期实现，保留备查，新流程不用。

---

## 5. 台账、报告与知识

### 5.1 ledger.py — 个股线台账 [既有]

- JSONL 事件台账 + `ledger_summary.json` sidecar（增量消费：记 byte_offset，避免全表重扫）。
- 事件：`proposal|local|holdout|platform|decision`；`latest()` 按 candidate_id 合并。
- `hashes_for(labels)`：生成排除集（公式×标签分离）——只测过 c2c 的公式在 c2c+vwap 联测时会
  重新生成以补齐证据；显式事件（proposed/rejected/duplicate）不论标签一律排除。
- `count_tested()`：唯一 (candidate_id, label) 测试对 → **多重检验分母**。

### 5.2 gate_ledger.py — 门控线台账 [v2.8.23 新增]

- **三级指纹**：`candidate_fp`（规格元组）/ `family_fp`（(层级,基础量) = 近克隆家族）/
  `value_fp`（复用 `evaluator.value_fingerprint` 对 p 序列做值等价判定）。
- **候选身份不随测试窗口改变**；窗口/形式/因子集/数据版本/**n_perm/n_boot** 属 `caliber`。
- 状态判定：`completed`（该 caliber 已测）/ `untested` / `stale`（口径已变）。
  `pending(specs, caliber)` = 只返回需执行的（对应 `ledger.hashes_for`）。
- append-only JSONL + `gate_ledger_summary.json`（byte_offset 增量消费）。
- **口径缺口修正**：`CALIBER_FIELDS` 加入 `n_perm`/`n_boot`——实测 G3 判定在 n_perm 100→60 时
  **翻转**，不记入口径会误判 `completed` 而跳过必要的重测。
- **验证**：同口径二次运行 → 待执行 **0/40**（completed）；改窗口/形式/因子集/分辨率 → **40/40 stale**。

### 5.3 report.py / knowledge.py [既有]

- `report.py`：短名单 JSON → Markdown 会话报告（含贪婪 ΔComb 段）。
- `knowledge.py`：见 §2.3。

### 5.4 confirm_candidate.py — 候选正式确认流程 [v2.8.23 新增]

- 流程：**台账先查**（避免把同批数据的旧跑当独立新证据）→ 记 `provenance` 事件
  （含选择偏差标注）→ 冻结口径跑 F1 → `f1` 事件回写（含"同批数据"标注 + 层级检验范围）。
- **`GM_limit_ratio_ratio` 结果**：`provenance` 记「**搜索幸存者，待正式确认**」
  （54 规格 → 36 eligible → 40 全跑 → 短名单 1 个；存在选择偏差，不得因其唯一幸存而改阈值）；
  冻结口径（n_perm=200/n_boot=2000）下 **h3_pass 7/20、FDR 7/20**（delta 中位 +0.01754、20/20 同向）。
  **模型级对照（外部生产引擎）未运行**（TECH §8.17 E1）；独立确认须用未参与候选筛选的前向数据。见 `results/merge_v2/gate_search_report.md §10`。

### 5.5 tag_search_windows.py — 已搜索候选打标 [v2.8.16 新增]

- 给已搜索候选追加 `search_start`/`search_end`/`label_available_max`/`evidence` 四列 +
  产出 `search_registry.csv`；原件备份 `results/merge_v2/_untagged_backup/`，`--restore` 可还原。
- 窗口由 `boundary_log` 反查推出，不硬编码；**自测/调试读取（读到文件最大日）被显式排除**。

---

## 6. 门控流水线编排

### 6.1 run_gate_pipeline.py — S0→S7 串联 [v2.8.23 新增]

- **与个股线两段式分工对齐（2026-10-13）**：搜索 = `gate_search_space.py`；**本脚本即
  "筛选+排名"命令**（S2 八层 → G2/G3 → S4 排名 → G4 → S7）；`--topn N`：短名单只保留
  S4 排名前 N（0=全部存活者）——**只影响短名单，族校验仍按全族 720**；逐个检验/模型级
  增量分别为独立人工步骤与仓库根生产脚本（见 §8.16 更新注）。
- **只跑台账中未完成/过期的**（`gate_ledger.pending`）。
- S2 预筛链（本实现含）：①`candidate_fp` 去重 ②`family_fp` 记账 ③**规格合法性**（分位窗<63、层级未知 → 拒）
  ④**尘埃检测**（p 与时间的秩相关 ≥0.9 → 退化为时间趋势）⑤G4 门控间冗余 ⑥**value_fp 去重**
  （**与个股线 `run_search.evaluate_fast` 的 vfp 语义一致：仿射/镜像/重标定等价 → 同指纹 → 拒**；
  实测 `vfp(a)=vfp(2a+3)=vfp(1−a)` 而 `vfp(a²)` 不同）
  ⑦G0+方向中性度 ⑨**稀疏性**（on 占比/独立值数，**告警不设门**）。
- S4：`gate_score`（预注册权重，**仅用于截断**）+ 停止规则（`gate_stop`）。
  **`gate_score` 不含家族惩罚项**——原设计的 `family_penalty × (family_multi>1)` 是二值项，
  而搜索空间里同一基础量的 4 个操作化构成等规模家族（`family_multi` 恒为 4），
  **对所有候选取值相同 → 恒无区分力**。家族内近克隆的取舍属 S6 去冗余（`dedup`/G4），
  不应伪装成排序分项；`family_multi` 仅作报告列。
- S5：G2/G3（复用 `gate_g2g3`）+ 方向分层；S7：短名单 + 台账回写。
- **首次运行**：40 门控 49 秒 → 短名单 1 个。**回归**：与 `gate_g2g3.py` 逐位一致（\|Δ\|=0.00e+00，36/36）。

### 6.2 verify_sector_inputs.py — 行业侧接入前验证 [v2.8.23 新增]

- 三项验证（Codex 要求）：**日期对齐**（与 STOCK IC 矩阵 1115 天全交集）、
  **行业成分/缺失/样本量**（成分<5 只的行业 7/131，§8.2.4 要求单独报告）、
  **泄漏风险**（静态 SW2 映射的 D5 已知前视风险；非重叠前视探测 ρ=−0.049 → 通过）。
- **实现教训**：初版用 `S1_raw.shift(-1)` 做前视探测——但 S1_raw 是 20 日累计，
  相邻窗重叠 19/20 天，必然高相关（+0.878），**是机械重叠不是泄漏**。改用非重叠 `shift(-20)`。

### 6.3 smoke_pipeline.py — 全流程冒烟闸门 [v2.8.23 新增]

- **上服务器大规模搜索前的验证闸门**。逐阶段调**真实 CLI**：
  S0 生成 → S2 预筛 → 市场流水线 → 行业流水线 → **增量对账（二次运行应 0 pending）**
  → 台账统计 + 停止状态。
- 默认清空台账保证可复现；运行前删除目标产物以避免"假通过"。
- **⚠ 小样设置不产生统计结论**（样本量/置换次数/分辨率均压缩）。
- **当前状态：14/14 通过、113 秒。**

---

## 7. 执行、导出与自测

### 7.1 run_search.py — 入口 [既有]

- 子命令：`search` / `rescore` / `upload` / `export` / `intake` / `propose` / `report` /
  `ledger` / `fixtures` / `platform-ingest` / `calibrate` / `decay-check`。
- search 主循环（`SearchRun.run`）：`_seeds()`（UCB 机制轴配额）→ `evaluate_fast`（**9 层预筛链**：
  hash → struct_fp → lint → dust → 腿冗余 → vfp → screens/tier1 → dual → stability/sparse）→
  精炼（mutate/crossover/window_neighbor + `mutation_stats.json` 分支增益加权）→ 幸存者截断 →
  `evaluate_full` → 短名单 + greedy → `ledger.log`。
- **停止参数**：`--n`（撒种预算）、`--full-cap`（全评上限）、`--stall`（连续无改进早停，0=关）。

### 7.2 executor.py — 平台导出 [既有]

- 只写 `candidates.txt` + meta（经 `batch.py` 校验，**零平台调用**）。

### 7.3 自测清单

| 脚本 | 覆盖 |
|---|---|
| `selftest_phase_a.py` | 5 门：求值器回放 / 筛选一致性 / 轴检测 / **池边际** / 导出格式 |
| `selftest_supplements.py` | 8 门：parser/lint/explain/knowledge/intake/greedy/report/llm/local-context |
| `selftest_b0/b1/b2/b3.py` | 面板构造、筛选、grammar v2 模板族、专项 |
| **`selftest_gates.py`** | 4 项：**S1 排序依赖**（T1）/ G0 覆盖率口径（T2）/ 门控因果性（T3）/ 掩码等价性（T4） |
| **`selftest_stop.py`** | 10 项：stall 触发与重置 / 预算扣减 / 连续两批暂停 / 批次过小保护 / 持久化 |
| `fixtures.py` | 固件因子定义 |

### 7.4 待建 [见 MERGE_V2_PLAN §13.1]

- `folds.py`、`contract.py`、`run_version.py`。

---

## 8. 折表、版本与边界纪律（v2 discipline）

- 折表由同一参数化函数生成，stage15 与 merge 共用（`folds.py` 待建），不得复制。
- 版本记录带内容哈希（池、门控、折表、网格、评估规则）；评估脚本读取哈希，不一致即报错。
- **`boundary.py`（§1.4）是门控/池/诊断线的数据读取入口；个股进化线不经过它**
  （`run_search.py`/`fitness.py`/`evaluator.py` 均不 import，`evaluator.py:57` 直接
  `pd.read_parquet`）→ **个股线的数据接触范围当前不可审计**（§8.3、PLAN §20.7-1）。
- **`windows.py`（§1.5）是门控线的窗口事实源；不是全项目单一事实源**
  （个股线窗口硬编码于 `fitness.py:29-31`）。
- 证据等级三级：**E1 探索、E2 滚动回测、E3 前向确认**。只有 E3 可用于"确认有效"。
  每份输出头部标注等级与数据边界。**状态术语见 §0.3.1**（`qualified_stat` ≠ `E3 confirmed`）。
- **选择窗与监控段分离**：一切"做决定"的步骤只用 `≤2025-12-31`；2026-01-05~2026-09-11
  只作衰减监控（`E1-diagnostic`），**不得回流进任何选择环节**。
  **但该纪律目前只对门控线生效**（`windows.check_window` 未被个股线调用），
  且**该监控段 154/169 天已被个股线 holdout 消耗**（§8.3）——
  **"监控段"这一角色本身已部分失效，须按 PLAN §20.6 建档。**

### 8.1 三种评估级别（不得混用）

| 级别 | 规模与配置 | 产物用途 | 明确不能做 |
|---|---|---|---|
| **Smoke** | 小样（缩小因子数/置换次数/窗口），`smoke_pipeline.py` | 验证 CLI 串联、数据接口、产物、状态机、错误处理 | **不作统计结论**；不得暗示预测能力 |
| **Full research evaluation** | 全量面板、**冻结口径**、完整置换/bootstrap | 统计判定（H2/G2-G3/F1/FDR） | 不消除候选选择偏差；不等于 E3 |
| **Forward confirmation** | 规则/模型/判定阈值**全部冻结后**，用未参与选择的确认数据 | E3 | 一旦用其结果指导新版本选择，该区间即失去独立性 |

**两条纪律**：

1. **小样通过 ≠ 全量性能验收**。全量运行还须检查：峰值内存、临时文件空间、
   异常中断恢复、重复执行幂等性、并行任务是否改变统计结果。
2. **全量评估通过 ≠ 证据等级自动升级**。完整 F1 + bootstrap + placebo 只说明候选
   在当前数据与检验设计下通过条件，**不会消除**"先搜索几十个候选再选唯一幸存者"的选择偏差
   （本项目实测为 Case B，见 PLAN §9.6.2）。

### 8.2 契约与推断目标（指向 PLAN）

| 主题 | 规格位置 |
|---|---|
| 跨模块接口契约（三分哈希 / 四时间字段 / 阻断规则） | **PLAN §18**（草案，只写规格不实现代码） |
| FDR 推断目标与数据独立性约束（Case B 已确认） | **PLAN §9.6**（草案） |
| 目标一致性诊断（H2 vs `y_prod`，三层验证） | **PLAN §19**（草案，先冻结设计再执行） |
| 数据接触审计与窗口治理 | **PLAN §20** + `results/merge_v2/data_contact_audit_2026-10-11.md` |
| 实施清单四态矩阵 | **PLAN §17.9** |

### 8.3 两条线是**两套窗口、两套纪律**（2026-10-11 审计确认）

> 审计报告：`results/merge_v2/data_contact_audit_2026-10-11.md`。本节是**事实澄清**，
> 更正此前"`windows.py` 是窗口单一事实源"的表述。

| | 个股进化搜索线 | 门控 / 池 / 诊断线 |
|---|---|---|
| 权威定义 | `fitness.py:29-31` **硬编码** | `windows.py` |
| 数据窗 | **2021-09-01 ~ 2026-08-21** | **2021-06-01 ~ 2025-12-31** |
| 内部 holdout | **2025-07-01 ~ 2026-08-21**（280 交易日） | 无 |
| holdout 用途 | **进 `rank_score`**（`min(train,holdout)`）→ 影响精英保留与撒种 | — |
| 一级（快评）窗 | **2024-08-21 ~ 2026-08-21**（抽样 1800 只，**58% 在 holdout 内**） | 不抽样 |
| 二级（全量）窗 | 2021-09-01 ~ 2026-08-21（全 5126 只） | — |
| 读取审计 | **无**（不 import `boundary.py`） | 有（`boundary_log.jsonl`，`read_labels ≤2025-12-31`） |
| 版本控制 | **无**（`git ls-files pandaai_online` = 0） | 无 |
| 台账覆盖 | `ledger.jsonl` 仅 **2026-09-15 ~ 2026-10-03（10 天）** | — |

**决定性证据**：`ledger.jsonl` 的 6 条 `holdout` 事件的 `monthly_excess`
覆盖 **2025-07 ~ 2026-07（13 个月）** → 进化搜索的 holdout **实际评估了 2026 年数据**，
且月度结果回写台账并进入 `decision` 事件。

**后果（数据段接触矩阵）**：

- 衰减监控段 `2026-01-05 ~ 2026-09-11` 共 169 交易日，
  **154 天（91%）已被进化搜索 holdout 消耗**；
- 疑似未触碰仅 **2026-08-22 ~ 2026-09-11（15 天）**；
- E3 前向区（2026-09-21 起）**当前可用数据 = 0 交易日**。

> **因此不存在可用的历史独立确认区间。** 任何"2026 可作 E3"的说法必须先核对本矩阵。

**治理规则（PLAN §20.6，**已于 2026-10-11 批准**；本节为记录，不含独立代码）**：
搜索不得触及 `date ≥ 2026-01-05`；holdout 只能在搜索池内部划分且必须声明
是"搜索内验证集"还是"独立确认集"；个股线必须接入读取审计与版本控制；
**先审计、后动代码**，不改 `fitness.py` 的 `START`/`END`/`HOLDOUT_START`。

### 8.4 结论等级与"已确认"的边界（回应三方审查）

| 产物 | 可声称 | 不可声称 |
|---|---|---|
| 门控 F1（`GM_limit_ratio_ratio` 的 `7/20`） | 同批数据的**探索性**结果 | "已确认有效的交互"、训练池准入依据 |
| G2/G3 通过 | 通过指定统计门槛（`qualified_stat`） | `E3 confirmed` |
| 进化搜索 winner（`base20`） | 历史搜索结果（E1/E2） | 干净 OOS 表现 |
| 冒烟 14/14 | **工程证据**（管道/接口/台账/错误捕获） | 统计或经济证据 |
| 2026 段任一统计量 | 诊断参考 | 独立确认（该段 91% 已被消耗） |

### 8.5 指纹数据截止日（`data_cut`）——候选身份 = 公式 × 数据截止日

> 2026-10-11 用户指令 + 批准。规格见 **PLAN §20.6.4**。

**动机**：2026 年数据在架构探索期被用于搜索（`fitness.END=2026-08-21`，holdout 实测
评估至 2026-07），这些历史指纹既不能被当成"与新搜索同口径"，也不能一删了之。
解法：把**数据截止日提升为候选身份的一部分**。

| 项 | 值/位置 |
|---|---|
| 默认截止日 | `windows.DEFAULT_DATA_CUT = FACTOR_SCREEN_MAX = 2025-12-31` |
| 遗留截止日 | `windows.LEGACY_DATA_CUT = "2026-08-21"` |
| 规范化 | `windows.normalize_data_cut()`（`auto`/`default`/`legacy`/ISO） |
| 守门 | `windows.check_data_cut()`：`≥ DECAY_MONITOR_START` 即拒绝；CLI 逃生门 `--allow-2026-cut` |
| CLI | `run_search.py --data-cut`（全局）、`--use-legacy-evidence` |
| 记录 | `ledger.log/log_raw` 写 `data_cut`；`Ledger.data_cut` 为默认值 |
| 摘要 | sidecar `cuts` / `tested_pairs`（三元组 `cut␀cid␀lab`）/ `vfps`（`cut\|vfp`） |
| 查询 | `hashes_for(labels, data_cut=)` / `vfps_for()` / `evidence_pool_for()` / `count_tested(data_cut=)` |
| 迁移 | `migrate_data_cut.py`（`--apply`；备份 + 校验 + 删 sidecar 强制重建） |
| 验收 | `selftest_cut.py` **10/10** |

**两条不变量（回归测试 T3/T4/T5 锁定）**：

1. `hashes_for()` 默认只返回**当前截止日**下的已测公式 → 遗留记录不参与新搜索的排除集；
2. `--data-cut` **同时是评估窗上界** → `DataContext(end=)`/`build_fast_frame(end=)` 绑定它，
   **指纹记的截止日 = 实际用到的数据上界**（否则记录说谎）。

**实测（最小真实搜索 seed 993）**：快评 `n_periods=42`×3=126 天 = 2025-07-01~2025-12-31；
全量 `350`×3=1050 天 = 2021-09-01~2025-12-31；记录 `data_cut=2025-12-31`；遗留截止日已测=0。

**副作用（正确语义，非 bug）**：截止日变更后排除集/腿库/精英回流**默认为空**——
此前没有任何因子是在 2025-12-31 窗下评估过的 → 各机制轴回到未测状态、搜索冷启动。
`--use-legacy-evidence` 可复用旧 winner（默认关闭，因其带旧选择偏差）。

**裁决（2026-10-11）**：用户更正原则为「**holdout 只在 `cut_date` 之前**，
切点可落在 2025 年内」→ 当前默认（`cut=2025-07-01 < data_cut=2025-12-31`）**即符合**，
无需改动。`SearchRun` 的 `--cut < --data-cut` 检查正是该原则的形式化。详见 **PLAN §20.10**。
两条限定：① 该检查只证明**日期关系**合法，不是数据隔离的充分证明（见 §8.6）；
② 2025 不因此成为独立确认集——它是参与 `rank_score` 的内部 holdout。

### 8.6 数据截止日边界审计（`audit_cut_paths.py`，**9/0 通过**）

> 触发（Codex）：「`cut < data_cut` 只证明日期关系合法，不能证明所有数据访问路径都遵守边界」。
> 规格见 **PLAN §20.12**。

**修出的 4 个真实越界点**（都含 2026 数据、都在搜索路径上）：

| # | 位置 | 后果 | 处置 |
|---|---|---|---|
| 1 | `pool_legacy.WINDOW_END` / `ALIGNED_END = '2026-07-10'` | `--greedy-top`（ΔComb）与 `refresh_member_sia` 用含 2026 的池回测窗 | `windows.cut_bound` 裁剪 |
| 2 | `dedup.monthly_spearman(end='2026-07-10')` | 池相关窗含 2026 | 新增 `data_cut` 参数（默认安全） |
| 3 | `pool_sim.build_panel_daily` 硬编码 `'2023-03-01'~'2026-07-10'` | ΔComb 宇宙构造 | 窗口参数化，调用方必须传 |
| 4 | **`export`/`upload --end` 默认 `'20260821'`** | **传给平台 `batch.py` 的回测窗** → 平台指标含 2026 → 污染 `platform` 事件与 `calibration.json` | 默认改为跟随 `--data-cut` |

**新工具**：

- `windows.cut_bound(default_end, data_cut)` — 把遗留硬编码上界裁到截止日；
  `CUT_OFF`（`'off'`）显式关闭（历史重放/单测）。`DATA_CUT_MIN` 放宽为 `2021-06-01`。
- `audit_cut_paths.py` — 静态 A1（2026+ 代码字面量扫描，排除注释/文档串）+ A2（前视算子）
  + A3（标签位移）；动态 B1–B6（`DataContext` / `build_fast_frame` / `monthly_spearman` /
  `pool_legacy` / `pool_sim` / `ledger` 隔离）。**用法：`python audit_cut_paths.py`（exit 0 = 通过）**。

**诚实标注（不可声称"已完全隔离"）**：

- `build_label_series` 的 `shift(-n)` 是**标签定义本身**——按"按信号日设界、标签自然前瞻"
  的既有约定，`label_available_max` 须随记录落盘；
- `ensure_local_fields` 已核查**无前视算子**（同日截面 + 60 日滚动，全部后视）；
- `build_extra_panel` 读宽经 `ctx` 行左连接后使用 → **有效使用受界**，属数据最小化瑕疵；
- 旧台账 15,329 条的 `data_cut` 是**事后回填**（`data_cut_legacy: true`），非当时元数据。

**G3/G4/S6 改为显式 SKIP**：`amihud120`/`highpos120` 已于 2026-10-03 出池，
这些期望值是遗留池/遗留窗读数。实测归因：S6 在 `CUT_OFF`（本轮改动前的行为）下
base=0.2058、`selected=[]`，**同样不符 0.1864** ⇒ **数据漂移**，与本轮改动无关。
`selftest_phase_a` 现输出 `G1:PASS | G2:PASS | G3:SKIP | G4:SKIP | G5:PASS`（exit 0）。

### 8.7 两层问题必须分开（Codex 提醒）

| 层次 | 问题 | 状态 |
|---|---|---|
| **身份隔离** | 不同截止日的候选身份不同；旧结果不混入新搜索 | **已实施 + 审计通过** |
| **统计有效性** | 新搜索的快评/holdout/晋级规则能否找到稳定有效的因子 | **未验证**（见 PLAN §20.14） |

**指纹机制不能证明统计有效性。** 个股线快层（抽样 1800 + 24 月尾窗）**从未做过精度基准**，
而门控线的抽样路线已被实测否决（§3.11）——两者统计量与淘汰规则不同，结论不可搬用。
须做 PLAN §20.14 的**四组对照**（A 抽样+24月 / B 全量+24月 / C 抽样+完整窗 / D 全量+完整窗）
+ 五类指标（重点：**错误淘汰率**）+ **单独**测量 `--full-cap 60` 截断损失。
**基准完成前不得修改快评阈值、`--full-cap` 或生产筛选流程。**

**用户裁定（2026-10-11）**：不再做误差来源归因，判据收敛为**「遗漏的优秀因子数量」**。

**n=100 / cap=60 读数（已执行）**：A（现行产线快评）**top-10 漏 0、top-20 漏 1、top-60 漏 5**
（recall@60 = 91.7%；对照 B 4/60、C 3/60）。原因拆解：**阈值淘汰 4 个 / 截断淘汰 1 个**——
唯一的截断淘汰是 D#59（A_sia 0.00409 已过阈值，但在 keep 集内排第 **63**，被 `cap=60` 截掉）。
5 个遗漏全部是 A **低估** D ⇒ 误差方向**保守**（偏向淘汰，不偏向放行）。

**基准自身 bug 已修**：`fast_kept_mask` 原为「keep 集 ∩ 全局前 cap」，产线是
「对 keep 集按分数排序取前 cap」（`run_search.py:825-838`）；修正后经 **200 组随机配置**验证
**逐位等价** ⇒ **n=40 旧读数作废**。

详见 `results/merge_v2/bench_fast_findings_2026-10-11.md` + PLAN §20.14。
**未做任何生产参数改动。**

### 8.8 平台先验作用域（`calibration.json` / `--platform-prior`）

**两者都是纯 c2c 口径，vwap 路径零依赖 ⇒ 不需要隔离**（PLAN §20.15）。

- `cmd_calibrate`（`run_search.py:1821`）**显式**只配对 `label in (None, 'c2c')`；
- `--platform-prior` 是 opt-in（默认 `False`），只乘进 **c2c 的 `rank_score`**（`575-583`）；
  快评的 `rank_score`（`509`）不传该参数；
- vwap 榜按 `vwap_*` 排序（`1581-1584, 1633-1642`），其 `score` tie-breaker 由 `_rescore_rows`
  （`1492`）从**本地 c2c** 指标派生；
- `platform` 事件 `label=None`，不进 `['vwap']` 排除集、不进证据池、不进 UCB（`_ingest` 只收 `fast/local`）。

**既有 scoped limitation**：`platform` 事件与 `calibration.json` 的历史读数是在**含 2026 的窗**上
算出的（§8.6 第 4 项）。只影响 c2c 侧 opt-in 先验；**vwap 侧不受影响**。
将来若要把 c2c 先验纳入决策，须先「标记 → 隔离 → 再评估重标定」。
### 8.9 个股线快评的抽样范围与比例（2026-10-11 实测）

> 规格见 **PLAN §20.16**。回答「抽样是不是 2021~2025 的数据？比例多少？」

**抽样是截面（股票）抽样，时间上不抽样。**

| 维度 | 配置 | 实测 |
|---|---|---|
| 截面（股票） | `--fast-stocks 2500` | **2500 / 5126 = 48.77%** |
| 时间 | 不抽样；窗 = `data_cut − --fast-months(26)` | **2023-10-31 ~ 2025-12-31（529 交易日）** |

**⇒ 快评窗不是 2021~2025；2021~2025 是「全量评估窗」**（2021-09-01 ~ 2025-12-31，1050 交易日）。

**两个维度叠加后的真实压缩率**：

| 窗 | 全量行数 | 抽样行数 | 压缩率 |
|---|---|---|---|
| 全量窗 | 5,112,506 | 2,495,883 | 48.82% |
| 快评窗 | 2,683,517 | 1,307,461 | 48.72% |
| **快评 vs 全量（实际被评估的行）** | **5,112,506** | **1,307,461** | **25.57%** |

**⇒ 快评实际只算全量评估的 25.57% 行数（≈3.9× 压缩）**——这就是「抽样换速度」的完整量化。

> **配置变更（2026-10-11 用户指令）**：`--fast-stocks` 1800 → **2500**、`--fast-months` 24 → **26**。
> 旧配置实测：抽样 35.12%、窗 2024-01-02~2025-12-31（485 日）、压缩 **16.90%**（≈5.9×）。
> **§8.7 引用的 top-60 遗漏 5/60 读数对应旧配置**，新配置未重测（用户指示）。

其他实测事实：

- **抽样总体 = 全量未裁面板的股票池（5126 只）**，非快评窗内存在的股票；
  实测 1800 只抽样股票**全部**在快评窗与全量窗内有数据 → **有效样本 = 名义样本 = 1800**。
- `build_fast_frame` 实际返回 **1,381,835 行 / 1800 只 / 2022-09-30 ~ 2025-12-31**；
  多出的 **15 个月是 warm-up 垫底期**（`warmup_months=15`），只供滚动算子取历史，
  由 `DataContext(start=2024-01-02)` 在评估时裁掉，**不参与任何统计量**。
- 抽样**确定性可复现**：`seed=11` + `random.Random` + `pick_stocks`（2026-10-11 从
  `build_fast_frame` 抽出，抽股集合逐只不变）；`--fast-stratify size` 可按市值五分位分层（默认关）。

### 8.10 两条线的面板不是同一张（但重叠区完全一致）

| | 个股线（搜索） | 门控线 |
|---|---|---|
| 文件 | `cache/stock_features_20200101_20260918.parquet` | `factor_lab_data/factors.parquet` |
| 行数 | **7,555,041** | **6,272,523** |
| 日期 | **2020-01-02 ~ 2026-09-18**（1629 交易日） | **2021-06-01 ~ 2026-09-18**（1289 交易日） |
| 股票 | 5126 | 5126 |
| 读取入口 | `evaluator.load_panel()` → `cache_utils.find_stock_features()` | `boundary.read_factors()` |

**重叠区（2021-06-01 ~ 2026-09-18）逐项对账（实测）**：行数差 **0**；
`(stock_code, date)` 键集合**完全相同**（双方独有键各 0）；交易日历**相同**（1289 日）；
股票池**相同**（5126）。

⇒ **无口径漂移风险**；差异仅为 cache 多出的 2020-01-02 ~ 2021-05-31 前段
（1,282,518 行，被 `fitness.START=2021-09-01` 裁掉）。
仍记录一处**起窗差异**：门控线自 `FACTORS_FIRST_SIGNAL_DATE=2021-06-01` 起，
个股线自 `START=2021-09-01` 起 ⇒ `2021-06-01 ~ 2021-08-31`（65 交易日）**只有门控线用**。

> 更正：此前审计报告与 PLAN §20.8 冻结快照中的「面板 6,272,523 行」描述的是 `factors.parquet`，
> **不是搜索实际读的那张**。搜索读的是 cache（7,555,041 行）。

### 8.11 cache 文件名硬编码已消除（2026-10-11 修复）

**问题**：`cache_utils` 的模块契约明写「文件名带日期后缀且会被管线重生成，**禁止硬编码**」，
但实测有 **3 处**硬编码 `stock_features_20200101_20260918.parquet`：

| 位置 | 原写法 |
|---|---|
| `boundary.py:read_cache` | `os.path.join(ROOT, "cache", "stock_features_20200101_20260918.parquet")` |
| `gates.py:23` | 同上 |
| `gate_search_space.py:47` | 同上 |

而个股线 `evaluator.load_panel()` 用的是**自动发现**（`cache_utils.find_stock_features`，
规则 = **开始最早 / 期间最长 / 并列取 mtime 最新**）。

**风险（静默分裂）**：今天两者解析到同一文件（`..._20200101_20260918`），
但一旦管线生成**同起始、更晚结束**的新缓存（`cache/` 下已存在多个更晚结束的候选文件，
如 `..._20221201_20261009.parquet`），自动发现会切换而硬编码处不会
→ **个股线与门控线静默读到不同面板**，正是本项目反复出现的「两套」问题。

**修复**：3 处统一改为 `find_stock_features()`（`boundary.py` / `gates.py` 加 `import _paths`，
`gate_search_space.py` 复用 `gates.CACHE`）。

**验证（行为中性）**：修复后四处解析结果**完全一致**，仍为 `..._20200101_20260918`；
`boundary.read_cache` 实测 `file_max_date=2026-09-18`，读到的行数与边界过滤正确。

### 8.12 数据版本：**只作溯源，不作指纹作用键**（2026-10-11 用户裁定）

> **本节在本日内被推翻过一次，保留结论与理由以免重蹈。**
> 上一版曾把「数据版本」加进个股线指纹作用键（`formula × cut_date × data_version`），
> 理由是「数据重生成后同一 (公式, cut_date) 的数值可能已变」。
> **用户否决**：cache 是**日频重生成**的 ⇒ 每个交易日全部指纹失效，「已测过」机制失去意义。

#### 8.12.1 为什么不能把内容哈希当作用键（实测依据）

| 文件 | 实测更新方式 |
|---|---|
| `cache/stock_features_*` | **日频重生成**：`..._20221201_20261008.parquet`（mtime 10-09 00:13）→ `..._20261009.parquet`（10-09 22:15）；另有逐日小文件 `..._MMDD_MMDD.parquet` |
| `factors_normal_1008/1009.parquet` | 10-09 01:03 / 23:08 ⇒ **factors 也有日频变体** |
| `factors.parquet`（门控线读的） | 稳定文件名，间隔更新（mtime 10-05） |

**若用内容哈希作作用键**：日频更新 → 文件名/内容变 → 全部指纹失效 → 每天重测全量。
而日频更新对**窗内（≤ `cut_date`）**历史数据的影响极小（追加的是窗后数据；
比率型量（`pct_chg`、比率型因子、比率型标签）对复权不敏感）。

⇒ **指纹的作用键只有 `cut_date`。**

#### 8.12.2 三层分离（最终设计）

| 层 | 内容 | 行为 |
|---|---|---|
| **① 作用键（指纹）** | `(公式, cut_date)` | **唯一**决定「已测过 · 跳过」。**不含数据版本** |
| **② 溯源记录** | `data_version`（声明式）写进每条事件 | **不过滤**；供审计回答「这个数字出自哪个数据世代」 |
| **③ 显式换代杠杆** | 个股：`run_search --retest-before <ISO>`；门控：`data_version.py --refresh` | **只由人主动触发**；日常更新不动 |

#### 8.12.3 两类更新的正确处理

| 情形 | 该做什么 |
|---|---|
| **日常日频更新**（追加新交易日、个别修订） | **什么都不做**。指纹只用 `--data-cut`，不会失效 |
| **实质变化**（批量复权重算 / 数据源更换 / 修数据） | ① 门控：`python data_version.py --refresh --kind factors --note "原因"` ⇒ 声明新版本 ⇒ 门控 caliber 变化 ⇒ 自动补测；② 个股：`run_search --retest-before <变化日>` ⇒ 只把「最后事件早于该时刻」的公式重新纳入（不是全量重测） |

#### 8.12.4 声明式版本（`results/merge_v2/data_version.json`）

**门控线的 `data_version` 必须是声明式，不是内容哈希**——否则日频更新会让门控 caliber
每日失效、40 个门控全部重跑。故：

- 源：`windows.declared_data_version(kind)` 读 `results/merge_v2/data_version.json`；
- 只有 `python data_version.py --refresh --kind <factors|panel>` 才更新它；
- 原先散在 3 个文件的字面量 `"factors_abe94a59"` 已收拢到该文件（`boundary.factor_data_version()`）；
- **bootstrap 值与既有字面量一致**：`factors` = `factors_abe94a59`、`panel` = `panel_c2ece310` ⇒ **零 caliber 失效**。

（`windows.file_data_version` 保留为**工具**：只在 `--refresh` 或 `--inspect` 时计算内容哈希，
带 `(size, mtime_ns)` 缓存；实测首次 12.0s/1.4s，命中 0.014s。）

#### 8.12.5 验收

- `selftest_cut.py` **13/13**：
  **T11a** 生产口径不按版本过滤（三个版本都在排除集里）；
  **T11b** 审计口径仍可按版本追溯；
  **T11c** `retest_before` 未来时刻 ⇒ 全部重新纳入（换代杠杆生效）；
- `audit_cut_paths.py` **9/0**；门控冒烟 **14/14，exit=0**（`factors_abe94a59` 未变，无 caliber 失效）。

#### 8.12.6 残余（诚实标注）

- 门控 caliber 的 `data_version` 覆盖 `factors.parquet`；构造门控序列 `g` 用的 **cache**、
  以及 `labels_prod.parquet` **未纳入**该字段。按本节的设计，这**不是缺陷**——
  若需要，应在**实质变化**时用 `--refresh` 显式声明，而不是让它随日频更新自动变。
- 个股线的 `data_version` 只是溯源标签：若数据实质性变化而用户未执行 `--retest-before`，
  旧结论会被沿用。**这是有意的取舍**（日常稳定性优先），代价是依赖人的判断。

### 8.13 数据代际管理操作规程（SOP）【可执行】

> §8.12 说明**为什么**这样设计；本节说明**遇到数据更新时具体该敲什么命令**。
> 目标：**日常更新零动作**，**实质换代一次显式动作**，且两者都不会误伤指纹。

#### 8.13.1 指纹作用键总表（唯一口径，改动即需批准）

| 线 | 「已测过」由什么决定 | 台账字段 | 代码位置 |
|---|---|---|---|
| **个股** | `(表达式哈希, data_cut)` | `data_cut` | `ledger.hashes_for(labels, data_cut=)` （`run_search` 传 `self.data_cut`；`vfps_for` / `evidence_pool_for` 同口径） |
| **门控** | `(candidate_fp / family_fp, caliber_key)`；`caliber_key = sha1(window_end ‖ form ‖ factor_set ‖ label ‖ data_version ‖ n_perm ‖ n_boot)` | `caliber` / `caliber_key` | `gate_ledger.CALIBER_FIELDS` |

**共同点**：两线的口径都**不含文件内容哈希**。个股只吃 `data_cut`；门控吃 `window_end` 与**声明式** `data_version`。

#### 8.13.2 内容哈希（cache md5）的定位

`windows.file_data_version(path, kind)` → `f"{kind}_{md5(内容)[:8]}"`。
带 `(size, mtime_ns)` 缓存（`results/merge_v2/data_version_cache.json`）；实测首次 12.0s（factors 7.3GB）/ 1.4s（cache 679MB），**命中 0.014s**。

**它不参与任何过滤**——既不是个股指纹的一部分，也不是门控 caliber 的直接来源。它只有三个用途：

1. `data_version.py --refresh` 时**计算**并写入声明文件（换代动作的数据来源）；
2. `data_version.py --inspect` 时**对照**（看当前文件与声明是否已不同）；
3. 作为事件里的 `data_version` 字段值，供审计回答「这个数字出自哪个数据世代」。

⇒ **看到 md5 变了 ≠ 指纹失效。** 只有跑过 `--refresh` 才会改变门控 caliber。

#### 8.13.3 情形判定（先看这张表）

| 情形 | 典型表现 | 动作 |
|---|---|---|
| **日常日频更新** | `cache/stock_features_*` 出现新日期后缀文件；追加了新交易日；个别股票补数据 | **① 什么都不做**（指纹只用 `data_cut`，不受影响） |
| **实质变化** | 批量复权因子重算（历史价格整体缩放）；行情数据源更换；大面积修数据（停牌标记、涨跌停标记口径变更）；`factors.parquet` 被重新生成 | **② 走 §8.13.4 的换代流程**（门控 refresh + 个股 retest-before） |
| **不确定** | 只看到文件变大/变小 | 先跑 `python data_version.py --inspect` 看内容哈希，再判断是否需要换代 |

**判据要点**：问自己「**窗内（≤ `data_cut`）的历史数值会不会变**」。
只追加窗后数据 → 不会变 → 不用动作；窗内被重算 → 会变 → 换代。

#### 8.13.4 换代流程（两步，顺序建议：先门控后个股）

**第 1 步：门控线 —— 声明新版本，触发自动补测**

```powershell
# 变更前先看当前声明（可选）
python data_version.py

# 门控数据世代刷新（factors.parquet 是门控的个股因子来源）
python data_version.py --refresh --kind factors --note "2026-11 批量复权重算"
```

**预期输出**：`{"kind":"factors","old":"factors_abe94a59","new":"factors_xxxxxxxx","changed":true,...}` → 若 `changed=true` 会打印提醒：门控 caliber 变化 ⇒ 依赖该 caliber 的
`gate_g2g3` / `f1_batch` / `confirm_candidate` 记录判为 **stale**，`run_gate_pipeline.py --all`
会自动重跑（只跑 stale，不全量）。

```powershell
# 触发补测（只跑 pending/stale；产物在 results/merge_v2/gates_search/）
python run_gate_pipeline.py --all
```

> 若变更同时影响门控序列 `g` 的构造（它读 **cache**），请**一并**刷新 panel：
> `python data_version.py --refresh --kind panel --note "..."`。
> 注意：`panel` 目前**未**进入门控 caliber（§8.12.6），所以它只改溯源标签；
> 要让门控也感知 cache 换代，需把 panel 版本并入 caliber——**属口径变更，须先批准**。

**第 2 步：个股线 —— 按时间点重测，不做全量重测**

```powershell
# 只把「最后事件早于 <变化日>」的公式重新纳入候选（≈ 强制重测换代前测过的那些）
python run_search.py search --retest-before 2026-11-01 --n 40 --seed 1 ...
```

**生效范围**：`ledger.hashes_for(..., retest_before=<ISO>)` 逐 hash 比较 `last_at`（= 该公式最后一次 fast/local 事件的 `created_at`）。`last_at >= retest_before` 的**仍被排除**（它们已是新数据下测过的）；早于的**重新纳入**。

**与 `--no-exclude` 的区别**（不要混用）：

| 杠杆 | 范围 | 适用 |
|---|---|---|
| `--retest-before <ISO>` | 只重测「最后事件早于该时刻」的公式 | **换代**（推荐） |
| `--no-exclude` | **全部**历史公式重测（排除集清空） | 极端情形 / 全量回归对照 |

**第 3 步：换代后固化证据**

```powershell
# 冒烟闸门（14 项；确认换代后流程仍通）
python smoke_pipeline.py
# 指纹作用键回归（13 项；含 T11a/b/c 换代杠杆语义）
python selftest_cut.py
```

然后更新 `results/merge_v2/PROGRESS.md` 与 `data_ledger.md`（记录换了什么、为什么、影响范围）。

#### 8.13.5 日常更新的 1 分钟自检（确认「什么都不用做」）

```powershell
python data_version.py            # 看声明值；与上次一致即无需动作
python data_version.py --inspect  # 看当前文件内容哈希（仅供对照，不是作用键）
```

- 声明值**不变** ⇒ 门控 caliber 不变 ⇒ 无补测；
- `--inspect` 显示内容哈希变了**但你没做实质改动** ⇒ **正常**（日频追加），**不要**跑 `--refresh`；
- 只有当你**确实**做了换代改动时，才执行 §8.13.4。

#### 8.13.6 回滚

声明文件是纯 JSON，可手工改回：`results/merge_v2/data_version.json` 里把 `factors` 改回旧值
⇒ 门控 caliber 恢复 ⇒ 旧判定重新可用（台账未删，`gate_ledger.jsonl` append-only）。
个股侧无状态（`--retest-before` 只是当次的排除集参数），**不需要回滚**。

#### 8.13.7 三条不得违反的约束

1. **不得**把内容哈希（cache md5）加进任何指纹作用键（日频更新会让指纹每日失效）。
2. **不得**在无实质变化时跑 `data_version.py --refresh`（会无谓触发门控全量补测）。
3. **不得**用 `--no-exclude` 代替 `--retest-before` 做换代（成本差一个量级）。

### 8.14 门控搜索空间冗余诊断与去重判据修正（2026-10-12）

**背景**：用户提出"看上去在搜索，其实搜出来是同一个东西的不同表达"。
Codex 裁决：先核查候选的公式与实际生成值，能确定重复的直接改；不建新框架。

**结论：无可确认的重复，搜索空间不变。** 三层可用判据（`--data-cut 2025-12-31`）：

| 层 | 判据 | 结果 |
|---|---|---|
| 基础量定义（10 个） | raw Spearman | max **0.898**，无 ≥0.9 ⇒ 不重叠 |
| 操作化（4 个 op） | raw Spearman | 中位 **0.114** ⇒ 不冗余（唯一部分重叠 `level`–`ratio` 中位 0.761，共享 mean20 分子，但 ≠1 不删） |
| **门控实际状态（40 个 p）** | p 序列 Spearman | 中位 **0.066**，仅 **3/780 ≥0.8** ⇒ **不冗余** |

**效应向量口径（ΔIC 相关）不可用**：20 个因子自身共线（因子间 \|ρ\| 中位 **0.825**）
⇒ 门控效应向量部分单维（12/36 与共同方向 \|r\|≥0.9），
"132/630 对 ≥0.8" 是**构造必然**，不是冗余证据。白化后结论随正则化翻转
（0/630 ↔ 246/630，奇异谱条件数 **1391**）⇒ 亦不可用。

**修正的去重判据（2 处，仅改判据，候选集零变化）**：

1. `gate_ledger.gate_order_fp()`（新增）：门控序列的**排序指纹** ——
   直接对原值取秩并哈希。单调变换 ⇔ 同秩 ⇔ 同一门控。
   **不做镜像规范化**（`p` 与 `1−p` 激活方向相反，不是同一门控）；
   **不做量化**（按有效数字量化会破坏秩不变性，实测 FAIL）。
2. `gate_ledger.structure_check()`（新增）：输入结构自检（n/uniq/std），
   `ok=False` 时调用方**必须拒绝输出结论**。
3. `run_gate_pipeline.py` S2 第 ⑥ 层：`gate_value_fp` → `gate_order_fp`
   （verdict `reject_vfp` → `reject_order_dup`）；新增第 ⑧ 层 `reject_structure`。

**为何旧判据失效**（回答"现有去重为何没拦住"）：

- `evaluator.value_fingerprint` 只取**边际分布**的 9 个分位点，为**截面因子值**设计
  （对截面因子，仿射等价 ⇔ 同一排序 ⇔ 同一因子）。
- 门控 p 是**时序**对象："何时开启"由时序对齐决定，**边际相同 ≠ 同一门控**。
- 实测：行业门控 p = 逐日跨行业 `rank(pct=True)` → 边际恒为 [0,1] 均匀
  （std 恰 = 1/√12 = 0.2887，仅 131 个取值）→ **14 个门控只产出 2 个指纹**。
  若用作拒绝判据会**误拒 12/14**。
- 该缺陷此前**潜伏**：市场门控的滚动分位 p 边际各不相同（未触发），
  行业门控不走本流水线（`gate_g2g3_sector.py` 不 import `value_fingerprint`）。
  **一旦行业门控接入统一流水线会立刻误拒 12/14。**
- 原注释称"与个股线 `run_search.evaluate_fast` 的 vfp 语义一致"——**错误**，已修正。

**验收**：T1 合成夹具（线性/非线性单调变换同指纹；反向/独立/微扰不同指纹）PASS；
T2 结构自检 PASS；T3 行业 14/14 唯一（旧判据 2）PASS；T4 市场 40/40 唯一 PASS；
T5 性能 0.1ms/14 万点；**T6 流水线回归：`pipeline_all.csv` 40 行、
短名单 3 行（`GM_turn_med_accel`/`GM_amt_hhi_chg`/`GM_limit_ratio_ratio`，与修复前逐字一致）、
neutral 36/40 与 `screen.csv` 对账一致、台账 40 pass / 0 reject ⇒ 零假阳性。**

**报告**：`results/merge_v2/gate_space_redundancy_audit_2026-10-12.md`（含附录 A 的 3 次仪器陷阱）。

#### 8.14.1 诊断工具的强制前置（防复现）

本项目已发生 **3 次**"程序正常输出、结果无意义"：

| # | 陷阱 | 表现 | 已发生 |
|---|---|---|---|
| 1 | 把 `(date, sector)` 按日期均值压平 | std=0 / uniq=1 → 假"同指纹"、假"13 簇" | **2 次**（G4 v1、L2a 初测） |
| 2 | 用共线因子作效应基 | 假高相关（0.504） | 1 次（L3） |
| 3 | 未声明正则化的白化 | 结论随参数翻转 | 1 次（L3） |

⇒ **任何门控诊断工具必须以 `structure_check` 自检对象结构**（陷阱 1）；
陷阱 2/3 属**分析口径**，须由调用方**显式声明**（如"效应基已白化，λ=…"），
不能靠函数兜底。**未声明口径的诊断结果不得写入任何报告。**

### 8.15 搜索族预注册清单与 G4 口径裁决（2026-10-12，批次 2）

**族机制（F1v2）**：`hypothesis_registry.csv` 收录搜索族 F1 的 **720 个预注册候选**
（id HYP-0141~0860；HYP-0041~0140 为已删除旧门控世代的保留段；F2 的 40 行保留）。
`f1_family.py --from-registry --gen 2` 生成清单（family_id=**F1v2**，
members_hash=**f80a10ce6f0a**，spec_cap=720 = 36 eligible×20 的冻结设计规模）；
`judge_f1.py --family` 对输入**逐一对应校验**（缺/多/方向/重复/超 cap 即拒绝；
结果文件带 `family_id`/`family_status`/`family_members_hash` 血缘三列）。
**验收**：validate 720/720 ok；族约束重判=旧基准（h3_pass **15/720**、联合 FDR **0/720**）
⇒ 族机制接入判定不改变结论，只把 FDR 分母的来源从"输入文件"换成"预注册清单"（§9.6）。
FDR 族扩展须**换新世代清单**（不得沿用旧哈希，`--gen` 参数）。

**G4 口径裁决 = series（时序）**：接入 `run_gate_pipeline.py` S6 ⑤——评估后集合上计算，
**记录+告警不设硬门**（§17.5 纪律；台账 g4 事件 verdict=diagnostic）。
效应向量口径**剔除**（20 因子自身共线中位 \|ρ\|=0.825、84% 方差单中心；白化随正则化
0/630↔246/630 翻转）⇒ 仪表陷阱问题。两口径产物并存**不得混用**：
独立 CLI（eligible 36 → 34 簇）vs 管线（评估后 40 → 37 簇），后者带 `_pipeline` 后缀。
G4 只答独立性不答有效性。

**S7 状态机**：短名单候选台账记 `decided` 事件，`state=qualified_g2g3` ——
**≠ 入池**（须 F1 候选层 + 外部生产引擎对照）；运行报告 `pipeline_report.md` 含同语义。
生命周期（TECH §0.3.1）中 `qualified_stat`→`tested_increment`/`frozen_prod` 的
执行载体**已由批次 4 实现并获批**（`s8_state.py` 两条状态机，退出条件代码强制，
2026-10-12，PLAN §16.29）。

详细记录：PLAN §16.27；矩阵更新：PLAN §17.9 #5/#14/#16/#17。

### 8.16 批次 4：FeatureContract / S8 状态机 / Test B（2026-10-12）

> **2026-10-13 更新（用户裁定）**：`s8_test_b.py` **已删除**——"不把 LightGBM 模型级
> 增量诊断放进 merge skill"；`tested_increment` 的证据改为**外部生产引擎对照运行清单**
> （`main_alpha_stage15_hp_search.py --features_file` 双运行 + `compare_stage15.py`
> 对拍；清单格式见 `s8_state` docstring）。对象表/诊断值见 PLAN §16.29（历史记录）与
> §16.29.1（删除记录）。feature_definition 现 **4 哈希**（2026-10-13 拆出
> label_definition_hash；现行值见下方表格与 PLAN §16.29.1）。

三个新模块（全部为**实现代码**，详细记录 PLAN §16.29）：

| 模块 | 作用 | 现况 |
|---|---|---|
| `feature_contract.py` | **三层哈希**（feature_definition / dataset / experiment 互补不重复）+ 四时间字段；`--show/--out/--check`（不一致 exit=2 ⇒ 禁止静默复用旧产物）。2026-10-12 值：fd=`3a3a6decb69c` / ds=`2b752f583083` / exp=`34eb1a0b55c4` | 落盘 `contract_hashes.json` |
| `s8_state.py` | **两条状态机**（`s8_gate`：candidate→qualified_stat→tested_increment→frozen_prod；`s8_stock_pool`：…lifetime→E3_confirmed），退出条件**代码强制**：qualified_stat 需 F1 judged 行 h3∧FDR∧family registered∧hash=现行 F1v2；tested_increment 需 Test B 报告 mode=formal ∧ 五指标 ∧ experiment_hash=现行；冻结级必须 `--approve "<理由>"`；**个股线无 `pool_state.json` 一律代码级阻断**。事件写**独立** `s8_state_ledger.jsonl`（首次实测曾误写生产台账 1 条真实拒绝事件——留档） | **零合格（诚实）：门控线无任何满足 qualified_stat 的候选**（F1v2 联合 FDR 0/720） |
| `s8_test_b.py` | **Test B 模型级增量**：contract folds 外层 8 窗（18/3/step3/embargo5d，测试窗锚定 signal_max 往回布——实测修掉双重步长与 `'-31'` 非法月日两个 bug）；LightGBM 单配置（同预算）；基线 base20（baseline_pool_not_frozen）；候选特征 = `d·x·max(s·g,0)`（与 F1 同式）；**五项预规定指标** top1_return/n_trades/drawdown/turnover/fold_stability（Top1 逐日，r3=生产成交代理，报告明示未计滑点/涨跌停折算）；`--mode formal` **要求 state=qualified_stat**，diagnostic 报告被 s8_state 校验拒绝（机制实测成立） | diagnostic 实跑通过（2 候选 × 2 窗 × 双臂，137s；Top1 中位 基线 +0.00407 → 增强 −0.00792，**仅诊断记录不构成验收**）；formal 通道就绪、无合格候选故未运行 |

**与个股进化搜索的两段式对应**（2026-10-13 用户指定分工）：搜索 = `gate_search_space.py`
（生成空间与面板）；**筛选与排名 = 本脚本**（含 8 层资格筛选 → G2/G3 → S4 排名 → G4 →
S7 短名单；`--topn N` 可只输出排名前 N 的存活门控，0=全部通过筛选者）；逐个检验 =
`f1_batch.py` + `judge_f1.py --family`（族校验要求全族整体运行）；模型级增量 =
**仓库根生产脚本**（merge 不内嵌）。

**验收**：导入 74/74；FeatureContract 哈希落盘；状态机（①FDR 未过→拒 ②非法跳跃→拒）机制实
测成立；Test B diagnostic 实跑出五项指标。**仍待**：外部生产引擎对照（E1，须 qualified_stat
候选；Test B 接口已随裁定移除）、
S8_Stock_Pool 冻结流程（须 pool_state.json）。

### 8.17 集中待决清单（2026-10-13 建；每次裁决后更新本节）

> 目的："待批准"不再混着出现——每项标明类别（§A–E）、当前状态、阻断范围、
> 需要用户回答的问题、对应章节。**代码实现完成 ≠ 该事项获批/验收**。

**A. 文档滞后（已全部清零）**：曾有的 4 处过期"待批准"已于 2026-10-13 按事实修正
（§8.3 影子段、§8.9 引段、§8.12 正文、§8.15 状态机段）——今后"待批准"措辞出现前
须先对照本清单，防止再造。

**B. 研究规格待裁决（真裁决，阻断正式判定）**

| # | 事项（白话） | 代号 | 现状 | 阻断范围 | 需要你回答 | 章节 |
|---|---|---|---|---|---|---|
| B1 | **组合资格线用哪把尺**——**已裁决（2026-10-13，§16.30）：正式=斜率尺（form g，回冻结原文）；"极端日尺"（form t）降为诊断，旧 t 批次改名 `diag_*`；720 已按 g 重算重判（族 ok 720/720，h3 20/720，**联合 FDR 仍 0/720**，q_min=0.1279）⇒ 口径裁决不改"无合格候选"现实**；管线 caliber 顺带如实化（form=g + panel_sha 增量筛选兜底） | G2 口径 & form | 已关闭 | 无 | 无（本项关闭；参见 §16.30 与 `g2_caliber_adjudication_2026-10-13.md`） |

**C. 提案待处置（已按用户判据落定，2026-10-13）**

| # | 事项 | 处置 |
|---|---|---|
| C1 | 个股池路线图（`pool_design_v1.md`） | **部分实现、未整体批准**；§4 池冻结路线图 = S8_Stock_Pool 冻结流程的设计参考；Q-D3/Q-D4 待裁决（见 E 类） |
| C2 | 搜索架构统一（`search_arch_design.md`） | **部分被实现取代**；个股线薄编排层进 backlog（不阻塞——个股线当前可完整运行） |

**D. 预注册签字未完成（保持阻断）**

| # | 事项 | 现状 | 阻断范围 | 需要你回答 |
|---|---|---|---|---|
| D1 | 9 个签字项未签（A3/A5/A8、B1/B8/B9/B10、D3/D4/D8） | **未签**（自 2026-10-10 起） | **签字前不得运行正式版本实验**（(spec 述于 PLAN §16.2），即 V1–V8 或 P 类运行 | 逐项审查后签署；或明确改约束（需记录） |

**E. 工程已实现但未过正式研究验收（诚实门，代码正确 ≠ 统计验收）**

| # | 事项（白话） | 代号 | 现状 | 前置条件（就位才能跑） |
|---|---|---|---|---|
| E1 | 把候选特征**真的喂进选股模型**对照基线的正式版未跑 | 模型级增量对照（原"检验 B"） | **merge 内实现已删除**（用户裁定 2026-10-13："模型级增量诊断不放 merge skill"，§16.29.1）。正式对照 = 仓库根生产脚本 **双运行**（`--features_file`：基线集 / 基线+候选）+ `compare_stage15.py` 对拍，`tested_increment` 证据=外部运行清单（`s8_state` 只核引用与文件存在） | 外部生产对照运行与对拍完成；候选须 qualified_stat（现状 0/720 无合格候选） |
| E2 | 个股因子池"版本冻结"未执行 | S8_Stock_Pool 冻结 | 状态机代码级阻断中（`pool_state.json` 不存在） | 按 pool_design §4 路线图执行入池/去重/代表/冻结并获审批 |
| E3 | 整个系统的"生产验收" | — | 从未声明，也不因工程验收而自动达成 | E1+E2 完成 + 正式结果证据齐 + D1 签字（按其约束的范围） |

## 9. 专题文档索引

| 文档 | 内容 | 与本文关系 |
|---|---|---|
| `docs/nonlinear_selection.md` | 非线性筛选与模型适配（MI/JSD/CMIM/JMI） | §3.3 的设计依据 |
| `docs/POOL_PRUNING_ARCHITECTURE.md` | 池精简架构、删除实验、噪声带 | §4 的删除门依据 |
| `docs/tier2_batch_plan_20260930.md` | Tier2 批次计划 | 历史批次记录 |
| `docs/skill_review_20260924.md` | skill 评审记录 | 历史评审 |
| `docs/INITIAL_POOL_INFO_AUDIT_PROTOCOL.md` | 初始池信息审计协议 | 历史审计 |
| `docs/deepseek_task_sector_factors.md` | 行业因子任务记录 | 历史任务 |
| `docs/MERGE_V2_PLAN.md` | v2 完整链条实现规格（含 **§17 统一候选流水线架构**） | **v2 实施的唯一依据** |
| `docs/CODEMAP.md` | 项目级代码-职责对照表 | 项目级索引 |
| `results/merge_v2/gate_search_report.md` | 门控搜索与重评（含 §10 候选确认、§11 停止规则） | §2.6–2.8、§3.8、§5.4 的证据 |
| `results/merge_v2/gate_form_redesign_report.md` | 门控形式重设计 + 评估加速 + 冗余口径辨析 | §3.4、§3.8、§3.11 的证据 |
| `results/merge_v2/gate_forensics_2026-10-10.md` | 门控取证（M3 误杀、n_eff、组合形式）——**文件已随 2026-10-12 旧门控删除** | §3.7、§3.9 的证据（结论保留在有此标记的正文） |
| `results/merge_v2/stability_diag_report.md` | 因子时间稳定性诊断（结论为负） | §3.5 的证据 |
| `results/merge_v2/b1_restore_report.md` | 平台池 legacy API 恢复 | §4.2 的依据 |
| ~~`results/merge_v2/joint_eval_report.md`~~ | （**误删，2026-10-12**——F2 联合评估证据；未被 git 跟踪无法恢复） | F2 结论以 §2.5 正文为准 |
| `results/merge_v2/sector_input_verify.md` | 行业侧接入前验证 | §6.2 的产物 |
| `results/merge_v2/arch_review_v1.md` | Merge Skill 架构审查报告 v1（E1） | 三路径与模块职责核对结论 |
| `results/merge_v2/arch_gap_review.md` | 架构差异审查（现状—目标差距表） | §0.2 的设计输入 |
| `results/merge_v2/pool_design_v1.md` | 因子搜索体系设计 v1（**处置 2026-10-13：部分实现、池冻结路线图为 S8 设计参考、未整体批准**——文件头有处置标记） | 三路径设计、池构建路线图 |
| `results/merge_v2/search_arch_design.md` | 统一搜索架构设计（**处置 2026-10-13：部分被实现取代、残余进 backlog、未整体批准**——文件头有处置标记） | 薄编排层 + 三后端 |

---

## 10. 版本标注约定

- 本文档内所有 `[vX.Y]` / `[v2.8/...]` 标记表示该功能引入或修改的版本。
- `[既有]` 表示 v2.8 之前已存在的功能。
- `[待建]` 表示已规划未实现，附带对应的 MERGE_V2_PLAN 章节号。
- 每次完成功能后，更新本文对应模块节（插入具体位置），并在 MERGE_V2_PLAN 的修订摘要中登记，
  **不得另起"版本流水账"文档**；**也不得另起架构文档**（§0.6）。
