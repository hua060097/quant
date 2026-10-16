看完你补齐的 `grammar.py`、`run_search.py`、`run_gate_pipeline.py` 和 `h3_eval.py` 后，整套 `pandaai_online/merge` 的全貌已经百分之百清晰了。

正如你所判断的，**只靠局部修补 Bug 无法让这套系统成为顶级架构**。因为从全局代码看，你现在的系统存在“四个结构性错位”：

1. **双底座割裂（两个平行宇宙）**：左半边的个股搜索（`run_search.py` + `build_top.py`）跑在 `stock_features_*.parquet` 和动态 `vwap` 标签上；右半边的门控与 F1 检验（`run_gate_pipeline.py` + `f1_batch.py`）却绑在外部已损坏裁剪的 `factors.parquet`（262列变155列，缺18个 `base20` 因子） 和静态 `base20.json` 上。左半边搜出再好的因子，也流不到右半边去做门控合成。


2. **语法树越界污染**：`grammar.py` 里的 `m17_context_factor` 正在个股进化树里直接生成**纯行业广播因子**（如 `TS_MEAN(IND_RET, w)`，每日仅 131 个行业值），和 `gate_search_space.py` 的 14 个行业门控职能重叠且稀释了个体 Alpha 的搜索预算。


3. **门控交互的数学公式与树模型机理冲突**：`h3_eval.py` 定义的 $F_1 = d \cdot x \cdot \max(s \cdot g, 0)$ 是“清零弃权算子”。在行业门控上，它把 50%~70% 非激活行业的股票全变成 `0`，排在了激活行业负分股票（$d \cdot x < 0$）前面，导致 14 个行业门控 $\Delta\text{IC}$ 全部坍缩为 `-0.0204`；在市场门控上，激活日同乘正数不改变单因子截面排序，只能测“空仓择时”，测不出 LightGBM 多因子动态权重的提升。


4. **缺少统一的自定义因子（Python / 论文 / Alpha101）插件入口**：目前 `intake.py` 只能收平台 DSL 字符串公式，`refs.py` 只能读固定的外部 JSON。当你写复杂的论文因子或 Alpha101（比如含行业中性化残差、矩阵运算或复杂截面回归）时，无法无缝注入到同一个评估、去重和门控交互流水线中。



下面为你设计一套“Merge V4.1 统一因子搜索与门控合成引擎”总体重构方案（V4原案 + Gemini核验 + Owner拍板后的执行版）。**本文件是merge skill总体实施方案的唯一入口**：`FACTOR_SEARCH_ARCHITECTURE_V3_PROPOSAL.md`与`TECHNICAL.md`已废弃（仅保留备查，不再更新）。`MERGE_V2_PLAN.md`冻结红线（§8.4 F1公式、§16.27 F1v2族、D7/D8入模红线、停止点）仍然有效，本方案只做增量、不覆盖冻结。

Owner拍板（2026-10-15）：1）零文件删除，只标DEPRECATED；2）VWAP标签维持`vwap_formula_only`，搜索期保持`rank_sia`，重排期统一`lgb_score T_ref=0.3`；3）门控保持54规格不变，纯市场/行业广播禁直入；4）F1保留`g/t`旧族，中心化增益另建`F1v3_gain`新族；5）先做完merge重构，Stage15衔接后续再定（疑问记入文末待定问题）。

Owner拍板V4.2（2026-10-15晚）：1）门控筛选离线化——`build_gate_bank.py`产`certified_gate_bank.json`（G2符号锁定s，G4复用管线簇文件），日常F1/`run_gate_pipeline`经`--gate-bank`只读消费，零重复筛选；2）交互靶标转向偏科池——`select_specialists.py`从vwap rank#21~150按`|t_slope|>=2.5`选market偏科因子（`specialist_pool.json`+值面板+分form cands），top20直入Class1只留5~6机制代表作switch底座；3）三算子——`gain/flip/switch`同属F1v3新族（`--form`分支，`f1_family --from-cands`建清单，`judge_f1`验证逻辑不动），旧`g/t`与F1v2冻结不动；4）Owner显式豁免2~3个正交纯行业状态（`direct_sector_states`，`|rho|<0.7`贪心，D7市场1xT仍为0），澄清D8只禁行业轮动策略不禁选股背景特征。

---

## 一、 顶级因子引擎总体架构图（Merge V4 架构）

新架构将原有的两套割裂系统收敛为“一个物理底座、三路因子摄入（GP进化 + 自定义插件 + 门控网格）、一个正交中枢、一套连续调制交互引擎、双端导出适配”：

```text
===================================================================================================
                 [Layer 0: 统一时点数据与向量化算子底座 (Unified PIT Data & FastRanks)]
  • 唯一主数据源: cache/stock_features_*.parquet (经 boundary.py PyArrow 物理截断 date <= data_cut)
  • 唯一主标签源: fitness.build_label_series (默认 vwap 3日调仓口径; c2c 仅作比赛支线标签)
  • 基础预计算层 (ensure_local_fields):
    - 基础量价: OPEN, HIGH, LOW, CLOSE, VWAP(TP), RET1, LOGC, GAP1, BODY, VOLUME, AMOUNT, TURNOVER, MARKET_CAP
    - 本地上下文: MKT_RET (1×T), IND_RET (S×T), IND_EXCESS (N×T), DISP (S×T)
    - 个股栖息地底座 (Class 2): BETA20, BETA60, IVOL20 (20日特质波动率), AMIHUD20
===================================================================================================
            │                                      │                                      │
            ▼ (摄入源 A: 公式进化搜索)              ▼ (摄入源 B: 自定义/论文/Alpha101)     ▼ (摄入源 C: 门控网格枚举·低频离线)
┌───────────────────────────┐        ┌───────────────────────────┐        ┌───────────────────────────┐
│ 1A. AST 遗传进化引擎       │        │ 1B. 自定义因子插件注册中心 │        │ 1C. 市场/行业状态门控库    │
│  (grammar.py + run_search)│        │  (custom_registry.py)     │        │  (gate_search_space.py)   │
├───────────────────────────┤        ├───────────────────────────┤        ├───────────────────────────┤
│ • 仅搜 N×T 纯个股与行业相对│        │ • 支持两类自定义接入:     │        │ • 严禁进化搜索，防过拟合  │
│   Alpha (M1~M16)          │        │   ① DSL公式字典 (Alpha101)│        │ • 枚举: 基础量×操作×分位窗│
│ • 删掉 M17 纯行业广播模板  │        │   ② @register_py_factor   │        │ • 规格54冻结: 市场40+行业14│
│ • UCB 机制轴配额 + 精英回流│        │     原生 Pandas/NumPy 函数│        │ • 每data_cut→bank-v1(s锁定)│
└─────────────┬─────────────┘        └─────────────┬─────────────┘        └─────────────┬─────────────┘
              └─────────────────┬──────────────────┘                                    │
                                ▼                                                       │
=========================================================================               │
     [Layer 2: 双通道筛选与全局正交中枢 (Unified Screening & Orthogonality)]              │
  • Stage 2.1 快评 (Fast Screen, 2500股×26月):                                          │
    - 通道 A (全场高分): vwap_lgb_score >= 阈值 或 Top 50%                               │
    - 通道 C (栖息地补捞): 全场一般，但在强势股栖息地 (cond_ic: hot5_q80) |t| >= 2.5 豁免晋级│
  • Stage 2.2 全评与指纹去重 (Full Eval):                                                │
    - 结构指纹 (struct_fp) + 分布指纹 (vfp) + 时序衰减检验 (decay_ratio >= 0.6)          │
  • Stage 2.3 全局正交排榜中枢 (build_top.py 升级版):                                    │
    - 汇总 [mined(GP) + custom(自定义/Alpha101) + refs(外部库)] 全部存活候选              │
    - 按 LGB 复合分降序，执行逐日截面 Spearman |ρ| <= 0.7 贪心正交化                     │
    - 产出 1: 【入选纯个股 Alpha 池】(Top 60~100，直通 LightGBM)                         │
    - 产出 2: 【动态靶标面板 vwap_top20.json/factors.parquet + 偏科池 specialist_pool.json/factors.parquet】(top20机制均衡；偏科按t_slope选)│
              的日内居中百分位秩 u_{i,t} = d*(rank_pct - 0.5) ∈ [-0.5, +0.5])           │
=========================================================================               │
                                │                                                       │
                                └───────────────────────────┬───────────────────────────┘
                                                            ▼
===================================================================================================
              [Layer 3: 门控资格离线认证与连续调制交互合成引擎 (Gate & Interaction Engine)]
  • Stage 3.1 门控资格离线筛选（低频，每data_cut一次，与日常因子搜索彻底解耦）:
    - 离线泳道 build_gate_bank.py: 短名单 → G0卫生门+方向中性 → G4聚类去冗余(|ρ|>=0.8取簇代表)
      → G2/G3组级检验(锁符号s) → certified_gate_bank.json (bank-v1，旧版归档禁静默覆盖)
      + direct_sector_states.parquet (正交纯行业状态，D8豁免，只随data_cut刷新)
    - 日常泳道只读消费 bank (--gate-bank certified_gate_bank.json): 过滤到认证门控集+s锁定覆盖，
      不再逐日重跑G2/G3筛选——门控资格是一次性权威结论，日常因子搜索循环不含任何门控筛选
    - run_gate_pipeline.py 保留原筛选能力但仅用于bank刷新/诊断（--gate-bank时转为bank门控+cands生成）
  • Stage 3.2 显式交互特征构造与 H3/FDR 检验 (f1_batch.py/f1_sector.py --form gain|flip|switch):
    - gain【中心化增益调制】: F = u*clip(1+0.5*s*g, 0.2, 1.8)；flip【方向翻转】: F = u*(s*g)，|g|<0.10 死区弃权；
      switch【双因子软切换】: F = p*u_off+(1-p)*u_def
    - 市场门控 (1×T): gain 测底仓倾斜增量；flip/switch 测单因子截面 ΔRankIC（switch 对照静态共识，tilt 仅辅助列）
      + 200次块置换 Placebo + BH-FDR；三式分开跑 FDR，严禁合并
    - 行业门控 (S×T): 三式统一测单因子截面 ΔRankIC + 日期整块行业置换 Placebo + BH-FDR
    - 通过检验的配对经 materialize_interactions.py 实例化为数值矩阵: approved_interactions.parquet (N×T)
     （flip 死区仅 u 有限处填 0.0，u 缺失处保持 NaN；gain/switch 不填）
===================================================================================================
                                                            │
                                                            ▼
===================================================================================================
                       [Layer 4: 下游双端导出适配器 (Downstream Adapters)]
  • 适配器 A (LightGBM 训练矩阵组装器 export_lgb_dataset.py):
    组装 [Class 1: 正交个股Alpha] + [Class 4: 合格门控交互列(禁做单日截面Zscore)]
       + [Class 2: 6个个股栖息地底座] + [Class 3: 2~3个G4聚类代表行业状态]，生成 FeatureContract Manifest
  • 适配器 B (C2C 因子大赛导出器 executor.py):
    过滤 local_only=False 且为 AST 公式的候选，导出平台兼容 candidates.txt
===================================================================================================

```

---

## 二、 因子四分类法与入模铁律（架构级强约束）

在重构后的代码中，所有特征严格归入以下 **4 个互斥类别（Class 1 ~ Class 4）**，彻底消除“兼具多重类别不知如何评估”的混乱：

| 类别代号 | 物理张量粒度 | 严格定义与边界（处理跨类模糊因子） | 来源模块 | 评估指标与去重规则 | 入模 LightGBM 规则 |
| --- | --- | --- | --- | --- | --- |
| **Class 1**<br>

<br>`STOCK_ALPHA`<br>

<br>（个股选股 Alpha） | `STOCK×DATE`<br>

<br>($N \times T$) | 凡是最终输出为每只股票每日一个值、且旨在直接预测未来横截面收益排序的因子。**含利用行业/市场算出的个股相对因子**（如 `RET1 - IND_RET`、`(RET1 - IND_RET)/DISP`、残差动量 `RET1 - BETA60*MKT_RET`）。

 | 1. GP 搜索 (`grammar.py`)<br>

<br>2. 自定义库 (`custom_registry.py`)<br>

<br>3. 外部库 (`refs.py`)

 | 全场 `lgb_score`（含 `mono` 与换手软惩罚）**或** 栖息地 `cond_ic` 显著；截面 Spearman $\vert{}\rho\vert{} \le 0.7$ 正交去重。

 | **主力入模**（日内截面 RankPct 或 Z-score 标准化后入模）。 |
| **Class 2**<br>

<br>`STOCK_HABITAT`<br>

<br>（个股风险/栖息地底座） | `STOCK×DATE`<br>

<br>($N \times T$) | 个股级风险暴露或所处状态：固定为 `BETA20`、`BETA60`、`IVOL20`（特质波动率）、`AMIHUD20`、`TURNOVER_20`、`LOG_MV` 共 6 个核心指标。它们单看长期均值为 0（无独立 Alpha 方向），但截面方差大。

 | `evaluator.py`<br>

<br>`ensure_local_fields` 预计算

 | 仅做覆盖率（$\ge 80\%$）与无前视检查，**豁免单因子 IC 淘汰**。

 | **全部直接入模**（供 LightGBM 切分大/小盘、高/低弹性个股分支），同时作为 AST 叶节点供 Class 1 组合。

 |
| **Class 3**<br>

<br>`REGIME_GATE`<br>

<br>（市场与行业状态门控） | `MARKET×DATE` ($1 \times T$)<br>

<br>`SECTOR×DATE` ($S \times T$) | 描述大盘整体环境（离散度、涨跌停比、大小盘分化、市场波动）或申万二级行业状态（行业动量排名、行业内分化度、行业换手变化）。

 | `gate_search_space.py`<br>

<br>（合并 `cond_regime_ic.py`）

 | G0 卫生门 $\to$ 方向中性度 $\to$ G4 聚类去重（$\vert{}\rho\vert{} \ge 0.8$ 留 1）$\to$ G2/G3 组级条件斜率显著。

 | **$1 \times T$ 纯市场状态严禁入模！**<br>

<br>$S \times T$ 行业状态仅取 G4 聚类后的 **2~3 个簇中心代表**直接广播入模；其余全部转入 Class 4。

 |
| **Class 4**<br>

<br>`GATED_INTERACT`<br>

<br>（显式门控交互因子） | `STOCK×DATE`<br>

<br>($N \times T$) | 由 **偏科/机制代表因子 ($u_{i,t}$) $\otimes$ bank认证门控 ($g_t / g_{s,t}$)** 经`gain/flip/switch`三算子显式合成的特征。 | `f1_batch.py`/`f1_sector.py`内部分支（`--form`），族清单`f1_family --from-cands`

 | 市场门控：gain 测**底仓倾斜 $\Delta\text{RankIC}$**，flip/switch 测**单因子截面 $\Delta\text{RankIC}$**（switch 对照静态共识，tilt 仅辅助列）；行业门控三式统一测**单因子截面 $\Delta\text{RankIC}$**；均须通过块置换 Placebo $p < 0.05$ 与 BH-FDR $q \le 0.10$（三式分族，严禁合并）。

 | **原值直接入模！**（严禁在入模前对其再做单日横截面 Z-score/Rank，否则会约掉同日市场门控乘数）。 |

---

## 三、 现有代码“留、改、删、增”全景裁决表

为了让执行重构的大模型动作干净利落，下面对 `pandaai_online/merge/` 下的所有文件给出明确处置指令：

### 1. 零删除原则（DEPRECATED — 保留证据链，只标职责边界）

Owner裁决：取消原V4删除6文件的操作（PLAN §16.26有误删教训）。以下文件全部保留原位，仅在文件头加`DEPRECATED`注释并做引用断链检查：

| 文件名 | 裁决 | 说明 |
| --- | --- | --- |
| `confirm_candidate.py` | **保留/DEPRECATED** | 单一门控一次性脚本，逻辑由统一交互管线承接，但不删除。 |
| `gate_search_prepare_f1.py` | **保留/DEPRECATED** | 脆弱胶水脚本，由`run_gate_pipeline.py`直接衔接，但不删除。 |
| `corr_compare.py` | **保留/DEPRECATED** | 一次性研究脚本，非生产管线，不删除。 |
| `experiments_log_ab.py` | **保留/DEPRECATED** | log政策A/B结论已固化，不删除。 |
| `cond_regime_ic.py` | **保留** | 独有`mkt_vol_20`/`sector_vol_20`/`sector_mom_60`如需吸收另立项，不合并删除。 |
| `cond_supplement.py` | **保留** | 通道C逻辑不前移（快评recall@60已91.7%，不加豁免），不删除。 |

### 2. 原样保留（KEEP — 底座扎实，直接复用）

* **基础工具与契约**：`_paths.py`、`windows.py`、`data_version.py`、`audit_cut_paths.py`、`bench_fast_precision.py`、`feature_contract.py`。


* **算子与评估底层**：`parser.py`、`explain.py`、`fixtures.py`、`dedup.py`、`fastranks.py`、`cond_ic.py`、`gates.py`、`gate_validate.py`、`gate_ledger.py`、`gate_stop.py`、`gate_g4_clusters.py`、`f1_family.py`、`executor.py`、`ledger.py`。



### 3. 核心重构（REFACTOR — 共 10 个文件，打通全链路）

* `boundary.py` & `evaluator.py` & `field_catalog.py`（PyArrow物理隔离+`ensure_extended_fields`安全扩展，不动`ensure_local_fields`4字段）


* `grammar.py` & `fitness.py` & `run_search.py`（M17默认关闭不删除、搜索期保持`rank_sia`、不加通道C豁免、接入自定义因子）


* `build_top.py`（接管`custom`+`mined`+`refs`全池正交去重，输出`vwap_top20.json`+面板供`--base-panel`选用，不替代`base20.json`/`factors.parquet`）


* `gate_search_space.py` & `run_gate_pipeline.py`（保持54规格不变，G4复用管线簇文件去重，`--gate-bank`只读消费bank的s锁定） & `build_gate_bank.py`（离线bank构建：短名单+G4簇复用+s锁定+direct states） & `select_specialists.py`（偏科池：t_slope精确计算+值面板+cands/族清单直连）


* `f1_batch.py` + `f1_sector.py` + `h3_eval.py`（保留`g/t`冻结公式零改动；新增`--form gain/flip/switch`为F1v3新族分支，`--base-panel`加`specialist`，`--gate-bank`过滤+s覆盖；`f1_family --from-cands`建新族清单，`judge_f1`不动） + `selftest_interaction_forms.py`（三算子合成自测6/6）



### 4. 新增模块（NEW — 共 2 个核心文件）

1. **`custom_registry.py`（自定义/论文/Alpha101 因子插件中心）**：支持用装饰器 `@register_py_factor` 编写任意复杂的 Pandas/NumPy 因子或注册 Alpha101 公式字典，自动接入缓存、快评、全评、正交榜单与门控交互。须过`audit_cut_paths A2`（禁`shift(-n)/center`）+长度对齐检查。求值统一走`ensure_extended_fields`（超集，含 BETA20/IND_EXCESS/IVOL20；教训：曾只调`ensure_local_fields`导致示例因子 `BETA20` KeyError，由 `audit_pit` warn 捕获）。
   附带 **`audit_pit` 时间穿越扰动审计**：`custom_registry.audit_pit(spec, frame)` 取 3 个随机 T，把 T+1 后行情列改随机数重算，断言≤T 值 `allclose` 不变；每 spec_hash 每进程一次，大面板采样 200 只防爆内存，warn-only 不阻断。合成验证：正常滚动因子通过，前视 `shift(-1)` 作弊 3 探针全抓。教训：`price_cols` 必须大小写兼容（真实面板大写列），无列时拒测，杜绝静默通过。
2. **`export_lgb_dataset.py`（LightGBM 训练特征矩阵一键导出器，Stage15衔接暂缓）**：按四分类铁律组装特征矩阵与`skip_daily_cs_norm_cols`白名单。Stage15桥接（`factors_with_merge_cand.parquet`/`bridge_stage15.py`）后续再定，本期只保证merge内部产物完整，疑问记入文末待定问题。

---

## 四、 分阶段重构详细执行规范（供 AI 编码助手直接执行）

> **给执行大模型的全局指令**：请按以下 Stage 1 至 Stage 6 的顺序依次修改 `to/integrated_stock_selection/pandaai_online/merge/` 下的代码。每完成一个 Stage，运行对应的验证命令确保无回归错误。

---

### Stage 1：统一物理读取边界与扩充 Class 2 栖息地底座

**涉及文件**：`boundary.py`、`field_catalog.py`、`evaluator.py`

1. **`boundary.py` 改造（强制 PyArrow 物理过滤）**：


* 新增内部辅助函数 `_read_parquet_bounded(fp: str, cols: list[str], max_date: str) -> pd.DataFrame`：先用 `pyarrow.parquet.read_schema(fp)` 检查 `date` 列类型，在 `pd.read_parquet(fp, columns=cols, filters=[("date", "<=", bound)])` 中直接传入 `filters`，从磁盘 I/O 层杜绝读取 `max_date` 之后的确认集数据。


* 在 `read_factors`、`read_labels`、`read_cache` 中统一调用该函数。


* 新增可选 `read_base_panel(data_max: str)`：读取 `results/merge_v2/vwap_top20_factors.parquet`（由 `build_top.py` 生成的动态正交面板，供`--base-panel vwap_top20`选用）。默认链路仍读`factors.parquet`+`base20.json`以保证历史可复现，不彻底摆脱硬依赖。




2. **`field_catalog.py` 改造**：


* 将 `LOCAL_ONLY_FIELDS` 扩展为：
```python
LOCAL_ONLY_FIELDS = (
    'VWAP', 'MKT_RET', 'IND_RET', 'IND_EXCESS', 'DISP',
    'BETA20', 'BETA60', 'IVOL20', 'AMIHUD20'
)

```




3. **`evaluator.py` 改造**：


* `load_panel(data_cut: str | None = None)`保持默认`None`不变（不破坏现有调用方），仅在显式传入`data_cut`时透传PyArrow `filters`裁剪。`ensure_local_fields`原4字段不动；新增可选`ensure_extended_fields(frame)`按需计算`IND_EXCESS/BETA20/IVOL20`，避免15,329条历史台账指纹变动。


* 重写 `ensure_local_fields(frame: pd.DataFrame)`（注意：内部所有滚动算子严禁使用 `shift(-n)` 或 `center=True`，确保 `audit_cut_paths.py` 的 `A2` 测试通过）：


```python
def ensure_local_fields(frame: pd.DataFrame) -> pd.DataFrame:
    if all(c in frame.columns for c in LOCAL_ONLY_FIELDS):
        return frame
    codes = frame['stock_code']
    dates = frame['date']
    # 1. 日内均价代理 VWAP 与日收益
    frame['VWAP'] = (frame['OPEN'] + frame['HIGH'] + frame['LOW'] + frame['CLOSE']) / 4.0
    ret = frame.groupby(codes, sort=False)['CLOSE'].pct_change(fill_method=None)
    # 2. 市场与行业上下文
    frame['MKT_RET'] = ret.groupby(dates, sort=False).transform('mean')
    ind = codes.map(sw2_mapping()).fillna('__NA__')
    frame['IND_RET'] = ret.groupby([dates, ind], sort=False).transform('mean')
    frame['IND_EXCESS'] = ret - frame['IND_RET']
    frame['DISP'] = ret.groupby([dates, ind], sort=False).transform('std')

    # 3. 滚动 BETA20 / BETA60 与 20日特质波动率 IVOL20
    m = frame['MKT_RET']
    rm = ret * m
    m2 = m * m
    r2 = ret * ret
    for w, bcol in ((20, 'BETA20'), (60, 'BETA60')):
        def _rmean(s: pd.Series, win: int = w) -> pd.Series:
            return s.groupby(codes, sort=False).rolling(win, min_periods=win) \
                .mean().reset_index(level=0, drop=True)
        er, em = _rmean(ret), _rmean(m)
        cov = _rmean(rm) - er * em
        var_m = _rmean(m2) - em * em
        beta = cov / var_m.where(var_m > 1e-12)
        frame[bcol] = beta
        if w == 20:
            var_r = _rmean(r2) - er * er
            resid_var = (var_r - beta * beta * var_m).clip(lower=0.0)
            frame['IVOL20'] = np.sqrt(resid_var)

    # 4. AMIHUD20 非流动性底座
    ill = (ret.abs() / frame['AMOUNT'].where(frame['AMOUNT'] > 0)) * 1e9
    ill_mean20 = ill.groupby(codes, sort=False).rolling(20, min_periods=15) \
        .mean().reset_index(level=0, drop=True)
    frame['AMIHUD20'] = np.log1p(ill_mean20.clip(lower=0.0))
    return frame

```





---

### Stage 2：构建自定义因子扩展中心（支持自定义论文因子 & Alpha101）

**新增文件**：`custom_registry.py` 与目录 `custom_factors/__init__.py`、`custom_factors/alpha101_examples.py`

**设计目标**：让你后续写任何论文因子或 Alpha101 因子时，只需在一个文件里写一个函数或一行公式，就能自动进入 `run_search.py` 和 `build_top.py` 参与评估、条件 IC 诊断、正交去重和门控交互。

**`custom_registry.py` 完整实现规范**：

```python
# -*- coding: utf-8 -*-
"""自定义因子（论文 / Alpha101 / Python 向量化函数）统一注册与求值中心。

支持两种注册方式：
  1. DSL 公式型：register_formula_factor(factor_id, formula, axis, description)
  2. Python 函数型：@register_py_factor(factor_id, axis, description)
     函数签名：fn(frame: pd.DataFrame, ev: Evaluator) -> pd.Series | np.ndarray
     传入的 frame 已按 (stock_code, date) 排序且已调用 ensure_local_fields(frame)，
     可直接访问 OPEN/HIGH/LOW/CLOSE/VWAP/VOLUME/AMOUNT/TURNOVER/MARKET_CAP/
     MKT_RET/IND_RET/IND_EXCESS/DISP/BETA20/BETA60/IVOL20 等全部列。
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from evaluator import Evaluator, ensure_local_fields
from parser import parse_formula


@dataclass
class CustomFactorSpec:
    factor_id: str
    axis: str                  # 对应 AXIS_OF 的机制轴，如 'volatility', 'pv_relation', 'trend_position'
    family: str                # 默认 'custom:<axis>'
    kind: str                  # 'formula' | 'python'
    formula_str: str           # 公式型存公式字符串；Python 型存函数源码签名摘要
    fn: Callable | None = None
    description: str = ''
    competition_uploadable: bool = False

    @property
    def spec_hash(self) -> str:
        if self.kind == 'formula':
            payload = f"formula|{self.factor_id}|{self.formula_str.strip()}"
        else:
            src = inspect.getsource(self.fn) if self.fn is not None else self.factor_id
            payload = f"python|{self.factor_id}|{src}"
        return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


REGISTRY: dict[str, CustomFactorSpec] = {}


def register_formula_factor(factor_id: str, formula: str, axis: str = 'custom',
                            description: str = '') -> CustomFactorSpec:
    from executor import competition_status
    ok_comp, _ = competition_status(formula)
    spec = CustomFactorSpec(
        factor_id=factor_id, axis=axis, family=f'custom_{axis}',
        kind='formula', formula_str=formula, fn=None,
        description=description, competition_uploadable=ok_comp
    )
    REGISTRY[factor_id] = spec
    return spec


def register_py_factor(factor_id: str, axis: str = 'custom', description: str = ''):
    def _decorator(fn: Callable) -> Callable:
        spec = CustomFactorSpec(
            factor_id=factor_id, axis=axis, family=f'custom_{axis}',
            kind='python', formula_str=f"py:{fn.__name__}", fn=fn,
            description=description, competition_uploadable=False
        )
        REGISTRY[factor_id] = spec
        return fn
    return _decorator


def evaluate_custom_factor(spec: CustomFactorSpec, frame: pd.DataFrame,
                           ev: Evaluator | None = None) -> np.ndarray:
    """在给定面板上求值自定义因子，返回与 frame 行对齐的 float64 数组。"""
    ensure_local_fields(frame)
    ev = ev or Evaluator(frame)
    if spec.kind == 'formula':
        node = parse_formula(spec.formula_str)
        vals = ev.values(node)
    else:
        assert spec.fn is not None
        res = spec.fn(frame, ev)
        vals = res.to_numpy(dtype=float) if isinstance(res, pd.Series) else np.asarray(res, dtype=float)
    if len(vals) != len(frame):
        raise ValueError(f"Custom factor {spec.factor_id} returned length {len(vals)} != frame {len(frame)}")
    return np.where(np.isfinite(vals), vals, np.nan)


def load_all_custom_Specs() -> dict[str, CustomFactorSpec]:
    """自动导入 custom_factors/ 目录下所有模块以触发装饰器注册。"""
    import importlib
    import pkgutil
    from pathlib import Path
    pkg_dir = Path(__file__).resolve().parent / 'custom_factors'
    if pkg_dir.exists():
        for _, mod_name, _ in pkgutil.iter_modules([str(pkg_dir)]):
            importlib.import_module(f'custom_factors.{mod_name}')
    return dict(REGISTRY)

```

同时，在 `custom_factors/alpha101_examples.py` 中预置几个典型示例（展示如何用公式或原生 Python 写 Alpha101 与论文因子）：

```python
# -*- coding: utf-8 -*-
"""自定义 Alpha101 与论文因子示例库（在此文件或同目录新建 .py 即可自动注册）。"""
import numpy as np
import pandas as pd
from custom_registry import register_formula_factor, register_py_factor

# 方式 1：直接用平台/本地 DSL 公式注册 Alpha101
register_formula_factor(
    factor_id="alpha101_006",
    formula="0 - CORR(OPEN, VOLUME, 10)",
    axis="pv_relation",
    description="Alpha#6: -1 * correlation(open, volume, 10)"
)

register_formula_factor(
    factor_id="alpha101_012",
    formula="SIGN(DIFF(VOLUME, 1)) * (0 - DIFF(CLOSE, 1))",
    axis="pv_relation",
    description="Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))"
)

# 方式 2：用原生 Python / Pandas 编写复杂论文因子或带截面中性化的 Alpha101
@register_py_factor(
    factor_id="paper_idio_mom_20d",
    axis="trend_position",
    description="剥离市场Beta与行业收益后的20日特质收益风险调整动量"
)
def paper_idio_mom_20d(frame: pd.DataFrame, ev) -> pd.Series:
    codes = frame['stock_code']
    # 个股超额行业且剥离市场 Beta 后的日残差
    ret1 = frame.groupby(codes, sort=False)['CLOSE'].pct_change(fill_method=None)
    resid = ret1 - frame['IND_RET'] - frame['BETA20'] * (frame['MKT_RET'] - frame['IND_RET'])
    g = resid.groupby(codes, sort=False)
    mean20 = g.transform(lambda s: s.rolling(20, min_periods=15).mean())
    std20 = g.transform(lambda s: s.rolling(20, min_periods=15).std())
    return mean20 / std20.where(std20 > 1e-8)

```

---

### Stage 3：重构个股进化搜索与双通道筛选（清理 M17、升级 LGB 主分、内嵌通道 C）

**涉及文件**：`grammar.py`、`fitness.py`、`run_search.py`

1. **`grammar.py` 改造（严守 Class 1 边界）**：


* **M17默认关闭、不删除**：`m17_context_factor`生成纯行业广播变量（$S \times T$），按PLAN §10.9降为legacy。先只读统计`ledger.jsonl`中`family=='M17_context_factor'`的已测次数/通过率/最高vwap分，确认后再通过`--local-context`开关默认关闭，代码与`AXIS_OF`映射保留。


* **升级 `m14_context` 与 `m16_ctx_interaction**`：将 Stage 1 新增的 `IND_EXCESS`、`BETA20`、`IVOL20`、`VWAP` 加入 `_M16_LEG_BUILDERS` 和 `m14_context` 腿池中，让进化引擎能自由搜出 `TS_MEAN(IND_EXCESS, 20) / IVOL20` 或 `CORR(VWAP, VOLUME, 20)` 等高质量个股因子！




2. **`fitness.py` 改造（统一自用 LightGBM 评价函数）**：


* `fitness.py`口径冻结：搜索期（`run_search.py`）保持`de99122`的`rank_sia = |RankIC|*|RankICIR|*win`不变（快评抽样下`mono`噪声大，不误杀）；重排/榜单期统一用`refs.lgb_score`现有公式（`T_ref=0.3`，含`|mono|`），取消`0.35/clip(0.2,1.0)`等新参数。如需`vwap_lgb_score` helper，必须复用`lgb_score`实现，不得另起数字。




3. **`run_search.py` 改造（3 处关键升级）**：


* **升级 1（保持 `label_search_value`）**：`run_search.py`的`label_search_value(tier,label)`在`vwap`时保持`rank_sia`不变，不调用含`mono`/换手的新函数，避免快评误杀。


* **升级 2（通道C豁免暂缓）**：现行快评`recall@60 91.7%`（PLAN §20.14.2），不内嵌`hot5_q80 |t|>=2.5`豁免。如后续要加，必须先做分层漏检校准并预注册阈值，本期只保留诊断输出。


* **升级 3（支持 `--include-custom` 一键评测自定义因子）**：
在 `SearchRun.run()` 中，自动加载 `custom_registry.load_all_custom_specs()` 中未测过的自定义因子，与 `proposal_nodes` 一起直接送入 `evaluate_full` 评估并记入 `ledger.jsonl`。





---

### Stage 4：升级 `build_top.py` 为全局正交中枢（新增动态面板，不替代 `base20.json` 与 `factors.parquet`）

**涉及文件**：`build_top.py`

在 `build_top.py` 中完成三件事，使其成为连接“左半边个股挖掘”与“右半边门控合成”的唯一枢纽：

1. **三源合一候选池**：在 `main()` 中汇总三路候选：① `eval_mined`（台账挖掘的 AST 公式）、② `norm_refs`（`refs.py` 外部因子库）、③ **`eval_custom`（`custom_registry.py` 中注册的所有自定义论文/Alpha101 因子）**。


2. **扩充正交榜单容量**：将 `vwap_orthogonal_indices` 的保留数量从 `keep=20` 提升为可配置参数 `--keep-pool 60`（用于直通 LightGBM 的主特征池）和 `--keep-base 25`（用于门控交互测试的机制平衡靶标面板）。


3. **导出动态门控靶标面板 `vwap_top20.json` + `vwap_top20_factors.parquet`**：
在正交入选后按`family`均衡抽取前20个正交因子（每轴限额，保证机制覆盖），`direction 1/0`映射为`+1/-1`存`vwap_top20.json`，因子值存`vwap_top20_factors.parquet`（另存IC矩阵`ic_vwap_top20.parquet`可选）。
> **兼容收益**：不淘汰`factors.parquet`/`base20.json`/`ic_stock_2025.parquet`；门控线新增`--base-panel base20|vwap_top20`（默认`base20`保证历史可复现）。有向居中秩`u=d*(RankPct-0.5)`只用于`--form gain`新族输入，不改旧族。
>
>



---

### Stage 5：重构门控与交互合成引擎（修复 3 大数学 Bug + 合并市场/行业流水线）

**涉及文件**：`gate_search_space.py`、`run_gate_pipeline.py`、新增/合并 `f1_interaction.py`（替代原 `f1_batch.py` + `f1_sector.py`）

#### 1. 扩充 `gate_search_space.py`（吸收 `cond_regime_ic.py`）

：

* 保持Gate Grid V1 54规格（40市场+14行业）冻结不变，不扩到45+18，不新增`--qwins`多窗口枚举。如需`sec_mom_20/sec_vol_20`，另立项并走新族预注册，本期不动。



#### 2. 升级 `run_gate_pipeline.py`（统一处理 Market 与 Sector 门控 + G4 聚类选代表）

：

* **市场/行业分线执行不变**：`run_gate_pipeline.py`仍跑market，sector走专用后端，不强行统一入口，避免回归爆炸。组级基准默认读`ic_stock_2025.parquet`，`--base-panel vwap_top20`时才读`ic_vwap_top20.parquet`。
* **G4 复用管线簇文件（诊断口径不变）**：bank构建复用`g4_clusters_market_pipeline.csv`（series/0.8），每簇留`score`最高1个；G4仍只记录+告警，不升级为淘汰门（§17.5纪律）。
* **离线bank（`build_gate_bank.py`，低频）**：输入短名单+sector G2G3+registry，输出`certified_gate_bank.json`（`bank-v1`，含data_cut/面板sha/G2符号锁定s/来源）；sector取`|delta_median|`贪心+flatten p向量`|rho|<0.7`选至多`--max-sector-direct`（默认2，上限3）落`direct_sector_states.parquet`。日常F1经`--gate-bank`过滤+s覆盖，`run_gate_pipeline`经`--gate-bank`用bank门控+s生成cands；不传保持历史行为。


* **自动确定门控作用符号 $s \in \{+1, -1\}$**：
不再像旧代码那样盲目硬编码 `s = +1`！如果某个门控 $g$ 在 G2 中呈现显著的**负斜率**（即 `median_b < 0` 且 4 折同为负，说明低状态时个股 Alpha 更强），自动将该门控的预注册方向设为 **`s = -1`**！



#### 3. 新增交互分支 `--form gain/flip/switch` 为 `F1v3` 新族（保留`g/t`冻结公式，先复现再立项）：

`f1_batch.py`/`f1_sector.py`/`h3_eval.py`的`d*x*max(s*g,0)`一律不动，旧720结论保留。`--form gain`分支已落地；本次补齐`flip`/`switch`分支（以内部分支实现，不新建`f1_interaction.py`，避免入口分裂）：

* **输入**：`--base-panel base20|vwap_top20|specialist`（specialist读`specialist_factors.parquet`+`specialist_pool.json`方向）；门控经`--gate-bank`过滤+s覆盖，不传则用cands自带值。
* **靶标转向偏科池**：`select_specialists.py`从vwap rank#21~150按`|t_slope|>=2.5`（`h3_eval.slope_test`，IC_t~g_t精确回归）选market偏科因子，无条件`|rank_ic|<0.05`；高低分组IC异号建议flip否则gain；switch对取trend轴代表×波动/分布/流动轴代表×bank前2门控（上限12对，V1启发式）。top20直入Class1，只留机制代表作switch底座。
* **统一交互特征构造公式（中心化增益调制）**：
对于任意配对 $(u_{k,i,t}, \; g, \; s)$，定义调制权重与交互特征：

$$w_{i,t} = \text{clip}\big(1.0 + 0.5 \cdot s \cdot g_{i,t}, \; 0.2, \; 1.8\big)$$


$$F_{k,g,i,t} = u_{k,i,t} \cdot w_{i,t}$$



*(其中对市场门控 $g_{i,t} \equiv g_t$；对行业门控 $g_{i,t} \equiv g_{\text{sec}(i), t}$。)*
* **方向翻转（`--form flip`）**：$F_{k,g,i,t} = u_{k,i,t} \cdot (s \cdot g_{i,t})$，允许负权重实现跨状态方向倒转。更正V4.2初版表述：市场`flip`在单因子日度RankIC下退化为纯符号翻转（同日同乘常数，幅度$|g_t|$被约掉），故$|g_t|<0.10$中性日置NaN弃权（不强行押注方向；特征落盘时填0.0）；市场/行业统一走单因子ΔIC，placebo对含死区的权重函数整体置换。
* **双因子软切换（`--form switch`）**：$F_{i,t} = p_{i,t} \cdot u^{\text{off}}_{i,t} + (1-p_{i,t}) \cdot u^{\text{def}}_{i,t}$，$p=(g+1)/2$；市场/行业统一对照为单因子静态等权共识`0.5*u_off+0.5*u_def`的截面ΔIC（市场底仓倾斜保留为`tilt_delta`辅助列，不作主判定；switch 独立`--out`文件保证 CSV schema 自洽）；cands须带`base_off/base_def/d_off/d_def`，输出`base`记复合键`off__def`以保族校验逐一对应；`--base-panel`取列时复合键排除，只取真实列（否则面板取列 KeyError）。
* 三式权重函数收敛于`h3_eval.build_gain_weight/build_flip_weight/build_switch_blend`（`selftest_interaction_forms.py` 6/6合成自测）；新族清单经`f1_family --from-cands --family F1v3 --form <gain|flip|switch>`生成，`judge_f1`验证逻辑零改动；三式与`g/t`分开跑FDR，严禁合并。
* **分粒度评估准则（H3 + Placebo + BH-FDR）**：
1. **当门控为行业门控 (`level == 'sector'`) 时**：
* 因为同一天不同行业的 $w_{s,t} \in [0.5, 1.5]$ 不同，直接改变全市场个股排序，且 $w_{s,t} > 0$ 恒成立，**彻底消除了旧代码 `max(s*g, 0)` 带来的非激活行业全员 0 值塌缩 Bug**！
* 逐日计算 $\text{IC}_t(F_{k,g}) = \text{Spearman}_t(F_{k,g,i,t}, y^{\text{vwap}}_{i,t})$ 与 $\text{IC}_t(u_k)$ 的配对差：

$$\Delta\text{IC}_t = \text{IC}_t(F_{k,g}) - \text{IC}_t(u_k)$$




2. **当门控为市场门控 (`level == 'market'`) 时（仅 `--form gain`）**：
* 为解决单日全市场同乘常数 $w_t$ 不改变单因子内部排序的盲区，采用**多因子底仓倾斜测试（Baseline-Tilt Test）**：
令底仓得分为 $\bar{U}_{i,t} = \frac{1}{K}\sum_{m=1}^{K} u_{m,i,t}$。对比加入动态门控调节前后的多因子截面排序增量：

$$S^{\text{static}}_{i,t} = \bar{U}_{i,t} + 0.25 \cdot u_{k,i,t}, \quad S^{\text{gated}}_{i,t} = \bar{U}_{i,t} + 0.25 \cdot F_{k,g,i,t}$$


$$\Delta\text{IC}_t = \text{Spearman}_t(S^{\text{gated}}_{i,t}, y^{\text{vwap}}_{i,t}) - \text{Spearman}_t(S^{\text{static}}_{i,t}, y^{\text{vwap}}_{i,t})$$

* `--form flip/switch` 的市场分支改走单因子截面ΔIC（`switch`对照静态共识，tilt只记`tilt_delta`辅助列），见上文三算子定义。




3. **统一的显著性判定（复用 `h3_eval.py` 与 `judge_f1.py`）**：


* 块自助法（`BLOCK_BOOT = 5`，2000 次）95% 置信区间下界 `ci_lo > 0`；


* 6 个连续时间块中至少 4 块 $\Delta\text{IC} > 0$（`nblocks_pos >= 4`）；


* 200 次日期块置换（`PLACEBO_BLOCK = 20`，市场门控置换日序列 $g_t$，行业门控按日期块整体置换行业矩阵 $G[p_{\text{idx}}, :]$）经验 $p < 0.05$；


* 在当前短名单配对族上执行 BH-FDR 校正 `fdr_q <= 0.10`。






* **自动特征落盘**（`materialize_interactions.py`，已实现并合成验证）：所有满足 `h3_pass & fdr_sig` 的配对，按本节三式定义重算实例化为 $N \times T$ 列写入 `results/merge_v2/approved_interactions.parquet`；flip 死区仅 `isfinite(u)&isfinite(g)&~isfinite(F)` 处填 `0.0`（停牌/未上市保持 NaN，缺门控日保持 NaN）；方向以 cands 的 `d/d_off/d_def` 为准。`fdr_sig` 为 0 时落空表（分支畅通已由合成 2 特征验证：5.38M 行，死区填 0 共 439,892 格）。

---

### Stage 6：新增 LightGBM 训练矩阵组装器 (`export_lgb_dataset.py`，Stage15桥接待定)

**新增文件**：`export_lgb_dataset.py`（merge内部交付层；Stage15对接后续再定）

这个脚本负责把四类特征按比例与预处理契约打包成特征文件与 Manifest。Stage15桥接（`factors_with_merge_cand.parquet`/`bridge_stage15.py`/`features_baseline|enhanced.json`）本期不实施，疑问记入§六待定问题：

```python
# -*- coding: utf-8 -*-
"""LightGBM 训练特征矩阵与 FeatureContract 一键组装导出器。

用法：
    python export_lgb_dataset.py --max-alpha 80 --max-interact 25 --out-dir ../../results/merge_v2/lgb_ready
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import _paths  # noqa: F401
import windows as W
from evaluator import Evaluator, ensure_local_fields, load_panel

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / 'results' / 'merge_v2'

#: Class 2 个股栖息地/风险状态固定底座（6个）
HABITAT_COLS = ('BETA20', 'BETA60', 'IVOL20', 'AMIHUD20', 'TURNOVER', 'MARKET_CAP')


def cs_rank_centered(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """对 Class 1 个股 Alpha 做逐日截面居中百分位秩 [-0.5, 0.5] 标准化。"""
    out = df[['stock_code', 'date']].copy()
    g = df.groupby('date', sort=False)
    for c in cols:
        out[c] = (g[c].rank(pct=True) - 0.5).astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-cut', default=W.DEFAULT_DATA_CUT)
    ap.add_argument('--max-alpha', type=int, default=80, help='Class 1 正交个股Alpha上限')
    ap.add_argument('--max-interact', type=int, default=25, help='Class 4 门控交互因子上限')
    ap.add_argument('--max-sector', type=int, default=3, help='Class 3 行业广播状态上限（严禁超3个）')
    ap.add_argument('--out-dir', default=str(RESULTS / 'lgb_ready'))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载底层面板与 Class 2 个股栖息地底座
    panel = load_panel(data_cut=args.data_cut)
    ensure_local_fields(panel)
    base_df = panel[['stock_code', 'date'] + list(HABITAT_COLS)].copy()
    base_df['LOG_MV'] = np.log(np.maximum(base_df.pop('MARKET_CAP').to_numpy(dtype=float), 1.0)).astype(np.float32)
    habitat_features = ['BETA20', 'BETA60', 'IVOL20', 'AMIHUD20', 'TURNOVER', 'LOG_MV']

    # 2. 合并 Class 1 正交个股 Alpha (来自 build_top.py 导出的正交池)
    alpha_path = RESULTS / 'vwap_orth_alpha_pool.parquet'
    alpha_features = []
    if alpha_path.exists():
        adf = pd.read_parquet(alpha_path)
        adf['date'] = pd.to_datetime(adf['date'])
        adf = adf[adf['date'] <= pd.Timestamp(args.data_cut)]
        acols = [c for c in adf.columns if c not in ('stock_code', 'date')][:args.max_alpha]
        adf_norm = cs_rank_centered(adf, acols)
        base_df = base_df.merge(adf_norm, on=['stock_code', 'date'], how='left')
        alpha_features = acols

    # 3. 合并 Class 3 行业状态代表 (严格限 2~3 个，防同行业 Top5 抱团)
    sec_path = RESULTS / 'gates_search' / 'approved_sector_states.parquet'
    sector_features = []
    if sec_path.exists():
        sdf = pd.read_parquet(sec_path)
        sdf['date'] = pd.to_datetime(sdf['date'])
        sdf = sdf[sdf['date'] <= pd.Timestamp(args.data_cut)]
        scols = [c for c in sdf.columns if c not in ('stock_code', 'date', 'sector')][:args.max_sector]
        if 'sector' in sdf.columns and 'stock_code' not in sdf.columns:
            from field_catalog import sw2_mapping
            base_df['_sec'] = base_df['stock_code'].map(sw2_mapping())
            base_df = base_df.merge(sdf[['date', 'sector'] + scols],
                                    left_on=['date', '_sec'], right_on=['date', 'sector'], how='left')
            base_df.drop(columns=['_sec', 'sector'], errors='ignore', inplace=True)
        else:
            base_df = base_df.merge(sdf[['stock_code', 'date'] + scols], on=['stock_code', 'date'], how='left')
        sector_features = scols

    # 4. 合并 Class 4 显式门控交互因子 (严禁再做单日横截面 Z-score/Rank!)
    inter_path = RESULTS / 'approved_interactions.parquet'
    interact_features = []
    if inter_path.exists():
        idf = pd.read_parquet(inter_path)
        idf['date'] = pd.to_datetime(idf['date'])
        idf = idf[idf['date'] <= pd.Timestamp(args.data_cut)]
        icols = [c for c in idf.columns if c not in ('stock_code', 'date')][:args.max_interact]
        base_df = base_df.merge(idf[['stock_code', 'date'] + icols], on=['stock_code', 'date'], how='left')
        interact_features = icols

    all_features = alpha_features + interact_features + habitat_features + sector_features
    feat_file = out_dir / 'lgb_features.parquet'
    base_df.to_parquet(feat_file, index=False)

    manifest = {
        'data_cut': args.data_cut,
        'n_total_features': len(all_features),
        'features_all': all_features,
        'by_class': {
            'class1_stock_alpha': alpha_features,
            'class2_stock_habitat': habitat_features,
            'class3_sector_state': sector_features,
            'class4_gated_interact': interact_features,
            'market_broadcast_1xT': [],  # 铁律：永远为空！
        },
        # 供下游训练脚本读取：以下列严禁在训练前做逐日横截面标准化
        'skip_daily_cs_norm_cols': interact_features + sector_features,
    }
    man_file = out_dir / 'features_manifest.json'
    man_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"导出完成: {len(all_features)} 列特征 -> {feat_file}")
    print(f"Manifest -> {man_file}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

```

---

## 五、 重构后的日常标准工作流（SOP 4 条命令跑通全局）

当大模型按上述 6 个 Stage 完成重构后，你平时的研究与生产只需要极其清爽的 **4 条命令**：

> **频率契约（门控筛选与日常因子搜索彻底解耦）**：
> - **低频离线泳道**（每`data_cut`一次，通常月级）：`build_gate_bank.py` → `certified_gate_bank.json`（bank-v1：G0/中性度/G4簇代表/G2G3已做完、s锁定）+ `direct_sector_states.parquet`（D8豁免行业状态）。旧版按`bank_v1_<data_cut>.json`归档，禁静默覆盖。
> - **日常高频泳道**（每次研究迭代）：步骤一~二→四~六全部**不包含任何门控筛选**——只经`--gate-bank`只读消费 bank 的门控集合与 s，BN 均值快评/进化搜索/正交排榜/偏科选择/三算子 F1 检验/落盘/组装均不触碰 G2/G3 重算。
> - 唯一例外：`run_gate_pipeline.py` 本体仍保留完整筛选能力，但只在 bank 刷新/诊断时使用；不传`--gate-bank`时维持历史行为（兼容旧复现）。

1. **当你有新灵感、新论文或 Alpha101 因子时**：
直接在 `custom_factors/` 下写 Python 函数 `@register_py_factor` 或公式 `register_formula_factor`。
2. **步骤一：运行个股进化搜索 + 自定义因子评估**
```bash
python run_search.py search --n 60 --seed 42

```


*(默认 `local-vwap` 口径，搜索期用`rank_sia`排序，重排期用`lgb_score T_ref=0.3`，快评不加通道C豁免；`--include-custom`同步评估自定义因子。)*
3. **步骤二：运行全局正交中枢排榜（汇总 Mined + Custom + Refs）**
```bash
python build_top.py --top 200 --keep-pool 80 --keep-base 20

```


*(自动完成 `|ρ| <= 0.7` 逐日截面正交化，输出主力个股Alpha池与20个机制平衡门控靶标 `vwap_top20.json` + `vwap_top20_factors.parquet`，不删除base20。)*
4. **步骤三（低频，每data_cut一次）：构建认证门控字典**
```bash
python build_gate_bank.py --max-market 8 --max-sector 5 --max-sector-direct 2

```


*(输入短名单+sector G2G3+registry+G4簇复用；输出`certified_gate_bank.json`（s锁定）+`direct_sector_states.parquet`。日常不再重跑筛选。)*
5. **步骤四：偏科池选择与三算子交互合成（日常）**
```bash
python select_specialists.py --rank-lo 21 --rank-hi 150 --t-min 2.5 --max-factors 60
python f1_batch.py --cands results/merge_v2/specialist_cands_gain.json --form gain --base-panel specialist --gate-bank results/merge_v2/gates_search/certified_gate_bank.json --out results/merge_v2/gates_search/f1_specialist_gain.csv --tag spgain
python f1_batch.py --cands results/merge_v2/specialist_cands_flip.json --form flip --base-panel specialist --gate-bank results/merge_v2/gates_search/certified_gate_bank.json --out results/merge_v2/gates_search/f1_specialist_flip.csv --tag spflip
python f1_batch.py --cands results/merge_v2/specialist_cands_switch.json --form switch --base-panel specialist --gate-bank results/merge_v2/gates_search/certified_gate_bank.json --out results/merge_v2/gates_search/f1_specialist_switch.csv --tag spswitch
python f1_family.py --from-cands results/merge_v2/specialist_cands_gain.json --family F1v3 --form gain --out results/merge_v2/gates_search/family_manifest_F1v3_gain.json
python judge_f1.py --family results/merge_v2/gates_search/family_manifest_F1v3_gain.json results/merge_v2/gates_search/f1_specialist_gain.csv

```


*(top20直入Class1；偏科池走gain/flip/switch三式，sector线同理用`f1_sector.py`；三式分开跑FDR，严禁合并。)*
6. **步骤五：合格交互落盘（`materialize_interactions.py`）**
```bash
python materialize_interactions.py --judged results/merge_v2/gates_search/f1_specialist_gain_judged.csv --cands results/merge_v2/specialist_cands_gain.json --base-panel specialist --label-max 2025-12-31

```


*(只实例化 `h3_pass & fdr_sig` 行；flip 死区仅 u 有限处填 0.0；无通过行时落空表。性能注意：面板须先按 date 稳定排序再算日内秩，否则 `blockwise_rank` 碎块数≈行数直接卡死——本脚本已内置排序。)*
7. **步骤六：一键组装 LightGBM 训练特征集（Stage15桥接待定）**
```bash
python export_lgb_dataset.py --max-alpha 80 --max-interact 25

```



你可以直接将本回答复制给你的编码大模型（如 Claude Code 或 Cursor），让它按照 **第三节的删改清单** 和 **第四节的 Stage 1 ~ Stage 6** 逐一执行重构。如果某个 Stage 在执行时需要更细的某一个文件全量替换代码，随时告诉我！

---

## 六、 待定问题（Owner后续再定，本期只记录不实施）

1. **Stage15桥接（后续零拷贝方案，已定方向未实施）**：不复制7.3GB `factors.parquet`；`export_lgb_dataset.py`只输出merge新增列`lgb_features.parquet`（~100MB），Stage15侧加`--extra_factors_path`做内存left merge（`_SHAPE_FEATURES`与基线列保留，`features_sha1`自动防错配）。生产训练入口改动待单独立项。
2. **F1v3参数（冻结为预注册默认，不做同样本网格调优）**：`gain`系数`0.5`、`clip(0.2,1.8)`、底仓倾斜`0.25`（现仅作switch辅助列）、`BLOCK_BOOT=5/2000次`、`PLACEBO_BLOCK=20`冻结；后续变更须开新族版本并用样本外理由，不许回测调参。`flip`中性死区`|g|<0.10`置NaN（落盘填0.0），阈值0.10/0.15二选一已按Owner指令取0.10。
3. **门控扩容与bank刷新（冻结）**：54规格不扩容；`certified_gate_bank.json`仅随`data_cut`推进刷新一次，旧版按`bank_v1_<data_cut>.json`归档保留，禁止静默覆盖。
4. **VWAP口径与通道C（冻结）**：维持`vwap_formula_only`，masked版本另开label；`rank_sia`/`lgb_score`分工不变；快评永久不加通道C豁免（偏科打捞已由`select_specialists.py`承担，避免双重打捞的选择偏差；#150名开外的栖息地效应仍 uncovered，特此记录）。
5. **面板指纹隔离（冻结为永久规范）**：`ensure_extended_fields`与`load_panel(data_cut=None)`已过回归，作为永久规范。
6. **自定义因子PIT审计（已闭环）**：`custom_registry.audit_pit`复用PLAN §10.3时间穿越扰动思想（3个随机T，T+1后改随机数重算，断言≤T值`allclose`不变），warn-only不阻断；`--include-custom`计入`full-cap`/检验族不变。
7. **sector偏科预筛（下一步，不阻塞主链）**：bank仅2个行业门控，无需131行业逐回归；后续对#21~150候选算2门控全截面无置换ΔRankIC快速预筛（`fastranks`约260列），`|t|>=2.5`送`f1_sector --form gain/flip`跑200次置换。预筛只作discovery分流，最终FDR以预注册族清单为准，不得把无置换t当证据。
8. **switch静态对照（已统一）**：市场/行业统一用单因子静态等权共识`0.5*u_off+0.5*u_def`作主判定基准；市场底仓倾斜保留为辅助列。
9. **direct states数量（冻结）**：`|rho|<0.7`贪心选出几个算几个（实测1个），不为凑2~3放宽阈值；上限3、默认2、`export --max-sector`默认0三件套不变。

---

## 七、 实施状态（2026-10-15 merge内部重构已落地，Stage15桥接待定）

- Phase1：`boundary._read_parquet_bounded`物理隔离、`evaluator.load_panel(data_cut=None)`兼容+`ensure_extended_fields`、`custom_registry`+`custom_factors/`已实现；`audit_cut_paths 9/0/1NOTE`、`selftest_profiles 10/10`通过。
- Phase2：`build_top --keep-pool/--keep-base/--include-custom/--export-panel`+`vwap_top20.json`（family限4均衡）+可选`vwap_top20_factors.parquet`；门控/F1 `—base-panel base20|vwap_top20`（默认base20）；`run_search --include-custom`（公式型走全量、Python型走custom通道记分母）。
- Phase3：`f1_batch/f1_sector --form gain`为F1v3_gain新分支（sector单因子ΔIC、market底仓倾斜，200次块置换，精确重算）；`g/t`冻结逻辑零改动；输出不得与g/t合并FDR，待敏感性分析后冻结参数。
- `export_lgb_dataset.py`为merge内部版（Class1+4+栖息地，Class3默认0，1xT恒空）；Stage15桥（`factors_with_merge_cand`/`bridge_stage15`）未做。
- 待定校准仍在§六：通道C、VWAP auto-fail、门控扩容、Stage15桥（F1v3参数已冻结为预注册默认）。

## 七点二、 V4.2增量（2026-10-15晚，Owner四项裁决落地）

- 离线bank：`build_gate_bank.py`（短名单+G4簇文件复用+s锁定+G2符号，`certified_gate_bank.json bank-v1`）；`direct_sector_states.parquet`（sector greedy `|rho|<0.7`，默认2上限3，Owner豁免D8）；日常F1/`run_gate_pipeline`经`--gate-bank`只读消费（过滤+s覆盖，不传保持历史）。
- 偏科靶标：`select_specialists.py`（vwap rank#21~150 × bank市场门控，`slope_test`精确t，`|t|>=2.5`且无条件`<0.05`；高低异号建议flip；switch对为趋势轴×波动/分布/流动轴×bank前2门控，上限12）；输出`specialist_pool.json`+值面板+三分form cands+`F1v3_<form>`族清单直连。
- 三算子：`gain`（已落地不动）+`flip`（`u*s*g`，`|g|<0.10`死区NaN弃权（落盘仅u有限处填0），market单因子ΔIC/sector单因子ΔIC）+`switch`（`p*u_off+(1-p)*u_def`，market/行业统一单因子静态共识主判定，`tilt_delta`仅辅助列）；权重函数收敛`h3_eval`三helper，`selftest_interaction_forms.py` 7/7（含F2死区用例）；`f1_family --from-cands`建清单，`judge_f1`零改动；`--base-panel`加`specialist`（复合键排除）。
- 落盘：`materialize_interactions.py`已实现并合成验证（2特征、5,384,420行、flip死区填0共439,892格、约1分钟；面板内置date稳定排序——教训：`blockwise_rank`要求日块连续，否则碎块数≈行数直接卡死）；`fdr_sig=0`落空表（分支畅通已预验证）。
- `export_lgb_dataset.py`优先读`direct_sector_states.parquet`（兼容旧名）；回归：AST/导入/`--help`/profiles 10/10/audit 9/0/1NOTE/forms 7/7全绿。
- bank已实跑：`certified_gate_bank.json bank-v1`（market 3：turn_med_accel/amt_hhi_chg s=-1、limit_ratio_ratio s=+1；sector 2：sec_amp_med/sec_xs_disp s=-1，delta≈-0.02045与塌缩数值吻合）；G4复用37簇文件去重；direct states贪心`|rho|<0.7`选中1个（第2个与之共线被排除，符合预期）。新族清单`--from-cands`→`--validate`闭环ok:true。specialist全量待`build_top` flat产物（out/下暂无）后再跑，t_slope数学已合成验证。
- V4.2收敛（Gemini终审+Owner指令）：flip市场单因子死区`|g|<0.10`；switch市场/行业统一单因子静态共识主判定（tilt转辅助）；F1v3参数冻结为预注册默认；`audit_pit`闭环——实跑教训：`paper_idio_mom_20d`因求值只走`ensure_local_fields`报`BETA20`告警，已修为`ensure_extended_fields`超集（合成验证132/200有限值，NaN为warm-up正常）；sector预筛走2门控无置换ΔRankIC（下一步）；Stage15零拷贝桥接已定方向未实施；§六9项中#3#4#5#9冻结，#2#6#8代码闭环，#1#7下一步。
- 首次实跑记录（2026-10-15，`build_top --top 200 --keep-pool 80 --include-custom --export-panel`）：refs 1,672行/台账候选4,403/合选345；评估实算206个（242个台账复用）33分钟，非卡死系正常重负载，被Owner叫停（后台PID 22260已杀）；副作用按只追不删保留——`ledger.jsonl`新增约412条`local`事件（重跑自动复用跳过）+`out/value_cache/a2d87b758d02824e/`缓存+boundary_log 2条v4check。正交扫描及后续Export未到达。
- 提交：V4.2代码+文档已commit `6684a3e`（21文件 +2972/-61），产物（certified_gate_bank.json等）按gitignore不入库。

---

## 八、 V4.3分期计划（Owner已拍板：V4.3a→V4.3b→V4.3c依次执行；本节只定计划，代码未动）

### 已裁决事项（Gemini两处认同 + Muse两处修正，均有磁盘证据）

1. **市场G2主刀是`4折全同+CI不含0`，不是`same_frac≥0.60`**（证据：`pipeline_all.csv` 40门控全跑完，`same_frac`多数量过0.6如`newhigh_ratio_accel` 0.91/3折仍挂，`turn_med_level` 0.78/4折挂于CI；另4门含最强的`newhigh_ratio_chg`死于G0中性门）。风格对冲是同一机制的另一面：`median_b`被拉到0附近（如`xs_iqr_ratio` 0.000079）→折摇摆+CI跨0。结论：放行逻辑不变，改**判定分母**（见下3）。
2. **行业bank当前是2门控×5因子冒烟残余**（证据：`g2g3_sector.csv`仅2行，mtime与panel差2分钟；`screen.csv`14行业门控全eligible，12个从未跑G2；且`gate_g2g3_sector.py:116` `max(s·vv,0)`截断致`delta≈-0.02045`、s锁死-1）。结论：V4.3a废弃此CSV重算。
3. **市场G2改良走先验机制分簇，拒绝`|t|≥2.0`响应子集**（后者=同一统计量先选后测，选择偏差）。簇源直接用`roles.csv`现成`group`列（8组：volatility_and_liquidity 30/directional_alpha 28/momentum_and_position 28/risk_and_exhaustion 15/t2_promoted 12/tail_protect 6/sector_context 3/capital_flow_risk 2；`t2_promoted`为晋批次标签，检验时单独成簇并注明）：**任一簇内4折同号+簇内`same_frac≥0.65`+CI不含0即认证**，记录受益簇。阈值0.65与孪生阈值同属预注册冻结。
4. **W4（可交易掩码）修正**：掩码代码已存在（`fitness.py:87-90`，`universe=True`时`is_can_buy==1 & is_st==0 & ~一字板`），但全链路零启用（grep `universe=True`零调用）。故V5动作是"**开掩码+台账基线切换**"，不是新标签；开启即改变全部历史指纹可比性，须配基线版本号。

### V4.3a：行业G2无塌缩重算 → bank-v2（纠偏，不动市场；代码已落实，待服务器全量跑）

- `gate_g2g3_sector.py`（已改，未跑全量）：F1改`gain`式`u·clip(1+0.5·s·g)`（与F1v3_gain同式），含4修复——①门控中心化`g=2p-1`（面板`p∈(0,1]`，旧式权重恒≥1只有放大；市场`_g`列/f1_sector早已中心化，仅sector G2漏网）；②`u=d*(pct-0.5)`外层预计算（消Level-Leak+去循环内重复`to_numeric`，与f1_batch/f1_sector同定义）；③`load_panel`按date稳定排序+对齐断言（`factors.parquet`为stock-major行序，`PanelRanker.day`为date-major码，旧代码`m`行与`day`码错位——bank-v1行业脏数据的第三病因，比截断更致命；两门控delta撞到1e-5即旁证）；④G3双侧主判据（新族预注册`g3_rule=two-sided`，方向由bank s锁定；旧`form="t"`单侧冻结规格保留为诊断列，旧CSV/台账行不动，市场`gate_g2g3.py`+D4自测不动）。`--ledger`默认生产台账（新`form="gain"` caliber区分；`--no-g3`不写台账）。
- 合成验证（已过）：权重对称1.45/0.55无清零；常数因子与旧式同秩（无退化）；均匀信息因子delta≤0（不乱奖）；偏科因子d=+1/-1双向均为delta=+0.0795（定向修复后对称检出）。
- 三联动修复（Gemini抓出+Muse验尸确认，同源bug类）：①`f1_sector.py`——`m`按date稳定排序+对齐断言（stock-major错位同源）；gain/flip分支`u`改调`f1_batch._centered_u(xv)`（旧代码对已乘d的z排名后再乘d，d=-1负负得正方向全反；base20有16/20个d=-1）；旧g/t分支冻结公式不动，只吃排序修复。②`select_specialists.py`——flat `direction` 0/1映射`_dsign`（0→-1；`refs_cond_t1.py:98`同口径），否则反向因子`"d":0`使落盘全零列；pool/cands/dirmap三处统一。③`materialize_interactions.py`——sector分支`prow.reindex(udays)`（sgates从2020-01起，不补则门控错位1.5年；合成验证：未修取-0.8，已修取0.8）。④自修：`gate_g2g3_sector`两处`_centered_u`调用改传原始值（helper本身无错，错在传入了`d*xv`）。`H.block_placebo`本来就是双侧，F1消费侧无盲区。
- 待跑（服务器）：14 eligible门控全量G2（约60分钟离线）→ 按通过者`--g3-gates`跑G3；`build_gate_bank.py`输入换新CSV，bank升`v2`（v1归档），`direct`重选（目标真2~3正交，`|rho|<0.7`不变）。
- 验收：`g2g3_sector.csv` 14行；`delta_median`脱离-0.020塌缩带；bank-v2 sector≥2；回归7/7不动。
- 追加三修复（已落实`5a6f1ed`）：`f1_batch`漏排序（旧g/t经`_prepared`/日期键免疫，只有新三式中招）+对齐断言；`select_specialists`个体`s_pair`（有向斜率符号锁定，specialist面板时bank只过滤不覆盖）；`materialize`三批合并落盘（空通过不清空）+`build_gate_bank` sector base/op解析。
- 全仓审计结论（已落实）：生产链其余`d`写入方全源自`base20.json`（±1安全）；中心化其余路径全为`2p-1`；日期其余全为日期键映射或自带排序。诊断工具（corr_compare/stability/cond）自洽，无新增修复。
- 数据状态更新：`factors.parquet`已恢复262列完整版（Owner确认，mtime 10-05），store第四源=136列全量，sector G2按124 STOCK全跑。155列警示解除。
- 漏网三修复之上再补四修复（已落实，待服务器跑）：①habitat估值断路——`val_store`只含LGB前150名，Class2低LGB因子永不进入；修为机制家族过滤（context/vol/shape/M7/M14/store）+上限150按需补算，与扫描值合并为`hab_vals`。②去重口径——席位/股性池改窗内逐日截面Spearman均值（与扫描矩阵同口径），禁500万行混合corrcoef。③data-cut铁律——`build_top`加`--data-cut`（默认2025-12-31），`load_panel`/`DataContext`/`pos_me`/`B.read_factors`全链路钳制（旧默认END='20260821'曾放行2026）；panel store列批量单读。④分簇接线——`run_gate_pipeline --by-group`写`median_b_eff/slope_group_*`列（`gate_g2g3.py`独跑是断链的）；`build_gate_bank`读eff锁定s并升`bank-v2`（覆写前旧版自动归档）。合成验证：孪生体被store优先丢弃、bank-v2归档+s取簇符号（全池-0.0002→簇+0.009锁+1）。
- Owner三拍板（已生效）：G3双侧（新gain族）、habitat通道默认开（`--no-habitat-track`对照）、262列按全量跑；`export_lgb_dataset`最后一步显式`--max-sector 2`。
- 全仓8点收口（已落实）：①direct长宽表——bank侧pivot宽表（中心化2p-1，列即gate_id），export侧长/宽兼容（旧长表现场转宽；2门控以上旧路径会1254万行爆炸）；②四源helper收敛——`decorrelate_top.factor_full_values`统一6处手抄（select specialist双循环+build_top cond/dom/面板；cond_supplement诊断脚本暂列已知缺口）；③habitat断路/④截面去重/⑤data-cut/⑥分簇接线（上一条）；⑦`audit_pit`衍生列缓存盲区——扰动后强制drop 9列重算，合成验证`VWAP.shift(-1)`作弊3探针全抓（旧实现100%逃逸）；⑧`_store_aligned`全局单次调用+`_centered_u`收敛至`fastranks.centered_u`（f1_batch别名保留，sector/switch分支改调）。脚本级冗余（decorrelate_top/cond_supplement/cond_regime_ic等）按零删除静置，decorrelate_top/cond_supplement仅保留被import的函数。
- 合成验证：centered_u三处同值；audit好通过/作弊抓；长宽pivot正确；helper四分支；孪生优先；bank-v2归档。supplements全绿、forms 7/7。长跑零启动（用户服务器跑）。
- 全链路4断路收口（已落实）：①`mined_prefilter`并集——台账`local`事件记`habitat_score`，prefilter取sia-top200 ∪ habitat-top50（合成验证h08/h09低sia高habitat被捞入）；②面板全量落盘——`vwap_top20_factors.parquet`=`order[:keep_pool]`+席位（json同步扩全集供F1方向表，门控基准仍取前keep_base）；③specialist窗口`--rank-hi 400`/`--max-factors 150`+sector三式cands及`F1v3_sector_*`清单（bank sector s组级预注册）；④direct全序贪心（g2_pass优先+|delta|，14/14全测故无需screen未测量回填，诚实排除未测）+`--max-sector`默认2（>3告警）。
- 还差的长跑（服务器）：面板重算63规格→pipeline `--by-group`→sector G2/G3全量→bank-v2→build_top全量→specialists→三式F1（market+sector双线）→judge→materialize→export（`--max-sector 2`）。

### V4.3b：行业基础量+3 → bank-v3（扩池，市场不动；代码已落实`182183f`，面板重算待服务器）

- `gate_search_space.py`：`SECTOR_BASES`加`sec_mom_xrank`（行业20日累计跨行业分位，仅level）、`sec_vol`（行业20日收益波动）、`sec_breadth`（行业内上涨占比）；涨停集中度需先验证涨停标记字段可用性，不可用则本期放弃。`sec_ret_xrank`（单日）保留，新mom为多日梯度，不重复。全程索引对齐（不reset错位），合成验证因果性通过。
- 待跑（服务器）：sector面板重算（规格54→63）→ V4.3a同口径G2 → bank-v3。
- 验收：新基础量覆盖率≥80%（G0口径）；bank-v3 `direct`稳定2~3个。

### V4.3c：store第四源 + 双向去重闭环 + 机制分簇G2（代码已落实，待服务器跑）

- `store_sync.py`新建：与快评**同构**fast面板上算存量列vfp（同函数同参同seed，行序无关可比），`store_vfps_<data_cut>.json`缓存（factors sha换代自失效）；`run_search`启动自动并入`seen_vfp`（`--no-store-sync`可关；失败只告警不阻断）。注意：现155列版下store可用仅49列（roles 136名中87名不在库），262列恢复后自动扩到全量。
- `build_top.py`加`eval_store`分支（同`DataContext`双标签口径，lib='store'，不写台账）+`--include-store`（默认开）+正交扫描孪生双遍（先扫store占位，非store与已选store孪生`|ρ|≥0.95`丢弃留store列名；0.70~0.95贪心择优已有规则覆盖）；`select_specialists.py`零改动（读flat自动获存量偏科因子）。
- `promote_to_store.py`新建：显式离线回写（列冲突拒绝、备份+`roles.csv`同步+作废vfps缓存，dry-run默认）。合成dry-run通过。
- 栖息地尖兵席位：`habitat_seats`（窗内cond口径与LGB同窗，`hot5_q80 |t|≥2.5`+ndays≥10，正交`|ρ|≤0.7`，配额10）进面板/json/`habitat_seats.json`；`export_lgb_dataset`席位强制收录（专属配额不占max_alpha）。快评通道C豁免禁令不动。
- Class2股性榜单：`habitat_pool`（精确`habitat_score=jsd+0.5|mkt_up-mkt_dn|`，正交Top10）→`vwap_habitat_pool.parquet`；`export_lgb_dataset` Class2=6固定底座（原值冻结）+股性池（居中秩）+manifest `class2_habitat_pool`。
- `gate_g2g3.py`加`--by-group`（默认关；任一簇4折+CI+簇内`same_frac≥0.65`即认证并记录受益簇；t2_promoted单列）。合成验证分簇释放被抵消斜率（0.0003→0.0199）。
- 验收（服务器跑后）：store列全进flat；孪生对判留store；席位/股性池产物存在；`--by-group`输出group列。

### Class2独立搜索与评价（代码已落实，与V4.3c同批；三断路全部打通）

- 接线：`field_catalog.EXTENDED_ONLY_FIELDS`（VWAP/IND_EXCESS/BETA20/IVOL20/AMIHUD20；`ensure_local_fields`原4字段不动）→ `evaluator`惰性`ensure_extended_fields`分支 → `parser`/`grammar.validate`/`formula_uses_local`（上传硬拦）/`lint`尺度表全放行；`run_search` local_context改调超集。合成：m14 120个零无效、新腿35/120，m16 beta腿57/120。
- 生成：`m14` form6~9（市场/行业任意窗口Beta、市场残差平滑、残量信息比）+`m16_ctx` Beta/IVOL/BETA20腿；`_mutate`换字段池+`allow_documented`透传（默认关行为不变，C2C由adapter拦截）。
- 评价：`habitat_fast_score`（jsd+0.5|牛熊pooled反差；合成均匀0.0016 vs 偏科1.0128）+快评`--habitat-track`（默认开，名额full_cap×0.15）+`--habitat-frac/--no-store-sync`。
- 导出：见V4.3c股性榜单+export Class2。

### V5 backlog（parking，等V4.3首批Stage15实盘对拍后再启动；本期不展开设计）

- W1目标错位（全市场RankIC vs 极值二分类Top5）：需Stage15侧代码对拍（`multihorizon_35`定义、槽位规则我方未见源码，暂不背书具体数字），方向为尾部鉴别评分。
- W2残差/错题靶向搜索（`--label residual`、Top50头部分鉴别）：待W1对齐后立项。
- W3多元共线（VIF/残差比≥0.35）：便宜，可提前到V4.3c附带（终选环节加一道检查，不动搜索）。
- W4开`universe`掩码+台账基线切换；`sw2_mapping`静态漂移（PLAN D5已知）。
- W5单表7.3GB瓶颈：`--extra_factors_path`侧挂增量表（方向已定，V4.2收敛在案）。