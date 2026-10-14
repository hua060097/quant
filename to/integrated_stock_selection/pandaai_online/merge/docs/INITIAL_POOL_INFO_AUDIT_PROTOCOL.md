# Initial Pool Information Audit Protocol（初始因子池信息审计规范）

- 版本：v1.0（2026-09-30 落稿，待审查后冻结）
- 作者：DeepSeek（本仓库执行端）；审查人：GLM / ChatGPT（外部独立审查）；批准人：用户
- 状态：**待审批**。批准前不执行全量审计；批准后若修改任何判定条件，须在本文件追加 changelog 并全量重跑。
- 一句话定义：对 73 个 tier2 候选做**模型完全隔离**的信息审计，产出五级工程分类（EXACT / REPRESENTATION 复核 / HIGH-SIM 暂缓 / UNIQUE 入队 / UNRESOLVED 挂起）与 Information Map。**不删除任何候选；registry v2 冻结不动，结论挂 annex。**

---

## 0. 性质、边界与声明

- 本审计回答的问题：**"73 个候选中，哪些代表重复信息、哪些是不同表达、哪些值得进入实现队列"**。它是工程优先级审计，不是因子筛选，不是模型选择。
- **Initial Pool ≠ 工作池**：Initial Pool 是研究宇宙（registry 393 + 审计 annex）；工作池（CURATED，现 71）是模型层概念。不得把 Initial Pool 写入 CURATED。
- **模型侧结论归档不作废**：2×2 容量交互、四池 gain 构成、228 浅树一致性等结论与本审计解耦，归属工作池项目。
- **跨系统口径声明**：候选值来自 merge 引擎（cache/stock_features 面板），池值来自 factor_lab（factors.parquet）。本审计的 ρ 是**实现优先级证据，不是最终信息等价证据**；实现后必须用 factor_lab 真值复核孪生 + 卫生。
- **as-of 声明**：月末截面统计属全窗计算；本审计不宣称干净 OOS——它只做信息治理，不产生任何性能声明。
- **no-deletion**：审计不产出任何删除动作；未实现 ≠ 删除；registry 全量保留。

---

## 1. 输入与数据源（全部只读，运行前打印实际文件名）

| 输入 | 路径 | 用途 |
|---|---|---|
| registry | `pandaai_online/merge/out/INITIAL_POOL_registry_*.json`（最新；当前 v2 = 393 条，tier2 = 73：49 mined + 24 refs） | tier2 名单、family/channels/prior_verdict、roster 连接键 |
| flat | `pandaai_online/merge/out/ALIGNED_TOP2_*_flat.csv`（最新） | mined 候选的 `formula_cn` |
| ref specs | `pandaai_online/merge/decorrelate_top.py::load_all_specs()`（980 条） | refs 结构化定义：`base` / `transform` / `params.window` / `category_cn` / `formula_cn` |
| merge 面板 | `cache/stock_features_*.parquet`（`evaluator.load_panel`） | 候选值现算 + 月末行集 |
| 池值 | `factor_lab_data/factors.parquet`（228 因子列） | 池侧月末值 |
| value_cache | `build_top.py::_value_cache_dir`（按面板指纹） | 候选月末值缓存，避免重复现算 |
| 旁证（不参与判定） | `INITIAL_POOL_me_corr_*.csv` / `INITIAL_POOL_audit_*.csv` / `INITIAL_POOL_me_rankmat_*.parquet` | 池侧已验证产物；冒烟对照 + 广播列清单 |

---

## 2. Information Map 字段规范（本次覆盖 tier2；池侧缺项标 UNKNOWN）

| 字段 | 来源 | 置信度规则 | 说明 |
|---|---|---|---|
| domain | refs：spec `category_cn` 映射；mined：registry `domain_hint` | refs=medium；mined=low | 十个信息域之一 |
| economic_family | refs：`category_cn`；mined：registry `merge_family` | refs=medium；mined=low | 经济族（粗粒度） |
| base_measure | refs：spec `base`；mined：公式内基础字段集（CLOSE/VOLUME/AMOUNT/TURNOVER 等） | refs=medium；mined=low | 底层度量 |
| raw_horizon | refs：`params.window`；mined：公式窗口参数集合 | 单窗口=medium；多/无窗口=low | 可为多值列表；主窗口=最大值 |
| horizon_bucket | 由主窗口推导（≤5 / 5–20 / 20–60 / >60） | — | **仅存储，永不作判据** |
| state_class | 见 §3 检测规则 | 检测命中=high | MARKET_STATE / INDUSTRY_STATE / HYBRID / CROSS_SECTIONAL / UNKNOWN |
| representation.cs_transform | 公式含 `RANK(` → rank；否则 none | mined=medium；refs=由 spec | 截面变换 |
| representation.transform | refs：`transform`（smooth/compress/confirm_volume/raw…）；mined：外层函数 | refs=medium；mined=low | 时序/非线性变换 |
| representation.norm | refs：`params.norm`（如存在） | refs=medium | 归一化 |
| sign | registry 历史注记 `direction` | low | 仅报告，不参与判定 |

**`none` ≠ `unknown`**：无窗口概念的纯形态因子 → `raw_horizon = none`（none==none 可算匹配）；解析失败 → `unknown`（永不匹配，永不因它暂缓）。

---

## 3. 相关口径（钉死，引用已验证实现）

**观察单位 = 月末截面**（与 build_top 正交扫描、model_suitability、initial_pool_audit 全部一致）：

- **标准 A（个股类）**：各月末截面内逐列 pct-rank，跨月平均 Pearson = 月均截面秩相关。实现同 `initial_pool_audit.py` L113-122（秩）+ L138-156（跨月平均）。
- **标准 B（广播类）**：月末日期序列（每月一值）Spearman。实现同 `initial_pool_audit.py` L158-174。
- **混合坐标轴（广播列 × 个股列）不比较**（NaN），同 L169-171。
- **广播判定**：月末日期中"日内截面为常数"（std<1e-9）的日期占比 ≥ 0.90（同 L118-122）。
- **行业状态判定**：月末日期中"行业内为常数"的日期占比 ≥ 0.90（新增，用于 state_class；行业映射 `field_catalog.sw2_mapping`）。
- **候选×池**：在 `(stock_code, 月末日)` 交集行上按标准 A/B 计算（同 `model_suitability.py` L184-203 的 join 口径）；每月有效配对 < 50 跳过（同 L91）。
- **候选×候选**：merge 面板全额月末行上按标准 A/B。
- **报告与禁入**：Spearman 为主；辅助量（覆盖对数、Pearson）**只报告、不进判定**。
- 已知微小差异声明：秩按列在其有效行内计算后取配对交点做中心化点积；在两列均全覆盖时与逐对 rankdata 口径严格一致；存在 NaN 时有微小近似。

---

## 4. 决策矩阵（五级，预先注册；任何条件不得因模型结果改变）

| 级别 | 自动条件（满足其一） | 处置 |
|---|---|---|
| **EXACT_REDUNDANT** | 同公式（tier2 内部公式 hash/`formula_cn` 相同）；登记显式别名；人工确认数学恒等 | 不实现（登记保留） |
| **UNRESOLVED_REPRESENTATION** | 与池或 tier2 任一成员 `max|ρ| ≥ 0.9999`（含符号翻转对；符号翻转不自动判等，进人工复核） | 人工复核（暂不实现） |
| **HIGH_SIMILARITY** | `max|ρ| ≥ 0.95` **且** 五项维度全同且两侧标签置信度达标：domain、economic_family、raw_horizon（比值 ≤ 1.25 或 none==none）、state_class、representation（cs_transform+transform 同） | 暂缓实现（登记保留，可回滚） |
| **UNIQUE_CANDIDATE** | 无上述充分证据 | 进入实现队列 |
| **UNRESOLVED** | 求值失败 / 月末覆盖率 < 0.60 / 近常数（top 值占比 ≥ 0.999） | 挂起（注明原因） |

**保守路由（关键）**：任一维度标签 unknown 或置信度不足 → 不满足"全维度匹配" → **不暂缓**，进实现队列。宁可多实现，不误判重复。

**阈值（预注册）**：`REP_RHO=0.9999`、`HIGH_SIM_RHO=0.95`、`HORIZON_RATIO_MAX=1.25`、`MIN_PAIRS=50`、`BCAST_FRAC=0.90`、`COV_MIN=0.60`、`TOP_RATIO=0.999`。敏感性：annex 汇总 0.95/0.97/0.99 各档计数与实际 max|ρ| 值。

**禁入清单（违反即流程违规）**：Gain、LGB、A_official、mono、CV、OOS、JMI(label)、num_leaves、feature_fraction、topN 收益、池模拟收益、任何 quick/deploy 指标。

---

## 5. 输出（全部新文件，落 merge/out/）

- `TIER2_INFO_AUDIT_<ts>.csv`：逐候选一行，列为 §2 字段 + `eval_status` / `coverage_me` / `bcast_frac` / `max_rho_pool` / `who_pool` / `max_rho_t2` / `who_t2` / `sign_flip` / `cls` / `reason`（未被暂缓的具体原因）。
- `TIER2_INFO_AUDIT_pairs_<ts>.csv`：所有 `|ρ| ≥ 0.90` 对 + 全部 `≥0.9999` 对（含 axis=pool/t2、method=A/B、月数）。
- `TIER2_INFO_AUDIT_<ts>.md`：annex——分类计数、各级名单（UNIQUE 按队列建议序）、覆盖表（domain × {tier1, tier2}；范围外域声明）、声明、冒烟校验记录。
- 覆盖表范围外声明：fundamental / valuation / earnings 等域在本系统**无数据源**，标"范围外"而非缺口。

---

## 6. 后续（不属本审计，另行批准）

1. UNIQUE 队列 → 分批实现 → 生成 → **factor_lab 真值孪生复核 + 卫生** → Initial Pool v1.1 增量。
2. 池侧 family/representation 标签缺口定向补齐（仅对 UNIQUE 中 `max_rho_pool ≥ 0.95` 的少量候选）→ 重分类复核（把"无法暂缓"降为"复核后可暂缓"）。
3. 工作池 × 容量研究：另立项。

---

## 7. 版本与复跑

- 复跑触发：registry 更新 / 池列变化 / 口径代码变更 / 阈值变更。
- 任何阈值或条件变更：在本文件 changelog 记录 → 全量重跑 → 旧 annex 标注作废。
- changelog：
  - v1.0（2026-09-30）：首版。
