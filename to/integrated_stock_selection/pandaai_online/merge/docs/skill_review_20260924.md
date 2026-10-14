# pandaai-factor-merge skill 优化分析报告（2026-09-23，v1.1 修订于 2026-09-24）

> v1.1：经 skill 原作者（DeepSeek）核查与我方实测复核，修订 6 处（含撤回 3 条错误论断），
> 详见文末"修订记录"。初稿中已证伪的内容在正文以 ~~删除线~~/（v1.1 …）标注保留，便于追溯。
>
> **实施记录（2026-09-24）**：经用户确认，双方"认同"清单已实施（标签三元诊断不做；
> vwap 掩码/TP 口径/方向冻结按用户决定不动）。实施明细见文末"实施记录"节。

## Context

应用户要求审查 skill `pandaai-factor-merge`（SKILL.md + 引擎代码
`D:\.trae-cn\pythons\path\to\integrated_stock_selection\pandaai_online\merge`，约 8700 行 + 台账 17.3MB），
重点评估四个方面：① vwap 口径（为本地模型搜索因子）② 搜索组合机制是否有遗漏
③ 因子的真实金融含义 ④ 历史记录对未来搜索的助力。另参考外部前沿论文/开源框架吸取经验。

**用户澄清（修正初稿前提）：目前尚无真实可用的 LightGBM 模型**（advanced_quant_model.py
并非目标模型），因子现阶段只能依靠自身评估（榜2 闭式复合分等因子级指标）。

**本次仅交付分析报告与修改建议，不实施任何修改、不改任何代码。**（用户已确认：vwap 口径
暂不动、真模型对齐暂缓——相关建议保留在清单中供后续决策）

调研方式：3 个代码探查 agent（grammar 搜索空间 / fitness+pool 池机制 / 历史-知识闭环）
+ 1 个网络调研 agent（LLM 因子挖掘论文与框架，经 arXiv API 与 GitHub API 核实）
+ 我本人直接核读 field_catalog.py / fitness.py。

---

## 一、总体评价

这是一个完成度相当高的自研闭环引擎：三路径发现（机制搜索/LLM 提案/外部因子筛选）汇入同一台账、
两级评估淘汰、holdout 冻结方向、池级 ΔComb、多重检验计数、语义 lint、中文机制反解释——
这套骨架与 2025-2026 顶会论文（AlphaMemo/RD-Agent(Q)/Hubble）的共识结构高度同构。

主要短板集中在**"最后一公里"**：搜索空间有几条轴整体缺失（VWAP/OPEN/布尔门）、
自评估榜单缺少"与既有因子集互补性"的正交化环节、历史知识只在台账层面被消费
（无 UCB/无失败 veto/无精英回流）、平台远程反馈断链。以下分主题详述。

---

## 二、vwap 口径分析

### 2.1 标签定义现状（fitness.py:51-59）

`buy=open(T+1)`，`sell=0.7·TP(T+2)+0.3·TP(T+3)`，`TP=(O+H+L+C)/4`，`label=(sell-buy)/(buy+1e-12)`，
仅 cycle=3。docstring 自认：生产版 daily_atr_label.py 的 ST/涨跌停/amount<1e7/复权掩码**未实现**。

### 2.2 发现的问题（按严重度）

1. **停牌=行序位移的可交易性失配**（v1.1 修正：非"前视"）：`groupby().shift(-N)` 按行位移，
   停牌股票取到复牌日价格，等于假设停牌期间可持仓穿越。标签不含未来信息，属可交易性/口径问题
   而非前视偏差；平台侧大概率同样按行位移（c2c 校准差 ≤8% 间接支持），是否构成失配待核。
2. ~~无复权~~（v1.1 撤回）：数据为前复权（用户确认；独立复核 755 万行 pct_chg vs
   close.pct_change() 差异>2pp 仅 18 行 ≈0.0002%，无系统性除权跳空）。顺带发现 34 行 ≤-25%、
   40 行 ≥+32% 的极端收益（疑似新股首日/脏点，非除权）——反而佐证"buy 防护/脏点过滤"项的价值。
3. **无涨跌停/一字板约束**：T+1 一字板买不进、T+2/T+3 跌停卖不出的极端样本全留在标签里
   ——恰是高 IC 来源，系统性抬高 vwap 榜 IC/ICIR。
   （v1.1 注：全仓三套宇宙口径并存——c2c 大赛分析全样本对齐平台（勿动，排除反而破坏 ≤8% 校准）/
   独立 screen 默认过滤 / 池模拟剔除。vwap 掩码建议维持"暂不动、仅文档化"决定。）
4. **TP(OHLC4) vs 真 VWAP**（v1.1 重新定性：非缺陷，属执行匹配的经验问题）：用户执行为
   T+1 开盘买、T+2/T+3 分钟择时卖（止损止盈路径），OHLC4 的 H/L 保留盘中极值信息，对退出路径
   的刻画天然优于"全天均匀成交均价"；且 AMOUNT/VOLUME 有本引擎实证过的语义/尺度风险
   （amihud ×1e9 事故即 AMOUNT 语义坑）。**维持 OHLC4**；可选做"OHLC4 vs AMOUNT/VOLUME vs
   分钟真实成交"三元 RMSE/corr 诊断一锤定音。
   （v1.1 撤回"factors.parquet 已有 volume_weighted_price 列"——实测两个 factors.parquet
   216 列与 merge 缓存 40 列均无此列；generate_daily_features.py 中的同名特征是 20 日滚动
   量加权价，属另一特征体系，非当日 VWAP。）
5. **除法防零弱**：`buy+1e-12` 是加性偏置；极小 buy → 巨大有限 label 污染截面
   （`np.isfinite` 只挡 inf 不挡 1e6）。clip±2σ 发生在 zscore 之后，一个离群值先抬高 sd 把其他票压扁。
6. **vwap 方向 in-sample**（v1.1 降级：影响有限）：build_top.py 的 vwap 榜在全窗口自动定向后
   同窗报告；但榜2 的 LGB 复合分用 |RankIC|/|RankICIR|（方向不变量，refs.py:385-386 实测确认），
   方向选择仅二阶影响 rank_ic_win_rate 与报告符号，不影响榜内排序。优先级最低。
7. **cycle=3 硬锁** + 标签后延 3 天：有效评估窗实际截止约 2026-08-14。

### 2.3 澄清：目前没有真实可用的 LightGBM 模型（用户澄清，修正初稿论断）

- **用户澄清：advanced_quant_model.py 不是目标模型；当前没有真实可用的 LightGBM 模型。**
  因此"对模型有用的因子"现阶段只能依靠**因子自评估**——即榜2 的闭式复合分
  （refs.py:374-388：`|RankIC|×|RankICIR|×rank胜率×max(0,mono)×T_ref/max(turn,T_ref)`）。
- merge/ 目录 grep `lightgbm` 零命中是符合预期的：自评估口径本就不含模型，初稿据此推断
  "与生产模型脱节"的前提不成立。
- 该启发式作为因子级代理是合理的，但缺两类信息：
  1. **与既有候选集的互补性**：榜2 按分数排序，不惩罚"与已入选 vwap 因子高度相关"的重复信息
     （decorrelate_top.py 有去重流程但独立于榜2 排序）；
  2. **组合层面的交互价值**：单因子指标无法预见两因子组合才显现的价值——无需真模型，
     可用两两 IC 增量或岭回归等权组合做零算力代理（AlphaGen 池级 ΔIC 思想的简化版）。
- run_search.py L610-617 把 vwap_top 写 JSONL 是单向无 schema 的导出，暂无消费方——
  模型落地前维持现状即可。

### 2.4 建议（用户已决定暂不动口径，以下供后续立项）

- 口径修复（v1.1 收缩）：buy 极小值防护（`np.where(buy>eps,…,nan)`）、vwap 标签接可交易掩码
  （is_limit_up/is_can_buy/is_st 列现成，fitness.py L79-80 的 universe 逻辑可直接复用）、
  （可选，最低优先级）build_top 方向冻结。~~复权列检查~~（前提不成立，撤回）；
  ~~TP 换真 vwap 列~~（撤回，维持 OHLC4，见 2.2 第 4 条）。修复后 IC 数字会下降
  （剔高 IC 极端样本），需重跑 qs2-R741 式校准——这是"暂不动"的合理理由，但**至少应在
  SKILL.md 文档化上述已知偏差与三套宇宙口径**（c2c 全样本对齐平台 / 独立 screen 默认过滤 /
  池模拟剔除——"涨跌停口径我们好像改过"的记忆即源于三套并存）。
- 不依赖模型即可做的低成本改进（P2 级，建议纳入）：
  - 榜2 排序加入"与已入选 vwap 因子集的最大 |Spearman| 正交惩罚"，让榜单偏向互补信息；
  - 两两 IC 增量或岭回归等权组合代理"组合边际贡献"（零算力版池级 ΔIC）。
- 真模型落地后再立项（P3，暂缓，前提=存在可用模型）：
  - schema 化桥接：把公式因子值写入特征库；
  - leave-one-factor-out ΔAUC / 对模型 OOF 残差求 IC。

---

## 三、搜索组合机制分析

### 3.1 现有覆盖

13 个模板族（M1 动量/M2 流动性含 Amihud·量比·换手期限结构/M3 分布/M4 位置/M5 趋势/
M6 量价相关/M7 波动/M8 条件门/M9 复合/M10 形状含效率比·双均线·让利回吐·下行波动占比/
M11 状态门/M12 次序统计(D)/M13 技术指标(D)）→ 10 个机制轴；变异五档（改窗/换算子/子树替换/
改指数/换字段）+ 同类位点交叉；机制配额 + 腿库交叉重组 + 结构指纹查重。覆盖面在同类系统中
已属中上。

### 3.2 遗漏的组合（按影响排序）

1. **VWAP 模板族缺失**（v1.1 修正表述：是模板/变异轴缺失，非语法不可表达）：PLATFORM_FIELDS
   无 VWAP 字段，`(C-VWAP)/VWAP` 类不可表达；但 **AMOUNT/VOLUME 是合法表达式**，VWAP 乖离族
   今天就可用 `(C-AMOUNT/VOLUME)/(AMOUNT/VOLUME)` 表达——缺的是模板族与尺度政策处理。
   → 可探针 AMOUNT/VOLUME 虚拟序列 AVGN（注意 volume 单位手/股：若是手，AVGN=100×每股均价，
   水平比较需 ×0.01，且与"AMOUNT 分母显式 ×1e9"政策有交互，需单独探针）。
2. **OPEN 模板 0 覆盖**（v1.1 修正表述：OPEN 在 PLATFORM_FIELDS 中合法，grammar.py 中出现
   0 次，仅变异换字段可能偶然触及）：隔夜缺口（OPEN/DELAY(CLOSE,1)-1）、日内实体（(C-O)/O）
   类机制模板不可达——隔夜收益类是 A 股短周期文献强因子。修法是加模板族
   （GAP1/BODY 虚拟序列仅为便利，非必需）。
3. **截面变换轴极窄**：CS_OPS 仅 RANK/SCALE/ZSCORE，后两者 DOCUMENTED 默认禁用；
   无截面 zscore 模板、无中性化、无分位阈值扫描（仅硬编码 RANK>0.5 一档）。
4. **布尔组合不可达**：AND/OR 是 DOCUMENTED 且无模板生成，random_condition 只产单一 GT；
   GE/LE/MAX/MIN/COUNT 四个 VERIFIED 算子完全闲置。
5. **复合腿只有等权 RANK-ADD**：无减法腿/加权/非 RANK 多腿；腿库限 node_count≤12。
6. **变异缺**：无常数/门限微调（常数白名单仅 0/0.5/1/2/1e9）、无 NEG 翻转、无腿数变异；
   mutate 的 rate 参数是死参数（grammar.py:651）。
7. **回归残差不能进公式**：TS_REGRESSION 未实现（evaluator.py:341）；正交化仅本地诊断
   （resid.py，--resid-weight 默认 0）。
8. **精英不跨代**：单代内精英保留，跨代仅靠台账配额与腿库间接继承；grammar.py:708 的
   `elites` 参数从未被 search 使用。
9. 小项：TEMA/ADV 闲置；_axis_plan 注释（8+1 轴）与实际 10 轴不符；lint 缺"AMOUNT 分母
   必须×1e9"规则（政策只内嵌模板）。

### 3.3 建议

- 补 VWAP 虚拟字段 + M14_vwap 模板族（乖离/动量/C 相对滚动 vwap/CORR(RET1,AVGN)）、
  GAP1/BODY 虚拟序列与配套模板（P1，先本地后平台 probe）。
- 变异补 NEG 翻转/门限微调/腿数±1；GE/LE 进 random_condition；尝试 AND/OR 平台 probe。
- 复合支持减法腿；分位阈值进常数白名单（0.3/0.8）供门限变异。
- 拥挤防控：高频 struct_fp top-N 作黑名单，mutate 对高频子树降权（frequent-subtree avoidance，
  Alpha Jungle arXiv:2505.11122 的做法）。

---

## 四、因子的真实金融含义

### 4.1 现有机制

已具备：lint 语义规则库（量纲/数学域/稀疏并列/MEAN 陷阱七类）、explain.py 中文反解释、
知识库机制条目（validated/rule/failed/warning/candidate 分级）、Amihud 等有金融语义的模板族命名。
这在同类 skill 中是少见的优点。

### 4.2 缺口

1. **反解释不参与打分**：explain.py 只在报告层，"解释不出机制"不降权。
2. **假设-公式无一致性校验**：提案通道记录 hypothesis 字段但从不校验公式是否兑现假设。
3. **失败教训不拦截生成**：fail-sparse-tie 教训只在 intake 冒烟拦截外部提案，
   自家 grammar 生成的 M8/M11 离散事件候选无同样检查（2026-09-15 平台事故 F-B04/B08
   稀疏并列换手 88% 的教训没有进入生成约束）。
4. **KB 与搜索零交互**：run_search.py grep knowledge 0 命中；--add-json 新增机制只影响
   LLM few-shot，不改变撒种/变异/淘汰任何行为。
5. **KB 覆盖失衡**：实际 26 条（SKILL.md 写 18 条过时）；M1/M5/M6/M7/M10/M11/M12/M13
   八个族无机制条目，而这 8 族台账显示被测 1600+ 次（M13_tech 339、M8_gate 310、M1_return 287…）。
6. lint 无"AMOUNT 分母必须×1e9"规则。

### 4.3 建议

- **假设-公式一致性门**（AlphaAgent arXiv:2502.16789 / AlphaSchema 五元组
  Event/Context/Qualities/Direction/Output arXiv:2607.26642）：intake 时 LLM 对
  "假设↔公式"打 1-5 分，<3 不进全量评估——省回测算力又保金融含义。
- 反解释前置到评估层：shortlist 成员必须附 explain 输出，解释矛盾者降权。
- fail-sparse-tie 检查（截面独立值中位数<100）应用为生成期 lint 规则（不只 intake）。
- KB 自动回流：search 结束把低分/无效事件按 (family, lint-code) 聚合自动 append failed 条目；
  8 个空白族的机制条目补齐（每族 2-3 条含公式与教训）。
- lint 补 AMOUNT×1e9 规则。

---

## 五、历史记录对未来搜索的助力

### 5.1 现状

台账（5397 行，fast 2338/local 3047/ref:* 1844）驱动：canonical 哈希排除集、轴配额权重、
腿库 top40、rescore、多重检验分母（唯一 (公式,标签) 对 4303）。这是合格的"结果账本"。

### 5.2 缺口（按影响排序）

1. **轴配额无 bandit/UCB**（run_search.py:74-122）：`w=0.5+0.5×(best-ever/global-best)+0.25×未测`。
   无衰减/无不确定性项；**冠军轴永久锁 w=1.0，从未测轴 w=0.75 反而低于冠军 → 探索被系统性压制**
   （旧版"未测×3"被删后矫枉过正——SKILL.md 的"未测轴权重×3"表述也已过时，代码中不存在）。
   best[axis] 还混用 fast 窗与 full 窗两种 SiA 口径（乐观偏差）。counts[axis] 已统计，改 UCB 只差一步。
2. **失败不 veto**：高置信失败模式（如离散计数稀疏并列）无硬否决；正例软加分的非对称机制缺失
   （AlphaMemo arXiv:2606.20625 的 APV："负证据比正证据更跨周期稳定"）。
3. **精英/知识不回流生成**：generate_population 的 elites 参数闲置（见四.4/三.8）。
4. **结构指纹不落盘**：seen_fp 只在本次 run 内存有效（run_search.py:205-207）；跨 run 近克隆
   重新全评；台账 3093 id/2132 hash（重复 1.45×）；多重检验分母虚增。
5. **平台反馈断链**：ledger.py:4 声明 platform 事件但全仓 0 条 0 写入者；batch.py 只写 CSV
   不回填；平台 IC vs 本地 IC 校准只能人工读 CSV 手写 knowledge（仅 3 条）。
6. **台账越老越慢**：一次 search ≈6 次全量 JSON 扫（实测 0.61s/次；17.3MB 且按当前增速每月翻倍）
   ——与"历史越大越有用"方向相反。无索引/无 sidecar 摘要。
7. **refs 路径被排除**：refs.py 写 log_raw（无 tier1 键、family=ref:* 不在 AXIS_OF）→
   34% 台账事件对配额/腿库/rescore 全部不可见；rescore 实际不覆盖 refs（与 docstring 声明矛盾）。
8. 次要：腿库均匀抽样无 SiA 权重/无多样性约束；提案 family 固定 'proposal' 不反哺配额；
   holdout/decision 事件 6 条无写入者（遗留）；knowledge.ensure_seed 每次 load 整文件重写
   （破坏 append-only）；多重检验只打印不用（Bonferroni 在 n=4303 已无区分度，stability 仅 tiebreak）。

### 5.3 建议

- **轴配额改 UCB**：arm=(axis, family)，reward=近 90 天滑窗 mean(sia)（fast/full 分开归一），
  `w = mean + c×sqrt(ln(N)/n_arm)`；n≥20 且 mean≈0 的已证伪轴降权（RD-Agent(Q) arXiv:2505.15155
  的 bandit 调度思想）。
- **非对称记忆**：高置信失败（重复出现≥K 次的 (family, lint-code) 失败对）作生成期 veto；
  成功经验作软加分——置信门 `c=n/(n+κ)·|μ|/(σ+ε)`，warm-up 期回退纯台账（AlphaMemo §4.4）。
- **精英回流**：search 启动时从台账取 winner 公式 + knowledge.retrieve 查 (axis,tags)，
  喂给 generate_population(elites=…)。
- **sidecar 摘要**（ledger_summary.json：per-axis best/counts、hash→latest、struct_fp 全集），
  写事件时增量更新，search O(1) 读；struct_fp 写入事件字段。
- **platform 事件回填**：upload/run 后解析 candidates_*.report.csv 的平台读数 append 台账；
  --calibrate 拟合 platform≈f(local,turnover) 作 rank_score 先验。
- refs.py 补 tier1 键 + AXIS_OF 映射；腿库按 SiA 加权抽样 + 字段集去重；stability 升为可选
  硬门槛 `--min-stability`；BH-FDR 替代 Bonferroni。

---

## 六、其他工程问题（杂项）

- 三模块三种月末截面定义 + 三个时间窗（dedup 2023-06~/decorrelate 2021-09~/pool 2023-03~），
  "|ρ|>0.7"判据跨脚本不可比。
- evaluator.load_panel 不做 inf→NaN（pool_sim 有）；SLOPE 快路径要求 mp==w 否则 10-100x 减速。
- decorrelate_top.py：全零相关矩阵→去重静默失效；NaN 候选被误当高相关剔除；死代码 L138-140。
- pool_matrix.py L114-115 用键名子串匹配决定删谁（脆弱）。
- Comb 代理：age_months 死参数、pool_sim 的 decay 表未接线、NA 在 rawA≈0.056 饱和到 0.7
  → ΔComb 退化为纯 NC 竞赛；SiA（旧窗全宇宙静态快照）与 NC（新窗剔ST宇宙）不同窗线性相加。
- 贪婪 ΔComb **只加不换不删**：leave_one_out 已给出"谁拖后腿"但 greedy 不用；
  reserve 入选后不挪 members（重复考虑）；dirs/sia 字典污染（pool.py:146-147）；无 OOS/bootstrap。
- SKILL.md 事实修正：KB 种子 18→26；"8轴+文档轴"→10 轴；"未测轴权重×3"已不存在；
  "LGB复合分"应注明是启发式而非真 LightGBM；rescore 不覆盖 refs。

---

## 七、优化建议总表（供后续决策，本次不实施）

| 优先级 | 项目 | 回应 | 工作量 |
|---|---|---|---|
| P0 | platform 反馈回填 + --calibrate | 历史助力/校准 | 小 |
| P0 | ledger sidecar 摘要 + struct_fp 落盘 | 效率/重复挖掘 | 小 |
| P0 | refs 事件补 tier1+AXIS_OF | 历史助力 | 小 |
| P0 | SKILL.md 事实修正（含 vwap 已知偏差文档化） | 文档可信 | 小 |
| P0 | vwap 标签 buy 防护（bug 级） | 口径 | 极小 |
| P1 | 轴配额改 UCB + 已证伪轴降权 | 历史助力 | 中 |
| P1 | 精英/知识回流 generate_population(elites) | 历史助力 | 小 |
| P1 | 变异补齐（NEG 翻转/门限/腿数）+ GE/LE | 组合遗漏 | 小 |
| P1 | M14 VWAP 模板族（AMOUNT/VOLUME 表达式合法，尺度探针先行；虚拟字段可选） | 组合遗漏/vwap | 中 |
| P1 | 隔夜缺口/日内实体模板族（OPEN 进模板；GAP1/BODY 虚拟序列可选） | 组合遗漏 | 小 |
| P1 | ΔComb 支持 replace/remove + 池相关硬门槛 + decay 接线 | 池机制 | 中 |
| P1 | fail-sparse-tie 进生成期 lint | 金融含义 | 小 |
| P2 | 假设-公式一致性门（LLM 打分） | 金融含义 | 中 |
| P2 | KB 自动回流 + 8 族条目补齐 | 金融含义/历史 | 中 |
| P2 | 拥挤模板黑名单（负通道） | 组合遗漏 | 中 |
| P2 | stability 硬门槛 + BH-FDR | 统计纪律 | 小 |
| P2 | 榜2 排序加正交惩罚（并入排序层，与 decorrelate_top 分工）+ 两两 IC 增量代理（仅对榜2 top，控 O(n²) 成本） | vwap 核心诉求 | 中 |
| P3 | 真模型落地后：特征库桥接 + LOFO ΔAUC/残差 IC | vwap 核心诉求 | 大（暂缓，前提=可用模型） |
| 不建议 | GFlowNet/RL/MCTS 重型引擎；台账换 sqlite | 成本不值 | — |

（远期可选：AlphaMemo 式 AST-diff motif 记忆，工作量大不进近期。）

## 八、外部前沿借鉴来源（Agent4 经 arXiv/GitHub API 核实）

- AlphaMemo（arXiv:2606.20625，jarrettyu/AlphaMemo）：搜索台账先验 + motif 记忆 + 非对称 veto
- RD-Agent(Q)（arXiv:2505.15155，microsoft/RD-Agent）：知识森林 + bandit 方向调度
- Hubble（arXiv:2604.09601）：正/负双通道 RAG + family-aware 准入 + AST 沙箱
- AlphaAgent（arXiv:2502.16789）：AST 原创性正则 + 假设-因子语义对齐
- AlphaGen（KDD 2023，arXiv:2306.12964，ICT-FinD-Lab/alphagen）：池级 ΔIC 作 fitness
- Alpha Jungle（arXiv:2505.11122）：LLM-MCTS + frequent-subtree avoidance
- AlphaSchema（arXiv:2607.26642）：五元组语义 schema（金融含义对齐最结构化做法）
- AlphaEval（arXiv:2508.13174）：五维评价（有效性/稳定性/稳健性/多样性/可解释性）
- LLM-SR（arXiv:2404.18400）：LLM 出结构假设、数据拟合参数的分工原则
- quantskills 组织：skill-llm-alpha-generator（LLM→校验→GP warm-start→AlphaEval→解释）、
  skill-quant-factor-skill-factory（factor_index.json 显式查重）
- WorldQuant 101 Formulaic Alphas（arXiv:1601.00991）：量价类 alpha 大量使用 vwap 字段
  ——是"平台补 VWAP 虚拟字段"的直接文献支撑

## 修订记录

- **v1.1（2026-09-24）**：经 skill 原作者（DeepSeek）核查 + 我方实测复核，修订 6 处：
  1. 撤回"无复权"条目——数据为前复权（用户确认；独立复核 755 万行 pct_chg vs close.pct_change
     差异>2pp 仅 18 行 ≈0.0002%，无系统性除权跳空）；
  2. 撤回"factors.parquet 已有 volume_weighted_price 列"——实测两个 factors.parquet（216 列）
     与 merge 缓存（40 列）均无此列；初稿此条来自未核实的调研转述，**系初稿最大错误**；
  3. "VWAP 轴语法上不可表达"修正为"模板族缺失"（AMOUNT/VOLUME 为合法表达式）；
     "OPEN 不可达"修正为"模板 0 使用"（实测 grammar.py 词频：OPEN/GAP1/BODY=0、AMOUNT=6、
     VOLUME=12）；
  4. "停牌=前视"修正为"可交易性失配"（标签不含未来信息，非前视偏差；平台侧行为待核）；
  5. 撤回"TP 换真 VWAP"建议——标签职责是逼近生产 P&L 路径（T+1 开盘买、T+2/T+3 分钟择时卖），
     OHLC4 的 H/L 对止损止盈路径刻画更贴，**维持 OHLC4**；可选 OHLC4 vs AMOUNT/VOLUME vs
     分钟真实成交三元诊断；
  6. "build_top 方向冻结"降级为最低优先级（lgb_score 用 |RankIC|×|RankICIR| 方向不变量，
     方向仅二阶影响 win_rate 项与报告符号）。
  核查方法：pyarrow 读 parquet schema（两个 factors.parquet + merge 缓存）；755 万行收益
  一致性/极端跳空实测；grammar.py 词频统计；refs.py lgb_score 源码复核。
- 另确认涨跌停口径"三套并存"现状（c2c 全样本对齐平台 / 独立 screen 默认过滤 / 池模拟剔除），
  与原作者结论一致；vwap 掩码维持"暂不动、仅文档化"决定。

## 实施记录（2026-09-24，仅双方认同项）

| # | 实施 | 文件 | 验证 |
|---|---|---|---|
| 1 | vwap 标签 buy≤1e-6 脏行置 NaN（替换 buy+1e-12） | fitness.py | 合成用例（buy=0→NaN，正常行数值不变） |
| 2 | load_panel inf→NaN（与 pool_sim 对齐）；模块头陈旧注释修正 | evaluator.py | — |
| 3 | lint 新增 amount-scale 规则（AMOUNT 作分母无 ×1e9 → warning） | lint.py + selftest_supplements S2 | 命中/不误报 4 用例 |
| 4 | ledger sidecar（ledger_summary.json：byte_offset 增量、hash→labels、fps、tested_pairs、events_min、evidence_pool）；struct_fp 落盘（search fast/local 事件） | ledger.py / run_search.py | 真账 5397 事件全量对照等价（0.32s 建账，增量 0ms） |
| 5 | 轴配额改 UCB（fast/full 分窗归一、近 90 天滑窗、base=max(近窗,0.5×全期)、未测轴乐观初始化、证伪轴×0.3） | run_search._axis_plan | E4 合成用例 PASS；真实台账 UCB(0/0.3/1.0) 行为合理分化 |
| 6 | 精英回流（25% 撒种来自台账 winner 变异，--no-elite-seed/--elite-seed-frac）；腿库 SiA 加权抽样 + 结构指纹多样性 | run_search._seeds | 加权抽样 95% 命中高权重；各轴 winner 就绪 |
| 7 | 变异补齐：NEG 翻转、门限微调（0.3/0.8 进常数白名单）、复合腿数±1；GE/LE/LT 进 random_condition | grammar.py / field_catalog.py | 3000 次变异全过 validate；各分支可达性统计验证 |
| 8 | fail-sparse-tie 生成期拦截（截面独立值中位数<100 不进全量，--no-sparse-tie-gate 可关；与 intake 共享 sparse_unique_median） | run_search / intake | b0 E3 集成 PASS |
| 9 | KB 自动回流（auto_reflow：n≥8 且均 SiA<0.0005 → failed 条目，幂等）；ensure_seed 无变化不重写；8 族（M1/M5/M6/M7/M10/M11/M12/M13）机制条目补齐（种子 26→34） | knowledge.py | 幂等/保留自定义条目/检索命中验证 |
| 10 | 假设↔公式一致性机检（关键词↔字段/算子，warning 级不拒绝；LLM 打分留 P2） | intake.py | 4 用例（命中/一致/偏度/无假设） |
| 11 | Comb 代理接线 DECAY_TABLE（age_months 活化，默认 3 行为逐位不变）；refresh_member_sia（aligned_v1 窗同窗化，--refresh-sia opt-in）；greedy 试算字典不再污染共享 dicts；reserve 晋级显式标注（不自动改 pool_state） | pool.py / run_search | age3=旧值、age18 走 0.85/1.0 档；污染/晋级单测 PASS |
| 12 | decorrelate_top：零方差列相关 NaN（不再静默通过去重）、cnt==0 返回 None、死代码清除、NaN 显式保守剔除；pool_matrix 子串匹配改显式定义（顺带修复一处名实不符） | decorrelate_top.py / pool_matrix.py | 合成矩阵用例（高相关保留/常数列 NaN/None 回退） |
| 13 | 榜2 排序层正交：build_top 主路径 LGB 降序扫描前 60，月末截面 \|ρ\|>0.7 只留最优（复用 decorrelate 矩阵实现；from-flat 路径不变） | build_top.py | 渲染分支 + 矩阵用例 |
| 14 | platform 反馈回填：platform-ingest（report.csv+meta+txt→platform 事件，label=None 不污染排除集）+ calibrate（platform_sia≈a×local_sia×T^b log-log OLS→out/calibration.json）+ search --platform-prior（T^b 相对项 clip[0.5,2.0]，opt-in） | run_search.py / fitness.rank_score | 真实 al1 产物 ingest 3 事件；已知真值 (a=0.5,b=-0.3) 拟合复现；排除集/分母不受影响 |
| 15 | SKILL.md 事实修正（种子 26→34、UCB 描述替代"未测×3"、LGB 复合分注明启发式、refs 并入后 rescore 覆盖）、宇宙口径三套 + vwap 已知偏差两节、新增命令文档 | SKILL.md | — |

未实施（按决定排除）：标签三元诊断（用户排除）、vwap 掩码/TP 换 VWAP/方向冻结（口径暂不动）、
VWAP/GAP1 模板族与 AND/OR 探针（需平台探针，未在双方明确认同清单）、ΔComb replace/remove、
stability 硬门槛/BH-FDR、两两 IC 增量代理（榜2 正交已覆盖主要价值）、AlphaMemo 式 motif 记忆（远期）。

## 实施记录（续，2026-09-24 层1/层2）

| # | 实施 | 文件 | 验证 |
|---|---|---|---|
| 16 | 层2 本地上下文：`--local-context` 注入 MKT_RET/IND_RET/BETA60（evaluator 惰性计算，7.5M 行 ~11s，广播性/覆盖校验过）+ M14_context 模板族 6 形（市况门/行业相对强度/风险调整/个股-市场相关/行业门/alpha 分解）+ context 轴（不走 UCB，保底 max(2,n/8)）+ local_only 分流（c2c 短名单不占位，仅进 vwap 榜）+ export/upload formula_uses_local 硬闸门 + parser 认识本地字段（render→parse 往返/榜单重评通） | field_catalog/evaluator/grammar/run_search/parser/lint | 200 样本 validate 双向门全对；6 形全部可求值（cov 0.99+） |
| 17 | 层1 中性化诊断：resid 正交化链首插 SW2 行业均值剥离（成员≥5）+ BETA60 秩正交化 → drop_ind；search 终选成员自动附 resid_ind；resid.py --context | resid.py/run_search | mktcap 健全性锚点 drop_ind≈0.85（300股冒烟）/0.997（600股） |
| 18 | S9 验收门（M14 生成/双向门/广播/drop_ind）；SKILL.md 文档化 | selftest_supplements/SKILL.md | 全套验收 |
| 19 | 层2 端到端集成验证：--local-context 小规模 search（临时台账）——M14 保底撒种→进全量→c2c 短名单隔离（0 泄漏）→台账 struct_fp→终选 ind_drop 打印 | run_search | 断言全过（n_local_only≥1、短名单无本地字段公式） |

S9 锚点修正记录：最初用"mktcap drop_ind≈1"作断言是概念错误——行业均值剥离（链首）会把
绝对市值变为行业内相对市值，其 IC 可保留（样本依赖读数）。正确不变量是链**最后一步**的
Gram-Schmidt 正交性：逐截面 Σ resid·rank(BETA60)=0（实测 max|dot|/den≈3.4e-19）。

未做（按讨论结论）：层3 市场/行业因子独立 grammar 搜索轨道（factor_lab 手写构造 + mkt_sector_test
枚举评估已是正确形态）；中性化诊断进 rank_score（观察 drop_ind 一段时间后再决定）。

## 实施记录（续，2026-09-25 条件 IC 与提速）

| # | 实施 | 文件 | 验证 |
|---|---|---|---|
| 20 | cond_ic.py：主管线 COND-IC 同口径移植（hot5/hot2×70/80/90+cold）+ mkt_up/dn 新族；键名 cond_{tag}_rank_ic_* 与主管线一致；search 终选自动附带（--no-cond-ic）+ report 节 + S11 | merge/cond_ic.py 等 | 掩码嵌套/互补/键名/IC 有限全过；首读数：zturn20 栖息地内 -0.086 vs 全市场 -0.055（栖息地增强） |
| 21 | B1：validator 慢路径复用共享日期序（跳过每因子 sort_values+factorize）；prep 放开 label-NaN（NaN 感知 lr + pos_by_label）；修复空日 reduceat 越界（dcodes 压缩为连续出现日） | hierarchical_factor_validator.py | **2136 键零差异**（12 因子×~178 键，atol 1e-10） |
| 22 | B2：cond IC 压缩到子集行（热行仅 10-30%），fast 路径全压缩、slow 路径压缩排名+旧 lr_c 语义逐位保留 | 同上 | 同上等价验证覆盖 |
| 23 | B3：test_factors 分块流式载入（每块 载入→merge→一次性inf→条件列→validate→释放），--no-streaming 回退 / --stream-chunk N；inf 逐列循环改整块一次替换 | main_alpha_*_妖股_copy.py | py_compile 通过；逐因子复用同一验证函数（结构等价）；建议先 --factor-names 冒烟 |

## 附：未核实项

- VWAP 因子文献细节与 LightGBM 特征筛选从业者共识（null importance/SHAP/adversarial
  validation/IC 相关矩阵层次聚类）来自模型既有知识，**未联网核实**（该调研 agent 因环境网络
  策略未完成收尾）。其中 null-importance、adversarial validation、SHAP 筛选、层次聚类去冗余
  属业界标准做法，P3 立项时可直接采用。
- FAMA 论文未能核实为因子挖掘文献（Agent4 多渠道检索未命中）。
