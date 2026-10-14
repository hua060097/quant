# 非线性筛选与模型适配（MI/JSD/CMIM/JMI，2026-09-27）

> 位置：`merge/nonlinear.py`（估计器）+ `merge/model_suitability.py`（筛选脚本）+
> `build_top.py` 扫描列/`run_search.py` full 阶段透出。
> 本文档回答：实现原理是什么、文献根据是什么、怎么用、结合 quick 结果怎么判读。

## 1. 为什么需要（问题定义）

榜2 的 LGB 复合分全是单调线性统计（`|RankIC|×|RankICIR|×|mono|`），结构上抓不到
非单调（U 形）因子；同时单因子总量指标回答不了"相对现池的边际价值"。
MI/JSD/CMIM/JMI 补这两块，但分工不同（见 §3），不可互相替代。

## 2. 实现原理

### 2.1 口径（与正交去重同基）

- 只用**月末截面**（`pos_me`，60 个月末 × ~5000 股）：与 `month_end_spearman_matrix`、
  dom 诊断同一数据基。候选值复用正交扫描已求值（`out/value_cache/<面板指纹>/`），
  **零新增因子求值**；池侧（69 CURATED）从 `factor_lab_data/factors.parquet` 按月末
  日期过滤直读。
- 面板指纹 = 源 parquet 文件名|大小|mtime + 面板行数 + 月末行数；面板更新自动失效。

### 2.2 估计器（`nonlinear.py`，纯 numpy，确定性，无外部依赖）

| 函数 | 定义 | 参数 |
|---|---|---|
| `mi_monthly` | 逐月截面分位分箱 MI（类 IC 口径），返回均值/Std/归一化 NMI | 10×10 格，`n≥50`/月 |
| `jsd_tails` | 全窗 pooled 顶/底分位（默认上下 20%）标签分布的 JSD，[0, ln2) | 标签 20 格 |
| `cond_mi_monthly` | 逐月 `I(Y;X\|Z)`，链式法则经分箱直方图 | 5×5×5 格 |
| `joint_mi_monthly` | 逐月 `I((X,Z);Y)`，JMI 排序项构件 | 5×5×5 格 |
| `month_end_index` | 行日期 →（月末位置，对齐月标签），与 build_top 的 pos_me 同语义 | — |

### 2.3 已知的估计器局限（诚实记录，勿踩第二遍）

- **直方图偏差**：10×10 格在 n≈100/月时偏差底约 0.4（合成噪声 MI≈0.49）；真实月末
  n≈5000/月，偏差约 0.008，可忽略。**MI 绝对值不可跨样本量比较，只做同批排序。**
- **自条件恒零**：`I(Y;X|X)=0`（合成验证通过），可作冒烟断言。
- **KSG 不升级**：KSG（Kraskov 2004）是连续 MI 金标准，但当前用法（同批排序、
  大样本）下直方图已够用；且 kNN 类估计器对极强依赖需指数级样本（Gao 2015）。
  升级触发器：① 做高维条件估计直方图稀疏化；② 小样本月份（<500 行）成主流；
  ③ 需要跨轮绝对值比较（仍需归一化，无免费午餐）。
- **CMIM 比率噪声底**：`min I(Y;X|Z)/I(Y;X)` 的估计噪声底约 0.1（自匹配实测 0.101），
  否决阈值 0.1 贴着噪声底——twin 否决承担主力，CMIM 是第二道网，不要倒置。

## 3. 文献根据（为什么是这套，不是拍脑袋）

- **总量 vs 边际**：Brown et al. 2012（JMLR 统一框架）——单看 `I(因子;标签)` 叫 MIM，
  是信息论特征选择里最弱的一档（假设特征独立）；正确做法是 Max-Relevance
  Min-Redundancy（Peng-Long-Ding 2005），更好的两档是 **JMI**（Brown 实证推荐：
  精度/稳定性/小样本最优）和 **CMIM**（Fleuret 2004：`max_n min_l I(Y;Xn|Xν(l))`，
  与任一已选特征条件独立即丢弃）。
- **本管线的映射**：LGB 排序 + `|ρ|>0.7` 正交贪心 ≈ mRMR（relevance 排序 +
  pairwise 去重，已有）；CMIM 否决 + JMI 排序 = 缺失的 conditional 环节（本次补上）；
  模型 add-one = wrapper（金标准，已有）。三段缺一不可，MI 单独不能替代边际评估。
- **多重检验**：Harvey-Liu-Zhu（2016，2026-02 NBER 更新重申）：几千次搜索选出的
  第一名必须过 t>3。注意 HLZ 只回答"premium 是否真实"，不回答"进富模型是否有增量"
  （t=12 的因子也可能零边际——I1 实证）。
- **成本**：Novy-Marx/Velikov 2016（换手<50%/月才可能净存活；buy/hold spread 最有效）；
  TCA factors（2024 JoF）：去留看**含成本的组合边际贡献**，不是单因子净值——
  所以 `net_h3` 永不单独否决，只做连续项与合取参考（`net<0 AND 冗余` 出局；
  `net<0 AND 新信息` 送组合测试）。

## 4. 用法

### 4.1 build_top 接线（诊断列，不进排序；c2c 不动）

- `vwap_orthogonal_indices(..., label='vwap', y_me=...)` 对扫描候选算 MI/JSD，
  回填 flat 列（`mi/mi_std/nmi/jsd`）+ 榜2 展示列；c2c 行保持 NaN。
- `run_search.evaluate_full`（及 rescore 复评点）对各标签算 MI/JSD 进台账 tier1；
  `_ledger_candidates` 透出（老记录缺失为 None）；**fast 阶段不算**（保速度）。

### 4.2 模型适配筛选（`model_suitability.py`，独立脚本，不进 build_top）

```powershell
python model_suitability.py [--flat PATH] [--top 150] [--twin 0.7] [--cmim-ratio 0.1] [--topk 5]
```

判定链（顺序执行）：
1. **twin 否决**：候选 vs CURATED 全员，月均截面 Spearman `max|ρ|>0.7` → 出局
   （build_top 正交规则在现池上的延伸）。
2. **CMIM 否决**：`min over top-5 池代表元 I(标签;候选|代表元) / I(标签;候选) < 0.1`
   → 出局（单个池成员解释掉 90%+ 即无条件增量）。
3. **JMI 排序**：幸存者按 `Σ_top5 I((候选,代表元);标签)` 降序；短名单**内部**再按
   pairwise |ρ|>0.7 去重（只留 JMI 高者）——教训：2026-09-27 首版漏掉此步，
   `29fa/18e2`、`a030/9f77` 等精确否定对同时入选，已修复（见 §6 事故记录）。
4. **成本标注**：net/turnover 只标注 + 合取参考，不单独否决。
5. 产物：`out/MODEL_SUITABILITY_<ts>.md/.json/_full.csv/_shortlist.csv`。
   **短名单是 add-one 候选输入，不是进池结论**；进 CURATED 仍需模型 add-one ≥2 窗口。

## 5. 结合 quick 结果怎么看（判读规范）

以下为 2026-09-27 I1/18e2 两轮确立的判读规则，后续照此执行：

1. **先验有效性**：确认实验跑的池子确实含候选（`[FEAT] X=(…,70)` + 新指纹 +
   importance 表有该因子行）。反例：2026-09-27 191038 轮 X=69 且日志有
   `因子库缺少以下因子`（frozen 数据快照过期），冠军数与基线逐位相同——
   **测的是旧池，结论作废**，教训：跑前必做 `frozen_runner.py update` 并核对列。
2. **训练确定性**：同输入下 quick 结果逐位可复现（0923≡041628 全 56 格×8 指标
   maxdiff=0）。因此**跨池差异是因果效应，不是噪音**——但 slot 类路径依赖指标
   除外（同配置 H8@cp300 slot 三轮 10.98→5.30→1.30，摆动 8 倍，不可做决策指标；
   以 Rank1 median/mean 分布为准）。
3. **分系报告，不看全局冠军**：add-one verdict 必须按 HP 系分别报告各折增减
   （本轮：H12 系 4~6/6 折改善、H8 系多恶化——结论是"HP 相关"，不是"全局改善"）。
   全局冠军换人（H12@cp200→H14@cp450→H12@cp400）本身不携带信息。
4. **门的形式要件**：候选在目标 HP 系 ≥2 walk-forward 折（quick 的 6 folds 即
   walk-forward 窗口）median/mean 同向改善 → 可留（I1：H8 最好的格子被打掉，
   不满足 → 出局；18e2：H12 系满足 → 条件保留）。
5. **监控条件写法**：保留必须附带可证伪的后续条件（如"H12 系改善消失或恶化蔓延
   到 H14/H15 则拿掉"），写入报告备注，不留开放式保留。

## 6. 已验证案例（2026-09-27）

- **I1（bcdb，榜2 #1，LGB=0.00939）**：twin 最高 0.633（G2）未触发否决但偏高；
  resid-drop 0.67（风格暴露）；net_h3<0；add-one 后 H8 最好的格子 median
  0.0096→0.0012 → **出局**（已从 CURATED 移除）。
- **18e2（CORR120，榜2 #7，LGB=0.00429）**：twin 通过；CMIM 0.53；换手 7.0% 最低；
  净 +1.18%；add-one 后 H12 系 4~6/6 折改善 → **条件保留**（监控条件见 §5.5）。
- **M4-120/250（MI 发现，0.0817/0.077）**：MI 高但 resid-drop 0.88/0.93，且
  M4-250 与在池 `distance_from_52w_low` ρ=0.85（孪生）→ **出局**。教训：MI 高 +
  drop 高 = 穿非线性外衣的风格暴露；MI 高且 drop 小才是真候选。
- **29fa（-CORR120）**：与 18e2 精确孪生（|ρ|=1.0），短名单内部去重后只留其一。
- **d8f9（换手20/60比+STDDEV(AMOUNT,14)，榜2 #3，LGB=0.00687）**：twin 最高 0.67
  （G2）未触发；CMIM 0.40；add-one（032326，71 池 OFF）H8 系 6/7+5/7 格改善、
  H12 系 1/7+1/7 崩 → **条件保留**（与 18e2 互补：18e2 保 H12、d8f9 保 H8）。
- **H16（干净63叶，2026-09-29 R1）**：七格全灭（mean_avg -0.0008 全组最低、
  minfold_avg -0.0207 最深）→ **删除**。结论：63 叶在轻正则/低 min_leaf 下是
  "峰谷双输"死亡区，不是中间甜蜜点；叶子数轴非单调。
- **qs_atr_ratio_cv5（2026-09-29 R3）**：残差 t≈-4 但 add-one 28 格
  mean/median/minfold 三项平均全负，且拖累 H15 全系 → **删除**。教训："残差≠增益"
  又一实证；twin 0.97（在池 cv10）的孪生担忧事后看是对的。
- **0bccd6cf5a3a（EMA 行业残差动量，榜2 #9）**：vwap 榜 15 因子残差筛查最强新信号
  （逐日 IC +0.04~+0.06，四组符号一致，|t| 至 3.21；与 2ec4 取向一致互证）；
  待 IND_RET 实现后 add-one。**b5aeddd8fedf** 次之（|t|~3.2，一致负向）。
  18e2/1222/d8f9/bcdb 在本轮筛查中弱或无信号（|t|≤2.6）。

## 7. 待办（按优先级）

1. OOS 门（已批准未实施）：扫描期用已缓存月末值做 holdout 方向一致性检查，
   零新增求值；防 winner's curse（HLZ），与 CMIM 正交。
2. 池正交 resid 列：把 resid.py 的 3 轴版升级为相对 CURATED 池主成分版，
   做成常规筛选列（factors.parquet 69 列现成，无需求值）。
3. KSG 升级：见 §2.3 触发器，未到不做。
4. 0bccd 实现（IND_RET 行业映射 plumbing）+ add-one；先查 sector 因子如何取
   行业映射，复用同一通道（2026-09-29 发现，待做）。
5. deploy 口径纪律：OFF 口径选的冠军（如 H8@cp350）不得到 ON 口径部署；
   hardneg 在 deploy 分钟口径三次验证有效（H8@cp300 +0.80 vs -0.60；
   H15@cp200 +0.13；H14@cp350 +0.21 vs -0.54），在 quick 口径一律有害
   （mean 平均 -0.007）。两边用不同口径预期读数，禁止混用结论。
