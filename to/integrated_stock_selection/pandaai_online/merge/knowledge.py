"""机制知识库：已验证公式、失败教训、平台规则与出处（检索 + 追加）。

对应 Alpha-GPT Knowledge Library 的本地轻量版：先用结构化检索（标签/字段/算子/关键词），
不依赖 embedding；后续可升级向量检索。

文件：knowledge/mechanisms.jsonl（每行一个 JSON 对象，append-only）
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import _paths  # noqa: F401
import windows

KB_PATH = Path(__file__).resolve().parent / 'knowledge' / 'mechanisms.jsonl'

_SEED = [
    {
        'id': 'winner-amihud120',
        'status': 'validated',
        'mechanism': '非流动性水平：单位成交额驱动的价格冲击越大，流动性溢价越高（慢水平因子）',
        'formula': 'LOG(1+TS_MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,120))',
        'direction': 1,
        'family': 'M2_liquidity',
        'tags': ['liquidity', 'illiquidity', 'slow', 'level', 'winner'],
        'fields': ['CLOSE', 'AMOUNT'],
        'ops': ['LOG', 'TS_MEAN', 'ABS', 'DELAY'],
        'evidence': '本地c2c_3d net +11.16%；平台full换手30.68% vs 本地2.5%（量纲口径差异，以平台为准）',
        'source': 'LESSONS.md #12; qs_factory_panel; candidates_full',
    },
    {
        'id': 'winner-mktcap',
        'status': 'validated',
        'mechanism': '小市值规模效应：总市值越低预期收益越高（与流动性轴部分同源，实测与amihud120相关-0.813）',
        'formula': 'MARKET_CAP',
        'direction': 0,
        'family': 'FUND',
        'tags': ['size', 'smallcap', 'fundamental', 'winner'],
        'fields': ['MARKET_CAP'],
        'ops': [],
        'evidence': '平台5年：RankIC -0.0346 / net +22.20% / 换手4.60% / 单调0.92；本地log(total_mv) net +18.51%；与amihud120截面Spearman -0.813',
        'source': 'REPORT_R4_FUND.md; REPORT_R3.md; merge selftest G3',
    },
    {
        'id': 'winner-comp6',
        'status': 'validated',
        'mechanism': 'RANK和复合：把5个已验证慢因子（距高点/偏度/流动性/位置/斜率/换手）作截面排名后等权相加，分散单因子噪声',
        'formula': ('RANK(0-(CLOSE/TS_MAX(HIGH,250)-1))+RANK(0-TS_SKEW(CLOSE/DELAY(CLOSE,1)-1,120))'
                    '+RANK(LOG(1+TS_MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,120)))'
                    '+RANK(0-(CLOSE/TS_MAX(HIGH,120)-1))+RANK(0-SLOPE(LOG(CLOSE),60))'
                    '+RANK(0-TS_MEAN(TURNOVER,20))'),
        'direction': 1,
        'family': 'composite',
        'tags': ['composite', 'rank-sum', 'winner', 'best-single'],
        'fields': ['CLOSE', 'HIGH', 'AMOUNT', 'TURNOVER'],
        'ops': ['RANK', 'TS_MAX', 'TS_SKEW', 'TS_MEAN', 'SLOPE', 'LOG', 'DELAY', 'ABS'],
        'evidence': '平台full：RankIC +0.0641 / ICIR 0.2448 / net +13.03% / 换手12.25%（当时最强单体）',
        'source': 'comp_variants.py docstring; candidates_r3_full.txt',
    },
    {
        'id': 'winner-vr120s20',
        'status': 'validated',
        'mechanism': '量比平滑：当日成交量相对120日均量的偏离，经EMA(20)平滑后换手从48%降到15%',
        'formula': 'EMA((VOLUME/TS_MEAN(VOLUME,120))-1,20)',
        'direction': 0,
        'family': 'M2_liquidity',
        'tags': ['volume', 'ratio', 'smoothing', 'winner'],
        'fields': ['VOLUME'],
        'ops': ['EMA', 'TS_MEAN'],
        'evidence': '平台net +3.38%（原版vr120换手49% net -20.09）；平滑是降换手的有效路径',
        'source': 'LESSONS.md #4; candidates_r3_full.txt',
    },
    {
        'id': 'winner-dist250-high',
        'status': 'validated',
        'mechanism': '距一年高点距离：越接近250日高点动量越强（或反之，方向由数据定）',
        'formula': 'CLOSE/TS_MAX(HIGH,250)-1',
        'direction': 0,
        'family': 'M4_location',
        'tags': ['position', 'high', 'momentum', 'winner'],
        'fields': ['CLOSE', 'HIGH'],
        'ops': ['TS_MAX'],
        'evidence': 'F-A08平台多头+8.68；注意与本地跨度/基准差异（CALIBRATION_C2C）',
        'source': 'CALIBRATION_C2C.md; calibration_top5.py',
    },
    {
        'id': 'winner-skew120',
        'status': 'validated',
        'mechanism': '收益偏度：过去120日日收益左偏（负偏度/崩盘风险溢价）与未来收益的关系',
        'formula': 'TS_SKEW(CLOSE/DELAY(CLOSE,1)-1,120)',
        'direction': 0,
        'family': 'M3_distribution',
        'tags': ['skew', 'distribution', 'slow', 'winner'],
        'fields': ['CLOSE'],
        'ops': ['TS_SKEW', 'DELAY'],
        'evidence': 'comp6成员之一；平台已验证公式片段',
        'source': 'candidates_r3_full.txt; qs_factory_panel',
    },
    {
        'id': 'winner-zturn20',
        'status': 'validated',
        'mechanism': '换手水平：20日均换手率（截面ZSCORE与当日截面单调等价，公式可省ZSCORE）',
        'formula': 'TS_MEAN(TURNOVER,20)',
        'direction': 0,
        'family': 'M2_liquidity',
        'tags': ['turnover', 'level', 'winner'],
        'fields': ['TURNOVER'],
        'ops': ['TS_MEAN'],
        'evidence': 'comp6成员之一；R3结论指出 zturn20≡ZSCORE(TS_MEAN(TURNOVER,N)) 截面单调等价',
        'source': 'qs_r3.py; candidates_r3_full.txt',
    },
    {
        'id': 'fail-time-series-z',
        'status': 'failed',
        'mechanism': '慢水平因子加20日时序z变换（去漂移）：把持续高位抹成噪声，收益由正转负',
        'formula': None,
        'direction': None,
        'family': 'transform',
        'tags': ['zscore', 'transform', 'failed', 'lesson'],
        'fields': [],
        'ops': ['TS_ZSCORE'],
        'evidence': 'amihud120 加20日时序z后本地net +11.16% → -27.55%',
        'source': 'LESSONS.md #5',
    },
    {
        'id': 'fail-sparse-tie',
        'status': 'failed',
        'profile': 'competition-c2c',
        'mechanism': '离散事件计数因子（连涨/触碰）稀疏并列：平台tie-breaking放大换手至88%，本地只看到1-2%',
        'formula': None,
        'direction': None,
        'family': 'M8_gate',
        'tags': ['sparse', 'tie', 'event', 'failed', 'lesson'],
        'fields': [],
        'ops': ['COUNT', 'EVERY', 'EXIST'],
        'evidence': 'F-B04 b_up_count / F-B08 b_touch：本地换手1~2% vs 平台88%，平台侧为准，禁用',
        'source': 'CALIBRATION_C2C.md',
    },
    {
        'id': 'warn-amount-unit',
        'status': 'warning',
        'mechanism': '【已更正 2026-09-16，见 info-amount-caliber】AMOUNT 数据本身与平台逐位一致（单位=元）；'
                     '本地/平台唯一差异是复权口径。amihud 平台换手异常指向平台侧对 ~1e-11 量级数值的'
                     '处理，而非数据口径。',
        'formula': None,
        'direction': None,
        'family': 'liquidity',
        'tags': ['amount', 'unit', 'turnover', 'warning', 'corrected'],
        'fields': ['AMOUNT'],
        'ops': [],
        'evidence': '000001.SZ 2025-01-02~15 逐字段对比：volume/amount 完全一致；close 比值 1.0876 恒定（复权差）',
        'source': '用户平台样本; diag_amount*.py; 2026-09-16 更正',
    },
    {
        'id': 'fail-value-family',
        'status': 'failed',
        'mechanism': '价值族（BM/股息率/PS/EV-EBITDA）在基本面探针中全灭——这一信息轴只有规模因子有效',
        'formula': None,
        'direction': None,
        'family': 'FUND',
        'tags': ['value', 'fundamental', 'failed', 'lesson'],
        'fields': [],
        'ops': [],
        'evidence': '基本面6探针：mktcap成功，价值类全灭',
        'source': 'REPORT_R4_FUND.md',
    },
    {
        'id': 'rule-probe-window',
        'status': 'rule',
        'profile': 'competition-c2c',
        'mechanism': '远程probe窗口必须装得下回看：250日因子probe窗口≥2年，否则平台fail',
        'formula': None,
        'direction': None,
        'family': 'process',
        'tags': ['process', 'probe', 'rule'],
        'fields': [],
        'ops': [],
        'evidence': 'LESSONS.md #11',
        'source': 'LESSONS.md',
    },
    {
        'id': 'rule-turnover-gate',
        'status': 'rule',
        'mechanism': '换手口径（2026-09-16 修订，以官方评分规则为准）：官方 A/B 公式不含换手/净；'
                     '换手只出现在池级 C 的分母 max(Turnover_month, 0.3)。单因子阶段不做换手硬门槛，'
                     '改用官方 C 同源的换手软惩罚 ref/max(T_month, ref)（默认 ref=0.3）。'
                     '历史上自造的"单次换手<30%/15%"门槛已删除（它把"月换手 30%"错当单次，'
                     '且会拒绝平台自己的 amihud120=30.68%）。',
        'formula': None,
        'direction': None,
        'family': 'process',
        'profile': 'competition-c2c',
        'tags': ['process', 'turnover', 'rule', 'official'],
        'fields': [],
        'ops': [],
        'evidence': '官方规则原文（飞书总文档，2026-09-16 用户提供）：rawC=max(Rex,0)/max(Turnover_month,0.3)×SR×(1−1.2MaxDD)；'
                    'Turnover_month=当月各调仓日组合换手之和；A/B 公式仅 |RankIC|×|ICIR|×胜率。',
        'source': 'official rules; merge fitness.turnover_penalty; selftest_b3 T1',
    },
    {
        'id': 'rule-official-scoring-20260916',
        'status': 'rule',
        'mechanism': '官方三段式：Comb=0.20NA+0.35NB+0.45NC；A=池内因子 S=|RankIC|×|ICIR|×胜率(|IC|>0.02)'
                     '（样本内近5年+全部历史样本外），锚 0.08、Decay 1.0/0.9/0.85、新池<6月封顶0.7；'
                     'B=仅生效后新增样本外同款指标，锚 0.06、无衰减；C=池等权合成组合的实时日频账本'
                     '（rawC 同 rule-turnover-gate），锚 0.6。B/C 是前瞻项，提交前不可观测——'
                     '本地历史最优只能做代理；搜索目标=可观测的 A + 稳健性 + 换手软惩罚。',
        'formula': None,
        'direction': None,
        'family': 'process',
        'profile': 'competition-c2c',
        'tags': ['process', 'scoring', 'official', 'rule', 'anchors'],
        'fields': [],
        'ops': [],
        'evidence': '飞书总文档三段式评分原文 + 用户 2026-09-16 提供；本地对齐见 selftest_b3 T1。',
        'source': 'official rules; merge/fitness.py; selftest_b3',
    },
    {
        'id': 'rule-c2c-caliber',
        'status': 'rule',
        'profile': 'competition-c2c',
        'mechanism': '口径三件套：收盘→收盘前向收益、池统一周期采样、单边0.3%成本；开盘口径会得出相反结论',
        'formula': None,
        'direction': None,
        'family': 'process',
        'tags': ['process', 'caliber', 'rule', 'labels'],
        'fields': [],
        'ops': [],
        'evidence': '开盘口径86显著/30正net 在c2c口径下几乎全灭（隔夜缺口端IC −0.05~−0.09）',
        'source': 'CALIBRATION_C2C.md; CALIBRATION_LEDGER.md',
    },
    {
        'id': 'info-comp-variants',
        'status': 'candidate',
        'mechanism': 'comp6的边际变体（本地c2c_3d，未平台验证）：加/换成员以提升Comb',
        'formula': None,
        'direction': 1,
        'family': 'composite',
        'tags': ['composite', 'candidate', 'variant'],
        'fields': [],
        'ops': ['RANK'],
        'evidence': 'comp7_mktcap RankIC+0.0619/net+14.00；comp7_noh_mkt +0.0668/+14.44；comp8_mkt_vr +0.0684/+13.79（本地）',
        'source': 'comp_variants.py; REPORT_R3.md',
    },
    {
        'id': 'info-p7-marginal',
        'status': 'candidate',
        'mechanism': 'P7池边际：highpos120最弱，用mktcap替换的理论Comb最高（0.1969）',
        'formula': None,
        'direction': None,
        'family': 'pool',
        'tags': ['pool', 'marginal', 'candidate', 'october'],
        'fields': [],
        'ops': [],
        'evidence': 'P7 0.1839 / +mktcap 0.1884 / -highpos+mktcap 0.1969 / -amihud+mktcap 0.1804（本地Comb代理）',
        'source': 'marginal_test3.py; merge selftest G4',
    },
    {
        'id': 'rule-verified-ops',
        'status': 'rule',
        'mechanism': '平台已验证算子白名单：TS_MEAN/DELAY/LOG/MAX/TS_MIN/RANK/TS_MAX/STDDEV/SLOPE/ABS/IF/CORR/TS_SKEW/COUNT/EMA/MIN/TS_ZSCORE（含提交run记录）',
        'formula': None,
        'direction': None,
        'family': 'process',
        'tags': ['process', 'operators', 'rule', 'platform'],
        'fields': [],
        'ops': ['TS_MEAN', 'DELAY', 'LOG', 'MAX', 'TS_MIN', 'RANK', 'TS_MAX', 'STDDEV',
                'SLOPE', 'ABS', 'IF', 'CORR', 'TS_SKEW', 'COUNT', 'EMA', 'MIN', 'TS_ZSCORE'],
        'evidence': 'candidates_*.txt 全量提交公式的操作符统计',
        'source': 'candidates_*.txt; merge field_catalog.py',
    },
    {
        'id': 'warn-crowding',
        'status': 'rule',
        'mechanism': '拥挤度先验：Alpha101/公开因子族已被大量参赛者覆盖，优先冷门机制（隔夜结构/停牌恢复/涨跌停生态/财报衍生）',
        'formula': None,
        'direction': None,
        'family': 'process',
        'tags': ['process', 'crowding', 'rule'],
        'fields': [],
        'ops': [],
        'evidence': 'skill2 排除清单将 Alpha191 标为拥挤；比赛排名由相对强度决定',
        'source': 'skill-factor-mining-pandaai（概念）; merge 计划',
    },
    {
        'id': 'info-log-ab-20260916',
        'status': 'rule',
        'mechanism': 'log 政策 A/B（800股/24月，c2c_3d）：MARKET_CAP 加 log 使 SiA 翻倍（0.0009→0.0019，'
                     'RankIC 不变、ICIR/胜率改善）；TURNOVER 水平加 log 使 SiA 下降；VOLUME 水平略降；'
                     'AMOUNT 略升。结论：市值一律 log；换手/成交量水平保留 raw 为主、log 小配额探索。',
        'formula': None,
        'direction': None,
        'family': 'transform',
        'tags': ['log', 'transform', 'ab-test', 'size', 'turnover'],
        'fields': ['MARKET_CAP', 'TURNOVER', 'VOLUME', 'AMOUNT'],
        'ops': ['LOG', 'TS_MEAN'],
        'evidence': 'experiments_log_ab.py 输出；pair deltas：MARKETCAP +0.0010，TURNOVER20 -0.0032/'
                    '60 -0.0021/120 -0.0010，VOLUME -0.0002~-0.0005，AMOUNT +0.0002。'
                    'RankIC 因单调不变性基本不变（±0.006 内）。',
        'source': 'merge/experiments_log_ab.py; selftest_b2 T1',
    },
    {
        'id': 'info-resid-size-axis',
        'status': 'rule',
        'mechanism': '风险轴残差 IC（秩空间顺序正交化 mktcap→turnover→vol20）：MARKET_CAP 过轴后恒零；'
                     'amihud120 残差 SiA 掉幅约 74%（与 mktcap 相关 -0.813）。残差 IC 用于识别'
                     '"伪装的风格暴露"。',
        'formula': None,
        'direction': None,
        'family': 'process',
        'tags': ['residual', 'orthogonalization', 'risk-axis', 'diagnostic'],
        'fields': ['MARKET_CAP', 'TURNOVER'],
        'ops': ['STDDEV'],
        'evidence': 'selftest_b2 R1：MARKET_CAP drop=1.0（残差恒零）；amihud120 raw_sia 0.0012→resid 0.0003。',
        'source': 'merge/resid.py; selftest_b2 R1',
    },
    {
        'id': 'info-overnight-dual-20260916',
        'status': 'rule',
        'profile': 'competition-c2c',
        'mechanism': '口径发现（2026-09-16 双标签实测）：c2c 候选在 vwap 口径（T+1开盘买入、'
                     '0.7·TP(T+2)+0.3·TP(T+3) 卖出）下 SiA 比值仅 0.01~0.19 —— alpha 集中在 '
                     'T→T+1 隔夜段，T+1 开盘执行不可收割。政策修订（同日）：vwap 不再是必过门槛——'
                     '实测池内 8 个因子自己只有 skew120 过 vwap 双主（比值 0.30），8/8 按双主口径都要被拒；'
                     '比赛评分也不使用 vwap。现行：c2c 记分，vwap 仅诊断并按 SiA 输出 top20，'
                     '供用户 LightGBM 项目使用。',
        'formula': None,
        'direction': None,
        'family': 'process',
        'tags': ['dual-label', 'overnight', 'execution', 'rule', 'caliber', 'vwap'],
        'fields': [],
        'ops': [],
        'evidence': 'shortlist_20260916_010658_seed7：15/15 候选 vwap/c2c 比值 0.01~0.19；'
                    '池内双主复评仅 skew120 通过；71 个 c2c 优质低换手候选中 65 个死于 fast vwap 门槛，'
                    '全量复评证实其 vwap 比值 0.01~0.14。',
        'source': 'merge run_search; selftest_b3 T2; 2026-09-16 用户决议',
    },
    {
        'id': 'info-platform-rsc-20260916',
        'status': 'validated',
        'profile': 'competition-c2c',
        'mechanism': '平台校准（2026-09-16，12 个 M2 流动性候选上平台，窗口 20230301-20260710、'
                     'cycle3、10 组）：本地全窗指标与平台高度一致（净超额差 ≤0.7pt）。'
                     'MG-M2-e69118f1=EMA(VOLUME/TS_MEAN(VOLUME,250)-1,20) dir=0：平台 RankIC -0.0584、'
                     '换手 14.26%、净 +4.14%（本地 +4.31%），超过池内 vr120_s20（RankIC -0.0543/换手 15.22%/净 +3.38%）；'
                     '但不及 comp6（净 +13.03%）与 dist250（+8.86%）。其余 11 个高换手候选（20~63%）平台净 −0.6~−22.5%，'
                     '与本地预测方向一致——换手软惩罚确实在筛掉成本死亡区。',
        'formula': 'EMA(((VOLUME)/(TS_MEAN(VOLUME,250)))-(1),20)',
        'direction': 0,
        'family': 'M2_liquidity',
        'tags': ['platform', 'calibration', 'liquidity', 'batch', 'validated'],
        'fields': ['VOLUME'],
        'ops': ['EMA', 'DIV', 'TS_MEAN', 'SUB'],
        'evidence': 'out/candidates_rsc1.report.csv、candidates_rsc2.report.csv（平台）；'
                    'rescore_20260916_170634.jsonl（本地）；余额 318→292（12 run）。',
        'source': 'batch.py rsc1/rsc2 2026-09-16',
    },
    {
        'id': 'info-amount-caliber-20260916',
        'status': 'rule',
        'profile': 'competition-c2c',
        'mechanism': 'AMOUNT 数据校准（2026-09-16）：平台与本地 amount/volume 逐位一致（样本 000001.SZ 十日），'
                     '唯一差异是复权口径（本地前复权、平台不复权，比值 1.0876 恒定）。amihud 平台换手异常'
                     '（30.68% vs 本地 2.6%）不是数据错误：本地平滑天数扫描显示平台行为更像"未平滑的日度'
                     '信号"（平台 bot=0.711≈本地 D1 的 0.706；平台 amihud60/120 换手几乎相同而本地差 2 倍），'
                     '疑为平台侧对 ~1e-11 量级数值的存储/精度处理。结论：AMOUNT 系因子的本地换手不可信，'
                     '一律以平台读数为准；已提交 ×1e9 尺度探针验证。',
        'formula': None,
        'direction': None,
        'family': 'process',
        'tags': ['amount', 'caliber', 'amihud', 'warning', 'rule', 'corrected'],
        'fields': ['AMOUNT', 'VOLUME', 'CLOSE'],
        'ops': [],
        'evidence': '000001.SZ 20250102-20250115：volume/amount 完全一致；close 比值 1.0876（复权差）。'
                    '非 AMOUNT 因子平台/本地换手一致（slope60 12.6/12.63、skew120 9.9/10.03、'
                    'dist250 10.8/10.67、highpos120 14.4/14.29）。',
        'source': '用户平台样本对比; diag_amount.py; diag_amount2.py; 2026-09-16',
    },
    {
        'id': 'info-amount-probe-result-20260916',
        'status': 'validated',
        'profile': 'competition-c2c',
        'mechanism': 'AMOUNT 尺度探针结论（2026-09-16，实测扣 4 credit）：“平台对 ~1e-11 数值处理异常”'
                     '假说成立。同窗口同周期对比：原版（/AMOUNT，值~1e-11）换手 30.68%/成本 15.46%/'
                     '净 −0.50%；×1e9 版（值~1e-2）换手 4.93%/成本 2.48%/净 +13.86%。'
                     '强制规则：AMOUNT 做分母必须在公式内显式 ×1000000000。',
        'formula': 'LOG(1+TS_MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT*1000000000,120))',
        'direction': 1,
        'family': 'process',
        'tags': ['amount', 'probe', 'verified', 'rule', 'scale'],
        'fields': ['AMOUNT'],
        'ops': ['LOG', 'TS_MEAN', 'ABS', 'DELAY', 'MUL'],
        'evidence': 'SC-amt120：IC 0.0230 / RankIC 0.0309 / turn 4.93% / net +13.86% / mono 0.95；'
                    '对照 F-E01 amihud120：turn 30.68% / net −0.50%。',
        'source': 'pandaai_online/merge/out/amt_scale_probe.report.csv; 2026-09-16',
    },
    {
        'id': 'warn-amihud-platform-cost',
        'status': 'rule',
        'profile': 'competition-c2c',
        'mechanism': 'amihud120 平台实测换手 30.68%/调仓 → 年化成本 15.46% → 净超额 −0.50%/年（方向1）；'
                     '近期窗口（2025-01~03）换手 77%、更差。池决策以平台读数为准：考虑移除 amihud120'
                     '（comp6 内部该腿权重 1/6、被稀释后平台换手 12.25%/净 +13.03%，可保留）。',
        'formula': None,
        'direction': 1,
        'family': 'process',
        'tags': ['amihud', 'turnover', 'cost', 'pool', 'rule'],
        'fields': ['AMOUNT'],
        'ops': ['LOG', 'TS_MEAN', 'ABS', 'DELAY'],
        'evidence': 'candidates_qs3_full.report.csv：F-E01 turn=30.68 cost=15.46 long=14.96 net=-0.50；'
                    '分组换手 g1=71.1%、g2~g8=80~85%、g10=30.7%（全分布排序噪声）。',
        'source': 'pandaai_online/candidates_qs3_full.report.csv; 2026-09-16',
    },
    # ---- 2026-09-24：8 族机制条目补齐（M1/M5/M6/M7/M10/M11/M12/M13 此前无条目，
    #      而台账显示这 8 族被测 1600+ 次；status 用 rule/candidate，不虚标 validated） ----
    {
        'id': 'rule-m1-return-momentum',
        'status': 'rule',
        'mechanism': '动量/反转（M1）：中期动量 TS_MEAN(RET1,w) 与短期反转常并存；'
                     'A 股 20 日内偏反转、120 日上偏动量，方向须由训练窗冻结',
        'family': 'M1_return',
        'tags': ['momentum', 'reversal', 'return'],
        'fields': ['CLOSE'],
        'ops': ['TS_MEAN', 'STDDEV', 'EMA'],
        'evidence': '台账 M1_return 287 次评估；方向随 horizon 翻转是该族主要风险',
        'source': 'seed 2026-09-24（8 族机制条目补齐）',
    },
    {
        'id': 'rule-m5-trend-slope',
        'status': 'rule',
        'mechanism': '趋势斜率（M5）：对数价格回归斜率度量持续趋势；风险调整形式'
                     ' SLOPE/STDDEV(RET1,w) 比裸斜率稳健',
        'family': 'M5_trend',
        'tags': ['trend', 'slope', 'log-price'],
        'fields': ['CLOSE'],
        'ops': ['SLOPE', 'STDDEV', 'EMA'],
        'evidence': '台账 M5_trend 118 次评估；与 M4 位置族高相关，池组合注意去冗余',
        'source': 'seed 2026-09-24（8 族机制条目补齐）',
    },
    {
        'id': 'rule-m6-pvcorr',
        'status': 'rule',
        'mechanism': '量价相关（M6）：CORR(价/收益, 量/换手, w) 度量量价配合度；'
                     '量价背离常预示反转，量价齐升偏向延续',
        'family': 'M6_pvcorr',
        'tags': ['volume-price', 'correlation', 'divergence'],
        'fields': ['CLOSE', 'VOLUME', 'TURNOVER'],
        'ops': ['CORR'],
        'evidence': '台账 M6_pvcorr 102 次评估；CORR 全窗实现较慢（6.5s/面板）',
        'source': 'seed 2026-09-24（8 族机制条目补齐）',
    },
    {
        'id': 'rule-m7-volatility',
        'status': 'rule',
        'mechanism': '波动率（M7）：低波动异象（低波组收益更高）为 A 股稳定结构；'
                     '变异系数与相对价格波动是尺度稳健变体',
        'family': 'M7_volatility',
        'tags': ['volatility', 'low-vol', 'anomaly'],
        'fields': ['CLOSE'],
        'ops': ['STDDEV', 'TS_MEAN', 'ABS'],
        'evidence': '台账 M7_volatility 81 次评估；与市值/换手相关，风险轴残差诊断推荐常开',
        'source': 'seed 2026-09-24（8 族机制条目补齐）',
    },
    {
        'id': 'rule-m10-shape-path',
        'status': 'rule',
        'mechanism': '路径形状（M10）：效率比（净位移/路径长度）区分平滑趋势与震荡行进；'
                     '下行波动占比（Sortino 式）与让利回吐刻画回撤路径',
        'family': 'M10_shape',
        'tags': ['path', 'efficiency-ratio', 'downside'],
        'fields': ['CLOSE', 'LOW'],
        'ops': ['ABS', 'TS_MEAN', 'STDDEV', 'EMA', 'DELAY'],
        'evidence': '台账 M10_shape 214 次评估；OOS 中 shape 族存活率较高（Hubble 类似结论）',
        'source': 'seed 2026-09-24（8 族机制条目补齐）',
    },
    {
        'id': 'rule-m11-regime-gate',
        'status': 'rule',
        'mechanism': '状态门（M11）：高波/量价相关/换手 z 等状态条件下的条件动量；'
                     'TS_MEAN(IF(cond,RET1,0),w) 产生大量零值 → 稀疏并列换手风险，'
                     '上平台前必须过 sparse-tie 检查',
        'family': 'M11_regime',
        'tags': ['regime', 'conditional', 'sparse-tie-risk'],
        'fields': ['TURNOVER', 'VOLUME'],
        'ops': ['IF', 'TS_MEAN', 'TS_ZSCORE', 'CORR', 'STDDEV'],
        'evidence': '台账 M11_regime 251 次评估；F-B04/B08 稀疏并列事故高发族',
        'source': 'seed 2026-09-24（8 族机制条目补齐）',
    },
    {
        'id': 'rule-m12-orderstats',
        'status': 'candidate',
        'mechanism': '次序统计（M12，DOCUMENTED）：TS_RANK/TS_ARGMAX/TS_MEDIAN/TS_KURT 等'
                     '位置型统计量；首版未平台冒烟，上平台前需单独 probe',
        'family': 'M12_orderstats',
        'tags': ['order-stats', 'documented', 'needs-probe'],
        'fields': ['CLOSE', 'HIGH'],
        'ops': ['TS_RANK', 'TS_ARGMAX', 'TS_MEDIAN', 'TS_KURT', 'SIGNEDPOWER'],
        'evidence': '台账 M12 早期配额实验曾获 2.5× 预算后证明性价比低（2026-09-16 修订背景）',
        'source': 'seed 2026-09-24（8 族机制条目补齐）',
    },
    {
        'id': 'rule-m13-tech-indicators',
        'status': 'candidate',
        'mechanism': '技术指标（M13，DOCUMENTED）：RSI/BIAS/CCI/BOLL%/KDJ/MACD_DIF/OBV 斜率等；'
                     '拥挤度最高的族，同质化风险大，与历史因子池相关性检查必过',
        'family': 'M13_tech',
        'tags': ['technical', 'documented', 'crowded'],
        'fields': ['CLOSE', 'HIGH', 'LOW', 'VOLUME'],
        'ops': ['RSI', 'BIAS', 'CCI', 'BOLL_WIDTH', 'KDJ_K', 'MACD_DIF', 'SLOPE'],
        'evidence': '台账 M13_tech 339 次评估（最多）；建议优先做拥挤度负通道过滤',
        'source': 'seed 2026-09-24（8 族机制条目补齐）',
    },
]


def ensure_seed(path: Path | None = None) -> Path:
    """确保种子条目为最新：同 id 种子条目用最新文本覆盖；自定义条目保留。

    2026-09-24 修复：合并结果与现文件逐条一致时不再整文件重写（旧实现每次 load()
    都重写，破坏 append-only 语义——手工编辑/追加的自定义条目时间戳会被抹掉）。
    """
    path = Path(path or KB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        existing = [json.loads(x) for x in
                    path.read_text(encoding='utf-8').splitlines() if x.strip()]
    seed_by_id = {e.get('id'): e for e in _SEED}
    out: list[dict] = []
    have_seed = set()
    for record in existing:
        rid = record.get('id')
        if rid in seed_by_id:
            updated = dict(seed_by_id[rid])
            updated.setdefault('created_at', record.get('created_at')
                               or datetime.now().isoformat(timespec='seconds'))
            out.append(updated)
            have_seed.add(rid)
        else:
            out.append(record)
    for entry in _SEED:
        if entry.get('id') in have_seed:
            continue
        record = dict(entry)
        record.setdefault('created_at', datetime.now().isoformat(timespec='seconds'))
        out.append(record)

    def _canon(records: list[dict]) -> list[str]:
        return [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records]

    if _canon(out) == _canon(existing):
        return path
    with path.open('w', encoding='utf-8') as f:
        for record in out:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path


def load(path: Path | None = None) -> list[dict]:
    path = ensure_seed(path)
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def append(entry: dict, path: Path | None = None) -> dict:
    path = ensure_seed(path)
    record = dict(entry)
    record.setdefault('id', f"kb-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    record.setdefault('created_at', datetime.now().isoformat(timespec='seconds'))
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record


def auto_reflow(ledger=None, path: Path | None = None,
                min_events: int = 8, sia_cutoff: float = 0.0005,
                label: str = 'c2c', data_cut: str | None = None,
                profile: str | None = None) -> list[dict]:
    """按目标标签/数据截止日回流低产出 family，避免 C2C 负证据污染本地 VWAP。

    此前失败教训只进台账（低分 fast/local 事件），知识库 failed 条目全靠人工
    （--add-json 唯一调用点是其自身 main）。改为每次 search 结束自动聚合：
    family 事件数 ≥ min_events 且平均 SiA < sia_cutoff（或全部无 SiA）→ append
    profile/label/family scoped failed 条目（幂等：已存在则跳过）。
    返回本次新增条目列表。
    """
    profile = profile or ('local-vwap' if label == 'vwap' else 'competition-c2c')
    cut = windows.normalize_data_cut(data_cut) if data_cut is not None else None
    rows = None
    if ledger is not None:
        rows = ledger.summary()['events_min']
    else:
        from ledger import Ledger as _L
        rows = _L().summary()['events_min']
    agg: dict[str, list] = {}
    for row in rows:
        # 行格式：5 元组（旧）或 6 元组（含数据截止日，2026-10-11 起）
        _created, family, event_label, _ev, sia = row[:5]
        event_cut = row[5] if len(row) > 5 else None
        if event_label != label and not (label == 'c2c' and event_label is None):
            continue
        if cut is not None and event_cut != cut:
            continue
        if not family or family in ('proposal', 'intake'):
            continue
        agg.setdefault(family, []).append(sia)
    existing_ids = {e.get('id') for e in load(path)}
    added = []
    for family, sias in sorted(agg.items()):
        if len(sias) < min_events:
            continue
        vals = [float(s) for s in sias if s is not None]
        if not vals:
            # 2026-09-24 修复：全部事件无有效 SiA 读数（如旧 schema 的 ref:* 事件）
            # 是"数据不足"不是"失败"——此前会误标为 failed（首跑即误标 5 个 ref 族）
            continue
        mean = sum(vals) / len(vals)
        if mean >= sia_cutoff:
            continue
        mean_txt = f'{mean:.5f}'
        entry = {
            'id': f'auto-fail-{profile}-{family}',
            'profile': profile,
            'label': label,
            'data_cut': cut,
            'status': 'failed',
            'family': family,
            'mechanism': f'自动回流：{family} 台账 {len(sias)} 次评估平均 SiA={mean_txt}'
                         f'（<{sia_cutoff} 或无有效读数），历史无有效产出，新搜索降权',
            'evidence': f'source=ledger events_min; n_events={len(sias)}; '
                        f'mean_sia={mean_txt}; cutoff={sia_cutoff}',
            'tags': ['auto', 'failed', 'low-yield', f'profile:{profile}', f'label:{label}'],
            'source': f'knowledge.auto_reflow {datetime.now().strftime("%Y-%m-%d")}',
        }
        if entry['id'] in existing_ids:
            continue
        append(entry, path)
        existing_ids.add(entry['id'])
        added.append(entry)
    return added


def _tokens(text: str) -> list[str]:
    text = text.lower()
    return [t for t in re.split(r'[\s,，、;；:：()（）\[\]/]+', text) if len(t) >= 2]


def retrieve(query: str = '', tags=None, fields=None, ops=None, k: int = 5,
             path: Path | None = None, profile: str | None = None) -> list[dict]:
    tags = {t.lower() for t in (tags or [])}
    fields = {f.upper() for f in (fields or [])}
    ops = {o.upper() for o in (ops or [])}
    query_tokens = _tokens(query)
    scored = []
    for entry in load(path):
        if profile:
            entry_profile = entry.get('profile')
            if entry_profile is not None and entry_profile not in ('shared', profile):
                continue
            # Legacy automatic failures were generated by the historical C2C-primary
            # search and have no profile field; do not treat them as VWAP evidence.
            is_legacy_auto_failure = (str(entry.get('id', '')).startswith('auto-fail-')
                                      and entry_profile is None)
            if is_legacy_auto_failure and profile != 'competition-c2c':
                continue
        text = ' '.join(str(entry.get(x, '')) for x in
                        ('mechanism', 'formula', 'evidence', 'family', 'source')).lower()
        entry_tags = {str(t).lower() for t in entry.get('tags', [])}
        score = 0.0
        score += 3.0 * len(tags & entry_tags)
        score += 2.0 * len(fields & {str(f).upper() for f in entry.get('fields', [])})
        score += 1.5 * len(ops & {str(o).upper() for o in entry.get('ops', [])})
        if query:
            if query.lower() in text:
                score += 4.0
            score += sum(1.0 for t in query_tokens if t in text)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:k]]


def stats(path: Path | None = None) -> dict:
    entries = load(path)
    by_status: dict = {}
    for entry in entries:
        by_status[entry.get('status', '?')] = by_status.get(entry.get('status', '?'), 0) + 1
    return {'n': len(entries), 'by_status': by_status}


def main() -> None:
    import argparse

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='机制知识库：检索/统计/追加')
    ap.add_argument('--query', default='')
    ap.add_argument('--tags', default='')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--add-json', default='')
    args = ap.parse_args()
    if args.add_json:
        print(json.dumps(append(json.loads(args.add_json)), ensure_ascii=False, indent=2))
        return
    if args.stats:
        print(json.dumps(stats(), ensure_ascii=False, indent=2))
        return
    hits = retrieve(args.query,
                    tags=[t for t in args.tags.split(',') if t], k=args.k)
    for i, entry in enumerate(hits, 1):
        print(f"{i}. [{entry.get('status')}] {entry.get('id')}  {entry.get('mechanism')}")
        if entry.get('formula'):
            print(f"   formula: {entry['formula']}")
        print(f"   evidence: {entry.get('evidence')}")


if __name__ == '__main__':
    main()
