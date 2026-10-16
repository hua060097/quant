# -*- coding: utf-8 -*-
"""门控统一批量流水线（MERGE_V2_PLAN §17.3，S0→S7）。

**解决什么**：门控线与进化搜索线此前是两条不同构的流程，且门控线**跨批次重复评估**
（每次重跑 54 个）。本流水线把 §17.3 的八阶段串起来，并以 `gate_ledger` 的指纹
对账，**只跑未测过或口径已变的候选**。

阶段与实现映射（§17.3）：
  S0 候选摄入     `gate_search_space.generate_space()`
  S1 身份与指纹    `gate_ledger`（candidate_fp / family_fp / order_fp）
  S2 廉价预筛链    §17.5 的 9 层——本实现含：
                   ① candidate_fp 去重 ② family_fp 记账 ③ 规格合法性（B4：qwin
                   读实际生成值）④ 尘埃检测 ⑥ order_fp ⑦ G0 + 方向中性度
                   ⑧ 输入结构自检 ⑨ 稀疏性（告警不设门）
                   （⑤ G4 属 #S6：依赖当前门控集合，评估后计算）
  S3 快层评估      **不启用**（§4.3 实测：抽样使符号一致率降到 1/3）
  S4 排序与停止    `gate_score`（预注册权重，仅用于截断）
  S5 全层评估      G2/G3（`gate_g2g3`）→ 方向分层
  S6 去冗余与校正  ⑤ G4（**series 口径**，2026-10-12 裁决；记录+告警不设硬门，
                   §17.5 纪律）+ 检验族分母：F1 批次的族 = `f1_family` 清单
                   （F1v2，720，预注册于 hypothesis_registry.csv），
                   由 `judge_f1.py --family` **强制逐一对应**（不匹配即拒绝），
                   台账记 family_id/成员哈希
  S7 短名单与台账  短名单 CSV + **报告 MD（pipeline_report.md）** +
                   decided 事件（state=qualified_g2g3，≠ 入池）+ 台账回写

**注意**：F1（候选级 ΔIC + placebo + FDR）是逐 (门控,因子) 对的昂贵层，
由 `f1_batch.py --gates-parquet` 承接；本流水线产出 **gate 级短名单**后调用它。

用法：
    python run_gate_pipeline.py                 # 全流程（只跑 pending）
    python run_gate_pipeline.py --all           # 忽略台账，全跑（回归对照用）
    python run_gate_pipeline.py --no-f1         # 抑制"下一步"打印
    python run_gate_pipeline.py --all --topn 5  # 筛选+排名，只保留前 5 进短名单

**与个股进化搜索的两段式对应**（2026-10-13 用户要求的分工）：
  搜索命令 = `gate_search_space.py`（生成候选空间与门控面板，一次落盘）
  筛选命令 = **本脚本**（8 层资格筛选 + G2/G3 + S4 排名 + G4 + S7 短名单/[--topn]）
  逐个检验 = `f1_batch.py` + `judge_f1.py --family`（**人工衔接**——族校验要求
             全族 720 整体运行，短名单 topn 不改变族规模）
  模型级增量 = 仓库根生产脚本（merge 不内嵌，见 s8_state docstring）
  冒烟（小样一条龙）= `smoke_pipeline.py`（仅为流程闸门，产物在独立目录）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_ledger as GL  # noqa: E402
import gate_search_space as GSS  # noqa: E402
import gate_stop as GS  # noqa: E402
import gates as G  # noqa: E402
import boundary as B  # noqa: E402
import windows as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
SPACE = os.path.join(RESULTS, "gates_search")
CUT = W.FACTOR_SCREEN_MAX
def make_caliber(n_perm: int, n_boot: int, form: str = "t",
                 factor_set: str = "", panel_sha: str | None = None) -> dict:
    return {"window_end": CUT, "form": form, "factor_set": factor_set or "base20",
            "label": "r3", "data_version": B.factor_data_version(),
            "n_perm": int(n_perm), "n_boot": int(n_boot), "panel_sha": panel_sha}

#: S4 排序分权重（预注册；仅用于截断，不作判定）
#: **注意**：此处**不含**家族惩罚项。原设计有 `family_penalty × (family_multi>1)`，
#: 但搜索空间里同一基础量的 4 个操作化构成等规模家族（family_multi 恒为 4），
#: 该二值项对所有候选取值相同 → **恒无区分力**。家族内近克隆的取舍属 S6 去冗余
#: 阶段（`dedup` / G4），不应伪装成排序分项。`family_multi` 仅作报告列保留。
GATE_SCORE_W = {"abs_stat": 1.0, "same_frac": 0.5, "neutrality": 0.5}


# ------------------------------------------------------------------ S2 新增层
def spec_legality(spec: dict, qwin_actual: int | None = None) -> tuple[bool, str]:
    """③ 规格合法性（2026-10-12 修复 B4：qwin 读实际生成值，旧恒等 252 为死逻辑）。

    三项检查中的两项由**构造保证**（OPS 为固定 lambda：`ratio` 恒为同期同量纲的
    mean20/mean120；聚合方式由 `generate_space` 按层级受限），此处复核记录在案：
    1. 分位窗 = 实际生成值 ≥ 63（B4：gate_search_registry.json 记录的生成 qwin）
    2. 层级 ∈ {market, sector}（未知层级直接拒绝）
    3. **聚合与层级匹配**：`sec_ret_xrank`/`sec_amt_share_chg` 本身即变化/排名口径，
       生成器只取 `level`；此处复读同一约束，防未来放宽生成器时漏检
    """
    op = spec.get("op")
    qwin = int(qwin_actual or 252)
    if qwin < 63:
        return False, f"实际生成分位窗 {qwin} < 63，估计噪声过大"
    if spec.get("level") not in ("market", "sector"):
        return False, f"未知层级 {spec.get('level')}"
    if spec.get("base") in ("sec_ret_xrank", "sec_amt_share_chg") and op != "level":
        return False, f"{spec.get('base')} 本身为变化/排名口径，只允许 level 操作化"
    return True, ""


def dust_check(p: np.ndarray) -> tuple[bool, float]:
    """④ 尘埃检测：p 序列与时间趋势的秩相关过高 → 退化为时间趋势（非状态量）。

    对照 `run_search.dust_errors`（ADD/SUB 数量级差 ≥100× → 装饰项）的语义：
    这里检测"门控值其实只是时间在走"。（B7：删除恒为 None 的 `t_idx` 死参数）
    """
    v = np.asarray(p, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 60:
        return True, float("nan")
    t = np.arange(len(v), dtype=float)[ok]
    rho = float(pd.Series(v[ok]).corr(pd.Series(t), method="spearman"))
    return (abs(rho) < 0.9), rho          # |ρ|≥0.9 → 判为退化


def sparsity_check(p: np.ndarray, hi: float = G.THRESH_HI) -> dict:
    """⑨ 稀疏性：on 天数占比 / p 序列独立值个数。仅告警。"""
    v = np.asarray(p, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return {"on_frac": float("nan"), "n_unique": 0, "warn": True}
    on = float((v[ok] >= hi).mean())
    nu = int(pd.Series(np.round(v[ok], 4)).nunique())
    return {"on_frac": on, "n_unique": nu,
            "warn": bool(on < 0.05 or nu < 20)}


def gate_score(row: dict) -> float:
    """S4 排序分（预注册权重；仅用于截断）。

    三项均可变且与"门控质量"正相关：效应量 / 组内一致性 / 方向中性度。
    家族重叠**不在**此列（见 `GATE_SCORE_W` 的说明）。
    """
    w = GATE_SCORE_W
    return (w["abs_stat"] * abs(row.get("median_b") or 0)
            + w["same_frac"] * (row.get("same_frac") or 0)
            + w["neutrality"] * (1 - abs(row.get("dir_corr") or 0)))


# ------------------------------------------------------------------ 主流程
def read_ic_cache(path: str) -> pd.DataFrame:
    """B2（2026-10-12）：读市场线 IC 缓存 + **血缘 sidecar + CUT 守卫**。

    - 守卫：缓存最大日期若 > CUT（FACTOR_SCREEN_MAX）→ 截断并告警（越界防泄漏）
    - sidecar（`<cache>.prov.json`）记录生成时的因子集哈希/日期范围/数据版本；
      若存在且 `factor_hash` 与当前文件实际列不符 → **拒绝运行**（口径被静默更换）
    - sidecar 缺失 → 首次运行自动补写（标注 auto_created），此后每次运行比对
    """
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if len(df.index) and df.index.max() > pd.Timestamp(CUT):
        n0 = len(df.index)
        df = df[df.index <= pd.Timestamp(CUT)]
        print(f"  ⚠ IC 缓存越界截断：{os.path.basename(path)} {n0}→{len(df)} 行 "
              f"（max {df.index.max().date()}；CUT={CUT}）", flush=True)
    cols = [str(c) for c in df.columns]
    fh = hashlib.sha1("|".join(sorted(cols)).encode()).hexdigest()[:12]
    prov_path = path + ".prov.json"
    if os.path.exists(prov_path):
        try:
            with open(prov_path, encoding="utf-8") as f:
                prov = json.load(f)
            if str(prov.get("factor_hash")) != fh:
                raise SystemExit(
                    f"IC 缓存血缘不一致：{os.path.basename(path)} factor_hash "
                    f"{prov.get('factor_hash')} ≠ 当前 {fh}（因子集被静默更换）——"
                    f"重新生成缓存或显式 --ic-cache 指向一致版本")
        except (ValueError, OSError):
            pass
    else:
        prov = {"auto_created": True, "created": pd.Timestamp.today().isoformat(),
                "factor_hash": fh, "n_factors": len(cols),
                "date_min": str(df.index.min().date()) if len(df.index) else None,
                "date_max": str(df.index.max().date()) if len(df.index) else None,
                "data_version": B.factor_data_version()}
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(prov, f, ensure_ascii=False, indent=1)
        print(f"  血缘 sidecar 自动创建 → {os.path.basename(prov_path)}（首次运行）",
              flush=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="门控统一批量流水线（S0→S7）")
    ap.add_argument("--all", action="store_true", help="忽略台账，全跑")
    ap.add_argument("--no-f1", action="store_true",
                    help="（旧名，已废弃）抑制 F1 下一步命令打印；等价 --quiet-next")
    ap.add_argument("--quiet-next", action="store_true",
                    help="抑制 F1 下一步命令打印（B6：流水线从不自动运行 F1，"
                         "旧名 --no-f1 名不符实，保留为别名）")
    ap.add_argument("--space-dir", default=SPACE)
    ap.add_argument("--ledger", default=GL.DEFAULT_PATH)
    # B2（2026-10-12）：市场线 IC 缓存此前硬编码且无血缘记录——数据截止日移动会
    # 静默使用旧 IC。现给参数 + 血缘 sidecar + CUT 守卫（见 read_ic_cache）。
    ap.add_argument("--ic-cache", default=os.path.join(RESULTS, "ic_stock_2025.parquet"))
    ap.add_argument("--corr-max", type=float, default=0.75,
                    help="方向中性上限（V4.3放宽：flip/switch 以牛熊趋势门控为最佳"
                         "开关，旧 0.60 误杀 newhigh 系；>0.80 纯指数克隆仍拦截）")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--stop-state", default=GS.DEFAULT_PATH,
                    help="停止规则状态文件（§17.7）")
    ap.add_argument("--limit-gates", type=int, default=0,
                    help=">0 只跑前 N 个门控（**冒烟用**，不作统计结论）")
    ap.add_argument("--topn", type=int, default=0,
                    help="S7 短名单只保留 S4 排名的前 N（0=全部 G2∧中性存活者；"
                         "**注意**：topn 只影响短名单，逐个检验全族 720 仍按 F1v2 清单整体运行，"
                         "族逐一对应校验不受 topn 影响）")
    ap.add_argument("--base-panel", choices=("base20", "vwap_top20"), default="base20",
                    help="F1下一步候选的因子面板（默认base20保证历史可复现；"
                         "vwap_top20读build_top导出的动态正交面板，不改变G2/G3组级统计）")
    ap.add_argument("--gate-bank", default=None,
                    help="已认证门控字典certified_gate_bank.json：cands的门控与s改从bank读取"
                         "（离线G2符号锁定）；不传则用本轮短名单×s=+1历史行为")
    ap.add_argument("--by-group", action="store_true", default=True,
                    help="V4.3c 机制分簇G2（默认开）：按 roles.csv group 先验簇分别跑斜率"
                         "口径；t2_promoted/ungrouped不成簇；全池已过用全池median_b，"
                         "仅全池未过时任一真机制簇达标才拯救；旧全池结论保留")
    ap.add_argument("--no-by-group", dest="by_group", action="store_false",
                    help="关闭机制分簇G2（纯全池口径对照）")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.time()
    lg = GL.GateLedger(a.ledger)

    # ---- B1（2026-10-12）：IC 缓存先读，caliber 记录**实际因子集** ----
    # 旧实现写死 factor_set="base20" 但实际用 124 因子 → 口径记录不实，
    # 因子集变更不会被 stale 检测发现。真实 factor_set = f"ic{实际列数}"。
    icdf = read_ic_cache(a.ic_cache)
    # V4.3c 机制分簇映射（roles.csv group；t2_promoted/ungrouped 不成簇，
    # 仅留全池——循环论证嫌疑）。
    _groups: dict[str, list[int]] = {}
    if a.by_group:
        try:
            _roles = pd.read_csv(os.path.join(RESULTS, "roles.csv"))
            _gmap = dict(zip(_roles["name"].astype(str), _roles["group"].astype(str)))
            for _j, _c in enumerate(icdf.columns):
                _g = _gmap.get(str(_c), "ungrouped")
                if _g in ("t2_promoted", "ungrouped"):
                    continue
                _groups.setdefault(_g, []).append(_j)
            print(f"  分簇G2：{len(_groups)} 真机制簇 "
                  f"{sorted((g, len(v)) for g, v in _groups.items())}", flush=True)
        except Exception as exc:
            print(f"  warning: 分簇映射失败（{str(exc)[:100]}），退化为全池", flush=True)
            _groups = {}
    # 面板内容哈希（2026-10-13 增量筛选补丁）：不进 caliber 就会出现"同 spec
    # 面板被重生成（gate_search_space 重跑）→ 台账误判 completed 而漏重筛"。
    reg_path_p = os.path.join(a.space_dir, "gate_search_registry.json")
    _panel_sha = None
    if os.path.exists(reg_path_p):
        try:
            with open(reg_path_p, encoding="utf-8") as f:
                _panel_sha = json.load(f).get("market_sha12") or None
        except (ValueError, OSError):
            _panel_sha = None
    if not _panel_sha:
        print("  ⚠ panel_sha 缺失（registry 不存在/无 market_sha12）——caliber 将不含面板哈希",
              flush=True)
    CALIBER = make_caliber(a.n_perm, a.n_boot, factor_set=f"ic{icdf.shape[1]}",
                           panel_sha=_panel_sha, form="g")
    print(f"caliber form = g（管线 G2/G3 的门控值 = 2p−1 斜率尺，§16.30 裁决）",
          flush=True)
    print(f"caliber factor_set = {CALIBER['factor_set']}（实际读入 {icdf.shape[1]} 因子；"
          f"base20 是 F1 候选标签集，不是 G2 的组级统计分母）", flush=True)

    # ---- S0 候选摄入 ----
    specs = [s for s in GSS.generate_space() if s["level"] == "market"]
    if a.limit_gates:
        specs = specs[:a.limit_gates]
    print(f"S0 摄入 {len(specs)} 个市场级规格"
          f"{'（冒烟：仅前 %d 个）' % a.limit_gates if a.limit_gates else ''}")

    # ---- S1 指纹 + 台账对账 ----
    if a.all:
        pend = [{**s, "_state": "forced"} for s in specs]
    else:
        pend = lg.pending(specs, CALIBER)
    n_new = sum(1 for p in pend if p["_state"] == "untested")
    n_stale = sum(1 for p in pend if p["_state"] == "stale")
    print(f"S1 指纹对账：待执行 {len(pend)}/{len(specs)} "
          f"（untested {n_new} / stale {n_stale} / completed "
          f"{len(specs) - len(pend)}）")
    if not pend:
        print("无待执行候选（台账已覆盖当前口径）")
        return 0

    # ---- 载入门控序列 + 方向 ----
    gm = pd.read_parquet(os.path.join(a.space_dir, "gate_search_market.parquet"))
    gm["date"] = pd.to_datetime(gm["date"])
    gm = gm[(gm["date"] >= W.FACTORS_FIRST_SIGNAL_DATE) & (gm["date"] <= CUT)]
    mkt = G.load_market_daily()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date").reset_index(drop=True)
    mkt["cum20"] = mkt["r_m"].rolling(20, min_periods=20).sum()
    mkt["cum60"] = mkt["r_m"].rolling(60, min_periods=60).sum()
    for w_ in (20, 60):
        mkt[f"cum{w_}_p"] = G._roll_quantile(mkt[f"cum{w_}"], G.QWIN,
                                             G.QMIN["market"]).to_numpy()
    md = mkt.set_index("date")[["cum20_p", "cum60_p"]]

    import gate_validate as GV
    import gate_g2g3 as G23

    # B1：icdf 已在 S1 前读入（read_ic_cache：血缘 + CUT 守卫）
    orient = np.sign(icdf.mean(axis=0, skipna=True).to_numpy(dtype=float))
    orient[orient == 0] = 1.0
    ic = icdf.mul(orient, axis=1).to_numpy(dtype=float)

    # B4（2026-10-12）：规格合法性的分位窗读**实际生成值**（生成器 registry 记录），
    # 旧实现读 spec（specs 无 qwin 键）恒等 252 → 检查恒不触发（死逻辑）。
    reg_path = os.path.join(a.space_dir, "gate_search_registry.json")
    qwin_actual = 252
    if os.path.exists(reg_path):
        try:
            with open(reg_path, encoding="utf-8") as f:
                qwin_actual = int(json.load(f).get("qwin") or 252)
        except (ValueError, OSError):
            pass
    else:
        print("  ⚠ 未找到 gate_search_registry.json（spec_legality 退回默认 qwin=252）",
              flush=True)

    rows = []
    n_leg_fail = n_dust = n_sparse = n_vfp_dup = n_struct = 0
    seen_ofp: dict[str, str] = {}
    print(f"\n{'gate':<28s}{'合法':>4s}{'尘埃':>6s}{'稀疏':>6s}{'G0':>6s}"
          f"{'dir':>6s}{'G2':>5s}{'G3':>5s}{'score':>8s}")
    for spec in pend:
        gid = spec["gate_id"]
        sub = gm[gm["gate_id"] == gid].sort_values("date")
        if not len(sub):
            continue
        d = pd.DataFrame({"date": sub["date"].to_numpy(),
                          "p": sub["p"].to_numpy(dtype=float)})

        # ---- S2 预筛链 ----
        ok_leg, why_leg = spec_legality(spec, qwin_actual)
        if not ok_leg:
            n_leg_fail += 1
            lg.log("screened", spec, caliber=CALIBER, verdict="reject_legality",
                   data={"reason": why_leg})
            continue
        ok_dust, rho_t = dust_check(d["p"].to_numpy())
        if not ok_dust:
            n_dust += 1
            lg.log("screened", spec, caliber=CALIBER, verdict="reject_dust",
                   data={"rho_time": rho_t, "reason": "退化为时间趋势"})
            continue
        sp = sparsity_check(d["p"].to_numpy())
        if sp["warn"]:
            n_sparse += 1
        # ⑥ 排序等价去重（2026-10-12 修正）：判据 = 门控**开启时点**相同。
        # **不是**个股线的 value_fp 语义——那条线作用于截面**因子值**（仿射等价 ⇔
        # 同一排序 ⇔ 同一因子）；门控是**时序**对象，边际相同 ≠ 同一门控。
        # 实测：行业门控 p = 逐日跨行业 rank → 边际恒均匀 → value_fp 只出 2 个
        # 指纹，用作拒绝会把 12/14 误判为重复。详见
        # results/merge_v2/gate_space_redundancy_audit_2026-10-12.md §3.2
        ofp = GL.gate_order_fp(d["p"].to_numpy())
        if ofp and ofp in seen_ofp:
            n_vfp_dup += 1
            lg.log("screened", spec, caliber=CALIBER, verdict="reject_order_dup",
                   data={"order_fp": ofp, "dup_of": seen_ofp[ofp],
                         "reason": "开启时点与已有门控完全相同（单调等价）"})
            continue
        if ofp:
            seen_ofp[ofp] = gid
        g0b = GV.g0_binary(d["date"], d["p"])
        g0c = GV.g0_continuous(d["date"], 2 * d["p"] - 1,
                               eval_start=W.FACTORS_FIRST_SIGNAL_DATE)
        dd = d.set_index("date").join(md, how="left")
        c20 = float(dd["p"].corr(dd["cum20_p"], method="spearman"))
        c60 = float(dd["p"].corr(dd["cum60_p"], method="spearman"))
        cmax = float(np.nanmax(np.abs([c20, c60])))
        neutral = cmax < a.corr_max
        vfp = GL.gate_value_fp(d["p"].to_numpy())   # 诊断记录（不作判定）
        # 输入结构自检（2026-10-12）：对象被压平/退化时**拒绝继续**，
        # 防复现 G4 v1 与 L2a 的"程序正常输出、结果无意义"错误
        ok_st, st_rep = GL.structure_check(d["p"].to_numpy(), name=gid)
        if not ok_st:
            n_struct += 1
            lg.log("screened", spec, caliber=CALIBER, verdict="reject_structure",
                   data={"structure": st_rep,
                         "reason": "输入对象退化：" + "; ".join(st_rep["why"])})
            continue
        lg.log("screened", spec, caliber=CALIBER,
               data={"g0_binary": g0b["g0_status"], "g0_cont": g0c["g0_status"],
                     "cov": g0c.get("cov"), "std": g0c.get("std"),
                     "seg_med": g0b.get("seg_med"), "flips_py": g0b.get("flips_py"),
                     "dir_corr": cmax, "neutral": bool(neutral),
                     "value_fp": vfp, "order_fp": ofp, "sparsity": sp,
                     "rho_time": rho_t, "structure": st_rep})

        # ---- S5 组级评估（G2/G3）----
        gser = 2 * d["p"].to_numpy(dtype=float) - 1
        gser = pd.Series(gser, index=d["date"]).reindex(icdf.index).to_numpy(float)
        st = G23.g2_stat(ic, gser)
        lo, hi = G23.g2_ci_bootstrap(ic, gser, n_boot=a.n_boot)
        g3 = G23.g3_placebo(ic, gser, n_perm=a.n_perm)
        g2_pass = bool((st["folds_same"] == G23.G2_FOLDS)
                       and G23.ci_excludes_zero(lo, hi)
                       and st["same_frac"] >= G23.G2_SAME_SIGN_MIN)
        # V4.3c 机制分簇 G2（新检验族版本）：全池已过则用全池 median_b
        #（slope_group 记 full_pool），仅全池未过时启用分簇拯救。
        _grp, _geff = "full_pool", float(st["median_b"])
        _gsame, _gfolds = float("nan"), 0
        _gok = False
        if _groups and not g2_pass:
            _best = None
            for _gn, _cols in _groups.items():
                if len(_cols) < 5:
                    continue
                _sic = ic[:, _cols]
                _sst = G23.g2_stat(_sic, gser)
                _lo, _hi = G23.g2_ci_bootstrap(_sic, gser, n_boot=a.n_boot)
                _ok = (_sst["folds_same"] == G23.G2_FOLDS) \
                    and G23.ci_excludes_zero(_lo, _hi) \
                    and _sst["same_frac"] >= 0.65
                if _ok and (_best is None
                            or abs(_sst["median_b"]) > abs(_best[1])):
                    _best = (_gn, _sst["median_b"], _sst["same_frac"],
                             _sst["folds_same"])
            if _best is not None:
                _grp, _geff, _gsame, _gfolds = _best
                _gok = True
                g2_pass = True
        row = {"gate_id": gid, "level": spec["level"], "base": spec["base"],
               "op": spec["op"], "state": spec["_state"],
               "g0_cont": g0c["g0_status"], "neutral": neutral,
               "dir_corr": cmax, "median_b": st["median_b"],
               "same_frac": st["same_frac"], "folds_same": st["folds_same"],
               "ci_lo": lo, "g2_pass": g2_pass, "g3_pass": g3["pass"],
               "g3_p": g3.get("p_two_sided"), "value_fp": vfp,
               "on_frac": sp["on_frac"], "rho_time": rho_t,
               "slope_group": _grp, "median_b_eff": _geff,
               "slope_group_same": _gsame, "slope_group_folds": _gfolds,
               "slope_group_pass": bool(_gok)}
        lg.log("g2g3", spec, caliber=CALIBER,
               verdict=("candidate" if (g2_pass and neutral) else
                        "proxy" if not neutral else
                        "weak" if g2_pass else "reject"),
                data={"median_b": st["median_b"], "same_frac": st["same_frac"],
                      "folds_same": st["folds_same"], "ci_lo": lo,
                      "g2_pass": g2_pass, "g3_pass": g3["pass"],
                      "g3_p": g3.get("p_one_sided"), "value_fp": vfp,
                      "dir_corr": cmax, "slope_group": _grp,
                      "median_b_eff": _geff, "slope_group_pass": bool(_gok)})
        rows.append(row)
        print(f"{gid:<28s}{'OK':>4s}{'OK':>6s}"
              f"{('WARN' if sp['warn'] else 'OK'):>6s}{g0c['g0_status']:>6s}"
              f"{cmax:>6.2f}{str(g2_pass):>5s}{str(g3['pass']):>5s}")

    df = pd.DataFrame(rows)
    if not len(df):
        print("\n无存活候选")
        return 0

    # 家族规模：**仅作报告列**（B7：删除上述矛盾注释——`gate_score` 自 2026-10-11 起
    # 不含家族惩罚项：同一基础量的 4 个操作化构成等规模家族，family_multi 恒为 4，
    # 二值惩罚对所有候选取值相同 → 恒无区分力；取舍属 S6 去冗余，不属排序分）。
    fam_size = {}
    for spec in pend:
        fam_size[spec["gate_id"]] = sum(
            1 for s2 in pend if s2["level"] == spec["level"]
            and s2["base"] == spec["base"])
    df["family_multi"] = df["gate_id"].map(fam_size).fillna(0).astype(int)
    df["score"] = df.apply(lambda r: gate_score(r.to_dict()), axis=1)
    df = df.sort_values("score", ascending=False)

    # ---- S7 短名单 ----
    short = df[(df["g2_pass"]) & (df["neutral"])].copy()
    if a.topn and len(short) > a.topn:
        short = short.head(a.topn)
        n_alive = int((df["g2_pass"] & df["neutral"]).sum())
        print(f"  --topn={a.topn}: 短名单按 S4 排名截取 {len(short)} 个"
              f"（全部存活者 {n_alive}）")
    dst = os.path.join(a.space_dir, "pipeline_shortlist.csv")
    df.to_csv(os.path.join(a.space_dir, "pipeline_all.csv"),
              index=False, encoding="utf-8")
    short.to_csv(dst, index=False, encoding="utf-8")
    print(f"\n=== S4/S7 汇总 ===")
    print(f"  预筛淘汰：规格非法 {n_leg_fail} / 退化尘埃 {n_dust} / "
          f"排序等价重复 {n_vfp_dup} / 输入结构退化 {n_struct}")
    print(f"  稀疏告警：{n_sparse}（告警不设门）")
    print(f"  G2 且方向中性（短名单）：{len(short)}/{len(df)}")
    for _, r in short.iterrows():
        print(f"    {r['gate_id']:<28s} 中位斜率 {r['median_b']:+.5f} "
              f"同号 {r['same_frac']:.2f} dir={r['dir_corr']:.2f} "
              f"G3={r['g3_pass']} score={r['score']:.4f}")

    # ---- S6 ⑤ G4 独立性（2026-10-12 口径裁决：**series（时序）**；记录+告警，不设硬门）
    # 依据：时序口径中位 |ρ|=0.064 稳定且与"门控何时开启"（研究对象）一致；
    # 效应向量口径被 20 因子自身共线污染（中位 |ρ|=0.825）且白化结论随正则化翻转
    # （0/630↔246/630）⇒ 剔除出 G4 候选（见冗余审计 §10）。
    # 纪律（§17.5）：先"记录+告警"，**不得**未经论证就变成淘汰条件。
    g4_rep = None
    if len(df) >= 2:
        try:
            import gate_g4_clusters as G4C
            eval_ids = list(df["gate_id"])          # 只对**实际评估过**的门控算
            ser_g4 = (gm.pivot_table(index="date", columns="gate_id",
                                     values="p", aggfunc="mean")
                      .sort_index().reindex(columns=eval_ids))
            rho, mode_note = G4C.corr_matrix(ser_g4, "market", "series")
            thr = 0.8
            cl = G4C.cluster(rho, thr)
            vals = rho.to_numpy(float)
            off = vals[np.triu_indices(len(eval_ids), 1)]
            fin = off[np.isfinite(off)]
            n_over = int((fin >= thr).sum())
            multi = [c for c in cl if len(c) > 1]
            g4_rep = {"mode": "series", "thr": thr, "n_gates": len(eval_ids),
                      "n_pairs_finite": int(len(fin)), "n_pairs_over": n_over,
                      "rho_median": (float(np.median(fin)) if len(fin) else float("nan")),
                      "rho_max": (float(np.max(fin)) if len(fin) else float("nan")),
                      "n_clusters": len(cl), "n_multi_clusters": len(multi),
                      "note": mode_note}
            lg.log("g4", {"level": "market", "op": None}, caliber=CALIBER,
                   verdict="diagnostic", data=g4_rep)
            if n_over:
                print(f"  ⚠ G4(series, 0.8)：{n_over} 对超阈（中位 |ρ|={np.median(fin):.3f}"
                      f"，max={fin.max():.3f}）→ {len(cl)} 簇；候选**非全部独立**，"
                      f"证据不得按 {len(eval_ids)} 份独立证据计数（台账 g4 事件已记录）")
            else:
                print(f"  G4(series, 0.8)：无超阈对（中位 |ρ|={np.median(fin):.3f}）"
                      f"→ {len(cl)} 簇")
            # 落盘（**_pipeline 后缀**：不得覆盖独立 CLI `gate_g4_clusters.py` 的
            # 基准产物——其口径是 screen eligible 36 门控；管线是评估后集合，
            # 两者集合不同，结果也不同（路径纪律，见冗余审计 §11））
            rho.to_csv(os.path.join(a.space_dir, "g4_rho_market_pipeline.csv"),
                       encoding="utf-8")
            with open(os.path.join(a.space_dir, "g4_clusters_market_pipeline.csv"),
                      "w", encoding="utf-8") as f:
                f.write("cluster_id,n_members,members\n")
                for i_, c in enumerate(cl, 1):
                    f.write('%d,%d,"%s"\n' % (i_, len(c), ";".join(c)))
        except Exception as e:                      # G4 失败不阻断判定（记录+告警层）
            print(f"  G4 失败（不影响判定）：{type(e).__name__}: {str(e)[:120]}")

    # ---- S7.2 decided 事件（状态机：短名单 = qualified_g2g3，⇒ ≠ 入池）----
    for _, r in short.iterrows():
        lg.log("decided",
               {"level": r["level"], "base": r["base"], "op": r["op"]},
               caliber=CALIBER, verdict="candidate",
               data={"state": "qualified_g2g3", "gate_id": r["gate_id"],
                     "median_b": float(r["median_b"]),
                     "same_frac": float(r["same_frac"]),
                     "folds_same": int(r["folds_same"]),
                     "g2_pass": bool(r["g2_pass"]), "g3_pass": bool(r["g3_pass"]),
                     "g3_p": (None if pd.isna(r["g3_p"]) else float(r["g3_p"])),
                     "dir_corr": float(r["dir_corr"]), "score": float(r["score"]),
                     "note": "qualified_g2g3 ≠ 入池：仍须 F1（候选层）与"
                             "模型级增量对照（外部生产引擎 engine=stage15 "
                             "双运行，§17.9 #8 未做）"})

    # ---- S7.3 报告 MD（每次运行覆盖；§17.3 S7 的报告义务）----
    rep_path = os.path.join(a.space_dir, "pipeline_report.md")
    rep = ["# 门控统一流水线运行报告", "",
           f"- 运行时间：{pd.Timestamp.today().isoformat(timespec='seconds')}",
           f"- 口径：window_end={CUT} / form=双口径 / label=r3 / "
           f"data_version={CALIBER.get('data_version')} / factor_set={CALIBER.get('factor_set')} "
           f"/ n_boot={a.n_boot} / n_perm={a.n_perm}",
           f"- 输入 {len(specs)} 个（--all={'是' if a.all else '否'}）；"
           f"S2 预筛淘汰：非法 {n_leg_fail} / 尘埃 {n_dust} / 排序等价 {n_vfp_dup} / "
           f"结构 {n_struct}；稀疏告警 {n_sparse}",
           f"- 短名单（G2 且方向中性）：{len(short)}/{len(df)}；"
           f"台账：{json.dumps(lg.stats(), ensure_ascii=False)}",
           ""]
    if g4_rep is not None:
        rep += [f"- G4(series, |ρ|≥{thr})：超阈 {g4_rep['n_pairs_over']} 对 / "
                f"{g4_rep['n_gates']} 门控 → {g4_rep['n_clusters']} 簇"
                f"（中位 {g4_rep['rho_median']:.3f}）；记录+告警，不设硬门", ""]
    if len(short):
        rep += ["## 短名单（qualified_g2g3 ≠ 入池）", "",
                "| gate | ΔIC中位 | 同号 | 4折 | CI下界 | dir | G3 | score |", "|---|---|---|---|---|---|---|---|"]
        for _, r in short.iterrows():
            rep.append(f"| {r['gate_id']} | {r['median_b']:+.5f} | "
                       f"{r['same_frac']:.2f} | {int(r['folds_same'])}/4 | "
                       f"{r['ci_lo']:+.5f} | {r['dir_corr']:.2f} | "
                       f"{'pass' if r['g3_pass'] else 'fail'} | {r['score']:.4f} |")
        rep += ["", "**状态语义**：qualified_g2g3 = 获得候选层验证资格；"
                    "仍须 F1（f1_batch + judge_f1 --family）与模型级增量"
                    "（外部生产引擎对照，merge 内不实现）。"]
    else:
        rep += ["## 短名单", "", "（无）"]
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")

    # ---- §17.7 停止规则：硬预算 + 方向级暂停（keeper = 通过 G2 且方向中性）----
    stop = GS.StopState.load(a.stop_state)
    pause, why = stop.record_batch(len(df), len(short))
    stop.save(a.stop_state)
    print(f"\n=== 停止规则（§17.7）===")
    print(stop.report())
    if pause:
        print(f"  ⚠ 触发方向级暂停：{why}")
        print(f"  （暂停=诊断该方向，不自动永久终止；重新启动需记录原因）")

    print(f"\n输出 → {dst}")
    print(f"台账 → {a.ledger}")
    print(f"统计 → {json.dumps(lg.stats(), ensure_ascii=False)}")
    print(f"用时 {time.time() - t0:.0f}s")

    if not (a.quiet_next or a.no_f1) and len(short):
        cp = os.path.join(a.space_dir, "cands_pipeline_shortlist.json")
        if a.base_panel == "vwap_top20":
            _vt = json.loads(open(os.path.join(RESULTS, "vwap_top20.json"),
                                  encoding="utf-8").read())
            b20 = [{"factor": x["factor"], "direction": int(x["direction"])}
                   for x in _vt.get("base20", [])]
            print(f"  F1候选面板=vwap_top20（{len(b20)}因子；G2/G3组级统计仍用ic缓存，不受影响）",
                  flush=True)
        else:
            b20 = json.loads(open(os.path.join(RESULTS, "base20.json"),
                                  encoding="utf-8").read())["base20"]
        if a.gate_bank:
            bank = json.loads(open(a.gate_bank, encoding="utf-8").read())
            blist = [e for e in bank.get("entries", []) if e.get("level") == "market"]
            print(f"  cands门控来源=bank {a.gate_bank}（{len(blist)}市场门控，s按G2锁定）",
                  flush=True)
        else:
            blist = [{"gate_id": r["gate_id"], "s": 1} for _, r in short.iterrows()]
        cands = []
        for b in blist:
            for x in b20:
                cands.append({"name": f"{x['factor']}__{b['gate_id']}__pos",
                              "base": x["factor"], "gate": b["gate_id"],
                              "gate_type": "market", "s": int(b["s"]),
                              "d": int(x["direction"])})
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(cands, f, ensure_ascii=False)
        gp = os.path.join(a.space_dir, "market_gates_search.parquet")
        print(f"\n下一步（F1 候选级评估，{len(cands)} 个候选；正式口径=斜率尺 g，"
              f"§16.30 裁决，form t 已降为诊断）：")
        print(f"  python f1_batch.py --cands {cp} "
              f"--out {os.path.join(a.space_dir, 'f1_pipeline.csv')} "
              f"--tag pipe --form g --base-panel {a.base_panel} --gates-parquet {gp}")
        print(f"  python judge_f1.py --family "
              f"{os.path.join(a.space_dir, 'family_manifest_F1.json')} "
              f"{os.path.join(a.space_dir, 'f1_pipeline.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
