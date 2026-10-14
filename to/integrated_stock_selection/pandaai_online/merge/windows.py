# -*- coding: utf-8 -*-
"""数据窗口单一事实源（§3.4 / §10.4 / §11.2）。

此前窗口上界散落在各脚本的默认参数与 `base20.json` 的 `label_max` 里，
且没有记录**下界**，导致"已搜索过什么、用的哪段数据"无法审计。本模块把
计划规定的三个日期与 V1–V8 版本窗口集中定义，并提供窗口构造/校验工具。

三个日期不可混用（§10.4 边界表）：
- SIGNAL_MAX_DATE  允许进入池/训练/选择的最晚**信号日**
- LABEL_MAX_DATE   为这些信号算标签时，行情最晚可用日（label_available_max_date）
- FORWARD_FIRST_SIGNAL_DATE  第一个独立前向信号日 = E3 起点（§11.2）

D13：数据截至 2026-09-18 足够架构研究；之后数据继续更新，流程向前滚动。
因此新增数据后应重算生产窗口（见 `refresh_prod_window`），但**已冻结版本的
D_k 不得随之改动**（C17/C19）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path

import hashlib

# ---------------------------------------------------------------- 三个关键日期
#: 数据上界：允许读取的最晚**信号日**（标签完整）。
SIGNAL_MAX_DATE = "2026-09-11"
#: 标签完整可得日：2026-09-11 信号的 R5 需要 2026-09-18 的价格。
LABEL_MAX_DATE = "2026-09-18"
#: 第一个独立前向信号日（E3 起点）。2026-09-14~09-18 的行情已用于生产训练
#: 标签的形成，**不得**作为独立前向数据（§11.2）。
FORWARD_FIRST_SIGNAL_DATE = "2026-09-21"

# ---------------------------------------------------------------- 选择/监控分界
# 2026-10-10 用户决定：把「选择」与「监控」彻底分开。
FACTOR_SCREEN_MAX = "2025-12-31"
CV_MAX = "2025-12-31"
TRAIN_SIGNAL_MAX = "2025-12-31"
DECAY_MONITOR_START = "2026-01-05"

# ---------------------------------------------------------------- 已消耗区段（审计认定）
#: 【2026-10-11 审计认定】个股进化搜索 holdout 实际覆盖的区段。
#: 证据：ledger.jsonl 的 6 条 holdout 事件 monthly_excess 覆盖 2025-07 ~ 2026-07，
#: 且 holdout 的 SiA 进入 fitness.rank_score（min(train,holdout)）→ 影响精英保留与撒种。
#: 后果：该区段（含 2026 年内 154 个交易日）**不得追认为独立确认段**。
#: 详见 results/merge_v2/data_contact_audit_2026-10-11.md。
CONSUMED_BY_STOCK_SEARCH = ("2025-07-01", "2026-08-21")
#: 2026 段中**疑似未触碰**的残余区间（仅 15 个交易日；是否可用须先完成审计 §4-2）。
UNTOUCHED_2026_REMNANT = ("2026-08-22", "2026-09-11")

# ---------------------------------------------------------------- 指纹数据截止日
#: **指纹的数据截止日**（2026-10-11 用户指令）：候选身份 = 公式 × 数据截止日。
#: 默认 = 选择窗上界。搜索/筛选一律使用该截止日的指纹；历史遗留截止日的指纹不参与。
DEFAULT_DATA_CUT = FACTOR_SCREEN_MAX  # "2025-12-31"
#: 历史遗留截止日：该期间个股线 END=2026-08-21（fitness.py），2026 年数据已被消耗。
#: 仅用于**标记**旧台账记录，不得作为新搜索的截止日。
LEGACY_DATA_CUT = "2026-08-21"
#: 可接受的截止日：不早于面板首个信号日、不晚于 SIGNAL_MAX_DATE。
#: （2026-10-11 放宽：原下限 2024-12-26 是人为设定，会挡住"更早截止日"的合法重放；
#: 真正的下限是"面板里有没有数据"，即 FACTORS_FIRST_SIGNAL_DATE = "2021-06-01"。）
DATA_CUT_MIN = "2021-06-01"


def normalize_data_cut(value: str | None) -> str:
    """把截止日参数规范成 ISO 日期串；None/空 → 默认截止日。

    支持 `auto` / `default`（→ DEFAULT_DATA_CUT）与 `legacy`（→ LEGACY_DATA_CUT）。
    """
    if value is None:
        return DEFAULT_DATA_CUT
    s = str(value).strip()
    if not s:
        return DEFAULT_DATA_CUT
    low = s.lower()
    if low in ("auto", "default", "prod", "selection"):
        return DEFAULT_DATA_CUT
    if low == "legacy":
        return LEGACY_DATA_CUT
    d = date.fromisoformat(s.replace("/", "-"))
    if d < date.fromisoformat(DATA_CUT_MIN):
        raise ValueError(f"data_cut {s} 早于允许下限 {DATA_CUT_MIN}")
    if d > date.fromisoformat(SIGNAL_MAX_DATE):
        raise ValueError(f"data_cut {s} 晚于数据上界 {SIGNAL_MAX_DATE}")
    return d.isoformat()


def check_data_cut(value: str | None) -> list[str]:
    """返回违规项（空 = 合规）。新搜索/筛选的截止日必须 ≤ DECAY_MONITOR_START 前一日。"""
    bad: list[str] = []
    try:
        cut = normalize_data_cut(value)
    except ValueError as exc:
        return [str(exc)]
    if cut >= DECAY_MONITOR_START:
        bad.append(f"data_cut {cut} 触及衰减监控段（≥{DECAY_MONITOR_START}）："
                   f"新搜索/筛选不得使用包含该段数据的截止日")
    return bad


#: 显式关闭截止日裁剪（历史重放 / 单元测试用）。**不得**用于生产搜索。
CUT_OFF = "off"


def _root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


#: 数据版本标识的缓存（按 (size, mtime_ns) 缓存 path→内容哈希，避免每次重算 GB 级文件）
DATA_VERSION_CACHE = _root() / "results" / "merge_v2" / "data_version_cache.json"
#: **声明式**数据版本（指纹口径用）。只在用户显式刷新时改变，**不随日常更新变化**。
DECLARED_VERSION_FILE = _root() / "results" / "merge_v2" / "data_version.json"

#: 默认声明的数据版本（bootstrap 用；当前 factors.parquet / cache 面板的内容哈希前缀）
DEFAULT_DECLARED_VERSIONS = {"factors": "factors_abe94a59", "panel": "panel_c2ece310"}


def declared_data_version(kind: str, source_path: str | Path | None = None,
                          auto_declare: bool = True) -> str:
    """返回**声明式**数据版本（`results/merge_v2/data_version.json` 的值）。

    **为什么不直接用 `file_data_version`（内容哈希）作指纹口径**（2026-10-11 用户裁定）：

    `cache/stock_features_*` 是**日频重生成**的（文件名带日期、内容每日追加），
    `factors` 也有日频变体。若指纹以内容哈希为作用键，则**每个交易日全部指纹失效**，
    「已测过，跳过」这个机制就彻底失去意义——而日频更新对**窗内（≤ cut_date）**
    历史数据的影响极小（追加的是窗后数据；比率型量对复权不敏感）。

    因此：**指纹的作用键只有 `cut_date`**；数据版本降级为**溯源记录**
    （写进每条事件，供审计回答「这个数字出自哪个数据世代」），**不参与过滤**。

    本函数提供**唯一**的、**显式**的换代杠杆：数据发生**实质变化**时
    （批量复权重算、数据源更换、修数据），用 `data_version.py --refresh --kind X`
    声明新版本——门控 caliber 会随之改变并触发补测。日常更新**不会**变。

    文件缺失且 `auto_declare=True` 时，按当前内容哈希 bootstrap 一次并落盘
    （默认值见 `DEFAULT_DECLARED_VERSIONS`，保证首次即与既有 caliber 一致）。
    """
    fp = DECLARED_VERSION_FILE
    data: dict = {}
    if fp.exists():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    cur = data.get(kind) if isinstance(data, dict) else None
    if cur:
        return str(cur)
    if not auto_declare:
        return ""
    ver = (file_data_version(source_path, kind) if source_path is not None
           else DEFAULT_DECLARED_VERSIONS.get(kind, f"{kind}_undeclared"))
    if isinstance(data, dict):
        data[kind] = ver
        data.setdefault("_note", "声明式数据版本：只由 data_version.py --refresh 更新；"
                                 "不随日常 cache 更新变化（见 windows.declared_data_version）")
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp = fp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(fp)
        except Exception:
            pass
    return ver


def refresh_declared_version(kind: str, source_path: str | Path,
                             note: str = "") -> dict:
    """**显式**刷新某类数据的声明式版本（内容哈希）。返回落盘后的完整声明表。

    仅在数据发生实质变化时调用（批量复权、数据源更换、修数据）。日常更新不要调用。
    """
    ver = file_data_version(source_path, kind)
    fp = DECLARED_VERSION_FILE
    data: dict = {}
    if fp.exists():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    old = data.get(kind)
    data[kind] = ver
    data[f"{kind}_refreshed_at"] = datetime.now().isoformat(timespec="seconds")
    if note:
        data[f"{kind}_note"] = note
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(fp)
    return {"kind": kind, "old": old, "new": ver, "changed": old != ver,
            "source": str(source_path), "file": str(fp)}


def file_data_version(path: str | Path, kind: str,
                      cache_path: str | Path | None = None) -> str:
    """数据文件的内容版本标识：``f"{kind}_{md5(内容)[:8]}"``。

    **为什么指纹必须带数据版本**（2026-10-11）：候选身份此前只含 `(公式, data_cut)`。
    数据被管线重生成后（行情修正、复权调整、状态标记补全等），同一 `(公式, data_cut)`
    会被误判为「已测过」而复用旧指纹——但**数值可能已经变了**。
    故版本必须进入身份，与 `data_cut` 一起构成「在哪个数据版本上、用哪段窗」的完整口径。

    **实现**：md5 内容前 8 位；按 `(size, mtime_ns)` 缓存到
    `results/merge_v2/data_version_cache.json`，命中则免重算（7.3 GB 文件哈希约十几秒，
    但只在文件未变时命中缓存）。文件缺失时返回 `f"{kind}_absent"`。

    兼容性：`factors.parquet` 当前 md5 前缀为 `abe94a59`，与既有硬编码字面量
    `"factors_abe94a59"` **完全一致** ⇒ 替换为自动计算**不改变任何既有 caliber**。
    """
    p = Path(path)
    if not p.is_file():
        return f"{kind}_absent"
    st = p.stat()
    key = f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    cp = Path(cache_path) if cache_path else DATA_VERSION_CACHE
    cache: dict = {}
    if cp.exists():
        try:
            cache = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    hit = cache.get(p.resolve().as_posix()) if isinstance(cache, dict) else None
    if isinstance(hit, dict) and hit.get("key") == key and hit.get("version"):
        return str(hit["version"])
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    ver = f"{kind}_{h.hexdigest()[:8]}"
    if not isinstance(cache, dict):
        cache = {}
    cache[p.resolve().as_posix()] = {"key": key, "version": ver,
                                     "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(cp)
    except Exception:
        pass
    return ver


def cut_bound(default_end: str, data_cut: str | None = DEFAULT_DATA_CUT) -> str:
    """把某个"遗留硬编码窗上界"裁剪到数据截止日：返回 min(default_end, data_cut)。

    **为什么需要**：代码里存在若干硬编码的窗上界（如 `pool_legacy.ALIGNED_END='2026-07-10'`、
    `dedup.monthly_spearman(end='2026-07-10')`），它们是历史遗留、**含 2026 数据**。
    仅给指纹打标签不足以隔离——这些路径必须同样被裁到截止日，否则"边界"只存在于
    一份字段里而不存在于实际读取中（Codex 2026-10-11 提醒：日期守门 ≠ 数据隔离证明）。

    - `data_cut` 省略 → `DEFAULT_DATA_CUT`（安全默认）
    - `data_cut = CUT_OFF`（'off'）→ 不裁剪，返回 `default_end`（显式逃生门，历史重放用）
    """
    if isinstance(data_cut, str) and data_cut.strip().lower() == "off":
        return default_end
    cut = normalize_data_cut(data_cut)
    return min(str(default_end), cut)

#: factors.parquet 的首个信号日（数据下界，非人为设定）。
FACTORS_FIRST_SIGNAL_DATE = "2021-06-01"
#: labels_prod.parquet 的首个信号日。
LABELS_FIRST_SIGNAL_DATE = "2020-01-02"

#: 历史遗留窗口（label_max=2024-12-26）。保留以便审计"旧结果属于哪段数据"。
#:
#: 【2026-10-11 审计更正】原注释称此值"由 boundary_log.jsonl 153 条读取记录反查"得到
#: "已完成搜索实际使用的历史窗口"。实测该日志中 data_max=2024-12-26 的记录仅 10 条，
#: 且 label 为 selftest-neg / audit-dir / h3calib / dbg2~dbg5 / calib2 / m4diag
#: ——**全部来自门控与诊断线，无一条来自个股进化搜索**；该日志也不含搜索侧记录
#: （搜索线不 import boundary，见 data_contact_audit_2026-10-11.md §1.4）。
#: 因此本常量**不代表个股搜索的窗口**，引用时不得用它推断"搜索未接触 2025 年"。
#: 个股线的实际窗口由 fitness.py 的 START/END/HOLDOUT_START 决定（审计报告 §1.1）。
LEGACY_SEARCH_START = "2021-06-01"
LEGACY_SEARCH_END = "2024-12-26"

# ---------------------------------------------------------------- V1–V8 版本窗口
#: §3.4 表：截止日 D_k、外层评估窗（信号日）、严格边界 S_k。
#: 这些是框架 E2 的设计窗口，**不随数据更新而改动**（C17 版本化）。
VERSION_WINDOWS: tuple[dict, ...] = (
    {"version": "V1", "cutoff": "2024-06-21", "outer_start": "2024-07-01",
     "outer_end": "2024-09-27", "strict_S": "2022-12-16"},
    {"version": "V2", "cutoff": "2024-09-20", "outer_start": "2024-09-30",
     "outer_end": "2025-01-02", "strict_S": "2022-12-16"},
    {"version": "V3", "cutoff": "2024-12-25", "outer_start": "2025-01-03",
     "outer_end": "2025-04-10", "strict_S": "2023-03-23"},
    {"version": "V4", "cutoff": "2025-04-02", "outer_start": "2025-04-11",
     "outer_end": "2025-07-14", "strict_S": "2023-06-28"},
    {"version": "V5", "cutoff": "2025-07-07", "outer_start": "2025-07-15",
     "outer_end": "2025-10-17", "strict_S": "2023-09-25"},
    {"version": "V6", "cutoff": "2025-10-10", "outer_start": "2025-10-20",
     "outer_end": "2026-01-16", "strict_S": "2023-12-29"},
    {"version": "V7", "cutoff": "2026-01-09", "outer_start": "2026-01-19",
     "outer_end": "2026-04-24", "strict_S": "2024-04-09"},
    {"version": "V8", "cutoff": "2026-04-17", "outer_start": "2026-04-27",
     "outer_end": "2026-07-28", "strict_S": "2024-07-11"},
    {"version": "P", "cutoff": SIGNAL_MAX_DATE, "outer_start": None,
     "outer_end": None, "strict_S": None},
)
#: 截断窗 T（§3.4）：数据进入生产训练，但不属于任何已评估外层窗（C25）。
TRUNCATION_WINDOW = ("2026-07-29", "2026-09-11")


@dataclass(frozen=True)
class Window:
    """一次搜索/评估所用的数据窗口。start/end 均为**信号日**。"""

    start: str
    end: str
    label_available_max: str
    evidence: str = "E1"
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def overlaps_forward(self) -> bool:
        """是否触及 E3 前向区（触及即失去确认性资格）。"""
        return self.end >= FORWARD_FIRST_SIGNAL_DATE

    def __str__(self) -> str:
        return f"{self.start}~{self.end} (labels≤{self.label_available_max})"


def prod_window(factors_first: str = FACTORS_FIRST_SIGNAL_DATE) -> Window:
    """**选择窗**（因子筛选 / 建池 / CV / 超参 / 训练）：≤ 2025-12-31。"""
    return Window(start=factors_first, end=FACTOR_SCREEN_MAX,
                  label_available_max=LABEL_MAX_DATE, evidence="E1",
                  note="选择窗：一切『做决定』的步骤都只用这段；"
                       "证据等级恒为 E1（历史数据已被多次使用，C8）")


def decay_monitor_window() -> Window:
    """**衰减监控窗**：2026-01-05 ~ 2026-09-11（169 个交易日）。

    用于部署前检查因子相对历史的衰减，**只作提示项，不作门禁**。
    重要：该段已被 stage15 / 部署诊断看过（data_ledger 标注「不得作确认集」），
    且本窗一旦回流到任何选择环节即失去监控意义。evidence 标 E1-diagnostic。
    """
    return Window(start=DECAY_MONITOR_START, end=SIGNAL_MAX_DATE,
                  label_available_max=LABEL_MAX_DATE, evidence="E1-diagnostic",
                  note="衰减监控：只报告、不进门禁；不得回流到筛选/建池/CV/超参")


def legacy_search_window() -> Window:
    """已完成搜索所用的历史窗口（boundary_log 反查：label_max=2024-12-26）。"""
    return Window(start=LEGACY_SEARCH_START, end=LEGACY_SEARCH_END,
                  label_available_max=LEGACY_SEARCH_END, evidence="E1",
                  note="P0/P1 已完成搜索的实际窗口；比新的选择窗少 246 个交易日，"
                       "故其统计量与新窗口不同口径，需重跑才可比")


def version_window(version: str) -> Window:
    """框架版本 V1–V8/P 的截止日窗口（C17：冻结后不得改动）。"""
    for v in VERSION_WINDOWS:
        if v["version"] == version:
            return Window(start=FACTORS_FIRST_SIGNAL_DATE, end=v["cutoff"],
                          label_available_max=LABEL_MAX_DATE, evidence="E2",
                          note=f"外层窗 {v['outer_start']}~{v['outer_end']}"
                               if v["outer_start"] else "生产版，无外层窗")
    raise KeyError(f"未知版本 {version!r}；可选 {[v['version'] for v in VERSION_WINDOWS]}")


def refresh_prod_window(new_signal_max: str, new_label_max: str) -> Window:
    """数据更新后构造新的生产窗口（D13 向前滚动）。

    注意：这只影响**新**搜索；已冻结的 V1–V8 与已打标的旧结果窗口不变。
    若 new_signal_max ≥ FORWARD_FIRST_SIGNAL_DATE，该窗口已进入 E3 前向区，
    任何用它做的筛选都只能得到 E3 之外层确认，不得作为 E3。
    """
    w = Window(start=FACTORS_FIRST_SIGNAL_DATE, end=new_signal_max,
               label_available_max=new_label_max, evidence="E1",
               note="数据更新后的新生产窗口（D13 向前滚动）")
    return w


def check_window(w: Window) -> list[str]:
    """返回违规项列表（空 = 合规）。供 §10.4/§10.5 边界测试调用。

    额外检查：**选择窗不得触及监控段**——若选择窗 end ≥ 衰减监控起点，说明
    2026 数据已回流到选择环节，监控段失去意义。
    """
    bad: list[str] = []
    if w.start > w.end:
        bad.append(f"start {w.start} > end {w.end}")
    if w.label_available_max < w.end:
        bad.append(f"label_available_max {w.label_available_max} < end {w.end}："
                   f"末尾信号的标签不完整")
    if w.start < FACTORS_FIRST_SIGNAL_DATE:
        bad.append(f"start {w.start} 早于 factors.parquet 起始 "
                   f"{FACTORS_FIRST_SIGNAL_DATE}")
    if w.evidence.startswith("E1") and w.end >= DECAY_MONITOR_START:
        bad.append(f"end {w.end} 已触及衰减监控段（≥{DECAY_MONITOR_START}）："
                   f"选择窗（evidence={w.evidence}）不得进入该区间")
    return bad


def dump() -> dict:
    """写入 contract/registry 用的快照。"""
    return {
        "signal_max_date": SIGNAL_MAX_DATE,
        "label_available_max_date": LABEL_MAX_DATE,
        "forward_first_signal_date": FORWARD_FIRST_SIGNAL_DATE,
        "factors_first_signal_date": FACTORS_FIRST_SIGNAL_DATE,
        "factor_screen_max": FACTOR_SCREEN_MAX,
        "cv_max": CV_MAX,
        "train_signal_max": TRAIN_SIGNAL_MAX,
        "decay_monitor_start": DECAY_MONITOR_START,
        "default_data_cut": DEFAULT_DATA_CUT,
        "legacy_data_cut": LEGACY_DATA_CUT,
        "consumed_by_stock_search": list(CONSUMED_BY_STOCK_SEARCH),
        "untouched_2026_remnant": list(UNTOUCHED_2026_REMNANT),
        "selection_window": prod_window().as_dict(),
        "decay_monitor_window": decay_monitor_window().as_dict(),
        "legacy_search_window": legacy_search_window().as_dict(),
        "version_windows": [dict(v) for v in VERSION_WINDOWS],
        "truncation_window": list(TRUNCATION_WINDOW),
    }


if __name__ == "__main__":
    print(json.dumps(dump(), ensure_ascii=False, indent=2))
    for w in (prod_window(), decay_monitor_window(), legacy_search_window(),
              version_window("V8")):
        print(f"{w}\n    evidence={w.evidence}  违规={check_window(w) or '无'}")
