"""JSONL 事件台账（schema-v3）：本地/保留/平台指标 + 决策，全局假设计数器。

事件行（append-only）：
  {"schema_version":3,"event":"proposal|local|holdout|platform|decision",
   "candidate_id","expression_hash","formula","direction","family","hypothesis",
   "created_at","struct_fp","data":{...}}
latest() 按 candidate_id 合并事件；count_tested() = 唯一 (candidate_id,label) 测试对（多重检验分母）。

Sidecar 摘要（2026-09-24 新增）：ledger.jsonl 17MB 后每次 search 全量扫 6 次 ≈3.6s 且线性变慢，
与"历史越大越有用"方向相反。改为 ledger_summary.json（与台账同目录）：
  - 增量消费：记录 byte_offset，新事件只解析尾部；offset 不一致/损坏/缺失时全量重建
  - 内容：hash→labels 排除集、struct_fp→hashes、tested_pairs、事件计数、
    events_min（fast/local 的 created_at/family/label/event/sia，供 UCB 配额）、
    evidence_pool（c2c 证据公式池，供腿库/精英回流，首见去重保持旧语义）
  - log()/log_raw() 不触碰摘要（append-only）；下次 summary() 自动吃进尾部增量

**数据截止日 scoping（2026-10-11 用户指令，SUMMARY_SCHEMA 2）**：
候选身份 = 公式 × 数据截止日。每条事件带 `data_cut`；摘要按 cut 分别登记
（`hashes[h]['cuts']`、`tested_pairs` 三元组）。`hashes_for(labels, data_cut=...)`
默认只返回**该截止日**下的已测公式，因此历史遗留截止日（`windows.LEGACY_DATA_CUT`，
即个股线用过 2026 数据的期间）的记录**不参与新搜索的排除集**——它们被标记而非混用。
缺 `data_cut` 字段的旧记录按遗留截止日处理（`_record_cut`），无需改写历史台账；
`migrate_data_cut.py` 可把该标记**显式落盘**。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import _paths  # noqa: F401
import windows
from grammar import Node, candidate_id as make_candidate_id, expression_hash, render

DEFAULT_PATH = Path(__file__).resolve().parent / 'ledger.jsonl'
SCHEMA_VERSION = 3
SUMMARY_SCHEMA = 3
LEG_MAX_TURNOVER = 0.20
#: 记录/查询使用的默认截止日（选择窗上界）。可用环境或参数覆盖。
DEFAULT_DATA_CUT = windows.DEFAULT_DATA_CUT


def _record_cut(rec: dict) -> str:
    """记录的数据截止日；缺字段的旧记录 → 遗留截止日（标记而非改写）。"""
    return str(rec.get('data_cut') or windows.LEGACY_DATA_CUT)


def _record_version(rec: dict) -> str:
    """记录的数据版本；缺字段的旧记录 → 空串（=「无版本」，不匹配任何具名版本）。

    见 `windows.file_data_version`：数据被重生成后，同一 (公式, data_cut) 的**数值可能已变**，
    故版本必须进入候选身份。旧记录没有版本 ⇒ 在带版本的查询下**不构成排除依据**
    （与 `data_cut` 对遗留记录的处理同构：宁可重测，不可误判为「已测过」）。
    """
    return str(rec.get('data_version') or '')


class Ledger:
    def __init__(self, path: Path | str = DEFAULT_PATH,
                 data_cut: str | None = DEFAULT_DATA_CUT,
                 data_version: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.path.with_name(self.path.stem + '_summary.json')
        self._summary: dict | None = None
        #: 写入时默认使用的数据截止日（log/log_raw 未显式传入时生效）。
        self.data_cut = windows.normalize_data_cut(data_cut)
        #: 写入时默认使用的数据版本（见 `windows.file_data_version`）。
        #: None = 不标记版本（旧行为）；生产搜索应设为本面板的内容版本。
        self.data_version = data_version

    # ------------------------------------------------------------------ 基础
    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def log(self, event: str, node: Node, direction: int, family: str = '',
            hypothesis: str = '', data: dict | None = None,
            label: str | None = None, struct_fp: str | None = None,
            data_cut: str | None = None,
            data_version: str | None = None) -> dict:
        record = {
            'schema_version': SCHEMA_VERSION,
            'event': event,
            'candidate_id': make_candidate_id(node, direction),
            'expression_hash': expression_hash(node),
            'formula': render(node),
            'direction': int(direction),
            'family': family,
            'hypothesis': hypothesis,
            'label': label,
            'data_cut': windows.normalize_data_cut(data_cut or self.data_cut),
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'data': data or {},
        }
        ver = data_version if data_version is not None else self.data_version
        if ver:
            record['data_version'] = str(ver)
        if struct_fp:
            record['struct_fp'] = struct_fp
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return record

    def log_raw(self, event: str, formula: str, direction: int | None = None,
                family: str = '', hypothesis: str = '', data: dict | None = None,
                label: str | None = None, struct_fp: str | None = None,
                data_cut: str | None = None,
                data_version: str | None = None) -> dict:
        """无法解析为 AST 的原始公式事件（parse 失败/重复提案）。"""
        raw = ''.join(str(formula).lower().split())
        expr_hash = hashlib.sha256(raw.encode()).hexdigest()
        cid = (hashlib.sha256(f'{raw}|{int(direction)}'.encode()).hexdigest()[:12]
               if direction is not None else expr_hash[:12])
        record = {
            'schema_version': SCHEMA_VERSION,
            'event': event,
            'candidate_id': cid,
            'expression_hash': expr_hash,
            'formula': str(formula),
            'direction': direction,
            'family': family,
            'hypothesis': hypothesis,
            'label': label,
            'data_cut': windows.normalize_data_cut(data_cut or self.data_cut),
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'data': data or {},
        }
        ver = data_version if data_version is not None else self.data_version
        if ver:
            record['data_version'] = str(ver)
        if struct_fp:
            record['struct_fp'] = struct_fp
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return record

    # ------------------------------------------------------------- sidecar
    def _empty_summary(self) -> dict:
        return {
            'schema': SUMMARY_SCHEMA,
            'byte_offset': 0,
            'n_events': 0,
            'events': {},
            'hashes': {},
            'fps': {},
            'vfps': [],
            'cuts': {},
            'versions': {},
            'tested_pairs': [],
            'tested_by_label': {},
            'candidate_ids': [],
            'events_min': [],
            'evidence_pool': [],
        }

    def _ingest(self, s: dict, recs: list[dict]) -> None:
        hashes = s['hashes']
        fps = s['fps']
        pairs = set(s['tested_pairs'])
        cids = set(s['candidate_ids'])
        cuts = s['cuts']
        versions = s['versions']
        for rec in recs:
            ev = rec.get('event')
            cut = _record_cut(rec)
            ver = _record_version(rec)
            s['events'][ev] = s['events'].get(ev, 0) + 1
            s['n_events'] += 1
            if cut not in cuts:
                cuts[cut] = {'events': 0, 'tested_pairs': 0, 'explicit': 0}
            else:
                for _sk, _sv in (('events', 0), ('tested_pairs', 0), ('explicit', 0)):
                    cuts[cut].setdefault(_sk, _sv)
            cuts[cut]['events'] += 1
            vkey = ver or '(无版本)'
            if vkey not in versions:
                versions[vkey] = {'events': 0, 'tested_pairs': 0}
            else:
                for _sk, _sv in (('events', 0), ('tested_pairs', 0)):
                    versions[vkey].setdefault(_sk, _sv)
            versions[vkey]['events'] += 1
            h = rec.get('expression_hash')
            if h:
                entry = hashes.setdefault(
                    h, {'labels': [], 'explicit': False, 'cuts': [], 'versions': [],
                        'last_at': ''})
                entry.setdefault('versions', [])
                entry.setdefault('last_at', '')
                _ts = rec.get('created_at') or ''
                if _ts > entry['last_at']:
                    entry['last_at'] = _ts
                if cut not in entry['cuts']:
                    entry['cuts'].append(cut)
                if ver not in entry['versions']:
                    entry['versions'].append(ver)
                if ev in ('fast', 'local'):
                    lab = rec.get('label')
                    if lab and lab not in entry['labels']:
                        entry['labels'].append(lab)
                elif ev in ('proposed', 'rejected', 'duplicate'):
                    entry['explicit'] = True
                    cuts[cut]['explicit'] += 1
            fp = rec.get('struct_fp')
            if fp and h:
                bucket = fps.setdefault(fp, [])
                if h not in bucket:
                    bucket.append(h)
            if ev in ('fast', 'local'):
                cid = rec.get('candidate_id')
                lab = rec.get('label')
                pairs.add(f'{cut}\x00{cid}\x00{lab}')
                cuts[cut]['tested_pairs'] += 1
                versions[vkey]['tested_pairs'] += 1
                key = lab or 'unlabeled'
                s['tested_by_label'][key] = s['tested_by_label'].get(key, 0) + 1
                if cid:
                    cids.add(cid)
                d = rec.get('data') or {}
                vfp = d.get('vfp')
                if vfp:
                    tagged = f'{cut}|{ver}|{vfp}'
                    if tagged not in s['vfps']:
                        s['vfps'].append(tagged)
                sia = (d.get('tier1') or {}).get('sia')
                s['events_min'].append([rec.get('created_at') or '',
                                        rec.get('family') or '', lab, ev,
                                        float(sia) if sia is not None else None, cut, ver])
                # evidence_pool：保留全部 c2c 证据行（不去重），换手过滤/首见去重在
                # 消费端做——与旧 _leg_library 语义逐条一致（过滤先于去重）
                if (lab in (None, 'c2c') and sia is not None and h
                        and rec.get('formula')):
                    t = d.get('tier1') or {}
                    s['evidence_pool'].append({
                        'hash': h, 'formula': rec['formula'],
                        'family': rec.get('family') or '',
                        'direction': int(rec.get('direction', 1)),
                        'sia': float(sia), 'turnover': t.get('turnover'),
                        'cut': cut, 'version': ver,
                    })
        s['tested_pairs'] = sorted(pairs)
        s['candidate_ids'] = sorted(cids)
        s['cuts'] = cuts
        s['versions'] = versions

    def _read_from(self, offset: int) -> tuple[list[dict], int]:
        if not self.path.exists() or self.path.stat().st_size <= offset:
            return [], offset
        with self.path.open('rb') as f:
            f.seek(offset)
            chunk = f.read()
        parts = chunk.split(b'\n')
        recs = []
        for raw in parts[:-1]:
            line = raw.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line.decode('utf-8')))
            except Exception:
                continue
        consumed = sum(len(p) + 1 for p in parts[:-1])
        return recs, offset + consumed

    def _save_summary(self, s: dict) -> None:
        tmp = self.summary_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(s, ensure_ascii=False), encoding='utf-8')
        tmp.replace(self.summary_path)

    def summary(self) -> dict:
        size = self.path.stat().st_size if self.path.exists() else 0
        if self._summary is not None and self._summary['byte_offset'] == size:
            return self._summary
        cached = None
        if self.summary_path.exists():
            try:
                cand = json.loads(self.summary_path.read_text(encoding='utf-8'))
                if (cand.get('schema') == SUMMARY_SCHEMA
                        and 0 <= cand.get('byte_offset', -1) <= size):
                    cached = cand
            except Exception:
                cached = None
        if cached is None:
            s = self._empty_summary()
            recs, offset = self._read_from(0)
            self._ingest(s, recs)
            s['byte_offset'] = offset
        else:
            s = cached
            # 旧格式 sidecar 补齐新增键（向后兼容：新增汇总字段时不强制全量重建）
            for _k, _v0 in self._empty_summary().items():
                s.setdefault(_k, _v0)
            if s['byte_offset'] < size:
                recs, offset = self._read_from(s['byte_offset'])
                self._ingest(s, recs)
                s['byte_offset'] = offset
        self._summary = s
        self._save_summary(s)
        return s

    def fps_covered(self, seen_hash: set[str]) -> set[str]:
        """跨 run 结构指纹覆盖集：fp 组内任一 hash 已在排除集 → 该近克隆家族视为已覆盖。

        与 run 内 seen_fp 语义（同族一票否决）对齐：先到者定族，后到近克隆跳过。
        """
        return {fp for fp, hs in self.summary()['fps'].items()
                if any(h in seen_hash for h in hs)}

    # ------------------------------------------------------------- 消费 API
    def latest(self) -> dict:
        merged: dict = {}
        for rec in self.read():
            key = rec['candidate_id']
            entry = merged.setdefault(key, {'candidate_id': key,
                                            'expression_hash': rec['expression_hash'],
                                            'formula': rec['formula'],
                                            'direction': rec['direction'],
                                            'events': {}})
            entry['formula'] = rec['formula']
            entry['events'][rec['event']] = rec
            if rec.get('family'):
                entry['family'] = rec['family']
        return merged

    def hashes(self, data_cut: str | None = None,
               data_version: str | None = None) -> set:
        """全部历史哈希。`data_cut=None` = 不按截止日过滤（旧行为）。

        传入 `data_cut` 时只返回在该截止日下出现过事件的哈希；
        传入 `data_version` 时再要求版本匹配（见 `windows.file_data_version`）。
        """
        hashes = self.summary()['hashes']
        if data_cut is None and data_version is None:
            return set(hashes)
        cut = windows.normalize_data_cut(data_cut) if data_cut is not None else None
        out = set()
        for k, v in hashes.items():
            if cut is not None and cut not in (v.get('cuts') or []):
                continue
            if data_version is not None and data_version not in (v.get('versions') or []):
                continue
            out.add(k)
        return out

    def hashes_for(self, labels: list[str] | None = None,
                   data_cut: str | None = DEFAULT_DATA_CUT,
                   data_version: str | None = None,
                   retest_before: str | None = None) -> set[str]:
        """生成排除集（公式×标签分离策略 + **数据截止日 scoping**）。

        **指纹的作用键只有 `data_cut`**（2026-10-11 用户裁定）。数据版本**不作作用键**——
        `cache/stock_features_*` 是日频重生成的，若以内容哈希为作用键，则每个交易日全部
        指纹失效，「已测过，跳过」机制失去意义。`data_version` 参数**仅供审计查询**
        （回答「哪些公式是在某个数据世代下测过的」），**生产搜索不得传**。

        **换代时的显式杠杆**：`retest_before`（ISO 日期/时间）——只把**最后事件早于**
        该时刻的公式重新纳入候选（= 强制重测数据换代前测过的那些）。用于批量复权重算、
        数据源更换等**实质变化**场景；日常更新不要用。

        labels=None：该口径下的全部哈希。
        labels=[L1,..]：仅当公式在**所有**请求标签下都有 fast/local 证据才排除；
        显式事件（proposed/rejected/duplicate）不论标签一律排除（避免重复生成提案）。
        这样"只测过 c2c 的公式"在 c2c+vwap 联测时会重新生成并补齐 vwap 证据，
        而同一进程内因子值只计算一次，不浪费算力。
        """
        hashes = self.summary()['hashes']
        if data_cut is not None:
            cut = windows.normalize_data_cut(data_cut)
            hashes = {k: v for k, v in hashes.items()
                      if cut in (v.get('cuts') or [])}
        if data_version is not None:
            hashes = {k: v for k, v in hashes.items()
                      if data_version in (v.get('versions') or [])}
        if retest_before:
            tb = str(retest_before)
            hashes = {k: v for k, v in hashes.items()
                      if (v.get('last_at') or '') >= tb}
        explicit = {k for k, v in hashes.items() if v['explicit']}
        if labels is None:
            return set(hashes)
        per_label = []
        for lab in labels:
            per_label.append({k for k, v in hashes.items() if lab in v['labels']})
        tested_all = set.intersection(*per_label) if per_label else set()
        return tested_all | explicit

    def count_tested(self, data_cut: str | None = None) -> int:
        """多重检验分母：唯一 (数据截止日, candidate_id, label) 测试对（fast/local 事件）。

        截止日不同的测试属于不同检验族（窗口变了，统计量不可比），故默认计入全部；
        传 `data_cut` 则只统计该截止日。
        """
        pairs = self.summary()['tested_pairs']
        if data_cut is None:
            return len(pairs)
        cut = windows.normalize_data_cut(data_cut)
        return sum(1 for p in pairs if p.split('\x00', 1)[0] == cut)

    def tested_by_label(self) -> dict:
        return dict(self.summary()['tested_by_label'])

    def cut_stats(self) -> dict:
        """按数据截止日的事件/测试对数（审计用）。"""
        return {k: dict(v) for k, v in sorted(self.summary()['cuts'].items())}

    def version_stats(self) -> dict:
        """按数据版本的事件/测试对数（审计用）。`(无版本)` = 未标记版本的旧记录。"""
        return {k: dict(v) for k, v in sorted(self.summary()['versions'].items())}

    def evidence_pool_for(self, data_cut: str | None = DEFAULT_DATA_CUT,
                          data_version: str | None = None) -> list[dict]:
        """该口径下的 c2c 证据池（腿库 / 精英回流的原料）。

        2026-10-11：历史 winner 是用**遗留截止日**（含 2026 数据）选出来的，直接复用
        会把旧的选择偏差带进新搜索——故默认只返回当前截止日的证据。截止日变更后
        池子为空是**正确**的（答案：目前没有任何一个因子是在新窗口下被评估过的），
        此时腿库/精英回流为空、各机制轴回到未测状态（乐观初始化），搜索照常进行。
        `data_cut=None` 返回全量（历史重放 / 显式要求复用旧证据时使用）。

        `data_version` 非空时再要求版本匹配（数据重生成后旧证据不再算数）。
        """
        pool = self.summary().get('evidence_pool') or []
        if data_cut is None and data_version is None:
            return list(pool)
        cut = windows.normalize_data_cut(data_cut) if data_cut is not None else None
        out = []
        for e in pool:
            c = e.get('cut') or windows.LEGACY_DATA_CUT
            if cut is not None and c != cut:
                continue
            if data_version is not None and (e.get('version') or '') != data_version:
                continue
            out.append(e)
        return out

    def vfps_for(self, data_cut: str | None = DEFAULT_DATA_CUT,
                 data_version: str | None = None) -> set[str]:
        """该口径下的值指纹集合。

        值指纹由面板分位数派生（`evaluator.value_fingerprint`），**随窗口与数据版本改变**，
        故必须与 (截止日, 版本) 绑定；跨口径比较无意义。
        摘要以 `cut|version|vfp` 存储；`data_cut=None, data_version=None` 返回全部（旧行为）。
        """
        raw = self.summary().get('vfps') or []
        if data_cut is None and data_version is None:
            return {x.split('|', 2)[2] if x.count('|') >= 2
                    else (x.split('|', 1)[1] if '|' in x else x) for x in raw}
        cut = windows.normalize_data_cut(data_cut) if data_cut is not None else None
        out = set()
        for x in raw:
            if x.count('|') >= 2:
                c, v, fp = x.split('|', 2)
                if cut is not None and c != cut:
                    continue
                if data_version is not None and v != data_version:
                    continue
                out.add(fp)
            elif '|' in x:  # 旧格式 cut|vfp（无版本段）
                c, fp = x.split('|', 1)
                if cut is not None and c != cut:
                    continue
                if data_version is not None:
                    continue
                out.add(fp)
            else:  # 更旧：裸 vfp
                if cut is not None and cut != windows.LEGACY_DATA_CUT:
                    continue
                if data_version is not None:
                    continue
                out.add(x)
        return out

    def multiple_testing_threshold(self, alpha: float = 0.05,
                                   data_cut: str | None = None) -> float:
        return alpha / max(self.count_tested(data_cut), 1)

    def stats(self) -> dict:
        s = self.summary()
        return {'events': dict(s['events']), 'n_events': s['n_events'],
                'unique_candidates': len(s['candidate_ids']),
                'tested': self.count_tested(),
                'tested_by_label': self.tested_by_label(),
                'p_threshold': self.multiple_testing_threshold(),
                'ledger_data_cut': self.data_cut,
                'ledger_data_version': self.data_version,
                'module_default_cut': DEFAULT_DATA_CUT,
                'tested_by_cut': {k: self.count_tested(k) for k in self.cut_stats()},
                'cuts': self.cut_stats(),
                'versions': self.version_stats()}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description='台账统计')
    ap.add_argument('--path', default=str(DEFAULT_PATH))
    ap.add_argument('--data-cut', default=None,
                    help='只看该数据截止日（默认全部；auto=选择窗上界）')
    args = ap.parse_args()
    stats = Ledger(args.path).stats()
    if args.data_cut:
        cut = windows.normalize_data_cut(args.data_cut)
        stats['filter_cut'] = cut
        stats['filter_cut_hashes'] = len(Ledger(args.path).hashes(cut))
        stats['filter_cut_tested'] = Ledger(args.path).count_tested(cut)
        stats['filter_cut_p_threshold'] = Ledger(args.path).multiple_testing_threshold(
            0.05, cut)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
