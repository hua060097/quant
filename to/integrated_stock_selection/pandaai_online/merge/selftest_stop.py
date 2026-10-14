# -*- coding: utf-8 -*-
"""停止规则验收（§17.7）：与进化搜索同构的硬预算 + 可选早停 + 方向级暂停。"""
from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_stop as GS  # noqa: E402

ok_all = True


def chk(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f'  [{"PASS" if cond else "FAIL"}] {name:<48s} {detail}')


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 78)
    print("停止规则验收")
    print("=" * 78)

    # T1 stall 早停（对应 --stall）
    st = GS.StopState(cfg=GS.StopConfig(stall=3))
    hits = [st.note(s) for s in (0.5, 0.4, 0.3, 0.2)]      # 首个设 best，后 3 个无改进
    chk("T1 stall=3：连续 3 个无改进触发早停", hits == [False, False, False, True],
        f"hits={hits} stall={st.stall_count}")
    st2 = GS.StopState(cfg=GS.StopConfig(stall=3))
    hits2 = [st2.note(s) for s in (0.5, 0.4, 0.6, 0.2)]    # 中间有改进 → 重置
    chk("T1 stall 被改进重置", hits2 == [False, False, False, False],
        f"hits={hits2} stall={st2.stall_count}")

    # T2 stall=0 关闭
    st3 = GS.StopState(cfg=GS.StopConfig(stall=0))
    chk("T2 stall=0 永不早停", not any(st3.note(0.1) for _ in range(50)))

    # T3 硬预算（对应 --n / --full-cap）
    st4 = GS.StopState(cfg=GS.StopConfig(n_seeds=40, full_cap=12))
    st4.record_batch(40, 5)
    left = st4.budget_left()
    chk("T3 预算扣减", left == {"n_seeds_left": 0, "full_cap_left": 7}, str(left))

    # T4 方向级暂停：连续两批 <10%
    st5 = GS.StopState(cfg=GS.StopConfig())
    p1, _ = st5.record_batch(40, 2)     # 5%
    p2, why = st5.record_batch(40, 1)   # 2.5%
    chk("T4 连续两批 <10% → 暂停", p1 is False and p2 is True, why)

    # T5 只有一批不够
    st6 = GS.StopState(cfg=GS.StopConfig())
    p1, _ = st6.record_batch(40, 1)
    chk("T5 仅一批不触发", p1 is False)

    # T6 批次过小不按比例判定
    st7 = GS.StopState(cfg=GS.StopConfig())
    st7.record_batch(5, 0)
    p, why = st7.record_batch(5, 0)
    chk("T6 批次 <20 不按比例判定", p is False and "过小" in why, why)

    # T7 一批正常 + 一批低 → 不触发
    st8 = GS.StopState(cfg=GS.StopConfig())
    st8.record_batch(40, 8)     # 20%
    p, why = st8.record_batch(40, 1)
    chk("T7 非连续低率不触发", p is False, why)

    # T8 持久化
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.json")
        st9 = GS.StopState(cfg=GS.StopConfig(stall=2))
        st9.record_batch(30, 4)
        st9.note(0.5)
        st9.save(p)
        st10 = GS.StopState.load(p)
        chk("T8 持久化往返", st10.batches == st9.batches
            and st10.cfg.stall == 2 and st10.best == st9.best,
            f"batches={st10.batches}")
        chk("T8 缺文件时安全回落", isinstance(GS.StopState.load(p + ".nope"), GS.StopState))

    print(f'\nSUMMARY {"PASS" if ok_all else "FAIL"}')
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
