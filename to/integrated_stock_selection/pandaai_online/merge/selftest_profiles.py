"""本地 VWAP / C2C profile 与竞赛导出适配器的合成回归测试。

不读取行情/因子数据、不调用平台；验证 local-only 与 documented 候选不会在
本地目标筛选阶段被 C2C 能力边界误删，而竞赛导出只保留平台已验证表达式。
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import _paths  # noqa: F401
from executor import prepare_competition_rows
from grammar import expression_hash
from knowledge import append as kb_append, retrieve as kb_retrieve
from ledger import Ledger
from parser import parse_formula
from llm_client import _user_prompt
from run_search import label_search_metric, label_search_value, resolve_search_profile, valid_rows_for_label


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    profile, primary, labels = resolve_search_profile(SimpleNamespace(label=None))
    checks.append(("default profile is local VWAP",
                   profile == "local-vwap" and primary == "vwap" and labels == ["vwap"],
                   f"profile={profile} primary={primary} labels={labels}"))

    profile, primary, labels = resolve_search_profile(
        SimpleNamespace(profile="local-vwap", label=["c2c"]))
    checks.append(("C2C is an additional measurement in local profile",
                   profile == "local-vwap" and primary == "vwap" and labels == ["vwap", "c2c"],
                   f"profile={profile} primary={primary} labels={labels}"))

    profile, primary, labels = resolve_search_profile(
        SimpleNamespace(profile=None, label=["c2c", "vwap"]))
    checks.append(("legacy explicit label ordering remains compatible",
                   profile == "competition-c2c" and primary == "c2c"
                   and labels == ["c2c", "vwap"],
                   f"profile={profile} primary={primary} labels={labels}"))

    metric = {"sia": 0.02, "rank_ic": -0.10, "rank_ic_ir": 0.50,
              "rank_ic_win": 0.60}
    checks.append(("primary scoring metric follows target profile",
                   label_search_metric("vwap") == "rank_sia"
                   and abs(label_search_value(metric, "vwap") - 0.03) < 1e-12
                   and label_search_metric("c2c") == "sia"
                   and label_search_value(metric, "c2c") == 0.02,
                   f"vwap={label_search_value(metric, 'vwap'):.4f} c2c={label_search_value(metric, 'c2c'):.4f}"))

    local_prompt = _user_prompt("行业强弱", 2, [], profile="local-vwap")
    c2c_prompt = _user_prompt("行业强弱", 2, [], profile="competition-c2c")
    checks.append(("proposal fields follow backend capability, not C2C for local",
                   "IND_RET" in local_prompt and "IND_RET" not in c2c_prompt,
                   f"local_ind={('IND_RET' in local_prompt)} c2c_ind={('IND_RET' in c2c_prompt)}"))

    valid_formula = "TS_MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1),20)"
    local_formula = "TS_MEAN(IND_RET,20)"
    documented_formula = "TS_KURT(CLOSE,20)"
    rows = [
        {"name": "verified", "formula": valid_formula, "direction": 0,
         "label_directions": {"vwap": 0, "c2c": 1}, "local_only": False,
         "competition_uploadable": True,
         "labels": {"vwap": {"valid": True, "sia": 0.04},
                    "c2c": {"valid": True, "sia": 0.03}}},
        {"name": "industry-local", "formula": local_formula, "direction": 0,
         "label_directions": {"vwap": 0, "c2c": 1}, "local_only": True,
         "competition_uploadable": False,
         "labels": {"vwap": {"valid": True, "sia": 0.08},
                    "c2c": {"valid": True, "sia": 0.02}}},
        {"name": "documented-local", "formula": documented_formula, "direction": 1,
         "label_directions": {"vwap": 1, "c2c": 0}, "local_only": False,
         "competition_uploadable": False,
         "labels": {"vwap": {"valid": True, "sia": 0.06},
                    "c2c": {"valid": True, "sia": 0.01}}},
    ]

    vwap_valid = valid_rows_for_label(rows, "vwap")
    c2c_valid = valid_rows_for_label(rows, "c2c")
    checks.append(("local-only candidates stay in local per-label universes",
                   len(vwap_valid) == 3 and len(c2c_valid) == 3
                   and any(r["local_only"] for r in vwap_valid),
                   f"vwap={len(vwap_valid)} c2c={len(c2c_valid)}"))

    uploaded, rejected = prepare_competition_rows(rows, "c2c", "vwap")
    checks.append(("C2C adapter filters only export set and uses C2C direction",
                   [r["name"] for r in uploaded] == ["verified"]
                   and uploaded[0]["direction"] == 1
                   and uploaded[0]["sia"] == 0.03
                   and {r["name"] for r in rejected} == {"industry-local", "documented-local"},
                   f"uploaded={[r['name'] for r in uploaded]} rejected="
                   f"{[(r['name'], r['reason']) for r in rejected]}"))

    with tempfile.TemporaryDirectory(prefix="merge_profiles_") as temp:
        ledger = Ledger(Path(temp) / "ledger.jsonl")
        node = parse_formula("TS_MEAN(CLOSE,20)/CLOSE")
        for label, score in (("c2c", 0.01), ("vwap", 0.03)):
            ledger.log("local", node, 1, family="M1_return", label=label,
                       data_cut="2025-12-31",
                       data={"tier1": {"sia": score, "turnover": 0.1}})
        c2c_pool = ledger.evidence_pool_for("2025-12-31", label="c2c")
        vwap_pool = ledger.evidence_pool_for("2025-12-31", label="vwap")
        checks.append(("UCB/elite evidence pools stay label-isolated",
                       len(c2c_pool) == len(vwap_pool) == 1
                       and c2c_pool[0]["sia"] == 0.01
                       and vwap_pool[0]["sia"] == 0.03
                       and ledger.count_tested("2025-12-31", label="c2c") == 1
                       and ledger.count_tested("2025-12-31", label="vwap") == 1,
                       f"c2c={len(c2c_pool)} vwap={len(vwap_pool)}"))

        local_proposal = parse_formula("TS_MEAN(CLOSE,10)/CLOSE")
        c2c_proposal = parse_formula("CLOSE/TS_MEAN(CLOSE,10)")
        ledger.log("proposed", local_proposal, 1, family="M1_return", label="vwap",
                   data_cut="2025-12-31", data={"profile": "local-vwap"})
        ledger.log("proposed", c2c_proposal, 1, family="M1_return", label="c2c",
                   data_cut="2025-12-31", data={"profile": "competition-c2c"})
        vwap_exclude = ledger.hashes_for(["vwap"], data_cut="2025-12-31")
        c2c_exclude = ledger.hashes_for(["c2c"], data_cut="2025-12-31")
        local_hash = expression_hash(local_proposal)
        c2c_hash = expression_hash(c2c_proposal)
        checks.append(("proposal duplicate suppression is label-scoped",
                       local_hash != c2c_hash
                       and local_hash in vwap_exclude and local_hash not in c2c_exclude
                       and c2c_hash in c2c_exclude and c2c_hash not in vwap_exclude
                       and ledger.count_tested("2025-12-31", label="vwap") == 1
                       and ledger.count_tested("2025-12-31", label="c2c") == 1
                       and len(vwap_exclude) == 2 and len(c2c_exclude) == 2,
                       f"vwap={len(vwap_exclude)} c2c={len(c2c_exclude)}"))

        kb_path = Path(temp) / "knowledge.jsonl"
        kb_append({"id": "c2c-only-test", "profile": "competition-c2c",
                   "status": "failed", "mechanism": "profile_scope_marker_c2c",
                   "tags": ["profile-scope-marker"]}, kb_path)
        kb_append({"id": "vwap-only-test", "profile": "local-vwap",
                   "status": "rule", "mechanism": "profile_scope_marker_vwap",
                   "tags": ["profile-scope-marker"]}, kb_path)
        local_hits = kb_retrieve("profile_scope_marker", k=10, path=kb_path,
                                 profile="local-vwap")
        c2c_hits = kb_retrieve("profile_scope_marker", k=10, path=kb_path,
                               profile="competition-c2c")
        checks.append(("auto-learned proposal knowledge is profile-scoped",
                       any(e.get("id") == "vwap-only-test" for e in local_hits)
                       and not any(e.get("id") == "c2c-only-test" for e in local_hits)
                       and any(e.get("id") == "c2c-only-test" for e in c2c_hits),
                       f"local={len(local_hits)} c2c={len(c2c_hits)}"))

    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    failed = sum(not ok for _, ok, _ in checks)
    print(f"profile/export contract: {len(checks) - failed}/{len(checks)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
