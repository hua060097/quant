"""LLM 提案通道（OpenAI 兼容 API，可选）：交易想法 → 平台公式候选（知识库 few-shot）。

对应 Alpha-GPT 的 Knowledge Compiler + LLM：注入字段/算子目录与知识库示例，约束输出为
可校验的公式 JSON；产出后由 intake.check_candidate 完成 parse/validate/冒烟/去重。

无 OPENAI_API_KEY/OPENAI_BASE_URL 时 available()==False：走 inbox 模式（手写提案 JSONL
交给 intake.py），默认不发送任何网络请求。凭证只从环境变量读取，不写入任何文件。
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

import _paths  # noqa: F401
from field_catalog import (
    ARITY, DOCUMENTED_OPS, LOCAL_ONLY_FIELDS, LOCAL_ONLY_OPS,
    PLATFORM_FIELDS, VERIFIED_OPS, WINDOWS,
)
from knowledge import retrieve

DEFAULT_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')


def available() -> bool:
    return bool(os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_BASE_URL'))


def _endpoint() -> str:
    base = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    return base.rstrip('/') + '/chat/completions'


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.8,
         timeout: int = 90) -> str:
    api_key = os.environ.get('OPENAI_API_KEY', 'EMPTY')
    payload = json.dumps({'model': model or DEFAULT_MODEL, 'messages': messages,
                          'temperature': temperature}).encode('utf-8')
    req = urllib.request.Request(
        _endpoint(), data=payload,
        headers={'Content-Type': 'application/json',
                 'Authorization': f'Bearer {api_key}'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return data['choices'][0]['message']['content']


def _system_prompt(profile: str = 'local-vwap') -> str:
    if profile == 'competition-c2c':
        target = ('当前是 C2C 竞赛适配 profile。只生成 C2C 公式模式可表达的候选；'
                  '平台兼容性只约束这个竞赛提案，不代表 merge 本地搜索的全部能力。')
    else:
        target = ('当前是本地 VWAP 搜索 profile。可使用给定的本地市场/行业上下文字段和本地算子；'
                  '不可上传的平台表达式仍然是有效的本地候选。')
    return (
        '你是一名量化研究员，负责把交易想法翻译成股票日频因子公式。'
        + target + ' 输出必须是合法公式，且只能使用本 profile 给定字段与算子；'
        '窗口只能取给定集合；不要使用未来函数；滚动均值必须用 TS_MEAN/MA，不得用 MEAN(X,N)。')


def _user_prompt(hypothesis: str, k: int, examples: list[dict],
                 profile: str = 'local-vwap') -> str:
    example_text = '\n'.join(
        f"- [{e.get('status')}] {e.get('mechanism')} | formula: {e.get('formula') or 'N/A'}"
        for e in examples[:5]) or '（无）'
    if profile == 'competition-c2c':
        fields = list(PLATFORM_FIELDS)
        ops = sorted(VERIFIED_OPS)
        backend_note = '这些候选将进入 C2C 竞赛兼容性适配器。'
    else:
        fields = list(PLATFORM_FIELDS) + list(LOCAL_ONLY_FIELDS)
        ops = sorted((set(VERIFIED_OPS) | set(DOCUMENTED_OPS) | set(LOCAL_ONLY_OPS))
                     & set(ARITY))
        backend_note = ('本地字段/算子可本地求值与评估；其 competition_uploadable 状态由后端校验，'
                        '不要为迎合上传能力而删掉行业/市场上下文。')
    return f"""交易想法：{hypothesis}
profile: {profile}
{backend_note}

可用字段（公式模式，日频）：
{', '.join(fields)}
虚拟序列：RET1=CLOSE/DELAY(CLOSE,1)-1；LOGC=LOG(CLOSE)

可用算子（按 profile 能力表）：
{', '.join(ops)}
窗口集合：{', '.join(str(w) for w in WINDOWS)}；可用形式：OP(X,N)，如 TS_MEAN(VOLUME,20)
语法：算术 + - * /、比较 > >= < <=、逻辑 && ||、IF(cond,a,b)、CORR(A,B,N)、
复合表达式如 RANK(0-(CLOSE/TS_MAX(HIGH,250)-1))+RANK(TS_SKEW(RET1,120))

知识库参考（成功/失败教训）：
{example_text}

请给出 {k} 个候选（彼此机制不同，优先冷门机制，避免与知识库已失败的机制重复）。
只输出 JSON 数组，每个元素：
{{"profile": "{profile}", "name": "短名", "formula": "公式", "direction": 0或1, "mechanism": "一句话机制", "tags": ["tag"]}}
direction 含义：1=做多因子值高的一端，0=做多低的一端。"""
def _extract_json_array(text: str) -> list[dict]:
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    items = []
    for chunk in re.findall(r'\{[^{}]*\}', text):
        try:
            item = json.loads(chunk)
            if isinstance(item, dict) and 'formula' in item:
                items.append(item)
        except json.JSONDecodeError:
            continue
    return items


def propose(hypothesis: str, k: int = 8, examples: list[dict] | None = None,
            model: str | None = None, temperature: float = 0.8,
            profile: str = 'local-vwap') -> list[dict]:
    if profile not in ('local-vwap', 'competition-c2c'):
        raise ValueError(f'unknown proposal profile {profile!r}')
    if not available():
        return []
    if examples is None:
        examples = retrieve(hypothesis, k=5, profile=profile)
    messages = [{'role': 'system', 'content': _system_prompt(profile)},
                {'role': 'user', 'content': _user_prompt(hypothesis, k, examples, profile)}]
    text = chat(messages, model=model, temperature=temperature)
    proposals = _extract_json_array(text)
    for item in proposals:
        item.setdefault('source', 'llm')
        item['profile'] = profile
    return proposals[:k]


def write_inbox_template(path: str, hypothesis: str, k: int = 8,
                         profile: str = 'local-vwap') -> str:
    """无 LLM 通道时的人工/agent 提案模板。"""
    if profile not in ('local-vwap', 'competition-c2c'):
        raise ValueError(f'unknown proposal profile {profile!r}')
    lines = [json.dumps({'hypothesis': hypothesis, 'formula': '<FORMULA>',
                         'direction': None, 'tags': [], 'source': 'inbox',
                         'profile': profile}, ensure_ascii=False)]
    lines += [f'# 再补 {k - 1} 行同类 JSON；公式语法见 parser.py；示例：' +
              'RANK(0-TS_SKEW(CLOSE/DELAY(CLOSE,1)-1,120))'
              for _ in range(1)]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def main() -> None:
    import argparse
    import sys

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='LLM 提案通道（无key时只打印说明）')
    ap.add_argument('hypothesis')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--profile', choices=('local-vwap', 'competition-c2c'),
                    default='local-vwap')
    ap.add_argument('--inbox', default='')
    args = ap.parse_args()
    if not available():
        path = args.inbox or 'proposals_inbox.jsonl'
        write_inbox_template(path, args.hypothesis, args.k, profile=args.profile)
        print('未检测到 OPENAI_API_KEY/OPENAI_BASE_URL：已生成 inbox 模板 ' + path)
        print('填写后运行：python pandaai_online/merge/intake.py ' + path)
        return
    items = propose(args.hypothesis, args.k, profile=args.profile)
    print(json.dumps(items, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
