"""本机Ollama配对观察：一次基础请求，再修改一个条件发起追问。

预设历史固定实验条件；模型请求真实执行。工具仅作为可用工具定义传入，
本脚本不执行工具，也不复现官方完整ReAct文件任务或厂商计费。
"""

import json
import random
import statistics
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from openai import OpenAI

MODEL = 'qwen3:0.6b'
REPEATS = 3
MODES = {
    'correct': '稳定前缀，追加追问',
    'dynamic_system': '修改系统开头的时间信息',
    'dynamic_profile': '修改系统中的用户额度',
    'shuffled_tools': '改变工具定义顺序',
    'sliding_window': '系统保留，历史仅留最后5条',
    'text_format': '结构化历史改成一条纯文本消息',
}
BACKGROUND = (
    'An agent reads messages, preserves relevant facts, uses available tools when needed, '
    'and answers using evidence. Stable prefixes can reuse previously computed states.\n'
) * 35
TOOLS = [
    {'type': 'function', 'function': {
        'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']},
    }}
    for name, description in [
        ('read_file', '读取指定路径的文本文件。本轮问题不需要访问文件。'),
        ('list_files', '列出指定目录中的文件。本轮问题不需要访问目录。'),
        ('file_info', '查看指定文件的基本信息。本轮问题不需要访问文件。'),
    ]
]


def build_case(mode, trial):
    marker = uuid4().hex
    expected = f'K{730 + trial}'
    system = (
        f'实验编号：{marker}\n当前时间：2026-10-11 18:00:00\n用户剩余额度：100\n'
        '你是中文助手。根据对话回答，不需要调用工具。'
        '被问归档编号时，只输出原文中的编号；历史没有编号则只输出UNKNOWN。\n'
        + BACKGROUND
    )
    # 预先编写历史，避免各组因模型早期回复不同而改变实验条件。
    history = [
        {'role': 'user', 'content': f'请记住：项目叫海燕，归档编号为{expected}。'},
        {'role': 'assistant', 'content': '已记录项目信息。'},
    ]
    for task in ('整理目录', '检查格式', '准备归档'):
        history.extend([
            {'role': 'user', 'content': f'下一项工作是{task}，请回复收到。'},
            {'role': 'assistant', 'content': '收到'},
        ])
    history.append({'role': 'user', 'content': '请仅回复收到。'})
    baseline = {'messages': [{'role': 'system', 'content': system}] + history, 'tools': deepcopy(TOOLS)}
    return baseline, expected


def build_followup(baseline, baseline_answer, mode):
    request = deepcopy(baseline)
    request['messages'].extend([
        {'role': 'assistant', 'content': baseline_answer},
        {'role': 'user', 'content': '海燕项目的归档编号是什么？仅输出编号；找不到则输出UNKNOWN。'},
    ])
    if mode == 'dynamic_system':
        request['messages'][0]['content'] = request['messages'][0]['content'].replace('18:00:00', '18:00:01', 1)
    elif mode == 'dynamic_profile':
        request['messages'][0]['content'] = request['messages'][0]['content'].replace('剩余额度：100', '剩余额度：099', 1)
    elif mode == 'shuffled_tools':
        # 循环移位确保顺序确实变化，避免随机打乱恰好得到原顺序。
        request['tools'] = request['tools'][1:] + request['tools'][:1]
    elif mode == 'sliding_window':
        request['messages'] = request['messages'][:1] + request['messages'][1:][-5:]
    elif mode == 'text_format':
        history_text = '\n'.join(f"{m['role'].upper()}: {m['content']}" for m in request['messages'][1:])
        request['messages'] = [request['messages'][0], {'role': 'user', 'content': history_text}]
    return request


def measure(client, request):
    started = time.perf_counter()
    first_content = None
    fragments, chunks = [], []
    usage, finish = None, None
    with client.chat.completions.create(
        model=MODEL, **request, stream=True, stream_options={'include_usage': True},
        reasoning_effort='none', temperature=0, max_tokens=64,
    ) as stream:
        for chunk in stream:
            chunks.append(chunk.model_dump(mode='json'))
            if chunk.usage is not None:
                usage = chunk.usage.model_dump(mode='json')
            for choice in chunk.choices:
                if choice.finish_reason is not None:
                    finish = choice.finish_reason
                if choice.delta.content:
                    if first_content is None:
                        first_content = time.perf_counter() - started
                    fragments.append(choice.delta.content)
    total = time.perf_counter() - started
    usage = usage or {}
    prompt = usage.get('prompt_tokens')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    answer = ''.join(fragments).strip()
    return {
        'request': request, 'answer': answer, 'finish_reason': finish,
        'first_content_seconds': first_content, 'total_seconds': total,
        'prompt_tokens': prompt, 'cached_tokens': cached,
        'cached_fraction': cached / prompt if cached is not None and prompt else None,
        'completion_tokens': usage.get('completion_tokens'), 'usage': usage, 'raw_chunks': chunks,
        'request_completed': bool(answer) and finish == 'stop',
    }


def display(value):
    return '未知' if value is None else f'{value:.4f}'


def summarize(pairs):
    summary = {}
    print('\n只汇总追问轮；预热及各对的基础请求不计入：')
    for mode, description in MODES.items():
        group = [p for p in pairs if p['mode'] == mode]
        rows = [p['followup'] for p in group]
        times = [r['first_content_seconds'] for r in rows if r['first_content_seconds'] is not None]
        all_usage = bool(rows) and all(r['prompt_tokens'] is not None and r['cached_tokens'] is not None for r in rows)
        prompt_total = sum(r['prompt_tokens'] for r in rows) if all_usage else None
        fraction = sum(r['cached_tokens'] for r in rows) / prompt_total if prompt_total else None
        item = {
            'samples': len(rows),
            'median_first_content_seconds': statistics.median(times) if times else None,
            'mean_prompt_tokens': statistics.mean(r['prompt_tokens'] for r in rows) if rows and all(r['prompt_tokens'] is not None for r in rows) else None,
            'reported_cached_token_fraction': fraction,
            'original_code_recovered': sum(p['original_code_recovered'] for p in group),
            'input_evidence_followed': sum(p['input_evidence_followed'] for p in group),
        }
        summary[mode] = item
        print(f"{mode}: n={len(rows)} 首段延迟中位数={display(item['median_first_content_seconds'])}s "
              f"平均输入token={display(item['mean_prompt_tokens'])} 缓存比例={display(fraction)} "
              f"找回原编号={item['original_code_recovered']}/{len(rows)} "
              f"按当前输入作答={item['input_evidence_followed']}/{len(rows)}")
    return summary


def main():
    client = OpenAI(base_url='http://127.0.0.1:11434/v1', api_key='ollama', timeout=180, max_retries=0)
    result_dir = Path(__file__).resolve().parents[1] / '运行结果' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    result_dir.mkdir(parents=True)
    path = result_dir / 'record.json'
    result = {
        'model': MODEL, 'repeats_per_mode': REPEATS, 'pairs': [], 'warmup': None,
        'failure': None, 'observation_completed': False,
        'scope': '本地配对上下文对照，历史预设，真实模型请求，工具定义仅供观察且不执行。',
        'notes': [
            '每对先用原始上下文请求一次，再只修改指定条件追加追问；共1+6*3*2=37次请求。',
            '两组请求保存完整输入；每对使用唯一实验编号以减轻跨对前缀复用，但服务可能仍复用公共工具前缀。',
            '汇总只计追问轮。缓存比例来自服务字段，缺失记未知；首段正文延迟近似TTFT。',
            '滑动窗口保留稳定系统，所以仍可能有较高缓存比例；删除了最早编号，UNKNOWN是符合当前证据的回答。',
            '纯文本格式首次转换会改变输入序列；长期稳定的纯文本本身仍可能缓存，不能认定必然低缓存。',
            '工具在服务端模板中的具体位置和序列化方式由Ollama决定，以实测输入token及缓存字段为准。',
            '每组3次是探索性观察，不替代官方Kimi完整ReAct文件任务、计费或任务成功率评估。',
        ],
    }

    def save():
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print('2-3：预热1次，六种方式各3对，共37次本机请求；不执行任何工具。', flush=True)
    save()
    try:
        warmup_request, _ = build_case('correct', 0)
        result['warmup'] = measure(client, warmup_request)
        if not result['warmup']['request_completed']:
            raise RuntimeError('预热未正常返回正文；若模型提出工具调用，本脚本不会执行。')
        schedule = [(mode, trial) for mode in MODES for trial in range(REPEATS)]
        random.Random(42).shuffle(schedule)
        for index, (mode, trial) in enumerate(schedule, 1):
            print(f'\n[{index}/{len(schedule)}] {mode}：{MODES[mode]}，第{trial + 1}次', flush=True)
            request, expected = build_case(mode, trial)
            baseline = measure(client, request)
            pair = {'mode': mode, 'trial': trial + 1, 'expected_code': expected, 'baseline': baseline}
            result.setdefault('attempts', []).append(pair)
            save()
            if not baseline['request_completed']:
                raise RuntimeError('基础请求未正常完成；请检查保存的finish_reason及原始响应。')
            target = build_followup(request, baseline['answer'], mode)
            followup = measure(client, target)
            evidence_present = any(expected in m['content'] for m in target['messages'])
            grounded_answer = expected if evidence_present else 'UNKNOWN'
            pair.update({
                'followup': followup, 'code_present_in_input': evidence_present,
                'original_code_recovered': followup['answer'] == expected,
                'input_evidence_followed': followup['answer'] == grounded_answer,
            })
            result['pairs'].append(pair)
            save()
            print('回答：', followup['answer'], '；原编号在本轮输入中：', evidence_present)
            print('输入token：', followup['prompt_tokens'], '缓存token：', followup['cached_tokens'],
                  '首段正文延迟：', display(followup['first_content_seconds']), '秒', flush=True)
            if not followup['request_completed']:
                raise RuntimeError('追问未正常完成（可能截断或提出工具请求）；本脚本不会执行工具。')
    except Exception as exc:
        result['failure'] = f'{type(exc).__name__}: {exc}'
        print('本次未完成：', result['failure'], flush=True)
    finally:
        result['summary'] = summarize(result['pairs'])
        result['observation_completed'] = result['failure'] is None and len(result['pairs']) == len(MODES) * REPEATS
        save()
        print('本次记录：', path)
    if not result['observation_completed']:
        print('KV_OBSERVATION_INCOMPLETE：部分结果已保存，请反馈输出。')
        raise SystemExit(1)
    print('KV_OBSERVATION_COMPLETE：六种方式的配对请求完成，结论以实际汇总为准。')


if __name__ == '__main__':
    main()
