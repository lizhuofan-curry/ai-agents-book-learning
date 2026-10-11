"""相同五道订单决策题，比较六组提示词；真实调用本机模型，不执行工具。

本地单步工具决策观察，不是官方完整τ-bench业务任务成功率。
"""

import json
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from openai import OpenAI

MODEL = 'qwen3:0.6b'
GROUPS = {
    'baseline': {'tone': 'professional', 'shuffle': False, 'remove_descriptions': False},
    'tone_casual': {'tone': 'casual', 'shuffle': False, 'remove_descriptions': False},
    'tone_emphatic': {'tone': 'emphatic', 'shuffle': False, 'remove_descriptions': False},
    'wiki_shuffle': {'tone': 'professional', 'shuffle': True, 'remove_descriptions': False},
    'no_tool_desc': {'tone': 'professional', 'shuffle': False, 'remove_descriptions': True},
    'combined': {'tone': 'casual', 'shuffle': True, 'remove_descriptions': True},
}
TONES = {
    'professional': '你是专业的订单客服，请准确依据业务规则选择处理方式。',
    'casual': '你是订单客服小助手，咱们轻松点，照着业务规则把订单处理好吧～',
    'emphatic': '你是订单客服！务必准确处理订单！务必依据业务规则！',
}
# 规则条件完整且互斥；随机排列时只改变组织顺序，不删减规则。
RULES = [
    ('已退款', '订单status为refunded时，只告知已退款，不得重复退款。'),
    ('超过期限', '订单未退款且签收超过30天时，拒绝退款；拆封和确认状态不影响这个结论。'),
    ('已拆封', '订单未退款、签收不超过30天且已经拆封时，转人工处理。'),
    ('尚未确认', '订单未退款、签收不超过30天、未拆封且用户未确认退款时，先向用户请求确认。'),
    ('可以退款', '订单未退款、签收不超过30天、未拆封且用户已确认退款时，发起退款。'),
]
# 使用不透露动作含义的工具名，明确隔离工具描述的作用；这会放大文档消融难度。
TOOL_ACTIONS = [
    ('op_a', '发起退款', '为满足退款规则且用户已确认的订单发起退款。'),
    ('op_b', '拒绝退款', '对超过退款期限的订单拒绝退款。'),
    ('op_c', '转人工', '将需要人工处理的订单转交人工客服。'),
    ('op_d', '请求确认', '向尚未确认退款的用户请求确认，暂不退款。'),
    ('op_e', '告知已退款', '告知用户此订单已退款，避免重复退款。'),
]
CASES = [
    {'id': 'T1', 'title': '期限边界：30天可退', 'order': {'order_id': 'A100', 'status': 'paid', 'days_since_delivery': 30, 'opened': False, 'refund_confirmed': True}, 'expected_tool': 'op_a'},
    {'id': 'T2', 'title': '超过期限：31天拒绝', 'order': {'order_id': 'A101', 'status': 'paid', 'days_since_delivery': 31, 'opened': False, 'refund_confirmed': True}, 'expected_tool': 'op_b'},
    {'id': 'T3', 'title': '已拆封：转人工', 'order': {'order_id': 'A102', 'status': 'paid', 'days_since_delivery': 8, 'opened': True, 'refund_confirmed': True}, 'expected_tool': 'op_c'},
    {'id': 'T4', 'title': '用户未确认：先确认', 'order': {'order_id': 'A103', 'status': 'paid', 'days_since_delivery': 2, 'opened': False, 'refund_confirmed': False}, 'expected_tool': 'op_d'},
    {'id': 'T5', 'title': '已经退款：不得重复', 'order': {'order_id': 'A104', 'status': 'refunded', 'days_since_delivery': 10, 'opened': False, 'refund_confirmed': True}, 'expected_tool': 'op_e'},
]


def build_request(group, case):
    config = GROUPS[group]
    rules = deepcopy(RULES)
    if config['shuffle']:
        random.Random(42).shuffle(rules)
        rule_text = '\n'.join(text for _, text in rules)
    else:
        rule_text = '\n'.join(f'【{title}】{text}' for title, text in rules)
    system = TONES[config['tone']] + '\n' + rule_text
    system += '\n请仅提出一个符合规则的工具调用，order_id必须与输入一致。不要只输出解释，不得猜测缺失信息。'
    tools = []
    for name, _, description in TOOL_ACTIONS:
        parameter = {'type': 'string'}
        function = {
            'name': name,
            'parameters': {'type': 'object', 'properties': {'order_id': parameter}, 'required': ['order_id'], 'additionalProperties': False},
        }
        if not config['remove_descriptions']:
            function['description'] = description
            parameter['description'] = '需要处理的订单编号，必须原样取自输入order_id。'
        tools.append({'type': 'function', 'function': function})
    return {
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': '请处理这笔订单：\n' + json.dumps(case['order'], ensure_ascii=False)},
        ],
        'tools': tools,
    }


def evaluate(case, message):
    calls = message.tool_calls or []
    single = len(calls) == 1
    name = calls[0].function.name if single else None
    argument_string = calls[0].function.arguments if single else None
    arguments = None
    parse_error = None
    if single:
        try:
            arguments = json.loads(argument_string)
        except (json.JSONDecodeError, TypeError) as exc:
            parse_error = str(exc)
    tool_correct = single and name == case['expected_tool']
    arguments_correct = single and isinstance(arguments, dict) and arguments == {'order_id': case['order']['order_id']}
    return {
        'single_call': single, 'tool_name': name, 'argument_string': argument_string,
        'parsed_arguments': arguments, 'argument_parse_error': parse_error,
        'tool_correct': tool_correct, 'arguments_correct': arguments_correct,
        'decision_correct': tool_correct and arguments_correct,
        'tool_executed': False,
    }


def summarize(rows):
    summary = {}
    lines = ['# 本地提示工程工具决策对照', '',
             '每组相同5题。分数只检查单次工具名与参数，不是完整业务任务成功率。', '',
             '| 组别 | 样本数 | 工具选择正确 | 参数正确 | 两者均正确 |',
             '| --- | --- | --- | --- | --- |']
    print('\n五题工具决策汇总（预热不计入）：')
    for group in GROUPS:
        selected = [r for r in rows if r['group'] == group]
        item = {'samples': len(selected)}
        for metric in ('tool_correct', 'arguments_correct', 'decision_correct'):
            item[metric] = sum(r['evaluation'][metric] for r in selected)
        summary[group] = item
        n = len(selected)
        print(f"{group}: n={n} 工具正确={item['tool_correct']}/{n} "
              f"参数正确={item['arguments_correct']}/{n} 两者均正确={item['decision_correct']}/{n}")
        lines.append(f"| {group} | {n} | {item['tool_correct']}/{n} | {item['arguments_correct']}/{n} | {item['decision_correct']}/{n} |")
    lines.extend(['', '仅5个固定样例，不能推断普遍排序；每项原始请求与回答见record.json。',
                  '工具名op_a～op_e刻意不透露用途，移除描述会放大困难；真实系统可能从名称推断用途。'])
    return summary, '\n'.join(lines) + '\n'


def main():
    client = OpenAI(base_url='http://127.0.0.1:11434/v1', api_key='ollama', timeout=180, max_retries=0)
    result_dir = Path(__file__).resolve().parents[1] / '运行结果' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    result_dir.mkdir(parents=True)
    result = {
        'model': MODEL, 'group_definitions': GROUPS, 'cases': CASES,
        'rows': [], 'warmup': None, 'failure': None, 'observation_completed': False,
        'scope': '本机真实模型的单步工具决策；不执行工具，不是τ-bench完整客服环境。',
        'parameters': {'reasoning_effort': 'none', 'temperature': 0, 'max_tokens': 256},
    }

    def save():
        (result_dir / 'record.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print('2-4：预热1次 + 六组各5题 = 31次本机请求。每题只判断一次，不执行工具。', flush=True)
    save()
    try:
        result['warmup'] = client.chat.completions.create(
            model=MODEL, messages=[{'role': 'user', 'content': '请仅回复准备好了。'}],
            reasoning_effort='none', temperature=0, max_tokens=64,
        ).model_dump(mode='json')
        schedule = [(group, case) for group in GROUPS for case in CASES]
        random.Random(42).shuffle(schedule)
        for index, (group, case) in enumerate(schedule, 1):
            request = build_request(group, case)
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL, **request, tool_choice='auto', **result['parameters'],
            )
            elapsed = time.perf_counter() - started
            message = response.choices[0].message
            evaluation = evaluate(case, message)
            finish = response.choices[0].finish_reason
            row = {
                'group': group, 'case_id': case['id'], 'case_title': case['title'],
                'expected_tool': case['expected_tool'], 'expected_order_id': case['order']['order_id'],
                'request': request, 'response': response.model_dump(mode='json'),
                'total_seconds': elapsed, 'evaluation': evaluation,
                'request_completed': finish in ('stop', 'tool_calls'),
            }
            result['rows'].append(row)
            save()
            print(f"[{index}/{len(schedule)}] {group} / {case['title']}", flush=True)
            print('  预期：', case['expected_tool'], {'order_id': case['order']['order_id']})
            print('  实际：', evaluation['tool_name'], evaluation['argument_string'],
                  '；判定：', '正确' if evaluation['decision_correct'] else '未通过', flush=True)
            if not message.tool_calls:
                print('  未提出工具调用，正文：', message.content)
    except Exception as exc:
        result['failure'] = f'{type(exc).__name__}: {exc}'
        print('测量中断：', result['failure'], flush=True)
    finally:
        result['summary'], report = summarize(result['rows'])
        result['observation_completed'] = (
            result['failure'] is None and len(result['rows']) == len(GROUPS) * len(CASES)
            and all(r['request_completed'] for r in result['rows'])
        )
        save()
        (result_dir / '对照汇总.md').write_text(report, encoding='utf-8')
        print('本次记录：', result_dir)
    if not result['observation_completed']:
        print('PROMPT_ABLATION_INCOMPLETE：请求未全部正常完成，部分结果已保存。')
        raise SystemExit(1)
    print('PROMPT_ABLATION_COMPLETE：六组五题测量结束；回答对错请看汇总，不执行工具。')


if __name__ == '__main__':
    main()
