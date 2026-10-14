"""用户或助手离线核对：从已保存的第3轮响应恢复答案，零API/搜索请求。

只支持完整保存且工具动作已结束的最终响应。原record.json不改写。
"""

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

from safe_records import atomic_json_dump


CODE_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('online_recovery', CODE_DIR / '06_模型自主搜索与阅读.py')
online = importlib.util.module_from_spec(spec)
spec.loader.exec_module(online)
ROOT = online.ROOT
DEFAULT_RECORD = ROOT / '运行结果/deepseek_online_20261014_183831_601088/record.json'


def recover(path):
    path = Path(path).resolve()
    original_bytes = path.read_bytes()
    record = json.loads(original_bytes.decode('utf-8'))
    turns = record.get('turns', [])
    if not turns or record.get('stage') != 'deepseek_autonomous_online_tool_loop':
        raise ValueError('不是预期的DeepSeek轨迹')
    if record.get('status') != 'interrupted' or record.get('error', {}).get('type') != 'OSError':
        raise ValueError('本工具仅恢复本地写入错误中断的已保存响应')
    last = turns[-1].get('raw_response', {})
    choices = last.get('choices', [])
    if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
        raise ValueError('没有完整保存的正常最终响应，不能离线补答案')
    message = choices[0].get('message', {})
    answer = message.get('content')
    if message.get('tool_calls') or not isinstance(answer, str) or not answer.strip():
        raise ValueError('最后响应仍需执行工具或没有正文，不能伪造完成')
    # 校验此前每一个正式工具请求都有唯一匹配的已执行事件。
    calls = []
    for turn in turns[:-1]:
        prior_choices = turn.get('raw_response', {}).get('choices', [])
        if len(prior_choices) != 1 or prior_choices[0].get('finish_reason') not in ('stop', 'tool_calls'):
            raise ValueError('此前响应不完整')
        calls.extend(prior_choices[0].get('message', {}).get('tool_calls') or [])
    ids = [call.get('id') for call in calls]
    events = record.get('tool_events', [])
    event_ids = [event.get('tool_call_id') for event in events]
    if not ids or len(set(ids)) != len(ids) or ids != event_ids:
        raise ValueError('工具请求与已执行事件未完整对应')
    event_by_id = {event['tool_call_id']: event for event in events}
    for call in calls:
        event = event_by_id[call['id']]
        name, arguments = online.parse_call(call)
        if event.get('name') != name or event.get('arguments') != arguments or 'result' not in event:
            raise ValueError('工具调用参数/返回与事件不一致')
    # 最后一次请求必须确实携带这些工具结果，而非仅在别处保存过。
    sent_results = {m.get('tool_call_id'): m.get('content')
                    for m in turns[-1]['request']['messages'] if m.get('role') == 'tool'}
    for event in events:
        if event['tool_call_id'] not in sent_results or json.loads(sent_results[event['tool_call_id']]) != event['result']:
            raise ValueError('最终请求没有对应工具结果')
    pages = {}
    for source in record.get('sources', []):
        identifier = source.get('id', '')
        if not re.fullmatch(r'S[0-9]+', identifier):
            raise ValueError('来源ID不符合固定快照格式')
        raw = (path.parent / f'{identifier}.html').read_bytes()
        text = (path.parent / f'{identifier}.txt').read_text(encoding='utf-8')
        if (online.qa.sha(raw) != source.get('raw_html_sha256') or text != source.get('text') or
                online.qa.sha(text.encode('utf-8')) != source.get('text_sha256')):
            raise ValueError('网页快照哈希或内容不一致')
        pages[online.official_url(source['url'])] = source
    search_successes = sum(event.get('name') == 'search_web' and event['result'].get('ok') is True for event in events)
    checks = online.check_final(answer, SimpleNamespace(pages=pages, search_successes=search_successes))
    ordered_chain = any(event.get('name') == 'fetch_webpage' and event['result'].get('ok') is True and
                        any(previous.get('name') == 'search_web' and previous['result'].get('ok') is True
                            for previous in events[:index]) for index, event in enumerate(events))
    usage = [turn['raw_response'].get('usage') for turn in turns]
    if any(not isinstance(value, dict) for value in usage):
        raise ValueError('模型用量记录不完整')
    prompt = sum(value['prompt_tokens'] for value in usage)
    completion = sum(value['completion_tokens'] for value in usage)
    if (len(turns) != record.get('model_request_attempts') or prompt != record.get('all_prompt_tokens') or
            completion != record.get('all_completion_tokens')):
        raise ValueError('模型请求数或token汇总不一致')
    recovered = {'stage': 'offline_recovery_of_saved_final_response',
                 'recovered_at': datetime.now().astimezone().isoformat(),
                 'original_record': str(path), 'original_record_sha256': online.qa.sha(original_bytes),
                 'original_status': record.get('status'), 'original_error': record.get('error'),
                 'status': 'saved_final_response_recovered', 'final_answer': answer, 'checks': checks,
                 'observed_chain_completed': ordered_chain, 'request_tool_result_correlations_verified': True,
                 'webpage_hashes_verified': True, 'original_model_request_attempts': len(turns),
                 'original_search_requests': record.get('search_request_attempts'),
                 'original_web_get_attempts': record.get('web_get_attempts'),
                 'original_prompt_tokens': prompt, 'original_completion_tokens': completion,
                 'model_requests_during_recovery': 0, 'search_requests_during_recovery': 0,
                 'web_requests_during_recovery': 0,
                 'source_selection': [{'url': event.get('canonical_url'),
                                       'was_in_search_results': event.get('was_in_search_results')}
                                      for event in events if event.get('name') == 'fetch_webpage'],
                 'scope': '离线确认已有搜索/读取/最终响应证据；不将原中断运行改写成正常退出，也不代表六策略完成'}
    result_path = path.parent / '恢复记录.json'
    atomic_json_dump(result_path, recovered)
    parsed = json.loads(answer) if checks['fields_valid'] else None
    readable = ('# 已保存回答的离线恢复\n\n'
                '原程序在记录写入时中断；原始record.json保留。本文件提取已保存的最终API响应，没有新增API、搜索或网页请求。\n\n'
                f'技术链路证据：{ordered_chain}。原运行{len(turns)}次模型请求、{record.get("search_request_attempts")}次搜索、{record.get("web_get_attempts")}次网页GET；模型总输入{prompt}、输出{completion} token。\n\n'
                '## 模型原始最终回答\n\n' + (parsed['answer'] if parsed else answer) + '\n\n'
                + ('来源：' + parsed['source_url'] + '\n\n' if parsed else '')
                + f'非空白字符：{checks["answer_chars"]}；100字符要求：{checks["within_100_chars"]}；来源已实际读取：{checks["source_was_read"]}。\n\n'
                '语言/事实仍需人工复核。此恢复不删除原IO错误，不代表原程序正常退出，也不是完整六策略研究。\n')
    (path.parent / '恢复结果.md').write_text(readable, encoding='utf-8')
    if path.read_bytes() != original_bytes:
        raise RuntimeError('原记录意外发生变化')
    return recovered, result_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()
    recovered, output = recover(args.record)
    print('已从原始记录恢复模型回答，无新增API调用：\n', recovered['final_answer'])
    print('技术链路证据：', recovered['observed_chain_completed'])
    print('机械检查：', json.dumps(recovered['checks'], ensure_ascii=False))
    print('恢复记录：', output)
    print('SAVED_RESPONSE_RECOVERED：原中断状态保留，恢复过程未请求模型。')


if __name__ == '__main__':
    main()
