"""用户运行：用项目已有DeepSeek API完成搜索/阅读小任务，Serper默认。

用户已选择云端路线；最多7次模型请求、2次搜索、2次读取，可能计费。
不重写06记录、不下载模型、不预填正文或URL、不强迫程序代替模型选择工具。
"""

import argparse
from copy import deepcopy
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI
import requests

from safe_records import atomic_json_dump


CODE_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('online_web_tools', CODE_DIR / '06_模型自主搜索与阅读.py')
online = importlib.util.module_from_spec(spec)
spec.loader.exec_module(online)
ROOT = online.ROOT
qa = online.qa
search = online.search


def scrub(value, keys):
    for key in keys:
        value = search.redact(value, key)
    return value


def run_loop(client, tools, record, save, keys):
    messages = [{'role': 'system', 'content': online.SYSTEM}, {'role': 'user', 'content': online.QUESTION}]
    record['messages'] = messages
    seen_ids = set()
    for number in range(1, record['max_rounds'] + 1):
        kwargs = {'model': record['model'], 'messages': deepcopy(messages), 'tools': online.TOOLS,
                  'tool_choice': 'auto', 'stream': False, 'max_tokens': 1024,
                  'extra_body': {'thinking': {'type': 'disabled'}}}
        wire_body = {k: v for k, v in kwargs.items() if k != 'extra_body'}
        wire_body.update(kwargs['extra_body'])
        # 小任务成本保护，用保守字节代理单独命名，非DeepSeek真实窗口溢出。
        bound = len(json.dumps(wire_body, ensure_ascii=False).encode('utf-8')) + 2048
        if bound > 32000:
            record.update({'status': 'teaching_input_budget_blocked', 'budget_proxy_bytes': bound})
            return
        turn = {'round': number, 'request': deepcopy(wire_body)}
        record['turns'].append(turn)
        save()
        print(f'第{number}/{record["max_rounds"]}轮，DeepSeek正在决定下一步……', flush=True)
        response = client.chat.completions.create(**kwargs)
        turn['raw_response'] = response.model_dump(mode='json')
        usage = response.usage
        turn['usage'] = usage.model_dump(mode='json') if usage is not None else None
        turn['prompt_tokens'] = usage.prompt_tokens if usage is not None else None
        turn['completion_tokens'] = usage.completion_tokens if usage is not None else None
        save()
        if not response.choices:
            raise ValueError('DeepSeek没有返回choices')
        choice = response.choices[0]
        turn['finish_reason'] = choice.finish_reason
        if choice.finish_reason not in ('stop', 'tool_calls'):
            record['status'] = 'generation_incomplete'
            return
        message = choice.message
        assistant = {'role': 'assistant', 'content': message.content}
        calls = message.tool_calls or []
        if calls:
            assistant['tool_calls'] = [call.model_dump(mode='json', exclude_none=True) for call in calls]
        # 即使意外返回思考内容，也按官方tools协议保留，不静默删除。
        reasoning = getattr(message, 'reasoning_content', None)
        if isinstance(reasoning, str):
            assistant['reasoning_content'] = reasoning
        messages.append(assistant)
        if calls:
            ids = [call.id for call in calls]
            if any(not isinstance(identifier, str) or not identifier for identifier in ids):
                raise ValueError('工具调用缺少有效ID，不能伪造关联编号')
            if len(set(ids)) != len(ids) or any(identifier in seen_ids for identifier in ids):
                raise ValueError('工具调用ID重复，停止而不是错误关联结果')
            seen_ids.update(ids)
            if number == record['max_rounds']:
                turn['pending_tools_not_executed'] = True
                record['status'] = 'round_budget_reached'
                return
            for index, call in enumerate(calls):
                raw_call = call.model_dump(mode='json', exclude_none=True)
                event = {'round': number, 'index': index, 'tool_call_id': call.id, 'raw_call': raw_call}
                record['tool_events'].append(event)
                save()
                try:
                    name, arguments = online.parse_call(raw_call)
                    event.update({'name': name, 'arguments': arguments})
                    print('模型请求工具：', name, json.dumps(arguments, ensure_ascii=False), flush=True)
                    result = tools.execute(name, arguments, event)
                except Exception as error:
                    result = {'ok': False, 'error_type': type(error).__name__, 'message': str(error)}
                result = scrub(result, keys)
                event['result'] = result
                # DeepSeek采用OpenAI协议，必须使用实际返回的tool_call_id。
                messages.append({'role': 'tool', 'tool_call_id': call.id,
                                 'content': json.dumps(result, ensure_ascii=False)})
                print('工具结果：', '成功' if result['ok'] else result['message'], flush=True)
                save()
        else:
            content = message.content
            if not isinstance(content, str) or not content.strip():
                record['status'] = 'empty_final_response'
                return
            record['final_answer'] = content
            record['checks'] = online.check_final(content, tools)
            events = record['tool_events']
            record['chain_completed'] = any(
                event.get('name') == 'fetch_webpage' and event.get('result', {}).get('ok') is True and
                any(previous.get('name') == 'search_web' and previous.get('result', {}).get('ok') is True
                    for previous in events[:index])
                for index, event in enumerate(events))
            record['status'] = 'answered'
            print('最终回答：\n', content)
            return
    record['status'] = 'round_budget_reached'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=search.PROVIDERS, default='serper')
    parser.add_argument('--max-rounds', type=int, choices=range(1, 8), default=7)
    args = parser.parse_args()
    config = dotenv_values(search.CONFIG_FILE) if search.CONFIG_FILE.is_file() else {}

    def setting(name, default=''):
        return (os.environ.get(name) or config.get(name) or default).strip()

    api_key = setting('DEEPSEEK_API_KEY')
    search_key_name = search.PROVIDERS[args.provider][0]
    search_key = setting(search_key_name)
    base_url = setting('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    model = setting('DEEPSEEK_MODEL', 'deepseek-flash')
    keys = [api_key, search_key]
    directory = ROOT / '运行结果' / ('deepseek_online_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True)
    record = {'stage': 'deepseek_autonomous_online_tool_loop', 'created_at': datetime.now().astimezone().isoformat(),
              'model_provider': 'deepseek', 'model': model, 'base_url': base_url,
              'search_provider': args.provider, 'max_rounds': args.max_rounds, 'max_retries': 0,
              'status': 'prepared', 'turns': [], 'tool_events': [], 'chain_completed': False,
              'measurement_completed': False, 'scope': 'DeepSeek在线小任务；未执行六策略压缩',
              'record_writer_version': 'atomic_records_v2',
              'source_hashes': {p.name: qa.sha(p.read_bytes()) for p in (Path(__file__),
                  CODE_DIR / 'safe_records.py',
                  CODE_DIR / '06_模型自主搜索与阅读.py', CODE_DIR / '02_真实搜索接口检查.py',
                  CODE_DIR / '03_读取官方网页正文.py', CODE_DIR / '04_依据真实网页回答.py')}}
    tools = None

    def save():
        if tools is not None:
            record.update({'search_request_attempts': tools.searches, 'web_get_attempts': tools.get_attempts,
                           'read_tool_calls': tools.fetches, 'sources': deepcopy(list(tools.pages.values()))})
        record['model_request_attempts'] = len(record['turns'])
        for field in ('prompt_tokens', 'completion_tokens'):
            values = [turn.get(field) for turn in record['turns']]
            record['all_' + field] = sum(values) if all(isinstance(value, int) for value in values) else None
        atomic_json_dump(directory / 'record.json', scrub(record, keys))

    save()
    print(f'DeepSeek模型：{model}；搜索：{args.provider}；使用项目根目录.env。', flush=True)
    print(f'最多{args.max_rounds}次云端模型请求、2次搜索、2次读取；可能按账号规则计费，无自动重试。', flush=True)
    try:
        if not api_key or not search_key:
            missing = [name for name, value in (('DEEPSEEK_API_KEY', api_key), (search_key_name, search_key)) if not value]
            raise ValueError('根目录.env缺少：' + ', '.join(missing))
        timeout = float(setting('DEEPSEEK_TIMEOUT', '180'))
        if not 0 < timeout <= 300:
            raise ValueError('DEEPSEEK_TIMEOUT需在0到300秒之间')
        with requests.Session() as external, OpenAI(api_key=api_key, base_url=base_url,
                                                    timeout=timeout, max_retries=0) as client:
            tools = online.WebTools(args.provider, search_key, external, directory)
            run_loop(client, tools, record, save, keys)
            record['measurement_completed'] = True
    except (Exception, KeyboardInterrupt) as error:
        detail = {'type': type(error).__name__, 'message': str(error),
                  'http_status': getattr(error, 'status_code', None),
                  'pending_record_path': getattr(error, 'pending_record_path', None)}
        record.update({'status': 'interrupted', 'error': detail})
        print('运行中断：', scrub(detail, keys))
    finally:
        record['finished_at'] = datetime.now().astimezone().isoformat()
        save()
        print('状态：', record['status'], '；搜索→读取→最终回复链路：', record['chain_completed'])
        print('模型请求：', record['model_request_attempts'], '；搜索：', record.get('search_request_attempts', 0),
              '；网页GET：', record.get('web_get_attempts', 0))
        print('输入token：', record['all_prompt_tokens'], '；输出token：', record['all_completion_tokens'])
        print('最终机械检查：', json.dumps(record.get('checks', {}), ensure_ascii=False))
        print('本次记录：', directory / 'record.json')
    if not record['measurement_completed']:
        print('DEEPSEEK_ONLINE_RUN_INCOMPLETE：保留失败，不自动换模型或填工具动作。')
        raise SystemExit(1)
    print('DEEPSEEK_ONLINE_OBSERVATION_COMPLETE：测量结束，链路与回答要求需分别检查。')


if __name__ == '__main__':
    main()
