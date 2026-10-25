"""真实本机模型的两组工具选择；目录检索本地执行，业务工具只返回模拟结果。"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catalog_step = load('catalog_source', Path(__file__).with_name('01_阅读工具目录与按需展示.py'))
http = load('bounded_http_source', ROOT.parent/'4.2感知工具/我的代码/04_本机模型选择并调用MCP工具.py')
sdk = http.sdk
MODEL = 'qwen3.5:9b'
MAX_CALLS = 5
MAX_ACTIONS = 4
TOP_K = 5
DISCOVER = {'type': 'function', 'function': {'name': 'discover_tools',
    'description': '根据当前缺少的能力查找工具说明。用need描述具体需求，返回至多5个候选；候选会成为可调用工具。',
    'parameters': {'type': 'object', 'properties': {'need': {'type': 'string', 'minLength': 1}},
                   'required': ['need'], 'additionalProperties': False}}}
SYSTEM = '''你正在参加工具选择教学实验。用户需要的专业工具位于本地目录中，根据当前可用工具决定调用或结束。
业务工具只返回明确标注的模拟数据，不会真实查询论文或下载文件。可以用模拟结果继续填写下一步参数，但不能声称真实任务已经完成。
每次只选择一个工具，名称和参数必须来自当前可用定义；若缺少能力，可使用当前提供的发现接口描述需求。
总计最多4次工具动作和5次模型请求。根据工具实际返回继续，不能虚构已执行的动作。参数应符合定义。
信息足够或预算不足时，用简洁中文说明实际选择过程和限制。不输出内部思维过程。'''


def terms(text):
    normalized = text.lower().replace('_', ' ')
    english = re.findall(r'[a-z0-9]+', normalized)
    chinese = re.findall(r'[\u4e00-\u9fff]+', normalized)
    tokens = ['en:'+word for word in english]
    for part in chinese:
        tokens.extend('zh:'+c for c in part)
        tokens.extend('bi:'+part[i:i+2] for i in range(len(part)-1))
    return Counter(tokens)


class ToolIndex:
    """固定词项TF-IDF余弦检索，不用嵌入模型或任务标准答案。"""
    def __init__(self, tools):
        self.tools = {t['function']['name']: t for t in tools}
        raw = {name: terms(json.dumps(t['function'], ensure_ascii=False)) for name, t in self.tools.items()}
        df = Counter(term for counts in raw.values() for term in counts)
        self.idf = {term: math.log((1+len(raw))/(1+freq))+1 for term, freq in df.items()}
        self.vectors = {name: self.vector(counts) for name, counts in raw.items()}

    def vector(self, counts):
        values = {term: (1+math.log(count))*self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v*v for v in values.values()))
        return {term: value/norm for term, value in values.items()} if norm else {}

    def search(self, need, excluded=()):
        if not isinstance(need, str) or not need.strip() or len(need) > 500:
            raise ValueError('发现需求须为1至500字符的非空文本')
        query = self.vector(terms(need))
        scores = []
        for name, vector in self.vectors.items():
            if name in excluded:
                continue
            score = sum(value*vector.get(term, 0) for term, value in query.items())
            if score > 0:
                scores.append({'name': name, 'score': score})
        scores.sort(key=lambda x: (-x['score'], x['name']))
        return scores[:TOP_K]


def prepare():
    views, manifest = catalog_step.prepare()
    tools = views['full']
    by = {t['function']['name']: t for t in tools}
    initial = [by['calculator'], by['get_current_time'], DISCOVER]
    return tools, initial, manifest


def selected(message, available):
    from jsonschema import Draft202012Validator
    calls = message.get('tool_calls')
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError('每次只接受一个工具动作，不自动拆分或代填')
    function = calls[0].get('function', {})
    name, args = function.get('name'), function.get('arguments')
    if not isinstance(name, str) or name not in available or not isinstance(args, dict):
        raise ValueError('调用未知或尚未暴露的工具，或参数不是对象')
    schema = available[name]['function']['parameters']
    Draft202012Validator(schema).validate(args)
    if not set(args) <= set(schema.get('properties', {})):
        raise ValueError('不接受额外参数')
    if name == 'discover_tools' and (not args['need'].strip() or len(args['need']) > 500):
        raise ValueError('发现需求为空或过长')
    return name, deepcopy(args)


def simulated_result(name, args):
    # 模拟值是教学占位，域名不可用于真实下载；不调用官方TOOL_IMPLS或任何外部服务。
    if name == 'arxiv_search':
        count = args['max_results']
        if type(count) is not int or not 1 <= count <= 10:
            return {'simulated': True, 'success': False, 'error': '教学模拟支持1至10条结果'}
        papers = [{'title': '教学模拟Transformer论文'+str(i),
                   'pdf_url': f'https://example.invalid/teaching/transformer-{i}.pdf',
                   'notice': '虚构教学占位，不是真实论文'} for i in range(1, min(count, 3)+1)]
        return {'simulated': True, 'success': True, 'results': papers}
    if name == 'download_file':
        return {'simulated': True, 'success': True, 'requested_url': args['url'],
                'requested_path': args['path'], 'file_created': False,
                'notice': '仅记录模拟下载选择，没有网络请求或文件下载'}
    return {'simulated': True, 'success': False, 'tool': name, 'notice': '已记录该工具选择；此教学任务没有为它执行真实业务'}


def trace_metrics(arm):
    actions = arm.get('actions', [])
    business = [a for a in actions if a['name'] != 'discover_tools']
    names = [a['name'] for a in business]
    bindings = []
    urls_seen = set()
    search_chosen = False
    for action in business:
        if action['name'] == 'arxiv_search' and action.get('result', {}).get('success') is True:
            search_chosen = True
            urls_seen.update(p['pdf_url'] for p in action['result']['results'])
        if action['name'] == 'download_file':
            bindings.append(action['arguments']['url'] in urls_seen)
    return {'model_requests': len(arm.get('calls', [])), 'tool_actions': len(actions),
            'discovery_actions': sum(a['name'] == 'discover_tools' for a in actions),
            'selected_business_tools': names,
            'search_then_simulated_download_with_returned_url': search_chosen and bool(bindings) and all(bindings),
            'real_papers_searched': False, 'real_files_downloaded': 0,
            'initial_schema_characters': arm['initial_schema_metrics']['json_characters'],
            'summed_request_tool_schema_characters': sum(c['schema_metrics']['json_characters'] for c in arm.get('calls', [])),
            'prompt_tokens_reported': sum(c.get('prompt_tokens', 0) or 0 for c in arm.get('calls', [])),
            'output_tokens_reported': sum(c.get('output_tokens', 0) or 0 for c in arm.get('calls', [])),
            'semantics': 'final answer and relevance require review; no real-task completion score'}


async def call_model(request, arm, save):
    import httpx2
    if len(arm['calls']) >= MAX_CALLS:
        raise ValueError('单组模型请求预算耗尽')
    entry = {'status': 'started', 'request': deepcopy(request),
             'schema_metrics': catalog_step.metrics(request.get('tools', []))}
    arm['calls'].append(entry)
    save()
    started = time.perf_counter()
    try:
        async with httpx2.AsyncClient(trust_env=False, follow_redirects=False,
                                      timeout=httpx2.Timeout(600, connect=10)) as client:
            response = await client.post(http.URL, json=request)
            entry.update(http_status=response.status_code, raw_http_body=response.text)
            save()
            response.raise_for_status()
        raw = json.loads(response.text, object_pairs_hook=http.unique)
        entry['response'] = raw
        save()
        if raw.get('model') != MODEL or raw.get('done') is not True or raw.get('done_reason') != 'stop':
            raise ValueError('模型名或停止状态异常')
        message = raw.get('message')
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            raise ValueError('模型消息结构异常')
        entry.update(status='completed', prompt_tokens=raw.get('prompt_eval_count'), output_tokens=raw.get('eval_count'))
        return message
    except BaseException:
        entry['status'] = 'incomplete'
        raise
    finally:
        entry['elapsed_seconds'] = time.perf_counter()-started
        save()


async def run_arm(mode, full, initial, arm, save, model_call=call_model):
    start_tools = full if mode == 'full' else initial
    available = {t['function']['name']: deepcopy(t) for t in start_tools}
    index = ToolIndex(full)
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': catalog_step.TASK}]
    arm.update(status='running', actions=[], calls=[], initial_schema_metrics=catalog_step.metrics(start_tools),
               initial_available_names=list(available))
    save()
    started = time.perf_counter()
    try:
        for turn in range(MAX_CALLS):
            print(mode, '模型请求', turn+1, '；可见工具', len(available), flush=True)
            exposed = list(available.values()) if len(arm['actions']) < MAX_ACTIONS else None
            request = http.payload(messages, exposed)
            message = await model_call(request, arm, save)
            arm.setdefault('assistant_messages', []).append(message)
            save()
            if not message.get('tool_calls'):
                answer = message.get('content')
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError('模型既未调用也未回答')
                arm.update(status='answered_review_pending', final_answer=answer)
                print(mode, '最终回答：', answer, flush=True)
                return
            if len(arm['actions']) >= MAX_ACTIONS:
                raise ValueError('工具动作预算耗尽')
            name, args = selected(message, available)
            action = {'name': name, 'arguments': args}
            arm['actions'].append(action)
            save()
            print(mode, '选择：', name, json.dumps(args, ensure_ascii=False), flush=True)
            if name == 'discover_tools':
                ranking = index.search(args['need'], excluded=set(available))
                candidates = [index.tools[r['name']] for r in ranking]
                action['ranking'] = ranking
                action['added_schema_metrics'] = catalog_step.metrics(candidates)
                available.update({t['function']['name']: t for t in candidates})
                result = {'discovery_is_local': True, 'candidates': candidates,
                          'notice': '这些定义已加入后续可用工具；业务调用仅模拟'}
                print(mode, '发现候选：', [r['name'] for r in ranking], flush=True)
            else:
                result = simulated_result(name, args)
                print(mode, '教学模拟返回：', json.dumps(result, ensure_ascii=False), flush=True)
            action['result'] = result
            messages.extend([message, {'role': 'tool', 'tool_name': name, 'content': json.dumps(result, ensure_ascii=False)}])
            save()
        raise ValueError('模型请求预算耗尽，尚未回答')
    except (Exception, KeyboardInterrupt) as error:
        arm.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        print(mode, '未完成，不重试：', arm['error'], flush=True)
        if isinstance(error, KeyboardInterrupt):
            raise
    finally:
        arm['elapsed_seconds'] = time.perf_counter()-started
        arm['metrics'] = trace_metrics(arm)
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != sdk.ENVIRONMENT.resolve():
        if not sdk.PYTHON.is_file():
            raise RuntimeError('未找到已配置的第四章环境')
        print('切换已有第四章环境；不安装依赖。', flush=True)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.execv(str(sdk.PYTHON), [str(sdk.PYTHON), str(Path(__file__)), *sys.argv[1:]])
    sdk.use_sdk_environment()
    full, initial, manifest = prepare()
    print('4-1第二步：全量126工具 vs 初始基础工具+按需发现；同一本机qwen3.5:9b。')
    print('任务：', catalog_step.TASK)
    print('每组最多5次模型请求、4次工具动作；总计最多10次请求。无预热、重试、云端、嵌入或下载。')
    print('候选由本地TF-IDF词项检索产生，业务工具只返回教学模拟；不实际搜索论文或下载文件。')
    if args.preview:
        ToolIndex(full)
        print('TOOL_DISCOVERY_COMPARE_PREVIEW_ONLY：来源/126目录/初始3工具检查通过，无模型或业务调用。')
        return 0
    directory = ROOT/'运行结果'/('tool_choice_compare_'+datetime.now(sdk.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'
    record = {'status': 'running', 'created_at': datetime.now(sdk.CHINA).isoformat(),
              'script_sha256': sdk.sha(Path(__file__)), 'catalog_helper_sha256': sdk.sha(Path(catalog_step.__file__)),
              'payload_helper_sha256': sdk.sha(Path(http.__file__)), 'manifest': manifest,
              'model': MODEL, 'task': catalog_step.TASK, 'maximum_model_requests': 10,
              'per_arm_model_budget': MAX_CALLS, 'per_arm_tool_action_budget': MAX_ACTIONS,
              'candidate_top_k': TOP_K, 'retrieval': 'local TF-IDF cosine on English tokens and Chinese uni/bigrams',
              'automatic_retries': 0, 'cloud_requests': 0, 'real_business_operations': 0,
              'arms': {}, 'design': 'full first, discovery second; independent histories; native tools interface; shared sampling and budget',
              'limitations': 'single known teaching case; different initial tool availability; simulated business results; not official text-action/embedding campaign or causal accuracy benchmark'}

    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        tmp.replace(path)

    async def compare():
        for mode in ('full', 'discovery'):
            arm = {}
            record['arms'][mode] = arm
            await run_arm(mode, full, initial, arm, save)

    save()
    print('记录路径：', path, flush=True)
    try:
        asyncio.run(compare())
        record['status'] = 'comparison_saved_review_pending'
        for name, arm in record['arms'].items():
            print(name, '状态：', arm['status'], '统计：', json.dumps(arm['metrics'], ensure_ascii=False))
        print('TOOL_DISCOVERY_COMPARISON_SAVED：两组轨迹保存，按各组状态和语义核对，不代表真实下载完成。')
        return 0
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='incomplete', error={'type': type(error).__name__, 'message': str(error)})
        print('对照中断，不重试：', record['error'])
        return 1
    finally:
        save()
        print('记录路径：', path)


if __name__ == '__main__':
    raise SystemExit(main())
