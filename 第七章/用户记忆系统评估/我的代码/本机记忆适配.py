"""官方CardBuilder/MemoryAgent的本机传输适配；不改官方源码。"""
from copy import deepcopy
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PLAN = ROOT / '测试材料/JSONCards单案例方案.json'
MODEL = 'qwen3.5:9b'


class StopPipeline(BaseException):
    """越过上游except Exception，阻止CardBuilder内置重试发送第二次请求。"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_plan():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    required = {
        'upstream_commit': 'dbc046eb896ac4e39aa19c7774c8bf49583b89a6',
        'case_id': 'layer1_01_bank_account',
        'case_source': 'chapter3/user-memory-evaluation/test_cases/layer1/01_bank_account_setup.yaml',
        'case_sha256': 'e9f788fa393a0f07c83f32e90f05d0b2212355ba62e0a5c3288a469b5feb75b6',
        'model': MODEL, 'endpoint': 'http://127.0.0.1:11434/api/chat',
        'maximum_model_requests': 2, 'stage_maximum_requests': {'card_build': 1, 'cards_answer': 1},
        'num_ctx': 16384, 'num_predict': {'card_build': 4096, 'cards_answer': 1024},
        'temperature': 0, 'seed': 42, 'think': False, 'automatic_retries': False,
        'new_embedding_forward_batches': 0, 'retrieval_requests': 0, 'judge_requests': 0,
    }
    if any(plan.get(k) != v for k, v in required.items()):
        raise ValueError('本步固定单题、两次本机请求与参数，不自动更换方案')
    source = ROOT / '官方资料/upstream' / plan['case_source']
    if sha(source) != plan['case_sha256']:
        raise ValueError('固定案例已变化')
    return plan


def load_official():
    if os.environ.get('PYTHON_DOTENV_DISABLED') != '1':
        raise ValueError('须先禁用dotenv读取')
    source = ROOT / '官方资料/upstream/chapter7/user-memory-system-evaluation/experiment.py'
    name = 'pinned_memory_experiment_7_4'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def local_post(payload):
    # 只接受固定回环地址；无环境代理、重定向、云端回退或自动重试。
    connection = http.client.HTTPConnection('127.0.0.1', 11434, timeout=300)
    try:
        connection.request('POST', '/api/chat', body=json.dumps(payload, ensure_ascii=False).encode(),
            headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        return response.status, response.read().decode('utf-8')
    finally:
        connection.close()


class LocalChat:
    def __init__(self, official, plan, record, save, post=local_post):
        self.official, self.plan, self.record, self.save, self.post = official, plan, record, save, post
        self.stage = None

    def complete(self, messages, tools=None, tool_choice=None, json_object=False):
        stage = self.stage
        calls = self.record['model_calls']
        if stage not in self.plan['stage_maximum_requests']:
            raise StopPipeline('未指定已授权阶段')
        if len(calls) >= 2 or any(c['stage'] == stage for c in calls):
            self.record.setdefault('blocked_dispatches', []).append({'stage': stage, 'reason': 'stage/total request budget; no new HTTP request'})
            self.save()
            raise StopPipeline('禁止上游内置重试或超出两次请求预算')
        if tools or tool_choice is not None or json_object != (stage == 'card_build'):
            raise StopPipeline('本步只允许提取JSON和无工具回答')
        payload = {'model': MODEL, 'messages': deepcopy(messages), 'stream': False, 'think': False,
            'options': {'num_ctx': self.plan['num_ctx'], 'num_predict': self.plan['num_predict'][stage],
                'temperature': 0, 'seed': 42}}
        if json_object:
            payload['format'] = 'json'
        entry = {'stage': stage, 'status': 'started', 'request': payload}
        calls.append(entry)
        self.save()
        print(f'第{len(calls)}次本机模型请求：{stage}', flush=True)
        started = time.perf_counter()
        try:
            status, raw = self.post(deepcopy(payload))
            entry.update(http_status=status, raw_http_body=raw, client_wall_ms=(time.perf_counter() - started) * 1000)
            self.save()
            if status != 200:
                raise ValueError('本机模型HTTP状态：' + str(status))
            response = json.loads(raw)
            entry['response'] = response
            self.save()
            msg = response.get('message', {})
            if (response.get('model') != MODEL or response.get('done') is not True
                    or response.get('done_reason') != 'stop' or msg.get('role') != 'assistant'
                    or msg.get('tool_calls') or not isinstance(msg.get('content'), str) or not msg['content'].strip()):
                raise ValueError('响应异常、截断、空回答或意外工具调用；原样保留')
            pin, pout = response.get('prompt_eval_count'), response.get('eval_count')
            if any(type(v) is not int or v < 0 for v in (pin, pout)):
                raise ValueError('模型用量未知，不将未知token补0')
            if json_object:
                obj = json.loads(msg['content'])
                if not isinstance(obj, dict) or set(obj) != {'cards'}:
                    raise ValueError('提取响应必须为仅含cards字段的JSON对象')
            entry.update(status='completed', reported_prompt_tokens=pin, reported_output_tokens=pout,
                cached_input_tokens=None, local_billed_money=None)
            # 上游Usage的0货币/缓存字段仅为内部接口默认值；对外报告保留未知。
            usage = self.official.Usage(input_tokens=pin, output_tokens=pout,
                unpriced_tokens=pin + pout, unpriced_requests=1)
            return self.official.ChatTurn(msg['content'], [], usage, entry['client_wall_ms'],
                {'role': 'assistant', 'content': msg['content']})
        except Exception as exc:
            entry.update(status='failed_or_incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            self.save()


def validate_cards(cards, case):
    if not isinstance(cards, list) or not cards:
        raise ValueError('没有生成非空卡片列表')
    allowed = {h.conversation_id for h in case.conversation_histories}
    required = {'category', 'card_key', 'backstory', 'date_created', 'person', 'relationship',
        'facts', 'source_conversation_ids', 'status', 'memory_tier'}
    keys = []
    for card in cards:
        if not isinstance(card, dict) or required - set(card):
            raise ValueError('卡片缺少官方要求字段')
        key, ids = card['card_key'], card['source_conversation_ids']
        if not isinstance(key, str) or not key.strip() or key in keys:
            raise ValueError('卡片key为空或重复')
        keys.append(key)
        if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) or x not in allowed for x in ids)
                or not isinstance(card['facts'], dict) or not card['facts']
                or card['status'] not in {'current', 'superseded', 'uncertain'}
                or card['memory_tier'] not in {'core', 'supporting'}):
            raise ValueError('卡片类型、来源ID、状态或分层异常')
    return {'nonempty_unique_cards': True, 'required_fields_and_source_ids_valid': True,
        'semantic_faithfulness': 'pending source-level review; valid IDs are not proof of support'}


def run_cards_path(official, case, plan, record, save, directory, post=local_post):
    chat = LocalChat(official, plan, record, save, post)
    chat.stage = 'card_build'
    cards, _, _ = official.CardBuilder(chat).build(case)
    record['cards'] = cards
    record['card_checks'] = validate_cards(cards, case)
    dump(directory / 'cards.json', {'cards': cards})
    save()
    print(f'已生成{len(cards)}张卡片；字段与来源ID检查通过，内容忠实性待核对。', flush=True)
    chat.stage = 'cards_answer'
    result = official.MemoryAgent(chat).cards_only(case.user_question, cards)
    record['answer'] = result.answer
    record['agent_result'] = {'answer': result.answer, 'steps': result.steps, 'tool_calls': result.tool_calls,
        'latency_ms': result.latency_ms, 'trace': result.trace,
        'reported_prompt_tokens': result.usage.input_tokens, 'reported_output_tokens': result.usage.output_tokens,
        'cached_input_tokens': None, 'local_billed_money': None}
    (directory / 'answer.txt').write_text(result.answer, encoding='utf-8')
    build, answer = record['model_calls']
    original_build = build['request']['messages']
    expected_answer = [{'role': 'system', 'content': official.SYSTEM_RULES + '\nResident Advanced JSON Cards:\n' + json.dumps(cards, ensure_ascii=False)},
        {'role': 'user', 'content': case.user_question}]
    checks = {
        'exactly_two_completed_requests_within_budget': len(record['model_calls']) == 2 and all(x['status'] == 'completed' for x in record['model_calls']),
        'card_builder_received_only_source_history': len(original_build) == 1 and original_build[0]['role'] == 'user'
            and case.evaluation_criteria not in original_build[0]['content'] and case.user_question not in original_build[0]['content']
            and all(h.conversation_id in original_build[0]['content'] for h in case.conversation_histories),
        'answer_received_only_cards_and_question': answer['request']['messages'] == expected_answer,
        'independent_two_message_answer_session': len(answer['request']['messages']) == 2,
        'answer_one_step_zero_tools': result.steps == 1 and result.tool_calls == 0 and result.retrieved_chunks == [],
        'saved_artifacts_match_generated_output': json.loads((directory / 'cards.json').read_text()) == {'cards': cards}
            and (directory / 'answer.txt').read_text() == result.answer,
    }
    if not all(checks.values()):
        raise ValueError('链路检查未通过：' + json.dumps(checks))
    record['checks'] = checks
    record['reported_usage'] = {'prompt_tokens': sum(x['reported_prompt_tokens'] for x in record['model_calls']),
        'output_tokens': sum(x['reported_output_tokens'] for x in record['model_calls']),
        'cached_input_tokens': None, 'local_billed_money': None}
    print('回答：\n' + result.answer, flush=True)
    return cards, result
