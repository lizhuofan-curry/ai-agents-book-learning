"""沿用已有DeepSeek配置，三条固定原答案各评分一次；可能计费，无重试。"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import http.client
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import 本机记忆适配 as base
import RAG本机适配 as rag

ROOT, PROJECT = base.ROOT, base.PROJECT
PLAN = ROOT / '测试材料/三路裁判评分方案.json'
DIMENSIONS = ('precision', 'recall', 'reasoning', 'proactivity')
GRADES = {1: 'fail', 2: 'pass', 3: 'good', 4: 'excellent'}
CHINA = timezone(timedelta(hours=8))


def strict_json(raw):
    def reject(value):
        raise ValueError('JSON中出现非有限数字：' + value)
    return json.loads(raw, parse_constant=reject)


def redact(value, secret=''):
    if isinstance(value, str):
        value = value.replace(secret, '[REDACTED]') if secret else value
        return re.sub(r'sk-[A-Za-z0-9_-]{8,}', '[REDACTED]', value)
    if isinstance(value, list):
        return [redact(x, secret) for x in value]
    if isinstance(value, dict):
        return {k: ('[REDACTED]' if k.lower() in {'authorization', 'api_key'} else redact(v, secret)) for k, v in value.items()}
    return value


def load_config(plan):
    from dotenv import dotenv_values
    settings = dotenv_values(PROJECT / '.env')
    get = lambda k, default='': os.environ.get(k) or settings.get(k) or default
    endpoint = get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com').rstrip('/')
    p = urlsplit(endpoint)
    if (p.scheme != 'https' or p.hostname != 'api.deepseek.com' or p.port not in (None, 443)
            or p.username or p.password or p.query or p.fragment or p.path not in ('', '/v1')):
        raise ValueError('裁判只接受DeepSeek官方HTTPS端点，不回退第三方')
    model = get('DEEPSEEK_MODEL', 'deepseek-flash')
    timeout = float(get('DEEPSEEK_TIMEOUT', '180'))
    if model != plan['judge_model'] or not 0 < timeout <= 300:
        raise ValueError('裁判配置模型变化或超时不合法；不自动更换方案')
    return {'model': model, 'endpoint': endpoint, 'path': p.path + '/chat/completions',
        'timeout': timeout, 'key': get('DEEPSEEK_API_KEY')}


class CapturedPrompt(BaseException):
    pass


def original_judge_messages(official, case, answer, memory):
    evaluator = object.__new__(official.LLMEvaluator)
    captured = []

    def capture(messages):
        captured.append(deepcopy(messages))
        raise CapturedPrompt()

    evaluator._call_llm = capture
    try:
        evaluator.evaluate(case, answer, memory)
    except CapturedPrompt:
        pass
    if len(captured) != 1 or len(captured[0]) != 2:
        raise ValueError('无法捕获原裁判的两消息输入')
    return captured[0]


def load_inputs():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    if (plan['upstream_commit'] != 'dbc046eb896ac4e39aa19c7774c8bf49583b89a6'
            or plan['case_id'] != 'layer1_01_bank_account' or plan['judge_provider'] != 'deepseek'
            or plan['judge_model'] != 'deepseek-flash' or plan['maximum_requests'] != 3
            or plan['per_answer_requests'] != 1 or plan['max_output_tokens'] != 4096
            or plan['temperature'] != 0 or plan['thinking'] != 'disabled' or plan['automatic_retries'] is not False
            or [x['arm'] for x in plan['sources']] != ['advanced_json_cards', 'rag', 'hybrid']):
        raise ValueError('固定单题三条原答案及评分预算不符')
    case_path = ROOT / '官方资料/upstream/chapter3/user-memory-evaluation/test_cases/layer1/01_bank_account_setup.yaml'
    if base.sha(case_path) != plan['case_sha256']:
        raise ValueError('固定权威案例变化')
    guard = rag.load(Path(__file__).with_name('01_官方案例与评分规则离线检查.py'), 'judge_reference_guard')
    references = guard.verify_reference()
    rag.augment_dependencies()
    official = base.load_official()
    case = official.UserMemoryEvaluationFramework().get_test_case(plan['case_id'])
    inputs = []
    parent = (ROOT / '本地运行结果').resolve()
    for item in plan['sources']:
        directory = (parent / item['source_run']).resolve()
        if directory.parent != parent:
            raise ValueError('评分来源越界')
        r = json.loads((directory / 'record.json').read_text())
        review = json.loads((directory / '核对记录.json').read_text())
        if (base.sha(directory / 'record.json') != item['record_sha256'] or base.sha(directory / '核对记录.json') != item['review_sha256']
                or review['source_record_sha256'] != item['record_sha256']
                or not all(r['checks'].values()) or not all(review['checks'].values())):
            raise ValueError('固定来源或技术核对变化；语义负面结果仍可评分')
        for name, digest in r['artifacts_sha256'].items():
            if base.sha(directory / name) != digest:
                raise ValueError('来源产物变化')
        answer = (directory / 'answer.txt').read_text()
        call = r['model_calls'][-1]
        if (answer != r['answer'] or answer != call['response']['message']['content']
                or json.loads(call['raw_http_body']) != call['response'] or call['response']['done_reason'] != 'stop'):
            raise ValueError('原始答案与响应不一致')
        if item['arm'] == 'advanced_json_cards':
            memory = json.dumps(r['cards'], ensure_ascii=False)
        elif item['arm'] == 'rag':
            memory = '\n'.join(c['text'] for c in r['agent_result']['retrieved_chunks'])
        else:
            core = json.loads((directory / 'core_cards.json').read_text())
            memory = json.dumps(core, ensure_ascii=False) + '\n' + '\n'.join(c['text'] for c in r['agent_result']['retrieved_chunks'])
        messages = original_judge_messages(official, case, answer, memory)
        inputs.append({'arm': item['arm'], 'source_run': item['source_run'], 'answer_sha256': base.sha(directory / 'answer.txt'),
            'request': {'model': plan['judge_model'], 'messages': messages, 'temperature': 0,
                'max_tokens': 4096, 'stream': False, 'response_format': {'type': 'json_object'}, 'thinking': {'type': 'disabled'}}})
    return plan, inputs, references, official, case


def validate_judgment(data):
    if not isinstance(data, dict) or not isinstance(data.get('dimensions'), dict) or set(data['dimensions']) != set(DIMENSIONS):
        raise ValueError('缺少四个独立评分维度')
    for name in DIMENSIONS:
        d = data['dimensions'][name]
        if (not isinstance(d, dict) or type(d.get('score')) is not int or d['score'] not in GRADES
                or d.get('grade') not in GRADES.values() or not isinstance(d.get('reasoning'), str) or not d['reasoning'].strip()
                or not isinstance(d.get('evidence'), list) or any(not isinstance(x, str) for x in d['evidence'])
                or (d.get('boundary_case') is not None and not isinstance(d['boundary_case'], str))):
            raise ValueError('评分维度类型/范围不符：' + name)
    h = data.get('hallucination')
    if (not isinstance(h, dict) or type(h.get('detected')) is not bool
            or not isinstance(h.get('reasoning'), str) or not h['reasoning'].strip()
            or any(not isinstance(h.get(k), list) or any(not isinstance(x, str) for x in h[k]) for k in ['claims', 'evidence'])):
        raise ValueError('幻觉否决必须为明确布尔值并附说明')
    if not isinstance(data.get('overall_reasoning'), str) or not data['overall_reasoning'].strip():
        raise ValueError('缺少评分说明')
    info = data.get('required_info_found', {})
    if not isinstance(info, dict):
        raise ValueError('required_info_found必须为对象')
    values = list(info.values())
    if values and not (all(type(v) is bool for v in values) or all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values)):
        raise ValueError('信息项评分须统一为布尔或有限的0—1数字，拒绝隐式混合转换')


def parse_judgment(official, case_id, raw):
    data = strict_json(raw)
    validate_judgment(data)
    evaluator = object.__new__(official.LLMEvaluator)
    result = evaluator._parse_evaluation_response(raw, case_id)
    scores = [data['dimensions'][d]['score'] for d in DIMENSIONS]
    veto = data['hallucination']['detected']
    expected = 0.0 if veto else round(float(sum(Decimal(s - 1) / 3 for s in scores) / 4), 6)
    if (set(result.dimensions) != set(DIMENSIONS) or result.hallucination is None or result.reward != expected
            or result.passed != (not veto and all(s >= 3 for s in scores[:3])) or result.veto_applied != veto):
        raise ValueError('原评分解析与独立门槛/奖励核算不一致')
    return {'raw_judge_json': data, 'official_result': result.model_dump(mode='json'),
        'grade_labels_normalized_by_score': [d for d in DIMENSIONS if data['dimensions'][d]['grade'] != GRADES[data['dimensions'][d]['score']]],
        'all_dimensions_have_evidence': all(data['dimensions'][d]['evidence'] for d in DIMENSIONS),
        'reason_and_evidence_semantics': 'pending source-level review; parsed score is not proof of judge correctness'}


class TransportFailure(Exception):
    pass


def remote_post(payload, config):
    connection = http.client.HTTPSConnection('api.deepseek.com', timeout=config['timeout'])
    try:
        connection.request('POST', config['path'], body=json.dumps(payload, ensure_ascii=False).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + config['key']})
        response = connection.getresponse()
        return response.status, response.read().decode('utf-8')
    finally:
        connection.close()


def judge_once(item, record, save, official, case_id, post):
    if len(record['judge_calls']) >= 3 or any(c['arm'] == item['arm'] for c in record['judge_calls']):
        raise RuntimeError('每条原答案最多一次，总计最多三次，不重试')
    entry = {'arm': item['arm'], 'status': 'started', 'request': deepcopy(item['request'])}
    record['judge_calls'].append(entry)
    record['judge_requests_started'] = len(record['judge_calls'])
    save()
    started = time.perf_counter()
    try:
        try:
            status, raw = post(deepcopy(item['request']))
        except Exception as exc:
            raise TransportFailure('裁判连接未完成：' + type(exc).__name__) from exc
        entry.update(http_status=status, raw_http_body=raw, client_wall_ms=(time.perf_counter() - started) * 1000)
        save()
        if status != 200:
            raise TransportFailure('裁判HTTP状态：' + str(status))
        response = strict_json(raw)
        entry['response'] = response
        save()
        choices = response.get('choices')
        if (not isinstance(response.get('model'), str) or not response['model'].lower().startswith('deepseek')
                or not isinstance(choices, list) or len(choices) != 1 or choices[0].get('finish_reason') != 'stop'):
            raise ValueError('裁判模型身份/结束状态/选择数量异常，不能作为有效评分')
        message = choices[0].get('message', {})
        if message.get('role') != 'assistant' or message.get('tool_calls') or not isinstance(message.get('content'), str):
            raise ValueError('裁判响应正文或角色异常')
        entry['judgment'] = parse_judgment(official, case_id, message['content'])
        entry.update(status='structured_judgment_saved_semantic_review_pending', reported_usage=response.get('usage'))
    except TransportFailure as exc:
        entry.update(status='provider_error', error=str(exc))
        raise
    except Exception as exc:
        entry.update(status='invalid_or_incomplete_judgment', error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        save()
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='只核对来源、构造三份原提示并显示预算，0次请求')
    args = parser.parse_args()
    os.environ.update(PYTHON_DOTENV_DISABLED='1', MAX_RETRIES='1')
    plan, inputs, references, official, case = load_inputs()
    config = load_config(plan)
    print('第七章记忆评估第五步：同一原对话、三条原始答案，DeepSeek独立评分。')
    print('裁判：' + config['model'] + '；最多3次官方DeepSeek请求，每条1次，可能计费；不重跑生成、嵌入或检索。')
    print('仅发送公开合成案例、三条原始答案及相应记忆轨迹；原审阅结论不交给裁判。')
    if args.preview:
        for item in inputs:
            print(item['arm'] + '：两消息独立请求，输入字符数=' + str(sum(len(m['content']) for m in item['request']['messages'])))
        print('INDEPENDENT_JUDGE_PREVIEW：输入与预算已核对，凭据配置存在=' + str(bool(config['key'])) + '，未发出网络请求。')
        return 0
    if not config['key']:
        raise ValueError('已有DEEPSEEK_API_KEY未配置；0次请求，不回退其他裁判')
    directory = ROOT / '本地运行结果' / ('judge_three_' + datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'created_at': datetime.now(CHINA).isoformat(), 'plan': plan,
        'script_sha256': base.sha(Path(__file__)), 'input_plan_sha256': base.sha(PLAN),
        'official_reference_hashes': references, 'judge_endpoint': config['endpoint'], 'judge_model': config['model'],
        'prepared_inputs': inputs, 'judge_calls': [], 'maximum_judge_requests': 3,
        'new_answer_generation_requests': 0, 'new_embedding_forward_batches': 0, 'retrieval_requests': 0,
        'judge_requests_started': 0, 'billed_money': None, 'automatic_retries': 0, 'scope': plan['scope']}

    def save():
        base.dump(directory / 'record.json', redact(record, config['key']))

    def post(payload):
        status, raw = remote_post(payload, config)
        return status, redact(raw, config['key'])

    save()
    print('记录路径：' + str(directory / 'record.json'), flush=True)
    try:
        for item in inputs:
            print('独立评分：' + item['arm'], flush=True)
            try:
                entry = judge_once(item, record, save, official, case.test_id, post)
            except TransportFailure:
                record['not_attempted_after_provider_error'] = [x['arm'] for x in inputs if x['arm'] not in {c['arm'] for c in record['judge_calls']}]
                break
            if 'judgment' in entry:
                result = entry['judgment']['official_result']
                print(json.dumps({'arm': item['arm'], 'scores': {k: v['score'] for k, v in result['dimensions'].items()},
                    'reward': result['reward'], 'passed': result['passed'], 'veto_applied': result['veto_applied']}, ensure_ascii=False), flush=True)
            else:
                print('本条评分未完整，保留原输出，不将无效评分写成任务0分。', flush=True)
        _, fresh_inputs, fresh_refs, _, _ = load_inputs()
        complete = len(record['judge_calls']) == 3 and all('judgment' in c for c in record['judge_calls'])
        record['checks'] = {'sources_and_inputs_unchanged': fresh_inputs == inputs and fresh_refs == references,
            'at_most_three_unique_requests': len(record['judge_calls']) <= 3 and len({c['arm'] for c in record['judge_calls']}) == len(record['judge_calls']),
            'three_structured_judgments_available': complete}
        record['status'] = 'three_judgments_saved_semantic_review_pending' if complete and all(record['checks'].values()) else 'incomplete_judge_campaign'
        print('INDEPENDENT_JUDGE_SAVED：原评分与失败已保存；模型裁判理由仍待来源复核，不代表人工校准或60题完成。', flush=True)
        return 0 if record['status'] == 'three_judgments_saved_semantic_review_pending' else 1
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='incomplete_judge_campaign', error=redact({'type': type(exc).__name__, 'message': str(exc)}, config['key']))
        return 1
    finally:
        save()


if __name__ == '__main__':
    raise SystemExit(main())
