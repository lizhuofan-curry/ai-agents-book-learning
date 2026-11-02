"""可变案例/卡片数量的受限Cards支架；旧单题适配与结果只读。"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time

import 本机记忆适配 as base
import RAG本机适配 as rag

ROOT, PROJECT = base.ROOT, base.PROJECT
PLAN = ROOT / '测试材料/跨层三案例JSONCards方案.json'
CHINA = timezone(timedelta(hours=8))


def strict_json(text):
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ValueError('JSON重复字段：' + k)
            obj[k] = v
        return obj
    def reject(value):
        raise ValueError('JSON含非有限数值：' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def persist(path, record):
    payload = deepcopy(record)
    payload.pop('checkpoint_sha256', None)
    payload['checkpoint_sha256'] = digest(payload)
    base.dump(path, payload)  # 单文件原子替换；校验和只防意外损坏，不是签名。


def load_checkpoint(path):
    obj = strict_json(path.read_text())
    checksum = obj.pop('checkpoint_sha256')
    if digest(obj) != checksum:
        raise ValueError('断点内容校验失败，不自动修复或重发')
    return obj


def implementation_hashes():
    names = ['09_跨层三案例JSONCards.py', '批量JSONCards适配.py', '本机记忆适配.py',
        'RAG本机适配.py', '08_Qwen提取输入长度核验.py', '07_六十案例离线预检.py',
        '01_官方案例与评分规则离线检查.py']
    return {n: base.sha(Path(__file__).with_name(n)) for n in names}


def verify_inputs():
    plan = strict_json(PLAN.read_text())
    fixed = {'protocol_version': 'cards-cross-layer-smoke-v1', 'model': 'qwen3.5:9b',
        'endpoint': 'http://127.0.0.1:11434/api/chat', 'num_ctx': 16384,
        'num_predict': {'card_build': 4096, 'cards_answer': 1024}, 'temperature': 0, 'seed': 42,
        'think': False, 'engineering_headroom_tokens': 256, 'maximum_model_requests': 6,
        'maximum_per_case': 2, 'maximum_per_stage': 1, 'judge_requests': 0,
        'retrieval_requests': 0, 'embedding_forward_batches': 0, 'automatic_retries': False}
    if any(type(plan.get(k)) is not type(v) or plan[k] != v for k, v in fixed.items()):
        raise ValueError('固定三案例、参数或请求预算变化')
    expected = ['layer1_02_insurance_claim', 'layer2_01_multiple_vehicles', 'layer3_01_travel_coordination']
    if [c['case_id'] for c in plan['cases']] != expected:
        raise ValueError('仅允许预先固定的三条新案例，不回跑旧银行题或扩展60题')
    length_guard = rag.load(Path(__file__).with_name('08_Qwen提取输入长度核验.py'), 'batch_length_guard')
    length_plan, cache, _, _, assets = length_guard.verify_inputs()
    parent = (ROOT / '本地运行结果').resolve()
    source = (parent / plan['length_run']).resolve()
    if source.parent != parent or not source.name.startswith('qwen_length_'):
        raise ValueError('本人08来源路径越界')
    length = strict_json((source / 'record.json').read_text())
    review = strict_json((source / '核对记录.json').read_text())
    if (base.sha(source / 'record.json') != plan['length_record_sha256']
            or base.sha(source / '核对记录.json') != plan['length_review_sha256']
            or review['source_record_sha256'] != plan['length_record_sha256']
            or not all(length['checks'].values()) or not all(review['checks'].values())
            or length['script_sha256'] != base.sha(Path(length_guard.__file__))):
        raise ValueError('本人08来源或审阅变化')
    guard = rag.load(Path(__file__).with_name('01_官方案例与评分规则离线检查.py'), 'batch_reference_guard')
    refs = guard.verify_reference()
    for item in plan['cases']:
        path = (ROOT / '官方资料/upstream' / item['source_path']).resolve()
        if not path.is_relative_to((ROOT / '官方资料/upstream').resolve()) or base.sha(path) != item['source_sha256']:
            raise ValueError('案例来源变化或越界')
    provenance = {'plan_sha256': base.sha(PLAN), 'implementation_sha256': implementation_hashes(),
        'official_reference_hashes': refs, 'installed_manifest_sha256': length_plan['manifest_sha256'],
        'tokenizer_assets_sha256': assets, 'length_record_sha256': plan['length_record_sha256']}
    return plan, cache, provenance


def fresh_record(plan, provenance):
    return {'created_at': datetime.now(CHINA).isoformat(), 'status': 'prepared', 'plan': deepcopy(plan),
        'provenance': deepcopy(provenance), 'cases': [], 'network_audit': {'allowed_events': [], 'blocked_events': []},
        'generation_request_cap': 6, 'judge_requests': 0, 'retrieval_requests': 0,
        'embedding_forward_batches': 0, 'external_network_requests': 0,
        'semantic_review': 'pending source-level review; no judge score or task pass inferred',
        'full_sixty_or_three_arm_runner_completed': False}


def request_count(record):
    # 已持久化的started即占预算，即使未知是否到达服务端，也绝不退还。
    return sum(len(c['model_calls']) for c in record['cases'])


def check_resume(record, plan, provenance, directory):
    if record['plan'] != plan or record['provenance'] != provenance:
        raise ValueError('断点方案、源码、分词资产或官方资料变化，不跨协议恢复')
    ids = [c['case_id'] for c in record['cases']]
    order = [c['case_id'] for c in plan['cases']]
    if ids != order[:len(ids)] or request_count(record) > 6:
        raise ValueError('断点案例顺序或请求上限异常')
    for row in record['cases']:
        stages = [c['stage'] for c in row['model_calls']]
        if stages not in [[], ['card_build'], ['card_build', 'cards_answer']]:
            raise ValueError('断点阶段重复或乱序')
        for name, expected in row.get('artifacts_sha256', {}).items():
            if name not in {'cards.json', 'answer.txt'} or base.sha(directory / row['case_id'] / name) != expected:
                raise ValueError('断点产物变化，不自动覆盖')


class BatchChat:
    def __init__(self, official, case, row, record, plan, tokenizer, save, provenance_guard, post=base.local_post):
        self.official, self.case, self.row, self.record = official, case, row, record
        self.plan, self.tokenizer, self.save, self.provenance_guard, self.post = plan, tokenizer, save, provenance_guard, post
        self.stage = 'card_build'

    def complete(self, messages, tools=None, tool_choice=None, json_object=False):
        stage = self.stage
        calls = self.row['model_calls']
        try:
            if (stage not in {'card_build', 'cards_answer'} or any(x['stage'] == stage for x in calls)
                    or len(calls) >= 2 or request_count(self.record) >= 6):
                raise ValueError('每阶段1次、每题2次、全批6次预算；不重试')
            if tools or tool_choice is not None or json_object != (stage == 'card_build'):
                raise ValueError('本批只允许JSON提取与独立无工具回答')
            if stage == 'card_build':
                if (len(messages) != 1 or messages[0]['role'] != 'user'
                        or hashlib.sha256(messages[0]['content'].encode()).hexdigest() != self.row['card_prompt_sha256']):
                    raise ValueError('提取输入须为冻结原CardBuilder提示，不加入问题或裁判资料')
            else:
                expected = [{'role': 'system', 'content': self.official.SYSTEM_RULES + '\nResident Advanced JSON Cards:\n'
                    + json.dumps(self.row['cards'], ensure_ascii=False)}, {'role': 'user', 'content': self.case.user_question}]
                if messages != expected:
                    raise ValueError('回答必须仅接收本题原卡片与问题的独立两消息上下文')
            # 每次请求前复核当前清单/源码/来源，随后检查动态生成的实际消息。
            self.provenance_guard()
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                tokenize=True, enable_thinking=False)
            nout = self.plan['num_predict'][stage]
            length = {'local_input_tokens': len(ids), 'output_reserve': nout,
                'engineering_headroom_tokens': 256, 'required_total': len(ids) + nout + 256, 'num_ctx': 16384}
            self.row.setdefault('length_checks', []).append({'stage': stage, **length})
            self.save()
            if length['required_total'] > 16384:
                raise ValueError('实际动态上下文超限；不截断、不发送、不重试')
            payload = {'model': base.MODEL, 'messages': deepcopy(messages), 'stream': False, 'think': False,
                'options': {'num_ctx': 16384, 'num_predict': nout, 'temperature': 0, 'seed': 42}}
            if json_object:
                payload['format'] = 'json'
            entry = {'stage': stage, 'status': 'started', 'request': payload, 'length_check': length}
            calls.append(entry)
            self.save()  # 发HTTP之前落盘；中断/超时也占预算，禁止不确定重发。
            print(self.case.test_id + ' → ' + stage + '，本批已占请求预算 ' + str(request_count(self.record)) + '/6', flush=True)
            started = time.perf_counter()
            status, raw = self.post(deepcopy(payload))
            entry.update(http_status=status, raw_http_body=raw, client_wall_ms=(time.perf_counter() - started) * 1000)
            self.save()
            if status != 200:
                raise ValueError('本机HTTP失败：' + str(status))
            response = strict_json(raw)
            entry['response'] = response
            self.save()
            msg = response.get('message', {})
            if (response.get('model') != base.MODEL or response.get('done') is not True
                    or response.get('done_reason') != 'stop' or msg.get('role') != 'assistant'
                    or msg.get('tool_calls') or not isinstance(msg.get('content'), str) or not msg['content'].strip()):
                raise ValueError('异常、截断、空回答或意外工具调用；原响应保留')
            pin, pout = response.get('prompt_eval_count'), response.get('eval_count')
            if any(type(v) is not int or v < 0 for v in [pin, pout]) or pout > nout:
                raise ValueError('用量缺失/类型错误/超限，不填0')
            if pin != len(ids):
                raise ValueError('无工具模板的本地计数与实际请求报告不一致，停止本题后续阶段')
            if json_object:
                obj = strict_json(msg['content'])
                if not isinstance(obj, dict) or set(obj) != {'cards'}:
                    raise ValueError('提取须为仅含cards的JSON对象')
                base.validate_cards(obj['cards'], self.case)
            entry.update(status='completed', reported_prompt_tokens=pin, reported_output_tokens=pout,
                cached_input_tokens=None, local_billed_money=None)
            self.save()
            return self.official.ChatTurn(msg['content'], [], self.official.Usage(input_tokens=pin,
                output_tokens=pout, unpriced_tokens=pin+pout, unpriced_requests=1), entry['client_wall_ms'],
                {'role': 'assistant', 'content': msg['content']})
        except Exception as exc:
            if 'entry' in locals():
                entry.update(status='failed_or_incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
                self.save()
            # 越过官方CardBuilder的except Exception，禁止内置重试。
            raise base.StopPipeline(str(exc)) from exc


def run_cases(official, selected, record, plan, tokenizer, directory, save, provenance_guard, post=base.local_post):
    by_id = {c.test_id: c for c in selected}
    touched = {r['case_id'] for r in record['cases']}
    for row in record['cases']:
        if row['status'] == 'started':
            row.update(status='interrupted_not_retried', recovery_note='Prior case start retained; no additional request for this case.')
    save()
    for spec in plan['cases']:
        if spec['case_id'] in touched:
            continue
        case = by_id[spec['case_id']]
        row = {**deepcopy(spec), 'status': 'started', 'model_calls': [], 'semantic_review': 'pending'}
        record['cases'].append(row)
        save()  # 整题先占位；恢复不会重新提取或补发已开始题。
        target = directory / case.test_id
        target.mkdir(exist_ok=False)
        try:
            chat = BatchChat(official, case, row, record, plan, tokenizer, save, provenance_guard, post)
            cards, _, _ = official.CardBuilder(chat).build(case)
            row['cards'] = cards
            row['card_structure_checks'] = base.validate_cards(cards, case)
            base.dump(target / 'cards.json', {'cards': cards})
            save()
            chat.stage = 'cards_answer'
            result = official.MemoryAgent(chat).cards_only(case.user_question, cards)
            row['answer'] = result.answer
            row['agent_result'] = {'steps': result.steps, 'tool_calls': result.tool_calls, 'trace': result.trace}
            (target / 'answer.txt').write_text(result.answer, encoding='utf-8')
            row['checks'] = {'two_completed_calls': len(row['model_calls']) == 2 and all(c['status'] == 'completed' for c in row['model_calls']),
                'independent_answer_no_tools': result.steps == 1 and result.tool_calls == 0,
                'actual_messages_fit_without_truncation': all(x['required_total'] <= 16384 for x in row['length_checks']),
                'saved_cards_match': strict_json((target / 'cards.json').read_text()) == {'cards': cards},
                'saved_answer_matches': (target / 'answer.txt').read_text() == result.answer}
            if not all(row['checks'].values()):
                raise ValueError('执行一致性检查失败')
            row['status'] = 'answer_saved_semantic_review_pending'
            print('回答：' + result.answer, flush=True)
        except (Exception, base.StopPipeline) as exc:
            row.update(status='failed_or_incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
            print('本题失败原样保留，不重试：' + str(exc), flush=True)
        finally:
            row['artifacts_sha256'] = {p.name: base.sha(p) for p in target.iterdir() if p.name in {'cards.json', 'answer.txt'} and p.is_file()}
            save()
    summarize(record)
    save()


def summarize(record):
    calls = [x for row in record['cases'] for x in row['model_calls']]
    def reported(key):
        values = [c.get('response', {}).get(key) for c in calls]
        good = [v for v in values if type(v) is int and v >= 0]
        return {'reported_sum': sum(good), 'calls_with_valid_count': len(good), 'unknown_calls': len(calls)-len(good)}
    complete = sum(c['status'] == 'answer_saved_semantic_review_pending' for c in record['cases'])
    record['summary'] = {'cases_touched': len(record['cases']), 'answers_saved': complete,
        'failed_or_interrupted_cases': len(record['cases']) - complete, 'reserved_model_requests': len(calls),
        'completed_model_requests': sum(c['status'] == 'completed' for c in calls),
        'prompt_tokens': reported('prompt_eval_count'), 'output_tokens': reported('eval_count'),
        'cached_input_tokens': None, 'local_billed_money': None, 'formal_task_pass_rate': None}
    record['status'] = 'three_case_cards_saved_review_pending' if complete == 3 else 'three_case_cards_incomplete_failures_preserved'
