"""只改未证实的完成状态，最多1次DeepSeek评分；Codex撰写对照，不重跑原模型。"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import 本机记忆适配 as base
import RAG本机适配 as rag

ROOT = base.ROOT
PLAN = ROOT / '测试材料/裁判正向对照方案.json'
JUDGE_SCRIPT = Path(__file__).with_name('05_三路结果独立裁判评分.py')


def load_control():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    if (plan['source_arm'] != 'advanced_json_cards' or plan['maximum_judge_requests'] != 1
            or plan['judge_model'] != 'deepseek-flash' or plan['max_output_tokens'] != 4096
            or plan['automatic_retries'] is not False
            or plan['replace_from'] != ' and is already configured for direct deposit'
            or plan['replace_to'] != '. You can set up direct deposit'):
        raise ValueError('对照改动、模型或单次预算不符')
    source = (ROOT / '本地运行结果' / plan['source_judge_run']).resolve()
    if source.parent != (ROOT / '本地运行结果').resolve() or not source.name.startswith('judge_three_'):
        raise ValueError('对照来源越界')
    r = json.loads((source / 'record.json').read_text())
    review = json.loads((source / '核对记录.json').read_text())
    if (base.sha(source / 'record.json') != plan['source_record_sha256']
            or base.sha(source / '核对记录.json') != plan['source_review_sha256']
            or review['source_record_sha256'] != plan['source_record_sha256']
            or not all(r['checks'].values()) or not all(review['checks'].values())
            or base.sha(JUDGE_SCRIPT) != r['script_sha256']):
        raise ValueError('固定原评分或实现变化')
    judge = rag.load(JUDGE_SCRIPT, 'supported_control_judge')
    original_plan, inputs, refs, official, case = judge.load_inputs()
    if original_plan != r['plan'] or inputs != r['prepared_inputs']:
        raise ValueError('原输入变化')
    source_item = next(x for x in original_plan['sources'] if x['arm'] == plan['source_arm'])
    old_run = ROOT / '本地运行结果' / source_item['source_run']
    old = json.loads((old_run / 'record.json').read_text())
    original_answer = old['answer']
    if original_answer.count(plan['replace_from']) != 1:
        raise ValueError('原答案中待改状态表述不唯一')
    control_answer = original_answer.replace(plan['replace_from'], plan['replace_to'], 1)
    # 同一原始记忆轨迹与官方提示，只替换答案，不提供原裁判结论。
    messages = judge.original_judge_messages(official, case, control_answer, json.dumps(old['cards'], ensure_ascii=False))
    original_request = next(x['request'] for x in inputs if x['arm'] == plan['source_arm'])
    if (messages[0] != original_request['messages'][0]
            or original_request['messages'][1]['content'].count(original_answer) != 1
            or messages[1]['content'] != original_request['messages'][1]['content'].replace(original_answer, control_answer, 1)):
        raise ValueError('答案以外的裁判上下文发生变化')
    source_text = '\n'.join(m.content for h in case.conversation_histories for m in h.messages)
    if not all(x in source_text for x in ['4429853327', '123006800', 'First National Bank', 'TechCorp Industries']):
        raise ValueError('正向对照关键事实缺少来源')
    request = deepcopy(original_request)
    request['messages'] = messages
    item = {'arm': 'supported_answer_control', 'source_run': source_item['source_run'], 'request': request}
    baseline = next(c['judgment']['official_result'] for c in r['judge_calls'] if c['arm'] == plan['source_arm'])
    return plan, item, original_answer, control_answer, baseline, refs, judge, official, case


def score_once(item, record, save, judge, official, case_id, post):
    if record['judge_calls'] or record.get('http_requests_started', 0):
        raise RuntimeError('本对照只允许一次请求，不重试')

    def limited(payload):
        if record.get('http_requests_started', 0) >= 1:
            raise RuntimeError('已达一次实际HTTP请求预算')
        record['http_requests_started'] = 1
        save()
        return post(payload)

    return judge.judge_once(item, record, save, official, case_id, limited)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true', help='只核对改动和输入；0次请求')
    args = parser.parse_args()
    os.environ.update(PYTHON_DOTENV_DISABLED='1', MAX_RETRIES='1')
    plan, item, original, control, baseline, refs, judge, official, case = load_control()
    config = judge.load_config(plan)
    print('第七章裁判正向对照：仅将未证实的完成状态改为设置用途。')
    print('最多1次已有DeepSeek官方请求，可能计费；0次原模型生成、检索或嵌入，无重试。')
    print('本条是Codex撰写的受控答案示例，不是qwen修复成绩，也不是人工金标。')
    print('对照答案：' + control)
    if args.preview:
        print('JUDGE_CONTROL_PREVIEW：同一权威原文、原记忆轨迹及参数，仅答案指定短语变化；未发送请求。')
        return 0
    if not config['key']:
        raise ValueError('已有DeepSeek Key未配置；0次请求')
    directory = ROOT / '本地运行结果' / ('judge_positive_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    record = {'status': 'running', 'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'plan': plan, 'script_sha256': base.sha(Path(__file__)),
        'input_plan_sha256': base.sha(PLAN), 'original_judge_script_sha256': base.sha(JUDGE_SCRIPT),
        'original_answer': original, 'control_answer': control, 'prepared_input': item,
        'baseline_result_for_client_comparison_only': baseline, 'official_reference_hashes': refs,
        'judge_calls': [], 'http_requests_started': 0, 'maximum_judge_requests': 1,
        'new_writer_requests': 0, 'new_embedding_forward_batches': 0, 'retrieval_requests': 0,
        'billed_money': None, 'human_calibration_completed': False, 'scope': plan['scope']}

    def save():
        base.dump(directory / 'record.json', judge.redact(record, config['key']))

    def post(payload):
        status, raw = judge.remote_post(payload, config)
        return status, judge.redact(raw, config['key'])

    save()
    print('记录路径：' + str(directory / 'record.json'), flush=True)
    try:
        entry = score_once(item, record, save, judge, official, case.test_id, post)
        if 'judgment' not in entry:
            record['status'] = 'incomplete_control_judgment'
            print('CONTROL_INCOMPLETE：原输出保留，不重试，不将无效评分记作0分。')
            return 1
        result = entry['judgment']['official_result']
        record['comparison'] = {'baseline_veto': baseline['veto_applied'], 'control_veto': result['veto_applied'],
            'baseline_reward': baseline['reward'], 'control_reward': result['reward'],
            'scope': 'Observed wording-control difference; not writer performance gain or human agreement rate.'}
        print(json.dumps({'scores': {k: v['score'] for k, v in result['dimensions'].items()},
            'reward': result['reward'], 'passed': result['passed'], 'veto_applied': result['veto_applied']}, ensure_ascii=False))
        fresh = load_control()
        record['checks'] = {'fixed_input_and_sources_unchanged': fresh[0] == plan and fresh[1] == item,
            'only_one_request_started': record['http_requests_started'] == len(record['judge_calls']) == 1}
        record['status'] = 'positive_control_saved_semantic_review_pending' if all(record['checks'].values()) else 'failed_source_checks'
        print('JUDGE_CONTROL_SAVED：正向对照评分已保存；理由仍需复核，一条对照不能证明裁判已校准。')
        return 0 if record['status'] == 'positive_control_saved_semantic_review_pending' else 1
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='incomplete_control_judgment', error=judge.redact({'type': type(exc).__name__, 'message': str(exc)}, config['key']))
        print('CONTROL_INCOMPLETE：请求未完成，保留结果，不重试。')
        return 1
    finally:
        save()


if __name__ == '__main__':
    raise SystemExit(main())
