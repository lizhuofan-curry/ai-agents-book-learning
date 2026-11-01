"""独立接收方只看已核对的模型交接原稿；按原始任务规则核验决定，不改稿不重试。"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
GENERATION = ROOT/'运行结果/generated_handoff_20261101_151431_417834/record.json'
REVIEW = GENERATION.parent/'交接稿核对.json'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transport = load(ROOT/'我的代码/08_任务交接信息对照.py', 'generated_receiver_transport')
verifier = load(ROOT/'我的代码/11_离线核验任务交接结果.py', 'generated_receiver_verifier')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    generated = json.loads(GENERATION.read_text())
    review = json.loads(REVIEW.read_text())
    verifier.require(review['source_record_sha256'] == sha(GENERATION), '核对意见不对应当前交接稿')
    verifier.require(review['review_result'] == 'acceptable_for_this_synthetic_receiver_trial', '交接稿尚未完成本例语义核对')
    call = generated['generation_call']
    raw = json.loads(call['raw_http_body'])
    verifier.require(raw == call['response'] and json.loads(raw['message']['content']) == generated['handoff'], '原始响应与交接稿不一致')
    verifier.require(raw.get('done') is True and raw.get('done_reason') == 'stop', '整理模型未正常完成')
    # base_record 是冻结的来源记录，可能包含另一台机器的绝对路径。
    # 只核对已审阅案例的项目内路径；实际读取当前 checkout，并继续核对原始哈希。
    base_relative = Path('运行结果/handoff_within_window_20261101_150027_129403/record.json')
    expected_parts = ('第四章', '4.5协作工具', *base_relative.parts)
    recorded_parts = PurePosixPath(generated['base_record'].replace('\\', '/')).parts
    verifier.require(recorded_parts[-len(expected_parts):] == expected_parts and '..' not in recorded_parts,
                     '本入口仅用于已经核对的默认案例')
    base_path = (ROOT/base_relative).resolve()
    verifier.require(base_path.is_relative_to(ROOT.resolve()), '原始任务记录路径超出当前案例目录')
    verifier.require(sha(base_path) == generated['base_sha256'], '原始任务记录已变化')
    base = json.loads(base_path.read_text())
    # 核算只在客户端使用；不把原始材料、旧回答、期望决定或核对意见发给接收方。
    _, original_materials = verifier.extract_materials(base, 'manual_context')
    facts = verifier.compute_rule('DEMO-001', original_materials)
    return generated, facts


def make_payload(handoff):
    request = transport.make_payload([])
    request['messages'][1]['content'] = (
        '请完成下列模型整理的交接任务。你只能看到这份交接稿。'
        'evidence_ids使用材料H编号；source_ids是溯源指针，其原始S记录没有交给你。\n'
        '[FROM_MAIN_AGENT：模型整理的原稿，未人工改写]\n'+json.dumps(handoff, ensure_ascii=False))
    return request


def check_decision(answer, handoff, facts):
    """基于原始任务规则核算；不将整理稿中的改写当作新的业务规则。"""
    result = {'rule_check_against_original_source': deepcopy(facts), 'model_decision': answer['decision'],
              'can_execute_action': False, 'can_forward_full_result': False,
              'reason_semantics': 'pending human review', 'model_output_changed': False}
    cited = set(answer['evidence_ids'])
    by_id = {item['id']: item for item in handoff['materials']}
    covered = {sid for hid in cited for sid in by_id[hid]['source_ids']}
    needed = {'R1':'S2','O1':'S3','O2':'S4','O3':'S5'}
    required = {needed[x] for x in facts['required_evidence_ids']}
    result['cited_original_source_ids_via_handoff'] = sorted(covered)
    if answer['decision'] != facts['expected_decision']:
        result['gate'] = 'blocked_decision_mismatch'
    elif answer['missing_information']:
        result['gate'] = 'blocked_answer_inconsistency'
    elif not required <= covered:
        result['gate'] = 'blocked_missing_supporting_citations'
    else:
        result['gate'] = 'decision_verified_reason_review_pending'
    # 来源编号覆盖不证明改写或回答的语义支持关系，仍保留人工审阅。
    return result


def receive(record, save, model_call=transport.local_call):
    entry = {'request': make_payload(record['handoff_as_received']), 'status': 'started'}
    record['receiver_call'] = entry
    save()
    try:
        raw = model_call(deepcopy(entry['request']), entry, save)
        entry['response'] = raw
        save()
        allowed = [x['id'] for x in record['handoff_as_received']['materials']]
        answer = transport.inspect_response(raw, allowed)
        record['receiver_answer'] = answer
        record['decision_verification'] = check_decision(answer, record['handoff_as_received'], record['original_rule_facts'])
        entry.update(status='completed', prompt_tokens_reported=raw.get('prompt_eval_count'), output_tokens_reported=raw.get('eval_count'))
        record['status'] = 'receiver_answer_saved_review_pending'
        print(json.dumps(answer, ensure_ascii=False, indent=2), flush=True)
        print('决定核验：'+json.dumps(record['decision_verification'],ensure_ascii=False), flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        entry.update(status='incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        record['status'] = 'incomplete'
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    generated, facts = prepare()
    print('任务交接第四步：独立接收方只看已核对的模型交接原稿。')
    print('最多1次本机qwen3.5:9b请求；不重生成交接稿、不重新跑旧基线。')
    print('只把交接原稿交给模型，原始S记录和规则核算结果仅供客户端核验。')
    if args.preview:
        print('GENERATED_RECEIVER_PREVIEW_ONLY：来源及核对意见检查通过，无模型请求、无结果目录。')
        return 0
    directory = ROOT/'运行结果'/('generated_receiver_'+datetime.now(transport.CHINA).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'record.json'
    hashes = {'generation': sha(GENERATION), 'review': sha(REVIEW), 'base': generated['base_sha256']}
    record = {'created_at': datetime.now(transport.CHINA).isoformat(), 'status': 'running',
              'generation_record': str(GENERATION), 'review_record': str(REVIEW), 'input_sha256': hashes,
              'handoff_as_received': deepcopy(generated['handoff']), 'original_rule_facts': facts,
              'script_sha256': {p.name: sha(p) for p in (Path(__file__), Path(transport.__file__), Path(verifier.__file__))},
              'maximum_new_model_requests': 1, 'generation_requests_this_run': 0, 'automatic_retries': 0,
              'cloud_requests': 0, 'actions_executed': 0,
              'scope': 'one reviewed, known synthetic handoff; human-selected checkpoint; independent receiver request, not autonomous delegation or controlled architecture comparison'}
    def save():
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        tmp.replace(path)
    save()
    print('记录路径：'+str(path), flush=True)
    try:
        receive(record, save)
        verifier.require(sha(GENERATION)==hashes['generation'] and sha(REVIEW)==hashes['review'], '输入记录发生变化')
        print('GENERATED_RECEIVER_SAVED：回答与核验已保存；失败原样保留，理由待人工核对。')
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='incomplete', error={'type': type(exc).__name__, 'message': str(exc)})
        print('本步未完成，不重试：'+str(exc))
        return 1
    finally:
        save()
        print('记录路径：'+str(path), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
