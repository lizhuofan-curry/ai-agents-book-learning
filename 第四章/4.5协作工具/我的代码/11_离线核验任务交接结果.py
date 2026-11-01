"""只读回放交接记录，按实际交接材料独立核算决定；不请求模型，不修改原记录。"""
import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CHINA = timezone(timedelta(hours=8))
RUNS = ['handoff_within_window_20261101_150027_129403',
        'handoff_overdue_20261101_150332_851732']
CONTEXT_TAG = '\n\n[FROM_MAIN_AGENT：人工整理的背景]\n'
POLICY = r'虚构商家规则：申请日减购买日的自然日差不超过([0-9]+)天，且商品未使用，才符合退款条件。两项须同时满足；与金额无关。'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract_materials(record, arm_name):
    """从接收方实际请求取材料，核对记录中的材料副本，避免只看case名称。"""
    arm = record['arms'][arm_name]
    messages = arm['request']['messages']
    require([m['role'] for m in messages] == ['system', 'user'], '预期独立的system/user消息')
    content = messages[1]['content']
    task = record['task']
    task_match = re.fullmatch(r'判断订单([A-Z0-9-]+)是否符合给定商家的退款条件，并说明依据。', task)
    require(task_match is not None, '任务不在本教学核验器支持的格式内')
    if arm_name == 'title_only':
        require(content == task, '仅标题组包含非预期背景')
        return task_match[1], []
    require(arm_name == 'manual_context', '不支持的实验组')
    require(content.startswith(task + CONTEXT_TAG), '未找到约定的交接背景')
    materials = json.loads(content[len(task + CONTEXT_TAG):])
    require(materials == record['materials'], '实际请求材料与记录副本不一致')
    return task_match[1], materials


def compute_rule(order_id, materials):
    """仅支持明确的教学语法；未知规则/缺失/冲突拒绝核算，不猜测自然语言。"""
    require(isinstance(materials, list) and len(materials) == 4, '需要四条明确的规则与订单材料')
    by_id = {}
    for item in materials:
        require(isinstance(item, dict) and set(item) == {'id', 'text'}, '材料结构异常')
        require(isinstance(item['id'], str) and isinstance(item['text'], str), '材料字段类型异常')
        require(item['id'] not in by_id, '材料编号重复')
        by_id[item['id']] = item['text']
    require(set(by_id) == {'R1', 'O1', 'O2', 'O3'}, '材料编号缺失或未知')
    policy = re.fullmatch(POLICY, by_id['R1'])
    require(policy is not None, '不支持的规则文本，不能套用默认14天')
    limit = int(policy[1])
    require(0 <= limit <= 365, '期限超出本教学核验器支持范围')
    oid = re.escape(order_id)
    purchase = re.fullmatch(r'订单'+oid+r'，购买日期为([0-9]{4}-[0-9]{2}-[0-9]{2})。', by_id['O1'])
    state = re.fullmatch(r'订单'+oid+r'的商品(未使用|已使用)。', by_id['O2'])
    application = re.fullmatch(r'本次退款申请日期为([0-9]{4}-[0-9]{2}-[0-9]{2})。以该日期判断，不使用电脑当前日期。', by_id['O3'])
    require(purchase is not None and state is not None and application is not None,
            '日期、订单号或商品状态格式不受支持')
    purchased_at = date.fromisoformat(purchase[1])
    applied_at = date.fromisoformat(application[1])
    days = (applied_at - purchased_at).days
    require(days >= 0, '申请日早于购买日，需人工检查材料')
    unused = state[1] == '未使用'
    expected = 'eligible' if days <= limit and unused else 'ineligible'
    if days > limit:
        required_ids = {'R1', 'O1', 'O3'}
    elif not unused:
        required_ids = {'R1', 'O2'}
    else:
        required_ids = {'R1', 'O1', 'O2', 'O3'}
    return {'purchase_date': purchase[1], 'application_date': application[1],
            'elapsed_days': days, 'limit_days': limit, 'unused': unused,
            'expected_decision': expected, 'required_evidence_ids': sorted(required_ids)}


def read_answer(arm):
    """对照原始HTTP、解析响应、保存回答；这些一致也不证明任务正确。"""
    require(arm.get('http_status') == 200, 'HTTP未成功')
    raw = json.loads(arm['raw_http_body'])
    require(raw == arm['response'], '原始HTTP与保存响应不一致')
    require(raw.get('model') == arm['request']['model'] and raw.get('done') is True
            and raw.get('done_reason') == 'stop', '模型名或结束状态不符合要求')
    message = raw['message']
    require(message.get('role') == 'assistant' and not message.get('tool_calls'), '回答角色或工具调用异常')
    answer = json.loads(message['content'])
    require(answer == arm['answer'], '保存回答与原始回答不一致')
    require(isinstance(answer, dict) and set(answer) == {'decision', 'reason', 'evidence_ids', 'missing_information'}, '回答字段不正确')
    require(answer['decision'] in ('need_info', 'eligible', 'ineligible'), '决定枚举非法')
    require(isinstance(answer['reason'], str) and bool(answer['reason'].strip()), '理由为空')
    for key in ('evidence_ids', 'missing_information'):
        require(isinstance(answer[key], list) and all(isinstance(v, str) and bool(v.strip()) for v in answer[key]), key+'格式不正确')
    return answer


def verify_arm(record, name):
    # 不实现任何退款/文件执行工具；决定核对通过后，文字理由仍待人工审阅。
    result = {'gate': 'blocked_invalid_record_or_materials', 'can_execute_action': False,
              'can_forward_full_result': False,
              'free_text_reason_review': 'not_automated; human review required',
              'model_output_changed': False}
    try:
        order_id, materials = extract_materials(record, name)
        answer = read_answer(record['arms'][name])
        result['model_decision'] = answer['decision']
        allowed = {m['id'] for m in materials}
        require(set(answer['evidence_ids']) <= allowed, '引用了未交接的材料')
        if not materials:
            if answer['decision'] == 'need_info' and answer['missing_information']:
                result['gate'] = 'needs_more_information_no_action'
            else:
                result.update(gate='blocked_unsupported_decision', error='只有标题，不能确定退款资格')
            return result
        facts = compute_rule(order_id, materials)
        result['rule_check'] = facts
        if answer['decision'] != facts['expected_decision']:
            result.update(gate='blocked_decision_mismatch', error='模型决定与交接材料独立核算结果不一致')
        elif answer['missing_information']:
            result.update(gate='blocked_answer_inconsistency', error='确定决定同时声明材料缺失，需人工核对')
        elif not set(facts['required_evidence_ids']) <= set(answer['evidence_ids']):
            result.update(gate='blocked_missing_supporting_citations', error='缺少支持此决定的必要材料引用')
        else:
            result['gate'] = 'decision_verified_reason_review_pending'
        return result
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        result['error'] = str(exc)
        return result


def verify_record(path):
    blob = path.read_bytes()
    record = json.loads(blob)
    result = {'source_record': str(path.resolve()), 'source_sha256': hashlib.sha256(blob).hexdigest(),
              'arms': {name: verify_arm(record, name) for name in ('title_only', 'manual_context')}}
    require(path.read_bytes() == blob, '核验期间源记录发生变化')
    result['source_unchanged'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, action='append', help='指定记录，可重复；默认回放现有两例')
    args = parser.parse_args()
    paths = args.record or [ROOT/'运行结果'/name/'record.json' for name in RUNS]
    print('任务交接第二步：只读回放；0次模型请求，不用启动Ollama。')
    reports = []
    try:
        for path in paths:
            report = verify_record(path)
            reports.append(report)
            print('\n记录：'+path.parent.name)
            for name, item in report['arms'].items():
                print(name+' → '+item['gate'])
                if 'rule_check' in item:
                    facts = item['rule_check']
                    print(f"  日差={facts['elapsed_days']}，期限={facts['limit_days']}；规则期望={facts['expected_decision']}，模型决定={item['model_decision']}")
                if 'error' in item:
                    print('  原因：'+item['error'])
        directory = ROOT/'运行结果'/('handoff_verification_'+datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_%f'))
        directory.mkdir(parents=True, exist_ok=False)
        output = directory/'verification.json'
        output.write_text(json.dumps({
            'created_at': datetime.now(CHINA).isoformat(),
            'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reports': reports, 'new_model_requests': 0, 'actions_executed': 0,
            'scope': 'fixed teaching grammar; independently check structured decision against provided rule and dates; free-text reason semantics not automatically verified; no model repair',
        }, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print('\nHANDOFF_VERIFICATION_SAVED：阻断项也会保存；保存成功不表示每组通过。')
        print('核验报告：'+str(output))
        # 发现并正确报告历史失败是回放成功；每组是否通过必须看gate。
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print('核验未完成：'+str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
