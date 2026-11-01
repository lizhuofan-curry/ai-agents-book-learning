"""核验器的边界、拒绝路径及原始证据保持检查；不请求模型。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'handoff_verifier', ROOT/'第四章/4.5协作工具/我的代码/11_离线核验任务交接结果.py')
step = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(step)


def fixture(purchase='2026-10-20', application='2026-11-01', decision='eligible', used=False):
    materials = [
        {'id': 'R1', 'text': '虚构商家规则：申请日减购买日的自然日差不超过14天，且商品未使用，才符合退款条件。两项须同时满足；与金额无关。'},
        {'id': 'O1', 'text': f'订单DEMO-001，购买日期为{purchase}。'},
        {'id': 'O2', 'text': '订单DEMO-001的商品'+('已使用。' if used else '未使用。')},
        {'id': 'O3', 'text': f'本次退款申请日期为{application}。以该日期判断，不使用电脑当前日期。'},
    ]
    task = '判断订单DEMO-001是否符合给定商家的退款条件，并说明依据。'
    def arm(context, d):
        answer = {'decision': d, 'reason': '合成测试理由',
                  'evidence_ids': ['R1','O1','O2','O3'] if context else [],
                  'missing_information': [] if context else ['规则及订单事实']}
        raw = {'model': 'synthetic-local', 'done': True, 'done_reason': 'stop',
               'message': {'role': 'assistant', 'content': json.dumps(answer)}}
        return {'http_status': 200, 'raw_http_body': json.dumps(raw), 'response': raw, 'answer': answer,
                'request': {'model': 'synthetic-local', 'messages': [
                    {'role': 'system', 'content': '合成教学要求'},
                    {'role': 'user', 'content': task+(step.CONTEXT_TAG+json.dumps(materials,ensure_ascii=False) if context else '')}]}}
    return {'case': 'deliberately_misleading_label', 'task': task, 'materials': materials,
            'arms': {'title_only': arm(False, 'need_info'), 'manual_context': arm(True, decision)}}


def update_materials(record):
    content = record['task'] + step.CONTEXT_TAG + json.dumps(record['materials'], ensure_ascii=False)
    record['arms']['manual_context']['request']['messages'][1]['content'] = content


def change_answer(record, **changes):
    arm = record['arms']['manual_context']
    arm['answer'].update(changes)
    arm['response']['message']['content'] = json.dumps(arm['answer'], ensure_ascii=False)
    arm['raw_http_body'] = json.dumps(arm['response'], ensure_ascii=False)


class HandoffVerifierChecks(unittest.TestCase):
    def test_real_saved_failure_caught_without_altering_record(self):
        paths = [step.ROOT/'运行结果'/name/'record.json' for name in step.RUNS]
        snapshots = [p.read_bytes() for p in paths]
        results = [step.verify_record(p) for p in paths]
        self.assertEqual(results[0]['arms']['manual_context']['gate'], 'decision_verified_reason_review_pending')
        self.assertEqual(results[1]['arms']['manual_context']['gate'], 'blocked_decision_mismatch')
        self.assertEqual(results[1]['arms']['manual_context']['rule_check']['expected_decision'], 'ineligible')
        self.assertEqual([p.read_bytes() for p in paths], snapshots)

    def test_fourteen_included_fifteen_excluded(self):
        for purchase, decision, days in [('2026-10-18','eligible',14), ('2026-10-17','ineligible',15)]:
            result = step.verify_arm(fixture(purchase=purchase, decision=decision), 'manual_context')
            self.assertEqual(result['gate'], 'decision_verified_reason_review_pending')
            self.assertEqual(result['rule_check']['elapsed_days'], days)

    def test_leap_day_and_used_item(self):
        result = step.verify_arm(fixture(purchase='2024-02-28',application='2024-03-01'), 'manual_context')
        self.assertEqual(result['rule_check']['elapsed_days'], 2)
        result = step.verify_arm(fixture(used=True, decision='ineligible'), 'manual_context')
        self.assertEqual(result['gate'], 'decision_verified_reason_review_pending')

    def test_future_and_invalid_dates_block(self):
        for purchase in ['2026-11-02', '2026-02-30']:
            self.assertEqual(step.verify_arm(fixture(purchase=purchase), 'manual_context')['gate'], 'blocked_invalid_record_or_materials')

    def test_unknown_rule_not_defaulted(self):
        record = fixture()
        record['materials'][0]['text'] = '任意日期都可以退款。'
        update_materials(record)
        self.assertEqual(step.verify_arm(record, 'manual_context')['gate'], 'blocked_invalid_record_or_materials')

    def test_limit_read_from_material_not_fixed_case_name(self):
        record = fixture(purchase='2026-10-10')
        record['materials'][0]['text'] = record['materials'][0]['text'].replace('14天', '30天')
        update_materials(record)
        result = step.verify_arm(record, 'manual_context')
        self.assertEqual(result['rule_check']['limit_days'], 30)
        self.assertEqual(result['gate'], 'decision_verified_reason_review_pending')

    def test_missing_duplicate_and_wrong_order_materials(self):
        for kind in ['missing', 'duplicate', 'wrong_order']:
            record = fixture()
            if kind == 'missing':
                record['materials'].pop()
            elif kind == 'duplicate':
                record['materials'][2]['id'] = 'O1'
            else:
                record['materials'][1]['text'] = '订单OTHER-001，购买日期为2026-10-20。'
            update_materials(record)
            self.assertEqual(step.verify_arm(record, 'manual_context')['gate'], 'blocked_invalid_record_or_materials')

    def test_request_material_copy_mismatch(self):
        record = fixture()
        record['materials'][1]['text'] = '订单DEMO-001，购买日期为2026-10-10。'
        self.assertEqual(step.verify_arm(record, 'manual_context')['gate'], 'blocked_invalid_record_or_materials')

    def test_saved_answer_tampering_caught(self):
        record = fixture()
        record['arms']['manual_context']['answer']['decision'] = 'ineligible'
        self.assertEqual(step.verify_arm(record, 'manual_context')['gate'], 'blocked_invalid_record_or_materials')

    def test_missing_and_unseen_citations(self):
        record = fixture()
        change_answer(record, evidence_ids=['R1','O1','O3'])
        self.assertEqual(step.verify_arm(record,'manual_context')['gate'], 'blocked_missing_supporting_citations')
        change_answer(record, evidence_ids=['UNKNOWN'])
        self.assertEqual(step.verify_arm(record,'manual_context')['gate'], 'blocked_invalid_record_or_materials')

    def test_no_actions_or_free_text_semantic_claim(self):
        record = fixture()
        change_answer(record, reason='不符合退款条件。')
        result = step.verify_arm(record, 'manual_context')
        self.assertEqual(result['gate'], 'decision_verified_reason_review_pending')
        self.assertFalse(result['can_forward_full_result'])
        self.assertFalse(result['can_execute_action'])
        self.assertIn('human review', result['free_text_reason_review'])

    def test_title_only_not_given_hidden_full_materials(self):
        record = fixture()
        self.assertEqual(step.verify_arm(record, 'title_only')['gate'], 'needs_more_information_no_action')
        arm = record['arms']['title_only']
        arm['answer']['decision'] = 'eligible'
        arm['response']['message']['content'] = json.dumps(arm['answer'])
        arm['raw_http_body'] = json.dumps(arm['response'])
        self.assertEqual(step.verify_arm(record, 'title_only')['gate'], 'blocked_unsupported_decision')

    def test_source_bytes_preserved_on_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'record.json'
            original = json.dumps(fixture(),ensure_ascii=False).encode('utf-8')
            path.write_bytes(original)
            report = step.verify_record(path)
            self.assertTrue(report['source_unchanged'])
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
