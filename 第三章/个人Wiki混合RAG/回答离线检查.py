"""回答入口的假响应、独立上下文、预算和摘引校验；0次真实模型请求。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wiki_answer', ROOT/'我的代码/05_依据融合证据回答三题.py')
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)
TEXT = '智能体依据环境观察做决策，通过工具采取行动。\r\n工具执行结果进入后续上下文。'
MATERIALS = [{'chunk_id': 'D1:C1', 'text': TEXT}]
PLAN = {'temperature': 0, 'seed': 42, 'num_predict': 1024, 'num_ctx': 8192}


def row(name='Q1'):
    return {'query_id': name, 'query': name+'是什么', 'hits': [
        {'chunk_id': 'D1:C1', 'path': 'wiki/concepts/example.md', 'source_line_start': 1,
         'source_line_end': 2, 'text': TEXT}]}


def answer():
    return {'status': 'supported', 'claims': [{'text': '智能体依据观察做决策并采取行动。',
        'citations': [{'chunk_id': 'D1:C1', 'quote': '智能体依据环境观察做决策'}]}], 'missing_information': []}


def response(value=None, stop='stop'):
    return {'model': step.MODEL, 'done': True, 'done_reason': stop,
            'message': {'role': 'assistant', 'content': json.dumps(value or answer(), ensure_ascii=False)},
            'prompt_eval_count': 10, 'eval_count': 20}


class AnswerChecks(unittest.TestCase):
    def test_request_has_only_current_question_and_frozen_evidence(self):
        payload = step.build_request(row(), PLAN)
        self.assertEqual([m['role'] for m in payload['messages']], ['system', 'user'])
        material = json.loads(payload['messages'][1]['content'])
        self.assertEqual(material['question'], 'Q1是什么')
        self.assertEqual(material['materials'][0]['text'], TEXT)
        self.assertNotIn('tools', payload)
        self.assertNotIn('expected_answer', material)
        self.assertEqual(payload['format'], step.SCHEMA)

    def test_supported_partial_and_insufficient_status_shapes(self):
        self.assertTrue(step.validate_answer(answer(), MATERIALS)['passed'])
        partial = answer()
        partial.update(status='partial', missing_information=['缺少一般定义'])
        self.assertTrue(step.validate_answer(partial, MATERIALS)['passed'])
        insufficient = {'status': 'insufficient', 'claims': [], 'missing_information': ['缺少相关材料']}
        self.assertTrue(step.validate_answer(insufficient, MATERIALS)['passed'])
        invalid = deepcopy(insufficient)
        invalid['missing_information'] = []
        self.assertFalse(step.validate_answer(invalid, MATERIALS)['passed'])
        invalid['status'] = []
        self.assertFalse(step.validate_answer(invalid, MATERIALS)['passed'])

    def test_unknown_id_and_invented_quote_rejected(self):
        value = answer()
        value['claims'][0]['citations'][0]['chunk_id'] = 'OTHER'
        self.assertFalse(step.validate_answer(value, MATERIALS)['passed'])
        value = answer()
        value['claims'][0]['citations'][0]['quote'] = '这是材料中完全不存在的句子'
        self.assertFalse(step.validate_answer(value, MATERIALS)['passed'])

    def test_real_quote_does_not_certify_claim_meaning(self):
        value = answer()
        value['claims'][0]['text'] = '故意夸大为任何问题都能解决。'
        result = step.validate_answer(value, MATERIALS)
        self.assertTrue(result['passed'])
        self.assertEqual(result['semantic_faithfulness'], 'pending_review')

    def test_only_line_ending_normalization_allowed(self):
        value = answer()
        value['claims'][0]['citations'][0]['quote'] = '通过工具采取行动。\n工具执行结果进入后续上下文。'
        result = step.validate_answer(value, MATERIALS)
        self.assertTrue(result['passed'])
        self.assertEqual(result['citation_matches'][0]['match'], 'line_endings_only')
        value['claims'][0]['citations'][0]['quote'] = '智能体 依据环境观察做决策'
        self.assertFalse(step.validate_answer(value, MATERIALS)['passed'])

    def test_three_fresh_fake_requests_and_no_fourth_or_same_question(self):
        record = {'model_calls': []}
        requests = []
        def fake(payload, entry, save):
            requests.append(payload)
            raw = response()
            entry.update(http_status=200, raw_http_body=json.dumps(raw, ensure_ascii=False))
            return raw
        first = step.answer_once(row('Q1'), PLAN, record, lambda: None, fake)
        self.assertTrue(first['validation']['passed'])
        with self.assertRaises(RuntimeError):
            step.answer_once(row('Q1'), PLAN, record, lambda: None, fake)
        for name in ['Q2', 'Q3']:
            step.answer_once(row(name), PLAN, record, lambda: None, fake)
        with self.assertRaises(RuntimeError):
            step.answer_once(row('Q4'), PLAN, record, lambda: None, fake)
        self.assertEqual(len(requests), 3)
        for number, payload in enumerate(requests, 1):
            self.assertEqual(len(payload['messages']), 2)
            self.assertEqual(json.loads(payload['messages'][1]['content'])['question'], f'Q{number}是什么')

    def test_truncation_saved_without_parsing_or_retry(self):
        record = {'model_calls': []}
        raw = response(stop='length')
        entry = step.answer_once(row(), PLAN, record, lambda: None, lambda *args: raw)
        self.assertEqual(len(record['model_calls']), 1)
        self.assertEqual(entry['response'], raw)
        self.assertNotIn('answer', entry)
        self.assertEqual(entry['status'], 'failed_or_incomplete')


if __name__ == '__main__':
    unittest.main(verbosity=2)
