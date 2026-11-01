"""三路裁判适配回归：假HTTP，0次真实DeepSeek或本机模型请求。"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.update(PYTHON_DOTENV_DISABLED='1', MAX_RETRIES='1')
sys.path.insert(0, str(Path(__file__).parent / '我的代码'))
import RAG本机适配 as rag
judge = rag.load(Path(__file__).parent / '我的代码/05_三路结果独立裁判评分.py', 'three_path_judge_test')


def judgment():
    return {'dimensions': {d: {'score': 4, 'grade': 'excellent', 'reasoning': 'Offline fixture',
        'evidence': ['Offline fixture evidence'], 'boundary_case': None} for d in judge.DIMENSIONS},
        'hallucination': {'detected': False, 'claims': [], 'evidence': [], 'reasoning': 'Offline fixture'},
        'overall_reasoning': 'Offline fixture only', 'required_info_found': {}}


def response(data=None, reason='stop'):
    return 200, json.dumps({'model': 'deepseek-fake-offline', 'choices': [{'finish_reason': reason,
        'message': {'role': 'assistant', 'content': json.dumps(data if data is not None else judgment())}}]})


class JudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.plan, cls.inputs, _, cls.official, cls.case = judge.load_inputs()

    def test_three_inputs_use_original_source_and_no_review_verdict(self):
        self.assertEqual([x['arm'] for x in self.inputs], ['advanced_json_cards', 'rag', 'hybrid'])
        systems = []
        for item in self.inputs:
            messages = item['request']['messages']
            self.assertEqual(len(messages), 2)
            systems.append(messages[0]['content'])
            self.assertIn('AUTHORITATIVE CONVERSATION SOURCE', messages[1]['content'])
            self.assertIn('agent trace; NOT an authoritative source', messages[1]['content'])
            self.assertNotIn('retrieval_top5_omitted_existing_answer_evidence', messages[1]['content'])
            self.assertNotIn('correct_target_number_but_answer_not_accepted', messages[1]['content'])
        self.assertEqual(len(set(systems)), 1)

    def test_original_reward_threshold_veto_and_grade_normalization(self):
        data = judgment()
        data['dimensions']['recall'].update(score=2, grade='pass')
        data['dimensions']['precision']['grade'] = 'fail'
        parsed = judge.parse_judgment(self.official, self.case.test_id, json.dumps(data))
        self.assertGreater(parsed['official_result']['reward'], 0)
        self.assertFalse(parsed['official_result']['passed'])
        self.assertIn('precision', parsed['grade_labels_normalized_by_score'])
        data['hallucination']['detected'] = True
        parsed = judge.parse_judgment(self.official, self.case.test_id, json.dumps(data))
        self.assertEqual(parsed['official_result']['reward'], 0)
        self.assertTrue(parsed['official_result']['veto_applied'])

    def test_coercion_and_missing_dimension_fail_closed(self):
        mutations = [lambda d: d['dimensions']['precision'].update(score=2.5),
            lambda d: d['dimensions']['precision'].update(score=True),
            lambda d: d['hallucination'].update(detected='false'),
            lambda d: d['dimensions'].pop('recall'),
            lambda d: d.update(required_info_found={'a': True, 'b': 0.5})]
        for mutate in mutations:
            data = judgment()
            mutate(data)
            with self.assertRaises(ValueError):
                judge.parse_judgment(self.official, self.case.test_id, json.dumps(data))

    def test_one_call_per_answer_and_three_total(self):
        record, calls = {'judge_calls': []}, []
        def post(payload):
            calls.append(payload)
            return response()
        for item in self.inputs:
            judge.judge_once(item, record, lambda: None, self.official, self.case.test_id, post)
        self.assertEqual(len(calls), 3)
        with self.assertRaises(RuntimeError):
            judge.judge_once(self.inputs[0], record, lambda: None, self.official, self.case.test_id, post)
        self.assertEqual(len(calls), 3)
        self.assertIsNone(record['judge_calls'][0]['reported_usage'])

    def test_invalid_or_truncated_judge_is_not_a_fake_zero_score(self):
        for reply in [(200, '{bad-json'), (200, '{"usage":NaN}'), response(reason='length')]:
            record = {'judge_calls': []}
            entry = judge.judge_once(self.inputs[0], record, lambda: None, self.official, self.case.test_id, lambda p: reply)
            self.assertEqual(entry['status'], 'invalid_or_incomplete_judgment')
            self.assertNotIn('judgment', entry)
            self.assertIn('raw_http_body', entry)
        record = {'judge_calls': []}
        with self.assertRaises(judge.TransportFailure):
            judge.judge_once(self.inputs[0], record, lambda: None, self.official, self.case.test_id, lambda p: (402, 'fake provider error'))
        self.assertEqual(record['judge_calls'][0]['status'], 'provider_error')
        self.assertNotIn('judgment', record['judge_calls'][0])

    def test_secret_redaction_and_official_destination(self):
        secret = 'sk-' + 'offline-secret-' * 3
        value = judge.redact({'authorization': secret, 'body': 'echo ' + secret}, secret)
        self.assertNotIn(secret, json.dumps(value))
        for endpoint in ['http://api.deepseek.com', 'https://third-party.example', 'https://api.deepseek.com/private']:
            with patch('dotenv.dotenv_values', return_value={}), patch.dict(os.environ,
                    {'DEEPSEEK_BASE_URL': endpoint}, clear=True), self.assertRaises(ValueError):
                judge.load_config(self.plan)


if __name__ == '__main__':
    print('裁判离线检查：全部假HTTP，0次真实网络/模型请求。', flush=True)
    def block(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
            raise RuntimeError('Offline tests prohibit all sockets')
    sys.addaudithook(block)
    unittest.main()
