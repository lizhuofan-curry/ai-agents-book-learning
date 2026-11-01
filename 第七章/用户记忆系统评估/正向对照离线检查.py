"""只检查对照输入与单次请求约束，全部为假HTTP。"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import unittest

os.environ.update(PYTHON_DOTENV_DISABLED='1', MAX_RETRIES='1')
sys.path.insert(0, str(Path(__file__).parent / '我的代码'))
import RAG本机适配 as rag
control = rag.load(Path(__file__).parent / '我的代码/06_支持性答案正向对照.py', 'positive_control_test')


def fake_reply():
    data = {'dimensions': {d: {'score': 4, 'grade': 'excellent', 'reasoning': 'Fake fixture',
        'evidence': [], 'boundary_case': None} for d in ['precision', 'recall', 'reasoning', 'proactivity']},
        'hallucination': {'detected': False, 'claims': [], 'evidence': [], 'reasoning': 'Fake fixture'},
        'overall_reasoning': 'Fake fixture', 'required_info_found': {}}
    return 200, json.dumps({'model': 'deepseek-fake-offline', 'choices': [{'finish_reason': 'stop',
        'message': {'role': 'assistant', 'content': json.dumps(data)}}]})


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.plan, cls.item, cls.original, cls.edited, cls.baseline, _, cls.judge, cls.official, cls.case = control.load_control()

    def test_exactly_the_designated_answer_phrase_changes(self):
        self.assertEqual(self.edited, self.original.replace(self.plan['replace_from'], self.plan['replace_to'], 1))
        self.assertNotIn('already configured', self.edited)
        source = control.ROOT / '本地运行结果' / self.plan['source_judge_run'] / 'record.json'
        baseline = json.loads(source.read_text())['prepared_inputs'][0]['request']
        restored = json.loads(json.dumps(self.item['request']))
        restored['messages'][1]['content'] = restored['messages'][1]['content'].replace(self.edited, self.original, 1)
        self.assertEqual(restored, baseline)

    def test_original_judge_verdict_is_not_sent_to_the_control(self):
        self.assertTrue(self.baseline['veto_applied'])
        self.assertEqual(self.baseline['reward'], 0)
        payload = json.dumps(self.item['request'])
        self.assertNotIn('baseline_result_for_client_comparison_only', payload)
        self.assertNotIn('semantic_review_notes', payload)

    def test_one_request_guard_blocks_even_if_called_again(self):
        record, calls = {'judge_calls': [], 'http_requests_started': 0}, []
        def post(payload):
            calls.append(payload)
            return fake_reply()
        entry = control.score_once(self.item, record, lambda: None, self.judge, self.official, self.case.test_id, post)
        self.assertIn('judgment', entry)
        with self.assertRaises(RuntimeError):
            control.score_once(self.item, record, lambda: None, self.judge, self.official, self.case.test_id, post)
        self.assertEqual(len(calls), 1)
        self.assertEqual(record['http_requests_started'], 1)

    def test_provider_failure_is_preserved_without_fake_score(self):
        record = {'judge_calls': [], 'http_requests_started': 0}
        with self.assertRaises(self.judge.TransportFailure):
            control.score_once(self.item, record, lambda: None, self.judge, self.official, self.case.test_id,
                lambda p: (402, 'fake provider error'))
        self.assertEqual(record['http_requests_started'], 1)
        self.assertEqual(record['judge_calls'][0]['status'], 'provider_error')
        self.assertNotIn('judgment', record['judge_calls'][0])


if __name__ == '__main__':
    print('正向对照离线检查：假HTTP，0次真实模型/网络请求。', flush=True)
    def block(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
            raise RuntimeError('Offline tests prohibit socket access')
    sys.addaudithook(block)
    unittest.main()
