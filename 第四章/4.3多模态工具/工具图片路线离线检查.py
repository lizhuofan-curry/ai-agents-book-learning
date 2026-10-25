"""假HTTP验证真实分派路径、角色隔离、失败返回和全流程预算；不连接模型。"""
import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('image_tool_route', ROOT/'我的代码/06_模型按需调用图片分析工具.py')
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)


class FakeSession:
    requests = []
    responses = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, **kwargs):
        assert url == route.native.URL and self.trust_env is False
        assert kwargs['allow_redirects'] is False and kwargs['timeout'] == (10, 600)
        type(self).requests.append(deepcopy(kwargs['json']))
        status, body = type(self).responses.pop(0)
        return type('Response', (), {'status_code': status, 'text': body})()


def reply(message, reason='stop'):
    return 200, json.dumps({'model': route.native.MODEL, 'done': True, 'done_reason': reason,
                           'message': {'role': 'assistant', **message}, 'prompt_eval_count': 10, 'eval_count': 5})


def choice(name='analyze_image', args=None):
    return {'content': '', 'tool_calls': [{'function': {'name': name, 'arguments': args or {'question': '请核查图中数据与单位'}}}]}


def fake_call(request, kind, record, save, directory):
    return route.call_model(request, kind, record, save, directory, FakeSession)


class Checks(unittest.TestCase):
    def setUp(self):
        FakeSession.requests = []
        FakeSession.responses = []

    def test_tool_chain_and_role_isolation(self):
        focus = '测试：核查可见标注'
        observation = '离线工具占位证据，不是模型实测'
        FakeSession.responses = [reply(choice(args={'question': focus})), reply({'content': observation}), reply({'content': '离线最终回答'})]
        with tempfile.TemporaryDirectory() as directory:
            record = {}
            route.run_agent(record, lambda: None, directory, fake_call, ROOT)
            requests = FakeSession.requests
            self.assertEqual(len(requests), 3)
            self.assertEqual(record['model_requests'], 3)
            self.assertEqual([x['kind'] for x in record['model_calls']], ['controller', 'analyst', 'controller'])
            self.assertTrue(all('images' not in m for m in requests[0]['messages'] + requests[2]['messages']))
            self.assertEqual(requests[1]['messages'][1]['content'], focus)
            image = base64.b64decode(requests[1]['messages'][1]['images'][0], validate=True)
            self.assertEqual(image, route.native.load_input(ROOT))
            tool_message = requests[2]['messages'][-1]
            self.assertEqual(tool_message['role'], 'tool')
            self.assertEqual(json.loads(tool_message['content'])['analysis'], observation)
            self.assertNotIn('tools', requests[2])
            usage = route.comparison(record, ROOT)['usage']['tool_route']
            self.assertEqual(usage['prompt_tokens_reported'], 30)
            self.assertEqual(usage['analyst_requests'], 1)
            self.assertTrue(record['evidence_received'])
            with self.assertRaises(RuntimeError):
                route.run_agent(record, lambda: None, directory, fake_call, ROOT)

    def test_early_answer_does_not_force_tool_or_claim_evidence(self):
        FakeSession.responses = [reply({'content': '尚无图片证据'})]
        with tempfile.TemporaryDirectory() as directory:
            record = {}
            route.run_agent(record, lambda: None, directory, fake_call, ROOT)
            self.assertEqual(len(FakeSession.requests), 1)
            self.assertEqual(record['tool_calls'], [])
            self.assertFalse(record['evidence_received'])
            self.assertEqual(record['status'], 'answer_without_image_evidence_review_pending')

    def test_invalid_tool_not_executed_or_fixed(self):
        FakeSession.responses = [reply(choice(name='other_tool'))]
        with tempfile.TemporaryDirectory() as directory:
            record = {}
            with self.assertRaises(ValueError):
                route.run_agent(record, lambda: None, directory, fake_call, ROOT)
            self.assertEqual(len(FakeSession.requests), 1)
            self.assertEqual(record['tool_calls'], [])
        for args in ({'question': 'x', 'extra': 1}, {'question': 3}, {'question': ' '}):
            with self.assertRaises(ValueError):
                route.selected_question(choice(args=args))

    def test_analyst_failure_is_real_error_returned_without_retry(self):
        FakeSession.responses = [reply(choice()), (500, '离线视觉故障'), reply({'content': '工具失败，无法确定'})]
        with tempfile.TemporaryDirectory() as directory:
            record = {}
            route.run_agent(record, lambda: None, directory, fake_call, ROOT)
            self.assertEqual(len(FakeSession.requests), 3)
            self.assertFalse(record['evidence_received'])
            self.assertEqual(record['model_calls'][1]['raw_http_body'], '离线视觉故障')
            result = json.loads(FakeSession.requests[2]['messages'][-1]['content'])
            self.assertFalse(result['success'])
            self.assertNotIn('analysis', result)

    def test_second_tool_request_is_stopped(self):
        FakeSession.responses = [reply(choice()), reply({'content': '离线证据'}), reply(choice())]
        with tempfile.TemporaryDirectory() as directory:
            record = {}
            with self.assertRaises(ValueError):
                route.run_agent(record, lambda: None, directory, fake_call, ROOT)
            self.assertEqual(len(record['tool_calls']), 1)
            self.assertEqual(len(FakeSession.requests), 3)
            self.assertNotIn('final_answer', record)

    def test_truncated_controller_reply_saved_and_stopped(self):
        FakeSession.responses = [reply({'content': '未完成'}, 'length')]
        with tempfile.TemporaryDirectory() as directory:
            record = {}
            with self.assertRaises(ValueError):
                route.run_agent(record, lambda: None, directory, fake_call, ROOT)
            self.assertEqual(len(FakeSession.requests), 1)
            self.assertIn('length', record['model_calls'][0]['raw_http_body'])

    def test_budget_blocks_hidden_extra_request(self):
        with tempfile.TemporaryDirectory() as directory:
            record = {'model_calls': [{'kind': 'controller'}, {'kind': 'analyst'}, {'kind': 'controller'}]}
            with self.assertRaises(RuntimeError):
                fake_call({}, 'analyst', record, lambda: None, directory)
            self.assertEqual(FakeSession.requests, [])


if __name__ == '__main__':
    unittest.main()
