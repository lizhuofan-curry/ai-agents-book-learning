"""单题RAG控制流的模拟检查；不访问Ollama、不运行正式题。"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1]/'第三章/3.8Agentic_RAG/我的代码/03_单题普通RAG与自主检索.py'
spec = importlib.util.spec_from_file_location('live_step', PATH)
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)


def search(query):
    return {'action': 'search', 'query': query, 'claims': [], 'limitations': ''}


def answer(ref='fake_1'):
    return {'action': 'answer', 'query': '', 'claims': [{'text': '测试事实', 'citations': [ref]}],
            'limitations': '仅合成材料'}


class FakeRetriever:
    def search(self, query, top_k):
        return [{'chunk_id': 'fake_1', 'text': '第一条 合成测试事实', 'score': 1}]


class FakeSession:
    def __init__(self, actions, reason='stop'):
        self.actions = iter(actions)
        self.reason = reason
        self.requests = []
    def post(self, url, **kwargs):
        self.requests.append(kwargs['json'])
        action = next(self.actions)
        content = action if isinstance(action, str) else json.dumps(action, ensure_ascii=False)
        raw = {'model': 'fake', 'done': True, 'done_reason': self.reason,
               'message': {'content': content}, 'prompt_eval_count': 10, 'eval_count': 5}
        class Response:
            status_code = 200
            text = json.dumps(raw, ensure_ascii=False)
            def raise_for_status(self):
                pass
        return Response()


class LiveChecks(unittest.TestCase):
    def run_arm(self, mode, actions, reason='stop'):
        arm = {}
        session = FakeSession(actions, reason)
        with contextlib.redirect_stdout(io.StringIO()):
            step.run_arm(mode, '合成问题', FakeRetriever(), session, 'fake', arm, lambda: None)
        return arm, session

    def test_ordinary_one_search_one_call(self):
        arm, session = self.run_arm('ordinary', [answer()])
        self.assertEqual(arm['status'], 'answered_review_pending')
        self.assertEqual(len(arm['searches']), 1)
        self.assertEqual(len(session.requests), 1)

    def test_agent_decides_second_query_and_stops(self):
        arm, session = self.run_arm('agentic', [search('甲'), search('乙'), answer()])
        self.assertEqual(arm['status'], 'answered_review_pending')
        self.assertEqual([s['query'] for s in arm['searches']], ['甲', '乙'])
        self.assertEqual(len(session.requests), 3)
        self.assertNotIn('检索工具返回', session.requests[0]['messages'][-1]['content'])
        self.assertIn('检索工具返回', session.requests[1]['messages'][-1]['content'])

    def test_budget_without_forced_answer(self):
        arm, session = self.run_arm('agentic', [search('甲'), search('乙'), search('丙'), search('丁')])
        self.assertEqual(arm['status'], 'budget_exhausted')
        self.assertEqual(len(arm['searches']), 3)
        self.assertEqual(len(session.requests), 4)
        self.assertNotIn('answer', arm)

    def test_repeated_query_stops(self):
        arm, _ = self.run_arm('agentic', [search('甲'), search('甲')])
        self.assertEqual(arm['status'], 'repeated_query_stopped')
        self.assertEqual(len(arm['searches']), 1)

    def test_unseen_citation_and_premature_answer_rejected(self):
        for mode, actions in [('ordinary', [answer('invented')]), ('agentic', [answer()])]:
            arm, _ = self.run_arm(mode, actions)
            self.assertEqual(arm['status'], 'failed')
            self.assertNotIn('answer', arm)
        # 检索确实执行但未命中时，允许明确说明信息不足。
        empty = {'action': 'answer', 'query': '', 'claims': [], 'limitations': '未检索到材料'}
        self.assertEqual(step.validate(empty, set(), False, has_searched=True), empty)

    def test_truncation_bad_json_duplicate_keys_no_retry(self):
        for action, reason in [(answer(), 'length'), ('{', 'stop'),
                               ('{"action":"search","action":"answer"}', 'stop')]:
            arm, session = self.run_arm('ordinary', [action], reason)
            self.assertEqual(arm['status'], 'failed')
            self.assertEqual(len(session.requests), 1)
            self.assertIn('raw_http_body', arm['calls'][0])

    def test_ordinary_cannot_search(self):
        arm, _ = self.run_arm('ordinary', [search('甲')])
        self.assertEqual(arm['status'], 'failed')
        self.assertEqual(len(arm['searches']), 1)

    def test_transport_failure_saved(self):
        class Broken:
            def post(self, *args, **kwargs):
                raise TimeoutError('synthetic timeout')
        arm = {}
        with contextlib.redirect_stdout(io.StringIO()):
            step.run_arm('ordinary', '合成问题', FakeRetriever(), Broken(), 'fake', arm, lambda: None)
        self.assertEqual(arm['status'], 'failed')
        self.assertEqual(arm['calls'][0]['error']['type'], 'TimeoutError')


if __name__ == '__main__':
    unittest.main()
