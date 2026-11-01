"""RAG适配回归：全部为假HTTP/假向量，0次模型请求或嵌入前向。"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ['PYTHON_DOTENV_DISABLED'] = '1'
sys.path.insert(0, str(Path(__file__).parent / '我的代码'))
import 本机记忆适配 as base
import RAG本机适配 as rag


def response(content='', calls=None, reason='stop', tokens=True):
    raw = {'model': base.MODEL, 'done': True, 'done_reason': reason,
        'message': {'role': 'assistant', 'content': content}, 'prompt_eval_count': 100, 'eval_count': 30}
    if calls is not None:
        raw['message']['tool_calls'] = calls
    if not tokens:
        del raw['prompt_eval_count']
    return 200, json.dumps(raw)


def tool(name='search_memory', arguments=None):
    return {'function': {'name': name, 'arguments': {'query': 'checking account direct deposit'} if arguments is None else arguments}}


class FakeEmbedder:
    def __init__(self, official):
        self.calls = []
        self.last_usage, self.last_latency_ms = official.Usage(), 0.1

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float('checking' in t.lower()), 1.0] for t in texts]


class RAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rag.augment_dependencies()
        cls.official = base.load_official()
        cls.case = cls.official.UserMemoryEvaluationFramework().get_test_case('layer1_01_bank_account')
        cls.plan = rag.verify_plan()[0]

    def run_fake(self, replies):
        record = {'model_calls': [], 'retrieval_requests': 0}
        requests = []
        embedder = FakeEmbedder(self.official)

        def post(payload):
            requests.append(deepcopy(payload))
            if len(requests) > len(replies):
                self.fail('unexpected extra HTTP request')
            return replies[len(requests) - 1]

        with tempfile.TemporaryDirectory(prefix='rag-adapter-offline-') as name:
            directory = Path(name)
            save = lambda: base.dump(directory / 'record.json', record)
            error = None
            try:
                with redirect_stdout(io.StringIO()):
                    rag.run_rag_path(self.official, self.case, self.plan, record, save, directory, post, embedder)
            except (Exception, base.StopPipeline) as exc:
                error = exc
            preserved = (directory / 'retrieval.json').exists()
        return record, requests, embedder, error, preserved

    def test_real_model_tool_mapping_and_source_only_top5_contract(self):
        record, requests, embed, error, preserved = self.run_fake([
            response(calls=[tool()]), response('Checking number 4429853327; routing number 123006800.')])
        self.assertIsNone(error)
        self.assertTrue(all(record['checks'].values()))
        self.assertEqual(len(requests), 2)
        self.assertEqual([len(x) for x in embed.calls], [11, 1])
        self.assertEqual(embed.calls[1], ['checking account direct deposit'])
        self.assertEqual(record['retrieval_requests'], 1)
        self.assertTrue(preserved)
        self.assertEqual(requests[0]['messages'], [{'role': 'system', 'content': self.official.SYSTEM_RULES},
            {'role': 'user', 'content': self.case.user_question}])
        self.assertIn('tools', requests[0])
        self.assertNotIn('tools', requests[1])
        result = next(m for m in requests[1]['messages'] if m['role'] == 'tool')
        self.assertEqual(result['tool_name'], 'search_memory')
        self.assertEqual(len(json.loads(result['content'])), 5)
        self.assertNotIn('tool_call_id', result)
        self.assertIsNone(record['reported_usage']['local_billed_money'])
        self.assertIsNone(record['reported_usage']['cached_input_tokens'])

    def test_no_actual_tool_request_is_failure_not_fabricated_retrieval(self):
        record, requests, embed, error, preserved = self.run_fake([response('I do not know.')])
        self.assertIsInstance(error, ValueError)
        self.assertEqual(len(requests), 1)
        self.assertEqual(record['retrieval_requests'], 0)
        self.assertEqual(len(embed.calls), 1)
        self.assertFalse(preserved)

    def test_invalid_tools_and_extra_calls_are_rejected_before_search(self):
        variants = [[tool('Bash')], [tool(), tool()], [tool(arguments={'query': ''})],
            [tool(arguments={'query': 'x', 'command': 'bad'})], [tool(arguments='not-an-object')]]
        for calls in variants:
            with self.subTest(calls=calls):
                record, requests, _, error, _ = self.run_fake([response(calls=calls)])
                self.assertIsInstance(error, ValueError)
                self.assertEqual(len(requests), 1)
                self.assertEqual(record['retrieval_requests'], 0)

    def test_response_errors_truncation_and_unknown_usage_do_not_retry(self):
        for reply in [(500, 'fake error'), response(calls=[tool()], reason='length'), response(calls=[tool()], tokens=False)]:
            with self.subTest(reply=reply):
                record, requests, _, error, _ = self.run_fake([reply])
                self.assertIsInstance(error, ValueError)
                self.assertEqual(len(requests), 1)
                self.assertEqual(record['retrieval_requests'], 0)
                self.assertIn('raw_http_body', record['model_calls'][0])

    def test_additional_search_after_top5_is_rejected_and_evidence_preserved(self):
        record, requests, _, error, preserved = self.run_fake([response(calls=[tool()]), response(calls=[tool()])])
        self.assertIsInstance(error, ValueError)
        self.assertEqual(len(requests), 2)
        self.assertEqual(record['retrieval_requests'], 1)
        self.assertTrue(preserved)
        self.assertNotIn('answer', record)

    def test_third_model_request_is_blocked_before_http(self):
        chat = rag.RAGChat(self.official, self.plan, {'model_calls': [{}, {}]}, lambda: None,
            lambda p: self.fail('budget must stop before HTTP'))
        with self.assertRaises(base.StopPipeline):
            chat.complete([])


if __name__ == '__main__':
    print('RAG离线检查：假HTTP与假向量，0次真实模型、嵌入前向或网络请求。', flush=True)
    def block(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
            raise RuntimeError('Offline tests prohibit all socket access')
    sys.addaudithook(block)
    unittest.main()
