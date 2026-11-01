"""Hybrid条件分支测试：假HTTP/假查询向量，0次生成或嵌入前向。"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

os.environ['PYTHON_DOTENV_DISABLED'] = '1'
sys.path.insert(0, str(Path(__file__).parent / '我的代码'))
import 本机记忆适配 as base
import RAG本机适配 as rag
import Hybrid本机适配 as hybrid


def response(text='', calls=None, reason='stop', tokens=True):
    value = {'model': base.MODEL, 'done': True, 'done_reason': reason,
        'message': {'role': 'assistant', 'content': text}, 'prompt_eval_count': 100, 'eval_count': 30}
    if calls is not None:
        value['message']['tool_calls'] = calls
    if not tokens:
        del value['prompt_eval_count']
    return 200, json.dumps(value)


def tool(name='search_memory'):
    return {'function': {'name': name, 'arguments': {'query': 'checking account number'}}}


class FakeCachedBGE(hybrid.CachedDocumentBGE):
    def embed(self, texts):
        if self.calls == 0:
            return super().embed(texts)
        self.calls += 1
        if self.calls != 2 or texts != ['checking account number']:
            raise base.StopPipeline('unexpected fake query')
        self.last_usage, self.last_latency_ms = self.official.Usage(), 0.1
        # 仅供编排回归，使用旧向量作假响应；正式路径不会复用旧查询。
        return np.load(self.source['rag']['directory'] / 'query.npy', allow_pickle=False).tolist()


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rag.augment_dependencies()
        cls.official = base.load_official()
        cls.case = cls.official.UserMemoryEvaluationFramework().get_test_case('layer1_01_bank_account')
        cls.plan, cls.source = hybrid.verify_inputs()

    def run_fake(self, replies):
        record = {'model_calls': [], 'retrieval_requests': 0, 'embedding_forward_batches': 0}
        requests = []

        def post(payload):
            requests.append(deepcopy(payload))
            if len(requests) > len(replies):
                self.fail('unexpected HTTP request')
            return replies[len(requests) - 1]

        with tempfile.TemporaryDirectory(prefix='hybrid-adapter-offline-') as name:
            directory = Path(name)
            save = lambda: base.dump(directory / 'record.json', record)
            embedder = FakeCachedBGE(self.official, self.source, record, save, directory)
            error = None
            try:
                with redirect_stdout(io.StringIO()):
                    hybrid.run_hybrid_path(self.official, self.case, self.plan, self.source,
                        record, save, directory, post, embedder)
            except (Exception, base.StopPipeline) as exc:
                error = exc
            core = json.loads((directory / 'core_cards.json').read_text())
            retrieval_preserved = (directory / 'retrieval.json').exists()
        return record, requests, embedder, error, core, retrieval_preserved

    def test_direct_answer_is_one_real_choice_zero_search(self):
        record, requests, embed, error, core, _ = self.run_fake([response('Account details are in resident memory.')])
        self.assertIsNone(error)
        self.assertEqual(len(requests), 1)
        self.assertEqual(record['observed_path'], 'core_cards_only')
        self.assertEqual(record['retrieval_requests'], 0)
        self.assertEqual(embed.calls, 1)
        self.assertEqual(record['embedding_forward_batches'], 0)
        self.assertEqual(core, [c for c in self.source['cards_data'] if c['memory_tier'] == 'core'])
        self.assertEqual(len(core), 11)
        self.assertTrue(all(record['checks'].values()))
        self.assertNotIn('retrieval', record)
        self.assertIsNone(record['reported_incremental_usage']['local_billed_money'])

    def test_optional_actual_search_uses_two_requests_and_top5(self):
        record, requests, embed, error, _, preserved = self.run_fake([response(calls=[tool()]), response('Answer from supplied evidence.')])
        self.assertIsNone(error)
        self.assertEqual(len(requests), 2)
        self.assertEqual(embed.calls, 2)
        self.assertEqual(record['observed_path'], 'core_cards_plus_search')
        self.assertEqual(record['retrieval_requests'], 1)
        self.assertTrue(preserved)
        self.assertNotIn('tools', requests[1])
        self.assertEqual(len(json.loads(next(m['content'] for m in requests[1]['messages'] if m['role'] == 'tool'))), 5)
        self.assertTrue(all(record['checks'].values()))

    def test_invalid_tool_or_extra_calls_do_not_trigger_search(self):
        for calls in [[tool('Bash')], [tool(), tool()]]:
            with self.subTest(calls=calls):
                record, requests, embed, error, _, _ = self.run_fake([response(calls=calls)])
                self.assertIsInstance(error, ValueError)
                self.assertEqual(len(requests), 1)
                self.assertEqual(record['retrieval_requests'], 0)
                self.assertEqual(embed.calls, 1)

    def test_truncation_unknown_usage_and_http_failure_do_not_retry(self):
        for reply in [response('partial', reason='length'), response('unknown', tokens=False), (500, 'fake')]:
            with self.subTest(reply=reply):
                record, requests, _, error, _, _ = self.run_fake([reply])
                self.assertIsInstance(error, ValueError)
                self.assertEqual(len(requests), 1)
                self.assertEqual(record['model_calls'][0]['status'], 'failed_or_incomplete')

    def test_followup_tool_is_rejected_with_evidence_preserved(self):
        record, requests, _, error, _, preserved = self.run_fake([response(calls=[tool()]), response(calls=[tool()])])
        self.assertIsInstance(error, ValueError)
        self.assertEqual(len(requests), 2)
        self.assertEqual(record['retrieval_requests'], 1)
        self.assertTrue(preserved)

    def test_mismatched_document_order_and_third_dispatch_are_blocked(self):
        record = {'model_calls': [{}, {}], 'retrieval_requests': 0}
        embed = hybrid.CachedDocumentBGE(self.official, self.source, record, lambda: None, Path('/tmp'))
        with self.assertRaises(ValueError):
            embed.embed(['wrong source'])
        chat = hybrid.HybridChat(self.official, self.plan, record, lambda: None,
            lambda p: self.fail('budget must stop before HTTP'))
        with self.assertRaises(base.StopPipeline):
            chat.complete([])


if __name__ == '__main__':
    print('Hybrid离线检查：假HTTP/假查询向量，0次真实生成、嵌入前向或网络请求。', flush=True)
    def block(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
            raise RuntimeError('Offline tests prohibit socket access')
    sys.addaudithook(block)
    unittest.main()
