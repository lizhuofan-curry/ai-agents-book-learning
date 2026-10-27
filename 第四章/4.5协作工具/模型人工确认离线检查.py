"""假模型HTTP返回+真实MCP进程；自动测试回复不计正式人工决定。"""
import asyncio
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('model_human_test', ROOT/'我的代码/05_模型请求人工确认并依据结果回答.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeHTTP:
    requests = []
    responses = []
    def __init__(self, **kwargs):
        assert kwargs['trust_env'] is False and kwargs['follow_redirects'] is False
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False
    async def post(self, url, json):
        assert url == module.URL
        type(self).requests.append(deepcopy(json))
        body = type(self).responses.pop(0)
        return type('Response', (), {'status_code': 200, 'text': body})()


def reply(message, reason='stop'):
    return json.dumps({'model': module.MODEL, 'done': True, 'done_reason': reason,
                       'message': {'role': 'assistant', **message}, 'prompt_eval_count': 10, 'eval_count': 5})


def choice(name='create_request', args=None):
    return {'content': '', 'tool_calls': [{'function': {'name': name, 'arguments': args if args is not None else {}}}]}


class Checks(unittest.TestCase):
    def setUp(self):
        FakeHTTP.requests = []
        FakeHTTP.responses = []

    def execute(self, decision, responses, expected_error=False):
        FakeHTTP.responses = responses
        with tempfile.TemporaryDirectory() as directory, patch('httpx2.AsyncClient', FakeHTTP):
            record = {'input_source': 'test_fixture_not_real_model_or_human'}
            def input_fn(human, save):
                human['input_source'] = 'automated_test_input_not_human'
                module.sdk.local.resolve(human, decision)
                save()
            coro = module.exercise(record, lambda: None, directory, input_fn=input_fn)
            if expected_error:
                with self.assertRaises(Exception):
                    asyncio.run(asyncio.wait_for(coro, 25))
            else:
                asyncio.run(asyncio.wait_for(coro, 25))
            exists = (Path(directory)/'MCP教学结果.txt').exists()
            return record, exists

    def test_fixed_timeout_constraint_is_visible_and_matches_validation(self):
        catalog = {'tools': [{'name': 'create_request', 'inputSchema': {
            'type': 'object', 'properties': {'timeout_seconds': {'type': 'number', 'default': 60}}}}]}
        original = deepcopy(catalog)
        tools = module.model_tools(catalog)
        self.assertEqual(tools[0]['function']['parameters']['properties']['timeout_seconds']['enum'], [60])
        self.assertFalse(tools[0]['function']['parameters']['additionalProperties'])
        self.assertEqual(catalog, original)
        self.assertEqual(module.select_arguments(choice(args={'timeout_seconds': 60}), catalog), {'timeout_seconds': 60})
        self.assertEqual(module.select_arguments(choice(), catalog), {})
        with self.assertRaises(Exception):
            module.select_arguments(choice(args={'timeout_seconds': 30}), catalog)

    def test_saved_thirty_second_choice_stops_with_readable_leaf_error(self):
        # 复现原始保存响应中的参数，不请求真实模型；仍不得静默改写为60。
        record, exists = self.execute('approved', [reply(choice(args={'timeout_seconds': 30}))], True)
        self.assertFalse(exists)
        self.assertEqual(len(FakeHTTP.requests), 1)
        self.assertNotIn('mcp_calls', record)
        grouped = ExceptionGroup('outer', [ExceptionGroup('inner', [ValueError('fixed timeout mismatch')])])
        self.assertEqual(module.error_details(grouped)['exceptions'][0]['exceptions'][0]['type'], 'ValueError')
        self.assertEqual(module.leaf_errors(grouped), ['ValueError: fixed timeout mismatch'])

    def test_approve_chain_and_context_isolation(self):
        record, exists = self.execute('approved', [reply(choice()), reply({'content': '测试回答：批准后已生成'})])
        self.assertTrue(exists)
        self.assertEqual(len(record['model_calls']), 2)
        self.assertEqual([x['name'] for x in record['mcp_calls']], ['create_request','submit_decision','finish_request'])
        first, final = FakeHTTP.requests
        self.assertEqual([x['function']['name'] for x in first['tools']], ['create_request'])
        self.assertNotIn('tools', final)
        observation = json.loads(final['messages'][-1]['content'])
        self.assertTrue(observation['server_execution_result']['action']['executed'])
        self.assertEqual(final['messages'][-1]['role'], 'tool')
        self.assertTrue(all(record['artifact_checks'].values()))

    def test_reject_returned_verbatim_and_file_absent(self):
        record, exists = self.execute('rejected', [reply(choice()), reply({'content': '测试回答：拒绝，未生成'})])
        self.assertFalse(exists)
        self.assertEqual(record['server_decision']['status'], 'rejected')
        observation = json.loads(FakeHTTP.requests[1]['messages'][-1]['content'])
        self.assertFalse(observation['server_execution_result']['action']['executed'])

    def test_no_input_is_not_approval(self):
        record, exists = self.execute('input_unavailable', [reply(choice()), reply({'content': '无输入，未执行'})])
        self.assertFalse(exists)
        self.assertEqual(record['server_decision']['status'], 'input_unavailable')

    def test_early_answer_does_not_force_tool_or_wait(self):
        record, exists = self.execute('approved', [reply({'content': '测试提前回答'})])
        self.assertFalse(exists)
        self.assertNotIn('human_input', record)
        self.assertNotIn('mcp_calls', record)
        self.assertEqual(len(FakeHTTP.requests), 1)
        self.assertEqual(record['status'], 'answered_without_human_request_review_pending')

    def test_model_cannot_submit_approval(self):
        record, exists = self.execute('approved', [reply(choice('submit_decision', {'request_id': 'fake','decision':'approved'}))], True)
        self.assertFalse(exists)
        self.assertNotIn('mcp_calls', record)
        self.assertEqual(len(FakeHTTP.requests), 1)

    def test_second_tool_call_does_not_execute_again(self):
        record, exists = self.execute('rejected', [reply(choice()), reply(choice())], True)
        self.assertFalse(exists)
        self.assertEqual(len(record['mcp_calls']), 3)
        self.assertNotIn('final_answer', record)

    def test_truncated_model_response_preserved_no_retry(self):
        record, exists = self.execute('approved', [reply({'content':'未完成'}, 'length')], True)
        self.assertFalse(exists)
        self.assertEqual(len(FakeHTTP.requests), 1)
        self.assertIn('length', record['model_calls'][0]['raw_http_body'])
        self.assertNotIn('mcp_calls', record)


if __name__ == '__main__':
    if Path(sys.prefix).resolve() != module.sdk.SDK_ENV.resolve():
        if not module.sdk.PYTHON.is_file():
            raise RuntimeError('未找到已有SDK环境')
        print('切换已有第四章SDK环境运行检查；不安装依赖。', flush=True)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.execv(str(module.sdk.PYTHON), [str(module.sdk.PYTHON), str(Path(__file__)), *sys.argv[1:]])
    unittest.main()
