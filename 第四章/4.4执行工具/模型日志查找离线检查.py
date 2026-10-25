"""模拟模型与真实MCP工具检查；无Ollama调用。"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/'我的代码'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('model_log_lookup', '11_模型查找完整日志证据.py')
server = load('full_log_reader', '10_完整日志只读SDK服务.py')


def tool(name, args):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': name, 'arguments': args}}]}


class LogLookupChecks(unittest.TestCase):
    def test_real_mcp_search_then_read_with_fake_model(self):
        requests = []
        found = []

        async def fake(request, record, save):
            requests.append(request)
            if len(requests) == 1:
                text = json.dumps(request, ensure_ascii=False)
                for hidden in ('log 0500:', '501', 'middle-only teaching marker', 'middle_read'):
                    self.assertNotIn(hidden, text)
                return tool('search_log', {'pattern': 'CHECKPOINT'})
            actual = json.loads(request['messages'][-1]['content'])
            if len(requests) == 2:
                self.assertTrue(actual['success'])
                self.assertNotIn('lines', actual['message'])
                found.append(actual['message']['line_numbers'][0])
                return tool('read_log_lines', {'start_line': found[0], 'end_line': found[0]})
            self.assertNotIn('tools', request)
            line = actual['message']['lines'][0]
            self.assertEqual(line['line_number'], found[0])
            self.assertIn('CHECKPOINT', line['text'])
            return {'role': 'assistant', 'content': '模拟回答：已取得该行原文，不是模型实测。'}

        record = {}
        asyncio.run(m.exercise(record, lambda: None, fake))
        self.assertEqual(len(requests), 3)
        self.assertEqual(record['status'], 'log_evidence_answer_saved_review_pending')
        self.assertEqual([x['name'] for x in record['tool_calls']], ['search_log', 'read_log_lines'])
        self.assertEqual(m.sdk.sha(m.SOURCE/'stdout_full.txt'), m.LOG_SHA)

    def test_no_read_answer_kept_as_unverified(self):
        async def fake(request, record, save):
            return {'role': 'assistant', 'content': '模拟无依据回答：头尾没有，所以不存在。'}
        record = {}
        asyncio.run(m.exercise(record, lambda: None, fake))
        self.assertEqual(record['tool_calls'], [])
        self.assertEqual(record['status'], 'answer_without_marker_source_review_pending')

    def test_read_bounds_and_search_limit(self):
        with patch.dict(os.environ, {'TEACHING_LOG_PATH': str(m.SOURCE/'stdout_full.txt'), 'TEACHING_LOG_SHA256': m.LOG_SHA}):
            self.assertTrue(server.read_log_lines(1, 21).is_error)
            self.assertTrue(server.read_log_lines(0, 1).is_error)
            self.assertTrue(server.read_log_lines(1000, 1001).is_error)
            result = json.loads(server.search_log('log').content[0].text)['message']
            self.assertEqual(result['match_count'], 1000)
            self.assertEqual(len(result['line_numbers']), 20)
            self.assertTrue(result['truncated'])

    def test_changed_full_log_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'log.txt'
            path.write_text('changed\n')
            with patch.dict(os.environ, {'TEACHING_LOG_PATH': str(path), 'TEACHING_LOG_SHA256': m.LOG_SHA}):
                self.assertTrue(server.search_log('changed').is_error)


if __name__ == '__main__':
    unittest.main()
